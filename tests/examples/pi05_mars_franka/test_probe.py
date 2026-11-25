"""The probe's pure and offline parts: the station report and action contract, the raw-report tally
with its (report id, byte length) pairs and both int16 triples, the SpaceMouse verdict over both
device layouts, the configuration numbers, the steered-mode chunk timing, the lines-on-failure/
traceback rules, and main reporting an absent station instead of crashing.

No c3po, no torch, no checkpoint, no SpaceMouse. A fake station supplies the manifest that c3po's
describe handshake would give, the tally is fed synthetic reports in the module's layout (the report
id, then three and six int16 counts), and main's hardware gates are patched, so every test is
deterministic on a machine with a device and on one without.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import bridge
import numpy as np
import probe
import pytest
import stats as stats_module
from config import load_config
from station import StationManifest

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
SHIPPED = EXAMPLE_DIR / "configs" / "franka.toml"

ARM = "follower"
ACTION_KEYS = (f"{ARM}/joint_velocity", f"{ARM}/gripper_position")
CHECKPOINT_WIDTH = 32
SESSION_Q0 = [0.010, -0.629, -0.002, -2.511, -0.010, 1.906, 0.007]


# ---------------------------------------------------------------- the fakes


def _report(report_id: int, x: int, y: int, z: int) -> bytes:
    """One raw hidraw translation/rotation report: the id, then three little-endian int16 counts."""
    return bytes([report_id]) + struct.pack("<hhh", x, y, z)


def _report6(report_id: int, *values: int) -> bytes:
    """One raw six-axis report: the id, then two int16 triples (13 bytes, the 6-axis shape)."""
    return bytes([report_id]) + struct.pack("<hhhhhh", *values)


def _tally(data: bytes, tally: probe.Tally | None = None, node: str = "/dev/hidraw0") -> probe.Tally:
    tally = probe.Tally() if tally is None else tally
    probe._tally(tally, node, data)
    return tally


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "franka.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _stats_file(tmp_path: Path) -> Path:
    """A hermetic norm_stats.json: 32 wide, meaningful in the eight station dims, zeros after."""
    q01, q99 = np.zeros(CHECKPOINT_WIDTH), np.zeros(CHECKPOINT_WIDTH)
    q01[:8] = [-0.5, -0.7, -0.4, -0.9, -0.6, -0.6, -0.7, 0.0]
    q99[:8] = [0.5, 0.8, 0.4, 0.8, 0.6, 0.7, 0.7, 1.0]
    group = {"mean": [0.0] * CHECKPOINT_WIDTH, "std": [1.0] * CHECKPOINT_WIDTH}
    path = tmp_path / "norm_stats.json"
    payload = {
        "state": {**group, "q01": q01.tolist(), "q99": q99.tolist()},
        "actions": {**group, "q01": q01.tolist(), "q99": q99.tolist()},
    }
    path.write_text(json.dumps({"norm_stats": payload}), encoding="utf-8")
    return path


def _config_text(stats_path: Path | None = None, q0: str = "[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]") -> str:
    """A complete config for the configuration section: one corruption, one mode, a q0."""
    stats = "" if stats_path is None else f'stats = "{stats_path}"\n'
    return f"""
[station]
host = "127.0.0.1"
port = 4242
arm = "{ARM}"
rate = 10.0

[cameras]
exterior = "cam_a"
wrist = "cam_b"
third = "cam_c"

[policy]
path = "lerobot/pi05_droid"
{stats}device = "cpu"

[command]
translation_gain = 0.2
rotation_gain = 0.3
q0 = {q0}

[readout]
corruption = {{rotation_z_deg = 20}}
noise = 0.2
seed = 3

[modes.fc]
mode = "flow_control"
tau = 4
lam = 2
ic_gain = 7.0

[modes.mars-f]
mode = "frs"
adapter_matrix = "corruption"

[modes.mars-u]
mode = "frs"
adapter_user = "assets/user/u/user.pt"

[modes.mars]
mode = "frs"
adapter_user = "assets/user/u/user.pt"
adapter_task = "assets/task/t/task.pt"
"""


def _observation() -> dict[str, np.ndarray]:
    """The reset observation c3po's describe-derived keys imply for the shipped config."""
    frame = np.full((8, 8, 3), 40, dtype=np.uint8)
    return {
        f"{ARM}/joint_position": np.zeros(7),
        f"{ARM}/gripper_position": np.zeros(1),
        "scene_zed": frame.copy(),
        "wrist_zed": frame.copy(),
    }


def _fake_station(*, action_keys=ACTION_KEYS, native: float = 15.0, command_mode: str = "joint_velocity"):
    """A c3po Robot stand-in: the manifest c3po would build, and its read-only step(None)."""
    raw = {
        "station_model": "franka",
        "api_version": "1.0",
        "arms": [{"name": ARM, "command_mode": command_mode}],
    }
    robot = SimpleNamespace(
        _manifest=raw,
        joint_names={ARM: ["panda_joint1"]},
        step=lambda action=None: {key: value.copy() for key, value in _observation().items()},
    )
    manifest = StationManifest(
        arms=(ARM,), cameras=("scene_zed", "wrist_zed"), action_keys=action_keys, control_rate=native
    )
    return SimpleNamespace(manifest=manifest, _robot=robot)


def _mouse_tally() -> probe.Tally:
    """The session's shape: hidraw8 delivered two translation reports, no rotation on any node."""
    tally = probe.Tally()
    _tally(_report(1, 350, -40, 12), tally, "/dev/hidraw8")
    _tally(_report(1, 300, 45, -5), tally, "/dev/hidraw8")
    return tally


def _source_facts(*, rest=(0.154, 0.0, 0.0, 0.0, 0.0, 0.0)) -> probe.SourceFacts:
    return probe.SourceFacts(
        path="/dev/hidraw8",
        tare=True,
        rest=np.asarray(rest, dtype=np.float64),
        at_rest=np.zeros(6),
    )


def _mouse_report(tally: probe.Tally, source=None, error=None, config=None) -> tuple[list[str], int]:
    return probe._mouse_report(
        [], 0, load_config(SHIPPED) if config is None else config, tally, source, error
    )


# ---------------------------------------------------------------- the station report


def test_the_station_report_names_the_manifest_and_checks_the_contract(monkeypatch):
    """One reset, its keys/shapes/dtypes, the reset pose, the camera feeds, the hold window and the
    two action keys: the facts the first teleop run depends on, each with a PASS/FAIL line."""
    monkeypatch.setattr(probe, "HOLD_SECONDS", 0.01)
    lines, failures, unavailable = probe.station_section(
        load_config(SHIPPED), _fake_station(), _observation(), None
    )
    text = "\n".join(lines)
    assert unavailable is None and failures == 0, text
    assert f"manifest: arms ['{ARM}'], cameras ['scene_zed', 'wrist_zed']" in text
    assert f"action keys {list(ACTION_KEYS)}, native rate 15 Hz" in text
    assert f"{ARM}/joint_position: (7,) float64" in text
    assert "reset pose q: [+0.000 +0.000 +0.000 +0.000 +0.000 +0.000 +0.000]" in text
    assert "scene_zed: (8, 8, 3) uint8, mean 40.0/255" in text
    assert "camera third: omitted" in text
    assert "[PASS] holds without a command: max |dq| 0.0000 rad" in text
    assert f"[PASS] {ARM}/joint_velocity: joint velocities" in text
    assert f"[PASS] {ARM}/gripper_position: gripper bit" in text
    assert "[PASS] command mode: advertised 'joint_velocity'" in text


def test_the_contract_fails_a_missing_key_and_names_it(monkeypatch):
    monkeypatch.setattr(probe, "HOLD_SECONDS", 0.01)
    station = _fake_station(action_keys=(f"{ARM}/gripper_position",), native=10.0)
    lines, failures, _ = probe.station_section(load_config(SHIPPED), station, _observation(), None)
    text = "\n".join(lines)
    assert failures == 2  # the missing velocity key and the native rate below the requested 15 Hz
    assert f"[FAIL] {ARM}/joint_velocity: not in action_keys ['{ARM}/gripper_position']" in text
    assert "[FAIL] rate: native 10 vs requested 15 Hz" in text


def test_a_position_command_mode_is_refused(monkeypatch):
    monkeypatch.setattr(probe, "HOLD_SECONDS", 0.01)
    lines, failures, _ = probe.station_section(
        load_config(SHIPPED), _fake_station(command_mode="joint_position"), _observation(), None
    )
    assert failures == 1
    assert "[FAIL] command mode: advertised 'joint_position'" in "\n".join(lines)


def test_an_absent_station_is_one_unavailable_line():
    lines, failures, unavailable = probe.station_section(
        load_config(SHIPPED), None, None, ImportError("no c3po")
    )
    assert unavailable == "ImportError: no c3po"
    assert lines == ["[UNAVAILABLE] ImportError: no c3po"] and failures == 0


# ---------------------------------------------------------------- the raw-report tally


def test_the_tally_counts_pairs_per_node_and_keeps_the_first_triple_maxima():
    """A pair is (report id, byte length): a 7-byte report is a triple after the id, and a different
    length is a different report shape even under the same id."""
    tally = _tally(_report(1, 350, -40, 12), node="/dev/hidraw0")
    _tally(_report(1, 300, 45, -5), tally, "/dev/hidraw0")
    _tally(_report(2, -300, 210, 55), tally, "/dev/hidraw1")
    _tally(bytes([3, 1, 0, 0, 0]), tally, "/dev/hidraw1")
    assert tally.seen == {
        "/dev/hidraw0": {(1, 7): 2},
        "/dev/hidraw1": {(2, 7): 1, (3, 5): 1},
    }
    assert tally.peaks[("/dev/hidraw0", 1, 7)] == ((350, 45, 12), None)  # |−40| < 45; |−5| < 12
    assert tally.peaks[("/dev/hidraw1", 2, 7)] == ((300, 210, 55), None)
    assert tally.last[("/dev/hidraw0", 1, 7)] == ((300, 45, -5), None)


def test_a_short_report_counts_toward_its_pair_but_cannot_move_a_peak():
    """Which ids arrive is the finding, so a truncated report is counted; its counts only exist when
    the three int16s are all there."""
    tally = _tally(bytes([2, 3]))
    assert tally.seen == {"/dev/hidraw0": {(2, 2): 1}} and tally.peaks == {} and tally.last == {}


def test_a_six_axis_report_shows_its_second_triple_at_offset_seven():
    """The session's rotation question: a 13-byte report under any id exposes triple two, so a
    six-axis single report (or rotation in a wider report) is visible instead of lost."""
    tally = _tally(_report6(1, 350, -40, 12, 100, -20, 7))
    assert tally.seen == {"/dev/hidraw0": {(1, 13): 1}}
    assert tally.peaks[("/dev/hidraw0", 1, 13)] == ((350, 40, 12), (100, 20, 7))
    assert tally.last[("/dev/hidraw0", 1, 13)] == ((350, -40, 12), (100, -20, 7))


def test_the_second_triple_is_not_printed_for_a_seven_byte_report():
    tally = _tally(_report(1, 350, -40, 12))
    lines = probe._tally_lines(tally)
    assert any(
        "translation (offset 1) [+350 +40 +12]" in line and "rotation (offset 7)" not in line
        for line in lines
    )


def test_the_tally_lines_print_every_pair_and_its_maxima():
    tally = _tally(_report(1, 350, -40, 12), node="/dev/hidraw8")
    _tally(_report6(1, 12, -350, 4, 3, 4, -5), tally, "/dev/hidraw8")
    text = "\n".join(probe._tally_lines(tally))
    assert "/dev/hidraw8 reports: id 1 (7 bytes) x 1, id 1 (13 bytes) x 1" in text
    assert "/dev/hidraw8 id 1 (7 bytes): max |int16| translation (offset 1) [+350 +40 +12]" in text
    assert (
        "/dev/hidraw8 id 1 (13 bytes): max |int16| translation (offset 1) [+12 +350 +4], "
        "rotation (offset 7) [+3 +4 +5]" in text
    )


# ---------------------------------------------------------------- the SpaceMouse report


def test_translation_passes_and_missing_rotation_fails_with_the_diagnosis():
    """The session's shape: translation arrived on hidraw8; no id 2 and no six-axis second triple, so
    rotation is a FAIL whose diagnosis says to twist again and that the tally is where another
    id/shape would show."""
    lines, failures = _mouse_report(_mouse_tally(), source=_source_facts())
    text = "\n".join(lines)
    assert failures == 1
    assert "[PASS] id 1 (translation) reports: max |counts| x 350 y 45 z 12" in text
    assert "[FAIL] id 2 (rotation) reports: none arrived on any node" in text
    assert "[DIAG] no rotation motion arrived in either layout" in text
    assert "twist each rotation axis to its stop again" in text
    assert "second triple" in text and "13 bytes" in text


def test_a_combined_six_axis_report_passes_rotation_and_reports_its_scale():
    """The finding: no id 2 ever arrives, but the id 1 report's second triple carries the rotation, so
    rotation is present and its observed peak is the scale to copy."""
    tally = _mouse_tally()
    _tally(_report6(1, 350, -40, 12, 350, -249, 100), tally, "/dev/hidraw8")
    lines, failures = _mouse_report(tally, source=_source_facts())
    text = "\n".join(lines)
    assert failures == 0
    assert "[PASS] rotation in the combined report: max |counts| x 350 y 249 z 100" in text
    assert "[DIAG]" not in text
    assert "copy into [spacemouse] rotation_full_scale = 350" in text
    assert "the observed peak: the id 1 second triple" in text


def test_a_six_axis_second_triple_at_rest_is_not_rotation_motion():
    """A combined-layout device sends its translation reports under id 1 whether or not the stick is
    twisted, so a second triple within the noise floor is not rotation: the FAIL/DIAG path stays."""
    tally = _mouse_tally()
    _tally(_report6(1, 350, -40, 12, 3, -4, 5), tally, "/dev/hidraw8")
    lines, failures = _mouse_report(tally, source=_source_facts())
    text = "\n".join(lines)
    assert failures == 1
    assert "[FAIL] id 2 (rotation) reports: none arrived on any node" in text
    assert "[DIAG] no rotation motion arrived in either layout" in text
    assert "+5 counts" in text


def test_the_rest_vector_uses_the_combined_layouts_second_triple():
    """The fallback rest six-vector (no source, or no tare) reads rotation from the newest id 1
    six-axis report when no id 2 arrived."""
    tally = _tally(_report6(1, 0, 0, 0, 175, 0, -70), node="/dev/hidraw8")
    rest = probe._rest_from_tally(tally, load_config(SHIPPED))
    assert rest[3] == pytest.approx(0.5)  # 175 counts / the default rotation full scale 350
    assert rest[4] == pytest.approx(0.0)
    assert rest[5] == pytest.approx(-0.2)


def test_a_rotation_report_passes_the_check():
    tally = _mouse_tally()
    _tally(_report(2, -300, 210, 55), tally, "/dev/hidraw8")
    lines, failures = _mouse_report(tally, source=_source_facts())
    text = "\n".join(lines)
    assert failures == 0
    assert "[PASS] id 2 (rotation) reports: max |counts| x 300 y 210 z 55" in text
    assert "[DIAG]" not in text
    assert "copy into [spacemouse] rotation_full_scale = 300" in text


def test_no_reports_at_all_fails_both_checks():
    lines, failures = _mouse_report(probe.Tally(), source=_source_facts())
    assert failures == 2
    assert "[FAIL] id 1 (translation) reports: none arrived" in "\n".join(lines)


def test_the_source_line_names_the_pick_the_rest_vector_and_the_tare():
    """The report the session needs next: which node the source takes (not the first by name), the
    rest six-vector it captured, and that it tared (the session's x = +0.154 rest offset)."""
    lines, failures = _mouse_report(_mouse_tally(), source=_source_facts())
    text = "\n".join(lines)
    assert failures == 1  # only rotation: the source itself is fine
    assert "source would pick: /dev/hidraw8" in text
    assert "source tare: on" in text
    assert "rest six-vector [x y z rx ry rz]: [+0.154 +0.000 +0.000 +0.000 +0.000 +0.000]" in text
    assert "at rest after tare: [+0.000 +0.000 +0.000 +0.000 +0.000 +0.000]" in text


def test_an_absent_source_fails_and_the_rest_vector_still_prints_from_the_tally():
    lines, failures = _mouse_report(
        _mouse_tally(), source=None, error=FileNotFoundError("no node delivered motion")
    )
    text = "\n".join(lines)
    assert failures == 2  # rotation and the source
    assert "[FAIL] source: FileNotFoundError: no node delivered motion" in text
    assert "rest six-vector [x y z rx ry rz]: [+0.857 +0.129 -0.014" in text  # 300/350, 45/350...
    assert "the source is absent" in text


def test_tare_off_prints_the_untared_rest_vector():
    facts = probe.SourceFacts(path="/dev/hidraw8", tare=False, rest=None, at_rest=np.zeros(6))
    lines, _ = _mouse_report(_mouse_tally(), source=facts)
    text = "\n".join(lines)
    assert "source tare: off" in text
    assert "no tare captured" in text


# ---------------------------------------------------------------- the nodes


def _sysfs_node(root: Path, name: str, uevent: str, descriptor: bytes | None = None) -> None:
    device = root / name / "device"
    device.mkdir(parents=True)
    (device / "uevent").write_text(uevent, encoding="utf-8")
    if descriptor is not None:
        (device / "report_descriptor").write_bytes(descriptor)


def test_a_3dconnexion_node_line_names_its_hid_name_and_descriptor_size(tmp_path):
    """The session could not see why hidraw3 was silent: the report descriptor's size and HID_NAME
    are per-node facts, and a reader must know which node is which interface."""
    sysfs = tmp_path / "hidraw"
    _sysfs_node(
        sysfs,
        "hidraw3",
        "DRIVER=hid-generic\nHID_ID=0003:0000256F:0000C652\nHID_NAME=3Dconnexion SpaceMouse Pro\n",
        descriptor=b"\x05\x01\x09\x02\xa1\x01",
    )
    _sysfs_node(sysfs, "hidraw0", "HID_ID=0003:000026CE:000001A2\nHID_NAME=Some keyboard\n")
    line = probe._node_line("/dev/hidraw3", True, sysfs=sysfs)
    assert "HID_ID 0003:0000256F:0000C652 (3Dconnexion)" in line
    assert "HID_NAME '3Dconnexion SpaceMouse Pro'" in line
    assert "report descriptor 6 bytes" in line
    other = probe._node_line("/dev/hidraw0", False, sysfs=sysfs)
    assert "(other)" in other and "report descriptor" not in other


def test_a_node_without_a_readable_descriptor_says_so(tmp_path):
    sysfs = tmp_path / "hidraw"
    _sysfs_node(sysfs, "hidraw3", "HID_ID=0003:0000256F:0000C652\n")
    line = probe._node_line("/dev/hidraw3", True, sysfs=sysfs)
    assert "report descriptor unavailable" in line and "HID_NAME ?" in line


# ---------------------------------------------------------------- the configuration section


def test_the_configuration_section_prints_its_numbers_from_a_temp_config(tmp_path):
    """The section's channel: the config tree, the resolved q0/F7/gain, the noise sigma from the
    statistics file — computed with bridge.py and stats.py, never re-derived here."""
    stats_path = _stats_file(tmp_path)
    path = _write(tmp_path, _config_text(stats_path))
    config = load_config(path)
    lines, failures, unavailable = probe.configuration_section(config, path, None, None)
    text = "\n".join(lines)
    assert unavailable is None and failures == 0, text
    assert "127.0.0.1:4242, rate 10 Hz" in text
    assert "exterior 'cam_a', wrist 'cam_b', third cam_c" in text
    assert "fc=flow_control(tau 4, lam 2, ic_gain 7)" in text
    assert "[command] q0" in text
    q0 = np.asarray(config.command.q0, dtype=np.float64)
    matrix7 = bridge.corruption_lift(
        bridge.matrix6_from_spec(config.readout.corruption), q0, config.command.twist_length
    )
    norm = float(np.linalg.norm(matrix7, 2))
    assert sum(line.startswith("  [") for line in lines) == 7  # the seven rows: the session's F7
    assert f"||F7||2 = {norm:.4f}" in text
    assert f"gain: {1.0 / norm:.4f} (auto = 1/||F7||2)" in text
    sigma = config.readout.noise * stats_module.action_std(config.policy.stats)
    assert (
        f"noise sigma (8 dims: 7 command dims + gripper; the loop's noise is the first 7): {probe._vec(sigma)}"
        in text
    )
    assert f"({config.readout.noise:g} x action std)" in text


def test_the_modes_line_names_each_modes_adapter(tmp_path):
    """A session record has to show what an arm composes: the corruption matrix, the user module and
    the task module each appear beside their mode, not only their knobs."""
    path = _write(tmp_path, _config_text(_stats_file(tmp_path)))
    lines, _, _ = probe.configuration_section(load_config(path), path, None, None)
    modes = next(line for line in lines if line.startswith("modes: "))
    assert "mars-f=frs(n_reversal_steps 10, pinned_steps 8, adapter matrix corruption)" in modes
    mars_u = next(part for part in modes.split("; ") if part.startswith("mars-u="))
    assert "adapter user " in mars_u and mars_u.endswith("assets/user/u/user.pt)")
    mars = next(part for part in modes.split("; ") if part.startswith("mars="))
    assert "adapter user " in mars and "assets/user/u/user.pt" in mars
    assert "task " in mars and "assets/task/t/task.pt" in mars


def test_the_lift_falls_back_to_the_station_reset_pose_when_q0_is_empty(tmp_path):
    path = _write(tmp_path, _config_text(q0="[]"))
    config = load_config(path)
    observation = {f"{ARM}/joint_position": np.arange(7, dtype=np.float64) / 10.0}
    lines, failures, _ = probe.configuration_section(config, path, observation, None)
    text = "\n".join(lines)
    assert failures == 0
    assert "q0: [+0.000 +0.100 +0.200 +0.300 +0.400 +0.500 +0.600] (the station reset pose)" in text
    assert "||F7||2 = " in text


def test_an_unresolvable_corruption_is_reported_as_information_not_an_identity(tmp_path):
    path = _write(tmp_path, _config_text(q0="[]"))
    lines, failures, _ = probe.configuration_section(load_config(path), path, None, ImportError("no c3po"))
    text = "\n".join(lines)
    assert failures == 0
    assert "[INFO] q0: the reset pose is unavailable (ImportError: no c3po)" in text
    assert "[INFO] F7, gain: unresolved without q0" in text
    assert "F7 (7x7" not in text


# ---------------------------------------------------------------- the shipped config


def test_the_shipped_config_pins_the_session_reset_pose_and_turns_the_tare_on():
    """The measured reset pose is pinned so F7, the mars-f adapter and the trainer's replay agree
    across sessions; its lift is the session's 1.3846, which the pinned q0 reproduces."""
    config = load_config(SHIPPED)
    np.testing.assert_allclose(config.command.q0, SESSION_Q0)
    assert config.spacemouse.tare is True
    matrix7 = bridge.corruption_lift(
        bridge.matrix6_from_spec(config.readout.corruption),
        np.asarray(config.command.q0, dtype=np.float64),
        config.command.twist_length,
    )
    assert float(np.linalg.norm(matrix7, 2)) == pytest.approx(1.3846, abs=1e-3)
    lines, failures, _ = probe.configuration_section(config, SHIPPED, None, None)
    text = "\n".join(lines)
    assert failures == 0
    assert "q0: [+0.010 -0.629 -0.002 -2.511 -0.010 +1.906 +0.007] (from [command] q0)" in text


# ---------------------------------------------------------------- the command


def _patch_absent_hardware(monkeypatch):
    """The station, the hidraw enumeration and the CUDA gate all absent, as on a laptop."""
    import station as station_module

    def no_c3po(*args, **kwargs):
        raise ImportError("c3po is not installed; from the lerobot checkout run `uv pip install -e ../c3po`")

    monkeypatch.setattr(station_module, "C3poStation", no_c3po)
    monkeypatch.setattr(probe, "hidraw_nodes", lambda: [])
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)


def test_main_reports_an_unavailable_station_and_exits_non_zero(tmp_path, monkeypatch, capsys):
    """No c3po here: the station section says unavailable and the probe continues, then exits non-zero."""
    path = _write(tmp_path, _config_text(q0="[]"))
    _patch_absent_hardware(monkeypatch)
    assert probe.main(["--config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "=== 1/4 configuration ===" in output
    assert "=== 2/4 station ===" in output and "UNAVAILABLE" in output
    assert "c3po is not installed" in output
    assert "=== 3/4 spacemouse ===" in output and "no /dev/hidraw* node" in output
    assert "=== 4/4 policy ===" in output
    assert "probe: 1/4 sections passed" in output


def test_a_section_that_raises_keeps_its_collected_lines_and_prints_the_traceback(
    tmp_path, monkeypatch, capsys
):
    """The session's loss: checkpoint facts collected, then the policy step raised and the section
    returned one line. A raising section must print what it collected *and* the stack."""
    path = _write(tmp_path, _config_text(q0="[]"))
    _patch_absent_hardware(monkeypatch)

    def exploding(config, observation, lines=None):
        lines.append("checkpoint: type pi05; chunk_size 8 (the facts survived the raise)")
        raise TypeError("'numpy.int64' object is not iterable")

    monkeypatch.setattr(probe, "policy_section", exploding)
    assert probe.main(["--config", str(path)]) == 1
    captured = capsys.readouterr()
    assert "checkpoint: type pi05; chunk_size 8 (the facts survived the raise)" in captured.out
    assert "[FAIL] the section raised TypeError: 'numpy.int64' object is not iterable" in captured.out
    assert "status: FAIL (1 checks failed)" in captured.out
    assert "Traceback (most recent call last)" in captured.err
    assert "TypeError: 'numpy.int64' object is not iterable" in captured.err
    # The sections after the raising one still ran: a diagnostic does not hide the rest.
    assert "probe: 1/4 sections passed" in captured.out


def test_an_unavailable_section_keeps_the_lines_it_collected_before_the_gate(tmp_path, monkeypatch, capsys):
    """The session's exact loss: the checkpoint facts were collected, then prepare failed and the
    section returned only `[UNAVAILABLE]`; the facts must be printed too."""
    path = _write(tmp_path, _config_text(q0="[]"))
    _patch_absent_hardware(monkeypatch)

    def gated(config, observation, lines=None):
        lines.append("checkpoint: type pi05; chunk_size 8 (collected before the gate)")
        return probe._unavailable(lines, "prepare + act: TypeError: boom")

    monkeypatch.setattr(probe, "policy_section", gated)
    assert probe.main(["--config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "checkpoint: type pi05; chunk_size 8 (collected before the gate)" in output
    assert "[UNAVAILABLE] prepare + act: TypeError: boom" in output
    assert "status: UNAVAILABLE: prepare + act: TypeError: boom" in output


def _install_fake_cuda_and_checkpoint(monkeypatch):
    """A GPU-looking torch and a checkpoint config, so the policy section runs on a laptop.

    The real checkpoint is not downloaded (the HF cache is not on the workspace): the config stands
    in so the section reaches its prepare step, which is the point of the session's failure.
    """
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "lerobot.policies", types.ModuleType("lerobot.policies"))
    checkpoint = SimpleNamespace(
        type="pi05",
        chunk_size=8,
        n_action_steps=8,
        num_inference_steps=10,
        max_action_dim=32,
        max_state_dim=32,
        use_relative_actions=False,
        dtype="bfloat16",
        input_features={"observation.images.exterior": None},
    )
    configs = types.ModuleType("lerobot.configs.policies")
    configs.PreTrainedConfig = SimpleNamespace(from_pretrained=lambda path: checkpoint)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", configs)
    monkeypatch.setattr(probe.shutil, "which", lambda name: "nvidia-smi")


class _BrokenBundle:
    """The session's failure: the checkpoint loaded, then prepare raised a TypeError."""

    def prepare(self, batch):
        raise TypeError("'numpy.int64' object is not iterable")


def test_the_policy_section_prints_its_checkpoint_facts_before_a_failing_prepare(
    tmp_path, monkeypatch, capsys
):
    """The session's exact loss: the checkpoint facts and statistics were collected, then prepare
    raised. The facts stay in the report and the stack is printed; nothing is swallowed."""
    import policy as policy_module

    path = _write(tmp_path, _config_text(_stats_file(tmp_path), q0="[]"))
    _patch_absent_hardware(monkeypatch)
    _install_fake_cuda_and_checkpoint(monkeypatch)
    monkeypatch.setattr(policy_module, "load_bundle", lambda config: _BrokenBundle())
    assert probe.main(["--config", str(path)]) == 1
    captured = capsys.readouterr()
    assert "checkpoint: lerobot/pi05_droid; type pi05; chunk_size 8" in captured.out
    assert "load_bundle: " in captured.out
    assert "statistics file:" in captured.out
    assert "action mean (8)" in captured.out
    assert "Traceback (most recent call last)" in captured.err
    assert "TypeError: 'numpy.int64' object is not iterable" in captured.err
    assert "probe: 1/4 sections passed" in captured.out


# ---------------------------------------------------------------- the steered timing


def test_the_chunk_line_prints_the_pause_only_when_the_chunk_overruns():
    """``chunk_ms - n_action_steps * 1000/rate``, when positive: the number that decides whether the
    15 Hz loop stalls."""
    config = load_config(SHIPPED)
    overrun = probe._chunk_line("fc", "flow_control", 2000.0, config)
    assert overrun == ("steered chunk time fc (flow_control): 2000 ms; pause at n_action_steps 8: 1467 ms")
    assert probe._chunk_line("frs", "frs", 500.0, config) == "steered chunk time frs (frs): 500 ms"


class _TimedBundle:
    """A bundle stand-in: ``act`` records the command, ``reset`` counts the chunk resets, no model."""

    def __init__(self):
        self.commands = []
        self.resets = 0

    def act(self, batch, command=None):
        self.commands.append(np.array(command, dtype=np.float64))
        return np.zeros(8)

    def reset(self):
        self.resets += 1


def test_the_steered_timing_times_each_modes_chunk_act_with_a_nonzero_command(monkeypatch):
    """The autonomous chunk says nothing about FRS's reversal: the probe builds each steered mode and
    times one act with a dummy command above the deadband, so the reversal path is what it times."""
    import policy as policy_module

    config = load_config(SHIPPED)  # ships [modes.fc] and [modes.frs]
    bundle = _TimedBundle()
    built = []

    def fake_make_steering(a_bundle, table):
        built.append(table.mode)
        return SimpleNamespace()

    monkeypatch.setattr(policy_module, "make_steering", fake_make_steering)
    clock = iter([0.0, 2.0, 5.0, 9.0])  # two chunk acts: 2 s for fc, 4 s for frs
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    lines = []
    probe._steered_timing(config, {"batch": True}, 6.0, bundle, lines)
    text = "\n".join(lines)
    assert built == ["flow_control", "frs"]  # each mode's own table reached make_steering
    assert (
        "steered chunk time fc (flow_control): 2006 ms (prepare 6 + act 2000); "
        "pause at n_action_steps 8: 1473 ms" in text
    )
    assert (
        "steered chunk time frs (frs): 4006 ms (prepare 6 + act 4000); "
        "pause at n_action_steps 8: 3473 ms" in text
    )
    # The dummy command crosses the deadband in dims 0-2 and nowhere else, so FRS takes its reversal
    # path (an all-zero command would take the unsteered one and time the base policy again).
    np.testing.assert_array_equal(bundle.commands[0], [0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
    assert np.max(np.abs(bundle.commands[0])) > config.command.deadband
    assert bundle.resets == 4  # before each mode's act and after it


def test_the_steered_timing_names_a_missing_mode_and_times_the_rest(tmp_path, monkeypatch):
    """A config without one of the two tables is not a failure: the probe says which is missing and
    still times the other."""
    import policy as policy_module

    config = load_config(_write(tmp_path, _config_text(_stats_file(tmp_path))))
    bundle = _TimedBundle()
    monkeypatch.setattr(policy_module, "make_steering", lambda a_bundle, table: SimpleNamespace())
    monkeypatch.setattr(probe.time, "monotonic", lambda: 0.0)
    lines = []
    probe._steered_timing(config, {}, 0.0, bundle, lines)
    text = "\n".join(lines)
    assert "[INFO] steered timing frs: the config has no [modes.frs] table" in text
    assert "steered chunk time fc" in text


def test_a_steering_policy_that_cannot_build_is_an_information_line(monkeypatch):
    """A mode whose checkpoint is absent must not take the whole policy section down: the probe says
    which mode did not build and times the next."""
    import policy as policy_module

    config = load_config(SHIPPED)
    bundle = _TimedBundle()

    def make_steering(a_bundle, table):
        if table.mode == "flow_control":
            raise ValueError("the checkpoint's action statistics are not installed")
        return SimpleNamespace()

    monkeypatch.setattr(policy_module, "make_steering", make_steering)
    monkeypatch.setattr(probe.time, "monotonic", lambda: 0.0)
    lines = []
    probe._steered_timing(config, {}, 0.0, bundle, lines)
    text = "\n".join(lines)
    assert "[INFO] steered timing fc (flow_control) did not run: ValueError" in text
    assert "steered chunk time frs (frs)" in text


def test_an_unreadable_config_is_a_user_error(tmp_path, capsys):
    """A config the loader refuses is the one failure with no section to report: one line, exit 2."""
    assert probe.main(["--config", str(tmp_path / "absent.toml")]) == 2
    assert "mars-franka probe:" in capsys.readouterr().err


def test_importing_the_probe_needs_neither_torch_nor_c3po_nor_lerobot():
    """The hardware imports are lazy: a laptop (and this test suite) imports the probe without them."""
    code = (
        "import sys, probe; "
        "assert 'torch' not in sys.modules, 'probe imported torch'; "
        "assert 'c3po' not in sys.modules, 'probe imported c3po'; "
        "assert 'lerobot' not in sys.modules, 'probe imported lerobot'; "
        "assert hasattr(probe, 'main') and hasattr(probe, '_tally')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
