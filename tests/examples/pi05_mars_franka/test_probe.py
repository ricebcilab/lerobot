"""The probe's pure and offline parts: the station report and action contract, the raw-report tally
and its summary, the configuration numbers, and main reporting an absent station instead of crashing.

No c3po, no torch, no checkpoint, no SpaceMouse. A fake station supplies the manifest that c3po's
describe handshake would give, the tally is fed synthetic reports in the module's layout (three int16
counts after the report id), and main's hardware gates are patched, so every test is deterministic on
a machine with a device and on one without.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import bridge
import numpy as np
import probe
import stats as stats_module
from config import load_config
from station import StationManifest

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
SHIPPED = EXAMPLE_DIR / "configs" / "franka.toml"

ARM = "follower"
ACTION_KEYS = (f"{ARM}/joint_velocity", f"{ARM}/gripper_position")
CHECKPOINT_WIDTH = 32


# ---------------------------------------------------------------- the fakes


def _report(report_id: int, x: int, y: int, z: int) -> bytes:
    """One raw hidraw report: the id, then three little-endian int16 counts."""
    return bytes([report_id]) + struct.pack("<hhh", x, y, z)


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


def _tally(data: bytes, ids=(1, 2, 3), seen=None, peaks=None, last=None):
    seen, peaks, last = (
        seen if seen is not None else {},
        peaks if peaks is not None else {},
        last if last is not None else {},
    )
    probe._tally("/dev/hidraw0", data, ids, seen, peaks, last)
    return seen, peaks, last


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


def test_the_tally_counts_report_ids_per_node_and_keeps_the_peak_counts():
    seen, peaks, last = {}, {}, {}
    probe._tally("/dev/hidraw0", _report(1, 350, -40, 12), (1, 2, 3), seen, peaks, last)
    probe._tally("/dev/hidraw0", _report(1, 300, 45, -5), (1, 2, 3), seen, peaks, last)
    probe._tally("/dev/hidraw1", _report(2, -300, 210, 55), (1, 2, 3), seen, peaks, last)
    probe._tally("/dev/hidraw1", bytes([3, 1, 0, 0, 0]), (1, 2, 3), seen, peaks, last)
    assert seen == {"/dev/hidraw0": {1: 2}, "/dev/hidraw1": {2: 1, 3: 1}}
    assert peaks == {1: (350, 45, 12), 2: (300, 210, 55)}  # |−40| < 45; |−5| < 12
    assert last == {1: (300, 45, -5), 2: (-300, 210, 55)}


def test_a_short_report_counts_toward_its_id_but_cannot_move_a_peak():
    """Which ids arrive is the finding, so a truncated report is counted; its counts only exist when
    the three int16s are all there."""
    seen, peaks, last = _tally(bytes([2, 3]))
    assert seen == {"/dev/hidraw0": {2: 1}} and peaks == {} and last == {}


def test_the_mouse_summary_prints_the_max_counts_and_the_scales_to_copy():
    """The summary lines are the artifact: max |counts| per axis, the two full scales to write into
    [spacemouse], the configured mapping and the rest six-vector."""
    seen, peaks, last = {}, {}, {}
    probe._tally("/dev/hidraw0", _report(1, 350, -40, 12), (1, 2, 3), seen, peaks, last)
    probe._tally("/dev/hidraw1", _report(2, -300, 210, 55), (1, 2, 3), seen, peaks, last)
    lines, failures = probe._mouse_report([], 0, load_config(SHIPPED), seen, peaks, last, (1, 2, 3))
    text = "\n".join(lines)
    assert failures == 0
    assert "/dev/hidraw0: id 1 x 1" in text and "/dev/hidraw1: id 2 x 1" in text
    assert "max |counts| x 350 y 40 z 12" in text
    assert "max |counts| x 300 y 210 z 55" in text
    assert "copy into [spacemouse] full_scale = 350" in text
    assert "copy into [spacemouse] rotation_full_scale = 300" in text
    assert "configured source [0, 1, 2, 3, 4, 5]" in text
    assert "rest-state six-vector [x y z rx ry rz]" in text


def test_no_rotation_reports_is_a_failed_check():
    seen, peaks, last = {}, {}, {}
    probe._tally("/dev/hidraw0", _report(1, 350, 0, 0), (1, 2, 3), seen, peaks, last)
    lines, failures = probe._mouse_report([], 0, load_config(SHIPPED), seen, peaks, last, (1, 2, 3))
    assert failures == 1
    assert "[FAIL] id 2 (rotation) reports: none arrived" in "\n".join(lines)


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
    assert f"||F7||2 = {norm:.4f}" in text
    assert f"gain: {1.0 / norm:.4f} (auto = 1/||F7||2)" in text
    sigma = config.readout.noise * stats_module.action_std(config.policy.stats)
    assert f"noise sigma (7, action units): {probe._vec(sigma)}" in text


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


# ---------------------------------------------------------------- the command


def test_main_reports_an_unavailable_station_and_exits_non_zero(tmp_path, monkeypatch, capsys):
    """No c3po here: the station section says unavailable and the probe continues, then exits non-zero.
    The CUDA gate and the hidraw enumeration are patched too, so this never touches a real device."""
    path = _write(tmp_path, _config_text(q0="[]"))
    import station as station_module

    def no_c3po(*args, **kwargs):
        raise ImportError("c3po is not installed; from the lerobot checkout run `uv pip install -e ../c3po`")

    monkeypatch.setattr(station_module, "C3poStation", no_c3po)
    monkeypatch.setattr(probe, "hidraw_nodes", lambda: [])
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    assert probe.main(["--config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "=== 1/4 configuration ===" in output
    assert "=== 2/4 station ===" in output and "UNAVAILABLE" in output
    assert "c3po is not installed" in output
    assert "=== 4/4 policy ===" in output
    assert "probe: 1/4 sections passed" in output


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
