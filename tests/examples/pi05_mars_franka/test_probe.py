"""The probe's pure and offline parts: its report formatters, the action contract, the raw-report
tally, the configuration numbers, and ``main`` reporting an absent station instead of crashing.

No c3po, no torch, no checkpoint, no SpaceMouse. The hardware sections are replaced where a test has
to reach them (the station opener, the CUDA gate, the hidraw enumeration), and the one test that runs
``main`` also asserts that the import path stayed torch-free. The SpaceMouse bytes the tests feed the
tally are synthetic reports in the module's layout: three int16 counts after the report id.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

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


# ---------------------------------------------------------------- the fakes


def _report(report_id: int, x: int, y: int, z: int) -> bytes:
    """One raw hidraw report: the id, then three little-endian int16 counts."""
    return bytes([report_id]) + struct.pack("<hhh", x, y, z)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "franka.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _stats_file(tmp_path: Path) -> Path:
    """A hermetic ``norm_stats.json``: 32 wide, meaningful in the eight station dims, zeros after."""
    q01 = np.zeros(CHECKPOINT_WIDTH)
    q99 = np.zeros(CHECKPOINT_WIDTH)
    q01[:8] = [-0.5, -0.7, -0.4, -0.9, -0.6, -0.6, -0.7, 0.0]
    q99[:8] = [0.5, 0.8, 0.4, 0.8, 0.6, 0.7, 0.7, 1.0]
    group = {"mean": [0.0] * CHECKPOINT_WIDTH, "std": [1.0] * CHECKPOINT_WIDTH}
    path = tmp_path / "norm_stats.json"
    path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {**group, "q01": q01.tolist(), "q99": q99.tolist()},
                    "actions": {**group, "q01": q01.tolist(), "q99": q99.tolist()},
                }
            }
        ),
        encoding="utf-8",
    )
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


# ---------------------------------------------------------------- the report formatting


def test_the_manifest_report_names_the_stations_four_facts():
    """A fake manifest formats as the four lines c3po's describe handshake exposes, plus joint names."""
    manifest = StationManifest(
        arms=("follower",),
        cameras=("front_rgb", "wrist_rgb"),
        action_keys=ACTION_KEYS,
        control_rate=15.0,
    )
    assert probe.format_manifest(manifest) == [
        "arms: follower",
        "cameras: front_rgb, wrist_rgb",
        f"action keys: {', '.join(ACTION_KEYS)}",
        "control rate: 15 Hz (the station's native rate)",
    ]
    named = probe.format_manifest(manifest, {"follower": ["panda_joint1", "panda_joint2"]})
    assert named[-1] == "joint names for follower: panda_joint1, panda_joint2"
    assert (
        probe.format_manifest(manifest, {"follower": []})[-1] == "joint names for follower: none advertised"
    )


# ---------------------------------------------------------------- the action contract


def test_the_contract_passes_when_the_manifest_advertises_the_two_action_keys():
    checks = probe.action_contract(ARM, ACTION_KEYS, "joint_velocity")
    assert [(check.label, check.status) for check in checks] == [
        (f"{ARM}/joint_velocity", "PASS"),
        (f"{ARM}/gripper_position", "PASS"),
        ("command mode", "PASS"),
    ]
    assert "joint_velocity" in checks[2].detail


@pytest.mark.parametrize(
    ("action_keys", "failed"),
    [
        ((f"{ARM}/gripper_position",), f"{ARM}/joint_velocity"),
        ((f"{ARM}/joint_velocity",), f"{ARM}/gripper_position"),
    ],
)
def test_the_contract_fails_a_missing_action_key_and_names_it(action_keys, failed):
    """The loop sends two keys; a manifest without one refuses every ``step`` (c3po raises)."""
    checks = probe.action_contract(ARM, action_keys, "joint_velocity")
    missing = [check for check in checks if check.label == failed][0]
    assert missing.status == "FAIL"
    assert "action_keys" in missing.detail
    assert sum(check.status == "FAIL" for check in checks) == 1


def test_the_contract_fails_both_keys_when_the_manifest_advertises_neither():
    """An empty action-key list is the clearest failure: both keys the loop writes are refused."""
    checks = probe.action_contract(ARM, (), None)
    assert [check.status for check in checks] == ["FAIL", "FAIL", "INFO"]


def test_the_contract_refuses_a_position_command_mode():
    """An arm table that advertises joint_position would read the loop's velocities as positions."""
    checks = probe.action_contract(ARM, ACTION_KEYS, "joint_position")
    assert checks[2].status == "FAIL"
    assert "joint_position" in checks[2].detail


def test_a_manifest_without_a_command_mode_is_information_not_a_failure():
    """libraries older than the arm-table mode do not publish it; the action key is still the contract."""
    checks = probe.action_contract(ARM, ACTION_KEYS, None)
    assert [check.status for check in checks] == ["PASS", "PASS", "INFO"]
    assert "action key" in checks[2].detail


# ---------------------------------------------------------------- the raw-report tally


def test_the_tally_counts_report_ids_per_node_and_keeps_the_peak_counts():
    """id 1/id 2 carry three int16 counts; the tally keeps the largest |count| per axis, per id."""
    tally = probe.ReportTally()
    tally.add("/dev/hidraw0", _report(probe.TRANSLATION_REPORT, 350, -40, 12))
    tally.add("/dev/hidraw0", _report(probe.TRANSLATION_REPORT, 300, 45, -5))
    tally.add("/dev/hidraw1", _report(probe.ROTATION_REPORT, -300, 210, 55))
    tally.add("/dev/hidraw1", bytes([probe.BUTTON_REPORT, 1, 0, 0, 0]))
    tally.add("/dev/hidraw0", b"")  # an empty read is not a report

    assert tally.ids("/dev/hidraw0") == {probe.TRANSLATION_REPORT: 2}
    assert tally.ids_by_report(probe.TRANSLATION_REPORT) == {"/dev/hidraw0": 2}
    assert tally.peaks[probe.TRANSLATION_REPORT] == (350, 45, 12)
    assert tally.peaks[probe.ROTATION_REPORT] == (300, 210, 55)
    assert tally.suggestion() == (350.0, 300.0)
    assert tally.reports == 4
    lines = tally.summary()
    assert lines[0] == f"/dev/hidraw0: id {probe.TRANSLATION_REPORT} x 2"
    assert f"/dev/hidraw1: id {probe.ROTATION_REPORT} x 1, id {probe.BUTTON_REPORT} x 1" in lines
    assert f"peak |counts| id {probe.ROTATION_REPORT} (rotation): x 300, y 210, z 55" in lines


def test_a_short_report_still_counts_toward_its_id_but_cannot_move_a_peak():
    """Which ids arrive is the finding, so a truncated report is counted; its counts are only usable
    when the three int16s are all there."""
    tally = probe.ReportTally()
    tally.add("/dev/hidraw0", bytes([probe.ROTATION_REPORT, 3]))
    assert tally.ids("/dev/hidraw0") == {probe.ROTATION_REPORT: 1}
    assert tally.peaks == {}
    assert tally.summary()[1] == f"peak |counts| id {probe.TRANSLATION_REPORT} (translation): no reports"


def test_the_rest_vector_is_the_last_counts_through_the_configured_mapping():
    """At rest the device sends a zeroing report; the vector is mapped and scaled like the twist."""
    tally = probe.ReportTally()
    tally.add("/dev/hidraw0", _report(probe.TRANSLATION_REPORT, 100, -50, 0))
    tally.add("/dev/hidraw0", _report(probe.ROTATION_REPORT, 0, 0, 20))
    rest = tally.rest_vector(
        source=[0, 1, 2, 3, 4, 5],
        sign=[1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        full_scale=350.0,
        rotation_full_scale=350.0,
    )
    np.testing.assert_allclose(rest, np.array([100, -50, 0, 0, 0, -20]) / 350.0)

    tally.add("/dev/hidraw0", _report(probe.TRANSLATION_REPORT, 0, 0, 0))
    tally.add("/dev/hidraw0", _report(probe.ROTATION_REPORT, 0, 0, 0))
    np.testing.assert_allclose(tally.rest_vector([0, 1, 2, 3, 4, 5], [1.0] * 6, 350.0, 350.0), np.zeros(6))


# ---------------------------------------------------------------- the configuration section


def test_the_configuration_section_prints_its_numbers_from_a_temp_config(tmp_path):
    """The section's channel: the config tree, the resolved q0/F7/gain, and the noise sigma from the
    statistics file — computed with bridge.py and stats.py, never re-derived here."""
    stats_path = _stats_file(tmp_path)
    path = _write(tmp_path, _config_text(stats_path))
    config = load_config(path)

    section = probe.configuration_section(config, path=path)

    assert section.status == "PASS", "\n".join(section.lines)
    text = "\n".join(section.lines)
    assert "127.0.0.1:4242" in text and "rate 10 Hz" in text
    assert "exterior 'cam_a', wrist 'cam_b', third 'cam_c'" in text
    assert "q0: [+0.100 -0.200 +0.300 -0.400 +0.500 -0.600 +0.700] (from [command] q0)" in text
    assert "fc: flow_control, tau 4, lam 2, ic_gain 7" in text
    assert "noise 0.2 sigma" in text

    q0 = np.asarray(config.command.q0, dtype=np.float64)
    matrix7 = bridge.corruption_lift(
        bridge.matrix6_from_spec(config.readout.corruption), q0, config.command.twist_length
    )
    norm = float(np.linalg.norm(matrix7, 2))
    assert f"||F7||2 = {norm:.4f}" in text
    assert f"gain: {1.0 / norm:.4f}" in text
    sigma = config.readout.noise * stats_module.action_std(config.policy.stats)
    assert probe._vector_text(sigma) in text


def test_the_lift_falls_back_to_the_station_reset_pose_when_q0_is_empty(tmp_path):
    """[command] q0 = [] means the loop captures the pose at connect; the section can read it from a
    reset observation, which is what a hardware session has."""
    path = _write(tmp_path, _config_text(q0="[]"))
    config = load_config(path)
    observation = {f"{ARM}/joint_position": np.arange(7, dtype=np.float64) / 10.0}

    section = probe.configuration_section(config, path=path, reset_observation=observation)

    text = "\n".join(section.lines)
    assert section.status == "PASS"
    assert "q0: [+0.000 +0.100 +0.200 +0.300 +0.400 +0.500 +0.600] (the station's reset pose)" in text
    assert "||F7||2 = " in text


def test_an_unresolvable_corruption_is_reported_as_information_not_as_an_identity(tmp_path):
    """Without a station and without a configured q0 the lift cannot be evaluated: the section says so
    instead of printing the identity, which is not this session's corruption."""
    path = _write(tmp_path, _config_text(q0="[]"))
    config = load_config(path)

    section = probe.configuration_section(config, path=path)

    text = "\n".join(section.lines)
    assert section.status == "PASS"
    assert "[INFO] F7: not resolved" in text
    assert "F7 (7x7" not in text


# ---------------------------------------------------------------- the command


def test_main_reports_an_unavailable_station_and_exits_non_zero(tmp_path, monkeypatch, capsys):
    """No c3po here: the station section says unavailable and the probe continues, then exits non-zero.
    The CUDA gate and the hidraw enumeration are patched as well so this test is deterministic on a
    CUDA machine and never opens a real device."""
    path = _write(tmp_path, _config_text(q0="[]"))

    def no_c3po(config):
        raise ImportError("c3po is not installed; from the lerobot checkout run `uv pip install -e ../c3po`")

    monkeypatch.setattr(probe, "_open_station", no_c3po)
    monkeypatch.setattr(probe, "_cuda_available", lambda: False)
    monkeypatch.setattr(probe, "hidraw_nodes", lambda: [])

    assert probe.main(["--config", str(path)]) == 1
    output = capsys.readouterr().out
    assert "=== 1/4 configuration ===" in output
    assert "=== 2/4 station ===" in output
    assert "UNAVAILABLE" in output
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
        "assert hasattr(probe, 'main') and hasattr(probe, 'ReportTally')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
