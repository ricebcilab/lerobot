"""The Franka settings: a base YAML and per-arm files that extend it, and the steering kwargs the runner builds.

The base file spells out every default, so it pins the documented YAML shape against the dataclasses as well as
the loader. The per-arm files are the interesting case: they inherit the base's sections key by key and
override one or two, which is how `mars_fixed.yaml` adds a matrix to `frs`.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import yaml
from config import ExperimentConfig
from interfaces import ArmSpec

BASE_YAML = """
station: {host: "10.42.0.1", port: 9090, arm: null}
cameras: {exterior: null, wrist: null}
policy: {path: "lerobot/pi05_droid", prompt_generic: "do something", n_action_steps: null}
command: {qdot_full: 0.5, twist_full: 0.11}
arm:
  q_min: null
  q_max: null
  qdot_max: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  gripper_range: [0.0, 1.0]
  gripper_closed_at: 1.0
readout: {corruption: null, input_noise: 0.0}
steering:
  mode: frs
  tau: 14
  lam: 1
  ic_gain: 1.0
  n_reversal_steps: 5
  pinned_steps: null
  adapter_user: null
  adapter_matrix: null
trial: {trials: 20, max_steps: 300}
"""


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def _config(tmp_path, **sections) -> ExperimentConfig:
    """A config written from the sections given, everything else left at its default."""
    return ExperimentConfig.from_yaml(_write(tmp_path, "c.yaml", yaml.safe_dump(sections)))


def _matrix():
    return np.eye(7).tolist()


# ---------------------------------------------------------------- the base file


def test_base_yaml_round_trips_to_the_defaults(tmp_path):
    config = ExperimentConfig.from_yaml(_write(tmp_path, "base.yaml", BASE_YAML))
    assert config == ExperimentConfig()
    assert config.station.host == "10.42.0.1" and config.station.port == 9090
    assert config.station.arm is None  # the manifest's only arm
    assert config.cameras.exterior is None and config.cameras.wrist is None
    assert config.policy.path == "lerobot/pi05_droid"
    assert config.command.qdot_full == 0.5 and config.command.twist_full == 0.11
    assert config.arm.qdot_max == (0.5,) * 7
    assert config.arm.q_min is None and config.arm.q_max is None
    assert config.arm.gripper_range == (0.0, 1.0) and config.arm.gripper_closed_at == 1.0
    assert config.readout.input_noise == 0.0 and config.readout.corruption is None
    assert config.steering.mode == "frs" and config.steering.tau == 14
    assert config.trial.trials == 20 and config.trial.max_steps == 300


def test_the_settings_are_frozen(tmp_path):
    config = _config(tmp_path, station={"host": "h"})
    with pytest.raises(FrozenInstanceError):
        config.station.host = "other"
    with pytest.raises(FrozenInstanceError):
        config.trial.trials = 3


# ---------------------------------------------------------------- extends


def test_extends_merges_sections_key_by_key(tmp_path):
    _write(
        tmp_path,
        "base.yaml",
        "policy: {path: 'lerobot/pi05_droid', n_action_steps: 10}\n"
        "command: {qdot_full: 0.5, twist_full: 0.11}\n"
        "steering: {mode: frs, tau: 14}\n",
    )
    overlay = {
        "extends": "base.yaml",
        "command": {"qdot_full": 0.25},
        "steering": {"mode": "mars_fixed", "adapter_matrix": _matrix()},
    }
    merged = _write(tmp_path, "mars_fixed.yaml", yaml.safe_dump(overlay))
    config = ExperimentConfig.from_yaml(merged)
    assert config.policy.path == "lerobot/pi05_droid" and config.policy.n_action_steps == 10
    assert config.command.qdot_full == 0.25  # overridden
    assert config.command.twist_full == 0.11  # inherited from the same section
    assert config.steering.mode == "mars_fixed" and config.steering.tau == 14
    assert config.steering.adapter_matrix == _matrix()


def test_extends_names_the_file_it_came_from(tmp_path):
    _write(tmp_path, "base.yaml", "trial: {trials: 4}\n")
    broken = _write(tmp_path, "arm.yaml", "extends: base.yaml\ntrial: {trialz: 4}\n")
    with pytest.raises(ValueError, match="trial.trialz"):
        ExperimentConfig.from_yaml(broken)
    with pytest.raises(ValueError, match=str(broken)):
        ExperimentConfig.from_yaml(broken)


def test_extends_rejects_a_cycle(tmp_path):
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="extends"):
        ExperimentConfig.from_yaml(tmp_path / "a.yaml")


def test_a_null_section_keeps_the_defaults(tmp_path):
    config = _config(tmp_path, arm=None, steering={"mode": "fc"})
    assert config.arm.qdot_max == (0.5,) * 7
    assert config.steering.mode == "fc" and config.steering.ic_gain == 1.0


# ---------------------------------------------------------------- rejected keys and values


def test_unknown_keys_are_named(tmp_path):
    with pytest.raises(ValueError, match="station.portt"):
        _config(tmp_path, station={"host": "h", "portt": 9090})
    with pytest.raises(ValueError, match="command.qdotfull"):
        _config(tmp_path, command={"qdot_full": 0.5, "qdotfull": 0.5})
    with pytest.raises(ValueError, match="stationz"):
        _config(tmp_path, stationz={"host": "h"})


def test_a_section_must_be_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="'station' must be a mapping"):
        _config(tmp_path, station=[1, 2])


def test_a_negative_full_speed_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="command.qdot_full"):
        _config(tmp_path, command={"qdot_full": -0.5})
    with pytest.raises(ValueError, match="command.twist_full"):
        _config(tmp_path, command={"twist_full": 0.0})


def test_a_bad_velocity_cap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="arm.qdot_max"):
        _config(tmp_path, arm={"qdot_max": []})
    with pytest.raises(ValueError, match="arm.qdot_max"):
        _config(tmp_path, arm={"qdot_max": [0.5] * 6})
    with pytest.raises(ValueError, match="arm.qdot_max"):
        _config(tmp_path, arm={"qdot_max": [0.5] * 6 + [-0.5]})


def test_joint_limits_are_per_joint_and_paired(tmp_path):
    with pytest.raises(ValueError, match="arm.q_max"):
        _config(
            tmp_path,
            arm={
                "q_min": [-2.0] * 7,
                "q_max": [-3.0] * 7,
                "qdot_max": [0.5] * 7,
                "gripper_range": [0.0, 1.0],
                "gripper_closed_at": 1.0,
            },
        )
    with pytest.raises(ValueError, match="arm.q_min"):
        _config(tmp_path, arm={"q_min": [-2.0] * 7})


def test_the_gripper_threshold_must_lie_in_its_range(tmp_path):
    with pytest.raises(ValueError, match="arm.gripper_closed_at"):
        _config(tmp_path, arm={"gripper_range": [0.0, 1.0], "gripper_closed_at": 2.0})


def test_an_unknown_mode_is_named(tmp_path):
    with pytest.raises(ValueError, match="steering.mode"):
        _config(tmp_path, steering={"mode": "mars"})


def test_trial_counts_are_positive(tmp_path):
    with pytest.raises(ValueError, match="trial.trials"):
        _config(tmp_path, trial={"trials": 0, "max_steps": 300})


def test_noise_cannot_be_negative(tmp_path):
    with pytest.raises(ValueError, match="readout.input_noise"):
        _config(tmp_path, readout={"input_noise": -0.1})


# ---------------------------------------------------------------- the steering kwargs


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("teleop", "none"),
        ("floor", "none"),
        ("ceiling", "none"),
        ("fc", "flow_control"),
        ("frs", "frs"),
        ("mars_fixed", "frs"),
        ("mars_learned", "frs"),
    ],
)
def test_steering_config_maps_each_arm(tmp_path, mode, expected):
    adapter = {}
    if mode == "mars_fixed":
        adapter["adapter_matrix"] = _matrix()
    if mode == "mars_learned":
        adapter["adapter_user"] = "user.pt"
    config = _config(tmp_path, steering={"mode": mode, **adapter})
    assert config.steering_config()["mode"] == expected


def test_teleop_arms_steer_nothing(tmp_path):
    config = _config(tmp_path, steering={"mode": "teleop"})
    kwargs = config.steering_config()
    assert kwargs["mode"] == "none"
    assert kwargs["adapter_user"] is None and kwargs["adapter_matrix"] is None


def test_steering_config_fixes_the_franka_action_layout(tmp_path):
    config = _config(tmp_path, command={"qdot_full": 0.4})
    kwargs = config.steering_config()
    assert kwargs["n_dims"] == 7 and kwargs["gripper_dim"] == 7
    assert kwargs["operator_dims"] is None  # the stick constrains every commanded joint
    assert kwargs["reference_step"] == 0.4
    assert kwargs["pinned_steps"] is None  # null = the executed prefix


def test_pinned_steps_follow_the_executed_prefix(tmp_path):
    config = _config(tmp_path, policy={"n_action_steps": 10})
    assert config.steering_config()["pinned_steps"] == 10


def test_the_flow_knobs_reach_the_policy(tmp_path):
    config = _config(
        tmp_path,
        steering={"mode": "fc", "tau": 20, "lam": 2, "ic_gain": 0.5, "n_reversal_steps": 3},
    )
    kwargs = config.steering_config()
    assert kwargs["tau"] == 20 and kwargs["lam"] == 2 and kwargs["ic_gain"] == 0.5
    assert kwargs["n_reversal_steps"] == 3


def test_an_adapter_needs_a_steering_arm(tmp_path):
    with pytest.raises(ValueError, match="adapter_user"):
        _config(tmp_path, steering={"mode": "floor", "adapter_user": "user.pt"}).steering_config()
    with pytest.raises(ValueError, match="adapter_matrix"):
        _config(tmp_path, steering={"mode": "teleop", "adapter_matrix": _matrix()}).steering_config()


def test_a_mars_arm_needs_its_adapter(tmp_path):
    with pytest.raises(ValueError, match="adapter_matrix"):
        _config(tmp_path, steering={"mode": "mars_fixed"}).steering_config()
    with pytest.raises(ValueError, match="adapter_user"):
        _config(tmp_path, steering={"mode": "mars_learned"}).steering_config()


# ---------------------------------------------------------------- what the station and runner read


def test_arm_overrides_build_an_arm_spec(tmp_path):
    config = _config(
        tmp_path,
        arm={
            "q_min": [-2.9, -1.7, -2.9, -3.0, -2.9, -0.0, -2.9],
            "q_max": [2.9, 1.7, 2.9, -0.1, 2.9, 3.7, 2.9],
            "qdot_max": [0.5] * 7,
            "gripper_range": [0.0, 0.04],
            "gripper_closed_at": 0.0,
        },
    )
    spec = ArmSpec(joint_names=tuple(f"panda_joint{i + 1}" for i in range(7)), **config.arm_overrides())
    assert spec.q_min[0] == -2.9 and spec.qdot_max == (0.5,) * 7
    assert spec.gripper_closed_at == 0.0


def test_arm_overrides_leave_unset_joint_limits_to_the_station(tmp_path):
    overrides = _config(tmp_path).arm_overrides()
    assert "q_min" not in overrides and "q_max" not in overrides
    assert overrides["qdot_max"] == (0.5,) * 7
    assert overrides["gripper_range"] == (0.0, 1.0) and overrides["gripper_closed_at"] == 1.0


def test_camera_names_wait_for_the_probe(tmp_path):
    config = _config(tmp_path, cameras={"exterior": "cam_high"})
    with pytest.raises(ValueError, match="cameras.wrist"):
        config.camera_names()
    probed = _config(tmp_path, cameras={"exterior": "cam_high", "wrist": "cam_wrist"})
    assert probed.camera_names() == ("cam_high", "cam_wrist")


def test_a_condition_file_may_pin_its_own_prefix(tmp_path):
    """`steering.pinned_steps` reaches the steering config, and the executed prefix is the fallback."""
    from config import ExperimentConfig

    path = tmp_path / "arm.yaml"
    path.write_text(
        "steering: {mode: frs, pinned_steps: 5}\n"
        "policy: {path: x, prompt_generic: y, n_action_steps: 15}\n"
    )
    assert ExperimentConfig.from_yaml(path).steering_config()["pinned_steps"] == 5
    path.write_text("steering: {mode: frs}\npolicy: {path: x, prompt_generic: y, n_action_steps: 15}\n")
    assert ExperimentConfig.from_yaml(path).steering_config()["pinned_steps"] == 15
