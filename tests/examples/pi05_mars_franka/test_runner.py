"""The runner's wiring and one trial: the clamp, the hold, the record, and the start-pose guard.

Everything here is offline. The station is `station.FakeStation`, the steering policy returns fixed numbers,
the Jacobian is a two-by-seven linear map, and the run directory is `tmp_path`; no checkpoint, no c3po and no
arm are anywhere in the test.
"""

import json
from datetime import datetime

import numpy as np
import pytest
import torch
import yaml
from arms import fixed_adapter_matrix
from config import ExperimentConfig, SteeringSettings, TrialSettings
from interfaces import ArmSpec
from record import initial_state_hash
from runner import DROID_RESET_Q, PolicyBundle, build, require_start_pose, run, run_trial
from station import FakeStation

ARM = "follower"
NOW = lambda: datetime(2026, 1, 1, 12, 0, 0)  # noqa: E731 - a clock the record's timestamp can be pinned to


# ---------------------------------------------------------------- fakes


def observation(q, camera_names=("exterior", "wrist"), size=8):
    """One station observation: a joint pose, an open gripper and zero frames."""
    obs = {
        f"{ARM}/joint_position": np.asarray(q, dtype=np.float64),
        f"{ARM}/gripper_position": np.array([0.0]),
    }
    for name in camera_names:
        obs[name] = np.zeros((size, size, 3), dtype=np.uint8)
    return obs


class FakePolicy:
    """A steering policy whose chunks are two fixed actions and one fixed plan."""

    FIRST = torch.tensor([[1.0, -1.0, 0.4, 0.0, 0.6, -0.6, 0.2, 1.5]])
    SECOND = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])
    PLAN = torch.tensor([[0.2, -0.2, 0.1, 0.0, 0.1, -0.1, 0.05, 0.8]])

    def __init__(self):
        self.resets = 0
        self.batches = []
        self.last_policy_action = None

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        chunk = (self.FIRST, self.SECOND)[min(len(self.batches), 1)]
        self.batches.append(batch)
        self.last_policy_action = self.PLAN.clone()
        return chunk.clone()


class FakePostprocessor:
    """A postprocessor that offsets every action, so a test can see it was applied."""

    def __init__(self):
        self.calls = 0

    def __call__(self, action):
        self.calls += 1
        return action + 0.1


class LinearJacobian:
    """A Jacobian that is one fixed matrix: `twist = J @ qdot`, `qdot = pinv(J) @ twist`."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.pinv = np.linalg.pinv(self.matrix)

    def flange_pose(self, q):
        return np.zeros(3)

    def twist(self, q, qdot):
        return self.matrix @ np.asarray(qdot, dtype=np.float64)

    def joint_velocity(self, q, twist):
        return self.pinv @ np.asarray(twist, dtype=np.float64)


def _arm_config(mode="frs", **steering):
    return ExperimentConfig(
        steering=SteeringSettings(mode=mode, **steering),
        trial=TrialSettings(trials=1, max_steps=4),
    )


def _wired(config, *, control_rate=30.0, start_pose=None):
    """The composition root over the fakes; the station's rate makes one action two ticks long."""
    station = FakeStation([observation(DROID_RESET_Q)], control_rate=control_rate)
    policy = FakePolicy()
    post = FakePostprocessor()
    jacobian = LinearJacobian(np.arange(6 * 7, dtype=np.float64).reshape(6, 7) / 10.0 + np.eye(6, 7))

    def policy_factory(cfg, stats_path, rename_map):
        return PolicyBundle(policy, preprocessor=lambda batch: batch, postprocessor=post)

    wired = build(
        config,
        stats_path=None,
        station_factory=lambda cfg: station,
        policy_factory=policy_factory,
        jacobian_factory=lambda: jacobian,
        now=NOW,
        headless=True,
        start_pose=start_pose,
    )
    return wired, station, policy, post


def _run(wired, tmp_path, **kwargs):
    options = {
        "trial": 0,
        "prompt": "do something",
        "task_description": "pick the red block",
        "start_pose": DROID_RESET_Q,
        "tolerance": 0.01,
        "max_steps": 4,
        "run_dir": tmp_path,
        "seed": 0,
    }
    options.update(kwargs)
    return run_trial(wired, **options)


# ---------------------------------------------------------------- the trial


def test_the_station_gets_the_clamped_postprocessed_action(tmp_path):
    config = _arm_config()
    wired, station, policy, post = _wired(config)
    _run(wired, tmp_path)

    # FakePostprocessor adds 0.1; the velocity cap is 0.5 rad/s and the gripper's range is [0, 1].
    raw = FakePolicy.FIRST.numpy()[0] + 0.1
    expected = np.concatenate([np.clip(raw[:7], -0.5, 0.5), [min(raw[7], 1.0)]])
    sent = station.actions[0]
    assert sent[f"{ARM}/joint_velocity"] == pytest.approx(expected[:7])
    assert sent[f"{ARM}/gripper_position"] == pytest.approx([expected[7]])
    assert post.calls >= 1


def test_one_action_is_held_for_every_tick_of_the_interval(tmp_path):
    control_rate = 30.0  # 15 Hz actions: ceil(30 / 15) = 2 station ticks each
    wired, station, policy, _ = _wired(_arm_config(), control_rate=control_rate)
    _run(wired, tmp_path, max_steps=4)

    assert len(station.actions) == 4
    for first, second in ((0, 1), (2, 3)):
        assert np.array_equal(
            station.actions[first][f"{ARM}/joint_velocity"],
            station.actions[second][f"{ARM}/joint_velocity"],
        )
        assert np.array_equal(
            station.actions[first][f"{ARM}/gripper_position"],
            station.actions[second][f"{ARM}/gripper_position"],
        )
    assert not np.array_equal(
        station.actions[0][f"{ARM}/joint_velocity"], station.actions[2][f"{ARM}/joint_velocity"]
    )
    assert len(policy.batches) == 2  # one plan per action, not one per tick


def test_the_record_has_every_key_and_shape(tmp_path):
    wired, station, _, _ = _wired(_arm_config())
    result = _run(wired, tmp_path)
    record = result.record

    with np.load(tmp_path / "trial_000.npz") as data:
        assert data["action"].shape == (4, 8)
        assert data["user_translation"].shape == (4, 3)
        assert data["user_translation_raw"].shape == (4, 3)
        assert data["user_gripper"].shape == (4,)
        assert data["user_reads"].shape == (4,)
        assert data["policy_translation"].shape == (4, 3)
        assert data["joint_pos"].shape == (4, 7)
        assert data["t"].shape == (4,)
        assert bool(data["success"]) is False
        assert str(data["task_description"]) == "pick the red block"
        assert str(data["vla_prompt"]) == "do something"
        assert str(data["mode"]) == "shared_flow_reversal_steering"
        # The plan was unnormalized by the same postprocessor as the action.
        assert data["policy_translation"][0] == pytest.approx(FakePolicy.PLAN.numpy()[0][:3] + 0.1)
    assert record["user_reads"] == 2  # one read at each action boundary
    assert record["initial_state_hash"] == initial_state_hash(
        observation(DROID_RESET_Q), ARM, station.manifest.cameras
    )
    for key in (
        "pair_id",
        "initial_state_hash",
        "trial",
        "task_description",
        "vla_prompt",
        "mode",
        "corruption_matrix",
        "reversal_adapter_matrix",
        "tau",
        "lambda",
        "ic_gain",
        "n_reversal_steps",
        "pinned_steps",
        "reversal_adapter",
        "prompt",
        "operator",
        "n_action_steps",
        "success",
    ):
        assert key in record, key


def test_a_trial_writes_one_json_line_and_one_npz(tmp_path):
    wired, _, _, _ = _wired(_arm_config())
    _run(wired, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["trial_000.npz", "trials.jsonl"]
    lines = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trial"] == 0


def test_a_trial_with_a_bad_start_pose_writes_nothing(tmp_path):
    wired, _, _, _ = _wired(_arm_config())
    with pytest.raises(ValueError, match="reset"):
        _run(wired, tmp_path, start_pose=DROID_RESET_Q + 0.5)
    assert list(tmp_path.iterdir()) == []


def test_require_start_pose_refuses_a_pose_outside_the_tolerance():
    with pytest.raises(ValueError, match="reset"):
        require_start_pose(np.full(7, 0.2), np.zeros(7), 0.05)
    require_start_pose(np.full(7, 0.04), np.zeros(7), 0.05)  # inside: no raise
    require_start_pose(np.full(7, 5.0), None, 0.05)  # no configured pose: no check


def test_reset_hash_changes_with_the_pose_and_the_frame():
    cameras = ("exterior", "wrist")
    a = initial_state_hash(observation(np.zeros(7)), ARM, cameras)
    assert a != initial_state_hash(observation(np.ones(7)), ARM, cameras)
    moved = observation(np.zeros(7))
    moved["exterior"] = np.ones((8, 8, 3), dtype=np.uint8)
    assert a != initial_state_hash(moved, ARM, cameras)


# ---------------------------------------------------------------- the fixed adapter


def test_fixed_adapter_matrix_on_a_two_joint_toy():
    jac = np.array([[1.0, 0.5], [0.2, 2.0]])
    corruption = np.array([[0.8, 0.1], [-0.3, 1.2]])
    toy = LinearJacobian(jac)
    fake = np.array([[0.0, 0.0]])
    matrix = fixed_adapter_matrix(toy, np.zeros(2), corruption)

    assert matrix == pytest.approx(np.linalg.pinv(jac) @ corruption @ jac)
    # A zero corruption asks for no joint motion, whatever the operator requested.
    assert fixed_adapter_matrix(toy, np.zeros(2), np.zeros((2, 2))) == pytest.approx(np.zeros((2, 2)))


def test_fixed_adapter_matrix_embeds_a_three_axis_corruption_in_six():
    jac = np.arange(42, dtype=np.float64).reshape(6, 7) / 7.0 + np.eye(6, 7)
    corruption = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    block = np.eye(6)
    block[:3, :3] = corruption
    matrix = fixed_adapter_matrix(LinearJacobian(jac), np.zeros(7), corruption)
    assert matrix.shape == (7, 7)
    assert matrix == pytest.approx(np.linalg.pinv(jac) @ block @ jac)


def test_build_fills_mars_fixeds_matrix_at_the_start_pose():
    config = _arm_config(mode="mars_fixed")
    wired, _, _, _ = _wired(config, start_pose=np.zeros(7))
    matrix = np.asarray(wired.config.steering.adapter_matrix, dtype=np.float64)
    assert matrix.shape == (7, 7)
    assert wired.config.steering_config()["adapter_matrix"] == matrix.tolist()


def test_build_skips_the_policy_for_teleop():
    station = FakeStation([observation(np.zeros(7))], control_rate=10.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("teleop must not load a policy")

    wired = build(
        _arm_config(mode="teleop"),
        stats_path=None,
        station_factory=lambda cfg: station,
        policy_factory=forbidden,
        jacobian_factory=lambda: LinearJacobian(np.eye(6, 7)),
        now=NOW,
        headless=True,
        start_pose=np.zeros(7),
    )
    assert wired.policy is None
    assert wired.arm.joint_names == tuple(f"{ARM}/joint_{i + 1}" for i in range(7))
    assert isinstance(wired.arm, ArmSpec)

def test_a_run_writes_the_flat_config_yaml(tmp_path):
    config = ExperimentConfig(steering=SteeringSettings(mode="frs"), trial=TrialSettings(trials=2, max_steps=2))
    wired, _, _, _ = _wired(config)
    records = run(
        wired,
        prompt="do something",
        task_description="pick the red block",
        seed=0,
        start_pose=DROID_RESET_Q,
        tolerance=0.01,
        run_dir=tmp_path,
    )

    assert len(records) == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "config.yaml",
        "trial_000.npz",
        "trial_001.npz",
        "trials.jsonl",
    ]
    flat = yaml.safe_load((tmp_path / "config.yaml").read_text())
    for key in ("mode", "tau", "lambda", "n_reversal_steps", "reversal_adapter", "prompt", "operator",
                "n_action_steps", "pinned_steps", "qdot_full"):
        assert key in flat, key
    assert flat["arm"] == "frs"
    assert flat["mode"] == "shared_flow_reversal_steering"
    assert flat["prompt"] == "do something"
    assert flat["operator"] == "human"
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 2


def test_the_command_line_can_set_the_corruption_and_the_adapter(tmp_path):
    """The two knobs a block varies: the injected corruption and, for `mars_learned`, the module."""
    from runner import parse_args

    args = parse_args(
        ["--arm", "mars_learned", "--corruption", "{rotation_z_deg: 20}", "--adapter", "u.pt"]
    )
    assert args.corruption == "{rotation_z_deg: 20}" and args.adapter == "u.pt"
