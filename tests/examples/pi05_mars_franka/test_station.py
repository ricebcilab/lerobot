"""The station port: the key mapping, the fake the dry run drives, the manifest, and the letterbox."""

import numpy as np
import pytest
from station import (
    C3poStation,
    FakeStation,
    action_dict,
    build_arm_spec,
    manifest_from_observations,
    resize_with_pad,
    state_from_obs,
)


def _observation(joint_position: np.ndarray | None = None, gripper_position: float = 0.3) -> dict:
    """One c3po-shaped observation from a follower arm and its two cameras."""
    joint_position = np.zeros(7) if joint_position is None else joint_position
    return {
        "follower/joint_position": joint_position,
        "follower/gripper_position": np.array([gripper_position]),
        "exterior": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def test_action_dict_splits_the_eight_numbers_and_clips_nothing():
    action = np.array([0.5, -2.0, 3.0, -0.1, 0.0, 0.25, -0.75, 0.6])
    sent = action_dict(action, "follower")
    assert set(sent) == {"follower/joint_velocity", "follower/gripper_position"}
    assert sent["follower/joint_velocity"].shape == (7,)
    assert sent["follower/gripper_position"].shape == (1,)
    # The 3.0 and -2.0 survive: the clamps live in actions.py, not here.
    np.testing.assert_array_equal(sent["follower/joint_velocity"], action[:7])
    np.testing.assert_array_equal(sent["follower/gripper_position"], action[7:])


@pytest.mark.parametrize("shape", [(7,), (9,), (8, 1)])
def test_action_dict_rejects_anything_but_one_eight_vector(shape):
    with pytest.raises(ValueError):
        action_dict(np.zeros(shape), "follower")


def test_state_from_obs_concatenates_the_joint_positions_and_the_gripper():
    joint_position = np.arange(7, dtype=np.float64)
    obs = {"follower/joint_position": joint_position, "follower/gripper_position": np.array([0.4])}
    state = state_from_obs(obs, "follower")
    assert state.shape == (8,)
    np.testing.assert_array_equal(state, np.append(joint_position, 0.4))


def test_state_from_obs_rejects_a_wrong_joint_or_gripper_shape():
    with pytest.raises(ValueError):
        state_from_obs(
            {"follower/joint_position": np.zeros(6), "follower/gripper_position": np.zeros(1)}, "follower"
        )
    with pytest.raises(ValueError):
        state_from_obs(
            {"follower/joint_position": np.zeros(7), "follower/gripper_position": np.zeros(2)}, "follower"
        )


def test_the_fake_station_cycles_observations_and_records_actions():
    observations = [_observation(np.full(7, float(i))) for i in range(3)]
    station = FakeStation(observations, control_rate=15.0)
    try:
        assert station.manifest.arms == ("follower",)
        assert station.manifest.control_rate == 15.0
        np.testing.assert_array_equal(station.reset()["follower/joint_position"], np.zeros(7))

        actions = [action_dict(np.full(8, float(i)), "follower") for i in range(3)]
        np.testing.assert_array_equal(station.step(actions[0])["follower/joint_position"], np.ones(7))
        np.testing.assert_array_equal(station.step(actions[1])["follower/joint_position"], np.full(7, 2.0))
        # The sequence is a loop, not a clip: the third step wraps to the first observation.
        np.testing.assert_array_equal(station.step(actions[2])["follower/joint_position"], np.zeros(7))
        assert [float(action["follower/gripper_position"][0]) for action in station.actions] == [
            0.0,
            1.0,
            2.0,
        ]
        # A None read advances the station but is not a command, so it is not recorded.
        station.step(None)
        assert len(station.actions) == 3
        # reset rewinds the loop.
        np.testing.assert_array_equal(station.reset()["follower/joint_position"], np.zeros(7))
    finally:
        station.close()


def test_the_manifest_finds_the_arm_and_the_cameras_from_the_observation_keys():
    manifest = manifest_from_observations([_observation()], control_rate=30.0)
    assert manifest.arms == ("follower",)
    assert manifest.arm() == "follower"
    assert manifest.cameras == ("exterior", "wrist")
    assert set(manifest.action_keys) == {"follower/joint_velocity", "follower/gripper_position"}
    assert manifest.control_rate == 30.0


def test_two_arms_keep_both_and_the_station_then_needs_one_named():
    obs = {}
    for arm in ("left", "right"):
        obs[f"{arm}/joint_position"] = np.zeros(7)
        obs[f"{arm}/gripper_position"] = np.zeros(1)
    manifest = manifest_from_observations([obs], control_rate=10.0)
    assert manifest.arms == ("left", "right")
    with pytest.raises(ValueError):
        manifest.arm()


def test_build_arm_spec_uses_the_overrides_over_the_defaults():
    manifest = manifest_from_observations([_observation()], control_rate=30.0)
    overrides = {
        "q_min": (-0.5,) * 7,
        "q_max": (0.5,) * 7,
        "qdot_max": tuple(0.05 * i for i in range(1, 8)),
        "gripper_range": (0.0, 0.8),
        "gripper_closed_at": 0.8,
    }
    spec = build_arm_spec(manifest, overrides, default_qdot_max=0.9)
    assert spec.q_min == overrides["q_min"]
    assert spec.q_max == overrides["q_max"]
    assert spec.qdot_max == overrides["qdot_max"]
    assert spec.gripper_range == overrides["gripper_range"]
    assert spec.gripper_closed_at == 0.8


def test_build_arm_spec_falls_back_to_the_panda_defaults():
    manifest = manifest_from_observations([_observation()], control_rate=30.0)
    spec = build_arm_spec(manifest, {})
    assert spec.qdot_max == (0.5,) * 7
    assert len(spec.joint_names) == 7
    assert all(hi > lo for lo, hi in zip(spec.q_min, spec.q_max, strict=True))


def test_build_arm_spec_takes_the_robots_joint_names_when_it_has_them():
    manifest = manifest_from_observations([_observation()], control_rate=30.0)
    names = tuple(f"panda_joint{i}" for i in range(1, 8))
    # robot.joint_names[arm] is a mapping, so both spellings of the argument are accepted.
    assert build_arm_spec(manifest, {}, joint_names={"follower": names}).joint_names == names
    assert build_arm_spec(manifest, {}, joint_names=names).joint_names == names


def test_build_arm_spec_rejects_a_six_joint_speed_cap():
    manifest = manifest_from_observations([_observation()], control_rate=30.0)
    with pytest.raises(ValueError):
        build_arm_spec(manifest, {"qdot_max": (0.5,) * 6})


def test_resize_with_pad_letterboxes_a_landscape_frame_without_stretching():
    source = np.full((480, 640, 3), 255, dtype=np.uint8)
    resized = resize_with_pad(source)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8

    rows, cols = np.nonzero(resized[..., 0])
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    # 640x480 scales by 224/640, so the image is 224x168 and needs 28 black rows top and bottom.
    assert (top, bottom, left, right) == (28, 195, 0, 223)
    assert np.count_nonzero(resized[:top]) == 0
    assert np.count_nonzero(resized[bottom + 1 :]) == 0
    # The content box keeps the source's 4:3 aspect: no pixel was stretched into the padding.
    width, height = right - left + 1, bottom - top + 1
    assert np.isclose(width / height, 640 / 480, atol=0.01)


def test_the_station_is_configured_at_the_rate_it_is_given(monkeypatch):
    """c3po asks r2d2 for 50 Hz unless told otherwise; the station's own rate is what the actions assume."""
    import sys
    import types

    calls = {}

    class FakeRobot:
        rate = 50.0

        def __init__(self, *, host, port, rate=None):
            calls.update(host=host, port=port, rate=rate)
            self.rate = 50.0 if rate is None else rate
            self.arms, self.cameras, self.action_keys = ("follower",), ("exterior", "wrist"), ()
            self.control_rate = 15.0
            self.joint_names = {"follower": [f"panda_joint{i}" for i in range(1, 8)]}

    monkeypatch.setitem(sys.modules, "c3po", types.SimpleNamespace(Robot=FakeRobot))
    default = C3poStation("10.0.0.1", 9090)
    assert calls["rate"] is None and default.configured_rate == 50.0  # c3po's own default

    asked = C3poStation("10.0.0.1", 9090, 15.0)
    assert calls["rate"] == 15.0 and asked.configured_rate == 15.0
