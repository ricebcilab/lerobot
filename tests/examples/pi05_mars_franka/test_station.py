"""Properties of the station port: the manifest, the action dict, the observation batch, the fake.

No torch and no c3po here. The manifest derivations are called directly on a stand-in object with
the four c3po ``Robot`` attributes, because constructing ``C3poStation`` would need a station; the
subprocess test pins that importing the module does not import c3po at all. The letterbox test
checks aspect preservation against a frame whose content can only land where a letterbox puts it.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bridge import ARM_DIMS
from station import (
    IMAGE_SIZE,
    STATE_DIMS,
    FakeStation,
    StationManifest,
    _manifest_from_observations,
    _manifest_from_robot,
    action_dict,
    observation_batch,
    resize_with_pad,
)

CAMERAS = {"exterior": "scene_rgb", "wrist": "wrist_rgb"}
"""The config's camera role -> station feed mapping for a two-camera observation."""


def _frame(height: int = 720, width: int = 1280) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _observation() -> dict[str, np.ndarray]:
    """One station observation: two camera feeds plus the arm state, and a spare third feed."""
    exterior = _frame()
    exterior[:, : exterior.shape[1] // 2] = 255  # left half white, right half black
    return {
        "follower/joint_position": np.arange(ARM_DIMS, dtype=np.float64) / 10.0,
        "follower/gripper_position": np.array([0.25]),
        "scene_rgb": exterior,
        "wrist_rgb": _frame(480, 640),
        "extra_rgb": _frame(240, 320),
    }


def _canned(count: int = 3) -> list[dict[str, np.ndarray]]:
    """``count`` distinguishable observations, for the fake's sequence."""
    return [
        {
            "follower/joint_position": np.full(ARM_DIMS, float(index)),
            "follower/gripper_position": np.full(1, 0.5 * index),
            "front_rgb": np.full((8, 8, 3), 10 * index, dtype=np.uint8),
        }
        for index in range(count)
    ]


def _batch(obs: dict[str, np.ndarray], cameras: dict | None = None) -> dict:
    return observation_batch(
        obs, arm="follower", cameras=CAMERAS if cameras is None else cameras, task="pick the cube"
    )


# ---------------------------------------------------------------- the manifest


def test_the_manifest_from_a_robot_is_the_four_c3po_properties():
    """The adapter boundary: c3po's four ``Robot`` properties become the frozen manifest."""
    robot = SimpleNamespace(
        arms=["follower"],
        cameras=["front_rgb", "wrist_rgb"],
        action_keys=["follower/joint_velocity", "follower/gripper_position"],
        control_rate=15,
    )
    assert _manifest_from_robot(robot) == StationManifest(
        arms=("follower",),
        cameras=("front_rgb", "wrist_rgb"),
        action_keys=("follower/joint_velocity", "follower/gripper_position"),
        control_rate=15.0,
    )


def test_the_manifest_from_observations_orders_arms_and_actions():
    """The fake's manifest: arms in observation order, the remaining keys as cameras, action keys
    ``joint_velocity`` then ``gripper_position`` per arm (what ``action_dict`` writes)."""
    obs = {
        "left_follower/joint_position": np.zeros(ARM_DIMS),
        "right_follower/joint_position": np.zeros(ARM_DIMS),
        "left_follower/gripper_position": np.zeros(1),
        "right_follower/gripper_position": np.zeros(1),
        "front_rgb": _frame(8, 8),
    }
    manifest = _manifest_from_observations(obs, 15.0)
    assert manifest.arms == ("left_follower", "right_follower")
    assert manifest.cameras == ("front_rgb",)
    assert manifest.action_keys == (
        "left_follower/joint_velocity",
        "left_follower/gripper_position",
        "right_follower/joint_velocity",
        "right_follower/gripper_position",
    )
    assert manifest.control_rate == 15.0


# ---------------------------------------------------------------- the images


def test_the_batch_images_are_float32_channel_first_in_unit_range():
    obs = _observation()
    batch = _batch(obs)
    for feature, feed in (
        ("observation.images.base_0_rgb", "scene_rgb"),
        ("observation.images.left_wrist_0_rgb", "wrist_rgb"),
    ):
        image = batch[feature]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert image.dtype == np.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
        expected = np.moveaxis(resize_with_pad(obs[feed]).astype(np.float32) / 255.0, -1, 0)
        np.testing.assert_array_equal(image, expected)
    assert batch["task"] == "pick the cube"


def test_the_letterbox_stretches_nothing_and_bars_top_and_bottom():
    """A 720x1280 frame keeps its 16:9 shape: scaled to 224 wide it is 126 high, so the 49-row
    bars are top and bottom and the frame reaches both side edges (no left or right bar)."""
    frame = _frame()
    frame[:, :640] = 255
    padded = resize_with_pad(frame)
    assert padded.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert padded.dtype == np.uint8

    content_height = round(720 * IMAGE_SIZE / 1280)
    assert content_height * 1280 == 720 * IMAGE_SIZE  # aspect preserved, not stretched
    top = (IMAGE_SIZE - content_height) // 2
    inside = padded[top : top + content_height]
    assert inside.shape == (content_height, IMAGE_SIZE, 3)
    assert np.all(padded[:top] == 0)  # top bar
    assert np.all(padded[top + content_height :] == 0)  # bottom bar
    assert np.all(inside[:, 0] == 255)  # left edge is the frame's own white half
    assert np.all(inside[:, -1] == 0)  # right edge is the frame's own black half
    middle = padded[IMAGE_SIZE // 2]
    assert np.all(middle[:100] == 255) and np.all(middle[124:] == 0)

    white = resize_with_pad(np.full((720, 1280, 3), 255, dtype=np.uint8))
    assert np.all(white[top : top + content_height] == 255)  # full width: no side bars


def test_the_batch_image_places_the_letterboxed_content_in_the_model_frame():
    """Channel-first means the bars are the first and last rows, not the first and last columns."""
    image = _batch(_observation())["observation.images.base_0_rgb"]
    assert np.all(image[:, :49, :] == 0.0)  # top bar, all three channels
    assert np.all(image[:, :49, :].shape == (3, 49, IMAGE_SIZE))
    assert np.all(image[:, 49:175, 0] == 1.0)  # the white half spans the width...
    assert np.all(image[:, 49:175, -1] == 0.0)  # ...and the black half is the right edge


def test_the_third_image_is_omitted_until_a_feed_is_named():
    """DESIGN.md: an absent third feature is -1 with a zero mask; a zeros image would be a real
    unmasked input, so the key is omitted rather than filled."""
    obs = _observation()
    batch = _batch(obs)
    assert set(batch) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.state",
        "task",
    }
    assert "observation.images.right_wrist_0_rgb" not in _batch(obs, {**CAMERAS, "third": None})

    named = _batch(obs, {**CAMERAS, "third": "extra_rgb"})["observation.images.right_wrist_0_rgb"]
    assert named.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and named.dtype == np.float32
    expected = np.moveaxis(resize_with_pad(obs["extra_rgb"]).astype(np.float32) / 255.0, -1, 0)
    np.testing.assert_array_equal(named, expected)


# ---------------------------------------------------------------- the state


def test_the_state_is_the_eight_real_dims_padded_with_zeros_to_32():
    obs = _observation()
    state = _batch(obs)["observation.state"]
    assert state.shape == (STATE_DIMS,)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state[:ARM_DIMS], obs["follower/joint_position"].astype(np.float32))
    np.testing.assert_array_equal(state[ARM_DIMS : ARM_DIMS + 1], np.array([0.25], dtype=np.float32))
    assert np.all(state[ARM_DIMS + 1 :] == 0.0)


def test_a_wrong_shape_joint_position_raises_naming_it():
    obs = _observation()
    obs["follower/joint_position"] = np.zeros(ARM_DIMS - 1)
    with pytest.raises(ValueError, match=r"follower/joint_position.*\(7,\)"):
        _batch(obs)


def test_a_missing_joint_position_raises_naming_it():
    obs = _observation()
    del obs["follower/joint_position"]
    with pytest.raises(ValueError, match="follower/joint_position"):
        _batch(obs)


def test_a_wrong_shape_gripper_position_raises_naming_it():
    obs = _observation()
    obs["follower/gripper_position"] = np.zeros(2)
    with pytest.raises(ValueError, match=r"gripper_position.*\(1,\)"):
        _batch(obs)


# ---------------------------------------------------------------- the action dict


def test_action_dict_clips_the_joints_and_binarizes_the_gripper():
    received = action_dict(np.array([2.0, -3.0, 0.5, -0.25, 1.0, -1.0, 0.0, 0.9]), "follower")
    assert set(received) == {"follower/joint_velocity", "follower/gripper_position"}
    np.testing.assert_array_equal(
        received["follower/joint_velocity"], np.array([1.0, -1.0, 0.5, -0.25, 1.0, -1.0, 0.0])
    )
    assert received["follower/gripper_position"].shape == (1,)
    assert received["follower/gripper_position"][0] == 1.0


@pytest.mark.parametrize(
    ("value", "expected"), [(0.5, 0.0), (0.5000001, 1.0), (1.0, 1.0), (0.0, 0.0), (-0.5, 0.0)]
)
def test_the_gripper_binarizes_strictly_above_half(value, expected):
    """The Robotiq takes a position, so the policy's continuous gripper dim becomes open or shut."""
    received = action_dict(np.array([0, 0, 0, 0, 0, 0, 0, value], dtype=np.float64), "follower")
    assert received["follower/gripper_position"][0] == expected


@pytest.mark.parametrize(
    "action", [np.zeros(ARM_DIMS), np.zeros(ARM_DIMS + 2), np.zeros((1, ARM_DIMS + 1))], ids=["7", "9", "2d"]
)
def test_action_dict_rejects_a_wrong_width(action):
    with pytest.raises(ValueError, match=r"action8 must be \(8,\)"):
        action_dict(action, "follower")


# ---------------------------------------------------------------- the fake station


def test_the_fake_returns_its_observations_in_order_and_wraps():
    station = FakeStation(_canned(3), rate=15.0)
    np.testing.assert_array_equal(station.reset()["follower/joint_position"], np.zeros(ARM_DIMS))
    for expected in (1.0, 2.0, 0.0, 1.0):
        observation = station.step(np.zeros(ARM_DIMS + 1))
        np.testing.assert_array_equal(observation["follower/joint_position"], np.full(ARM_DIMS, expected))
    np.testing.assert_array_equal(station.reset()["follower/joint_position"], np.zeros(ARM_DIMS))


def test_the_fake_returns_copies_of_its_observations():
    station = FakeStation(_canned(3), rate=15.0)
    first = station.reset()
    first["follower/joint_position"][0] = 99.0
    first["front_rgb"][:] = 0
    np.testing.assert_array_equal(station.reset()["follower/joint_position"], np.zeros(ARM_DIMS))
    second = station.step(np.zeros(ARM_DIMS + 1))
    second["follower/joint_position"][0] = 99.0
    np.testing.assert_array_equal(
        station.step(np.zeros(ARM_DIMS + 1))["follower/joint_position"], np.full(ARM_DIMS, 2.0)
    )


def test_the_fake_records_only_the_actions_it_was_sent():
    station = FakeStation(_canned(3), rate=15.0)
    station.reset()
    actions = [np.arange(ARM_DIMS + 1, dtype=np.float64) + index for index in range(3)]
    station.step(actions[0])
    station.step(None)  # a read-only step, as c3po's step(None) is
    station.step(actions[1])
    station.step(actions[2])
    assert len(station.actions) == 3
    for recorded, sent in zip(station.actions, actions, strict=True):
        assert isinstance(recorded, np.ndarray)
        np.testing.assert_array_equal(recorded, sent)


def test_the_fake_manifest_is_derived_from_its_observations():
    manifest = FakeStation(_canned(3), rate=15.0).manifest
    assert manifest.arms == ("follower",)
    assert manifest.cameras == ("front_rgb",)
    assert manifest.action_keys == ("follower/joint_velocity", "follower/gripper_position")
    assert manifest.control_rate == 15.0


def test_the_fake_rejects_a_wrong_width_action():
    station = FakeStation(_canned(1), rate=15.0)
    station.reset()
    with pytest.raises(ValueError, match=r"action8 must be \(8,\)"):
        station.step(np.zeros(ARM_DIMS))


# ---------------------------------------------------------------- the boundaries


def test_a_missing_camera_feed_raises_naming_it():
    with pytest.raises(ValueError, match="missing_rgb"):
        _batch(_observation(), {"exterior": "missing_rgb", "wrist": "wrist_rgb"})


def test_a_camera_role_that_names_no_feed_raises_naming_the_role():
    with pytest.raises(ValueError, match="exterior"):
        _batch(_observation(), {"exterior": None, "wrist": "wrist_rgb"})


def test_a_camera_feed_that_is_not_a_uint8_frame_raises_naming_it():
    obs = _observation()
    obs["scene_rgb"] = obs["scene_rgb"].astype(np.float32)
    with pytest.raises(ValueError, match="scene_rgb"):
        _batch(obs)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((10, 10, 4), dtype=np.uint8),
    ],
    ids=["2d", "float", "rgba"],
)
def test_resize_with_pad_accepts_only_an_rgb_uint8_frame(frame):
    with pytest.raises(ValueError, match="uint8"):
        resize_with_pad(frame)


def test_the_module_imports_without_c3po():
    """c3po is optional: it is imported inside ``C3poStation.__init__``, never at module level."""
    example_dir = str(Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka")
    code = (
        "import sys, station; "
        "assert 'c3po' not in sys.modules, 'station imported c3po at module level'; "
        "assert hasattr(station, 'C3poStation')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": example_dir},
    )
