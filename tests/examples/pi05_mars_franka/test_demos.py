"""The demonstration loader: discovery from a fake dataset's metadata, sampling, split, open.

Hermetic and offline: a fake ``LeRobotDataset`` (metadata, ``fps``, ``__len__``, ``__getitem__``)
stands in for r2d2's dataset, so no ``dataset`` extra, station, or torch is needed. The fake's frames
are keyed by the dataset's feature names and its images are the station's ``(H, W, 3)`` uint8, which
is the contract the loader reads; the tests assert the loader's refusals name the feature they could
not map, because the real schema is only verified on the first recording.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from bridge import ARM_DIMS
from demos import ACTION_DIMS, discover, open_demos, split

FPS = 15.0
CAMERAS = {"exterior": "scene_zed", "wrist": "wrist_zed"}
IMAGE_HW = (8, 6)  # deliberately not 224: a letterbox would show


# ---------------------------------------------------------------- the fake dataset


def _features(
    *,
    arm: str = "follower",
    action_prefix: str = "action/follower",
    single_action: bool = False,
    cameras: tuple[str, ...] = ("scene_zed", "wrist_zed"),
    image_style: str = "bare",
    observed_velocity: bool = True,
    joint_shape: tuple[int, ...] = (ARM_DIMS,),
    extra: dict | None = None,
) -> dict:
    """A dataset's feature mapping: the arm state, the recorded actions, cameras, bookkeeping.

    ``single_action`` is the other accepted schema: one ``action`` column of shape ``(8,)`` instead of
    the separated ``<prefix>/joint_velocity`` + ``<prefix>/gripper_position`` pair.
    """
    features = {
        f"{arm}/joint_position": {"dtype": "float32", "shape": joint_shape},
        f"{arm}/gripper_position": {"dtype": "float32", "shape": (1,)},
        "timestamp": {"dtype": "float32", "shape": (1,)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
    }
    if single_action:
        features["action"] = {"dtype": "float32", "shape": (ACTION_DIMS,)}
    else:
        features[f"{action_prefix}/joint_velocity"] = {"dtype": "float32", "shape": (ARM_DIMS,)}
        features[f"{action_prefix}/gripper_position"] = {"dtype": "float32", "shape": (1,)}
    if observed_velocity:
        features[f"{arm}/joint_velocity"] = {"dtype": "float32", "shape": (ARM_DIMS,)}
    for name in cameras:
        styled = f"observation.images.{name}" if image_style == "lerobot" else name
        features[styled] = {"dtype": "video", "shape": (*IMAGE_HW, 3)}
    if extra:
        features.update(extra)
    return features


class _Meta:
    """What ``discover`` reads: ``.features`` and ``.fps``."""

    def __init__(self, features: dict, fps: float = FPS) -> None:
        self.features = features
        self.fps = fps


def _frame_action(index: int) -> np.ndarray:
    """The action recorded at global frame ``index``: seven joint velocities then the gripper."""
    return np.concatenate([np.full(ARM_DIMS, index / 100.0), np.array([index / 1000.0])])


def _dataset_frames(
    *,
    episode_lengths: tuple[int, ...],
    arm: str = "follower",
    action_prefix: str = "action/follower",
    single_action: bool = False,
    camera_features: tuple[str, ...] = ("scene_zed", "wrist_zed"),
) -> list[dict]:
    """Frames ordered by episode then position, keyed by feature name, with deterministic values."""
    frames = []
    for episode, length in enumerate(episode_lengths):
        for position in range(length):
            index = len(frames)
            frame = {
                "episode_index": np.array([episode], dtype=np.int64),
                "frame_index": np.array([position], dtype=np.int64),
                "index": np.array([index], dtype=np.int64),
                "timestamp": np.array([position / FPS], dtype=np.float32),
                "task_index": np.array([0], dtype=np.int64),
                f"{arm}/joint_position": np.arange(ARM_DIMS, dtype=np.float64) + index / 100.0,
                f"{arm}/gripper_position": np.array([float(index % 2)]),
                f"{arm}/joint_velocity": np.full(ARM_DIMS, -index / 100.0),
            }
            if single_action:
                frame["action"] = _frame_action(index)
            else:
                frame[f"{action_prefix}/joint_velocity"] = _frame_action(index)[:ARM_DIMS]
                frame[f"{action_prefix}/gripper_position"] = np.array([_frame_action(index)[ARM_DIMS]])
            for offset, name in enumerate(camera_features):
                frame[name] = (
                    ((np.arange(int(np.prod(IMAGE_HW)) * 3) + index + 7 * offset) % 256)
                    .astype(np.uint8)
                    .reshape(*IMAGE_HW, 3)
                )
            frames.append(frame)
    return frames


def _fake_dataset_class(metadata: _Meta, frames: list[dict]):
    """A stand-in for ``LeRobotDataset``: metadata, fps, length, frame dicts, and its own filter."""

    class FakeLeRobotDataset:
        created: list[FakeLeRobotDataset] = []

        def __init__(
            self,
            repo_id: str = "",
            root: str | Path | None = None,
            episodes: list[int] | None = None,
            return_uint8: bool = False,
            **kwargs,
        ) -> None:
            self.repo_id = repo_id
            self.root = root
            self.episodes = episodes
            self.return_uint8 = return_uint8
            self.meta = metadata
            self._frames = [
                frame for frame in frames if episodes is None or int(frame["episode_index"][0]) in episodes
            ]
            FakeLeRobotDataset.created.append(self)

        @property
        def fps(self) -> float:
            return metadata.fps

        def __len__(self) -> int:
            return len(self._frames)

        def __getitem__(self, index: int) -> dict:
            return {key: np.array(value, copy=True) for key, value in self._frames[index].items()}

    return FakeLeRobotDataset


def _install(monkeypatch, dataset_class) -> None:
    """Make ``from lerobot.datasets import LeRobotDataset`` resolve to the fake."""
    module = types.ModuleType("lerobot.datasets")
    module.LeRobotDataset = dataset_class
    monkeypatch.setitem(sys.modules, "lerobot.datasets", module)


def _open(monkeypatch, features: dict, frames: list[dict], **kwargs):
    """Open a fake dataset through ``open_demos``; defaults are a two-episode demonstration set."""
    metadata = _Meta(features)
    dataset_class = _fake_dataset_class(metadata, frames)
    _install(monkeypatch, dataset_class)
    kwargs.setdefault("chunk_size", 4)
    kwargs.setdefault("n_action_steps", 2)
    demos = open_demos(Path("/datasets/teleop"), CAMERAS, **kwargs)
    return demos, dataset_class


# ---------------------------------------------------------------- discovery


def test_discovery_maps_the_station_roles_from_bare_station_names():
    features = discover(_Meta(_features()), CAMERAS)

    assert features.arm == "follower"
    assert features.joint_position == "follower/joint_position"
    assert features.gripper == "follower/gripper_position"
    assert features.joint_velocity == "action/follower/joint_velocity"
    assert features.gripper_action == "action/follower/gripper_position"
    assert features.action == (
        "action/follower/joint_velocity",
        "action/follower/gripper_position",
    )
    assert features.cameras == CAMERAS


def test_discovery_accepts_lerobot_style_camera_feature_names():
    features = discover(_Meta(_features(image_style="lerobot")), CAMERAS)

    assert features.cameras == {
        "exterior": "observation.images.scene_zed",
        "wrist": "observation.images.wrist_zed",
    }


def test_discovery_matches_a_configured_full_name_to_a_bare_camera_feature():
    configured = {"exterior": "observation.images.scene_zed", "wrist": "scene_wrist"}

    features = discover(
        _Meta(_features(cameras=("scene_zed",))), {"exterior": "observation.images.scene_zed"}
    )

    assert features.cameras == {"exterior": "scene_zed"}
    with pytest.raises(ValueError, match="wrist"):
        discover(_Meta(_features(cameras=("scene_zed",))), configured)


def test_discovery_reads_the_actions_outside_the_arms_observed_velocity_namespace():
    """The station also streams ``follower/joint_velocity``; only the action prefix's is the command."""
    with_observed = discover(_Meta(_features(observed_velocity=True)), CAMERAS)
    without = discover(_Meta(_features(observed_velocity=False)), CAMERAS)

    assert with_observed.joint_velocity == without.joint_velocity == "action/follower/joint_velocity"


def test_discovery_accepts_a_single_eight_wide_action_column():
    """LeRobot's other convention: one ``action`` column, dims 0-6 joints and dim 7 the gripper."""
    features = discover(_Meta(_features(single_action=True)), CAMERAS)

    assert features.action == ("action",)
    assert features.joint_velocity == features.gripper_action == "action"
    assert features.arm == "follower" and features.gripper == "follower/gripper_position"
    assert features.cameras == CAMERAS


def test_discovery_prefers_the_separated_form_when_both_are_recorded():
    features = _features()
    features["action"] = {"dtype": "float32", "shape": (ACTION_DIMS,)}

    discovered = discover(_Meta(features), CAMERAS)

    assert discovered.joint_velocity == "action/follower/joint_velocity"
    assert discovered.gripper_action == "action/follower/gripper_position"
    assert discovered.action == (
        "action/follower/joint_velocity",
        "action/follower/gripper_position",
    )


def test_discovery_refuses_a_seven_wide_action_column():
    features = _features(single_action=True)
    features["action"]["shape"] = (ARM_DIMS,)

    with pytest.raises(ValueError, match="gripper") as error:
        discover(_Meta(features), CAMERAS)
    assert "'action'" in str(error.value)


def test_discovery_refuses_a_non_float_action_column():
    features = _features(single_action=True)
    features["action"]["dtype"] = "int64"

    with pytest.raises(ValueError, match="float"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_an_action_column_of_the_wrong_shape():
    features = _features(single_action=True)
    features["action"]["shape"] = (ACTION_DIMS + 1,)

    with pytest.raises(ValueError, match="shape"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_dataset_without_a_joint_position_feature():
    features = _features()
    del features["follower/joint_position"]

    with pytest.raises(ValueError, match="joint_position"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_two_arms():
    features = _features(extra={"other/joint_position": {"dtype": "float32", "shape": (ARM_DIMS,)}})

    with pytest.raises(ValueError, match="arms"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_missing_gripper_observation():
    features = _features()
    del features["follower/gripper_position"]

    with pytest.raises(ValueError, match="gripper_position"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_camera_role_that_matches_no_feed():
    with pytest.raises(ValueError, match="exterior"):
        discover(_Meta(_features()), {"exterior": "front_zed", "wrist": "wrist_zed"})


def test_discovery_refuses_a_camera_role_that_matches_two_feeds():
    features = _features(extra={"observation.images.scene_zed": {"dtype": "video", "shape": (*IMAGE_HW, 3)}})

    with pytest.raises(ValueError, match="matches 2"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_an_unknown_action_feature_in_the_action_namespace():
    features = _features(extra={"action/follower/joint_torque": {"dtype": "float32", "shape": (ARM_DIMS,)}})

    with pytest.raises(ValueError, match="joint_torque"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_dataset_without_a_recorded_joint_velocity():
    features = _features()
    del features["action/follower/joint_velocity"]

    with pytest.raises(ValueError, match="joint_velocity"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_missing_action_gripper():
    features = _features()
    del features["action/follower/gripper_position"]

    with pytest.raises(ValueError, match="gripper_position"):
        discover(_Meta(features), CAMERAS)


def test_discovery_refuses_a_wrong_arm_or_action_width():
    with pytest.raises(ValueError, match="joint_position"):
        discover(_Meta(_features(joint_shape=(ARM_DIMS - 1,))), CAMERAS)
    features = _features()
    features["action/follower/joint_velocity"]["shape"] = (ARM_DIMS - 1,)
    with pytest.raises(ValueError, match="joint_velocity"):
        discover(_Meta(features), CAMERAS)
    features = _features()
    del features["follower/joint_position"]["shape"]
    with pytest.raises(ValueError, match="declares no shape"):
        discover(_Meta(features), CAMERAS)


def test_discovery_accepts_a_configuration_with_no_cameras():
    assert discover(_Meta(_features()), {}).cameras == {}


def test_discovery_refuses_an_unusable_frame_rate():
    with pytest.raises(ValueError, match="fps"):
        discover(_Meta(_features(), fps=0.0), CAMERAS)


def test_the_recorded_action_order_is_kept_while_the_chunk_uses_the_checkpoint_layout(monkeypatch):
    """``action`` records the dataset's order; the chunk is dims 0-6 joints, 7 the gripper."""
    features = _features()
    joint_velocity = features.pop("action/follower/joint_velocity")
    gripper_action = features.pop("action/follower/gripper_position")
    features["action/follower/gripper_position"] = gripper_action  # recorded before joint_velocity
    features["action/follower/joint_velocity"] = joint_velocity

    assert discover(_Meta(features), CAMERAS).action == (
        "action/follower/gripper_position",
        "action/follower/joint_velocity",
    )
    demos, _ = _open(monkeypatch, features, _dataset_frames(episode_lengths=(6,)), chunk_size=2)
    np.testing.assert_array_equal(demos.chunk(1)[0, :ARM_DIMS], _frame_action(2)[:ARM_DIMS])
    assert demos.chunk(1)[0, ARM_DIMS] == _frame_action(2)[ARM_DIMS]


# ---------------------------------------------------------------- sampling and observation


def test_one_sample_per_chunk_boundary_with_the_chunk_clamped_at_the_episode_end(monkeypatch):
    demos, _ = _open(monkeypatch, _features(), _dataset_frames(episode_lengths=(10,)), chunk_size=4)

    # boundaries at frames 0, 2, 4, 6, 8 with n_action_steps=2; each chunk is the next four actions,
    # clamped at the episode's end (frame 9 repeats) when fewer than four remain
    assert len(demos) == 5
    assert demos.fps == FPS
    for sample, start in enumerate((0, 2, 4, 6, 8)):
        chunk = demos.chunk(sample)
        assert chunk.shape == (4, ACTION_DIMS)
        assert chunk.dtype == np.float64
        indices = [min(start + offset, 9) for offset in range(4)]
        np.testing.assert_array_equal(chunk, np.stack([_frame_action(index) for index in indices]))


def test_the_single_action_column_is_read_in_the_checkpoint_layout(monkeypatch):
    frames = _dataset_frames(episode_lengths=(6,), single_action=True)
    demos, _ = _open(monkeypatch, _features(single_action=True), frames, chunk_size=2, n_action_steps=1)

    assert len(demos) == 6
    for sample in range(len(demos)):
        indices = [min(sample + offset, 5) for offset in range(2)]
        np.testing.assert_array_equal(
            demos.chunk(sample), np.stack([_frame_action(index) for index in indices])
        )
    np.testing.assert_array_equal(demos.chunk(0)[0, :ARM_DIMS], _frame_action(0)[:ARM_DIMS])
    assert demos.chunk(0)[0, ARM_DIMS] == _frame_action(0)[ARM_DIMS]
    np.testing.assert_array_equal(
        demos.observation(0)["follower/joint_position"], frames[0]["follower/joint_position"]
    )


def test_the_separated_form_is_read_when_a_redundant_action_column_is_also_declared(monkeypatch):
    """The frames carry only the separated keys, so a loader that preferred ``action`` would fail."""
    features = _features()
    features["action"] = {"dtype": "float32", "shape": (ACTION_DIMS,)}
    demos, _ = _open(monkeypatch, features, _dataset_frames(episode_lengths=(4,)), chunk_size=2)

    np.testing.assert_array_equal(demos.chunk(0), np.stack([_frame_action(0), _frame_action(1)]))


def test_observation_returns_the_raw_station_dict_with_full_uint8_frames(monkeypatch):
    frames = _dataset_frames(episode_lengths=(4,))
    demos, _ = _open(monkeypatch, _features(), frames, chunk_size=2, n_action_steps=1)

    observation = demos.observation(1)

    assert set(observation) == {
        "follower/joint_position",
        "follower/gripper_position",
        "scene_zed",
        "wrist_zed",
    }
    assert demos.episode(1) == 0
    np.testing.assert_array_equal(
        observation["follower/joint_position"], frames[1]["follower/joint_position"]
    )
    np.testing.assert_array_equal(
        observation["follower/gripper_position"], frames[1]["follower/gripper_position"]
    )
    for name in ("scene_zed", "wrist_zed"):
        assert observation[name].shape == (*IMAGE_HW, 3)
        assert observation[name].dtype == np.uint8
        np.testing.assert_array_equal(observation[name], frames[1][name])


def test_channel_first_float_frames_become_the_stations_uint8_hwc(monkeypatch):
    """LeRobot hands ``image`` features back as float CHW; the station dict wants raw HWC uint8."""
    frames = _dataset_frames(episode_lengths=(3,))
    originals = []
    for frame in frames:
        for name in ("scene_zed", "wrist_zed"):
            originals.append(frame[name].copy())
            frame[name] = np.moveaxis(frame[name].astype(np.float32) / 255.0, -1, 0)
    demos, _ = _open(monkeypatch, _features(), frames, chunk_size=2, n_action_steps=1)

    frame = 2
    for name, expected in zip(("scene_zed", "wrist_zed"), originals[2 * 2 : 2 * 2 + 2], strict=True):
        np.testing.assert_array_equal(demos.observation(frame)[name], expected)


def test_the_episode_filter_selects_whole_episodes(monkeypatch):
    demos, dataset_class = _open(
        monkeypatch, _features(), _dataset_frames(episode_lengths=(4, 6, 5)), episodes=[1], chunk_size=2
    )

    assert dataset_class.created[-1].episodes == [1]
    assert len(demos) == 3  # episode 1's six frames, boundaries at 0, 2, 4
    assert {demos.episode(index) for index in range(len(demos))} == {1}


def test_a_short_episode_is_refused_by_name(monkeypatch):
    with pytest.raises(ValueError, match="episode 0"):
        _open(monkeypatch, _features(), _dataset_frames(episode_lengths=(2, 6)), chunk_size=4)


def test_a_frame_without_an_episode_index_is_refused(monkeypatch):
    frames = _dataset_frames(episode_lengths=(4,))
    del frames[2]["episode_index"]

    with pytest.raises(ValueError, match="episode_index"):
        _open(monkeypatch, _features(), frames, chunk_size=2)


def test_an_empty_dataset_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="no frames"):
        _open(monkeypatch, _features(), [], chunk_size=2)


def test_open_demos_passes_the_dataset_location_through(monkeypatch, tmp_path):
    metadata = _Meta(_features())
    dataset_class = _fake_dataset_class(metadata, _dataset_frames(episode_lengths=(4,)))
    _install(monkeypatch, dataset_class)

    open_demos(Path("teleop"), CAMERAS, chunk_size=2, n_action_steps=2, root=tmp_path)

    created = dataset_class.created[-1]
    assert created.repo_id == "teleop" and created.root == tmp_path
    assert created.return_uint8 is True


def test_open_demos_names_the_dataset_extra_when_lerobot_datasets_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.datasets", None)

    with pytest.raises(ImportError) as error:
        open_demos(Path("/datasets/teleop"), CAMERAS, chunk_size=2, n_action_steps=2)

    assert "`dataset` extra" in str(error.value)


# ---------------------------------------------------------------- the split


def test_split_holds_out_whole_episodes(monkeypatch):
    # one sample per episode: three episodes give three samples, two of which are held out
    demos, _ = _open(
        monkeypatch, _features(), _dataset_frames(episode_lengths=(5, 5, 5)), chunk_size=2, n_action_steps=5
    )
    assert len(demos) == 3

    train, val = split(demos, val_frac=0.5, seed=0)

    assert isinstance(train, np.ndarray) and isinstance(val, np.ndarray)
    assert sorted(train.tolist() + val.tolist()) == [0, 1, 2]
    assert len(val) == 2
    val_episodes = {demos.episode(index) for index in val.tolist()}
    train_episodes = {demos.episode(index) for index in train.tolist()}
    assert len(val_episodes) == 2 and not val_episodes & train_episodes
    again = split(demos, val_frac=0.5, seed=0)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(val, again[1])


def test_split_falls_back_to_a_seeded_row_split_for_one_episode(monkeypatch):
    demos, _ = _open(monkeypatch, _features(), _dataset_frames(episode_lengths=(12,)), chunk_size=2)
    assert len(demos) == 6

    train, val = split(demos, val_frac=0.5, seed=0)

    assert len(train) == 3 and len(val) == 3
    assert sorted(train.tolist() + val.tolist()) == list(range(6))
    again = split(demos, val_frac=0.5, seed=0)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(val, again[1])


def test_split_refuses_a_set_that_cannot_hold_anything_out(monkeypatch):
    demos, _ = _open(
        monkeypatch, _features(), _dataset_frames(episode_lengths=(2,)), chunk_size=2, n_action_steps=2
    )

    with pytest.raises(ValueError, match="at least two"):
        split(demos, val_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="val_frac"):
        split(demos, val_frac=0.0, seed=0)
    with pytest.raises(ValueError, match="val_frac"):
        split(demos, val_frac=1.5, seed=0)
