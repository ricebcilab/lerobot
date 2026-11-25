"""The demonstrations a ``teleop --record`` session produces, as training samples.

``teleop --record NAME`` (Phase A) leaves a LeRobot v3.0 dataset: the station's camera frames, the
arm's joint positions and the actions that were sent -- the seven joint velocities in action units
and the gripper position. Phase C's task module is trained on it (DESIGN.md, "Phase C"). This module
turns that dataset into samples: one per chunk boundary of every episode, each an observation the
policy's raw batch starts from plus the demonstrated action chunk.

**The feature names are discovered, not assumed.** The dataset is r2d2's, and its schema is unknown
offline, so :func:`discover` reads ``meta.features`` and maps what it finds onto the station's roles,
refusing anything it cannot map by name:

- the arm is the prefix of the one ``<arm>/joint_position`` observation feature, and its gripper
  observation is ``<arm>/gripper_position``;
- the recorded actions are one ``<prefix>/joint_velocity`` feature and its sibling
  ``<prefix>/gripper_position``, under a prefix other than the arm's (the observation namespace), so
  the measured joint velocity the station also streams cannot be mistaken for the command;
- a camera feed is an image/video feature (``dtype`` ``image``/``video``, or a ``uint8`` HWC frame),
  matched to a configured ``[cameras]`` role by full name or last path component, so ``scene_zed``
  and ``observation.images.scene_zed`` both name the exterior camera.

A refusal names the feature or role that is missing or ambiguous rather than silently mapping it.

**Units and layout are the recording's.** ``Demos.chunk`` holds the recorded action values verbatim,
in the checkpoint's 8-wide layout (dims 0-6 the joint velocities, dim 7 the gripper), and never
normalizes them: the trainer applies the checkpoint's statistics. ``Demos.observation`` returns the
raw station observation dict -- full ``(H, W, C)`` uint8 frames, not letterboxed or scaled, which is
why the trainer feeds it to ``station.observation_batch`` and then its own pre-processors; doing that
here would hide the station's one letterbox behind a second one.

**Sampling.** The control loop asks the policy for a chunk once every ``n_action_steps`` ticks
(``loop.Loop._tick``), so a sample starts at each of those boundaries and its action chunk is the
next ``chunk_size`` recorded actions, clamped at the episode's end by repeating the last one. An
episode shorter than ``chunk_size`` has no whole chunk to demonstrate and is refused by episode.

``open_demos`` is the only place that imports ``lerobot.datasets`` (lazily, inside the function):
the ``dataset`` extra is optional, so this module and its tests stay numpy-only. :func:`split` is
``train_user.split_states``' rule for samples: whole episodes are held out when there are several,
else a seeded row permutation; it is reimplemented here rather than imported because
``train_user`` pulls in torch and the trainer, and this loader must stay importable on a dry run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from bridge import ARM_DIMS
from numpy.typing import NDArray

ACTION_DIMS = ARM_DIMS + 1
"""The checkpoint's action width: the seven joint velocities then the gripper (DESIGN.md)."""


class DatasetMetadata(Protocol):
    """What ``discover`` reads: a ``LeRobotDatasetMetadata`` or anything shaped like one."""

    @property
    def features(self) -> Mapping[str, Mapping[str, Any]]: ...

    @property
    def fps(self) -> float: ...


@dataclass(frozen=True)
class Features:
    """The discovered dataset features: which one plays which of the station's roles.

    ``cameras`` maps a configured ``[cameras]`` role (``exterior``, ``wrist``, ``third``) to the
    dataset feature holding that feed, so it is exactly the mapping ``station.observation_batch``
    needs. ``action`` is the recorded action features in the dataset's own order; the chunk is built
    in the checkpoint's layout (see the module docstring), so the order is provenance, not layout.
    """

    arm: str
    joint_position: str
    gripper: str
    cameras: dict[str, str]
    joint_velocity: str
    gripper_action: str
    action: tuple[str, ...]


@dataclass(frozen=True)
class _Sample:
    """One demonstration sample: a chunk-boundary observation and the action chunk that followed."""

    episode: int
    observation: Mapping[str, NDArray]
    chunk: NDArray


# ---------------------------------------------------------------- discovery


def discover(meta: DatasetMetadata, cameras: Mapping[str, str]) -> Features:
    """The dataset's features as the station's roles, or a ``ValueError`` naming what is missing.

    ``meta`` is a ``LeRobotDatasetMetadata`` (or anything with ``.features`` and ``.fps``); its
    ``features`` map every feature name to at least ``{"dtype": ..., "shape": ...}``. The rules are
    the module docstring's, and every refusal says which feature/role was expected and what the
    dataset has instead.
    """
    features = dict(meta.features)
    _positive_fps(meta)  # a dataset without a usable rate is not a dataset this loader can sample

    arm, joint_position = _single_arm(features)
    gripper = f"{arm}/gripper_position"
    _require_feature(features, gripper, "the arm's gripper observation")
    _require_vector(features, joint_position, ARM_DIMS, "the arm's joint positions")
    _require_vector(features, gripper, 1, "the arm's gripper observation")

    joint_velocity, gripper_action = _action_features(features, arm)
    _require_vector(features, joint_velocity, ARM_DIMS, "the recorded joint velocities")
    _require_vector(features, gripper_action, 1, "the recorded gripper commands")

    action = tuple(name for name in features if name in (joint_velocity, gripper_action))
    return Features(
        arm=arm,
        joint_position=joint_position,
        gripper=gripper,
        cameras=_camera_features(features, cameras),
        joint_velocity=joint_velocity,
        gripper_action=gripper_action,
        action=action,
    )


def _single_arm(features: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """The one ``<arm>/joint_position`` feature's arm and name: this example drives exactly one arm."""
    found: dict[str, str] = {}
    for name in features:
        split = _split_suffix(name, "joint_position")
        if split is not None:
            found[split[0]] = name
    if not found:
        raise ValueError(
            "the dataset has no arm observation: expected one '<arm>/joint_position' feature; "
            f"its features are {_names(features)}"
        )
    if len(found) > 1:
        raise ValueError(
            f"the dataset has {len(found)} arms {sorted(found)}; this example drives exactly one"
        )
    (arm, joint_position) = next(iter(found.items()))
    return arm, joint_position


def _action_features(features: Mapping[str, Mapping[str, Any]], arm: str) -> tuple[str, str]:
    """The recorded action's ``joint_velocity`` and its sibling ``gripper_position`` feature names.

    The command namespace is the prefix of the one ``*/joint_velocity`` feature that is not the arm's
    own (the station streams a measured ``<arm>/joint_velocity`` too, which is not an action). Any
    other feature under that prefix, or any other ``*/joint_velocity``/``*/gripper_position`` feature
    outside the arm's namespace, is refused: the actions are exactly those two.
    """
    candidates: dict[str, str] = {}
    for name in features:
        split = _split_suffix(name, "joint_velocity")
        if split is not None and split[0] != arm:
            candidates[split[0]] = name
    if not candidates:
        raise ValueError(
            "the dataset records no action joint_velocity: expected one '*/joint_velocity' feature "
            f"under a prefix other than the arm's {arm!r}; its features are {_names(features)}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"the dataset records {len(candidates)} action joint_velocity features "
            f"{sorted(candidates.values())}; expected exactly one action namespace"
        )
    (prefix, joint_velocity) = next(iter(candidates.items()))
    gripper_action = f"{prefix}/gripper_position"
    _require_feature(features, gripper_action, "the recorded gripper command (the sibling of joint_velocity)")
    for name in features:
        if _parent(name) == prefix and name not in (joint_velocity, gripper_action):
            raise ValueError(
                f"the recorded actions are exactly {joint_velocity!r} and {gripper_action!r}; "
                f"the dataset also records {name!r} under {prefix!r}, which this loader cannot map"
            )
    for name in features:
        split = _split_suffix(name, "gripper_position")
        if split is not None and split[0] not in (arm, prefix):
            raise ValueError(
                f"the recorded actions are exactly {joint_velocity!r} and {gripper_action!r}; "
                f"the dataset also records {name!r}, an action-like feature under another prefix"
            )
    return joint_velocity, gripper_action


def _camera_features(features: Mapping[str, Mapping[str, Any]], cameras: Mapping[str, str]) -> dict[str, str]:
    """Each configured role's dataset feature, refusing an unfound or ambiguous camera feed."""
    image_names = [name for name, feature in features.items() if _is_image(feature)]
    resolved: dict[str, str] = {}
    for role, wanted in cameras.items():
        matches = [name for name in image_names if _name_matches(name, wanted)]
        if not matches:
            raise ValueError(
                f"cameras[{role!r}] names {wanted!r}, but the dataset has no image or video feature "
                f"matching it (image features: {sorted(image_names)})"
            )
        if len(matches) > 1:
            raise ValueError(
                f"cameras[{role!r}] names {wanted!r}, which matches {len(matches)} features "
                f"{sorted(matches)}; a camera role maps to exactly one feature"
            )
        resolved[role] = matches[0]
    return resolved


# ---------------------------------------------------------------- the samples


class Demos:
    """Demonstration samples in memory, split-ready: one per chunk boundary of every episode.

    ``features`` is what :func:`discover` found and ``fps`` the recording's rate. A sample's
    observation is the raw station observation dict at its chunk boundary (full camera frames, the
    arm's joint positions and gripper), ready for ``station.observation_batch``; its chunk is the
    recorded ``(chunk_size, 8)`` action values in action units, unmodified.
    """

    def __init__(self, features: Features, fps: float, samples: Sequence[_Sample]) -> None:
        if not samples:
            raise ValueError("a Demos needs at least one sample; the dataset contributed none")
        self.features = features
        self.fps = float(fps)
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def observation(self, i: int) -> dict[str, NDArray]:
        """Sample ``i``'s chunk-boundary observation: the station dict ``observation_batch`` reads."""
        return dict(self._samples[i].observation)

    def chunk(self, i: int) -> NDArray:
        """Sample ``i``'s demonstrated actions, ``(chunk_size, 8)`` in action units, verbatim."""
        return np.array(self._samples[i].chunk, copy=True)

    def episode(self, i: int) -> int:
        """The dataset episode sample ``i`` came from (the recording's 0-based episode index)."""
        return self._samples[i].episode


def _load(
    dataset: Any,
    cameras: Mapping[str, str],
    *,
    chunk_size: int,
    n_action_steps: int,
) -> Demos:
    """Every chunk boundary of every selected episode as a sample: the one pass over the dataset.

    The dataset is read once, frame by frame: the frames at the boundaries (every ``n_action_steps``
    within an episode) keep their observation, and every frame contributes its action. The sample's
    chunk is assembled after the episode's actions are known, so it can be clamped at the episode's
    end; an episode shorter than ``chunk_size`` is refused by its index.
    """
    features = discover(dataset.meta, cameras)
    fps = _positive_fps(dataset)
    count = len(dataset)
    if not count:
        raise ValueError("the dataset has no frames; there are no demonstrations to load")
    actions: dict[int, list[NDArray]] = {}
    observations: dict[int, list[Mapping[str, NDArray]]] = {}
    for index in range(count):
        frame = dataset[index]
        episode = _episode_index(frame, index)
        if episode not in actions:
            actions[episode] = []
            observations[episode] = []
        position = len(actions[episode])  # the frame's 0-based position inside its episode
        actions[episode].append(_frame_action(frame, features))
        if position % n_action_steps == 0:
            observations[episode].append(_frame_observation(frame, features))
    samples = [
        sample
        for episode in actions
        for sample in _episode_samples(
            episode, actions[episode], observations[episode], chunk_size=chunk_size, stride=n_action_steps
        )
    ]
    return Demos(features, fps, samples)


def _episode_samples(
    episode: int,
    actions: Sequence[NDArray],
    observations: Sequence[Mapping[str, NDArray]],
    *,
    chunk_size: int,
    stride: int,
) -> list[_Sample]:
    """One sample per boundary of one episode, its chunk clamped at the episode's last action."""
    recorded = np.stack(actions)
    if recorded.shape[0] < chunk_size:
        raise ValueError(
            f"episode {episode} is {recorded.shape[0]} frames long, shorter than one "
            f"{chunk_size}-action chunk; there is no whole chunk to demonstrate"
        )
    samples = []
    for boundary, observation in enumerate(observations):
        start = boundary * stride
        chunk = recorded[start : start + chunk_size]
        missing = chunk_size - chunk.shape[0]
        if missing:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], missing, axis=0)])
        samples.append(_Sample(episode=episode, observation=dict(observation), chunk=chunk))
    return samples


# ---------------------------------------------------------------- opening a dataset


def open_demos(
    path: Path,
    cameras: Mapping[str, str],
    *,
    chunk_size: int,
    n_action_steps: int,
    episodes: Sequence[int] | None = None,
    root: Path | None = None,
) -> Demos:
    """Open a downloaded ``teleop --record`` dataset as samples; ``lerobot.datasets`` is imported here.

    ``path`` is the dataset directory the recording downloaded to (a Hub repo id is accepted when it
    does not exist locally), and ``root`` overrides the local directory it is read from.
    ``episodes`` selects episodes, as ``LeRobotDataset``'s own filter does. ``chunk_size`` is the
    demonstrated chunk's horizon and ``n_action_steps`` the control loop's chunk boundary spacing.

    The ``dataset`` extra is optional in this checkout, so the import is lazy and its absence is
    reported as an ``ImportError`` naming that extra rather than as a bare ``lerobot.datasets``
    failure.
    """
    _positive_int(chunk_size, "chunk_size")
    _positive_int(n_action_steps, "n_action_steps")
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise ImportError(
            "opening demonstrations needs LeRobotDataset from lerobot.datasets, which the `dataset` "
            f"extra provides (from the lerobot checkout: `uv sync --extra dataset`): {error}"
        ) from error
    repo_id, dataset_root = _dataset_location(Path(path), root)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=None if episodes is None else list(episodes),
        return_uint8=True,
    )
    return _load(dataset, cameras, chunk_size=chunk_size, n_action_steps=n_action_steps)


def _dataset_location(path: Path, root: Path | None) -> tuple[str, Path | None]:
    """The ``(repo_id, local root)`` ``LeRobotDataset`` is opened with: a downloaded set reads from disk."""
    if root is not None:
        return path.name or "dataset", Path(root)
    if path.is_dir():
        return path.name or "dataset", path
    return str(path), None


# ---------------------------------------------------------------- the split


def split(demos: Demos, *, val_frac: float, seed: int) -> tuple[NDArray, NDArray]:
    """(train, val) sample indices: whole episodes held out when there are several, else a row split.

    This is ``train_user.split_states``' rule (reimplemented here so this loader does not import the
    torch-side trainer): a row split would put one trajectory's neighbouring samples on both sides of
    the boundary, so with more than one episode whole episodes are held out -- the seeded permutation
    decides which, the last one always stays for training, and ``val`` may overshoot ``val_frac`` by
    up to one episode's sample count. A single-episode set has no groups to keep apart and its
    samples are split by a seeded permutation of the samples instead.
    """
    _fraction(val_frac, "val_frac")
    count = len(demos)
    if count < 2:
        raise ValueError(f"a split needs at least two samples to hold one out, got {count}")
    n_val = min(count - 1, max(1, int(round(val_frac * count))))
    episodes = np.array([demos.episode(index) for index in range(count)])
    groups = np.unique(episodes)
    generator = np.random.default_rng(seed)
    if len(groups) > 1:
        val: list[int] = []
        for episode in generator.permutation(groups)[:-1]:  # one episode stays for training
            val.extend(np.flatnonzero(episodes == episode).tolist())
            if len(val) >= n_val:
                break
        held = set(val)
        train = [index for index in range(count) if index not in held]
        return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)
    order = generator.permutation(count)
    return order[n_val:].astype(np.int64), order[:n_val].astype(np.int64)


# ---------------------------------------------------------------- reading a frame


def _frame_action(frame: Mapping[str, Any], features: Features) -> NDArray:
    """One frame's recorded action as ``(8,)`` in the checkpoint's layout: joints then gripper."""
    joints = _frame_vector(frame, features.joint_velocity, ARM_DIMS)
    gripper = _frame_vector(frame, features.gripper_action, 1)
    return np.concatenate([joints, gripper])


def _frame_observation(frame: Mapping[str, Any], features: Features) -> dict[str, NDArray]:
    """One frame's raw station observation dict: the arm state and each camera's full frame."""
    observation = {
        features.joint_position: _frame_vector(frame, features.joint_position, ARM_DIMS),
        features.gripper: _frame_vector(frame, features.gripper, 1),
    }
    for name in features.cameras.values():
        observation[name] = _image_hwc(frame, name)
    return observation


def _frame_vector(frame: Mapping[str, Any], name: str, width: int) -> NDArray:
    """One per-frame feature as finite float64 ``(width,)``, refused by name like every other mapping."""
    if name not in frame:
        raise ValueError(f"the dataset frame has no {name!r} (keys: {sorted(frame)})")
    value = np.asarray(frame[name], dtype=np.float64).reshape(-1)
    if value.shape != (width,):
        raise ValueError(f"{name!r}: expected shape ({width},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name!r}: expected finite numbers, got {value}")
    return value


def _image_hwc(frame: Mapping[str, Any], name: str) -> NDArray:
    """One camera frame as the station's ``(H, W, 3)`` uint8, whatever layout the dataset returned.

    ``LeRobotDataset`` hands video frames back channel-first (and image features as float [0, 1] after
    its PIL transform), while the station's own feeds are channel-last uint8; the raw station dict
    needs the latter. The last axis decides the layout: three channels means channel-last, a first
    axis of three means channel-first (a real frame is much wider than three pixels, so the two cannot
    be confused in practice).
    """
    if name not in frame:
        raise ValueError(f"the dataset frame has no {name!r} (keys: {sorted(frame)})")
    image = np.asarray(frame[name])
    if image.ndim != 3:
        raise ValueError(f"{name!r}: expected an image frame, got shape {image.shape}")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"{name!r}: expected three channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"{name!r}: expected a uint8 frame or a [0, 1] float frame, got {image.dtype}")
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name!r}: the frame has non-finite values")
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return image


def _episode_index(frame: Mapping[str, Any], index: int) -> int:
    """One frame's 0-based episode index (a LeRobot dataset's own default feature)."""
    if "episode_index" not in frame:
        raise ValueError(
            f"frame {index} has no 'episode_index'; a LeRobot dataset's frames name the episode they "
            "belong to, and the split and the chunk boundaries are grouped by it"
        )
    value = np.asarray(frame["episode_index"]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"frame {index}: episode_index must be a scalar, got shape {value.shape}")
    return int(value[0])


# ---------------------------------------------------------------- features and validation


def _split_suffix(name: str, suffix: str) -> tuple[str, str] | None:
    """``(prefix, "/")`` when ``name`` is ``<prefix>/<suffix>`` with a non-empty prefix, else None."""
    prefix, _, last = name.rpartition("/")
    if not prefix or last != suffix:
        return None
    return prefix, "/"


def _parent(name: str) -> str:
    """A feature name's parent path: ``<parent>/<leaf>`` -> ``<parent>``, or the empty string."""
    parent, separator, _ = name.rpartition("/")
    return parent if separator else ""


def _is_image(feature: Mapping[str, Any]) -> bool:
    """Whether a feature's dtype/shape marks a camera: a video/image, or a ``uint8`` HWC frame."""
    dtype = str(feature.get("dtype", ""))
    if dtype in ("image", "video"):
        return True
    shape = _shape(feature)
    return dtype in ("uint8", "uint16") and shape is not None and len(shape) == 3 and shape[-1] in (1, 3, 4)


def _name_matches(name: str, wanted: str) -> bool:
    """Whether a feature is the camera a role names: its full name or its last path component."""
    return name == wanted or _last_component(name) == _last_component(wanted)


def _last_component(name: str) -> str:
    """A feature name's last path component, whichever separator it uses (``images.scene_zed``)."""
    return name.replace("/", ".").rsplit(".", 1)[-1]


def _shape(feature: Mapping[str, Any]) -> tuple[int, ...] | None:
    """A feature's declared shape as a tuple of ints, whatever list/tuple the metadata stored."""
    shape = feature.get("shape")
    if shape is None:
        return None
    return tuple(int(value) for value in shape)


def _require_feature(features: Mapping[str, Mapping[str, Any]], name: str, what: str) -> None:
    """Refuse a missing feature, naming the role it would have played and the dataset's names."""
    if name not in features:
        raise ValueError(f"the dataset has no {name!r}, {what}; its features are {_names(features)}")


def _require_vector(features: Mapping[str, Mapping[str, Any]], name: str, width: int, what: str) -> None:
    """Refuse a mapped vector feature whose declared shape is not this example's arm/action width."""
    shape = _shape(features[name])
    if shape is None:
        raise ValueError(f"{name!r} ({what}): the feature declares no shape; expected ({width},)")
    if shape != (width,):
        raise ValueError(f"{name!r} ({what}): expected shape ({width},), got {shape}")


def _names(features: Mapping[str, Any]) -> str:
    """A feature mapping's names as one sorted list for an error message."""
    return ", ".join(sorted(features))


def _positive_fps(source: Any) -> float:
    """A usable frame rate from a metadata object or dataset, named in the refusal."""
    value = getattr(source, "fps", None)
    try:
        fps = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"fps: expected a positive number, got {value!r}") from error
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps: expected a positive number, got {value!r}")
    return fps


def _positive_int(value: Any, name: str) -> int:
    """A positive integer knob (a chunk horizon or a boundary stride), refused by name."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}: expected a positive integer, got {value!r}")
    return value


def _fraction(value: Any, name: str) -> float:
    """A ``(0, 1]`` fraction, the same rule ``[training] val_frac`` is parsed by."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name}: expected a number in (0, 1], got {value!r}")
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError(f"{name}: expected a number in (0, 1], got {value!r}")
    return fraction
