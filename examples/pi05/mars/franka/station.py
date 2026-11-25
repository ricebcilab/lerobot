"""The c3po station port: the real station, the fake, and the two dictionaries between them.

DESIGN.md freezes this module's interface. A ``Station`` is a dict-in, dict-out MDP port:
``reset()`` and ``step(action8)`` hand back the raw c3po observation, and this module says what the
policy reads from it (the 32-wide state, the letterboxed camera images as [0,1] float) and what the
station receives back (the two action keys). ``RecordingStation`` is the port ``run_recording``
needs: the same station plus c3po's ``record``/``download``. c3po is imported inside ``C3poStation`` only, so the
fake, ``--dry-run`` and every test here run with no c3po — and no torch — installed. The adapters
are the previous project's ``examples-from-unrelated-project/droid.py`` (openpi's own DROID
client); the letterbox's provenance is on ``resize_with_pad``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from bridge import ARM_DIMS
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from c3po._recording import Recording  # c3po is optional: typing only, never imported at runtime

IMAGE_SIZE = 224
"""The checkpoint's image size: pi05_droid's example input is (224, 224, 3)."""

STATE_DIMS = 32
"""PI05 declares ``observation.state`` (32,). The model pads actions itself but not state, so the
station sends 32 wide: the arm and gripper in dims 0-7, zeros beyond (DESIGN.md)."""


# ---------------------------------------------------------------- the manifest


@dataclass(frozen=True)
class StationManifest:
    """What a station is: the arms, camera feeds and action keys it exposes, and its control rate."""

    arms: tuple[str, ...]
    cameras: tuple[str, ...]
    action_keys: tuple[str, ...]
    control_rate: float


class _ManifestSource(Protocol):
    """The four c3po ``Robot`` properties a manifest is built from (c3po itself is optional)."""

    arms: list[str]
    cameras: list[str]
    action_keys: list[str]
    control_rate: float


def _manifest_from_robot(robot: _ManifestSource) -> StationManifest:
    """The manifest c3po's ``describe`` advertises, without asking c3po for permission at import."""
    return StationManifest(
        arms=tuple(robot.arms),
        cameras=tuple(robot.cameras),
        action_keys=tuple(robot.action_keys),
        control_rate=float(robot.control_rate),
    )


def _manifest_from_observations(obs: Mapping[str, NDArray], rate: float) -> StationManifest:
    """The manifest a canned observation implies: arm prefixes, the other keys as cameras.

    Action keys are ``{arm}/joint_velocity`` then ``{arm}/gripper_position`` per arm, in the
    observation's own order: ``action_dict`` writes exactly this pair.
    """
    arms = tuple(key.removesuffix("/joint_position") for key in obs if key.endswith("/joint_position"))
    return StationManifest(
        arms=arms,
        cameras=tuple(key for key in obs if not _is_arm_state_key(key, arms)),
        action_keys=tuple(f"{arm}/{kind}" for arm in arms for kind in ("joint_velocity", "gripper_position")),
        control_rate=float(rate),
    )


def _is_arm_state_key(key: str, arms: tuple[str, ...]) -> bool:
    """True for a per-arm observation key, so the manifest's cameras are the remaining keys."""
    arm, _, kind = key.rpartition("/")
    return arm in arms and kind in {"joint_position", "joint_velocity", "gripper_position"}


# ---------------------------------------------------------------- the port and its implementations


class Station(Protocol):
    """The port the loop drives: c3po owns the clock, one ``step`` per control step.

    ``arm`` is the single arm this example drives (a station may expose none or two); ``cancel_episode``
    is c3po's recording-loop contract: after the ``r`` flag the researcher discards the in-progress
    episode before re-recording it. A station that records nothing implements the cancel as a no-op;
    ``C3poStation`` forwards it to ``robot.cancel_episode()``.
    """

    @property
    def manifest(self) -> StationManifest: ...

    @property
    def arm(self) -> str: ...

    def reset(self) -> dict[str, NDArray]: ...
    def step(self, action8: NDArray) -> dict[str, NDArray]: ...
    def close(self) -> None: ...
    def cancel_episode(self) -> None: ...


@runtime_checkable
class RecordingStation(Station, Protocol):
    """A station that can record a dataset: c3po's recording contract, ``Loop.run_recording``'s port.

    ``record`` is c3po's ``robot.recording(name)`` context manager and ``download`` its
    ``robot.download_dataset`` (call it after the recording context and before ``close``).
    ``C3poStation`` implements both; ``FakeStation`` neither, because it records nothing — so the loop
    can refuse a station that is not one instead of hoping for the attributes.
    """

    def record(self, name: str) -> Recording: ...

    def download(self, name: str, dest: Path) -> Path: ...


class C3poStation:
    """The real station through c3po. c3po is imported here, not at module level."""

    def __init__(self, host: str, port: int, rate: float) -> None:
        try:
            from c3po import Robot
        except ImportError as error:  # the example's one optional hardware dependency
            raise ImportError(
                "c3po is not installed; from the lerobot checkout run `uv pip install -e ../c3po`"
            ) from error

        self._robot = Robot(host=host, port=port, rate=rate)
        self._manifest = _manifest_from_robot(self._robot)

    @property
    def manifest(self) -> StationManifest:
        return self._manifest

    @property
    def arm(self) -> str:
        """The single arm this example drives; a c3po station may have none or two."""
        arms = self._manifest.arms
        if len(arms) != 1:
            raise ValueError(f"the station has {len(arms)} arms {arms}; this example drives exactly one")
        return arms[0]

    def reset(self) -> dict[str, NDArray]:
        return self._robot.reset()

    def step(self, action8: NDArray) -> dict[str, NDArray]:
        return self._robot.step(action_dict(action8, self.arm))

    def close(self) -> None:
        self._robot.close()

    def cancel_episode(self) -> None:
        """c3po's ``robot.cancel_episode()``: discard the episode in progress on re-record."""
        self._robot.cancel_episode()

    def record(self, name: str) -> Recording:
        """c3po's ``robot.recording(name)``: the keyboard-controlled recording context manager."""
        return self._robot.recording(name)

    def download(self, name: str, dest: Path) -> Path:
        """c3po's ``robot.download_dataset``: call after the recording context exits and before
        ``close()``, because the dataset's URL is published by r2d2's ``dataset_ready``
        notification, which only arrives while the connection is open."""
        return self._robot.download_dataset(name, dest)


class FakeStation:
    """A deterministic station for ``--dry-run`` and the loop tests: canned observations, in order.

    ``step`` returns the next canned observation and wraps, so a loop runs to whatever step count
    the caller asks for. It records the ``(8,)`` actions it was sent; a ``None`` action is a
    read-only step and is not recorded (c3po's ``step(action=None)``).
    """

    def __init__(self, observations: Sequence[Mapping[str, NDArray]], rate: float) -> None:
        if not observations:
            raise ValueError("a FakeStation needs at least one observation")
        self._observations = [_copy_observation(observation) for observation in observations]
        self._manifest = _manifest_from_observations(observations[0], rate)
        self._next = 1 % len(self._observations)
        self._actions: list[NDArray] = []

    @property
    def manifest(self) -> StationManifest:
        return self._manifest

    @property
    def actions(self) -> list[NDArray]:
        """The action8s sent, in order — the policy's numbers, not the c3po action dicts."""
        return list(self._actions)

    @property
    def arm(self) -> str:
        """The single arm this example drives; a fake station may have none or two."""
        arms = self._manifest.arms
        if len(arms) != 1:
            raise ValueError(f"the station has {len(arms)} arms {arms}; this example drives exactly one")
        return arms[0]

    def reset(self) -> dict[str, NDArray]:
        self._next = 1 % len(self._observations)
        return _copy_observation(self._observations[0])

    def step(self, action8: NDArray | None) -> dict[str, NDArray]:
        if action8 is not None:
            self._actions.append(np.array(_action_vector(action8), copy=True))
        observation = self._observations[self._next]
        self._next = (self._next + 1) % len(self._observations)
        return _copy_observation(observation)

    def close(self) -> None:
        """Nothing to release: a fake station is a list."""

    def cancel_episode(self) -> None:
        """Nothing to discard: a fake station records no dataset (the recording loop's no-op)."""


# ---------------------------------------------------------------- the two dictionaries


def action_dict(action8: NDArray, arm: str) -> dict[str, NDArray]:
    """One ``(8,)`` policy action as c3po's two action keys for ``arm``.

    Joint velocities are clipped to [-1, 1] and the gripper is binarized at 0.5, as openpi's own
    DROID client does. The gripper is a position: a half-closed command means nothing to a
    Robotiq, so the station turns the policy's continuous dim into open (0.0) or closed (1.0).
    """
    action = _action_vector(action8)
    return {
        f"{arm}/joint_velocity": np.clip(action[:ARM_DIMS], -1.0, 1.0),
        f"{arm}/gripper_position": np.where(action[ARM_DIMS:] > 0.5, 1.0, 0.0),
    }


def observation_batch(
    obs: Mapping[str, NDArray], *, arm: str, cameras: Mapping[str, str | None], task: str
) -> dict[str, NDArray | str]:
    """One c3po observation as the policy's raw batch, before any processor runs.

    Images are float32 (3, 224, 224) in [0, 1] because ``PI05Policy`` does ``img*2-1`` for SigLIP:
    a uint8 frame would be multiplied as 0-255 and destroyed. The third image is omitted unless
    ``cameras["third"]`` names a feed: ``PI05Policy.prepare_images`` fills an absent image feature
    with a -1 image and a zero mask, whereas a zeros image would be an unmasked real input.
    """
    batch: dict[str, NDArray | str] = {
        "observation.images.base_0_rgb": _camera_image(obs, _camera_feed(cameras, "exterior")),
        "observation.images.left_wrist_0_rgb": _camera_image(obs, _camera_feed(cameras, "wrist")),
        "observation.state": _state(obs, arm),
        "task": task,
    }
    third = cameras.get("third")
    if third is not None:
        batch["observation.images.right_wrist_0_rgb"] = _camera_image(obs, third)
    return batch


def resize_with_pad(image: NDArray, size: int = IMAGE_SIZE) -> NDArray:
    """Letterbox one ``(H, W, 3)`` uint8 frame into ``(size, size, 3)``, stretching nothing.

    Copies ``openpi_client.image_tools.resize_with_pad`` via the previous project's
    ``examples-from-unrelated-project/droid.py``, the preparation pi0.5-DROID was trained on: one
    scale ratio for both axes, so nothing is stretched, and the remainder is black padding. PIL
    sizes are ``(width, height)`` where the array is ``(height, width, channels)``; a frame
    already at ``size`` is returned as is, which is what the reference does.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 frame, got shape {image.shape} dtype {image.dtype}")
    if image.shape[:2] == (size, size):
        return image
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / size, cur_height / size)
    resized = Image.fromarray(image).resize(
        (int(cur_width / ratio), int(cur_height / ratio)), resample=Image.BILINEAR
    )
    padded = Image.new(resized.mode, (size, size), 0)
    padded.paste(resized, (int((size - resized.width) / 2), int((size - resized.height) / 2)))
    return np.asarray(padded)


# ---------------------------------------------------------------- validated pieces


def _camera_feed(cameras: Mapping[str, str | None], role: str) -> str:
    """The feed name ``cameras[role]`` names, refusing a role that names nothing."""
    feed = cameras.get(role)
    if not feed:
        raise ValueError(f"cameras[{role!r}] names no feed; the {role} image cannot be built")
    return feed


def _camera_image(obs: Mapping[str, NDArray], feed: str) -> NDArray:
    """One c3po camera feed as the batch image: float32 (3, 224, 224) in [0, 1]."""
    if feed not in obs:
        raise ValueError(f"the observation has no camera feed {feed!r} (keys: {sorted(obs)})")
    frame = np.asarray(obs[feed])
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"camera feed {feed!r}: expected an (H, W, 3) uint8 frame, "
            f"got shape {frame.shape} dtype {frame.dtype}"
        )
    image = resize_with_pad(frame).astype(np.float32) / 255.0
    return np.moveaxis(image, -1, 0)


def _state(obs: Mapping[str, NDArray], arm: str) -> NDArray:
    """The policy's state: ``[joint_position (7,), gripper_position (1,)]`` then zeros to 32."""
    joint = _obs_vector(obs, f"{arm}/joint_position", ARM_DIMS)
    gripper = _obs_vector(obs, f"{arm}/gripper_position", 1)
    state = np.zeros(STATE_DIMS, dtype=np.float32)
    state[: ARM_DIMS + 1] = np.concatenate([joint, gripper])
    return state


def _obs_vector(obs: Mapping[str, NDArray], key: str, size: int) -> NDArray:
    """One observation vector as float64 (size,), validated by name: a wrong shape is a wild arm."""
    if key not in obs:
        raise ValueError(f"the observation has no {key!r} (keys: {sorted(obs)})")
    value = np.asarray(obs[key], dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"{key!r}: expected shape ({size},), got {value.shape}")
    return value


def _action_vector(action8: NDArray) -> NDArray:
    """The policy's ``(8,)`` action, validated: any other width is a wild arm, not a rounding error."""
    action = np.asarray(action8)
    if action.shape != (ARM_DIMS + 1,):
        raise ValueError(f"action8 must be ({ARM_DIMS + 1},), got shape {action.shape}")
    return action


def _copy_observation(obs: Mapping[str, NDArray]) -> dict[str, NDArray]:
    """A copy, so a caller that mutates the observation cannot mutate the fake's canned sequence."""
    return {key: np.array(value, copy=True) for key, value in obs.items()}
