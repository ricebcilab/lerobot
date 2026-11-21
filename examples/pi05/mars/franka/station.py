"""The station port: c3po on the real arm, and the fake that the dry run and the tests drive.

``interfaces.RobotPort`` is the contract: a manifest from the handshake, a reset, a step, a close.
The real station is a thin forwarder around c3po's ``Robot``; ``C3poStation`` imports c3po inside its
constructor, so this module and every test import on a machine that has never installed it.

The station takes joint velocities, not Cartesian targets (``DESIGN.md``): ``action_dict`` splits the
policy's eight numbers into ``{arm}/joint_velocity`` and ``{arm}/gripper_position`` and does nothing
else. No clamping happens here on purpose: the clamps read the arm's ``ArmSpec`` in ``actions.py``,
and a second, silent clamp would hide exactly the policy behaviour an experiment is measuring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from interfaces import ACTION_DIMS, ARM_DIMS, GRIPPER_DIM, ArmSpec, StationManifest
from PIL import Image

IMAGE_SIZE = 224
"""pi05_droid's input size: openpi's DROID client resizes both cameras to 224x224 before the request."""

# The panda's joint ranges from mujoco_menagerie panda.xml. These are UNVERIFIED as this arm's
# limits: slice 0's probe reads the station's real ranges and replaces them. They exist so
# ``build_arm_spec`` can return a valid ``ArmSpec`` before the hardware is present, and so an
# impossible override is still rejected rather than passed through.
_PANDA_Q_MIN = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
_PANDA_Q_MAX = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)

# The station's gripper is a normalized position, 0 open and 1 closed: the DROID adapter binarizes a
# policy's gripper at 0.5 to those two ends for the Robotiq, so closed_at is the top of the range.
_GRIPPER_RANGE = (0.0, 1.0)

# The observation keys that name an arm, and the action keys ``step`` accepts for it.
_ARM_OBS_SUFFIXES = ("/joint_position", "/gripper_position")
_ARM_ACTION_SUFFIXES = ("/joint_velocity", "/gripper_position")


def _seven(value: object, name: str) -> tuple[float, ...]:
    """One per-joint quantity as seven floats; a scalar broadcasts to every joint."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (float(value),) * ARM_DIMS
    values = tuple(float(v) for v in value)
    if len(values) != ARM_DIMS:
        raise ValueError(f"{name} needs {ARM_DIMS} values, got {len(values)}")
    return values


def manifest_from_robot(robot: Any) -> StationManifest:
    """What c3po's ``Robot`` says it is, as the frozen manifest.

    The handshake properties are copied into tuples because c3po may hand back lists: the manifest
    is a value, and two connections to the same station should compare equal.
    """
    return StationManifest(
        arms=tuple(robot.arms),
        cameras=tuple(robot.cameras),
        action_keys=tuple(robot.action_keys),
        control_rate=float(robot.control_rate),
    )


def manifest_from_observations(
    observations: Sequence[Mapping[str, np.ndarray]], control_rate: float
) -> StationManifest:
    """Derive the manifest from observation keys, for the fake station and ``--dry-run``.

    The arms are the prefixes of the ``{arm}/joint_position`` keys, and every remaining observation
    key is a camera. The order is the observation's own, so the fake's manifest names things in the
    order the script's fixtures do.
    """
    observations = tuple(observations)
    if not observations:
        raise ValueError("a station needs at least one observation to describe itself")
    keys = tuple(observations[0])
    arms = tuple(
        dict.fromkeys(key.removesuffix("/joint_position") for key in keys if key.endswith("/joint_position"))
    )
    if not arms:
        raise ValueError("the observations name no arm: expected a {arm}/joint_position key")

    arm_obs_keys = {f"{arm}{suffix}" for arm in arms for suffix in _ARM_OBS_SUFFIXES}
    return StationManifest(
        arms=arms,
        cameras=tuple(key for key in keys if key not in arm_obs_keys),
        action_keys=tuple(f"{arm}{suffix}" for arm in arms for suffix in _ARM_ACTION_SUFFIXES),
        control_rate=float(control_rate),
    )


def action_dict(action: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    """Split one ``(8,)`` policy action into c3po's two action keys for ``arm``.

    Nothing is clipped or thresholded: the seven velocities go out as they arrive, and so does the
    gripper. ``actions.py`` owns the limits, so this function stays a pure shape check.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIMS,):
        raise ValueError(f"one action must be ({ACTION_DIMS},), got {action.shape}")
    return {
        f"{arm}/joint_velocity": action[:GRIPPER_DIM].copy(),
        f"{arm}/gripper_position": action[GRIPPER_DIM:].copy(),
    }


def state_from_obs(obs: Mapping[str, np.ndarray], arm: str) -> np.ndarray:
    """The policy's proprioception state ``(8,)``: seven joint positions, then the gripper.

    The checkpoint's own DROID client concatenates the two keys in this order, so the runner does
    the same rather than asking the policy to normalize two arrays separately.
    """
    joint_position = np.asarray(obs[f"{arm}/joint_position"], dtype=np.float64)
    if joint_position.shape != (ARM_DIMS,):
        raise ValueError(f"{arm}/joint_position must be ({ARM_DIMS},), got {joint_position.shape}")
    gripper_position = np.asarray(obs[f"{arm}/gripper_position"], dtype=np.float64)
    if gripper_position.shape != (1,):
        raise ValueError(f"{arm}/gripper_position must be (1,), got {gripper_position.shape}")
    return np.concatenate([joint_position, gripper_position])


def build_arm_spec(
    manifest: StationManifest,
    overrides: Mapping[str, Any],
    default_qdot_max: float = 0.5,
    *,
    joint_names: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
) -> ArmSpec:
    """The arm's limits: the config's overrides first, the station's names, then safe defaults.

    ``overrides`` may carry ``joint_names``, ``q_min``, ``q_max``, ``qdot_max``, ``gripper_range``
    and ``gripper_closed_at``. Joint names come from ``joint_names`` (c3po's
    ``robot.joint_names[arm]``, as a mapping or as the seven names themselves) or from a manifest
    that happens to expose them; a scalar ``qdot_max`` broadcasts to all seven joints. The panda
    limits are the unverified defaults named above, and the probe replaces them.
    """
    arm = manifest.arm()
    names: object = joint_names
    if names is None:
        names = getattr(manifest, "joint_names", None)
    if isinstance(names, Mapping):
        names = names.get(arm)
    if names is None:
        # A placeholder until the probe reads c3po's names: ArmSpec needs seven, and inventing
        # plausible-looking ones would be worse than a name that says it is provisional.
        names = tuple(f"{arm}/joint_{i + 1}" for i in range(ARM_DIMS))
    else:
        names = tuple(names)
        if len(names) != ARM_DIMS:
            raise ValueError(f"{arm} needs {ARM_DIMS} joint names, got {len(names)}")

    gripper_range = tuple(float(v) for v in overrides.get("gripper_range", _GRIPPER_RANGE))
    if len(gripper_range) != 2:
        raise ValueError(f"gripper_range needs two values, got {len(gripper_range)}")

    return ArmSpec(
        joint_names=names,
        q_min=_seven(overrides.get("q_min", _PANDA_Q_MIN), "q_min"),
        q_max=_seven(overrides.get("q_max", _PANDA_Q_MAX), "q_max"),
        qdot_max=_seven(overrides.get("qdot_max", default_qdot_max), "qdot_max"),
        gripper_range=gripper_range,
        gripper_closed_at=float(overrides.get("gripper_closed_at", gripper_range[1])),
    )


def resize_with_pad(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Letterbox one ``(H, W, 3)`` uint8 frame into a ``(size, size, 3)`` one.

    This reproduces ``openpi_client.image_tools.resize_with_pad``, the preparation pi05_droid was
    trained on: one scale ratio for both axes, so nothing is stretched, and black padding fills the
    remainder. The station's cameras are 16:9-ish, so stretching them square would put the policy
    off-distribution before it ever saw a pixel. PIL sizes are ``(width, height)`` while the array
    is ``(height, width, channels)``.
    """
    if image.shape[:2] == (size, size):
        return image
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / size, current_height / size)
    resized = Image.fromarray(image).resize(
        (int(current_width / ratio), int(current_height / ratio)), resample=Image.BILINEAR
    )
    padded = Image.new(resized.mode, (size, size), 0)
    padded.paste(resized, (int((size - resized.width) / 2), int((size - resized.height) / 2)))
    return np.asarray(padded)


class C3poStation:
    """A live N-Droids Franka through c3po.

    The connection is made once, in the constructor, and the manifest is read at the same time: a
    station that changed its arms mid-episode is not a case this example has to survive. reset,
    step and close are forwarded unchanged, because the station already has the semantics
    ``RobotPort`` describes and re-implementing them here would only add somewhere to disagree.
    """

    def __init__(self, host: str, port: int) -> None:
        import c3po

        self.robot = c3po.Robot(host=host, port=port)
        self._manifest = manifest_from_robot(self.robot)

    @property
    def manifest(self) -> StationManifest:
        return self._manifest

    def reset(self) -> dict[str, np.ndarray]:
        """Start a new episode and return the observation after the reset."""
        return self.robot.reset()

    def step(self, action: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        """Send one action, or ``None`` to read, and wait for the next observation."""
        return self.robot.step(action)

    def close(self) -> None:
        self.robot.close()


class FakeStation:
    """A deterministic station for the tests and ``--dry-run``: observations in, actions out.

    The observations are a loop, not a clip: every ``step`` advances the index and wraps, so a
    policy can run for as long as the caller wants without the script having to know the length.
    ``reset`` rewinds to the first one. ``actions`` keeps the command dictionaries in order; a
    ``None`` read is not a command and is not recorded, which is what makes a dry run's log mean
    "commands sent" rather than "ticks elapsed". ``control_rate`` defaults to 10 Hz for a dry run;
    a real schedule passes the rate the config already uses.
    """

    def __init__(self, observations: Sequence[Mapping[str, np.ndarray]], control_rate: float = 10.0) -> None:
        observations = tuple(dict(observation) for observation in observations)
        if not observations:
            raise ValueError("a fake station needs at least one observation")
        self._observations = observations
        self._manifest = manifest_from_observations(observations, control_rate)
        self._index = 0
        self.actions: list[dict[str, np.ndarray]] = []

    @property
    def manifest(self) -> StationManifest:
        return self._manifest

    def reset(self) -> dict[str, np.ndarray]:
        """Rewind to the first observation and return it."""
        self._index = 0
        return dict(self._observations[0])

    def step(self, action: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        """Record a command (unless ``None``) and return the next observation in the loop."""
        if action is not None:
            self.actions.append(dict(action))
        self._index = (self._index + 1) % len(self._observations)
        return dict(self._observations[self._index])

    def close(self) -> None:
        """Nothing to release: the observations own no resources."""


__all__ = [
    "C3poStation",
    "FakeStation",
    "action_dict",
    "build_arm_spec",
    "manifest_from_observations",
    "manifest_from_robot",
    "resize_with_pad",
    "state_from_obs",
]
