"""Ports and shared types for the Franka shared-autonomy example.

The example steers a π0.5-DROID checkpoint on an N-Droids Franka station through c3po. Three things vary
between machines: the station (its manifest, its action keys), the arm's kinematics (the Jacobian), and the
operator's device. Each is a protocol here, so every module above it is pure and testable without hardware.

Unit conventions, which the rest of the example depends on:

- ``q`` and ``qdot`` are joint positions and velocities, radians and radians per second, seven of them.
- A *twist* is the six-vector ``(vx, vy, vz, wx, wy, wz)`` of the flange in the station's base frame:
  metres per second, then radians per second.
- An *action* is ``(qdot_1..qdot_7, gripper)``: eight numbers in the checkpoint's action units. The
  checkpoint's postprocessor unnormalizes them; the station adapter sends them.
- A *command* is what the operator asks for: seven numbers in ``[-1, 1]``, one per joint, where 1 is a full
  deflection. It is the reference the steering policy steers toward, not the action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ARM_DIMS = 7
"""Joints of the arm, and the dims a command and a user module act on."""

ACTION_DIMS = ARM_DIMS + 1
"""The checkpoint's action: seven joint velocities, then the gripper position."""

GRIPPER_DIM = ARM_DIMS
"""Index of the gripper in the action (the eighth dim)."""


@dataclass(frozen=True)
class StationManifest:
    """What the station says it is, from c3po's handshake.

    ``action_keys`` is the set of keys ``step()`` accepts; the adapter sends the arm's joint velocity and the
    gripper position, so both must be present. ``cameras`` are observation keys. ``control_rate`` is the
    station's native loop rate in Hz, which decides how long one action is held.
    """

    arms: tuple[str, ...]
    cameras: tuple[str, ...]
    action_keys: tuple[str, ...]
    control_rate: float

    def arm(self) -> str:
        """The arm to drive: the only one, or ``"follower"``. Raises when neither applies."""
        if len(self.arms) == 1:
            return self.arms[0]
        if "follower" in self.arms:
            return "follower"
        raise ValueError(f"the station has arms {self.arms}; name one in the config")


@dataclass(frozen=True)
class ArmSpec:
    """The arm's limits, which every clamp reads.

    ``q_min``/``q_max``/``qdot_max`` are per joint. ``qdot_max`` is a safety cap, not the arm's datasheet
    limit: the runner never commands more, whatever the policy says. ``gripper_range`` is the station's
    gripper travel, and ``gripper_closed_at`` the value a closed command takes (the station's gripper is a
    position, so a discrete button becomes one of the two ends).
    """

    joint_names: tuple[str, ...]
    q_min: tuple[float, ...]
    q_max: tuple[float, ...]
    qdot_max: tuple[float, ...]
    gripper_range: tuple[float, float]
    gripper_closed_at: float

    def __post_init__(self) -> None:
        if len(self.joint_names) != ARM_DIMS:
            raise ValueError(f"the arm needs {ARM_DIMS} joint names, got {len(self.joint_names)}")
        for name in ("q_min", "q_max", "qdot_max"):
            if len(getattr(self, name)) != ARM_DIMS:
                raise ValueError(f"{name} needs {ARM_DIMS} values, got {len(getattr(self, name))}")
        if not all(hi > lo for lo, hi in zip(self.q_min, self.q_max, strict=True)):
            raise ValueError("every q_max must exceed its q_min")
        if not all(cap > 0 for cap in self.qdot_max):
            raise ValueError("every qdot_max must be positive")
        lo, hi = self.gripper_range
        if not (hi > lo and lo <= self.gripper_closed_at <= hi):
            raise ValueError(
                f"gripper_closed_at must lie in gripper_range {self.gripper_range}, got {self.gripper_closed_at}"
            )

    @property
    def qdot_cap(self) -> np.ndarray:
        """``qdot_max`` as an array, for broadcasting over a seven-vector."""
        return np.asarray(self.qdot_max, dtype=np.float64)


@dataclass(frozen=True)
class CommandSpec:
    """What one unit of the operator's command means.

    ``qdot_full`` is the joint speed a full deflection asks for, in rad/s: the scale that turns a stick
    position into a command, and the FRS reference's step. ``twist_full`` is the flange speed the stick asks
    for, in m/s, before the Jacobian; it and ``qdot_full`` are related by the arm's pose, so the runner reports
    both and the config fixes the one the operator feels.
    """

    qdot_full: float
    twist_full: float

    def __post_init__(self) -> None:
        if not (self.qdot_full > 0 and self.twist_full > 0):
            raise ValueError(f"both speeds must be positive, got {self}")


class RobotPort(Protocol):
    """A station: a reset, a step, and what it says it is."""

    @property
    def manifest(self) -> StationManifest: ...

    def reset(self) -> dict[str, np.ndarray]:
        """A new episode. Returns the observation after the reset."""
        ...

    def step(self, action: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
        """Send one action and wait for the next observation; ``None`` reads without commanding."""
        ...

    def close(self) -> None: ...


class JacobianPort(Protocol):
    """The arm's differential kinematics at the flange, in the base frame."""

    def flange_pose(self, q: np.ndarray) -> np.ndarray:
        """The flange position, ``(3,)``, in metres."""
        ...

    def twist(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """The flange twist ``(6,)`` for a joint velocity ``(7,)``."""
        ...

    def joint_velocity(self, q: np.ndarray, twist: np.ndarray) -> np.ndarray:
        """The damped least-squares joint velocity ``(7,)``, rad/s, for a flange twist ``(6,)``."""
        ...


__all__ = [
    "ACTION_DIMS",
    "ARM_DIMS",
    "GRIPPER_DIM",
    "ArmSpec",
    "CommandSpec",
    "JacobianPort",
    "RobotPort",
    "StationManifest",
]
