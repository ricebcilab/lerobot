"""The action-space arithmetic for the Franka example: units, clamps, and the gripper's two ends.

The station integrates the joint velocity it is sent, which makes every safety decision the arithmetic's. Three
scales meet here and are easy to confuse: the operator's command is a fraction of full deflection in ``[-1, 1]``,
the policy's action is a joint velocity in rad/s, and the station's gripper is a position in its own units. Each
function does one conversion, so a runner can say which scale it is in.

Nothing here reads the hardware: the limits arrive as arrays and the step as ``dt``, so the same clamp is tested
offline and used on the arm.
"""

from __future__ import annotations

import math

import numpy as np
from interfaces import ARM_DIMS

__all__ = [
    "binarize_gripper",
    "clamp_joint_velocity",
    "command_from_qdot",
    "qdot_from_command",
    "ticks_per_action",
]


def _joints(values, name: str) -> np.ndarray:
    """``values`` as a float64 ``(7,)`` array, or a ValueError naming the argument and its shape."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (ARM_DIMS,):
        raise ValueError(f"{name} must have shape ({ARM_DIMS},), got {array.shape}")
    return array


def _speed(value, name: str) -> float:
    """``value`` as a positive speed; a full deflection of zero would divide every command by nothing."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive speed, got {value!r}") from None
    if not speed > 0 or not math.isfinite(speed):
        raise ValueError(f"{name} must be a positive speed, got {value!r}")
    return speed


def command_from_qdot(qdot: np.ndarray, qdot_full: float) -> np.ndarray:
    """The operator's command for a joint velocity: ``qdot / qdot_full``, clipped to ``[-1, 1]``.

    ``qdot_full`` is the joint speed a full stick deflection asks for, so the command says how much of the
    arm's travel the operator is using. Clipping is what keeps a policy or Jacobian overshoot from becoming an
    unbounded push: the command never asks for more than the full deflection the reference was built for.
    """
    return np.clip(_joints(qdot, "qdot") / _speed(qdot_full, "qdot_full"), -1.0, 1.0)


def qdot_from_command(command: np.ndarray, qdot_full: float) -> np.ndarray:
    """The joint velocity a command asks for: ``command * qdot_full``, in rad/s.

    The inverse of :func:`command_from_qdot`, for the runner that needs the desired velocity next to the command
    (to report it, or to hand it to the station without a policy). It does not clip, because the inverse of a
    clipped command is already a speed the arm may be asked for.
    """
    return _joints(command, "command") * _speed(qdot_full, "qdot_full")


def clamp_joint_velocity(
    qdot: np.ndarray,
    q: np.ndarray,
    qdot_max: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    dt: float,
) -> np.ndarray:
    """A joint velocity the arm can execute for one step of ``dt`` seconds without crossing a limit.

    First the per-joint speed cap, then the band that reaches a limit exactly: at ``q`` the fastest the joint may
    move is ``(q_limit - q) / dt``, so a joint a centimetre from its stop cannot be sent at a speed that would
    overshoot it. The band binds before the cap only near a limit, which is the case the cap cannot see.

    A joint already outside its limits is the one case the band overrides the cap: there the band points back
    inside, and returning to the range is the safe action even at more than ``qdot_max``.
    """
    qdot = _joints(qdot, "qdot")
    q = _joints(q, "q")
    cap = _joints(qdot_max, "qdot_max")
    lower = _joints(q_min, "q_min")
    upper = _joints(q_max, "q_max")
    if np.any(cap <= 0) or np.any(upper <= lower):
        raise ValueError("qdot_max must be positive and every q_max must exceed its q_min")
    step = float(dt)
    if not step > 0 or not math.isfinite(step):
        raise ValueError(f"dt must be a positive step in seconds, got {dt!r}")
    return np.clip(np.clip(qdot, -cap, cap), (lower - q) / step, (upper - q) / step)


def binarize_gripper(value: float, closed_at: float, gripper_range: tuple[float, float]) -> float:
    """The gripper position a continuous value means: one end of the station's travel, or the other.

    The operator's gripper is a button and the policy's is a number, so the number is snapped to the end it asks
    for. ``closed_at`` says which end is closed (the station's closed position, at either end of its travel) and
    is the threshold: a value on the closed side of it snaps closed, everything else opens.
    """
    if len(gripper_range) != 2:
        raise ValueError(f"gripper_range must be (low, high), got {gripper_range!r}")
    lo, hi = float(gripper_range[0]), float(gripper_range[1])
    if not hi > lo:
        raise ValueError(f"gripper_range must be (low, high), got {gripper_range!r}")
    closed_at, value = float(closed_at), float(value)
    if not lo <= closed_at <= hi:
        raise ValueError(f"closed_at must lie in gripper_range [{lo}, {hi}], got {closed_at}")
    if not lo <= value <= hi:
        raise ValueError(f"the gripper value {value} is outside gripper_range [{lo}, {hi}]")
    closed_end = hi if hi - closed_at <= closed_at - lo else lo
    if closed_end == hi:
        return hi if value >= closed_at else lo
    return lo if value <= closed_at else hi


def ticks_per_action(control_rate: float, action_rate: float) -> int:
    """How many station ticks one action is held for: ``ceil(control_rate / action_rate)``, at least one.

    The checkpoint's actions are one policy step apart; the station's loop runs at its own rate. Holding each
    action for whole ticks is the only way not to resample the policy's timing, so a round-up is used and a
    station slower than the policy still gets one tick per action.
    """
    control = float(control_rate)
    action = float(action_rate)
    if not control > 0 or not action > 0:
        raise ValueError(
            f"control_rate and action_rate must be positive, got control_rate={control_rate!r}, "
            f"action_rate={action_rate!r}"
        )
    return max(1, math.ceil(control / action))
