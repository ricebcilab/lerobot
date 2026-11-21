"""The Franka's differential kinematics: the flange pose, the twist, and the damped inverse.

Everything here is the previous project's ``inverse_kinematics.py``. The chain is mujoco_menagerie's
``panda.xml``, validated on this arm by the lab's ``impedance_loop.py``, which records the 2026-08-27
hardware fault that a hand-derived DH table caused there. Do not re-derive the chain and do not retune
its numbers; transpose them.

Units follow ``interfaces.py``: ``q`` in radians, ``qdot`` in radians per second, and a twist as
``(vx, vy, vz, wx, wy, wz)`` in the station's base frame. Pure: numpy only, no state, no I/O.
"""

from __future__ import annotations

import numpy as np

LAMBDA = 1e-2
"""Damped-solve regularization weight: FC App. E, and the value the previous project shipped."""

# mujoco_menagerie franka_emika_panda/panda.xml, as impedance_loop.py's PandaModel reads it: each
# link's static transform relative to its parent, (position, unit quaternion w,x,y,z), with hinge i
# about that frame's z-axis. The table is copied, never recomputed.
_STATIC_POS = (
    (0.0, 0.0, 0.333),
    (0.0, 0.0, 0.0),
    (0.0, -0.316, 0.0),
    (0.0825, 0.0, 0.0),
    (-0.0825, 0.384, 0.0),
    (0.0, 0.0, 0.0),
    (0.088, 0.0, 0.0),
)
_SQ2 = np.sqrt(2.0) / 2.0
_STATIC_QUAT = (
    (1.0, 0.0, 0.0, 0.0),
    (_SQ2, -_SQ2, 0.0, 0.0),  # Rx(-90 deg)
    (_SQ2, _SQ2, 0.0, 0.0),  # Rx(+90 deg)
    (_SQ2, _SQ2, 0.0, 0.0),
    (_SQ2, -_SQ2, 0.0, 0.0),
    (_SQ2, _SQ2, 0.0, 0.0),
    (_SQ2, _SQ2, 0.0, 0.0),
)
# The flange origin in link 7's frame. The flange's fixed -45 deg yaw (panda.xml's
# ``quat="0.9238795 0 0 -0.3826834"``, Rz) rotates that frame but does not move its origin, and only
# the origin enters the linear Jacobian, so the yaw cancels here.
_HAND_POS = np.array([0.0, 0.0, 0.107])


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """The unit quaternion's rotation matrix, as impedance_loop.py computes it."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rz(theta: float) -> np.ndarray:
    """A rotation about z, for one hinge angle."""
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _seven(value: object, name: str) -> np.ndarray:
    """One joint-space vector as seven floats, validated at the public boundary."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (7,):
        raise ValueError(f"{name} must be (7,), got {vector.shape}")
    return vector


def _frames(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Link frame origins (8, 3) and rotations (7, 3, 3), all in the Franka base frame.

    Hinge *i* turns about the z-axis through ``origins[i + 1]``: the CHILD frame's origin, not the
    parent's. Using the parent's overstates the lever arm wherever the two origins differ, and
    impedance_loop.py records the 2026-08-27 joint-4 hardware fault that came from exactly that.
    The offset is therefore load-bearing; the chain below is not off by one.
    """
    assert q.shape == (7,), f"q must be (7,), got {q.shape}"
    origins = np.zeros((8, 3))
    rotations = np.zeros((7, 3, 3))
    p, r = np.zeros(3), np.eye(3)
    for i in range(7):
        p = p + r @ np.asarray(_STATIC_POS[i])
        r = r @ _quat_to_rot(*_STATIC_QUAT[i]) @ _rz(q[i])
        origins[i + 1] = p
        rotations[i] = r
    return origins, rotations


class PandaJacobian:
    """The arm's differential kinematics at the flange, in the base frame (``JacobianPort``).

    Stateless: every call rebuilds the chain from ``q``. Seven rotations are cheaper than a cache
    that could go stale, and the runner calls this once per replan rather than per station tick.
    """

    def flange_pose(self, q: np.ndarray) -> np.ndarray:
        """The flange frame origin, ``(3,)`` in metres: the Jacobian's reference point."""
        origins, rotations = _frames(_seven(q, "q"))
        return origins[7] + rotations[6] @ _HAND_POS

    def geometric_jacobian(self, q: np.ndarray, point: np.ndarray) -> np.ndarray:
        """The (6, 7) geometric Jacobian of ``point``: three linear rows, then three angular.

        The angular block is the hinge axis ``z_i``, and it is not decoration. The stick's twist has
        a zero angular part, and these three rows are what hold the flange's orientation still while
        it translates. Solving the linear rows alone leaves the wrist free, and the minimum-norm
        solution then spends the command on rotation nobody asked for: at the DROID reset pose a
        pure "up" came out as 0.27 rad/s of wrist spin alongside 0.11 m/s of translation.
        """
        q = _seven(q, "q")
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"point must be (3,), got {point.shape}")
        origins, rotations = _frames(q)
        jac = np.zeros((6, 7))
        for i in range(7):
            axis = rotations[i][:, 2]
            jac[:3, i] = np.cross(axis, point - origins[i + 1])
            jac[3:, i] = axis
        return jac

    def twist(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """The flange twist ``(6,)`` for a joint velocity ``(7,)``."""
        q = _seven(q, "q")
        qdot = _seven(qdot, "qdot")
        jac = self.geometric_jacobian(q, self.flange_pose(q))
        return jac @ qdot

    def joint_velocity(self, q: np.ndarray, twist: np.ndarray) -> np.ndarray:
        """The damped least-squares joint velocity ``(7,)``, rad/s, for a flange twist ``(6,)``.

        ``qdot = J^T (J J^T + lambda^2 I)^-1 v`` with ``LAMBDA``. The damping is what keeps the
        solve finite at a singular pose; it also means the forward twist of the answer is a fraction
        of the request rather than the request itself, which the test pins rather than hides.
        """
        q = _seven(q, "q")
        twist = np.asarray(twist, dtype=np.float64)
        if twist.shape != (6,):
            raise ValueError(f"twist must be (6,), got {twist.shape}")
        jac = self.geometric_jacobian(q, self.flange_pose(q))
        return jac.T @ np.linalg.solve(jac @ jac.T + LAMBDA**2 * np.eye(6), twist)


__all__ = ["LAMBDA", "PandaJacobian"]
