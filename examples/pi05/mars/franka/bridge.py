"""The MARS-on-Franka bridge: the kinematic chain, the damped inverse, the units, the readout.

Every function here is a pure function of its arguments: numpy only, no I/O, no state, no c3po, no
torch. The last one is why the matrix-spec helpers below are re-derived rather than imported from
``lerobot.teleoperators.readout`` (that module imports torch at module scope, and ``--dry-run`` must
not load torch); the tests check the fixed forms against the readout's.

The chain is not derived here. It is mujoco_menagerie's ``panda.xml`` as already validated on this
arm by r2d2's ``impedance_loop.py`` (``PandaModel``), which records the 2026-08-27 hardware fault a
hand-derived DH table caused; it is copied, comments included, from the previous project's
``examples-from-unrelated-project/inverse_kinematics.py``. Do not re-derive it or retune its
numbers. The units are DESIGN.md's: ``q`` radians, ``qdot`` rad/s, a twist as
``(vx, vy, vz, wx, wy, wz)`` in the base frame, one action unit as 0.2 rad per control step at
15 Hz = 3 rad/s (flow-steering/docs/plant.md measures the plant's 0.2 rad per step; bb8's
``impedance_executor.py`` integrates a unit that way).
"""

from __future__ import annotations

from typing import Any

import numpy as np

ARM_DIMS = 7
TWIST_DIMS = 6
ACTION_UNIT = 3.0
"""rad/s per action unit: 0.2 rad per 15 Hz control step = 3 rad/s (DESIGN.md "Units")."""

LAMBDA = 1e-2
"""Damped-solve regularization weight: FC App. E, and the value the reference implementation
shipped (``inverse_kinematics.LAMBDA``)."""

# mujoco_menagerie franka_emika_panda/panda.xml, via impedance_loop.py's PandaModel: the
# static transform of each link relative to its parent, as (position, unit quaternion
# w,x,y,z), with hinge i about the z-axis of the frame this transform reaches.
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
# Flange (hand) frame origin in link 7's frame. Its fixed -45 deg yaw (Rz, panda.xml's
# ``quat="0.9238795 0 0 -0.3826834"``) rotates that frame without moving the origin, and
# only the origin enters a linear Jacobian — so the yaw cancels here.
_HAND_POS = np.array([0.0, 0.0, 0.107])


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """impedance_loop.py's, unchanged."""
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
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _frames(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Link frame origins (8, 3) and rotations (7, 3, 3), Franka base frame.

    Hinge *i* turns about the z-axis through ``origins[i + 1]``: the CHILD frame's origin,
    not the parent's. impedance_loop.py records that as load-bearing — using the parent's
    overstates the lever arm wherever the two are not parallel (the 2026-08-27 joint-4
    error).
    """
    assert q.shape == (ARM_DIMS,), f"q must be ({ARM_DIMS},), got {q.shape}"
    origins = np.zeros((8, 3))
    rotations = np.zeros((ARM_DIMS, 3, 3))
    p, r = np.zeros(3), np.eye(3)
    for i in range(ARM_DIMS):
        p = p + r @ np.asarray(_STATIC_POS[i])
        r = r @ _quat_to_rot(*_STATIC_QUAT[i]) @ _rz(q[i])
        origins[i + 1] = p
        rotations[i] = r
    return origins, rotations


def flange_pose(q: np.ndarray) -> np.ndarray:
    """Flange frame origin (3,) in the base frame: the Jacobian's reference point."""
    origins, rotations = _frames(np.asarray(q, dtype=np.float64))
    return origins[7] + rotations[6] @ _HAND_POS


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """(6, 7) geometric Jacobian at the flange: three linear rows, then three angular.

    The angular block is the hinge axis ``z_i``, and it is not decoration. FC App. E
    commands a ``v_user`` whose angular part is zero, and those rows are what hold the
    flange's orientation still while it translates. Solving the linear rows alone leaves
    the wrist free, and the minimum-norm solution then spends much of the command on
    rotation nobody asked for: at the DROID reset pose a pure "up" came out as 0.27 rad/s
    of wrist spin alongside 0.11 m/s of translation.
    """
    q = np.asarray(q, dtype=np.float64)
    origins, rotations = _frames(q)
    point = origins[7] + rotations[6] @ _HAND_POS
    jac = np.zeros((TWIST_DIMS, ARM_DIMS))
    for i in range(ARM_DIMS):
        axis = rotations[i][:, 2]
        jac[:3, i] = np.cross(axis, point - origins[i + 1])
        jac[3:, i] = axis
    return jac


def _unit_scale(length: float) -> np.ndarray:
    """(6, 6) that puts a twist in metres: the angular rows are multiplied by ``length``."""
    if not length > 0:
        raise ValueError(
            f"length is the mixed-unit twist length in metres, so it must be positive, got {length}"
        )
    return np.diag([1.0, 1.0, 1.0, length, length, length])


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    """``value`` as a finite float (size,), validated at the public boundary."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must be ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite, got {vector}")
    return vector


def joint_velocity(twist: np.ndarray, q: np.ndarray, length: float = 0.2) -> np.ndarray:
    """(7,) rad/s: damped least squares of ``twist`` at ``q``, with mixed units.

    The units trap: ``twist`` mixes m/s and rad/s, and 0.5 rad/s is not comparable to 0.11 m/s
    until the angular rows are multiplied by the 0.2 m twist length (0.5 rad/s * 0.2 m = 0.1
    m/s). Both the angular rows of ``J`` and the angular part of the twist carry ``length``, so
    the linear and angular residuals enter the least-squares in the same unit and neither block
    dominates. The same diagonal on both sides leaves the equation ``J qdot = twist`` intact, so
    the solution is already rad/s; only the damping weighting differs from the unweighted solve.
    """
    twist = _vector(twist, TWIST_DIMS, "twist")
    scale = _unit_scale(length)
    scaled = scale @ geometric_jacobian(q)
    return scaled.T @ np.linalg.solve(scaled @ scaled.T + LAMBDA**2 * np.eye(TWIST_DIMS), scale @ twist)


def action_units(qdot: np.ndarray) -> np.ndarray:
    """(7,) joint velocity in the checkpoint's action units (one unit = ``ACTION_UNIT`` rad/s)."""
    return _vector(qdot, ARM_DIMS, "qdot") / ACTION_UNIT


def operator_command(twist: np.ndarray, q: np.ndarray, length: float = 0.2) -> np.ndarray:
    """(7,) action units: the operator's command for a station twist (IK, then the unit scale)."""
    return action_units(joint_velocity(twist, q, length))


def _right_inverse(jac: np.ndarray, length: float) -> np.ndarray:
    """(7, 6) ``J+`` with ``J J+ == I6`` for the mixed-unit map ``joint_velocity`` uses.

    ``corruption_lift``'s inverse identity needs ``J J+ == I6`` exactly (see the algebra there);
    the damped inverse ``joint_velocity`` solves with is short of that by O(lambda^2), so the
    lift takes the Moore-Penrose right inverse of the same scaled Jacobian instead. The nominal
    pose is fixed, so this is evaluated once and not near a singularity by construction.
    """
    scale = _unit_scale(length)
    return np.linalg.pinv(scale @ jac) @ scale


def corruption_lift(matrix6: np.ndarray, q0: np.ndarray, length: float = 0.2) -> np.ndarray:
    """(7, 7) ``F = I + J+(q0) (M6 - I) J(q0)``: the corruption lifted into joint space.

    ``M6`` lives in the twist frame, which rotates with ``q``; applying it once, in the nominal
    commanded subspace at ``q0``, makes the distortion a fixed 7x7 on the command: ``M6`` on
    ``range(J+(q0))`` and the identity on its orthogonal complement. Its exact inverse is
    ``lift_inverse``, because ``J J+ == I6`` makes ``(M - I)`` and ``(M^-1 - I)`` cancel on that
    subspace. That is what makes the MARS learning problem well-posed. ``J+`` here is the exact
    (Moore-Penrose) right inverse: the damping stays in ``joint_velocity``, because only
    ``J J+ == I6`` makes the stated inverse exact.
    """
    matrix6 = _matrix(matrix6, TWIST_DIMS, "matrix6")
    jac = geometric_jacobian(q0)
    jac_plus = _right_inverse(jac, length)
    return np.eye(ARM_DIMS) + jac_plus @ (matrix6 - np.eye(TWIST_DIMS)) @ jac


def lift_inverse(matrix6: np.ndarray, q0: np.ndarray, length: float = 0.2) -> np.ndarray:
    """(7, 7) the exact inverse of ``corruption_lift(matrix6, q0)``: the same with ``M6^-1``."""
    matrix6 = _matrix(matrix6, TWIST_DIMS, "matrix6")
    return corruption_lift(np.linalg.inv(matrix6), q0, length)


_SPEC_FORMS = "{rotation_z_deg: d}, {scale: [sx, sy, sz]}, {M: rows}, {random: {...}}, or bare rows"


def _scalar(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return number


def _matrix(value: Any, size: int, name: str) -> np.ndarray:
    """``value`` as a finite float (size, size), validated at the public boundary."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: entries must be numbers ({e})") from e
    if matrix.shape != (size, size):
        raise ValueError(f"{name}: expected a {size}x{size} matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}: entries must be finite, got {matrix}")
    return matrix


def _rotation_z(degrees: float) -> np.ndarray:
    """3x3 rotation about z: ``readout.matrix_spec``'s 3-D rotation form, same numbers."""
    cos, sin = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle(axis: np.ndarray, radians: float) -> np.ndarray:
    """Rodrigues' formula, ``axis`` a unit vector: the rotation LIBERO's random family draws."""
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(radians) * cross + (1.0 - np.cos(radians)) * (cross @ cross)


def _range(value: Any, where: str, low: float, high: float, low_open: bool = False) -> tuple[float, float]:
    """``value`` as an ordered (lo, hi) pair of finite numbers inside [low, high] (or (low, high])."""
    try:
        lo, hi = (float(item) for item in value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: expected a [low, high] pair of numbers, got {value!r}") from e
    inside = (low < lo if low_open else low <= lo) and hi <= high
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi and inside):
        bounds = f"{'(' if low_open else '['}{low:g}, {high:g}]"
        raise ValueError(
            f"{where}: expected a [low, high] pair with low <= high inside {bounds}, got {value!r}"
        )
    return lo, hi


def _random_matrix(block: Any, rng: np.random.Generator | None, where: str) -> np.ndarray:
    """One draw from LIBERO's family ``M = R(axis, angle) @ diag(scale)``.

    The angle is uniform in ``rotation_deg``, each gain log-uniform in ``scale``, and the rotation
    axis uniform on the sphere -- LIBERO's ``RandomCorruption`` samples exactly this, and its
    fields are the block's keys. The fixed forms in the same grammar rotate about z instead. The
    family's semantics on the Franka are unverified; only its invertibility is tested.
    """
    if not isinstance(block, dict) or not set(block) <= {"rotation_deg", "scale"}:
        raise ValueError(f"{where}: random takes rotation_deg and scale ranges, got {block!r}")
    angle = _range(block.get("rotation_deg", (0.0, 0.0)), f"{where}.rotation_deg", 0.0, 180.0)
    scale = _range(block.get("scale", (1.0, 1.0)), f"{where}.scale", 0.0, np.inf, low_open=True)
    generator = _generator(rng)
    axis = generator.normal(size=3)
    axis /= np.linalg.norm(axis)
    radians = np.radians(generator.uniform(*angle))
    gains = np.exp(generator.uniform(np.log(scale[0]), np.log(scale[1]), size=3))
    return _axis_angle(axis, radians) @ np.diag(gains)


def matrix6_from_spec(spec: Any, rng: np.random.Generator | None = None) -> np.ndarray:
    """(6, 6) corruption matrix: the 3x3 translation block, then the identity on the twist.

    The grammar is ``readout.matrix_spec(spec, 3)``'s, so a spec copied from a LIBERO config means
    here what it means there: ``{rotation_z_deg: d}`` (the 3-D rotation about z; ``rotation_deg``
    is only the 2-D form), ``{scale: [sx, sy, sz]}``, ``{M: rows}`` (or bare rows), and
    ``{random: {...}}``, one draw from the per-trial family. ``F`` is the reversal-adapter
    grammar, not a corruption key. ``None`` is the off state and gives the identity, so a config's
    explicit ``corruption = "none"`` parses to it before this call. A curl or a quantize
    corruption is not a matrix, so it is rejected.
    """
    if spec is None:
        return np.eye(TWIST_DIMS)
    where = "matrix6_from_spec"
    if isinstance(spec, str):
        raise ValueError(
            f"{where}: {spec!r} is not a matrix spec; the off state is None (config parses 'none' to it)"
        )
    if isinstance(spec, dict):
        keys = set(spec)
        if keys & {"curl", "quantize"}:
            raise ValueError(f"{where}: a curl or quantize corruption has no matrix; got {sorted(keys)}")
        if keys == {"random"}:
            block = _random_matrix(spec["random"], rng, f"{where}.random")
        elif keys == {"F"}:
            raise ValueError(
                f"{where}: the corruption grammar's literal matrix key is 'M', not 'F' "
                f"(F is the reversal-adapter grammar); expected {_SPEC_FORMS}"
            )
        elif keys == {"rotation_deg"}:
            raise ValueError(
                f"{where}: the 3-D rotation key is 'rotation_z_deg', not 'rotation_deg' "
                "(rotation_deg is readout.matrix_spec's 2-D plane rotation)"
            )
        elif keys == {"rotation_z_deg"}:
            block = _rotation_z(_scalar(spec["rotation_z_deg"], f"{where}.rotation_z_deg"))
        elif keys == {"scale"}:
            scale = spec["scale"]
            if not isinstance(scale, (list, tuple)) or len(scale) != 3:
                raise ValueError(f"{where}: scale must be three numbers, got {scale!r}")
            block = np.diag([_scalar(item, f"{where}.scale") for item in scale])
        elif keys == {"M"}:
            block = _matrix(spec["M"], 3, f"{where}.M")
        else:
            raise ValueError(f"{where}: expected {_SPEC_FORMS}, got {sorted(keys)}")
    else:
        block = _matrix(spec, 3, where)
    matrix6 = np.eye(TWIST_DIMS)
    matrix6[:3, :3] = block
    return matrix6


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    """The generator given, or numpy's default one (fresh entropy per call when none is given)."""
    return np.random.default_rng() if rng is None else rng


def readout(
    intent7: np.ndarray,
    matrix7: np.ndarray | None,
    noise_std7: np.ndarray | None,
    gain: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """(7,) served command: ``clip(gain * F @ intent7 + sigma * eps, -1, 1)``, eps ~ N(0, 1).

    The clip is the one nonlinearity the chain cannot lift away: it is why ``gain = "auto"``
    exists, so a corrupted command must stay inside [-1, 1] before the clip acts. ``noise_std7``
    is a scalar or one std per joint; None draws nothing.
    """
    intent = _vector(intent7, ARM_DIMS, "intent7")
    served = intent if matrix7 is None else _matrix(matrix7, ARM_DIMS, "matrix7") @ intent
    served = _scalar(gain, "gain") * served
    if noise_std7 is not None:
        sigma = np.asarray(noise_std7, dtype=np.float64)
        if sigma.shape not in ((), (ARM_DIMS,)):
            raise ValueError(f"noise_std7 must be a scalar or ({ARM_DIMS},), got {sigma.shape}")
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError(f"noise_std7 must be finite and non-negative, got {noise_std7!r}")
        served = served + sigma * _generator(rng).standard_normal(ARM_DIMS)
    return np.clip(served, -1.0, 1.0)


def nominal_pose(obs: dict, arm: str) -> np.ndarray:
    """(7,) the arm's joint angles from a station observation: its ``{arm}/joint_position`` entry."""
    key = f"{arm}/joint_position"
    if key not in obs:
        raise ValueError(f"the observation has no {key!r} (keys: {sorted(obs)})")
    return _vector(obs[key], ARM_DIMS, key)
