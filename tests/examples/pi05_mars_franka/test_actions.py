"""The action arithmetic: a command's units, the clamp that keeps the arm safe, and the gripper's two ends.

The clamps are the part worth testing carefully. The station integrates the velocity it is sent, so a step that
looks small in radians per second can still push a joint past its limit if the arm is already next to it.
"""

import numpy as np
import pytest
from actions import (
    binarize_gripper,
    clamp_joint_velocity,
    command_from_qdot,
    qdot_from_command,
    ticks_per_action,
)

Q_MIN = np.full(7, -2.0)
Q_MAX = np.full(7, 2.0)
Q = np.zeros(7)
QDOT_MAX = np.full(7, 0.5)


def _v(value: float) -> np.ndarray:
    """A seven-vector with `value` in every joint."""
    return np.full(7, value)


# ---------------------------------------------------------------- the command's units


def test_command_from_qdot_scales_to_full_deflection():
    qdot = np.array([0.25, -0.25, 0.5, -0.5, 0.1, 0.0, 0.4])
    assert command_from_qdot(qdot, 0.5) == pytest.approx([0.5, -0.5, 1.0, -1.0, 0.2, 0.0, 0.8])


def test_command_from_qdot_clips_an_overdriven_joint():
    assert command_from_qdot(_v(2.0), 0.5) == pytest.approx(_v(1.0))
    assert command_from_qdot(_v(-2.0), 0.5) == pytest.approx(_v(-1.0))


def test_command_from_qdot_stays_inside_the_unit_cube():
    rng = np.random.default_rng(0)
    command = command_from_qdot(rng.normal(scale=5.0, size=7), 0.5)
    assert np.all(np.abs(command) <= 1.0)


def test_qdot_from_command_inverts_the_normalisation():
    rng = np.random.default_rng(1)
    qdot = rng.uniform(-0.4, 0.4, size=7)
    assert qdot_from_command(command_from_qdot(qdot, 0.4), 0.4) == pytest.approx(qdot)


def test_qdot_from_command_inverts_a_clipped_command():
    qdot = np.array([1.0, -1.0, 0.2, -0.2, 0.0, 0.3, -0.3])
    assert qdot_from_command(command_from_qdot(qdot, 0.5), 0.5) == pytest.approx(np.clip(qdot, -0.5, 0.5))


# ---------------------------------------------------------------- the velocity and limit clamp


def test_clamp_joint_velocity_respects_the_velocity_cap():
    qdot = np.array([5.0, -5.0, 0.5, -0.5, 0.0, 0.49, 0.51])
    clamped = clamp_joint_velocity(qdot, Q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)
    assert clamped == pytest.approx([0.5, -0.5, 0.5, -0.5, 0.0, 0.49, 0.5])


def test_clamp_joint_velocity_stops_a_joint_at_its_limit():
    q = Q.copy()
    q[0] = Q_MIN[0] + 0.01  # a centimetre of travel left at 5 cm/s: -0.2 rad/s
    clamped = clamp_joint_velocity(_v(-0.5), q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)
    assert clamped[0] == pytest.approx(-0.2)
    # away from the limit the cap is what binds
    assert clamp_joint_velocity(_v(-0.5), q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)[1] == pytest.approx(-0.5)
    assert clamp_joint_velocity(_v(0.5), q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)[0] == pytest.approx(0.5)


def test_clamp_joint_velocity_respects_the_step_to_the_upper_limit():
    q = Q.copy()
    q[3] = Q_MAX[3] - 0.01
    clamped = clamp_joint_velocity(_v(0.5), q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)
    assert clamped[3] == pytest.approx(0.2)  # 0.01 / 0.05
    assert clamped[4] == pytest.approx(0.5)


def test_clamp_joint_velocity_never_crosses_a_limit():
    rng = np.random.default_rng(2)
    dt = 0.02
    for _ in range(100):
        q = rng.uniform(Q_MIN, Q_MAX)
        qdot = rng.normal(scale=2.0, size=7)
        stepped = q + clamp_joint_velocity(qdot, q, QDOT_MAX, Q_MIN, Q_MAX, dt=dt) * dt
        assert np.all(stepped >= Q_MIN - 1e-12) and np.all(stepped <= Q_MAX + 1e-12)


def test_clamp_joint_velocity_brings_back_a_joint_already_out_of_range():
    q = Q.copy()
    q[0] = Q_MIN[0] - 0.01
    clamped = clamp_joint_velocity(_v(-0.5), q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.05)
    assert clamped[0] == pytest.approx(0.2)  # the only way back inside is positive


def test_clamp_joint_velocity_rejects_a_bad_step():
    with pytest.raises(ValueError, match="dt"):
        clamp_joint_velocity(_v(0.1), Q, QDOT_MAX, Q_MIN, Q_MAX, dt=0.0)
    with pytest.raises(ValueError, match="qdot_max"):
        clamp_joint_velocity(_v(0.1), Q, _v(0.0), Q_MIN, Q_MAX, dt=0.02)


# ---------------------------------------------------------------- the gripper


def test_binarize_gripper_snaps_at_the_threshold():
    assert binarize_gripper(1.0, 1.0, (0.0, 1.0)) == 1.0
    assert binarize_gripper(0.9, 1.0, (0.0, 1.0)) == 0.0  # closed only at the closed end
    assert binarize_gripper(0.6, 0.5, (0.0, 1.0)) == 1.0
    assert binarize_gripper(0.5, 0.5, (0.0, 1.0)) == 1.0  # the threshold itself closes
    assert binarize_gripper(0.4, 0.5, (0.0, 1.0)) == 0.0


def test_binarize_gripper_handles_a_closed_at_the_low_end():
    assert binarize_gripper(0.0, 0.0, (0.0, 0.04)) == 0.0
    assert binarize_gripper(0.02, 0.0, (0.0, 0.04)) == 0.04
    assert binarize_gripper(0.04, 0.0, (0.0, 0.04)) == 0.04


def test_binarize_gripper_rejects_a_value_outside_the_range():
    with pytest.raises(ValueError, match="gripper_range"):
        binarize_gripper(1.5, 1.0, (0.0, 1.0))
    with pytest.raises(ValueError, match="gripper_range"):
        binarize_gripper(-0.1, 1.0, (0.0, 1.0))


# ---------------------------------------------------------------- holding an action


def test_ticks_per_action_rounds_up_and_never_reaches_zero():
    assert ticks_per_action(50, 15) == 4
    assert ticks_per_action(15, 15) == 1
    assert ticks_per_action(10, 30) == 1


def test_ticks_per_action_rejects_a_bad_rate():
    with pytest.raises(ValueError, match="action_rate"):
        ticks_per_action(50, 0)


# ---------------------------------------------------------------- what the errors say


def test_shapes_are_named():
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        command_from_qdot(np.zeros(6), 0.5)
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        qdot_from_command(np.zeros((7, 1)), 0.5)
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        clamp_joint_velocity(_v(0.1), np.zeros(8), QDOT_MAX, Q_MIN, Q_MAX, dt=0.02)


def test_the_full_speed_must_be_positive():
    with pytest.raises(ValueError, match="qdot_full"):
        command_from_qdot(_v(0.1), 0.0)
    with pytest.raises(ValueError, match="qdot_full"):
        qdot_from_command(_v(0.1), -0.5)
