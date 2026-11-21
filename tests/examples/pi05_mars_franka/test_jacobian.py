"""The Franka Jacobian: the chain's pose, its six rows, and the damped solve that inverts them."""

import numpy as np
import pytest
from jacobian import _HAND_POS, _STATIC_POS, _STATIC_QUAT, PandaJacobian, _quat_to_rot

READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
"""panda.xml's ``home`` keyframe: the arm forward and bent, well away from a singular pose."""


def test_the_geometric_jacobian_is_six_by_seven_with_unit_angular_rows():
    arm = PandaJacobian()
    jac = arm.geometric_jacobian(READY, arm.flange_pose(READY))
    assert jac.shape == (6, 7)
    # Each angular row is a hinge axis, so it must be a unit vector whatever q is.
    np.testing.assert_allclose(np.linalg.norm(jac[3:], axis=0), 1.0, atol=1e-12)


def test_the_flange_pose_at_zero_is_the_static_chain_summed():
    # At q = 0 every hinge rotation is identity, so the pose is the static frames composed. The
    # test composes them itself from the module's table rather than reading a magic number: that
    # checks the chain's order and the flange offset, which a constant lifted from the module
    # would not.
    position = np.zeros(3)
    rotation = np.eye(3)
    for offset, quat in zip(_STATIC_POS, _STATIC_QUAT, strict=True):
        position = position + rotation @ np.asarray(offset)
        rotation = rotation @ _quat_to_rot(*quat)
    expected = position + rotation @ _HAND_POS

    pose = PandaJacobian().flange_pose(np.zeros(7))
    assert pose.shape == (3,)
    np.testing.assert_allclose(pose, expected, atol=1e-12)


def test_the_inverse_reproduces_every_twist_component_to_one_e_minus_six():
    arm = PandaJacobian()
    # The damping leaves a lambda^2 residual, so the reproduction is exact only in the limit of a
    # small twist. At 1e-4 the residual is ~1e-7, well inside the stated tolerance.
    for i in range(6):
        twist = np.zeros(6)
        twist[i] = 1e-4
        qdot = arm.joint_velocity(READY, twist)
        assert qdot.shape == (7,)
        np.testing.assert_allclose(arm.twist(READY, qdot), twist, atol=1e-6)


def test_a_translation_only_twist_comes_back_without_wrist_spin():
    arm = PandaJacobian()
    twist = np.array([1e-3, 2e-3, -1e-3, 0.0, 0.0, 0.0])
    forward = arm.twist(READY, arm.joint_velocity(READY, twist))
    # The angular rows are in the solve even though this twist asks for none of them: without
    # them the wrist absorbs command it was never given.
    assert np.max(np.abs(forward[3:])) < 1e-6
    np.testing.assert_allclose(forward[:3], twist[:3], atol=1e-5)


def test_the_kinematics_reject_the_wrong_width():
    arm = PandaJacobian()
    with pytest.raises(ValueError):
        arm.flange_pose(np.zeros(6))
    with pytest.raises(ValueError):
        arm.twist(READY, np.zeros(6))
    with pytest.raises(ValueError):
        arm.joint_velocity(READY, np.zeros(5))
    with pytest.raises(ValueError):
        arm.geometric_jacobian(READY, np.zeros(2))
