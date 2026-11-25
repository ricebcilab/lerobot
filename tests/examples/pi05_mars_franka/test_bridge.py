"""Properties of the bridge: geometry against finite differences and the reference's numbers, the
corruption lift as a similarity, the mixed units, and the readout's noise and clip.

No torch and no c3po here: numpy, pytest, and ``lerobot.teleoperators.readout`` (whose fixed matrix
forms the spec parser is checked against) only. Nothing re-derives the chain. The finite difference
tests the convention (a wrong frame or a transposed quaternion fails it), the pinned pose tests the
transcription against the previous project's validated copy, and the lift tests the algebra that
makes the corruption well-posed.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from bridge import (
    ACTION_UNIT,
    ARM_DIMS,
    TWIST_DIMS,
    action_units,
    corruption_lift,
    flange_pose,
    geometric_jacobian,
    joint_velocity,
    lift_inverse,
    matrix6_from_spec,
    nominal_pose,
    operator_command,
    readout,
)

from lerobot.teleoperators import readout as readout_module

RESET = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])
"""DROID's reset pose (``droid/robot_env.py``), via impedance_loop.py's DROID_RESET_JOINTS_7."""

FOLDED = np.array([0.3, -1.2, 0.4, -1.9, 1.1, 2.2, -0.7])
"""A second pose: a Jacobian that is only right at the reset pose is not right."""

# fmt: off
PINNED_FLANGE = np.array([0.35970336660478397, 2.364467769449871e-16, 0.48994129029098277])

PINNED_JACOBIAN = np.array(
    [
        [-2.364467769449871e-16, 0.15694129029098278, -1.4117926684250425e-16, 0.1472003632456299, -3.588173029053979e-17, 0.10700000000000022, 6.162975822039155e-33],
        [0.35970336660478397, -7.987019192796296e-17, 0.38325391242594387, -2.7268902815483815e-17, 0.07456955173858613, 1.585377451109838e-17, -1.4099117546531708e-17],
        [0.0, -0.359703366604784, -8.248348318226992e-17, 0.47869960429327263, 1.3289272958488033e-17, 0.08800000000000018, -5.065470576696889e-33],
        [0.0, 0.0, -0.5877852522924732, 2.6102908825208415e-16, 0.951056516295154, 2.610290882520842e-16, 3.8702835219370217e-16],
        [0.0, 1.0000000000000002, -4.2406746031409825e-17, -1.0000000000000007, 5.127047662811686e-16, -1.000000000000001, 3.592757177872433e-16],
        [1.0, -2.220446049250313e-16, 0.8090169943749479, -1.3723111286221176e-16, -0.3090169943749478, -1.3723111286221183e-16, -1.0000000000000009],
    ]
)
"""The Jacobian at RESET, computed by evaluating the previous project's
``examples-from-unrelated-project/inverse_kinematics.py`` (the same panda.xml chain; its own tests
check this pose's flange position against impedance_loop.py's PandaModel on the hardware station).
It pins the transcription, not the chain: a mistranscribed transform fails it."""

REFERENCE_QDOT_AT_ZERO = np.array(
    [1.4651501962998145e-17, 0.15695313035186467, 4.6138660726908685e-17, -2.3247860638697654e-04, 7.8757842543085269e-18, 0.15717310234836396, -1.0379179306963910e-18]
)
# fmt: on
"""The reference implementation's joint velocity (rad/s) for 0.11 m/s along +x at q = 0."""


@pytest.mark.parametrize("q", [RESET, FOLDED], ids=["reset", "folded"])
def test_the_linear_jacobian_is_the_differentiated_flange_pose(q):
    """Central differences with h = 1e-6 m: truncation O(h^2) * |f'''| ~ 1e-12 and roundoff
    ~ eps_machine / h ~ 2e-10, so h = 1e-6 sits at the knee and 1e-6 absolute is the reference
    test's tolerance; a wrong frame or hinge convention lands around 1e-3 and fails it."""
    jac = geometric_jacobian(q)
    step = 1e-6
    for joint in range(ARM_DIMS):
        plus, minus = q.copy(), q.copy()
        plus[joint] += step
        minus[joint] -= step
        derivative = (flange_pose(plus) - flange_pose(minus)) / (2 * step)
        np.testing.assert_allclose(jac[:3, joint], derivative, atol=1e-6)


def test_the_angular_block_is_a_unit_axis_per_joint():
    """DESIGN.md: the angular rows are the hinge axes ``z_i``; each column is therefore a unit axis."""
    jac = geometric_jacobian(RESET)
    np.testing.assert_allclose(np.linalg.norm(jac[3:], axis=0), np.ones(ARM_DIMS), atol=1e-12)


def test_the_flange_pose_at_the_reset_pose_is_the_reference_number():
    """The reference's test pins this flange position against impedance_loop.py's PandaModel."""
    np.testing.assert_allclose(flange_pose(RESET), PINNED_FLANGE, rtol=1e-9, atol=1e-12)


def test_the_jacobian_at_the_reset_pose_is_the_reference_number():
    """See PINNED_JACOBIAN: the previous project's implementation evaluated at the same pose."""
    np.testing.assert_allclose(geometric_jacobian(RESET), PINNED_JACOBIAN, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "spec",
    [
        {"rotation_z_deg": 20},
        {"scale": [2.0, 1.0, 1.0]},
        {"random": {"rotation_deg": [30, 90], "scale": [1, 4]}},
    ],
    ids=["rotation", "scale", "random"],
)
def test_the_lift_inverts_itself(spec):
    """DESIGN.md's whole argument for command-space corruption: the lift is a fixed invertible
    7x7, so ``lift_inverse`` (the same with M6^-1) undoes it."""
    matrix6 = matrix6_from_spec(spec, rng=np.random.default_rng(0))
    np.testing.assert_allclose(
        lift_inverse(matrix6, q0=RESET) @ corruption_lift(matrix6, q0=RESET), np.eye(ARM_DIMS), atol=1e-9
    )


def test_the_lift_is_the_corruption_on_the_commanded_subspace_and_identity_elsewhere():
    """DESIGN.md: ``F`` is ``M6`` on ``range(J+(q0))`` and the identity on its orthogonal
    complement. An operator only commands the row space of ``J`` (the twist image); the rest is
    the arm's redundant direction and must pass through untouched.

    On the row space the test is in twist coordinates (``J F x == M6 J x``); on the complement it
    uses the right null vector of ``J`` (the last SVD right vector)."""
    matrix6 = matrix6_from_spec({"rotation_z_deg": 20})
    lift = corruption_lift(matrix6, RESET)
    jac = geometric_jacobian(RESET)
    scale = np.diag([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
    jac_plus = np.linalg.pinv(scale @ jac) @ scale  # the lift's own right inverse
    twist = np.array([0.02, -0.01, 0.005, 0.0, 0.0, 0.0])
    commanded = jac_plus @ twist
    np.testing.assert_allclose(lift @ commanded, jac_plus @ (matrix6 @ twist), atol=1e-9)
    np.testing.assert_allclose(jac @ (lift @ commanded), matrix6 @ (jac @ commanded), atol=1e-9)

    null = np.linalg.svd(jac)[2][-1]
    np.testing.assert_allclose(lift @ null, null, atol=1e-9)


@pytest.mark.parametrize("q", [RESET, FOLDED], ids=["reset", "folded"])
def test_the_damped_inverse_tracks_a_small_twist(q):
    """``J @ qdot ~= twist`` up to the damping: each singular direction of the scaled Jacobian is
    scaled by sigma^2 / (sigma^2 + lambda^2), so the relative residual is O(lambda^2 / sigma_min^2).
    A per-component 1e-3 m/s error would already be a unit error (0.11 m/s is full deflection)."""
    twist = np.array([0.005, -0.002, 0.001, 0.004, -0.003, 0.002])
    qdot = joint_velocity(twist, q)
    residual = geometric_jacobian(q) @ qdot - twist
    assert np.linalg.norm(residual) < 1e-2 * np.linalg.norm(twist)


def test_a_pure_translation_at_q_zero_has_the_reference_signs():
    """The reference implementation's qdot for 0.11 m/s along +x at q = 0 (REFERENCE_QDOT_AT_ZERO).
    The mixed-unit damping perturbs the magnitudes, so the test pins the structure the units must
    not change: the same signs on the joints that move, and the same ~0.157 rad/s scale."""
    qdot = joint_velocity(np.array([0.11, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(ARM_DIMS))
    moving = np.abs(REFERENCE_QDOT_AT_ZERO) > 1e-6
    np.testing.assert_array_equal(np.sign(qdot[moving]), np.sign(REFERENCE_QDOT_AT_ZERO[moving]))
    carrying = np.abs(REFERENCE_QDOT_AT_ZERO) > 1e-2
    np.testing.assert_allclose(qdot[carrying], REFERENCE_QDOT_AT_ZERO[carrying], rtol=0.01)
    np.testing.assert_allclose(qdot[~moving], 0.0, atol=1e-3)


def test_action_units_are_the_plant_scale():
    """DESIGN.md: one unit = 0.2 rad per control step at 15 Hz = 3 rad/s (flow-steering's constants
    and bb8's impedance_executor, which integrates a unit as 0.2 rad per step)."""
    assert ACTION_UNIT == 3.0
    np.testing.assert_allclose(action_units(np.full(ARM_DIMS, 3.0)), np.ones(ARM_DIMS))
    np.testing.assert_allclose(action_units(np.full(ARM_DIMS, 0.6)), np.full(ARM_DIMS, 0.2))


def test_the_operator_command_is_ik_in_action_units():
    """The command is the twist's IK divided by ACTION_UNIT: one action unit moves the flange at
    0.2 rad/step, so ten percent of full speed is 0.011 m/s, not 0.11."""
    command = operator_command(np.array([0.011, 0.0, 0.0, 0.0, 0.0, 0.0]), RESET)
    twist = geometric_jacobian(RESET) @ (ACTION_UNIT * command)
    np.testing.assert_allclose(twist, np.array([0.011, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-4)


ROWS = [[1.0, 0.2, -0.3], [0.0, 2.0, 0.4], [0.5, 0.0, 1.0]]


@pytest.mark.parametrize(
    "spec",
    [
        {"rotation_z_deg": 20.0},
        {"rotation_z_deg": -35.0},
        {"scale": [2.0, 0.5, 1.0]},
        {"M": ROWS},
        ROWS,
    ],
    ids=["rotation+20", "rotation-35", "scale", "M", "bare-rows"],
)
def test_the_fixed_forms_are_exactly_the_readouts_numbers(spec):
    """Contract test, not a shape check: the corruption grammar is ``readout.matrix_spec(spec, 3)``'s,
    so a spec copied from a LIBERO config must produce exactly the same numbers here (bit-for-bit,
    ``assert_array_equal``). That equality is why this module may duplicate the two helpers instead
    of importing the readout (which would pull torch onto the ``--dry-run`` path)."""
    block = readout_module.matrix_spec(spec, 3)
    expected = np.eye(TWIST_DIMS)
    expected[:3, :3] = block
    np.testing.assert_array_equal(matrix6_from_spec(spec), expected)


def test_the_two_other_names_are_rejected_naming_the_right_key():
    """``F`` is the reversal-adapter grammar's key and ``rotation_deg`` the 2-D plane rotation; a spec
    that mixes them up must say which key this grammar uses."""
    with pytest.raises(ValueError, match="literal matrix key is 'M'"):
        matrix6_from_spec({"F": ROWS})
    with pytest.raises(ValueError, match="rotation_z_deg"):
        matrix6_from_spec({"rotation_deg": 20.0})


def test_scale_is_its_diagonal_and_only_the_translation_block():
    matrix6 = matrix6_from_spec({"scale": [2.0, 0.5, 1.0]})
    np.testing.assert_allclose(matrix6[:3, :3], np.diag([2.0, 0.5, 1.0]))
    np.testing.assert_allclose(matrix6[3:, 3:], np.eye(3))
    assert np.count_nonzero(matrix6[:3, 3:]) == 0 and np.count_nonzero(matrix6[3:, :3]) == 0


def test_a_literal_matrix_fills_the_block_and_the_rest_is_identity():
    np.testing.assert_array_equal(matrix6_from_spec({"M": ROWS})[:3, :3], ROWS)
    np.testing.assert_array_equal(matrix6_from_spec(ROWS)[:3, :3], ROWS)


def test_the_off_state_is_none_and_the_string_none_is_the_configs_job():
    """``None`` is the off state; a config's ``corruption = "none"`` is parsed to ``None`` before
    this call, so the raw string is rejected here rather than silently meaning something else."""
    np.testing.assert_array_equal(matrix6_from_spec(None), np.eye(TWIST_DIMS))
    with pytest.raises(ValueError, match="off state is None"):
        matrix6_from_spec("none")


def test_a_random_family_member_is_reproducible_and_inside_its_ranges():
    """A per-trial draw: the same seeded generator gives the same matrix, a different one does not.
    ``M = R @ diag(gains)`` with an orthogonal R, so the singular values are the drawn gains, which
    must lie inside the requested scale range."""
    spec = {"random": {"rotation_deg": [30, 90], "scale": [1, 4]}}
    first = matrix6_from_spec(spec, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(first, matrix6_from_spec(spec, rng=np.random.default_rng(0)))
    assert not np.allclose(first, matrix6_from_spec(spec, rng=np.random.default_rng(1)))
    singular = np.linalg.svd(first[:3, :3], compute_uv=False)
    assert np.all(singular >= 1.0 - 1e-12) and np.all(singular <= 4.0 + 1e-12)
    np.testing.assert_allclose(first[3:, 3:], np.eye(3))


@pytest.mark.parametrize(
    "spec",
    [
        {"curl": 1.5, "tau": 0.1},
        {"quantize": "axes"},
        {"F": ROWS},
        {"rotation_deg": 20.0},
        {"rotation_deg": 20.0, "scale": [1.0, 1.0, 1.0]},
        "spiral",
    ],
    ids=["curl", "quantize", "F-key", "rotation_deg", "two-keys", "string"],
)
def test_a_non_matrix_spec_is_rejected(spec):
    with pytest.raises(ValueError, match="matrix6_from_spec"):
        matrix6_from_spec(spec)


def test_a_random_spec_cannot_be_combined_or_malformed():
    with pytest.raises(ValueError, match="random"):
        matrix6_from_spec({"random": {"rotation_deg": [90, 30]}})
    with pytest.raises(ValueError, match="random"):
        matrix6_from_spec({"random": {"shear": [1, 2]}})
    with pytest.raises(ValueError, match="matrix6_from_spec"):
        matrix6_from_spec({"random": {"rotation_deg": [0, 90]}, "scale": [1, 1, 1]})


def test_the_readout_clips_to_the_servable_range():
    """DESIGN.md: the served command is clipped to [-1, 1]; the clip is the nonlinearity the lift
    exists to avoid, which is why gain = "auto" keeps it inactive."""
    np.testing.assert_allclose(
        readout(np.full(ARM_DIMS, 0.9), 2.0 * np.eye(ARM_DIMS), None), np.ones(ARM_DIMS)
    )
    np.testing.assert_allclose(
        readout(-np.full(ARM_DIMS, 0.9), 2.0 * np.eye(ARM_DIMS), None), -np.ones(ARM_DIMS)
    )


def test_gain_scales_the_corrupted_command_before_noise_and_clip():
    """``gain * (F @ intent7)``: a scalar commutes with the linear map, so scaling the intent is
    the same thing, and either way the clip sees the gained command."""
    intent = np.linspace(-0.3, 0.3, ARM_DIMS)
    matrix = np.eye(ARM_DIMS)[::-1]  # a permutation, so the corruption is visible
    half = readout(intent, matrix, None, gain=0.5)
    np.testing.assert_allclose(half, np.clip(0.5 * (matrix @ intent), -1.0, 1.0))
    np.testing.assert_allclose(half, readout(0.5 * intent, matrix, None))


def test_a_seeded_rng_reproduces_the_noise():
    """eps is one standard normal per joint drawn from the rng, so the same seed is the same draw."""
    intent = np.zeros(ARM_DIMS)
    sigma = np.full(ARM_DIMS, 0.1)
    expected = np.clip(sigma * np.random.default_rng(3).standard_normal(ARM_DIMS), -1.0, 1.0)
    first = readout(intent, None, sigma, rng=np.random.default_rng(3))
    np.testing.assert_allclose(first, expected)
    np.testing.assert_array_equal(first, readout(intent, None, sigma, rng=np.random.default_rng(3)))
    assert not np.allclose(first, readout(intent, None, sigma, rng=np.random.default_rng(4)))


def test_no_noise_is_drawn_when_sigma_is_none():
    """With ``noise_std7`` None the rng is not touched, so a caller's stream does not depend on
    whether noise is on."""
    generator = np.random.default_rng(0)
    state = copy.deepcopy(generator.bit_generator.state)
    np.testing.assert_allclose(
        readout(np.full(ARM_DIMS, 0.4), None, None, gain=0.5, rng=generator), np.full(ARM_DIMS, 0.2)
    )
    assert generator.bit_generator.state == state


def test_nominal_pose_reads_the_arms_joint_position():
    obs = {
        "follower/joint_position": np.arange(ARM_DIMS, dtype=np.float64),
        "follower/gripper_position": np.array([0.5]),
    }
    np.testing.assert_array_equal(nominal_pose(obs, "follower"), np.arange(ARM_DIMS, dtype=np.float64))
    with pytest.raises(ValueError, match="leader/joint_position"):
        nominal_pose(obs, "leader")
    with pytest.raises(ValueError, match="joint_position"):
        nominal_pose({"follower/joint_position": np.zeros(ARM_DIMS - 1)}, "follower")


def test_the_example_modules_import_without_torch_or_c3po():
    """DESIGN.md: ``--dry-run`` never imports torch; bridge is on that path, so it must stay pure
    (which is why the spec helpers are re-derived rather than imported from the readout)."""
    example_dir = str(Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka")
    code = (
        "import sys, bridge, config; "
        "assert 'torch' not in sys.modules, 'a module imported torch'; "
        "assert 'c3po' not in sys.modules, 'a module imported c3po'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": example_dir},
    )
