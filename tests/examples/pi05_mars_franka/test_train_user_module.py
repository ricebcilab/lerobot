"""The Franka user-module trainer's pure arithmetic: the chunk windows, the two command maps, the readout.

A label crosses three spaces before the regression sees it: the policy's joint-velocity chunk, the
operator's translation push, and the joint command the steering core steers toward. The tests pin the
two maps between them (including the exact inverse on a non-singular pose) and the readout's
reproducibility, none of which needs a checkpoint, a dataset or an arm.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from jacobian import PandaJacobian

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
TRAIN_DIR = EXAMPLE_DIR / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

user_module = importlib.import_module("user_module")

READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
"""panda.xml's ``home`` keyframe: the arm forward and bent, well away from a singular pose."""

TWIST_FULL = 0.11  # m/s at a full deflection
QDOT_FULL = 0.5  # rad/s at a full deflection


class _FixedTwist:
    """A jacobian port with one fixed flange twist: enough to pin what the command maps read of it."""

    def __init__(self, twist):
        self.twist_value = np.asarray(twist, dtype=np.float64)
        self.widths: list[tuple] = []

    def twist(self, q, qdot):
        self.widths.append(np.asarray(qdot).shape)
        return self.twist_value


def _realizable_label(arm: PandaJacobian, q: np.ndarray, seed: int = 0) -> np.ndarray:
    """A label the translation-only command can carry exactly: no flange rotation and no null-space part.

    A command has three translation dims, so it cannot express a label that spins the flange
    (``J_w L = 0`` is what makes step 5's zero angular request consistent) nor the one joint
    direction that changes nothing (the minimum-norm inverse drops ``null(J)``). Projected onto
    both, the round trip label -> command -> label is exact, which is what the inverse test pins.
    """
    jacobian = arm.geometric_jacobian(q, arm.flange_pose(q))
    angular = jacobian[3:]
    w = np.random.default_rng(seed).normal(size=7)
    no_rotation = w - angular.T @ np.linalg.solve(angular @ angular.T, angular @ w)
    in_row_space = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T, jacobian @ no_rotation)
    return in_row_space / np.linalg.norm(in_row_space) * 0.01


# ---------------------------------------------------------------- the chunk boundaries


@pytest.mark.parametrize(
    ("n_frames", "n_action_steps", "expected"),
    [
        (0, 5, []),
        (1, 5, [0]),
        (5, 5, [0]),
        (9, 4, [0, 4, 8]),
        (15, 15, [0]),
        (31, 15, [0, 15, 30]),
    ],
)
def test_windows_are_the_chunk_boundaries(n_frames, n_action_steps, expected):
    assert user_module.windows(n_frames, n_action_steps) == expected


def test_windows_reject_a_step_that_cannot_advance():
    with pytest.raises(ValueError, match="n_action_steps"):
        user_module.windows(10, 0)
    with pytest.raises(ValueError, match="n_frames"):
        user_module.windows(-1, 3)


# ---------------------------------------------------------------- step 3: the operator's push


def test_label_command_reads_the_translation_of_the_arm_jacobian_only():
    # The angular part is absurd on purpose: if it leaked into the command, the assertion below fails.
    jacobian = _FixedTwist([0.022, -0.033, 0.011, 500.0, -500.0, 500.0])
    label = np.linspace(-0.2, 0.2, 7)
    command = user_module.label_command(jacobian, np.zeros(7), label, TWIST_FULL)
    np.testing.assert_allclose(command, np.array([0.022, -0.033, 0.011]) / TWIST_FULL)
    assert jacobian.widths == [(7,)]  # only the seven joints reach the Jacobian

    # An eight-number action's gripper must not reach it either.
    np.testing.assert_array_equal(
        user_module.label_command(jacobian, np.zeros(7), np.append(label, 0.9), TWIST_FULL), command
    )


def test_label_command_clips_an_overdriven_push():
    jacobian = _FixedTwist([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    command = user_module.label_command(jacobian, np.zeros(7), np.zeros(8), TWIST_FULL)
    np.testing.assert_array_equal(command, [1.0, -1.0, 0.0])


# ---------------------------------------------------------------- step 5: the reference back


def test_reference_command_inverts_label_command_on_a_realizable_label():
    arm = PandaJacobian()
    label = _realizable_label(arm, READY)

    command = user_module.label_command(arm, READY, label, TWIST_FULL)
    assert np.all(np.abs(command) < 1.0)  # a small label, so no clipping confuses the round trip
    recovered = user_module.reference_command(arm, READY, command, TWIST_FULL, QDOT_FULL)

    np.testing.assert_allclose(recovered, label / QDOT_FULL, atol=1e-6)


def test_reference_command_holds_the_flange_orientation_still():
    arm = PandaJacobian()
    command = user_module.reference_command(arm, READY, np.array([0.5, -0.2, 0.1]), TWIST_FULL, QDOT_FULL)
    twist = arm.twist(READY, command * QDOT_FULL)
    # The angular rows are part of the solve, so the reference asks for no wrist spin.
    np.testing.assert_allclose(twist[3:], 0.0, atol=1e-12)


def test_reference_command_clips_a_full_push_to_a_command():
    arm = PandaJacobian()
    command = user_module.reference_command(arm, READY, np.full(3, 100.0), TWIST_FULL, QDOT_FULL)
    assert np.all(np.abs(command) <= 1.0)
    assert np.max(np.abs(command)) == 1.0


# ---------------------------------------------------------------- step 4: the readout


def test_served_command_is_seeded_and_an_idle_push_stays_idle():
    push = np.array([0.4, -0.2, 0.1])
    first = user_module.served_command(push, 0.05, seed=7)
    second = user_module.served_command(push, 0.05, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, user_module.served_command(push, 0.05, seed=8))
    np.testing.assert_array_equal(user_module.served_command(np.zeros(3), 0.3, seed=7), np.zeros(3))


# ---------------------------------------------------------------- the CLI's two files


def _argv(tmp_path, dataset, stats) -> list[str]:
    return [
        "--dataset",
        str(dataset),
        "--policy",
        "lerobot/pi05_droid",
        "--stats",
        str(stats),
        "--out",
        str(tmp_path / "out"),
    ]


def test_the_cli_refuses_a_missing_dataset(tmp_path, capsys):
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")  # the path check only; the file is read when the model is built
    with pytest.raises(SystemExit):
        user_module.parse_args(_argv(tmp_path, tmp_path / "no_such_dataset", stats))
    assert "--dataset" in capsys.readouterr().err


def test_the_cli_refuses_a_missing_stats_file(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(SystemExit):
        user_module.parse_args(_argv(tmp_path, dataset, tmp_path / "no_such_stats.json"))
    assert "--stats" in capsys.readouterr().err
