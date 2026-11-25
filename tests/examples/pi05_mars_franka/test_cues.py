"""The cues: the pilot's command lengths, the twist draw and the stick's own IK.

No torch, no checkpoint, no station: numpy and the pure bridge. These are the properties the trainer
and the validator rely on -- the lengths are the recorded stick pushes with the fallback when the pilot
never pushed, a cue is a unit-sphere translation direction at a pool speed with no rotation and is
reproducible from its generator, and a cue command is exactly ``bridge.operator_command`` at the state's
``q`` (the space a stick can actually command).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import bridge
import numpy as np
import pytest
from bridge import ARM_DIMS, TWIST_DIMS
from config import CommandConfig
from cues import cue_commands, cue_twists, magnitudes

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"

Q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
"""A DROID-like state's joint angles: the IK's ``q``."""

GAIN = CommandConfig().translation_gain
LENGTH = CommandConfig().twist_length


def _deflections(pushes: list[tuple[float, float, float]]) -> np.ndarray:
    """Pilot rows from translation pushes; the rotation axes are filled with a nonzero value so a test
    can prove they are ignored (the study commands translation only)."""
    rows = [[x, y, z, 0.5, -0.25, 0.75] for x, y, z in pushes]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------- the pilot's lengths


def test_magnitudes_recovers_the_pilots_translation_lengths():
    deflections = _deflections([(0.2, 0.0, 0.0), (0.0, -0.5, 0.0), (0.3, 0.0, 0.4)])
    lengths = magnitudes(deflections, GAIN)
    expected = np.linalg.norm(deflections[:, :3], axis=1) * GAIN
    np.testing.assert_allclose(lengths, expected)
    np.testing.assert_allclose(lengths, [0.2 * GAIN, 0.5 * GAIN, 0.5 * GAIN])  # rotation ignored


def test_magnitudes_falls_back_inside_the_range_when_the_pilot_never_pushed():
    zeros = np.zeros((5, TWIST_DIMS))
    lengths = magnitudes(zeros, GAIN, fallback=(0.2, 0.6), rng=np.random.default_rng(0))
    assert lengths.shape == (5,)
    assert np.all(lengths >= 0.2) and np.all(lengths <= 0.6)
    assert not np.all(lengths == 0.0)
    replay = magnitudes(zeros, GAIN, fallback=(0.2, 0.6), rng=np.random.default_rng(0))
    np.testing.assert_array_equal(lengths, replay)


# ---------------------------------------------------------------- the twist draw


def test_cue_twists_draws_a_unit_translation_direction_and_no_rotation():
    twists = cue_twists(40, np.ones(4), np.random.default_rng(0))
    assert twists.shape == (40, TWIST_DIMS)
    np.testing.assert_allclose(np.linalg.norm(twists[:, :3], axis=1), np.ones(40), atol=1e-12)
    np.testing.assert_array_equal(twists[:, 3:], np.zeros((40, 3)))


def test_cue_twists_draws_its_speeds_from_the_pool_and_replays_from_the_seed():
    pool = np.array([0.2, 0.4, 0.6])
    one = cue_twists(20, pool, np.random.default_rng(11))
    two = cue_twists(20, pool, np.random.default_rng(11))
    np.testing.assert_array_equal(one, two)
    speeds = np.linalg.norm(one[:, :3], axis=1)
    assert np.all(speeds >= pool.min() - 1e-12) and np.all(speeds <= pool.max() + 1e-12)
    # uniform over the observed ones: every drawn speed is one the pilot made
    for speed in speeds:
        assert np.min(np.abs(pool - speed)) < 1e-12
    other = cue_twists(20, pool, np.random.default_rng(12))
    assert not np.allclose(one, other)


def test_cue_twists_refuses_an_empty_speed_pool():
    with pytest.raises(ValueError, match="empty"):
        cue_twists(2, np.array([]), np.random.default_rng(0))


# ---------------------------------------------------------------- the operator's IK


def test_cue_commands_is_the_bridges_operator_command_at_the_states_q():
    twists = cue_twists(5, np.array([0.3]), np.random.default_rng(2))
    commands = cue_commands(twists, Q, LENGTH)
    assert commands.shape == (5, ARM_DIMS)
    for twist, command in zip(twists, commands, strict=True):
        np.testing.assert_array_equal(command, bridge.operator_command(twist, Q, LENGTH))


def test_cue_commands_refuses_a_q_that_is_not_the_arms_shape():
    twists = np.zeros((2, TWIST_DIMS))
    with pytest.raises(ValueError, match=rf"q must be \({ARM_DIMS},\)"):
        cue_commands(twists, np.zeros(TWIST_DIMS), LENGTH)


def test_cue_commands_refuses_a_twist_batch_that_is_not_six_wide():
    with pytest.raises(ValueError, match=rf"\(count, {TWIST_DIMS}\)"):
        cue_commands(np.zeros((2, 5)), Q, LENGTH)


# ---------------------------------------------------------------- the import boundary


def test_importing_the_cues_needs_neither_torch_nor_lerobot():
    """The cues are on the trainer's and validator's offline path: bridge is numpy-only, nothing here
    touches the policy stack."""
    code = (
        "import sys, cues; "
        "assert 'torch' not in sys.modules, 'cues imported torch'; "
        "assert 'lerobot' not in sys.modules, 'cues imported lerobot'; "
        "assert hasattr(cues, 'cue_commands')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
