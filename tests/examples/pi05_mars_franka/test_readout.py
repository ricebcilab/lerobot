"""The session's readout as a value: the lift, the gain, the noise, the adapter and the served command.

No station, no checkpoint, no torch: a ``Config`` and arrays. These are the loop's former private
resolution restated as a public value -- ``serve`` is bridge's formula, the auto gain keeps the
corrupted command inside the clip, a fixed spec resolves to one value and a ``{random: ...}`` family to
one per (seed, episode), the ``mars-f`` adapter follows the lift or its exact inverse, and the noise is
the configured std in action-standard-deviation units or refused when there is no scale to express it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import bridge
import numpy as np
import pytest
from bridge import ARM_DIMS
from config import CommandConfig, Config, ReadoutConfig
from numpy.typing import NDArray
from readout import Readout, resolve

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"

Q0 = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
"""A DROID-like nominal pose: the lift's ``q0``."""

INTENT = np.array([0.3, -0.2, 0.4, 0.05, -0.05, 0.2, -0.1])
"""A 7-wide command in action units."""

STD8 = np.linspace(0.2, 0.9, 8)
"""An 8-wide action standard deviation: the last dim (the gripper) must be sliced away."""

SPEC = {"scale": [2.0, 1.0, 1.0]}
"""(6, 6)'s translation block doubled on x: a fixed corruption with |F| != 1."""

FAMILY = {"random": {"rotation_deg": [0.0, 90.0], "scale": [0.5, 2.0]}}
"""A per-episode corruption family: its draw is ``default_rng([seed, episode])``'s."""

TWIST_LENGTH = CommandConfig().twist_length


def _config(*, corruption=SPEC, noise: float = 0.0, seed: int = 0, gain: str | float = "auto") -> Config:
    base = Config()
    return replace(
        base,
        readout=ReadoutConfig(corruption=corruption, noise=noise, seed=seed),
        command=replace(base.command, gain=gain),
    )


def _lift(spec, *, length: float = TWIST_LENGTH) -> NDArray:
    """The fixed 7x7 the loop would lift ``spec`` to at ``Q0``."""
    return bridge.corruption_lift(bridge.matrix6_from_spec(spec), Q0, length)


# ---------------------------------------------------------------- the served command


def test_serve_is_the_bridges_gain_lift_and_one_noise_draw_clipped():
    """``serve`` is ``clip(gain * F @ intent + sigma * eps, -1, 1)`` for the generator it is handed;
    the intent is large enough that the clip is part of the answer, not decoration."""
    matrix7, sigma7, gain = _lift(SPEC), 0.3 * STD8[:ARM_DIMS], 0.7
    readout = Readout(matrix7=matrix7, gain=gain, sigma7=sigma7, adapter=None)
    served = readout.serve(INTENT * 6.0, np.random.default_rng(3))
    eps = np.random.default_rng(3).standard_normal(ARM_DIMS)
    expected = np.clip(gain * (matrix7 @ (INTENT * 6.0)) + sigma7 * eps, -1.0, 1.0)
    np.testing.assert_array_equal(served, expected)
    assert np.any(np.abs(served) == 1.0)  # the clip acted


def test_serve_draws_nothing_when_the_noise_is_off():
    """``sigma7 is None`` is "no draw", not zeros: two serves on one generator are identical and the
    generator's stream is untouched."""
    readout = Readout(matrix7=None, gain=1.0, sigma7=None, adapter=None)
    rng = np.random.default_rng(1)
    first, second = readout.serve(INTENT, rng), readout.serve(INTENT, rng)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.clip(INTENT, -1.0, 1.0))
    np.testing.assert_array_equal(rng.random(3), np.random.default_rng(1).random(3))


# ---------------------------------------------------------------- the resolved value


def test_the_auto_gain_is_the_inverse_of_the_lifts_spectral_norm():
    readout = resolve(_config(), q0=Q0, action_std7=STD8)
    np.testing.assert_array_equal(readout.matrix7, _lift(SPEC))
    np.testing.assert_allclose(readout.gain, 1.0 / float(np.linalg.norm(_lift(SPEC), 2)))
    assert readout.adapter is None


def test_a_configured_gain_is_used_instead_of_auto():
    readout = resolve(_config(gain=0.5), q0=Q0, action_std7=STD8)
    assert readout.gain == pytest.approx(0.5)


def test_the_corruption_off_serves_the_intent_unlifted_at_gain_one():
    readout = resolve(_config(corruption=None), q0=Q0, action_std7=STD8)
    assert readout.matrix7 is None
    assert readout.gain == 1.0
    assert readout.adapter is None
    np.testing.assert_array_equal(readout.serve(INTENT, np.random.default_rng(0)), np.clip(INTENT, -1, 1))


def test_a_fixed_corruption_resolves_to_the_same_value_every_time():
    first = resolve(_config(), q0=Q0, action_std7=STD8, episode=0)
    second = resolve(_config(), q0=Q0, action_std7=STD8, episode=7)  # episode is not a seed for fixed
    np.testing.assert_array_equal(first.matrix7, second.matrix7)
    assert first.gain == second.gain
    assert first.label == second.label
    np.testing.assert_array_equal(first.sigma7, second.sigma7)


def test_a_random_family_draws_from_the_seed_and_the_episode():
    """The same (seed, episode) is the same matrix; another episode is another draw; and the matrix is
    exactly ``bridge.corruption_lift(matrix6_from_spec(spec, default_rng([seed, episode])))``."""
    config = _config(corruption=FAMILY, noise=0.0, seed=5)
    zeroth = resolve(config, q0=Q0, action_std7=STD8, episode=0)
    replay = resolve(config, q0=Q0, action_std7=STD8, episode=0)
    first = resolve(config, q0=Q0, action_std7=STD8, episode=1)
    np.testing.assert_array_equal(zeroth.matrix7, replay.matrix7)
    assert not np.allclose(zeroth.matrix7, first.matrix7)
    expected = bridge.corruption_lift(
        bridge.matrix6_from_spec(FAMILY, np.random.default_rng([5, 0])), Q0, TWIST_LENGTH
    )
    np.testing.assert_array_equal(zeroth.matrix7, expected)


# ---------------------------------------------------------------- the noise


def test_the_noise_is_the_configured_std_in_action_std_units():
    readout = resolve(_config(noise=0.3), q0=Q0, action_std7=STD8)
    np.testing.assert_array_equal(readout.sigma7, 0.3 * STD8[:ARM_DIMS])


def test_the_noise_off_hands_the_readout_no_sigma_at_all():
    assert resolve(_config(noise=0.0), q0=Q0, action_std7=STD8).sigma7 is None
    assert resolve(_config(noise=0.0), q0=Q0, action_std7=None).sigma7 is None


def test_a_missing_action_std_with_noise_refuses_the_noise():
    """Noise is in action-standard-deviation units: with no std there is no unit, and drawing none
    silently would quietly change what the session ran. The error names the missing statistics."""
    with pytest.raises(ValueError, match="action statistics"):
        resolve(_config(noise=0.3), q0=Q0, action_std7=None)


# ---------------------------------------------------------------- the mars-f adapter


def test_the_adapter_is_the_lift_its_inverse_or_none():
    adapter_from_corruption = resolve(
        _config(), q0=Q0, action_std7=STD8, adapter_from_corruption="corruption"
    )
    inverse = resolve(
        _config(), q0=Q0, action_std7=STD8, adapter_from_corruption="corruption_inverse"
    ).adapter
    np.testing.assert_array_equal(adapter_from_corruption.adapter, _lift(SPEC))
    np.testing.assert_array_equal(
        inverse, bridge.lift_inverse(bridge.matrix6_from_spec(SPEC), Q0, TWIST_LENGTH)
    )
    np.testing.assert_allclose(inverse @ _lift(SPEC), np.eye(ARM_DIMS), atol=1e-12)
    assert resolve(_config(), q0=Q0, action_std7=STD8, adapter_from_corruption=None).adapter is None


def test_the_adapter_with_the_corruption_off_is_the_identity_lift():
    """A ``mars-f`` session with no corruption still holds a matrix: the identity, so the holder is
    never stale. That is the same behaviour the loop relied on."""
    readout = resolve(_config(corruption=None), q0=Q0, action_std7=STD8, adapter_from_corruption="corruption")
    np.testing.assert_array_equal(readout.adapter, np.eye(ARM_DIMS))


# ---------------------------------------------------------------- the run header


def test_a_fixed_corruption_carries_the_run_header_label():
    readout = resolve(_config(noise=0.35), q0=Q0, action_std7=STD8)
    assert readout.label == f"fixed corruption {SPEC!r} (gain {readout.gain:.4f}, noise 0.35 sigma)"


def test_the_random_family_and_the_off_state_print_nothing():
    """The loop printed a header only for a fixed spec; the label says so, and the loop prints only a
    non-empty one."""
    assert resolve(_config(corruption=FAMILY), q0=Q0, action_std7=STD8, episode=0).label == ""
    assert resolve(_config(corruption=None), q0=Q0, action_std7=STD8).label == ""


# ---------------------------------------------------------------- the import boundary


def test_importing_the_readout_needs_neither_torch_nor_c3po():
    """The readout is on ``--dry-run``'s path: bridge is numpy-only and config is a TYPE_CHECKING name."""
    code = (
        "import sys, readout; "
        "assert 'torch' not in sys.modules, 'readout imported torch'; "
        "assert 'c3po' not in sys.modules, 'readout imported c3po'; "
        "assert hasattr(readout, 'Readout') and hasattr(readout, 'resolve')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
