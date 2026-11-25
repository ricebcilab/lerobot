"""The pilot's cues: the Cartesian stick pushes the user module is trained and validated on.

A cue is what a person would command in a state: a direction and a speed, nothing else. The study's
operator commands translation only (DESIGN.md, decision 4), so a cue is a 3-D push -- a direction
uniform on the unit sphere times a speed the pilot actually used -- with the rotation axes zero. Two
functions keep the two spaces apart:

- ``cue_twists`` draws the cue in physical units (m/s, rad/s): the *intention*, expressible without a
  pose, and the quantity the validator compares an executed joint velocity against (through ``J(q)``).
- ``cue_commands`` pushes each twist through the operator's own ``bridge.operator_command`` at the
  state's ``q``: the *intention at a pose*, in the seven action-unit joint velocities a stick can
  actually command (the six-dimensional image of the Jacobian), not an arbitrary 7-vector that the
  readout and the policy would never see.

``magnitudes`` recovers the pilot's speeds from their recorded deflections, so the cue's speeds are the
study's own command-length distribution rather than a guess (uniform over the observed ones, or over a
fallback range when the pilot never pushed).

Pure numpy, no torch, no lerobot (``bridge`` is numpy-only), so a trainer's offline test -- and this
module's -- runs without a checkpoint or a station. No function draws unless given an ``rng`` as an
argument: a cue set is reproducible from the trainer's seed.
"""

from __future__ import annotations

import bridge
import numpy as np
from bridge import ARM_DIMS, TWIST_DIMS
from numpy.typing import NDArray

DEFAULT_FALLBACK = (0.2, 0.6)
"""The default command-length range (m/s) drawn when the pilot never pushed: a stand-in, not a measurement."""


def magnitudes(
    deflections: NDArray,
    translation_gain: float,
    fallback: tuple[float, float] = DEFAULT_FALLBACK,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """(N,) translational command lengths in m/s: ``|deflection[:3]| * translation_gain`` per row.

    ``deflections`` is a pilot recording's (N, 6) deflection rows, each axis in [-1, 1]; the first
    three are the translation the study's operator commands, and the rotation axes are ignored. A
    pilot who never pushed leaves every length zero -- no distribution to draw a speed from -- and then
    one uniform draw per row over ``fallback`` (m/s) stands in, so the pool keeps the pilot's row count.
    An empty pilot stays empty: there are no rows to aim a cue from, and ``cue_twists`` refuses it.
    """
    array = np.asarray(deflections, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != TWIST_DIMS:
        raise ValueError(f"deflections must be (N, {TWIST_DIMS}), got {array.shape}")
    gain = _positive_or_zero(translation_gain, "translation_gain")
    lengths = np.linalg.norm(array[:, :3], axis=1) * gain
    if lengths.size and not np.any(lengths > 0):
        lengths = _generator(rng).uniform(*_range(fallback), size=lengths.size)
    return lengths


def cue_twists(count: int, magnitudes: NDArray, rng: np.random.Generator) -> NDArray:
    """(count, 6) cue twists in (m/s, rad/s): a unit-sphere translation direction at a pool speed.

    The speed is drawn uniformly from ``magnitudes`` -- the pool ``magnitudes()`` returned, i.e. the
    pilot's observed command lengths (or the fallback's draws when the pilot never pushed). The
    rotation rows are zero because the study's operator commands translation only; a cue is as real as
    a person's push, and a person here pushes the three translation axes.
    """
    pool = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if pool.size == 0:
        raise ValueError("magnitudes is empty: there is no pilot speed to draw a cue from")
    steps = _count(count)
    generator = _generator(rng)
    speeds = generator.choice(pool, size=steps)
    directions = generator.normal(size=(steps, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    twists = np.zeros((steps, TWIST_DIMS))
    twists[:, :3] = directions * speeds[:, None]
    return twists


def cue_commands(twists: NDArray, q: NDArray, length: float) -> NDArray:
    """(count, 7) action units: each cue twist through ``bridge.operator_command`` at ``q``.

    This is the operator's own IK (DESIGN.md's signal chain, "the same bridge the operator uses"), so a
    cue lies in the subspace a stick can actually command at that pose. ``length`` is the mixed-unit
    twist length, ``command.twist_length``: metres and radians enter the solve with the same weight.
    """
    array = np.asarray(twists, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != TWIST_DIMS:
        raise ValueError(f"twists must be (count, {TWIST_DIMS}), got {array.shape}")
    pose = np.asarray(q, dtype=np.float64)
    if pose.shape != (ARM_DIMS,):
        raise ValueError(f"q must be ({ARM_DIMS},), got {pose.shape}")
    scale = _positive(length, "length")
    if array.shape[0] == 0:
        return np.empty((0, ARM_DIMS))
    return np.stack([bridge.operator_command(twist, pose, scale) for twist in array])


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    """The generator given, or numpy's default one (fresh entropy per call when none is given)."""
    return np.random.default_rng() if rng is None else rng


def _count(value: int) -> int:
    """A cue count: a whole number of cues, zero allowed (an empty batch)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"count must be a non-negative integer, got {value!r}")
    return value


def _range(fallback: tuple[float, float]) -> tuple[float, float]:
    """``fallback`` as an ordered (low, high) pair of non-negative finite speeds."""
    try:
        low, high = (float(value) for value in fallback)
    except (TypeError, ValueError) as e:
        raise ValueError(f"fallback must be a [low, high] pair of speeds in m/s, got {fallback!r}") from e
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low <= high):
        raise ValueError(f"fallback must be a [low, high] pair with 0 <= low <= high, got {fallback!r}")
    return low, high


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _positive_or_zero(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def _positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if not number > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number
