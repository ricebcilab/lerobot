"""The session's readout as a value: the corruption lift, the gain, the noise, and the served command.

Phase A's loop resolved the ``[readout]`` section inline (its private ``_Readout`` and
``_resolve_readout``): the corruption spec through ``bridge.matrix6_from_spec``, lifted at ``q0`` into a
fixed 7x7 ``F``, the ``auto`` gain ``1 / ||F||2`` that keeps a corrupted command inside the clip, and the
noise std in action units. Phase B's learner needs exactly that object for a state it did not collect --
the same ``F``, gain and sigma its session ran under, applied to a cue -- so this module is that object
and the pure resolution that builds it. ``loop`` keeps only *when* to resolve (once for a fixed spec, per
episode for a ``{random: ...}`` family), the noise generator, and the ``mars-f`` holder assignment.

``resolve`` is a pure function of its arguments: the config, the pose ``q0`` the lift is taken at, the
action standard deviations the noise is expressed in, the episode -- which with ``readout.seed`` is the
whole state of a random family's draw -- and the loop's ``adapter_from_corruption`` switch. It reads no
clock, no file and no loop state, so a states file's every row can be replayed with the readout its
session resolved.

Units (DESIGN.md, "Units and the signal chain"): ``F`` is dimensionless on the command (action units in,
action units out), ``gain`` scales that command, and ``sigma7`` is per-joint noise in action units.
``readout.noise`` is in action-standard-deviation units, hence the multiplication by the action std: the
caller supplies that std because only it knows whether the session has a bundle or just a statistics
file, and it may pass None when the noise is off (and then it is not even read).

Pure numpy: importing this pulls neither torch nor c3po, so it stays on ``--dry-run``'s path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import bridge
import numpy as np
from bridge import ARM_DIMS, TWIST_DIMS
from numpy.typing import NDArray

if TYPE_CHECKING:
    from config import Config

ADAPTER_SOURCES = ("corruption", "corruption_inverse")
"""How ``mars-f``'s fixed adapter follows the episode's corruption; None leaves the holder alone."""


@dataclass(frozen=True)
class Readout:
    """One episode's readout in force: ``clip(gain * matrix7 @ intent7 + sigma7 * eps, -1, 1)``.

    ``matrix7`` is the corruption lifted into command space at the session's ``q0`` (None = corruption
    off); ``gain`` is what the command is scaled by before ``F`` (the config's, or ``auto``'s
    ``1 / ||F||2`` so the corrupted command cannot clip); ``sigma7`` is the per-joint Gaussian std in
    action units (None = draw nothing); ``adapter`` is the matrix the ``mars-f`` steering policy's
    holder gets (``F``, its exact inverse, or None to leave the holder alone); ``label`` is the run
    header's corruption text for a fixed spec ("" when there is nothing fixed to print -- the random
    family redraws every episode and never had a header, so the loop prints only a non-empty label).
    """

    matrix7: NDArray | None = field(repr=False)  # F, the lifted corruption in action units; None = off
    gain: float  # the scalar the command is scaled by before F
    sigma7: NDArray | None = field(repr=False)  # per-joint Gaussian std in action units; None = off
    adapter: NDArray | None = field(repr=False)  # mars-f's fixed matrix (F or its inverse), else None
    label: str = ""  # the human-readable corruption label the loop prints, once

    def serve(self, intent7: NDArray, rng: np.random.Generator | None = None) -> NDArray:
        """(7,) served command: the corruption, the gain and one noise draw, clipped to [-1, 1].

        ``bridge.readout``'s formula unchanged -- this is a delegation, not a re-derivation, so a
        session's seeded noise sequence is what the loop drew before the extraction.
        """
        return bridge.readout(intent7, self.matrix7, self.sigma7, self.gain, rng)


def resolve(
    config: Config,
    *,
    q0: NDArray,
    action_std7: NDArray | None,
    episode: int = 0,
    adapter_from_corruption: str | None = None,
) -> Readout:
    """One episode's readout: ``F`` at ``q0``, the gain, the noise std and the ``mars-f`` adapter.

    ``q0`` is the lift's nominal pose (``command.q0`` when configured, else the reset pose the loop
    captured); it is part of what the value means even with the corruption off, so it is always passed.
    ``action_std7`` is the per-dim action standard deviation the noise is expressed in, or None when
    there is no scale (refused only if ``readout.noise`` is not zero). ``episode`` is the random
    family's whole seed state: its draw is ``default_rng([readout.seed, episode])``, so the same
    (seed, episode) is the same matrix and a different episode is a different one.
    ``adapter_from_corruption`` names the matrix ``mars-f``'s holder gets: ``F``, its exact inverse
    through ``bridge.lift_inverse``, or None to leave the holder alone. An unknown name is refused here
    as well as at the loop's construction, so a hand-built call cannot silently hold no adapter.
    """
    spec = config.readout.corruption
    random_family = isinstance(spec, dict) and set(spec) == {"random"}
    rng = np.random.default_rng([int(config.readout.seed), episode]) if random_family else None
    matrix6 = bridge.matrix6_from_spec(spec, rng) if spec is not None else np.eye(TWIST_DIMS)
    length = float(config.command.twist_length)
    # The lift is computed when either the served command or the adapter needs it; the served matrix is
    # still None when the corruption is off, as it always was.
    lift = (
        bridge.corruption_lift(matrix6, q0, length) if spec is not None or adapter_from_corruption else None
    )
    gain = _gain(config, spec, lift)
    return Readout(
        matrix7=lift if spec is not None else None,
        gain=gain,
        sigma7=_noise(config, action_std7),
        adapter=_adapter(adapter_from_corruption, matrix6, lift, q0, length),
        label=_label(spec, random_family, gain, config.readout.noise),
    )


def _adapter(
    source: str | None, matrix6: NDArray, lift: NDArray | None, q0: NDArray, length: float
) -> NDArray | None:
    """The 7x7 ``mars-f`` holds: ``F``, its exact inverse, or None; a fixed spec's F is the lift."""
    if source is None:
        return None
    if source == "corruption":
        return lift
    if source == "corruption_inverse":
        return bridge.lift_inverse(matrix6, q0, length)
    raise ValueError(
        f"adapter_from_corruption: expected None, 'corruption' or 'corruption_inverse', got {source!r}"
    )


def _gain(config: Config, spec: dict[str, Any] | None, lift: NDArray | None) -> float:
    """``command.gain``, with ``auto`` = ``1 / ||F||_2`` when the corruption is on, else 1.0."""
    gain = config.command.gain
    if gain != "auto":
        return float(gain)
    return 1.0 if spec is None or lift is None else 1.0 / float(np.linalg.norm(lift, 2))


def _noise(config: Config, action_std7: NDArray | None) -> NDArray | None:
    """``readout.noise`` in action units: the configured std times the action std, or None off.

    The action std is in the caller's hands because teleop reads it from the statistics file while the
    shared modes take it from the bundle; a session with no scale left at all (a bundle built without a
    statistics file) has no unit for the noise, and that is refused rather than silently drawing none.
    """
    noise = float(config.readout.noise)
    if not noise:
        return None
    if action_std7 is None:
        raise ValueError(
            "readout noise is in action-standard-deviation units and there are no action statistics "
            "(action_std7 is None): set [policy] stats and reload, or set [readout] noise = 0.0"
        )
    return noise * np.asarray(action_std7, dtype=np.float64)[:ARM_DIMS]


def _label(spec: dict[str, Any] | None, random_family: bool, gain: float, noise: float) -> str:
    """The run header's corruption text: the fixed spec and its resolved gain/noise, once.

    A ``{random: ...}`` family redraws at every episode and the loop never printed a header for it;
    keeping the label empty preserves that exactly.
    """
    if spec is None or random_family:
        return ""
    return f"fixed corruption {spec!r} (gain {gain:.4f}, noise {noise:g} sigma)"
