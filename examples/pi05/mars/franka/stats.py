"""The checkpoint's action statistics: openpi's DROID quantiles installed on the QUANTILES steps.

pi0.5-DROID's published pre/post-processor JSONs carry no features and no statistics: the cached
configs are ``features: {}`` and the ``normalization_mapping`` says ``QUANTILES`` for STATE and ACTION,
so the q01/q99 statistics have to come from openpi's ``assets/droid/norm_stats.json`` before the
pipelines mean anything (``DESIGN.md``, "What the checkpoint's own code settles"). ``install_stats``
puts them where lerobot's normalization mixin reads them and returns the 8-wide action (mean, std) the
steering cores use; ``action_std`` reads the same file's action quantiles as a per-dim standard
deviation, for teleop's readout noise (teleop must not load the checkpoint).

Where the steps read them (``src/lerobot/processor/normalize_processor.py``):

- ``_NormalizationMixin._normalize_observation`` (lines 266-285) walks the step's ``features`` dict and
  normalizes every observation key in it, so ``observation.state`` needs a feature entry; the cached
  JSONs' ``features: {}`` is why the installer populates it.
- ``_normalize_action`` (lines 287-301) reaches the action through ``_apply_transform`` (lines 302-401),
  whose QUANTILES branch (lines 386-401) reads only ``q01``/``q99`` from ``stats["action"]``: no feature
  entry is needed for the action, but both keys are declared so each step's feature set is the
  checkpoint's 32-wide one.
- Assigning ``step.stats`` is not enough: the mixin keeps ``_tensor_stats`` from construction
  (``hotswap_stats``, lines 559-584, rebuilds it). ``_install_on`` assigns the dicts and calls the
  mixin's ``to`` (lines 156-174), which re-derives ``_tensor_stats`` from them.

The meaningful width is the file's A5 contract: dims 0-7 meaningful (the arm's joints and the gripper),
zero after. It is the first index whose q01 == q99 == 0; it must equal ``dims``, nothing may be
non-zero beyond it, and the file is 32 wide (the checkpoint's padded state/action vectors), so a file
that disagrees is refused rather than silently normalizing a different action space.

This module imports numpy only at module level: the lerobot feature/step types it checks and the core
whose statistics conversion it uses are imported inside ``install_stats``, so importing it pulls no
torch (``policy.py``, which is on ``--dry-run``'s path, imports it for that reason).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

CHECKPOINT_WIDTH = 32
"""pi0.5-DROID's padded state/action width; the statistics are installed at this width."""

STATION_DIMS = 8
"""The Franka's action width, which is also the statistics' meaningful width: seven joints and the gripper."""


def install_stats(
    preprocessor: Any, postprocessor: Any, stats_path: Path, dims: int = STATION_DIMS
) -> tuple[NDArray, NDArray]:
    """Install openpi's DROID statistics on the steps that read them; return the 8-wide action (mean, std).

    ``stats_path`` is ``assets/droid/norm_stats.json``: ``{"norm_stats": {"state": {...}, "actions":
    {...}}}``, each group ``mean``/``std``/``q01``/``q99`` 32 wide, meaningful (not q01 == q99 == 0) in
    the first ``dims`` dims and zero after. The checkpoint's ``normalization_mapping`` is ``QUANTILES``
    for STATE and ACTION, so a ``NormalizerProcessorStep``/``UnnormalizerProcessorStep`` reads the
    group's **q01/q99** (never mean/std) for the keys it sees: the pre-processor's normalizer gets both
    ``observation.state`` and ``action``, the post-processor's unnormalizer gets ``action``. A step only
    touches the keys in its ``features`` dict (and the ACTION key through ``stats`` alone), and the
    checkpoint's JSONs ship ``features: {}``, so the feature entries are added too and the step's tensor
    statistics are rebuilt.

    The ``(mean, std)`` returned for the steering cores is the affine form of that same quantile map,
    ``(q01 + q99) / 2`` and ``(q99 - q01) / 2`` (the cores' own conversion,
    ``FlowControlSteering.action_mean_std``, is what computes it; this does not re-derive the map).
    """
    path = Path(stats_path)
    groups = _groups(path)

    state = _quantile_pair(groups, "state", path, dims)
    actions = _quantile_pair(groups, "actions", path, dims)

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    stats = {
        "observation.state": {"q01": state["q01"], "q99": state["q99"]},
        "action": {"q01": actions["q01"], "q99": actions["q99"]},
    }
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(CHECKPOINT_WIDTH,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(CHECKPOINT_WIDTH,)),
    }
    normalizers = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    unnormalizers = [step for step in postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)]
    if not normalizers:
        raise ValueError(f"{path}: the pre-processor has no NormalizerProcessorStep to install statistics on")
    if not unnormalizers:
        raise ValueError(
            f"{path}: the post-processor has no UnnormalizerProcessorStep to install statistics on"
        )
    for step in (*normalizers, *unnormalizers):
        _install_on(step, stats, features)

    from lerobot.policies.steering import FlowControlSteering

    mean, std = FlowControlSteering.action_mean_std(postprocessor)
    return mean[:dims], std[:dims]


def action_std(stats_path: Path, dims: int = STATION_DIMS) -> NDArray:
    """The per-dim action standard deviation of ``norm_stats.json``, the first ``dims`` dims.

    ``max((q99 - q01) / 2, 1e-6)``: the same affine form of the QUANTILES map that ``install_stats``
    returns from the post-processor, read from the file directly. Teleop's readout noise is defined in
    action-standard-deviation units, and teleop must not load the checkpoint (nor torch), so the loop
    reads the statistics here instead. The quantiles are the 32-wide file's, width-checked by
    ``_quantile_pair``; the 1e-6 floor keeps a constant dim from dividing by zero downstream.
    """
    dims = _width(dims)
    path = Path(stats_path)
    actions = _quantile_pair(_groups(path), "actions", path, STATION_DIMS)
    return np.maximum((actions["q99"] - actions["q01"]) / 2.0, 1e-6)[:dims]


def _groups(path: Path) -> dict:
    """The ``norm_stats`` table of the statistics file, parsed once for every reader."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    groups = data.get("norm_stats") if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        raise ValueError(f"{path}: expected a 'norm_stats' table, got {type(groups).__name__}")
    return groups


def _width(dims: int) -> int:
    """The number of per-dim statistics a caller asks for: between one and the station's 8."""
    if isinstance(dims, bool) or not isinstance(dims, int) or not 0 < dims <= STATION_DIMS:
        raise ValueError(f"dims: expected an integer in (0, {STATION_DIMS}], got {dims!r}")
    return dims


def _quantile_pair(groups: dict, name: str, path: Path, dims: int) -> dict[str, NDArray]:
    """One group of ``norm_stats`` (``state`` or ``actions``): its 32-wide q01/q99, width-checked."""
    group = groups.get(name)
    if not isinstance(group, dict):
        raise ValueError(f"{path}: expected a 'norm_stats.{name}' table, got {type(group).__name__}")
    keys = ("mean", "std", "q01", "q99")
    missing = [key for key in keys if key not in group]
    if missing:
        raise ValueError(f"{path}: norm_stats.{name} is missing {', '.join(missing)}")
    arrays = {key: np.asarray(group[key], dtype=np.float64).reshape(-1) for key in keys}
    widths = sorted({array.shape[0] for array in arrays.values()})
    if widths != [CHECKPOINT_WIDTH]:
        raise ValueError(
            f"{path}: norm_stats.{name} must be {CHECKPOINT_WIDTH} wide, got widths {widths} "
            "(pi0.5-DROID's padded state and action vectors)"
        )
    q01, q99 = arrays["q01"], arrays["q99"]
    # The meaningful width is the first index whose q01 == q99 == 0 (the pad dims); nothing beyond the
    # expected width may be non-zero (dims 0-7 are the arm and the gripper).
    zero = (q01 == 0) & (q99 == 0)
    meaningful = int(np.argmax(zero)) if zero.any() else CHECKPOINT_WIDTH
    if meaningful != dims or bool((~zero[dims:]).any()):
        raise ValueError(
            f"{path}: norm_stats.{name} is meaningful in {meaningful} dims (zeros from index {meaningful}); "
            f"expected {dims} meaningful dims and zeros beyond"
        )
    return {"q01": q01, "q99": q99}


def _install_on(step: Any, stats: dict, features: dict) -> None:
    """Merge the two keys into a normalization step and rebuild its tensor statistics from them."""
    step.stats = {**(step.stats or {}), **stats}
    step.features = {**(step.features or {}), **features}
    step.to()  # _NormalizationMixin.to: re-derives _tensor_stats from .stats
