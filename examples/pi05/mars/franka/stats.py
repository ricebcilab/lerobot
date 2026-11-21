"""The checkpoint's normalization statistics, which `lerobot/pi05_droid` does not ship.

The published DROID port carries `config.json`, `model.safetensors` and the two processor JSONs, and its
processor JSONs list `features: {}`. There are no `policy_*_step_*.safetensors` beside them, which is where a
finetuned checkpoint (the LIBERO one) keeps its statistics, and `config.json` has no `dataset_stats`. Left
alone, the normalizer and unnormalizer pass state and action through untransformed, and
`FlowControlSteering.action_mean_std` raises, so FRS cannot build its reference.

OpenPI keeps the same numbers in the checkpoint's `assets/droid/norm_stats.json`. This module reads that file
and turns it into the `stats` dict a LeRobot normalizer takes, in the shape `LeRobotDataset.meta.stats` has
(`{feature: {stat: array}}`), so the two pipelines can be built with it as an override:

    pre, post = make_pre_post_processors(
        policy_cfg, pretrained_path=policy_path,
        preprocessor_overrides={"normalizer_processor": {"stats": stats}, ...},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats}},
    )

`_NormalizationMixin` preserves explicitly given statistics through `load_state_dict`, which is why the
override survives the checkpoint's empty ones.

The feature names and the statistics that matter: the pipeline normalizes with `QUANTILES`, so `q01`/`q99`
are the ones that decide the map, and `mean`/`std`/`min`/`max` are carried for the other modes and for
`action_mean_std`'s affine reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

STATE = "observation.state"
ACTION = "action"
"""The two features a DROID policy normalizes, in LeRobot's names."""

_OPENPI_KEYS = {"state": STATE, "actions": ACTION, "action": ACTION}
"""OpenPI's keys for the same two features (it says `actions`; older exports say `action`)."""

_STATS = ("mean", "std", "min", "max", "q01", "q99")
"""The statistics a normalizer reads. `q10`/`q50`/`q90` are not needed by any mode the DROID policy uses."""


def read_openpi_stats(path: str | Path) -> dict[str, dict[str, Any]]:
    """OpenPI's `norm_stats.json` as a LeRobot `stats` dict.

    Accepts the exported form (`{"norm_stats": {...}}`) and a bare mapping. Every feature needs `q01` and
    `q99`: they are what the quantile normalizer applies, and a file without them would silently normalize
    with zeros. `mean`/`std` are derived from the two quantiles when absent, which is the affine map
    `action_mean_std` would compute anyway.
    """
    raw = json.loads(Path(path).read_text())
    block = raw.get("norm_stats", raw)
    if not isinstance(block, dict):
        raise ValueError(f"{path}: expected a mapping of features, got {type(block).__name__}")

    stats: dict[str, dict[str, Any]] = {}
    for key, feature in block.items():
        name = _OPENPI_KEYS.get(key)
        if name is None:
            continue
        if not isinstance(feature, Mapping):
            raise ValueError(f"{path}: {key!r} must map a statistic's name to its values")
        missing = [stat for stat in ("q01", "q99") if stat not in feature]
        if missing:
            raise ValueError(f"{path}: {key!r} has no {', '.join(missing)}; a quantile normalizer needs both")
        entry = {stat: np.asarray(feature[stat], dtype=np.float64) for stat in _STATS if stat in feature}
        lo, hi = entry["q01"], entry["q99"]
        if lo.shape != hi.shape:
            raise ValueError(f"{path}: {key!r} has q01 {lo.shape} but q99 {hi.shape}")
        entry.setdefault("mean", (lo + hi) / 2)
        entry.setdefault("std", np.maximum((hi - lo) / 2, 1e-6))
        entry.setdefault("min", lo)
        entry.setdefault("max", hi)
        stats[name] = entry

    if set(stats) != {STATE, ACTION}:
        raise ValueError(
            f"{path}: expected statistics for both {STATE!r} and {ACTION!r}, found {sorted(stats)} "
            f"(OpenPI's keys are {sorted(_OPENPI_KEYS)})"
        )
    return stats


def overrides(
    stats: Mapping[str, Mapping[str, Any]], rename_map: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The pre- and post-processor overrides that install `stats` on a checkpoint's pipelines.

    The pre-processor's rename step takes the camera map in the same override, so a caller passes both at
    once and cannot install the statistics without the names they belong to.
    """
    pre: dict[str, Any] = {"normalizer_processor": {"stats": dict(stats)}}
    if rename_map is not None:
        pre["rename_observations_processor"] = {"rename_map": dict(rename_map)}
    post: dict[str, Any] = {"unnormalizer_processor": {"stats": dict(stats)}}
    return pre, post


def action_width(stats: Mapping[str, Mapping[str, Any]]) -> int:
    """How many action dims the statistics describe: the checkpoint's real action width, not its padding."""
    return int(np.asarray(stats[ACTION]["q01"]).shape[0])


__all__ = ["ACTION", "STATE", "action_width", "overrides", "read_openpi_stats"]
