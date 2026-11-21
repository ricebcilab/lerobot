#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The run's artefacts: one trial's per-step rows, and the identity a trial is paired on.

The `.npz` is LIBERO's shape (`action`, `user_translation`, `user_translation_raw`, `user_gripper`,
`user_reads`, `policy_translation`, `joint_pos`, `t`), so `../libero/notebooks/analyze.py` reads a Franka
trial without a branch. The reset hash is what lets the report refuse to pair two arms' trials that did not
start from the same scene; the pair id is what they pair on when they did.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from arms import _adapter_label, _matrix_list
from batches import DROID_IMAGE_KEYS

__all__ = [
    "TrialRecorder",
    "flat_settings",
    "initial_state_hash",
    "nan_matrix",
    "pair_id_for",
    "provenance",
]

DROID_N_ACTION_STEPS = 15
"""pi05_droid's chunk length, recorded when neither the file nor the checkpoint names one."""

MODE_LABELS = {
    "teleop": "teleop",
    "floor": "policy",
    "ceiling": "policy",
    "fc": "shared_flow_control",
    "frs": "shared_flow_reversal_steering",
    "mars_fixed": "shared_flow_reversal_steering",
    "mars_learned": "shared_flow_reversal_steering",
}
"""The arm in LIBERO's `mode` vocabulary, which `../libero/notebooks/analyze.py`'s `method_of` reads."""


def initial_state_hash(obs: Mapping[str, np.ndarray], arm: str, cameras: Sequence[str]) -> str:
    """The reset's digest: the joint pose to four decimals, then every camera frame's bytes.

    The pose is rounded because the station's encoders and the config's start pose need not agree to the last
    bit; the frames are digested as they arrive, because a different scene is exactly what the pairing must
    not hide. Cheap: one pass over each frame, once per trial.
    """
    digest = hashlib.sha256()
    q = np.asarray(obs[f"{arm}/joint_position"], dtype=np.float64)
    digest.update(np.round(q, 4).tobytes())
    for name in cameras:
        digest.update(np.ascontiguousarray(np.asarray(obs[name])).tobytes())
    return digest.hexdigest()


def pair_id_for(seed: int, trial: int) -> str:
    """The identity two arms' trials pair on: the run's seed and the trial's index, not the reset.

    The reset is not in the id, because pairing happens across arms and the check that they matched is the
    `initial_state_hash`; folding the hash into the id would make a mismatched reset unpairable instead of
    visibly unmatched.
    """
    return hashlib.sha256(
        json.dumps({"seed": int(seed), "trial": int(trial)}, sort_keys=True).encode()
    ).hexdigest()


class TrialRecorder:
    """Collects one trial's per-step rows and writes them as a .npz in LIBERO's shape."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.start = time.time()

    @property
    def total_reads(self) -> int:
        """How many times the operator's input was read during this trial."""
        return int(sum(row["user_reads"] for row in self.rows))

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def add(
        self,
        *,
        q: np.ndarray,
        action: np.ndarray,
        plan: np.ndarray,
        served: np.ndarray,
        raw: np.ndarray,
        gripper: float,
        reads: int,
    ) -> None:
        """One station tick's row: what was executed, what the operator asked for and what the policy wanted."""
        self.rows.append(
            {
                "t": self.elapsed,
                "action": np.asarray(action, dtype=np.float32),
                "policy_translation": np.asarray(plan, dtype=np.float32)[:3],
                "user_translation": np.asarray(served, dtype=np.float32)[:3],
                "user_translation_raw": np.asarray(raw, dtype=np.float32)[:3],
                "user_gripper": float(gripper),
                "user_reads": int(reads),
                "joint_pos": np.asarray(q, dtype=np.float32),
            }
        )

    def save(self, path: Path, **scalars: Any) -> None:
        """Stack the rows into arrays and write them with the trial's scalars alongside."""
        arrays = {}
        if self.rows:
            for key in self.rows[0]:
                arrays[key] = np.stack([np.asarray(row[key]) for row in self.rows])
        np.savez_compressed(path, **arrays, **{key: np.asarray(value) for key, value in scalars.items()})


# ---------------------------------------------------------------- the record's scalar fields


def nan_matrix(matrix: list[list[float]] | None, size: int) -> np.ndarray:
    """A matrix for the .npz; NaN when off, so the file needs no pickle to hold a None."""
    return np.full((size, size), np.nan) if matrix is None else np.asarray(matrix, dtype=np.float64)


def _n_action_steps(wired: Any) -> int:
    """The executed prefix length: the file's, the checkpoint's, or pi05_droid's known chunk."""
    base = getattr(wired.policy, "base_policy", None) if wired.policy is not None else None
    checkpoint = getattr(getattr(base, "config", None), "n_action_steps", None)
    if checkpoint is not None:
        return int(checkpoint)
    return int(wired.config.policy.n_action_steps or DROID_N_ACTION_STEPS)


def flat_settings(
    wired: Any,
    *,
    prompt: str,
    task_description: str,
    seed: int,
    start_pose: np.ndarray | None,
    tolerance: float,
    max_steps: int,
) -> dict:
    """The flat keys LIBERO's analysis reads, plus the run's identity, as plain JSON values."""
    config = wired.config
    steering = config.steering
    pinned = steering.pinned_steps if steering.pinned_steps is not None else config.policy.n_action_steps
    return {
        "arm": steering.mode,
        "mode": MODE_LABELS[steering.mode],
        "operator": "human",
        "prompt": prompt,
        "vla_prompt": prompt,
        "task_description": task_description,
        "tau": int(steering.tau),
        "lambda": int(steering.lam),
        "ic_gain": steering.ic_gain,
        "n_reversal_steps": steering.n_reversal_steps,
        "pinned_steps": pinned,
        "n_action_steps": _n_action_steps(wired),
        "reversal_adapter": _adapter_label(config),
        "corruption_matrix": _matrix_list(config.readout.corruption),
        "reversal_adapter_matrix": (
            None
            if steering.adapter_matrix is None
            else [[float(value) for value in row] for row in steering.adapter_matrix]
        ),
        "qdot_full": float(config.command.qdot_full),
        "twist_full": float(config.command.twist_full),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "start_pose": None if start_pose is None else [float(value) for value in np.asarray(start_pose)],
        "reset_tolerance_rad": float(tolerance),
    }


def provenance(
    wired: Any, *, prompt: str, task_description: str, seed: int, start_pose, tolerance: float
) -> dict:
    """The run directory's flat config.yaml, in the shape `../libero/notebooks/analyze.py` loads."""
    config = wired.config
    settings = flat_settings(
        wired,
        prompt=prompt,
        task_description=task_description,
        seed=seed,
        start_pose=start_pose,
        tolerance=tolerance,
        max_steps=config.trial.max_steps,
    )
    settings.update(
        {
            "name": config.steering.mode,
            "trials": int(config.trial.trials),
            "corruption": config.readout.corruption,
            "readout_input_noise": float(config.readout.input_noise),
            "station": {"host": config.station.host, "port": int(config.station.port)},
            "cameras": {
                "exterior": wired.cameras.sources[DROID_IMAGE_KEYS[0]],
                "wrist": wired.cameras.sources[DROID_IMAGE_KEYS[1]],
            },
            "camera_map": dict(wired.cameras.sources),
            "policy_path": config.policy.path,
            "started_at": wired.now().isoformat(timespec="seconds"),
        }
    )
    return settings
