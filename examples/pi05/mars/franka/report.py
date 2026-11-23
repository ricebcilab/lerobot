#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Summarise Franka runs: success per arm, the measured command authority, and the paired difference.

    python report.py <run_dir> [<run_dir> ...] [--pair ARM_A ARM_B]

Each run directory is the one `runner.py` wrote: `config.yaml` (flat, with `mode`, `arm`, `qdot_full`) and
`trials.jsonl`, plus one `.npz` per trial. One line per arm carries its trials, successes, Wilson's 95%
interval and mean duration, then the mean measured authority over the pushing steps of its step files.

Authority is read off the arrays, not scheduled: with `a` the executed joint velocity, `p` the policy's own
plan and `r` the operator's served command (both in the same rad/s units, so `r` is the recorded unit command
scaled by the checkpoint's `qdot_full`), it is the barycentric coordinate of `a` on the segment `p -> r`,
`<a - p, r - p> / |r - p|^2`, clipped to [0, 1], averaged over the steps whose operator input leaves the
deadband. A NaN in either vector makes the step unmeasurable and it is skipped.

Two arms are compared on `pair_id`, one pair per trial, and only where the two records' `initial_state_hash`
agrees: a pair whose resets differ was never the same trial and is reported as skipped, not compared. The
difference is in paired success, with an exact two-sided sign test over the discordant pairs. Nothing here
needs scipy: `wilson` and `sign_test` are the two formulas, written out.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEADBAND = 0.05
"""The operator is idle below this (max |component|), matching `lerobot.teleoperators.readout.DEADBAND`."""

Z_95 = 1.959963984540054
"""The two-sided 95% normal quantile Wilson's interval uses."""


def wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson's score interval for a binomial proportion; NaN, NaN when there were no trials.

    The interval is the one the LIBERO analysis uses, but inline so a report needs no scipy: the centre is
    the shrunk estimate and the half-width uses the proportion's own variance plus the shrinkage's.
    """
    if n <= 0:
        return math.nan, math.nan
    rate = successes / n
    denominator = 1.0 + Z_95**2 / n
    centre = (rate + Z_95**2 / (2.0 * n)) / denominator
    half = Z_95 * math.sqrt(rate * (1.0 - rate) / n + Z_95**2 / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(wins: int, losses: int) -> float:
    """The exact two-sided sign test's p-value over `wins` + `losses` discordant pairs; NaN for none.

    Under the null each discordant pair is a fair coin, so the two-sided tail is twice the probability of
    the observed split or a more extreme one, clipped at 1 because the two tails can overlap.
    """
    n = wins + losses
    if n <= 0:
        return math.nan
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def trial_authority(path: str | Path, qdot_full: float, deadband: float = DEADBAND) -> float:
    """The mean measured authority of one trial's step file, or NaN when no step is measurable.

    `action` is the executed joint velocity, `policy_translation` the policy's plan, and `user_translation`
    the served command in operator units; scaling the command by `qdot_full` puts all three in rad/s. Only
    steps whose raw operator input leaves the deadband have a direction to be measured against.
    """
    try:
        with np.load(Path(path)) as data:
            action = np.asarray(data["action"], dtype=np.float64)[:, :3]
            plan = np.asarray(data["policy_translation"], dtype=np.float64)[:, :3]
            served = np.asarray(data["user_translation"], dtype=np.float64)[:, :3]
            raw = np.asarray(data["user_translation_raw"], dtype=np.float64)[:, :3]
    except (KeyError, OSError, ValueError):
        return math.nan  # no step arrays (an aborted first tick), or not a step file
    command = served * float(qdot_full)
    delta = command - plan
    denominator = (delta * delta).sum(axis=1)
    pushing = np.max(np.abs(raw), axis=1) >= deadband
    finite = np.isfinite(action).all(axis=1) & np.isfinite(plan).all(axis=1)
    usable = pushing & finite & (denominator > 1e-12)
    if not usable.any():
        return math.nan
    values = ((action - plan) * delta).sum(axis=1)[usable] / denominator[usable]
    return float(np.clip(values, 0.0, 1.0).mean())


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One run's flat config and its trial records, in file order."""
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    lines = (run_dir / "trials.jsonl").read_text().splitlines()
    return config, [json.loads(line) for line in lines if line.strip()]


@dataclass
class ArmSummary:
    """One arm's trials across the runs given, with the statistics the report prints."""

    arm: str
    records: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def successes(self) -> int:
        return sum(bool(record["success"]) for _, record in self.records)

    @property
    def mean_duration(self) -> float:
        values = [float(record["duration_s"]) for _, record in self.records if "duration_s" in record]
        return float(np.mean(values)) if values else math.nan

    def authority(self) -> float:
        """The mean over trials of each trial's measured authority; trials with no push are skipped."""
        values = []
        for run_dir, record in self.records:
            name = record.get("steps_file") or f"trial_{int(record.get('trial', 0)):03d}.npz"
            path = run_dir / name
            if not path.exists():
                continue
            value = trial_authority(path, qdot_full=float(record.get("qdot_full", 1.0)))
            if not math.isnan(value):
                values.append(value)
        return float(np.mean(values)) if values else math.nan


def summarize(runs: Sequence[tuple[Path, dict[str, Any], list[dict[str, Any]]]]) -> list[ArmSummary]:
    """Group the runs' trials by arm, in the order the arms first appear."""
    summaries: dict[str, ArmSummary] = {}
    for run_dir, config, trials in runs:
        for trial in trials:
            arm = str(config.get("arm") or trial.get("arm") or config.get("mode") or "unknown")
            summaries.setdefault(arm, ArmSummary(arm)).records.append((run_dir, trial))
    return list(summaries.values())


def _summary_line(summary: ArmSummary) -> str:
    """One arm's line: n, successes, Wilson's interval, mean duration, measured authority."""
    low, high = wilson(summary.successes, summary.n)
    rate = summary.successes / summary.n if summary.n else math.nan
    authority = summary.authority()
    measured = "   n/a" if math.isnan(authority) else f"{authority:5.2f}"
    return (
        f"{summary.arm:<12} n={summary.n:<3d} success={summary.successes}/{summary.n} ({rate:5.0%})  "
        f"wilson [{low:.2f}, {high:.2f}]  mean {summary.mean_duration:5.1f}s  authority {measured}"
    )


def paired_difference(first: ArmSummary, second: ArmSummary) -> dict[str, Any]:
    """The paired success difference over `pair_id`s whose recorded reset hashes agree.

    Pairs present in only one arm, and pairs whose hashes differ, are counted but not compared: the first
    never met the other condition and the second did not start from the same scene.
    """
    first_by = {record["pair_id"]: record for _, record in first.records if "pair_id" in record}
    second_by = {record["pair_id"]: record for _, record in second.records if "pair_id" in record}
    shared = first_by.keys() & second_by.keys()
    matched = skipped = wins = losses = 0
    differences: list[float] = []
    for pair in sorted(shared):
        left, right = first_by[pair], second_by[pair]
        if left.get("initial_state_hash") != right.get("initial_state_hash"):
            skipped += 1
            continue
        matched += 1
        left_success, right_success = bool(left["success"]), bool(right["success"])
        wins += int(left_success and not right_success)
        losses += int(right_success and not left_success)
        differences.append(float(left_success) - float(right_success))
    return {
        "matched": matched,
        "skipped": skipped,
        "wins": wins,
        "losses": losses,
        "difference": float(np.mean(differences)) if differences else math.nan,
        "p_value": sign_test(wins, losses),
    }


def _paired_line(first: ArmSummary, second: ArmSummary) -> str:
    """The two arms' paired comparison, or the count of pairs that could not be compared."""
    result = paired_difference(first, second)
    if not result["matched"]:
        return (
            f"paired {first.arm} vs {second.arm}: 0 matched pairs "
            f"({result['skipped']} hash-mismatched, {len(first.records)} and {len(second.records)} trials)"
        )
    return (
        f"paired {first.arm} vs {second.arm}: {result['matched']} matched pair(s) "
        f"({result['skipped']} hash-mismatched), success difference {result['difference']:+.2f}, "
        f"W/L {result['wins']}/{result['losses']}, sign-test p={result['p_value']:.3f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report for the run directories given."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, metavar="RUN_DIR")
    parser.add_argument(
        "--pair", nargs=2, metavar=("ARM_A", "ARM_B"), default=None, help="the two arms to compare"
    )
    args = parser.parse_args(argv)
    runs = []
    for run_dir in args.runs:
        try:
            config, trials = load_run(run_dir)
        except OSError as error:
            raise SystemExit(f"{run_dir}: {error}") from error
        runs.append((run_dir, config, trials))
    summaries = summarize(runs)
    for summary in summaries:
        print(_summary_line(summary))
    by_arm = {summary.arm: summary for summary in summaries}
    pair = args.pair or (list(by_arm) if len(by_arm) == 2 else None)
    if pair is not None:
        missing = [arm for arm in pair if arm not in by_arm]
        if missing:
            parser.error(f"--pair: no trials for {', '.join(missing)} (have {', '.join(by_arm)})")
        print("\n" + _paired_line(by_arm[pair[0]], by_arm[pair[1]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
