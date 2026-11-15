"""Explicit scene assignments and independent random streams for matched trials."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrialSpec:
    suite: str
    task_id: int
    init_state_id: int
    env_seed: int
    policy_seed: int
    operator_seed: int
    input_noise_seed: int

    def metadata(self) -> dict:
        values = asdict(self)
        values["pair_id"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        return values


def corruption_rng(spec: TrialSpec) -> np.random.Generator:
    """The stream a trial's random corruption is drawn from: a function of the reset alone, so every
    method draws the same corruption on the same trial; derived from a seed already in the spec so
    `pair_id` stays comparable with earlier runs."""
    return np.random.default_rng([spec.input_noise_seed, 0x434F5252])


def trial_specs(
    schedule: list[int], suite: str, seed: int, state_counts: dict[int, int], first_occurrence: int = 0
) -> list[TrialSpec]:
    """Match each task's kth occurrence across methods, depth blocks and task orders.

    Each task receives a seeded permutation of its available initial states,
    reshuffled after each complete pass. Prefixes are stable when trial count
    or task order changes. Different seeds can share states; they do not define
    disjoint development and evaluation sets. `first_occurrence` does: a task's
    kth trial is its occurrence `first_occurrence + k`, with the state and seeds
    a longer run would give that occurrence, so a run from 5 continues where a
    5-per-task run stopped (disjoint states within a pass of a task's states).
    """
    counts: Counter = Counter()
    rngs = {task: np.random.default_rng([seed, task, 0x53544154]) for task in set(schedule)}
    permutations = {}
    for task_id in rngs:  # draw each task's permutations through the passes the offset skips
        n_states = state_counts[task_id]
        if n_states < 1:
            raise ValueError(f"Task {task_id} has no initial states")
        for skipped in range(first_occurrence):
            if skipped % n_states == 0:
                permutations[task_id] = rngs[task_id].permutation(n_states)
        counts[task_id] = first_occurrence
    specs = []
    for task_id in schedule:
        n_states = state_counts[task_id]
        occurrence = counts[task_id]
        if occurrence % n_states == 0:
            permutations[task_id] = rngs[task_id].permutation(n_states)
        state_id = int(permutations[task_id][occurrence % n_states])
        counts[task_id] += 1
        streams = np.random.SeedSequence([seed, task_id, occurrence, 0x54524941]).generate_state(4)
        specs.append(TrialSpec(suite, task_id, state_id, *(int(s) for s in streams)))
    return specs
