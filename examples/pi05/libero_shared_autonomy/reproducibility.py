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


def trial_specs(schedule: list[int], suite: str, seed: int, state_counts: dict[int, int]) -> list[TrialSpec]:
    """Match each task's kth occurrence across methods, depth blocks and task orders.

    Each task receives a seeded permutation of its available initial states,
    reshuffled after each complete pass. Prefixes are stable when trial count
    or task order changes. Different seeds can share states; they do not define
    disjoint development and evaluation sets.
    """
    counts: Counter = Counter()
    rngs = {task: np.random.default_rng([seed, task, 0x53544154]) for task in set(schedule)}
    permutations = {}
    specs = []
    for task_id in schedule:
        n_states = state_counts[task_id]
        if n_states < 1:
            raise ValueError(f"Task {task_id} has no initial states")
        occurrence = counts[task_id]
        if occurrence % n_states == 0:
            permutations[task_id] = rngs[task_id].permutation(n_states)
        state_id = int(permutations[task_id][occurrence % n_states])
        counts[task_id] += 1
        streams = np.random.SeedSequence([seed, task_id, occurrence, 0x54524941]).generate_state(4)
        specs.append(TrialSpec(suite, task_id, state_id, *(int(s) for s in streams)))
    return specs
