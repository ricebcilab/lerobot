"""Load and summarize runs written by experiment.py.

A run directory holds `config.yaml` (resolved settings, schedule and the
matrices in force), `trials.jsonl` (one record per completed trial) and per
trial a `.npz` of step arrays. Runs that share suite, seed and task_order see
the same task schedule, so trial i is the same task across them: comparisons
are paired.
"""

import json
from itertools import combinations
from math import atan2, degrees, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from lerobot.policies.pi05.steering import DEADBAND

MODE_SHORT = {
    "shared_flow_control": "FC",
    "shared_flow_reversal_steering": "FRS",
    "shared_override": "override",
    "teleop": "teleop",
    "policy": "policy",
}
AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def find_runs(root: Path) -> list[Path]:
    """Run directories under `root` that recorded at least one trial, oldest first."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "trials.jsonl").exists())


def _adapter_matrix(config: dict):
    """The resolved reversal adapter F of a run (pre-rename runs stored it as flow_adapter_matrix)."""
    return config.get("reversal_adapter_matrix", config.get("flow_adapter_matrix"))


def label_for(config: dict) -> str:
    """Short condition label from what actually varied: mode, corruption M, adapter F."""
    mode = MODE_SHORT.get(config["mode"], config["mode"])
    if config["mode"] == "shared_flow_control":
        mode += f" n_guided_steps={config.get('n_guided_steps', config.get('tau'))}"  # tau: pre-rename runs
    elif config["mode"] == "shared_flow_reversal_steering":
        depth = config.get("n_reversal_steps")
        mode += " full" if depth is None else f" n_reversal_steps={depth}"
    tags = []
    if config.get("corruption_matrix") is not None:
        tags.append("+M")
    if _adapter_matrix(config) is not None:
        tags.append("+F")
    return mode + "".join(tags)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    trials = [
        json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines() if line.strip()
    ]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    columns = ["trial", "task_id", "success", "steps", "duration_s", "steps_file"]
    df = pd.DataFrame(trials, columns=columns) if not trials else pd.DataFrame(trials)
    df["run"] = run_dir.name
    df["label"] = label_for(config)
    df["mode"] = config["mode"]
    df["corrupted"] = config.get("corruption_matrix") is not None
    df["adapted"] = _adapter_matrix(config) is not None
    df["run_dir"] = str(run_dir)
    return df, config


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, dict]]:
    frames, configs = [], {}
    for run_dir in run_dirs:
        df, config = load_run(run_dir)
        frames.append(df)
        configs[Path(run_dir).name] = config
    return pd.concat(frames, ignore_index=True), configs


def describe_rotation(matrix: np.ndarray) -> str:
    """Say whether a 3x3 is a rotation about z, and by how much."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        return ""
    if np.allclose(matrix, np.eye(3)):
        return "identity (no-op)"
    orthonormal = np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-3) and np.isclose(
        np.linalg.det(matrix), 1, atol=1e-3
    )
    angle = degrees(atan2(matrix[1, 0], matrix[0, 0]))
    return (
        f"rotation about z by {angle:+.0f} deg"
        if orthonormal
        else f"not a rotation (det={np.linalg.det(matrix):.3f})"
    )


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def success_table(trials: pd.DataFrame, configs: dict[str, dict]) -> pd.DataFrame:
    """Per run: trials, successes, rate, Wilson 95% CI, median steps of successes."""
    rows = []
    for run in configs:
        group = trials[trials["run"] == run]
        base = {
            "run": run,
            "label": label_for(configs[run]),
            "mode": configs[run]["mode"],
            "corrupted": configs[run].get("corruption_matrix") is not None,
            "adapted": _adapter_matrix(configs[run]) is not None,
        }
        if len(group) == 0:
            rows.append(
                {
                    **base,
                    "trials": 0,
                    "successes": 0,
                    "success_rate": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "median_steps_success": np.nan,
                }
            )
            continue
        n, k = len(group), int(group["success"].sum())
        low, high = wilson(k, n)
        rows.append(
            {
                **base,
                "trials": n,
                "successes": k,
                "success_rate": k / n if n else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "median_steps_success": group.loc[group["success"].astype(bool), "steps"].median(),
            }
        )
    return pd.DataFrame(rows).set_index("run")


def differing_settings(configs: dict[str, dict], keys: list[str]) -> pd.DataFrame:
    """Settings that are not identical across runs (rows = keys, columns = run labels).

    Nested values (matrix specs) are compared by their JSON form so dicts do not
    break pandas' uniqueness check.
    """

    def flat(value):
        return json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else value

    setup = pd.DataFrame({label_for(c): {k: flat(c.get(k)) for k in keys} for c in configs.values()})
    setup.loc["schedule"] = [str(c.get("schedule")) for c in configs.values()]
    return setup[setup.nunique(axis=1, dropna=False) > 1]


def compare(trials: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of runs a and b on the trials both completed (McNemar exact test)."""
    pair = trials.pivot_table(index="trial", columns="run", values="success", aggfunc="first")[
        [a, b]
    ].dropna()
    a_only = int((pair[a].astype(bool) & ~pair[b].astype(bool)).sum())
    b_only = int((~pair[a].astype(bool) & pair[b].astype(bool)).sum())
    discordant = a_only + b_only
    p = stats.binomtest(b_only, discordant, 0.5).pvalue if discordant else np.nan
    return {
        "A": a,
        "B": b,
        "paired_trials": len(pair),
        "A_rate": float(pair[a].mean()),
        "B_rate": float(pair[b].mean()),
        "A_only": a_only,
        "B_only": b_only,
        "discordant": discordant,
        "mcnemar_p": p,
    }


def paired_comparisons(trials: pd.DataFrame) -> pd.DataFrame:
    runs = list(dict.fromkeys(trials["run"]))
    return pd.DataFrame([compare(trials, a, b) for a, b in combinations(runs, 2)])


def input_activity(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-trial operator engagement from the step arrays, joined onto `trials`."""

    def one(row) -> pd.Series:
        z = np.load(Path(row["run_dir"]) / row["steps_file"])
        raw, served = z["user_translation_raw"], z["user_translation"]
        active = np.abs(raw).max(axis=1) >= DEADBAND
        return pd.Series(
            {
                "n_steps": len(raw),
                "commanding_frac": float(active.mean()),
                "mean_speed_when_active": float(np.linalg.norm(raw[active], axis=1).mean())
                if active.any()
                else 0.0,
                "mean_corruption_shift": float(np.linalg.norm((served - raw)[active], axis=1).mean())
                if active.any()
                else 0.0,
                "gripper_closed_frac": float((z["user_gripper"] > 0).mean()),
                "reads_this_trial": int(z["user_reads"].sum()),
            }
        )

    return trials.join(trials.apply(one, axis=1), rsuffix="_npz")
