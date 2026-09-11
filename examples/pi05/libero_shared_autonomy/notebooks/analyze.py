"""Load and compare runs written by experiment.py.

A run directory holds `config.yaml` (resolved settings, schedule and the
matrices in force), `trials.jsonl` (one record per completed trial) and per
trial a `.npz` of step arrays. Every trial row is tagged with the run's
*method* (FC = shared flow control, FRS = native flow reversal steering,
FRS+F = flow reversal with the reversal adapter, or a policy-only anchor:
floor = the arm's prompt, ceiling = the scene's own instruction), its *depth*
(the denoising steps the operator controls: `n_guided_steps` for FC,
`n_reversal_steps` for the reversal arms) and who the operator was.

Pairing requires recorded pair IDs and matching initial-state hashes. Legacy
runs only shared a task schedule, which does not establish matched resets.

Sections: loading, describing what was run, rates, paired tests, expressed
behaviour, step arrays, plots.
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
METHODS = ("FC", "FRS", "FRS+F", "policy (floor)", "policy (ceiling)")
AXES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]
PROMPT_FROM_TASK = "task"


# ---------------------------------------------------------------- loading


def find_runs(root: Path) -> list[Path]:
    """Run directories under `root` that recorded at least one trial, oldest first."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "trials.jsonl").exists())


def adapter_matrix(config: dict):
    """The resolved reversal adapter F of a run (pre-rename runs stored it as flow_adapter_matrix)."""
    return config.get("reversal_adapter_matrix", config.get("flow_adapter_matrix"))


def method_of(config: dict) -> str:
    """Which method a run's resolved config describes (see METHODS)."""
    mode = config["mode"]
    if mode == "shared_flow_control":
        return "FC"
    if mode == "shared_flow_reversal_steering":
        return "FRS+F" if adapter_matrix(config) is not None else "FRS"
    if mode == "policy":
        return "policy (ceiling)" if config.get("prompt") == PROMPT_FROM_TASK else "policy (floor)"
    return MODE_SHORT.get(mode, mode)


def depth_of(config: dict) -> float:
    """Denoising steps the operator controls; NaN for modes without a depth."""
    mode = config["mode"]
    if mode == "shared_flow_control":
        return float(config.get("n_guided_steps", config.get("tau", np.nan)))  # tau: pre-rename runs
    if mode == "shared_flow_reversal_steering":
        depth = config.get("n_reversal_steps")
        return 10.0 if depth is None else float(depth)  # None = full reversal through all 10 steps
    return np.nan


def label_for(config: dict) -> str:
    """Short run label from what actually varied: mode and depth, corruption M (+M), adapter F (+F)."""
    mode = config["mode"]
    label = MODE_SHORT.get(mode, mode)
    if mode == "shared_flow_control":
        label += f" n_guided_steps={int(depth_of(config))}"
    elif mode == "shared_flow_reversal_steering":
        depth = config.get("n_reversal_steps")
        label += " full" if depth is None else f" n_reversal_steps={depth}"
    if config.get("corruption_matrix") is not None:
        label += "+M"
    if adapter_matrix(config) is not None:
        label += "+F"
    return label


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One run's trials (tagged with run, label, method, depth, operator, ...) and its config."""
    run_dir = Path(run_dir)
    trials = [
        json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines() if line.strip()
    ]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    columns = ["trial", "task_id", "success", "steps", "duration_s", "steps_file"]
    df = pd.DataFrame(trials, columns=columns) if not trials else pd.DataFrame(trials)
    if "mode_expressed" not in df:
        df["mode_expressed"] = None
    df["success"] = df["success"].astype(bool)
    df["run"] = run_dir.name
    df["run_dir"] = str(run_dir)
    df["label"] = label_for(config)
    df["mode"] = config["mode"]
    df["method"] = method_of(config)
    df["depth"] = depth_of(config)
    df["operator"] = config.get("operator", "human")
    timing = config.get("policy_operator", {})
    df["operator_profile"] = (
        json.dumps(
            {
                "update_every_steps": timing.get("update_every_steps") or config.get("n_action_steps", 10),
                "delay_steps": timing.get("delay_steps", 0),
            },
            sort_keys=True,
        )
        if config.get("operator", "human") == "policy"
        else "human"
    )
    df["corrupted"] = config.get("corruption_matrix") is not None
    df["adapted"] = adapter_matrix(config) is not None
    return df, config


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """All trials of `run_dirs` in one frame, plus {run name: config}. `load_runs(find_runs(root))` loads a directory."""
    frames, configs = [], {}
    for run_dir in run_dirs:
        df, config = load_run(run_dir)
        frames.append(df)
        configs[Path(run_dir).name] = config
    versions = {c.get("implementation_version", 1) for c in configs.values()}
    if len(versions) > 1:
        raise ValueError(
            "Do not pool legacy and corrected runs; load a separate output directory per version."
        )
    profiles = {
        profile for frame in frames for profile in frame["operator_profile"].unique() if profile != "human"
    }
    if len(profiles) > 1:
        raise ValueError(
            "Do not pool operator timing profiles; load a separate output directory per profile."
        )
    return pd.concat(frames, ignore_index=True), configs


# ---------------------------------------------------------------- describing what was run


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


# ---------------------------------------------------------------- rates


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(trials: pd.DataFrame, by: list[str], metric: str = "success") -> pd.DataFrame:
    """Rate of a boolean `metric` per group of `by` columns, with trial counts and Wilson 95% CI.

    Groups are sorted in METHODS order when `by` contains "method", then by the other keys.
    """
    rows = []
    for key, group in trials.groupby(by, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(group), int(group[metric].astype(bool).sum())
        low, high = wilson(k, n)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "trials": n,
                "successes": k,
                "rate": k / n,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    table = pd.DataFrame(rows, columns=[*by, "trials", "successes", "rate", "ci95_low", "ci95_high"])
    if "method" in by:
        order = {m: i for i, m in enumerate(METHODS)}
        table = table.assign(_o=table["method"].map(order).fillna(len(order)))
        table = table.sort_values(["_o", *[b for b in by if b != "method"]]).drop(columns="_o")
    return table.reset_index(drop=True)


def success_table(trials: pd.DataFrame, configs: dict[str, dict]) -> pd.DataFrame:
    """Per run: label, mode, flags, trial and success counts, rate with Wilson 95% CI, median steps of successes.

    Runs in `configs` without any trial are kept, with zero counts and NaN rates.
    """
    rates = rate_table(trials, ["run"]).set_index("run").reindex(list(configs))
    rates[["trials", "successes"]] = rates[["trials", "successes"]].fillna(0).astype(int)
    rates = rates.rename(columns={"rate": "success_rate"})
    info = pd.DataFrame(
        {
            run: {
                "label": label_for(c),
                "mode": c["mode"],
                "corrupted": c.get("corruption_matrix") is not None,
                "adapted": adapter_matrix(c) is not None,
            }
            for run, c in configs.items()
        }
    ).T
    wins = trials[trials["success"]]
    median = wins.groupby("run")["steps"].median().reindex(list(configs))
    return info.join(rates).assign(median_steps_success=median)


def per_task_table(trials: pd.DataFrame, metric: str = "success") -> pd.DataFrame:
    """Rows = task ("<id>: <description>"), columns = (method, depth): rate of `metric`."""
    labelled = trials.assign(task=trials["task_id"].astype(str) + ": " + trials["task_description"])
    return labelled.pivot_table(
        index="task",
        columns=["method", "depth"],
        values=metric,
        aggfunc=lambda s: s.astype(bool).mean(),
        dropna=False,
    )


def best_depth(trials: pd.DataFrame, metric: str = "success") -> pd.DataFrame:
    """Per steering method: the depth with the highest rate of `metric`, pooled over tasks."""
    curve = rate_table(trials, ["method", "depth"], metric)
    curve = curve[curve["depth"].notna()]
    return curve.loc[curve.groupby("method")["rate"].idxmax()].reset_index(drop=True)


# ---------------------------------------------------------------- paired tests


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """McNemar's exact test on paired boolean outcomes `a` and `b` (same length, same trials)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    discordant = a_only + b_only
    return {
        "paired_trials": len(a),
        "A_rate": float(a.mean()) if len(a) else np.nan,
        "B_rate": float(b.mean()) if len(b) else np.nan,
        "A_only": a_only,
        "B_only": b_only,
        "discordant": discordant,
        "mcnemar_p": stats.binomtest(b_only, discordant, 0.5).pvalue if discordant else np.nan,
    }


def paired_test(
    trials: pd.DataFrame,
    by: str,
    a: str,
    b: str,
    pair_on: list[str] | None = None,
    metric: str = "success",
) -> dict:
    """McNemar's exact test on verified matching resets, defaulting to recorded pair IDs.

    `paired_test(trials, "run", a, b)` compares two runs trial by trial;
    `paired_test(trials[trials.depth == 4], "method", "FRS", "FRS+F", ["task_id", "trial"])`
    compares two methods at one depth, paired by scene reset and pooled over tasks.
    """
    required = ["pair_id", "initial_state_hash"]
    selected = trials[trials[by].isin([a, b])]
    if any(k not in selected for k in required) or selected[required].isna().any().any():
        raise ValueError(
            "Unverified pairing: pair_id and initial_state_hash are required. "
            "Legacy task/trial numbers do not identify matching resets; rerun with explicit initial states."
        )
    pair_on = ["pair_id"] if pair_on is None else list(pair_on)
    arms = [selected[selected[by] == arm] for arm in (a, b)]
    if any(arm.duplicated(pair_on).any() for arm in arms):
        raise ValueError("Duplicate pairing keys: select one condition per arm or use pair_on=['pair_id'].")
    pa, pb = [arm.set_index(pair_on) for arm in arms]
    both = pa.index.intersection(pb.index)
    for key in required:
        va, vb = pa.loc[both].reset_index()[key], pb.loc[both].reset_index()[key]
        if not va.equals(vb):
            raise ValueError(
                f"Unmatched {key}: the requested pairs do not share the same trial setup and state."
            )
    return {"A": a, "B": b, **mcnemar(pa.loc[both, metric].values, pb.loc[both, metric].values)}


def paired_comparisons(
    trials: pd.DataFrame, by: str = "run", pair_on: list[str] | None = None
) -> pd.DataFrame:
    """`paired_test` over every pair of values of `by`, in order of first appearance."""
    values = list(dict.fromkeys(trials[by]))
    return pd.DataFrame([paired_test(trials, by, a, b, pair_on) for a, b in combinations(values, 2)])


# ---------------------------------------------------------------- expressed behaviour


def task_targets(trials: pd.DataFrame) -> dict[int, str | None]:
    """The scene element each task is about: what moved most in the ceiling anchor's successes.

    Falls back to the most common element among *all* successful trials of the task,
    and to None when the task never succeeded (then mode accuracy is undefined).
    """
    targets: dict[int, str | None] = {}
    for task_id, group in trials.groupby("task_id"):
        wins = group[group["success"] & group["mode_expressed"].notna()]
        ceiling = wins[wins["method"] == "policy (ceiling)"]
        pool = ceiling if len(ceiling) else wins
        targets[int(task_id)] = pool["mode_expressed"].mode().iloc[0] if len(pool) else None
    return targets


def with_mode_accuracy(trials: pd.DataFrame) -> pd.DataFrame:
    """Add `target` (the task's element) and `on_target` (this trial expressed it, finished or not)."""
    targets = task_targets(trials)
    out = trials.copy()
    out["target"] = out["task_id"].map(targets)
    out["on_target"] = (out["mode_expressed"] == out["target"]) & out["target"].notna()
    return out


# ---------------------------------------------------------------- step arrays


def step_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-trial metrics from the step arrays, joined onto `trials`.

    Operator engagement: `commanding_frac` (steps pushing beyond the deadband),
    `mean_speed_when_active`, `mean_corruption_shift` (|served - raw| while pushing),
    `gripper_closed_frac`, `reads_this_trial`. Intent transmission (needs `action`):
    `cos_raw`, the mean cosine between the executed translation and the operator's *raw*
    command (the clean intent), and `cos_served`, the same against the corrupted command
    the policy was given -- a method that undoes the corruption has `cos_raw` above `cos_served`.
    """

    def mean_cos(a: np.ndarray, b: np.ndarray) -> float:
        norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        ok = norms > 1e-9
        return float(((a * b).sum(axis=1)[ok] / norms[ok]).mean()) if ok.any() else np.nan

    def one(row) -> pd.Series:
        z = np.load(Path(row["run_dir"]) / row["steps_file"])
        raw, served = z["user_translation_raw"], z["user_translation"]
        active = np.abs(raw).max(axis=1) >= DEADBAND
        executed = z["action"][:, :3] if "action" in z.files else None
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
                "gripper_closed_frac": float((z["user_gripper"] > 0).mean())
                if "user_gripper" in z.files
                else np.nan,
                "reads_this_trial": int(z["user_reads"].sum()) if "user_reads" in z.files else 0,
                "cos_raw": mean_cos(raw[active], executed[active]) if executed is not None else np.nan,
                "cos_served": mean_cos(served[active], executed[active]) if executed is not None else np.nan,
            }
        )

    return trials.join(trials.apply(one, axis=1), rsuffix="_npz")


# ---------------------------------------------------------------- plots


def plot_performance_vs_depth(trials: pd.DataFrame, ax, metric: str = "success", title: str | None = None):
    """One line per steering method: rate of `metric` against controlled denoising steps.

    Pooled over the tasks in `trials`, with Wilson 95% error bars. Methods without a
    depth (the policy-only anchors) are drawn as dashed horizontal bands. Returns the table.
    """
    curve = rate_table(trials, ["method", "depth"], metric)
    for method, group in curve.groupby("method", sort=False):
        group = group.sort_values("depth")
        if group["depth"].isna().all():
            rate = group["rate"].iloc[0]
            ax.axhline(rate, linestyle="--", linewidth=1, label=f"{method} ({rate:.0%})")
            ax.axhspan(group["ci95_low"].iloc[0], group["ci95_high"].iloc[0], alpha=0.08)
            continue
        err = np.vstack([group["rate"] - group["ci95_low"], group["ci95_high"] - group["rate"]])
        ax.errorbar(group["depth"], group["rate"], yerr=err, marker="o", capsize=3, label=method)
    ax.set_xlabel("method depth (FC guided steps / FRS reversal steps)")
    ax.set_ylabel(f"{metric} rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(sorted(curve["depth"].dropna().unique()))
    ax.set_title(title or f"{metric} vs. controlled steps, pooled over {trials['task_id'].nunique()} tasks")
    ax.grid(alpha=0.3)
    ax.legend()
    return curve
