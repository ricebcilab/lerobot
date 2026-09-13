"""Load and compare runs written by experiment.py.

A run directory holds `config.yaml` (resolved settings, schedule and the
matrices in force), `trials.jsonl` (one record per completed trial) and per
trial a `.npz` of step arrays. Every trial row is tagged with the run's
*method* (FC = shared flow control, FRS = native flow reversal steering,
FRS-RA = flow reversal with the reversal adapter, or a policy-only anchor:
floor = the arm's prompt, ceiling = the scene's own instruction), its *depth*
(the denoising steps the operator controls: `n_guided_steps` for FC,
`n_reversal_steps` for the reversal arms), its *authority* (the share of the
denoising steps under the operator's constraint, on one 0-1 axis for both
conventions) and who the operator was.

Pairing requires recorded pair IDs and matching initial-state hashes.

Sections: loading, tables, paired tests, per-trial metrics from the step arrays, plots.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator
from scipy import stats

from lerobot.policies.pi05.steering import DEADBAND

METHODS = ("FC", "FRS", "FRS-RA", "policy (floor)", "policy (ceiling)")
STEERED = ("FC", "FRS", "FRS-RA")
# One fixed colour per method in every figure (identity is never cycled). The three
# steered hues are colourblind-safe as a set (worst adjacent CVD dE 9.2); the anchors
# are reference lines, so they take a neutral.
METHOD_COLORS = {
    "FC": "#2a78d6",
    "FRS": "#eb6834",
    "FRS-RA": "#1baf7a",
    "policy (floor)": "#6b6b6b",
    "policy (ceiling)": "#6b6b6b",
}
PAPER_STYLE = Path(__file__).with_name("paper.mplstyle")  # plt.style.use(analyze.PAPER_STYLE)
# Axis labels for the per-trial metrics, as they would read in a figure caption.
METRIC_LABELS = {
    "success": "Success rate",
    "on_target": "Right behaviour selected",
    "progress_max": "Best task progress",
    "progress_final": "Final task progress",
    "goal_distance_min": "Closest approach to goal (m)",
    "path_length": "End-effector path length (m)",
    "displacement": "End-effector displacement (m)",
    "path_efficiency": "Path efficiency",
    "time_to_success": "Steps to success",
    "commanding_frac": "Fraction of steps commanding",
    "mean_speed_when_active": "Mean command magnitude",
    "mean_corruption_shift": "Mean corruption shift",
    "gripper_closed_frac": "Fraction of steps gripper closed",
    "reads_this_trial": "Input reads per trial",
    "cos_raw": "Intent agreement (cosine to raw command)",
    "cos_served": "Intent agreement (cosine to served command)",
    "user_authority": "Measured user authority",
    "user_authority_pushing": "Measured user authority while pushing",
    "policy_authority": "Measured policy authority",
}
X_LABELS = {"authority": "Scheduled user authority", "depth": "Method depth (denoising steps)"}
NUM_INFERENCE_STEPS = 10  # pi0.5's denoising steps; the depth of every arm counts a subset of them
PROMPT_FROM_TASK = "task"  # config.PROMPT_FROM_TASK


# ---------------------------------------------------------------- loading


def find_runs(root: Path) -> list[Path]:
    """Run directories under `root` that recorded at least one trial, oldest first."""
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "trials.jsonl").exists())


def method_of(config: dict) -> str:
    """Which method a run's resolved config describes (see METHODS)."""
    mode = config["mode"]
    if mode == "shared_flow_control":
        return "FC"
    if mode == "shared_flow_reversal_steering":
        return "FRS-RA" if config.get("reversal_adapter_matrix") is not None else "FRS"
    if mode == "policy":
        return "policy (ceiling)" if config.get("prompt") == PROMPT_FROM_TASK else "policy (floor)"
    return mode  # shared_override / teleop: not part of the study


def depth_of(config: dict) -> float:
    """Denoising steps the operator controls; NaN for modes without a depth."""
    mode = config["mode"]
    if mode == "shared_flow_control":
        return float(config["n_guided_steps"])
    if mode == "shared_flow_reversal_steering":
        depth = config.get("n_reversal_steps")
        return float(NUM_INFERENCE_STEPS if depth is None else depth)  # None = full reversal
    return np.nan


def authority(method: str, depth: float, num_steps: int = NUM_INFERENCE_STEPS) -> float:
    """The operator's authority on a 0-1 axis shared by both depth conventions.

    Both methods end with the policy denoising freely from some point of its
    `num_steps`-step schedule. FC clamps the command onto the state for the first
    `n_guided_steps`, so the operator constrains `depth / num_steps` of the schedule.
    FRS reverses the reference `n_reversal_steps` of the way to noise and the policy
    denoises those steps back freely, so the operator constrains the remaining
    `(num_steps - depth) / num_steps`. Authority 1 is the whole schedule under the
    operator (FC: every step clamped; FRS: the reference executed as is), 0 is the
    policy alone. NaN for the anchors.
    """
    if method == "FC":
        return depth / num_steps
    if method in ("FRS", "FRS-RA"):
        return (num_steps - depth) / num_steps
    return np.nan


def operator_profile(config: dict) -> str:
    """Who drove: "human", or the synthetic operator's resolved timing as a JSON string."""
    if config.get("operator", "human") != "policy":
        return "human"
    timing = config.get("policy_operator") or {}
    profile = {
        "update_every_steps": timing.get("update_every_steps") or config.get("n_action_steps", 10),
        "delay_steps": timing.get("delay_steps", 0),
    }
    return json.dumps(profile, sort_keys=True)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One run's trials (tagged with run, label, method, depth, authority, operator, ...) and its config."""
    run_dir = Path(run_dir)
    trials = [
        json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines() if line.strip()
    ]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    columns = ["trial", "task_id", "success", "steps", "duration_s", "steps_file", "mode_expressed"]
    df = pd.DataFrame(trials, columns=columns) if not trials else pd.DataFrame(trials)
    df["success"] = df["success"].astype(bool)
    method, depth = method_of(config), depth_of(config)
    df["run"] = run_dir.name
    df["run_dir"] = str(run_dir)
    df["mode"] = config["mode"]
    df["method"] = method
    df["depth"] = depth
    df["authority"] = authority(method, depth)
    df["operator"] = config.get("operator", "human")
    df["operator_profile"] = operator_profile(config)
    return df, config


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """All trials of `run_dirs` in one frame, plus {run name: config}. `load_runs(find_runs(root))` loads a directory."""
    frames, configs = [], {}
    for run_dir in run_dirs:
        df, config = load_run(run_dir)
        frames.append(df)
        configs[Path(run_dir).name] = config
    profiles = {
        profile for frame in frames for profile in frame["operator_profile"].unique() if profile != "human"
    }
    if len(profiles) > 1:
        raise ValueError(
            "Do not pool operator timing profiles; load a separate output directory per profile."
        )
    return pd.concat(frames, ignore_index=True), configs


# ---------------------------------------------------------------- tables


def wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion; NaN when n == 0."""
    if n == 0:
        return (np.nan, np.nan)
    ci = stats.binomtest(successes, n).proportion_ci(confidence_level=0.95, method="wilson")
    return (ci.low, ci.high)


def _is_binary(values: pd.Series) -> bool:
    """A boolean column, or a numeric one whose non-NaN values are all 0/1 (success, on_target, ...)."""
    if values.dtype == bool:
        return True
    v = values.dropna()
    return bool(len(v)) and bool(v.isin([0, 1]).all())


def _groups(trials: pd.DataFrame, by: list[str]):
    """`groupby(by)` yielding ({key column: value}, group), keeping NaN keys and first-appearance order."""
    for key, group in trials.groupby(by, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(by, key, strict=True)), group


def _in_method_order(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sort by METHODS order when `by` contains "method", then by the other keys."""
    if "method" in by:
        order = {m: i for i, m in enumerate(METHODS)}
        table = table.assign(_o=table["method"].map(order).fillna(len(order)))
        table = table.sort_values(["_o", *[b for b in by if b != "method"]]).drop(columns="_o")
    return table.reset_index(drop=True)


def summary_table(trials: pd.DataFrame, by: list[str], metric: str = "success") -> pd.DataFrame:
    """Per group of `by`: `n`, `value` and a 95% CI for any per-trial `metric`.

    A binary metric (success, on_target) gives the rate with a Wilson interval; a
    continuous one the mean of its non-NaN values with a t interval. Same columns
    either way, so tables and plots need not care which.
    """
    binary = _is_binary(trials[metric])
    rows = []
    for key, group in _groups(trials, by):
        values = group[metric].dropna().to_numpy(dtype=float)
        n = len(values)
        if binary:
            value = values.mean() if n else np.nan
            low, high = wilson(int(values.sum()), n)
        elif n > 1:
            value = values.mean()
            low, high = stats.t.interval(0.95, n - 1, loc=value, scale=stats.sem(values))
        else:
            value, low, high = (values.mean() if n else np.nan), np.nan, np.nan
        rows.append({**key, "n": n, "value": value, "ci95_low": low, "ci95_high": high})
    return _in_method_order(pd.DataFrame(rows, columns=[*by, "n", "value", "ci95_low", "ci95_high"]), by)


def best_depth(trials: pd.DataFrame, metric: str = "success") -> pd.DataFrame:
    """Per steering method: the depth (and authority) with the highest `metric`, pooled over tasks; ties to the lowest depth."""
    curve = summary_table(trials, ["method", "depth", "authority"], metric)
    curve = curve[curve["depth"].notna()]
    return curve.loc[curve.groupby("method")["value"].idxmax()].reset_index(drop=True)


def best_vs_best(
    trials: pd.DataFrame, metric: str = "success"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Each steering method at the depth where it scored best *on these trials*, and their paired tests.

    Selection and test use the same data, so the comparison is optimistic for every
    method alike. Returns (best, comparisons, selected): the `best_depth` table, one
    `paired_test` row per pair of steered methods (paired by recorded reset), and the
    trials at those depths.
    """
    best = best_depth(trials[trials["method"].isin(STEERED)], metric)
    chosen = best.set_index("method")["depth"]
    selected = trials[trials["method"].isin(chosen.index)]
    selected = selected[selected["depth"] == selected["method"].map(chosen)]
    return best, paired_comparisons(selected, "method", metric=metric), selected


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
    `paired_test(trials[trials.depth == 4], "method", "FRS", "FRS-RA", ["task_id", "trial"])`
    compares two methods at one depth, paired by scene reset and pooled over tasks.
    """
    required = ["pair_id", "initial_state_hash"]
    selected = trials[trials[by].isin([a, b])]
    if any(k not in selected for k in required) or selected[required].isna().any().any():
        raise ValueError("Unverified pairing: pair_id and initial_state_hash are required on every trial.")
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
    trials: pd.DataFrame, by: str = "run", pair_on: list[str] | None = None, metric: str = "success"
) -> pd.DataFrame:
    """`paired_test` over every pair of values of `by` (METHODS order for methods, else first appearance)."""
    values = list(dict.fromkeys(trials[by]))
    if by == "method":
        values.sort(key=lambda m: METHODS.index(m) if m in METHODS else len(METHODS))
    return pd.DataFrame([paired_test(trials, by, a, b, pair_on, metric) for a, b in combinations(values, 2)])


# ---------------------------------------------------------------- per-trial metrics from the step arrays


def _reference_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """The ceiling anchor's successes, or every success when there is no anchor."""
    wins = trials[trials["success"] & trials["mode_expressed"].notna()]
    ceiling = wins[wins["method"] == "policy (ceiling)"]
    return ceiling if len(ceiling) else wins


def task_targets(trials: pd.DataFrame) -> dict[int, str | None]:
    """The scene element each task is about: what moved most in the reference trials, None if it never succeeded."""
    targets: dict[int, str | None] = {}
    for task_id, group in trials.groupby("task_id"):
        pool = _reference_trials(group)
        targets[int(task_id)] = pool["mode_expressed"].mode().iloc[0] if len(pool) else None
    return targets


def _locate(z, element: str):
    """The (kind, (T, d)) track of a scene element in a trial's arrays, or None."""
    for kind, names_key, track_key in (
        ("object", "object_names", "object_pos"),
        ("articulation", "articulation_names", "articulation_qpos"),
    ):
        if names_key not in z.files:
            continue
        names = [str(n) for n in z[names_key]]
        if element in names:
            track = np.asarray(z[track_key][:, names.index(element)], dtype=np.float64)
            return kind, track.reshape(len(track), -1)
    return None


def task_goals(trials: pd.DataFrame) -> dict[int, dict | None]:
    """Where each task's target element ends up when the task is done.

    The target is `task_targets`; the goal is the set of its final positions
    (objects, (G, 3)) or joint values (articulations, (G, 1)) over the reference
    trials. None when the task never succeeded or the arrays lack the element.
    """
    goals: dict[int, dict | None] = {}
    for task_id, target in task_targets(trials).items():
        goals[int(task_id)] = None
        if target is None:
            continue
        group = trials[(trials["task_id"] == task_id) & (trials["mode_expressed"] == target)]
        finals, kind = [], None
        for _, row in _reference_trials(group).iterrows():
            with np.load(Path(row["run_dir"]) / row["steps_file"]) as z:
                located = _locate(z, target)
            if located is not None:
                kind, track = located
                finals.append(track[-1])
        if finals:
            goals[int(task_id)] = {"element": target, "kind": kind, "positions": np.stack(finals)}
    return goals


def step_authority(action: np.ndarray, policy: np.ndarray, command: np.ndarray) -> np.ndarray:
    """Per-step user authority in [0, 1]: how far the executed translation moved from the
    policy's own plan toward the operator's command.

    With `a` executed, `p` the plan and `r` the command, it is the barycentric coordinate
    of `a`'s projection onto the segment p -> r, `<a - p, r - p> / |r - p|^2`, clipped to
    [0, 1]. Swapping p and r gives exactly 1 minus it (before and after clipping), so the
    policy's authority is the complement; motion perpendicular to the segment belongs
    to neither and is ignored. NaN where p = r, when there was nothing to decide.
    """
    a, p, r = (np.asarray(x, dtype=np.float64) for x in (action, policy, command))
    d = r - p
    denominator = (d * d).sum(axis=1)
    out = np.full(len(a), np.nan)
    defined = denominator > 1e-12
    out[defined] = np.clip(((a - p) * d).sum(axis=1)[defined] / denominator[defined], 0.0, 1.0)
    return out


def _mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = norms > 1e-9
    return float(((a * b).sum(axis=1)[ok] / norms[ok]).mean()) if ok.any() else np.nan


def trial_metrics(trials: pd.DataFrame) -> pd.DataFrame:
    """Per-trial metrics from the step arrays (one read per trial), joined onto `trials`.

    Operator engagement: `commanding_frac` (steps pushing beyond the deadband),
    `mean_speed_when_active`, `mean_corruption_shift` (|served - raw| while pushing),
    `gripper_closed_frac`, `reads_this_trial`. Intent transmission: `cos_raw`, the mean
    cosine between the executed translation and the operator's *raw* command (the
    clean intent), and `cos_served`, the same against the corrupted command the policy
    was given -- a method that undoes the corruption has `cos_raw` above `cos_served`.

    Path: `path_length` (end-effector distance travelled, m), `displacement`
    (straight-line start to end) and `path_efficiency` (their ratio). Time:
    `time_to_success` is `steps` for successes and NaN for failures, which run to the
    step cap -- average it over successes only, or use `success_over_time`. Progress
    measures the task's target element against the goal set (`task_goals`):
    `goal_distance_min` is the closest it came, `progress_final` and `progress_max`
    are 1 - distance / initial distance at the end and at best, clipped to [0, 1];
    NaN when the task has no goal or the trial lacks the element.

    Authority (`step_authority`, needs the `policy_translation` array): `user_authority`
    is the mean over the trial's steps, `user_authority_pushing` the mean over the steps
    where the operator commanded beyond the deadband, `policy_authority` its complement.
    """
    goals = task_goals(trials)

    def one(row) -> pd.Series:
        with np.load(Path(row["run_dir"]) / row["steps_file"]) as z:
            raw, served, executed = z["user_translation_raw"], z["user_translation"], z["action"][:, :3]
            gripper, reads, eef = (
                z["user_gripper"],
                z["user_reads"],
                np.asarray(z["eef_pos"], dtype=np.float64),
            )
            plan = z["policy_translation"] if "policy_translation" in z.files else None
            goal = goals.get(int(row["task_id"]))
            located = None if goal is None else _locate(z, goal["element"])
        active = np.abs(raw).max(axis=1) >= DEADBAND
        path_length = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else np.nan
        displacement = float(np.linalg.norm(eef[-1] - eef[0])) if len(eef) > 1 else np.nan
        out = {
            "n_steps": len(raw),
            "commanding_frac": float(active.mean()),
            "mean_speed_when_active": float(np.linalg.norm(raw[active], axis=1).mean())
            if active.any()
            else 0.0,
            "mean_corruption_shift": float(np.linalg.norm((served - raw)[active], axis=1).mean())
            if active.any()
            else 0.0,
            "gripper_closed_frac": float((gripper > 0).mean()),
            "reads_this_trial": int(reads.sum()),
            "cos_raw": _mean_cos(raw[active], executed[active]),
            "cos_served": _mean_cos(served[active], executed[active]),
            "path_length": path_length,
            "displacement": displacement,
            "path_efficiency": displacement / path_length if path_length else np.nan,
            "time_to_success": float(row["steps"]) if row["success"] else np.nan,
            "user_authority": np.nan,
            "user_authority_pushing": np.nan,
            "policy_authority": np.nan,
            "goal_distance_min": np.nan,
            "progress_final": np.nan,
            "progress_max": np.nan,
        }
        if plan is not None:
            authority = step_authority(executed, plan, served)
            if np.isfinite(authority).any():
                out["user_authority"] = float(np.nanmean(authority))
                out["policy_authority"] = 1.0 - out["user_authority"]
            if np.isfinite(authority[active]).any():
                out["user_authority_pushing"] = float(np.nanmean(authority[active]))
        if located is not None:
            _, track = located
            distance = np.linalg.norm(track[:, None, :] - goal["positions"][None, :, :], axis=2).min(axis=1)
            out["goal_distance_min"] = float(distance.min())
            if distance[0] > 0:
                progress = np.clip(1.0 - distance / distance[0], 0.0, 1.0)
                out["progress_final"], out["progress_max"] = float(progress[-1]), float(progress.max())
        return pd.Series(out)

    return trials.join(trials.apply(one, axis=1), rsuffix="_npz")


def success_over_time(trials: pd.DataFrame, by: list[str], horizon: int | None = None) -> pd.DataFrame:
    """Fraction of each group's trials that had succeeded by step k, for k = 0..horizon.

    Failures count as never succeeding (they ran to the cap), so this is the empirical
    CDF of the success time with censoring handled by construction. `horizon` defaults
    to the longest trial.
    """
    horizon = int(trials["steps"].max()) if horizon is None else horizon
    steps = np.arange(horizon + 1)
    rows = []
    for key, group in _groups(trials, by):
        done = np.sort(group.loc[group["success"], "steps"].to_numpy())
        fraction = np.searchsorted(done, steps, side="right") / len(group)
        rows.append(pd.DataFrame({**key, "step": steps, "fraction": fraction}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- plots


def plot_metric(
    trials: pd.DataFrame, ax, metric: str = "success", x: str = "authority", title: str | None = None
):
    """One line per steering method: `metric` against `x`.

    Any per-trial metric works: a binary one (success, on_target) is drawn as a rate
    with Wilson 95% error bars, a continuous one (path length, progress, ...) as a
    mean with t-based 95% error bars. Pooled over the tasks in `trials`. Methods
    without a depth (the policy-only anchors) are dashed horizontal bands. `x` is
    "authority" (default: the scheduled share of the denoising schedule, the same
    reading for both depth conventions), "depth", or any per-trial metric such as
    `user_authority`, in which case each method/depth cell sits at that metric's cell
    mean. Returns the table.
    """
    binary = _is_binary(trials[metric])
    if x in X_LABELS:
        table = summary_table(trials, ["method", x], metric)
    else:  # a measured quantity: one point per method/depth cell, at the cell's mean of x
        table = summary_table(trials, ["method", "depth"], metric)
        means = trials.groupby(["method", "depth"], dropna=False)[x].mean().rename(x).reset_index()
        table = table.merge(means, on=["method", "depth"], how="left")
    for method, group in table.groupby("method", sort=False):
        group = group.sort_values(x)
        color = METHOD_COLORS.get(method)
        if group[x].isna().all():
            value = group["value"].iloc[0]
            label = f"{method} ({value:.0%})" if binary else f"{method} ({value:.2f})"
            style = "--" if method == "policy (ceiling)" else ":"
            ax.axhline(value, linestyle=style, linewidth=1, color=color, label=label)
            if group["ci95_low"].notna().all():
                ax.axhspan(group["ci95_low"].iloc[0], group["ci95_high"].iloc[0], alpha=0.08, color=color)
            continue
        err = np.vstack([group["value"] - group["ci95_low"], group["ci95_high"] - group["value"]])
        ax.errorbar(
            group[x],
            group["value"],
            yerr=np.nan_to_num(err),
            marker="o",
            capsize=3,
            label=method,
            color=color,
        )
    if x in X_LABELS:
        ax.set_xlabel(X_LABELS[x])
        ax.set_xticks(sorted(table[x].dropna().unique()))  # the sampled settings, nothing in between
    else:
        ax.set_xlabel(METRIC_LABELS.get(x, x.replace("_", " ").capitalize()))
        ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize()))
    if binary:
        ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend()
    return table


def plot_success_over_time(trials: pd.DataFrame, ax, by: str = "method", title: str | None = None):
    """Fraction of trials succeeded by step k, one curve per value of `by`. Returns the table."""
    curve = success_over_time(trials, [by])
    for key, group in curve.groupby(by, sort=False):
        method = next((m for m in METHODS if str(key).startswith(m)), None)  # "FRS d2" -> FRS
        ax.step(
            group["step"], group["fraction"], where="post", label=str(key), color=METHOD_COLORS.get(method)
        )
    ax.set_xlabel("Control step")
    ax.set_ylabel("Fraction of trials succeeded")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend()
    return curve


def plot_best_vs_best(best: pd.DataFrame, comparisons: pd.DataFrame, ax, title: str | None = None):
    """Head-to-head at each method's best depth: a bar per method with its Wilson 95% interval,
    the depth and authority under it, and a bracket per pair giving the McNemar result
    (discordant pairs won by each side, and p). Takes the tables `best_vs_best` returns.
    """
    best = best.reset_index(drop=True)
    x = np.arange(len(best))
    colors = [METHOD_COLORS.get(m) for m in best["method"]]
    err = np.vstack([best["value"] - best["ci95_low"], best["ci95_high"] - best["value"]])
    ax.bar(x, best["value"], width=0.5, color=colors, yerr=err, capsize=4, ecolor="#444444", linewidth=0)
    for xi, value, n in zip(x, best["value"], best["n"], strict=True):
        ax.text(
            xi,
            0.02,
            f"{value:.0%}\n{int(round(value * n))}/{n}",
            ha="center",
            va="bottom",
            color="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{m}\ndepth {int(d)}, authority {a:.1f}"
            for m, d, a in zip(best["method"], best["depth"], best["authority"], strict=True)
        ]
    )
    ax.set_ylabel("Success rate")
    if title:
        ax.set_title(title)
    # One bracket per comparison, stacked above the bars.
    pos = {m: i for i, m in enumerate(best["method"])}
    top = float(np.nanmax(best["ci95_high"])) + 0.06
    step = 0.11
    for k, row in enumerate(comparisons.itertuples(index=False)):
        a, b = pos[row.A], pos[row.B]
        y = top + k * step
        ax.plot([a, a, b, b], [y - 0.02, y, y, y - 0.02], color="#444444", linewidth=1)
        p = "no discordant pairs" if np.isnan(row.mcnemar_p) else f"p = {row.mcnemar_p:.3f}"
        wins = f"{row.B} {row.B_only} · {row.A} {row.A_only} of {row.paired_trials} pairs"
        ax.text(
            (a + b) / 2,
            y + 0.01,
            f"{row.B_rate - row.A_rate:+.0%} · {p}\n{wins}",
            ha="center",
            va="bottom",
            fontsize="small",
        )
    ax.set_ylim(0, top + len(comparisons) * step + 0.06)
    ax.set_yticks(np.linspace(0, 1, 6))  # the headroom above 100% only holds the brackets
    ax.spines["left"].set_bounds(0, 1)
    return best
