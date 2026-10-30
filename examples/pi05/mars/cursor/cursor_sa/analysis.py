"""Analysis helpers: the shared figure style and scene drawing, per-trial kinematics
of simulated rollouts, goal-choice statistics of the goal-hidden policy, and
statistics of logged trial records (a study's blocks)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK_2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5df"


# ---------------------------------------------------------------- figures


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 150,
                         "savefig.bbox": "tight", "text.color": INK})


def draw_scene(ax, layout, task, show_goal=True, obstacle_style=None):
    style = obstacle_style or dict(facecolor="#dedcd6", edgecolor="#a5a49e", lw=0.8)
    for x, y, r in layout.real_obstacles(0):
        ax.add_patch(plt.Circle((x, y), r, zorder=1, **style))
    for j in np.flatnonzero(layout.mask[0]):
        is_goal = show_goal and j == layout.goal[0]
        ax.add_patch(plt.Circle(layout.targets[0, j], task.target_radius, zorder=1,
                                facecolor="#fbe3a1" if is_goal else "#ecebe6",
                                edgecolor=INK if is_goal else "#c9c8c1", lw=1.2 if is_goal else 0.8))
    ax.plot(*layout.start[0], "o", color=INK, ms=5, zorder=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)


# ---------------------------------------------------------------- kinematics


def hold_last(path, n):
    """Pad paths to n points by holding the last position."""
    return np.concatenate([path, np.repeat(path[:, -1:], max(0, n - path.shape[1]), 1)], 1)[:, :n]


def trial_metrics(env, path, layouts, task):
    """Per-trial outcome and kinematics. The movement phase runs from the start
    to the first time the cursor enters the goal (or the end of the trial)."""
    B, T = path.shape[:2]
    rows, idx = np.arange(B), np.arange(T)[None]
    start, goal = layouts.start, layouts.goal_pos
    dist = np.linalg.norm(goal - start, axis=1)
    along = (goal - start) / dist[:, None]
    across = np.stack([-along[:, 1], along[:, 0]], 1)

    inside = (np.linalg.norm(path - goal[:, None], axis=-1) < task.target_radius) & (idx <= env.steps[:, None])
    arrive = np.where(inside.any(1), inside.argmax(1), env.steps)
    moving = idx[:, :-1] < arrive[:, None]
    step = np.diff(path, axis=1)
    length = (np.linalg.norm(step, axis=-1) * moving).sum(1)

    lateral = np.where(idx <= arrive[:, None], ((path - start[:, None]) * across[:, None]).sum(-1), 0)
    bow = lateral[rows, np.abs(lateral).argmax(1)] / dist
    accel = np.linalg.norm(np.diff(step, axis=1), axis=-1) / task.dt ** 2
    accel = (accel * moving[:, 1:]).sum(1) / np.maximum(moving[:, 1:].sum(1), 1)

    speed = np.linalg.norm(step, axis=-1) / task.dt
    grid = np.linspace(0, 1, 50)
    profile = np.stack([np.interp(grid * max(a, 1), np.arange(T - 1), s) for a, s in zip(arrive, speed)])

    return dict(success=env.success, wrong=(env.selected >= 0) & ~env.success, collision=env.collided,
                timeout=(env.selected < 0) & ~env.collided, time_s=env.steps * task.dt, efficiency=np.linalg.norm(path[rows, arrive] - start, axis=1)
                / np.maximum(length, 1e-9), bow=bow, accel=accel, profile=profile)


# ---------------------------------------------------------------- goal choice


def eligible(layouts, task):
    """Targets that could have been the goal in training (far enough from the start)."""
    d = np.linalg.norm(layouts.targets - layouts.start[:, None], axis=-1)
    return layouts.mask & (d >= task.min_reach)


def spread(counts, elig):
    """Entropy of each layout's choices over its eligible targets, divided by
    log(n eligible): 1 is an even spread, 0 is always the same target.
    Only layouts with 2+ eligible targets and at least one selection count."""
    picked = counts.sum(1)
    use = (elig.sum(1) > 1) & (picked > 0)
    if not use.any():
        return float("nan")
    p = counts[use] / picked[use, None]
    ent = -(p * np.log(np.where(p > 0, p, 1))).sum(1)
    return float(np.mean(ent / np.log(elig[use].sum(1))))


def choice_stats(selected, layouts, task, rng):
    """selected [L, R]: target picked on each of R runs per layout (-1: none)."""
    L, R = selected.shape
    N = layouts.mask.shape[1]
    counts = (selected[..., None] == np.arange(N)).sum(1)
    elig = eligible(layouts, task)
    # The same statistic for an ideal policy (uniform over eligible targets), with the same R runs.
    ideal = np.zeros_like(counts)
    for b in range(L):
        ideal[b] = np.bincount(rng.choice(np.flatnonzero(elig[b]), R), minlength=N)
    picked = max(counts.sum(), 1)
    return dict(selected_any=float((selected >= 0).mean()), no_selection=float((selected < 0).mean()),
                ineligible=float(counts[~elig].sum() / picked),
                spread=spread(counts, elig), spread_ideal=spread(ideal, elig))


# ---------------------------------------------------------------- trial records


def bootstrap_mean(x, rng, n_boot=4000):
    """[mean, lower, upper]: the mean of x with a bootstrap 95% CI."""
    x = np.asarray(x, dtype=float)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_lead(trials, focus, other, rng, n_boot=4000):
    """Hit rate of arm `focus` minus that of arm `other` over the layouts both played,
    paired by (participant, run, layout_id), with a paired bootstrap 95% CI: [lead, lower, upper]."""
    hits = {}
    for r in trials:
        if r["arm"] in (focus, other):
            hits.setdefault((r.get("participant"), r["run"], r["layout_id"]), {})[r["arm"]] = float(r["outcome"] == "hit")
    return bootstrap_mean([h[focus] - h[other] for h in hits.values() if focus in h and other in h], rng, n_boot)


def cumulative_hits(trials, times):
    """The fraction of all trials that have selected the goal by each time in `times`."""
    t_hit = np.array([r["duration_s"] for r in trials if r["outcome"] == "hit"])
    return np.array([(t_hit <= t).sum() / len(trials) for t in times])


def aligned_path(record):
    """The trial in reach coordinates: the start at the origin, the goal at (1, 0), lengths
    in units of the reach. Returns (path [T, 2], targets [N, 2], target radius)."""
    start, targets = np.asarray(record["start"], float), np.asarray(record["targets"], float)
    goal = targets[record["goal"]]
    reach = np.linalg.norm(goal - start)
    along = (goal - start) / reach
    basis = np.stack([along, [-along[1], along[0]]], 1)  # columns: along the reach, across it
    to = lambda p: (np.asarray(p, float) - start) @ basis / reach
    return to(record["path"]), to(targets), record["target_radius"] / reach


def path_efficiency(record):
    """Straight-line distance over path length, from the start to the first entry into
    the goal; None unless the trial is a hit."""
    if record["outcome"] != "hit":
        return None
    path, _, radius = aligned_path(record)
    inside = np.linalg.norm(path - [1.0, 0.0], axis=1) < radius
    if not inside.any():
        return None
    end = inside.argmax()
    length = np.linalg.norm(np.diff(path[:end + 1], axis=0), axis=1).sum()
    return float(np.linalg.norm(path[end]) / length) if length > 0 else None
