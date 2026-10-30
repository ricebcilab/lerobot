"""Tune-then-test sweeps of the steering methods with a simulated agent, on the portal's mechanism.

    python run_sweep.py --sweep gain                                   # configs/sweeps/gain.yaml
    python run_sweep.py --sweep study_condition --tune-trials 50 --test-trials 50   # a quick look
    python run_sweep.py --sweep task_space --resume                    # continue an interrupted sweep
    python run_sweep.py --sweep grid --cells U1xT3 U2xT4 --out grid_part1   # some cells, in a process of their own
    python run_sweep.py --sweep grid --merge grid_part1 grid_part2          # combine the parts into grid
    python run_sweep.py --sweep grid --fixed-depths results/sim/sweeps/grid_s0/depths.yaml --out grid_again   # no tuning

A sweep config (configs/sweeps/<name>.yaml) names the agent, the base arm, the
methods (overrides of the arm), the depths to try (one list, or a list per
mode: flow_control's tau counts chunk steps of 16, frs's n_reverse_steps
Euler steps of flow_steps), the grid and the trial
counts. The grid is a corruption family's levels x input noises, optionally x
task axes (`grid.axes`: arm parameters such as targets, min_target_sep, reach or
dwell_s, each with a list of values, or a mapping name -> value; every
combination is a cell), or a USERS x TASKS grid (`grid.users`: named corruption
profiles with their user-module checkpoint, `grid.tasks`: named obstacle specs
with their task-module checkpoint and optionally a time limit of their own
(`time_limit_s`), `grid.train`: the cells trained on; a
method's reversal_adapter may say `cell` in its user or task slot, resolved per
cell to that cell's checkpoint, and a method's `model` or adapter slot may be
given per user or per task, nested in either order, e.g. a policy retrained for
each task with user modules calibrated on it). For every
cell each method's depth (tau for flow_control, n_reverse_steps for frs) is
tuned on the tuning trials, then every method is tested at its tuned depth on
new trials (new layouts and noise); the focus method's lead over the others
comes with a paired bootstrap 95% CI, since every method sees the same layouts.
The showcase cell (the focus method's largest tuned lead) is chosen on the
tuning trials only. Every trial runs through session.run_trials, exactly as a
participant's. Cells whose layouts cannot be sampled (too many targets for the
spacing) are skipped and listed.

Writes results/sim/sweeps/<name>/: tuning.json, test.json, depths.yaml (the
tuned depths, to copy into arm configs), ranking.csv (every cell, best lead
first), log.txt (everything the run printed) and figs/ (a users x tasks grid:
grid.png, the focus method's maps, and summary.png, every method on every
cell). test.json keeps every test trial of a cell (`trials`: its
outcome and duration per method, and the side of the reach the expert's route
passes the obstacle on) and, with obstacles, summarizes the cell by that side
(`by_side`). Results are saved after every group of cells, so --resume
continues where a run stopped. Every cell draws its layouts, noise and bootstrap
from seeds of its own, so its results do not depend on which cells run with it:
--cells runs a part of a sweep (several parts can run at once, each with its own
--out) and --merge combines the parts' outputs. --fixed-depths skips tuning and tests
every method at the depths a depths.yaml gives (a cell per key, as a sweep writes it): the test trials
are the same ones a tuned run would test at those depths, tuning.json is empty and depths.yaml is a copy
of the input.
"""

import argparse
import csv
import itertools
import json
import shutil
import time
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from cursor_sa import config
from cursor_sa.agents import make_agent
from cursor_sa.analysis import INK, INK_2, bootstrap_mean, draw_scene, style
from cursor_sa.env import ROUTE_CLEARANCE, TaskConfig, center_out, curl_steady_state, plan_route, route_side, sample_layouts
from cursor_sa.paths import RESULTS
from cursor_sa.records import logged, summarize
from cursor_sa.session import arm_task, channel_for, load_checkpoints, run_trials

CURL_TAU = 0.1  # s, cursor inertia for the curl family
FAMILIES = {  # level -> arm overrides; level -> the family's reversal adapter (the known corruption)
    "gain": (lambda g: {"corruption": {"scale": [1.0, float(g)]}}, lambda g: {"translation": "corruption"}),
    "rotation": (lambda d: {"corruption": {"rotation_deg": float(d)}}, lambda d: {"translation": "corruption"}),
    "curl": (lambda b: {"velocity_tau": CURL_TAU, "curl": float(b)},
             lambda b: {"F": np.linalg.inv(curl_steady_state(CURL_TAU, float(b))).tolist()}),
    "none": (lambda _: {}, lambda _: {"translation": "corruption"}),
}
FAMILY_LABEL = {"gain": "Vertical gain g", "rotation": "Rotation (deg)", "curl": "Curl b (1/s)", "none": "", "grid": ""}
COLORS = ["#8a8984", "#eb6834", "#1baf7a", "#2a78d6", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_LABEL = {"teleop": "Teleop", "flow_control": "Flow control", "frs": "FRS", "mars_user": "MARS-U", "mars_task": "MARS-T",
                "mars": "MARS", "retrained_mars_user": "Retrained policy + MARS-U", "mars_fixed": "MARS (F = M)"}
SWEEP_KEYS = {"name", "agent", "arm", "methods", "depths", "grid", "trials", "focus", "against", "seed", "traces"}
AXIS_KEYS = set(config.ARM_KEYS.values()) - {"mode", "tau", "lam", "n_reverse_steps"}  # what grid.axes may vary
GRID_KEYS = {"users", "tasks", "noises", "train"}           # a users x tasks grid
FAMILY_GRID_KEYS = {"family", "levels", "noises", "axes"}   # a corruption family's levels x noises (x task axes)


def load_sweep(path):
    path = Path(path)
    if not path.exists():
        path = config.CONFIG_DIR / "sweeps" / (path.name if path.suffix else path.name + ".yaml")
    data = yaml.safe_load(path.read_text())
    unknown = set(data) - SWEEP_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    depths = data.get("depths", list(range(1, 11)))
    if isinstance(depths, dict):  # a list per mode (flow_control's tau counts chunk steps, the others Euler steps)
        if set(depths) - set(DEPTH_MODES) or not all(isinstance(v, list) and v for v in depths.values()):
            raise ValueError(f"{path}: depths must be a list, or a mapping of {DEPTH_MODES} to lists")
    sweep = dict(name=data.get("name", path.stem), agent=data.get("agent", "expert"), arm=data.get("arm", "base.yaml"),
                 methods=data["methods"], depths=depths,
                 trials={"tune": 200, "test": 500, **data.get("trials", {})}, seed=data.get("seed", 100),
                 focus=data.get("focus"), against=data.get("against"), traces=data.get("traces"))
    grid = data.get("grid", {})
    if "users" in grid or "tasks" in grid:  # a users x tasks grid: every cell is a (corruption, obstacle) pair with its adapters
        if set(grid) - GRID_KEYS:
            raise ValueError(f"{path}: a users x tasks grid takes {sorted(GRID_KEYS)}, got {sorted(grid)}")
        cells = grid_cells(grid, path)
        for m, spec in data["methods"].items():  # a model or adapter slot given per user or task names every one
            slots = spec.get("reversal_adapter") if isinstance(spec.get("reversal_adapter"), dict) else {}
            for where, value in [("model", spec.get("model"))] + [(f"reversal_adapter.{k}", slots.get(k)) for k in ("user", "task")]:
                check_cell_mapping(value, grid, f"{path}: methods.{m}.{where}")
        sweep["grid"] = dict(family="grid", levels=list(cells), noises=[float(s) for s in grid["noises"]], axes={},
                             axis_values={}, cells=cells, users=list(grid["users"]), tasks=list(grid["tasks"]))
        return finish_sweep(sweep, path)
    if set(grid) - FAMILY_GRID_KEYS:
        raise ValueError(f"{path}: grid takes {sorted(FAMILY_GRID_KEYS)} (or users, tasks, noises, train), got {sorted(grid)}")
    axes = grid.get("axes") or {}
    bad = [k for k, v in axes.items() if k not in AXIS_KEYS or not isinstance(v, (list, dict))]
    if bad:
        raise ValueError(f"{path}: grid.axes must map arm parameters (not the mode or depth) to lists of values, or to "
                         f"named values; check {bad}")
    # an axis is a list of values, or a mapping name -> value; cells carry the names, which make the cell keys
    values = {k: dict(v) for k, v in axes.items() if isinstance(v, dict)}
    sweep["grid"] = dict(family=grid.get("family", "none"), levels=grid.get("levels", [None]),
                         noises=[float(s) for s in grid["noises"]],
                         axes={k: (list(v) if isinstance(v, dict) else v) for k, v in axes.items()}, axis_values=values)
    if sweep["grid"]["family"] not in FAMILIES:
        raise ValueError(f"{path}: grid.family must be one of {sorted(FAMILIES)}")
    return finish_sweep(sweep, path)


def finish_sweep(sweep, path):
    if sweep["agent"] not in ("expert", "expert_chunk"):
        raise ValueError(f"{path}: agent must be expert or expert_chunk")
    sweep["focus"] = sweep["focus"] or list(sweep["methods"])[-1]
    sweep["against"] = sweep["against"] or [m for m in sweep["methods"] if m != sweep["focus"]]
    return sweep


def grid_cells(grid, where):
    """The cells of a users x tasks grid, 'UxT' -> settings, in user-major order."""
    users, tasks = grid.get("users") or {}, grid.get("tasks") or {}
    for kind, items, keys in (("users", users, {"corruption", "adapter"}), ("tasks", tasks, {"obstacle", "adapter", "time_limit_s"})):
        if not isinstance(items, dict) or not items:
            raise ValueError(f"{where}: grid.{kind} must map names to settings")
        for name, spec in items.items():
            if not isinstance(spec, dict) or set(spec) - keys:
                raise ValueError(f"{where}: grid.{kind}.{name} takes {sorted(keys)}, got {spec!r}")
    train = [tuple(pair) for pair in grid.get("train") or []]
    for u, t in train:
        if u not in users or t not in tasks:
            raise ValueError(f"{where}: grid.train names an unknown cell {u} x {t}")
    return {f"{u}x{t}": dict(user=u, task=t, corruption=user.get("corruption"), obstacle=task.get("obstacle"),
                             time_limit_s=task.get("time_limit_s"), user_adapter=user.get("adapter"),
                             task_adapter=task.get("adapter"), train=(u, t) in train)
            for u, user in users.items() for t, task in tasks.items()}


def check_cell_mapping(value, grid, where):
    """A value given per user or per task of a users x tasks grid (nested in either order) names every one."""
    if not (isinstance(value, dict) and value):
        return
    for kind in ("users", "tasks"):
        names = set(grid.get(kind) or ())
        if set(value) & names:
            if set(value) != names:
                raise ValueError(f"{where} must name every one of grid.{kind} {sorted(names)}, got {sorted(value)}")
            for k, v in value.items():
                check_cell_mapping(v, grid, f"{where}.{k}")
            return


def per_cell(value, cell, grid):
    """A value given per user or per task of the grid (nested in either order), resolved to the cell's."""
    while isinstance(value, dict) and value and (set(value) <= set(grid["users"]) or set(value) <= set(grid["tasks"])):
        value = value[cell["user"] if set(value) <= set(grid["users"]) else cell["task"]]
    return value


def cell_adapter(spec, cell, grid):
    """A method's reversal_adapter resolved for a cell: a user or task slot given per user or per task gives the
    cell's checkpoint, and `cell` the cell's own (None where the cell has none, e.g. the task slot of an
    obstacle-free task)."""
    if not (isinstance(spec, dict) and spec and set(spec) <= {"user", "task"}):
        return spec
    def resolve(slot):
        value = per_cell(spec.get(slot), cell, grid)
        return cell[f"{slot}_adapter"] if value == "cell" else value
    return {slot: resolve(slot) for slot in ("user", "task")}


def method_arm(sweep, method, level, axes=None):
    """The arm a method plays at a grid level (a corruption level, or a users x tasks cell) and task
    axes; channel noise is set per trial later."""
    grid = sweep["grid"]
    values = grid.get("axis_values", {})
    axes = {k: values[k][v] if k in values else v for k, v in (axes or {}).items()}
    spec = {k: v for k, v in sweep["methods"][method].items() if k != "channel"}
    if grid["family"] == "grid":
        cell = grid["cells"][level]
        limit = {} if cell["time_limit_s"] is None else {"time_limit_s": cell["time_limit_s"]}  # a task's own, if it sets one
        over = {"corruption": cell["corruption"], "obstacle": cell["obstacle"], **limit, **axes, **spec}
        if "model" in spec:  # a policy per task (or user), e.g. one retrained for each task
            over["model"] = per_cell(spec["model"], cell, grid)
        if "reversal_adapter" in spec:
            over["reversal_adapter"] = cell_adapter(spec["reversal_adapter"], cell, grid)
    else:
        level_over, family_adapter = FAMILIES[grid["family"]]
        over = {**level_over(level), **axes, **spec}
        if over.get("reversal_adapter") == "family":
            over["reversal_adapter"] = family_adapter(level)
    return config.load_arm(sweep["arm"], over)


def with_depth(arm, depth):
    arm = dict(arm)
    if arm["mode"] == "flow_control":
        arm["tau"] = depth
    elif arm["mode"] == "frs":
        arm["n_reverse_steps"] = None if depth == arm["flow_steps"] else depth
    return arm


def has_depth(arm):
    return arm["mode"] in DEPTH_MODES


DEPTH_MODES = ("flow_control", "frs")


def depths_for(sweep, mode):
    """The depths a sweep tries for a mode: one list for every mode, or a mapping mode -> list (flow_control's
    tau counts chunk steps of 16, the other depths Euler steps of flow_steps, so their ranges differ)."""
    depths = sweep["depths"]
    if isinstance(depths, dict):
        if mode not in depths:
            raise ValueError(f"{sweep['name']}: depths gives no list for mode {mode!r}")
        return list(depths[mode])
    return list(depths)


def run_cell_batch(sweep, arm, method, layouts, noises, agent, policy, adapters, device, seed):
    """One method at one depth over all noises of a level at once; per-noise records."""
    n = len(layouts)
    off = sweep["methods"][method].get("channel") in ("off", False)  # YAML reads `off` as False
    channel = channel_for(arm, np.random.default_rng(seed), np.repeat(noises, n), off)
    torch.manual_seed(seed)
    records = run_trials(arm, layouts.tile(len(noises)), agent, channel, policy, adapters, device)
    return {str(s): records[i * n:(i + 1) * n] for i, s in enumerate(noises)}


def pass_sides(layouts, target_radius):
    """Per trial, the side of the reach the expert's first route passes its obstacles on (`env.route_side`:
    left, right or straight); straight without obstacles."""
    out = []
    for b in range(len(layouts)):
        obstacles = layouts.real_obstacles(b)
        route = None if not len(obstacles) else plan_route(layouts.start[b], layouts.goal_pos[b], obstacles, ROUTE_CLEARANCE,
                                                            avoid=layouts.distractors(b, target_radius))
        out.append("straight" if route is None else route_side(route, layouts.start[b], layouts.goal_pos[b]))
    return out


def best_depth(per_depth):
    """Highest success, then the shortest median time."""
    return max(per_depth, key=lambda d: (per_depth[d]["success"], -(per_depth[d]["time_s"] or 1e9)))


# ---------------------------------------------------------------- cells


def cell_groups(sweep):
    """Every (level, task axes) combination of the grid, in grid order. The cells
    of a group differ only in input noise and run together."""
    axes = sweep["grid"]["axes"]
    combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    return [(level, combo) for level in sweep["grid"]["levels"] for combo in combos]


def _fmt(value):
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt(v) for v in value)
    if isinstance(value, dict):
        return "-".join(f"{k}{_fmt(v)}" for k, v in value.items())
    return "none" if value is None else str(value)


def cell_key(level, noise, axes=None):
    """'<level>|<noise>' plus '|<parameter>=<value>' for every task axis."""
    return f"{'-' if level is None else level}|{noise}" + "".join(f"|{k}={_fmt(v)}" for k, v in (axes or {}).items())


def group_label(sweep, level, axes):
    family = FAMILY_LABEL[sweep["grid"]["family"]].split(" (")[0]
    parts = ([f"{family} {level}".strip()] if level is not None else []) + [f"{k} {_fmt(v)}" for k, v in axes.items()]
    return ", ".join(parts)


def cell_label(sweep, cell):
    return ", ".join(filter(None, [group_label(sweep, cell["level"], cell.get("axes", {})), f"noise {cell['noise']}"]))  # older cells: no axes


# ---------------------------------------------------------------- figures


def fig_depth_curves(sweep, tuning, out, groups):
    """Success against depth on the tuning trials: a row per group, a column per noise."""
    noises = sweep["grid"]["noises"]
    fig, axs = plt.subplots(len(groups), len(noises), figsize=(2.6 * len(noises), 2.2 * len(groups)),
                            sharex=True, sharey=True, squeeze=False)
    for r, (level, axes) in enumerate(groups):
        shown = set()
        for c, noise in enumerate(noises):
            ax, cell = axs[r, c], tuning[cell_key(level, noise, axes)]
            for j, m in enumerate(sweep["methods"]):
                color = COLORS[j % len(COLORS)]
                if "None" in cell[m]:
                    ax.axhline(100 * cell[m]["None"]["success"], color=color, ls="--", lw=1.4)
                else:
                    depths = depths_for(sweep, method_arm(sweep, m, level, axes)["mode"])  # each mode's own list
                    shown.update(depths)
                    ax.plot(depths, [100 * cell[m][str(d)]["success"] for d in depths], color=color, lw=1.8)
            depths = sorted(shown) or [0]
            ax.set(ylim=(0, 102), xticks=[depths[0], depths[len(depths) // 2], depths[-1]])
            label = group_label(sweep, level, axes) if len(groups) > 1 else ""
            ax.set_title(", ".join(filter(None, [label, f"noise {noise}"])), fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel("Success (%)")
            if r == len(groups) - 1:
                ax.set_xlabel("depth (tau, lambda or n_reverse_steps)")
    handles = [plt.Line2D([], [], color=COLORS[j % len(COLORS)], lw=1.8) for j in range(len(sweep["methods"]))]
    fig.legend(handles, list(sweep["methods"]), loc="lower center", ncol=min(5, len(sweep["methods"])),
               bbox_to_anchor=(0.5, -0.04 if len(groups) > 2 else -0.14), fontsize=8)
    where = f" at {group_label(sweep, *groups[0])}" if len(groups) == 1 and groups[0][1] else ""
    fig.suptitle(f"{sweep['name']}: tuning trials{where}, success against depth (dashed: no depth)", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "depth_curves.png")
    plt.close(fig)


def fig_test(sweep, test, out):
    levels, noises, focus = sweep["grid"]["levels"], sweep["grid"]["noises"], sweep["focus"]
    if len(levels) > 1 and len(noises) > 1:  # maps: a baseline's success, and the focus method's lead
        base = sweep["against"][0]
        a = np.array([[test[cell_key(l, s)]["metrics"][base]["success"] for s in noises] for l in levels])
        lead = np.array([[test[cell_key(l, s)]["lead_over_best"][0] for s in noises] for l in levels])
        sig = np.array([[test[cell_key(l, s)]["beats_all"] for s in noises] for l in levels])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        from matplotlib.colors import LinearSegmentedColormap
        seq = LinearSegmentedColormap.from_list("seq", ["#f2f1ec", "#8a8984", "#3a3936"])
        div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ec", "#2a78d6"])
        lim = max(5.0, np.abs(100 * lead).max())
        for ax, data, cmap, vmin, vmax, fmt, title in [
                (axes[0], 100 * a, seq, 0, 100, "{:.0f}", f"{base} success (%)"),
                (axes[1], 100 * lead, div, -lim, lim, "{:+.0f}", f"{focus} lead over the best other method (points)")]:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set(xticks=range(len(noises)), xticklabels=noises, yticks=range(len(levels)), yticklabels=levels,
                   xlabel="Input noise", ylabel=FAMILY_LABEL[sweep["grid"]["family"]], title=title)
            ax.grid(False)
            for r in range(len(levels)):
                for c in range(len(noises)):
                    v = data[r, c]
                    dark = (v > 60) if ax is axes[0] else (abs(v) > 0.6 * lim)
                    ax.text(c, r, fmt.format(v) + ("*" if ax is axes[1] and sig[r, c] else ""), ha="center",
                            va="center", fontsize=9, color="white" if dark else INK)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.text(0.5, -0.02, f"Test trials, every method at its tuned depth. * {focus} beats each of "
                 f"{', '.join(sweep['against'])} (paired bootstrap 95% CI above 0).", ha="center", color=INK_2, fontsize=8)
        fig.tight_layout()
    else:  # lines over the varying axis
        xs = noises if len(noises) > 1 else levels
        keys = [cell_key(levels[0], s) for s in noises] if len(noises) > 1 else [cell_key(l, noises[0]) for l in levels]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for j, m in enumerate(sweep["methods"]):
            y = [100 * test[k]["metrics"][m]["success"] for k in keys]
            lo = [100 * test[k]["success_ci"][m][1] for k in keys]
            hi = [100 * test[k]["success_ci"][m][2] for k in keys]
            ax.fill_between(xs, lo, hi, color=COLORS[j % len(COLORS)], alpha=0.12, lw=0)
            ax.plot(xs, y, "-o", color=COLORS[j % len(COLORS)], lw=2.2 if m == focus else 1.6, ms=4, label=m)
        ax.set(xlabel="Input noise" if len(noises) > 1 else FAMILY_LABEL[sweep["grid"]["family"]],
               ylabel="Test success (%)", ylim=(0, 105), xticks=xs)
        ax.set_title(f"{sweep['name']}: every method at its tuned depth (95% CI)", color=INK, fontsize=9)
        ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out / "test.png", bbox_inches="tight")
    plt.close(fig)


def fig_axes(sweep, test, out):
    """The focus method's lead over the best other method against every varied
    parameter: one dot per test cell, the mean over the other parameters as a bar."""
    grid, focus = sweep["grid"], sweep["focus"]
    varied = {k: v for k, v in grid["axes"].items() if len(v) > 1}
    if len(grid["levels"]) > 1:
        varied = {"level": grid["levels"], **varied}
    if len(grid["noises"]) > 1:
        varied["noise"] = grid["noises"]
    value_of = lambda cell, k: cell["level"] if k == "level" else cell["noise"] if k == "noise" else cell["axes"][k]
    cells, rng = list(test.values()), np.random.default_rng(0)
    fig, axs = plt.subplots(1, max(1, len(varied)), figsize=(3.0 * max(1, len(varied)), 3.4), sharey=True,
                            squeeze=False)
    for ax, (k, values) in zip(axs[0], varied.items()):
        for i, v in enumerate(values):
            here = [c for c in cells if _fmt(value_of(c, k)) == _fmt(v)]
            lead = np.array([100 * c["lead_over_best"][0] for c in here])
            sig = np.array([c["beats_all"] for c in here], dtype=bool)
            x = i + rng.uniform(-0.18, 0.18, len(here))
            ax.scatter(x[~sig], lead[~sig], s=12, color="#8a8984", alpha=0.6, lw=0)
            ax.scatter(x[sig], lead[sig], s=12, color="#2a78d6", alpha=0.85, lw=0)
            if len(lead):
                ax.plot([i - 0.3, i + 0.3], [lead.mean()] * 2, color=INK, lw=2)
        ax.axhline(0, color=INK_2, lw=0.8, ls="--")
        ax.set(xticks=range(len(values)), xticklabels=[_fmt(v) for v in values], xlabel=k)
        ax.tick_params(axis="x", labelsize=8)
    axs[0, 0].set_ylabel(f"{focus} lead over the best other (points)")
    fig.suptitle(f"{sweep['name']}: test trials, tuned depths; one dot per cell, bar: mean. "
                 f"Blue: {focus} beats every other method (95% CI)", color=INK, fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "axes.png", bbox_inches="tight")
    plt.close(fig)


def fig_grid(sweep, test, out):
    """Users x tasks maps, one row per noise: the focus method's success, its lead over the best other
    method (* where it beats every method in `against`), and its collision rate; train cells outlined."""
    from matplotlib.colors import LinearSegmentedColormap
    cells, focus, noises = sweep["grid"]["cells"], sweep["focus"], sweep["grid"]["noises"]
    users, tasks = sweep["grid"]["users"], sweep["grid"]["tasks"]
    seq = LinearSegmentedColormap.from_list("seq", ["#f2f1ec", "#8a8984", "#3a3936"])
    div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ec", "#2a78d6"])
    fig, axes = plt.subplots(len(noises), 3, figsize=(10.5, 0.9 + 0.55 * len(users) * len(noises) + 1.2), squeeze=False)
    for r, noise in enumerate(noises):
        data = {k: np.full((len(users), len(tasks)), np.nan) for k in ("success", "lead", "collision")}
        sig, trained = np.zeros_like(data["lead"], dtype=bool), np.zeros_like(data["lead"], dtype=bool)
        for name, c in cells.items():
            t = test.get(cell_key(name, noise))
            if t is None:
                continue
            i, j = users.index(c["user"]), tasks.index(c["task"])
            data["success"][i, j] = 100 * t["metrics"][focus]["success"]
            data["lead"][i, j] = 100 * t["lead_over_best"][0]
            data["collision"][i, j] = 100 * t["metrics"][focus].get("collision", 0.0)
            sig[i, j], trained[i, j] = t["beats_all"], c["train"]
        lim = max(5.0, np.nanmax(np.abs(data["lead"])) if np.isfinite(data["lead"]).any() else 5.0)
        for ax, (k, cmap, vmin, vmax, fmt, title) in zip(axes[r], [
                ("success", seq, 0, 100, "{:.0f}", f"{focus} success (%)"),
                ("lead", div, -lim, lim, "{:+.0f}", f"{focus} lead over the best other (points)"),
                ("collision", seq, 0, 100, "{:.0f}", f"{focus} collisions (%)")]):
            im = ax.imshow(data[k], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set(xticks=range(len(tasks)), xticklabels=tasks, yticks=range(len(users)), yticklabels=users,
                   title=title + (f", noise {noise}" if len(noises) > 1 else ""))
            ax.grid(False)
            for i in range(len(users)):
                for j in range(len(tasks)):
                    v = data[k][i, j]
                    if np.isfinite(v):
                        dark = (v > 60) if k != "lead" else (abs(v) > 0.6 * lim)
                        ax.text(j, i, fmt.format(v) + ("*" if k == "lead" and sig[i, j] else ""), ha="center", va="center",
                                fontsize=8, color="white" if dark else INK)
                    if trained[i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.6))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.5, -0.01, f"Test trials, every method at its tuned depth. Outlined: cells trained on. * {focus} beats each of "
             f"{', '.join(sweep['against'])} (paired bootstrap 95% CI above 0).", ha="center", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "grid.png", bbox_inches="tight")
    plt.close(fig)


def pooled_success(cells, method, rng, n_boot=2000):
    """[mean, lower, upper]: a method's success averaged over cells, with a 95% CI that resamples every
    cell's test trials."""
    hits = [np.array([o == "hit" for o in c["trials"]["outcome"][method]], dtype=float) for c in cells]
    boot = np.mean([h[rng.integers(0, len(h), (n_boot, len(h)))].mean(1) for h in hits], 0)
    return [float(np.mean([h.mean() for h in hits])), *np.percentile(boot, [2.5, 97.5])]


def rounded_bar(ax, x0, x1, y0, y1, color, horizontal=False, radius=4):
    """A bar from its baseline (y0, or x0 when horizontal) to its value, square at the baseline and rounded
    at the value's end with a radius of `radius` pixels. Set the axes' limits first."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    (ax0, ay0), (ax1, ay1) = ax.transData.transform([(0, 0), (1, 1)])
    sx, sy = abs(ax1 - ax0), abs(ay1 - ay0)  # pixels per data unit
    if horizontal:
        r = min(radius, (y1 - y0) * sy / 2, (x1 - x0) * sx)
        rx, ry = r / sx, r / sy
        verts = [(x0, y0), (x1 - rx, y0), (x1, y0), (x1, y0 + ry), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0, y1), (x0, y0)]
    else:
        r = min(radius, (x1 - x0) * sx / 2, (y1 - y0) * sy)
        rx, ry = r / sx, r / sy
        verts = [(x1, y0), (x1, y1 - ry), (x1, y1), (x1 - rx, y1), (x0 + rx, y1), (x0, y1), (x0, y1 - ry), (x0, y0), (x1, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", lw=0, zorder=2))


def obstacle_label(spec):
    if not spec:
        return "no obstacle"
    if spec["kind"] == "ring":
        return f"ring around the {spec['around']}"
    return {"wall": "wall with a gap"}.get(spec["kind"], spec["kind"])


def corruption_label(spec):
    if not spec:
        return "no corruption"
    if "scale" in spec:
        return "gain ({:g}, {:g})".format(*spec["scale"])
    if "rotation_deg" in spec:
        return f"rotation {spec['rotation_deg']:g}°"
    if "curl" in spec:
        return f"curl {spec['curl']:g}/s"
    return str(spec)


def fig_summary(sweep, test, out):
    """Every method on the users x tasks grid (the first noise): its success averaged over the cells with an
    obstacle and over those without, then per task and user the best of `against` and every other method,
    with 95% CIs. The baselines (`against`) are gray, the focus method the accent blue. Methods that play the
    same arm on a task (one that lacks a module they differ by) are named below the figure."""
    from matplotlib.patches import Patch
    grid, focus, against, methods = sweep["grid"], sweep["focus"], sweep["against"], list(sweep["methods"])
    noise, users, tasks = grid["noises"][0], grid["users"], grid["tasks"]
    cells = {name: test[cell_key(name, noise)] for name in grid["cells"] if cell_key(name, noise) in test}
    obstacle = {grid["cells"][n]["task"]: grid["cells"][n]["obstacle"] for n in grid["cells"]}
    limit = {t: method_arm(sweep, methods[0], f"{users[0]}x{t}")["time_limit_s"] for t in tasks}
    corruption = {grid["cells"][n]["user"]: grid["cells"][n]["corruption"] for n in grid["cells"]}
    others = iter(["#6da7ec", "#eda100", "#eb6834", *COLORS[2:]])  # gray context, the focus in the accent (dataviz checks)
    color = {m: "#8a8984" if m in against else "#1c5cab" if m == focus else next(others) for m in methods}
    label = lambda m: METHOD_LABEL.get(m, m)
    lower = lambda s: s if s[1:2].isupper() else s[:1].lower() + s[1:]
    played = lambda m, t: json.dumps([{**{k: v for k, v in method_arm(sweep, m, f"{u}x{t}").items() if k != "reversal_adapter"},
                                       "channel": sweep["methods"][m].get("channel")} for u in users],
                                     sort_keys=True, default=lambda o: np.asarray(o).tolist())  # as played: the derived adapter
    same = {}  # methods that play the same arm on a task (one that lacks a module they differ by), so the same trials
    for t in tasks:
        arms = {}
        for m in methods:
            arms.setdefault(played(m, t), []).append(m)
        text = "; ".join(" = ".join(label(m) for m in ms) for ms in arms.values() if len(ms) > 1)
        if text:
            same.setdefault(text, []).append(t)
    surface, grid_ink, base_ink, muted = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781"
    fig = plt.figure(figsize=(13, 8.4), dpi=150, facecolor=surface)
    fig.text(0.04, 0.972, f"Test success on the {sweep['name']}: {len(users)} users × {len(tasks)} tasks",
             fontsize=13, weight="semibold", color=INK, va="top")
    fig.text(0.04, 0.938, f"Simulated {sweep['agent']} user at input noise {noise:g}, {sweep['trials']['test']} test "
             f"trials per cell, every method at its depth tuned on {sweep['trials']['tune']} other trials. "
             "Lines: 95% bootstrap CIs.", fontsize=9, color=INK_2, va="top")

    def frame(ax, value_axis):
        ax.set_facecolor(surface)
        for side, spine in ax.spines.items():
            spine.set_visible(side == ("left" if value_axis == "x" else "bottom"))
            spine.set_color(base_ink)
        ax.grid(False)
        ax.grid(True, axis=value_axis, color=grid_ink, lw=0.8, zorder=0)
        ax.tick_params(colors=muted, length=0, labelsize=8)

    # one bar per method: the mean over the cells with an obstacle, and over those without
    groups = [(f"Obstacle tasks ({', '.join(t for t in tasks if obstacle[t])}): mean of {{n}} cells", True),
              (f"No obstacle ({', '.join(t for t in tasks if not obstacle[t])}): mean of {{n}} cells", False)]
    groups = [(title, [c for c in cells.values() if bool(obstacle[c["task"]]) == has]) for title, has in groups]
    groups = [(title.format(n=len(cs)), cs) for title, cs in groups if cs]
    rng = np.random.default_rng(0)
    for g, (title, cs) in enumerate(groups):
        ax = fig.add_axes([0.22 + g * 0.4, 0.6, 0.34, 0.27])
        frame(ax, "x")
        ax.set(xlim=(0, 112), ylim=(-0.6, len(methods) - 0.4), xticks=[0, 25, 50, 75, 100], yticks=[])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        for i, m in enumerate(methods):
            y, (s, lo, hi) = len(methods) - 1 - i, (100 * v for v in pooled_success(cs, m, rng))
            rounded_bar(ax, 0, s, y - 0.21, y + 0.21, color[m], horizontal=True)
            ax.plot([lo, hi], [y, y], color=INK_2, lw=0.9, zorder=3, solid_capstyle="butt")
            ax.text(hi + 1.5, y, f"{s:.0f}%", va="center", fontsize=8.5, color=INK,
                    weight="semibold" if m == focus else "normal")
            if g == 0:
                ax.text(-2, y, label(m), ha="right", va="center", fontsize=9, color=INK,
                        weight="semibold" if m == focus else "normal")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=8)

    # per task and user: the best baseline and every other method
    shown = ["best"] + [m for m in methods if m not in against]
    colors = [color[against[0]]] + [color[m] for m in shown[1:]]
    width, gap = 0.8 / len(shown), 0.012
    fig.legend([Patch(color=c) for c in colors],
               ["Best of " + ", ".join(lower(label(m)) for m in against[:-1]) + f" and {lower(label(against[-1]))}"]
               + [label(m) for m in shown[1:]], loc="upper left", bbox_to_anchor=(0.04, 0.515), ncol=len(shown),
               fontsize=8.5, handlelength=1.0, handleheight=1.0, columnspacing=1.6, frameon=False)
    left, span, step = 0.06, 0.92, 0.012
    w = (span - step * (len(tasks) - 1)) / len(tasks)
    for j, t in enumerate(tasks):
        ax = fig.add_axes([left + j * (w + step), 0.12, w, 0.32])
        frame(ax, "y")
        ax.set(xlim=(-0.5, len(users) - 0.5), ylim=(0, 100), yticks=[0, 25, 50, 75, 100], xticks=range(len(users)))
        ax.set_yticklabels(["0", "25", "50", "75", "100%"] if j == 0 else [])
        ticks = []
        for i, u in enumerate(users):
            c = cells.get(f"{u}x{t}")
            trained = grid["cells"][f"{u}x{t}"]["train"] and obstacle[t]
            ticks.append(u + ("*" if trained else ""))
            if c is None:
                continue
            for k, m in enumerate(shown):
                m = c["best_other"] if m == "best" else m
                s, lo, hi = (100 * v for v in c["success_ci"][m])
                x0 = i - 0.4 + k * width
                rounded_bar(ax, x0 + gap / 2, x0 + width - gap / 2, 0, s, colors[k])
                ax.plot([x0 + width / 2] * 2, [lo, hi], color=INK_2, lw=0.8, zorder=3, solid_capstyle="butt")
        ax.set_xticklabels(ticks, color=INK, fontsize=8.5)
        ax.set_title(f"{t}  ·  {obstacle_label(obstacle[t])}, {limit[t]:g} s", loc="left", fontsize=9.5, color=INK, pad=6)
    note = "   ".join(f"{u} {corruption_label(corruption[u])}" for u in users)
    starred = any(grid["cells"][f"{u}x{t}"]["train"] and obstacle[t] for u in users for t in tasks)
    lines = [f"Users: {note}." + ("   * the user the task's module was trained with." if starred else "")]
    lines += [f"The same arm, so the same trials, on {', '.join(ts)}: {text}." for text, ts in same.items()]
    for k, line in enumerate(lines):
        fig.text(0.04, 0.045 - 0.022 * k, line, fontsize=8, color=INK_2)
    fig.savefig(out / "summary.png", facecolor=surface, bbox_inches=None)
    plt.close(fig)


def write_grid_summary(sweep, test, path):
    """grid_summary.md: every cell, then the focus method's mean lead by user, by task, and for the
    cells trained on against the held-out ones."""
    focus, methods = sweep["focus"], list(sweep["methods"])
    order = {n: i for i, n in enumerate(sweep["grid"]["tasks"])}
    rows = sorted(test.values(), key=lambda t: (t["user"], order[t["task"]], t["noise"]))
    pct = lambda x: f"{100 * x:.1f}%"
    lines = [f"# {sweep['name']}: {focus} on the users x tasks grid", "",
             "| Cell | Noise | Trained on | " + " | ".join(methods) + f" | Best other | {focus} lead [95% CI] |",
             "|---|---|---|" + "---|" * len(methods) + "---|---|"]
    for t in rows:
        lo = t["lead_over_best"]
        lines.append(f"| {t['level']} | {t['noise']} | {'yes' if t['train_cell'] else 'held-out'} | "
                     + " | ".join(pct(t["metrics"][m]["success"]) + (f" (c {pct(t['metrics'][m]['collision'])})"
                                                                     if t["metrics"][m].get("collision") else "")
                                  for m in methods)
                     + f" | {t['best_other']} | {100 * lo[0]:+.1f} [{100 * lo[1]:+.1f}, {100 * lo[2]:+.1f}]"
                     + ("*" if t["beats_all"] else "") + " |")
    def block(title, groups):
        out = ["", f"## {title}", "", f"| {title} | Cells | Mean {focus} success | Mean lead | Beats all |", "|---|---|---|---|---|"]
        for name, ts in groups:
            if ts:
                out.append(f"| {name} | {len(ts)} | {pct(np.mean([t['metrics'][focus]['success'] for t in ts]))} | "
                           f"{100 * np.mean([t['lead_over_best'][0] for t in ts]):+.1f} | {sum(t['beats_all'] for t in ts)} / {len(ts)} |")
        return out
    users, tasks = sweep["grid"]["users"], sweep["grid"]["tasks"]
    lines += block("User", [(u, [t for t in rows if t["user"] == u]) for u in users])
    lines += block("Task", [(k, [t for t in rows if t["task"] == k]) for k in tasks])
    lines += block("Cells", [("trained on", [t for t in rows if t["train_cell"]]), ("held-out", [t for t in rows if not t["train_cell"]])])
    sided = [t for t in rows if t.get("by_side")]
    if sided:  # success by the side of the reach the route passes the obstacle on
        lines += ["", "## By the side passed", "", "| Cell | Side | Trials | " + " | ".join(methods) + " |",
                  "|---|---|---|" + "---|" * len(methods)]
        for t in sided:
            for side, b in t["by_side"].items():
                lines.append(f"| {t['level']} | {side} | {b['n']} | "
                             + " | ".join(pct(b["metrics"][m]["success"]) for m in methods) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def fig_showcase(sweep, test, key, out):
    cell = test[key]
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(sweep["methods"]), 3.2))
    for i, m in enumerate(sweep["methods"]):
        s, (_, lo, hi) = cell["metrics"][m]["success"], cell["success_ci"][m]
        ax.bar(i, 100 * s, color=COLORS[i % len(COLORS)], width=0.6)
        ax.errorbar(i, 100 * s, yerr=[[100 * (s - lo)], [100 * (hi - s)]], color=INK, capsize=4, lw=1.2)
        d = cell["depths"][m]
        if d is not None:
            ax.text(i, 3, f"depth {d}", ha="center", color="white", fontsize=8)
    ax.set(xticks=range(len(sweep["methods"])), xticklabels=list(sweep["methods"]), ylim=(0, 100), ylabel="Success (%)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(f"Test trials at {cell_label(sweep, cell)} (95% CI)", color=INK, fontsize=8, wrap=True)
    fig.savefig(out / "showcase.png", bbox_inches="tight")
    plt.close(fig)


def fig_traces(sweep, test, agent, policies, adapters, device, out):
    """Center-out reaches for every method at its tuned depth, at the trace levels."""
    tr = sweep["traces"]
    levels, noise = tr.get("levels", sweep["grid"]["levels"]), tr.get("noise", sweep["grid"]["noises"][0])
    methods = list(sweep["methods"])
    fig, axes = plt.subplots(len(methods), len(levels), figsize=(2.5 * len(levels), 2.5 * len(methods)), squeeze=False)
    for c, level in enumerate(levels):
        cell = test[cell_key(level, noise)]
        for r, m in enumerate(methods):
            arm = method_arm(sweep, m, level)
            if cell["depths"][m] is not None:
                arm = with_depth(arm, cell["depths"][m])
            layouts = center_out()
            recs = run_cell_batch(sweep, arm, m, layouts, [noise], agent, policies[arm["model"]], adapters, device,
                                  sweep["seed"] + 99)[str(noise)]
            ax = axes[r, c]
            draw_scene(ax, layouts[:1], TaskConfig(), show_goal=False)
            for rec in recs:
                p = np.array(rec["path"])
                ax.plot(p[:, 0], p[:, 1], color=COLORS[r % len(COLORS)] if rec["outcome"] == "hit" else "#e34948",
                        lw=1.3, zorder=3)
            hits = sum(rec["outcome"] == "hit" for rec in recs)
            d = cell["depths"][m]
            ax.text(0.03, 0.03, (f"depth {d}, " if d is not None else "") + f"{hits}/{len(recs)} hit",
                    transform=ax.transAxes, fontsize=8, color=INK_2)
            if r == 0:
                ax.set_title(f"{FAMILY_LABEL[sweep['grid']['family']].split(' (')[0]} {level}", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(m, color=INK)
    fig.suptitle(f"{sweep['name']}: center-out reaches at input noise {noise}, tuned depths (red: missed)",
                 color=INK, y=1.0, fontsize=10)
    fig.savefig(out / "traces.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- outputs


def write_ranking(sweep, test, path):
    """Every test cell as a row, the best lead first (by the lower end of its 95% CI)."""
    methods, axes = list(sweep["methods"]), list(sweep["grid"]["axes"])
    cell_cols = ["user", "task", "train_cell"] if sweep["grid"]["family"] == "grid" else []
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "level", "noise", *cell_cols, *axes, *(f"{m}_success" for m in methods),
                    *(f"{m}_collision" for m in methods), *(f"{m}_depth" for m in methods), "best_other", "lead", "lead_lo",
                    "lead_hi", "beats_all"])
        for k, t in sorted(test.items(), key=lambda kv: -kv[1]["lead_over_best"][1]):
            w.writerow([k, t["level"], t["noise"], *(t.get(c) for c in cell_cols), *(_fmt(t["axes"][a]) for a in axes),
                        *(round(100 * t["metrics"][m]["success"], 1) for m in methods),
                        *(round(100 * t["metrics"][m].get("collision", 0.0), 1) for m in methods),
                        *(t["depths"][m] for m in methods),
                        t["best_other"], *(round(100 * x, 1) for x in t["lead_over_best"]), t["beats_all"]])


def print_cells(sweep, rows):
    methods = list(sweep["methods"])
    for k, t in rows:
        cols = "  ".join(f"{m} {100 * t['metrics'][m]['success']:5.1f}%"
                         + (f" c{100 * t['metrics'][m]['collision']:.0f}" if t["metrics"][m].get("collision") else "")
                         + (f" ({t['depths'][m]})" if t["depths"][m] else "") for m in methods)
        lo = t["lead_over_best"]
        print(f"{k}: {cols}   lead {100 * lo[0]:+.1f} [{100 * lo[1]:+.1f}, {100 * lo[2]:+.1f}]"
              + ("  *" if t["beats_all"] else ""))


# ---------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="sweep config (bare names are looked up in configs/sweeps/)")
    p.add_argument("--tune-trials", type=int, help="override trials.tune")
    p.add_argument("--test-trials", type=int, help="override trials.test")
    p.add_argument("--depths", type=int, nargs="+", help="override the depths")
    p.add_argument("--resume", action="store_true", help="keep the cells already in the output folder, run the rest")
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--out", help="output folder name under results/sim/sweeps/ (default: the sweep name)")
    p.add_argument("--cells", nargs="+", help="run only these grid levels (e.g. U1xT3; a family's levels as written)")
    p.add_argument("--merge", nargs="+", help="output folders of parts of this sweep: combine them into --out, run nothing")
    p.add_argument("--fixed-depths", help="a depths.yaml: skip tuning, test every method at the depths it gives per cell")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    sweep = load_sweep(args.sweep)
    if args.tune_trials:
        sweep["trials"]["tune"] = args.tune_trials
    if args.test_trials:
        sweep["trials"]["test"] = args.test_trials
    if args.depths:
        sweep["depths"] = args.depths
    out = Path(args.results) / "sim" / "sweeps" / (args.out or sweep["name"])
    (out / "figs").mkdir(parents=True, exist_ok=True)
    with logged(out / "log.txt", append=args.resume) as log:
        for part in args.merge or []:  # a merged sweep's log starts with its parts' logs
            src = Path(args.results) / "sim" / "sweeps" / part / "log.txt"
            if src.exists():
                log.write(f"=== {part}\n{src.read_text()}")
        run(args, sweep, out)


def run(args, sweep, out):
    """The sweep itself, into its folder `out` (main sets up the folder and the log)."""
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    grid, methods, noises = sweep["grid"], list(sweep["methods"]), sweep["grid"]["noises"]
    focus, against = sweep["focus"], sweep["against"]
    groups = cell_groups(sweep)
    run = groups
    if args.cells:
        unknown = set(args.cells) - {str(level) for level in grid["levels"]}
        if unknown:
            raise ValueError(f"--cells: {sorted(unknown)} are not levels of {sweep['name']} ({grid['levels']})")
        run = [(level, axes) for level, axes in groups if str(level) in args.cells]

    # every arm the sweep will play, so the checkpoints load once (the task axes do not change them)
    policies = adapters = None
    if not args.merge or sweep["traces"]:
        arms = {f"{m}@{l}": method_arm(sweep, m, l) for m in methods for l in grid["levels"]}
        policies, adapters = load_checkpoints(arms, device)
    agent = make_agent(sweep["agent"])
    run_args = dict(sweep=sweep, tune_trials=sweep["trials"]["tune"], test_trials=sweep["trials"]["test"],
                    seed=sweep["seed"], agent=sweep["agent"], device=device)

    fixed = None
    if args.fixed_depths and not args.merge:  # every cell to run, with a depth for every method that has one
        fixed = yaml.safe_load(Path(args.fixed_depths).read_text()) or {}
        for level, axes in run:
            for s in noises:
                k = cell_key(level, s, axes)
                if k not in fixed or set(fixed[k]) != set(methods):
                    raise ValueError(f"--fixed-depths {args.fixed_depths}: cell {k} must give a depth (or null) for "
                                     f"every method {methods}, got {fixed.get(k)}")
                for m in methods:
                    if (fixed[k][m] is None) == has_depth(method_arm(sweep, m, level, axes)):
                        raise ValueError(f"--fixed-depths {args.fixed_depths}: cell {k} gives {m} the depth "
                                         f"{fixed[k][m]!r}, but {m} {'takes' if fixed[k][m] is None else 'has'} no depth")

    tuning, test, skipped = {}, {}, {}
    if args.merge:
        for part in args.merge:
            src = Path(args.results) / "sim" / "sweeps" / part
            saved = json.loads((src / "test.json").read_text())
            test.update(saved["test"])
            skipped.update(saved.get("skipped", {}))
            tuning.update(json.loads((src / "tuning.json").read_text())["tuning"])
        order = [cell_key(level, s, axes) for level, axes in groups for s in noises]  # the grid's order, as one run writes it
        test, tuning, skipped = ({k: d[k] for k in order if k in d} for d in (test, tuning, skipped))
        run = []
        print(f"merging {len(args.merge)} parts: {len(test)} cells, {len(skipped)} skipped", flush=True)
    elif args.resume and (out / "test.json").exists():
        saved = json.loads((out / "test.json").read_text())
        test, skipped = saved["test"], saved.get("skipped", {})
        tuning = {k: v for k, v in json.loads((out / "tuning.json").read_text())["tuning"].items() if k in test}
        print(f"resuming: {len(test)} cells done, {len(skipped)} skipped", flush=True)

    def tuned_depths(cell):
        return {m: (None if "None" in cell[m] else int(best_depth(cell[m]))) for m in methods}

    def showcase():
        """The cell where the focus method's tuned lead is largest, on the tuning trials."""
        if not tuning:
            return None
        tuned = lambda k, m: tuning[k][m][str(tuned_depths(tuning[k])[m])]["success"]
        return max(tuning, key=lambda k: (tuned(k, focus) - max(tuned(k, m) for m in against), -tuned(k, against[0])))

    def save():
        depths = {k: tuned_depths(v) for k, v in tuning.items()}
        with open(out / "tuning.json", "w") as f:
            json.dump(dict(args=run_args, tuning=tuning, depths=depths), f, indent=1)
        if fixed is not None:  # the depths tested are the input's
            if Path(args.fixed_depths).resolve() != (out / "depths.yaml").resolve():
                shutil.copyfile(args.fixed_depths, out / "depths.yaml")
        else:  # merged fixed-depth parts have no tuning: their cells keep the depths they were tested at
            depths.update({k: v["depths"] for k, v in test.items() if k not in depths})
            with open(out / "depths.yaml", "w") as f:
                yaml.safe_dump(depths, f, sort_keys=False)
        with open(out / "test.json", "w") as f:
            json.dump(dict(args=run_args, showcase=showcase(), skipped=skipped, test=test), f, indent=1)

    tic = time.time()
    if args.merge:
        save()
    for g, (level, axes) in enumerate(run, 1):
        keys = {s: cell_key(level, s, axes) for s in noises}
        if all(k in test or k in skipped for k in keys.values()):
            continue
        label = group_label(sweep, level, axes) or "the grid"
        arm0 = method_arm(sweep, methods[0], level, axes)
        task, targets = arm_task(arm0), tuple(arm0["targets"])
        try:
            tune_layouts = sample_layouts(np.random.default_rng(sweep["seed"]), sweep["trials"]["tune"], targets, task)
            test_layouts = sample_layouts(np.random.default_rng(sweep["seed"] + 10), sweep["trials"]["test"], targets, task)
        except ValueError as e:
            skipped.update({k: str(e) for k in keys.values()})
            save()
            print(f"{g}/{len(run)} skipping {label}: {e}", flush=True)
            continue

        # tuning: every method at every depth, all noises at once (none with --fixed-depths)
        if fixed is not None:
            depths = {s: dict(fixed[keys[s]]) for s in noises}
        else:
            for k in keys.values():
                tuning[k] = {m: {} for m in methods}
            for m in methods:
                arm = method_arm(sweep, m, level, axes)
                for d in (depths_for(sweep, arm["mode"]) if has_depth(arm) else [None]):
                    per_noise = run_cell_batch(sweep, with_depth(arm, d) if d else arm, m, tune_layouts, noises, agent,
                                               policies[arm["model"]], adapters, device, sweep["seed"] + 1)
                    for s, recs in per_noise.items():
                        tuning[keys[float(s)]][m][str(d)] = summarize(recs)
            depths = {s: tuned_depths(tuning[keys[s]]) for s in noises}

        # test: every method at its tuned depth, on new layouts
        hit, records = {s: {} for s in noises}, {s: {} for s in noises}
        for m in methods:
            arm = method_arm(sweep, m, level, axes)
            by_depth = {}
            for s in noises:
                by_depth.setdefault(depths[s][m], []).append(s)
            for d, ss in by_depth.items():
                per_noise = run_cell_batch(sweep, with_depth(arm, d) if d else arm, m, test_layouts, ss, agent,
                                           policies[arm["model"]], adapters, device, sweep["seed"] + 11)
                for s, recs in per_noise.items():
                    hit[float(s)][m] = np.array([r["outcome"] == "hit" for r in recs])
                    records[float(s)][m] = recs
        cell = grid["cells"][level] if grid["family"] == "grid" else {}
        sides = pass_sides(test_layouts, task.target_radius)
        for s in noises:  # the focus method's paired lead over every other method; best_other and beats_all over `against`
            boot = np.random.default_rng([0, zlib.crc32(keys[s].encode())])  # a cell's own bootstrap
            lead = {m: bootstrap_mean(hit[s][focus].astype(float) - hit[s][m], boot) for m in methods if m != focus}
            best_other = max(against, key=lambda m: hit[s][m].mean())
            test[keys[s]] = dict(level=level, noise=s, axes=axes, user=cell.get("user"), task=cell.get("task"),
                                 train_cell=cell.get("train"), depths=depths[s],
                                 metrics={m: summarize(records[s][m]) for m in methods},
                                 success_ci={m: bootstrap_mean(hit[s][m], boot) for m in methods},
                                 lead=lead, best_other=best_other, lead_over_best=lead[best_other],
                                 beats_all=bool(all(lead[m][1] > 0 for m in against)))
            if set(sides) - {"straight"}:  # the outcomes by the side the route passes the obstacle on
                test[keys[s]]["by_side"] = {
                    side: dict(n=sides.count(side), metrics={m: summarize([r for r, d in zip(records[s][m], sides) if d == side])
                                                             for m in methods})
                    for side in ("left", "right", "straight") if side in sides}
            test[keys[s]]["trials"] = dict(side=sides, outcome={m: [r["outcome"] for r in records[s][m]] for m in methods},
                                           duration_s={m: [r["duration_s"] for r in records[s][m]] for m in methods})
        save()
        elapsed = time.time() - tic
        print(f"{g}/{len(run)} {label} ({elapsed:.0f}s, {elapsed / 60 * (len(run) - g) / g:.0f} min left)", flush=True)

    if not test:
        print("no cell could be run" + (f"; skipped: {skipped}" if skipped else ""))
        return
    write_ranking(sweep, test, out / "ranking.csv")
    if grid["family"] == "grid":
        write_grid_summary(sweep, test, out / "grid_summary.md")
    show = showcase()  # None without tuning (--fixed-depths, or merged fixed-depth parts): no depth curves, no showcase
    style()
    done = [(l, a) for l, a in groups if cell_key(l, noises[0], a) in tuning]
    if show is not None:
        few = done if len(done) <= 6 else [(test[show]["level"], test[show]["axes"])]
        fig_depth_curves(sweep, tuning, out / "figs", [(test[show]["level"], test[show]["axes"])] if grid["axes"] else few)
    if grid["axes"]:
        fig_axes(sweep, test, out / "figs")
    elif grid["family"] == "grid":
        fig_grid(sweep, test, out / "figs")
        fig_summary(sweep, test, out / "figs")  # every method on every cell, in place of one showcase cell
    elif all(cell_key(level, noises[0], axes) in test for level, axes in groups):
        fig_test(sweep, test, out / "figs")
    if grid["family"] != "grid" and show is not None:
        fig_showcase(sweep, test, show, out / "figs")
    if sweep["traces"]:
        fig_traces(sweep, test, agent, policies, adapters, device, out / "figs")

    print(f"\n{sweep['name']}: test trials, tuned depths ({time.time() - tic:.0f}s). {focus} lead over the "
          f"best of {against}: points [95% CI]" + (f"; {len(skipped)} cells skipped" if skipped else ""))
    if grid["axes"] or grid["family"] == "grid":
        ranked = sorted(test.items(), key=lambda kv: -kv[1]["lead_over_best"][1])
        print(f"top {min(15, len(ranked))} of {len(ranked)} cells by the lower end of the lead's CI (all in ranking.csv):")
        print_cells(sweep, ranked[:15])
    else:
        print_cells(sweep, list(test.items()))
    print(f"showcase (chosen on tuning trials): {show}; depths in {out / 'depths.yaml'}")


if __name__ == "__main__":
    main()
