#!/usr/bin/env python
"""Report for the grid rederived with the vision task module, and its decoy control.

Reads ``<results>/sim/sweeps/<run>_s<k>/test.json`` for the seeds that exist of ``grid_vision`` (teleop, ft_flow_control,
ft_frs, mars_task, mars), ``grid`` (mars_user) and ``grid_ft`` (ft_mars_user), and of ``grid_vision_decoy`` (the four
trained-on obstacle cells with the obstacle placed about a non-goal target, every arm at the grid's tuned settings), and
writes ``report.md`` and ``report.json`` to ``--out``. Metrics are the sweeps' fractions (``success``, ``collision``,
``wrong``), shown in %; a cell's value is the mean over the seeds; a task's the mean over its users.

    python reports/report_grid_vision.py --results ../results --out ../results/sim/reports/grid_vision
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

GRID_METHODS = {  # Figure 3d's arms, in the figure's order -> the run they come from
    "teleop": "grid_vision", "ft_flow_control": "grid_vision", "ft_frs": "grid_vision", "mars_task": "grid_vision",
    "mars_user": "grid", "ft_mars_user": "grid_ft", "mars": "grid_vision",
}
DECOY_METHODS = {  # the decoy sweep's arms -> (the grid run, the method there) the same cell is compared with
    "teleop": ("grid_vision", "teleop"), "ft_flow_control": ("grid_vision", "ft_flow_control"),
    "ft_frs": ("grid_vision", "ft_frs"), "ft_mars_user": ("grid_ft", "ft_mars_user"), "mars_user": ("grid", "mars_user"),
    "mars_task_param": ("grid", "mars_task"), "mars_param": ("grid", "mars"), "mars_task": ("grid_vision", "mars_task"),
    "mars": ("grid_vision", "mars"),
}
LABEL = {
    "teleop": "Teleop", "ft_flow_control": "Fine-tuned + FC", "ft_frs": "Fine-tuned + FRS", "mars_task": "MARS-T (vision)",
    "mars_user": "MARS-U", "ft_mars_user": "Fine-tuned + MARS-U", "mars": "MARS-UT (vision)",
    "mars_task_param": "MARS-T (parametric)", "mars_param": "MARS-UT (parametric)",
}
TASKS = ("T1", "T2", "T3", "T4", "T5")
OBSTACLE_TASKS = TASKS[1:]
METRICS = ("success", "collision", "wrong")


def _task(key: str) -> str:
    return re.match(r"U\d+x(T\d+)\|", key).group(1)


def load_runs(sweeps: Path, name: str) -> dict[int, dict]:
    """The seeds of ``name`` that exist: seed -> test.json."""
    out = {}
    for k in range(10):
        path = sweeps / f"{name}_s{k}" / "test.json"
        if path.is_file():
            out[k] = json.loads(path.read_text())
    return out


def cell_metric(runs: dict[int, dict], key: str, method: str, metric: str) -> list[float]:
    """One value per seed (%), for the seeds whose run has the cell and the method."""
    return [100 * r["test"][key]["metrics"][method][metric] for r in runs.values()
            if key in r["test"] and method in r["test"][key]["metrics"]]


def grid_tables(runs: dict[str, dict[int, dict]]) -> dict:
    """Per cell and per task, every Figure 3d arm's success and collision (mean over seeds; a task's over its users)."""
    keys = sorted({k for r in runs.get("grid_vision", {}).values() for k in r["test"]} or
                  {k for r in runs.get("grid", {}).values() for k in r["test"]})
    cells, seeds_used = {}, {}
    for key in keys:
        cells[key] = {}
        for m, run in GRID_METHODS.items():
            vals = {met: cell_metric(runs.get(run, {}), key, m, met) for met in METRICS}
            if vals["success"]:
                cells[key][m] = {met: round(float(np.mean(v)), 1) for met, v in vals.items() if v}
                cells[key][m]["sd_seeds"] = round(float(np.std(vals["success"])), 1)
                seeds_used[m] = len(vals["success"])
    tasks = {}
    for t in TASKS:
        tasks[t] = {}
        for m in GRID_METHODS:
            vals = [cells[k][m] for k in cells if _task(k) == t and m in cells[k]]
            if vals:
                tasks[t][m] = {met: round(float(np.mean([v[met] for v in vals])), 1) for met in ("success", "collision") if all(met in v for v in vals)}
    obstacle = {}
    for m in GRID_METHODS:
        vals = [cells[k][m]["success"] for k in cells if _task(k) in OBSTACLE_TASKS and m in cells[k]]
        if vals:
            obstacle[m] = dict(success=round(float(np.mean(vals)), 1), n_cells=len(vals))
    return dict(cells=cells, tasks=tasks, obstacle_cells=obstacle, seeds=seeds_used)


def decoy_table(runs: dict[str, dict[int, dict]]) -> list[dict]:
    """Per decoy cell and arm: success, collision and wrong target on the decoy layouts (mean over seeds), beside the
    same arm in the same cell of the grid, and the difference in success (decoy minus grid)."""
    decoy = runs.get("grid_vision_decoy", {})
    rows = []
    for key in sorted({k for r in decoy.values() for k in r["test"]}):
        for m, (grid_run, grid_m) in DECOY_METHODS.items():
            d = {met: cell_metric(decoy, key, m, met) for met in METRICS}
            if not d["success"]:
                continue
            g = {met: cell_metric(runs.get(grid_run, {}), key, grid_m, met) for met in METRICS}
            row = dict(cell=key.split("|")[0], arm=m, seeds=len(d["success"]),
                       **{f"decoy_{met}": round(float(np.mean(v)), 1) for met, v in d.items() if v},
                       **{f"grid_{met}": round(float(np.mean(v)), 1) for met, v in g.items() if v})
            if "grid_success" in row:
                row["delta_success"] = round(row["decoy_success"] - row["grid_success"], 1)
            rows.append(row)
    return rows


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.1f}"


def write_markdown(report: dict, path: Path) -> None:
    g, lines = report["grid"], ["# The grid with the vision task module, and its decoy control", ""]
    lines += [f"Seeds per arm: {g['seeds']}", "", "## Success per task (%; mean over users and seeds)", ""]
    methods = [m for m in GRID_METHODS if any(m in g["tasks"][t] for t in TASKS)]
    lines += ["| task | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for t in TASKS:
        lines.append(f"| {t} | " + " | ".join(_fmt(g["tasks"][t].get(m, {}).get("success")) for m in methods) + " |")
    lines.append("| obstacle cells | " + " | ".join(_fmt(g["obstacle_cells"].get(m, {}).get("success")) for m in methods) + " |")
    lines += ["", "## Collisions per task (%)", "", "| task | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for t in TASKS:
        lines.append(f"| {t} | " + " | ".join(_fmt(g["tasks"][t].get(m, {}).get("collision")) for m in methods) + " |")
    lines += ["", "## Success per cell (%; mean over seeds, sd over seeds in brackets)", "",
              "| cell | " + " | ".join(LABEL[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]
    for key, cell in g["cells"].items():
        lines.append(f"| {key.split('|')[0]} | " + " | ".join(
            f"{cell[m]['success']:.1f} ({cell[m]['sd_seeds']:.1f})" if m in cell else "-" for m in methods) + " |")
    rows = report["decoy"]
    lines += ["", "## Decoy control (the obstacle placed about a non-goal target; the grid's four trained-on cells)", ""]
    if rows:
        lines += ["Each arm at the grid's tuned settings. `decoy` columns: the decoy layouts; `grid` columns: the same arm in the same",
                  "cell of the grid (obstacle about the goal); `delta`: decoy success minus grid success (points). Mean over seeds.", "",
                  "| cell | arm | seeds | decoy success | decoy collision | decoy wrong target | grid success | grid collision | grid wrong target | delta |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['cell']} | {LABEL[r['arm']]} | {r['seeds']} | {_fmt(r.get('decoy_success'))} | {_fmt(r.get('decoy_collision'))} | "
                         f"{_fmt(r.get('decoy_wrong'))} | {_fmt(r.get('grid_success'))} | {_fmt(r.get('grid_collision'))} | "
                         f"{_fmt(r.get('grid_wrong'))} | {_fmt(r.get('delta_success'))} |")
    else:
        lines.append("No decoy run found.")
    if report["checks"]:
        lines += ["", "## Checks", ""] + [f"- {c}" for c in report["checks"]]
    path.write_text("\n".join(lines) + "\n")


def checks(runs: dict[str, dict[int, dict]]) -> list[str]:
    """teleop's outcomes in grid_vision_s<k> equal grid_s<k>'s (the same trials), per seed present in both."""
    out = []
    for k, r in runs.get("grid_vision", {}).items():
        g = runs.get("grid", {}).get(k)
        if g is None:
            continue
        same = all(r["test"][key]["trials"]["outcome"]["teleop"] == g["test"][key]["trials"]["outcome"]["teleop"]
                   for key in r["test"] if key in g["test"])
        out.append(f"seed {k}: teleop outcomes {'equal' if same else 'DIFFER from'} grid_s{k}'s (the trials are the grid's)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results"))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    sweeps = Path(a.results) / "sim" / "sweeps"
    runs = {name: load_runs(sweeps, name) for name in ("grid_vision", "grid", "grid_ft", "grid_vision_decoy")}
    report = dict(runs={n: sorted(r) for n, r in runs.items()}, grid=grid_tables(runs), decoy=decoy_table(runs), checks=checks(runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    write_markdown(report, out / "report.md")
    print(out / "report.md")


if __name__ == "__main__":
    main()
