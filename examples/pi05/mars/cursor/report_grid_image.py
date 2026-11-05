#!/usr/bin/env python
"""Report for the image task modules: gate verdicts, grid tables, pooled statistic.

Reads ``<results>/sim/sweeps/<run>/test.json`` of the gate run (``grid_task_check_image``), the grid
runs (``grid_image_s<k>``) and optionally the fine-tuned runs (``grid_ft_s<k>``), and writes
``report.md`` and ``report.json`` to ``--out``. The gate gives each image arm one of three verdicts against two
reference rows of the check run, ``oracle_task`` (the parametric module) and ``oracle_ft`` (the fine-tuned policy,
the floor): pass (within 10 points of parametric hits, collisions at most parametric's + 10 points), marginal
(more hits than the fine-tuned policy but not a pass, collisions within the same bar) or FAIL. Without ``--grid`` only the gate is reported (the arms are the
check run's ``oracle_task_<x>`` methods), so the gate can be read before the grid exists.
"""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np

MARGIN = 0.10  # gate: hits may trail the parametric module, collisions and hidden hits may exceed theirs, by this much


def _context(key: str) -> str | None:
    m = re.match(r"U\d+x(T\d+)\|", key)
    return m.group(1) if m else None


def _user(key: str) -> str | None:
    m = re.match(r"(U\d+)x", key)
    return m.group(1) if m else None


def _within(a: float, b: float) -> bool:
    """``a >= b`` without float noise: fractions on 500 trials differ by multiples of 0.002, so nine places suffice."""
    return round(a - b, 9) >= 0


def gate_table(check: dict, image_arms: list[str]) -> list[dict]:
    """Per context (from the cell key ``U0xT<k>|...``) and image arm, the gate verdict.

    Reads ``metrics`` only. Metrics are fractions in [0, 1], so the rule is in fractions (10 points = 0.10).
    hits and collision are from ``oracle_task_<x>``; ref_hits and ref_collision from ``oracle_task`` (the
    parametric module); ft_hits from ``oracle_ft`` (the fine-tuned policy, None when the run has no such row).
    ``collisions_ok`` = collision <= ref_collision + 0.10. ``pass`` = hits >= ref_hits - 0.10 and collisions_ok.
    Else ``marginal`` = hits > ft_hits and collisions_ok, else ``FAIL``; pass wins where it overlaps the floor.
    Without an ``oracle_ft`` row no arm is marginal. ``flag`` = hidden_hits > ref_hidden_hits + 0.10, with
    hidden_hits from ``policy_task_<x>`` and ref_hidden_hits from ``policy_task``.
    """
    rows = []
    for key, cell in sorted(check.get("test", {}).items()):
        ctx = _context(key)
        met = cell.get("metrics", {})
        if ctx is None:
            continue
        ref_c = met.get("oracle_task", {}).get("collision")
        ref_hits = met.get("oracle_task", {}).get("success")
        ft_hits = met.get("oracle_ft", {}).get("success")
        ref_h = met.get("policy_task", {}).get("success")
        for arm in image_arms:
            o, p = met.get(f"oracle_task_{arm}"), met.get(f"policy_task_{arm}")
            if o is None:
                continue
            hits, col = o["success"], o["collision"]
            hidden = p["success"] if p else None
            collisions_ok = ref_c is not None and _within(ref_c + MARGIN, col)
            passes = ref_hits is not None and _within(hits, ref_hits - MARGIN) and collisions_ok
            above_floor = ft_hits is not None and hits > ft_hits
            verdict = "pass" if passes else ("marginal" if above_floor and collisions_ok else "FAIL")
            rows.append(
                dict(
                    context=ctx,
                    arm=arm,
                    hits=hits,
                    ref_hits=ref_hits,
                    ft_hits=ft_hits,
                    collision=col,
                    ref_collision=ref_c,
                    verdict=verdict,
                    **{"pass": verdict == "pass"},
                    hidden_hits=hidden,
                    ref_hidden_hits=ref_h,
                    flag=bool(hidden is not None and ref_h is not None and hidden > ref_h + MARGIN),
                )
            )
    return sorted(rows, key=lambda r: (r["context"], r["arm"]))


def join_ft(grid_run: dict, ft_run: dict, arm: str = "ft_mars_user") -> dict:
    """Copy the fine-tuned arm into a copy of the grid run, after checking the trials are the same.

    The fine-tuned run may have more cells than the grid (it also runs T1); the result has the grid's cells.
    Raises ValueError unless every grid cell is in the fine-tuned run and, cell for cell, ``trials.side`` and
    the ``teleop`` outcomes agree.
    """
    g, f = grid_run["test"], ft_run["test"]
    if not set(g) <= set(f):
        raise ValueError(f"trials differ: grid cells missing from the fine-tuned run {sorted(set(g) - set(f))}")
    out = copy.deepcopy(grid_run)
    for key, cell in out["test"].items():
        fc = f[key]
        gt, ftr = cell.get("trials", {}), fc.get("trials", {})
        if gt.get("side") != ftr.get("side"):
            raise ValueError(f"trials.side differs in cell {key}")
        if gt.get("outcome", {}).get("teleop") != ftr.get("outcome", {}).get("teleop"):
            raise ValueError(f"trials: teleop outcomes differ in cell {key}")
        cell["metrics"][arm] = copy.deepcopy(fc["metrics"][arm])
        if arm in fc.get("settings", {}):
            cell.setdefault("settings", {})[arm] = copy.deepcopy(fc["settings"][arm])
        for field in ("outcome", "duration_s"):
            if arm in ftr.get(field, {}):
                cell.setdefault("trials", {}).setdefault(field, {})[arm] = copy.deepcopy(ftr[field][arm])
    return out


def pooled_over_seeds(runs: list[dict], method: str) -> dict:
    """Success in percent: per seed the mean over its cells, then mean and 95% t-CI over seeds.

    With one seed, or identical per-seed values, lo = hi = mean.
    """
    per_seed = [
        100.0 * float(np.mean([c["metrics"][method]["success"] for c in r["test"].values()])) for r in runs
    ]
    mean = float(np.mean(per_seed))
    if len(per_seed) < 2:
        return dict(mean=mean, lo=mean, hi=mean, per_seed=per_seed)
    from scipy import stats

    sem = float(np.std(per_seed, ddof=1)) / np.sqrt(len(per_seed))
    half = float(stats.t.ppf(0.975, len(per_seed) - 1)) * sem
    if not np.isfinite(half):
        half = 0.0
    return dict(mean=mean, lo=mean - half, hi=mean + half, per_seed=per_seed)


def _load(results: Path, name: str) -> dict:
    return json.loads((results / "sim" / "sweeps" / name / "test.json").read_text())


def _fmt(v, pct=True, none="-"):
    if v is None:
        return none
    return f"{100 * v:.1f}" if pct else f"{v:.1f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    esc = lambda cell: str(cell).replace("|", "\\|")  # noqa: E731  a cell key like U1xT2|1.0 holds a pipe
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return out + [""]


def _methods(run: dict) -> list[str]:
    seen: list[str] = []
    for c in run["test"].values():
        for m in c.get("metrics", {}):
            if m not in seen:
                seen.append(m)
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", required=True, help="gate run name")
    ap.add_argument("--grid", nargs="*", default=[], help="grid run names, one per seed (default: the gate alone)")
    ap.add_argument("--ft", nargs="*", default=[], help="fine-tuned run names, paired with --grid by position")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.ft and not a.grid:
        ap.error("--ft needs --grid: it pairs with the grid runs by position")
    if a.ft and len(a.ft) != len(a.grid):
        ap.error("--ft must pair with --grid by position")

    check = _load(a.results, a.check)
    grids = [_load(a.results, n) for n in a.grid]
    if a.ft:
        grids = [join_ft(g, _load(a.results, n)) for g, n in zip(grids, a.ft, strict=True)]
    if grids:
        gmethods = _methods(grids[0])
        arms = [m[len("mars_"):] for m in gmethods if m.startswith("mars_")]
    else:
        gmethods = []
        arms = [m[len("oracle_task_"):] for m in _methods(check) if m.startswith("oracle_task_")]

    gate = gate_table(check, arms)
    md = ["# Image task modules: gate and grid", "", "## Gate (success and collision in %)", ""]
    if gate:
        md += _table(
            ["context", "arm", "hits", "ref hits", "ft hits", "collision", "ref collision", "verdict",
             "hidden hits", "ref hidden", "flag"],
            [
                [r["context"], r["arm"], _fmt(r["hits"]), _fmt(r["ref_hits"], none="n/a"),
                 _fmt(r["ft_hits"], none="n/a"), _fmt(r["collision"]), _fmt(r["ref_collision"]),
                 r["verdict"], _fmt(r["hidden_hits"]), _fmt(r["ref_hidden_hits"]),
                 "FLAG" if r["flag"] else "ok"]
                for r in gate
            ],
        )
    else:
        md += ["The gate run has no cells; no gate verdicts.", ""]

    pooled_methods = [m for m in ["mars", *[f"mars_{x}" for x in arms], "ft_mars_user"] if m in gmethods]
    pooled = {m: pooled_over_seeds(grids, m) for m in pooled_methods}
    if grids:
        md += [f"## Pooled over cells, {len(grids)} seed(s), success % (mean, 95% t-CI over seeds)", ""]
        ci = (lambda p, k: f"{p[k]:.1f}") if len(grids) > 1 else (lambda p, k: "n/a")  # one seed has no interval
        md += _table(
            ["method", "mean", "lo", "hi", "per seed"],
            [[m, f"{p['mean']:.1f}", ci(p, "lo"), ci(p, "hi"), ", ".join(f"{v:.1f}" for v in p["per_seed"])]
             for m, p in pooled.items()],
        )

    per_cell, per_ctx = {}, {}
    for name, run in zip(a.grid, grids, strict=True):
        ms = _methods(run)
        keys = sorted(run["test"])
        md += [f"## Per-cell success %, {name}", ""]
        md += _table(  # train: a cell the modules were trained on (`train_cell` in test.json)
            ["cell", "train", *ms],
            [
                [k, "train" if run["test"][k].get("train_cell") else "", *[
                    _fmt(run["test"][k]["metrics"].get(m, {}).get("success")) for m in ms]]
                for k in keys
            ],
        )
        per_cell[name] = {k: {m: run["test"][k]["metrics"].get(m, {}).get("success") for m in ms} for k in keys}
        ctxs = sorted({c for c in map(_context, keys) if c})
        ctx_rows, per_ctx[name] = [], {}
        for c in ctxs:
            ck = [k for k in keys if _context(k) == c]
            users = {_user(k) for k in ck}
            means = {
                m: float(np.mean([run["test"][k]["metrics"][m]["success"] for k in ck if m in run["test"][k]["metrics"]]))
                for m in ms
            }
            per_ctx[name][c] = means
            ctx_rows.append([f"{c} ({len(users)} users)", *[_fmt(means[m]) for m in ms]])
        md += [f"## Per-context success % (mean over users), {name}", ""]
        md += _table(["context", *ms], ctx_rows)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    (a.out / "report.json").write_text(
        json.dumps(dict(gate=gate, pooled=pooled, per_cell=per_cell, per_context=per_ctx), indent=2)
    )


if __name__ == "__main__":
    main()
