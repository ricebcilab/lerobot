#!/usr/bin/env python
"""Report for the image task modules: gate verdicts, grid tables, pooled statistic.

Reads ``<results>/sim/sweeps/<run>/test.json`` of the gate run (``grid_task_check_image``), the grid
runs (``grid_image_s<k>``) and optionally the fine-tuned runs (``grid_ft_s<k>``), and writes
``report.md`` and ``report.json`` to ``--out``.
"""

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np

HIT_MIN = 0.80  # gate: oracle_task_<x> must hit at least this often (metrics are fractions in [0, 1])
MARGIN = 0.10  # gate: collision may exceed the reference by this much; hidden hits likewise


def _context(key: str) -> str | None:
    m = re.match(r"U\d+x(T\d+)\|", key)
    return m.group(1) if m else None


def _user(key: str) -> str | None:
    m = re.match(r"(U\d+)x", key)
    return m.group(1) if m else None


def gate_table(check: dict, image_arms: list[str]) -> list[dict]:
    """Per context (from the cell key ``U0xT<k>|...``) and image arm, the gate verdict.

    Reads ``metrics`` only. Metrics are fractions in [0, 1], so the rule is in fractions:
    ``pass`` = hits >= 0.80 and collision <= ref_collision + 0.10, with hits and collision from
    ``oracle_task_<x>`` and ref_collision from ``oracle_task`` (the same rule as 80 / +10 percent).
    ``flag`` = hidden_hits > ref_hidden_hits + 0.10, with hidden_hits from ``policy_task_<x>`` and
    ref_hidden_hits from ``policy_task``. Values are reported as the metrics give them.
    """
    rows = []
    for key, cell in sorted(check.get("test", {}).items()):
        ctx = _context(key)
        met = cell.get("metrics", {})
        if ctx is None:
            continue
        ref_c = met.get("oracle_task", {}).get("collision")
        ref_h = met.get("policy_task", {}).get("success")
        for arm in image_arms:
            o, p = met.get(f"oracle_task_{arm}"), met.get(f"policy_task_{arm}")
            if o is None:
                continue
            hits, col = o["success"], o["collision"]
            hidden = p["success"] if p else None
            rows.append(
                dict(
                    context=ctx,
                    arm=arm,
                    hits=hits,
                    collision=col,
                    ref_collision=ref_c,
                    **{"pass": bool(hits >= HIT_MIN and ref_c is not None and col <= ref_c + MARGIN)},
                    hidden_hits=hidden,
                    ref_hidden_hits=ref_h,
                    flag=bool(hidden is not None and ref_h is not None and hidden > ref_h + MARGIN),
                )
            )
    return sorted(rows, key=lambda r: (r["context"], r["arm"]))


def join_ft(grid_run: dict, ft_run: dict, arm: str = "ft_mars_user") -> dict:
    """Copy the fine-tuned arm into a copy of the grid run, after checking the trials are the same.

    Raises ValueError unless the cell keys, ``trials.side`` and the ``teleop`` outcomes agree.
    """
    g, f = grid_run["test"], ft_run["test"]
    if set(g) != set(f):
        raise ValueError(f"trials differ: cell keys {sorted(set(g) ^ set(f))}")
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


def _fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{100 * v:.1f}" if pct else f"{v:.1f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
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
    ap.add_argument("--grid", nargs="+", required=True, help="grid run names, one per seed")
    ap.add_argument("--ft", nargs="*", default=[], help="fine-tuned run names, paired with --grid by position")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.ft and len(a.ft) != len(a.grid):
        ap.error("--ft must pair with --grid by position")

    check = _load(a.results, a.check)
    grids = [_load(a.results, n) for n in a.grid]
    if a.ft:
        grids = [join_ft(g, _load(a.results, n)) for g, n in zip(grids, a.ft, strict=True)]
    gmethods = _methods(grids[0])
    arms = [m[len("mars_"):] for m in gmethods if m.startswith("mars_")]

    gate = gate_table(check, arms)
    md = ["# Image task modules: gate and grid", "", "## Gate (success and collision in %)", ""]
    if gate:
        md += _table(
            ["context", "arm", "hits", "collision", "ref collision", "pass", "hidden hits", "ref hidden", "flag"],
            [
                [r["context"], r["arm"], _fmt(r["hits"]), _fmt(r["collision"]), _fmt(r["ref_collision"]),
                 "pass" if r["pass"] else "FAIL", _fmt(r["hidden_hits"]), _fmt(r["ref_hidden_hits"]),
                 "FLAG" if r["flag"] else "ok"]
                for r in gate
            ],
        )
    else:
        md += ["The gate run has no cells; no gate verdicts.", ""]

    pooled_methods = [m for m in ["mars", *[f"mars_{x}" for x in arms], "ft_mars_user"] if m in gmethods]
    pooled = {m: pooled_over_seeds(grids, m) for m in pooled_methods}
    md += [f"## Pooled over cells, {len(grids)} seed(s), success % (mean, 95% t-CI over seeds)", ""]
    md += _table(
        ["method", "mean", "lo", "hi", "per seed"],
        [[m, f"{p['mean']:.1f}", f"{p['lo']:.1f}", f"{p['hi']:.1f}", ", ".join(f"{v:.1f}" for v in p["per_seed"])]
         for m, p in pooled.items()],
    )

    per_cell, per_ctx = {}, {}
    for name, run in zip(a.grid, grids, strict=True):
        ms = _methods(run)
        keys = sorted(run["test"])
        md += [f"## Per-cell success %, {name}", ""]
        md += _table(
            ["cell", *ms], [[k, *[_fmt(run["test"][k]["metrics"].get(m, {}).get("success")) for m in ms]] for k in keys]
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
