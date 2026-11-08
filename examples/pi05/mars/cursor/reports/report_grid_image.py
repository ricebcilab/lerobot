#!/usr/bin/env python
"""Report for the image task modules' gates: verdicts, scan crossings, recipe choice and width check.

Reads ``<results>/sim/sweeps/<run>/test.json`` of a gate run (``grid_task_check_*``) and writes ``report.md`` and
``report.json`` to ``--out``. The gate gives each image arm one of three verdicts against two reference rows of the
check run, ``oracle_task`` (the parametric module) and ``oracle_ft`` (the fine-tuned policy, the floor): pass (within 10
points of parametric hits, collisions at most parametric's + 10 points), marginal (more hits than the fine-tuned policy
but not a pass, collisions within the same bar) or FAIL. The grid's tables are ``report_grid_vision.py``'s.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

MARGIN = 0.10  # gate: hits may trail the parametric module, collisions and hidden hits may exceed theirs, by this much
LEVEL = 0.03  # scan: "level" with the parametric module is hits and collisions within this much of its
ADOPT = 0.03  # recipe choice: a change is adopted when its mean hits beat the base's by more than this
WIDTH_DROP = 0.05  # width check: the reduced run may trail the full-width run by at most this much
CHANGES = {  # recipe choice: the changes measured against the base, per arm prefix
    "views": ("span1", "nomask", "noaffine"),  # round 3, the two-view modules
    "rel": ("jit", "rigid", "both"),  # round 4, the relative module augmented
}


def _context(key: str) -> str | None:
    m = re.match(r"U\d+x(T\d+)\|", key)
    return m.group(1) if m else None


def _within(a: float, b: float) -> bool:
    """``a >= b`` without float noise: fractions on 500 trials differ by multiples of 0.002, so nine places suffice."""
    return round(a - b, 9) >= 0


def _hidden_flag(hidden: float | None, ref: float | None) -> bool:
    """Goal-hidden hits strictly more than the reference's + MARGIN, without float noise; False when either is missing."""
    return hidden is not None and ref is not None and not _within(ref + MARGIN, hidden)


def _flag_cell(r: dict) -> str:
    """The flag column: n/a when the goal-hidden row or its reference was not measured."""
    if r["hidden_hits"] is None or r["ref_hidden_hits"] is None:
        return "n/a"
    return "FLAG" if r["flag"] else "ok"


def gate_table(check: dict, image_arms: list[str]) -> list[dict]:
    """Per context (from the cell key ``U0xT<k>|...``) and image arm, the gate verdict.

    Reads ``metrics`` only. Metrics are fractions in [0, 1], so the rule is in fractions (10 points = 0.10).
    hits and collision are from ``oracle_task_<x>``; ref_hits and ref_collision from ``oracle_task`` (the
    parametric module); ft_hits from ``oracle_ft`` (the fine-tuned policy, None when the run has no such row).
    ``collisions_ok`` = collision <= ref_collision + 0.10. ``pass`` = hits >= ref_hits - 0.10 and collisions_ok.
    Else ``marginal`` = hits > ft_hits and collisions_ok, else ``FAIL``; pass wins where it overlaps the floor.
    Without an ``oracle_ft`` row no arm is marginal. ``flag`` = hidden_hits > ref_hidden_hits + 0.10 (``_hidden_flag``),
    with hidden_hits from ``policy_task_<x>`` and ref_hidden_hits from ``policy_task``; False when either is missing.
    The wrong-target shares (the metrics' ``wrong``; None when the row or the metric is missing): wrong_target from
    ``oracle_task_<x>`` (goal given), hidden_wrong_target from ``policy_task_<x>`` and ref_hidden_wrong_target from
    ``policy_task`` (goal hidden).
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
        ref_hw = met.get("policy_task", {}).get("wrong")
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
                    flag=_hidden_flag(hidden, ref_h),
                    wrong_target=o.get("wrong"),
                    hidden_wrong_target=p.get("wrong") if p else None,
                    ref_hidden_wrong_target=ref_hw,
                )
            )
    return sorted(rows, key=lambda r: (r["context"], r["arm"]))


def _context_means(run: dict, method: str) -> dict[str, dict[str, float]]:
    """Per context, the mean of ``success`` and ``collision`` of ``method`` over the run's cells that have it."""
    by_ctx: dict[str, list[dict]] = {}
    for key, cell in sorted(run.get("test", {}).items()):
        ctx = _context(key)
        if ctx is not None and method in cell.get("metrics", {}):
            by_ctx.setdefault(ctx, []).append(cell["metrics"][method])
    return {
        ctx: {k: float(np.mean([m[k] for m in ms])) for k in ("success", "collision")}
        for ctx, ms in by_ctx.items()
    }


def scan_rows(runs: dict[int, dict], prefix: str = "views") -> list[dict]:
    """Per context and demonstration count, how the image module compares with the parametric one.

    ``runs`` maps a count N to that count's run; the module rows are its ``oracle_task_<prefix>_n<N>`` and
    ``policy_task_<prefix>_n<N>`` methods, the reference ``oracle_task`` / ``policy_task``. ``level`` = hits >=
    ref_hits - LEVEL and collision <= ref_collision + LEVEL; ``in_band`` the same with MARGIN; ``flag`` as in
    ``gate_table``. Raises ValueError when a context's ``oracle_task`` metrics differ between runs (different layouts).
    """
    refs: dict[str, dict] = {}
    rows = []
    contexts = {c for run in runs.values() for c in map(_context, run.get("test", {})) if c}
    for n, run in sorted(runs.items()):
        for method in ("oracle_task", f"oracle_task_{prefix}_n{n}"):
            have = _context_means(run, method)
            for ctx in sorted(contexts - set(have)):
                raise ValueError(f"the run for {n} demonstrations has no {method} row for context {ctx}")
        for ctx, ref in _context_means(run, "oracle_task").items():
            if ctx in refs and any(abs(refs[ctx][k] - ref[k]) > 1e-9 for k in ref):
                raise ValueError(f"oracle_task differs between runs for {ctx}: the runs used different layouts")
            refs[ctx] = ref
        views = _context_means(run, f"oracle_task_{prefix}_n{n}")
        hidden = _context_means(run, f"policy_task_{prefix}_n{n}")
        ref_hidden = _context_means(run, "policy_task")
        for ctx, v in views.items():
            ref = refs.get(ctx)
            if ref is None:
                continue
            h, rh = hidden.get(ctx, {}).get("success"), ref_hidden.get(ctx, {}).get("success")
            ok = lambda margin: _within(v["success"], ref["success"] - margin) and _within(  # noqa: E731
                ref["collision"] + margin, v["collision"]
            )
            rows.append(
                dict(
                    context=ctx,
                    demos=n,
                    hits=v["success"],
                    collision=v["collision"],
                    ref_hits=ref["success"],
                    ref_collision=ref["collision"],
                    level=ok(LEVEL),
                    in_band=ok(MARGIN),
                    hidden_hits=h,
                    ref_hidden_hits=rh,
                    flag=_hidden_flag(h, rh),
                )
            )
    return sorted(rows, key=lambda r: (r["context"], r["demos"]))


def first_count(rows: list[dict], key: str) -> dict[str, int | None]:
    """Per context, the smallest demonstration count whose row has ``key`` true (None when none does)."""
    out: dict[str, int | None] = {}
    for r in rows:
        out.setdefault(r["context"], None)
        if r[key] and (out[r["context"]] is None or r["demos"] < out[r["context"]]):
            out[r["context"]] = r["demos"]
    return out


def crossing(counts: list[int], values: list[float], threshold: float) -> tuple[float | None, bool]:
    """Where the curve of ``values`` over ``counts``, linear in log10(count), first reaches ``threshold``.

    The first count when the curve starts at or above it. Past the last count the last segment is continued when
    it rises, and the result is marked True (an extrapolation). ``(None, False)`` when it never gets there, or when
    the extrapolated count is more than 10 times the largest count given (a nearly flat last segment).
    """
    x = [float(np.log10(c)) for c in counts]
    for i, v in enumerate(values):
        if _within(v, threshold):
            if i == 0:
                return float(counts[0]), False
            frac = (threshold - values[i - 1]) / (v - values[i - 1])
            return float(10 ** (x[i - 1] + frac * (x[i] - x[i - 1]))), False
    if len(values) >= 2 and values[-1] > values[-2]:
        slope = (values[-1] - values[-2]) / (x[-1] - x[-2])
        n = float(10 ** min(x[-1] + (threshold - values[-1]) / slope, 300.0))  # the cap keeps a flat slope finite
        return (n, True) if round(10 * max(counts) - n, 6) >= 0 else (None, False)
    return None, False


def joint_crossing(
    counts: list[int], hits: list[float], collisions: list[float], hits_bar: float, collision_bar: float
) -> tuple[float | None, bool]:
    """Where both the hits curve has risen to ``hits_bar`` and the collisions curve has fallen to ``collision_bar``.

    Each from ``crossing`` (collisions negated); the later of the two counts, marked extrapolated when that one is
    (on a tie when either is). ``(None, False)`` when either curve does not get there on its trend.
    """
    h = crossing(counts, hits, hits_bar)
    c = crossing(counts, [-v for v in collisions], -collision_bar)
    if h[0] is None or c[0] is None:
        return None, False
    if h[0] == c[0]:
        return h[0], h[1] or c[1]
    return max(h, c, key=lambda t: t[0])


def _arm_method(arm: str, prefix: str) -> str:
    """The row of an arm: ``oracle_task_<prefix>_<arm>``, or ``oracle_task_<prefix>`` for an arm named as the prefix
    (the unaugmented module, e.g. ``oracle_task_rel``)."""
    return f"oracle_task_{prefix}" if arm == prefix else f"oracle_task_{prefix}_{arm}"


def _arm_means(check: dict, arms: list[str], prefix: str = "views") -> dict[str, float]:
    """Mean hits of each arm's row (``_arm_method``) over the contexts (each context's mean over its cells)."""
    out = {}
    have = {arm: _context_means(check, _arm_method(arm, prefix)) for arm in arms}
    contexts = set().union(*have.values())
    for arm in arms:
        per_ctx = have[arm]
        if not per_ctx:
            raise ValueError(f"{_arm_method(arm, prefix)} has no rows in the run")
        for ctx in sorted(contexts - set(per_ctx)):
            raise ValueError(f"{_arm_method(arm, prefix)} has no row for context {ctx}")
        out[arm] = float(np.mean([m["success"] for m in per_ctx.values()]))
    return out


def select_recipe(
    check: dict, base: str = "base", changes: list[str] | tuple[str, ...] | None = None, prefix: str = "views"
) -> dict:
    """Mean hits of each recipe over the contexts, and the changes whose mean beats the base's by more than ADOPT.

    The rows are ``oracle_task_<prefix>_<arm>`` (``_arm_method``); ``changes`` defaults to the prefix's ``CHANGES``.
    ``per_context`` holds the rows behind the means: ``{arm: {context: {"hits": .., "collision": ..}}}``.
    """
    if changes is None:
        if prefix not in CHANGES:
            raise ValueError(f"no default changes for prefix {prefix!r}; known: {sorted(CHANGES)}")
        changes = CHANGES[prefix]
    means = _arm_means(check, [base, *changes], prefix=prefix)
    adopted = [c for c in changes if round(means[c] - means[base] - ADOPT, 9) > 0]
    per_context = {
        arm: {
            ctx: dict(hits=m["success"], collision=m["collision"])
            for ctx, m in sorted(_context_means(check, _arm_method(arm, prefix)).items())
        }
        for arm in (base, *changes)
    }
    return dict(means=means, adopted=adopted, per_context=per_context)


def width_check(
    check: dict, reduced: str = "base", full: str = "full", context: str = "T4", prefix: str = "views"
) -> dict:
    """The reduced-width run's hits against the full-width run's, in ``context``; ok within WIDTH_DROP. The rows are
    ``oracle_task_<prefix>_<arm>`` (``_arm_method``: an arm named like the prefix reads the bare row)."""
    hits = []
    for arm in (reduced, full):
        method = _arm_method(arm, prefix)
        met = _context_means(check, method).get(context)
        if met is None:
            raise ValueError(f"{method} has no row for context {context}")
        hits.append(met["success"])
    r, f = hits
    return dict(reduced_hits=r, full_hits=f, ok=_within(r, f - WIDTH_DROP))


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


def _scan_report(a, ap: argparse.ArgumentParser) -> None:
    runs, names = {}, {}
    for name in a.scan:
        run = _load(a.results, name)
        pattern = re.compile(rf"oracle_task_{re.escape(a.prefix)}_n(\d+)$")
        counts = {int(m.group(1)) for m in map(pattern.match, _methods(run)) if m}
        if len(counts) != 1:
            raise SystemExit(
                f"{name}: expected exactly one oracle_task_{a.prefix}_n<N> method, found {sorted(counts)}"
            )
        n = counts.pop()
        if n in runs:
            ap.error(f"--scan: two runs ({names[n]} and {name}) are for {n} demonstrations; give one run per count")
        runs[n], names[n] = run, name
    rows = scan_rows(runs, prefix=a.prefix)
    largest = max(runs)
    first_level, first_band = first_count(rows, "level"), first_count(rows, "in_band")
    crossings = {}
    for ctx in sorted(first_level):
        cr = [r for r in rows if r["context"] == ctx]
        ref, ref_c = cr[0]["ref_hits"], cr[0]["ref_collision"]
        crossings[ctx] = {}
        for name, margin in (("level", LEVEL), ("in_band", MARGIN)):
            n, extra = joint_crossing(
                [r["demos"] for r in cr], [r["hits"] for r in cr], [r["collision"] for r in cr], ref - margin,
                ref_c + margin,
            )
            crossings[ctx][name] = dict(demos=n, extrapolated=extra)

    def show(n):
        return f"more than {largest}" if n is None else str(n)

    def show_cross(c):
        if c["demos"] is None:
            return "not reached on this trend"
        return f"{c['demos']:.0f}" + (" (extrapolated)" if c["extrapolated"] else "")

    md = ["# Image task modules: demonstration scan", "", "## Scan (success and collision in %)", ""]
    md += _table(
        ["context", "demos", "hits", "ref hits", "collision", "ref collision", "level", "in band", "hidden hits",
         "ref hidden", "flag"],
        [
            [r["context"], r["demos"], _fmt(r["hits"]), _fmt(r["ref_hits"]), _fmt(r["collision"]),
             _fmt(r["ref_collision"]), "yes" if r["level"] else "no", "yes" if r["in_band"] else "no",
             _fmt(r["hidden_hits"]), _fmt(r["ref_hidden_hits"]), _flag_cell(r)]
            for r in rows
        ],
    )
    for ctx in sorted(first_level):
        md += [
            f"- {ctx}: first level count {show(first_level[ctx])}; first in-band count {show(first_band[ctx])}; "
            f"crossing of the level bar on hits and collisions {show_cross(crossings[ctx]['level'])}; "
            f"crossing of the in-band bar on hits and collisions {show_cross(crossings[ctx]['in_band'])}",
        ]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    (a.out / "report.json").write_text(
        json.dumps(
            dict(scan=rows, first_level=first_level, first_in_band=first_band, crossings=crossings), indent=2
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", help="gate run name")
    mode.add_argument("--select", metavar="RUN", help="recipe-choice run name")
    mode.add_argument("--width", metavar="RUN", help="width-check run name")
    mode.add_argument("--scan", nargs="+", metavar="RUN", help="demonstration-scan run names, one per count")
    ap.add_argument("--reduced", default="base", help="--width: the reduced-width arm (oracle_task_<prefix>_<arm>)")
    ap.add_argument("--full", default="full", help="--width: the full-width arm (oracle_task_<prefix>_<arm>)")
    ap.add_argument("--context", default="T4", help="--width: the context the two arms are compared in")
    ap.add_argument("--prefix", default="views", help="--select, --scan, --width: the arm rows' prefix "
                    "(oracle_task_<prefix>_<arm>)")
    ap.add_argument("--base", default="base", help="the base arm of --select; an arm named like --prefix reads the bare "
                    "oracle_task_<prefix> row")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.select or a.width:
        if a.select:
            out = select_recipe(_load(a.results, a.select), base=a.base, prefix=a.prefix)
        else:
            out = width_check(_load(a.results, a.width), reduced=a.reduced, full=a.full, context=a.context, prefix=a.prefix)
        print(json.dumps(out, indent=2))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / ("select.json" if a.select else "width.json")).write_text(json.dumps(out, indent=2))
        return
    if a.scan:
        return _scan_report(a, ap)

    check = _load(a.results, a.check)
    arms = [m[len("oracle_task_"):] for m in _methods(check) if m.startswith("oracle_task_")]

    gate = gate_table(check, arms)
    md = ["# Image task modules: gate", "", "## Gate (success and collision in %)", ""]
    if gate:
        md += _table(
            ["context", "arm", "hits", "ref hits", "ft hits", "collision", "ref collision", "verdict",
             "hidden hits", "ref hidden", "flag", "wrong target", "hidden wrong", "ref hidden wrong"],
            [
                [r["context"], r["arm"], _fmt(r["hits"]), _fmt(r["ref_hits"], none="n/a"),
                 _fmt(r["ft_hits"], none="n/a"), _fmt(r["collision"]), _fmt(r["ref_collision"]),
                 r["verdict"], _fmt(r["hidden_hits"]), _fmt(r["ref_hidden_hits"]),
                 _flag_cell(r), _fmt(r["wrong_target"], none="n/a"), _fmt(r["hidden_wrong_target"], none="n/a"),
                 _fmt(r["ref_hidden_wrong_target"], none="n/a")]
                for r in gate
            ],
        )
    else:
        md += ["The gate run has no cells; no gate verdicts.", ""]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    (a.out / "report.json").write_text(
        json.dumps(
            dict(scan=rows, first_level=first_level, first_in_band=first_band, crossings=crossings), indent=2
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", help="gate run name")
    mode.add_argument("--select", metavar="RUN", help="recipe-choice run name")
    mode.add_argument("--width", metavar="RUN", help="width-check run name")
    mode.add_argument("--scan", nargs="+", metavar="RUN", help="demonstration-scan run names, one per count")
    ap.add_argument("--reduced", default="base", help="--width: the reduced-width arm (oracle_task_<prefix>_<arm>)")
    ap.add_argument("--full", default="full", help="--width: the full-width arm (oracle_task_<prefix>_<arm>)")
    ap.add_argument("--context", default="T4", help="--width: the context the two arms are compared in")
    ap.add_argument("--prefix", default="views", help="--select, --scan, --width: the arm rows' prefix "
                    "(oracle_task_<prefix>_<arm>)")
    ap.add_argument("--base", default="base", help="the base arm of --select; an arm named like --prefix reads the bare "
                    "oracle_task_<prefix> row")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.select or a.width:
        if a.select:
            out = select_recipe(_load(a.results, a.select), base=a.base, prefix=a.prefix)
        else:
            out = width_check(_load(a.results, a.width), reduced=a.reduced, full=a.full, context=a.context, prefix=a.prefix)
        print(json.dumps(out, indent=2))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / ("select.json" if a.select else "width.json")).write_text(json.dumps(out, indent=2))
        return
    if a.scan:
        return _scan_report(a, ap)

    check = _load(a.results, a.check)
    arms = [m[len("oracle_task_"):] for m in _methods(check) if m.startswith("oracle_task_")]

    gate = gate_table(check, arms)
    md = ["# Image task modules: gate", "", "## Gate (success and collision in %)", ""]
    if gate:
        md += _table(
            ["context", "arm", "hits", "ref hits", "ft hits", "collision", "ref collision", "verdict",
             "hidden hits", "ref hidden", "flag", "wrong target", "hidden wrong", "ref hidden wrong"],
            [
                [r["context"], r["arm"], _fmt(r["hits"]), _fmt(r["ref_hits"], none="n/a"),
                 _fmt(r["ft_hits"], none="n/a"), _fmt(r["collision"]), _fmt(r["ref_collision"]),
                 r["verdict"], _fmt(r["hidden_hits"]), _fmt(r["ref_hidden_hits"]),
                 _flag_cell(r), _fmt(r["wrong_target"], none="n/a"), _fmt(r["hidden_wrong_target"], none="n/a"),
                 _fmt(r["ref_hidden_wrong_target"], none="n/a")]
                for r in gate
            ],
        )
    else:
        md += ["The gate run has no cells; no gate verdicts.", ""]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    (a.out / "report.json").write_text(
        json.dumps(dict(gate=gate), indent=2)
    )


if __name__ == "__main__":
    main()
