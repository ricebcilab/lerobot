"""The port's acceptance check: a re-run sweep's success per cell and method against the original run's.

    python research/port_check.py --old results/sim/sweeps/grid_s0/test.json --new results/sim/sweeps/port_check/test.json

For every cell and method of either test.json it prints the old success, the new success, the old run's 95%
Wilson interval (from its hits and trials) and whether the new value lies inside it. A cell or method found in
only one of the two files fails the check, as does a new value outside its interval. Writes
results/research/port_check/port_check.json (every row and a summary: how many values lie inside, how many are
exactly equal, how many test trials agree trial for trial, the largest absolute difference and where) and exits 1
if any value fails, or if the two files hold no values at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import math

from cursor_sa.paths import RESULTS

Z95 = 1.959963984540054  # the standard normal's 97.5% quantile


def wilson(hits, n, z=Z95):
    """The Wilson score interval of a binomial proportion hits / n."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def compare(old, new):
    """One row per (cell, method) of either run's `test`; the summary of the rows."""
    rows = []
    for cell in sorted(set(old) | set(new), key=lambda k: (k not in old, list(old).index(k) if k in old else 0, k)):
        o_metrics = old.get(cell, {}).get("metrics", {})
        n_metrics = new.get(cell, {}).get("metrics", {})
        for m in list(o_metrics) + [m for m in n_metrics if m not in o_metrics]:
            o, n = o_metrics.get(m), n_metrics.get(m)
            row = {"cell": cell, "method": m, "old": None if o is None else o["success"],
                   "new": None if n is None else n["success"], "old_n": None if o is None else o["n"],
                   "new_n": None if n is None else n["n"]}
            if o is None or n is None:
                row.update(lo=None, hi=None, inside=False, equal=False, diff=None, trials_identical=False,
                           why=f"only in the {'new' if o is None else 'old'} run")
            else:
                lo, hi = wilson(o["hits"], o["n"])
                o_trials = old[cell].get("trials", {}).get("outcome", {}).get(m)
                n_trials = new[cell].get("trials", {}).get("outcome", {}).get(m)
                row.update(lo=lo, hi=hi, inside=bool(lo <= n["success"] <= hi), equal=o["success"] == n["success"],
                           diff=n["success"] - o["success"], trials_identical=o_trials is not None and o_trials == n_trials,
                           why=None if lo <= n["success"] <= hi else "outside the old 95% Wilson interval")
            rows.append(row)
    compared = [r for r in rows if r["diff"] is not None]
    largest = max(compared, key=lambda r: abs(r["diff"]), default=None)
    summary = {"values": len(rows), "inside": sum(r["inside"] for r in rows), "failed": sum(not r["inside"] for r in rows),
               "equal": sum(r["equal"] for r in rows), "trials_identical": sum(r["trials_identical"] for r in rows),
               "largest_abs_diff": None if largest is None else abs(largest["diff"]),
               "largest_at": None if largest is None else f"{largest['cell']} {largest['method']}",
               "trial_counts_differ": [f"{r['cell']} {r['method']}" for r in compared if r["old_n"] != r["new_n"]]}
    return rows, summary


def pct(v):
    return "    -" if v is None else f"{100 * v:5.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", required=True, help="the original run's test.json")
    p.add_argument("--new", required=True, help="the re-run's test.json")
    p.add_argument("--out", default=str(RESULTS / "research" / "port_check"), help="folder for port_check.json")
    args = p.parse_args(argv)

    old = json.loads(Path(args.old).read_text())["test"]
    new = json.loads(Path(args.new).read_text())["test"]
    rows, summary = compare(old, new)
    print(f"{'cell':12s} {'method':20s} {'old':>5s} {'new':>5s}   old 95% Wilson   inside")
    for r in rows:
        ci = "      -       " if r["lo"] is None else f"[{pct(r['lo'])}, {pct(r['hi'])}]"
        print(f"{r['cell']:12s} {r['method']:20s} {pct(r['old'])} {pct(r['new'])}   {ci}   "
              f"{'yes' if r['inside'] else 'NO  ' + r['why']}")
    print(f"\n{summary['inside']}/{summary['values']} inside, {summary['equal']} exactly equal, {summary['trials_identical']} "
          f"identical trial for trial; largest |new - old| {summary['largest_abs_diff']} at {summary['largest_at']}")
    if summary["trial_counts_differ"]:
        print(f"trial counts differ: {summary['trial_counts_differ']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "port_check.json", "w") as f:
        json.dump({"old": str(args.old), "new": str(args.new), "summary": summary, "rows": rows}, f, indent=1)
    print(f"written to {out / 'port_check.json'}")
    if not summary["values"]:
        print("nothing compared: both runs have no values")
        return 1
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
