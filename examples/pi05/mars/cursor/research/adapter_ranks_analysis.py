"""The rank knobs' results: MARS per variant against the grid's own MARS (step rank 4, cross-step rank 0, 100
demonstrations; results/sim/sweeps/grid_s0, seed 0) on the same trials. Success (%) on T1 (the user module alone), on
the 16 obstacle cells and on U4's obstacle cells, with the paired difference from the grid's MARS (95% CI resampling the
trials within each cell, the cells fixed).

    python research/adapter_ranks_analysis.py

Writes results/research/adapter_ranks/ranks_stats.json."""
import json, sys, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cursor_sa.paths import RESULTS  # noqa: E402
SW = RESULTS / "sim" / "sweeps"
U, T, OBS = ["U1", "U2", "U3", "U4"], ["T1", "T2", "T3", "T4", "T5"], ["T2", "T3", "T4", "T5"]
VARIANTS = ["sr1_n10", "sr2_n10", "sr4_n10", "sr1_n100", "sr2_n100", "cr2_n100", "cr8_n100", "cr32_n100"]
LABEL = {"sr1_n10": "step rank 1, 10 demos", "sr2_n10": "step rank 2, 10 demos", "sr4_n10": "step rank 4, 10 demos",
         "sr1_n100": "step rank 1, 100 demos", "sr2_n100": "step rank 2, 100 demos", "base": "step rank 4, 100 demos (grid)",
         "cr2_n100": "cross-step rank 2, 100", "cr8_n100": "cross-step rank 8, 100", "cr32_n100": "cross-step rank 32, 100"}
grid = json.loads((SW / "grid_s0" / "test.json").read_text())["test"]
cells = {}
for u in U:
    for t in T:
        f = SW / "adapter_ranks" / f"{u}x{t}" / "test.json"
        cells[f"{u}x{t}"] = json.loads(f.read_text())["test"][f"{u}x{t}|1.0"] if f.exists() else None
hits = lambda d, m: np.array([o == "hit" for o in d["trials"]["outcome"][m]], float)
def get(cell, v):
    return hits(grid[f"{cell}|1.0"], "mars") if v == "base" else hits(cells[cell], f"mars_{v}")
def summary(v, keys, n_boot=4000):
    a = [get(c, v) for c in keys]; b = [get(c, "base") for c in keys]
    d = [x - y for x, y in zip(a, b)]
    rng = np.random.default_rng(zlib.crc32((v + "".join(keys)).encode()))
    boot = [np.mean([di[rng.integers(0, len(di), len(di))].mean() for di in d]) for _ in range(n_boot)]
    return 100 * np.mean([x.mean() for x in a]), 100 * np.mean([x.mean() for x in d]), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)
done = [c for c, d in cells.items() if d is not None]
print(f"cells done: {len(done)}/20")
groups = {"T1 (user module alone)": [f"{u}xT1" for u in U], "16 obstacle cells": [f"{u}x{t}" for u in U for t in OBS],
          "U4 (curl) obstacle cells": [f"U4x{t}" for t in OBS]}
out = {}
for g, keys in groups.items():
    if not all(k in done for k in keys):
        continue
    print(f"\n== {g}: MARS success %, and the difference from the grid's MARS [95% CI]")
    for v in ["base"] + VARIANTS:
        m, d, lo, hi = summary(v, keys)
        out.setdefault(v, {})[g] = dict(success=round(m, 1), diff=[round(d, 1), round(lo, 1), round(hi, 1)])
        print(f"  {LABEL[v]:32s} {m:5.1f}" + ("" if v == "base" else f"   {d:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"))
print("\n== per cell, MARS success % (tuned depth)")
print(f"{'cell':7s}" + "".join(f"{v:>11s}" for v in ["base"] + VARIANTS))
for c in done:
    row = [f"{100 * get(c, v).mean():5.1f}" for v in ["base"] + VARIANTS]
    print(f"{c:7s}" + "".join(f"{r:>11s}" for r in row))
(RESULTS / "research" / "adapter_ranks" / "ranks_stats.json").write_text(json.dumps(out, indent=1))
