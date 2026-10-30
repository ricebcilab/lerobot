"""The adapter's rank knobs on the grid (seed 0's trials and task modules): user modules retrained per variant, then
MARS (user + task module; on T1 the user module alone) on every cell, its depth tuned as the grid tunes it. GPU 1 only
(GPU 0 drives the desktop); at most 3 trainings and 6 processes at once; a job starts once its checkpoints exist.
Variants: step rank {1, 2, 4} at 10 and 100 demonstrations (4 is the default, every step's matrix free, which spans the
same matrices), cross-step rank {2, 8, 32} at 100; step rank 4 / cross-step rank 0 at 100 demonstrations is the
grid's own user module, whose results are the grid's (results/sim/sweeps/grid_s0).

    python research/adapter_ranks_sched.py

Writes results/research/adapter_ranks/ (ranks.yaml, sched.log, logs/) and results/sim/sweeps/adapter_ranks/; the user
modules are checkpoints/adapters/user_<user>_expert_<variant>_s0 (training seed 0, after the demonstration count)."""
import os, subprocess, sys, time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cursor_sa.paths import CONFIGS, RESULTS, adapter_path  # noqa: E402
S = RESULTS / "research" / "adapter_ranks"
PY = sys.executable
U = {"U1": ("gain025y", "{scale: [1.0, 0.25]}"), "U2": ("gain025x", "{scale: [0.25, 1.0]}"),
     "U3": ("rot60", "{rotation_deg: 60}"), "U4": ("curl20", "{curl: 20, tau: 0.1}")}
VARIANTS = {"sr1_n10": (10, 1, 0), "sr2_n10": (10, 2, 0), "sr4_n10": (10, None, 0), "sr1_n100": (100, 1, 0),
            "sr2_n100": (100, 2, 0), "cr2_n100": (100, None, 2), "cr8_n100": (100, None, 8), "cr32_n100": (100, None, 32)}
STEPS = {10: 100, 100: 500}   # the data-efficiency recipe: 5 steps per demonstration, at least 100
name = lambda s, v: f"user_{s}_expert_{v}_s0"
ckpt = lambda n: adapter_path(n)



def sweep_config():
    """The ranks sweep: the grid's arm, cells, depths, trials and seed, one MARS method per variant."""
    grid = yaml.safe_load((CONFIGS / "sweeps" / "grid.yaml").read_text())
    return dict(name="adapter_ranks", agent="expert", arm=grid["arm"],
                methods={f"mars_{v}": dict(mode="frs", reversal_adapter=dict(user={u: name(s, v) for u, (s, _) in U.items()}, task="cell"))
                         for v in VARIANTS},
                depths=dict(frs=grid["depths"]["frs"]), grid=grid["grid"], trials=grid["trials"], focus="mars_cr8_n100",
                against=[f"mars_{v}" for v in VARIANTS if v != "cr8_n100"], seed=grid["seed"])


if __name__ == "__main__":
    (S / "logs").mkdir(parents=True, exist_ok=True)
    (S / "ranks.yaml").write_text("# written by adapter_ranks_sched.py\n" + yaml.safe_dump(sweep_config(), sort_keys=False, width=140))
    jobs = []
    for v, (n, sr, cr) in VARIANTS.items():
        for s, c in U.values():
            knobs = (f" --step-rank {sr}" if sr else "") + (f" --cross-step-rank {cr}" if cr else "")
            jobs.append(dict(name=f"train_{s}_{v}", kind="train", deps=[],
                             cmd=f"{PY} -u train/train_adapter.py --kind user --corruption '{c}' --name {name(s, v)} "
                                 f"--n-demos {n} --steps {STEPS[n]} --frozen-noise --log-every 50{knobs}"))
    users = [name(s, v) for v in VARIANTS for s, _ in U.values()]
    for cell in [f"{u}x{t}" for t in ["T5", "T4", "T3", "T2", "T1"] for u in U]:
        jobs.append(dict(name=f"eval_{cell}", kind="eval", deps=[ckpt(n) for n in users],
                         cmd=f"{PY} -u run_sweep.py --sweep {S}/ranks.yaml --cells {cell} --out adapter_ranks/{cell}"))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="4")
    log = open(S / "sched.log", "a")
    say = lambda m: (log.write(f"{time.strftime('%H:%M:%S')} {m}\n"), log.flush())
    say(f"{len(jobs)} jobs")
    running, failed = [], []
    while jobs or running:
        for p, j in list(running):
            if p.poll() is not None:
                running.remove((p, j)); say(f"{'done' if p.returncode == 0 else 'FAILED'} {j['name']}")
                if p.returncode:
                    failed.append(j["name"])
        for j in list(jobs):
            if len(running) >= 6:
                break
            if j["kind"] == "train" and sum(1 for _, r in running if r["kind"] == "train") >= 3:
                continue
            if all(Path(d).exists() for d in j["deps"]):
                p = subprocess.Popen(j["cmd"], shell=True, cwd=ROOT, env=env, stdout=open(S / "logs" / f"{j['name']}.log", "w"),
                                     stderr=subprocess.STDOUT)
                running.append((p, j)); jobs.remove(j); say(f"start {j['name']}")
        time.sleep(10)
    say(f"all finished; failed: {failed}")
