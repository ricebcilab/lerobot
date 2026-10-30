"""Analyze experiment runs: per participant and pooled, human or simulated.

    python analyze_results.py                               # every run under results/p*/ and results/sim/
    python analyze_results.py --participants p1 p2 sim      # some participants
    python analyze_results.py --protocol study --out study  # one protocol, named output folder

Practice and aborted trials are excluded. Writes to results/analysis/<out>/:

    trials.csv            one row per trial (participant, run, block, arm, trial, outcome, duration, ...)
    summary.csv           per participant x run x arm: n, success, wrong-target and timeout rates, median time of the hits
    summary.md            the same, readable, plus the pooled table across people
    success_by_arm.png    hit rate per arm for each participant, and the mean over people
    learning_curves.png   hit rate against trial index within a block, per arm, pooled over people
    paths_<participant>.png   cursor paths of every trial, one panel per arm
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cursor_sa.analysis import INK, INK_2, style
from cursor_sa.env import TaskConfig
from cursor_sa.paths import RESULTS
from cursor_sa.records import load_run, summarize

ARM_COLORS = ["#8a8984", "#eb6834", "#1baf7a", "#2a78d6", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def find_runs(results, participants=None, protocol=None):
    """(participant, run name, run dir) for every run with a manifest."""
    runs = []
    for pdir in sorted(Path(results).iterdir()):
        if not pdir.is_dir() or not (pdir.name == "sim" or (pdir.name[0] == "p" and pdir.name[1:].isdigit())):
            continue
        if participants and pdir.name not in participants:
            continue
        for rdir in sorted(pdir.iterdir()):
            if (rdir / "manifest.json").exists():
                manifest = json.loads((rdir / "manifest.json").read_text())
                if protocol is None or manifest["protocol"]["name"] == protocol:
                    runs.append((pdir.name, rdir.name, rdir))
    return runs


def load_trials(runs):
    trials, arms_in_order = [], []
    for participant, run, rdir in runs:
        manifest, records = load_run(rdir)
        for arm in manifest["block_order"]:
            if arm not in arms_in_order:
                arms_in_order.append(arm)
        trials += [r for r in records if not r["practice"] and r["outcome"] != "aborted"]
    return trials, arms_in_order


def by_arm(trials):
    """Per (participant, run, arm) outcome summary, in first-seen order."""
    groups = {}
    for r in trials:
        groups.setdefault((r["participant"], r["run"], r["arm"]), []).append(r)
    return [dict(participant=participant, run=run, arm=arm, method=rs[0]["method"], **summarize(rs))
            for (participant, run, arm), rs in groups.items()]


def pooled(rows, arms):
    """Mean and sem of the per-person success rate for each arm, people only (not sim)."""
    out = []
    for arm in arms:
        rates = [r["success"] for r in rows if r["arm"] == arm and r["participant"] != "sim"]
        sim = [r["success"] for r in rows if r["arm"] == arm and r["participant"] == "sim"]
        out.append(dict(arm=arm, n_people=len(rates), mean_success=float(np.mean(rates)) if rates else None,
                        sem_success=float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else None,
                        sim_success=float(np.mean(sim)) if sim else None))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_summary_md(path, rows, pool):
    fmt = lambda x, p=True: "—" if x is None else (f"{100 * x:.1f}%" if p else f"{x:.2f}")
    lines = ["# Experiment summary", "",
             "| Participant | Run | Arm | n | Hit | Wrong target | Collision | Timeout | Median time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['participant']} | {r['run']} | {r['arm']} | {r['n']} | {fmt(r['success'])} | {fmt(r['wrong'])} | "
              f"{fmt(r.get('collision', 0.0))} | {fmt(r['timeout'])} | {fmt(r['time_s'], False)} |" for r in rows]
    lines += ["", "## Pooled over people (mean of per-person hit rates)", "", "| Arm | People | Hit rate | sem | Expert (sim) |",
              "|---|---|---|---|---|"]
    lines += [f"| {p['arm']} | {p['n_people']} | {fmt(p['mean_success'])} | {fmt(p['sem_success'])} | {fmt(p['sim_success'])} |"
              for p in pool]
    Path(path).write_text("\n".join(lines) + "\n")


def fig_success(rows, pool, arms, out):
    participants = []
    for r in rows:
        if r["participant"] not in participants:
            participants.append(r["participant"])
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(participants) + 1.2, 3.4))
    width = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        xs, ys = [], []
        for i, p in enumerate(participants):
            rs = [r for r in rows if r["participant"] == p and r["arm"] == arm]
            if rs:
                xs.append(i + (j - (len(arms) - 1) / 2) * width)
                ys.append(100 * np.mean([r["success"] for r in rs]))
        ax.bar(xs, ys, width=width * 0.95, color=ARM_COLORS[j % len(ARM_COLORS)], label=arm)
        pooled_arm = [p for p in pool if p["arm"] == arm][0]
        if pooled_arm["mean_success"] is not None:
            x = len(participants) + (j - (len(arms) - 1) / 2) * width
            err = 100 * pooled_arm["sem_success"] if pooled_arm["sem_success"] else 0
            ax.bar([x], [100 * pooled_arm["mean_success"]], width=width * 0.95, color=ARM_COLORS[j % len(ARM_COLORS)],
                   yerr=[err], capsize=3, edgecolor=INK, lw=0.8)
    labels = participants + (["people (mean)"] if any(p["mean_success"] is not None for p in pool) else [])
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Hit rate (%)", ylim=(0, 105))
    ax.legend(fontsize=8, ncol=min(4, len(arms)), loc="upper left")
    fig.savefig(out / "success_by_arm.png")
    plt.close(fig)


def fig_learning(trials, arms, out, window=5):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for j, arm in enumerate(arms):
        rs = [r for r in trials if r["arm"] == arm and r["participant"] != "sim"] or \
             [r for r in trials if r["arm"] == arm]
        if not rs:
            continue
        n = max(r["trial"] for r in rs)
        by_trial = [[r["outcome"] == "hit" for r in rs if r["trial"] == i] for i in range(1, n + 1)]
        rate = np.array([np.mean(b) if b else np.nan for b in by_trial])
        kernel = np.ones(window) / window
        smooth = np.convolve(np.nan_to_num(rate), kernel, mode="same") / np.convolve(~np.isnan(rate), kernel, mode="same")
        ax.plot(range(1, n + 1), 100 * smooth, color=ARM_COLORS[j % len(ARM_COLORS)], lw=2, label=arm)
    ax.set(xlabel="Trial within block", ylabel=f"Hit rate (%), {window}-trial window", ylim=(0, 105))
    ax.legend(fontsize=8)
    ax.set_title("Learning within a block (people pooled; sim if no people)", fontsize=9, color=INK)
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)


def fig_paths(trials, arms, participant, out):
    rs = [r for r in trials if r["participant"] == participant]
    runs = sorted({r["run"] for r in rs})
    if not rs:
        return
    rs = [r for r in rs if r["run"] == runs[0]]
    fig, axes = plt.subplots(1, len(arms), figsize=(2.6 * len(arms), 2.8), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        arm_trials = [r for r in rs if r["arm"] == arm]
        for r in arm_trials:
            path = np.array(r["path"])
            color = {"hit": "#2a78d6", "wrong_target": "#e34948", "timeout": "#eda100", "collision": "#1f1f1e"}.get(r["outcome"], "#8a8984")
            ax.plot(path[:, 0], path[:, 1], color=color, lw=0.8, alpha=0.6)
            goal = np.array(r["targets"][r["goal"]])
            radius = r.get("target_radius", TaskConfig().target_radius)  # records before Oct 16 lack it
            ax.add_patch(plt.Circle(goal, radius, fill=False, edgecolor=INK_2, lw=0.4))
            for x, y, rad in r.get("obstacles", []):  # records before Oct 22 have no obstacles
                ax.add_patch(plt.Circle((x, y), rad, facecolor="#dedcd6", edgecolor="#a5a49e", lw=0.4, alpha=0.5))
        hits = sum(r["outcome"] == "hit" for r in arm_trials)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{arm}: {hits}/{len(arm_trials)} hits", fontsize=9, color=INK)
        ax.grid(False)
    fig.suptitle(f"{participant}, {runs[0]}: paths (blue hit, red wrong target, yellow timeout, black collision)", fontsize=9, color=INK)
    fig.savefig(out / f"paths_{participant}.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--participants", nargs="+", help="p1 p2 ... sim (default: all)")
    p.add_argument("--protocol", help="only runs of this protocol name")
    p.add_argument("--out", default="all", help="folder name under results/analysis/")
    args = p.parse_args()

    runs = find_runs(args.results, args.participants, args.protocol)
    if not runs:
        raise SystemExit("no runs found")
    trials, arms = load_trials(runs)
    rows = by_arm(trials)
    pool = pooled(rows, arms)
    out = Path(args.results) / "analysis" / args.out
    out.mkdir(parents=True, exist_ok=True)
    keep = ("participant", "run", "protocol", "block", "arm", "method", "trial", "layout_id", "outcome", "duration_s",
            "selected", "goal", "n_targets", "chunks", "steered_chunks", "t_start", "t_end")
    write_csv(out / "trials.csv", [{k: r.get(k) for k in keep} for r in trials])
    write_csv(out / "summary.csv", rows)
    write_summary_md(out / "summary.md", rows, pool)
    style()
    fig_success(rows, pool, arms, out)
    fig_learning(trials, arms, out)
    for participant in sorted({r["participant"] for r in trials}):
        fig_paths(trials, arms, participant, out)
    print((out / "summary.md").read_text())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
