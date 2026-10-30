"""What a goal-dropout policy does when no target is flagged as the goal.

    python research/evaluate_policy_goal_hidden.py --models flow_goaldrop regress_goaldrop

In the demos the goal is a uniformly random target at least min_reach from
the cursor start ("eligible"). So with no goal flag, the ideal policy picks one
eligible target at random and commits to it. Each layout is run --runs times to
see which targets the policy picks. Writes results/research/<out>/metrics.json and
figures to results/research/<out>/figs/ (--out, default policy_goal_hidden).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cursor_sa.analysis import INK, INK_2, choice_stats, draw_scene, style
from cursor_sa.env import TaskConfig, sample_layouts
from cursor_sa.paths import RESULTS, policy_path
from cursor_sa.policy import FlowPolicy
from cursor_sa.records import logged
from evaluate_policy_goal_given import COLORS, LABELS, rollout

HERE = Path(__file__).parent


def fig_rollouts(runs, test_sets, names, look, task, n_runs, out):
    counts = list(test_sets)
    fig, axes = plt.subplots(len(names), len(counts), figsize=(2.3 * len(counts), 2.3 * len(names)),
                             squeeze=False)
    for c, n in enumerate(counts):
        layout = test_sets[n][:1]
        for r, name in enumerate(names):
            env, path = runs[n][name]
            ax = axes[r, c]
            draw_scene(ax, layout, task, show_goal=False)
            ax.add_patch(plt.Circle(layout.start[0], task.min_reach, fill=False, ls="--",
                                    lw=0.8, color=INK_2, zorder=1))
            for b in range(n_runs):
                ax.plot(*path[b, :env.steps[b] + 1].T, color=look[name][0], lw=0.9, alpha=0.4, zorder=2)
            picks = np.bincount(env.selected[:n_runs][env.selected[:n_runs] >= 0], minlength=n)
            for j in np.flatnonzero(picks):
                ax.text(*layout.targets[0, j], str(picks[j]), ha="center", va="center", fontsize=7,
                        color=INK, zorder=5, fontweight="bold")
            ax.text(0.03, 0.03, f"{(env.selected[:n_runs] < 0).sum()}/{n_runs} no selection",
                    transform=ax.transAxes, color=INK_2, fontsize=8)
            if r == 0:
                ax.set_title(f"{n} targets", color=INK)
            if c == 0:
                ax.set_ylabel(look[name][1], color=INK)
    fig.suptitle(f"No goal flag: {n_runs} runs from the same start\n"
                 f"(numbers: runs that selected each target; dashed: never a goal inside this circle "
                 f"in training)", color=INK, y=1.02, fontsize=9)
    fig.savefig(out / "goal_free_rollouts.png")
    plt.close(fig)


def fig_summary(summary, names, look, out):
    counts = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for name in names:
        axes[0].plot(counts, [100 * summary[n][name]["selected_any"] for n in counts], "-o",
                     color=look[name][0], lw=2, ms=5, label=look[name][1])
        axes[1].plot(counts, [summary[n][name]["spread"] for n in counts], "-o",
                     color=look[name][0], lw=2, ms=5, label=look[name][1])
    axes[1].plot(counts, [summary[n][names[0]]["spread_ideal"] for n in counts], "--", color=INK_2,
                 lw=1.5, label="Ideal (uniform over eligible targets)")
    axes[0].set(xlabel="Targets on screen", ylabel="Runs that selected a target (%)", ylim=(0, 105),
                xticks=counts)
    axes[1].set(xlabel="Targets on screen", ylabel="Spread of choices\n(normalized entropy)",
                ylim=(0, 1.05), xticks=counts)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.16))
    fig.savefig(out / "goal_free_summary.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["flow_goaldrop", "regress_goaldrop"])
    p.add_argument("--layouts", type=int, default=200, help="test layouts per target count")
    p.add_argument("--runs", type=int, default=20, help="runs per layout")
    p.add_argument("--targets", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--execute", type=int, default=8)
    p.add_argument("--flow_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--out", default="policy_goal_hidden", help="output folder under results/research/")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RESULTS / "research" / args.out
    (out / "figs").mkdir(parents=True, exist_ok=True)
    with logged(out / "log.txt"):
        policies = {m: FlowPolicy.load(policy_path(m), device) for m in args.models}
        look = {m: (COLORS[pol.mode], LABELS.get(pol.mode, m)) for m, pol in policies.items()}
        names, task = list(policies), TaskConfig()

        test_sets, runs, summary = {}, {}, {}
        for n in args.targets:
            rng = np.random.default_rng(args.seed + n)
            layouts = test_sets[n] = sample_layouts(rng, args.layouts, n, task)
            rep = layouts[np.repeat(np.arange(args.layouts), args.runs)]
            runs[n], summary[n] = {}, {}
            for m, pol in policies.items():
                torch.manual_seed(args.seed + n)
                env, path = runs[n][m] = rollout(pol, rep, task, args.execute, args.flow_steps, device,
                                                 hide_goal=True)
                selected = env.selected.reshape(args.layouts, args.runs)
                summary[n][m] = choice_stats(selected, layouts, task, np.random.default_rng(n))
                done = env.selected >= 0
                summary[n][m]["time_s"] = float(np.median(env.steps[done] * task.dt)) if done.any() else None
            print(f"{n} targets: " + "  ".join(f"{m} selected {summary[n][m]['selected_any']:.1%}, "
                                               f"spread {summary[n][m]['spread']:.2f}" for m in names), flush=True)

        style()
        fig_rollouts(runs, test_sets, names, look, task, args.runs, out / "figs")
        fig_summary(summary, names, look, out / "figs")
        with open(out / "metrics.json", "w") as f:
            json.dump(dict(args=vars(args), summary={str(n): v for n, v in summary.items()}), f, indent=1)

        cols = ["selected_any", "no_selection", "ineligible", "spread", "spread_ideal", "time_s"]
        print(f"\n{'targets':>7} {'model':>17} " + " ".join(f"{c:>13}" for c in cols))
        for n in summary:
            for m in names:
                print(f"{n:>7} {m:>17} " + " ".join(f"{summary[n][m][c] if summary[n][m][c] is not None else float('nan'):>13.3f}"
                                                    for c in cols))


if __name__ == "__main__":
    main()
