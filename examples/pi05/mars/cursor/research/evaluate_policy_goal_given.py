"""Closed-loop evaluation of trained policies, with the demo generator as reference.

    python research/evaluate_policy_goal_given.py
    python research/evaluate_policy_goal_given.py --models flow --trials 200 --targets 1 4 16

For each target count, the same test layouts are run by the demo generator
(open loop) and by each policy (closed loop, replanning every --execute steps).
Writes results/research/<out>/metrics.json and figures to results/research/<out>/figs/
(--out, default policy_goal_given).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cursor_sa.analysis import INK, INK_2, draw_scene, hold_last, style, trial_metrics
from cursor_sa.demos import make_demos, replay
from cursor_sa.env import CursorEnv, TaskConfig, sample_layouts
from cursor_sa.paths import RESULTS, policy_log, policy_path
from cursor_sa.policy import FlowPolicy
from cursor_sa.records import logged
from cursor_sa.steering import observation

HERE = Path(__file__).parent
COLORS = {"demos": "#8a8984", "flow": "#2a78d6", "regress": "#eb6834"}
LABELS = {"demos": "Demos", "flow": "Flow policy", "regress": "Regression"}


def rollout(policy, layouts, task, execute=8, flow_steps=10, device="cpu", hide_goal=False, zero_hist=False):
    """Run the policy alone, closed loop, replanning every `execute` steps. hide_goal=True
    clears every goal flag, so the policy is not told which target is the goal;
    zero_hist=True hides its velocity history (a memory ablation).
    Returns the finished env and cursor paths [B, max_steps + 1, 2]."""
    env = CursorEnv(layouts, task, np.random.default_rng(0))
    hist = np.zeros((len(layouts), policy.hist, 2))
    path = [env.pos.copy()]
    while not env.done.all():
        k = env.t % execute
        if k == 0:
            obs = observation(layouts, env.pos, np.zeros_like(hist) if zero_hist else hist, device, not hide_goal)
            chunk = policy.sample(obs, steps=flow_steps).cpu().numpy()
        prev = env.pos.copy()
        env.step(chunk[:, k])
        hist = np.concatenate([hist[:, 1:], (env.pos - prev)[:, None]], 1)
        path.append(env.pos.copy())
    return env, hold_last(np.stack(path, 1), task.max_steps + 1)


def demo_rollout(rng, layouts, task):
    actions, _ = make_demos(rng, layouts, task)
    path = layouts.start[:, None] + np.cumsum(np.concatenate([np.zeros_like(actions[:, :1]), actions], 1), 1)
    return replay(actions, layouts, task), hold_last(path, task.max_steps + 1)


def summarize(m):
    ok = m["success"]
    med = lambda x: float(np.median(x[ok])) if ok.any() else float("nan")
    return dict(success=float(ok.mean()), wrong_target=float(m["wrong"].mean()),
                timeout=float(m["timeout"].mean()), time_s=med(m["time_s"]),
                path_efficiency=med(m["efficiency"]), bow_sd=float(m["bow"][ok].std()),
                mean_accel=med(m["accel"]))


def fig_rollouts(policies, test_sets, task, args, device, look, out):
    """Many runs from the same start, per method, across target counts."""
    counts = [n for n in (1, 4, 8, 16) if n in test_sets] or list(test_sets)[:4]
    names = ["demos", *policies]
    fig, axes = plt.subplots(len(names), len(counts), figsize=(2.3 * len(counts), 2.3 * len(names)),
                             squeeze=False)
    for c, n in enumerate(counts):
        layout = test_sets[n][:1]
        rep = layout[np.zeros(args.fan, dtype=int)]
        for r, name in enumerate(names):
            if name == "demos":
                env, path = demo_rollout(np.random.default_rng(c), rep, task)
            else:
                torch.manual_seed(c)
                env, path = rollout(policies[name], rep, task, args.execute, args.flow_steps, device)
            ax = axes[r, c]
            draw_scene(ax, layout, task)
            for b in range(args.fan):
                ax.plot(*path[b, :env.steps[b] + 1].T, color=look[name][0], lw=0.9, alpha=0.35, zorder=2)
            ax.text(0.03, 0.03, f"{env.success.mean():.0%} success", transform=ax.transAxes,
                    color=INK_2, fontsize=8)
            if r == 0:
                ax.set_title(f"{n} target{'s' * (n > 1)}", color=INK)
            if c == 0:
                ax.set_ylabel(look[name][1], color=INK)
    fig.suptitle(f"{args.fan} runs from the same start (goal outlined)", color=INK, y=1.0)
    fig.savefig(out / "rollouts.png")
    plt.close(fig)


def fig_by_target_count(summary, names, train_range, look, out):
    counts = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, key, label, scale in [(axes[0], "success", "Success rate (%)", 100),
                                  (axes[1], "time_s", "Median time to select goal (s)", 1)]:
        ax.axvspan(*train_range, color="#f3f2ed", zorder=0)
        for name in names:
            y = [summary[n][name][key] * scale for n in counts]
            ax.plot(counts, y, "-o", color=look[name][0], lw=2, ms=5, label=look[name][1])
        ax.set(xlabel="Targets on screen", ylabel=label, xticks=counts)
    axes[0].set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#f3f2ed"))
    labels.append(f"Training range ({train_range[0]}-{train_range[1]} targets)")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.16))
    fig.savefig(out / "by_target_count.png")
    plt.close(fig)


def fig_kinematics(pooled, names, look, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    bins = np.linspace(-0.4, 0.4, 33)
    for name in names:
        m = pooled[name]
        ok = m["success"]
        axes[0].hist(m["bow"][ok], bins, histtype="step", lw=2, color=look[name][0], density=True,
                     label=f"{look[name][1]} (sd {m['bow'][ok].std():.2f})")
        p = m["profile"][ok]
        x = np.linspace(0, 1, p.shape[1])
        lo, mid, hi = np.percentile(p, [25, 50, 75], axis=0)
        axes[1].fill_between(x, lo, hi, color=look[name][0], alpha=0.15, lw=0)
        axes[1].plot(x, mid, color=look[name][0], lw=2, label=look[name][1])
    axes[0].set(xlabel="Peak sideways deviation / reach distance", ylabel="Density",
                title="Path shape (successful trials)")
    axes[1].set(xlabel="Time, start to goal entry (normalized)", ylabel="Speed (screens/s)",
                title="Speed profile (median, IQR)")
    axes[0].set_ylim(0, 1.6 * axes[0].get_ylim()[1])  # headroom for the legend
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(out / "kinematics.png")
    plt.close(fig)


def fig_training(policy_names, look, out):
    fig, axes = plt.subplots(1, len(policy_names), figsize=(4 * len(policy_names), 2.8), squeeze=False)
    for ax, name in zip(axes[0], policy_names):
        log = json.loads(policy_log(name).read_text())["log"]
        step = [e["step"] for e in log]
        ax.plot(step, [e["train"] for e in log], color=look[name][0], lw=2, label="train")
        ax.plot(step, [e["val"] for e in log], color=look[name][0], lw=2, ls="--", label="val (EMA)")
        ax.set(yscale="log", xlabel="Step", ylabel="Loss", title=f"{look[name][1]} loss")
        ax.legend()
    fig.savefig(out / "training.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["flow", "regress"])
    p.add_argument("--trials", type=int, default=500, help="test layouts per target count")
    p.add_argument("--targets", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    p.add_argument("--execute", type=int, default=8, help="chunk steps executed before replanning")
    p.add_argument("--flow_steps", type=int, default=10, help="Euler steps per sampled chunk")
    p.add_argument("--fan", type=int, default=30, help="runs per layout in the rollout figure")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="policy_goal_given", help="output folder under results/research/")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RESULTS / "research" / args.out
    (out / "figs").mkdir(parents=True, exist_ok=True)
    with logged(out / "log.txt"):
        policies = {m: FlowPolicy.load(policy_path(m), device) for m in args.models}
        train_range = json.loads(policy_log(args.models[0])
                                 .read_text())["args"]["targets"]
        names = ["demos", *policies]
        look = {"demos": (COLORS["demos"], LABELS["demos"]),
                **{m: (COLORS[pol.mode], LABELS.get(m, m)) for m, pol in policies.items()}}
        task = TaskConfig()

        test_sets, summary, pooled = {}, {}, {n: [] for n in names}
        for n in args.targets:
            rng = np.random.default_rng(args.seed + n)
            layouts = test_sets[n] = sample_layouts(rng, args.trials, n, task)
            torch.manual_seed(args.seed + n)
            runs = {"demos": demo_rollout(rng, layouts, task)}
            runs.update({m: rollout(pol, layouts, task, args.execute, args.flow_steps, device)
                         for m, pol in policies.items()})
            metrics = {m: trial_metrics(*run, layouts, task) for m, run in runs.items()}
            summary[n] = {m: summarize(v) for m, v in metrics.items()}
            if train_range[0] <= n <= train_range[1]:
                for m in names:
                    pooled[m].append(metrics[m])
            print(f"{n:2d} targets: " + "  ".join(f"{m} {summary[n][m]['success']:.1%}" for m in names), flush=True)

        pooled = {m: {k: np.concatenate([d[k] for d in v]) for k in v[0]} for m, v in pooled.items()}
        style()
        fig_rollouts(policies, test_sets, task, args, device, look, out / "figs")
        fig_by_target_count(summary, names, train_range, look, out / "figs")
        fig_kinematics(pooled, names, look, out / "figs")
        fig_training(list(policies), look, out / "figs")

        with open(out / "metrics.json", "w") as f:
            json.dump(dict(args=vars(args), summary={str(n): v for n, v in summary.items()}), f, indent=1)

        cols = ["success", "wrong_target", "timeout", "time_s", "path_efficiency", "bow_sd", "mean_accel"]
        print(f"\n{'targets':>7} {'method':>8} " + " ".join(f"{c:>15}" for c in cols))
        for n in summary:
            for m in names:
                print(f"{n:>7} {m:>8} " + " ".join(f"{summary[n][m][c]:>15.3f}" for c in cols))


if __name__ == "__main__":
    main()
