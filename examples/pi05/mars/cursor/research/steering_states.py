"""Why steering fails through a retrained policy and not through the frozen policy with a task module: the
states each arm visits in closed loop, and each field's chance of colliding from them.

    python research/steering_states.py                        # U1xT4: gain (1, 0.25), the ring around the goal
    python research/steering_states.py --cell U3xT5 --trials 200

Runs the grid's U + T arm and its retrained arm closed loop on one cell, at the depths the grid tuned
(results/sim/sweeps/grid_s0/), recording the policy's observation and the served command at every replan of
every live trial (trials run in lockstep, and one that has ended keeps reporting its final position, so
those states are dropped). From every recorded state and command, both fields regenerate a chunk as the
controller would: the frozen policy with the user and task modules, and the retrained policy with its
user module. The share of chunks whose executed two steps run into a disc is the per-replan hazard that
compounds over a trial's replans. The demonstrations' states, with a synthetic command through the
user's channel, are the reference point. Writes results/research/steering_states/<cell>.json and log.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from argparse import Namespace

import numpy as np
import torch

import cursor_sa.steering as st
from cursor_sa.agents import make_agent
from cursor_sa.channel import matrix_spec
from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.env import obstacle_spec, sample_layouts
from cursor_sa.paths import RESULTS
from cursor_sa.records import logged
from cursor_sa.session import arm_adapter, arm_task, load_checkpoints
from cursor_sa.steering import ReversalAdapter, forward_flow, hide_goal, reverse_flow, to_action
from run_sweep import load_sweep, method_arm, run_cell_batch, with_depth
from train.train_adapter import collides, layouts_for

OUT = RESULTS / "research" / "steering_states"
ARMS = ("mars", "retrained_mars_user")
NEAR = 0.02  # a state this close to a disc is closer than the expert's route ever goes (its margin is 0.03)


def record_states(sweep, arm, method, layouts, depth, policies, adapters, device, seed):
    """The arm's trials, and at every replan of every live trial the policy's observation and the served command."""
    log, orig = [], st.SharedController.act

    def act(self, command, reference=None):
        if self.t >= self.cycle and self.mode != "teleop":
            obs = self.observation()
            log.append(({k: v.detach().cpu() for k, v in obs.items()}, torch.as_tensor(np.asarray(command), dtype=torch.float32)))
        return orig(self, command, reference)

    st.SharedController.act = act
    try:
        recs = run_cell_batch(sweep, with_depth(arm, depth), method, layouts, [arm["input_noise"]], make_agent(sweep["agent"]),
                              policies[arm["model"]], adapters, device, seed)[str(arm["input_noise"])]
    finally:
        st.SharedController.act = orig
    obs = {k: torch.stack([o[k] for o, _ in log]) for k in log[0][0]}  # [replans, trials, ...]
    cmd = torch.stack([c for _, c in log])
    alive = torch.zeros(cmd.shape[:2], dtype=torch.bool)             # the cursor still moves after this replan
    alive[:-1] = (obs["pos"][1:] != obs["pos"][:-1]).any(-1)
    keep = alive & (cmd.norm(dim=-1) > 0.05)                          # live, and pushing
    return recs, {k: v[keep].to(device) for k, v in obs.items()}, cmd[keep].to(device)


def demo_states(cell, policy, device, n_layouts=1000, n_states=4096, seed=777):
    """Held-out demonstration states of the cell's task (goal hidden, cursor travelling), with the expert's push
    through the cell's channel as the command."""
    rng = np.random.default_rng(seed)
    layouts, task = layouts_for(Namespace(targets=[4, 4], reach=[0.35, 0.5]), obstacle_spec(cell["obstacle"]), rng, n_layouts)
    actions, lengths = make_demos(rng, layouts, task, allow_failed=True)
    obs, label = DemoSampler(layouts, actions, lengths, policy.horizon, policy.hist, device).sample(n_states)
    obs = hide_goal(obs)
    keep = obs["obstacle_mask"].any(1) & (label.norm(dim=-1).sum(1) > 0.05)
    obs, label = {k: v[keep] for k, v in obs.items()}, label[keep]
    push = label[:, 0] / label[:, 0].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    corruption = cell["corruption"]
    if corruption and "curl" not in corruption:
        push = torch.clamp(push @ torch.tensor(matrix_spec(corruption), dtype=torch.float32, device=device).T, -1, 1)
    g = torch.Generator(device=device).manual_seed(seed)
    return obs, torch.clamp(push + torch.randn(push.shape, device=device, generator=g), -1, 1)


def clearance(obs):
    d = (obs["obstacles"][:, :, :2] - obs["pos"][:, None]).norm(dim=-1) - obs["obstacles"][:, :, 2]
    return torch.where(obs["obstacle_mask"], d, torch.full_like(d, 9.0)).min(1).values


@torch.no_grad()
def hazard(policy, adapter, obs, cmd, depth, steps=10):
    """From each state, the chunk MARS would execute for the held command at this depth (as frs_chunk): the
    share whose first two steps, and whose 16 steps, run into a disc."""
    ref = to_action(cmd[:, None].expand(-1, policy.horizon, -1), policy)
    forward = ReversalAdapter(task=adapter.task) if adapter is not None and adapter.task is not None else None
    z = reverse_flow(policy, obs, ref.clone(), depth, steps, adapter)
    out = forward_flow(policy, obs, z, 1.0 - depth / steps, 1.0, steps if depth == steps else steps - depth, forward)
    out = out * policy.action_scale
    return [100 * collides(obs, out[:, :2]).float().mean().item(), 100 * collides(obs, out).float().mean().item()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="U1xT4")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    with logged(OUT / "log.txt", append=(OUT / "log.txt").exists()):
        torch.manual_seed(0)
        sweep = load_sweep("grid")
        cell = sweep["grid"]["cells"][args.cell]
        grid = json.loads((RESULTS / "sim" / "sweeps" / "grid_s0" / "test.json").read_text())["test"]
        depths = grid[f"{args.cell}|{sweep['grid']['noises'][0]}"]["depths"]
        arms = {m: method_arm(sweep, m, args.cell) for m in ARMS}
        policies, adapters = load_checkpoints(arms, device)
        fields = {"frozen + user + task modules": (policies[arms[ARMS[0]]["model"]], arm_adapter(arms[ARMS[0]], adapters)),
                  "retrained policy + its user module": (policies[arms[ARMS[1]]["model"]], arm_adapter(arms[ARMS[1]], adapters))}
        layouts = sample_layouts(np.random.default_rng(args.seed), args.trials, tuple(arms[ARMS[0]]["targets"]), arm_task(arms[ARMS[0]]))
        out = {"cell": args.cell, "trials": args.trials, "depths": {m: depths[m] for m in ARMS}, "closed_loop": {}, "states": {}, "hazard": {}}
        print(f"{args.cell}: {args.trials} trials, depths {out['depths']}")
        sets = {}
        for m in ARMS:
            recs, obs, cmd = record_states(sweep, arms[m], m, layouts, depths[m], policies, adapters, device, args.seed + 1)
            outcomes = np.array([r["outcome"] for r in recs])
            out["closed_loop"][m] = {"success": float((outcomes == "hit").mean()), "collision": float((outcomes == "collision").mean()),
                                     "replans_per_trial": len(cmd) / args.trials}
            sets[f"states of {m}"] = (obs, cmd)
            print(f"  {m}: hit {100 * out['closed_loop'][m]['success']:.0f}%, collide {100 * out['closed_loop'][m]['collision']:.0f}%, "
                  f"{len(cmd)} live replan states ({len(cmd) / args.trials:.0f} per trial)")
        sets["demonstration states"] = demo_states(cell, policies[arms[ARMS[0]]["model"]], device)
        print(f"  clearance from the nearest disc (the expert keeps >= 0.03) and the last step's size:")
        for name, (obs, cmd) in sets.items():
            c, speed = clearance(obs), obs["hist"][:, -1].norm(dim=-1)
            out["states"][name] = {"n": len(c), "clearance_median": c.median().item(), f"share_within_{NEAR}": (c < NEAR).float().mean().item(),
                                   "step_median": speed.median().item(), "share_step_below_0.004": (speed < 0.004).float().mean().item()}
            s = out["states"][name]
            print(f"    {name:36s} n {s['n']:5d}  clearance median {s['clearance_median']:.3f}, within {NEAR}: "
                  f"{100 * s[f'share_within_{NEAR}']:.1f}%  | last step median {s['step_median']:.4f}")
        print(f"  per-replan hazard: the executed two steps collide / the whole chunk would (%), at the arms' tuned depths:")
        for name, (obs, cmd) in sets.items():
            out["hazard"][name] = {}
            for field, (policy, adapter) in fields.items():
                out["hazard"][name][field] = {str(d): hazard(policy, adapter, obs, cmd, d) for d in sorted({depths[m] for m in ARMS})}
                print(f"    {name:36s} {field:36s} " + "  ".join(f"depth {d}: {v[0]:4.1f} / {v[1]:4.1f}"
                                                                   for d, v in out["hazard"][name][field].items()))
        with open(OUT / f"{args.cell}.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {OUT / f'{args.cell}.json'}")


if __name__ == "__main__":
    main()
