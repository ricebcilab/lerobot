"""Train one module of the reversal adapter: a user module for a user's command corruption, or a task
module for an obstacle task with a user's frozen user module in the loop.

    python train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' --name user_gain025y_expert
    python train/train_adapter.py --kind task --obstacle '{kind: ring, around: goal, pass: left}' \
        --user-module user_rot60_expert --corruption '{rotation_deg: 60}' --name task_ring_goal_with_rot60

USER MODULE: open-loop calibration with an expert that knows the goal. On demonstrations of the open-field
task, the label is the demonstration's next chunk L, and the command is what the expert
(`agents.ExpertAgent`) pushes at that moment toward the goal, through the user's channel (corruption M or
curl, then noise of std sigma), held over the chunk as the controller holds it: a_bci. The user module
U(x_t, t) acts on the backward pass; the field objective pulls it toward the label's latent z_L (L reversed
through the policy's flow), along the straight path from z_L to a_bci:

    U(x_t, t) P(x_t, t) = a_bci - z_L,   x_t = (1 - t) z_L + t a_bci

so a full reversal of a_bci lands on z_L, which regenerates L. Two knobs set the module's flexibility (UserModule):
`--step-rank r` builds every step's matrix from r matrices shared by the chunk, and `--cross-step-rank q` adds a
rank-q coupling between the chunk's steps (`--cross-step-dims 0` or `0,1` picks the dims it couples; default both);
the defaults (every step's matrix free, no coupling) are the grid's. They are saved in the module's config.

CLOSED-LOOP RECALIBRATION (ReFIT-style, `--closed-loop <initial module>`): expert demonstrations always head
for the goal, so the calibration above never shows the off-course states closed loop produces (and the module
trusts the policy's course there). The recalibration runs the initial module in closed loop on the open-field
task (the expert pushing through the user's channel at `--closed-loop-noise`, the grid's controller), keeps
every visited state with the expert's push there (what the user meant: toward the goal), relabels the state
with the reach the demonstrator would make from it (a straight min-jerk reach to the goal, its first chunk),
and trains a fresh module on batches mixing the open-loop data with these (`--closed-loop-mix`). Several
initial modules pool their trials (DAgger-style rounds: the first module's states and the recalibrated one's):

    python train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' --name user_gain025y_refit \
        --closed-loop user_gain025y_expert --closed-loop-depth 6

TASK MODULE: the policy is frozen and never saw obstacles. On demonstrations that go around the task's
obstacles (along the expert's path), the task module T = (A, b) is trained by flow matching through the
adapted field, on the policy's own interpolation x_t = (1 - t) z + t a,

    L_fm = || A(x_t, t, v, obstacles) v + b(...) - (a - z) ||^2 + lambda_id (||A - I||^2 + ||b||^2),   v = P(x_t, t | o)

with the policy's goal dropout, so the adapted field reproduces the detours with the goal given and
hidden alike; a share `p_free` of the layouts has no obstacle, where the module should be the identity.
The user's frozen module is in the loop through lambda_field x the field regression above through both
modules, on that user's commands around the obstacles (the layouts with the obstacle only):

    U(x_t, t) (A v + b) = a_bci - z_L,   z_L = L reversed through the task-adapted flow.

DATA: `--n-demos` demonstrations (one reach each) train a module; the default 20000 is as many as the base
policy's own. Every batch draws the channel's noise anew, so even a few demonstrations give endless noisy
commands; `--frozen-noise` draws it once per demonstration step instead, as a recorded calibration has one
command per step (the validation split keeps fresh draws).

Saves checkpoints/adapters/<name>/{adapter.pt, train_log.json}; adapter.pt holds kind, config,
state_dict, args and the obstacle spec (None for a user module). The trainer up to git commit 5c2167cd
also trained user modules from the policy's own chunks (grid 1 and the study's gain025y_free) and task
modules without a user.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cursor_sa import config  # noqa: E402
from cursor_sa.agents import ExpertAgent  # noqa: E402
from cursor_sa.channel import DEADBAND, corruption_spec, curl_commands, serve_command  # noqa: E402
from cursor_sa.demos import DemoConfig, DemoSampler, make_demos, plan_reach  # noqa: E402
from cursor_sa.env import Layouts, TaskConfig, obstacle_spec, sample_layouts  # noqa: E402
from cursor_sa.paths import adapter_dir, adapter_path, policy_path  # noqa: E402
from cursor_sa.policy import FlowPolicy  # noqa: E402
from cursor_sa.session import arm_task, channel_for, run_trials  # noqa: E402
from cursor_sa.steering import (ReversalAdapter, TaskModule, UserModule, forward_flow, hide_goal,  # noqa: E402
                                reverse_flow, to_action)

from lerobot.policies.pi05.steering import task_field_loss, user_field_loss  # noqa: E402

# ---------------------------------------------------------------- data


def layouts_for(args, obstacle, rng, n, p_free=0.0):
    """n layouts of the task (a share p_free of them without its obstacle)."""
    task = TaskConfig(min_reach=args.reach[0], max_reach=args.reach[1], obstacle=obstacle)
    n_free = int(round(p_free * n)) if obstacle is not None else 0
    parts = [sample_layouts(rng, n - n_free, tuple(args.targets), task)] if n - n_free else []
    if n_free:
        parts.append(sample_layouts(rng, n_free, tuple(args.targets), replace(task, obstacle=None)))
    return (Layouts.concat(parts) if len(parts) > 1 else parts[0]), task


def expert_pushes(layouts, pos, rows_per_call=500):
    """What the expert (`agents.ExpertAgent`) pushes at every demo step [B, L, 2], watching the
    demonstration's positions pos [B, L + 1, 2]: toward the goal, around the task's obstacles."""
    B, L = pos.shape[0], pos.shape[1] - 1
    out = np.zeros((B, L, 2))
    for rows in np.array_split(np.arange(B), max(1, int(np.ceil(B / rows_per_call)))):
        sub, agent = layouts[rows], ExpertAgent()
        agent.reset(len(rows))
        for k in range(L):
            out[rows, k] = agent.command(pos[rows, k], sub)
    return out


def channel_core(push, corruption, curl):
    """What the user's channel makes of the pushes [B, L, 2] before the noise: M @ push clipped, or a curl's
    velocity driven from rest by the pushes in turn."""
    if curl is not None:
        return curl_commands(push, curl)
    return push if corruption is None else np.clip(push @ np.asarray(corruption).T, -1.0, 1.0)


class ExpertData:
    """One split of expert calibration data: the demonstrations (a DemoSampler) and, at every demo step,
    the expert's push and what the user's channel makes of it before noise."""

    def __init__(self, layouts, actions, lengths, policy, device, corruption, curl, frozen_noise=False):
        self.sampler = DemoSampler(layouts, actions, lengths, policy.horizon, policy.hist, device)
        pos = layouts.start[:, None] + np.concatenate([np.zeros((len(actions), 1, 2)), np.cumsum(actions, 1)], 1)
        push = expert_pushes(layouts, pos)
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.push, self.core = f32(push), f32(channel_core(push, corruption, curl))
        self.noise = torch.randn_like(self.core) if frozen_noise else None  # one draw per demo step, or fresh ones

    def noise_at(self, i, k):
        """The channel's noise draws at demo steps (i, k) [B, 2] when frozen, else None (fresh draws)."""
        return None if self.noise is None else self.noise[i, k]

    def sample(self, n, goal_dropout=0.0):
        """(obs, chunk, push, core, i, k): demo samples and the expert's push and its channel core there."""
        obs, chunk, i, k = self.sampler.sample(n, goal_dropout, indexed=True)
        return obs, chunk, self.push[i, k], self.core[i, k], i, k


def expert_splits(args, obstacle, policy, device, rng, corruption, curl, p_free=0.0):
    """ExpertData for the train and val splits of a task."""
    out = {}
    for split, n in [("train", args.n_demos), ("val", args.val_demos)]:
        layouts, task = layouts_for(args, obstacle, rng, n, p_free)
        tic = time.time()
        actions, lengths = make_demos(rng, layouts, task, allow_failed=True)
        out[split] = ExpertData(layouts, actions, lengths, policy, device, corruption, curl,
                                frozen_noise=split == "train" and args.frozen_noise)
        print(f"{split}: {n} demonstrations and the expert's pushes ({time.time() - tic:.0f}s)", flush=True)
    return out


STRAIGHT = DemoConfig(max_bow=0.0, p_correct=0.0)   # the relabel: a straight min-jerk reach to the goal


class ClosedLoopData:
    """One split of closed-loop recalibration data: the states a controller visited with the user's channel in
    the loop (trajectories as a DemoSampler, so an observation carries the real history), at every step the
    expert's push there and what the channel makes of it before noise, and as the label the first chunk of a
    straight min-jerk reach from that state to the goal (what the demonstrator would do from there)."""

    def __init__(self, layouts, actions, lengths, raw, policy, device, corruption, curl, rng, task):
        self.sampler = DemoSampler(layouts, actions, lengths, policy.horizon, policy.hist, device)
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.push, self.core = f32(raw), f32(channel_core(raw, corruption, curl))
        H = policy.horizon
        pos = layouts.start[:, None] + np.concatenate([np.zeros((len(actions), 1, 2)), np.cumsum(actions, 1)], 1)
        labels = np.zeros((len(actions), actions.shape[1], H, 2))
        for i in range(len(actions)):
            goal = layouts.goal_pos[i]
            for k in range(int(lengths[i])):
                if np.linalg.norm(goal - pos[i, k]) < 1e-6:
                    continue                                         # on the goal's centre: hold
                steps = np.diff(plan_reach(rng, pos[i, k], goal, task, STRAIGHT), axis=0)[:H]
                labels[i, k, :len(steps)] = steps
        self.labels = f32(labels)

    def sample(self, n, goal_dropout=0.0):
        """(obs, label chunk, push, core, i, k), as ExpertData.sample."""
        obs, _, i, k = self.sampler.sample(n, goal_dropout, indexed=True)
        return obs, self.labels[i, k], self.push[i, k], self.core[i, k], i, k


def closed_loop_splits(args, policy, model, device, rng):
    """ClosedLoopData for the train and val splits (a tenth of every module's trials): the open-field task under
    `--closed-loop-arm` with each initial module of `--closed-loop` in the reversal at `--closed-loop-depth`, the
    user's corruption and `--closed-loop-noise`, the expert pushing at the goal. Prints how those trials went."""
    corruption, curl = corruption_spec(args.corruption)
    parts, records, val = [], [], []
    for name in args.closed_loop:
        arm = config.load_arm(args.closed_loop_arm, dict(
            model=model, mode="frs", reversal_adapter={"user": name, "task": None}, n_reverse_steps=args.closed_loop_depth,
            corruption=args.corruption, input_noise=args.closed_loop_noise, obstacle=None, time_limit_s=args.closed_loop_time_limit,
            targets=list(args.targets), reach=list(args.reach)))
        adapters, task = {name: ReversalAdapter.load(adapter_path(name), device)}, arm_task(arm)
        tic, first = time.time(), len(records)
        for start in range(0, args.closed_loop_trials, 500):
            layouts = sample_layouts(rng, min(500, args.closed_loop_trials - start), tuple(args.targets), task)
            parts.append(layouts)
            records += run_trials(arm, layouts, ExpertAgent(), channel_for(arm, rng), policy, adapters, device)
        mine = records[first:]
        rate = lambda o: np.mean([r["outcome"] == o for r in mine])
        print(f"closed loop with {name} at depth {args.closed_loop_depth}, noise {args.closed_loop_noise}: {len(mine)} trials, "
              f"success {100 * rate('hit'):.1f}%, wrong target {100 * rate('wrong_target'):.1f}%, timeout {100 * rate('timeout'):.1f}%, "
              f"{sum(len(r['raw_commands']) for r in mine)} states ({time.time() - tic:.0f}s)", flush=True)
        val += list(range(len(records) - max(1, len(mine) // 10), len(records)))   # the last tenth of each module's trials
    layouts = Layouts.concat(parts) if len(parts) > 1 else parts[0]
    lengths = np.array([len(r["raw_commands"]) for r in records])
    actions, raw = np.zeros((len(records), lengths.max(), 2)), np.zeros((len(records), lengths.max(), 2))
    for b, r in enumerate(records):
        actions[b, :lengths[b]], raw[b, :lengths[b]] = np.diff(np.asarray(r["path"]), axis=0), np.asarray(r["raw_commands"])
    train = np.setdiff1d(np.arange(len(records)), val)
    split = lambda idx: ClosedLoopData(layouts[idx], actions[idx], lengths[idx], raw[idx], policy, device, corruption, curl, rng, task)
    return {"train": split(train), "val": split(np.array(val))}


def mixed_batch(open_data, closed_data, batch, mix):
    """A batch with a share `mix` of closed-loop samples: (obs, chunk, push, core)."""
    n_c = int(round(mix * batch))
    a, b = open_data.sample(batch - n_c), closed_data.sample(n_c)
    return ({k: torch.cat([a[0][k], b[0][k]]) for k in a[0]},) + tuple(torch.cat([x, y]) for x, y in zip(a[1:4], b[1:4]))


def bci_command(policy, push, core, sigma, noise=None):
    """The controller's reference for one served command per sample: noise on the channel core (where the
    push is active; `noise` the standard normal draws, default fresh), held over the chunk, in the policy's
    units [B, H, 2]."""
    served = serve_command(core, push, sigma, noise)
    return to_action(served[:, None].expand(-1, policy.horizon, -1), policy)


def draw_sigma(args, n, device):
    return args.sigma_min + (args.sigma_max - args.sigma_min) * torch.rand(n, device=device)


def active_of(push):
    return push.abs().amax(-1) >= DEADBAND


# ---------------------------------------------------------------- losses and checks


def task_loss(policy, task, obs, chunk, z, t, lambda_id):
    """Flow matching through the adapted field on the policy's interpolation x_t = (1 - t) z + t a (the cursor's
    time convention, t = 0 noise; a the chunk in normalized units), plus the pull of A, b toward the identity.
    Returns the loss and its two parts. The core's `task_field_loss` on this policy and observation."""
    return task_field_loss(task.bind(obs), lambda x, t_: policy(obs, x, t_), chunk / policy.action_scale, z, t,
                           lambda_id, n_dims=2)


def field_loss(policy, user, task, free, latent, bci, weight=None):
    """The regression along the straight path from `latent` to a_bci, x_t = (1 - t) latent + t a_bci (the cursor's
    time convention, t = 0 noise; t drawn uniformly here): U(x_t, t) T(v) = a_bci - latent, v the policy's field
    and T the task module (A v + b) if any. Gradients reach whichever module trains. `weight` [B] (e.g. the active
    commands) averages the per-sample errors. The core's `user_field_loss` on this policy and observation."""
    t = torch.rand(len(latent), device=latent.device)
    return user_field_loss(user, lambda x, t_: policy(free, x, t_), latent, bci, t,
                           None if task is None else task.bind(free), weight)


def collides(obs, chunk):
    """Per sample: whether the chunk (screen units [B, H, 2]) from the cursor runs into one of its discs."""
    pts = torch.cat([obs["pos"][:, None], obs["pos"][:, None] + torch.cumsum(chunk, 1)], 1)   # [B, H + 1, 2]
    p0, p1 = pts[:, :-1, None], pts[:, 1:, None]                                            # [B, H, 1, 2]
    c, r = obs["obstacles"][:, None, :, :2], obs["obstacles"][:, None, :, 2]                 # [B, 1, K, 2], [B, 1, K]
    d = p1 - p0
    tau = (((c - p0) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    dist = (c - (p0 + tau[..., None] * d)).norm(dim=-1)                                      # [B, H, K]
    return ((dist < r) & obs["obstacle_mask"][:, None]).any(-1).any(-1)


@torch.no_grad()
def chunk_collision(policy, adapter, obs, steps):
    """Share of the obstacle trials among `obs` whose chunk, sampled from the (adapted) forward flow,
    runs into a disc (every step's segment tested). NaN when no trial has an obstacle."""
    rows = obs["obstacle_mask"].any(1)
    if not rows.any():
        return float("nan")
    obs = {k: v[rows] for k, v in obs.items()}
    z = torch.randn(len(obs["pos"]), policy.horizon, 2, device=obs["pos"].device)
    return collides(obs, forward_flow(policy, obs, z, 0.0, 1.0, steps, adapter) * policy.action_scale).float().mean().item()


@torch.no_grad()
def regenerate(policy, free, bci, steps, user=None, task=None):
    """FRS at full reversal: a_bci reversed through the backward field (user and task modules), regenerated
    through the forward field (task module)."""
    back = ReversalAdapter(user=user, task=task) if user is not None or task is not None else None
    z = reverse_flow(policy, free, bci, steps, steps, back)
    return forward_flow(policy, free, z, 0.0, 1.0, steps, ReversalAdapter(task=task) if task is not None else None)


@torch.no_grad()
def expert_check(policy, data, args, user=None, task=None, n=2048):
    """Open loop on held-out demo states, at full reversal: how well the regenerated chunk heads where the
    label does (cosine of their displacements, over active commands), with no module, the user module, and
    the task module with it (and how often the regenerated chunks collide)."""
    obs, chunk, push, core, _, _ = data.sample(n)
    free, active = hide_goal(obs), active_of(push)
    bci = bci_command(policy, push, core, draw_sigma(args, n, push.device))
    label = chunk.sum(1)
    out = {}
    variants = [("none", None, None), ("user", user, None)] + ([("composed", user, task)] if task is not None else [])
    for name, u, tm in variants:
        regen = regenerate(policy, free, bci, args.flow_steps, u, tm) * policy.action_scale
        move = regen.sum(1)
        cos = nn.functional.cosine_similarity(move, label, dim=-1)
        out[f"heading_cos_{name}"] = float(cos[active].mean())
        if obs["obstacle_mask"].any():
            rows = active & obs["obstacle_mask"].any(1)
            out[f"collide_{name}"] = float(collides({k_: v[rows] for k_, v in obs.items()}, regen[rows]).float().mean())
    return out


# ---------------------------------------------------------------- the task module


def train_task(args, policy, device, rng):
    user = ReversalAdapter.load(adapter_path(args.user_module), device).user.requires_grad_(False)
    data = expert_splits(args, args.obstacle, policy, device, rng, *corruption_spec(args.corruption), args.p_free)
    samplers = {split: d.sampler for split, d in data.items()}
    task = TaskModule(policy.horizon, d=args.d, layers=args.layers).to(device)
    opt = torch.optim.AdamW(task.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    adapted = ReversalAdapter(task=task)
    log, running, skipped, tic = [], 0.0, 0, time.time()
    for step in range(1, args.steps + 1):
        obs, chunk, push, core, i, k = data["train"].sample(args.batch, args.goal_dropout)
        z, t = torch.randn_like(chunk), torch.rand(args.batch, device=device)
        loss, _ = task_loss(policy, task, obs, chunk, z, t, args.lambda_id)
        if args.lambda_field:  # the user's commands, through the frozen user module and this task module
            free = hide_goal(obs)
            latent = reverse_flow(policy, free, chunk / policy.action_scale, args.flow_steps, args.flow_steps, adapted)
            bci = bci_command(policy, push, core, draw_sigma(args, args.batch, device), data["train"].noise_at(i, k))
            weight = (active_of(push) & obs["obstacle_mask"].any(1)).float()  # on the obstacle-free layouts T stays I
            loss = loss + args.lambda_field * field_loss(policy, user, task, free, latent, bci, weight)
        skipped += not step_if_finite(opt, loss, task.parameters())
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % args.log_every == 0:
            task.eval()
            with torch.no_grad():
                n = min(2048, 8 * args.batch)
                obs, chunk = samplers["val"].sample(n, args.goal_dropout)
                _, parts = task_loss(policy, task, obs, chunk, torch.randn_like(chunk), torch.rand(n, device=device), 0.0)
                hidden, _ = samplers["val"].sample(n, 1.0)
                given, _ = samplers["val"].sample(n, 0.0)
                val = dict(val_fm=parts["fm"], chunk_collision_hidden=chunk_collision(policy, adapted, hidden, args.flow_steps),
                           chunk_collision_given=chunk_collision(policy, adapted, given, args.flow_steps),
                           chunk_collision_policy_hidden=chunk_collision(policy, None, hidden, args.flow_steps),
                           chunk_collision_policy_given=chunk_collision(policy, None, given, args.flow_steps))
                val.update(expert_check(policy, data["val"], args, user, task))
            task.train()
            log.append(dict(step=step, train=running / args.log_every, skipped=skipped, **val))
            print(f"step {step:6d}  train {running / args.log_every:.4f}  val fm {val['val_fm']:.4f}  chunk collisions: "
                  f"policy {val['chunk_collision_policy_hidden']:.3f} -> adapted {val['chunk_collision_hidden']:.3f} (goal hidden), "
                  f"{val['chunk_collision_policy_given']:.3f} -> {val['chunk_collision_given']:.3f} (given)"
                  f"  regenerated from the user's commands: collide user {val['collide_user']:.3f} -> composed "
                  f"{val['collide_composed']:.3f}, heading cos {val['heading_cos_user']:.3f} -> {val['heading_cos_composed']:.3f}"
                  + (f"  skipped {skipped}" if skipped else "") + f"  ({time.time() - tic:.0f}s)", flush=True)
            running = 0.0
    save(args, "task", task, log)


# ---------------------------------------------------------------- the user module


def train_user(args, policy, model, device, rng):
    """User module from expert labels on the open-field task, with closed-loop recalibration data mixed in
    when `--closed-loop` names an initial module (see the module docstring)."""
    corruption, curl = corruption_spec(args.corruption)
    data = expert_splits(args, None, policy, device, rng, corruption, curl)
    closed = closed_loop_splits(args, policy, model, device, np.random.default_rng(args.seed + 1)) if args.closed_loop else None
    user = UserModule(policy.horizon, step_rank=args.step_rank, cross_step_rank=args.cross_step_rank,
                      cross_step_dims=args.cross_step_dims).to(device)
    opt = torch.optim.AdamW(user.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    log, running, skipped, tic = [], 0.0, 0, time.time()
    for step in range(1, args.steps + 1):
        noise = None
        if closed is None:
            obs, chunk, push, core, i, k = data["train"].sample(args.batch)
            noise = data["train"].noise_at(i, k)
        else:
            obs, chunk, push, core = mixed_batch(data["train"], closed["train"], args.batch, args.closed_loop_mix)
        free = hide_goal(obs)
        latent = reverse_flow(policy, free, chunk / policy.action_scale, args.flow_steps, args.flow_steps)
        bci = bci_command(policy, push, core, draw_sigma(args, args.batch, device), noise)
        loss = field_loss(policy, user, None, free, latent, bci, active_of(push).float())
        skipped += not step_if_finite(opt, loss, user.parameters())
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % args.log_every == 0:
            user.eval()
            val = expert_check(policy, data["val"], args, user)
            if closed is not None:
                val.update({f"closed_{k}": v for k, v in expert_check(policy, closed["val"], args, user).items()})
            user.train()
            log.append(dict(step=step, train=running / args.log_every, skipped=skipped, **val))
            closed_text = (f"  closed loop: none {val['closed_heading_cos_none']:.3f}  U {val['closed_heading_cos_user']:.3f}"
                           if closed is not None else "")
            print(f"step {step:6d}  train {running / args.log_every:.4f}  regenerated heading cos: none "
                  f"{val['heading_cos_none']:.3f}  U {val['heading_cos_user']:.3f}" + closed_text
                  + (f"  skipped {skipped}" if skipped else "") + f"  ({time.time() - tic:.0f}s)", flush=True)
            running = 0.0
    save(args, "user", user, log)


# ---------------------------------------------------------------- main


def step_if_finite(opt, loss, params):
    """One optimizer step, unless the loss or the clipped gradient is not finite (a batch where a flow
    blew up): then the step is skipped and the weights stay whole. Returns whether a step was taken."""
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        return False
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(list(params), 1.0)
    if not torch.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def save(args, kind, module, log):
    out = adapter_dir(args.name)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=kind, config=module.config, state_dict=module.state_dict(), args=vars(args), obstacle=args.obstacle),
               out / "adapter.pt")
    with open(out / "train_log.json", "w") as f:
        json.dump(dict(args=vars(args), log=log), f, indent=1)
    print(f"saved {out / 'adapter.pt'}")


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = args.model or config.load_arm("base.yaml")["model"]
    policy = FlowPolicy.load(policy_path(model), device)
    policy.requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_task(args, policy, device, rng) if args.kind == "task" else train_user(args, policy, model, device, rng)


def dim_indices(text):
    """A comma list of dim indices ('0,1') as a list of ints."""
    try:
        return [int(i) for i in text.split(",") if i.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a comma list of dim indices, got {text!r}") from None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["user", "task"], default="user")
    p.add_argument("--name", required=True)
    p.add_argument("--model", help="policy under checkpoints/policies/ (default: base.yaml's)")
    p.add_argument("--corruption", type=yaml.safe_load, required=True,
                   help="the user's corruption, e.g. '{rotation_deg: 60}' or '{curl: 20, tau: 0.1}'")
    p.add_argument("--obstacle", type=yaml.safe_load, help="task: the task's obstacle spec")
    p.add_argument("--user-module", help="task: the user's frozen user module, trained with that --corruption")
    p.add_argument("--lambda-field", type=float, default=0.03, help="task: weight of the field loss through the user module")
    p.add_argument("--sigma-min", type=float, default=1.0, help="noise std of the commands is drawn from [sigma_min, sigma_max]")
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--closed-loop", nargs="+", help="user: recalibrate on closed-loop trials of these initial user modules "
                   "(ReFIT-style; several pool their trials)")
    p.add_argument("--closed-loop-trials", type=int, default=2000, help="user: closed-loop trials to run per initial module")
    p.add_argument("--closed-loop-noise", type=float, default=1.0, help="user: the channel's noise std in those trials")
    p.add_argument("--closed-loop-depth", type=int, default=6, help="user: the initial module's reversal depth in them")
    p.add_argument("--closed-loop-arm", default="condition_grid.yaml", help="user: the controller and task they run under")
    p.add_argument("--closed-loop-time-limit", type=float, default=5.0, help="user: their time limit (s)")
    p.add_argument("--closed-loop-mix", type=float, default=0.5, help="user: share of every batch from closed-loop states")
    p.add_argument("--p-free", type=float, default=0.25, help="task: share of layouts without the obstacle")
    p.add_argument("--goal-dropout", type=float, default=0.25, help="task: as the policy was trained")
    p.add_argument("--lambda-id", type=float, default=1e-3, help="task: pull of A, b toward the identity")
    p.add_argument("--d", type=int, default=128, help="task: width")
    p.add_argument("--layers", type=int, default=2, help="task: cross-attention layers")
    p.add_argument("--targets", type=int, nargs=2, default=[4, 4], help="min and max targets per layout")
    p.add_argument("--reach", type=yaml.safe_load, nargs=2, default=[0.35, 0.5], help="min and max reach (null: none)")
    p.add_argument("--step-rank", type=int, help="user: matrices shared by the chunk's steps (default: every step's free)")
    p.add_argument("--cross-step-rank", type=int, default=0, help="user: rank of the coupling between the chunk's steps")
    p.add_argument("--cross-step-dims", type=dim_indices, help="user: the dims the coupling covers, a comma list of "
                   "indices (e.g. 0 or 0,1; default: all)")
    p.add_argument("--n-demos", type=int, default=20000, help="training demonstrations (one reach each)")
    p.add_argument("--frozen-noise", action="store_true", help="draw the channel's noise once per demonstration step "
                   "(as a recorded calibration has it) instead of anew in every batch")
    p.add_argument("--val-demos", type=int, default=2000)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.kind == "task" and (args.obstacle is None or args.user_module is None):
        p.error("--kind task needs the task's --obstacle and the user's --user-module")
    if args.frozen_noise and args.closed_loop:
        p.error("--frozen-noise is for calibration on the demonstrations: not with --closed-loop")
    if args.cross_step_dims is not None and not args.cross_step_rank:
        p.error("--cross-step-dims needs --cross-step-rank")
    if args.kind == "user" and args.obstacle is not None:
        p.error("a user module calibrates on the open field: no --obstacle")
    args.obstacle = obstacle_spec(args.obstacle)
    return args


if __name__ == "__main__":
    train(parse_args())
