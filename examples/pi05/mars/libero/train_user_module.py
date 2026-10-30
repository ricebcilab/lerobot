#!/usr/bin/env python
"""Calibrate a MARS user module for one operator channel on LIBERO (the cursor project's
`train_adapter.py --kind user`, ported to pi0.5).

The user module U(x_t, t) multiplies the policy's velocity in the BACKWARD pass of Flow Reversal
Steering, so that reversing what the channel delivered lands on the latent of what the operator
meant. Open-loop calibration with an expert that knows the goal, as in the cursor project's grid:

  state o      a LIBERO observation from a rollout of the task-prompted policy (the synthetic operator)
  label L      the policy's chunk at o under the scene's real task (what the operator means)
  command c    the operator's push for L: the mean translation of the executed prefix, clipped
               (`PolicyOperator`'s rule), through the arm's channel: corruption M or curl, then noise
               of std sigma ~ U(sigma_min, sigma_max) where the push is active, held over the chunk
  a_bci        the uniform-velocity reference chunk FRS builds from that served command
  z_L          L reversed through the goal-free field (the arm's prompt, e.g. "do something")

and the field regression, in pi0.5's time convention (t = 1 noise, t = 0 action):

    U(x_t, t) P(x_t, t) = z_L - a_bci,   x_t = (1 - t) a_bci + t z_L,   t ~ U(0, 1)

over the 7 action dims, averaged over active commands, so a full reversal of a_bci lands on z_L,
which regenerates L. P is the frozen policy's velocity field (`PI05Policy.velocity_field`).

Usage (from the example directory, in the LIBERO environment):
    python train_user_module.py --config configs/experiment/mars_learned_rotz20.yaml \
        --out outputs/user_modules/rotz20 --episodes 20 --steps 3000
Writes <out>/user.pt (load it with `reversal_adapter: {learned: <out>/user.pt}`), train_log.json and
states.pt (the collected states and labels; `--states <out>/states.pt` reuses them). `--step-rank`,
`--cross-step-rank` and `--cross-step-dims` (e.g. dx,dy,dz) set the module's flexibility (see the README's
user-module section); they are saved in user.pt's config, which steering reads back.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.policies.pi05.steering import (  # noqa: E402
    ACTION_DIMS,
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    LearnedUserModule,
    ReversalAdapter,
    apply_user_module,
    forward_flow,
    reverse_flow,
    save_adapter_module,
    uniform_reference,
)

# ---------------------------------------------------------------- pure parts (unit-tested)


def operator_command(chunk_env: np.ndarray, n_action_steps: int) -> tuple[np.ndarray, float]:
    """`PolicyOperator`'s rule on an env-unit chunk (T, >= 7): the mean translation of the executed prefix,
    clipped to [-1, 1], and the gripper command (close iff the mean gripper is > 0)."""
    prefix = np.asarray(chunk_env, dtype=np.float64)[:n_action_steps]
    translation = np.clip(prefix[:, :3].mean(axis=0), -1.0, 1.0)
    gripper = GRIPPER_CLOSE if prefix[:, GRIPPER_DIM].mean() > 0 else GRIPPER_OPEN
    return translation, gripper


def parse_cross_step_dims(value) -> list[int] | None:
    """The user module's `cross_step_dims` from a config or the command line: None (all dims), a list, or a string
    ('dx,dy,dz', '[dx, dy, dz]', '0,1,2'), each entry a name from ACTION_DIMS or an index into them."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.strip().strip("[]").split(",") if v.strip()]
    out = []
    for v in value:
        if isinstance(v, str) and v in ACTION_DIMS:
            out.append(ACTION_DIMS.index(v))
        elif (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, str) and v.lstrip("-").isdigit()):
            out.append(int(v))
        else:
            raise ValueError(f"cross_step_dims: {v!r} is neither one of {list(ACTION_DIMS)} nor an index")
    # checked here, before collect_states, so a bad value fails in seconds rather than after state collection
    if not out or len(set(out)) != len(out) or not all(0 <= i < N_ACTION_DIMS for i in out):
        raise ValueError(f"cross_step_dims: {value!r} must be unique, non-empty indices in [0, {N_ACTION_DIMS})")
    return out


def channel_core(command: np.ndarray, matrix=None, curl=None, n_steps: int = 10) -> np.ndarray:
    """What the arm's channel serves before noise for a push held over `n_steps` control steps:
    clip(M @ c) for a matrix corruption, the curl field's velocity from rest after `n_steps` for a curl,
    the push itself without either. Idle pushes pass through unchanged, as in the channel."""
    command = np.asarray(command, dtype=np.float64)
    if np.max(np.abs(command)) < DEADBAND:
        return command.copy()
    if curl is not None:
        decay, drive = curl.discretization()
        vel = np.zeros(3)
        for _ in range(n_steps):
            vel = decay @ vel + drive @ command
        return np.clip(vel, -1.0, 1.0)
    if matrix is not None:
        return np.clip(np.asarray(matrix, dtype=np.float64) @ command, -1.0, 1.0)
    return command.copy()


def serve(core: np.ndarray, raw: np.ndarray, sigma, rng: np.random.Generator) -> np.ndarray:
    """The channel's last stage for a batch: `core` (B, 3) plus N(0, sigma^2) noise per axis (sigma (B,)),
    clipped, where the raw push (B, 3) is active; idle pushes are served as they are."""
    core, raw = np.atleast_2d(core), np.atleast_2d(raw)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (core.shape[0],))
    noisy = np.clip(core + rng.normal(size=core.shape) * sigma[:, None], -1.0, 1.0)
    active = np.max(np.abs(raw), axis=-1, keepdims=True) >= DEADBAND
    return np.where(active, noisy, raw)


def field_loss(
    user: LearnedUserModule, velocity, z_l: torch.Tensor, a_bci: torch.Tensor, t: torch.Tensor, weight=None
):
    """The regression along the straight path from a_bci (t = 0) to z_L (t = 1): U(x_t, t) P(x_t, t) = z_L - a_bci
    on the first n action dims (the module's `velocity`, cross-step coupling included); per-sample errors averaged
    with `weight` (B,) when given."""
    tt = t[:, None, None]
    x_t = (1 - tt) * a_bci + tt * z_l
    with torch.no_grad():
        v = velocity(x_t, t).float()
    n = user.n_dims
    pred = apply_user_module(user, x_t.float(), t, v[..., :n])
    err = ((pred - (z_l - a_bci)[..., :n].float()) ** 2).mean((1, 2))
    if weight is None:
        return err.mean()
    weight = weight.float()
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def heading_cosine(chunk_a: torch.Tensor, chunk_b: torch.Tensor) -> torch.Tensor:
    """Cosine between the two chunks' total translations (sum over steps of the first 3 dims), per sample."""
    return nn.functional.cosine_similarity(chunk_a[..., :3].sum(1), chunk_b[..., :3].sum(1), dim=-1)


@torch.no_grad()
def regenerate(velocity, a_bci: torch.Tensor, num_steps: int, user=None) -> torch.Tensor:
    """FRS at full reversal: a_bci reversed through the (user-adapted) backward field, regenerated forward."""
    adapter = ReversalAdapter(user_module=user) if user is not None else None
    z = reverse_flow(a_bci, velocity, num_steps, adapter=adapter)
    return forward_flow(z, velocity, num_steps)


def as_tensors(obj):
    """numpy arrays (and scalars) inside nested dicts/lists as tensors, so a cache loads with weights_only."""
    if isinstance(obj, dict):
        return {k: as_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [as_tensors(v) for v in obj]
    if isinstance(obj, np.ndarray | np.generic):
        return torch.as_tensor(np.asarray(obj))
    return obj


def as_numpy(obj):
    """The inverse of `as_tensors` for observations: tensors back to numpy arrays."""
    if isinstance(obj, dict):
        return {k: as_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [as_numpy(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.numpy()
    return obj


def save_states(states: list[dict], path) -> None:
    """The labelled states as a cache that loads with `weights_only=True` (tensors, numbers, strings)."""
    torch.save([as_tensors(state) for state in states], path)


def load_states(path) -> list[dict]:
    """A cache written by `save_states`: observations back as numpy, labels and latents as tensors."""
    states = []
    for state in torch.load(path, weights_only=True):
        state = as_numpy(state)
        state["label"], state["latent"] = torch.as_tensor(state["label"]), torch.as_tensor(state["latent"])
        states.append(state)
    return states


def stack_batches(batches: list[dict]) -> dict:
    """Batches of size 1 (tensors and lists) into one batch of size B."""
    out = {}
    for key in batches[0]:
        values = [b[key] for b in batches]
        out[key] = (
            torch.cat(values, dim=0) if torch.is_tensor(values[0]) else sum((list(v) for v in values), [])
        )
    return out


# ---------------------------------------------------------------- data from the simulator


def collect_states(session, settings, episodes: int, rng: np.random.Generator) -> list[dict]:
    """Observations at every chunk boundary of `episodes` rollouts of the policy prompted with the scene's
    real task (the states the synthetic operator produces), with that task's description."""
    task_ids = settings.task_ids or list(range(len(session.list_tasks(settings.session.suite))))
    states, n = [], settings.session.n_action_steps
    session.set_mode("policy")
    for episode in range(episodes):
        task_id = int(task_ids[episode % len(task_ids)])
        if (settings.session.suite, task_id) != (session.suite, session.task_id):
            session.set_scene(settings.session.suite, task_id)
        task = session.task_description

        def recorder(step, observation, _task=task, _task_id=task_id, **_):
            if step % n == 0:
                states.append({"observation": copy.deepcopy(observation), "task": _task, "task_id": _task_id})

        torch.manual_seed(int(rng.integers(2**31)))
        session.rollout(task, recorder=recorder)
        print(f"episode {episode + 1}/{episodes} (task {task_id}): {len(states)} states so far", flush=True)
    return states


@torch.no_grad()
def label_states(session, states: list[dict], generic_prompt: str, control, mean, std) -> None:
    """Add to every state its label L (normalized chunk, (chunk, 7)), its latent z_L under the goal-free field
    ((chunk, max_dim)), the operator's raw push, the channel's core and the gripper reference."""
    policy, cfg = session.policy, session.policy.config
    n_out = N_ACTION_DIMS
    for i, state in enumerate(states):
        chunk = session.policy_chunk(state["observation"], state["task"])[0, :, :n_out].float()
        env_chunk = chunk.cpu().numpy() * std + mean
        raw, gripper = operator_command(env_chunk, session.settings.n_action_steps)
        core = channel_core(
            raw, control.corruption_matrix, control.corruption_curl, session.settings.n_action_steps
        )
        padded = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, device=chunk.device)
        padded[0, :, :n_out] = chunk
        with session.autocast_ctx:
            velocity = policy.velocity_field(session.policy_batch(state["observation"], generic_prompt))
            z_l = reverse_flow(padded, velocity, cfg.num_inference_steps)
        state.update(
            label=chunk.cpu(),
            latent=z_l[0].float().cpu(),
            raw=raw,
            core=core,
            gripper_normalized=float((gripper - mean[GRIPPER_DIM]) / std[GRIPPER_DIM]),
        )
        if (i + 1) % 50 == 0:
            print(f"labelled {i + 1}/{len(states)} states", flush=True)


# ---------------------------------------------------------------- training


def train(args) -> None:
    from config import load_experiment_settings
    from session import Session

    from lerobot.policies.pi05.steering import get_action_mean_std

    settings = load_experiment_settings(args.config, {})
    control = settings.session.control
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    session = Session(settings.session)
    policy, cfg = session.policy, session.policy.config
    device = next(policy.parameters()).device
    mean, std = get_action_mean_std(session.postprocessor)
    mean, std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]

    if args.states:
        states = load_states(args.states)
        print(f"loaded {len(states)} labelled states from {args.states}")
    else:
        states = collect_states(session, settings, args.episodes, rng)
        label_states(session, states, settings.prompt, control, mean, std)
        save_states(states, out / "states.pt")
    n_val = max(1, int(round(args.val_frac * len(states))))
    perm = rng.permutation(len(states))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    user = LearnedUserModule(
        cfg.chunk_size,
        N_ACTION_DIMS,
        d=args.d,
        tdim=args.tdim,
        step_rank=args.step_rank,
        cross_step_rank=args.cross_step_rank,
        cross_step_dims=args.cross_step_dims,
    ).to(device)
    opt = torch.optim.AdamW(user.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / args.steps)))

    def batch_of(idx):
        rows = [states[i] for i in idx]
        batch = stack_batches([session.policy_batch(r["observation"], settings.prompt) for r in rows])
        with torch.inference_mode(), session.autocast_ctx:
            velocity = policy.velocity_field(batch)
        sigma = rng.uniform(args.sigma[0], args.sigma[1], size=len(rows))
        served = serve(np.stack([r["core"] for r in rows]), np.stack([r["raw"] for r in rows]), sigma, rng)
        a_bci = uniform_reference(
            served,
            np.array([r["gripper_normalized"] for r in rows]),
            mean,
            std,
            cfg.chunk_size,
            cfg.max_action_dim,
        ).to(device)
        z_l = torch.stack([r["latent"] for r in rows]).to(device)
        label = torch.stack([r["label"] for r in rows]).to(device)
        active = torch.as_tensor(
            np.max(np.abs(np.stack([r["raw"] for r in rows])), -1) >= DEADBAND, device=device
        )
        return velocity, a_bci, z_l, label, active

    log, running, skipped, tic = [], 0.0, 0, time.time()
    for step in range(1, args.steps + 1):
        velocity, a_bci, z_l, _, active = batch_of(
            rng.choice(train_idx, size=min(args.batch, len(train_idx)), replace=False)
        )
        t = torch.rand(len(z_l), device=device)
        loss = field_loss(user, velocity, z_l, a_bci, t, active)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            norm = nn.utils.clip_grad_norm_(user.parameters(), 1.0)
            if torch.isfinite(norm):
                opt.step()
            else:
                skipped += 1
        else:
            skipped += 1
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % args.log_every == 0 or step == args.steps:
            user.eval()
            velocity, a_bci, _, label, active = batch_of(val_idx[: args.batch])
            with torch.no_grad(), session.autocast_ctx:
                cos_none = heading_cosine(regenerate(velocity, a_bci, cfg.num_inference_steps).float(), label)
                cos_user = heading_cosine(
                    regenerate(velocity, a_bci, cfg.num_inference_steps, user).float(), label
                )
            user.train()
            entry = {
                "step": step,
                "train_loss": running / args.log_every,
                "skipped": skipped,
                "heading_cos_none": float(cos_none[active].mean()) if active.any() else None,
                "heading_cos_user": float(cos_user[active].mean()) if active.any() else None,
            }
            log.append(entry)
            print(
                f"step {step:6d}  loss {entry['train_loss']:.4f}  regenerated heading cos: none "
                f"{entry['heading_cos_none']}  U {entry['heading_cos_user']}  ({time.time() - tic:.0f}s)",
                flush=True,
            )
            running = 0.0
    meta = {
        "corruption": control.corruption,
        "policy_path": settings.session.policy_path,
        "prompt": settings.prompt,
        "n_action_steps": settings.session.n_action_steps,
        "sigma": list(args.sigma),
        "states": len(states),
        "episodes": args.episodes,
        "steps": args.steps,
        "config": str(args.config),
    }
    path = save_adapter_module(user.cpu(), out / "user.pt", kind="user", meta=meta)
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=1))
    print(f"saved {path}")
    session.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="an arm file: its corruption, policy, scene and prompt")
    p.add_argument("--out", required=True)
    p.add_argument("--states", default=None, help="reuse a states.pt from an earlier run (skips collection)")
    p.add_argument("--episodes", type=int, default=20, help="task-prompted rollouts to collect states from")
    p.add_argument(
        "--sigma", type=float, nargs=2, default=[0.0, 1.0], help="input-noise std range of the commands"
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--tdim", type=int, default=64)
    p.add_argument(
        "--step-rank",
        type=int,
        default=None,
        help="matrices shared by the chunk's steps (default: every step's matrix free)",
    )
    p.add_argument(
        "--cross-step-rank", type=int, default=0, help="rank of the coupling between the chunk's steps"
    )
    p.add_argument(
        "--cross-step-dims",
        default=None,
        help="dims the coupling covers: names from ACTION_DIMS or indices, e.g. dx,dy,dz (default: all 7)",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    try:
        args.cross_step_dims = parse_cross_step_dims(args.cross_step_dims)
    except ValueError as err:
        p.error(str(err))
    if args.cross_step_dims is not None and not args.cross_step_rank:
        p.error("--cross-step-dims needs --cross-step-rank")
    return args


if __name__ == "__main__":
    train(parse_args())
