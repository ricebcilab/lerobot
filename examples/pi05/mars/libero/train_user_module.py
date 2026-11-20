#!/usr/bin/env python
"""Calibrate a MARS user module for one operator readout on LIBERO (the cursor project's
`train_adapter.py --kind user`, ported to pi0.5).

The user module U(x_t, t) multiplies the policy's velocity in the BACKWARD pass of Flow Reversal
Steering, so that reversing what the readout delivered lands on the latent of what the operator
meant. Open-loop calibration with an expert that knows the goal, as in the cursor project's grid:

  state o      a LIBERO observation from a rollout of the task-prompted policy (the synthetic operator) or, with
               `--generic-episodes`, of the policy under the arm's own prompt: the unsteered policy, which may be
               heading for another task, so the command has to redirect it
  label L      the policy's chunk at o under the scene's real task (what the operator means)
  command c    the operator's push for L: the mean translation of the executed prefix, clipped
               (`PolicyOperator`'s rule), through the arm's readout: corruption M, curl or keypad, then
               noise of std sigma ~ U(sigma_min, sigma_max) where the served command is itself active
               (the rollouts' rule), held over the chunk
  a_bci        the uniform-velocity reference chunk FRS builds from that served command
  z_L          L reversed through the goal-free field (the arm's prompt, e.g. "do something")

and the field regression, in pi0.5's time convention (t = 1 noise, t = 0 action):

    U(x_t, t) P(x_t, t) = z_L - a_bci,   x_t = (1 - t) a_bci + t z_L,   t ~ U(0, 1)

over the 7 action dims, averaged over active commands, so a full reversal of a_bci lands on z_L,
which regenerates L. P is the frozen policy's velocity field (`PI05Policy.velocity_field`).

`--target command` replaces the label's latent by where steering takes a faithful command: the plain reversal
(through the same goal-free field) of the reference FRS builds from the operator's push itself, a path of N
points. The regression then runs at the reversal's own steps, on the translation dims (what steering keeps of a
command), so that the adapted reversal of a_bci closes its distance to that path over the first `--correct-by`
steps (default: all N) and follows it afterwards; a faithful command leaves the module the identity (plain FRS).

`--cues N` (with `--target command`) trains on a cued calibration session instead of the tasks' own pushes: at
every state N cues, each the state's push length in a direction drawn evenly on the sphere and held `--cue-hold`
chunks, served through the arm's readout and recorded once with its noise (`--cue-noise redrawn`: the noise drawn
anew at every batch), the label the reversal of the cued push's own reference. A task's pushes cluster in a few
directions, and a module fitted to noisy commands learns that cluster; even cues leave it the user's readout and
the noise to learn. The validation still serves the tasks' own pushes, which such a module has never seen.

Usage (from the example directory, in the LIBERO environment):
    python train_user_module.py --config configs/experiment/mars_learned_rotz20.yaml \
        --out outputs/user_modules/rotz20 --episodes 20 --steps 3000
The log's validation, on held-out states at the middle of the noise range: the chunk the rollouts' steering
computes (`FlowReversalSteering`, the executed prefix pinned) at each of `--val-depths`, from the same noise for
the policy's own plan, FRS on the user's command (`none`), the same through the module (`user`), FRS on a
faithful readout's command with the same noise draw (`clean`) and on the push itself (`ideal`). Reported: the
cosine between the executed steps' mean translation and the operator's push, where the policy's own plan
disagrees with the push (`d<depth>_off_*`) and where it agrees (`d<depth>_on_*`). A module repairs the command
if `off_user` moves from `off_none` toward `off_clean`; one that only weakens the steering raises `on_user` alone.

Writes <out>/user.pt (load it with `reversal_adapter: {learned: <out>/user.pt}`), train_log.json and
states.pt (the collected states and labels; `--states <file>` reuses them, for any arm: the commands are
served again through the arm's own readout). `--step-rank`,
`--cross-step-rank` and `--cross-step-dims` (e.g. dx,dy,dz) set the module's flexibility (see the README's
user-module section); they are saved in user.pt's config, which steering reads back.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.policies.steering import (  # noqa: E402
    ACTION_DIMS,
    GRIPPER_CLOSE,
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    AdaptedField,
    AdapterModules,
    Flow,
    FlowReversalSteering,
    LearnedUserModule,
)
from lerobot.teleoperators import readout  # noqa: E402

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
        elif (isinstance(v, int) and not isinstance(v, bool)) or (
            isinstance(v, str) and v.lstrip("-").isdigit()
        ):
            out.append(int(v))
        else:
            raise ValueError(f"cross_step_dims: {v!r} is neither one of {list(ACTION_DIMS)} nor an index")
    # checked here, before collect_states, so a bad value fails in seconds rather than after state collection
    if not out or len(set(out)) != len(out) or not all(0 <= i < N_ACTION_DIMS for i in out):
        raise ValueError(
            f"cross_step_dims: {value!r} must be unique, non-empty indices in [0, {N_ACTION_DIMS})"
        )
    return out


class _Push:
    """The operator's push as a teleop source, for the rollouts' readers."""

    translation = np.zeros(3)
    gripper = GRIPPER_OPEN


def readout_cores(pushes, episodes, steps, control) -> np.ndarray:
    """What the arm's readout serves before noise for the operator's pushes (B, 3) along the collected episodes,
    by the rollouts' own readers (`teleop.TranslationGain` at the arm's gain, then `teleop.ReadoutReader` with its
    corruption): read once per chunk boundary at control step `steps[i]`, as FRS reads them, a curl field carried
    from one chunk to the next and started from rest at each new entry of `episodes`. (B, 3)."""
    from teleop import ReadoutReader, TranslationGain

    source = _Push()
    reader = ReadoutReader(
        TranslationGain(source, control.resolved_translation_gain),
        control.corruption_matrix,
        curl=control.corruption_curl,
        quantize=control.corruption_quantize,
    )
    cores, current = [], object()
    for push, episode, step in zip(pushes, episodes, steps, strict=True):
        if episode != current:
            reader.reset()
            current = episode
        source.translation = np.asarray(push, dtype=np.float64)
        reader.advance(int(step))
        cores.append(reader.translation)
    return np.stack(cores)


def serve(core: np.ndarray, sigma, rng: np.random.Generator) -> np.ndarray:
    """The readout's last stage for a batch, by the rollouts' rule: `core` (B, 3) plus N(0, sigma^2) noise per
    axis (sigma (B,)), clipped, where the core is itself outside the deadband; an idle core is served as it is."""
    core = np.atleast_2d(core)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (core.shape[0],))
    return readout.add_noise(core, core, sigma, rng.normal(size=core.shape))


def calibration_control(control):
    """The arm's control settings for calibration: its readout (corruption, noise, gain) without its reversal
    adapter, so a run can start from the arm file that names the module it is about to train."""
    return dataclasses.replace(
        control,
        reversal_adapter=None,
        reversal_adapter_matrix=None,
        reversal_adapter_user_path=None,
        reversal_adapter_task_path=None,
    )


def faithful_control(control):
    """The arm's readout without its corruption (its gain kept): what a faithful readout serves for a push."""
    return dataclasses.replace(
        control, corruption=None, corruption_matrix=None, corruption_curl=None, corruption_quantize=False
    )


def label_commands(states: list[dict], control) -> None:
    """Set on every state, from its push `raw`, the `core` the arm's readout serves before noise
    (`readout_cores`) and the `intent`, what a faithful readout would serve for the same push. The rollouts and
    their labels do not depend on the user, so a states file collected for one arm serves another."""
    pushes = [state["raw"] for state in states]
    episodes, steps = [state["episode"] for state in states], [state["step"] for state in states]
    cores = readout_cores(pushes, episodes, steps, control)
    intents = readout_cores(pushes, episodes, steps, faithful_control(control))
    for state, core, intent in zip(states, cores, intents, strict=True):
        state["core"], state["intent"] = core, intent


def cue_pushes(lengths, episodes, rng: np.random.Generator, hold: int = 3) -> np.ndarray:
    """The pushes of a cued calibration session, one per state in collection order: the state's own push length
    (`lengths`) in a direction drawn evenly on the sphere, kept for `hold` consecutive states of an episode (a
    cue lasts a few chunks, so a readout with memory settles on it) and drawn anew at every new episode;
    components clipped to [-1, 1] as the operator's are. No direction is more likely than another, whatever the
    tasks the states come from. (B, 3)."""
    pushes, current, held, direction = [], object(), 0, None
    for length, episode in zip(lengths, episodes, strict=True):
        if episode != current or held == hold:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            current, held = episode, 0
        held += 1
        pushes.append(np.clip(length * direction, -1.0, 1.0))
    return np.stack(pushes)


def reversal_path(flow: Flow, reference: torch.Tensor, steps: int) -> torch.Tensor:
    """The plain reversal of `reference` (B, H, D) step by step: (B, steps + 1, H, D), the reference first and
    its latent last, the points FRS passes through on the policy's own schedule of 1 / steps."""
    h = flow.toward_noise(steps)
    path = [reference]
    for k in range(steps):
        path.append(flow.integrate(path[-1], (1.0 - flow.noise_time) + k * h, h, 1, backward=True))
    return torch.stack(path, dim=1)


def command_target(
    path: torch.Tensor, reference: torch.Tensor, k: torch.Tensor, correct_by: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the adapted reversal of the user's `reference` (B, H, D) should be at reversal step `k` (B,), and
    the velocity that takes it to step k + 1, for the regression U(x, t_k) P(x, t_k) = target: the reversal
    starts at the user's reference and closes its distance to the intended reversal `path` (`reversal_path` of
    the intended reference) in equal parts over the first `correct_by` steps (None: all of them), then follows
    it. A faithful reference gives the path's own points and the plain field's own steps, so the module stays
    the identity. pi0.5's time convention (the reversal steps toward t = 1)."""
    steps = path.shape[1] - 1
    span = steps if correct_by is None else correct_by
    rows = torch.arange(len(path), device=path.device)
    k = k.to(path.device)
    left = (1 - k / span).clamp(min=0)[:, None, None]  # the share of the distance still open at step k
    after = (1 - (k + 1) / span).clamp(min=0)[:, None, None]
    off = reference - path[:, 0]
    at = path[rows, k] + left * off
    target = (path[rows, k + 1] - path[rows, k] - (left - after) * off) * steps
    return at, target


def translation_loss(user, velocity, x: torch.Tensor, t, target: torch.Tensor, weight=None) -> torch.Tensor:
    """The regression U(x, t) P(x, t) = target on the translation dims only, the part of the reversal steering
    keeps from a command (rotation and gripper are handed back to the policy); the module's other rows get no
    gradient and stay the identity. `weight` (B,) averages the per-sample errors, as in `field_loss`."""
    with torch.no_grad():
        v = velocity(x, t).float()
    pred = LearnedUserModule.apply_correction(user, x.float(), t, v[..., : user.n_dims])
    err = ((pred[..., :3] - target[..., :3].float()) ** 2).mean((1, 2))
    if weight is None:
        return err.mean()
    weight = weight.float()
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def executed_summary(intent, plan, arms: dict, threshold: float = 0.5) -> dict:
    """The validation's numbers: the mean cosine between the executed motion ((B, 3), the plan's and each
    arm's) and the operator's push `intent`, over the states where the policy's own plan disagrees with the push
    (cosine below `threshold`: `off_*`) and where it agrees (`on_*`), with the counts `off_n`, `on_n`. States
    with an idle push are left out; an empty group reports None."""
    intent = np.asarray(intent, dtype=np.float64)

    def cosine(moved):
        moved = np.asarray(moved, dtype=np.float64)
        norm = np.linalg.norm(moved, axis=-1) * np.linalg.norm(intent, axis=-1)
        return (moved * intent).sum(-1) / np.maximum(norm, 1e-12)

    pushed, agrees = np.asarray(readout.is_active(intent)), cosine(plan) >= threshold
    out = {}
    for name, group in (("off", pushed & ~agrees), ("on", pushed & agrees)):
        out[f"{name}_n"] = int(group.sum())
        for arm, moved in {"plan": plan, **arms}.items():
            out[f"{name}_{arm}"] = float(cosine(moved)[group].mean()) if group.any() else None
    return out


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
    """A cache written by `save_states`: observations back as numpy, labels, latents and paths as tensors."""
    states = []
    for state in torch.load(path, weights_only=True):
        state = as_numpy(state)
        for key in ("label", "latent", "path", "cue_path"):
            if key in state:
                state[key] = torch.as_tensor(state[key])
        states.append(state)
    return states


def stack_batches(batches: list[dict]) -> dict:
    """Batches of size 1 into one batch of size B: tensors concatenated and lists joined; any other field (the
    preprocessor's empty action, its reward and done defaults, its info) is the first batch's."""
    out = {}
    for key in batches[0]:
        values = [b[key] for b in batches]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list | tuple):
            out[key] = [item for v in values for item in v]
        else:
            out[key] = values[0]
    return out


# ---------------------------------------------------------------- data from the simulator


def collect_states(
    session,
    settings,
    episodes: int,
    rng: np.random.Generator,
    generic_episodes: int = 0,
    generic_steps: int | None = None,
) -> list[dict]:
    """Observations at every chunk boundary of `episodes` rollouts of the policy prompted with the scene's
    real task (the states the synthetic operator produces), then of `generic_episodes` rollouts under the arm's own
    prompt, recorded up to control step `generic_steps` (None: the whole episode): the unsteered policy's states,
    where it may be heading for another task. Every state carries the scene's task (what the operator means) and
    `rollout`, "task" or "generic". Each kind cycles through the tasks, and each pass over them starts from a
    task's next initial state (0, then 1, ...)."""
    task_ids = settings.task_ids or list(range(len(session.list_tasks(settings.session.suite))))
    states, n = [], settings.session.n_action_steps
    session.set_mode("policy")
    plan = [("task", i) for i in range(episodes)] + [("generic", i) for i in range(generic_episodes)]
    for episode, (kind, index) in enumerate(plan):
        task_id = int(task_ids[index % len(task_ids)])
        if (settings.session.suite, task_id) != (session.suite, session.task_id):
            session.set_scene(settings.session.suite, task_id)
        task = session.task_description
        last = generic_steps if kind == "generic" else None

        def recorder(
            step, observation, _task=task, _task_id=task_id, _episode=episode, _kind=kind, _last=last, **_
        ):
            at = int(step) - 1  # the rollout reports the count after the step, with the observation before it
            if at % n == 0 and (_last is None or at < _last):
                states.append(
                    {
                        "observation": copy.deepcopy(observation),
                        "task": _task,
                        "task_id": _task_id,
                        "episode": _episode,
                        "step": at,
                        "rollout": _kind,
                    }
                )

        torch.manual_seed(int(rng.integers(2**31)))
        observation, _ = session.vec_env.reset(
            seed=int(rng.integers(2**31)), options={"init_state_id": index // len(task_ids)}
        )
        if kind == "task":
            session.rollout(task, recorder=recorder, initial_observation=observation)
        else:
            session.rollout(
                settings.prompt, recorder=recorder, initial_observation=observation, max_steps=last
            )
        print(
            f"episode {episode + 1}/{len(plan)} ({kind} rollout, task {task_id}): {len(states)} states so far",
            flush=True,
        )
    return states


@torch.no_grad()
def label_states(session, states: list[dict], generic_prompt: str, control, mean, std) -> None:
    """Add to every state its label L (normalized chunk, (chunk, 7)), its latent z_L under the goal-free field
    ((chunk, max_dim)), the operator's raw push, the readout's core (`readout_cores`: what the rollouts' reader
    serves at that chunk boundary of its episode) and the gripper reference."""
    policy, cfg = session.policy, session.policy.config
    n_out = N_ACTION_DIMS
    for i, state in enumerate(states):
        chunk = session.policy_chunk(state["observation"], state["task"])[0, :, :n_out].float()
        env_chunk = chunk.cpu().numpy() * std + mean
        raw, gripper = operator_command(env_chunk, session.settings.n_action_steps)
        padded = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, device=chunk.device)
        padded[0, :, :n_out] = chunk
        with session.autocast_ctx:
            velocity = policy.velocity_field(session.policy_batch(state["observation"], generic_prompt))
            z_l = Flow(velocity).reverse(padded, cfg.num_inference_steps)
        state.update(
            label=chunk.cpu(),
            latent=z_l[0].float().cpu(),
            raw=raw,
            gripper_normalized=float((gripper - mean[GRIPPER_DIM]) / std[GRIPPER_DIM]),
        )
        if (i + 1) % 50 == 0:
            print(f"labelled {i + 1}/{len(states)} states", flush=True)
    label_commands(states, control)


@torch.no_grad()
def reversal_paths(
    session, states: list[dict], intents, generic_prompt: str, mean, std, batch: int = 8
) -> list:
    """For every state, the plain reversal (`reversal_path`) through the goal-free field of the reference chunk
    FRS builds from the push `intents[i]` served by a faithful readout: where steering takes that command."""
    policy, cfg = session.policy, session.policy.config
    device = next(policy.parameters()).device
    paths = []
    for start in range(0, len(states), batch):
        rows = states[start : start + batch]
        reference = Flow.uniform_reference(
            np.stack(intents[start : start + batch]),
            np.array([r["gripper_normalized"] for r in rows]),
            mean,
            std,
            cfg.chunk_size,
            cfg.max_action_dim,
        ).to(device)
        with session.autocast_ctx:
            velocity = policy.velocity_field(
                stack_batches([session.policy_batch(r["observation"], generic_prompt) for r in rows])
            )
            paths.extend(reversal_path(Flow(velocity), reference, cfg.num_inference_steps).float().cpu())
        if (start // batch) % 10 == 0:
            print(f"reversed {start + len(rows)}/{len(states)} intended references", flush=True)
    return paths


def label_paths(session, states: list[dict], generic_prompt: str, mean, std, batch: int = 8) -> None:
    """Add to every state `path`, the reversal path (`reversal_paths`) of its own `intent`."""
    paths = reversal_paths(session, states, [s["intent"] for s in states], generic_prompt, mean, std, batch)
    for state, path in zip(states, paths, strict=True):
        state["path"] = path


def label_cues(
    session,
    states: list[dict],
    generic_prompt: str,
    control,
    mean,
    std,
    draws: int,
    hold: int,
    sigma,
    rng,
    batch=8,
) -> None:
    """Add a cued calibration session to the states, `draws` cues for each (`cue_pushes`): per state, arrays over
    its cues of what the arm's readout serves for the cued push before noise along the episode (`cue_core`), the
    command as recorded, once, with noise of std sigma ~ U(sigma) (`cue_served`), what a faithful readout would
    serve (`cue_intent`), and the reversal path of that push (`cue_path`): the label needs the cue alone, not an
    expert's motion."""
    lengths = [np.linalg.norm(state["raw"]) for state in states]
    episodes, steps = [state["episode"] for state in states], [state["step"] for state in states]
    sessions = []
    for draw in range(draws):
        pushes = cue_pushes(lengths, episodes, rng, hold)
        cores = readout_cores(pushes, episodes, steps, control)
        intents = readout_cores(pushes, episodes, steps, faithful_control(control))
        served = serve(cores, rng.uniform(sigma[0], sigma[1], size=len(states)), rng)
        print(f"cued session {draw + 1}/{draws}", flush=True)
        paths = reversal_paths(session, states, list(intents), generic_prompt, mean, std, batch)
        sessions.append((cores, served, intents, paths))
    for i, state in enumerate(states):
        state["cue_core"] = np.stack([cores[i] for cores, _, _, _ in sessions])
        state["cue_served"] = np.stack([served[i] for _, served, _, _ in sessions])
        state["cue_intent"] = np.stack([intents[i] for _, _, intents, _ in sessions])
        state["cue_path"] = torch.stack([paths[i] for _, _, _, paths in sessions])


# ---------------------------------------------------------------- the training loop
#
# Moved verbatim from lerobot.policies.steering (train_module, step_if_finite), which keeps no training loop.


def _step_if_finite(opt: torch.optim.Optimizer, loss: torch.Tensor, params, clip: float = 1.0) -> bool:
    """One optimizer step, unless the loss or the clipped gradient norm is not finite (a batch where a flow
    blew up): then the step is skipped and the weights stay whole. Returns whether a step was taken."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    norm = nn.utils.clip_grad_norm_(list(params), clip)
    if not torch.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def _train_module(
    module: nn.Module,
    steps: int,
    lr: float,
    batch_loss: Callable[[int], torch.Tensor],
    validate: Callable[[], dict] | None = None,
    log_every: int = 1000,
    weight_decay: float = 1e-4,
    clip: float = 1.0,
) -> list[dict]:
    """This trainer's loop for a MARS module (the cursor's trainer keeps its own copy): AdamW with a cosine
    schedule over ``steps``, gradients clipped to ``clip``, non-finite batches skipped (``_step_if_finite``).
    ``batch_loss(step)`` returns one batch's loss. Every ``log_every`` steps, and at the last, the module is put
    in eval mode for ``validate()`` (named numbers, or None) and {step, train_loss (the window's mean),
    skipped, **validation} is logged and printed. Returns the log."""
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    log: list[dict] = []
    running, skipped, tic = 0.0, 0, time.time()
    for step in range(1, steps + 1):
        loss = batch_loss(step)
        skipped += not _step_if_finite(opt, loss, module.parameters(), clip)
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % log_every == 0 or step == steps:
            module.eval()
            val = validate() if validate is not None else {}
            module.train()
            entry = {"step": step, "train_loss": running / log_every, "skipped": skipped, **val}
            log.append(entry)
            shown = "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in val.items())
            print(
                f"step {step:6d}  train {entry['train_loss']:.4f}  {shown}"
                + (f"  skipped {skipped}" if skipped else "")
                + f"  ({time.time() - tic:.0f}s)",
                flush=True,
            )
            running = 0.0
    return log


# ---------------------------------------------------------------- training


def train(args) -> None:
    from config import load_experiment_settings
    from session import Session

    from lerobot.policies.steering import FlowControlSteering

    settings = load_experiment_settings(args.config, {}, args.sets)
    control = calibration_control(settings.session.control)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    session = Session(dataclasses.replace(settings.session, control=control))
    policy, cfg = session.policy, session.policy.config
    device = next(policy.parameters()).device
    mean, std = FlowControlSteering.action_mean_std(session.postprocessor)
    mean, std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]

    if args.states:
        states = load_states(args.states)
        # through this arm's readout, whichever arm the file was collected for
        label_commands(states, control)
        print(f"loaded {len(states)} labelled states from {args.states}")
    else:
        states = collect_states(
            session, settings, args.episodes, rng, args.generic_episodes, args.generic_steps
        )
        label_states(session, states, settings.prompt, control, mean, std)
    if args.cues:  # its own generator: the split below stays the one of a run without cues
        label_cues(
            session, states, settings.prompt, control, mean, std,
            args.cues, args.cue_hold, args.sigma, np.random.default_rng([args.seed, 1]), args.batch,
        )  # fmt: skip
    elif args.target == "command" and "path" not in states[0]:
        label_paths(session, states, settings.prompt, mean, std, args.batch)
    if not args.states or Path(args.states).resolve() != (out / "states.pt").resolve():
        save_states(states, out / "states.pt")
    n_steps = cfg.num_inference_steps
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

    def batch_loss(step):
        rows = [states[i] for i in rng.choice(train_idx, size=min(args.batch, len(train_idx)), replace=False)]
        batch = stack_batches([session.policy_batch(r["observation"], settings.prompt) for r in rows])
        with torch.inference_mode(), session.autocast_ctx:
            velocity = policy.velocity_field(batch)
        sigma = rng.uniform(args.sigma[0], args.sigma[1], size=len(rows))
        if args.cues:  # one cue of the calibration session per state: its command and its own path
            draws = rng.integers(args.cues, size=len(rows))
            cue = lambda key: np.stack([r[key][d] for r, d in zip(rows, draws, strict=True)])  # noqa: E731
            served = cue("cue_served") if args.cue_noise == "recorded" else serve(cue("cue_core"), sigma, rng)
            path = torch.stack([r["cue_path"][d] for r, d in zip(rows, draws, strict=True)])
        else:  # the task's own push, its noise drawn anew
            served = serve(np.stack([r["core"] for r in rows]), sigma, rng)
            path = torch.stack([r["path"] for r in rows]) if args.target == "command" else None
        a_bci = Flow.uniform_reference(
            served,
            np.array([r["gripper_normalized"] for r in rows]),
            mean,
            std,
            cfg.chunk_size,
            cfg.max_action_dim,
        ).to(device)
        active = torch.as_tensor(readout.is_active(served), device=device)  # the commands FRS would steer on
        if args.target == "command":
            # onto the reversal of the intended reference, at the reversal's own steps: U(x, t_k) P(x, t_k) = target
            k = torch.randint(n_steps, (len(rows),), device=device)
            x, target = command_target(path.to(device), a_bci, k, args.correct_by)
            return translation_loss(user, velocity, x, k / n_steps, target, weight=active)
        z_l = torch.stack([r["latent"] for r in rows]).to(device)
        t = torch.rand(len(z_l), device=device)
        # The straight path from a_bci (t = 0) to z_L (t = 1): U(x_t, t) P(x_t, t) = z_L - a_bci
        return LearnedUserModule.field_loss(user, velocity, a_bci, z_l, t, weight=active)

    grip = {}  # the validation's reference holds each state's own gripper, as the calibration's does
    steering = FlowReversalSteering(
        policy,
        None,
        postprocessor=session.postprocessor,
        reference_fn=lambda commands: Flow.uniform_reference(
            commands, grip["rows"], mean, std, cfg.chunk_size, cfg.max_action_dim
        ),
    )

    def validate():
        # What the rollouts' steering would execute on every held-out state (the reversal depth, the pinned
        # executed prefix, the same noise for every arm and the policy's own plan), at the middle of the noise
        # range, against the operator's push; the same commands and noise at every log point.
        sigma, val_rng = sum(args.sigma) / 2, np.random.default_rng(args.seed + 1)
        torch_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        moved = {depth: {} for depth in args.val_depths}
        intents = []
        for start in range(0, len(val_idx), args.batch):
            rows = [states[i] for i in val_idx[start : start + args.batch]]
            batch = stack_batches([session.policy_batch(r["observation"], settings.prompt) for r in rows])
            grip["rows"] = np.array([r["gripper_normalized"] for r in rows])
            core, intent = np.stack([r["core"] for r in rows]), np.stack([r["intent"] for r in rows])
            z, seed = val_rng.normal(size=core.shape), int(val_rng.integers(2**31))
            noise_std = np.full(len(rows), sigma)
            commands = {
                "none": readout.add_noise(core, core, noise_std, z),  # FRS on the user's command
                "clean": readout.add_noise(intent, intent, noise_std, z),  # ... on a faithful readout's
                "ideal": intent,  # ... on the push itself
            }
            commands["user"] = commands["none"]  # the same command through the module
            intents.append(np.stack([r["raw"] for r in rows]))
            for depth in args.val_depths:
                steering.n_reversal_steps = depth
                for arm, command in commands.items():
                    torch.manual_seed(seed)
                    with torch.no_grad(), session.autocast_ctx:
                        chunk, plain = steering.steered_chunk(
                            batch, command, field=AdaptedField(None, user) if arm == "user" else None
                        )
                    for name, c in ((arm, chunk), ("plan", plain)):
                        if name == "plan" and arm != "none":
                            continue  # the plan is the same for every arm: the same noise
                        executed = c[:, : cfg.n_action_steps, :3].float().mean(1).cpu().numpy()
                        moved[depth].setdefault(name, []).append(executed * std[:3] + mean[:3])
        torch.set_rng_state(torch_state[0])
        torch.cuda.set_rng_state_all(torch_state[1])
        intent = np.concatenate(intents)
        out = {}
        for depth, arms in moved.items():
            arms = {name: np.concatenate(parts) for name, parts in arms.items()}
            summary = executed_summary(intent, arms.pop("plan"), arms)
            out.update({f"d{depth}_{key}": value for key, value in summary.items()})
        return out

    log = _train_module(user, args.steps, args.lr, batch_loss, validate, args.log_every)
    meta = {
        "corruption": control.corruption,
        "suite": settings.session.suite,
        "task_ids": settings.task_ids,
        "sets": list(args.sets),
        "policy_path": settings.session.policy_path,
        "prompt": settings.prompt,
        "n_action_steps": settings.session.n_action_steps,
        "sigma": list(args.sigma),
        "states": len(states),
        "generic_states": sum(s.get("rollout") == "generic" for s in states),
        "episodes": args.episodes,
        "generic_episodes": args.generic_episodes,
        "generic_steps": args.generic_steps,
        "steps": args.steps,
        "config": str(args.config),
        "target": args.target,
        "correct_by": args.correct_by,
        "cues": args.cues,
        "cue_hold": args.cue_hold,
        "cue_noise": args.cue_noise,
        "val_idx": [int(i) for i in val_idx],
    }
    path = AdapterModules.save_module(user.cpu(), out / "user.pt", kind="user", meta=meta)
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=1))
    print(f"saved {path}")
    session.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="an arm file: its corruption, policy, scene and prompt")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a key of the arm file, as experiment.py does (repeatable), e.g. the calibration scenes: "
        "--set scene.suite=libero_spatial",
    )
    p.add_argument("--states", default=None, help="reuse a states.pt from an earlier run (skips collection)")
    p.add_argument("--episodes", type=int, default=20, help="task-prompted rollouts to collect states from")
    p.add_argument(
        "--generic-episodes",
        type=int,
        default=0,
        help="further rollouts under the arm's own prompt (the unsteered policy), labelled with the scene's task",
    )
    p.add_argument(
        "--generic-steps",
        type=int,
        default=None,
        help="control steps of those rollouts to keep (default: all)",
    )
    p.add_argument(
        "--sigma", type=float, nargs=2, default=[0.0, 1.0], help="input-noise std range of the commands"
    )
    p.add_argument(
        "--target",
        choices=("label", "command"),
        default="label",
        help="what a reversed command should land on: the latent of the operator's whole chunk (label, the "
        "cursor's recipe) or the reversal of the reference FRS builds from the operator's push (command)",
    )
    p.add_argument(
        "--correct-by",
        type=int,
        default=None,
        help="with --target command: the reversal step by which the correction is complete (default: the last)",
    )
    p.add_argument(
        "--cues",
        type=int,
        default=0,
        help="with --target command: train on a cued calibration session instead of the tasks' own pushes, this "
        "many cues per state, each the state's push length in an evenly drawn direction (default 0: off)",
    )
    p.add_argument("--cue-hold", type=int, default=3, help="chunks a cue's direction is held")
    p.add_argument(
        "--cue-noise",
        choices=("recorded", "redrawn"),
        default="recorded",
        help="a cue's command as recorded once, noise included (a session of finite length), or its noise drawn "
        "anew at every batch (the limit of a long session)",
    )
    p.add_argument(
        "--val-depths", type=int, nargs="+", default=[10, 4], help="reversal depths the validation runs"
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
    if args.cues and (args.target != "command" or args.cues < 1 or args.cue_hold < 1):
        p.error(
            "--cues (>= 1, with --cue-hold >= 1) needs --target command: a cue's label is its own reference"
        )
    if args.correct_by is not None and (args.target != "command" or args.correct_by < 1):
        p.error("--correct-by is a reversal step (>= 1) and needs --target command")
    return args


if __name__ == "__main__":
    train(parse_args())
