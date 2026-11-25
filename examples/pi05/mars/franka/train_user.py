"""The user-module trainer: MARS's cue-based target-command calibration on a Franka session's states.

DESIGN.md's "Phase B" recipe, in the order ``train`` runs it:

1. the states a session visited (``statefiles``) are split into a training and a held-out set;
2. for each state and each of ``[user] cues``: a cue twist at the state's joint angles
   (``cues.cue_twists``, its speed drawn from the pilot's recorded deflections through
   ``cues.magnitudes``) goes through the operator's bridge to an intent command (``cues.cue_commands``),
   and the session's readout serves it (``readout.resolve``, then ``serve``) with one noise draw at a
   std from ``[user] sigma``, in action-standard-deviation units;
3. the steering core's own rule -- ``FlowReversalSteering.reference_chunk``, with the study's
   ``reference_step = 1.0`` -- turns a served command into the reference chunk ``a_bci`` FRS steers
   toward, on all seven joint dims (never a re-implementation of ``(command - mean) / std``);
4. the *intent*'s reference is reversed through the velocity field the study itself steers, conditioned
   on ``--prompt``: there is no generic prompt, because the field the module will correct is the
   prompted one. ``reversal_path`` unrolls ``Flow.reverse``'s Euler steps so every point is kept;
5. at a random reversal step ``k`` the adapted reversal is fitted to the straight-line closure
   (``command_target``): ``U(x, t) P(x, t) = target`` on **all seven** dims, ``U`` the
   ``LearnedUserModule`` and ``P`` the frozen policy's field, weighted by the rows the loop's
   ``[command] deadband`` counts as active (an idle cue is not a cue). A faithful command leaves ``U``
   the identity: a zero-initialised module is plain FRS.

Where the per-step field comes from. ``P`` is the policy's velocity field for one observation, which
``bundle.policy.velocity_field`` returns as a closure over the encoded prefix (the images and tokens on
the device). It is evaluated in the checkpoint's own loaded dtype (``[policy] dtype``) and without a
mixed-precision context, exactly as the control loop evaluates it; that dtype is recorded in the
checkpoint's meta and nothing else changes the field's precision, so the module is calibrated in the
field it corrects. The trainer keeps per state only the small arrays the regression reads -- the served
cue commands, (cues, 7), and their intent reversal paths, (cues, N + 1, H, 32), a few tens of kB each --
and a step re-prepares its ``batch`` sampled rows from the npz (``bundle.prepare``, one row at a time
exactly as the control loop does) and stacks them into one batch. Caching every state's prepared device
batch instead would hoard about 1.2 MB of images per state (hundreds of MB on the GPU) to save at most
``batch`` preprocessor calls a step; each state's velocity field is discarded once its paths are
computed for the same reason. The paths themselves are per (state, cue), because a cue's intent depends
on the state's pose and its path on the state's field.

The readout is the config's ``[readout]`` replayed from the file: the states carry the session's ``q0``
(the lift's pose) and ``episode`` (a ``{random: ...}`` corruption's draw), and the noise is the study's
``[user] sigma`` range, so the calibration is against the readout the module will correct, not against a
cleaner one. The file's ``meta`` records the readout its session resolved (the CLI writes it), and
``_check_readout`` compares that with the config: a different corruption, noise, seed or configured ``q0``
is refused, because replaying another readout would train another condition's inverse; a file without
that provenance cannot be compared, and the run says so once before using the config's readout.
``[user] correct_by`` completes the closure early instead of at the last reversal step; ``None`` (the
default) uses all of them, FRS's own schedule.

Everything torch-free stays in ``readout.py``/``cues.py``/``statefiles.py``; this module is the one
place the checkpoint is loaded, and its imports say so. It is the same trainer for a session's states on
the station and for the offline test's fake bundle (``load_bundle`` monkeypatched). The real velocity
field, the real statistics and a real session are verification on the station only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import readout as session_readout
import statefiles
import torch
from bridge import ARM_DIMS
from config import Config, load_config
from cues import cue_commands, cue_twists, magnitudes
from numpy.typing import NDArray
from policy import Bundle, load_bundle, make_steering
from stats import CHECKPOINT_WIDTH
from torch import nn

from lerobot.policies.steering import AdapterModules, Flow, LearnedUserModule
from lerobot.teleoperators import readout

# ---------------------------------------------------------------- the recipe's pure parts


def reversal_path(flow: Flow, reference: torch.Tensor, steps: int) -> torch.Tensor:
    """The plain reversal of ``reference`` (B, H, D) step by step: (B, steps + 1, H, D).

    The reference first and its latent last, every point FRS passes through on the policy's own
    schedule of ``1 / steps`` (the last point is ``Flow.reverse(reference, steps)``). The older LIBERO
    trainer's port: the ``--target command`` regression needs the intermediate points, not only the
    latent.
    """
    h = flow.toward_noise(steps)
    path = [reference]
    for k in range(steps):
        path.append(flow.integrate(path[-1], (1.0 - flow.noise_time) + k * h, h, 1, backward=True))
    return torch.stack(path, dim=1)


def command_target(
    path: torch.Tensor, reference: torch.Tensor, k: torch.Tensor, correct_by: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the adapted reversal of ``reference`` should be at step ``k``, and the velocity to get there.

    ``path`` is the intended reversal (B, steps + 1, H, D) (``reversal_path`` of the *intent*'s
    reference), ``reference`` the served reference ``a_bci`` (B, H, D), ``k`` (B,) reversal steps. The
    adapted reversal starts at ``a_bci`` and closes its distance to ``path`` in equal parts over the
    first ``correct_by`` steps (None: all of them), then follows it. Returns ``(x, target)`` for
    ``U(x, t_k) P(x, t_k) = target``; with a faithful ``reference`` the path's own points and the plain
    field's own steps come back, so the identity module is the solution. Ported from the older LIBERO
    trainer, adapted to any width (its loss was the translation's three dims).
    """
    steps = path.shape[1] - 1
    span = steps if correct_by is None else correct_by
    if span < 1:
        raise ValueError(f"correct_by: expected None or a reversal step >= 1, got {correct_by!r}")
    rows = torch.arange(len(path), device=path.device)
    k = k.to(path.device)
    left = (1 - k / span).clamp(min=0)[:, None, None]  # the share of the distance still open at step k
    after = (1 - (k + 1) / span).clamp(min=0)[:, None, None]
    off = reference - path[:, 0]
    at = path[rows, k] + left * off
    target = (path[rows, k + 1] - path[rows, k] - (left - after) * off) * steps
    return at, target


def command_loss(user, velocity: Callable, x: torch.Tensor, t, target: torch.Tensor, weight=None):
    """The field regression ``U(x, t) P(x, t) = target`` on every one of the module's dims.

    ``P = velocity(x, t)`` is the frozen field (no grad, float32), ``U`` the module's correction
    (``apply_correction``: its cross-step coupling and additive term included). ``target`` is the
    straight-line closure's velocity (B, H, D); the module's first ``n_dims`` dims and its ``horizon``
    chunk steps are the compared ones. ``weight`` (B,) averages the per-sample errors, as in
    ``LearnedUserModule.field_loss``: the rows the control loop counts as active.
    """
    with torch.no_grad():
        v = velocity(x, t).float()
    n = getattr(user, "n_dims", v.shape[-1])
    horizon = getattr(user, "horizon", v.shape[1])
    pred = LearnedUserModule.apply_correction(user, x.float(), t, v[..., :horizon, :n])
    err = ((pred - target[..., :horizon, :n].float()) ** 2).mean((1, 2))
    if weight is None:
        return err.mean()
    weight = weight.float()
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def split_states(
    rows: Sequence[dict], val_frac: float, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    """(train, val) row indices: whole episodes held out when there are several, else a seeded row split.

    The rows of one episode are consecutive chunk boundaries of one trajectory, so a row split would
    leak a state's neighbours into the held-out set; with more than one episode this holds out whole
    episodes (the shuffled order decides which) and keeps at least one for training, so ``val``
    overshoots ``val_frac`` by up to one episode's row count. A file recorded in a single episode has no
    groups to keep, and its rows are split by a seeded permutation instead.
    """
    count = len(rows)
    if count < 2:
        raise ValueError(f"a split needs at least two states to hold one out, got {count}")
    n_val = min(count - 1, max(1, int(round(val_frac * count))))
    episodes = np.array([row["episode"] for row in rows])
    groups = np.unique(episodes)
    if len(groups) > 1:
        val: list[int] = []
        for episode in rng.permutation(groups)[:-1]:  # one episode stays for training
            val.extend(np.flatnonzero(episodes == episode).tolist())
            if len(val) >= n_val:
                break
        held = set(val)
        return [index for index in range(count) if index not in held], val
    order = rng.permutation(count)
    return order[n_val:].tolist(), order[:n_val].tolist()


# ---------------------------------------------------------------- the training loop
#
# Moved verbatim from lerobot.policies.steering (train_module, step_if_finite), which keeps no training
# loop, and from the LIBERO and cursor trainers, which each kept a copy of these two.


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
    """This trainer's loop for a MARS module: AdamW with a cosine schedule over ``steps``, gradients clipped
    to ``clip``, non-finite batches skipped (``_step_if_finite``). ``batch_loss(step)`` returns one batch's
    loss. Every ``log_every`` steps, and at the last, the module is put in eval mode for ``validate()``
    (named numbers, or None) and {step, train_loss (the window's mean), skipped, **validation} is logged and
    printed. Returns the log."""
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


# ---------------------------------------------------------------- one run's fixed pieces


@dataclass(frozen=True)
class _Run:
    """What every step of one run reads: the loaded bundle, the core's reference rule, and the fixed knobs."""

    config: Config
    bundle: Bundle
    core: Any
    prompt: str
    device: str
    q0: NDArray  # the session's nominal pose: the corruption lift's pose, from the states file
    num_steps: int  # the checkpoint's num_inference_steps: the reversal's own step count
    chunk_size: int
    max_action_dim: int


# ---------------------------------------------------------------- the states


def _states_file(states_path: Path) -> Path:
    """The states file ``[user] states`` names: the path itself, or ``<dir>/states.npz`` for a session dir."""
    path = Path(states_path)
    file = path / "states.npz" if path.is_dir() else path
    if not file.is_file():
        raise ValueError(
            f"[user] states: {path} is neither a states file nor a directory holding states.npz "
            "(the file --record-states writes); collect one with `mars-franka ... --record-states NAME`"
        )
    return file


def _rows(arrays: dict[str, NDArray]) -> list[dict]:
    """One row per state: the chunk boundary's two images, the arm's pose, and the episode it came from."""
    return [
        {
            "image_exterior": arrays["image_exterior"][index],
            "image_wrist": arrays["image_wrist"][index],
            "joint_position": arrays["joint_position"][index],
            "gripper": arrays["gripper"][index],
            "episode": int(arrays["episode"][index]),
        }
        for index in range(len(arrays["joint_position"]))
    ]


_READOUT_KEYS = ("corruption", "noise", "seed")
"""The readout provenance a states file's meta carries: the effective ``[readout]`` and its seed."""


def _check_readout(config: Config, path: Path, session: dict, q0: NDArray) -> bool:
    """Raise unless the states file's recorded readout is this config's; return whether it could be checked.

    The trainer replays the config's ``[readout]`` at the file's ``q0`` and each row's episode, so a file
    collected under another readout would train the wrong condition's inverse: a different corruption,
    noise level or noise seed draws different served commands, and a different nominal pose moves the
    corruption's lift. The CLI records the effective ``[readout]`` and its seed in the npz's meta, and the
    lift's pose is the file's own ``q0``, which a configured ``[command] q0`` must reproduce. A file
    without that provenance (hand-built, or older) cannot be compared: the config's readout is used and
    the caller says so once, because the alternative would be a second, file-side override nobody asked
    for -- an experiment is run with the config that matches its data.
    """
    if not all(key in session for key in _READOUT_KEYS):
        return False
    problems = []
    if session["corruption"] != config.readout.corruption:
        problems.append(
            f"corruption {session['corruption']!r} in the file vs {config.readout.corruption!r} in the config"
        )
    if float(session["noise"]) != float(config.readout.noise):
        problems.append(f"noise {session['noise']!r} in the file vs {config.readout.noise!r} in the config")
    if int(session["seed"]) != int(config.readout.seed):
        problems.append(f"seed {session['seed']!r} in the file vs {config.readout.seed!r} in the config")
    pose = np.asarray(config.command.q0, dtype=np.float64)
    if pose.size and not np.array_equal(q0, pose):
        problems.append(f"q0 {q0.tolist()} in the file vs {list(config.command.q0)} as [command] q0")
    if problems:
        raise ValueError(
            f"states: {path} was collected under a different readout than this config -- "
            + "; ".join(problems)
            + ". The trainer replays the config's [readout], so these states would train another "
            "condition's inverse: run an experiment whose config matches the data it trains on"
        )
    return True


def _raw_batch(row: dict, prompt: str) -> dict:
    """A state as the model-ready raw batch ``station.observation_batch`` builds at a chunk boundary.

    The file's uint8 images are the station's ``frame / 255`` (the round trip is exact), the state is the
    checkpoint's 32-wide ``[joint_position, gripper]``, and the task text is the study's prompt.
    """
    state = np.zeros(CHECKPOINT_WIDTH, dtype=np.float32)
    state[:ARM_DIMS] = row["joint_position"]
    state[ARM_DIMS] = row["gripper"][0]
    return {
        "observation.images.base_0_rgb": row["image_exterior"].astype(np.float32) / 255.0,
        "observation.images.left_wrist_0_rgb": row["image_wrist"].astype(np.float32) / 255.0,
        "observation.state": state,
        "task": prompt,
    }


def _stack(batches: list[dict]) -> dict:
    """Batches of size one into one batch of size B: tensors concatenated, lists joined, other fields kept.

    The pre-processor adds the batch dimension (a raw observation is one unbatched row), so every
    prepared batch is B = 1; fields that are not per-sample (the pre-processor's empty action, its
    reward and done defaults, its info) are taken from the first, as the LIBERO trainer does.
    """
    out = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], (list, tuple)):
            out[key] = [item for value in values for item in value]
        else:
            out[key] = values[0]
    return out


def _field(run: _Run, rows: list[dict]) -> Callable:
    """The velocity field for ``rows``: each row through ``bundle.prepare``, stacked, then encoded once."""
    batch = _stack([run.bundle.prepare(_raw_batch(row, run.prompt)) for row in rows])
    with torch.no_grad():
        return run.bundle.policy.velocity_field(batch)


# ---------------------------------------------------------------- the cues and their paths


def _add_cues(run: _Run, rows: list[dict], pool: NDArray, rng: np.random.Generator) -> None:
    """Add each state's cue draws: its served commands and the reversal paths of their intents.

    The cue's speed comes from the pilot's deflection pool, its direction from a unit sphere, and its
    intent goes through the operator's own bridge at the state's joints. The readout is the session's
    (``resolve``: its corruption lifted at the file's ``q0``, its gain, its episode for a random family),
    with ``[user] sigma``'s std drawn per cue and ``serve`` drawing the noise, so the recorded command is
    the one a session of this length would have served. The paths are the plain reversal of each intent's
    own reference through the state's field, which is discarded after the state.
    """
    std7 = np.asarray(run.bundle.std[:ARM_DIMS], dtype=np.float64)
    for row in rows:
        session = session_readout.resolve(
            run.config, q0=run.q0, action_std7=run.bundle.std, episode=row["episode"]
        )
        twists = cue_twists(run.config.user.cues, pool, rng)
        intent = cue_commands(twists, row["joint_position"], run.config.command.twist_length)
        served = []
        for draw in range(len(intent)):
            sigma7 = float(rng.uniform(*run.config.user.sigma)) * std7
            served.append(dataclasses.replace(session, sigma7=sigma7).serve(intent[draw], rng))
        reference = run.core.reference_chunk(intent).float().to(run.device)
        with torch.no_grad():
            path = reversal_path(Flow(_field(run, [row])), reference, run.num_steps)
        row["served"] = np.stack(served)
        row["paths"] = path.float().cpu()
    print(f"cued {len(rows)} states, {run.config.user.cues} cues each", flush=True)


def _loss(run: _Run, user, rows: list[dict], draws: NDArray, k: torch.Tensor) -> torch.Tensor:
    """One batch's regression: each row's sampled cue, its served reference, and one reversal step.

    ``draws`` picks one of a state's cue draws, ``k`` the reversal step (B,); ``a_bci`` is the core's rule
    on that cue's served command and the intent's path is the precomputed one.
    """
    served = np.stack([row["served"][int(draw)] for row, draw in zip(rows, draws, strict=True)])
    path = torch.stack([row["paths"][int(draw)] for row, draw in zip(rows, draws, strict=True)])
    reference = run.core.reference_chunk(served).float().to(run.device)
    x, target = command_target(path.to(run.device), reference, k, run.config.user.correct_by)
    active = torch.as_tensor(readout.is_active(served, run.config.command.deadband), device=run.device)
    velocity = _field(run, rows)
    return command_loss(user, velocity, x, (k / run.num_steps).float(), target, weight=active)


def _held_out(run: _Run, user, rows: list[dict]) -> float:
    """The held-out loss: the same regression on every held-out state, cue draw 0 and a seeded step.

    The draws are recreated from the seed at every log point, so a curve across the log is the module
    alone and not a new noise sample.
    """
    rng = np.random.default_rng([run.config.training.seed, 2])
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), run.config.training.batch):
            group = rows[start : start + run.config.training.batch]
            k = torch.as_tensor(rng.integers(run.num_steps, size=len(group)), device=run.device)
            losses.append(float(_loss(run, user, group, np.zeros(len(group), dtype=int), k)))
    return float(np.mean(losses))


# ---------------------------------------------------------------- the composition root


def _steering(bundle: Bundle, config: Config):
    """The steering policy whose ``core.reference_chunk`` is the study's reference rule (``[modes.frs]``)."""
    mode = config.modes.get("frs")
    if mode is None:
        raise ValueError(
            "[modes.frs] is missing: the user module's reference is the steering core's own rule; add\n"
            '[modes.frs]\nmode = "frs"'
        )
    steering = make_steering(bundle, mode)
    if steering.config.mode != "frs":
        raise ValueError(
            f"[modes.frs].mode is {steering.config.mode!r}, not 'frs': the user module trains against "
            "FRS's reversal"
        )
    return steering


def _checkpoint(bundle: Bundle) -> dict[str, int]:
    """The checkpoint's own sampler facts the reversal needs, each named if the config lacks it."""
    config = getattr(bundle.policy, "config", None)
    facts = {}
    for name in ("chunk_size", "max_action_dim", "num_inference_steps", "n_action_steps"):
        value = getattr(config, name, None)
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"the checkpoint's config has no positive {name!r}: it is not a flow policy a user module "
                "can train on"
            )
        facts[name] = value
    return facts


def _run(config: Config, bundle: Bundle, *, prompt: str, device: str, q0: NDArray) -> _Run:
    """The fixed pieces of one training run: the steering core, the checkpoint's schedule and the pose."""
    facts = _checkpoint(bundle)
    correct_by = config.user.correct_by
    if correct_by is not None and not 1 <= correct_by <= facts["num_inference_steps"]:
        raise ValueError(
            f"[user] correct_by: expected None or a reversal step in [1, {facts['num_inference_steps']}] "
            f"(the checkpoint's num_inference_steps), got {correct_by!r}"
        )
    return _Run(
        config=config,
        bundle=bundle,
        core=_steering(bundle, config).core,
        prompt=str(prompt),
        device=device,
        q0=np.asarray(q0, dtype=np.float64),
        num_steps=facts["num_inference_steps"],
        chunk_size=facts["chunk_size"],
        max_action_dim=facts["max_action_dim"],
    )


def _module(config: Config, run: _Run) -> LearnedUserModule:
    """The module ``[user]`` describes: the seven joint dims, on the checkpoint's chunk (or ``horizon``)."""
    user = config.user
    horizon = run.chunk_size if user.horizon is None else user.horizon
    if not 1 <= horizon <= run.chunk_size:
        raise ValueError(
            f"[user] horizon: expected the checkpoint's chunk_size ({run.chunk_size}) or an integer in "
            f"[1, {run.chunk_size}], got {user.horizon!r}"
        )
    return LearnedUserModule(
        horizon,
        ARM_DIMS,
        d=user.d,
        tdim=user.tdim,
        time_embedding=user.time_embedding,
        step_rank=user.step_rank,
        cross_step_rank=user.cross_step_rank,
        additive=user.additive,
    )


def _meta(
    run: _Run, user: LearnedUserModule, states_path: Path, session: dict, counts: dict, checked: bool
) -> dict:
    """The checkpoint's provenance: what replaying this run has to be reproduced against."""
    config = run.config
    return {
        "prompt": run.prompt,
        "states": str(states_path),
        "states_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "states_rows": counts["rows"],
        "train_rows": counts["train"],
        "val_rows": counts["val"],
        "session": session,
        "policy": str(config.policy.path),
        "corruption": config.readout.corruption,
        "readout": {"noise": config.readout.noise, "seed": config.readout.seed, "checked": checked},
        "command": {
            "translation_gain": config.command.translation_gain,
            "twist_length": config.command.twist_length,
        },
        "checkpoint": {
            "chunk_size": run.chunk_size,
            "max_action_dim": run.max_action_dim,
            "num_inference_steps": run.num_steps,
            "n_action_steps": getattr(run.bundle.policy.config, "n_action_steps", None),
            "dtype": getattr(run.bundle.policy.config, "dtype", None),  # the field's loaded precision
        },
        "module": {
            "horizon": user.horizon,
            "n_dims": user.n_dims,
            "d": config.user.d,
            "tdim": config.user.tdim,
            "time_embedding": config.user.time_embedding,
            "step_rank": config.user.step_rank,
            "cross_step_rank": config.user.cross_step_rank,
            "additive": config.user.additive,
        },
        "sigma": list(config.user.sigma),
        "cues": config.user.cues,
        "correct_by": config.user.correct_by,
        "training": {
            "steps": config.training.steps,
            "batch": config.training.batch,
            "lr": config.training.lr,
            "val_frac": config.training.val_frac,
            "seed": config.training.seed,
            "log_every": config.training.log_every,
            "device": run.device,
        },
    }


def train(config: Config, *, prompt: str, states_path: Path, out: Path, device: str | None = None) -> Path:
    """Train one user module from ``states_path`` and write ``<out>/user.pt`` and ``<out>/train_log.json``.

    The states are the study's (``statefiles``): a session's chunk-boundary observations. The velocity
    field is the study's own prompt; the reference is the steering core's rule; the readout is the one the
    states' session resolved. ``device`` overrides ``[training] device`` (the checkpoint is loaded there
    too); ``device="cpu"`` is how the offline tests run a fake bundle. Returns the checkpoint's path.
    """
    path = _states_file(states_path)
    arrays, session = statefiles.load(path)
    rows = _rows(arrays)
    checked = _check_readout(config, path, session, arrays["q0"])
    if not checked:
        print(
            f"warning: {path} records no readout provenance; its readout is the config's [readout], unchecked",
            flush=True,
        )
    device = str(config.training.device if device is None else device)
    torch.manual_seed(config.training.seed)
    split_rng = np.random.default_rng([config.training.seed, 0])
    train_indices, val_indices = split_states(rows, config.training.val_frac, split_rng)
    train_rows = [rows[index] for index in train_indices]
    val_rows = [rows[index] for index in val_indices]
    bundle = load_bundle(config, device=device)
    run = _run(config, bundle, prompt=prompt, device=device, q0=arrays["q0"])
    user = _module(config, run).to(device)
    print(f"train-user: {len(train_rows)} train / {len(val_rows)} held-out states from {path}", flush=True)

    draw_rng = np.random.default_rng([config.training.seed, 1])
    pool = magnitudes(arrays["deflection"], config.command.translation_gain, rng=draw_rng)
    _add_cues(run, rows, pool, draw_rng)

    def batch_loss(step: int) -> torch.Tensor:
        size = min(config.training.batch, len(train_rows))
        index = split_rng.choice(len(train_rows), size=size, replace=False)
        group = [train_rows[int(i)] for i in index]
        draws = split_rng.integers(config.user.cues, size=size)
        k = torch.randint(run.num_steps, (size,), device=device)
        return _loss(run, user, group, draws, k)

    def validate() -> dict:
        return {"val_loss": _held_out(run, user, val_rows)}

    log = _train_module(
        user, config.training.steps, config.training.lr, batch_loss, validate, config.training.log_every
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"rows": len(rows), "train": len(train_rows), "val": len(val_rows)}
    meta = _meta(run, user, path, session, counts, checked)
    module_path = AdapterModules.save_module(user.cpu(), out / "user.pt", kind="user", meta=meta)
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=1), encoding="utf-8")
    return module_path


def main(argv: list[str] | None = None) -> int:
    """The script entry point: ``--config`` and ``--prompt``, with ``[user]`` naming the input and output.

    ``cli.py``'s ``train-user`` command is the shipped surface; this is the form for a run without it
    (``python train_user.py --config ... --prompt ...``). ``--states``/``--out`` override ``[user]``.
    """
    parser = argparse.ArgumentParser(
        prog="train-user", description="Train MARS's user module on a Franka session's states."
    )
    parser.add_argument("--config", type=Path, required=True, metavar="PATH", help="the config file")
    parser.add_argument("--prompt", required=True, metavar="TEXT", help="the velocity field's prompt")
    parser.add_argument("--states", type=Path, metavar="PATH", help="override [user] states")
    parser.add_argument("--out", type=Path, metavar="PATH", help="override [user] out")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = train(
        config,
        prompt=args.prompt,
        states_path=config.user.states if args.states is None else args.states,
        out=config.user.out if args.out is None else args.out,
    )
    print(f"user module: {path}", flush=True)
    return 0
