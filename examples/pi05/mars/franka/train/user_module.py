#!/usr/bin/env python
"""Calibrate a MARS user module for the Franka's joint-velocity operator readout, offline.

The user module U(x_t, t) multiplies the policy's velocity in the BACKWARD pass of Flow Reversal
Steering (FRS), so that reversing what the readout delivered lands on the latent of what the
operator meant. This is the Franka port of `examples/pi05/mars/libero/train_user_module.py`: the
recipe is the same, and the arm's joint-velocity action space changes only where it must. Open-loop
calibration from a recorded c3po episode, no robot time:

  state o      a frame of a `LeRobotDataset` (c3po writes one), at every `n_action_steps` boundary,
               with the task text the episode recorded
  label L      the policy's chunk at o under the scene's real task prompt, normalized, seven dims:
               the joint-velocity label
  command c    the operator's push for L: the mean joint velocity of the executed prefix through the
               arm's Jacobian, translation only (the stick commands no wrist rotation), clipped to
               [-1, 1]
  served       c through the operator's readout (`lerobot.teleoperators.readout.Readout`, 3-D, the
               robot's noise gate), seeded
  reference    u_hat, the joint command a served push asks for: the exact (Moore-Penrose) inverse of
               the full geometric Jacobian on [c_served * twist_full, 0, 0, 0], divided by
               qdot_full and clipped
  z_L          L reversed through the goal-free field (the deployment prompt, "do something")

and the field regression, in pi0.5's time convention (t = 1 noise, t = 0 action):

    U(x_t, t) P(x_t, t) = z_L - R,   x_t = (1 - t) R + t z_L,   t ~ U(0, 1)

over the seven joint dims, averaged over active commands. R is the reference chunk the steering
core builds from the same command (`FlowReversalSteering.reference_chunk`), in the policy's
normalized space: `(u_hat * reference_step - mean) / std` held over the chunk, padding beyond the
seven dims with zeros. `u_hat` itself is in command units, which is the unit change from LIBERO:
its reference went through `Flow.uniform_reference`'s translation arithmetic, while the Franka's
command is already a joint velocity and needs the checkpoint's action statistics applied here.

Writes `<out>/user.pt` (`AdapterModules.save_module`, kind "user") and `<out>/train_log.json`.
Load the checkpoint from a condition file with `steering.adapter_user: <out>/user.pt` on the
`mars_learned` arm.

Usage (offline, in the training environment, from the example directory):
    python train/user_module.py --dataset /data/c3po/two_blocks \
        --policy lerobot/pi05_droid --stats assets/droid/norm_stats.json \
        --out outputs/user_modules/two_blocks --steps 3000

The readout here has input noise and no corruption: `--sigma-min`/`--sigma-max` are its only
knobs, and a deployment readout with a corruption matrix is a flag to add when the runner's
condition file is wired in. The command speeds `--qdot-full` and `--twist-full` default to
`config.CommandSettings`'s, because the labels must be in the same units the runner will use.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # franka/, the example's own modules

from config import CommandSettings  # noqa: E402
from interfaces import ACTION_DIMS, ARM_DIMS  # noqa: E402
from jacobian import PandaJacobian  # noqa: E402
from stats import ACTION, read_openpi_stats  # noqa: E402

from lerobot.policies.steering import AdapterModules, Flow, LearnedUserModule  # noqa: E402
from lerobot.teleoperators.readout import Readout, is_active  # noqa: E402

COMMAND = CommandSettings()
"""The deployment's command speeds, as the flags' defaults: a full deflection's joint rate and flange speed."""

# ---------------------------------------------------------------- pure parts (unit-tested)


def _arm(value, name: str = "L") -> np.ndarray:
    """A label's arm part as a float64 ``(7,)``: the seven joints, never the gripper."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (ACTION_DIMS,):
        array = array[:ARM_DIMS]
    if array.shape != (ARM_DIMS,):
        raise ValueError(f"{name} must have shape ({ARM_DIMS},) or ({ACTION_DIMS},), got {array.shape}")
    return array


def _translation(value, name: str = "c") -> np.ndarray:
    """A command's translation part as a float64 ``(3,)``."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def _speed(value, name: str) -> float:
    """A positive, finite speed; a zero would divide every command by nothing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive speed, got {value!r}") from None
    if not number > 0 or not math.isfinite(number):
        raise ValueError(f"{name} must be a positive speed, got {value!r}")
    return number


def windows(n_frames: int, n_action_steps: int) -> list[int]:
    """The frames a policy chunk starts on: ``0, n_action_steps, 2 * n_action_steps, ...`` below ``n_frames``.

    A chunk is executed for ``n_action_steps`` control steps, so a new state (and its label) is
    collected every that many frames. A trailing partial chunk is not a window: its label would be
    shorter than the chunk the module is trained for.
    """
    for name, value in (("n_frames", n_frames), ("n_action_steps", n_action_steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be at least 1, got {n_action_steps}")
    return list(range(0, int(n_frames), int(n_action_steps)))


def label_command(jacobian, q, L, twist_full) -> np.ndarray:  # noqa: N803 - the recipe's L
    """The operator's push for a joint-velocity label: ``clip(J(q) L / twist_full)``, translation only.

    ``L`` is the label in the action's own units, rad/s, at ``q`` (rad). It is the caller's
    unnormalized chunk averaged over the executed prefix, LIBERO's `operator_command` rule: one
    push stands for the chunk. ``J(q) L`` is the flange twist; the operator commands translation and
    no wrist rotation, so only its first three rows are the push. Dividing by ``twist_full`` (m/s at
    a full deflection) makes it a fraction of full deflection, and clipping keeps an overshoot from
    asking for more than that fraction.
    """
    twist_full = _speed(twist_full, "twist_full")
    twist = np.asarray(jacobian.twist(_arm(q, "q"), _arm(L)), dtype=np.float64)
    if twist.shape != (6,):
        raise ValueError(f"jacobian.twist must return (6,), got {twist.shape}")
    return np.clip(twist[:3] / twist_full, -1.0, 1.0)


def reference_command(jacobian, q, c_served, twist_full, qdot_full) -> np.ndarray:
    """The joint command a served translation push asks for: ``clip(J+ [c * twist_full, 0, 0, 0] / qdot_full)``.

    The steering core multiplies a command by its ``reference_step`` (``qdot_full``) to get a joint
    velocity, so this is the command that asks for the push's joint motion: the inverse of the
    flange twist ``(c_served * twist_full, 0, 0, 0)`` at ``q``, then divided by ``qdot_full`` and
    clipped to the operator's range.

    The inverse is Moore-Penrose, not the port's damped ``joint_velocity``: a reference is a label,
    not a command sent to the arm, and the damping would shrink it by ``lambda^2 / sigma^2`` at
    every pose, teaching the regression the shrink. The three angular rows are in the solve with a
    zero request, which holds the flange's orientation still while it translates (solving the
    translation rows alone lets the wrist absorb the command, `jacobian.geometric_jacobian`). A
    label with a rotational component therefore lands on its closest realizable part: the operator
    asked for translation, and that is the part the command can carry.

    The Jacobian's columns come from the port's own ``twist`` for each joint's unit velocity, so the
    arithmetic uses the validated kinematics and never a hand-rebuilt table.
    """
    q = _arm(q, "q")
    c_served = _translation(c_served, "c_served")
    twist_full = _speed(twist_full, "twist_full")
    qdot_full = _speed(qdot_full, "qdot_full")

    twist = np.zeros(6)
    twist[:3] = c_served * twist_full
    columns = [np.asarray(jacobian.twist(q, basis), dtype=np.float64) for basis in np.eye(ARM_DIMS)]
    jacobian_matrix = np.stack(columns, axis=1)
    if jacobian_matrix.shape != (6, ARM_DIMS):
        raise ValueError(f"jacobian.twist must return (6,) per joint, got {jacobian_matrix.shape}")
    qdot_ref = np.linalg.lstsq(jacobian_matrix, twist, rcond=None)[0]
    return np.clip(qdot_ref / qdot_full, -1.0, 1.0)


def served_command(c, sigma, seed, corruption=None) -> np.ndarray:
    """What the operator's readout serves for a translation push ``c``: ``(3,)``, seeded and reproducible.

    The readout is the shared one in the robot's configuration (``n_dims=3``, ``noise_gate="command"``):
    a corruption matrix first (none here), then Gaussian noise of std ``sigma`` per axis on the
    command that is itself active, clipped. An idle push never leaves zero, and neither stage can
    move it, which is what makes an idle command the unsteered policy. ``seed`` fixes the noise draw,
    so the same push and seed give the same served command.
    """
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be a finite std >= 0, got {sigma!r}")
    readout = Readout(
        corruption=corruption,
        input_noise=sigma,
        n_dims=3,
        noise_gate="command",
        rng=np.random.default_rng(seed),
    )
    return readout.read(_translation(c))


# ---------------------------------------------------------------- the recorded episode


def _open_dataset(args):
    """The episode selection as a `LeRobotDataset`: the first `--episodes`, or all of them.

    The dataset is local (c3po recorded it), so its metadata is read first: selecting episodes at
    construction keeps the reader from loading the videos of episodes the run does not use.
    """
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    root = Path(args.dataset)
    repo_id = root.name
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    total = int(meta.total_episodes)
    count = total if args.episodes == 0 else min(args.episodes, total)
    if count < 1:
        raise ValueError(f"{args.dataset}: the dataset has no episodes to train on")
    episodes = list(range(count))
    return LeRobotDataset(repo_id, root=root, episodes=episodes), episodes


def _as_uint8(image: torch.Tensor) -> torch.Tensor:
    """One decoded frame as uint8 (C, H, W): the recorded 8-bit pixels, which is all the memory a state needs.

    A dataset frame arrives as float32 in [0, 1] (LeRobot's `ToTensor`); the videos are 8-bit, so the
    round trip is exact, and a state's two cameras cost a quarter of what they would as floats.
    """
    return (image.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _collect_states(dataset, episodes, n_action_steps, prompt, max_frames) -> list[dict]:
    """The dataset's frames at every chunk boundary: the states the labels are built at.

    One state per `windows` index of every selected episode, with the policy's proprioception (seven
    joint positions then the gripper), the camera frames, the episode and boundary frame, and the
    task text the episode recorded (`prompt` overrides it). `max_frames` stops early; 0 reads them all.
    """
    lengths = dataset.meta.episodes["length"]
    camera_keys = tuple(dataset.meta.camera_keys)
    if not camera_keys:
        raise ValueError(f"{dataset.root}: the dataset records no cameras, which a DROID policy needs")
    states: list[dict] = []
    offset = 0
    for episode in episodes:
        length = int(lengths[episode])
        for step in windows(length, n_action_steps):
            frame = dataset[offset + step]
            state = np.asarray(frame["observation.state"], dtype=np.float32)
            if state.shape != (ACTION_DIMS,):
                raise ValueError(
                    f"{dataset.root}: observation.state must be ({ACTION_DIMS},) (joints then gripper), "
                    f"got {state.shape}"
                )
            states.append(
                {
                    "observation": {
                        "state": state,
                        "images": {key: _as_uint8(frame[key]) for key in camera_keys},
                    },
                    "task": str(frame["task"]) if prompt is None else prompt,
                    "episode": int(episode),
                    "step": int(step),
                }
            )
            if max_frames and len(states) >= max_frames:
                return states
        offset += length
    return states


# ---------------------------------------------------------------- the model plumbing


@dataclass(frozen=True)
class _LoadedModel:
    """The checkpoint and everything that turns a stored state into the tensors it consumes."""

    policy: torch.nn.Module
    preprocessor: object
    mean: np.ndarray
    std: np.ndarray
    chunk_size: int
    max_action_dim: int
    n_action_steps: int
    device: torch.device
    use_amp: bool

    def batch(self, observation, prompt: str):
        """One stored observation under `prompt`, preprocessed into the batch the policy takes."""
        batch = {"observation.state": torch.as_tensor(observation["state"], dtype=torch.float32)}
        for key, image in observation["images"].items():
            batch[key] = image.to(torch.float32) / 255.0
        batch["task"] = prompt  # the batcher wraps the string, the tokenizer reads it
        return self.preprocessor(batch)

    def autocast(self):
        """The policy's autocast context, or a null one when the checkpoint does not use AMP."""
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type)


def _camera_map(dataset_keys, policy_keys, policy_path: str) -> dict[str, str]:
    """The rename step's map from the dataset's camera keys to the checkpoint's.

    A key that names the same camera in both needs no entry. The rest are paired in order: c3po's
    recorder names cameras for the station, not for DROID, and which is the exterior and which the
    wrist is the recorder's order. A checkpoint that reads more cameras than the dataset fills runs
    the missing ones through the model's masked empty frames, which is how the published three-camera
    checkpoint runs on a two-camera station.
    """
    dataset_keys, policy_keys = list(dataset_keys), list(policy_keys)
    if not policy_keys:
        return {}
    unmapped = [key for key in dataset_keys if key not in policy_keys]
    unused = [key for key in policy_keys if key not in dataset_keys]
    if len(unmapped) > len(unused):
        raise ValueError(
            f"the dataset has {len(dataset_keys)} cameras ({', '.join(dataset_keys)}) and {policy_path} reads "
            f"{len(policy_keys)} ({', '.join(policy_keys)}); {len(unmapped)} have nowhere to go"
        )
    return dict(zip(unmapped, unused, strict=False))


def _load_model(args, dataset) -> _LoadedModel:
    """The one place the checkpoint, its processors and the label statistics are built.

    The action statistics come from OpenPI's `norm_stats.json`, reduced the way pi0.5's `QUANTILES`
    normalizer reduces them: midpoint and half-width of `q01`/`q99`, not the file's `mean`/`std`. They
    are the same affine map `FlowControlSteering.action_mean_std` reads off the postprocessor, and
    they are what both the policy's chunk and the FRS reference are normalized by.
    """
    from lerobot.policies import PreTrainedConfig, make_policy, make_pre_post_processors

    stats = read_openpi_stats(args.stats)
    action = stats[ACTION]
    q01 = np.asarray(action["q01"], dtype=np.float64)
    q99 = np.asarray(action["q99"], dtype=np.float64)
    if q01.shape != q99.shape or q01.shape[0] < ARM_DIMS:
        raise ValueError(
            f"--stats {args.stats}: the action statistics must cover {ARM_DIMS} dims, got {q01.shape} and "
            f"{q99.shape}"
        )
    mean = (q01[:ARM_DIMS] + q99[:ARM_DIMS]) / 2
    std = np.maximum((q99[:ARM_DIMS] - q01[:ARM_DIMS]) / 2, 1e-6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PreTrainedConfig.from_pretrained(args.policy)
    config.pretrained_path = str(args.policy)
    config.device = device
    if getattr(config, "compile_model", False):
        config.compile_model = False  # one policy call per state: compiling buys nothing here
    cameras = _camera_map(dataset.meta.camera_keys, list(config.image_features), str(args.policy))

    policy = make_policy(config, ds_meta=dataset.meta, rename_map=cameras)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.policy),
        preprocessor_overrides={
            "device_processor": {"device": str(next(policy.parameters()).device)},
            "normalizer_processor": {"stats": stats},
            "rename_observations_processor": {"rename_map": cameras},
        },
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats}},
    )

    chunk_size = int(getattr(config, "chunk_size", 0))
    max_action_dim = int(getattr(config, "max_action_dim", 0))
    n_action_steps = int(getattr(config, "n_action_steps", 0) or chunk_size)
    if chunk_size < 1 or max_action_dim < ARM_DIMS:
        raise ValueError(
            f"{args.policy}: chunk_size {chunk_size} and max_action_dim {max_action_dim} cannot hold a label"
        )
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError(f"{args.policy}: n_action_steps {n_action_steps} must lie in [1, chunk_size]")
    action_width = int(config.output_features[ACTION].shape[0])
    if action_width < ARM_DIMS:
        raise ValueError(
            f"{args.policy}: the action has {action_width} dims, fewer than the {ARM_DIMS} joints"
        )
    return _LoadedModel(
        policy=policy,
        preprocessor=preprocessor,
        mean=mean,
        std=std,
        chunk_size=chunk_size,
        max_action_dim=max_action_dim,
        n_action_steps=n_action_steps,
        device=next(policy.parameters()).device,
        use_amp=bool(getattr(config, "use_amp", False)),
    )


@torch.no_grad()
def _label_states(model: _LoadedModel, jacobian, states: list[dict], args, rng) -> None:
    """Steps 2-6 on every state: its label, the served command, the reference and the latent.

    The policy's chunk under the real task prompt is the label; its mean over the executed prefix is
    the joint velocity the operator's one push stands for. That push goes through the arm's readout
    with a noise std drawn once per state, and back through the Jacobian as the 7-dim command the
    steering core consumes. The latent is the label's reversal through the deployment prompt's field,
    the same goal-free field the module adapts at run time. One policy call per state keeps the
    arithmetic identical to the deployment path; `--batch` is the training batch, not this one.
    """
    steps = int(model.policy.config.num_inference_steps)
    for index, state in enumerate(states):
        observation = state["observation"]
        q = np.asarray(observation["state"][:ARM_DIMS], dtype=np.float64)
        with model.autocast():
            chunk = model.policy.predict_action_chunk(model.batch(observation, state["task"]))
        label = chunk[0, :, :ARM_DIMS].float()
        if label.shape[0] != model.chunk_size:
            raise ValueError(f"{args.policy}: chunk has {label.shape[0]} steps, expected {model.chunk_size}")

        env = label.cpu().numpy() * model.std + model.mean  # rad/s, the action's own units
        prefix = env[: model.n_action_steps].mean(axis=0)  # LIBERO's operator: one push per chunk
        push = label_command(jacobian, q, prefix, args.twist_full)
        sigma = float(rng.uniform(args.sigma_min, args.sigma_max))
        served = served_command(push, sigma, seed=args.seed + index)
        u_hat = reference_command(jacobian, q, served, args.twist_full, args.qdot_full)

        padded = torch.zeros(
            1, model.chunk_size, model.max_action_dim, dtype=torch.float32, device=model.device
        )
        padded[0, :, :ARM_DIMS] = label
        with model.autocast():
            velocity = model.policy.velocity_field(model.batch(observation, args.deployment_prompt))
            latent = Flow(velocity).reverse(padded, steps)

        state.update(
            u_hat=u_hat,
            latent=latent[0].float().cpu().numpy(),
            active=bool(is_active(served)),
            sigma=sigma,
        )
        if (index + 1) % 50 == 0:
            print(f"labelled {index + 1}/{len(states)} states", flush=True)


# ---------------------------------------------------------------- the training loop
#
# Moved from examples/pi05/mars/libero/train_user_module.py, which moved it from
# lerobot.policies.steering (train_module, step_if_finite): the package keeps no training loop.


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
    batch_loss,
    validate=None,
    log_every: int = 1000,
    weight_decay: float = 1e-4,
    clip: float = 1.0,
) -> list[dict]:
    """This trainer's loop for a MARS module: AdamW with a cosine schedule over ``steps``, gradients
    clipped to ``clip``, non-finite batches skipped (``_step_if_finite``). ``batch_loss(step)`` returns
    one batch's loss. Every ``log_every`` steps, and at the last, the module is put in eval mode for
    ``validate()`` (named numbers, or None) and {step, train_loss (the window's mean), skipped,
    **validation} is logged and printed. Returns the log."""
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


def _stack_batches(batches: list[dict]) -> dict:
    """Batches of size 1 into one batch of size B: tensors concatenated and lists joined; any other field
    (the preprocessor's empty action, its reward and done defaults, its info) is the first batch's."""
    out = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if torch.is_tensor(values[0]):
            out[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list | tuple):
            out[key] = [item for value in values for item in value]
        else:
            out[key] = values[0]
    return out


# ---------------------------------------------------------------- training


def train(args) -> None:
    """Collect the states, label them, train the user module and write `user.pt` and `train_log.json`."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dataset, episodes = _open_dataset(args)
    model = _load_model(args, dataset)
    print(
        f"{len(episodes)} episode(s), chunk {model.chunk_size} x {model.max_action_dim}, "
        f"executing {model.n_action_steps} on {model.device}",
        flush=True,
    )

    states = _collect_states(dataset, episodes, model.n_action_steps, args.prompt, args.max_frames)
    if not states:
        raise ValueError(f"{args.dataset}: no chunk boundaries in the selected episodes")
    print(f"{len(states)} states at chunk boundaries", flush=True)
    _label_states(model, PandaJacobian(), states, args, rng)

    n_val = 0
    if args.val_frac > 0 and len(states) > 1:
        n_val = min(max(1, round(args.val_frac * len(states))), len(states) - 1)
    perm = rng.permutation(len(states))
    val_idx = [int(i) for i in perm[:n_val]]
    train_idx = [int(i) for i in perm[n_val:]]

    user = LearnedUserModule(model.chunk_size, ARM_DIMS, step_rank=args.step_rank).to(model.device)

    def batch_loss(rows: list[dict]) -> torch.Tensor:
        """The field regression on `rows`: latent R the steering core's reference, target z_L."""
        latent = np.zeros((len(rows), model.chunk_size, model.max_action_dim), dtype=np.float32)
        for i, row in enumerate(rows):
            latent[i, :, :ARM_DIMS] = (row["u_hat"] * args.qdot_full - model.mean) / model.std
        target = np.stack([row["latent"] for row in rows])
        batch = _stack_batches([model.batch(row["observation"], args.deployment_prompt) for row in rows])
        with torch.no_grad(), model.autocast():
            velocity = model.policy.velocity_field(batch)
        latent_t = torch.as_tensor(latent, device=model.device)
        target_t = torch.as_tensor(target, device=model.device)
        active = torch.as_tensor([row["active"] for row in rows], dtype=torch.bool, device=model.device)
        t = torch.rand(len(rows), device=model.device)
        return LearnedUserModule.field_loss(user, velocity, latent_t, target_t, t, weight=active)

    def validate() -> dict:
        """The held-out field loss, without gradients, over the states the split kept back."""
        if not val_idx:
            return {}
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch):
                rows = [states[i] for i in val_idx[start : start + args.batch]]
                total += float(batch_loss(rows)) * len(rows)
                count += len(rows)
        return {"val_loss": total / max(count, 1)}

    def one_batch(step: int) -> torch.Tensor:
        size = min(args.batch, len(train_idx))
        return batch_loss([states[int(i)] for i in rng.choice(train_idx, size=size, replace=False)])

    log_every = max(1, min(100, args.steps))
    log = _train_module(user, args.steps, args.lr, one_batch, validate, log_every)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": str(args.dataset),
        "episodes": episodes,
        "policy": str(args.policy),
        "stats": str(args.stats),
        "prompt": args.prompt,
        "deployment_prompt": args.deployment_prompt,
        "chunk_size": model.chunk_size,
        "max_action_dim": model.max_action_dim,
        "n_action_steps": model.n_action_steps,
        "n_dims": ARM_DIMS,
        "action_mean": model.mean.tolist(),
        "action_std": model.std.tolist(),
        "qdot_full": args.qdot_full,
        "twist_full": args.twist_full,
        "sigma": [args.sigma_min, args.sigma_max],
        "states": len(states),
        "active_states": sum(row["active"] for row in states),
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "step_rank": args.step_rank,
        "val_frac": args.val_frac,
        "val_idx": val_idx,
        "reference": "u_hat = clip(J+ [c_served * twist_full, 0, 0, 0] / qdot_full)",
        "normalization": "QUANTILES q01/q99 of the action, first seven dims",
    }
    path = AdapterModules.save_module(user.cpu(), out / "user.pt", kind="user", meta=meta)
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=1))
    print(f"saved {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="the root of the recorded c3po LeRobotDataset")
    p.add_argument(
        "--policy", required=True, help="the pi0.5-DROID checkpoint: a Hub id or a local directory"
    )
    p.add_argument("--stats", required=True, help="OpenPI's norm_stats.json for the checkpoint")
    p.add_argument(
        "--prompt",
        default=None,
        help="the task prompt the label is computed under (default: each frame's recorded task text)",
    )
    p.add_argument(
        "--deployment-prompt",
        default="do something",
        help="the goal-free prompt whose field the reversal runs under (the arm's own prompt)",
    )
    p.add_argument("--out", required=True, help="the directory for user.pt and train_log.json")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1, help="the share of states held out from training")
    p.add_argument("--sigma-min", type=float, default=0.0, help="the readout's input-noise std, lower bound")
    p.add_argument("--sigma-max", type=float, default=0.1, help="the readout's input-noise std, upper bound")
    p.add_argument(
        "--step-rank",
        type=int,
        default=None,
        help="matrices shared by the chunk's steps (default: every step's matrix free)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="how many episodes to read, from the first (default 0: every episode in the dataset)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="the most boundary states to label, for a quick run (default 0: no cap)",
    )
    p.add_argument(
        "--qdot-full",
        type=float,
        default=COMMAND.qdot_full,
        help="the joint speed a full deflection asks for, rad/s (the steering reference step)",
    )
    p.add_argument(
        "--twist-full",
        type=float,
        default=COMMAND.twist_full,
        help="the flange speed a full deflection asks for, m/s (before the Jacobian)",
    )
    args = p.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        p.error(f"--dataset {args.dataset}: no such directory (pass the root of the recorded LeRobotDataset)")
    stats = Path(args.stats)
    if not stats.is_file():
        p.error(f"--stats {args.stats}: no such file (OpenPI's norm_stats.json)")
    if args.steps < 1:
        p.error(f"--steps must be at least 1, got {args.steps}")
    if args.batch < 1:
        p.error(f"--batch must be at least 1, got {args.batch}")
    if not args.lr > 0:
        p.error(f"--lr must be positive, got {args.lr}")
    if not 0 <= args.val_frac < 1:
        p.error(f"--val-frac must lie in [0, 1), got {args.val_frac}")
    if args.sigma_min < 0 or args.sigma_max < args.sigma_min:
        p.error(
            f"--sigma-min/--sigma-max must satisfy 0 <= min <= max, got {args.sigma_min} and {args.sigma_max}"
        )
    if args.step_rank is not None and args.step_rank < 1:
        p.error(f"--step-rank must be at least 1, got {args.step_rank}")
    if args.episodes < 0:
        p.error(f"--episodes must not be negative, got {args.episodes}")
    if args.max_frames < 0:
        p.error(f"--max-frames must not be negative, got {args.max_frames}")
    for flag, value in (("--qdot-full", args.qdot_full), ("--twist-full", args.twist_full)):
        if not value > 0 or not math.isfinite(value):
            p.error(f"{flag} must be a positive speed, got {value}")
    return args


if __name__ == "__main__":
    train(parse_args())
