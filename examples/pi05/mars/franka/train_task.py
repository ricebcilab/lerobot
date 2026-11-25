"""The task-module trainer: main's flow matching through the frozen field, on recorded demonstrations.

DESIGN.md's "Phase C" training paragraph, in the order ``train`` runs it:

1. the demonstrations (``demos.open_demos``) -- one sample per chunk boundary of every recorded
   episode, from the dataset's own discovered feature names -- are split with the loader's rule
   (``demos.split``: whole episodes held out when the dataset has several, else a seeded row split),
   seeded with ``[[training] seed, 0]`` -- the seed sequence ``train_user.split_states`` takes, so the
   held-out set is the one a validator using the same rule would evaluate;
2. the checkpoint is loaded (``policy.load_bundle``, **no** task checkpoint: the module is what this run
   builds) and its sampler facts fix the demonstrated chunk (``chunk_size``) and the chunk boundary
   (``n_action_steps``) the samples are taken at;
3. a sample's row is the station's raw observation (``station.observation_batch``: the arm state, the
   two policy images and the prompt) plus the scene camera's frame under ``observation.scene_images``;
   the module's scene step (``scene.Scene``: the token source and render size it is built with) turns
   that frame into ``observation.scene_tokens``, and the bundle's pre-processor carries the row to the
   device. The rows are stacked into one batch, and beside them go ``kind="task"`` and the demonstrated
   ``(chunk_size, 8)`` action chunk in **raw** action units: ``SteeringPolicy.forward`` normalizes the
   chunk with the checkpoint's own action statistics;
4. ``SteeringPolicy.forward`` computes main's ``LearnedTaskModule.field_loss``: the flow matching of
   ``A v + b`` against ``a - z`` along the policy's interpolation, plus ``lambda_id``'s pull of ``A, b``
   toward the identity. This module supplies the data, the scene tokens and the wiring, not the loss --
   and a zero-initialised ``LearnedTaskModule`` is the identity, so a fresh module is plain FRS's field;
5. the shared loop (``training.train_module``) trains the adapter's parameters -- AdamW + cosine,
   gradients clipped, non-finite batches skipped -- and logs the held-out flow-matching loss every
   ``[training] log_every`` steps, drawn from the seed so the curve is the module alone;
6. ``AdapterModules.save_module(..., kind="task")`` writes ``<out>/task.pt`` and ``<out>/train_log.json``;
   the checkpoint's meta records the prompt, the demonstrations and their ``meta/info.json`` digest, the
   scene and module shapes, the split and the checkpoint facts, so a run can be reproduced from it.

Rows are prepared per minibatch, not per step for the whole set: a prepared batch carries the tokenized
prompt and two images, and the demonstrations' samples are full camera pictures, so caching every
prepared row would hoard hundreds of MB on the device to save at most ``[training] batch`` pre-processor
passes a step. The velocity field and the action statistics are the checkpoint's own, so the module is
trained in the field it will shape.

Two fail-closed rules, as in the user-module trainer and its validator. The demonstrations are the
dataset ``[task] demos`` names, opened locally and hashed, and the run names how they are made when
that dataset is missing. And a task module shapes one prompt-conditioned field, so a module recorded
under another prompt at either path this run is defined by -- the destination ``<out>/task.pt`` or
``[task] checkpoint``, the file the study's arms read -- is refused by name (``validate-user``'s rule
for a checkpoint it evaluates), rather than silently replaced.

Everything torch-free stays in ``demos.py``/``scene.py``/``station.py``; this module is the one place
the task trainer loads the checkpoint, and its imports say so. It is the same trainer for the station's
recorded dataset and for the offline test's fake bundle and fake scene step. The real dataset schema,
the real ``lerobot/pi05_base`` tokens, a real training run and every S2 number are verification on the
station (chopper) only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from config import CameraConfig, Config, load_config
from demos import Features, open_demos, split
from numpy.typing import NDArray
from policy import Bundle, _scene_row, load_bundle
from scene import Scene
from station import SCENE_IMAGES, observation_batch
from stats import STATION_DIMS
from train_user import _checkpoint, _stack
from training import train_module

from lerobot.policies.steering import (
    AdapterModules,
    LearnedTaskModule,
    SteeringConfig as PolicySteeringConfig,
    SteeringPolicy,
)

# ---------------------------------------------------------------- what this trainer reads


class Demonstrations(Protocol):
    """The sample reader a ``demos.Demos`` offers: its discovered features and its samples.

    ``demos.open_demos``'s object, or the offline test's stand-in: the trainer only needs one raw
    station observation and one demonstrated action chunk per sample, and the episode each came from
    (``demos.split`` reads the episodes).
    """

    @property
    def features(self) -> Features: ...

    def __len__(self) -> int: ...

    def observation(self, index: int) -> dict[str, NDArray]: ...

    def chunk(self, index: int) -> NDArray: ...

    def episode(self, index: int) -> int: ...


@dataclass(frozen=True)
class _Run:
    """What every step of one run reads: the loaded bundle, its loss-computing steering policy, the
    demonstrations and the fixed knobs."""

    config: Config
    bundle: Bundle
    steering: SteeringPolicy
    scene: Scene
    demos: Demonstrations
    prompt: str
    device: str
    facts: dict[str, int]
    _cache: dict = field(default_factory=dict, repr=False, compare=False, hash=False)

    def prepared(self, index: int) -> dict:
        """Sample ``index`` as one prepared row: raw station batch -> scene tokens -> pre-processor.

        The scene camera's frame is the demonstration's ``[cameras] exterior`` picture (the ZED2), and
        the scene step is the one the module's own checkpoint would build: the same token source and
        render size, so the tokens are the ones the module reads in a session.
        """
        observation = self.demos.observation(index)
        feed = self.demos.features.cameras["exterior"]
        raw = observation_batch(
            observation,
            arm=self.demos.features.arm,
            cameras=self.demos.features.cameras,
            task=self.prompt,
            scene=observation[feed],
            scene_size=self.scene.render_size,
        )
        prepared = self.bundle.prepare(self._scene_pipeline()(_scene_row(raw)))
        prepared.pop(SCENE_IMAGES, None)  # the tokens are the module's; the picture is not a batch feature
        return prepared

    def batch(self, indices: Sequence[int]) -> tuple[dict, torch.Tensor]:
        """The step's batch: the rows stacked, ``kind="task"`` and the demonstrated chunks in raw units."""
        rows = [self.prepared(int(index)) for index in indices]
        chunks = torch.as_tensor(
            np.stack([self.demos.chunk(int(index)) for index in indices]),
            dtype=torch.float32,
            device=self.device,
        )
        return _stack(rows), chunks

    def _scene_pipeline(self) -> Any:
        """The scene's one-step pipeline, built once and kept: its encoder and its token cache are state."""
        pipeline = self._cache.get("scene_pipeline")
        if pipeline is None:
            pipeline = self.scene.step(device=self.device)
            self._cache["scene_pipeline"] = pipeline
        return pipeline


# ---------------------------------------------------------------- the training loss


def _loss(run: _Run, indices: Sequence[int], rng: np.random.Generator | None = None) -> torch.Tensor:
    """One batch's loss through ``SteeringPolicy.forward``: main's task loss, no re-implementation.

    ``rng`` fixes the noise and the flow times (the held-out read); without it ``forward`` draws them,
    from the trainer's ``torch.manual_seed``.
    """
    batch, chunks = run.batch(indices)
    fields = {**batch, "kind": "task", "chunk": chunks}
    if rng is not None:
        fields["noise"] = torch.as_tensor(
            rng.standard_normal(tuple(chunks.shape)), dtype=torch.float32, device=run.device
        )
        fields["time"] = torch.as_tensor(rng.random(len(indices)), dtype=torch.float32, device=run.device)
    loss, _ = run.steering.forward(fields)
    return loss


def _held_out(run: _Run, indices: Sequence[int]) -> float:
    """The held-out loss: every held-out sample, a seeded noise and time drawn anew from ``[seed, 2]``.

    The draws are recreated at every log point, so a curve across the log is the module alone and not a
    new noise sample.
    """
    rng = np.random.default_rng([int(run.config.training.seed), 2])
    losses = []
    with torch.no_grad():
        batch = int(run.config.training.batch)
        for start in range(0, len(indices), batch):
            losses.append(float(_loss(run, indices[start : start + batch], rng)))
    return float(np.mean(losses))


# ---------------------------------------------------------------- the run's fixed pieces


def _camera_roles(cameras: CameraConfig) -> dict[str, str]:
    """The configured camera roles as the name->feed mapping ``demos.discover`` takes, third omitted if absent."""
    roles = {"exterior": cameras.exterior, "wrist": cameras.wrist, "third": cameras.third}
    return {role: feed for role, feed in roles.items() if feed is not None}


def _demos_path(path: str | Path) -> Path:
    """The ``[task] demos`` dataset directory, named with how it is made when it is not there."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(
            f"[task] demos: {path} is not a downloaded demonstrations dataset. Record demonstrations first "
            "with `mars-franka teleop --record NAME` (it downloads one to [recording] dir), then point "
            "[task] demos at that dataset"
        )
    return path


def _dataset_digest(path: Path) -> tuple[Path, str]:
    """The demonstration dataset's provenance: its ``meta/info.json`` and the digest of its bytes.

    The file is the LeRobot dataset's own metadata (the feature mapping and the frame count); hashing
    it pins which dataset a checkpoint was trained on without hashing the frames. A directory without
    one is not the dataset the loader opens, so it is named rather than guessed at.
    """
    info = path / "meta" / "info.json"
    if not info.is_file():
        raise ValueError(
            f"[task] demos: {path} has no meta/info.json: it is not a LeRobot dataset directory (the file "
            "the checkpoint's provenance hashes)"
        )
    return info, hashlib.sha256(info.read_bytes()).hexdigest()


def _check_prompt(config: Config, out: Path, prompt: str) -> None:
    """Refuse a task module recorded under another prompt, at either path this run is defined by.

    The rule is ``validate_user``'s (a checkpoint recorded under one prompt is not another prompt's
    module). A task module shapes the field the study's prompt makes, so replacing a differently
    prompted module -- the destination ``<out>/task.pt`` this run writes, or ``[task] checkpoint`` the
    study's arms read (the same file by convention) -- would leave the study with a module for another
    task. Both are checked before the checkpoint is loaded; moving the module, or pointing ``[task]``
    at a new dataset/out/checkpoint, is how another prompt is trained.
    """
    destinations = {
        "[task] out": out / "task.pt",
        "[task] checkpoint": Path(config.task.checkpoint),
    }
    seen: set[Path] = set()
    for label, destination in destinations.items():
        resolved = destination.resolve()
        if resolved in seen or not destination.is_file():
            continue
        seen.add(resolved)
        meta = AdapterModules.read_checkpoint(destination, mmap=True).get("meta") or {}
        recorded = meta.get("prompt")
        if recorded is not None and str(recorded) != str(prompt):
            raise ValueError(
                f"{label}: {destination} was trained on prompt {recorded!r}, not {prompt!r}: a task module "
                "shapes one prompt-conditioned field, so train-task takes the trainer's --prompt; point "
                "[task] out/checkpoint elsewhere (or move the module) to train another prompt"
            )


def _module_config(config: Config, chunk_size: int) -> dict:
    """The task module's constructor config: the scene spec, the checkpoint's chunk and ``[task]``'s width.

    ``LearnedTaskModule.spec`` is main's rule for the scene sources (the token geometry, the ``vlm_image``
    token count and width, the optional fixed reduction); the checkpoint's ``chunk_size`` is the module's
    horizon, because a demonstrated chunk is exactly what it learns to produce, and its n_dims are the
    station's 8 action dims.
    """
    spec = LearnedTaskModule.spec(
        config.task.token_source, config.task.render_size, scene_rank=config.task.scene_rank
    )
    return {
        "horizon": chunk_size,
        "n_dims": STATION_DIMS,
        "d": config.task.d,
        "tdim": config.task.tdim,
        **spec,
    }


def _steering(config: Config, bundle: Bundle, *, chunk_size: int, device: str) -> SteeringPolicy:
    """The mode-``"none"`` ``SteeringPolicy`` whose ``forward`` computes the task module's loss.

    Its adapter is the only trainable state: a task module built from ``[task]``'s scene and shape, held
    in the ``task_module`` config so ``SteeringPolicy`` builds it from the same recorded config a
    checkpoint carries. ``lambda_id`` is the identity pull ``forward`` reads, and the base policy type is
    the checkpoint's own; no steering core is built (there is nothing to steer here, only to fit).
    """
    base = getattr(bundle.policy, "config", None)
    fields = PolicySteeringConfig(
        mode="none",
        n_dims=STATION_DIMS,
        task_module=_module_config(config, chunk_size),
        lambda_id=config.task.lambda_id,
        base_policy_type=str(getattr(base, "type", "pi05")),
        device=device,
    )
    return SteeringPolicy(fields, base_policy=bundle.policy, postprocessor=bundle.postprocessor)


def _meta(run: _Run, *, demos: Path, info: Path, digest: str, counts: dict[str, int]) -> dict:
    """The checkpoint's provenance: what reproducing this run has to be reproduced against."""
    config = run.config
    return {
        "prompt": run.prompt,
        "demos": str(demos),
        "demos_info": str(info),
        "demos_sha256": digest,
        "samples": {"all": counts["samples"], "train": counts["train"], "val": counts["val"]},
        "split": {
            "val_frac": config.training.val_frac,
            "seed": config.training.seed,
            "sequence": [int(config.training.seed), 0],
        },
        "scene": {
            "token_source": config.task.token_source,
            "render_size": config.task.render_size,
            "scene_rank": config.task.scene_rank,
        },
        "module": dict(run.steering.adapter.task.config),
        "checkpoint": {
            "policy": str(config.policy.path),
            "chunk_size": run.facts["chunk_size"],
            "max_action_dim": run.facts["max_action_dim"],
            "num_inference_steps": run.facts["num_inference_steps"],
            "n_action_steps": run.facts["n_action_steps"],
            "dtype": getattr(run.bundle.policy.config, "dtype", None),
        },
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


# ---------------------------------------------------------------- the trainer


def train(config: Config, *, prompt: str, demos_path: Path, out: Path, device: str | None = None) -> Path:
    """Train one task module from the demonstrations at ``demos_path``; return ``<out>/task.pt``.

    The demonstrations are the dataset ``[task] demos`` names, the prompt is the study's velocity-field
    prompt, and ``out`` is the directory ``task.pt`` and ``train_log.json`` land in. ``device`` overrides
    ``[training] device`` (the checkpoint is loaded there too); ``device="cpu"`` is how the offline tests
    run a fake bundle. The checkpoint's sampler facts fix the chunk and the boundary, so the loader and
    the module agree with the policy that will be shaped.
    """
    demos_path = _demos_path(demos_path)
    info, digest = _dataset_digest(demos_path)
    out = Path(out)
    _check_prompt(config, out, prompt)
    scene = Scene(config.task.token_source, config.task.render_size)  # this phase's scope, refused by name
    device = str(config.training.device if device is None else device)
    torch.manual_seed(config.training.seed)

    bundle = load_bundle(config, device=device)
    facts = _checkpoint(bundle)
    steering = _steering(config, bundle, chunk_size=facts["chunk_size"], device=device)
    steering.adapter.to(device)
    demos = open_demos(
        demos_path,
        _camera_roles(config.cameras),
        chunk_size=facts["chunk_size"],
        n_action_steps=facts["n_action_steps"],
    )
    train_indices, val_indices = split(
        demos, val_frac=config.training.val_frac, seed=[int(config.training.seed), 0]
    )
    run = _Run(
        config=config,
        bundle=bundle,
        steering=steering,
        scene=scene,
        demos=demos,
        prompt=str(prompt),
        device=device,
        facts=facts,
    )
    print(
        f"train-task: {len(train_indices)} train / {len(val_indices)} held-out demonstrations from {demos_path}",
        flush=True,
    )

    sample_rng = np.random.default_rng([int(config.training.seed), 0])

    def batch_loss(step: int) -> torch.Tensor:
        size = min(config.training.batch, len(train_indices))
        chosen = sample_rng.choice(len(train_indices), size=size, replace=False)
        return _loss(run, [int(train_indices[index]) for index in chosen])

    def validate() -> dict:
        return {"val_loss": _held_out(run, val_indices)}

    log = train_module(
        steering.adapter,
        config.training.steps,
        config.training.lr,
        batch_loss,
        validate,
        config.training.log_every,
    )

    out.mkdir(parents=True, exist_ok=True)
    counts = {"samples": len(demos), "train": len(train_indices), "val": len(val_indices)}
    meta = _meta(run, demos=demos_path, info=info, digest=digest, counts=counts)
    module_path = AdapterModules.save_module(
        steering.adapter.task.cpu(), out / "task.pt", kind="task", meta=meta
    )
    (out / "train_log.json").write_text(json.dumps({"meta": meta, "log": log}, indent=1), encoding="utf-8")
    return module_path


def main(argv: list[str] | None = None) -> int:
    """The script entry point: ``--config`` and ``--prompt``, with ``[task]`` naming the dataset and the output.

    ``cli.py``'s ``train-task`` command is the shipped surface; this is the form for a run without it
    (``python train_task.py --config ... --prompt ...``). ``--demos``/``--out`` override ``[task]``.
    """
    parser = argparse.ArgumentParser(
        prog="train-task", description="Train MARS's task module on recorded Franka demonstrations."
    )
    parser.add_argument("--config", type=Path, required=True, metavar="PATH", help="the config file")
    parser.add_argument("--prompt", required=True, metavar="TEXT", help="the velocity field's prompt")
    parser.add_argument("--demos", type=Path, metavar="PATH", help="override [task] demos")
    parser.add_argument("--out", type=Path, metavar="PATH", help="override [task] out")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    path = train(
        config,
        prompt=args.prompt,
        demos_path=config.task.demos if args.demos is None else args.demos,
        out=config.task.out if args.out is None else args.out,
    )
    print(f"task module: {path}", flush=True)
    return 0
