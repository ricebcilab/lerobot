"""The task-module trainer: the wiring, the run on fakes, and the fail-closed rules.

Offline and hermetic, as ``test_train_user.py``: a fake base policy with a linear velocity field and a
fake postprocessor carrying action statistics (the pattern ``test_policy.py`` uses), a fake ``demos.Demos``
with the discovered-feature shape the loader produces, ``load_bundle``/``open_demos`` monkeypatched, and
``Scene.step`` returning a stub one-step pipeline that injects ``observation.scene_tokens`` -- so no CUDA,
no checkpoint, no ``dataset`` extra, no ``pi05_base`` and no VLM are touched. The loss itself is main's
``LearnedTaskModule.field_loss`` through ``SteeringPolicy.forward``; the real velocity field, the real
statistics, the real dataset schema and the real scene tokens are verification on chopper only.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import scene as scene_module
import torch
import train_task
import train_user
import training
from config import CameraConfig, Config, TaskConfig, TrainingConfig
from demos import Features
from policy import Bundle
from stats import CHECKPOINT_WIDTH, STATION_DIMS

from lerobot.policies.steering import (
    VLM_SCENE_DIM,
    VLM_SCENE_TOKENS,
    AdapterModules,
    LearnedTaskModule,
)
from lerobot.policies.steering.processor_steering import OBS_SCENE_TOKENS

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
GRIPPER = np.array([1.0])
PROMPT = "pick the cube"
HORIZON = 3  # the fake checkpoint's chunk_size
N_ACTION_STEPS = 2
WIDTH = CHECKPOINT_WIDTH
MEAN = np.zeros(STATION_DIMS)
STD = np.full(STATION_DIMS, 0.25)
RENDER = 32  # the scene render: small enough that the fake batch is instant
IMAGE_HW = (8, 6)  # deliberately not square: the station's letterbox would show

FEATURES = Features(
    arm="follower",
    joint_position="follower/joint_position",
    gripper="follower/gripper_position",
    cameras={"exterior": "scene_zed", "wrist": "wrist_zed"},
    joint_velocity="action/follower/joint_velocity",
    gripper_action="action/follower/gripper_position",
    action=("action/follower/joint_velocity", "action/follower/gripper_position"),
)
"""The discovered features a recorded dataset names, as ``demos.discover`` returns them."""


# ---------------------------------------------------------------- the fakes


class _FakePolicyConfig:
    """pi0.5's sampler interface, as the checkpoint facts and the steering policy read it."""

    chunk_size = HORIZON
    max_action_dim = WIDTH
    num_inference_steps = 4
    n_action_steps = N_ACTION_STEPS
    noise_time = 1.0
    type = "pi05"
    device = "cpu"
    dtype = "bfloat16"


class _LinearPolicy:
    """A base policy whose velocity field is enough for the task loss: no weights, no prefix."""

    def __init__(self) -> None:
        self.config = _FakePolicyConfig()
        self.fields = 0

    def reset(self) -> None:
        pass

    def velocity_field(self, batch: dict):
        self.fields += 1
        return lambda x, time: 0.25 * x


class _BatchingPreprocessor:
    """The little it takes of the real pre-processor: a prepared observation into a B = 1 tensor batch.

    ``Bundle.prepare`` is the adapter that makes the station's numpy arrays tensors and its prompt a
    one-element list (see ``policy._policy_batch``), so this fake preprocessor is the next step: the
    scene step's output is already a batch (``[1, 256, 2048]``) and is passed through, every other
    observation key gets the leading batch dimension the policies read.
    """

    def __call__(self, raw: dict) -> dict:
        out = {}
        for key, value in raw.items():
            if key == OBS_SCENE_TOKENS:
                out[key] = value if torch.is_tensor(value) else torch.as_tensor(value)
            elif isinstance(value, list):
                out[key] = value
            else:
                out[key] = torch.as_tensor(value)[None]
        return out


class _StatsStep:
    """An unnormalizer-like step for ``action_mean_std``: action mean/std in raw units."""

    def __init__(self) -> None:
        mean = np.zeros(WIDTH)
        mean[7] = 0.5  # the gripper's statistical mean, in action units
        std = np.full(WIDTH, 0.25)
        std[7] = 0.4
        self.stats = {"action": {"mean": mean, "std": std}}
        self.norm_map = {}


class _Postprocessor:
    """The bits of the real postprocessor the task batch's normalization uses: action statistics."""

    def __init__(self) -> None:
        self.steps = [_StatsStep()]


class _FakeDemos:
    """A ``demos.Demos``' read surface: one raw station observation and one action chunk per sample.

    The observation carries the dataset's feature names (full camera frames, the arm's state), which is
    the contract ``station.observation_batch`` reads, and the chunk is the recorded ``(chunk_size, 8)``
    in raw action units.
    """

    def __init__(self, episodes: tuple[int, ...] = (0, 0, 1, 1), horizon: int = HORIZON) -> None:
        self.features = FEATURES
        self._episodes = list(episodes)
        self._horizon = horizon

    def __len__(self) -> int:
        return len(self._episodes)

    def observation(self, index: int) -> dict[str, np.ndarray]:
        frame = np.full((*IMAGE_HW, 3), index + 1, dtype=np.uint8)
        return {
            "follower/joint_position": JOINT + 0.01 * index,
            "follower/gripper_position": GRIPPER.copy(),
            "scene_zed": frame,
            "wrist_zed": np.zeros((*IMAGE_HW, 3), dtype=np.uint8),
        }

    def chunk(self, index: int) -> np.ndarray:
        steps = np.arange(self._horizon)
        return np.stack([np.full(STATION_DIMS, 0.01 * (index + 1) + 0.001 * step) for step in steps])

    def episode(self, index: int) -> int:
        return self._episodes[index]


class _StubScenePipeline:
    """``Scene.step``'s one step without the VLM: injects the scene tokens the task module reads.

    Counts the pictures it encoded, so a test can see that each minibatch's rows go through it.
    """

    def __init__(self) -> None:
        self.pictures = 0

    def __call__(self, batch: dict) -> dict:
        pictures = batch["observation.scene_images"]
        self.pictures += len(pictures)
        tokens = torch.zeros((len(pictures), VLM_SCENE_TOKENS, VLM_SCENE_DIM), dtype=torch.float16)
        return {**batch, OBS_SCENE_TOKENS: tokens}


def _bundle() -> Bundle:
    return Bundle(
        policy=_LinearPolicy(),
        preprocessor=_BatchingPreprocessor(),
        postprocessor=_Postprocessor(),
        mean=MEAN,
        std=STD,
    )


def _demos_dir(tmp_path: Path) -> Path:
    """A local dataset directory with the metadata file the trainer hashes."""
    path = tmp_path / "demos" / "pilot"
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(json.dumps({"fps": 15.0}), encoding="utf-8")
    return path


def _config(demos: Path, out: Path, **task) -> Config:
    """A config whose ``[task]`` paths are all under the test's tmp_path (never the checkout's assets)."""
    return Config(
        cameras=CameraConfig(exterior="scene_zed", wrist="wrist_zed"),
        task=TaskConfig(
            demos=demos,
            out=out,
            checkpoint=task.pop("checkpoint", out / "task.pt"),
            render_size=RENDER,
            d=16,
            tdim=8,
            **task,
        ),
        training=TrainingConfig(steps=3, batch=2, log_every=1, device="cpu", seed=0),
    )


def _fake_run(monkeypatch, config: Config, demos: _FakeDemos | None = None) -> _StubScenePipeline:
    """The three seams the run needs: the checkpoint, the dataset and the scene's VLM encode."""
    monkeypatch.setattr(train_task, "load_bundle", lambda config, device=None: _bundle())
    monkeypatch.setattr(
        train_task,
        "open_demos",
        lambda path, cameras, *, chunk_size, n_action_steps: demos or _FakeDemos(),
    )
    pipeline = _StubScenePipeline()
    monkeypatch.setattr(scene_module.Scene, "step", lambda self, *, device, encoder=None: pipeline)
    return pipeline


# ---------------------------------------------------------------- one end-to-end run


def test_train_writes_a_loadable_task_checkpoint_and_a_log(tmp_path, monkeypatch):
    """The whole run on a fake bundle: ``task.pt`` loads back as a task module of the checkpoint's chunk
    and the station's 8 dims, and the log carries the held-out loss and the provenance the checkpoint is
    reproduced against."""
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    config = _config(demos_path, out)
    pipeline = _fake_run(monkeypatch, config)

    path = train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    assert path == out / "task.pt" and path.is_file()
    module, kind, meta = AdapterModules.load_module(path)
    assert kind == "task" and isinstance(module, LearnedTaskModule)
    assert module.horizon == HORIZON and module.n_dims == STATION_DIMS
    assert meta["prompt"] == PROMPT and meta["demos"] == str(demos_path)
    assert meta["demos_sha256"] == sha256((demos_path / "meta" / "info.json").read_bytes()).hexdigest()
    assert meta["scene"] == {"token_source": "vlm_image", "render_size": RENDER, "scene_rank": None}
    assert meta["module"]["token_source"] == "vlm_image" and meta["module"]["image_size"] == RENDER
    assert meta["checkpoint"]["chunk_size"] == HORIZON
    assert meta["checkpoint"]["n_action_steps"] == N_ACTION_STEPS
    assert meta["checkpoint"]["dtype"] == "bfloat16"
    assert meta["split"] == {"val_frac": 0.1, "seed": 0, "sequence": [0, 0]}
    assert meta["samples"] == {"all": 4, "train": 2, "val": 2}
    assert meta["training"]["seed"] == 0 and meta["training"]["device"] == "cpu"

    written = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert written["meta"] == meta
    assert [entry["step"] for entry in written["log"]] == [1, 2, 3]
    assert all(np.isfinite(entry["train_loss"]) for entry in written["log"])
    assert all(
        isinstance(entry["val_loss"], float) and np.isfinite(entry["val_loss"]) for entry in written["log"]
    )
    # every train and held-out row went through the scene step: 2 train + 2 held-out per logged step
    assert pipeline.pictures == 3 * (2 + 2)


def test_a_fresh_module_is_the_identity_and_its_loss_is_finite(tmp_path):
    """A zero-initialised module is plain FRS's field: ``A = I, b = 0``; and the task loss on a batch is
    a finite number (main's field loss through this policy's forward)."""
    config = _config(_demos_dir(tmp_path), tmp_path / "task")
    policy = train_task._steering(config, _bundle(), chunk_size=HORIZON, device="cpu")
    module = policy.adapter.task
    tokens = torch.zeros((2, VLM_SCENE_TOKENS, VLM_SCENE_DIM), dtype=torch.float16)
    x_t = torch.zeros((2, HORIZON, STATION_DIMS))
    velocity = torch.full_like(x_t, 0.5)

    a, b = module(x_t, torch.zeros(2), velocity, {"scene": {"vlm_image": tokens}})

    torch.testing.assert_close(a, torch.eye(STATION_DIMS).expand(2, HORIZON, STATION_DIMS, STATION_DIMS))
    torch.testing.assert_close(b, torch.zeros_like(b))


def test_the_training_loss_is_finite_on_a_batch(tmp_path, monkeypatch):
    """``_loss`` is ``SteeringPolicy.forward``: the adapter's parameters receive a finite loss."""
    config = _config(_demos_dir(tmp_path), tmp_path / "task")
    _fake_run(monkeypatch, config)
    run = _run(config)

    loss = train_task._loss(run, [0, 1])

    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in run.steering.parameters()
    )
    assert any(parameter.grad is not None for parameter in run.steering.adapter.parameters())


def _run(config: Config) -> train_task._Run:
    """A ``_Run`` over the fakes, as ``train`` wires it."""
    bundle = _bundle()
    facts = train_user._checkpoint(bundle)
    steering = train_task._steering(config, bundle, chunk_size=facts["chunk_size"], device="cpu")
    return train_task._Run(
        config=config,
        bundle=bundle,
        steering=steering,
        scene=scene_module.Scene(config.task.token_source, config.task.render_size),
        demos=_FakeDemos(),
        prompt=PROMPT,
        device="cpu",
        facts=facts,
    )


# ---------------------------------------------------------------- the split's edges


def test_the_split_takes_the_validators_seed_sequence(tmp_path, monkeypatch):
    """``demos.split`` gets ``[[training] seed, 0]``: the sequence ``train_user.split_states`` takes, so the
    held-out samples are the ones a validator using the same rule would hold out."""
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    config = _config(demos_path, out)
    _fake_run(monkeypatch, config)
    seen = {}
    real_split = train_task.split

    def spy(demos, *, val_frac, seed):
        seen.update(val_frac=val_frac, seed=seed)
        return real_split(demos, val_frac=val_frac, seed=seed)

    monkeypatch.setattr(train_task, "split", spy)
    train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    assert seen == {"val_frac": config.training.val_frac, "seed": [config.training.seed, 0]}


def test_a_demo_set_smaller_than_a_batch_still_trains(tmp_path, monkeypatch):
    """``[training] batch`` may be larger than the train and held-out sets: the batch is capped, an epoch
    is not refused (``validate-user`` batches its held-out rows the same way)."""
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    config = _config(demos_path, out)
    config = Config(
        cameras=config.cameras,
        task=config.task,
        training=TrainingConfig(steps=2, batch=8, log_every=1, device="cpu", seed=0),
    )
    _fake_run(monkeypatch, config)

    train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))["log"]
    assert [entry["step"] for entry in log] == [1, 2]
    assert all(np.isfinite(entry["val_loss"]) for entry in log)


def test_a_single_episode_demo_set_is_split_by_row(tmp_path, monkeypatch):
    """One episode has no groups to hold apart: ``demos.split`` falls back to a seeded row split, and the
    run trains on the rest."""
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    config = _config(demos_path, out)
    _fake_run(monkeypatch, config, _FakeDemos(episodes=(0, 0, 0, 0)))

    train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    meta = AdapterModules.load_module(out / "task.pt")[2]
    assert meta["samples"] == {"all": 4, "train": 3, "val": 1}
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))["log"]
    assert all(np.isfinite(entry["val_loss"]) for entry in log)


# ---------------------------------------------------------------- the fail-closed rules


def test_train_names_a_missing_demonstrations_dataset(tmp_path):
    missing = tmp_path / "demos" / "pilot"
    with pytest.raises(ValueError, match="teleop --record") as error:
        train_task.train(
            _config(missing, tmp_path / "task"), prompt=PROMPT, demos_path=missing, out=tmp_path / "task"
        )
    assert str(missing) in str(error.value) and "[task] demos" in str(error.value)


def test_train_names_a_dataset_without_its_metadata(tmp_path):
    path = tmp_path / "demos" / "pilot"
    path.mkdir(parents=True)  # a directory, but not a dataset
    with pytest.raises(ValueError, match="meta/info.json") as error:
        train_task.train(
            _config(path, tmp_path / "task"), prompt=PROMPT, demos_path=path, out=tmp_path / "task"
        )
    assert str(path) in str(error.value)


def test_train_refuses_a_scene_source_this_phase_does_not_build(tmp_path):
    demos_path = _demos_dir(tmp_path)
    config = _config(demos_path, tmp_path / "task", token_source="cnn")
    with pytest.raises(ValueError, match="out of scope"):
        train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=tmp_path / "task", device="cpu")


@pytest.mark.parametrize("where", ["out", "checkpoint"])
def test_train_refuses_a_task_module_trained_on_another_prompt(tmp_path, where):
    """A task module shapes one prompt-conditioned field: a checkpoint recorded under another prompt --
    the destination ``[task] out`` or the study's ``[task] checkpoint`` -- is refused by name, as
    ``validate-user`` refuses a module trained under another prompt."""
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    out.mkdir(parents=True)
    checkpoint = out / "task.pt" if where == "out" else tmp_path / "deployed" / "task.pt"
    config = _config(demos_path, out, checkpoint=checkpoint)
    module = LearnedTaskModule(**train_task._module_config(config, HORIZON))
    AdapterModules.save_module(module, checkpoint, kind="task", meta={"prompt": "a different task"})

    with pytest.raises(ValueError, match="trained on prompt") as error:
        train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    assert "a different task" in str(error.value) and PROMPT in str(error.value)
    assert f"[task] {where}:" in str(error.value) and str(checkpoint) in str(error.value)


def test_train_allows_a_task_module_trained_on_the_same_prompt(tmp_path, monkeypatch):
    demos_path = _demos_dir(tmp_path)
    out = tmp_path / "task"
    out.mkdir(parents=True)
    config = _config(demos_path, out)
    module = LearnedTaskModule(**train_task._module_config(config, HORIZON))
    AdapterModules.save_module(module, out / "task.pt", kind="task", meta={"prompt": PROMPT})
    _fake_run(monkeypatch, config)

    train_task.train(config, prompt=PROMPT, demos_path=demos_path, out=out, device="cpu")

    assert AdapterModules.load_module(out / "task.pt")[2]["prompt"] == PROMPT


# ---------------------------------------------------------------- the shared loop and the entry point


def test_the_two_trainers_share_one_training_loop():
    """``training.py`` is the one loop both trainers use: the refactor's point (no second copy)."""
    assert train_user._train_module is training.train_module
    assert train_user._step_if_finite is training.step_if_finite
    assert train_task.train_module is training.train_module


def test_main_trains_from_the_config_and_lets_the_flags_override_the_paths(tmp_path, monkeypatch):
    """The script entry point: ``--config``/``--prompt`` required, ``[task]`` the default paths."""
    config = tmp_path / "franka.toml"
    config.write_text(f'[task]\ndemos = "{tmp_path}/ignored"\nout = "{tmp_path}/out"\n', encoding="utf-8")
    seen = {}

    def fake_train(config, *, prompt, demos_path, out, device=None):
        seen.update(prompt=prompt, demos_path=Path(demos_path), out=Path(out))
        return Path(out) / "task.pt"

    monkeypatch.setattr(train_task, "train", fake_train)
    other = tmp_path / "other"
    assert (
        train_task.main(
            ["--config", str(config), "--prompt", PROMPT, "--demos", str(other), "--out", str(other / "out")]
        )
        == 0
    )
    assert seen == {"prompt": PROMPT, "demos_path": other, "out": other / "out"}
