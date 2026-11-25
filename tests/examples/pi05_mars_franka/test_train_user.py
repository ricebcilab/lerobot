"""The user-module trainer: the recipe's pure parts, the loss, and one end-to-end run on a fake bundle.

Offline and hermetic: a fake base policy with a linear velocity field and a fake postprocessor carrying
action statistics (the pattern ``test_policy.py`` uses), a two-state npz written by ``statefiles``, and
``load_bundle`` monkeypatched, so no CUDA, no checkpoint and no station are touched. The real field, the
real statistics and a real session are verification on the station (chopper) only.
"""

from __future__ import annotations

import dataclasses
import json
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
import train_user
from bridge import ARM_DIMS
from config import Config, ReadoutConfig, SteeringConfig, TrainingConfig, UserConfig
from policy import Bundle
from statefiles import Recorder
from stats import CHECKPOINT_WIDTH

from lerobot.policies.steering import AdapterModules, Flow, LearnedUserModule

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
Q0 = JOINT + 0.01
GRIPPER = np.array([1.0])
DEFLECTION = np.array([0.8, -0.7, 0.6, 0.0, 0.0, 0.0])  # a pilot push, past the readout's deadband
SERVED = np.linspace(-0.4, 0.6, ARM_DIMS)
PROMPT = "pick the cube"
READOUT_META = {"corruption": {"rotation_z_deg": 20}, "noise": 0.35, "seed": 0}
"""The readout provenance the CLI writes into a states file: this file's config's [readout]."""
N_STEPS = 5
HORIZON = 3
WIDTH = 32  # the checkpoint's padded chunk width


# ---------------------------------------------------------------- the fields the hands derive


def _constant(value: float):
    """A velocity field with no dependence on ``x``: the plain reversal is a straight line."""
    return lambda x, time: torch.full_like(x, value)


def _linear(rate: float):
    """A velocity field ``P(x, t) = rate * x``: the reversal is a geometric recursion."""
    return lambda x, time: rate * x


def _reference(batch: int = 2) -> torch.Tensor:
    """A (B, H, D) reference chunk with distinct values, so a transposed or flattened index shows."""
    values = torch.arange(1.0, 1.0 + batch * HORIZON * WIDTH).reshape(batch, HORIZON, WIDTH)
    return values / 10.0


# ---------------------------------------------------------------- reversal_path and command_target


def test_a_constant_field_reverses_the_reference_along_a_straight_line():
    """P = const: ``path[k] = reference + k / N * P``, and the last point is ``Flow.reverse``'s latent."""
    reference, flow = _reference(), Flow(_constant(0.5))
    path = train_user.reversal_path(flow, reference, N_STEPS)
    assert path.shape == (2, N_STEPS + 1, HORIZON, WIDTH)
    for k in range(N_STEPS + 1):
        torch.testing.assert_close(path[:, k], reference + (k / N_STEPS) * 0.5)
    torch.testing.assert_close(path[:, -1], flow.reverse(reference, N_STEPS))
    # The intermediate points are the plain reversal's own: each step is one Euler step of the field.
    for k in range(N_STEPS):
        torch.testing.assert_close(path[:, k + 1] - path[:, k], torch.full_like(reference, 0.5 / N_STEPS))


def test_command_target_on_a_constant_field_closes_the_distance_to_the_intent():
    """A served reference off the intent's by ``off`` gets ``at = path[k] + left * off`` and the velocity
    ``P - off``: the closure's closed form, checked at several reversal steps."""
    reference, flow = _reference(), Flow(_constant(0.5))
    path = train_user.reversal_path(flow, reference, N_STEPS)
    served = reference - 0.25
    for step in (0, 2, N_STEPS - 1):
        k = torch.full((len(reference),), step, dtype=torch.long)
        at, target = train_user.command_target(path, served, k)
        left, after = 1 - step / N_STEPS, 1 - (step + 1) / N_STEPS
        rows = torch.arange(len(reference))
        off = served - reference
        torch.testing.assert_close(at, path[rows, k] + left * off)
        torch.testing.assert_close(
            target, (path[rows, k + 1] - path[rows, k] - (left - after) * off) * N_STEPS
        )
        torch.testing.assert_close(target, torch.full_like(target, 0.75))  # `P - off`, off = -0.25


def test_a_linear_field_reverses_geometrically_and_its_velocity_is_hand_derived():
    """P = a x: ``path[k] = (1 + a / N)**k reference``, so the intended step is ``a / N * path[k]`` and
    ``command_target``'s velocity is ``a * path[k] - off``."""
    rate, reference = 0.3, _reference()
    path = train_user.reversal_path(Flow(_linear(rate)), reference, N_STEPS)
    for k in range(N_STEPS + 1):
        torch.testing.assert_close(path[:, k], reference * (1 + rate / N_STEPS) ** k, rtol=1e-4, atol=1e-6)
    k = torch.tensor([1, 3])
    served = reference + 0.1
    at, target = train_user.command_target(path, served, k)
    rows, off = torch.arange(len(reference)), served - reference
    torch.testing.assert_close(at, path[rows, k] + (1 - k / N_STEPS)[:, None, None] * off)
    torch.testing.assert_close(target, rate * path[rows, k] - off, rtol=1e-4, atol=1e-6)


def test_correct_by_finishes_the_closure_early_and_then_follows_the_path():
    """``correct_by`` is the step the correction is complete by: from it on the target is the plain
    field's own step, and before it the closure is shared equally."""
    reference, flow = _reference(), Flow(_constant(0.5))
    path = train_user.reversal_path(flow, reference, N_STEPS)
    served, k = reference + 0.2, torch.tensor([0, N_STEPS - 1])
    at, target = train_user.command_target(path, served, k, correct_by=2)
    rows, off = torch.arange(len(reference)), served - reference
    left = (1 - k / 2).clamp(min=0)[:, None, None]
    after = (1 - (k + 1) / 2).clamp(min=0)[:, None, None]
    torch.testing.assert_close(at, path[rows, k] + left * off)
    torch.testing.assert_close(target, (path[rows, k + 1] - path[rows, k] - (left - after) * off) * N_STEPS)
    torch.testing.assert_close(at[1], path[rows, k][1])  # k >= correct_by: on the path itself
    with pytest.raises(ValueError, match="correct_by"):
        train_user.command_target(path, served, k, correct_by=0)


def test_a_faithful_reference_gives_the_plain_steps_and_the_identity_module_solves_it():
    """With the served reference equal to the intent's, the target is the field's own velocity at ``at``:
    a zero-initialised module (the identity) has zero loss on a constant and on a linear field."""
    for field in (_constant(0.5), _linear(0.3)):
        reference = _reference()
        path = train_user.reversal_path(Flow(field), reference, N_STEPS)
        k = torch.tensor([0, 2])
        at, target = train_user.command_target(path, reference, k)
        rows = torch.arange(len(reference))
        torch.testing.assert_close(at, path[rows, k])
        user = LearnedUserModule(HORIZON, WIDTH, d=8, tdim=8)
        loss = train_user.command_loss(user, field, at, (k / N_STEPS).float(), target)
        assert float(loss.detach()) < 1e-10


# ---------------------------------------------------------------- the loss and the training loop


def _toy_problem(rate: float = 0.5, off: float = 0.3) -> tuple:
    """A reference, its path, and the served reference ``off`` away from it: target = P - off."""
    reference = _reference()
    path = train_user.reversal_path(Flow(_constant(rate)), reference, N_STEPS)
    k = torch.zeros(len(reference), dtype=torch.long)
    at, target = train_user.command_target(path, reference + off, k)
    return at, target, k


def test_the_gradient_reaches_the_module():
    """The loss is a function of the trained module: a non-trivial target's gradient is finite and non-zero."""
    user = LearnedUserModule(HORIZON, WIDTH, d=16, tdim=8)
    at, target, k = _toy_problem()
    loss = train_user.command_loss(user, _constant(0.5), at, (k / N_STEPS).float(), target)
    loss.backward()
    grads = [parameter.grad for parameter in user.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(gradient).all() for gradient in grads)
    assert any(float(gradient.abs().sum()) > 0 for gradient in grads)  # the zero-init head is reachable


def test_the_training_loop_reduces_a_constant_offset_with_a_toy_field():
    """A module fitted on the toy regression's own point: the old ``_train_module`` (AdamW + cosine) must
    take the loss down, and every logged window must be finite.

    A one-step, two-dim chunk keeps the toy's scale small enough for a sign-based optimizer: the target
    (``P - off``) is one constant, and the module's matrices can only close a fraction of it per step.
    """
    reference = torch.tensor([[[0.1, 0.2]]])
    path = train_user.reversal_path(Flow(_constant(0.5)), reference, N_STEPS)
    k = torch.zeros(1, dtype=torch.long)
    at, target = train_user.command_target(path, reference + 0.3, k)
    user = LearnedUserModule(1, 2, d=16, tdim=8)
    losses = []

    def batch_loss(step):
        loss = train_user.command_loss(user, _constant(0.5), at, (k / N_STEPS).float(), target)
        losses.append(float(loss.detach()))
        return loss

    log = train_user._train_module(user, 30, 1e-2, batch_loss, None, 10)
    assert [entry["step"] for entry in log] == [10, 20, 30]
    assert all(np.isfinite(entry["train_loss"]) for entry in log)
    assert log[-1]["train_loss"] < log[0]["train_loss"]
    assert losses[-1] < 0.5 * losses[0]


# ---------------------------------------------------------------- the split


def _rows(episodes: list[int]) -> list[dict]:
    return [{"episode": episode} for episode in episodes]


def test_the_split_holds_out_whole_episodes_when_the_file_has_several():
    """Rows of one episode are one trajectory's neighbours: a whole episode is held out, never half of it."""
    rows = _rows([0, 0, 0, 1, 1, 2])
    train, val = train_user.split_states(rows, 0.2, np.random.default_rng(0))
    assert set(val) & set(train) == set() and sorted(train + val) == list(range(len(rows)))
    val_episodes = {rows[index]["episode"] for index in val}
    train_episodes = {rows[index]["episode"] for index in train}
    assert val_episodes and train_episodes and val_episodes.isdisjoint(train_episodes)
    assert {rows[index]["episode"] for index in range(len(rows))} == val_episodes | train_episodes


def test_the_split_is_a_seeded_row_split_when_the_file_has_one_episode():
    """One episode has no groups to keep: its rows are split by a seeded permutation, reproducibly."""
    rows = _rows([0] * 10)
    first = train_user.split_states(rows, 0.3, np.random.default_rng(7))
    second = train_user.split_states(rows, 0.3, np.random.default_rng(7))
    assert first == second
    train, val = first
    assert len(val) == 3 and len(train) == 7
    with pytest.raises(ValueError, match="two states"):
        train_user.split_states(_rows([0]), 0.5, np.random.default_rng(0))


# ---------------------------------------------------------------- the run, on a fake bundle


class _FakePolicyConfig:
    """pi0.5's sampler interface, as the steering core and the trainer read it."""

    chunk_size = 4
    max_action_dim = CHECKPOINT_WIDTH
    num_inference_steps = 4
    n_action_steps = 2
    noise_time = 1.0
    type = "pi05"
    device = "cpu"


class _LinearPolicy:
    """A base policy whose velocity field is enough for the trainer: no weights, no prefix.

    The two members ``SteeringPolicy``'s sampler contract checks are stubs: the trainer never samples a
    chunk, it only builds the core for its ``reference_chunk`` rule.
    """

    def __init__(self) -> None:
        self.config = _FakePolicyConfig()

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        return torch.zeros(1, self.config.chunk_size, self.config.max_action_dim)

    def velocity_field(self, batch: dict):
        return _linear(0.25)


class _BatchingPreprocessor:
    """The little it takes of the real pre-processor: a prepared observation into a B = 1 tensor batch.

    ``Bundle.prepare`` is the adapter that makes the station's numpy arrays tensors and its prompt a
    one-element list; this fake preprocessor is the next step, so it only adds the batch dimension.
    """

    def __call__(self, raw: dict) -> dict:
        return {
            key: value if isinstance(value, list) else torch.as_tensor(value)[None]
            for key, value in raw.items()
        }


class _StatsStep:
    """An unnormalizer-like step for ``action_mean_std``: action mean/std in raw units."""

    def __init__(self) -> None:
        mean = np.zeros(CHECKPOINT_WIDTH)
        mean[7] = 0.5  # the gripper's statistical mean, in action units
        std = np.full(CHECKPOINT_WIDTH, 0.25)
        std[7] = 0.4
        self.stats = {"action": {"mean": mean, "std": std}}
        self.norm_map = {}


class _Postprocessor:
    """The bits of the real postprocessor the steering core uses: action stats and a callable."""

    def __init__(self) -> None:
        self.steps = [_StatsStep()]

    def __call__(self, action):
        return torch.as_tensor(action)


def _bundle() -> Bundle:
    return Bundle(
        policy=_LinearPolicy(),
        preprocessor=_BatchingPreprocessor(),
        postprocessor=_Postprocessor(),
        mean=np.zeros(8),
        std=np.full(8, 0.25),
    )


def _states(path: Path, episodes: tuple[int, ...] = (0, 1), meta: dict | None = None) -> Path:
    """A states file of ``len(episodes)`` chunk boundaries, written by the real recorder.

    Its meta carries the readout provenance the CLI records (``READOUT_META``), so the trainer's check
    passes; ``meta`` replaces the whole meta for a file collected under another readout or without one.
    """
    recorder = Recorder(
        {"config": "test.toml", "timestamp": "2026-11-25T00:00:00+00:00", **READOUT_META}
        if meta is None
        else {"config": "test.toml", "timestamp": "2026-11-25T00:00:00+00:00", **meta}
    )
    image = np.zeros((3, 224, 224), dtype=np.float32)
    batch = {"observation.images.base_0_rgb": image, "observation.images.left_wrist_0_rgb": image}
    for index, episode in enumerate(episodes):
        recorder.add(
            batch,
            joint_position=JOINT + 0.1 * index,
            gripper=GRIPPER,
            deflection=DEFLECTION,
            served=SERVED,
            q0=Q0,
            episode=episode,
            tick=0,
            prompt=PROMPT,
            mode="steered",
        )
    return recorder.save(path)


def _config(states: Path, out: Path, **user) -> Config:
    return Config(
        readout=ReadoutConfig(corruption={"rotation_z_deg": 20}, noise=0.35),
        modes=MappingProxyType({"frs": SteeringConfig(mode="frs", n_reversal_steps=None, pinned_steps=None)}),
        user=UserConfig(states=states, out=out, cues=user.pop("cues", 1), d=16, tdim=8, **user),
        training=TrainingConfig(steps=3, batch=1, log_every=1, device="cpu", seed=0),
    )


def _fake_bundle(monkeypatch) -> None:
    monkeypatch.setattr(train_user, "load_bundle", lambda config, device=None: _bundle())


def test_train_writes_a_loadable_checkpoint_and_a_log(tmp_path, monkeypatch):
    """The whole recipe on two states: ``user.pt`` loads back as a user module, and the log carries the
    held-out loss at every logged step and the provenance the checkpoint is reproduced against."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    out = tmp_path / "user"
    config = _config(states, out)
    _fake_bundle(monkeypatch)

    path = train_user.train(config, prompt=PROMPT, states_path=states, out=out, device="cpu")

    assert path == out / "user.pt" and path.is_file()
    module, kind, meta = AdapterModules.load_module(path)
    assert kind == "user" and isinstance(module, LearnedUserModule)
    assert module.horizon == _FakePolicyConfig.chunk_size and module.n_dims == ARM_DIMS
    assert meta["prompt"] == PROMPT and meta["states"] == str(states)
    assert meta["states_sha256"] == sha256(states.read_bytes()).hexdigest()
    assert meta["corruption"] == {"rotation_z_deg": 20} and meta["sigma"] == [0.2, 0.5]
    assert meta["cues"] == 1 and meta["module"]["step_rank"] is None and meta["module"]["additive"] is False
    assert meta["training"]["seed"] == 0 and meta["training"]["device"] == "cpu"
    assert meta["session"]["config"] == "test.toml"
    assert meta["readout"]["checked"] is True

    written = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert written["meta"] == meta
    assert [entry["step"] for entry in written["log"]] == [1, 2, 3]
    assert all(isinstance(entry["val_loss"], float) for entry in written["log"])
    assert all(np.isfinite(entry["train_loss"]) for entry in written["log"])


def test_train_reads_a_session_directory_and_refuses_a_missing_states_file(tmp_path, monkeypatch):
    """``[user] states`` is the directory ``--record-states`` writes: the trainer adds ``states.npz``; a
    path with neither names what is missing instead of a bare ``FileNotFoundError``."""
    session = tmp_path / "states" / "pilot"
    states = _states(session / "states.npz")
    out = tmp_path / "user"
    config = _config(session, out)
    _fake_bundle(monkeypatch)

    assert (
        train_user.train(config, prompt=PROMPT, states_path=session, out=out, device="cpu") == out / "user.pt"
    )
    with pytest.raises(ValueError, match="states.npz"):
        train_user.train(config, prompt=PROMPT, states_path=tmp_path / "nowhere", out=out, device="cpu")
    assert states.is_file()


def test_a_correct_by_past_the_checkpoints_schedule_is_named(tmp_path):
    """``[user] correct_by`` is a reversal step of this checkpoint's schedule: a larger one is refused."""
    config = _config(tmp_path / "states" / "pilot", tmp_path / "user", correct_by=99)
    with pytest.raises(ValueError, match="correct_by"):
        train_user._run(config, _bundle(), prompt=PROMPT, device="cpu", q0=Q0)


@pytest.mark.parametrize(
    ("key", "value"),
    [("corruption", {"rotation_z_deg": 30}), ("noise", 0.5), ("seed", 7)],
)
def test_a_states_file_from_another_readout_is_refused_naming_both(tmp_path, key, value):
    """The trainer replays the config's [readout]; a file collected under another corruption, noise
    level or seed is refused before the checkpoint is loaded, with both sides in the message."""
    meta = {**READOUT_META, key: value}
    states = _states(tmp_path / "states" / "pilot" / "states.npz", meta=meta)
    config = _config(states, tmp_path / "user")
    with pytest.raises(ValueError, match="different readout") as error:
        train_user.train(config, prompt=PROMPT, states_path=states, out=tmp_path / "user", device="cpu")
    message = str(error.value)
    assert f"{value!r} in the file" in message
    assert f"{getattr(config.readout, key)!r} in the config" in message


def test_a_configured_q0_must_match_the_files_and_a_matching_one_trains(tmp_path, monkeypatch):
    """The file's q0 is the corruption's lift pose; a configured [command] q0 must reproduce it."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    out = tmp_path / "user"
    config = _config(states, out)
    _fake_bundle(monkeypatch)
    matching = dataclasses.replace(
        config, command=dataclasses.replace(config.command, q0=tuple(float(value) for value in Q0))
    )
    assert (
        train_user.train(matching, prompt=PROMPT, states_path=states, out=out, device="cpu")
        == out / "user.pt"
    )
    other = dataclasses.replace(
        config, command=dataclasses.replace(config.command, q0=tuple(float(value) for value in Q0 + 1.0))
    )
    with pytest.raises(ValueError, match="q0") as error:
        train_user.train(other, prompt=PROMPT, states_path=states, out=tmp_path / "other", device="cpu")
    assert "in the file" in str(error.value) and "as [command] q0" in str(error.value)


def test_a_states_file_without_readout_provenance_warns_once_and_trains(tmp_path, monkeypatch, capsys):
    """A hand-built or older file cannot be compared with the config: its readout is the config's,
    unchecked, and the run says so once."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz", meta={"config": "old.toml"})
    out = tmp_path / "user"
    config = _config(states, out)
    _fake_bundle(monkeypatch)

    train_user.train(config, prompt=PROMPT, states_path=states, out=out, device="cpu")

    assert capsys.readouterr().out.count("records no readout provenance") == 1
    assert AdapterModules.load_module(out / "user.pt")[2]["readout"]["checked"] is False


def test_the_weight_is_the_loops_deadband_not_the_readout_modules_default(tmp_path):
    """A cue served at 0.04 is a cue for the loop ([command] deadband 0.02) and idle for the readout
    module's default (0.05): weighting it out would calibrate on a different active set than the one the
    module must correct."""
    config = _config(tmp_path / "states" / "pilot", tmp_path / "user")
    run = train_user._run(config, _bundle(), prompt=PROMPT, device="cpu", q0=Q0)
    user = LearnedUserModule(run.chunk_size, ARM_DIMS, d=8, tdim=8)
    served = np.full(ARM_DIMS, 0.04)
    assert not bool(train_user.readout.is_active(served))  # the readout module's default: idle
    assert bool(train_user.readout.is_active(served, config.command.deadband))  # the loop's: active
    intent = served + 0.2  # the intent's reference is another command: the target is not the plain field
    path = train_user.reversal_path(
        Flow(_linear(0.25)), run.core.reference_chunk(intent[None]), run.num_steps
    )
    row = {
        "image_exterior": np.zeros((3, 224, 224), dtype=np.uint8),
        "image_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        "joint_position": JOINT,
        "gripper": GRIPPER,
        "served": served[None].copy(),
        "paths": path.float().cpu(),
    }
    k, draws = torch.zeros(1, dtype=torch.long), np.zeros(1, dtype=int)

    loss = train_user._loss(run, user, [row], draws, k)
    assert float(loss.detach()) > 0.0  # the loop's deadband keeps the cue in the regression
    idle = dataclasses.replace(config, command=dataclasses.replace(config.command, deadband=0.05))
    idle_run = train_user._run(idle, _bundle(), prompt=PROMPT, device="cpu", q0=Q0)
    assert float(train_user._loss(idle_run, user, [row], draws, k).detach()) == 0.0  # the default drops it


def test_main_trains_from_the_config_and_lets_the_flags_override_the_paths(tmp_path, monkeypatch):
    """The script entry point: ``--config``/``--prompt`` required, ``[user]`` the default paths."""
    other = tmp_path / "other" / "states.npz"
    _states(other)
    config = tmp_path / "franka.toml"
    config.write_text(
        f'[user]\nstates = "{tmp_path}/ignored"\nout = "{tmp_path}/out"\n\n[modes.frs]\nmode = "frs"\n',
        encoding="utf-8",
    )
    seen = {}

    def fake_train(config, *, prompt, states_path, out, device=None):
        seen.update(prompt=prompt, states_path=Path(states_path), out=Path(out))
        return Path(out) / "user.pt"

    monkeypatch.setattr(train_user, "train", fake_train)
    assert train_user.main(["--config", str(config), "--prompt", PROMPT, "--states", str(other)]) == 0
    assert seen == {"prompt": PROMPT, "states_path": other, "out": tmp_path / "out"}
