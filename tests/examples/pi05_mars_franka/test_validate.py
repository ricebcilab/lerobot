"""The user module's offline validator: the metric, the arms, the held-out split, and one end-to-end run.

Offline and hermetic, as ``test_train_user.py``: a fake base policy (linear or zero velocity field, a
config with pi0.5's sampler fields), a fake postprocessor carrying action statistics, a tiny saved
``user.pt``, and a states npz written by ``statefiles``. ``validate.load_bundle`` is monkeypatched and,
for the arm-table test, the steering core is a recording fake so each arm's command and adapter field
can be asserted directly. No CUDA, no checkpoint, no station: the real field, the real statistics and a
real session are verification on chopper only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import bridge
import numpy as np
import pytest
import statefiles
import torch
import train_user
import validate
from config import Config, ReadoutConfig, SteeringConfig, TrainingConfig, UserConfig
from policy import Bundle
from statefiles import Recorder
from stats import CHECKPOINT_WIDTH

from lerobot.policies.steering import AdaptedField, AdapterModules, LearnedUserModule

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
Q0 = JOINT + 0.01
GRIPPER = np.array([1.0])
DEFLECTION = np.array([0.8, -0.7, 0.6, 0.0, 0.0, 0.0])  # a pilot push, past the readout's deadband
PROMPT = "pick the cube"
READOUT_META = {"corruption": {"rotation_z_deg": 20}, "noise": 0.35, "seed": 0}
HORIZON = 2  # the fake checkpoint's chunk_size
WIDTH = CHECKPOINT_WIDTH
MEAN = np.zeros(8)
STD = np.full(8, 0.25)
SERVED = np.array([0.11, -0.12, 0.13, -0.14, 0.15, -0.16, 0.17])
CLEAN = np.array([0.21, -0.22, 0.23, -0.24, 0.25, -0.26, 0.27])
INTENT = np.array([0.31, -0.32, 0.33, -0.34, 0.35, -0.36, 0.37])
MATRIX = np.eye(7) + 0.05  # a fixed 7x7 corruption lift: any invertible matrix serves
TWIST = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------- the metric


def _chunk_of(commands: np.ndarray, steps: int = HORIZON) -> np.ndarray:
    """A normalized (steps, 32) chunk that unnormalizes (MEAN/STD) to ``commands`` in dims 0-6."""
    executed = np.zeros((steps, 8))
    executed[:, :7] = commands
    chunk = np.zeros((steps, WIDTH))
    chunk[:, :8] = (executed - MEAN) / STD
    return chunk


def _twist_of(twist: np.ndarray) -> np.ndarray:
    """The action-unit command a cue twist becomes at JOINT, through the operator's own bridge."""
    return bridge.operator_command(twist, JOINT, 0.2)


def test_the_metric_reads_a_chunk_along_the_intent_as_cosine_one():
    twist = np.array([0.1, 0.05, -0.02, 0.0, 0.0, 0.0])
    chunk = _chunk_of(_twist_of(twist))
    executed = validate.executed_twist(chunk, JOINT, MEAN, STD, HORIZON)
    assert validate.translation_cosine(executed, twist) == pytest.approx(1.0, abs=1e-4)


def test_the_metric_reads_an_orthogonal_chunk_as_cosine_zero():
    chunk = _chunk_of(_twist_of(np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])))
    executed = validate.executed_twist(chunk, JOINT, MEAN, STD, HORIZON)
    assert abs(validate.translation_cosine(executed, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))) < 0.05


def test_the_metric_reads_an_anti_parallel_chunk_as_cosine_minus_one():
    twist = np.array([0.1, 0.02, -0.03, 0.0, 0.0, 0.0])
    chunk = _chunk_of(_twist_of(-twist))
    executed = validate.executed_twist(chunk, JOINT, MEAN, STD, HORIZON)
    assert validate.translation_cosine(executed, twist) == pytest.approx(-1.0, abs=1e-4)


def test_the_metric_is_zero_for_a_chunk_that_does_not_move():
    """A mean over cosines of nothing is undefined: an unmoved chunk reports 0.0, not NaN."""
    executed = validate.executed_twist(np.zeros((HORIZON, WIDTH)), JOINT, MEAN, STD, HORIZON)
    assert np.allclose(executed, 0.0)
    assert validate.translation_cosine(executed, TWIST) == 0.0
    assert validate.translation_cosine(TWIST, np.zeros(6)) == 0.0


def test_the_unexecuted_steps_do_not_count():
    """Only ``chunk[:, :n_action_steps]`` is executed: a different later step changes nothing."""
    twist = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    chunk = _chunk_of(_twist_of(twist))
    chunk[1] = 5.0
    executed = validate.executed_twist(chunk, JOINT, MEAN, STD, n_action_steps=1)
    assert validate.translation_cosine(executed, twist) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------- the fakes


class _FakePolicyConfig:
    """pi0.5's sampler interface, as the steering core reads it."""

    chunk_size = HORIZON
    max_action_dim = WIDTH
    num_inference_steps = 4
    n_action_steps = HORIZON
    noise_time = 1.0
    type = "pi05"
    device = "cpu"


class _ZeroFieldPolicy:
    """A base policy whose velocity field is zero: the executed chunk is exactly the reversed reference.

    That makes the arms' chunks the references themselves, so the metric on them is hand-checkable.
    """

    def __init__(self) -> None:
        self.config = _FakePolicyConfig()

    def reset(self) -> None:
        pass

    def velocity_field(self, batch: dict):
        return lambda x, time: torch.zeros_like(x)

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        size = next(value.shape[0] for value in batch.values() if torch.is_tensor(value))
        config = self.config
        noise = torch.randn(size, config.chunk_size, config.max_action_dim)
        noise_fn = kwargs.get("noise_fn")
        if noise_fn is not None:
            noise = noise_fn(lambda x, time: torch.zeros_like(x), noise)
        return noise  # every Euler step of a zero field adds nothing


class _RecordingCore:
    """A steering core stand-in that records each call's command, adapter field and reversal depth."""

    def __init__(self, plan_chunk: np.ndarray | None = None) -> None:
        self.calls: list[dict] = []
        self.n_reversal_steps: int | None = None
        self._plan = _chunk_of(np.zeros(7) if plan_chunk is None else plan_chunk)

    def steered_chunk(self, batch: dict, commands, field=None) -> tuple[torch.Tensor, torch.Tensor]:
        size = next(value.shape[0] for value in batch.values() if torch.is_tensor(value))
        chunk = np.repeat(self._plan[None], size, axis=0)
        if commands is not None:
            chunk[:, :, :7] = np.asarray(commands)[:, None, :]
        chunk = torch.as_tensor(chunk, dtype=torch.float32)
        self.calls.append(
            {
                "depth": self.n_reversal_steps,
                "commands": None if commands is None else np.asarray(commands, dtype=np.float64),
                "field": field,
            }
        )
        return chunk, chunk

    def calls_for(self, depth: int) -> list[dict]:
        return [call for call in self.calls if call["depth"] == depth]


class _BatchingPreprocessor:
    """The little it takes of the pre-processor: a prepared observation into a B = 1 tensor batch.

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
        mean = np.zeros(WIDTH)
        mean[7] = 0.5
        std = np.full(WIDTH, 0.25)
        std[7] = 0.4
        self.stats = {"action": {"mean": mean, "std": std}}
        self.norm_map = {}


class _Postprocessor:
    """The bits of the real postprocessor the steering core uses: action stats and a callable."""

    def __init__(self) -> None:
        self.steps = [_StatsStep()]

    def __call__(self, action):
        return torch.as_tensor(action)


def _bundle(policy=None) -> Bundle:
    return Bundle(
        policy=_ZeroFieldPolicy() if policy is None else policy,
        preprocessor=_BatchingPreprocessor(),
        postprocessor=_Postprocessor(),
        mean=MEAN,
        std=STD,
    )


# ---------------------------------------------------------------- fixtures


def _states(path: Path, episodes: tuple[int, ...] = (0, 1), meta: dict | None = None) -> Path:
    """A states file of ``len(episodes)`` chunk boundaries, written by the real recorder."""
    recorder = Recorder(
        {
            "config": "test.toml",
            "timestamp": "2026-11-25T00:00:00+00:00",
            **(READOUT_META if meta is None else meta),
        }
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


def _config(states: Path, out: Path, **training) -> Config:
    defaults = {
        "steps": 1,
        "batch": 2,
        "val_frac": 0.5,
        "seed": 0,
        "device": "cpu",
        "val_depths": (2, 1),
    }
    return Config(
        readout=ReadoutConfig(corruption={"rotation_z_deg": 20}, noise=0.35),
        modes=MappingProxyType({"frs": SteeringConfig(mode="frs", n_reversal_steps=None, pinned_steps=None)}),
        user=UserConfig(states=states, out=out, cues=1),
        training=TrainingConfig(**(defaults | training)),
    )


def _checkpoint(path: Path, prompt: str = PROMPT) -> Path:
    """A tiny saved user module: zero-initialised, so it is the identity (plain FRS)."""
    module = LearnedUserModule(HORIZON, 7, d=16, tdim=8)
    return AdapterModules.save_module(module, path, kind="user", meta={"prompt": prompt})


def _fakes(monkeypatch, *, core=None, cues=None, policy=None) -> Bundle:
    """Patch the bundle load and (optionally) the steering core and the cue draws; return the bundle."""
    bundle = _bundle(policy)
    monkeypatch.setattr(validate, "load_bundle", lambda config, device=None: bundle)
    if core is not None:
        monkeypatch.setattr(
            train_user,
            "make_steering",
            lambda bundle, steering: SimpleNamespace(config=SimpleNamespace(mode="frs"), core=core),
        )
    if cues is not None:
        monkeypatch.setattr(train_user, "served_cues", cues)
    return bundle


def _fixed_cues(config, *, q, episode, pool, q0, action_std7, rng, sigma=None, count=None) -> train_user.Cues:
    """One known cue draw: fixed commands and matrix, so the arm table's values are literals."""
    return train_user.Cues(
        twists=TWIST[None],
        intent=INTENT[None],
        served=SERVED[None],
        clean=CLEAN[None],
        matrix=MATRIX,
    )


# ---------------------------------------------------------------- the arms


def test_each_arm_passes_its_own_command_and_adapter_field_to_steered_chunk(tmp_path, monkeypatch):
    """``none`` steers the served command, ``clean`` the faithful one, ``ideal`` the intent, ``f`` adds the
    normalized lift and ``user`` the trained module; ``plan`` steers nothing. Each depth re-seeds the same
    flow noise and the plan is drawn once (it has no command, so no depth changes it)."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz", episodes=(0, 0))
    config = _config(states, tmp_path / "user")
    core = _RecordingCore()
    _fakes(monkeypatch, core=core, cues=_fixed_cues)

    result = validate.validate(
        config, prompt=PROMPT, states_path=states, checkpoint=_checkpoint(tmp_path / "user" / "user.pt")
    )

    assert len(result["held_out"]["indices"]) == 1  # one held-out state: one call per arm per depth
    assert [call["depth"] for call in core.calls] == [None] + [2] * 5 + [1] * 5
    plan = core.calls[0]
    assert plan["commands"] is None and plan["field"] is None
    for depth in (2, 1):
        arms = core.calls_for(depth)
        assert [arm["commands"] is not None for arm in arms] == [True] * 5
        for call, command in zip(arms, (SERVED, CLEAN, INTENT, SERVED, SERVED), strict=True):
            np.testing.assert_array_equal(call["commands"], command[None])
        fixed, user = arms[3]["field"], arms[4]["field"]
        assert isinstance(fixed, AdaptedField) and fixed.user_module is None
        assert fixed.matrix.shape == (1, 7, 7)  # one matrix per state of the batch
        np.testing.assert_allclose(fixed.matrix.numpy()[0], MATRIX, rtol=1e-6)  # uniform std: S^-1 F S = F
        assert isinstance(user, AdaptedField) and user.matrix is None
        assert user.user_module is not None and user.user_module.horizon == HORIZON
        assert all(arm["field"] is None for arm in arms[:3])
    # The plan arm is the policy's own chunk: its cosine is the metric on the core's no-command chunk.
    executed = validate.executed_twist(core._plan, JOINT, MEAN, STD, HORIZON)
    expected = validate.translation_cosine(executed, TWIST)
    assert result["results"]["2"]["plan"]["cosine"] == pytest.approx(expected)
    assert result["results"]["2"]["plan"] == result["results"]["1"]["plan"]  # the same chunk at both depths


# ---------------------------------------------------------------- the held-out split


def test_validate_evaluates_only_the_trainers_held_out_rows(tmp_path, monkeypatch):
    """The split is ``train_user.split_states``'s, with the trainer's seed and ``val_frac``, so the module
    never saw the evaluated states; the JSON records exactly its val indices."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz", episodes=(0, 1, 2, 3))
    config = _config(states, tmp_path / "user", val_frac=0.5)
    core = _RecordingCore()
    _fakes(monkeypatch, core=core, cues=_fixed_cues)

    result = validate.validate(
        config, prompt=PROMPT, states_path=states, checkpoint=_checkpoint(tmp_path / "user" / "user.pt")
    )

    arrays, _ = statefiles.load(states)
    rows = train_user._rows(arrays)
    _, expected = train_user.split_states(rows, config.training.val_frac, np.random.default_rng([0, 0]))
    assert result["held_out"]["indices"] == [int(index) for index in expected]
    assert result["held_out"]["rows"] == len(expected)
    assert result["held_out"]["episodes"] == sorted({rows[index]["episode"] for index in expected})
    assert result["count"] == len(expected)
    assert all(
        result["results"][str(depth)][arm]["count"] == len(expected)
        for depth in (2, 1)
        for arm in validate.ARMS
    )


def test_a_single_episode_file_falls_back_to_the_seeded_row_split(tmp_path, monkeypatch):
    """One episode has no groups to keep: the val half is the seeded permutation's first rows, as
    ``train_user`` documents -- and a different seed holds out different rows."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz", episodes=(0, 0, 0, 0))
    config = _config(states, tmp_path / "user", val_frac=0.5)
    _fakes(monkeypatch, core=_RecordingCore(), cues=_fixed_cues)
    seen = set()
    for seed in (0, 7):
        config = _config(states, tmp_path / "user", val_frac=0.5, seed=seed)
        result = validate.validate(
            config, prompt=PROMPT, states_path=states, checkpoint=_checkpoint(tmp_path / "user" / "user.pt")
        )
        rows = [{"episode": 0} for _ in range(4)]
        _, expected = train_user.split_states(rows, 0.5, np.random.default_rng([seed, 0]))
        assert result["held_out"]["indices"] == [int(index) for index in expected]
        seen.add(tuple(result["held_out"]["indices"]))
    assert len(seen) == 2  # a seeded split, and the seed is the trainer's


# ---------------------------------------------------------------- one end-to-end run


def test_validate_end_to_end_writes_the_json_and_a_dict_with_every_depth_and_arm(tmp_path, monkeypatch):
    """A two-state npz, a fake zero-field bundle and a tiny loaded ``user.pt``: every depth/arm is
    reported, the identity module changes nothing, the ideal arm follows the intent, and the JSON is the
    returned dict."""
    states = _states(tmp_path / "states" / "pilot" / "states.npz", episodes=(0, 1))
    out = tmp_path / "user"
    config = _config(states, out)
    _fakes(monkeypatch)

    result = validate.validate(
        config, prompt=PROMPT, states_path=states, checkpoint=_checkpoint(out / "user.pt")
    )

    path = out / "validation.json"
    assert path.is_file() and json.loads(path.read_text(encoding="utf-8")) == result
    assert result["out"] == str(path)
    assert result["depths"] == [2, 1] and result["arms"] == list(validate.ARMS)
    assert result["count"] == 1 and result["held_out"]["rows"] == 1
    assert result["checkpoint_prompt"] == PROMPT and result["readout"]["checked"] is True
    assert result["module"]["horizon"] == HORIZON and result["module"]["n_dims"] == 7
    assert "m/s" in result["units"] and "rad/s" in result["units"]
    for depth in (2, 1):
        entry = result["results"][str(depth)]
        assert set(entry) == set(validate.ARMS)
        assert all(arm["count"] == 1 for arm in entry.values())
        assert all(arm["cosine"] is not None for arm in entry.values())
        # The zero field makes the executed chunk the reference and the ideal arm the cue's own command.
        assert entry["ideal"]["cosine"] == pytest.approx(1.0, abs=1e-3)
        # A zero-initialised module is the identity, so it is plain FRS on the same command and noise.
        assert entry["user"]["cosine"] == pytest.approx(entry["none"]["cosine"], abs=1e-6)
        # A fixed matrix multiplying a zero field changes nothing either.
        assert entry["f"]["cosine"] == pytest.approx(entry["none"]["cosine"], abs=1e-6)


def test_validate_prints_the_table_with_the_gates_footer(tmp_path, monkeypatch, capsys):
    states = _states(tmp_path / "states" / "pilot" / "states.npz", episodes=(0, 1))
    config = _config(states, tmp_path / "user")
    _fakes(monkeypatch)
    validate.validate(
        config, prompt=PROMPT, states_path=states, checkpoint=_checkpoint(tmp_path / "user" / "user.pt")
    )
    printed = capsys.readouterr().out
    assert "validate-user: 1 held-out of 2 states" in printed
    for arm in validate.ARMS:
        assert f"\n{arm} " in printed or printed.startswith(f"{arm} ")
    assert "gate (the reader's, never an exit status)" in printed
    assert "|user - clean|" in printed


# ---------------------------------------------------------------- refusals


def test_validate_names_a_missing_states_file(tmp_path):
    with pytest.raises(ValueError, match="record-states"):
        validate.validate(
            _config(tmp_path / "states" / "pilot", tmp_path / "user"),
            prompt=PROMPT,
            states_path=tmp_path / "states" / "pilot",
            checkpoint=tmp_path / "user" / "user.pt",
        )


def test_validate_names_a_missing_checkpoint_file(tmp_path):
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    missing = tmp_path / "user" / "user.pt"  # never written
    with pytest.raises(ValueError, match="train-user") as error:
        validate.validate(
            _config(states, tmp_path / "user"), prompt=PROMPT, states_path=states, checkpoint=missing
        )
    assert str(missing) in str(error.value)


def test_validate_names_an_unavailable_cuda_device(tmp_path, monkeypatch):
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    checkpoint = tmp_path / "user" / "user.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not needed: the device is checked first")
    config = _config(states, tmp_path / "user", device="cuda")
    monkeypatch.setattr(validate.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        validate.validate(config, prompt=PROMPT, states_path=states, checkpoint=checkpoint)


def test_validate_refuses_a_checkpoint_trained_on_another_prompt(tmp_path):
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    out = tmp_path / "user"
    with pytest.raises(ValueError, match="trained on prompt") as error:
        validate.validate(
            _config(states, out),
            prompt=PROMPT,
            states_path=states,
            checkpoint=_checkpoint(out / "user.pt", prompt="a different task"),
        )
    assert "a different task" in str(error.value)


def test_validate_refuses_states_from_another_readout(tmp_path):
    states = _states(
        tmp_path / "states" / "pilot" / "states.npz",
        meta={**READOUT_META, "corruption": {"rotation_z_deg": 30}},
    )
    with pytest.raises(ValueError, match="different readout"):
        validate.validate(
            _config(states, tmp_path / "user"),
            prompt=PROMPT,
            states_path=states,
            checkpoint=tmp_path / "user" / "user.pt",
        )


def test_validate_refuses_a_task_module_checkpoint(tmp_path):
    states = _states(tmp_path / "states" / "pilot" / "states.npz")
    out = tmp_path / "user"
    out.mkdir(parents=True)
    from lerobot.policies.steering import LearnedTaskModule

    checkpoint = AdapterModules.save_module(
        LearnedTaskModule(HORIZON, d=8, n_dims=7), out / "task.pt", kind="task"
    )
    with pytest.raises(ValueError, match="not a user module"):
        validate.validate(_config(states, out), prompt=PROMPT, states_path=states, checkpoint=checkpoint)


# ---------------------------------------------------------------- the script entry point


def test_main_lets_the_flags_override_the_configs_paths(tmp_path, monkeypatch):
    """``[user]`` names the defaults; ``--states``/``--checkpoint``/``--out`` override them."""
    config = tmp_path / "franka.toml"
    config.write_text(
        f'[user]\nstates = "{tmp_path}/ignored.states"\n'
        f'checkpoint = "{tmp_path}/ignored.pt"\nout = "{tmp_path}/ignored-out"\n',
        encoding="utf-8",
    )
    seen: dict = {}

    def fake_validate(config, *, prompt, states_path, checkpoint, out=None):
        seen.update(prompt=prompt, states_path=Path(states_path), checkpoint=Path(checkpoint), out=Path(out))
        return {"out": str(Path(out) / "validation.json")}

    monkeypatch.setattr(validate, "validate", fake_validate)
    arguments = [
        "--config",
        str(config),
        "--prompt",
        PROMPT,
        "--states",
        str(tmp_path / "states.npz"),
        "--checkpoint",
        str(tmp_path / "user.pt"),
        "--out",
        str(tmp_path / "out"),
    ]
    assert validate.main(arguments) == 0
    assert seen == {
        "prompt": PROMPT,
        "states_path": tmp_path / "states.npz",
        "checkpoint": tmp_path / "user.pt",
        "out": tmp_path / "out",
    }
