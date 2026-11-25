"""``Bundle.prepare``/``Bundle.action``, ``make_steering`` and ``load_bundle``: the policy half, no checkpoint.

The steering-policy tests use a fake base policy that satisfies the sampler contract (a config with
pi0.5's fields, ``predict_action_chunk`` honouring the cores' hooks, ``reset``, ``velocity_field``) and
a fake postprocessor carrying action statistics, so no checkpoint, tokenizer or GPU is needed. The
statistics half (``install_stats`` against real normalization steps) is ``test_stats.py``. The
pre-processor regression tests need only the cached checkpoint's own config and processor JSONs and
openpi's statistics asset; the gated tokenizer is stubbed out, so they run on CPU and skip cleanly when
the cache or the asset is absent. Only the end-to-end test at the bottom needs the real pi0.5: it is
skipped unless the gated PaliGemma tokenizer, CUDA and openpi's droid statistics are all present (it
runs on chopper).
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import policy as policy_module
import pytest
import scene as scene_module
import torch
from config import (
    Config as ExampleConfig,
    PolicyConfig as ExamplePolicyConfig,
    SteeringConfig as ExampleSteering,
)
from policy import (
    ARM_DIMS,
    DEFAULT_DEADBAND,
    REFERENCE_STEP,
    STATION_DIMS,
    Bundle,
    fake_bundle,
    load_bundle,
    make_steering,
    steering_batch,
    teleop_action,
)
from scene import Scene
from station import IMAGE_SIZE, observation_batch
from stats import CHECKPOINT_WIDTH, install_stats

from lerobot.policies.steering import (
    VLM_SCENE_DIM,
    VLM_SCENE_TOKENS,
    AdapterModules,
    LearnedTaskModule,
    SteeringSceneTokensProcessorStep,
)
from lerobot.policies.steering.processor_steering import OBS_SCENE_TOKENS, steering_batch_to_transition
from lerobot.processor import PolicyProcessorPipeline

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
REAL_STATS = EXAMPLE_DIR / "assets" / "droid" / "norm_stats.json"
"""openpi's droid statistics, gitignored: present in a checkout that downloaded the assets."""


# ---------------------------------------------------------------- teleop and the fake bundle


def test_teleop_action_is_the_served_command_and_the_binarized_gripper():
    """DESIGN: teleop's action is ``[served7, 1.0 if closed else 0.0]``, no policy involved."""
    served = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    np.testing.assert_array_equal(teleop_action(served, False), np.append(served, 0.0))
    np.testing.assert_array_equal(teleop_action(served, True), np.append(served, 1.0))
    closed = teleop_action(served, True)
    assert closed.shape == (STATION_DIMS,) and closed.dtype == np.float64
    assert closed[ARM_DIMS] == 1.0 and closed[:ARM_DIMS].tolist() == served.tolist()
    with pytest.raises(ValueError, match="served7"):
        teleop_action(np.zeros(ARM_DIMS - 1), True)


def test_fake_bundle_returns_its_fixed_action_and_ignores_the_command():
    """``--dry-run``'s bundle: a fixed (8,) action whatever the batch and command, in float64."""
    action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])
    bundle = fake_bundle(tuple(action))
    for command in (None, np.zeros(ARM_DIMS), np.full(ARM_DIMS, 0.5)):
        out = bundle.action({"observation.state": np.zeros(CHECKPOINT_WIDTH)}, command)
        np.testing.assert_array_equal(out, action)
        assert out.shape == (STATION_DIMS,) and out.dtype == np.float64
    np.testing.assert_allclose(fake_bundle().action({}), np.array([0.1] * ARM_DIMS + [1.0]))
    bundle = fake_bundle()
    assert bundle.mean is None  # no statistics were installed
    np.testing.assert_allclose(bundle.std, np.full(STATION_DIMS, 0.3))  # the dry run's stand-in scale
    with pytest.raises(ValueError, match="fake action"):
        fake_bundle((0.1,) * ARM_DIMS)


def test_the_module_and_the_fake_bundle_need_neither_torch_nor_lerobot():
    """DESIGN: ``--dry-run`` never imports torch (the checkpoint's path). The fake bundle and its action
    are on that path, so importing this module must not pull torch or lerobot in."""
    code = (
        "import sys, numpy as np; "
        "import policy; "
        "bundle = policy.fake_bundle(); "
        "action = bundle.action({'observation.state': np.zeros(32)}, np.zeros(7)); "
        "assert action.shape == (8,), action.shape; "
        "assert 'torch' not in sys.modules, 'policy imported torch'; "
        "assert 'lerobot' not in sys.modules, 'policy imported lerobot'; "
        "assert 'c3po' not in sys.modules, 'policy imported c3po'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )


# ---------------------------------------------------------------- fakes for the steering policy


class _FakePolicyConfig:
    """pi0.5's sampler interface as the steering policy checks it."""

    chunk_size = 15
    max_action_dim = CHECKPOINT_WIDTH
    num_inference_steps = 10
    n_action_steps = 8
    noise_time = 1.0
    type = "pi05"
    device = "cpu"


class _FakePolicy:
    """A base policy satisfying ``SteeringPolicy``'s sampler contract (``_sampler``).

    ``predict_action_chunk`` starts from zeros and honours the cores' hooks: the x_t hook (flow control
    writes the command into the initial condition) and the noise function (FRS replaces the latent), so
    the steered chunk is the hook's result, deterministic and checkable.
    """

    def __init__(self):
        self.config = _FakePolicyConfig()
        self.chunks = 0
        self.resets = 0
        self.evaluations = 0

    def reset(self) -> None:
        self.resets += 1

    def eval(self):
        self.evaluations += 1
        return self

    def select_action(self, batch: dict) -> torch.Tensor:
        return torch.arange(self.config.max_action_dim, dtype=torch.float32)[None]

    def velocity_field(self, batch: dict):
        return lambda x, time: torch.zeros_like(x)

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        self.chunks += 1
        x_t = torch.zeros(1, self.config.chunk_size, self.config.max_action_dim)
        x_t_hook = kwargs.get("x_t_hook")
        if x_t_hook is not None:
            x_t = x_t_hook(0, 1.0, x_t)
        noise_fn = kwargs.get("noise_fn")
        if noise_fn is not None:
            x_t = noise_fn(self.velocity_field(batch), x_t)
        return x_t


class _FakeStatsStep:
    """An unnormalizer-like step for ``action_mean_std``: action mean/std in raw units."""

    def __init__(self):
        mean = np.zeros(CHECKPOINT_WIDTH)
        mean[7] = 0.5  # the gripper's statistical mean, in action units
        std = np.full(CHECKPOINT_WIDTH, 0.25)
        std[7] = 0.4
        self.stats = {"action": {"mean": mean, "std": std}}
        self.norm_map = {}


class _FakePostprocessor:
    """The bits of the real postprocessors the steering policy uses: action stats and a callable."""

    def __init__(self):
        self.steps = [_FakeStatsStep()]

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(action)


FAKE_ACTION_MEAN = np.asarray(_FakeStatsStep().stats["action"]["mean"])[:ARM_DIMS]
FAKE_ACTION_STD = np.asarray(_FakeStatsStep().stats["action"]["std"])[:ARM_DIMS]

FC = ExampleSteering(mode="flow_control", tau=3, lam=1, ic_gain=2.0, n_reversal_steps=6, pinned_steps=8)
FRS = ExampleSteering(mode="frs", n_reversal_steps=6, pinned_steps=8)


def _prepared_like_the_real_pipeline(raw_batch: dict) -> dict:
    """A stand-in for the real pre-processor: the types and batch dimension it leaves for the policy.

    The real pipeline turns the station's ``(32,)`` state into a ``(1, 32)`` tensor; the steering policy
    reads the batch size from the first tensor in the batch, so a fake preprocessor must add that
    dimension too -- a bare ``(32,)`` tensor would be read as a batch of 32 rows -- or the command's
    ``(1, 7)`` would not match it.
    """
    batch = dict(raw_batch)
    state = batch.get("observation.state")
    if state is not None:
        batch["observation.state"] = torch.as_tensor(np.asarray(state), dtype=torch.float32)[None]
    return batch


def _fake_bundle(*, with_stats: bool = True) -> Bundle:
    """A ``Bundle`` over the fake policy and postprocessor, with 8-wide statistics unless asked otherwise."""
    return Bundle(
        policy=_FakePolicy(),
        preprocessor=_prepared_like_the_real_pipeline,
        postprocessor=_FakePostprocessor(),
        mean=np.zeros(STATION_DIMS) if with_stats else None,
        std=np.full(STATION_DIMS, 0.25) if with_stats else None,
    )


# ---------------------------------------------------------------- make_steering


@pytest.mark.parametrize("mode", ["none", "flow_control", "frs"])
def test_make_steering_builds_the_modes_knobs(mode):
    """Each mode's config reaches the core that reads it: FC's tau/lambda/ic_gain and written dims,
    FRS's reference_step/n_reversal_steps/pinned_steps and its 7-wide command with delegation beyond."""
    bundle = _fake_bundle()
    steering = make_steering(bundle, dataclasses.replace(FC, mode=mode))

    assert steering.config.mode == mode
    assert steering.config.n_dims == ARM_DIMS
    assert steering.config.operator_dims is None
    assert steering.config.base_policy_type == "pi05"
    assert steering.config.device == "cpu"
    assert bundle._cache["steering"] is steering  # made once and cached for every control tick
    assert steering.base_policy is bundle.policy and steering._postprocessor is bundle.postprocessor

    if mode == "none":
        with pytest.raises(ValueError, match="no steering core"):
            _ = steering.core
        return
    if mode == "flow_control":
        core = steering.core
        assert (core.tau, core.lam, core.ic_gain) == (3, 1, 2.0)
        assert core._dims == slice(0, ARM_DIMS)  # the A1 patch: write dims 0..n_dims-1
        assert len(core._mean) == 3
        return
    core = steering.core
    assert core._reference_step == REFERENCE_STEP
    assert core.n_reversal_steps == 6
    assert core.pinned_steps == 8
    assert core.operator_dims is None
    assert core._command_dims == ARM_DIMS
    delegated = core._delegated
    assert delegated.shape == (core._policy.config.chunk_size, CHECKPOINT_WIDTH)
    assert delegated[:, ARM_DIMS:].all()  # the gripper's dim and the padding are the policy's
    assert not delegated[:8, :ARM_DIMS].any()  # the executed prefix is pinned
    assert delegated[8:].all()  # the rest of the chunk is the policy's


def test_make_steering_builds_the_adapter_from_the_resolved_matrix():
    """The adapter fields go straight into the ``SteeringConfig``: the fixed 7x7 F arrives resolved and
    ``SteeringPolicy`` builds the ``AdapterModules`` holder itself; an unset matrix leaves FRS plain."""
    second = np.arange(ARM_DIMS * ARM_DIMS, dtype=np.float64).reshape(ARM_DIMS, ARM_DIMS)
    steering = make_steering(
        _fake_bundle(), dataclasses.replace(FRS, adapter_matrix=tuple(map(tuple, second)))
    )
    assert steering.adapter.matrix.shape == (ARM_DIMS, ARM_DIMS)
    np.testing.assert_allclose(steering.adapter.matrix, second)
    assert steering.config.adapter_matrix == second.tolist()

    plain = make_steering(_fake_bundle(), FRS)
    assert plain.adapter.matrix is None and plain.adapter.kind == "none"

    with pytest.raises(ValueError, match="resolve"):
        make_steering(_fake_bundle(), dataclasses.replace(FRS, adapter_matrix="corruption"))


def test_make_steering_refuses_a_steering_mode_without_statistics():
    """The cores read the postprocessor's action statistics; a bundle without them cannot steer. Mode
    ``none`` is the base policy's own chunk and needs no statistics."""
    bundle = _fake_bundle(with_stats=False)
    for mode in ("flow_control", "frs"):
        with pytest.raises(ValueError, match="action statistics"):
            make_steering(bundle, dataclasses.replace(FC, mode=mode))
    assert make_steering(bundle, dataclasses.replace(FC, mode="none")).config.mode == "none"


# ---------------------------------------------------------------- the bundle's two action paths


class _CountingPreprocessor:
    """A preprocessor stand-in that counts how many times the raw observation goes through it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, raw_batch: dict) -> dict:
        self.calls += 1
        return _prepared_like_the_real_pipeline(raw_batch)


def test_steering_batch_is_the_preprocessor_and_the_caller_adds_the_command():
    """LIBERO's order: the pre-processor makes the batch, then the loop sets ``batch["command"]``; the
    batch helper must not add it (the autonomous path has no command)."""
    seen = []
    raw = {"observation.state": np.zeros(CHECKPOINT_WIDTH)}

    def preprocess(batch):
        seen.append(batch)
        return {"preprocessed": True}

    bundle = Bundle(policy=None, preprocessor=preprocess, postprocessor=None)
    batch = steering_batch(bundle, raw)
    assert batch == {"preprocessed": True} and "command" not in batch
    assert seen[0] is raw


def test_bundle_action_takes_the_first_eight_dims_after_the_postprocessor():
    """The fake policy returns ``0..31`` normalized; the fake postprocessor is the identity, so the
    station's numbers are dims 0-7 of the post-processor's output, float64."""
    out = _fake_bundle().action({"observation.state": np.zeros(CHECKPOINT_WIDTH)})
    np.testing.assert_array_equal(out, np.arange(STATION_DIMS, dtype=np.float64))
    assert out.shape == (STATION_DIMS,) and out.dtype == np.float64


def test_act_takes_a_prepared_batch_and_returns_the_eights():
    """``prepare`` is the pre-processor call, ``act`` is the rest of a tick: the base policy's action
    without a command and the steered one with it, both (8,) float64, no preprocessing in ``act``."""
    bundle = _fake_bundle()
    make_steering(bundle, FRS)
    batch = bundle.prepare({"observation.state": np.zeros(CHECKPOINT_WIDTH)})
    np.testing.assert_array_equal(bundle.act(batch), np.arange(STATION_DIMS, dtype=np.float64))
    out = bundle.act(batch, np.full(ARM_DIMS, 0.5))
    assert out.shape == (STATION_DIMS,) and out.dtype == np.float64
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- the tick's plan


class _FakeSteering:
    """A steering policy whose plan differs from the action it serves, so the study sees which is which."""

    def __init__(self) -> None:
        self.actions = 0
        self.last_policy_action = None

    def select_action(self, batch: dict) -> torch.Tensor:
        self.actions += 1
        self.last_policy_action = torch.full((1, CHECKPOINT_WIDTH), 3.0)
        return torch.full((1, CHECKPOINT_WIDTH), 9.0)

    def reset(self) -> None:
        self.last_policy_action = None


def test_plan8_is_the_steering_policys_plan_not_the_action():
    """A steered fake whose plan (3.0) differs from the action it serves (9.0): ``plan8`` must be the
    former, post-processed exactly as the action was."""
    bundle = _fake_bundle()
    steering = _FakeSteering()
    bundle._cache["steering"] = steering
    action = bundle.act({}, np.zeros(ARM_DIMS))
    assert steering.actions == 1
    np.testing.assert_array_equal(action, np.full(STATION_DIMS, 9.0))
    plan = bundle.plan8
    assert plan.shape == (STATION_DIMS,) and plan.dtype == np.float64
    np.testing.assert_array_equal(plan, np.full(STATION_DIMS, 3.0))
    assert not np.allclose(plan, action)


def test_plan8_is_the_policys_own_chunk_in_a_steered_mode():
    """The real ``SteeringPolicy``: in flow control the action carries the operator's written command
    while the policy's own plan is the unsteered chunk it started from, so the two differ."""
    bundle = _fake_bundle()
    make_steering(bundle, FC)
    batch = bundle.prepare({"observation.state": np.zeros(CHECKPOINT_WIDTH)})
    action = bundle.act(batch, np.full(ARM_DIMS, 0.5))
    plan = bundle.plan8
    np.testing.assert_allclose(plan, np.zeros(STATION_DIMS))  # the fake policy's own chunk
    assert not np.allclose(plan, action)  # the write is in the action, not in the plan


def test_plan8_is_the_action_itself_when_nothing_steered():
    """The autonomous mode has no steering policy: the action *is* the base policy's own chunk step,
    so the plan is that same step, snapshotted for the tick."""
    bundle = _fake_bundle()
    action = bundle.act(bundle.prepare({"observation.state": np.zeros(CHECKPOINT_WIDTH)}))
    np.testing.assert_array_equal(bundle.plan8, action)
    assert bundle.plan8 is not action  # a snapshot, not the caller's array


def test_plan8_is_the_action_for_a_mode_none_steering_table():
    """A ``mode = "none"`` table (the S2 task-module anchor) steers nothing: the steering policy's
    ``last_policy_action`` is the base's own chunk, and that is the action too."""
    bundle = _fake_bundle()
    make_steering(bundle, dataclasses.replace(FRS, mode="none"))
    action = bundle.act({}, np.zeros(ARM_DIMS))
    np.testing.assert_array_equal(bundle.plan8, action)


def test_the_fake_bundle_never_has_a_plan_and_reset_clears_it():
    """``--dry-run``'s fixed action is not a policy plan (None); a new episode drops the last plan."""
    fake = fake_bundle()
    fake.act(fake.prepare({}), np.zeros(ARM_DIMS))
    assert fake.plan8 is None

    bundle = _fake_bundle()
    bundle._cache["steering"] = _FakeSteering()
    bundle.act({}, np.zeros(ARM_DIMS))
    assert bundle.plan8 is not None
    bundle.reset()
    assert bundle.plan8 is None


def test_prepare_once_per_chunk_and_act_per_tick_matches_action_per_tick():
    """The policies read a batch only when their action queue is empty (once per ``n_action_steps``), so
    the loop can prepare once per chunk and act on the ticks in between: the same actions as calling the
    one-shot ``action`` every tick, with one preprocessing instead of ``n_action_steps``."""
    preprocessor = _CountingPreprocessor()
    bundle = Bundle(
        policy=_FakePolicy(),
        preprocessor=preprocessor,
        postprocessor=_FakePostprocessor(),
        mean=np.zeros(STATION_DIMS),
        std=np.full(STATION_DIMS, 0.25),
    )
    make_steering(bundle, FC)
    raw = {"observation.state": np.zeros(CHECKPOINT_WIDTH)}
    command = np.full(ARM_DIMS, 0.5)
    ticks = bundle.policy.config.n_action_steps

    batch = bundle.prepare(raw)
    staged = np.stack([bundle.act(batch, command) for _ in range(ticks)])
    assert preprocessor.calls == 1  # one chunk, one preprocessing, however many ticks it serves
    bundle.reset()
    consecutive = np.stack([bundle.action(raw, command) for _ in range(ticks)])
    assert preprocessor.calls == 1 + ticks  # the one-shot call preprocesses every tick
    np.testing.assert_array_equal(staged, consecutive)


def test_action_is_act_of_prepare():
    """The convenient call is exactly ``act(prepare(raw_batch), command7)`` (two fresh bundles, so both
    read their chunk from the same state)."""
    raw = {"observation.state": np.zeros(CHECKPOINT_WIDTH)}
    command = np.full(ARM_DIMS, 0.5)
    one = _fake_bundle()
    make_steering(one, FRS)
    staged = _fake_bundle()
    make_steering(staged, FRS)
    np.testing.assert_array_equal(one.action(raw, command), staged.act(staged.prepare(raw), command))


def test_bundle_action_with_a_command_flows_the_command_through_flow_control():
    """FC writes ``ic_gain * command`` (its units) into the commanded 7 dims of the chunk's initial
    condition; the fake postprocessor is the identity, so the returned action is that write and dim 7
    (the gripper) was not written."""
    bundle = _fake_bundle()
    make_steering(bundle, FC)
    command = np.full(ARM_DIMS, 0.5)
    out = bundle.action({}, command)
    assert out.shape == (STATION_DIMS,) and out.dtype == np.float64
    np.testing.assert_allclose(out[:ARM_DIMS], 2.0 * command, atol=1e-6)
    assert out[ARM_DIMS] == 0.0


def test_bundle_action_with_a_command_uses_the_frs_reference():
    """FRS at ``reference_step = 1.0``: the command held over the chunk as ``(command - mean) / std``
    over the 7 commanded dims; the fake zero-velocity field leaves the reference in place, and the
    delegated gripper comes back as the policy's (zero) value."""
    bundle = _fake_bundle()
    make_steering(bundle, FRS)
    command = np.full(ARM_DIMS, 0.5)
    out = bundle.action({}, command)
    assert out.shape == (STATION_DIMS,) and out.dtype == np.float64
    np.testing.assert_allclose(out[:ARM_DIMS], (command - FAKE_ACTION_MEAN) / FAKE_ACTION_STD, atol=1e-6)
    assert out[ARM_DIMS] == 0.0


def test_bundle_action_with_a_command_needs_make_steering():
    with pytest.raises(ValueError, match="make_steering"):
        _fake_bundle().action({}, np.zeros(ARM_DIMS))


def test_bundle_reset_starts_a_new_episode():
    """``Bundle.reset`` resets the base policy and the cached steering policy (whose core resets the base
    again, as the LIBERO session does)."""
    bundle = _fake_bundle()
    bundle.reset()
    assert bundle.policy.resets == 1
    make_steering(bundle, FC)
    bundle.reset()
    assert bundle.policy.resets == 3


# ---------------------------------------------------------------- load_bundle, without weights

INSTALLED_MEAN = np.zeros(STATION_DIMS)
INSTALLED_STD = np.full(STATION_DIMS, 0.25)


def _fake_loading(monkeypatch, tmp_path, *, with_stats: bool = True):
    """Patch ``load_bundle``'s three loading seams and the installer it wires, so its composition is
    tested without a checkpoint (``install_stats`` itself is ``test_stats.py``'s, against real steps)."""
    import lerobot.configs.policies as configs_policies
    import lerobot.policies as policies
    import lerobot.policies.factory as policy_factory

    stats = tmp_path / "stats.json"
    stats.write_text("{}", encoding="utf-8")
    cfg = ExampleConfig(
        policy=dataclasses.replace(ExamplePolicyConfig(), stats=stats if with_stats else None)
    )
    fake_config = SimpleNamespace(type="pi05")
    loaded = _FakePolicy()

    def preprocessor(raw_batch):
        return dict(raw_batch)

    postprocessor = SimpleNamespace(steps=[])
    installed = []

    class _PolicyClass:
        @classmethod
        def from_pretrained(cls, path, config=None, **kwargs):
            assert config is fake_config
            return loaded

    def install(preprocessor_, postprocessor_, path):
        installed.append((preprocessor_, postprocessor_, Path(path)))
        return INSTALLED_MEAN, INSTALLED_STD

    monkeypatch.setattr(
        configs_policies.PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, path, **kwargs: fake_config),
    )
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda kind: _PolicyClass)
    monkeypatch.setattr(policies, "make_pre_post_processors", lambda **kwargs: (preprocessor, postprocessor))
    monkeypatch.setattr(policy_module, "install_stats", install)
    return SimpleNamespace(
        cfg=cfg,
        fake_config=fake_config,
        loaded=loaded,
        installed=installed,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def test_load_bundle_puts_the_run_config_on_the_checkpoint_and_installs_stats(tmp_path, monkeypatch):
    ctx = _fake_loading(monkeypatch, tmp_path)
    bundle = load_bundle(ctx.cfg, device="cpu")

    assert ctx.fake_config.pretrained_path == "lerobot/pi05_droid"
    assert ctx.fake_config.device == "cpu"
    assert ctx.fake_config.dtype == "bfloat16"
    assert ctx.fake_config.n_action_steps == 8
    assert ctx.loaded.evaluations == 1  # the policy is evaluated before use
    assert bundle.policy is ctx.loaded
    assert bundle.preprocessor is ctx.preprocessor and bundle.postprocessor is ctx.postprocessor
    # the installer runs once, on both pipelines and on the configured statistics file
    assert len(ctx.installed) == 1
    called_preprocessor, called_postprocessor, called_path = ctx.installed[0]
    assert called_preprocessor is ctx.preprocessor and called_postprocessor is ctx.postprocessor
    assert called_path == ctx.cfg.policy.stats
    np.testing.assert_array_equal(bundle.mean, INSTALLED_MEAN)
    np.testing.assert_array_equal(bundle.std, INSTALLED_STD)
    assert bundle.deadband == DEFAULT_DEADBAND


def test_load_bundle_defaults_to_the_configured_device(tmp_path, monkeypatch):
    ctx = _fake_loading(monkeypatch, tmp_path)
    load_bundle(ctx.cfg)
    assert ctx.fake_config.device == ctx.cfg.policy.device


def test_load_bundle_without_statistics_refuses_the_steering_modes(tmp_path, monkeypatch):
    """DESIGN: ``cfg.policy.stats`` absent means no statistics are installed; the shared modes then
    refuse in ``make_steering`` and ``Bundle.action`` refuses to act in the wrong action space, instead
    of handing the station unnormalized numbers."""
    ctx = _fake_loading(monkeypatch, tmp_path, with_stats=False)
    bundle = load_bundle(ctx.cfg, device="cpu")
    assert bundle.mean is None and bundle.std is None
    assert ctx.installed == []
    with pytest.raises(ValueError, match="action statistics"):
        make_steering(bundle, FRS)
    with pytest.raises(ValueError, match="no action statistics"):
        bundle.action({"observation.state": np.zeros(CHECKPOINT_WIDTH)})


def test_load_bundle_treats_a_missing_statistics_file_as_absent(tmp_path, monkeypatch):
    """The openpi asset is gitignored: a config that names it when it is not on disk is the same
    "no statistics installed" state (teleop does not load this module; the shared modes refuse)."""
    ctx = _fake_loading(monkeypatch, tmp_path)
    cfg = dataclasses.replace(
        ctx.cfg, policy=dataclasses.replace(ctx.cfg.policy, stats=tmp_path / "not-downloaded.json")
    )
    bundle = load_bundle(cfg, device="cpu")
    assert bundle.mean is None and bundle.std is None
    assert ctx.installed == []


# ---------------------------------------------------------------- a task checkpoint's scene

SCENE_SIZE = 128


def _task_checkpoint(path: Path, source: str = "vlm_image") -> Path:
    """A MARS task module checkpoint built with no training: its config is the scene ``load_bundle`` reads."""
    module = LearnedTaskModule(**LearnedTaskModule.spec(source, SCENE_SIZE))
    return AdapterModules.save_module(module, path, kind="task")


class _FakeEncoder:
    """The scene step's encoder contract without the 14 GB VLM: float16 tokens per picture batch."""

    def image_tokens(self, images, batch=64) -> torch.Tensor:
        return torch.zeros((len(images), VLM_SCENE_TOKENS, VLM_SCENE_DIM), dtype=torch.float16)

    def prefix_tokens(self, images, batch=64) -> torch.Tensor:
        raise AssertionError("a vlm_image step reads image_tokens")


def _fake_scene_step(self, *, device, encoder=None) -> PolicyProcessorPipeline:
    """``Scene.step`` with the fake encoder: the real one-step pipeline, no VLM loaded."""
    return PolicyProcessorPipeline(
        steps=[
            SteeringSceneTokensProcessorStep(
                source=self.source, render_size=self.render_size, device=device, encoder=_FakeEncoder()
            )
        ],
        to_transition=steering_batch_to_transition,
    )


def test_load_bundle_builds_the_scene_of_a_task_checkpoint(tmp_path, monkeypatch):
    """A task checkpoint's recorded scene goes on the bundle; without one the bundle has none."""
    ctx = _fake_loading(monkeypatch, tmp_path)
    path = _task_checkpoint(tmp_path / "task.pt")

    bundle = load_bundle(ctx.cfg, device="cpu", task=path)

    assert bundle.scene == Scene(source="vlm_image", render_size=SCENE_SIZE)
    assert load_bundle(ctx.cfg, device="cpu").scene is None


def test_prepare_encodes_the_scene_pictures_before_the_base_preprocessor(tmp_path, monkeypatch):
    """``Bundle.prepare`` runs the scene step first: the batch the base pre-processor sees carries the
    scene's tokens (float16 ``[1, 256, 2048]``) beside the station's state and prompt."""
    ctx = _fake_loading(monkeypatch, tmp_path)
    path = _task_checkpoint(tmp_path / "task.pt")
    monkeypatch.setattr(scene_module.Scene, "step", _fake_scene_step)
    seen = {}

    def base_preprocessor(bundle, raw_batch: dict) -> dict:
        seen["batch"] = raw_batch
        return dict(raw_batch)

    monkeypatch.setattr(policy_module, "steering_batch", base_preprocessor)
    bundle = load_bundle(ctx.cfg, device="cpu", task=path)
    pictures = np.full((3, SCENE_SIZE, SCENE_SIZE), 5, dtype=np.uint8)
    raw_batch = {
        "observation.scene_images": pictures,
        "observation.state": np.zeros(32, dtype=np.float32),
        "task": "pick the cube",
    }

    prepared = bundle.prepare(raw_batch)

    tokens = prepared[OBS_SCENE_TOKENS]
    assert torch.is_tensor(tokens)
    assert tokens.shape == (1, VLM_SCENE_TOKENS, VLM_SCENE_DIM)
    assert tokens.dtype == torch.float16
    # the base pre-processor ran after the scene step, on its output adapted for the pipeline
    assert seen["batch"] is not raw_batch
    np.testing.assert_array_equal(seen["batch"][OBS_SCENE_TOKENS].numpy(), tokens.numpy())
    assert torch.is_tensor(seen["batch"]["observation.state"])
    np.testing.assert_array_equal(seen["batch"]["observation.state"].numpy(), raw_batch["observation.state"])
    assert seen["batch"]["task"] == ["pick the cube"]


def test_prepare_without_a_scene_adapts_the_stations_batch_for_the_preprocessor(tmp_path, monkeypatch):
    """Without a scene the batch is the station's plus ``_policy_batch``'s adapter: the numpy arrays are
    tensors and the bare prompt is a one-element list (the caller's own batch is not mutated)."""
    ctx = _fake_loading(monkeypatch, tmp_path)
    seen = {}

    def base_preprocessor(bundle, raw_batch: dict) -> dict:
        seen["batch"] = raw_batch
        return dict(raw_batch)

    monkeypatch.setattr(policy_module, "steering_batch", base_preprocessor)
    bundle = load_bundle(ctx.cfg, device="cpu")
    raw_batch = {"observation.state": np.zeros(32, dtype=np.float32), "task": "pick"}

    prepared = bundle.prepare(raw_batch)

    assert seen["batch"] is not raw_batch
    assert torch.is_tensor(seen["batch"]["observation.state"])
    assert seen["batch"]["observation.state"].shape == (32,)
    assert seen["batch"]["task"] == ["pick"]
    assert OBS_SCENE_TOKENS not in prepared
    assert raw_batch["task"] == "pick" and isinstance(raw_batch["observation.state"], np.ndarray)


@pytest.mark.parametrize("source", ["cnn", "discs"])
def test_load_bundle_refuses_a_task_checkpoint_whose_scene_is_out_of_scope(tmp_path, source):
    """The scene is read before the checkpoint or the VLM it would need is loaded: an out-of-scope task
    module's source is refused by name (DESIGN.md's Phase C scope)."""
    path = _task_checkpoint(tmp_path / "task.pt", source)
    with pytest.raises(ValueError, match="out of scope"):
        load_bundle(ExampleConfig(), device="cpu", task=path)


# ---------------------------------------------------------------- the station's types, the pipeline's

STATION_CAMERAS = {"exterior": "scene_rgb", "wrist": "wrist_rgb"}
"""The config's camera role -> station feed mapping, as ``station.observation_batch`` reads it."""


def _station_batch() -> dict:
    """One station observation as ``station.observation_batch`` hands it to ``Bundle.prepare``."""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    observation = {
        "follower/joint_position": np.arange(ARM_DIMS, dtype=np.float64) / 10.0,
        "follower/gripper_position": np.array([0.25]),
        "scene_rgb": frame,
        "wrist_rgb": np.flipud(frame),
    }
    return observation_batch(observation, arm="follower", cameras=STATION_CAMERAS, task="pick up the cube")


def test_prepare_adapts_the_stations_numpy_batch_to_the_pipelines_types():
    """The station hands numpy arrays and a bare prompt string; the pre-processor's batch step batches
    only tensors and pi0.5's state tokenizer iterates a list of prompts. ``prepare`` is the adapter: it
    converts the arrays with ``torch.from_numpy`` (keeping their unbatched shapes, which is what the
    batch step then grows) and wraps the prompt in a one-element list."""
    seen = {}
    bundle = Bundle(
        policy=None,
        preprocessor=lambda batch: seen.update(batch=batch) or batch,
        postprocessor=None,
    )
    raw_batch = _station_batch()

    prepared = bundle.prepare(raw_batch)

    adapted = seen["batch"]
    assert adapted is not raw_batch
    for key in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"):
        assert torch.is_tensor(adapted[key])
        assert adapted[key].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert adapted[key].dtype == torch.float32
        np.testing.assert_array_equal(adapted[key].numpy(), raw_batch[key])
    assert torch.is_tensor(adapted["observation.state"])
    assert adapted["observation.state"].shape == (CHECKPOINT_WIDTH,)
    assert adapted["task"] == ["pick up the cube"]
    assert prepared is adapted  # what the pre-processor returns is what prepare returns
    # the adapter copies the dict: the caller's batch and its values are untouched
    assert raw_batch["task"] == "pick up the cube"
    assert isinstance(raw_batch["observation.state"], np.ndarray)


def test_prepare_leaves_an_already_tensor_batch_and_a_prompt_list_alone():
    """The adapter converts only the station's types: a batch already in the pipeline's (tensors, the
    scene step's tokens, a caller's prompt list) goes through untouched, so nothing is batched or
    wrapped twice."""
    image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
    state = torch.zeros(CHECKPOINT_WIDTH)
    tokens = torch.zeros(1, VLM_SCENE_TOKENS, VLM_SCENE_DIM, dtype=torch.float16)
    prompt = ["pick up the cube"]
    raw_batch = {
        "observation.images.base_0_rgb": image,
        "observation.state": state,
        OBS_SCENE_TOKENS: tokens,
        "task": prompt,
    }
    seen = {}
    bundle = Bundle(
        policy=None,
        preprocessor=lambda batch: seen.update(batch=batch) or batch,
        postprocessor=None,
    )

    bundle.prepare(raw_batch)

    adapted = seen["batch"]
    assert adapted["observation.images.base_0_rgb"] is image
    assert adapted["observation.state"] is state
    assert adapted[OBS_SCENE_TOKENS] is tokens
    assert adapted["task"] is prompt


# ---------------------------------------------------------------- the real pre-processors (CPU)

PROCESSOR_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")
"""The checkpoint's own files the pre-processor regression needs: its config and its two pipelines."""


def _real_processors_ready() -> bool:
    """The cached checkpoint's config and processor JSONs and openpi's statistics: no weights needed."""
    if not REAL_STATS.is_file():
        return False
    from huggingface_hub import try_to_load_from_cache

    cached = [try_to_load_from_cache("lerobot/pi05_droid", name) for name in PROCESSOR_FILES]
    return all(isinstance(path, str) for path in cached)


@pytest.fixture
def real_processors(monkeypatch) -> Bundle:
    """The cached checkpoint's real pre/post-processors on CPU, the gated tokenizer stubbed out.

    The stub is what makes this runnable on a machine without the gated PaliGemma tokenizer:
    ``TokenizerProcessorStep.__post_init__`` would load it from the Hub and ``__call__`` would tokenize
    the prompt. The defect is about the batch handed to those two steps, not about token ids, so both
    are replaced by no-ops. The checkpoint's weights are not loaded either: ``policy`` stays ``None``
    (``prepare`` never reads it); the pre-processor is loaded from the cached snapshot directory so
    nothing touches the network.
    """
    if not _real_processors_ready():
        pytest.skip("needs the cached lerobot/pi05_droid processor JSONs and openpi's droid statistics")
    from huggingface_hub import snapshot_download

    import lerobot.processor.tokenizer_processor as tokenizer_processor
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    monkeypatch.setattr(tokenizer_processor.TokenizerProcessorStep, "__post_init__", lambda self: None)
    monkeypatch.setattr(
        tokenizer_processor.TokenizerProcessorStep, "__call__", lambda self, transition: transition
    )
    snapshot = snapshot_download("lerobot/pi05_droid", local_files_only=True)
    policy_cfg = PreTrainedConfig.from_pretrained(snapshot)
    policy_cfg.pretrained_path = snapshot
    policy_cfg.device = "cpu"
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=snapshot,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    mean, std = install_stats(preprocessor, postprocessor, REAL_STATS)
    return Bundle(policy=None, preprocessor=preprocessor, postprocessor=postprocessor, mean=mean, std=std)


def test_the_real_pre_processor_takes_the_stations_raw_batch(real_processors):
    """The chopper probe's section-4 failure: ``prepare`` on ``station.observation_batch``'s batch raised
    ``TypeError: 'numpy.int64' object is not iterable`` in pi0.5's state tokenizer (the state was never
    batched and the prompt was a bare string). Through the real cached pre-processor, ``prepare`` now
    hands the model two ``(1, 3, 224, 224)`` float32 images, a ``(1, 32)`` float32 state, and one prompt
    built from the whole task text rather than one per character."""
    raw_batch = _station_batch()
    assert isinstance(raw_batch["observation.state"], np.ndarray)
    assert raw_batch["observation.state"].shape == (CHECKPOINT_WIDTH,)
    assert raw_batch["observation.images.base_0_rgb"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    prepared = real_processors.prepare(raw_batch)

    for key in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"):
        assert torch.is_tensor(prepared[key])
        assert prepared[key].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert prepared[key].dtype == torch.float32
    state = prepared["observation.state"]
    assert torch.is_tensor(state) and state.shape == (1, CHECKPOINT_WIDTH) and state.dtype == torch.float32
    assert len(prepared["task"]) == 1
    prompt = prepared["task"][0]
    assert prompt.startswith("Task: pick up the cube, State: ")
    assert prompt.endswith(";\nAction: ")


def test_the_real_pre_processor_does_not_batch_a_tensor_batch_twice(real_processors):
    """A batch already in the pipeline's types goes straight through the real pre-processor: the images
    keep one batch dimension (not ``(1, 1, 3, 224, 224)``) and the prompt list stays one prompt."""
    raw_batch = _station_batch()
    batch = {
        key: (torch.from_numpy(value) if isinstance(value, np.ndarray) else [value])
        for key, value in raw_batch.items()
    }

    prepared = real_processors.prepare(batch)

    for key in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"):
        assert prepared[key].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert prepared["observation.state"].shape == (1, CHECKPOINT_WIDTH)
    assert len(prepared["task"]) == 1


# ---------------------------------------------------------------- the real checkpoint (chopper)


def _real_policy_ready() -> bool:
    """The end-to-end test's prerequisites: CUDA, the gated PaliGemma tokenizer and openpi's statistics."""
    if not torch.cuda.is_available():
        return False
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache("google/paligemma-3b-pt-224", "tokenizer_config.json")
    return isinstance(cached, str) and REAL_STATS.is_file()


@pytest.mark.skipif(
    not _real_policy_ready(),
    reason="needs CUDA, the gated PaliGemma tokenizer and openpi's droid statistics (runs on chopper)",
)
def test_the_real_pi05_droid_bundle_acts_and_builds_the_steering_core():
    """The full pipeline: the real checkpoint on the real statistics. [V] on the station; offline it is
    skipped (the weights are not downloaded and the tokenizer is gated)."""
    bundle = load_bundle(ExampleConfig(), device="cuda")
    assert bundle.mean is not None and bundle.mean.shape == (STATION_DIMS,)

    image = np.zeros((3, 224, 224), dtype=np.float32)
    raw_batch = {
        "observation.images.base_0_rgb": image,
        "observation.images.left_wrist_0_rgb": image,
        "observation.state": np.zeros(CHECKPOINT_WIDTH, dtype=np.float32),
        "task": "pick up the cube",
    }
    action = bundle.action(raw_batch, None)
    assert action.shape == (STATION_DIMS,) and action.dtype == np.float64
    assert np.isfinite(action).all()

    steering = make_steering(bundle, FRS)
    assert steering.core.operator_dims is None
    assert steering.core._command_dims == ARM_DIMS
