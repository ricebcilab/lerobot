"""``Bundle.action``, ``make_steering`` and ``load_bundle``: the policy half, tested without a checkpoint.

The steering-policy tests use a fake base policy that satisfies the sampler contract (a config with
pi0.5's fields, ``predict_action_chunk`` honouring the cores' hooks, ``reset``, ``velocity_field``) and
a fake postprocessor carrying action statistics, so no checkpoint, tokenizer or GPU is needed. The
statistics half (``install_stats`` against real normalization steps) is ``test_stats.py``. Only the
end-to-end test at the bottom needs the real pi0.5: it is skipped unless the gated PaliGemma tokenizer,
CUDA and openpi's droid statistics are all present (it runs on chopper).
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
from stats import CHECKPOINT_WIDTH

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


def _fake_bundle(*, with_stats: bool = True) -> Bundle:
    """A ``Bundle`` over the fake policy and postprocessor, with 8-wide statistics unless asked otherwise."""
    return Bundle(
        policy=_FakePolicy(),
        preprocessor=lambda raw_batch: dict(raw_batch),
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
    preprocessor, postprocessor = SimpleNamespace(steps=[]), SimpleNamespace(steps=[])
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
