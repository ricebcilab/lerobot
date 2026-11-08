#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SteeringConfig and SteeringPolicy as lerobot's policy contract has them (the cursor_flow precedent): the
feature-less config, the factory entries, lerobot's save/load format with the base policy outside the weights,
the policy's batch forms, and the one path it drives every base through (pi0.5's sampler interface: the cursor's
``CursorFlowPolicy`` and a pi0.5-style stub alike). Its chunks against the cursor example's are in
tests/examples/pi05_mars_cursor/test_steering_policy_cursor.py."""

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.cursor_flow import CursorFlowConfig, CursorFlowPolicy, FlowPolicy
from lerobot.policies.cursor_flow.processor_cursor_flow import make_cursor_flow_pre_post_processors
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.steering import (
    AdapterModules,
    Flow,
    FlowControlSteering,
    FlowReversalSteering,
    LearnedTaskModule,
    LearnedUserModule,
    SteeringConfig,
    SteeringPolicy,
)


def _obs(b=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, 4, dtype=torch.bool)
    goal = torch.zeros(b, 4, dtype=torch.bool)
    goal[:, 0] = True
    return {
        "pos": torch.rand(b, 2, generator=g),
        "hist": 0.01 * torch.randn(b, 4, 2, generator=g),
        "targets": torch.rand(b, 4, 2, generator=g),
        "mask": mask,
        "goal": goal,
        "obstacles": torch.rand(b, 2, 3, generator=g) * torch.tensor([1.0, 1.0, 0.1]),
        "obstacle_mask": torch.ones(b, 2, dtype=torch.bool),
    }


def _modules(tmp_path, horizon=16):
    torch.manual_seed(0)
    user = LearnedUserModule(horizon, n_dims=2, d=16, tdim=8)
    task = LearnedTaskModule(horizon=horizon, d=16, tdim=8, heads=2, layers=1)
    for m in (user, task):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn(p.shape))
    return (
        AdapterModules.save_module(user, tmp_path / "u.pt", "user"),
        AdapterModules.save_module(task, tmp_path / "t.pt", "task"),
    )


def _policy():
    """A small cursor base: a ``FlowPolicy`` behind the sampler contract (``CursorFlowPolicy.wrapping``)."""
    torch.manual_seed(0)
    return CursorFlowPolicy.wrapping(
        FlowPolicy(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123).eval()
    )


def _post(base):
    """The cursor base's postprocessor: its action statistics (mean 0, std ``action_scale``)."""
    return make_cursor_flow_pre_post_processors(base.config)[1]


def _steering(config, base=None):
    """A steering policy over ``base`` (a fresh ``_policy()`` by default) and its postprocessor."""
    base = _policy() if base is None else base
    return SteeringPolicy(config, base_policy=base, postprocessor=_post(base))


COMMAND = torch.tensor([[0.6, -0.2], [0.0, 0.0], [-0.4, 0.8]])


def test_the_config_has_the_feature_less_contract():
    config = SteeringConfig(device="cpu")
    assert config.type == "steering" and isinstance(
        PreTrainedConfig.get_choice_class("steering")(), SteeringConfig
    )
    assert config.observation_delta_indices is None and config.action_delta_indices is None
    assert config.reward_delta_indices is None and config.validate_features() is None
    assert isinstance(config.get_optimizer_preset(), AdamWConfig) and config.get_scheduler_preset() is None
    assert config.input_features == {} and config.output_features == {} and config.normalization_mapping == {}
    for bad in (
        {"mode": "teleop"},
        {"scene_source": "nope"},
        {"token_geometry": "nope"},
        {"operator_dims": ["wrist"]},
    ):
        with pytest.raises(ValueError):
            SteeringConfig(**bad)


def test_the_factory_knows_the_policy():
    assert get_policy_class("steering") is SteeringPolicy
    config = make_policy_config("steering", mode="flow_control", device="cpu")
    assert isinstance(config, SteeringConfig) and config.mode == "flow_control"


def test_the_policy_wraps_the_mode_core_over_the_base_policy(tmp_path):
    base = _policy()
    for mode, core in (("flow_control", FlowControlSteering), ("frs", FlowReversalSteering)):
        steering = _steering(SteeringConfig(mode=mode, device="cpu"), base)
        assert isinstance(steering.core, core) and steering.core._policy is base
        assert base.config.noise_time == 0.0
        assert (
            steering.state_dict() == {} and list(steering.get_optim_params()) == []
        )  # the base is not a weight
        torch.manual_seed(1)
        chunk = steering.predict_action_chunk({**_obs(), "command": COMMAND})
        assert chunk.shape == (3, 16, 2)
        torch.manual_seed(1)
        plain = base.predict_action_chunk(_obs())
        assert torch.equal(chunk[1], plain[1]) and not torch.equal(
            chunk[0], plain[0]
        )  # the idle row is the policy's
        steering.reset()
    with pytest.raises(ValueError, match="flow control writes one group"):
        _ = SteeringPolicy(
            SteeringConfig(mode="flow_control", operator_dims=["translation", "rotation"]), base_policy=base
        ).core


def test_save_pretrained_carries_the_adapter_and_from_pretrained_rebuilds_it(tmp_path):
    user_path, task_path = _modules(tmp_path)
    f = np.array([[0.9, -0.2], [0.3, 1.1]])
    config = SteeringConfig(
        adapter_matrix=f, adapter_user=str(user_path), adapter_task=str(task_path), device="cpu"
    )
    steering = _steering(config)
    assert config.user_module == AdapterModules.load(user_path).user.config  # recorded when read
    user_keys = {f"adapter.user.{k}" for k in AdapterModules.load(user_path).user.state_dict()}
    task_keys = {f"adapter.task.{k}" for k in AdapterModules.load(task_path).task.state_dict()}
    assert set(steering.state_dict()) == user_keys | task_keys  # the adapter's weights only, the base's not
    steering.save_pretrained(tmp_path / "steering")
    user_path.unlink()
    task_path.unlink()  # the saved policy no longer needs the module checkpoints
    base = _policy()
    loaded = SteeringPolicy.from_pretrained(
        tmp_path / "steering", base_policy=base, postprocessor=_post(base)
    )
    assert np.array_equal(loaded.config.fixed_matrix, f) and loaded.config.user_module == config.user_module
    assert loaded.state_dict().keys() == steering.state_dict().keys()
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in steering.state_dict().items())
    assert not any(
        p.requires_grad for p in loaded.parameters()
    )  # served frozen, as load_module serves a module
    for mode in ("flow_control", "frs"):
        steering.config.mode = loaded.config.mode = mode
        for policy in (steering, loaded):
            base = _policy()
            policy.set_base_policy(base, _post(base))
        torch.manual_seed(2)
        want = steering.predict_action_chunk({**_obs(), "command": COMMAND})
        torch.manual_seed(2)
        assert torch.equal(loaded.predict_action_chunk({**_obs(), "command": COMMAND}), want)


def test_the_base_policy_loads_from_the_configs_reference(tmp_path):
    """A cursor base saved with its pipelines (the factory's) loads with them: FRS reads its postprocessor."""
    model = _policy()
    base = CursorFlowPolicy(CursorFlowConfig(**model.model.config, action_scale=0.0123, device="cpu"))
    base.model.load_state_dict(model.model.state_dict())
    base.save_pretrained(tmp_path / "base")
    for pipeline in make_cursor_flow_pre_post_processors(base.config):
        pipeline.save_pretrained(tmp_path / "base")
    steering = SteeringPolicy(
        SteeringConfig(base_policy_type="cursor_flow", base_policy_path=str(tmp_path / "base"), device="cpu")
    )
    assert isinstance(steering.base_policy, CursorFlowPolicy)
    torch.manual_seed(4)
    got = steering.predict_action_chunk({**_obs(), "command": COMMAND})
    torch.manual_seed(4)
    assert torch.equal(
        got,
        _steering(SteeringConfig(device="cpu"), model).predict_action_chunk({**_obs(), "command": COMMAND}),
    )
    with pytest.raises(ValueError, match="needs a base policy"):
        _ = SteeringPolicy(SteeringConfig(device="cpu")).base_policy


def test_observation_has_one_rule(tmp_path):
    """The base gets the batch untouched less the steering policy's own keys, a cursor batch under its prefix and a
    pi0.5 batch alike; the adapter's view strips the prefix and puts ``observation.scene_tokens`` at
    ``scene[scene_source]`` (the task module reads the cursor's dict), so a prefixed batch steers as the plain one."""
    steering = _steering(SteeringConfig(scene_source="vlm_image", device="cpu"))
    obs = _obs()
    prefixed = {f"observation.{k}": v for k, v in obs.items()}
    tokens = torch.zeros(3, 256, 8, dtype=torch.float16)
    extra = {"observation.layouts": object(), "action": None, "next.reward": 0.0}
    batch = {**prefixed, "observation.scene_tokens": tokens, **extra, "command": COMMAND, "kind": "task"}
    seen = steering.observation(batch)
    assert list(seen) == [*prefixed, "observation.scene_tokens", *extra] and all(
        seen[k] is batch[k] for k in seen
    )
    pi05 = _pi05_batch()
    assert list(steering.observation({**pi05, "kind": "user"})) == [k for k in pi05 if k != "command"]
    view = steering._adapter_view(seen)
    assert list(view) == [*obs, "action", "next.reward", "scene"] and view["scene"] == {"vlm_image": tokens}
    assert view["pos"] is obs["pos"] and view["scene"]["vlm_image"] is tokens
    nameless = _steering(SteeringConfig(device="cpu"))
    with pytest.raises(ValueError, match="needs a scene_source"):
        nameless._adapter_view(nameless.observation({**prefixed, "observation.scene_tokens": tokens}))
    _, task_path = _modules(tmp_path)
    steering = _steering(SteeringConfig(adapter_task=str(task_path), device="cpu"))
    torch.manual_seed(5)
    want = steering.predict_action_chunk({**obs, "command": COMMAND})
    torch.manual_seed(5)
    assert torch.equal(steering.predict_action_chunk({**prefixed, "command": COMMAND.numpy()}), want)


def test_the_cursor_base_goes_through_the_same_path():
    """No cursor branch: the core is built on the base itself, FRS's reference is the held command in the
    postprocessor's units (``Flow.held_reference``'s numbers bit for bit), and a base without the contract is refused
    by name."""
    names = ("_cursor_base", "_flow_model", "_action_scale", "_reference")
    base = _policy()
    steering = _steering(SteeringConfig(device="cpu"), base)
    assert not any(hasattr(SteeringPolicy, n) or hasattr(steering, n) for n in names)
    step = steering.config.max_speed * steering.config.dt
    assert steering.core._policy is base and steering.core._reference_step == step
    per_step = torch.rand(3, 16, 2) * 2 - 1
    for command in (COMMAND, per_step):
        want = Flow.held_reference(command, 16, step, base.action_scale)
        assert torch.equal(steering.core.reference_chunk(command), want)
    with pytest.raises(TypeError, match="velocity_field"):
        SteeringPolicy(SteeringConfig(device="cpu"), base_policy=_SamplerBase())(
            {"observation.state": torch.zeros(2, 8), "kind": "user"}
        )
    with pytest.raises(
        TypeError, match=r"no predict_action_chunk, reset, config.num_inference_steps.*wrapping"
    ):
        _ = SteeringPolicy(SteeringConfig(device="cpu"), base_policy=base.model).core  # a bare FlowPolicy


def test_mode_none_ignores_the_command(tmp_path):
    """Mode none: the base's own chunk through the task module's velocity hook, whatever the command; no core."""
    _, task_path = _modules(tmp_path)
    base = _policy()
    steering = _steering(SteeringConfig(mode="none", adapter_task=str(task_path), device="cpu"), base)
    with pytest.raises(ValueError, match="mode 'none' has no steering core"):
        _ = steering.core
    obs = _obs()
    torch.manual_seed(3)
    got = steering.predict_action_chunk({**obs, "command": COMMAND})
    hook = FlowControlSteering.velocity_hook(steering.adapter.field(obs))
    torch.manual_seed(3)
    want = base.predict_action_chunk(obs, velocity_hook=hook)
    torch.manual_seed(3)
    assert torch.equal(got, want) and not torch.equal(got, base.predict_action_chunk(obs))
    torch.manual_seed(3)
    assert torch.equal(steering.select_action({**obs, "command": COMMAND}), want[:, 0])
    with pytest.raises(ValueError, match=r"command of shape \(2, 2\) for an observation batch of 3"):
        steering.predict_action_chunk({**obs, "command": COMMAND[:2]})  # checked though ignored


def test_forward_needs_the_module_of_its_kind(tmp_path):
    user_path, task_path = _modules(tmp_path)
    steering = _steering(SteeringConfig(adapter_user=str(user_path), device="cpu"))
    with pytest.raises(ValueError, match="needs a task module"):
        steering({**_obs(), "kind": "task", "chunk": torch.zeros(3, 16, 2)})
    with pytest.raises(ValueError, match="'task' or 'user'"):
        steering({**_obs(), "kind": "other"})
    fresh = SteeringConfig(
        user_module=dict(LearnedUserModule(16, n_dims=2, d=16, tdim=8).config), device="cpu"
    )
    trainable = _steering(fresh)
    loss, parts = trainable(
        {**_obs(), "kind": "user", "latent": torch.randn(3, 16, 2), "bci_target": torch.randn(3, 16, 2)}
    )
    loss.backward()
    assert parts["field"] == loss.item() and all(p.grad is not None for p in trainable.get_optim_params())
    with pytest.raises(ValueError, match="holds a task module"):
        SteeringPolicy(SteeringConfig(adapter_user=str(task_path), device="cpu"))


class _SamplerBase:
    """A pi0.5-style base: no ``sample_chunk``, pi0.5's sampler interface itself (no ``noise_time`` in its config:
    pi0.5's 1.0), a fixed linear field Euler-integrated from t = 1 as ``PI05Pytorch.sample_actions`` integrates, every
    hook it gets applied; it records the batches and the names of the hooks it gets."""

    def __init__(self):
        from types import SimpleNamespace

        self.config = SimpleNamespace(num_inference_steps=4, chunk_size=6, max_action_dim=8, n_action_steps=3)
        self.batches, self.hooks = [], []

    def reset(self):
        pass

    @staticmethod
    def velocity(x_t, t):
        t = torch.as_tensor(t, dtype=x_t.dtype)
        return 0.5 * x_t - 0.3 * (t.reshape(-1, 1, 1) if t.dim() else t)

    def predict_action_chunk(self, batch, **hooks):
        self.batches.append(batch)
        self.hooks.append(sorted(k for k, v in hooks.items() if v is not None))
        noise = hooks.get("noise")
        x = torch.randn(len(batch["observation.state"]), 6, 8) if noise is None else noise
        if hooks.get("noise_fn") is not None:
            x = hooks["noise_fn"](self.velocity, x)
        start = 1.0 if hooks.get("flow_start_time") is None else float(hooks["flow_start_time"])
        n = 4 if hooks.get("num_forward_steps") is None else int(hooks["num_forward_steps"])
        dt = -start / n
        for step in range(n):
            t = start + step * dt
            if hooks.get("x_t_hook") is not None:
                x = hooks["x_t_hook"](step, t, x)
            v = self.velocity(x, t)
            if hooks.get("velocity_hook") is not None:
                v = hooks["velocity_hook"](step, t, x, v)
            x = x + dt * v
        return x


def _pi05_batch():
    return {
        "observation.state": torch.zeros(2, 8),
        "observation.images.top": torch.zeros(2, 3, 4, 4),
        "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
        "task": ["pick", "place"],
        "command": torch.tensor([[0.5, -0.2, 0.1], [0.0, 0.0, 0.0]]),
    }


def _pi05_postprocessor(dims=8):
    """A pi0.5-style postprocessor: MEAN_STD statistics over ``dims`` action dims (by default the 8 padded ones, 7 of
    them the env's)."""
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        PolicyProcessorPipeline,
        UnnormalizerProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    stats = {
        "action": {"mean": 0.1 * torch.arange(float(dims)), "std": 0.5 + 0.25 * torch.arange(float(dims))}
    }
    step = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(dims,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats=stats,
    )
    return PolicyProcessorPipeline(
        steps=[step], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )


# the hooks the stub got, recorded before the steering policy had one path (21b5ee8c): the steered pass, then the
# policy's own plan for the idle row
PI05_HOOKS = {
    "flow_control": [["velocity_hook", "x_t_hook"], ["velocity_hook"]],
    "frs": [["noise_fn", "velocity_hook"], ["velocity_hook"]],
}


@pytest.mark.parametrize("mode", ["flow_control", "frs"])
def test_a_pi05_style_stub_is_driven_unchanged(tmp_path, mode):
    """A pi0.5-style base gets lerobot's batch as it is and the hooks it got before (``PI05_HOOKS``), and its steered
    chunk is the mode's core run directly over it (what ``SteeringPolicy`` ran for such a base before), its task module
    in every pass."""
    torch.manual_seed(0)
    task = LearnedTaskModule(horizon=6, d=16, tdim=8, heads=2, layers=1, n_dims=3)
    with torch.no_grad():
        for p in task.parameters():
            p.add_(0.05 * torch.randn(p.shape))
    path = AdapterModules.save_module(task, tmp_path / "t.pt", "task")
    config = SteeringConfig(
        mode=mode, operator_dims=["translation"], n_dims=3, adapter_task=str(path), device="cpu"
    )
    base, post = _SamplerBase(), _pi05_postprocessor()
    steering = SteeringPolicy(config, base_policy=base, postprocessor=post)
    batch = _pi05_batch()
    torch.manual_seed(1)
    chunk = steering.predict_action_chunk(batch)
    assert chunk.shape == (2, 6, 8)  # normalized: the postprocessor unnormalizes a pi0.5 chunk
    keys = ["observation.state", "observation.images.top", "observation.language.tokens", "task"]
    assert base.hooks == PI05_HOOKS[mode] and all(list(b) == keys for b in base.batches)
    obs = {k: v for k, v in batch.items() if k != "command"}
    if mode == "flow_control":
        core = FlowControlSteering(base, None, post, tau=config.tau, lam=config.lam, ic_gain=config.ic_gain)
    else:
        core = FlowReversalSteering(base, None, post, operator_dims=("translation",), pinned_steps=None)
    torch.manual_seed(1)
    want = core.steered_chunk(obs, batch["command"], field=steering.adapter.field(obs), paired=False)[0]
    assert torch.equal(chunk, want) and not torch.equal(chunk[0], chunk[1])


def test_the_reference_rule_is_the_action_spaces():
    """FRS's default reference is the action space's rule: in the 7-D operator space (pi0.5's statistics, no
    reference_step from the policy) the uniform chunk; without the gripper dim the held command at ``reference_step``,
    zero beyond the commanded dims; there, without a reference_step, a named error."""
    command = torch.tensor([[0.5, -0.2, 0.1], [0.0, 0.3, 0.0]])
    config = SteeringConfig(operator_dims=["translation"], n_dims=3, device="cpu")
    core = SteeringPolicy(config, base_policy=_SamplerBase(), postprocessor=_pi05_postprocessor(7)).core
    assert core._reference_step is None
    want = Flow.uniform_reference(command, core._gripper_ref, core._mean, core._std, 6, 8)
    assert torch.equal(core.reference_chunk(command), want)
    three, two = _pi05_postprocessor(3), command[:, :2]
    held = FlowReversalSteering(
        _SamplerBase(), None, three, operator_dims=None, pinned_steps=None, reference_step=0.5
    )
    reference = held.reference_chunk(two)
    mean, std = 0.1 * torch.arange(2.0), 0.5 + 0.25 * torch.arange(2.0)
    assert reference.shape == (2, 6, 8) and not reference[..., 2:].any()
    assert torch.equal(reference[..., :2], (two[:, None].expand(-1, 6, -1) * 0.5 - mean) / std)
    bare = FlowReversalSteering(_SamplerBase(), None, three, operator_dims=None, pinned_steps=None)
    with pytest.raises(ValueError, match="needs a reference_step or a reference_fn"):
        bare.reference_chunk(two)


class _FieldBase(_SamplerBase):
    """The stub with the contract's ``velocity_field``: its fixed field, whatever the batch."""

    def velocity_field(self, batch):
        return self.velocity


def test_a_task_batch_is_normalized_by_the_statistics_it_needs(tmp_path):
    """``forward``'s task chunk (action units) is normalized by the postprocessor's statistics over its n dims (a
    pi0.5 base's translation: the first 3 of 7); without a postprocessor it is refused by name, not misscaled."""
    torch.manual_seed(0)
    task = LearnedTaskModule(horizon=6, d=16, tdim=8, heads=2, layers=1, n_dims=3)
    path = AdapterModules.save_module(task, tmp_path / "t.pt", "task")
    config = SteeringConfig(
        operator_dims=["translation"], n_dims=3, adapter_task=str(path), lambda_id=0.01, device="cpu"
    )
    steering = SteeringPolicy(config, base_policy=_FieldBase(), postprocessor=_pi05_postprocessor(7))
    batch = {k: v for k, v in _pi05_batch().items() if k != "command"}
    g = torch.Generator().manual_seed(2)
    chunk, z, t = (
        torch.randn(2, 6, 3, generator=g),
        torch.randn(2, 6, 3, generator=g),
        torch.rand(2, generator=g),
    )
    loss, parts = steering({**batch, "kind": "task", "chunk": chunk, "noise": z, "time": t})
    normalized = (chunk - 0.1 * torch.arange(3.0)) / (0.5 + 0.25 * torch.arange(3.0))
    bound = steering.adapter.task.bind(steering._adapter_view(steering.observation(batch)))
    want, want_parts = LearnedTaskModule.field_loss(bound, _SamplerBase.velocity, normalized, z, t, 0.01, 3)
    assert torch.equal(loss, want) and parts == {k: v.item() for k, v in want_parts.items()}
    with pytest.raises(
        ValueError, match=r"a task batch's chunk .*base_policy_path.*policy_postprocessor\.json"
    ):
        SteeringPolicy(config, base_policy=_FieldBase())({**batch, "kind": "task", "chunk": chunk})


def test_select_action_serves_the_chunks_executed_steps_one_per_call():
    base = _policy()
    base.config.n_action_steps = 3  # the base's: the contract's field
    steering = _steering(SteeringConfig(device="cpu"), base)
    batch = {**_obs(), "command": COMMAND}
    torch.manual_seed(6)
    chunk = steering.predict_action_chunk(batch)
    torch.manual_seed(6)
    served = [steering.select_action(batch) for _ in range(3)]
    assert all(torch.equal(a, chunk[:, i]) for i, a in enumerate(served))
    torch.manual_seed(7)
    fresh = steering.predict_action_chunk(batch)
    torch.manual_seed(7)
    assert torch.equal(steering.select_action(batch), fresh[:, 0])  # the queue ran out: a new chunk
    steering.reset()
    torch.manual_seed(6)
    assert torch.equal(steering.select_action(batch), chunk[:, 0]) and len(steering._action_queue) == 2


# ---------------------------------------------------------------- off the cursor path


class _FolderBase(torch.nn.Module):
    """A pi0.5-style base saved in a folder: pi0.5's sampler interface (a small PI05Config), no ``sample_chunk``."""

    def __init__(self):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config

        super().__init__()
        self.config = PI05Config(
            device="cpu", chunk_size=6, n_action_steps=3, max_action_dim=8, num_inference_steps=4
        )

    @classmethod
    def from_pretrained(cls, path):
        assert (Path(path) / "stub.txt").is_file(), path
        return cls()

    def reset(self):
        pass

    def predict_action_chunk(self, batch, **hooks):
        noise = torch.zeros(len(batch["observation.state"]), 6, 8)
        if "noise_fn" in hooks:  # FRS: the reversed reference
            noise = hooks["noise_fn"](lambda x, t: torch.ones_like(x), noise)
        return noise


def _base_folder(root, with_processors):
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import (
        DeviceProcessorStep,
        PolicyProcessorPipeline,
        UnnormalizerProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    root.mkdir(parents=True)
    (root / "stub.txt").write_text("a base")
    if with_processors:
        stats = {"action": {"mean": torch.zeros(8), "std": torch.full((8,), 2.0)}}
        features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu")], name=POLICY_PREPROCESSOR_DEFAULT_NAME
        ).save_pretrained(root)
        PolicyProcessorPipeline(
            steps=[
                UnnormalizerProcessorStep(
                    features=features, norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD}, stats=stats
                )
            ],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ).save_pretrained(root)
    return str(root)


def test_a_base_loaded_from_its_folder_brings_its_processors_for_frs(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.policies.factory.get_policy_class", lambda name: _FolderBase)
    config = SteeringConfig(
        base_policy_type="stub",
        base_policy_path=_base_folder(tmp_path / "base", True),
        operator_dims=["translation"],
        n_dims=3,
        device="cpu",
    )
    steering = SteeringPolicy(config)
    batch = {
        "observation.state": torch.zeros(2, 8),
        "task": ["a", "b"],
        "command": torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    chunk = steering.predict_action_chunk(batch)
    assert chunk.shape == (2, 6, 8) and steering.base_preprocessor is not None
    np.testing.assert_allclose(steering.core._std, np.full(7, 2.0))  # the folder's action statistics
    bare = SteeringPolicy(
        SteeringConfig(
            base_policy_type="stub", base_policy_path=_base_folder(tmp_path / "bare", False), device="cpu"
        )
    )
    with pytest.raises(ValueError, match=r"set_base_policy\(policy, postprocessor=\.\.\.\)"):
        _ = bare.core
    with pytest.raises(FileNotFoundError, match="missing"):
        _ = SteeringPolicy(
            SteeringConfig(base_policy_path=str(tmp_path / "missing" / "base"), device="cpu")
        ).base_policy


def test_a_trainer_batch_through_the_pre_pipeline_reaches_forward_unchanged(tmp_path):
    from lerobot.policies.steering import make_steering_pre_post_processors

    user_path, task_path = _modules(tmp_path)
    config = SteeringConfig(adapter_user=str(user_path), adapter_task=str(task_path), device="cpu")
    steering = _steering(config)
    pre, _ = make_steering_pre_post_processors(config)
    g = torch.Generator().manual_seed(8)
    obs, prefixed = _obs(), {f"observation.{k}": v for k, v in _obs().items()}
    target = 0.01 * torch.randn(
        3, 16, 2, generator=g
    )  # every entry inside the deadband: a target, not a command
    user = {
        "kind": "user",
        "latent": torch.randn(3, 16, 2, generator=g),
        "bci_target": target,
        "weight": torch.ones(3),
        "time": torch.rand(3, generator=g),
    }
    task = {
        "kind": "task",
        "chunk": 0.01 * torch.randn(3, 16, 2, generator=g),
        "noise": torch.randn(3, 16, 2, generator=g),
        "time": torch.rand(3, generator=g),
    }
    for training in (user, task):
        batch = pre({**prefixed, **training})
        assert all(
            batch[k] is training[k] or torch.equal(batch[k], training[k]) for k in training if k != "kind"
        )
        got, want = steering(batch), steering({**obs, **training})
        assert torch.equal(got[0], want[0]) and got[1] == want[1]
    assert torch.equal(pre({**prefixed, **user})["bci_target"], target)  # not deadband-zeroed


def test_a_single_unbatched_trial_gets_its_batch_dimension():
    steering = _steering(SteeringConfig(device="cpu"))
    one = {k: v[0] for k, v in _obs().items()}
    torch.manual_seed(9)
    want = steering.predict_action_chunk({**{k: v[:1] for k, v in _obs().items()}, "command": COMMAND[:1]})
    torch.manual_seed(9)
    got = steering.predict_action_chunk({**one, "command": COMMAND[0]})
    assert got.shape == (1, 16, 2) and torch.equal(got, want)
    torch.manual_seed(9)
    assert torch.equal(steering.select_action({**one, "command": COMMAND[0]}), want[:, 0])
    with pytest.raises(ValueError, match=r"command of shape \(3, 2\) for an observation batch of 1"):
        steering.predict_action_chunk({**one, "command": COMMAND})


def test_a_ragged_adapter_matrix_is_refused_by_name():
    for bad in ([[1.0, 0.0], [0.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]):
        with pytest.raises(ValueError, match="adapter_matrix"):
            SteeringConfig(adapter_matrix=bad, device="cpu")
