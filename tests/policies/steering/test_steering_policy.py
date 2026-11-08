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
and the policy's batch forms. Its chunks against the cursor example's are in
tests/examples/pi05_mars_cursor/test_steering_policy_cursor.py."""

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.cursor_flow import CursorFlowConfig, CursorFlowPolicy, FlowPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.steering import (
    AdapterModules,
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


def _policy(**kw):
    torch.manual_seed(0)
    return FlowPolicy(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123).eval()


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
        steering = SteeringPolicy(SteeringConfig(mode=mode, device="cpu"), base_policy=base)
        assert isinstance(steering.core, core) and steering.core._policy.config.noise_time == 0.0
        assert (
            steering.state_dict() == {} and list(steering.get_optim_params()) == []
        )  # the base is not a weight
        torch.manual_seed(1)
        chunk = steering.predict_action_chunk({**_obs(), "command": COMMAND})
        assert chunk.shape == (3, 16, 2)
        torch.manual_seed(1)
        plain = base.sample_chunk(_obs(), 10) * base.action_scale
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
    steering = SteeringPolicy(config, base_policy=_policy())
    assert config.user_module == AdapterModules.load(user_path).user.config  # recorded when read
    user_keys = {f"adapter.user.{k}" for k in AdapterModules.load(user_path).user.state_dict()}
    task_keys = {f"adapter.task.{k}" for k in AdapterModules.load(task_path).task.state_dict()}
    assert set(steering.state_dict()) == user_keys | task_keys  # the adapter's weights only, the base's not
    steering.save_pretrained(tmp_path / "steering")
    user_path.unlink()
    task_path.unlink()  # the saved policy no longer needs the module checkpoints
    loaded = SteeringPolicy.from_pretrained(tmp_path / "steering", base_policy=_policy())
    assert np.array_equal(loaded.config.fixed_matrix, f) and loaded.config.user_module == config.user_module
    assert loaded.state_dict().keys() == steering.state_dict().keys()
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in steering.state_dict().items())
    assert not any(
        p.requires_grad for p in loaded.parameters()
    )  # served frozen, as load_module serves a module
    for mode in ("flow_control", "frs"):
        steering.config.mode = loaded.config.mode = mode
        steering.set_base_policy(_policy())
        loaded.set_base_policy(_policy())
        torch.manual_seed(2)
        want = steering.predict_action_chunk({**_obs(), "command": COMMAND})
        torch.manual_seed(2)
        assert torch.equal(loaded.predict_action_chunk({**_obs(), "command": COMMAND}), want)


def test_the_base_policy_loads_from_the_configs_reference(tmp_path):
    model = _policy()
    base = CursorFlowPolicy(CursorFlowConfig(**model.config, action_scale=0.0123, device="cpu"))
    base.model.load_state_dict(model.state_dict())
    base.save_pretrained(tmp_path / "base")
    steering = SteeringPolicy(
        SteeringConfig(base_policy_type="cursor_flow", base_policy_path=str(tmp_path / "base"), device="cpu")
    )
    assert isinstance(steering.base_policy, CursorFlowPolicy)
    torch.manual_seed(4)
    got = steering.predict_action_chunk({**_obs(), "command": COMMAND})
    torch.manual_seed(4)
    assert torch.equal(
        got,
        SteeringPolicy(SteeringConfig(device="cpu"), base_policy=model).predict_action_chunk(
            {**_obs(), "command": COMMAND}
        ),
    )
    with pytest.raises(ValueError, match="needs a base policy"):
        _ = SteeringPolicy(SteeringConfig(device="cpu")).base_policy


def test_the_batch_may_carry_the_observation_under_its_prefix(tmp_path):
    steering = SteeringPolicy(SteeringConfig(scene_source="vlm_image", device="cpu"), base_policy=_policy())
    obs = _obs()
    prefixed = {f"observation.{k}": v for k, v in obs.items()}
    tokens = torch.zeros(3, 256, 8, dtype=torch.float16)
    seen = steering.observation(
        {
            **prefixed,
            "observation.scene_tokens": tokens,
            "observation.layouts": object(),
            "command": COMMAND,
            "action": None,
            "next.reward": 0.0,
        }
    )
    assert list(seen) == [*obs, "scene"] and seen["scene"]["vlm_image"] is tokens
    assert list(steering.observation({**obs, "command": COMMAND, "kind": "task"})) == list(obs)
    torch.manual_seed(5)
    want = steering.predict_action_chunk({**obs, "command": COMMAND})
    torch.manual_seed(5)
    assert torch.equal(steering.predict_action_chunk({**prefixed, "command": COMMAND.numpy()}), want)


def test_forward_needs_the_module_of_its_kind(tmp_path):
    user_path, task_path = _modules(tmp_path)
    steering = SteeringPolicy(
        SteeringConfig(adapter_user=str(user_path), device="cpu"), base_policy=_policy()
    )
    with pytest.raises(ValueError, match="needs a task module"):
        steering({**_obs(), "kind": "task", "chunk": torch.zeros(3, 16, 2)})
    with pytest.raises(ValueError, match="'task' or 'user'"):
        steering({**_obs(), "kind": "other"})
    fresh = SteeringConfig(
        user_module=dict(LearnedUserModule(16, n_dims=2, d=16, tdim=8).config), device="cpu"
    )
    trainable = SteeringPolicy(fresh, base_policy=_policy())
    loss, parts = trainable(
        {**_obs(), "kind": "user", "latent": torch.randn(3, 16, 2), "bci_target": torch.randn(3, 16, 2)}
    )
    loss.backward()
    assert parts["field"] == loss.item() and all(p.grad is not None for p in trainable.get_optim_params())
    with pytest.raises(ValueError, match="holds a task module"):
        SteeringPolicy(SteeringConfig(adapter_user=str(task_path), device="cpu"))


class _SamplerBase:
    """A pi0.5-style base: no ``sample_chunk``, pi0.5's sampler interface itself; records the batches it gets."""

    def __init__(self):
        from types import SimpleNamespace

        self.config = SimpleNamespace(num_inference_steps=4, chunk_size=6, max_action_dim=8, n_action_steps=3)
        self.batches = []

    def reset(self):
        pass

    def predict_action_chunk(self, batch, **hooks):
        self.batches.append(batch)
        return torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8)


def test_a_pi05_style_base_gets_lerobots_batch_as_it_is():
    base = _SamplerBase()
    steering = SteeringPolicy(
        SteeringConfig(mode="flow_control", operator_dims=["translation"], device="cpu"), base_policy=base
    )
    batch = {
        "observation.state": torch.zeros(2, 8),
        "observation.images.top": torch.zeros(2, 3, 4, 4),
        "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
        "task": ["pick", "place"],
        "command": torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    chunk = steering.predict_action_chunk(batch)
    assert chunk.shape == (2, 6, 8)  # normalized: the postprocessor unnormalizes a pi0.5 chunk
    assert base.batches and all(
        list(b) == ["observation.state", "observation.images.top", "observation.language.tokens", "task"]
        for b in base.batches
    )


def test_select_action_serves_the_chunks_executed_steps_one_per_call():
    base = _policy()
    steering = SteeringPolicy(SteeringConfig(device="cpu", n_action_steps=3), base_policy=base)
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
    steering = SteeringPolicy(config, base_policy=_policy())
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
    steering = SteeringPolicy(SteeringConfig(device="cpu"), base_policy=_policy())
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
