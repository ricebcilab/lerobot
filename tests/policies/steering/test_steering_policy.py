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
tests/examples/pi05_mars_cursor/test_steering_policy.py."""

import numpy as np
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.cursor_flow import CursorFlowConfig, CursorFlowPolicy, FlowPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.steering import (
    AdapterModules,
    AdapterSpec,
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
    assert (
        not SteeringConfig().has_task_module
        and SteeringConfig(adapter=AdapterSpec(task_path="t.pt")).has_task_module
    )


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
        adapter=AdapterSpec(matrix=f, user_path=str(user_path), task_path=str(task_path)), device="cpu"
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
    assert np.array_equal(loaded.config.adapter.matrix, f) and loaded.config.user_module == config.user_module
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
        SteeringConfig(adapter=AdapterSpec(user_path=str(user_path)), device="cpu"), base_policy=_policy()
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
        {**_obs(), "kind": "user", "latent": torch.randn(3, 16, 2), "command": torch.randn(3, 16, 2)}
    )
    loss.backward()
    assert parts["field"] == loss.item() and all(p.grad is not None for p in trainable.get_optim_params())
    with pytest.raises(ValueError, match="holds a task module"):
        SteeringPolicy(SteeringConfig(adapter=AdapterSpec(user_path=str(task_path)), device="cpu"))
