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

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_hf_token  # noqa: E402


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise


def _small_policy():
    """A pi0.5 with 7-dim actions, batch built the same way as test_policy_instantiation
    (through the pre-processor), since `predict_action_chunk` requires the batch to already
    contain tokenized-language keys that only the pre-processor adds -- a raw batch with a
    `"task"` string is not enough.
    """
    set_seed(0)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32", num_inference_steps=4)
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    policy = PI05Policy(config).to("cuda").eval()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.zeros(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["do something"],
    }
    batch = preprocessor(batch)
    return policy, batch


@require_cuda
@require_hf_token
def test_sample_actions_noise_fn_replaces_starting_noise():
    policy, batch = _small_policy()
    seen = {}
    replacement_noise = None

    def noise_fn(velocity, noise):
        seen["noise_shape"] = tuple(noise.shape)
        v = velocity(noise, 1.0)  # the velocity closure must evaluate the field
        seen["velocity_shape"] = tuple(v.shape)
        nonlocal replacement_noise
        replacement_noise = torch.zeros_like(noise)
        return replacement_noise

    def x_t_hook(step, time, x_t):
        if step == 0:
            seen["step0_x_t"] = x_t.clone()
        return x_t

    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch, noise_fn=noise_fn, x_t_hook=x_t_hook)
    assert seen["noise_shape"] == (1, policy.config.chunk_size, policy.config.max_action_dim)
    assert seen["velocity_shape"] == seen["noise_shape"]
    # The loop starts from `x_t = noise` before the first hook call, so the x_t the
    # hook sees at step 0 must be exactly what noise_fn returned, not the original
    # sampled noise -- this is what proves the return value replaces the starting noise.
    assert torch.equal(seen["step0_x_t"], replacement_noise)
    assert chunk.shape[0] == 1


@require_cuda
@require_hf_token
def test_sample_actions_partial_schedule_lands_on_zero():
    policy, batch = _small_policy()
    times = []

    def x_t_hook(step, time, x_t):
        times.append(round(time, 6))
        return x_t

    with torch.inference_mode():
        policy.predict_action_chunk(batch, x_t_hook=x_t_hook, flow_start_time=0.5, num_forward_steps=2)
    assert times == [0.5, 0.25]  # two Euler steps of -0.25 from t=0.5 reach t=0

    with pytest.raises(ValueError, match="num_forward_steps"), torch.inference_mode():
        policy.predict_action_chunk(batch, flow_start_time=0.5, num_forward_steps=0)
