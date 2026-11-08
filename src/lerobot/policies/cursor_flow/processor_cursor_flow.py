#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_cursor_flow import CursorFlowConfig


def make_cursor_flow_pre_post_processors(
    config: CursorFlowConfig,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the cursor flow policy.

    The policy reads its own dict observation (``CursorFlowPolicy.observation``), so the pre-processing pipeline
    only renames (nothing, by default). The post-processing pipeline unnormalizes the policy's normalized chunks to
    screen units per step, ``chunk * std + mean`` with mean 0 and std ``config.action_scale`` on both dims (the
    model's ``chunk * action_scale``, bit for bit), and moves them to the CPU. Its unnormalizer step carries the
    action statistics the steering cores read.

    Args:
        config: The cursor flow policy configuration.

    Returns:
        A tuple containing the pre-processor pipeline and the post-processor pipeline.
    """
    input_steps = [RenameObservationsProcessorStep(rename_map={})]
    scale = config.action_scale
    output_steps = [
        UnnormalizerProcessorStep(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={ACTION: {"mean": [0.0, 0.0], "std": [scale, scale]}},
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
