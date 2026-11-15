#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Shared-autonomy steering of flow-matching policies: flow control, Flow Reversal Steering and MARS.

The package keeps lerobot's layout for a policy:

- ``configuration_steering``: ``SteeringConfig`` (registered ``"steering"``; its adapter fields and the grammar of a
  ``reversal_adapter`` spec, ``from_spec`` and ``build_adapter``, whose matrices are ``lerobot.teleoperators.readout``'s
  ``matrix_spec``), and the constants its validation reads.
- ``modeling_steering``: ``SteeringPolicy`` (the lerobot policy, driving every base through pi0.5's sampler
  interface, which ``PI05Policy`` and ``CursorFlowPolicy`` offer; the cursor example runs it) and the cores it
  wraps: the two methods, ``FlowControlSteering`` and ``FlowReversalSteering`` (the command per call, or pulled
  from an operator's ``TeleopSource``); the MARS modules
  (``LearnedUserModule``, ``LearnedTaskModule`` and their one holder, ``AdapterModules``, with their checkpoints);
  ``AdaptedField``; and ``Flow``, the one Euler integrator and what is built on it.
- ``processor_steering``: ``make_steering_pre_post_processors``, the steps ``SteeringSceneTokensProcessorStep`` and
  ``SteeringCommandProcessorStep``, and ``SceneEncoder``, the engine of the scene step.

The modules' training loop is not the package's: it lives in the trainers that run it."""

from .configuration_steering import (
    ACTION_DIMS,
    GRIPPER_DIM,
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    SCENE_SOURCES,
    TOKEN_GEOMETRIES,
    SteeringConfig,
)
from .modeling_steering import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    SCENE_IMAGE_SIZE,
    VLM_SCENE_DIM,
    VLM_SCENE_TOKENS,
    VLM_VIEWS_TOKENS,
    AdaptedField,
    AdapterModules,
    Flow,
    FlowControlSteering,
    FlowReversalSteering,
    LearnedTaskModule,
    LearnedUserModule,
    SteeringPolicy,
    TeleopSource,
    active_rows,
    as_adapter,
    sinusoidal_time_embedding,
)
from .processor_steering import (
    SceneEncoder,
    SteeringCommandProcessorStep,
    SteeringSceneTokensProcessorStep,
    make_steering_pre_post_processors,
)

__all__ = [
    # the config, the policy and its processors
    "SteeringConfig",
    "SteeringPolicy",
    "make_steering_pre_post_processors",
    "SteeringCommandProcessorStep",
    "SteeringSceneTokensProcessorStep",
    # the cores and the module classes
    "FlowControlSteering",
    "FlowReversalSteering",
    "TeleopSource",
    "active_rows",
    "LearnedUserModule",
    "LearnedTaskModule",
    "AdapterModules",
    "AdaptedField",
    "Flow",
    "SceneEncoder",
    "as_adapter",
    "sinusoidal_time_embedding",
    # the action space and the scene specs
    "ACTION_DIMS",
    "N_ACTION_DIMS",
    "GRIPPER_DIM",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSE",
    "OPERATOR_DIM_GROUPS",
    "DEADBAND",
    "SCENE_SOURCES",
    "TOKEN_GEOMETRIES",
    "VLM_SCENE_TOKENS",
    "VLM_VIEWS_TOKENS",
    "VLM_SCENE_DIM",
    "SCENE_IMAGE_SIZE",
]
