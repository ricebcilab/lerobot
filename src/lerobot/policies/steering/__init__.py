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

- ``configuration_steering``: ``SteeringConfig`` (registered ``"steering"``; its adapter fields, ``from_spec`` and
  the matrix helpers of a ``reversal_adapter`` spec), and the constants its validation reads.
- ``modeling_steering``: ``SteeringPolicy`` (the lerobot policy) and the cores it wraps: the two methods,
  ``FlowControlSteering`` and ``FlowReversalSteering``, over the operator's ``TeleopSource``; the MARS modules
  (``LearnedUserModule``, ``LearnedTaskModule``, the composed ``AdapterModules`` with their checkpoints, and the
  holders ``MatrixHolder`` / ``ReversalAdapter``); ``AdaptedField``; and ``Flow``, the one Euler integrator and what
  is built on it.
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
    VLM_SCENE_DIM,
    VLM_SCENE_TOKENS,
    VLM_VIEWS_TOKENS,
    AdaptedField,
    AdapterModules,
    Flow,
    FlowControlPolicy,
    FlowControlSteering,
    FlowReversalSteering,
    FlowReversalSteeringPolicy,
    LearnedTaskModule,
    LearnedUserModule,
    MatrixHolder,
    ReversalAdapter,
    SteeringPolicy,
    TeleopSource,
    apply_user_module,
    as_adapter,
    build_reversal_adapter,
    convert_cursor_module,
    delegation_mask,
    flow_control_write,
    format_matrix,
    forward_flow,
    frs_latent,
    frs_schedule,
    get_action_mean_std,
    heading_cosine,
    integrate,
    load_adapter_module,
    paired_chunks,
    parse_adapter_spec,
    reduce_tokens,
    regenerate,
    reverse_flow,
    rotation_about_z,
    save_adapter_module,
    sinusoidal_time_embedding,
    task_field_loss,
    task_velocity_hook,
    translation_matrix,
    uniform_reference,
    user_field_loss,
    validate_matrix,
)
from .processor_steering import (
    SceneEncoder,
    SteeringCommandProcessorStep,
    SteeringSceneTokensProcessorStep,
    encode_views,
    fit_token_reduction,
    make_steering_pre_post_processors,
    scene_for,
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
    "LearnedUserModule",
    "LearnedTaskModule",
    "AdapterModules",
    "AdaptedField",
    "MatrixHolder",
    "ReversalAdapter",
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
    # deprecated aliases (removed by the next commit)
    "FlowControlPolicy",
    "FlowReversalSteeringPolicy",
    "apply_user_module",
    "build_reversal_adapter",
    "convert_cursor_module",
    "delegation_mask",
    "encode_views",
    "fit_token_reduction",
    "flow_control_write",
    "format_matrix",
    "forward_flow",
    "frs_latent",
    "frs_schedule",
    "get_action_mean_std",
    "heading_cosine",
    "integrate",
    "load_adapter_module",
    "paired_chunks",
    "parse_adapter_spec",
    "reduce_tokens",
    "regenerate",
    "reverse_flow",
    "rotation_about_z",
    "save_adapter_module",
    "scene_for",
    "task_field_loss",
    "task_velocity_hook",
    "translation_matrix",
    "uniform_reference",
    "user_field_loss",
    "validate_matrix",
]
