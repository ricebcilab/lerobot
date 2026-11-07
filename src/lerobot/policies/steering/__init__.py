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

- ``configuration_steering``: ``SteeringConfig`` (registered ``"steering"``), the action space and the operator's
  view of it (``TeleopSource``, ``DEADBAND``, ``OPERATOR_DIM_GROUPS``), ``AdapterSpec`` (a ``reversal_adapter``
  spec: ``parse``, ``build`` and the matrix helpers) and ``MatrixHolder``.
- ``modeling_steering``: ``SteeringPolicy`` (the lerobot policy) and the cores it wraps: the MARS modules
  (``LearnedUserModule``, ``LearnedTaskModule``, the composed ``AdapterModules`` with their checkpoints and the
  holder ``ReversalAdapter``), ``AdaptedField``, ``Flow`` (the one Euler integrator and what is built on it) and
  the two methods, ``FlowControlSteering`` and ``FlowReversalSteering``.
- ``processor_steering``: ``SceneEncoder``, the steps ``SteeringSceneTokensProcessorStep`` and
  ``SteeringCommandProcessorStep``, and ``make_steering_pre_post_processors``.

The module-level functions and the two ``...Policy`` core names these classes replaced stay importable for one round
as deprecated aliases (each says so in its docstring; ``FlowControlPolicy`` is ``FlowControlSteering``,
``FlowReversalSteeringPolicy`` is ``FlowReversalSteering``). The modules' training loop is not the package's: it
lives in the trainers that run it."""

from .configuration_steering import (
    ACTION_DIMS,
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    SCENE_SOURCES,
    TOKEN_GEOMETRIES,
    VLM_VIEWS_TOKENS,
    AdapterSpec,
    MatrixHolder,
    SteeringConfig,
    TeleopSource,
    build_reversal_adapter,
    check_steps,
    format_matrix,
    parse_adapter_spec,
    rotation_about_z,
    scene_spec,
    translation_matrix,
    validate_matrix,
)
from .modeling_steering import (
    AdaptedField,
    AdapterModules,
    Flow,
    FlowControlPolicy,
    FlowControlSteering,
    FlowReversalSteering,
    FlowReversalSteeringPolicy,
    LearnedTaskModule,
    LearnedUserModule,
    ReversalAdapter,
    SteeringPolicy,
    apply_user_module,
    as_adapter,
    convert_cursor_module,
    delegation_mask,
    flow_control_write,
    forward_flow,
    frs_latent,
    frs_schedule,
    get_action_mean_std,
    heading_cosine,
    integrate,
    load_adapter_module,
    paired_chunks,
    reduce_tokens,
    regenerate,
    reverse_flow,
    save_adapter_module,
    sinusoidal_time_embedding,
    task_field_loss,
    task_velocity_hook,
    uniform_reference,
    user_field_loss,
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
    # the policy, its config and its processors
    "SteeringConfig",
    "SteeringPolicy",
    "SteeringCommandProcessorStep",
    "SteeringSceneTokensProcessorStep",
    "make_steering_pre_post_processors",
    # the cores and the module classes
    "AdaptedField",
    "AdapterModules",
    "AdapterSpec",
    "Flow",
    "FlowControlSteering",
    "FlowReversalSteering",
    "LearnedTaskModule",
    "LearnedUserModule",
    "MatrixHolder",
    "ReversalAdapter",
    "SceneEncoder",
    "TeleopSource",
    "as_adapter",
    "check_steps",
    "scene_spec",
    "sinusoidal_time_embedding",
    # the action space and the scene specs
    "ACTION_DIMS",
    "DEADBAND",
    "GRIPPER_CLOSE",
    "GRIPPER_DIM",
    "GRIPPER_OPEN",
    "N_ACTION_DIMS",
    "OPERATOR_DIM_GROUPS",
    "SCENE_SOURCES",
    "TOKEN_GEOMETRIES",
    "VLM_VIEWS_TOKENS",
    # deprecated aliases, kept for one round
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
