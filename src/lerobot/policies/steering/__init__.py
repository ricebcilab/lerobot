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

"""Shared-autonomy steering of flow-matching policies: flow control, Flow Reversal Steering and MARS."""

from .configuration_steering import (
    ACTION_DIMS,
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    AdapterSpec,
    MatrixHolder,
    TeleopSource,
    build_reversal_adapter,
    check_steps,
    format_matrix,
    parse_adapter_spec,
    rotation_about_z,
    translation_matrix,
    validate_matrix,
)
from .modeling_steering import (
    AdaptedField,
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    LearnedTaskModule,
    LearnedUserModule,
    ReversalAdapter,
    apply_user_module,
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
    regenerate,
    reverse_flow,
    save_adapter_module,
    sinusoidal_time_embedding,
    step_if_finite,
    task_field_loss,
    task_velocity_hook,
    train_module,
    uniform_reference,
    user_field_loss,
)

__all__ = [
    "ACTION_DIMS",
    "AdaptedField",
    "AdapterSpec",
    "DEADBAND",
    "FlowControlPolicy",
    "FlowReversalSteeringPolicy",
    "GRIPPER_CLOSE",
    "GRIPPER_DIM",
    "GRIPPER_OPEN",
    "LearnedTaskModule",
    "LearnedUserModule",
    "MatrixHolder",
    "N_ACTION_DIMS",
    "OPERATOR_DIM_GROUPS",
    "ReversalAdapter",
    "TeleopSource",
    "apply_user_module",
    "build_reversal_adapter",
    "check_steps",
    "convert_cursor_module",
    "delegation_mask",
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
    "regenerate",
    "reverse_flow",
    "rotation_about_z",
    "save_adapter_module",
    "sinusoidal_time_embedding",
    "step_if_finite",
    "task_field_loss",
    "task_velocity_hook",
    "train_module",
    "translation_matrix",
    "uniform_reference",
    "user_field_loss",
    "validate_matrix",
]
