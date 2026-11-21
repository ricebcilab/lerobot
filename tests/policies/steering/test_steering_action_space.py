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
"""The steering package's action-space genericity: the 7-D operator space's defaults are untouched, and a wider
action space (a pi0.5-DROID checkpoint on a Franka arm: 7 joint velocities, the gripper at index 7) gets the
configured reference step and gripper index instead of the operator space's own."""

import numpy as np
import pytest
import torch

from lerobot.policies.steering import (
    GRIPPER_DIM,
    Flow,
    FlowReversalSteering,
    SteeringConfig,
    SteeringPolicy,
)

from tests.policies.steering.test_steering_wrappers import CHUNK, MAX_DIM, FakePolicy, postprocessor


def test_the_7_dim_action_space_keeps_the_uniform_reference():
    """pi0.5's 7-D operator space (no ``reference_step`` on the config) still takes ``Flow.uniform_reference``: its
    statistics span the gripper dim, so the core has no held-command step."""
    policy = SteeringPolicy(
        SteeringConfig(mode="frs", device="cpu"), base_policy=FakePolicy(), postprocessor=postprocessor()
    )
    core = policy.core
    assert core._reference_step is None and core._gripper_index == GRIPPER_DIM
    command = torch.tensor([[1.0, -0.5, 0.25]])
    want = Flow.uniform_reference(command, core._gripper_ref, core._mean, core._std, CHUNK, MAX_DIM)
    assert torch.equal(core.reference_chunk(command), want)


def test_a_wider_action_space_takes_the_configured_reference_step():
    """A config ``reference_step`` wins over the action space's own rule, and the held reference spans the command's
    dims, the padding beyond them zero."""
    mean, std = 0.1 * np.arange(8.0), 0.5 + 0.25 * np.arange(8.0)
    policy = SteeringPolicy(
        SteeringConfig(mode="frs", n_dims=7, reference_step=0.5, device="cpu"),
        base_policy=FakePolicy(),
        postprocessor=postprocessor(mean=mean, std=std),
    )
    core = policy.core
    assert core._reference_step == 0.5
    command = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.5, 0.2, -0.1]])
    reference = core.reference_chunk(command)
    mean_t = torch.as_tensor(mean[:7], dtype=command.dtype)
    std_t = torch.as_tensor(std[:7], dtype=command.dtype)
    want = (command[:, None].expand(-1, CHUNK, -1) * 0.5 - mean_t) / std_t
    assert reference.shape == (1, CHUNK, MAX_DIM)
    torch.testing.assert_close(reference[..., :7], want)
    assert not reference[..., 7:].any()


def test_the_statistics_are_kept_whole():
    """Neither statistic is truncated to the 7-D operator space: the reference reads the dims its command spans."""
    core = FlowReversalSteering(
        FakePolicy(), None, postprocessor(mean=np.arange(8.0), std=np.ones(8))
    )
    assert core._mean.shape[0] == 8 and core._std.shape[0] == 8


def test_a_configured_gripper_index_is_held_and_delegated():
    """With ``gripper_dim``: the held gripper is read at that index (here 7), and the delegation mask leaves that dim
    to the policy while the operator's named groups (dims 0..6) stay pinned -- built at construction and rebuilt by
    the setters."""
    mean, std = np.zeros(8), np.ones(8)
    mean[6], std[6] = 3.0, 1.0
    mean[7], std[7] = 1.0, 2.0
    core = FlowReversalSteering(
        FakePolicy(),
        None,
        postprocessor(mean=mean, std=std),
        operator_dims=("translation", "rotation", "gripper"),
        pinned_steps=None,
        gripper_dim=7,
    )
    assert core._has_gripper is True and core._gripper_index == 7
    assert core._gripper_ref == pytest.approx((-1.0 - 1.0) / 2.0)  # GRIPPER_OPEN at the config's index
    assert core._delegated[:, 7].all() and not core._delegated[:, :7].any()
    core.operator_dims = ["translation", "rotation", "gripper"]  # the setter rebuilds it with the index
    assert core._delegated[:, 7].all() and not core._delegated[:, :7].any()
    core.pinned_steps = 2  # the tail is delegated anyway; the pinned prefix still holds the gripper's dim delegated
    assert core._delegated[:2, 7].all() and not core._delegated[:2, :7].any()

    named = FlowReversalSteering.delegation_mask(
        CHUNK, MAX_DIM, ("translation", "rotation", "gripper"), None, gripper_dim=7
    )
    assert named[:, 7].all() and not named[:, :7].any()
    every = FlowReversalSteering.delegation_mask(CHUNK, MAX_DIM, None, None, gripper_dim=7)
    assert every[:, 7].all() and not every[:, :7].any()  # the other commanded dims stay pinned


def test_the_action_space_knobs_are_validated():
    for bad in (-1, 0):
        with pytest.raises(ValueError, match="reference_step"):
            SteeringConfig(reference_step=bad)
    with pytest.raises(ValueError, match="gripper_dim"):
        SteeringConfig(gripper_dim=8)  # n_dims is 2 by default
    with pytest.raises(ValueError, match="gripper_dim"):
        SteeringConfig(gripper_dim=-1)
    assert SteeringConfig(reference_step=0.5, gripper_dim=1).gripper_dim == 1
