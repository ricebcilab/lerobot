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
"""``Flow`` in the cursor's convention and ``Flow.paired_chunks``' generator restore (pi0.5's convention is pinned
by test_steering_core.py and test_steering_wrappers.py)."""

import pytest
import torch

from lerobot.policies.cursor_flow.modeling_cursor_flow import CursorFlowPolicy, FlowPolicy
from lerobot.policies.steering import (
    Flow,
    FlowControlSteering,
    FlowReversalSteering,
    active_rows,
)


def _field():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(7, 7, generator=g) * 0.3

    def velocity(x, t):
        t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, 1, 1) if torch.is_tensor(t) else t
        return torch.tanh(x @ w) * (1 + t)

    return velocity, torch.randn(3, 6, 7, generator=g), torch.eye(7) + 0.1 * torch.randn(7, 7, generator=g)


def test_a_cursor_convention_flow_reverses_from_t_one_and_runs_forward_from_t_zero():
    velocity, x, f = _field()
    flow = Flow(velocity, f, noise_time=0.0, batch_time=True)
    assert torch.equal(flow.reverse(x, 5, 3), flow.integrate(x, 1.0, -1.0 / 5, 3, backward=True))
    assert torch.equal(flow.forward(x, 5, 2), flow.integrate(x, 0.0, 1.0 / 5, 3))


def test_paired_chunks_reuses_the_noise_and_restores_the_generator():
    class Sampler:
        def predict_action_chunk(self, batch, **kw):
            return torch.randn(2, 3) + kw.get("shift", 0.0)

    torch.manual_seed(0)
    steered, plain = Flow.paired_chunks(Sampler(), None, None, shift=1.0)
    after = torch.randn(1)
    torch.manual_seed(0)
    assert torch.equal(steered - 1.0, plain) and torch.equal(torch.randn(2, 3), plain)
    assert torch.equal(torch.randn(1), after)  # the stream is what it would be without the second pass


class _FieldPolicy(FlowPolicy):
    """The cursor's FlowPolicy (``sample_chunk`` draws its noise from torch's generator) over a fixed field."""

    def __init__(self):
        super().__init__(horizon=6, d_model=8, layers=1, heads=1, d_ff=8)

    def forward(self, obs, x=None, t=None):
        return torch.tanh(x + 0.3) * (1.5 - t.reshape(-1, 1, 1))


def test_a_steered_chunk_without_the_plan_is_the_paired_one_and_leaves_the_same_stream():
    """``paired=False`` skips the plain pass when every row steers: the chunk and the generator state afterwards are
    the paired call's (with an idle row the plan is still drawn, for that row)."""
    policy = _FieldPolicy().eval()
    sampler = CursorFlowPolicy.wrapping(policy)  # the cursor policy behind pi0.5's interface, 5 Euler steps
    sampler.config.num_inference_steps, sampler.config.n_action_steps = 5, 2
    obs = {"pos": torch.zeros(2, 2)}
    cores = (
        FlowControlSteering(sampler, None, tau=4, lam=2, ic_gain=1.5, dims=slice(None)),
        FlowReversalSteering(
            sampler,
            None,
            n_reversal_steps=3,
            operator_dims=None,
            pinned_steps=None,
            reference_fn=lambda c: Flow.held_reference(c, 6, 0.024, policy.action_scale),
        ),
    )
    for core in cores:
        for commands in (torch.tensor([[0.8, 0.1], [-0.5, 0.6]]), torch.tensor([[0.8, 0.1], [0.0, 0.0]])):
            torch.manual_seed(1)
            paired, plan = core.steered_chunk(obs, commands)
            after = torch.get_rng_state()
            torch.manual_seed(1)
            alone, none = core.steered_chunk(obs, commands, paired=False)
            assert torch.equal(alone, paired) and torch.equal(torch.get_rng_state(), after)
            assert (none is None) == bool(active_rows(commands, core.deadband).all()) and plan is not None


def test_a_flow_takes_only_the_two_time_conventions():
    """noise_time is 1.0 (pi0.5) or 0.0 (the cursor): any other value would make the shared Euler step 0."""
    velocity, _, _ = _field()
    assert (
        Flow(velocity, noise_time=1).toward_noise(5) == 0.2
        and Flow(velocity, noise_time=0.0).toward_noise(5) == -0.2
    )
    for bad in (0.5, -1.0, 2.0):
        with pytest.raises(ValueError, match="noise_time must be 1.0"):
            Flow(velocity, noise_time=bad)
