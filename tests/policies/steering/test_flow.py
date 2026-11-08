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

import torch

from lerobot.policies.steering import Flow


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
    steered, plain = Flow.paired_chunks(Sampler(), None, True, None, shift=1.0)
    after = torch.randn(1)
    torch.manual_seed(0)
    assert torch.equal(steered - 1.0, plain) and torch.equal(torch.randn(2, 3), plain)
    assert torch.equal(torch.randn(1), after)  # the stream is what it would be without the second pass
