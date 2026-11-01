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
"""The robot and the cursor run one implementation of each steering method: the same wrapper classes
(`FlowControlPolicy`, `FlowReversalSteeringPolicy`) drive pi0.5's sampler loop (``PI05Pytorch.sample_actions``,
t = 1 noise) and the cursor policy's (``FlowPolicy.sample_chunk``, t = 0 noise). On time-mirrored fields and the
same noise the two give the same chunk, for flow control at every tau and lambda and for FRS and MARS at every
reversal depth and pinned prefix: the samplers, the only per-side code, agree."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.cursor_flow.modeling_cursor_flow import FlowPolicy
from lerobot.policies.steering import (
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    FlowControlPolicy,
    FlowReversalSteeringPolicy,
    ReversalAdapter,
    frs_schedule,
)

STEPS, CHUNK = 10, 6


class MiniPi05:
    """pi0.5's sampler loop (``PI05Pytorch.sample_actions``: ``noise_fn``, ``flow_start_time`` /
    ``num_forward_steps``, ``x_t_hook``, ``velocity_hook``) over a given field, from a fixed noise."""

    def __init__(self, field, noise):
        self.field, self.noise = field, noise
        self.config = SimpleNamespace(
            num_inference_steps=STEPS, chunk_size=CHUNK, max_action_dim=noise.shape[-1], n_action_steps=2
        )

    def reset(self):
        pass

    def predict_action_chunk(self, batch, **kwargs):
        noise = self.noise.clone()
        if kwargs.get("noise_fn") is not None:
            noise = kwargs["noise_fn"](self.field, noise)
        start = kwargs.get("flow_start_time")
        start = 1.0 if start is None else float(start)
        n = kwargs.get("num_forward_steps") or STEPS
        dt, x_t = -start / n, noise
        for step in range(n):
            time = start + step * dt
            if kwargs.get("x_t_hook") is not None:
                x_t = kwargs["x_t_hook"](step, time, x_t)
            v = self.field(x_t, time)
            if kwargs.get("velocity_hook") is not None:
                v = kwargs["velocity_hook"](step, time, x_t, v)
            x_t = x_t + dt * v
        return x_t


class FieldFlowPolicy(FlowPolicy):
    """The cursor's ``FlowPolicy`` with its network replaced by a given field: its own ``sample_chunk`` loop."""

    def __init__(self, field):
        super().__init__(horizon=CHUNK, d_model=8, layers=1, heads=1, d_ff=8)
        self.field = field

    def forward(self, obs, x=None, t=None):
        return self.field(x, t)


class CursorSampler:
    """The cursor example's sampler (``cursor_sa.steering.CursorSampler``): ``FlowPolicy.sample_chunk`` behind
    pi0.5's interface, noise at t = 0, from a fixed noise."""

    def __init__(self, policy, noise):
        self.policy, self.noise = policy, noise
        self.config = SimpleNamespace(
            num_inference_steps=STEPS,
            chunk_size=CHUNK,
            max_action_dim=noise.shape[-1],
            n_action_steps=2,
            noise_time=0.0,
        )

    def reset(self):
        pass

    def predict_action_chunk(self, obs, **hooks):
        return self.policy.sample_chunk(obs, STEPS, noise=self.noise.clone(), **hooks)


def fields(dims, seed=0):
    """A nonlinear field in pi0.5's convention and its time mirror in the cursor's: dx/dt0 = -f(x, 1 - t0)."""
    g = torch.Generator().manual_seed(seed)
    w, c = 0.5 * torch.randn(dims, dims, generator=g), torch.randn(dims, generator=g)

    def pi05(x, s):
        return torch.tanh(x @ w.T) * (1.0 + s) + c

    def cursor(x, t):
        s = 1.0 - (t[:, None, None] if torch.is_tensor(t) else t)
        return -(torch.tanh(x @ w.T) * (1.0 + s) + c)

    return pi05, cursor


def postprocessor(dims):
    stats = {"mean": np.zeros(dims), "std": np.ones(dims)}
    return SimpleNamespace(steps=[SimpleNamespace(stats={"action": stats})])


class Source:
    def __init__(self, translation):
        self.translation, self.gripper = np.asarray(translation, dtype=np.float64), -1.0


OBS = {"pos": torch.zeros(1, 2)}


def test_frs_schedule_mirrors_the_reversal_in_either_convention():
    assert frs_schedule(10, None, 1.0) == (1.0, 10) and frs_schedule(10, None, 0.0) == (0.0, 10)
    assert frs_schedule(10, 4, 1.0) == pytest.approx((0.4, 4)) and frs_schedule(10, 4, 0.0) == pytest.approx(
        (0.6, 4)
    )
    with pytest.raises(ValueError, match="noise_time"):
        frs_schedule(10, 4, 0.5)


@pytest.mark.parametrize("depth", [1, 3, 5, 8, 9, None])
@pytest.mark.parametrize("pinned", [2, 4, None])
@pytest.mark.parametrize("mars", [False, True])
def test_frs_and_mars_run_one_implementation_on_both_samplers(depth, pinned, mars):
    pi05_field, cursor_field = fields(N_ACTION_DIMS)
    noise = torch.randn(1, CHUNK, N_ACTION_DIMS, generator=torch.Generator().manual_seed(1))
    command = np.array([0.6, -0.3, 0.2])
    adapter = ReversalAdapter(matrix=np.diag([1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0])) if mars else None
    robot = FlowReversalSteeringPolicy(
        MiniPi05(pi05_field, noise),
        Source(command),
        postprocessor(N_ACTION_DIMS),
        adapter=adapter,
        n_reversal_steps=depth,
        operator_dims=tuple(OPERATOR_DIM_GROUPS),
        pinned_steps=pinned,
    )
    reference, field = robot.reference_chunk(command), robot._adapter_field(noise)
    robot_chunk, _ = robot._sample_chunk(batch=None)
    cursor = FlowReversalSteeringPolicy(
        CursorSampler(FieldFlowPolicy(cursor_field), noise),
        None,
        n_reversal_steps=depth,
        operator_dims=None,
        pinned_steps=pinned,
        reference_fn=lambda c: reference,
    )
    cursor_chunk, _ = cursor.steered_chunk(OBS, command[None], field=field)
    torch.testing.assert_close(cursor_chunk, robot_chunk, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tau", [2, 4, CHUNK])
@pytest.mark.parametrize("lam", [0, 1, 3, STEPS])
def test_flow_control_runs_one_implementation_on_both_samplers(tau, lam):
    pi05_field, cursor_field = fields(3, seed=2)
    noise = torch.randn(1, CHUNK, 3, generator=torch.Generator().manual_seed(3))
    command = np.array([0.4, -0.7, 0.1])
    robot = FlowControlPolicy(
        MiniPi05(pi05_field, noise), Source(command), postprocessor(3), tau=tau, lam=lam, ic_gain=1.5
    )
    robot_chunk, _ = robot._sample_chunk(batch=None)
    cursor = FlowControlPolicy(
        CursorSampler(FieldFlowPolicy(cursor_field), noise),
        None,
        tau=tau,
        lam=lam,
        ic_gain=1.5,
        dims=slice(None),
    )
    cursor_chunk, _ = cursor.steered_chunk(OBS, torch.tensor(command[None], dtype=torch.float32))
    torch.testing.assert_close(cursor_chunk, robot_chunk, atol=1e-5, rtol=1e-5)


def test_idle_rows_get_the_plain_chunk():
    """A batch of commands, one per row: an idle row is the policy's own chunk from the same noise."""
    _, cursor_field = fields(2, seed=4)
    noise = torch.randn(3, CHUNK, 2, generator=torch.Generator().manual_seed(5))
    sampler = CursorSampler(FieldFlowPolicy(cursor_field), noise)
    commands = torch.tensor([[0.8, 0.1], [0.01, 0.0], [-0.5, 0.6]])
    plain = sampler.predict_action_chunk(OBS)
    for wrapper in (
        FlowControlPolicy(sampler, None, tau=4, ic_gain=1.5, dims=slice(None)),
        FlowReversalSteeringPolicy(
            sampler,
            None,
            n_reversal_steps=5,
            operator_dims=None,
            pinned_steps=None,
            reference_fn=lambda c: c[:, None].expand(-1, CHUNK, -1),
        ),
    ):
        chunk, own = wrapper.steered_chunk(OBS, commands)
        torch.testing.assert_close(own, plain)
        torch.testing.assert_close(chunk[1], plain[1])
        assert not torch.allclose(chunk[[0, 2]], plain[[0, 2]])
