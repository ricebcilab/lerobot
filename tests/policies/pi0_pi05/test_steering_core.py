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
"""The task-generic steering core: one Euler integrator, per-trial matrices, the obstacle task module."""

import torch

from lerobot.policies.pi05.steering import (
    AdaptedField,
    LearnedUserModule,
    ObstacleTaskModule,
    _adapt_velocity,
    forward_flow,
    integrate,
    load_adapter_module,
    reverse_flow,
    save_adapter_module,
)

# ---------------------------------------------------------------- the pi0.5 flows before the core


def old_reverse_flow(x, velocity, num_steps, adapter=None, n_reversal_steps=None):
    h = 1.0 / num_steps
    steps = num_steps if n_reversal_steps is None else n_reversal_steps
    x_t = x
    for step in range(steps):
        t = step * h
        v = _adapt_velocity(adapter, x_t, t, velocity(x_t, t), backward=True)
        x_t = x_t + h * v
    return x_t


def old_forward_flow(x, velocity, num_steps, stop_at=None, adapter=None):
    h = 1.0 / num_steps
    steps = num_steps if stop_at is None else num_steps - stop_at
    x_t = x
    for step in range(steps):
        t = 1.0 - step * h
        x_t = x_t - h * _adapt_velocity(adapter, x_t, t, velocity(x_t, t), backward=False)
    return x_t


def _linear_field(dim=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, dim, generator=g)
    c = torch.randn(dim, generator=g)
    return lambda x, t: x @ w.T * (1.0 + t) + c


def test_integrate_reproduces_pi05_flows_exactly():
    f = _linear_field()
    x = torch.randn(4, 10, 7, generator=torch.Generator().manual_seed(1))
    assert torch.equal(integrate(x, f, 0.0, 1 / 10, 10, backward=True), old_reverse_flow(x, f, 10))
    assert torch.equal(integrate(x, f, 1.0, -1 / 10, 6), old_forward_flow(x, f, 10, stop_at=4))
    # the wrappers themselves, with and without an adapter and a partial reversal
    matrix = torch.randn(7, 7, generator=torch.Generator().manual_seed(2))
    for adapter in (None, matrix, AdaptedField(matrix[:3, :3])):
        assert torch.equal(
            reverse_flow(x, f, 10, adapter=adapter), old_reverse_flow(x, f, 10, adapter=adapter)
        )
        assert torch.equal(
            reverse_flow(x, f, 10, adapter=adapter, n_reversal_steps=4),
            old_reverse_flow(x, f, 10, adapter=adapter, n_reversal_steps=4),
        )
        assert torch.equal(
            forward_flow(x, f, 10, adapter=adapter), old_forward_flow(x, f, 10, adapter=adapter)
        )
        assert torch.equal(
            forward_flow(x, f, 10, stop_at=4, adapter=adapter),
            old_forward_flow(x, f, 10, stop_at=4, adapter=adapter),
        )


def test_integrate_grad_flag_and_times():
    seen = []
    w = torch.ones(1, requires_grad=True)

    def velocity(x, t):
        seen.append(t)
        return x * w

    x = torch.ones(2, 3, 1)
    assert not integrate(x, velocity, 0.5, 0.25, 2).requires_grad
    assert seen == [0.5, 0.75]
    assert integrate(x, velocity, 0.5, 0.25, 2, grad=True).requires_grad
    seen.clear()
    integrate(x, velocity, 1.0, -0.5, 2, batch_time=True)
    assert [s.tolist() for s in seen] == [[1.0, 1.0], [0.5, 0.5]]


def test_adapted_field_takes_per_trial_matrices():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(3, 2, 2, generator=g)
    v = torch.randn(3, 5, 2, generator=g)
    x = torch.randn(3, 5, 2, generator=g)
    expect = torch.einsum("bij,bhj->bhi", f, v)
    torch.testing.assert_close(AdaptedField(f).adapt(x, 0.3, v, backward=True), expect)
    torch.testing.assert_close(_adapt_velocity(f, x, 0.3, v, backward=True), expect)
    assert torch.equal(AdaptedField(f).adapt(x, 0.3, v, backward=False), v)
    # a (2, 2) F on a 7-dim velocity changes only the first two dims
    f2 = torch.randn(2, 2, generator=g)
    v7 = torch.randn(3, 5, 7, generator=g)
    out = AdaptedField(f2).adapt(x, 0.3, v7, backward=True)
    torch.testing.assert_close(out[..., :2], torch.einsum("ij,bhj->bhi", f2, v7[..., :2]))
    assert torch.equal(out[..., 2:], v7[..., 2:])


def test_adapted_field_n_dims_overrides_the_modules():
    g = torch.Generator().manual_seed(0)
    user = LearnedUserModule(5, n_dims=2)
    with torch.no_grad():
        user.net[-1].weight.normal_(generator=g)

    def bare(x, t):  # a user module without an n_dims attribute
        return user(x, t)

    x = torch.randn(3, 5, 7, generator=g)
    v = torch.randn(3, 5, 7, generator=g)
    out = AdaptedField(None, user_module=bare, n_dims=2).adapt(x, 0.3, v, backward=True)
    torch.testing.assert_close(out, AdaptedField(None, user_module=user).adapt(x, 0.3, v, backward=True))
    assert torch.equal(out[..., 2:], v[..., 2:])


def _obstacle_obs(b=3, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    obstacles = torch.cat(
        [torch.rand(b, k, 2, generator=g), 0.05 + 0.1 * torch.rand(b, k, 1, generator=g)], -1
    )
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[0, -1] = False
    return {"pos": torch.rand(b, 2, generator=g), "obstacles": obstacles, "obstacle_mask": mask}


def test_obstacle_task_module_starts_at_identity_and_round_trips(tmp_path):
    torch.manual_seed(0)
    module = ObstacleTaskModule(horizon=16).eval()
    with_obs = _obstacle_obs()
    without = {"pos": with_obs["pos"]}
    empty = {
        **with_obs,
        "obstacles": with_obs["obstacles"][:, :0],
        "obstacle_mask": with_obs["obstacle_mask"][:, :0],
    }
    x, v, t = torch.randn(3, 16, 2), torch.randn(3, 16, 2), torch.rand(3)
    eye = torch.eye(2).expand(3, 16, 2, 2)
    for obs in (with_obs, without, empty):
        a, b = module(obs, x, t, v)
        assert torch.equal(a, eye) and torch.equal(b, torch.zeros(3, 16, 2))
        a, b = module.bind(obs)(x, 0.4, v)  # a float time too
        assert torch.equal(a, eye) and torch.equal(b, torch.zeros(3, 16, 2))
    assert module.bind(with_obs).n_dims == 2

    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path = save_adapter_module(module, tmp_path / "task.pt", kind="task", arch="obstacle", meta={"note": "x"})
    assert torch.load(path, weights_only=True)["arch"] == "obstacle"
    loaded, kind, meta = load_adapter_module(path)
    assert isinstance(loaded, ObstacleTaskModule) and kind == "task" and meta == {"note": "x"}
    module.requires_grad_(False)  # both on nn.MultiheadAttention's same (fast, inference) path
    for obs in (with_obs, without):
        a0, b0 = module(obs, x, t, v)
        a1, b1 = loaded(obs, x, t, v)
        assert torch.equal(a0, a1) and torch.equal(b0, b1)
        assert not torch.equal(a0, eye)
    # through the adapted field, bound to the observation
    field = AdaptedField(None, task_module=loaded.bind(with_obs))
    a, b = module(with_obs, x, 0.4, v)
    torch.testing.assert_close(
        field.adapt(x, 0.4, v, backward=False), torch.einsum("bhij,bhj->bhi", a, v) + b
    )


def test_old_task_checkpoints_without_arch_load_as_mlp(tmp_path):
    from lerobot.policies.pi05.steering import LearnedTaskModule

    module = LearnedTaskModule(4, n_dims=3)
    path = save_adapter_module(module, tmp_path / "task.pt", kind="task")
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["arch"] == "mlp"
    del ckpt["arch"]  # an old LIBERO file
    torch.save(ckpt, path)
    loaded, kind, _ = load_adapter_module(path)
    assert isinstance(loaded, LearnedTaskModule) and kind == "task"
