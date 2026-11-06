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
"""The composed adapter (``AdapterModules``: a fixed matrix and/or a user module, and a task module) and its
coercion ``as_adapter``. Its numbers on the cursor policy are pinned against the frozen reference in
tests/examples/pi05_mars_cursor/test_steering_equivalence.py; these tests pin its parts."""

import torch

from lerobot.policies.steering import (
    AdaptedField,
    AdapterModules,
    LearnedTaskModule,
    LearnedUserModule,
    as_adapter,
    save_adapter_module,
)

H = 4


def _user():
    torch.manual_seed(0)
    user = LearnedUserModule(H, n_dims=2, d=8, tdim=8)
    with torch.no_grad():  # off the identity, so a module's effect shows
        user.net[-1].weight.normal_(std=0.1)
    return user


def _task():
    torch.manual_seed(1)
    task = LearnedTaskModule(horizon=H, d=16, tdim=8, heads=2, layers=1)
    with torch.no_grad():
        for p in task.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return task


def _obs(b=3, device="cpu"):
    g = torch.Generator().manual_seed(2)
    return {
        "pos": torch.rand(b, 2, generator=g).to(device),
        "obstacles": torch.rand(b, 2, 3, generator=g).to(device),
        "obstacle_mask": torch.tensor([[True, False]] * b, device=device),
    }


def _xtv(b=3):
    g = torch.Generator().manual_seed(3)
    return torch.randn(b, H, 2, generator=g), torch.full((b,), 0.3), torch.randn(b, H, 2, generator=g)


def test_kind_per_combination():
    user, task, matrix = _user(), _task(), torch.eye(2)
    assert AdapterModules().kind == "none"
    assert AdapterModules(user=user).kind == "user"
    assert AdapterModules(matrix=matrix).kind == "user"
    assert AdapterModules(task=task).kind == "task"
    assert AdapterModules(user=user, task=task).kind == "composed"
    assert AdapterModules(matrix=matrix, task=task).kind == "composed"
    assert AdapterModules(user=user, matrix=matrix, task=task).kind == "composed"


def test_field_binds_the_task_module_and_moves_the_matrix():
    task, matrix = _task(), [[0.0, -1.0], [1.0, 0.0]]
    adapter = AdapterModules(task=task, matrix=matrix)
    assert adapter.matrix.dtype == torch.float32
    obs = _obs()
    field = adapter.field(obs)
    assert isinstance(field, AdaptedField)
    assert field.task_module.module is task and field.task_module.obs is obs  # bound to this observation
    x, t, v = _xtv()
    torch.testing.assert_close(field.task_module(x, t, v), task(x, t, v, obs))
    # the fixed matrix travels to the observation's device ("meta": a second device on a CPU-only run)
    meta = adapter.field({k: val.to("meta") for k, val in obs.items()})
    assert meta.matrix.device.type == "meta" and adapter.matrix.device.type == "cpu"
    # a task part without bind (any (x, t, v) -> (A, b) callable) goes in as it is
    plain = lambda x_t, t_, v_: (torch.eye(2).expand(*v_.shape[:-1], 2, 2), torch.zeros_like(v_))  # noqa: E731
    assert AdapterModules(task=plain).field(obs).task_module is plain


def test_adapt_is_the_fields():
    user, task = _user(), _task()
    adapter = AdapterModules(user=user, task=task, matrix=torch.diag(torch.tensor([1.0, 0.5])))
    obs, (x, t, v) = _obs(), _xtv()
    for backward in (False, True):
        torch.testing.assert_close(
            adapter.adapt(obs, x, t, v, backward), adapter.field(obs).adapt(x, t, v, backward)
        )
    forward, backward = adapter.adapt(obs, x, t, v, False), adapter.adapt(obs, x, t, v, True)
    assert not torch.allclose(forward, v) and not torch.allclose(forward, backward)


def test_n_dims_reaches_the_field():
    obs = _obs()
    assert AdapterModules(user=_user()).field(obs).n_dims == 2
    assert AdapterModules(user=_user(), n_dims=None).field(obs).n_dims is None
    assert as_adapter(torch.eye(3), n_dims=3).field(obs).n_dims == 3


def test_load_a_saved_user_and_task_module(tmp_path):
    obs, (x, t, v) = _obs(), _xtv()
    for kind, module in (("user", _user()), ("task", _task())):
        path = save_adapter_module(module, tmp_path / kind / "adapter.pt", kind, meta={"note": kind})
        loaded = AdapterModules.load(path)
        assert loaded.kind == kind and not loaded.training
        part = getattr(loaded, kind)
        assert type(part) is type(module) and part is not module
        assert getattr(loaded, "task" if kind == "user" else "user") is None and loaded.matrix is None
        for name, value in module.state_dict().items():
            torch.testing.assert_close(part.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(
            loaded.adapt(obs, x, t, v, True), AdapterModules(**{kind: module}).adapt(obs, x, t, v, True)
        )


def test_as_adapter_coercions():
    assert as_adapter(None) is None
    adapter = AdapterModules(matrix=torch.eye(2))
    assert as_adapter(adapter) is adapter
    task = _task()
    coerced = as_adapter(task)
    assert coerced.task is task and coerced.user is None and coerced.matrix is None
    user = _user()
    coerced = as_adapter(user)
    assert coerced.user is user and coerced.task is None
    callable_user = lambda x_t, t: torch.eye(2).expand(len(x_t), H, 2, 2)  # noqa: E731
    assert as_adapter(callable_user).user is callable_user
    coerced = as_adapter([[2.0, 0.0], [0.0, 1.0]])
    assert coerced.kind == "user" and coerced.user is None
    assert coerced.matrix.dtype == torch.float32 and coerced.matrix.device.type == "cpu"
    torch.testing.assert_close(coerced.matrix, torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    per_trial = as_adapter(torch.eye(2, dtype=torch.float64).expand(3, 2, 2))
    assert per_trial.matrix.shape == (3, 2, 2) and per_trial.matrix.dtype == torch.float32
    assert as_adapter(task, n_dims=3).n_dims == 3
