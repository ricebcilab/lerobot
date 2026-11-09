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
"""The adapter holder (``AdapterModules``: a fixed matrix and/or a user module, and a task module, the one holder of
the cores, the policy and the LIBERO session) and its coercion ``as_adapter``. Its numbers on the cursor policy are
pinned against the frozen reference in tests/examples/pi05_mars_cursor/test_steering_equivalence.py; these tests pin
its parts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lerobot.policies.steering as steering
from lerobot.policies.steering import (
    N_ACTION_DIMS,
    AdaptedField,
    AdapterModules,
    Flow,
    LearnedTaskModule,
    LearnedUserModule,
    SteeringConfig,
    SteeringPolicy,
    as_adapter,
    modeling_steering,
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
    assert (
        isinstance(adapter.matrix, np.ndarray) and adapter.matrix.dtype == np.float64
    )  # the operator's units
    obs = _obs()
    field = adapter.field(obs)
    assert isinstance(field, AdaptedField) and field.matrix.dtype == torch.float32
    assert field.task_module.module is task and field.task_module.obs is obs  # bound to this observation
    x, t, v = _xtv()
    torch.testing.assert_close(field.task_module(x, t, v), task(x, t, v, obs))
    # the fixed matrix travels to the observation's device ("meta": a second device on a CPU-only run)
    meta = adapter.field({k: val.to("meta") for k, val in obs.items()})
    assert meta.matrix.device.type == "meta" and field.matrix.device.type == "cpu"
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
        path = AdapterModules.save_module(module, tmp_path / kind / "adapter.pt", kind, meta={"note": kind})
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
    assert np.array_equal(coerced.matrix, [[2.0, 0.0], [0.0, 1.0]])
    field = coerced.field(_obs())
    assert field.matrix.dtype == torch.float32 and field.matrix.device.type == "cpu"
    torch.testing.assert_close(field.matrix, torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    per_trial = as_adapter(torch.eye(2, dtype=torch.float64).expand(3, 2, 2))
    assert per_trial.matrix.shape == (3, 2, 2) and per_trial.field(_obs()).matrix.dtype == torch.float32
    assert as_adapter(task, n_dims=3).n_dims == 3


def test_normalized_matrix_is_the_pinned_expression_and_uniform_std_is_identity():
    m = np.array([[1.0, 0.3], [0.1, 0.9]])
    std = np.array([2.0, 0.5])
    holder = AdapterModules(matrix=m)
    assert np.array_equal(holder.normalized_matrix(std), m * std[np.newaxis, :] / std[:, np.newaxis])
    assert holder.normalized_matrix(np.array([0.0123, 0.0123])) is holder.matrix


def test_field_follows_like_and_std():
    """The field's matrix: as held without ``std`` (float32 on the observation's device), normalized with it, of
    ``like``'s dtype on ``like``'s device when given; the field is None when the holder is off. ``Flow`` refuses a
    holder whose fixed matrix is in the operator's units and takes one without."""
    m, std = np.array([[1.0, 0.3], [0.1, 0.9]]), np.array([2.0, 0.5])
    holder, obs = AdapterModules(matrix=m, task=_task()), _obs()
    as_held = holder.field(obs)
    assert torch.equal(as_held.matrix, torch.as_tensor(m, dtype=torch.float32))
    assert as_held.task_module.obs is obs  # bound to the observation
    normalized = holder.field(std=std, like=torch.zeros(1, H, 2, dtype=torch.float64))
    assert normalized.matrix.dtype == torch.float64
    assert torch.equal(normalized.matrix, torch.as_tensor(m * std[np.newaxis, :] / std[:, np.newaxis]))
    assert normalized.task_module is holder.task  # no observation: the module as it is
    on_like = holder.field(obs, std=std, like=torch.zeros(1, H, 2, device="meta"))
    assert on_like.matrix.device.type == "meta" and on_like.matrix.dtype == torch.float32
    assert (
        AdapterModules().field(obs) is None
        and AdapterModules().field(std=std, like=normalized.matrix) is None
    )
    x, t, v = _xtv()
    with pytest.raises(ValueError, match=r"field\(std="):
        Flow.adapt_velocity(holder, x, t, v, backward=True)
    no_matrix = AdapterModules(user=_user())
    torch.testing.assert_close(
        Flow.adapt_velocity(no_matrix, x, t, v, backward=True), no_matrix.field(like=x).adapt(x, t, v, True)
    )
    assert Flow.adapt_velocity(AdapterModules(), x, t, v, backward=True) is v
    assert AdapterModules().adapt(obs, x, t, v, True) is v


def test_describe_load_modules_and_labels(tmp_path):
    holder = AdapterModules(
        n_dims=None
    )  # each module on its own dims, the fixed matrix checked at N_ACTION_DIMS
    assert holder.kind == "none" and holder.describe() == "MARS adapter: off" and holder.format() == "off"
    holder.matrix, holder.label = np.eye(N_ACTION_DIMS), "identity"
    description = holder.describe()
    assert holder.kind == "user" and description.startswith(
        "MARS adapter: MARS-U (fixed user matrix F = identity"
    )
    assert "(identity, a no-op) = [+1.00 +0.00" in description
    with pytest.raises(ValueError, match="adapter matrix"):
        holder.matrix = np.eye(3)
    with pytest.raises(ValueError, match="adapter matrix"):
        AdapterModules(matrix=np.eye(3))  # the cursor's two dims
    per_trial = np.stack([np.eye(2), 2 * np.eye(2)])
    assert np.array_equal(AdapterModules(matrix=per_trial).matrix, per_trial)
    assert (
        AdapterModules(matrix=per_trial).format() == "[+1.00 +0.00; +0.00 +1.00] [+2.00 +0.00; +0.00 +2.00]"
    )
    with pytest.raises(ValueError, match="finite"):
        AdapterModules(matrix=np.stack([np.eye(2), np.full((2, 2), np.nan)]))
    assert AdapterModules.format_rows(np.array([[0.94, -0.34], [0.0, 1.0]])) == "[+0.94 -0.34; +0.00 +1.00]"
    user = LearnedUserModule(horizon=H, n_dims=N_ACTION_DIMS, d=8, tdim=8)
    task = LearnedTaskModule(horizon=H, n_dims=N_ACTION_DIMS, d=16, tdim=8, heads=2, layers=1)
    u = AdapterModules.save_module(user, tmp_path / "u.pt", "user")
    t = AdapterModules.save_module(task, tmp_path / "t.pt", "task")
    holder.load_modules(user_path=u, task_path=t)
    assert holder.kind == "composed" and (holder.user_label, holder.task_label) == (str(u), str(t))
    assert type(holder.user) is LearnedUserModule and not holder.user.training
    assert type(holder.task) is LearnedTaskModule and not holder.task.training
    description = holder.describe()
    assert description.startswith("MARS adapter: MARS (") and f"learned user module {u}" in description
    assert f"task module {t}" in description
    holder.matrix = None
    assert holder.kind == "composed" and "MARS adapter: MARS (learned user module" in holder.describe()
    with pytest.raises(ValueError, match="not a user module"):
        holder.load_modules(user_path=t)
    with pytest.raises(ValueError, match="not a task module"):
        holder.load_modules(task_path=u)


def test_the_policy_normalizes_its_fixed_matrix_by_the_postprocessors_std():
    def post(std):
        return SimpleNamespace(steps=[SimpleNamespace(stats={"action": {"mean": 0 * std, "std": std}})])

    m, std = [[1.0, 0.3], [0.1, 0.9]], np.array([2.0, 0.5, 9.0])  # stats wider than the matrix: their first n
    policy = SteeringPolicy(SteeringConfig(adapter_matrix=m, device="cpu"), SimpleNamespace(), post(std))
    want = torch.as_tensor(policy.adapter.normalized_matrix(std[:2]), dtype=torch.float32)
    assert torch.equal(policy._field({}).matrix, want)
    policy.set_base_policy(SimpleNamespace(), post(np.full(2, 0.05)))  # uniform (the cursor's): F as it is
    assert torch.equal(policy._field({}).matrix, torch.as_tensor(m, dtype=torch.float32))
    policy.set_base_policy(SimpleNamespace())
    with pytest.raises(ValueError, match="postprocessor"):
        policy._field({})


def test_the_old_holders_are_gone():
    for name in ("ReversalAdapter", "MatrixHolder"):
        assert not hasattr(steering, name) and name not in steering.__all__
        assert not hasattr(modeling_steering, name)
