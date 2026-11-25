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
"""A task module narrower than the base policy's action width: pi0.5's field spans ``max_action_dim`` (32),
while the station's task module is 8 dims (7 joints plus the gripper). ``LearnedTaskModule.field_loss``,
``SteeringPolicy.forward`` and ``AdaptedField`` fit and apply the module on its own first ``n_dims`` dims of
a field-width chunk -- the dims ``AdaptedField.adapt`` already uses at inference. The cursor (2 dims) and
LIBERO (7) numbers are the ones they always were: the same arithmetic on a field no wider than themselves.

The fake base is pi0.5's sampler interface reduced to what ``forward`` reads: a config and a constant
``velocity_field`` returning a (B, H, 32) tensor. The losses are hand-computed (a fresh ``LearnedTaskModule``
has an exact zero head, so it is the identity: ``A = I``, ``b = 0``) or from the direct ``A v + b``
arithmetic -- never snapshotted from the implementation.
"""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.policies.steering import (
    N_ACTION_DIMS,
    AdaptedField,
    AdapterModules,
    LearnedTaskModule,
    SteeringConfig,
    SteeringPolicy,
)

H = 4  # the fake chunk's horizon
WIDTH = 32  # pi0.5's max_action_dim, the width of its velocity field
N_DIMS = 8  # the station's action dims: 7 joints and the gripper
B = 2
MEAN = 0.1 * np.arange(WIDTH, dtype=np.float64)  # the checkpoint's action statistics, non-zero in every dim
STD = 0.5 + 0.05 * np.arange(WIDTH, dtype=np.float64)


def _module(n_dims: int = N_DIMS) -> LearnedTaskModule:
    """A small fresh task module: zero-initialised head, so it is the identity exactly."""
    return LearnedTaskModule(horizon=H, n_dims=n_dims, d=16, tdim=8, heads=2, layers=1)


def _perturbed(module: LearnedTaskModule, seed: int = 0) -> LearnedTaskModule:
    """``module`` with small non-zero weights everywhere: ``A != I`` and the whole net gets gradients."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape))
    return module


def _obs(b: int = B) -> dict:
    """The observation a task module binds to: the discs source reads no scene tokens, ``pos`` only queries."""
    return {"pos": torch.zeros(b, 2)}


class _ConstantBase:
    """pi0.5's sampler interface for ``forward``: a config and a constant field ``v = value`` of ``x``'s width."""

    def __init__(self, value: float = 0.5):
        self.config = SimpleNamespace(
            num_inference_steps=4, chunk_size=H, max_action_dim=WIDTH, n_action_steps=H, noise_time=1.0
        )
        self.value = value
        self.xs: list[torch.Tensor] = []  # every x the field was called on

    def reset(self) -> None:
        pass

    def velocity_field(self, batch: dict):
        def velocity(x: torch.Tensor, t) -> torch.Tensor:
            self.xs.append(x)
            return torch.full_like(x, self.value)

        return velocity


class _StatsStep:
    """An unnormalizer-like step for ``action_mean_std``: the checkpoint's action statistics over 32 dims."""

    def __init__(self) -> None:
        self.stats = {"action": {"mean": MEAN, "std": STD}}
        self.norm_map = {}


class _Postprocessor:
    """The one part of the real postprocessor ``_normalized_chunk`` reads: the action statistics."""

    def __init__(self) -> None:
        self.steps = [_StatsStep()]


# ---------------------------------------------------------------- the flow-matching loss


def test_field_loss_fits_the_modules_dims_of_a_32_wide_field():
    """A 32-wide chunk and latent: the constant field's 32-wide velocity, a fresh (identity-head) 8-dim module
    and lambda_id = 0.25 give the hand-computed loss

        fm = mean((0.5 - (chunk - z)[..., :8]) ** 2),   identity = 0,   loss = fm,

    since ``A = I`` and ``b = 0`` are exact, the module is fit on ``v[..., :8]`` and the target is the first 8
    dims of ``chunk - z``. The gradient reaches the 8-dim module."""
    torch.manual_seed(0)
    chunk, z = torch.randn(B, H, WIDTH), torch.randn(B, H, WIDTH)
    t = torch.rand(B)
    module = _module()
    bound = module.bind(_obs())

    def velocity(x: torch.Tensor, time) -> torch.Tensor:
        return torch.full_like(x, 0.5)

    loss, parts = LearnedTaskModule.field_loss(bound, velocity, chunk, z, t, 0.25, N_DIMS)

    # the module's A is 8 x 8: it fits the first 8 dims of the caller's 32-wide field
    assert module(torch.zeros(B, H, WIDTH), t, torch.zeros(B, H, WIDTH), _obs())[0].shape == (
        B,
        H,
        N_DIMS,
        N_DIMS,
    )
    expect_fm = ((0.5 - (chunk - z)[..., :N_DIMS]) ** 2).mean()
    assert torch.equal(parts["fm"], expect_fm)
    assert parts["identity"] == 0.0  # A = I, b = 0 exactly
    assert torch.equal(loss.detach(), expect_fm) and bool(torch.isfinite(loss))
    loss.backward()
    assert module.out.weight.grad is not None and module.out.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(g).all() for g in (p.grad for p in module.parameters() if p.grad is not None))


def test_field_loss_is_the_direct_arithmetic_when_the_field_is_no_wider_than_the_module():
    """The cursor's 2 dims and LIBERO's 7: with a perturbed module the loss is computed here directly from
    ``A v + b`` on the whole chunk, and equals ``field_loss``'s bit for bit (the first ``n`` dims are all of
    them, so slicing them changes nothing)."""
    for n_dims in (2, 7):
        torch.manual_seed(n_dims)
        module = _perturbed(_module(n_dims))
        bound = module.bind(_obs())
        chunk, z = torch.randn(B, H, n_dims), torch.randn(B, H, n_dims)
        t = torch.rand(B)

        def velocity(x: torch.Tensor, time) -> torch.Tensor:
            return 0.5 * x - 0.3 * time.reshape(-1, 1, 1)

        x = (1 - t[:, None, None]) * z + t[:, None, None] * chunk
        with torch.no_grad():
            v = velocity(x, t)
            a, b = bound(x, t, v)
        fm = ((torch.einsum("bsij,bsj->bsi", a, v) + b - (chunk - z)) ** 2).mean()
        identity = ((a - torch.eye(n_dims)) ** 2).mean() + (b**2).mean()
        want = fm + 0.25 * identity
        assert torch.equal(LearnedTaskModule.field_loss(bound, velocity, chunk, z, t, 0.25, n_dims)[0], want)


def test_a_libero_shaped_module_on_pi05s_32_wide_field_keeps_liberos_numbers():
    """LIBERO's 7-dim task module over pi0.5's 32-wide field: the loss is exactly the one seven-wide tensors
    with the same first seven dims give -- the field's dims 7+ and the chunk's are never compared."""
    torch.manual_seed(1)
    module = _perturbed(_module(N_ACTION_DIMS))
    bound = module.bind(_obs())
    chunk7, z7 = torch.randn(B, H, N_ACTION_DIMS), torch.randn(B, H, N_ACTION_DIMS)
    # the wide pair: the same first 7 dims, arbitrary values beyond (the field's padding is never compared)
    chunk32, z32 = torch.randn(B, H, WIDTH), torch.randn(B, H, WIDTH)
    chunk32[..., :N_ACTION_DIMS], z32[..., :N_ACTION_DIMS] = chunk7, z7
    t = torch.rand(B)

    def field7(x: torch.Tensor, time) -> torch.Tensor:
        return 0.5 * x - 0.3 * time.reshape(-1, 1, 1)

    def field32(x: torch.Tensor, time) -> torch.Tensor:  # its first 7 dims are field7's
        wide = torch.full_like(x[..., N_ACTION_DIMS:], 9.0)
        return torch.cat([field7(x[..., :N_ACTION_DIMS], time), wide], -1)

    narrow = LearnedTaskModule.field_loss(bound, field7, chunk7, z7, t, 0.25, N_ACTION_DIMS)
    wide = LearnedTaskModule.field_loss(bound, field32, chunk32, z32, t, 0.25, N_ACTION_DIMS)
    assert torch.equal(narrow[0], wide[0])
    assert torch.equal(narrow[1]["fm"], wide[1]["fm"]) and torch.equal(
        narrow[1]["identity"], wide[1]["identity"]
    )


# ---------------------------------------------------------------- the policy's forward


def test_forward_pads_the_normalized_chunk_to_the_base_fields_width():
    """An 8-wide demonstrated batch over the 32-wide fake base: ``forward`` normalizes the chunk by the
    checkpoint's statistics and zero-pads it to 32 in that normalized space, pads a caller's 8-wide noise the
    same way, and the field sees a 32-wide x whose padding dims are exactly zero (the statistics' means are
    non-zero, so padding after normalizing is what keeps them zero). The identity module's loss is
    mean((0.5 - (normalized - z)[..., :8]) ** 2): the first 8 dims are the ones compared."""
    base = _ConstantBase()
    config = SteeringConfig(
        mode="none", n_dims=N_DIMS, task_module=dict(_module().config), lambda_id=0.25, device="cpu"
    )
    steering = SteeringPolicy(config, base_policy=base, postprocessor=_Postprocessor())
    module = steering.adapter.task
    chunk, z, t = torch.randn(B, H, N_DIMS), torch.randn(B, H, N_DIMS), torch.rand(B)
    batch = {"observation.state": torch.zeros(B, WIDTH), "task": ["pick", "place"]}

    loss, parts = steering({**batch, "kind": "task", "chunk": chunk, "noise": z, "time": t})

    assert len(base.xs) == 1 and base.xs[0].shape == (B, H, WIDTH)  # the field saw the chunk at its width
    assert torch.equal(
        base.xs[0][..., N_DIMS:], torch.zeros(B, H, WIDTH - N_DIMS)
    )  # normalized-space padding
    mean, std = (torch.as_tensor(a[:N_DIMS], dtype=torch.float32) for a in (MEAN, STD))
    normalized = (chunk - mean) / std
    expect_fm = ((0.5 - (normalized - z)) ** 2).mean()
    assert parts["fm"] == float(expect_fm) and parts["identity"] == 0.0  # the logged parts are plain floats
    assert torch.equal(loss.detach(), expect_fm) and bool(torch.isfinite(loss))
    loss.backward()
    assert module.out.weight.grad is not None and module.out.weight.grad.abs().sum() > 0

    # noise and time drawn here: randn_like of the padded chunk and rand at the batch size, still 32 wide
    steering.zero_grad(set_to_none=True)
    for _ in range(2):
        loss, _ = steering({**batch, "kind": "task", "chunk": chunk})
        assert bool(torch.isfinite(loss)) and len(base.xs) >= 2 and base.xs[-1].shape == (B, H, WIDTH)


# ---------------------------------------------------------------- the field's dims


def test_adapted_field_applies_an_8_dim_module_to_its_first_8_dims_of_a_wider_field():
    """A 7-wide holder with an 8-dim task module: the field acts on the module's own 8 dims (the velocity is
    32 wide) and passes dims 8+ through untouched, exactly ``A v[:8] + b``."""
    torch.manual_seed(0)
    module = _perturbed(_module())
    bound = module.bind(_obs())
    field = AdaptedField(
        None, task_module=bound, holder_n_dims=7
    )  # the holder's 7 does not shrink the module
    x, v = torch.randn(B, H, WIDTH), torch.randn(B, H, WIDTH)

    out = field.adapt(x, 0.3, v, backward=False)

    a, b = bound(x, 0.3, v[..., :N_DIMS])
    assert torch.equal(out[..., :N_DIMS], torch.einsum("bhij,bhj->bhi", a, v[..., :N_DIMS]) + b)
    assert torch.equal(out[..., N_DIMS:], v[..., N_DIMS:])


def test_adapted_field_dims_precedence():
    """The dims a module acts on: an explicit ``n_dims`` wins; else the module's own; else the holder's;
    else ``N_ACTION_DIMS``. The holder's (or the explicit override) is what the field's ``n_dims`` reports."""
    wide = _module()  # n_dims = 8
    three = lambda x_t, t_: torch.eye(3).expand(len(x_t), H, 3, 3)  # noqa: E731
    holder = AdaptedField(None, task_module=wide, holder_n_dims=7)
    assert holder._dims(wide) == N_DIMS  # the module's own beats the holder's
    assert holder._dims(three) == 7  # a module without one uses the holder's
    assert holder.n_dims == 7
    assert AdaptedField(None)._dims(three) == N_ACTION_DIMS  # neither: the action space's
    explicit = AdaptedField(None, task_module=wide, n_dims=3, holder_n_dims=7)
    assert (
        explicit._dims(wide) == 3 and explicit.n_dims == 3
    )  # explicit wins over the module's own and the holder's
    # ... and it is applied: a 3-wide callable touches dims 0-2, dims 3+ pass through
    applied = AdaptedField(None, user_module=three, n_dims=3, holder_n_dims=7)
    x, v = torch.randn(B, H, WIDTH), torch.randn(B, H, WIDTH)
    out = applied.adapt(x, 0.3, v, backward=True)
    assert torch.equal(out[..., 3:], v[..., 3:]) and torch.equal(out[..., :3], v[..., :3])


def test_the_steering_policys_field_takes_an_8_dim_task_module_in_a_7_wide_config():
    """The defect's inference side: a ``SteeringPolicy`` whose holder is the Franka's 7 dims
    (``config.n_dims``) holding the station's 8-dim task module. The field reports the holder's 7 as before
    but applies the module on its own 8 dims of the base's 32-wide velocity, and dims 8+ pass through."""
    base = _ConstantBase()
    config = SteeringConfig(
        mode="none", n_dims=N_ACTION_DIMS, task_module=dict(_module().config), device="cpu"
    )
    steering = SteeringPolicy(config, base_policy=base)
    view = steering._adapter_view({"observation.state": torch.zeros(B, WIDTH), "task": ["pick"]})

    field = steering._field(view)

    assert N_ACTION_DIMS == 7  # the Franka's operator space: translations, rotation and the gripper
    assert field.n_dims == N_ACTION_DIMS  # the holder's width, as before
    assert field._dims(field.task_module) == N_DIMS  # the module's own 8 dims win
    x, v = torch.randn(B, H, WIDTH), torch.randn(B, H, WIDTH)
    out = field.adapt(x, 0.3, v, backward=False)
    assert out.shape == v.shape and torch.equal(out[..., N_DIMS:], v[..., N_DIMS:])


def test_the_cursor_and_libero_holders_keep_their_own_arithmetic():
    """The cursor's ``AdapterModules(n_dims=2)`` and LIBERO's ``AdapterModules(n_dims=None)``: holder and
    module agree, so the field acts as it always did (``A v + b`` over their own dims) and reports the
    holder's width."""
    cursor = _perturbed(_module(2))
    field = AdapterModules(task=cursor, n_dims=2).field(_obs())
    x, v = torch.randn(B, H, 2), torch.randn(B, H, 2)
    a, b = cursor.bind(_obs())(x, 0.3, v)
    assert torch.equal(field.adapt(x, 0.3, v, backward=False), torch.einsum("bhij,bhj->bhi", a, v) + b)
    assert field.n_dims == 2 and field._dims(cursor.bind(_obs())) == 2

    libero = _perturbed(_module(N_ACTION_DIMS))
    field = AdapterModules(task=libero, n_dims=None).field(_obs())
    x, v = torch.randn(B, H, N_ACTION_DIMS), torch.randn(B, H, N_ACTION_DIMS)
    a, b = libero.bind(_obs())(x, 0.3, v)
    assert torch.equal(field.adapt(x, 0.3, v, backward=False), torch.einsum("bhij,bhj->bhi", a, v) + b)
    assert field.n_dims is None and field._dims(libero.bind(_obs())) == N_ACTION_DIMS
