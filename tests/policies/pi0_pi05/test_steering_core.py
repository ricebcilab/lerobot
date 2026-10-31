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
"""The task-generic steering core: one Euler integrator, per-trial matrices, the obstacle task module, the user
module's rank knobs."""

import pytest
import torch

from lerobot.policies.pi05 import steering
from lerobot.policies.pi05.steering import (
    AdaptedField,
    LearnedTaskModule,
    LearnedUserModule,
    _adapt_velocity,
    apply_user_module,
    forward_flow,
    integrate,
    load_adapter_module,
    reverse_flow,
    save_adapter_module,
    user_field_loss,
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


def test_task_module_starts_at_identity_and_round_trips(tmp_path):
    torch.manual_seed(0)
    module = LearnedTaskModule(horizon=16).eval()
    with_obs = _obstacle_obs()
    without = {"pos": with_obs["pos"]}
    empty = {
        **with_obs,
        "obstacles": with_obs["obstacles"][:, :0],
        "obstacle_mask": with_obs["obstacle_mask"][:, :0],
    }
    x, v, t = torch.randn(3, 16, 2), torch.randn(3, 16, 2), torch.rand(3)
    eye = torch.eye(2).expand(3, 16, 2, 2)
    for obs in (with_obs, without, empty, None):  # obstacles, none, an empty set, no observation at all
        a, b = module(x, t, v, obs)
        assert torch.equal(a, eye) and torch.equal(b, torch.zeros(3, 16, 2))
        a, b = (module if obs is None else module.bind(obs))(x, 0.4, v)  # a float time too
        assert torch.equal(a, eye) and torch.equal(b, torch.zeros(3, 16, 2))
    assert module.bind(with_obs).n_dims == 2

    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path = save_adapter_module(module, tmp_path / "task.pt", kind="task", meta={"note": "x"})
    assert "arch" not in torch.load(path, weights_only=True)
    loaded, kind, meta = load_adapter_module(path)
    assert isinstance(loaded, LearnedTaskModule) and kind == "task" and meta == {"note": "x"}
    module.requires_grad_(False)  # both on nn.MultiheadAttention's same (fast, inference) path
    for obs in (with_obs, without, None):
        a0, b0 = module(x, t, v, obs)
        a1, b1 = loaded(x, t, v, obs)
        assert torch.equal(a0, a1) and torch.equal(b0, b1)
        assert not torch.equal(a0, eye)
    # through the adapted field, bound to the observation
    field = AdaptedField(None, task_module=loaded.bind(with_obs))
    a, b = module(x, 0.4, v, with_obs)
    torch.testing.assert_close(
        field.adapt(x, 0.4, v, backward=False), torch.einsum("bhij,bhj->bhi", a, v) + b
    )


def test_task_checkpoints_of_the_obstacle_arch_load_and_mlp_ones_are_refused(tmp_path):
    """Files from when there were two task modules: arch "obstacle" is LearnedTaskModule; "mlp", or an MLP's
    weights with no arch (the first LIBERO files), is the removed MLP module, refused with a clear error."""
    module = LearnedTaskModule(horizon=4, d=16, layers=1)
    path = save_adapter_module(module, tmp_path / "task.pt", kind="task")
    ckpt = torch.load(path, weights_only=True)
    torch.save({**ckpt, "arch": "obstacle"}, path)
    assert isinstance(load_adapter_module(path)[0], LearnedTaskModule)
    for old in ({**ckpt, "arch": "mlp"}, {**ckpt, "state_dict": {"net.0.weight": torch.zeros(1)}}):
        torch.save(old, path)
        with pytest.raises(ValueError, match="MLP task module"):
            load_adapter_module(path)


# ---------------------------------------------------------------- the user module's rank knobs


def _old_mlp(d_in, d, d_out):
    """The core's ``_mlp`` as it was (Linear, SiLU, Linear, SiLU, Linear; the last zero-initialized), frozen."""
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, d),
        torch.nn.SiLU(),
        torch.nn.Linear(d, d),
        torch.nn.SiLU(),
        torch.nn.Linear(d, d_out),
    )
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    return net


class OldLearnedUserModule(torch.nn.Module):
    """``LearnedUserModule`` before the rank knobs (lerobot main 0cc6706c, the mars time embedding), frozen."""

    def __init__(self, horizon, n_dims=7, d=256, tdim=64):
        super().__init__()
        self.config = {"horizon": horizon, "n_dims": n_dims, "d": d, "tdim": tdim, "time_embedding": "mars"}
        self.horizon, self.n_dims, self.tdim = horizon, n_dims, tdim
        self.net = _old_mlp(horizon * n_dims + tdim, d, horizon * n_dims * n_dims)

    def forward(self, x_t, t):
        b, h, n = x_t.shape[0], self.horizon, self.n_dims
        x = x_t[:, :h, :n].reshape(b, h * n).to(torch.float32)
        emb = steering.sinusoidal_time_embedding(t, self.tdim, device=x.device).expand(b, -1)
        delta = self.net(torch.cat([x, emb], dim=-1)).view(b, h, n, n)
        return torch.eye(n, device=x.device, dtype=x.dtype) + delta


RANKS = [
    {},
    {"step_rank": 1},
    {"step_rank": 3},
    {"cross_step_rank": 2},
    {"cross_step_rank": 2, "cross_step_dims": [2, 0]},
    {"step_rank": 2, "cross_step_rank": 3},
    {"step_rank": 2, "cross_step_rank": 1, "cross_step_dims": [1]},
]


def _perturbed_user(seed=2, horizon=6, n_dims=3, **ranks):
    torch.manual_seed(seed)
    user = LearnedUserModule(horizon, n_dims, d=32, tdim=8, **ranks)
    with torch.no_grad():
        for p in user.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return user.eval()


def _inputs(b=3, horizon=6, n_dims=3, pad=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, horizon, n_dims + pad, generator=g)
    return x, torch.rand(b, generator=g), torch.randn(b, horizon, n_dims, generator=g)


def test_defaults_are_bitwise_the_old_user_module():
    """Same seed, same weights (the knobs draw nothing extra at their defaults), same config, same outputs, and
    every place the core applies the module (AdaptedField, the user loss) gives the old einsum's bits."""
    torch.manual_seed(5)
    old = OldLearnedUserModule(6, 3, d=32, tdim=8)
    torch.manual_seed(5)
    new = LearnedUserModule(6, 3, d=32, tdim=8)
    assert new.config == old.config and not hasattr(new, "cross")
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(
        torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values(), strict=True)
    )
    _perturb_both = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p, q in zip(old.parameters(), new.parameters(), strict=True):
            p.add_(0.05 * torch.randn(p.shape, generator=_perturb_both))
            q.copy_(p)
    x, t, v = _inputs()
    assert torch.equal(new(x, t), old(x, t))
    einsum = torch.einsum("bhij,bhj->bhi", old(x, t), v)
    assert torch.equal(new.velocity(x, t, v), einsum)
    assert torch.equal(apply_user_module(new, x, t, v), einsum)
    full = torch.cat([v, torch.randn(3, 6, 2)], -1)
    old_field = torch.cat([einsum, full[..., 3:]], -1)
    assert torch.equal(AdaptedField(None, new).adapt(x, t, full, backward=True), old_field)
    latent, target = torch.randn(3, 6, 3), torch.randn(3, 6, 3)

    def velocity(x_, t_):
        return 0.5 * x_ + t_[:, None, None]

    xs = (1 - t[:, None, None]) * latent + t[:, None, None] * target
    old_err = (torch.einsum("bsij,bsj->bsi", old(xs, t), velocity(xs, t)) - (target - latent)) ** 2
    old_loss = old_err.mean((1, 2)).mean()  # user_field_loss's body before the knobs
    assert torch.equal(user_field_loss(new, velocity, latent, target, t), old_loss)


def test_every_rank_setting_starts_at_the_identity():
    x, t, v = _inputs()
    for ranks in RANKS:
        user = LearnedUserModule(6, 3, d=32, tdim=8, **ranks)
        with torch.no_grad():
            assert torch.equal(user.velocity(x, t, v), v), ranks
            assert torch.equal(user(x, t), torch.eye(3).expand(3, 6, 3, 3)), ranks
        assert user.config == {
            "horizon": 6,
            "n_dims": 3,
            "d": 32,
            "tdim": 8,
            "time_embedding": "mars",
            **ranks,
        }


def test_gradient_reaches_the_coefficients_the_basis_and_both_cross_factors():
    """At the identity the coefficients and U get a gradient (the basis and V do not, each multiplied by a zero);
    after one step all four do."""
    x, t, v = _inputs()
    target = torch.randn(3, 6, 3)
    user = LearnedUserModule(6, 3, d=32, tdim=8, step_rank=2, cross_step_rank=2)
    n_basis, n_u = 3 * 3 * 2, 6 * 3 * 2
    opt = torch.optim.SGD(user.parameters(), lr=0.1)
    grads = []
    for _ in range(2):
        opt.zero_grad()
        ((user.velocity(x, t, v) - target) ** 2).mean().backward()
        head, cross = user.net[-1].weight.grad, user.cross.weight.grad
        grads.append(
            {
                "basis": head[:n_basis].abs().sum().item(),
                "coef": head[n_basis:].abs().sum().item(),
                "U": cross[:n_u].abs().sum().item(),
                "V": cross[n_u:].abs().sum().item(),
                "hidden": user.net[0].weight.grad.abs().sum().item(),
            }
        )
        opt.step()
    assert grads[0]["coef"] > 0 and grads[0]["U"] > 0 and grads[0]["basis"] == 0 and grads[0]["V"] == 0
    assert all(value > 0 for value in grads[1].values()), grads[1]


def test_ranked_modules_save_and_load_with_their_knobs(tmp_path):
    x, t, v = _inputs()
    for i, ranks in enumerate(RANKS):
        user = _perturbed_user(**ranks)
        path = save_adapter_module(user, tmp_path / f"u{i}.pt", kind="user")
        loaded, kind, _ = load_adapter_module(path)
        assert kind == "user" and loaded.config == user.config
        assert torch.equal(loaded.velocity(x, t, v), user.velocity(x, t, v))


def test_step_rank_shares_the_corrections_and_cross_step_rank_couples_the_stacked_chunk():
    x, t, v = _inputs(b=2)
    with torch.no_grad():
        # (B, H, 9); the rank to 1e-3 of the largest singular value, above the identity's float rounding
        delta = (_perturbed_user(step_rank=1)(x, t) - torch.eye(3)).flatten(2)
        assert torch.linalg.matrix_rank(delta[0], rtol=1e-3) == 1
        assert torch.linalg.matrix_rank(delta[1], rtol=1e-3) == 1
        free = (_perturbed_user()(x, t) - torch.eye(3)).flatten(2)[0]
        assert torch.linalg.matrix_rank(free, rtol=1e-3) == 6
        user = _perturbed_user(cross_step_rank=2)
        uv = user.cross(user._hidden(x, t)).view(2, 2, 6, 3, 2)
        u, w = uv[:, 0].reshape(2, 18, 2), uv[:, 1].reshape(2, 18, 2)  # the stacked chunk's (s a) x q factors
        coupling = (u @ w.transpose(1, 2) @ v.reshape(2, 18, 1)).view(2, 6, 3)
        per_step = torch.einsum("bsij,bsj->bsi", user(x, t), v)
        torch.testing.assert_close(user.velocity(x, t, v), per_step + coupling)
        assert torch.linalg.matrix_rank(u[0] @ w[0].T) == 2


def test_cross_step_dims_couple_only_the_listed_dims():
    x, t, v = _inputs()
    with torch.no_grad():
        user = _perturbed_user(cross_step_rank=2, cross_step_dims=[2, 0])
        term = user.velocity(x, t, v) - torch.einsum("bsij,bsj->bsi", user(x, t), v)
        assert torch.equal(term[..., 1], torch.zeros_like(term[..., 1]))
        assert term[..., 0].abs().min() > 0 and term[..., 2].abs().min() > 0
        moved = v.clone()
        moved[..., 1] += 1.0  # the uncoupled dim does not feed the coupling either
        term_moved = user.velocity(x, t, moved) - torch.einsum("bsij,bsj->bsi", user(x, t), moved)
        torch.testing.assert_close(term_moved, term)
        uv = user.cross(user._hidden(x, t)).view(3, 2, 6, 2, 2)  # dims in the listed order: 2 then 0
        m = torch.einsum("bsar,bsa->br", uv[:, 1], v[..., [2, 0]])
        expect = torch.einsum("bsar,br->bsa", uv[:, 0], m)
        torch.testing.assert_close(term[..., 2], expect[..., 0])
        torch.testing.assert_close(term[..., 0], expect[..., 1])


def test_cross_step_dims_none_equals_listing_every_dim():
    x, t, v = _inputs()
    every = _perturbed_user(cross_step_rank=2, cross_step_dims=[0, 1, 2])
    none = LearnedUserModule(6, 3, d=32, tdim=8, cross_step_rank=2)
    none.load_state_dict(every.state_dict())
    with torch.no_grad():
        assert torch.equal(none.eval().velocity(x, t, v), every.velocity(x, t, v))


def test_invalid_cross_step_dims_raise():
    for dims in ([], [0, 0], [3], [-1], [0, 5]):
        with pytest.raises(ValueError, match="cross_step_dims"):
            LearnedUserModule(6, 3, cross_step_rank=2, cross_step_dims=dims)
    with pytest.raises(ValueError, match="cross_step_rank"):
        LearnedUserModule(6, 3, cross_step_dims=[0])
    with pytest.raises(ValueError, match="step_rank"):
        LearnedUserModule(6, 3, step_rank=0)


def test_libero_shape_parameter_count_and_outputs():
    """H = 50 and n = 7 (LIBERO): the cross head has 2 * 50 * 7 * q = 700 q outputs; x_t carries pi0.5's padding."""
    q, d = 2, 256
    base = LearnedUserModule(50, 7)
    user = LearnedUserModule(50, 7, cross_step_rank=q)
    assert user.cross.out_features == 700 * q
    n_base = sum(p.numel() for p in base.parameters())
    assert sum(p.numel() for p in user.parameters()) == n_base + d * 700 * q + 700 * q
    x, v = torch.randn(2, 50, 32), torch.randn(2, 50, 7)
    assert user(x, 0.3).shape == (2, 50, 7, 7)
    assert user.velocity(x, torch.rand(2), v).shape == (2, 50, 7)
    field = AdaptedField(None, user).adapt(x, 0.3, torch.randn(2, 50, 32), backward=True)
    assert field.shape == (2, 50, 32)


def test_the_core_applies_a_ranked_module_through_its_velocity():
    """AdaptedField (backward only; the padding dims pass through), the flows built on it and the user loss call the
    module's velocity, coupling included; a callable without one keeps the per-step einsum."""
    x, t, v = _inputs()
    user = _perturbed_user(step_rank=2, cross_step_rank=2)
    full = torch.cat([v, torch.randn(3, 6, 2)], -1)
    with torch.no_grad():
        expect = torch.cat([user.velocity(x, t, v), full[..., 3:]], -1)
        assert torch.equal(AdaptedField(None, user).adapt(x, t, full, backward=True), expect)
        assert torch.equal(AdaptedField(None, user).adapt(x, t, full, backward=False), full)
        coupling = user.velocity(x, t, v) - torch.einsum("bsij,bsj->bsi", user(x, t), v)
        assert coupling.abs().max() > 1e-3  # the coupling is really there

        def velocity(x_, t_):
            return 0.5 * x_

        stepped = integrate(full, velocity, 0.0, 0.1, 1, AdaptedField(None, user), backward=True)
        torch.testing.assert_close(
            stepped, full + 0.1 * torch.cat([user.velocity(full, 0.0, 0.5 * v), 0.5 * full[..., 3:]], -1)
        )
        latent, target = torch.randn(3, 6, 3), torch.randn(3, 6, 3)
        xs = (1 - t[:, None, None]) * latent + t[:, None, None] * target
        loss = ((user.velocity(xs, t, velocity(xs, t)) - (target - latent)) ** 2).mean((1, 2)).mean()
        assert torch.equal(user_field_loss(user, velocity, latent, target, t), loss)
    matrices = user(x, t).detach()
    plain = apply_user_module(lambda x_, t_: matrices, x, t, v)
    assert torch.equal(plain, torch.einsum("bhij,bhj->bhi", matrices, v))


def test_steered_chunk_reversal_uses_the_coupling():
    """steered_chunk's frs (cursor convention) with a cross-step module: the reversal sees the coupling, so it
    differs from the same module with the coupling cut, while the policy mode (no reversal) does not."""
    user = _perturbed_user(seed=3, horizon=6, n_dims=2, cross_step_rank=2)
    cut = LearnedUserModule(6, 2, d=32, tdim=8)
    cut.load_state_dict({k: p for k, p in user.state_dict().items() if not k.startswith("cross")})
    command = torch.tensor([[0.8, 0.1], [-0.5, 0.6]])

    def velocity(x_, t_):
        return torch.tanh(x_) - 0.3 * t_[:, None, None]

    def chunk(module, mode):
        torch.manual_seed(0)
        return steering.steered_chunk(
            velocity,
            command,
            mode,
            horizon=6,
            to_reference=lambda c: c,
            adapter=AdaptedField(None, module.eval(), n_dims=2),
        )

    assert not torch.allclose(chunk(user, "frs"), chunk(cut, "frs"))
    assert torch.equal(chunk(user, "policy"), chunk(cut, "policy"))


def test_convert_cursor_module_maps_the_ranks():
    sd = {}
    plain = steering.convert_cursor_module(
        {"config": {"horizon": 16, "d": 256, "tdim": 64}, "state_dict": sd}
    )
    assert plain["config"] == {"horizon": 16, "n_dims": 2, "d": 256, "tdim": 64}
    cursor = {"horizon": 16, "d": 256, "tdim": 64, "step_rank": None, "cross_step_rank": 8}
    ranked = steering.convert_cursor_module({"kind": "user", "config": cursor, "state_dict": sd})
    assert ranked["config"] == {"horizon": 16, "n_dims": 2, "d": 256, "tdim": 64, "cross_step_rank": 8}
    cursor = {"horizon": 16, "d": 256, "tdim": 64, "step_rank": 2, "cross_step_rank": 0}
    ranked = steering.convert_cursor_module({"kind": "user", "config": cursor, "state_dict": sd})
    assert ranked["config"] == {"horizon": 16, "n_dims": 2, "d": 256, "tdim": 64, "step_rank": 2}
    assert "cross_step_dims" not in ranked["config"]


def test_frs_chunk_pinned_steps_in_paints_the_later_steps():
    """frs_chunk with pinned_steps (Tang et al., App. D): the first steps are the steered chunk's, the later ones the
    plain chunk's from the same noise (a field acting per step keeps the two apart); a pin covering the whole chunk,
    or None, is the unpinned chunk, for a full and a partial reversal."""
    g = torch.Generator().manual_seed(3)
    w, c = torch.randn(2, 2, generator=g), torch.randn(2, generator=g)
    field = lambda x, t: x @ w.T * (1.0 + t)[:, None, None] + c  # noqa: E731
    z, ref = torch.randn(4, 16, 2, generator=g), torch.randn(4, 16, 2, generator=g)
    steer = torch.ones(4, 1, 1, dtype=torch.bool)
    plain = integrate(z, field, 0.0, 1 / 10, 10, batch_time=True)
    for n in (None, 6):
        whole = steering.frs_chunk(field, z, ref, steer, n, 10)
        assert torch.equal(steering.frs_chunk(field, z, ref, steer, n, 10, pinned_steps=16), whole)
        pinned = steering.frs_chunk(field, z, ref, steer, n, 10, pinned_steps=5)
        torch.testing.assert_close(pinned[:, :5], whole[:, :5])
        if n is None:  # the later steps start from z itself, so they are the plain chunk
            torch.testing.assert_close(pinned[:, 5:], plain[:, 5:])
        else:  # from the plain flow at t = 0.4, run on in the partial reversal's step
            mid = integrate(z, field, 0.0, 1 / 10, 4, batch_time=True)
            torch.testing.assert_close(
                pinned[:, 5:], integrate(mid, field, 0.4, 0.6 / 4, 4, batch_time=True)[:, 5:]
            )


def test_steered_chunk_rejects_a_pin_outside_the_chunk():
    field = lambda x, t: x  # noqa: E731
    for bad in (0, 17):
        with pytest.raises(ValueError, match="pinned_steps"):
            steering.steered_chunk(
                field, torch.ones(2, 2), "frs", horizon=16, to_reference=lambda d: d, pinned_steps=bad
            )
