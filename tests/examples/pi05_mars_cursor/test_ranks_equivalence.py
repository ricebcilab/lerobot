"""The ranked user module (step_rank, cross_step_rank) on the lerobot core against the original cursor project's
(`_reference_ranks`, ricebcirig bd2a48f2): same weights, same corrected velocity, same backward field, and a
cursor-format checkpoint loads into the same module.

As in `test_steering_equivalence`, the core embeds flow time with `sinusoidal_time_embedding` (10 ** x) and the
cursor with `time_embedding` (exp(x log 10)): as shipped the two agree to that rounding (held to 1e-4); with the
cursor's embedding patched into the core (`exact`) they agree bitwise."""

import _reference_ranks as ref
import pytest
import torch
from cursor_sa import steering
from cursor_sa.policy import FlowPolicy, time_embedding

import lerobot.policies.pi05.steering as core

RANKS = [
    {},
    {"step_rank": 1},
    {"step_rank": 2},
    {"cross_step_rank": 2},
    {"cross_step_rank": 8},
    {"cross_step_rank": 32},
    {"step_rank": 2, "cross_step_rank": 4},
]


@pytest.fixture(params=[False, True], ids=["shipped", "exact"])
def exact(request, monkeypatch):
    """With True, the core's modules embed time with the cursor's `time_embedding`."""
    if request.param:

        def cursor_embedding(t, dim, device=None, dtype=None):
            return time_embedding(torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1), dim)

        monkeypatch.setattr(core, "sinusoidal_time_embedding", cursor_embedding)
    return request.param


def _close(new, old, exact):
    if exact:
        assert torch.equal(new, old), (new - old).abs().max()
    else:
        torch.testing.assert_close(new, old, atol=1e-4, rtol=1e-4)


def _reference(seed, scale=0.1, **ranks):
    torch.manual_seed(seed)
    user = ref.UserModule(**ranks)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for p in user.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))
    return user.eval()


def _twin(old):
    """The example's UserModule with the reference's weights, through the core's checkpoint conversion."""
    new = steering.UserModule(
        **core.convert_cursor_module({"kind": "user", "config": old.config, "state_dict": old.state_dict()})[
            "config"
        ]
    )
    new.load_state_dict(old.state_dict())
    return new.eval()


def test_a_fresh_module_draws_the_same_weights_as_the_cursor():
    for ranks in RANKS:
        torch.manual_seed(7)
        old = ref.UserModule(**ranks)
        torch.manual_seed(7)
        new = steering.UserModule(**ranks)
        assert old.state_dict().keys() == new.state_dict().keys(), ranks
        assert all(torch.equal(old.state_dict()[k], new.state_dict()[k]) for k in old.state_dict()), ranks


@pytest.mark.parametrize("ranks", RANKS, ids=lambda r: "_".join(f"{k}{v}" for k, v in r.items()) or "default")
def test_velocity_matches_the_cursor(exact, ranks):
    old = _reference(1, **ranks)
    new = _twin(old)
    g = torch.Generator().manual_seed(2)
    x, t, v = (
        torch.randn(5, 16, 2, generator=g),
        torch.rand(5, generator=g),
        torch.randn(5, 16, 2, generator=g),
    )
    with torch.no_grad():
        _close(new(x, t), old(x, t), exact)
        _close(new.velocity(x, t, v), old.velocity(x, t, v), exact)
        _close(steering.user_velocity(new, x, t, v), ref.user_velocity(old, x, t, v), exact)
        _close(
            steering.ReversalAdapter(user=new).adapt({}, x, t, v, backward=True),
            ref.ReversalAdapter(user=old).adapt({}, x, t, v, backward=True),
            exact,
        )


@pytest.mark.parametrize("ranks", [{"cross_step_rank": 8}, {"step_rank": 2, "cross_step_rank": 4}])
def test_reversal_through_the_policy_matches_the_cursor(exact, ranks):
    """MARS-U's reversal through a policy's field (the coupling acting at every step), cursor against core."""
    torch.manual_seed(0)
    policy = FlowPolicy().eval()
    old = _reference(3, scale=0.01, **ranks)  # small enough for ten Euler steps to stay finite
    new = _twin(old)
    g = torch.Generator().manual_seed(4)
    obs = {
        "pos": torch.rand(4, 2, generator=g),
        "hist": 0.01 * torch.randn(4, policy.hist, 2, generator=g),
        "targets": torch.rand(4, 5, 2, generator=g),
        "mask": torch.ones(4, 5, dtype=torch.bool),
        "goal": torch.zeros(4, 5, dtype=torch.bool),
    }
    chunk = 0.5 * torch.randn(4, policy.horizon, 2, generator=g)
    old_z = ref.reverse_flow(policy, obs, chunk, 10, 10, ref.ReversalAdapter(user=old))
    new_z = steering.reverse_flow(policy, obs, chunk, 10, 10, steering.ReversalAdapter(user=new))
    assert (
        torch.isfinite(old_z).all()
        and (old_z - steering.reverse_flow(policy, obs, chunk, 10, 10)).abs().max() > 1e-3
    )
    _close(new_z, old_z, exact)


def test_a_cursor_checkpoint_loads_into_the_same_module(exact, tmp_path):
    old = _reference(5, step_rank=None, cross_step_rank=2)
    torch.save(
        {"kind": "user", "config": old.config, "state_dict": old.state_dict(), "args": {}},
        tmp_path / "adapter.pt",
    )
    new = steering.ReversalAdapter.load(tmp_path / "adapter.pt").user
    assert new.config == {
        "horizon": 16,
        "n_dims": 2,
        "d": 256,
        "tdim": 64,
        "time_embedding": "mars",
        "cross_step_rank": 2,
    }
    g = torch.Generator().manual_seed(6)
    x, t, v = (
        torch.randn(3, 16, 2, generator=g),
        torch.rand(3, generator=g),
        torch.randn(3, 16, 2, generator=g),
    )
    with torch.no_grad():
        _close(new.velocity(x, t, v), old.velocity(x, t, v), exact)
