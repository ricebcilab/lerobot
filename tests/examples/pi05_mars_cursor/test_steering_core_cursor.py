"""The lerobot steering core against the frozen cursor steering (`_reference_steering`)."""

import _reference_steering as ref
import torch
from cursor_sa.policy import FlowPolicy

from lerobot.policies.pi05.steering import (
    LearnedUserModule,
    ObstacleTaskModule,
    convert_cursor_module,
    integrate,
    load_adapter_module,
)


def _obs(b=3, n=5, hist=4, k=2):
    g = torch.Generator().manual_seed(0)
    goal = torch.zeros(b, n, dtype=torch.bool)
    goal[torch.arange(b), torch.randint(n, (b,), generator=g)] = True
    obstacles = torch.cat(
        [torch.rand(b, k, 2, generator=g), 0.05 + 0.1 * torch.rand(b, k, 1, generator=g)], -1
    )
    return {
        "pos": torch.rand(b, 2, generator=g),
        "hist": 0.01 * torch.randn(b, hist, 2, generator=g),
        "targets": torch.rand(b, n, 2, generator=g),
        "mask": torch.ones(b, n, dtype=torch.bool),
        "goal": goal,
        "obstacles": obstacles,
        "obstacle_mask": torch.ones(b, k, dtype=torch.bool),
    }


def _perturb(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return module


def test_integrate_reproduces_cursor_flows_exactly():
    torch.manual_seed(0)
    policy, obs = FlowPolicy().eval(), _obs()
    x = torch.randn(3, policy.horizon, 2)

    def velocity(x, t):
        return policy(obs, x, t)

    steps = 10
    for n in (4, steps):
        expect = ref.reverse_flow(policy, obs, x, n, steps)
        assert torch.equal(
            integrate(x, velocity, 1.0, -1.0 / steps, n, backward=True, batch_time=True), expect
        )
        t0 = 1.0 - n / steps
        m = steps if n == steps else steps - n
        expect = ref.forward_flow(policy, obs, x, t0, 1.0, m)
        assert torch.equal(integrate(x, velocity, t0, (1.0 - t0) / m, m, batch_time=True), expect)


def test_cursor_user_module_converts_and_matches(tmp_path):
    torch.manual_seed(0)
    old = _perturb(ref.UserModule(16)).eval()
    ckpt = {
        "config": old.config,
        "state_dict": old.state_dict(),
        "args": {"name": "u", "seed": 0},
        "obstacle": None,
    }
    assert "kind" not in ckpt  # the first adapters were saved without one
    converted = convert_cursor_module(ckpt)
    assert converted["kind"] == "user" and converted["meta"] == {
        "args": {"name": "u", "seed": 0},
        "obstacle": None,
    }
    torch.save(converted, tmp_path / "user.pt")
    new, kind, _ = load_adapter_module(tmp_path / "user.pt")
    assert isinstance(new, LearnedUserModule) and kind == "user" and new.n_dims == 2
    x, t = torch.randn(3, 16, 2), torch.rand(3)
    with torch.no_grad():
        torch.testing.assert_close(new(x, t), old(x, t), atol=1e-5, rtol=0)


def test_cursor_task_module_converts_and_matches(tmp_path):
    torch.manual_seed(0)
    old = _perturb(ref.TaskModule(16)).eval()
    obstacle = {"kind": "disc"}
    ckpt = {
        "kind": "task",
        "config": old.config,
        "state_dict": old.state_dict(),
        "args": {},
        "obstacle": obstacle,
    }
    converted = convert_cursor_module(ckpt)
    assert converted["kind"] == "task" and converted["arch"] == "obstacle"
    torch.save(converted, tmp_path / "task.pt")
    new, kind, meta = load_adapter_module(tmp_path / "task.pt")
    assert isinstance(new, ObstacleTaskModule) and kind == "task" and meta["obstacle"] == obstacle
    obs = _obs()
    x, v, t = torch.randn(3, 16, 2), torch.randn(3, 16, 2), torch.rand(3)
    with torch.no_grad():
        for a, b in zip(new(obs, x, t, v), old(obs, x, t, v), strict=True):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)  # deeper net, more rounding
