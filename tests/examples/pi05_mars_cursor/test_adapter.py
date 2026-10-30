"""The composed reversal adapter: a user module (backward pass) and a task module (both passes)."""

import numpy as np
import pytest
import torch

from cursor_sa import config, session
from cursor_sa.agents import ExpertChunkAgent
from cursor_sa.env import TaskConfig, sample_layouts
from cursor_sa.paths import adapter_path
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import (FixedAdapter, ReversalAdapter, SharedController, TaskModule, UserModule, flow,
                                forward_flow, observation, reverse_flow, steered_chunk)

MODES = ("policy", "flow_control", "frs")


def _obs(B=3, N=5, hist=4, obstacles=0):
    g = torch.Generator().manual_seed(0)
    goal = torch.zeros(B, N, dtype=torch.bool)
    goal[torch.arange(B), torch.randint(N, (B,), generator=g)] = True
    obs = dict(pos=torch.rand(B, 2, generator=g), hist=0.01 * torch.randn(B, hist, 2, generator=g),
               targets=torch.rand(B, N, 2, generator=g), mask=torch.ones(B, N, dtype=torch.bool), goal=goal)
    if obstacles:
        obs["obstacles"] = torch.cat([torch.rand(B, obstacles, 2, generator=g), torch.full((B, obstacles, 1), 0.05)], -1)
        obs["obstacle_mask"] = torch.ones(B, obstacles, dtype=torch.bool)
    return obs


def _perturbed_task(seed=1):
    task = TaskModule()
    torch.manual_seed(seed)
    torch.nn.init.normal_(task.out.weight, std=0.3)
    torch.nn.init.normal_(task.out.bias, std=0.3)
    return task.eval()


def _chunk(model, obs, mode, command, adapter=None, seed=1):
    torch.manual_seed(seed)
    return steered_chunk(model, obs, command, mode, tau=3, n_reverse_steps=4, adapter=adapter)


def test_fresh_task_module_is_the_identity_in_every_mode():
    model, obs = FlowPolicy().eval(), _obs(obstacles=2)
    command = torch.tensor([[0.8, 0.0], [0.0, -0.5], [0.3, 0.3]])
    fresh = ReversalAdapter(task=TaskModule()).eval()
    assert fresh.kind == "task"
    for mode in MODES:
        torch.testing.assert_close(_chunk(model, obs, mode, command, fresh), _chunk(model, obs, mode, command))
    x = torch.randn(3, model.horizon, 2)
    torch.testing.assert_close(reverse_flow(model, obs, x, 4, 10, fresh), reverse_flow(model, obs, x, 4, 10))
    # without obstacle keys in the observation (the tests' plain obs) the module still runs, as the identity
    torch.testing.assert_close(_chunk(model, _obs(), "frs", command, fresh), _chunk(model, _obs(), "frs", command))


def test_composed_adapter_applies_task_forward_and_user_task_backward():
    model, obs = FlowPolicy().eval(), _obs(obstacles=2)
    task, F = _perturbed_task(), torch.diag(torch.tensor([1.0, 0.5]))
    adapter = ReversalAdapter(user=FixedAdapter(F), task=task).eval()
    assert adapter.kind == "composed"
    x, t = torch.randn(3, model.horizon, 2), torch.full((3,), 0.4)
    with torch.no_grad():
        v = model(obs, x, t)
        A, b = task(obs, x, t, v)
        expect_forward = torch.einsum("bsij,bsj->bsi", A, v) + b
        torch.testing.assert_close(adapter.adapt(obs, x, t, v, backward=False), expect_forward)
        torch.testing.assert_close(adapter.adapt(obs, x, t, v, backward=True), torch.einsum("ij,bsj->bsi", F, expect_forward))
        assert not torch.allclose(expect_forward, v)  # the perturbed task module does change the field
        # one Euler step through the adapted field, forward and backward
        v0 = model(obs, x, torch.zeros(3))
        A0, b0 = task(obs, x, torch.zeros(3), v0)
        torch.testing.assert_close(flow(model, obs, x, 0.0, 0.1, 1, adapter), x + 0.1 * (torch.einsum("bsij,bsj->bsi", A0, v0) + b0))
        v1 = model(obs, x, torch.ones(3))
        A1, b1 = task(obs, x, torch.ones(3), v1)
        back = torch.einsum("ij,bsj->bsi", F, torch.einsum("bsij,bsj->bsi", A1, v1) + b1)
        torch.testing.assert_close(reverse_flow(model, obs, x, 1, 10, adapter), x - 0.1 * back)
    # a user-only adapter leaves the forward pass alone
    user_only = ReversalAdapter(user=FixedAdapter(F))
    torch.testing.assert_close(forward_flow(model, obs, x, 0.0, 1.0, 10, user_only), forward_flow(model, obs, x, 0.0, 1.0, 10))


def test_old_adapter_checkpoint_loads_as_a_user_only_adapter():
    adapter = ReversalAdapter.load(adapter_path("gain025y_free"))
    assert adapter.kind == "user" and adapter.task is None and isinstance(adapter.user, UserModule)
    model, obs = FlowPolicy().eval(), _obs()
    x = torch.randn(3, model.horizon, 2)
    with torch.no_grad():
        manual, h = x.clone(), 0.1
        for i in range(4):
            t = torch.full((3,), 1.0 - i * h)
            manual = manual - h * torch.einsum("bsij,bsj->bsi", adapter.user(manual, t), model(obs, manual, t))
    torch.testing.assert_close(reverse_flow(model, obs, x, 4, 10, adapter), manual)


def test_adapter_specs_accept_single_and_composite_forms(tmp_path, monkeypatch):
    gain = {"corruption": {"scale": [1.0, 0.25]}}
    single = config.load_arm("base.yaml", {"mode": "frs", "reversal_adapter": {"learned": "gain025y_free"}, **gain})
    assert single["learned_adapter"] == "gain025y_free" and single["method"] == "mars" and config.adapter_names(single) == ["gain025y_free"]
    fixed = config.load_arm("base.yaml", {"mode": "frs", "reversal_adapter": {"translation": "corruption"}, **gain})
    np.testing.assert_allclose(fixed["adapter_matrix"], np.diag([1.0, 0.25]))
    assert fixed["learned_adapter"] is None and fixed["method"] == "mars" and config.adapter_names(fixed) == []
    with pytest.raises(ValueError, match="checkpoint name"):  # a composite's slots name checkpoints
        config.load_arm("base.yaml", {"mode": "frs", "reversal_adapter": {"user": {"translation": "corruption"}, "task": "t"}, **gain})
    both = config.load_arm("base.yaml", {"mode": "frs", "reversal_adapter": {"user": "tiny_user", "task": "tiny_task"}, **gain})
    assert both["learned_adapter"] == {"user": "tiny_user", "task": "tiny_task"} and both["adapter_matrix"] is None
    assert config.adapter_names(both) == ["tiny_user", "tiny_task"]
    task_only = config.load_arm("base.yaml", {"mode": "teleop", "reversal_adapter": {"task": "tiny_task"}})
    assert task_only["learned_adapter"] == {"user": None, "task": "tiny_task"} and task_only["method"] == "teleop"
    assert config.adapter_names(config.load_arm("teleop.yaml")) == []
    with pytest.raises(ValueError, match="reversal_adapter"):
        config.load_arm("base.yaml", {"reversal_adapter": {"bogus": 1}})

    # tiny checkpoints of both kinds, assembled into one composed adapter by the session
    for name, module, kind in (("tiny_user", UserModule(), "user"), ("tiny_task", TaskModule(), "task")):
        (tmp_path / name).mkdir()
        torch.save(dict(kind=kind, config=module.config, state_dict=module.state_dict(), args={}), tmp_path / name / "adapter.pt")
    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / name / "adapter.pt")
    policies, adapters = session.load_checkpoints({"a": both, "b": fixed, "c": task_only}, "cpu")
    assert set(adapters) == {"tiny_user", "tiny_task"}
    composed = session.arm_adapter(both, adapters)
    assert composed.kind == "composed" and isinstance(composed.user, UserModule) and isinstance(composed.task, TaskModule)
    with_fixed = session.arm_adapter(fixed, adapters)
    assert with_fixed.kind == "user" and isinstance(with_fixed.user, FixedAdapter)
    assert session.arm_adapter(task_only, adapters).kind == "task"  # served in every mode
    assert set(session.adapter_checkpoints({"a": both})) == {"adapters/tiny_user", "adapters/tiny_task"}
    assert session.adapter_checkpoints({"t": config.load_arm("teleop.yaml")}) == {}


def test_expert_chunk_plans_through_the_adapted_field():
    arm = config.load_arm("teleop.yaml")
    policies, _ = session.load_checkpoints({arm["name"]: arm}, "cpu")
    policy = policies[arm["model"]]
    layouts = sample_layouts(np.random.default_rng(2), 3, 4, TaskConfig())
    plain = SharedController(policy, layouts, "teleop", steps=10, execute=2)
    adapted = SharedController(policy, layouts, "teleop", steps=10, execute=2, adapter=ReversalAdapter(task=_perturbed_task()))
    fresh = SharedController(policy, layouts, "teleop", steps=10, execute=2, adapter=ReversalAdapter(task=TaskModule()))
    agent = ExpertChunkAgent()
    agent.reset(3)
    torch.manual_seed(0)
    a = agent.plan(plain, layouts)
    torch.manual_seed(0)
    b = agent.plan(adapted, layouts)
    torch.manual_seed(0)
    c = agent.plan(fresh, layouts)
    np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-6)
    assert not np.allclose(a, b)


def test_flow_core_is_differentiable_and_the_wrappers_are_not():
    model, obs = FlowPolicy().eval(), _obs()
    for p in model.parameters():
        p.requires_grad_(False)
    user = UserModule()
    x = torch.randn(3, model.horizon, 2)
    z = flow(model, obs, x, 1.0, -0.1, 3, ReversalAdapter(user=user))
    z.sum().backward()
    assert user.net[-1].weight.grad is not None and user.net[-1].weight.grad.abs().sum() > 0
    assert not reverse_flow(model, obs, x, 3, 10, ReversalAdapter(user=user)).requires_grad
    assert not forward_flow(model, obs, x, 0.0, 1.0, 10).requires_grad


def test_observation_carries_the_obstacles():
    layouts = sample_layouts(np.random.default_rng(0), 2, 4, TaskConfig(min_reach=0.35, max_reach=0.5, obstacle={
        "kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.0, "side": "random"}))
    obs = observation(layouts, layouts.start, np.zeros((2, 4, 2)))
    assert obs["obstacles"].shape == (2, 1, 3) and obs["obstacle_mask"].tolist() == [[True], [True]]


def _perturbed_user(seed=2, **ranks):
    torch.manual_seed(seed)
    user = UserModule(**ranks)
    with torch.no_grad():
        for p in user.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return user.eval()


def test_user_module_ranks_default_to_todays_module_and_start_at_the_identity(tmp_path):
    x, t, v = torch.randn(3, 16, 2), torch.rand(3), torch.randn(3, 16, 2)
    # older checkpoints load as they were
    assert UserModule().config == dict(horizon=16, n_dims=2, d=256, tdim=64, time_embedding="mars")
    user = _perturbed_user()
    torch.testing.assert_close(user.velocity(x, t, v), torch.einsum("bsij,bsj->bsi", user(x, t), v))
    for ranks in (dict(step_rank=1), dict(step_rank=2), dict(cross_step_rank=8), dict(step_rank=2, cross_step_rank=4),
                  dict(cross_step_rank=2, cross_step_dims=[1])):
        with torch.no_grad():
            torch.testing.assert_close(UserModule(**ranks).velocity(x, t, v), v)
        user = _perturbed_user(**ranks)   # trained modules save and load with their knobs
        torch.save(dict(kind="user", config=user.config, state_dict=user.state_dict(), args={}), tmp_path / "u.pt")
        loaded = ReversalAdapter.load(tmp_path / "u.pt").user
        assert loaded.config == user.config
        torch.testing.assert_close(loaded.velocity(x, t, v), user.velocity(x, t, v))


def test_step_rank_shares_the_steps_corrections_and_cross_step_rank_couples_the_steps():
    x, t, v = torch.randn(2, 16, 2), torch.rand(2), torch.randn(2, 16, 2)
    with torch.no_grad():
        delta = (_perturbed_user(step_rank=1)(x, t) - torch.eye(2)).flatten(2)   # [B, H, 4]
        assert torch.linalg.matrix_rank(delta[0]) == 1 and torch.linalg.matrix_rank(delta[1]) == 1
        assert torch.linalg.matrix_rank((_perturbed_user()(x, t) - torch.eye(2)).flatten(2)[0]) == 4
        user = _perturbed_user(cross_step_rank=3)
        uv = user.cross(user._hidden(x, t)).view(2, 2, 16, 2, 3)
        U, V = uv[:, 0].reshape(2, 32, 3), uv[:, 1].reshape(2, 32, 3)   # the stacked chunk's (c a) x q factors
        coupling = (U @ V.transpose(1, 2) @ v.reshape(2, 32, 1)).view(2, 16, 2)
        torch.testing.assert_close(user.velocity(x, t, v), torch.einsum("bsij,bsj->bsi", user(x, t), v) + coupling)
        assert torch.linalg.matrix_rank(U[0] @ V[0].T) == 3
        adapter = ReversalAdapter(user=user)   # the backward pass applies the coupling, the forward pass does not
        torch.testing.assert_close(adapter.adapt({}, x, t, v, backward=True), user.velocity(x, t, v))
        torch.testing.assert_close(adapter.adapt({}, x, t, v, backward=False), v)


def test_cross_step_dims_option_of_the_trainer():
    from train import train_adapter
    args = train_adapter.parse_args(["--name", "u", "--corruption", "{rotation_deg: 60}", "--cross-step-rank", "2",
                                     "--cross-step-dims", "1"])
    assert args.cross_step_dims == [1] and args.step_rank is None
    assert train_adapter.parse_args(["--name", "u", "--corruption", "{rotation_deg: 60}"]).cross_step_dims is None
    with pytest.raises(SystemExit):
        train_adapter.parse_args(["--name", "u", "--corruption", "{rotation_deg: 60}", "--cross-step-dims", "0"])
