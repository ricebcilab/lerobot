"""The composed reversal adapter: a user module (backward pass) and a task module (both passes)."""

import numpy as np
import pytest
import torch

from cursor_sa import config, session
from cursor_sa.agents import ExpertChunkAgent
from lerobot.envs.cursor import TaskConfig, sample_layouts
from cursor_sa.paths import adapter_path
from cursor_sa.policy import FlowPolicy
from lerobot.policies.steering import AdapterModules, LearnedTaskModule, LearnedUserModule

from cursor_sa.steering import SharedController, flow, forward_flow, reverse_flow, steered_chunk

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
    task = LearnedTaskModule()
    torch.manual_seed(seed)
    torch.nn.init.normal_(task.out.weight, std=0.3)
    torch.nn.init.normal_(task.out.bias, std=0.3)
    return task.eval()


def _steered(model, obs, mode, command, adapter=None, seed=1):
    torch.manual_seed(seed)
    return steered_chunk(model, obs, command, mode, tau=3, n_reverse_steps=4, adapter=adapter)


def test_fresh_task_module_is_the_identity_in_every_mode():
    model, obs = FlowPolicy().eval(), _obs(obstacles=2)
    command = torch.tensor([[0.8, 0.0], [0.0, -0.5], [0.3, 0.3]])
    fresh = AdapterModules(task=LearnedTaskModule()).eval()
    assert fresh.kind == "task"
    for mode in MODES:
        torch.testing.assert_close(_steered(model, obs, mode, command, fresh), _steered(model, obs, mode, command))
    x = torch.randn(3, model.horizon, 2)
    torch.testing.assert_close(reverse_flow(model, obs, x, 4, 10, fresh), reverse_flow(model, obs, x, 4, 10))
    # without obstacle keys in the observation (the tests' plain obs) the module still runs, as the identity
    torch.testing.assert_close(_steered(model, _obs(), "frs", command, fresh), _steered(model, _obs(), "frs", command))


def test_composed_adapter_applies_task_forward_and_user_task_backward():
    model, obs = FlowPolicy().eval(), _obs(obstacles=2)
    task, F = _perturbed_task(), torch.diag(torch.tensor([1.0, 0.5]))
    adapter = AdapterModules(task=task, matrix=F).eval()
    assert adapter.kind == "composed"
    x, t = torch.randn(3, model.horizon, 2), torch.full((3,), 0.4)
    with torch.no_grad():
        v = model(obs, x, t)
        A, b = task(x, t, v, obs)
        expect_forward = torch.einsum("bsij,bsj->bsi", A, v) + b
        torch.testing.assert_close(adapter.adapt(obs, x, t, v, backward=False), expect_forward)
        torch.testing.assert_close(adapter.adapt(obs, x, t, v, backward=True), torch.einsum("ij,bsj->bsi", F, expect_forward))
        assert not torch.allclose(expect_forward, v)  # the perturbed task module does change the field
        # one Euler step through the adapted field, forward and backward
        v0 = model(obs, x, torch.zeros(3))
        A0, b0 = task(x, torch.zeros(3), v0, obs)
        torch.testing.assert_close(flow(model, obs, x, 0.0, 0.1, 1, adapter), x + 0.1 * (torch.einsum("bsij,bsj->bsi", A0, v0) + b0))
        v1 = model(obs, x, torch.ones(3))
        A1, b1 = task(x, torch.ones(3), v1, obs)
        back = torch.einsum("ij,bsj->bsi", F, torch.einsum("bsij,bsj->bsi", A1, v1) + b1)
        torch.testing.assert_close(reverse_flow(model, obs, x, 1, 10, adapter), x - 0.1 * back)
    # a user-only adapter leaves the forward pass alone
    user_only = AdapterModules(matrix=F)
    torch.testing.assert_close(forward_flow(model, obs, x, 0.0, 1.0, 10, user_only), forward_flow(model, obs, x, 0.0, 1.0, 10))
    # the flow core is differentiable (the trainers' path), the wrappers are not
    model.requires_grad_(False)
    user = LearnedUserModule(16, n_dims=2)
    flow(model, obs, x, 1.0, -0.1, 3, AdapterModules(user=user)).sum().backward()
    assert user.net[-1].weight.grad is not None and user.net[-1].weight.grad.abs().sum() > 0
    assert not reverse_flow(model, obs, x, 3, 10, AdapterModules(user=user)).requires_grad


def test_old_adapter_checkpoint_loads_as_a_user_only_adapter():
    adapter = AdapterModules.load(adapter_path("learned_adapter/gain025y_free"))
    assert adapter.kind == "user" and adapter.task is None and isinstance(adapter.user, LearnedUserModule)
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
    single = config.load_arm("base.yaml", {"mode": "frs", "reversal_adapter": {"learned": "learned_adapter/gain025y_free"}, **gain})
    assert single["learned_adapter"] == "learned_adapter/gain025y_free" and single["method"] == "mars" and config.adapter_names(single) == ["learned_adapter/gain025y_free"]
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
    assert config.adapter_names(config.load_arm("base.yaml", {"mode": "teleop"})) == []
    with pytest.raises(ValueError, match="reversal_adapter"):
        config.load_arm("base.yaml", {"reversal_adapter": {"bogus": 1}})

    # tiny checkpoints of both kinds, assembled into one composed adapter by the session
    for name, module, kind in (("tiny_user", LearnedUserModule(16, n_dims=2), "user"), ("tiny_task", LearnedTaskModule(), "task")):
        (tmp_path / name).mkdir()
        torch.save(dict(kind=kind, config=module.config, state_dict=module.state_dict(), args={}), tmp_path / name / "adapter.pt")
    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / name / "adapter.pt")
    policies, adapters = session.load_checkpoints({"a": both, "b": fixed, "c": task_only}, "cpu")
    assert set(adapters) == {"tiny_user", "tiny_task"}
    composed = session.arm_adapter(both, adapters)
    assert composed.kind == "composed" and isinstance(composed.user, LearnedUserModule) and isinstance(composed.task, LearnedTaskModule)
    with_fixed = session.arm_adapter(fixed, adapters)
    assert with_fixed.kind == "user" and with_fixed.user is None and with_fixed.matrix is not None
    assert session.arm_adapter(task_only, adapters).kind == "task"  # served in every mode
    assert set(session.adapter_checkpoints({"a": both})) == {"adapters/tiny_user", "adapters/tiny_task"}
    assert session.adapter_checkpoints({"t": config.load_arm("base.yaml", {"mode": "teleop"})}) == {}


def test_expert_chunk_plans_through_the_adapted_field():
    arm = config.load_arm("base.yaml", {"mode": "teleop"})
    policies, _ = session.load_checkpoints({arm["name"]: arm}, "cpu")
    policy = policies[arm["model"]]
    layouts = sample_layouts(np.random.default_rng(2), 3, 4, TaskConfig())
    plain = SharedController(policy, layouts, "teleop", steps=10, execute=2)
    adapted = SharedController(policy, layouts, "teleop", steps=10, execute=2, adapter=AdapterModules(task=_perturbed_task()))
    fresh = SharedController(policy, layouts, "teleop", steps=10, execute=2, adapter=AdapterModules(task=LearnedTaskModule()))
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
