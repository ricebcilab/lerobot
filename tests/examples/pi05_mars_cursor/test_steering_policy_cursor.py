"""lerobot's SteeringPolicy on the cursor: its chunk is the example's steered_chunk for each shared mode, with every
adapter the example composes, and its training loss is the trainer's (task_loss, field_loss) on the same draws."""

import numpy as np
import pytest
import torch
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import ReversalAdapter, observation, steered_chunk
from train import train_adapter

from lerobot.envs.cursor import TaskConfig, obstacle_spec, sample_layouts
from lerobot.policies.steering import (
    AdapterModules,
    LearnedTaskModule,
    LearnedUserModule,
    SteeringConfig,
    SteeringPolicy,
)

B = 5


def _perturb(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))  # small: a composed adapter stays finite
    return module.eval()


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("modules")
    torch.manual_seed(0)
    policy = FlowPolicy().eval()
    user = _perturb(LearnedUserModule(policy.horizon, n_dims=2, cross_step_rank=1), 1)
    task = _perturb(LearnedTaskModule(policy.horizon), 2)
    paths = (
        AdapterModules.save_module(user, root / "u.pt", "user"),
        AdapterModules.save_module(task, root / "t.pt", "task"),
    )
    task_cfg = TaskConfig(
        min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    layouts = sample_layouts(np.random.default_rng(0), B, 4, task_cfg)
    g = np.random.default_rng(1)
    obs = observation(layouts, layouts.start, 0.01 * g.standard_normal((B, policy.hist, 2)))
    command = torch.as_tensor(g.uniform(-1, 1, (B, 2)), dtype=torch.float32)
    command[1] = 0.0  # an idle row: the policy's own chunk
    reference = torch.as_tensor(g.uniform(-1, 1, (B, policy.horizon, 2)), dtype=torch.float32)
    return policy, paths, obs, command, reference


ADAPTERS = ("none", "matrix", "user", "task", "user_task")


def _adapters(kind, paths):
    """The example's adapter and the config's adapter fields for one adapter kind."""
    f = np.array([[0.8, -0.3], [0.4, 1.1]])
    user_path, task_path = (str(p) for p in paths)
    if kind == "none":
        return None, {}
    if kind == "matrix":
        return f, {"adapter_matrix": f}
    user = AdapterModules.load(user_path).user if "user" in kind else None
    task = AdapterModules.load(task_path).task if "task" in kind else None
    spec = {
        "adapter_user": user_path if user is not None else None,
        "adapter_task": task_path if task is not None else None,
    }
    return ReversalAdapter(user=user, task=task), spec


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("mode", ["flow_control", "frs"])
@pytest.mark.parametrize("per_step", [False, True], ids=["command", "reference"])
def test_predict_action_chunk_is_the_examples_steered_chunk(setup, mode, adapter, per_step):
    policy, paths, obs, command, reference = setup
    example_adapter, spec = _adapters(adapter, paths)
    knobs = {"tau": 4, "n_reverse_steps": 6, "lam": 2, "ic_gain": 1.5, "pinned_steps": 12} if per_step else {}
    torch.manual_seed(3)
    want = steered_chunk(
        policy,
        obs,
        command,
        mode,
        adapter=example_adapter,
        reference=reference if per_step else None,
        **knobs,
    )
    config = SteeringConfig(mode=mode, **spec, device="cpu")
    if per_step:
        config.tau, config.n_reversal_steps, config.lam, config.ic_gain, config.pinned_steps = (
            4,
            6,
            2,
            1.5,
            12,
        )
    steering = SteeringPolicy(config, base_policy=policy)
    torch.manual_seed(3)
    got = steering.predict_action_chunk({**obs, "command": reference if per_step else command})
    assert torch.isfinite(want).all() and torch.equal(got, want)
    torch.manual_seed(3)
    assert torch.equal(
        steering.select_action({**obs, "command": reference if per_step else command}), want[:, 0]
    )


def test_forward_is_the_trainers_losses(setup):
    policy, paths, obs, command, _ = setup
    g = torch.Generator().manual_seed(4)
    chunk = 0.01 * torch.randn(B, policy.horizon, 2, generator=g)
    z, t = torch.randn(B, policy.horizon, 2, generator=g), torch.rand(B, generator=g)
    task = AdapterModules.load(paths[1]).task
    user = AdapterModules.load(paths[0]).user
    config = SteeringConfig(
        adapter_user=str(paths[0]), adapter_task=str(paths[1]), lambda_id=0.01, device="cpu"
    )
    steering = SteeringPolicy(config, base_policy=policy)
    loss, parts = steering({**obs, "kind": "task", "chunk": chunk, "noise": z, "time": t})
    want, want_parts = train_adapter.task_loss(policy, task, obs, chunk, z, t, 0.01)
    assert torch.equal(loss, want) and parts == want_parts
    latent, bci, weight = (
        torch.randn(B, policy.horizon, 2, generator=g),
        torch.randn(B, policy.horizon, 2, generator=g),
        torch.ones(B),
    )
    torch.manual_seed(5)
    want = train_adapter.field_loss(policy, user, task, obs, latent, bci, weight)  # draws t
    torch.manual_seed(5)
    loss, _ = steering({**obs, "kind": "user", "latent": latent, "bci_target": bci, "weight": weight})
    assert torch.equal(loss, want)


def test_the_adapters_change_the_steered_chunk(setup):
    policy, paths, obs, command, _ = setup
    chunks = {}
    for kind in ("none", "user_task"):
        steering = SteeringPolicy(
            SteeringConfig(**_adapters(kind, paths)[1], device="cpu"), base_policy=policy
        )
        torch.manual_seed(3)
        chunks[kind] = steering.predict_action_chunk({**obs, "command": command})
    assert not torch.equal(chunks["none"], chunks["user_task"])
