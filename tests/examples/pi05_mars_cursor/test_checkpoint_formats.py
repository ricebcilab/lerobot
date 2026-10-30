"""The cursor checkpoints in both formats: the cursor project's .pt files and the converted lerobot ones
(convert_checkpoints.py), read by FlowPolicy.load and ReversalAdapter.load with the same results."""

import convert_checkpoints
import pytest
import torch
from cursor_sa import paths
from cursor_sa.policy import CursorFlowPolicy, FlowPolicy, load_checkpoint
from cursor_sa.steering import ReversalAdapter, TaskModule, UserModule, steered_chunk

from lerobot.policies.pi05.steering import load_adapter_module


def _obs(b=3, n=5, k=2):
    g = torch.Generator().manual_seed(0)
    goal = torch.zeros(b, n, dtype=torch.bool)
    goal[:, 1] = True
    return {
        "pos": torch.rand(b, 2, generator=g),
        "hist": 0.01 * torch.randn(b, 4, 2, generator=g),
        "targets": torch.rand(b, n, 2, generator=g),
        "mask": torch.ones(b, n, dtype=torch.bool),
        "goal": goal,
        "obstacles": torch.cat([torch.rand(b, k, 2, generator=g), torch.full((b, k, 1), 0.05)], -1),
        "obstacle_mask": torch.ones(b, k, dtype=torch.bool),
    }


def _old_policy(root, name, **kw):
    torch.manual_seed(0)
    model = FlowPolicy(d_model=32, layers=1, heads=2, d_ff=64, action_scale=0.0077, **kw).eval()
    (root / "policies" / name).mkdir(parents=True)
    torch.save(
        {"config": model.config, "state_dict": model.state_dict(), "args": {}},
        root / "policies" / name / "policy.pt",
    )
    return model


def _perturb(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return module


@pytest.mark.parametrize("obstacle_tokens", [False, True])
def test_flow_policy_loads_both_formats_alike(tmp_path, obstacle_tokens):
    model = _old_policy(tmp_path, "p", obstacle_tokens=obstacle_tokens)
    d = tmp_path / "policies" / "p"
    unconverted = FlowPolicy.load(d)  # a trainer's policy.pt in the directory
    assert isinstance(unconverted, FlowPolicy)
    convert_checkpoints.convert_tree(tmp_path)
    converted = FlowPolicy.load(d)
    obs, x, t = _obs(), torch.randn(3, 16, 2), torch.rand(3)
    ref = model(obs, x, t)
    for loaded in (unconverted, converted, FlowPolicy.load(d / "policy.pt.orig")):
        assert loaded.config == model.config and not loaded.training
        assert torch.equal(loaded(obs, x, t), ref)
    ckpt = load_checkpoint(d)
    assert ckpt["config"] == model.config and ckpt["state_dict"].keys() == model.state_dict().keys()


def test_policy_chunk_is_the_lerobot_policy_chunk(tmp_path):
    """The example's plain policy chunk (steered_chunk in mode "policy") is CursorFlowPolicy.predict_action_chunk."""
    _old_policy(tmp_path, "p", obstacle_tokens=True)
    convert_checkpoints.convert_tree(tmp_path)
    policy = CursorFlowPolicy.from_pretrained(tmp_path / "policies" / "p", strict=True)
    obs = _obs()
    torch.manual_seed(11)
    ours = steered_chunk(policy.model, obs, torch.zeros(3, 2), "policy")
    torch.manual_seed(11)
    assert torch.equal(policy.predict_action_chunk(obs), ours)


def test_converted_adapters_load_as_before(tmp_path):
    user, task = _perturb(UserModule(), 1), _perturb(TaskModule(), 2)
    old_user = {
        k: user.config[k] for k in ("horizon", "d", "tdim")
    }  # the cursor's user config, saved without kind
    old_task = {k: v for k, v in task.config.items() if k != "n_dims"}
    obstacle = {"kind": "disc", "radius": 0.08, "along": 0.5}
    for name, ckpt in (
        ("u", {"config": old_user, "state_dict": user.state_dict(), "args": {}}),
        (
            "t",
            {
                "kind": "task",
                "config": old_task,
                "state_dict": task.state_dict(),
                "args": {},
                "obstacle": obstacle,
            },
        ),
    ):
        (tmp_path / "adapters" / name).mkdir(parents=True)
        torch.save(ckpt, tmp_path / "adapters" / name / "adapter.pt")
    before = {n: ReversalAdapter.load(tmp_path / "adapters" / n / "adapter.pt") for n in ("u", "t")}
    assert convert_checkpoints.convert_tree(tmp_path) == {
        "policies": 0,
        "adapters": 2,
        "arch": {"obstacle": 1},
        "skipped": 0,
    }
    after = {n: ReversalAdapter.load(tmp_path / "adapters" / n / "adapter.pt") for n in ("u", "t")}
    obs, x, t, v = _obs(), torch.randn(3, 16, 2), torch.rand(3), torch.randn(3, 16, 2)
    for n in ("u", "t"):
        assert after[n].kind == before[n].kind
        assert torch.equal(after[n].adapt(obs, x, t, v, True), before[n].adapt(obs, x, t, v, True))
    assert torch.load(tmp_path / "adapters" / "t" / "adapter.pt")["arch"] == "obstacle"


def test_unknown_task_arch_is_a_value_error(tmp_path):
    task = TaskModule()
    torch.save(
        {"kind": "task", "arch": "bogus", "config": task.config, "state_dict": task.state_dict(), "meta": {}},
        tmp_path / "adapter.pt",
    )
    with pytest.raises(ValueError, match="unknown task module arch 'bogus'"):
        ReversalAdapter.load(tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="unknown task module arch 'bogus'"):
        load_adapter_module(tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="unknown task module arch 'bogus'"):
        convert_checkpoints.task_arch({"arch": "bogus", "config": {}})
    assert convert_checkpoints.task_arch({"config": {"horizon": 16, "d": 64, "tdim": 16}}) == "mlp"


def test_policy_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    assert paths.policy_path("p") == tmp_path / "policies" / "p"
    assert paths.policy_weights("p") == tmp_path / "policies" / "p" / "model.safetensors"
    (tmp_path / "policies" / "p").mkdir(parents=True)
    (tmp_path / "policies" / "p" / "policy.pt").touch()
    assert paths.policy_weights("p") == tmp_path / "policies" / "p" / "policy.pt"


def test_load_warns_when_a_folder_holds_both_formats(tmp_path, caplog):
    model = _old_policy(tmp_path, "p")
    d = tmp_path / "policies" / "p"
    convert_checkpoints.convert_tree(tmp_path)
    torch.manual_seed(1)
    retrained = FlowPolicy(d_model=32, layers=1, heads=2, d_ff=64, action_scale=0.0077).eval()
    torch.save(
        {"config": retrained.config, "state_dict": retrained.state_dict(), "args": {}}, d / "policy.pt"
    )
    with caplog.at_level("WARNING"):
        loaded = FlowPolicy.load(d)
    assert "both policy.pt and model.safetensors; loading policy.pt" in caplog.text
    obs, x, t = _obs(), torch.randn(3, 16, 2), torch.rand(3)
    assert torch.equal(loaded(obs, x, t), retrained(obs, x, t))  # the trainer's newer output, as ruled
    assert not torch.equal(loaded(obs, x, t), model(obs, x, t))
    caplog.clear()
    (d / "policy.pt").unlink()
    FlowPolicy.load(d)
    assert "both policy.pt" not in caplog.text
