"""The cursor checkpoints in both formats: the cursor project's .pt files, which only the converter reads
(convert_checkpoints.py), and the converted lerobot ones. load_policy reads a converted policy folder with the
policy.pt's results; adapters are read in lerobot's format (AdapterModules.load reads AdapterModules.load_module's
format), which the trainer writes and the converter produces from old files."""

import convert_checkpoints
import pytest
import torch
from cursor_sa import paths
from cursor_sa.policy import CursorFlowPolicy, FlowPolicy, load_policy
from cursor_sa.steering import steered_chunk

from lerobot.policies.cursor_flow import make_cursor_flow_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.steering import AdapterModules, LearnedTaskModule, LearnedUserModule


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
def test_a_policy_pt_is_the_converters_and_the_converted_folder_loads_alike(
    tmp_path, obstacle_tokens, monkeypatch
):
    model = _old_policy(tmp_path, "p", obstacle_tokens=obstacle_tokens)
    d = tmp_path / "policies" / "p"
    monkeypatch.setattr(paths, "CHECKPOINTS", tmp_path)
    assert paths.policy_path("p") == d and paths.policy_weights("p") == d / "model.safetensors"
    with pytest.raises(ValueError, match=r"policy\.pt.*convert_checkpoints\.py"):
        load_policy(d)  # a trainer's policy.pt in the directory: the converter's to read
    ckpt = convert_checkpoints.read_policy_pt(d / "policy.pt")
    assert ckpt.keys() == {"config", "state_dict"} and ckpt["config"] == model.config
    assert ckpt["state_dict"].keys() == model.state_dict().keys()
    convert_checkpoints.convert_tree(tmp_path)
    loaded = load_policy(d)
    assert isinstance(loaded, CursorFlowPolicy) and not loaded.training and not loaded.model.training
    obs, x, t = _obs(), torch.randn(3, 16, 2), torch.rand(3)
    assert loaded.model.config == model.config and torch.equal(loaded.model(obs, x, t), model(obs, x, t))
    orig = convert_checkpoints.read_policy_pt(d / "policy.pt.orig")["state_dict"]
    assert all(torch.equal(orig[k], v) for k, v in loaded.model.state_dict().items())
    # a trainer's newer policy.pt beside the converted files: refused until it is converted
    torch.manual_seed(1)
    retrained = FlowPolicy(
        d_model=32, layers=1, heads=2, d_ff=64, action_scale=0.0077, obstacle_tokens=obstacle_tokens
    ).eval()
    torch.save(
        {"config": retrained.config, "state_dict": retrained.state_dict(), "args": {}}, d / "policy.pt"
    )
    with pytest.raises(ValueError, match=r"convert_checkpoints\.py"):
        load_policy(d)
    with pytest.raises(FileNotFoundError, match="no config.json"):
        load_policy(tmp_path / "policies" / "missing")


def test_policy_chunk_is_the_lerobot_policy_chunk(tmp_path):
    """The example's plain policy chunk (steered_chunk in mode "policy") is CursorFlowPolicy.predict_action_chunk,
    through its postprocessor (to screen units)."""
    _old_policy(tmp_path, "p", obstacle_tokens=True)
    convert_checkpoints.convert_tree(tmp_path)
    policy = CursorFlowPolicy.from_pretrained(tmp_path / "policies" / "p", strict=True)
    obs = _obs()
    torch.manual_seed(11)
    ours = steered_chunk(policy.model, obs, torch.zeros(3, 2), "policy")
    torch.manual_seed(11)
    post = make_cursor_flow_pre_post_processors(policy.config)[1]
    assert torch.equal(post(policy.predict_action_chunk(obs)), ours)


def test_converted_adapters_load_as_before(tmp_path):
    user, task = _perturb(LearnedUserModule(16, n_dims=2), 1), _perturb(LearnedTaskModule(), 2)
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
    frozen = lambda m: m.eval().requires_grad_(False)  # as load_module serves them  # noqa: E731
    before = {"u": AdapterModules(user=frozen(user)), "t": AdapterModules(task=frozen(task))}
    assert convert_checkpoints.convert_tree(tmp_path) == {
        "policies": 0,
        "adapters": 2,
        "tasks": 1,
        "skipped": 0,
    }
    after = {n: AdapterModules.load(tmp_path / "adapters" / n / "adapter.pt") for n in ("u", "t")}
    obs, x, t, v = _obs(), torch.randn(3, 16, 2), torch.rand(3), torch.randn(3, 16, 2)
    for n in ("u", "t"):
        assert after[n].kind == before[n].kind
        assert torch.equal(after[n].adapt(obs, x, t, v, True), before[n].adapt(obs, x, t, v, True))
    assert "arch" not in torch.load(tmp_path / "adapters" / "t" / "adapter.pt")


def test_an_adapter_folder_converts_to_a_steering_policy_that_steers_as_the_module(tmp_path):
    from lerobot.policies.steering import AdapterModules, SteeringPolicy

    torch.manual_seed(0)
    policy = FlowPolicy(d_model=32, layers=1, heads=2, d_ff=64, action_scale=0.0077).eval()
    user, task = _perturb(LearnedUserModule(16, n_dims=2), 1), _perturb(LearnedTaskModule(), 2)
    for name, kind, module in (("u", "user", user), ("group/t", "task", task)):
        AdapterModules.save_module(
            module, tmp_path / "adapters" / name / "adapter.pt", kind, meta={"args": {}}
        )
    before = {n: (tmp_path / "adapters" / n / "adapter.pt").read_bytes() for n in ("u", "group/t")}
    counts = convert_checkpoints.convert_tree(tmp_path, steering=True, base_policy_path=tmp_path / "base")
    assert counts == {"policies": 0, "adapters": 0, "tasks": 0, "skipped": 0, "steering": 2}
    assert (
        convert_checkpoints.convert_tree(tmp_path, steering=True)["steering"] == 0
    )  # a re-run converts nothing
    obs, command = _obs(), torch.tensor([[0.5, -0.3], [0.0, 0.0], [-0.7, 0.2]])
    base = CursorFlowPolicy.wrapping(policy)
    cursor_post = make_cursor_flow_pre_post_processors(base.config)[1]  # the base's: to screen units
    for n in ("u", "group/t"):
        folder = tmp_path / "adapters" / n
        assert folder.joinpath("adapter.pt").read_bytes() == before[n]  # adapter.pt stays as it was
        steering = SteeringPolicy.from_pretrained(folder, base_policy=base, postprocessor=cursor_post)
        assert (steering.config.mode, steering.config.base_policy_path) == ("frs", str(tmp_path / "base"))
        adapter = AdapterModules.load(folder / "adapter.pt")  # before the seed: building a module draws
        torch.manual_seed(3)
        want = steered_chunk(policy, obs, command, "frs", adapter=adapter)
        torch.manual_seed(3)
        assert torch.equal(cursor_post(steering.predict_action_chunk({**obs, "command": command})), want)
        pre, post = make_pre_post_processors(
            steering.config, pretrained_path=str(folder)
        )  # the factory's reload
        batch = pre({**{f"observation.{k}": v for k, v in obs.items()}, "command": command})
        torch.manual_seed(3)
        assert torch.equal(cursor_post(post(steering.predict_action_chunk(batch))), want)


def test_a_converted_image_task_module_keeps_its_scene_settings(tmp_path):
    from lerobot.policies.cursor_flow.processor_cursor_flow import CursorSceneRenderProcessorStep
    from lerobot.policies.steering import (
        AdapterModules,
        LearnedTaskModule,
        SteeringPolicy,
        SteeringSceneTokensProcessorStep,
    )

    task = LearnedTaskModule(
        horizon=16,
        d=16,
        tdim=8,
        heads=2,
        layers=1,
        token_geometry="relative",
        **LearnedTaskModule.spec("vlm_image", 48),
    )
    AdapterModules.save_module(task, tmp_path / "adapters" / "t" / "adapter.pt", "task")
    assert convert_checkpoints.convert_tree(tmp_path, steering=True)["steering"] == 1
    folder = tmp_path / "adapters" / "t"
    config = SteeringPolicy.from_pretrained(folder).config
    assert (config.scene_source, config.render_size, config.token_geometry) == ("vlm_image", 48, "relative")
    pre, _ = make_pre_post_processors(config, pretrained_path=str(folder))
    (step,) = (s for s in pre.steps if isinstance(s, SteeringSceneTokensProcessorStep))
    assert (step.source, step.render_size) == ("vlm_image", 48)
    # the folder's pre-processor reads the trials' layouts: the cursor renders them for the tokens step
    types = [type(s) for s in pre.steps]
    at = types.index(SteeringSceneTokensProcessorStep)
    assert types[at - 1] is CursorSceneRenderProcessorStep and pre.steps[at - 1].render_size == 48
