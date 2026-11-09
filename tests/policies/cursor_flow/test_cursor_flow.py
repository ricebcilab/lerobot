#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.envs.cursor import TaskConfig, obstacle_spec, render_layouts, sample_layouts
from lerobot.policies.cursor_flow.configuration_cursor_flow import CursorFlowConfig
from lerobot.policies.cursor_flow.modeling_cursor_flow import CursorFlowPolicy, FlowPolicy
from lerobot.policies.cursor_flow.processor_cursor_flow import (
    OBS_LAYOUTS,
    CursorSceneRenderProcessorStep,
    make_cursor_flow_pre_post_processors,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.steering import (
    AdapterModules,
    Flow,
    LearnedTaskModule,
    LearnedUserModule,
    SteeringConfig,
    SteeringPolicy,
)
from lerobot.policies.steering.processor_steering import OBS_SCENE_DIGEST, OBS_SCENE_IMAGES
from lerobot.types import TransitionKey

CONVERTER = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "pi05"
    / "mars"
    / "cursor"
    / "tools"
    / "convert_checkpoints.py"
)


def _converter():
    spec = importlib.util.spec_from_file_location("convert_checkpoints", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_policy(obstacle_tokens: bool, seed: int = 0) -> FlowPolicy:
    torch.manual_seed(seed)
    return FlowPolicy(
        d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123, obstacle_tokens=obstacle_tokens
    ).eval()


def _obs(b: int = 3, n_targets: int = 4, n_obstacles: int = 2, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(b, n_targets, dtype=torch.bool)
    mask[0, -1] = False
    goal = torch.zeros(b, n_targets, dtype=torch.bool)
    goal[:, 0] = True
    obstacle_mask = torch.ones(b, n_obstacles, dtype=torch.bool)
    if b > 1:
        obstacle_mask[1] = False
    return {
        "pos": torch.rand(b, 2, generator=g),
        "hist": 0.01 * torch.randn(b, 4, 2, generator=g),
        "targets": torch.rand(b, n_targets, 2, generator=g),
        "mask": mask,
        "goal": goal,
        "obstacles": torch.rand(b, n_obstacles, 3, generator=g) * torch.tensor([1.0, 1.0, 0.1]),
        "obstacle_mask": obstacle_mask,
    }


def _save_old_policy(model: FlowPolicy, root: Path, name: str) -> Path:
    path = root / "policies" / name / "policy.pt"
    path.parent.mkdir(parents=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "args": {"seed": 0}}, path)
    return path


@pytest.mark.parametrize("obstacle_tokens", [False, True])
def test_round_trip_is_exact(tmp_path, obstacle_tokens):
    model = _small_policy(obstacle_tokens)
    _save_old_policy(model, tmp_path, "p")
    _converter().convert_tree(tmp_path)
    out = tmp_path / "policies" / "p"
    assert (out / "config.json").is_file() and (out / "model.safetensors").is_file()
    assert not (out / "policy.pt").exists() and (out / "policy.pt.orig").is_file()
    assert (out / "policy_preprocessor.json").is_file() and (out / "policy_postprocessor.json").is_file()

    loaded = CursorFlowPolicy.from_pretrained(out, strict=True)
    assert isinstance(loaded.config, CursorFlowConfig) and loaded.config.obstacle_tokens is obstacle_tokens
    assert loaded.model.config == model.config
    assert loaded.config.action_scale == pytest.approx(0.0123)
    assert torch.equal(loaded.model.action_scale, model.action_scale)
    for obs in (_obs(), _obs(n_obstacles=0), {k: v for k, v in _obs().items() if "obstacle" not in k}):
        x, t = torch.randn(3, 16, 2), torch.rand(3)
        assert torch.equal(loaded.model(obs, x, t), model(obs, x, t))


PIPELINE_FILES = {
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}


def test_a_converted_folder_without_pipelines_gets_them(tmp_path):
    """A policy folder converted before the pipelines gets them on a re-run and nothing else changes; a further run
    changes nothing; a steering policy then loads the base by path with its action statistics."""
    model = _small_policy(False)
    _save_old_policy(model, tmp_path, "p")
    converter = _converter()
    converter.convert_tree(tmp_path)
    out = tmp_path / "policies" / "p"
    assert {f.name for f in out.iterdir()} == {
        "config.json",
        "model.safetensors",
        "policy.pt.orig",
        *PIPELINE_FILES,
    }
    for name in PIPELINE_FILES:
        (out / name).unlink()  # a folder converted before them
    before = {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in out.iterdir()}
    assert converter.convert_tree(tmp_path) == {"policies": 1, "adapters": 0, "tasks": 0, "skipped": 0}
    assert {f.name for f in out.iterdir()} == {f.name for f in before} | PIPELINE_FILES
    assert {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in before} == before
    stamps = {f: f.stat().st_mtime_ns for f in out.iterdir()}
    assert converter.convert_tree(tmp_path)["policies"] == 0
    assert {f: f.stat().st_mtime_ns for f in out.iterdir()} == stamps
    steering = SteeringPolicy(SteeringConfig(base_policy_path=str(out), device="cpu"))
    _ = steering.core  # FRS: the folder's action statistics
    chunk = torch.randn(3, 16, 2)
    assert torch.equal(steering._postprocessor(chunk), chunk * model.action_scale)


def test_predict_action_chunk_is_the_plain_policy_chunk():
    """The plain policy chunk, normalized: noise drawn first, then the core's 10-step Euler flow in the t = 0 noise
    convention (``FlowPolicy.sample_chunk``, the loop the steering wrappers drive); the postprocessor scales it to
    screen units."""
    model = _small_policy(True)
    policy = CursorFlowPolicy(CursorFlowConfig(**model.config, action_scale=0.0123, device="cpu"))
    policy.model.load_state_dict(model.state_dict())
    _, post = make_cursor_flow_pre_post_processors(policy.config)
    obs = _obs()
    torch.manual_seed(7)
    noise = torch.randn(3, model.horizon, 2)
    expected = Flow(lambda x, t: model(obs, x, t), batch_time=True).integrate(noise, 0.0, 0.1, 10)
    torch.manual_seed(7)
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (3, 16, 2) and torch.equal(chunk, expected)
    assert torch.equal(post(chunk), expected * model.action_scale)
    torch.manual_seed(7)
    assert torch.equal(policy.select_action(obs), expected[:, 0])
    with pytest.raises(NotImplementedError, match="train_policy.py"):
        policy.forward(obs)


def _contract_policy() -> CursorFlowPolicy:
    torch.manual_seed(0)
    return CursorFlowPolicy(
        CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123)
    ).eval()


def test_predict_action_chunk_is_normalized_and_passes_hooks():
    policy = CursorFlowPolicy(CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123))
    obs = _obs()
    z = torch.randn(3, policy.config.horizon, 2)
    seen = {}

    def hook(step, t, x):
        return seen.setdefault("steps", []).append(step) or x

    chunk = policy.predict_action_chunk(obs, noise=z, x_t_hook=hook)
    assert torch.equal(chunk, policy.model.sample_chunk(obs, 10, noise=z))
    assert seen["steps"] == list(range(10))
    assert policy.config.chunk_size == policy.horizon and policy.config.noise_time == 0.0


def test_the_postprocessor_restores_screen_units_exactly():
    policy = CursorFlowPolicy(CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123))
    _, post = make_cursor_flow_pre_post_processors(policy.config)
    torch.manual_seed(0)
    chunk = policy.predict_action_chunk(_obs())
    torch.manual_seed(0)
    want = policy.model.sample(_obs(), steps=10)
    assert torch.equal(post(chunk), want)


def test_prefixed_and_unbatched_observations():
    policy = _contract_policy()
    obs = _obs(b=1)
    one = {k: v[0] for k, v in obs.items()}  # (2,) pos
    prefixed = {f"observation.{k}": v for k, v in obs.items()}
    torch.manual_seed(0)
    a = policy.predict_action_chunk(obs)
    torch.manual_seed(0)
    b = policy.predict_action_chunk(prefixed)
    torch.manual_seed(0)
    c = policy.predict_action_chunk(one)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_select_action_serves_n_action_steps_then_replans():
    policy = _contract_policy()
    obs = _obs()
    torch.manual_seed(1)
    chunk = policy.predict_action_chunk(obs)
    second = policy.predict_action_chunk(obs)  # the next draw: the replan's
    torch.manual_seed(1)
    served = [policy.select_action(obs) for _ in range(policy.config.n_action_steps + 1)]
    assert policy.config.n_action_steps == 8
    assert all(torch.equal(a, chunk[:, i]) for i, a in enumerate(served[:8]))
    assert torch.equal(served[8], second[:, 0])
    policy.reset()
    torch.manual_seed(1)
    assert torch.equal(policy.select_action(obs), chunk[:, 0])


def test_velocity_field_is_the_model():
    policy = _contract_policy()
    obs = _obs()
    x, t = torch.randn(3, 16, 2), torch.rand(3)
    field = policy.velocity_field({f"observation.{k}": v for k, v in obs.items()})
    assert torch.equal(field(x, t), policy.model(obs, x, t))
    assert torch.equal(field(x, 0.3), policy.model(obs, x, torch.full((3,), 0.3)))


def test_wrapping_shares_the_model():
    model = _small_policy(True)
    policy = CursorFlowPolicy.wrapping(model)
    assert policy.model is model and policy.config.obstacle_tokens is True
    assert (policy.horizon, policy.hist, policy.config.d_model) == (16, 4, 32)
    assert policy.action_scale is model.action_scale and policy.config.action_scale == float(
        model.action_scale
    )
    torch.manual_seed(2)
    want = model.sample_chunk(_obs(), 10)
    torch.manual_seed(2)
    assert torch.equal(policy.predict_action_chunk(_obs()), want)


def test_regress_mode_returns_the_prediction():
    torch.manual_seed(0)
    policy = CursorFlowPolicy(CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, mode="regress")).eval()
    obs = _obs()
    assert torch.equal(policy.predict_action_chunk(obs), policy.model(obs))
    with pytest.raises(ValueError, match="regress"):
        policy.predict_action_chunk(obs, noise=torch.zeros(3, 16, 2))


def test_render_step_writes_pictures_and_digest():
    """The cursor's render step writes its layouts' render and the digest the steering step computed before the
    render moved here (a sha256 over start, targets, mask, obstacles and their mask: the values pinned at the move),
    and keeps the last render."""
    task = TaskConfig(
        min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    lay, other = (sample_layouts(np.random.default_rng(seed), 3, 4, task) for seed in (0, 1))
    step = CursorSceneRenderProcessorStep(render_size=32)

    def rendered(layouts):
        return step({TransitionKey.OBSERVATION: {OBS_LAYOUTS: layouts}})[TransitionKey.OBSERVATION]

    obs = rendered(lay)
    assert np.array_equal(obs[OBS_SCENE_IMAGES], render_layouts(lay, 32)) and obs[OBS_LAYOUTS] is lay
    assert obs[OBS_SCENE_DIGEST] == "35302b39f2e6b6512f5435209a9aab3301849889520ca3fa8c29092c470579a3"
    again = rendered(lay[np.arange(3)])  # equal layouts: the kept render
    assert (
        again[OBS_SCENE_IMAGES] is obs[OBS_SCENE_IMAGES] and again[OBS_SCENE_DIGEST] == obs[OBS_SCENE_DIGEST]
    )
    moved = rendered(other)
    assert np.array_equal(moved[OBS_SCENE_IMAGES], render_layouts(other, 32))
    assert moved[OBS_SCENE_DIGEST] == "3f6e3b1ec6d8ca8e34fbac3b6c87dbead8cde9267a4f5d4ad0f9afa71ecff9d6"
    with pytest.raises(ValueError, match="observation.layouts"):
        step({TransitionKey.OBSERVATION: {"observation.pos": torch.zeros(1, 2)}})
    assert CursorSceneRenderProcessorStep(**step.get_config()).render_size == 32


def test_factory_knows_cursor_flow():
    assert get_policy_class("cursor_flow") is CursorFlowPolicy
    config = make_policy_config("cursor_flow", device="cpu")
    assert isinstance(config, CursorFlowConfig) and config.type == "cursor_flow"
    assert config.observation_delta_indices is None and config.action_delta_indices is None
    pre, post = make_pre_post_processors(config)
    want = make_cursor_flow_pre_post_processors(config)
    assert [type(s) for s in pre.steps] == [type(s) for s in want[0].steps]
    assert [type(s) for s in post.steps] == [type(s) for s in want[1].steps]
    assert post.steps[0].stats["action"]["std"] == [config.action_scale] * 2


def _save_old_adapters(root: Path) -> None:
    torch.manual_seed(3)
    user = LearnedUserModule(16, n_dims=2, d=32, tdim=16)
    old_user = {"horizon": 16, "d": 32, "tdim": 16}  # the cursor's user module config: no n_dims, no kind
    (root / "adapters" / "u").mkdir(parents=True)
    torch.save(
        {"config": old_user, "state_dict": user.state_dict(), "args": {}},
        root / "adapters" / "u" / "adapter.pt",
    )
    task = LearnedTaskModule(16, d=32, layers=1)
    old_task = {k: v for k, v in task.config.items() if k != "n_dims"}
    obstacle = {"kind": "disc", "radius": 0.08, "along": 0.5}
    (root / "adapters" / "group" / "t").mkdir(parents=True)
    torch.save(
        {
            "kind": "task",
            "config": old_task,
            "state_dict": task.state_dict(),
            "args": {},
            "obstacle": obstacle,
        },
        root / "adapters" / "group" / "t" / "adapter.pt",
    )


def test_converter_is_idempotent(tmp_path):
    _save_old_policy(_small_policy(False), tmp_path, "flow")
    _save_old_policy(_small_policy(True), tmp_path, "flow_obstacles")
    _save_old_adapters(tmp_path)
    converter = _converter()
    first = converter.convert_tree(tmp_path)
    assert first == {"policies": 2, "adapters": 2, "tasks": 1, "skipped": 0}

    _, kind, meta = AdapterModules.load_module(tmp_path / "adapters" / "u" / "adapter.pt")
    assert kind == "user" and meta == {"args": {}, "obstacle": None}
    task, kind, meta = AdapterModules.load_module(tmp_path / "adapters" / "group" / "t" / "adapter.pt")
    assert kind == "task" and isinstance(task, LearnedTaskModule) and meta["obstacle"]["kind"] == "disc"
    assert (tmp_path / "adapters" / "u" / "adapter.pt.orig").is_file()
    assert json.loads((tmp_path / "policies" / "flow" / "config.json").read_text())["type"] == "cursor_flow"

    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    stamps = {p: p.stat().st_mtime_ns for p in files}
    second = converter.convert_tree(tmp_path)
    assert second == {"policies": 0, "adapters": 0, "tasks": 0, "skipped": 0}
    assert sorted(p for p in tmp_path.rglob("*") if p.is_file()) == files
    assert {p: p.stat().st_mtime_ns for p in files} == stamps


def _files(root: Path) -> dict:
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _crash_on_first_replace(monkeypatch):
    real = os.replace
    calls = []

    def replace(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise OSError("simulated crash")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)


def test_adapter_conversion_survives_a_crash(tmp_path, monkeypatch):
    _save_old_adapters(tmp_path)
    before = _files(tmp_path)
    converter = _converter()
    _crash_on_first_replace(monkeypatch)
    with pytest.raises(OSError, match="simulated crash"):
        converter.convert_tree(tmp_path)
    after = _files(tmp_path)
    assert {p: b for p, b in after.items() if p.suffix != ".orig"} == before  # every adapter.pt as it was
    first = Path("adapters/group/t/adapter.pt")  # backed up, then the crash hit its replace
    assert after[Path("adapters/group/t/adapter.pt.orig")] == before[first]
    assert set(after) == set(before) | {Path("adapters/group/t/adapter.pt.orig")}  # no .tmp left behind
    monkeypatch.undo()
    assert converter.convert_tree(tmp_path) == {
        "policies": 0,
        "adapters": 2,
        "tasks": 1,
        "skipped": 0,
    }
    for p, b in before.items():
        assert (tmp_path / p.with_name(p.name + ".orig")).read_bytes() == b
        AdapterModules.load_module(tmp_path / p)
    assert not list(tmp_path.rglob("*.tmp"))


def test_policy_conversion_survives_a_crash(tmp_path, monkeypatch):
    model = _small_policy(True)
    _save_old_policy(model, tmp_path, "p")
    before = (tmp_path / "policies" / "p" / "policy.pt").read_bytes()
    converter = _converter()
    _crash_on_first_replace(monkeypatch)
    with pytest.raises(OSError, match="simulated crash"):
        converter.convert_tree(tmp_path)
    d = tmp_path / "policies" / "p"
    assert (d / "policy.pt").read_bytes() == before  # still the loadable original
    assert sorted(p.name for p in d.iterdir()) == [
        "policy.pt",
        "policy.pt.orig",
    ]  # no partial files, no temp folder
    monkeypatch.undo()
    assert converter.convert_tree(tmp_path)["policies"] == 1
    assert (d / "policy.pt.orig").read_bytes() == before and not (d / "policy.pt").exists()
    obs, x, t = _obs(), torch.randn(3, 16, 2), torch.rand(3)
    assert torch.equal(CursorFlowPolicy.from_pretrained(d, strict=True).model(obs, x, t), model(obs, x, t))


def test_failed_policy_verification_leaves_nothing_behind(tmp_path, monkeypatch):
    _save_old_policy(_small_policy(False), tmp_path, "p")
    converter = _converter()
    wrong = CursorFlowPolicy(CursorFlowConfig(**_small_policy(False, seed=5).config, device="cpu"))
    monkeypatch.setattr(CursorFlowPolicy, "from_pretrained", classmethod(lambda cls, *a, **k: wrong))
    with pytest.raises(RuntimeError, match="does not reproduce"):
        converter.convert_tree(tmp_path)
    assert sorted(p.name for p in (tmp_path / "policies" / "p").iterdir()) == ["policy.pt"]


def test_a_differing_backup_is_never_overwritten(tmp_path, capsys):
    _save_old_policy(_small_policy(False), tmp_path, "p")
    _save_old_adapters(tmp_path)
    old_backups = {
        tmp_path / "policies" / "p" / "policy.pt.orig": b"an older policy",
        tmp_path / "adapters" / "u" / "adapter.pt.orig": b"an older adapter",
    }
    for p, b in old_backups.items():
        p.write_bytes(b)
    before = _files(tmp_path)
    counts = _converter().convert_tree(tmp_path)
    assert counts == {"policies": 0, "adapters": 1, "tasks": 1, "skipped": 2}
    out = capsys.readouterr().out
    assert out.count("warning: skipping") == 2 and "policy.pt.orig exists and differs" in out
    after = _files(tmp_path)
    for p in old_backups:
        assert after[p.relative_to(tmp_path)] == before[p.relative_to(tmp_path)]  # backup kept
        orig = p.with_name(p.name.removesuffix(".orig")).relative_to(tmp_path)
        assert after[orig] == before[orig]  # and the file left unconverted
    assert not (tmp_path / "policies" / "p" / "config.json").exists()
    assert not [p for p in (tmp_path / "policies" / "p").iterdir() if p.name.startswith(".convert-")]


def test_the_package_has_no_cursor_migration():
    assert not hasattr(AdapterModules, "from_cursor_checkpoint")
    assert not hasattr(FlowPolicy, "load")
    converter = _converter()
    sd = {}
    cursor = {"horizon": 16, "d": 256, "tdim": 64, "step_rank": None, "cross_step_rank": 8}
    converted = converter.from_cursor_checkpoint({"kind": "user", "config": cursor, "state_dict": sd})
    assert converted["config"] == {"horizon": 16, "n_dims": 2, "d": 256, "tdim": 64, "cross_step_rank": 8}
    assert converted["kind"] == "user" and converted["state_dict"] is sd
