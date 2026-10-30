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

import pytest
import torch

from lerobot.policies.cursor_flow.configuration_cursor_flow import CursorFlowConfig
from lerobot.policies.cursor_flow.modeling_cursor_flow import CursorFlowPolicy, FlowPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.pi05.steering import (
    LearnedUserModule,
    ObstacleTaskModule,
    load_adapter_module,
    steered_chunk,
)

CONVERTER = (
    Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "cursor" / "convert_checkpoints.py"
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

    loaded = CursorFlowPolicy.from_pretrained(out, strict=True)
    assert isinstance(loaded.config, CursorFlowConfig) and loaded.config.obstacle_tokens is obstacle_tokens
    assert loaded.model.config == model.config
    assert loaded.config.action_scale == pytest.approx(0.0123)
    assert torch.equal(loaded.model.action_scale, model.action_scale)
    for obs in (_obs(), _obs(n_obstacles=0), {k: v for k, v in _obs().items() if "obstacle" not in k}):
        x, t = torch.randn(3, 16, 2), torch.rand(3)
        assert torch.equal(loaded.model(obs, x, t), model(obs, x, t))


def test_predict_action_chunk_is_the_plain_policy_chunk():
    """The cursor example's plain policy chunk: the core's steered_chunk in mode "policy" (noise drawn first,
    a 10-step forward flow in the t = 0 noise convention), times the action scale."""
    model = _small_policy(True)
    policy = CursorFlowPolicy(CursorFlowConfig(**model.config, action_scale=0.0123, device="cpu"))
    policy.model.load_state_dict(model.state_dict())
    obs = _obs()
    torch.manual_seed(7)
    expected = (
        steered_chunk(
            lambda x, t: model(obs, x, t),
            torch.zeros(3, 2),
            "policy",
            horizon=model.horizon,
            to_reference=lambda c: c,
        )
        * model.action_scale
    )
    torch.manual_seed(7)
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (3, 16, 2) and torch.equal(chunk, expected)
    torch.manual_seed(7)
    assert torch.equal(policy.select_action(obs), expected[:, 0])
    with pytest.raises(NotImplementedError, match="train_policy.py"):
        policy.forward(obs)


def test_factory_knows_cursor_flow():
    assert get_policy_class("cursor_flow") is CursorFlowPolicy
    config = make_policy_config("cursor_flow", device="cpu")
    assert isinstance(config, CursorFlowConfig) and config.type == "cursor_flow"
    assert config.observation_delta_indices is None and config.action_delta_indices is None


def _save_old_adapters(root: Path) -> None:
    torch.manual_seed(3)
    user = LearnedUserModule(16, n_dims=2, d=32, tdim=16)
    old_user = {"horizon": 16, "d": 32, "tdim": 16}  # the cursor's user module config: no n_dims, no kind
    (root / "adapters" / "u").mkdir(parents=True)
    torch.save(
        {"config": old_user, "state_dict": user.state_dict(), "args": {}},
        root / "adapters" / "u" / "adapter.pt",
    )
    task = ObstacleTaskModule(16, d=32, layers=1)
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
    assert first == {"policies": 2, "adapters": 2, "arch": {"obstacle": 1}, "skipped": 0}

    _, kind, meta = load_adapter_module(tmp_path / "adapters" / "u" / "adapter.pt")
    assert kind == "user" and meta == {"args": {}, "obstacle": None}
    task, kind, meta = load_adapter_module(tmp_path / "adapters" / "group" / "t" / "adapter.pt")
    assert kind == "task" and isinstance(task, ObstacleTaskModule) and meta["obstacle"]["kind"] == "disc"
    assert torch.load(tmp_path / "adapters" / "group" / "t" / "adapter.pt")["arch"] == "obstacle"
    assert (tmp_path / "adapters" / "u" / "adapter.pt.orig").is_file()
    assert json.loads((tmp_path / "policies" / "flow" / "config.json").read_text())["type"] == "cursor_flow"

    files = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    stamps = {p: p.stat().st_mtime_ns for p in files}
    second = converter.convert_tree(tmp_path)
    assert second == {"policies": 0, "adapters": 0, "arch": {}, "skipped": 0}
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
        "arch": {"obstacle": 1},
        "skipped": 0,
    }
    for p, b in before.items():
        assert (tmp_path / p.with_name(p.name + ".orig")).read_bytes() == b
        load_adapter_module(tmp_path / p)
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
    assert counts == {"policies": 0, "adapters": 1, "arch": {"obstacle": 1}, "skipped": 2}
    out = capsys.readouterr().out
    assert out.count("warning: skipping") == 2 and "policy.pt.orig exists and differs" in out
    after = _files(tmp_path)
    for p in old_backups:
        assert after[p.relative_to(tmp_path)] == before[p.relative_to(tmp_path)]  # backup kept
        orig = p.with_name(p.name.removesuffix(".orig")).relative_to(tmp_path)
        assert after[orig] == before[orig]  # and the file left unconverted
    assert not (tmp_path / "policies" / "p" / "config.json").exists()
    assert not [p for p in (tmp_path / "policies" / "p").iterdir() if p.name.startswith(".convert-")]
