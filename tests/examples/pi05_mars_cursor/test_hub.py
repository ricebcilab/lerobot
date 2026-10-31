"""hub.py: the manifest of checkpoint folders the study and the grid use, and their download from the Hub
(huggingface_hub is monkeypatched; nothing here talks to the Hub)."""

import subprocess
import sys
from pathlib import Path

import hub
import pytest
import yaml

HUB = Path(hub.__file__)


def test_manifest_matches_configs():
    done = subprocess.run([sys.executable, str(HUB), "manifest", "--check"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_manifest_covers_study_grid_and_seeds():
    folders = hub.manifest_folders()
    assert folders == hub.read_manifest()
    assert {"policies/flow_goaldrop", "adapters/user_gain025y_expert_n100_s0"} <= set(folders["study"])
    assert {
        "policies/flow_goaldrop_ring_start",
        "adapters/user_rot60_expert_n100_s0_on_wall_gap",
        "adapters/task_ring_goal_n300_s0_with_rot60",
    } <= set(folders["grid"])
    seeds = folders["grid_seeds"]
    assert len(seeds) == 4 * (
        len(folders["grid"]) - 5
    )  # every seed-0 module, four times; the 5 policies stay one seed
    assert "adapters/user_curl20_expert_n100_s3_on_ring_goal" in seeds
    assert "adapters/task_wall_gap_n300_s4_with_gain025x" in seeds
    assert not any("grid_seeds/" in f for group in folders.values() for f in group)  # all seeds side by side


def test_seed_name_swaps_the_seed_0_tag():
    assert (
        hub.seed_name("user_rot60_expert_n100_s0_on_wall_gap", 3) == "user_rot60_expert_n100_s3_on_wall_gap"
    )
    assert hub.seed_name("task_ring_goal_n300_s0_with_rot60", 2) == "task_ring_goal_n300_s2_with_rot60"
    assert hub.seed_name("user_gain025y_expert_n100_s0", 1) == "user_gain025y_expert_n100_s1"
    with pytest.raises(ValueError):
        hub.seed_name("user_gain025y_expert_n100", 1)  # no _s0 tag


def test_check_fails_on_a_stale_manifest(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "hub_manifest.yaml"
    data = yaml.safe_load(hub.MANIFEST.read_text())
    data["folders"]["study"] = data["folders"]["study"][1:]
    stale.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(hub, "MANIFEST", stale)
    assert hub.main(["manifest", "--check"]) == 1
    assert "out of date" in capsys.readouterr().out


def fake_hub_folder(local_dir, folder):
    """Write a manifest folder's loader files as snapshot_download would leave them."""
    d = Path(local_dir) / folder
    d.mkdir(parents=True, exist_ok=True)
    for f in hub.POLICY_REQUIRED if folder.startswith("policies/") else hub.ADAPTER_REQUIRED:
        (d / f).write_text("x")


def test_download_targets(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        for pattern in kwargs["allow_patterns"]:
            fake_hub_folder(kwargs["local_dir"], pattern.removesuffix("/*"))
        return str(kwargs["local_dir"])

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "CHECKPOINTS", tmp_path / "checkpoints")
    assert hub.main(["download"]) == 0
    assert len(calls) == 1
    (call,) = calls
    folders = {f for group in hub.read_manifest().values() for f in group}
    assert call["repo_id"] == "ricebcilab/mars-cursor" and call["repo_type"] == "model"
    assert Path(call["local_dir"]) == tmp_path / "checkpoints"
    assert sorted(call["allow_patterns"]) == sorted(f"{f}/*" for f in folders)

    calls.clear()
    assert hub.main(["download", "--repo", "someone/else"]) == 0
    assert calls[0]["repo_id"] == "someone/else"


def test_download_fails_when_a_folder_is_missing(tmp_path, monkeypatch, capsys):
    folders = {f for group in hub.read_manifest().values() for f in group}
    left_out = "adapters/user_rot60_expert_n100_s0"
    assert left_out in folders

    def fake_snapshot_download(**kwargs):
        for pattern in kwargs["allow_patterns"]:
            if pattern != f"{left_out}/*":
                fake_hub_folder(kwargs["local_dir"], pattern.removesuffix("/*"))
        return str(kwargs["local_dir"])

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "CHECKPOINTS", tmp_path / "checkpoints")
    assert hub.main(["download"]) == 1
    out = capsys.readouterr().out
    assert f"{left_out}/adapter.pt" in out and "1 files missing" in out


def test_upload_refuses_an_unconverted_adapter(tmp_path, monkeypatch, capsys):
    import torch

    root = tmp_path / "checkpoints"
    ada = root / "adapters" / "a"
    ada.mkdir(parents=True)
    # the cursor project's adapter format: a config and a state dict, no kind or meta
    torch.save({"config": {"hidden": 8}, "state_dict": {}}, ada / "adapter.pt")
    (ada / "train_log.json").write_text("{}")
    monkeypatch.setattr(hub, "CHECKPOINTS", root)
    monkeypatch.setattr(hub, "read_manifest", lambda: {"study": ["adapters/a"]})
    calls = []

    class FakeApi:
        def __init__(self):
            calls.append("HfApi")

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert hub.main(["upload"]) == 1
    assert calls == []
    assert "adapters/a: adapter.pt is unconverted" in capsys.readouterr().out


def test_upload_creates_the_repo_and_sends_only_converted_files(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    pol, ada = root / "policies" / "p", root / "adapters" / "a"
    for d, files in (
        (pol, ["config.json", "model.safetensors", "train_log.json", "policy.pt.orig", "train.out"]),
        (ada, ["adapter.pt", "adapter.pt.orig", "train_log.json"]),
    ):
        d.mkdir(parents=True)
        for f in files:
            (d / f).write_text("x")
    monkeypatch.setattr(hub, "CHECKPOINTS", root)
    monkeypatch.setattr(hub, "read_manifest", lambda: {"study": ["policies/p", "adapters/a"]})
    monkeypatch.setattr(hub, "adapter_converted", lambda path: True)
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs))

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert hub.main(["upload"]) == 0
    (_, create), (_, upload) = calls
    assert create == {
        "repo_id": "ricebcilab/mars-cursor",
        "repo_type": "model",
        "private": False,
        "exist_ok": True,
    }
    assert upload["repo_id"] == "ricebcilab/mars-cursor" and Path(upload["folder_path"]) == root
    assert sorted(upload["allow_patterns"]) == sorted(
        [
            "policies/p/config.json",
            "policies/p/model.safetensors",
            "policies/p/train_log.json",
            "adapters/a/adapter.pt",
            "adapters/a/train_log.json",
        ]
    )

    calls.clear()
    (pol / "policy.pt").write_text("x")  # a trainer's unconverted policy: refuse before touching the Hub
    assert hub.main(["upload"]) == 1
    assert calls == []
