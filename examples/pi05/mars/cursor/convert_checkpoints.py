"""Convert the cursor project's checkpoints, in place, to the formats lerobot loads.

    python examples/pi05/mars/cursor/convert_checkpoints.py [--checkpoints DIR]

DIR defaults to examples/pi05/mars/checkpoints (cursor_sa.paths.CHECKPOINTS).

- every policies/**/policy.pt (the cursor trainer's {config, state_dict, args}) becomes a lerobot
  `CursorFlowPolicy` directory beside it: config.json + model.safetensors (`save_pretrained`). The
  policy.pt is kept as policy.pt.orig.
- every adapters/**/adapter.pt (the cursor's {kind?, config, state_dict, args, obstacle}) is rewritten in
  the `save_adapter_module` format (`convert_cursor_module`; kind, config, weights, meta, and for a task
  module an explicit arch). The original is kept as adapter.pt.orig.

Each converted file is written to a temporary name, loaded back and its weights compared with the
original; only then is the original copied to *.orig and the new file moved into place (one atomic
replace for an adapter; a policy's policy.pt, which FlowPolicy.load reads first, is removed last), so an
interrupted run leaves every checkpoint loadable and can simply be run again. An existing *.orig that
differs from the file is never overwritten: that file is skipped with a warning. Files already in the new
format are left untouched, so re-running is a no-op. The trainers
still write the cursor formats; run this again after training (FlowPolicy.load and ReversalAdapter.load
read both formats meanwhile).
"""

import argparse
import filecmp
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch

from lerobot.policies.cursor_flow import CursorFlowConfig, CursorFlowPolicy
from lerobot.policies.pi05.steering import (LearnedTaskModule, LearnedUserModule, ObstacleTaskModule, convert_cursor_module,
                                             load_adapter_module, save_adapter_module)

CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"  # = cursor_sa.paths.CHECKPOINTS
OBSTACLE_KEYS = {"heads", "layers", "scale"}  # config keys only the obstacle-reading task module has
TASK_ARCHS = {"obstacle": ObstacleTaskModule, "mlp": LearnedTaskModule}  # a task checkpoint's "arch"


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def backup(path):
    """Keep path's original as path.orig (a copy, so path stays loadable until it is replaced). False, with a
    warning, when a different .orig is already there: that backup is never overwritten."""
    orig = path.with_name(path.name + ".orig")
    if orig.exists():
        if filecmp.cmp(path, orig, shallow=False):  # an interrupted earlier conversion of the same file
            return True
        print(f"warning: skipping {path}: {orig.name} exists and differs from it; move the old backup aside to "
              f"convert this file")
        return False
    shutil.copy2(path, orig)
    return True


POLICY_FILES = ("config.json", "model.safetensors")


def convert_policy(path):
    """policies/<name>/policy.pt -> policies/<name>/{config.json, model.safetensors}; True if converted.
    The new files are written and verified in a temporary folder and only then moved into place; policy.pt
    (read first by FlowPolicy.load) is removed last, after its backup, so a crash leaves a loadable folder."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"]
    config = CursorFlowConfig(**ckpt["config"], action_scale=float(state["action_scale"]), device="cpu")
    policy = CursorFlowPolicy(config)
    policy.model.load_state_dict(state)
    tmp = Path(tempfile.mkdtemp(prefix=".convert-", dir=path.parent))
    try:
        policy.save_pretrained(tmp)
        loaded = CursorFlowPolicy.from_pretrained(tmp, strict=True)
        if loaded.model.config != ckpt["config"] or not _same_weights(loaded.model.state_dict(), state):
            raise RuntimeError(f"{path}: the converted policy does not reproduce the checkpoint")
        if not backup(path):
            return False
        for name in POLICY_FILES:
            os.replace(tmp / name, path.parent / name)
        path.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def task_arch(ckpt):
    """A cursor task checkpoint's module architecture: its arch if it names one; else "obstacle" when it carries
    obstacle info (the task's obstacle spec, or the obstacle module's config), "mlp" otherwise. Every cursor
    task module so far is the obstacle-reading one (the trainer requires an obstacle)."""
    arch = ckpt.get("arch")
    if arch is None:
        arch = "obstacle" if ckpt.get("obstacle") is not None or OBSTACLE_KEYS <= set(ckpt["config"]) else "mlp"
    if arch not in TASK_ARCHS:
        raise ValueError(f"unknown task module arch {arch!r}")
    return arch


def converted(ckpt):
    """The `save_adapter_module` format: a kind and a meta (the cursor formats have neither, or no meta)."""
    return "kind" in ckpt and "meta" in ckpt


def convert_adapter(path):
    """adapters/**/adapter.pt in place to the `save_adapter_module` format. Returns the task module's arch,
    "user" for a user module, or None when the file is already converted."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    if converted(ckpt):
        return None
    if "n_dims" not in ckpt["config"]:  # the cursor project's format
        new = convert_cursor_module(ckpt)
    else:  # the cursor trainer since the port: the core's module configs, the cursor's keys
        new = {"kind": ckpt["kind"], "config": dict(ckpt["config"]), "state_dict": ckpt["state_dict"],
               "meta": {"args": ckpt.get("args"), "obstacle": ckpt.get("obstacle")}}
    kind = new["kind"]
    arch = None
    if kind == "task":
        arch = task_arch(ckpt)
        module = TASK_ARCHS[arch](**new["config"])
    else:
        module = LearnedUserModule(**new["config"])
    module.load_state_dict(new["state_dict"])
    tmp = path.with_name(path.name + ".tmp")
    try:
        save_adapter_module(module, tmp, kind, meta=new["meta"], arch=arch)
        loaded, loaded_kind, _ = load_adapter_module(tmp)
        if loaded_kind != kind or not _same_weights(loaded.state_dict(), ckpt["state_dict"]):
            raise RuntimeError(f"{path}: the converted module does not reproduce the checkpoint")
        if not backup(path):
            return "skipped"
        os.replace(tmp, path)  # the one step that changes adapter.pt, atomic: old or new, never neither
    finally:
        tmp.unlink(missing_ok=True)
    return arch or "user"


def convert_tree(root=CHECKPOINTS):
    """Convert every policy and adapter checkpoint under root. Returns the counts converted (policies,
    adapters, the task modules' archs) and of the files skipped for a differing .orig backup."""
    root = Path(root)
    done = [convert_policy(p) for p in sorted((root / "policies").rglob("policy.pt"))]
    kinds = Counter(convert_adapter(p) for p in sorted((root / "adapters").rglob("adapter.pt")))
    kinds.pop(None, None)
    skipped = done.count(False) + kinds.pop("skipped", 0)
    adapters = sum(kinds.values())
    kinds.pop("user", None)
    return {"policies": done.count(True), "adapters": adapters, "arch": dict(kinds), "skipped": skipped}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS, help="the checkpoints folder")
    args = p.parse_args(argv)
    counts = convert_tree(args.checkpoints)
    print(f"converted {counts['policies']} policies and {counts['adapters']} adapters "
          f"(task module archs: {counts['arch'] or 'none'}; skipped {counts['skipped']}) under {args.checkpoints}")
    return counts


if __name__ == "__main__":
    main()
