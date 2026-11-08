"""Convert the cursor project's checkpoints, in place, to the formats lerobot loads.

    python examples/pi05/mars/cursor/tools/convert_checkpoints.py [--checkpoints DIR] [--steering [--base-policy P]]

DIR defaults to examples/pi05/mars/checkpoints (cursor_sa.paths.CHECKPOINTS).

- every policies/**/policy.pt (the cursor trainer's {config, state_dict, args}) becomes a lerobot
  `CursorFlowPolicy` directory beside it: config.json + model.safetensors (`save_pretrained`). The
  policy.pt is kept as policy.pt.orig.
- every adapters/**/adapter.pt (the cursor's {kind?, config, state_dict, args, obstacle}) is rewritten in
  the `AdapterModules.save_module` format (`AdapterModules.from_cursor_checkpoint`; kind, config, weights, meta). The original is
  kept as adapter.pt.orig.
- with --steering, every adapters/**/adapter.pt (once in that format) also becomes a lerobot `SteeringPolicy`
  directory beside it: config.json + model.safetensors (`save_pretrained`; MARS, mode frs, the module in its slot,
  its config and scene settings recorded so the folder loads without adapter.pt) and the policy's two processor
  pipelines (policy_preprocessor.json, policy_postprocessor.json: `make_steering_pre_post_processors`' for that config,
  which the factory's `make_pre_post_processors(config, pretrained_path=folder)` reloads), `--base-policy` the policy folder it steers (the
  config's base_policy_path; unset by default). adapter.pt stays, and stays loadable by AdapterModules.load.

Each converted file is written to a temporary name, loaded back and its weights compared with the
original; only then is the original copied to *.orig and the new file moved into place (one atomic
replace for an adapter; a policy's policy.pt, which FlowPolicy.load reads first, is removed last), so an
interrupted run leaves every checkpoint loadable and can simply be run again. An existing *.orig that
differs from the file is never overwritten: that file is skipped with a warning. Files already in the new
format are left untouched, so re-running is a no-op. The policy trainer still writes policy.pt, so run this
again after training a policy (FlowPolicy.load reads both formats meanwhile); the adapter trainer writes the
`AdapterModules.save_module` format itself, and ReversalAdapter.load reads only that.
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
from lerobot.policies.steering import (AdapterModules, AdapterSpec, LearnedTaskModule, LearnedUserModule, SteeringConfig,
                                       SteeringPolicy, make_steering_pre_post_processors)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

CHECKPOINTS = Path(__file__).resolve().parents[2] / "checkpoints"  # = cursor_sa.paths.CHECKPOINTS


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


def converted(ckpt):
    """The `AdapterModules.save_module` format: a kind and a meta (the cursor formats have neither, or no meta)."""
    return "kind" in ckpt and "meta" in ckpt


def convert_adapter(path):
    """adapters/**/adapter.pt in place to the `AdapterModules.save_module` format. Returns its kind ("user" or
    "task"), "skipped" for a differing backup, or None when the file is already converted."""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    if converted(ckpt):
        return None
    if "n_dims" not in ckpt["config"]:  # the cursor project's format
        new = AdapterModules.from_cursor_checkpoint(ckpt)
    else:  # the cursor trainer since the port: the core's module configs, the cursor's keys
        new = {"kind": ckpt["kind"], "config": dict(ckpt["config"]), "state_dict": ckpt["state_dict"],
               "meta": {"args": ckpt.get("args"), "obstacle": ckpt.get("obstacle")}}
    kind = new["kind"]
    if kind == "task" and ckpt.get("arch", "obstacle") != "obstacle":
        raise ValueError(f"{path}: task module arch {ckpt['arch']!r}; only LearnedTaskModule is left")
    module = (LearnedTaskModule if kind == "task" else LearnedUserModule)(**new["config"])
    module.load_state_dict(new["state_dict"])
    tmp = path.with_name(path.name + ".tmp")
    try:
        AdapterModules.save_module(module, tmp, kind, meta=new["meta"])
        loaded, loaded_kind, _ = AdapterModules.load_module(tmp)
        if loaded_kind != kind or not _same_weights(loaded.state_dict(), ckpt["state_dict"]):
            raise RuntimeError(f"{path}: the converted module does not reproduce the checkpoint")
        if not backup(path):
            return "skipped"
        os.replace(tmp, path)  # the one step that changes adapter.pt, atomic: old or new, never neither
    finally:
        tmp.unlink(missing_ok=True)
    return kind


STEERING_FILES = ("config.json", "model.safetensors", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                  f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")  # the files a converted folder has


def convert_adapter_to_policy(path, base_policy_path=None):
    """adapters/<name>/adapter.pt (in the `AdapterModules.save_module` format) -> adapters/<name>/{config.json,
    model.safetensors, the two processor JSONs}, a lerobot `SteeringPolicy` (MARS: mode frs, the module in its slot,
    its config and scene settings recorded) and its pipelines; True if converted, False when the folder already has
    them all. Written and verified (the module's weights reproduced by
    `SteeringPolicy.from_pretrained`) in a temporary folder, then moved into place; adapter.pt is left as it is."""
    path = Path(path)
    if all((path.parent / name).exists() for name in STEERING_FILES):
        return False
    adapter = AdapterModules.load(path)
    kind = adapter.kind
    spec = AdapterSpec(user_path=str(path)) if kind == "user" else AdapterSpec(task_path=str(path))
    config = SteeringConfig(mode="frs", adapter=spec, base_policy_path=None if base_policy_path is None else str(base_policy_path),
                            device="cpu")
    policy = SteeringPolicy(config)
    tmp = Path(tempfile.mkdtemp(prefix=".convert-", dir=path.parent))
    try:
        policy.save_pretrained(tmp)
        for pipeline in make_steering_pre_post_processors(policy.config):
            pipeline.save_pretrained(tmp)
        loaded = SteeringPolicy.from_pretrained(tmp, strict=True)
        module = getattr(loaded.adapter, kind)
        if module.config != getattr(adapter, kind).config or not _same_weights(module.state_dict(), getattr(adapter, kind).state_dict()):
            raise RuntimeError(f"{path}: the steering policy does not reproduce the module")
        for file in sorted(tmp.iterdir()):  # the policy's files and the pipelines' (with any step state)
            os.replace(file, path.parent / file.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def convert_tree(root=CHECKPOINTS, steering=False, base_policy_path=None):
    """Convert every policy and adapter checkpoint under root. Returns the counts converted (policies,
    adapters, of which task modules) and of the files skipped for a differing .orig backup; with `steering`, also
    the count of adapters given a SteeringPolicy folder (`convert_adapter_to_policy`)."""
    root = Path(root)
    done = [convert_policy(p) for p in sorted((root / "policies").rglob("policy.pt"))]
    kinds = Counter(convert_adapter(p) for p in sorted((root / "adapters").rglob("adapter.pt")))
    kinds.pop(None, None)
    skipped = done.count(False) + kinds.pop("skipped", 0)
    counts = {"policies": done.count(True), "adapters": sum(kinds.values()), "tasks": kinds["task"], "skipped": skipped}
    if steering:
        adapters = sorted(p for p in (root / "adapters").rglob("adapter.pt") if converted(torch.load(p, map_location="cpu")))
        counts["steering"] = sum(convert_adapter_to_policy(p, base_policy_path) for p in adapters)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, default=CHECKPOINTS, help="the checkpoints folder")
    p.add_argument("--steering", action="store_true", help="also write a SteeringPolicy folder beside every adapter.pt")
    p.add_argument("--base-policy", type=Path, default=None, help="with --steering: the policy folder the adapters steer")
    args = p.parse_args(argv)
    counts = convert_tree(args.checkpoints, args.steering, args.base_policy)
    print(f"converted {counts['policies']} policies and {counts['adapters']} adapters "
          f"({counts['tasks']} task modules; skipped {counts['skipped']}) under {args.checkpoints}"
          + (f"; {counts['steering']} steering policies" if args.steering else ""))
    return counts


if __name__ == "__main__":
    main()
