"""Single-seed FRS/U/T/UT pilot on matched utensil-holder layouts with the fixed rotation user."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from session import Session
from train_visual_task import AVOID_PROMPT, run_trials, save_json, trials
from visual_task_scene import PilotScene

from lerobot.policies.steering import AdapterModules

CONFIG = CONFIG_DIR / "study_goal_rotation.yaml"
ARMS = {"FRS": (False, False), "U": (True, False), "T": (False, True), "UT": (True, True)}
PAIR_FIELDS = (
    "pair_id",
    "task_id",
    "init_state_id",
    "initial_state_hash",
    "route",
    "center",
    "blocked",
    "layout_seed",
    "policy_seed",
    "operator_seed",
    "env_seed",
    "input_noise_seed",
)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot_settings(snapshot, task_ids):
    return [
        f"scene.task_ids={task_ids}",
        f"policy.path={snapshot}",
        f"prompt={AVOID_PROMPT}",
        "control.n_reversal_steps=4",
        "control.pinned_steps=10",
        "control.translation_noise=0.0",
        "server.headless=true",
    ]


def calibrate(out, snapshot):
    """Invoke the existing rotation trainer in its own process to release the GPU before evaluation."""
    flags = [sys.executable, str(Path(__file__).with_name("train_user_module.py"))]
    options = {
        "--config": CONFIG,
        "--out": out,
        "--episodes": 20,
        "--open-episodes": 20,
        "--open-steps": 120,
        "--steps": 3000,
        "--seed": 0,
        "--val-depths": 4,
        "--log-every": 500,
    }
    for key, value in options.items():
        flags.extend([key, str(value)])
    flags.extend(["--sigma", "0", "0"])
    for override in pilot_settings(snapshot, [0, 2, 3, 9]):
        flags.extend(["--set", override])
    subprocess.run(flags, check=True)


def set_arm(session, arm, user_path, task_path):
    user, task = ARMS[arm]
    # apply_control clears both slots before loading either head. The cached steering
    # policy shares this holder and is reset at the start of every episode.
    control = replace(
        session.settings.control,
        reversal_adapter_user_path=str(user_path) if user else None,
        reversal_adapter_task_path=str(task_path) if task else None,
    )
    session.apply_control(control)
    holder = session.adapter
    if (holder.user is not None, holder.task is not None) != (user, task) or holder.matrix is not None:
        raise ValueError(f"Unexpected adapter slots for {arm}")


def preflight(session):
    spec, route, seed = next(trials(session, "train", 1))
    session.set_scene("libero_goal", spec.task_id)
    scene = PilotScene(session.vec_env.envs[0], "holder")
    obs = scene.reset(spec, seed, route)
    session.chain.reset()
    session.operator.reset()
    chunk = np.zeros((session.settings.n_action_steps, 7))
    chunk[:, 0] = 0.5
    session.operator.refresh(chunk, step=0)
    command = session.chunk_command()
    rotation = session.settings.control.corruption_matrix
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.trace(rotation), 2.0, atol=1e-7)  # 1 + 2*cos(60 degrees)
    np.testing.assert_allclose(command[0].numpy(), rotation @ chunk[0, :3] * session.chain.gain.gain)
    calls = Counter()

    def count(name):
        def hook(*_):
            calls[name] += 1

        return hook

    with ExitStack() as hooks, torch.inference_mode(), session.autocast_ctx:
        for name, layer in (("U", session.adapter.user.net[-1]), ("T", session.adapter.task.out)):
            hooks.enter_context(layer.register_forward_hook(count(name)))
        steering = session.steering_policy()
        steering.reset()
        torch.manual_seed(0)
        action = steering.select_action({**session.policy_batch(obs, AVOID_PROMPT), "command": command})
        if not action.isfinite().all() or not all(calls[k] for k in ("U", "T")):
            raise RuntimeError(f"Composed steering failed the known-command check: {calls}")
    return {"command": command.tolist(), "head_calls": dict(calls)}


def report(out, runs, reference):
    keys = reference.keys()
    header = [
        "| Arm | Layout | Trials | Correct placement | Holder contact | Joint success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines = [
        "# Rotation-user holder pilot",
        "",
        "One fixed 60° rotation, seed-0 checkpoints, 20 paired resets per arm.",
        "",
        *header,
    ]
    for name, run in runs.items():
        if run.keys() != keys or any(run[k][f] != reference[k][f] for k in keys for f in PAIR_FIELDS):
            raise ValueError(f"Unmatched scheduled trials for {name}")
        if any(r["arm"] != name or r["prompt"] != AVOID_PROMPT for r in run.values()):
            raise ValueError(f"Trial arm or prompt mismatch for {name}")
        groups = {"All": list(run.values())}
        for task, goal in ((1, "stove"), (8, "plate")):
            for blocked in (False, True):
                group = [r for r in run.values() if r["task_id"] == task and r["blocked"] == blocked]
                if len(group) != 5:
                    raise ValueError("Expected five trials per layout")
                groups[f"{goal}, {'blocked' if blocked else 'clear'}"] = group
        for layout, rows in groups.items():
            cells = [
                name,
                layout,
                len(rows),
                *(sum(r[k] for r in rows) for k in ("success", "violation", "joint")),
            ]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += [
        "",
        "## Paired joint-success changes",
        "",
        "| Comparison | Gained | Lost | Net |",
        "|---|---:|---:|---:|",
    ]
    for a, b in (("U", "FRS"), ("T", "FRS"), ("UT", "FRS"), ("UT", "U"), ("UT", "T")):
        diff = [int(runs[a][k]["joint"]) - int(runs[b][k]["joint"]) for k in keys]
        lines.append(f"| {a} minus {b} | {diff.count(1)} | {diff.count(-1)} | {sum(diff):+d} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(out, mixed):
    source = json.loads((mixed / "manifest.json").read_text())
    task_path = mixed / "checkpoints/mixed_s0/task.pt"
    user_path = out / "rotation/user.pt"
    snapshot = snapshot_download(source["policy"], revision=source["revision"], local_files_only=True)
    settings = load_experiment_settings(CONFIG, {}, pilot_settings(snapshot, [1, 8]))
    if not user_path.exists():
        calibrate(user_path.parent, snapshot)
    u = AdapterModules.read_checkpoint(user_path, mmap=True)["meta"]
    t = AdapterModules.read_checkpoint(task_path, mmap=True)["meta"]
    if (
        u["task_ids"] != [0, 2, 3, 9]
        or u["sigma"] != [0, 0]
        or u["corruption"] != settings.session.control.corruption
        or Path(u["policy_path"]).resolve() != Path(snapshot).resolve()
        or u["prompt"] != AVOID_PROMPT
        or (t["condition"], t["train_seed"], t["revision"], t["prompt"])
        != ("mixed", 0, source["revision"], AVOID_PROMPT)
    ):
        raise ValueError("Checkpoint mismatch for the rotation grid")
    session = Session(replace(settings.session, headless=True))
    try:
        session.policy.requires_grad_(False).eval()
        reference = {}
        for spec, route, seed in trials(session, "test", 10):
            path = mixed / "test/mixed_s0" / f"{spec.task_id}-{spec.init_state_id}-{seed}.json"
            row = json.loads(path.read_text())
            if row["route"] != route or row["pair_id"] != spec.metadata()["pair_id"]:
                raise ValueError(f"Archived schedule differs at {path}")
            reference[row["pair_id"]] = {f: row[f] for f in PAIR_FIELDS}
        manifest = {
            "policy": source["policy"],
            "revision": source["revision"],
            "prompt": AVOID_PROMPT,
            "rotation": settings.session.control.corruption_matrix.tolist(),
            "noise": 0,
            "gain": session.chain.gain.gain,
            "depth": 4,
            "prefix": 10,
            "seed": 0,
            "user_sha256": file_hash(user_path),
            "task_sha256": file_hash(task_path),
            "task_checkpoint": str(task_path.resolve()),
            "schedule": list(reference.values()),
        }
        path = out / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Output directory contains a different experiment")
        save_json(path, manifest)
        set_arm(session, "UT", user_path, task_path)
        save_json(out / "preflight.json", preflight(session))
        args = SimpleNamespace(out=out, stage="grid", states=10, prompt=AVOID_PROMPT, obstacle="holder")
        runs = {}
        for arm in ARMS:
            set_arm(session, arm, user_path, task_path)
            rows = run_trials(session, "test", args, arm)
            runs[arm] = {r["pair_id"]: r for r in rows}
        report(out, runs, reference)
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/pi05_libero_holder_rotation_grid"))
    parser.add_argument("--mixed", type=Path, default=Path("outputs/pi05_libero_holder_mixed"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.out, args.mixed)
