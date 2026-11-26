"""Single-seed paired evaluation of C against the same recipe trained on both obstacles."""

import argparse
import fcntl
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from config import CONFIG_DIR, load_experiment_settings
from holder_contact_probe import probe
from holder_factorial import digest
from holder_intervention_train import fit, prefix
from holder_interventions import BASE, screen_case
from huggingface_hub import snapshot_download
from paired_obstacles import CHECKPOINT, REVISION, PairedGoalScene
from reproducibility import TrialSpec
from session import Session
from train_visual_task import AVOID_PROMPT, DemoChunks, episode, save_json

OBJECTS = ("utensil_holder", "pepper_mill")
GOALS = (1, 8)


def layout(spec, i, rep):
    seed = int(np.random.SeedSequence([spec.env_seed, 13471, rep]).generate_state(1)[0])
    return {"spec": asdict(spec), "holder": GOALS[(i + rep) % 2], "seed": seed}


def evaluate(s, scene, out, schedule, name, ckpt):
    dest = out / "test" / name
    dest.mkdir(parents=True, exist_ok=True)
    s.adapter.load_modules(task_path=ckpt, device=s.policy.config.device)
    rows = []
    for i, item in enumerate(schedule):
        pair = []
        for goal in GOALS:
            path = dest / f"{i}-{goal}.json"
            if not path.exists():
                spec = replace(TrialSpec(**item["spec"]), task_id=goal)
                row, _ = episode(s, scene, spec, item["holder"], item["seed"], "visual", prompt=AVOID_PROMPT)
                row.update(
                    image_hash=scene.image_hash,
                    route_obstacle=OBJECTS[int(goal != item["holder"])],
                    obstacles={str(d): c.tolist() for d, c in scene.centers.items()},
                )
                save_json(path, row)
            pair.append(json.loads(path.read_text()))
        for key in ("image_hash", "initial_state_hash", "obstacles"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Unpaired {name} layout {i}: {key}")
        rows.extend(pair)
    return rows


def summarize(rows):
    return {
        "joint_by_object": {o: sum(r["joint"] for r in rows if r["route_obstacle"] == o) for o in OBJECTS},
        "placement": sum(r["success"] for r in rows),
        "wrong": sum(
            not r["success"] and ("plate" if r["task_id"] == 1 else "stove") in r["bowl_final_on"]
            for r in rows
        ),
        "both_joint": sum(a["joint"] and b["joint"] for a, b in zip(rows[::2], rows[1::2], strict=True)),
        "first_contacts": dict(
            Counter(
                r["first_contact"]["obstacle"] + "/" + r["first_contact"]["with"]
                for r in rows
                if r["violation"]
            )
        ),
    }


def collect(s, scene, out, specs):
    directory = out / "pairs"
    directory.mkdir(exist_ok=True)
    counts = Counter()
    hits = Counter()
    rescues = []
    preserves = []
    attempts = 0
    for rep in range(4):
        for i, spec in enumerate(specs):
            item = layout(spec, i, rep)
            for goal in GOALS:
                needed = {o for o in OBJECTS if counts[o, goal] < 8}
                if not needed:
                    continue
                case = (replace(spec, task_id=goal), item["holder"], item["seed"])
                r = screen_case(s, scene, case, directory, preserve_blocked=True, needed_objects=needed)
                attempts += 1
                file = directory / f"{goal}-{spec.init_state_id}-{item['seed']}.npz"
                if r["contact"]:
                    obstacle = r["contact"]["obstacle"]
                    hits[obstacle, goal] += 1
                    if r["rescue"]:
                        counts[obstacle, goal] += 1
                        rescues.append(file)
                elif r["success"]:
                    preserves.append(file)
                report = {
                    "attempts": attempts,
                    "preserves": len(preserves),
                    "strata": {
                        f"{o}/{g}": {"contacts": hits[o, g], "rescues": counts[o, g]}
                        for o in OBJECTS
                        for g in GOALS
                    },
                }
                save_json(out / "collection.json", report)
                if len(rescues) == 32:
                    return rescues, preserves
    return rescues, preserves


def run(s, out):
    s.set_scene("libero_goal", 1)
    scene = PairedGoalScene(s.vec_env.envs[0], "two_obstacles")
    n = {1: s.initial_state_count("libero_goal", 1)}
    train = TrialSpec.for_schedule([1] * 20, "libero_goal", 0, n)
    test = TrialSpec.for_schedule([1] * 16, "libero_goal", 0, n, 30)
    assert not {r.init_state_id for r in train} & {r.init_state_id for r in test}
    schedule = [layout(spec, i, 0) for i, spec in enumerate(test)]
    initial = BASE / "checkpoints/post_s0/task.pt"
    manifest = {
        "checkpoint": str(initial),
        "collector": str(CHECKPOINT),
        "revision": REVISION,
        "prompt": AVOID_PROMPT,
        "schedule": schedule,
        "checkpoints_digest": digest([initial, CHECKPOINT]),
    }
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Changed experiment manifest")
    save_json(path, manifest)
    base = evaluate(s, scene, out, schedule, "baseline", CHECKPOINT)
    results = {"baseline": summarize(base)}
    save_json(out / "results.json", results)
    rescue, preserve = collect(s, scene, out, train)
    if len(rescue) != 32 or not preserve:
        save_json(out / "decision.json", {"status": "insufficient_training_examples"})
        return
    # Compare training's differentiable prefix to the actual collecting C checkpoint.
    f = rescue[0]
    raw, _ = DemoChunks([f], 50, AVOID_PROMPT).batch(np.random.default_rng(0), 1)
    with np.load(f) as d, torch.no_grad(), s.autocast_ctx:
        prediction = prefix(s, s.preprocessor(raw), d)[0]
        torch.testing.assert_close(
            prediction,
            torch.as_tensor(d["normalized"][:, :3], device=prediction.device),
            atol=0.02,
            rtol=0.01,
        )
    risk = probe(s, out, files=rescue + preserve, objects=OBJECTS)
    if risk is None:
        save_json(out / "decision.json", {"status": "contact_ranking_failed"})
        return
    risk[0].to(s.policy.config.device).requires_grad_(False)
    ref = Path(json.loads((BASE / "manifest.json").read_text())["reference"])
    original = DemoChunks(sorted((ref / "train/teacher").glob("*.npz")), 50, AVOID_PROMPT)
    ckpt = fit(s, out, "C", original, rescue, preserve, risk)
    new = evaluate(s, scene, out, schedule, "two_obstacle", ckpt)
    for a, b in zip(base, new, strict=True):
        assert all(
            a[k] == b[k]
            for k in (
                "image_hash",
                "initial_state_hash",
                "layout_seed",
                "task_id",
                "policy_seed",
                "operator_seed",
                "obstacles",
            )
        )
    results["two_obstacle"] = summarize(new)
    change = {}
    for o in OBJECTS:
        pairs = [(a, b) for a, b in zip(base, new, strict=True) if a["route_obstacle"] == o]
        assert len(pairs) == 16
        change[o] = {
            "gained": sum(b["joint"] and not a["joint"] for a, b in pairs),
            "lost": sum(a["joint"] and not b["joint"] for a, b in pairs),
        }
    results["paired"] = change
    save_json(out / "results.json", results)
    a, b = results["baseline"], results["two_obstacle"]
    ceiling = [o for o in OBJECTS if a["joint_by_object"][o] >= 15]
    passed = (
        all(x["gained"] - x["lost"] >= 2 for x in change.values())
        and b["placement"] >= a["placement"] - 1
        and b["wrong"] <= a["wrong"]
    )
    save_json(out / "decision.json", {"advance": passed, "ceiling_limited": ceiling})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/pi05_libero_two_obstacle_pilot"))
    out = p.parse_args().out
    out.mkdir(exist_ok=True, parents=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
        cfg.control.n_reversal_steps, cfg.control.pinned_steps = 4, 10
        snapshot = snapshot_download(cfg.policy_path, revision=REVISION, local_files_only=True)
        session = Session(replace(cfg, policy_path=snapshot, n_action_steps=10, headless=True))
        try:
            session.policy.requires_grad_(False).eval()
            run(session, out)
        finally:
            session.close()
