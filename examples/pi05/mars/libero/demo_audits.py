"""Audit archived teacher demonstrations and collect blinded, one-replan human challenges."""

import argparse
import json
from collections import Counter, deque
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from reproducibility import TrialSpec
from route_validation import CONFIG_DIR, HEAD, REVISION, Session, ValidationScene, load_experiment_settings
from train_visual_task import AVOID_PROMPT, save_json, write_video
from two_obstacle_pilot import layout
from visual_task_scene import PilotScene

OUT = Path("outputs/pi05_libero_demo_audits")
ARCHIVE = Path("outputs/pi05_libero_utensil_holder/train/teacher")
H = 10


def frame(obs):
    return np.concatenate([obs["pixels"][k][0, ::-1, ::-1] for k in ("image", "image2")], axis=1)


def teacher(s, scene, spec, row, enabled):
    obs = scene.reset(spec, row["layout_seed"], row["route"])
    assert s.initial_state_hash() == row["initial_state_hash"] and s.task_description == row["prompt"]
    pending = deque()
    actions, frames = [], []
    activation = changed = None
    for step in range(scene.env._max_episode_steps):
        frames.append(frame(obs))
        if not pending:
            torch.manual_seed((spec.operator_seed + step) % 2**32)
            pending.extend(s.env_actions(s.policy_chunk(obs, s.task_description))[:H])
        raw = pending.popleft()
        action = scene.teacher_action(raw) if enabled else raw
        if scene.waypoint is not None and activation is None:
            activation = step
        if changed is None and not np.array_equal(action, raw):
            changed = step
        actions.append(action)
        obs, done, success = scene.step(action)
        if done:
            break
    frames.append(frame(obs))
    return (
        np.array(actions),
        frames,
        {"success": success, "contact": scene.first_contact, "activated": activation, "changed": changed},
    )


def paired_video(path, a, b):
    write_video(
        path,
        [np.concatenate((a[min(i, len(a) - 1)], b[min(i, len(b) - 1)])) for i in range(max(len(a), len(b)))],
        fps=20,
    )


def audit_teacher(s):
    out = OUT / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for p in sorted(ARCHIVE.glob("*.npz")):
        row = json.loads(p.with_suffix(".json").read_text())
        assert row["joint"]
        cases.append((p, row))
    cells = set()

    def smoke(case):
        r = case[1]
        cell = (r["task_id"], r["blocked"])
        first = cell not in cells
        cells.add(cell)
        return not first

    cases.sort(key=smoke)
    goal = None
    results = []
    for p, row in cases:
        dest = out / p.with_suffix(".json").name
        if dest.exists():
            results.append(json.loads(dest.read_text()))
            continue
        spec = TrialSpec(**{f.name: row[f.name] for f in fields(TrialSpec)})
        if goal != spec.task_id:
            goal = spec.task_id
            s.set_scene(spec.suite, goal)
            scene = PilotScene(s.vec_env.envs[0], "holder")
        a, fa, on = teacher(s, scene, spec, row, True)
        with np.load(p) as data:
            original = data["action"]
        ok = (
            len(a) == len(original) == row["steps"]
            and np.allclose(a, original, atol=1e-5, rtol=0)
            and on["success"]
            and not on["contact"]
        )
        off = None
        fb = fa
        if not ok:
            label = "unreproduced"
        elif on["changed"] is None:
            label = "clean_without_intervention"
            off = on
        else:
            b, fb, off = teacher(s, scene, spec, row, False)
            t = on["changed"]
            assert np.allclose(a[:t], b[:t], atol=1e-5, rtol=0)
            label = (
                "collision_prevented"
                if off["contact"]
                else "placement_benefit"
                if not off["success"]
                else "both_clean"
            )
        clip = out / f"{goal}-{label}.mp4"
        if not clip.exists():
            paired_video(clip, fa, fb)
        result = {
            "trial": p.stem,
            "goal": goal,
            "blocked": row["blocked"],
            "outcome": label,
            "on": on,
            "off": off,
        }
        save_json(dest, result)
        results.append(result)
        print(p.stem, label, flush=True)
        if len(results) == 4 and all(r["outcome"] == "unreproduced" for r in results):
            raise RuntimeError("All four smoke cases diverged")
    assert len(results) == 100
    save_json(
        out / "summary.json",
        {
            str(cell): dict(Counter(r["outcome"] for r in results if (r["goal"], r["blocked"]) == cell))
            for cell in cells
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["teacher", "prepare", "score"])
    parser.add_argument("--answers", type=Path)
    args = parser.parse_args()
    if args.stage == "score" and args.answers is None:
        parser.error("score requires --answers answers.json")
    cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
    cfg.control.n_reversal_steps, cfg.control.pinned_steps = 4, H
    path = snapshot_download(cfg.policy_path, revision=REVISION, local_files_only=True)
    s = Session(replace(cfg, policy_path=path, n_action_steps=H, headless=True))
    try:
        s.policy.requires_grad_(False).eval()
        if args.stage == "teacher":
            audit_teacher(s)
        else:
            human_audit(s, args)
    finally:
        s.close()


def human_rollout(s, scene, item, goal, intervention=None):
    from unittest.mock import patch

    from train_visual_task import episode
    from visual_task_scene import BOWL

    spec = replace(TrialSpec(**item["spec"]), task_id=goal)
    records = {}
    frames = []
    actions = []
    original_step = scene.step
    original_read = s.chunk_command
    original_reset = scene.reset

    def reset(*args):
        obs = original_reset(*args)
        frames.append(frame(obs))
        return obs

    def step(action):
        obs, done, success = original_step(action)
        frames.append(frame(obs))
        actions.append(np.asarray(action).copy())
        return obs, done, success

    def command():
        c = original_read()
        t = scene.control_step
        records[t] = {
            "command": c[0].tolist(),
            "state": s.initial_state_hash(),
            "gripper": float(s.steering_policy().core._gripper_ref),
            "grasp": bool(
                scene.domain._check_grasp(
                    scene.domain.robots[0].gripper, scene.domain.objects_dict[BOWL].contact_geoms
                )
            ),
        }
        if intervention and t == intervention[0]:
            return c.new_tensor([intervention[1]])
        return c

    with (
        patch.object(scene, "step", step),
        patch.object(scene, "reset", reset),
        patch.object(s, "chunk_command", command),
    ):
        row, _ = episode(s, scene, spec, item["holder"], item["seed"], "visual", prompt=AVOID_PROMPT)
    return row, records, frames, np.array(actions)


def signature(row):
    return {k: row[k] for k in ("initial_state_hash", "first_contact", "success", "joint", "steps")}


def prepare_human(s, scene, out):
    import base64

    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    selected = []
    counts = Counter()
    specs = TrialSpec.for_schedule([1] * 20, "libero_goal", 0, {1: s.initial_state_count("libero_goal", 1)})
    candidates = []
    for rep in (23, 24, 25):
        for i, spec in enumerate(specs):
            item = layout(spec, i, rep)
            for g in (1, 8):
                candidates.append((item, g))
    for n, (item, g) in enumerate(candidates):
        cache = out / f"candidate-{n}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            if "case" in saved:
                selected.append(saved["case"])
                if not saved["case"]["practice"]:
                    counts[tuple(saved["key"])] += 1
            continue
        if len(selected) >= 18:
            break
        if n >= 2 and all(
            counts[(g, k, o)] >= (1 if k == "control" else 3)
            for k in ("control", "contact")
            for o in ("utensil_holder", "pepper_mill")
        ):
            continue
        row, recs, frames, _ = human_rollout(s, scene, item, g)
        contact = row["first_contact"]
        practice = n < 2
        kind = "contact" if contact else "control"
        obj = contact["obstacle"] if contact else ("utensil_holder" if g == item["holder"] else "pepper_mill")
        key = (g, kind, obj)
        if not practice and (counts[key] >= (3 if contact else 1) or not (contact or row["joint"])):
            save_json(cache, {"status": "outside quota"})
            continue
        if contact:
            t = max(0, (contact["step"] // H - 1) * H)
        else:
            matches = [c for c in selected if c["kind"] == "contact" and c["goal"] == g]
            target = matches[-1] if matches else None
            best_time = target["t"] if target else 40
            t = min(
                recs,
                key=lambda j: (
                    target is not None and recs[j]["grasp"] != target["record"]["grasp"],
                    abs(j - best_time),
                ),
            )
        case = {
            "item": item,
            "goal": g,
            "t": t,
            "record": recs[t],
            "baseline": signature(row),
            "kind": kind,
            "practice": practice,
        }
        # Confirm replay, including all closed-loop operator updates, before involving the human.
        repeat, probes, _, _ = human_rollout(s, scene, item, g, (t, recs[t]["command"]))
        if signature(repeat) != case["baseline"] or probes[t] != recs[t]:
            save_json(
                cache, {"status": "unreproduced", "source": signature(row), "repeat": signature(repeat)}
            )
            print("Excluded unreproduced candidate", n, flush=True)
            continue
        stem = f"candidate-{n}"
        write_video(out / f"{stem}.mp4", frames[max(0, t - 30) : t + 1] + [frames[t]] * 10, fps=20)
        Image.fromarray(frames[t]).save(out / f"{stem}.png")
        case["stem"] = stem
        save_json(cache, {"case": case, "key": list(key)})
        selected.append(case)
        if not practice:
            counts[key] += 1
        print("Selected", len(selected), flush=True)
    order = selected[:2] + [
        selected[i] for i in np.random.default_rng(592).permutation(np.arange(2, len(selected)))
    ]
    public = []
    for i, c in enumerate(order):
        trial = {"id": i, "goal": "stove" if c["goal"] == 1 else "plate", "practice": c["practice"]}
        for ext, mime in (("mp4", "video/mp4"), ("png", "image/png")):
            media = (out / f"{c['stem']}.{ext}").read_bytes()
            trial[ext] = f"data:{mime};base64," + base64.b64encode(media).decode("ascii")
        public.append(trial)
    save_json(out / "design.json", order)
    template = Path(__file__).with_name("human_command_form.html").read_text()
    (out / "commands.html").write_text(template.replace("/*CASES*/[]", json.dumps(public)))
    save_json(
        out / "preparation.json",
        {"selected": len(selected) - 2, "strata": {str(k): v for k, v in counts.items()}},
    )


def human_audit(s, args):
    s.set_scene("libero_goal", 1)
    scene = ValidationScene(s.vec_env.envs[0], "two_obstacles")
    s.adapter.load_modules(task_path=HEAD, device=s.policy.config.device)
    out = OUT / "human"
    if args.stage == "prepare":
        if not (out / "commands.html").exists():
            prepare_human(s, scene, out)
        return
    cases = json.loads((out / "design.json").read_text())
    answers = json.loads(args.answers.read_text())
    if set(answers) != set(map(str, range(len(cases)))):
        raise ValueError("A confirmed vector is required for every trial")
    scored = []
    for i, c in enumerate(cases):
        v = np.asarray(answers[str(i)], dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all() or np.max(abs(v)) > 1:
            raise ValueError(f"Invalid command for {i}")
        if c["practice"]:
            continue
        t = c["t"]
        base, rb, fb, ab = human_rollout(s, scene, c["item"], c["goal"], (t, c["record"]["command"]))
        other, rh, fh, ah = human_rollout(s, scene, c["item"], c["goal"], (t, v.tolist()))
        assert (
            signature(base) == c["baseline"]
            and rb[t] == rh[t] == c["record"]
            and np.array_equal(ab[:t], ah[:t])
        )
        outcome = (
            ("gain" if other["joint"] else "loss")
            if base["joint"] != other["joint"]
            else ("both_clean" if base["joint"] else "both_failed")
        )
        if outcome in ("gain", "loss"):
            paired_video(out / f"comparison-{i}.mp4", fb, fh)
        sim_vec = np.asarray(rb[t]["command"])
        sim_norm, human_norm = np.linalg.norm(sim_vec), np.linalg.norm(v)
        sim_hit, human_hit = base["first_contact"], other["first_contact"]
        scored.append(
            {
                "id": i,
                "goal": c["goal"],
                "kind": c["kind"],
                "takeover": t,
                "sim_command": rb[t]["command"],
                "human_command": v.tolist(),
                "sim_magnitude": float(sim_norm),
                "human_magnitude": float(human_norm),
                "direction_cosine": float(sim_vec @ v / (sim_norm * human_norm))
                if sim_norm and human_norm
                else None,
                "sim_rest": bool(np.max(abs(sim_vec)) < 0.05),
                "later_bowl_contact": bool(
                    sim_hit
                    and human_hit
                    and human_hit["with"] == "bowl"
                    and human_hit["step"] > sim_hit["step"]
                ),
                "sim": signature(base),
                "human": signature(other),
                "outcome": outcome,
            }
        )
    save_json(out / "scored.json", scored)
    save_json(
        out / "summary.json",
        {
            kind: dict(Counter(r["outcome"] for r in scored if r["kind"] == kind))
            for kind in ("control", "contact")
        },
    )


if __name__ == "__main__":
    main()
