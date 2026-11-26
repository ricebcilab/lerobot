"""Paired XYZ rescue screen for the clean 10/10 post-fusion holder pilot."""

import argparse
import fcntl
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from session import Session
from train_visual_task import AVOID_PROMPT, save_json, trials
from visual_task_scene import PilotScene

from lerobot.envs import preprocess_observation

H = 10
BASE = Path("outputs/pi05_libero_holder_camera_stages")
OUT = Path("outputs/pi05_libero_holder_interventions")


def reset(s, scene, case):
    spec, route, seed = case
    obs = scene.reset(spec, seed, route)
    s.steering_policy().reset()
    s.chain.reset()
    s.chain.readout.rng = np.random.default_rng(spec.input_noise_seed)
    s.operator.reset()
    return obs


def chunk(s, obs, spec, step, capture=False):
    steering = s.steering_policy()
    s.chain.advance(step)
    torch.manual_seed((spec.operator_seed + step // H) % 2**32)
    s._refresh_operator(obs, step)
    batch = s.policy_batch(obs, AVOID_PROMPT)
    command = batch["command"] = s.chunk_command()
    seed = (spec.policy_seed + step // H) % 2**32
    torch.manual_seed(seed)
    record = {}
    if capture:
        cfg = s.policy.config
        noise = s.policy.model.sample_noise((1, cfg.chunk_size, cfg.max_action_dim), cfg.device)
        torch.manual_seed(seed)
        inputs = s.env_preprocessor(preprocess_observation(obs))
        record = {
            "frame_0": np.stack([obs["pixels"][k][0, ::-1, ::-1] for k in ("image", "image2")]),
            "state": inputs["observation.state"].cpu().numpy(),
            "noise": noise[0].cpu().numpy(),
            "command": command[0].numpy(),
            "gripper_ref": float(steering.core._gripper_ref),
            "stride": H,
        }
    with torch.inference_mode(), s.autocast_ctx:
        first = steering.select_action(batch)
        prefix = torch.stack([first] + [steering.select_action(batch) for _ in range(H - 1)], dim=1)
    if capture:
        record["normalized"] = prefix[0].float().cpu().numpy()
    return s.env_actions(prefix), record


def execute(scene, actions):
    for action in actions:
        obs, done, success = scene.step(action)
        if scene.violation or done:
            return obs, True, bool(success)
    return obs, False, False


def source(s, scene, case):
    obs = reset(s, scene, case)
    records = {}
    actions = []
    for start in range(0, scene.env._max_episode_steps, H):
        prefix, record = chunk(s, obs, case[0], start, True)
        record.update(
            action=prefix,
            state_hash=s.initial_state_hash(),
            next_gripper_ref=float(s.steering_policy().core._gripper_ref),
        )
        records[start] = record
        for action in prefix:
            obs, done, success = scene.step(action)
            actions.append(action)
            if scene.violation or done:
                return records, actions, scene.first_contact, bool(success)
    return records, actions, None, False


def branch(s, scene, case, actions, records, start, prefix):
    obs = reset(s, scene, case)
    if start:
        obs, done, _ = execute(scene, actions[:start])
        if done:
            raise ValueError("Replay terminated before intervention")
    if s.initial_state_hash() != records[start]["state_hash"]:
        raise ValueError("Replay state does not match source")
    s.steering_policy().core._gripper_ref = records[start]["next_gripper_ref"]
    for step in range(start, scene.env._max_episode_steps, H):
        if step != start:
            prefix, _ = chunk(s, obs, case[0], step)
        obs, done, success = execute(scene, prefix)
        if done:
            return {
                "joint": success and not scene.violation,
                "end_step": scene.control_step,
                "success": success,
                "contact": scene.first_contact,
            }
    return {"joint": False, "success": False, "contact": None}


def perturb(prefix, direction, magnitude):
    corrected = prefix.copy()
    corrected[:, :3] = np.clip(prefix[:, :3] + magnitude * direction, -1, 1)
    return corrected


def lateral(prefix, scene):
    dx, dy = prefix[:, :2].mean(axis=0)
    if np.hypot(dx, dy) < 1e-5:
        dx, dy = (scene.goal - scene.start)[:2]
    return np.array([-dy, dx, 0.0]) / np.hypot(dx, dy)


def screen_case(s, scene, case, out, *, preserve_blocked=False, needed_objects=None):
    spec, route, seed = case
    key = f"{spec.task_id}-{spec.init_state_id}-{seed}"
    path = out / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    records, actions, contact, success = source(s, scene, case)
    row = dict(
        **spec.metadata(),
        route=route,
        layout_seed=seed,
        blocked=bool(scene.blocked),
        contact=contact,
        success=success,
        branches=[],
        rescue=False,
    )
    if contact and (needed_objects is None or contact["obstacle"] in needed_objects):
        starts = sorted((k for k in records if k < contact["step"]), reverse=True)[:2]
        for start in starts:
            check = branch(s, scene, case, actions, records, start, records[start]["action"])
            if not check["contact"] or any(
                check["contact"][k] != contact[k] for k in ("step", "geom", "obstacle")
            ):
                raise ValueError(f"Unmodified branch failed to reproduce {key} at {start}")
        for magnitude in (0.15, 0.30):
            for start in starts:
                record = records[start]
                prefix = record["action"]
                left = lateral(prefix, scene)
                for name, direction in (("up", [0, 0, 1]), ("left", left), ("right", -left)):
                    corrected = perturb(prefix, np.asarray(direction), magnitude)
                    result = branch(s, scene, case, actions, records, start, corrected)
                    row["branches"].append(dict(start=start, direction=name, magnitude=magnitude, **result))
                    if result["joint"]:
                        np.savez_compressed(
                            out / f"{key}.npz",
                            **dict(record, original=prefix, action=corrected[: result["end_step"] - start]),
                        )
                        row["rescue"] = True
                        break
                if row["rescue"]:
                    break
            if row["rescue"]:
                break
    elif not contact and success and (not row["blocked"] or preserve_blocked):
        start = sorted(records)[len(records) // 2]
        np.savez_compressed(
            out / f"{key}.npz",
            **dict(records[start], action=records[start]["action"][: len(actions) - start]),
        )
    save_json(path, row)
    return row


def collect(s, out):
    # All 20 train reset IDs before drawing another holder layout on a reset.
    cases = list(trials(s, "train", 20, repeats=100))
    ordered = [case for rep in range(100) for case in cases[rep::100]]
    plates = [c for c in ordered if c[0].task_id == c[1] == 8]
    others = [c for c in ordered if not (c[0].task_id == c[1] == 8)]
    contacts = rescues = plate_rescues = clear = 0
    task = scene = None
    plate_cases = iter(plates)
    other_cases = iter(others)
    alternate_plate = False
    while contacts < 200:
        plate = contacts < 20 or (plate_rescues < 16 and (alternate_plate or rescues - plate_rescues >= 16))
        case = next(plate_cases if plate else other_cases)
        spec, route, seed = case
        alternate_plate = not plate
        if task != spec.task_id:
            task = spec.task_id
            s.set_scene("libero_goal", task)
            scene = PilotScene(s.vec_env.envs[0], "holder")
        row = screen_case(s, scene, case, out)
        contacts += int(bool(row["contact"]))
        rescues += int(row["rescue"])
        plate_rescues += int(plate and row["rescue"])
        clear += int(not row["blocked"] and row["success"] and not row["contact"])
        counts = {"contacts": contacts, "rescues": rescues, "plate_rescues": plate_rescues, "clear": clear}
        save_json(out.parent / "collection.json", counts)
        print(counts, flush=True)
        if contacts == 20 and rescues < 5 or contacts >= 200 or rescues >= 32 and plate_rescues >= 16:
            break
    return counts


def wrist_probe(s, out):
    """On failed gripper-first cases add wrist tilt to the longest-clear XYZ branch."""
    directory = out / "wrist"
    directory.mkdir(exist_ok=True)
    rows = [json.loads(f.read_text()) for f in sorted((out / "pairs").glob("*.json"))]
    source_rows = {
        (r["task_id"], r["init_state_id"], r["layout_seed"]): r
        for r in rows
        if r["blocked"] and r["contact"] and r["contact"]["geom"].startswith("gripper0_") and not r["rescue"]
    }
    s.set_scene("libero_goal", 8)
    scene = PilotScene(s.vec_env.envs[0], "holder")
    for case in trials(s, "train", 20, repeats=100):
        spec, route, seed = case
        key = (spec.task_id, spec.init_state_id, seed)
        if key not in source_rows or spec.task_id != 8:
            continue
        path = directory / ("-".join(map(str, key)) + ".json")
        if path.exists():
            continue
        row = source_rows[key]
        candidates = row["branches"]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda b: (b["contact"] is None, (b["contact"] or {}).get("step", 0), -b["magnitude"]),
        )
        start = best["start"]
        records, actions, contact, _ = source(s, scene, case)
        assert contact["step"] == row["contact"]["step"] and contact["geom"] == row["contact"]["geom"]
        original = records[start]["action"]
        left = lateral(original, scene)
        direction = {"left": left, "right": -left, "up": np.array([0, 0, 1])}[best["direction"]]
        base = perturb(original, direction, best["magnitude"])
        control = branch(s, scene, case, actions, records, start, base)

        def outcome(branch_row):
            contact = branch_row["contact"]
            return branch_row["success"], (contact["step"], contact["geom"]) if contact else None

        if outcome(control) != outcome(best):
            raise ValueError(f"Failed to reproduce XYZ control for {key}")
        attempts = []
        for dim, sign in ((3, 1), (3, -1), (4, 1), (4, -1)):
            updated = base.copy()
            updated[:, dim] = np.clip(base[:, dim] + sign * 0.3, -1, 1)
            result = branch(s, scene, case, actions, records, start, updated)
            attempts.append(dict(dim=dim, delta=sign * 0.3, **result))
            if result["joint"]:
                np.savez_compressed(
                    path.with_suffix(".npz"),
                    **dict(
                        records[start],
                        original=original,
                        xyz_only=base,
                        action=updated[: result["end_step"] - start],
                    ),
                )
                break
        save_json(path, {"source": key, "xyz": best, "wrist": attempts})
        print("Wrist replay", key, attempts[-1]["joint"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("stage", nargs="?", choices=("collect", "train", "wrist"), default="collect")
    args = p.parse_args()
    out = args.out
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    with (out / ".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.stage, out)


def run(stage, out):
    base = json.loads((BASE / "manifest.json").read_text())
    ckpt = BASE / "checkpoints/post_s0/task.pt"
    snapshot = snapshot_download(base["policy"], revision=base["revision"], local_files_only=True)
    cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
    assert cfg.control.corruption is None and cfg.control.translation_noise == 0
    cfg.control.n_reversal_steps, cfg.control.pinned_steps = 4, H
    s = Session(replace(cfg, policy_path=snapshot, n_action_steps=H, headless=True))
    try:
        s.policy.requires_grad_(False).eval()
        s.adapter.load_modules(task_path=ckpt, device=s.policy.config.device)
        with ckpt.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {"base": base, "checkpoint": str(ckpt), "sha256": digest, "prompt": AVOID_PROMPT}
        path = out / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f"Output belongs to a different pilot: {out}")
        save_json(path, manifest)
        if stage == "collect":
            collect(s, out / "pairs")
        else:
            from holder_intervention_train import check_runtime, experiment

            check_runtime(s, out)
            if stage == "wrist":
                wrist_probe(s, out)
            else:
                experiment(s, out)
    finally:
        s.close()


if __name__ == "__main__":
    main()
