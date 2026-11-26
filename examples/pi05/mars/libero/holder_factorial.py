"""Post-fusion holder pilot using the archived real-view baseline."""

import argparse
import fcntl
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from session import Session
from train_visual_task import AVOID_PROMPT, episode, run_trials, save_json, train, trials
from visual_task_scene import PilotScene


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with p.open("rb") as stream:
            h.update(p.name.encode() + hashlib.file_digest(stream, "sha256").digest())
    return h.hexdigest()


def corrections(session, root, baseline, *, attempt_limit=200):
    dest = root / "recoveries"
    dest.mkdir(exist_ok=True)
    pools = [[], []]
    attempts = [0, 0]
    by_layout = {}
    horizon = session.settings.n_action_steps
    cases = list(trials(session, "train", 20))
    current = None
    for rep in range(attempt_limit):
        for spec, initial_route, _ in cases:
            route = initial_route if rep % 2 == 0 else 9 - initial_route
            pool = int(spec.task_id == route == 8)
            if len(pools[pool]) >= 16 or attempts[pool] >= attempt_limit:
                continue
            seed = int(np.random.SeedSequence([spec.env_seed, rep, 202611]).generate_state(1)[0])
            collector = attempts[pool] % 3
            attempts[pool] += 1
            path = dest / f"{spec.task_id}-{spec.init_state_id}-{seed}.json"
            archive = path.with_suffix(".npz")
            if path.exists():
                row = json.loads(path.read_text())
            else:
                if current != spec.task_id:
                    session.set_scene("libero_goal", spec.task_id)
                    scene = PilotScene(session.vec_env.envs[0], "holder")
                    current = spec.task_id
                session.adapter.load_modules(
                    task_path=baseline / f"checkpoints/post_s{collector}/task.pt",
                    device=session.policy.config.device,
                )
                trace = {"hashes": {}}
                row, _ = episode(
                    session, scene, spec, route, seed, "visual", prompt=AVOID_PROMPT, trace=trace
                )
                row.update(collector=collector, takeovers=[], accepted=False)
                if pool != int(row["blocked"] and spec.task_id == 8):
                    raise ValueError("Sampled holder layout differs from requested recovery stratum")
                if row["violation"]:
                    boundary = ((row["first_contact"]["step"] - 10) // horizon) * horizon
                    for step in (boundary, boundary - horizon):
                        if step < 0:
                            break
                        recovery, demo = episode(
                            session,
                            scene,
                            spec,
                            route,
                            seed,
                            "teacher",
                            True,
                            replay=(trace["actions"][:step], trace["hashes"][step]),
                        )
                        row["takeovers"].append(
                            {"step": step, "joint": recovery["joint"], "steps": recovery["steps"]}
                        )
                        if demo is not None:
                            tmp = archive.with_suffix(".tmp.npz")
                            np.savez_compressed(tmp, **demo)
                            tmp.replace(archive)
                            row["accepted"] = True
                            break
                save_json(path, row)
            layout = f"{spec.task_id}_{'blocked' if row['blocked'] else 'clear'}"
            counts = by_layout.setdefault(layout, {"attempted": 0, "contact": 0, "accepted": 0})
            counts["attempted"] += 1
            counts["contact"] += int(row["violation"])
            counts["accepted"] += int(row["accepted"])
            if row["accepted"]:
                if not archive.is_file():
                    raise FileNotFoundError(archive)
                pools[pool].append(archive)
            print("Recovery attempts", attempts, "accepted", [len(p) for p in pools], flush=True)
        if all(len(p) >= 16 or n >= attempt_limit for p, n in zip(pools, attempts, strict=True)):
            break
    save_json(root / "collection.json", by_layout)
    size = min(map(len, pools))
    return pools[0][:size] + pools[1][:size] if size >= 8 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "experiment", "mixed"])
    parser.add_argument("--baseline", type=Path, default=Path("outputs/pi05_libero_holder_camera_stages"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--recovery-source", type=Path, default=Path("outputs/pi05_libero_holder_factorial"))
    a = parser.parse_args()
    if a.out is None:
        a.out = Path(
            "outputs/pi05_libero_holder_mixed"
            if a.stage == "mixed"
            else "outputs/pi05_libero_holder_factorial"
        )
    if a.stage == "mixed" and a.out.resolve() == a.recovery_source.resolve():
        parser.error("Mixed-only training requires a separate output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ref = json.loads((a.baseline / "manifest.json").read_text())
        demo_dir = Path(ref["reference"]) / "train/teacher"
        if (
            ref["prompt"] != AVOID_PROMPT
            or ref["seeds"] != [0, 1, 2]
            or ref["batch"] != 2
            or digest(demo_dir.glob("*.npz")) != ref["demo_sha256"]
        ):
            raise ValueError("Unexpected baseline configuration or demonstration digest")
        if a.out.resolve() in (a.baseline.resolve(), demo_dir.parent.parent.resolve()):
            raise ValueError("Use a dedicated output directory for this experiment")
        manifest = dict(
            ref,
            baseline=str(a.baseline.resolve()),
            baseline_heads=digest(a.baseline / f"checkpoints/post_s{s}/task.pt" for s in ref["seeds"]),
        )
        if a.stage == "mixed":
            if json.loads((a.recovery_source / "manifest.json").read_text()) != manifest:
                raise ValueError("Recovery source and requested baseline differ")
            manifest.update(recovery_source=str(a.recovery_source.resolve()), attempt_limit=2000)
        path = a.out / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Output belongs to a different baseline")
        save_json(path, manifest)
        if a.stage == "mixed":
            # Completed immutable source trials can be resumed in the same deterministic order.
            shutil.copytree(a.recovery_source / "recoveries", a.out / "recoveries", dirs_exist_ok=True)
        cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
        cfg.control.n_reversal_steps, cfg.control.pinned_steps = ref["depth"], ref["prefix"]
        snapshot = snapshot_download(ref["policy"], revision=ref["revision"], local_files_only=True)
        session = Session(replace(cfg, policy_path=snapshot, headless=True))
        session.policy.requires_grad_(False).eval()
        try:
            paths = corrections(session, a.out, a.baseline, attempt_limit=manifest.get("attempt_limit", 200))
            save_json(a.out / "selected.json", [p.name for p in paths])
            if a.stage == "mixed" and not paths:
                raise RuntimeError("Extended collection yielded fewer than eight clean recoveries per pool")
            if a.stage in ("experiment", "mixed"):
                experiment(session, a, manifest, demo_dir, paths)
        finally:
            session.close()


def calibration(demo_dir, out):
    """Replay one archived clean demo per task and check the shared external camera projection."""
    from PIL import Image, ImageDraw
    from robosuite import macros
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    from lerobot.envs import close_envs

    panels, projection = [], None
    for task in (1, 8):
        file = next(iter(sorted(demo_dir.glob(f"{task}-*.npz"))))
        row = json.loads(file.with_suffix(".json").read_text())
        _, envs, vec = Session._build_env("libero_goal", task)
        try:
            scene = PilotScene(vec.envs[0], "holder")
            obs = scene.reset(SimpleNamespace(**row), row["layout_seed"], row["route"])
            h, w = obs["pixels"]["image"].shape[1:3]
            if h != w or macros.IMAGE_CONVENTION != "opengl":
                raise ValueError("Expected square LIBERO images in raw OpenGL orientation")
            matrix = get_camera_transform_matrix(scene.domain.sim, "agentview", h, w)[:3].copy()
            # OpenCV projection has +y down. The policy rotates bottom-up GL images 180 degrees,
            # leaving +y down and reversing x. Express patch centers in normalized image units.
            matrix[0] = matrix[2] - matrix[0] / w
            matrix[1] /= h
            if projection is not None:
                np.testing.assert_allclose(matrix, projection, atol=1e-6)
            projection = matrix
            with np.load(file) as demo:
                stride = int(demo["stride"])
                frames, lifted = [], []
                for step, action in enumerate(demo["action"]):
                    if step % stride == 0:
                        i = step // stride
                        xyz = demo["state"][i, :3]
                        np.testing.assert_allclose(obs["robot_state"]["eef"]["pos"][0], xyz, atol=1e-5)
                        live = obs["pixels"]["image"][0, ::-1, ::-1]
                        archived = demo[f"frame_{i}"][0]
                        uvz = matrix @ np.r_[xyz, 1]
                        xy = uvz[:2] / uvz[2] * [w, h]
                        frame = Image.fromarray(archived)
                        draw = ImageDraw.Draw(frame)
                        x, y = xy
                        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="lime", width=2)
                        frames.append(frame)
                        lifted.append(scene.position("akita_black_bowl_1")[2] > scene.start[2] + 0.025)
                        if i == 0:
                            # Mesa and NVIDIA differ in shading. Check orientation independently of that variation.
                            errors = [
                                np.abs(view.astype(float) - archived).mean()
                                for view in (live, live[::-1], live[:, ::-1], live[::-1, ::-1])
                            ]
                            if np.argmin(errors) != 0 or np.std(live) < 1:
                                raise ValueError(f"Archived/live camera orientation mismatch: {errors}")
                    obs, _, _ = scene.step(action)
                lift = lifted.index(True)
                panels.extend(frames[i] for i in (0, max(0, lift - 1), (lift + len(frames) - 1) // 2))
        finally:
            close_envs(envs)
    sheet = Image.new("RGB", (3 * w, 2 * h))
    for i, frame in enumerate(panels):
        sheet.paste(frame, ((i % 3) * w, (i // 3) * h))
    sheet.save(out / "gripper_projection.png")
    return projection.tolist()


def experiment(session, a, manifest, demo_dir, paths):
    from visual_task_report import factorial_report

    if a.stage == "mixed":
        matrix, arms = None, ("mixed",)
    else:
        matrix = calibration(demo_dir, a.out)
        arms = ("geometry", "mixed", "combined") if paths else ("geometry",)
    args = SimpleNamespace(**{**manifest, "out": a.out, "stage": "factorial", "demos": 100})
    provenance = dict(manifest, projection=matrix, recovery_count=len(paths), recovery_sha256=digest(paths))
    save_json(a.out / "training.json", provenance)
    train(session, args, provenance, demo_dir=demo_dir, arms=arms, projection=matrix, corrections=paths)
    for seed in args.seeds:
        for arm in arms:
            label = f"{arm}_s{seed}"
            session.adapter.load_modules(
                task_path=a.out / "checkpoints" / label / "task.pt", device=session.policy.config.device
            )
            run_trials(session, "test", args, "visual", label=label)
    factorial_report(a.out, a.baseline, args.seeds, arms)


if __name__ == "__main__":
    main()
