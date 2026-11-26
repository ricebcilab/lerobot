"""Clean-operator visual-T pilot. See the LIBERO README for the four stages."""

import argparse
import fcntl
import hashlib
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from reproducibility import TrialSpec
from session import Session
from visual_task_scene import DESTINATIONS, PilotScene

from lerobot.envs import preprocess_observation
from lerobot.policies.steering import AdapterModules, LearnedTaskModule
from lerobot.utils.io_utils import write_video

PROMPT = "do something"
BOWL_PROMPT = "place the black bowl somewhere"
AVOID_PROMPT = "place the black bowl somewhere, avoiding obstacles along the way"
SPLITS = {"train": (0, 20), "dev": (20, 10), "test": (30, 20)}


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def trials(session, split, count, repeats=1):
    offset, size = SPLITS[split]
    for task in DESTINATIONS:
        counts = {task: session.initial_state_count("libero_goal", task)}
        if counts[task] < 50:
            raise ValueError(f"Task {task} needs at least 50 distinct states")
        specs = TrialSpec.for_schedule([task] * min(count, size), "libero_goal", 0, counts, offset)
        for i, spec in enumerate(specs):
            for rep in range(repeats):
                # Balance the intended goal and the route on which the mark is placed.
                route = (1, 8)[(i + rep) % 2]
                seed = int(np.random.SeedSequence([spec.env_seed, rep]).generate_state(1)[0])
                yield spec, route, seed


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def episode(
    session,
    scene,
    spec,
    route,
    seed,
    arm,
    record_demo=False,
    video=None,
    prompt=PROMPT,
    *,
    trace=None,
    replay=None,
    after_step=None,
):
    obs = scene.reset(spec, seed, route)
    scene.start_pick_trace()
    state_hash = session.initial_state_hash()
    steering = session.steering_policy()
    steering.reset()
    session.chain.reset()
    session.chain.readout.rng = np.random.default_rng(spec.input_noise_seed)
    session.operator.reset()
    device = next(session.policy.parameters()).device
    horizon = session.settings.n_action_steps
    pending = deque()
    actions, images, states, frames, replans = [], [], [], [], []
    success = False
    start = 0
    if replay is not None:
        prefix, expected_hash = replay
        start = len(prefix)
        for action in prefix:
            obs, done, _ = scene.step(action)
            if done or scene.violation:
                raise ValueError("Recovery replay encountered contact or termination before takeover")
        if start % horizon or session.initial_state_hash() != expected_hash:
            raise ValueError("Recovery takeover failed to reproduce the recorded replan state")
    for step in range(start, scene.env._max_episode_steps):
        if trace is not None and step % horizon == 0:
            trace["hashes"][step] = session.initial_state_hash()
        if video is not None:
            frames.append(
                np.concatenate([obs["pixels"][k][0, ::-1, ::-1] for k in ("image", "image2")], axis=1)
            )
        if record_demo and step % horizon == 0:
            inputs = session.env_preprocessor(preprocess_observation(obs))
            images.append(np.stack([obs["pixels"][k][0, ::-1, ::-1] for k in ("image", "image2")]))
            states.append(inputs["observation.state"][0].cpu().numpy())
        if arm == "teacher":
            if not pending:
                torch.manual_seed((spec.operator_seed + step) % 2**32)
                pending.extend(
                    session.env_actions(session.policy_chunk(obs, session.task_description))[:horizon]
                )
            action = scene.teacher_action(pending.popleft())
        elif arm == "base":
            if not pending:
                torch.manual_seed((spec.policy_seed + step // horizon) % 2**32)
                batch = session.policy_batch(obs, prompt)
                sync(device)
                started = time.perf_counter()
                with torch.inference_mode(), session.autocast_ctx:
                    chunk = session.policy.predict_action_chunk(batch)
                sync(device)
                replans.append(1000 * (time.perf_counter() - started))
                pending.extend(session.env_actions(chunk[:, :horizon]))
            action = pending.popleft()
        else:
            new_chunk = step % horizon == 0
            batch = session.policy_batch(obs, prompt)
            if new_chunk:
                session.chain.advance(step)
                torch.manual_seed((spec.operator_seed + step // horizon) % 2**32)
                session._refresh_operator(obs, step=step)
                batch["command"] = session.chunk_command()
                if arm == "mismatch":
                    # Counterfactual view of this exact pose with the obstacle on the other route.
                    anchor = scene.position(DESTINATIONS[8 if route == 1 else 1])
                    other = scene.render_at(
                        scene.alternate if scene.obstacle == "holder" else (scene.start[:2] + anchor[:2]) / 2
                    )
                    try:
                        _, patches = session.policy.camera_field(
                            session.policy_batch(other, prompt), stage=session.adapter.task.camera_stage
                        )
                        batch["adapter_camera_tokens"] = patches
                    finally:
                        scene.render_at(scene.center)
                torch.manual_seed((spec.policy_seed + step // horizon) % 2**32)
                sync(device)
                started = time.perf_counter()
            with torch.inference_mode(), session.autocast_ctx:
                normalized = steering.select_action(batch)
            if new_chunk:
                sync(device)
                replans.append(1000 * (time.perf_counter() - started))
            action = (
                session.env_postprocessor({"action": session.postprocessor(normalized)})["action"][0]
                .cpu()
                .numpy()
            )
        actions.append(action)
        if arm not in ("base", "teacher"):
            session.operator.observe(action)
        obs, done, success = scene.step(action)
        scene.track_picks()
        if after_step is not None:
            after_step(action)
        if done or ((trace is not None or replay is not None) and scene.violation):
            break
    if trace is not None:
        trace["actions"] = actions
    if video is not None:
        frames.append(np.concatenate([obs["pixels"][k][0, ::-1, ::-1] for k in ("image", "image2")], axis=1))
        write_video(video, frames, fps=20)
    row = dict(
        **spec.metadata(),
        **scene.pick_result(),
        prompt=session.task_description if arm == "teacher" else prompt,
        layout_seed=seed,
        route=route,
        center=scene.center.tolist(),
        initial_state_hash=state_hash,
        arm=arm,
        blocked=bool(scene.blocked),
        success=success,
        violation=scene.violation,
        first_contact=scene.first_contact,
        joint=success and not scene.violation,
        steps=len(actions),
        replan_ms=float(np.mean(replans)) if replans else None,
    )
    demo = (
        dict(
            state=states,
            action=actions,
            stride=horizon,
            **{f"frame_{i}": pair for i, pair in enumerate(images)},
        )
        if record_demo and row["joint"]
        else None
    )
    return row, demo


def run_trials(session, split, args, arm, *, collect=False, label=None, prompt=None):
    label = label or arm
    rows = []
    path = args.out / split / label
    path.mkdir(parents=True, exist_ok=True)
    accepted = {(task, route): 0 for task in DESTINATIONS for route in DESTINATIONS}
    current_task = scene = None
    repeats = args.demos if collect else 1
    # For collection interleave resets before repeating a layout family.
    cases = list(trials(session, split, args.states, repeats))
    if collect:
        cases = [case for rep in range(repeats) for case in cases[rep::repeats]]
    for spec, route, seed in cases:
        cell = (spec.task_id, route)
        if collect and accepted[cell] >= args.demos // 4:
            continue
        name = f"{spec.task_id}-{spec.init_state_id}-{seed}"
        result = path / f"{name}.json"
        demo_path = path / f"{name}.npz"
        if result.exists():
            row = json.loads(result.read_text())
            if args.stage == "audit" and "picked" not in row:
                raise ValueError(f"{result} has no pickup trace; use a separate audit output directory")
            if collect and row["joint"] and not demo_path.exists():
                raise ValueError(f"Missing accepted demonstration {demo_path}")
        else:
            if current_task != spec.task_id:
                session.set_scene("libero_goal", spec.task_id)
                scene = PilotScene(session.vec_env.envs[0], args.obstacle)
                current_task = spec.task_id
            have_video = any(
                r["task_id"] == spec.task_id and (args.obstacle == "disk" or r["route"] == route)
                for r in rows
            )
            video = None if have_video else path / f"{name}.mp4"
            row, demo = episode(session, scene, spec, route, seed, arm, collect, video, prompt or args.prompt)
            if demo is not None:
                np.savez_compressed(demo_path, **demo)
            save_json(result, row)
        rows.append(row)
        accepted[cell] += int(row["joint"])
        print(f"{split} {label} {name} placement={row['success']} violation={row['violation']}", flush=True)
        if collect and sum(accepted.values()) == args.demos:
            break
    if collect and sum(accepted.values()) < args.demos:
        raise RuntimeError(f"Teacher collected fewer clean demonstrations than requested: {accepted}")
    return rows


def feasibility(rows):
    for task in DESTINATIONS:
        selected = [r for r in rows if r["task_id"] == task]
        if not any(r["joint"] and r["blocked"] and r["arm"] == "teacher" for r in selected):
            return False
        if not any(r["joint"] and not r["blocked"] and r["arm"] == "frs" for r in selected):
            return False
    return any(r["blocked"] and r["violation"] and r["arm"] == "frs" for r in rows)


def probe(session, args):
    teacher = run_trials(session, "dev", args, "teacher")
    candidates = []
    for depth, prefix in [(2, 5), (4, 10), (10, 10)]:
        session.n_reversal_steps, session.pinned_steps = depth, prefix
        rows = run_trials(session, "dev", args, "frs", label=f"frs_d{depth}_p{prefix}")
        candidates.append((np.mean([r["joint"] for r in rows]), depth, prefix, rows))
    eligible = [candidate for candidate in candidates if feasibility(teacher + candidate[3])]
    score, depth, prefix, rows = max(eligible or candidates, key=lambda x: x[0])
    result = {"depth": depth, "prefix": prefix, "joint": score, "passed": feasibility(teacher + rows)}
    save_json(args.out / "development.json", result)
    print(result)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "probe",
            "collect",
            "train",
            "evaluate",
            "report",
            "audit",
            "audit-report",
            "prompt-check",
            "camera-pilot",
        ],
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--reference", type=Path, help="source run for audit-report or camera-pilot")
    p.add_argument("--obstacle", choices=["disk", "holder"], default="disk")
    p.add_argument(
        "--states", type=int, default=50, help="cap the number of initial states per split and task"
    )
    p.add_argument("--prompt")
    p.add_argument("--demos", type=int, default=100)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    if args.stage == "camera-pilot":
        args.obstacle = "holder"
        args.prompt = args.prompt or AVOID_PROMPT
        args.reference = args.reference or Path("outputs/pi05_libero_utensil_holder")
        args.out = args.out or Path("outputs/pi05_libero_holder_camera_stages")
        if args.out.resolve() == args.reference.resolve():
            p.error("camera-pilot needs a separate output directory")
    if args.out is None:
        folder = "utensil_holder" if args.obstacle == "holder" else "visual_task"
        if args.stage in ("audit", "audit-report"):
            folder += "_pick_audit"
        args.out = Path("outputs") / f"pi05_libero_{folder}"
    if args.prompt is None:
        args.prompt = BOWL_PROMPT if args.obstacle == "holder" else PROMPT
    if min(args.states, args.demos, args.steps, args.batch) < 1 or args.demos % 4:
        p.error("positive counts required; demos must be a multiple of four")
    if min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        p.error("training seeds must be distinct and nonnegative")
    return args


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args):
    if args.stage == "audit-report":
        from visual_task_report import audit_report

        return audit_report(args.out, args.reference)
    if args.stage == "report":
        from visual_task_report import camera_report, report

        manifest = json.loads((args.out / "manifest.json").read_text())
        if "demo_sha256" in manifest:
            return camera_report(args.out, manifest["seeds"])
        return report(args.out, args.seeds)
    settings = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {})
    # Resolve once: the model and processors load from the exact same snapshot.
    manifest = args.out / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    camera_pilot = args.stage == "camera-pilot"
    reference = json.loads((args.reference / "manifest.json").read_text()) if camera_pilot else {}
    if camera_pilot and (
        reference.get("obstacle") != "holder" or reference["policy"] != settings.session.policy_path
    ):
        raise ValueError("Expected utensil-holder demonstrations from the configured policy")
    snapshot = Path(
        snapshot_download(
            settings.session.policy_path,
            revision=reference.get("revision") or previous.get("revision"),
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    if not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(snapshot / "model.safetensors")
    provenance = {
        "policy": settings.session.policy_path,
        "revision": snapshot.name,
        "split_seed": 0,
        "states": args.states,
    }
    if args.obstacle != "disk":
        provenance["obstacle"] = args.obstacle
    if args.obstacle == "holder" or args.prompt != PROMPT:
        provenance["prompt"] = args.prompt
    if camera_pilot:
        demo_dir = args.reference / "train/teacher"
        files = sorted(demo_dir.glob("*.npz"))
        if len(files) != args.demos:
            raise ValueError(f"Expected {args.demos} holder demonstrations, found {len(files)}")
        digest = hashlib.sha256()
        for file in files:
            with file.open("rb") as stream:
                digest.update(file.name.encode() + hashlib.file_digest(stream, "sha256").digest())
        development = json.loads((args.reference / "development.json").read_text())
        provenance.update(
            demo_sha256=digest.hexdigest(),
            reference=str(args.reference.resolve()),
            seeds=args.seeds,
            steps=args.steps,
            batch=args.batch,
            depth=development["depth"],
            prefix=development["prefix"],
        )
    if previous and previous != provenance:
        raise ValueError("Output directory belongs to a different checkpoint or split; use another --out")
    save_json(manifest, provenance)
    if args.stage == "audit":
        settings.session.control.n_reversal_steps, settings.session.control.pinned_steps = 4, 10
    elif args.stage != "probe":
        dev = development if camera_pilot else json.loads((args.out / "development.json").read_text())
        if not dev["passed"] and args.stage != "prompt-check":
            raise RuntimeError("Development feasibility check failed; inspect dev videos before training")
        settings.session.control.n_reversal_steps = dev["depth"]
        settings.session.control.pinned_steps = dev["prefix"]
    session = Session(replace(settings.session, policy_path=str(snapshot), headless=True))
    session.policy.requires_grad_(False).eval()
    try:
        if args.stage == "audit":
            for arm in ("frs", "base"):
                run_trials(session, "test", args, arm)
        elif args.stage == "prompt-check":
            # Fix these from the earlier disk pilot, independently of either prompt's holder outcomes.
            session.n_reversal_steps, session.pinned_steps = 4, 10
            for label, prompt in (("generic", PROMPT), ("bowl", BOWL_PROMPT)):
                for arm in ("frs", "base"):
                    name = f"{arm}_{label}"
                    if arm == "frs" and prompt == args.prompt:
                        name = "frs_d4_p10"
                    run_trials(session, "dev", args, arm, label=name, prompt=prompt)
        elif args.stage == "probe":
            probe(session, args)
        elif args.stage == "collect":
            run_trials(session, "train", args, "teacher", collect=True)
        elif camera_pilot:
            train(session, args, provenance, demo_dir=demo_dir, arms=("pre", "post"))
            run_trials(session, "test", args, "frs")
            for seed in args.seeds:
                for stage in ("pre", "post"):
                    label = f"{stage}_s{seed}"
                    path = args.out / "checkpoints" / label / "task.pt"
                    session.adapter.load_modules(task_path=path, device=session.policy.config.device)
                    for arm in ("visual", "mismatch"):
                        run_trials(session, "test", args, arm, label=f"{label}_{arm}")
            from visual_task_report import camera_report

            camera_report(args.out, args.seeds)
        elif args.stage == "train":
            train(session, args, provenance)
        else:
            evaluate(session, args)
    finally:
        session.close()


class DemoChunks:
    """Uniform sampling of recorded replan states with their subsequent executed actions."""

    def __init__(self, paths, horizon, prompt=PROMPT):
        self.prompt = prompt
        self.rows = []
        self.actions = {}
        self.states = {}
        self.horizon = horizon
        for path in paths:
            with np.load(path) as data:
                self.actions[path] = data["action"].astype(np.float32)
                self.states[path] = data["state"]
                stride = int(data["stride"])
                self.rows.extend((path, i, i * stride) for i in range(len(self.states[path])))

    def batch(self, rng, size, correction=None):
        rows = [
            self.rows[i] for i in rng.integers(len(self.rows), size=size if correction is None else size - 1)
        ]
        sources = [(self, row) for row in rows]
        if correction is not None:
            sources.append((correction, correction.rows[rng.integers(len(correction.rows))]))
        images, states, actions, masks = [], [], [], []
        for source, (path, i, start) in sources:
            with np.load(path) as data:
                images.append(data[f"frame_{i}"])
            states.append(source.states[path][i])
            chunk = source.actions[path][start : start + self.horizon]
            actions.append(np.pad(chunk, ((0, self.horizon - len(chunk)), (0, 0)), mode="edge"))
            masks.append(np.arange(self.horizon) < len(chunk))
        cams = torch.from_numpy(np.stack(images)).permute(1, 0, 4, 2, 3).float() / 255
        return {
            "observation.images.image": cams[0],
            "observation.images.image2": cams[1],
            "observation.state": torch.from_numpy(np.stack(states)),
            "action": torch.from_numpy(np.stack(actions)),
            "task": [self.prompt] * size,
        }, torch.from_numpy(np.stack(masks))


def train(
    session, args, provenance, *, demo_dir=None, arms=("visual", "masked"), projection=None, corrections=()
):
    cfg = session.policy.config
    paths = sorted((demo_dir or args.out / "train/teacher").glob("*.npz"))
    if len(paths) != args.demos:
        raise ValueError(f"Expected {args.demos} accepted demos, found {len(paths)}")
    dataset = DemoChunks(paths, cfg.chunk_size, args.prompt)
    extra = DemoChunks(corrections, cfg.chunk_size, args.prompt) if corrections else None
    for seed in args.seeds:
        for arm in arms:
            directory = args.out / "checkpoints" / f"{arm}_s{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            task = LearnedTaskModule(
                horizon=cfg.chunk_size,
                n_dims=3,
                d=64,
                heads=4,
                layers=1,
                mask_scene=arm == "masked",
                camera_stage="post" if arm in ("post", "geometry", "mixed", "combined") else "pre",
                gripper_projection=projection if arm in ("geometry", "combined") else None,
                **LearnedTaskModule.spec("pi05_cameras"),
            ).to(cfg.device)
            optim = torch.optim.AdamW(task.parameters(), lr=3e-4)
            meta = dict(
                provenance,
                train_seed=seed,
                steps=args.steps,
                batch=args.batch,
                demos=len(paths),
                parameters=sum(p.numel() for p in task.parameters()),
            )
            if arm in ("geometry", "mixed", "combined"):
                uses_recoveries = arm in ("mixed", "combined")
                if (uses_recoveries and not corrections) or (arm != "mixed" and projection is None):
                    raise ValueError(f"Missing recovery demonstrations or calibration for {arm}")
                meta.update(
                    condition=arm,
                    original_per_batch=args.batch - int(uses_recoveries),
                    recovery_per_batch=int(uses_recoveries),
                )
            path = directory / "task.pt"
            if path.exists():
                if AdapterModules.read_checkpoint(path, mmap=True)["meta"] != meta:
                    raise ValueError(
                        f"Different training settings at {path}; select a fresh output directory"
                    )
                continue
            for step in range(args.steps):
                batch, valid = dataset.batch(rng, args.batch, extra if arm in ("mixed", "combined") else None)
                xyz = batch["observation.state"][:, :3]
                batch = session.preprocessor(batch)
                actions = session.policy.prepare_action(batch)
                model = session.policy.model
                noise = model.sample_noise(actions.shape, actions.device)
                times = model.sample_time(len(actions), actions.device)
                velocity, patches = session.policy.camera_field(batch, stage=task.camera_stage)
                loss = task.pi05_field_loss(
                    velocity, patches, actions, noise, times, valid.to(actions.device), 0.01, xyz
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0, error_if_nonfinite=True)
                optim.step()
                if step % 100 == 0:
                    print(f"{arm} seed={seed} step={step} loss={loss.item():.5f}", flush=True)
            tmp = AdapterModules.save_module(task.cpu(), directory / "task.tmp", "task", meta)
            tmp.replace(path)


def evaluate(session, args):
    from visual_task_report import report

    run_trials(session, "test", args, "frs")
    if args.obstacle == "holder":
        run_trials(session, "test", args, "base")
    for seed in args.seeds:
        for arm in ("masked", "visual", "mismatch"):
            source = "visual" if arm == "mismatch" else arm
            path = args.out / "checkpoints" / f"{source}_s{seed}" / "task.pt"
            session.adapter.load_modules(task_path=path, device=session.policy.config.device)
            run_trials(session, "test", args, arm, label=f"{arm}_s{seed}")
    report(args.out, args.seeds)


if __name__ == "__main__":
    main()
