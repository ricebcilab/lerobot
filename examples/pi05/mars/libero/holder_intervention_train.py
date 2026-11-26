"""Matched corrective-data and executed-prefix continuations after the XYZ rescue screen."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from holder_factorial import digest
from holder_interventions import H, chunk, reset
from torch.utils.checkpoint import checkpoint
from train_visual_task import AVOID_PROMPT, DemoChunks, episode, save_json, trials
from visual_task_scene import PilotScene

from lerobot.policies.steering import AdapterModules
from lerobot.policies.steering.modeling_steering import Flow


def prefix(s, batch, data):
    """Runtime's partial reversal, delegation and regeneration, with a gradient to T."""
    steer = s.steering_policy()
    core = steer.core
    velocity, patches = s.policy.camera_field(batch, stage="post")
    field = s.adapter.field({"scene": {"pi05_cameras": patches}})

    # Recompute frozen action expert activations in backward for its input Jacobian.
    def velocity_step(x, time):
        return checkpoint(lambda z: velocity(z, time), x, use_reentrant=False)

    flow = Flow(velocity_step, field)
    core._gripper_ref = float(data["gripper_ref"])
    noise = torch.as_tensor(data["noise"], device=s.policy.config.device)[None]
    cmd = data["command"][None]
    steps = s.policy.config.num_inference_steps
    if core._active(cmd).item():
        ref = core.reference_chunk(cmd).to(noise)
        x = flow.latent(ref, noise, steps, core.n_reversal_steps, core._delegated, grad=True)
        start, count = Flow.schedule(steps, core.n_reversal_steps)
        x = flow.integrate(x, start, -start / count, count, grad=True)
    else:
        x = flow.forward(noise, steps)
    return x[:, :H, :3]


def check_runtime(s, out):
    """Compare runtime and differentiable prefixes on an actual steered training observation."""
    case = next(c for c in trials(s, "train", 20) if c[0].task_id == c[1] == 8)
    s.set_scene("libero_goal", 8)
    scene = PilotScene(s.vec_env.envs[0], "holder")
    obs = reset(s, scene, case)
    for step in range(0, scene.env._max_episode_steps, H):
        actions, record = chunk(s, obs, case[0], step, True)
        if s.steering_policy().core._active(record["command"][None]).item():
            break
        for a in actions:
            obs, done, _ = scene.step(a)
            if done:
                raise RuntimeError("No active FRS command in smoke episode")
    batch = s.policy_batch(obs, AVOID_PROMPT)
    task = s.adapter.task
    task.requires_grad_(True)
    with s.autocast_ctx:
        pred = prefix(s, batch, record)
        ref = torch.as_tensor(record["normalized"][:, :3], device=pred.device)[None]
        error = float((pred.detach() - ref).abs().max())
        torch.testing.assert_close(pred.detach(), ref, atol=0.02, rtol=0.01)
        grad = torch.autograd.grad(pred.square().mean(), task.out.weight)[0]
    if not grad.isfinite().all() or grad.norm() == 0 or any(p.requires_grad for p in s.policy.parameters()):
        raise RuntimeError("Invalid adapter-only gradient")
    save_json(
        out / "prefix_check.json", {"max_abs_error": error, "T_output_weight_grad_norm": grad.norm().item()}
    )


def fit(s, out, arm, original, corrections, preserves, risk=None):
    args = json.loads((out / "manifest.json").read_text())
    device = s.policy.config.device
    s.adapter.load_modules(task_path=args["checkpoint"], device=device)
    task = s.adapter.task
    task.requires_grad_(True)
    opt = torch.optim.AdamW(task.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    cfg = s.policy.config
    path = out / "checkpoints" / arm / "task.pt"
    meta = dict(
        args,
        arm=arm,
        steps=1000,
        batch=2,
        lr=3e-4,
        seed=0,
        prefix_loss_weight=int(arm in ("B", "C")),
        collections_sha256=digest(corrections + preserves),
    )
    if risk is not None:
        meta["risk_weight"] = 0.1
    if path.exists():
        if AdapterModules.read_checkpoint(path, mmap=True)["meta"] != meta:
            raise ValueError(f"Checkpoint provenance mismatch at {path}")
        return path
    for step in range(1000):
        pool = preserves if step % 2 else corrections
        file = pool[rng.integers(len(pool))]
        pair = DemoChunks([file], cfg.chunk_size, AVOID_PROMPT)
        batch, mask = original.batch(rng, 2, correction=pair)
        batch = s.preprocessor(batch)
        actions = s.policy.prepare_action(batch)
        noise = s.policy.model.sample_noise(actions.shape, device)
        times = s.policy.model.sample_time(2, device)
        vel, patches = s.policy.camera_field(batch, stage="post")
        fm = task.pi05_field_loss(vel, patches, actions, noise, times, mask.to(device), 0.01)
        opt.zero_grad(set_to_none=True)
        fm.backward()
        if arm in ("B", "C"):
            single = {
                k: v[1:] if torch.is_tensor(v) and v.ndim and len(v) == 2 else v for k, v in batch.items()
            }
            with np.load(file) as data, s.autocast_ctx:
                pred = prefix(s, single, data)
                if step == 0 and args.get("collector", args["checkpoint"]) == args["checkpoint"]:
                    expected = torch.as_tensor(data["normalized"][:, :3], device=device)
                    torch.testing.assert_close(pred.detach()[0], expected, atol=0.02, rtol=0.01)
                valid = mask[1, :H].to(device)
                loss = (pred[0, valid] - actions[1, :H, :3][valid]).square().mean()
                if risk is not None and file.name in risk[1] and valid.all():
                    risk_head, stored = risk
                    views, state = stored[file.name]
                    logits = risk_head(views[None].to(device), state[None].to(device), pred)
                    loss = loss + 0.1 * torch.nn.functional.softplus(logits).mean()
                if step == 0:
                    g = torch.autograd.grad(loss, task.out.weight, retain_graph=True)[0]
                    if not torch.isfinite(g).all() or g.norm() == 0:
                        raise RuntimeError("Executed prefix has no usable T gradient")
                loss.backward()
        torch.nn.utils.clip_grad_norm_(task.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            print(arm, step, float(fm.detach()), flush=True)
    if any(p.grad is not None for p in s.policy.parameters()):
        raise RuntimeError("Unexpected frozen-policy gradient")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = AdapterModules.save_module(task.cpu(), path.with_suffix(".tmp"), "task", meta)
    tmp.replace(path)
    return path


def evaluate(s, out, split, paths):
    rows = {}
    for arm, path in paths.items():
        s.adapter.load_modules(task_path=path, device=s.policy.config.device)
        directory = out / split / arm
        directory.mkdir(parents=True, exist_ok=True)
        task = scene = None
        rows[arm] = []
        for spec, route, seed in trials(s, split, 10, repeats=2):
            # Independent layout seeds; keep the same pairing for every arm.
            seed = int(np.random.SeedSequence([seed, 24017]).generate_state(1)[0])
            name = directory / f"{spec.task_id}-{spec.init_state_id}-{seed}.json"
            if not name.exists():
                if task != spec.task_id:
                    task = spec.task_id
                    s.set_scene("libero_goal", task)
                    scene = PilotScene(s.vec_env.envs[0], "holder")
                row, _ = episode(s, scene, spec, route, seed, "visual", prompt=AVOID_PROMPT)
                save_json(name, row)
            rows[arm].append(json.loads(name.read_text()))
    reference = rows["baseline"]
    summary = {}
    for arm, rr in rows.items():
        assert [r["initial_state_hash"] for r in rr] == [r["initial_state_hash"] for r in reference]
        summary[arm] = {}
        for task in (1, 8):
            for blocked in (True, False):
                pairs = [
                    (r, b)
                    for r, b in zip(rr, reference, strict=True)
                    if r["task_id"] == task and r["blocked"] == blocked
                ]
                assert len(pairs) == 10
                summary[arm][f"{task}_{'blocked' if blocked else 'clear'}"] = {
                    **{k: sum(r[k] for r, b in pairs) for k in ("success", "violation", "joint")},
                    "bowl_pickups": sum("akita_black_bowl_1" in r["picked"] for r, b in pairs),
                    "first_contact": dict(
                        Counter(
                            (
                                r["first_contact"]["link"]
                                + (" grasping" if r["first_contact"]["grasped"] else " open/free")
                            )
                            for r, b in pairs
                            if r["first_contact"]
                        )
                    ),
                    "gained": sum(r["joint"] and not b["joint"] for r, b in pairs),
                    "lost": sum(b["joint"] and not r["joint"] for r, b in pairs),
                }
    save_json(out / split / "summary.json", summary)
    return summary


def experiment(s, out):
    counts = json.loads((out / "collection.json").read_text())
    if counts["rescues"] < 32 or counts["plate_rescues"] < 16:
        raise ValueError("Both rescue quotas must be met before training")
    pairs, clear = [], []
    for f in sorted((out / "pairs").glob("*.npz")):
        r = json.loads(f.with_suffix(".json").read_text())
        if r["rescue"]:
            pairs.append(f)
        elif r["success"] and not r["blocked"] and not r["contact"]:
            clear.append(f)
    if not clear:
        raise ValueError("Missing successful clear-route preservation examples")
    m = json.loads((out / "manifest.json").read_text())
    original = DemoChunks(
        sorted((Path(m["base"]["reference"]) / "train/teacher").glob("*.npz")), 50, AVOID_PROMPT
    )
    paths = {"baseline": Path(m["checkpoint"])}
    for arm in ("A", "B"):
        paths[arm] = fit(s, out, arm, original, pairs, clear)
    result = evaluate(s, out, "dev", paths)

    def scores(arm):
        r = result[arm]
        return r["8_blocked"]["joint"], r["1_clear"]["success"] + r["8_clear"]["success"]

    base, place = scores("baseline")
    eligible = [arm for arm in ("A", "B") if scores(arm)[0] >= base + 2 and scores(arm)[1] >= place - 1]
    winner = max(eligible, key=scores) if eligible else None
    if winner is None:
        from holder_contact_probe import probe

        risk = probe(s, out)
        if risk is not None:
            risk[0].to(s.policy.config.device).requires_grad_(False)
            paths["C"] = fit(s, out, "C", original, pairs, clear, risk)
            result = evaluate(s, out, "dev", paths)
            if scores("C")[0] >= base + 2 and scores("C")[1] >= place - 1:
                winner = "C"
    save_json(out / "selection.json", {"selected": winner})
    if winner:
        evaluate(s, out, "test", {k: paths[k] for k in ("baseline", winner)})
