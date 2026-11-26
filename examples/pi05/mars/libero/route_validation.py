"""Run matched frozen-policy and FRS controls for both obstructed bowl routes."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from paired_obstacles import REVISION, PairedGoalScene
from reproducibility import TrialSpec
from session import Session
from train_visual_task import AVOID_PROMPT, episode, save_json
from visual_task_scene import BOWL

SOURCE = Path("outputs/pi05_libero_two_obstacle_pilot")
OUT = Path("outputs/pi05_libero_route_validation")
HEAD = SOURCE / "checkpoints/C/task.pt"


class ValidationScene(PairedGoalScene):
    clear = False

    @property
    def blocked(self):
        return not self.clear and super().blocked

    def reset(self, spec, layout_seed, route):
        obs = super().reset(spec, layout_seed, route)
        if self.clear:
            sim = self.domain.sim
            for name in ("mars_marker", "mars_pepper_mill"):
                idx = sim.model.body_mocapid[sim.model.body_name2id(name)]
                sim.data.mocap_pos[idx] = [0, 0, -2]
            sim.forward()
            obs = self.observation(self.domain._get_observations(force_update=True))
        assert next(self.obstacle_contacts(), None) is None
        self.image_hash = hashlib.sha256(
            b"".join(obs["pixels"][k].tobytes() for k in ("image", "image2"))
        ).hexdigest()
        return obs


def replay(scene, case, actions):
    scene.reset(*case)
    for a in actions:
        _, done, success = scene.step(a)
        if done:
            break
    return {
        "steps": scene.control_step,
        "success": success,
        "joint": success and not scene.violation,
        "contact": scene.first_contact,
    }


def run_episode(s, scene, case, arm, instruction):
    actions, poses = [], []

    def record(a):
        actions.append(a.copy())
        grip = scene.domain._check_grasp(
            scene.domain.robots[0].gripper, scene.domain.objects_dict[BOWL].contact_geoms
        )
        poses.append(
            [
                *scene.domain.sim.data.site_xpos[scene.domain.robots[0].eef_site_id],
                *scene.position(BOWL),
                float(grip),
            ]
        )

    spec, seed, route = case
    row, _ = episode(
        s,
        scene,
        spec,
        route,
        seed,
        "base" if arm in "AB" else "frs",
        prompt=instruction if arm in "AB" else AVOID_PROMPT,
        after_step=record,
    )
    return row, actions, poses


def run(s, manifest):
    s.set_scene("libero_goal", 1)
    scene = ValidationScene(s.vec_env.envs[0], "two_obstacles")
    s.adapter.load_modules(task_path=HEAD, device=s.policy.config.device)
    task = s.adapter.task
    rows = []
    for i, item in enumerate(manifest["schedule"]):
        layout = []
        for goal in (1, 8):
            case = (replace(TrialSpec(**item["spec"]), task_id=goal), item["seed"], item["holder"])
            instruction = "put the bowl on the " + ("stove" if goal == 1 else "plate")
            pair = {}
            for arm in "ABCD":
                path = OUT / f"{i}-{goal}-{arm}.json"
                scene.clear = arm == "A"
                s.adapter.task = task if arm == "D" else None
                assert s.adapter.user is None and s.adapter.matrix is None
                if not path.exists():
                    row, actions, poses = run_episode(s, scene, case, arm, instruction)
                    assert s.task_description == instruction
                    row.update(
                        condition=arm,
                        layout=i,
                        goal=goal,
                        image_hash=scene.image_hash,
                        centers={str(k): v.tolist() for k, v in scene.centers.items()},
                    )
                    np.savez_compressed(path.with_suffix(".npz"), actions=actions, gripper_bowl_grasp=poses)
                    save_json(path, row)
                pair[arm] = json.loads(path.read_text())
            if pair["A"]["joint"] and pair["B"]["joint"]:
                path = OUT / f"{i}-{goal}-replay.json"
                if not path.exists():
                    actions = np.load(OUT / f"{i}-{goal}-A.npz")["actions"]
                    scene.clear = True
                    check = replay(scene, case, actions)
                    assert check["joint"] and check["steps"] == len(actions)
                    scene.clear = False
                    save_json(path, replay(scene, case, actions))
            layout.extend(pair.values())
        assert len({r["initial_state_hash"] for r in layout}) == 1
        assert len({str(r["centers"]) for r in layout}) == 1
        assert len({r["image_hash"] for r in layout if r["condition"] == "A"}) == 1
        assert len({r["image_hash"] for r in layout if r["condition"] != "A"}) == 1
        assert layout[0]["image_hash"] != layout[1]["image_hash"]
        rows.extend(layout)
        print("Finished layout", i, flush=True)
    return rows


def summarize(rows):
    def counts(group):
        return {k: sum(bool(r[k]) for r in group) for k in ("success", "violation", "joint")}

    report = {}
    for arm in "ABCD":
        group = [r for r in rows if r["condition"] == arm]
        layout_wins = Counter(r["layout"] for r in group if r["joint"])
        report[arm] = dict(
            **counts(group),
            both_goals=sum(v == 2 for v in layout_wins.values()),
            by_goal={g: counts([r for r in group if r["goal"] == g]) for g in (1, 8)},
            contacts=dict(
                Counter(
                    f"{c['obstacle']}/{c['link']}/grasped={c['grasped']}"
                    for r in group
                    if (c := r["first_contact"])
                )
            ),
        )
        if arm != "A":
            report[arm]["by_obstacle"] = {
                name: counts([r for r in group if (r["route"] == r["goal"]) == holder])
                for name, holder in (("utensil_holder", True), ("pepper_mill", False))
            }
    cases = {(r["layout"], r["goal"], r["condition"]): r for r in rows}
    transitions = Counter()
    gains, losses = [], []
    for i, goal, arm in cases:
        if arm != "A":
            continue
        a, b, c, d = [cases[i, goal, key] for key in "ABCD"]
        if a["success"]:
            transitions[
                "joint" if b["joint"] else "placed_with_contact" if b["success"] else "failed_placement"
            ] += 1
        if c["joint"] != d["joint"]:
            (gains if d["joint"] else losses).append([i, goal])
    report.update(clear_success_to_blocked=dict(transitions), T_gains=gains, T_losses=losses)
    replays = [json.loads(p.read_text()) for p in OUT.glob("*-replay.json")]
    report["replays"] = {
        "total": len(replays),
        "joint": sum(r["joint"] for r in replays),
        "contact": sum(r["contact"] is not None for r in replays),
    }
    return report


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
    cfg.control.n_reversal_steps, cfg.control.pinned_steps = 4, 10
    manifest = {
        "schedule": json.loads((SOURCE / "manifest.json").read_text())["schedule"],
        "policy_revision": REVISION,
        "task_sha256": hashlib.sha256(HEAD.read_bytes()).hexdigest(),
        "prompt": AVOID_PROMPT,
    }
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    save_json(path, manifest)
    snapshot = snapshot_download(cfg.policy_path, revision=REVISION, local_files_only=True)
    s = Session(replace(cfg, policy_path=snapshot, headless=True, n_action_steps=10))
    try:
        s.policy.requires_grad_(False).eval()
        rows = run(s, manifest)
        save_json(OUT / "results.json", rows)
        save_json(OUT / "summary.json", summarize(rows))
    finally:
        s.close()
