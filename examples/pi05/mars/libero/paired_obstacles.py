"""Evaluate two possible intentions on identical two-obstacle LIBERO scenes."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from config import CONFIG_DIR, load_experiment_settings
from huggingface_hub import snapshot_download
from PIL import Image
from reproducibility import TrialSpec
from session import Session
from train_visual_task import AVOID_PROMPT, episode, save_json
from visual_task_scene import BOWL, BOWL_RADIUS, DESTINATIONS, RADIUS, PilotScene, segment_distance

CHECKPOINT = Path("outputs/pi05_libero_holder_interventions/checkpoints/C/task.pt")
REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"


class PairedGoalScene(PilotScene):
    def reset(self, spec, layout_seed, route):
        obs = super().reset(spec, layout_seed, route)
        # Use the same physical task/reset for both operator intentions. LIBERO
        # uses this predicate for both its reward and its termination check.
        target = {1: "flat_stove_1_cook_region", 8: "plate_1"}[spec.task_id]
        self.domain.parsed_problem["goal_state"] = [["on", BOWL, target]]
        self.env.task_description = "put the bowl on the " + ("stove" if spec.task_id == 1 else "plate")
        for dest, center in self.centers.items():
            if (
                segment_distance(self.start[:2], self.position(DESTINATIONS[dest])[:2], center)
                > RADIUS + BOWL_RADIUS
            ):
                raise RuntimeError("Obstacle missed its assigned direct route")
        self.external_view = obs["pixels"]["image"][0, ::-1, ::-1].copy()
        self.image_hash = hashlib.sha256(
            b"".join(obs["pixels"][k].tobytes() for k in ("image", "image2"))
        ).hexdigest()
        return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts", type=int, default=2, help="Even number, half with each obstacle assignment")
    p.add_argument("--out", type=Path, default=Path("outputs/pi05_libero_paired_obstacles"))
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    a = p.parse_args()
    if a.layouts < 2 or a.layouts % 2 or a.layouts > 20:
        p.error("--layouts must be even and between 2 and 20")
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_settings(CONFIG_DIR / "visual_task.yaml", {}).session
    snapshot = snapshot_download(cfg.policy_path, revision=REVISION, local_files_only=True)
    cfg.control.n_reversal_steps, cfg.control.pinned_steps = 4, 10
    s = Session(replace(cfg, policy_path=snapshot, n_action_steps=10, headless=True))
    try:
        s.set_scene("libero_goal", 1)
        scene = PairedGoalScene(s.vec_env.envs[0], "two_obstacles")
        s.adapter.load_modules(task_path=a.checkpoint, device=s.policy.config.device)
        specs = TrialSpec.for_schedule(
            [1] * a.layouts, "libero_goal", 0, {1: s.initial_state_count("libero_goal", 1)}, 30
        )
        for i, spec in enumerate(specs):
            holder = (1, 8)[i % 2]
            seed = int(np.random.SeedSequence([spec.env_seed, 671]).generate_state(1)[0])
            pair = []
            for goal in DESTINATIONS:
                row, _ = episode(
                    s, scene, replace(spec, task_id=goal), holder, seed, "visual", prompt=AVOID_PROMPT
                )
                row.update(
                    image_hash=scene.image_hash,
                    physical_task=1,
                    holder_route=holder,
                    operator_prompt=s.task_description,
                    obstacles={
                        ("utensil_holder" if d == holder else "pepper_mill"): c.tolist()
                        for d, c in scene.centers.items()
                    },
                    checkpoint=str(a.checkpoint),
                    policy_revision=REVISION,
                )
                pair.append(row)
            for key in ("initial_state_hash", "image_hash", "obstacles"):
                if pair[0][key] != pair[1][key]:
                    raise RuntimeError(f"Goal pair has different {key}")
            Image.fromarray(scene.external_view).save(a.out / f"layout_{i}.png")
            save_json(a.out / f"layout_{i}.json", pair)
    finally:
        s.close()


if __name__ == "__main__":
    main()
