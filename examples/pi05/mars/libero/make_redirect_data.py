#!/usr/bin/env python
"""Redirect episodes for the prompt-dropout fine-tune: demonstrations that leave one task's path for another's.

The fine-tuned policy chooses its goal in the first chunk and is then committed by where the robot is: a state on
the way to one goal occurs only in that goal's demonstrations, so hiding the prompt leaves nothing to choose there,
and a committed policy cannot be steered (nor, mostly, told in words) onto another goal. Redirect episodes put the
missing demonstrations in: the policy runs under task A's instruction for `switch` steps, then under task B's to
the end, in B's scene. Only what follows the switch is kept, labelled with B's instruction, and only when the
benchmark's check for B passes: from a state on A's path, the way to B. Fine-tuned on them with prompt dropout
(`train_generic_prompt.py`), the policy keeps more than one goal open along the way.

Start states: the benchmark's initial states the study does not use (its development and test trials take each
task's first `--held-out` states of the seeded permutation in `reproducibility.trial_specs`).

Usage (from the repo root, through run.sh's environment; one process per shard, each on its own GPU and port):
    examples/pi05/mars/libero/run.sh python examples/pi05/mars/libero/make_redirect_data.py \
        --config examples/pi05/mars/libero/configs/experiment/study_goal.yaml --set policy.path=<checkpoint> \
        --out outputs/pi05_libero_redirect/shard_0 --shard 0 6
Writes a LeRobot dataset (the training dataset's features) under --out and attempts.json beside it; merge the
shards and the training dataset with `lerobot.datasets.dataset_tools.merge_datasets`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

FEATURES = {  # HuggingFaceVLA/libero's, what the policy was trained on
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}
FPS = 10  # the training dataset's
IMAGE_SIZE = 256  # the training dataset's images; the simulator renders 360


@dataclasses.dataclass(frozen=True)
class Attempt:
    """One redirect: `first` task's instruction for `switch` steps, then `meant` task's, from start state `start`."""

    first: int
    meant: int
    switch: int
    start: int


def study_states(task: int, n_states: int, seed: int, n: int, suite: str = "libero_goal") -> list[int]:
    """The initial states the study's first `n` trials of `task` use (`reproducibility.trial_specs`)."""
    from reproducibility import trial_specs

    return [s.init_state_id for s in trial_specs([task] * n, suite, seed=seed, state_counts={task: n_states})]


def redirect_plan(
    tasks, switches, n_starts: int, n_states: int, seed: int, held_out: int, suite: str = "libero_goal"
) -> list[Attempt]:
    """Every ordered pair of `tasks` at every switch step, from the meant task's first `n_starts` initial states
    the study does not use, grouped by the meant task (one scene at a time)."""
    plan = []
    for meant in sorted(tasks):
        used = set(study_states(meant, n_states, seed, held_out, suite))
        starts = [s for s in range(n_states) if s not in used][:n_starts]
        for first in sorted(tasks):
            if first != meant:
                plan += [Attempt(first, meant, switch, start) for switch in switches for start in starts]
    return plan


def shard(plan: list[Attempt], index: int, count: int) -> list[Attempt]:
    """Shard `index` of `count`: every count-th attempt, so each shard covers every pair."""
    return plan[index::count]


def dataset_frame(processed: dict, action, task: str, size: int = IMAGE_SIZE) -> dict:
    """One training frame from what the policy consumes (`env_preprocessor`'s output, batch of 1: images
    (1, 3, H, W) in [0, 1], state (1, 8)) and the executed action (1, 7) in the dataset's units. The simulator
    renders larger images than the dataset holds; they are resized to the dataset's `size`."""
    frame = {"task": task}
    for key in ("observation.images.image", "observation.images.image2"):
        image = processed[key].float().clamp(0, 1)
        if image.shape[-2:] != (size, size):
            image = torch.nn.functional.interpolate(
                image, size=(size, size), mode="bilinear", antialias=True, align_corners=False
            ).clamp(0, 1)
        frame[key] = np.round(image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frame["observation.state"] = processed["observation.state"][0].float().cpu().numpy().astype(np.float32)
    frame["action"] = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
    return frame


def main(args) -> None:
    from config import load_experiment_settings
    from session import Session
    from train_user_module import calibration_control

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.utils import preprocess_observation

    settings = load_experiment_settings(args.config, {}, args.sets)
    session = Session(
        dataclasses.replace(settings.session, control=calibration_control(settings.session.control))
    )
    suite = settings.session.suite
    session.set_mode("policy")
    prompts = {}
    for task in args.tasks:
        session.set_scene(suite, task)
        prompts[task] = session.task_description
    n_states = min(Session.initial_state_count(suite, task) for task in args.tasks)
    plan = shard(
        redirect_plan(args.tasks, args.switch_steps, args.starts, n_states, args.seed, args.held_out, suite),
        *args.shard,
    )
    out = Path(args.out)
    dataset = LeRobotDataset.create(
        repo_id=f"local/{out.name}",
        fps=FPS,
        features=FEATURES,
        root=out,
        robot_type="panda",
        use_videos=False,
    )
    rng = np.random.default_rng([args.seed, args.shard[0]])
    log, last, scene = [], {}, None
    for i, attempt in enumerate(plan):
        if attempt.meant != scene:
            session.set_scene(suite, attempt.meant)
            scene, env_step = attempt.meant, session.vec_env.step

            def remember(action, _step=env_step):  # the observation the next rollout starts from
                result = _step(action)
                last["observation"], last["ended"] = result[0], bool(result[2][0] or result[3][0])
                return result

            session.vec_env.step = remember
        torch.manual_seed(int(rng.integers(2**31)))
        observation, _ = session.vec_env.reset(
            seed=int(rng.integers(2**31)), options={"init_state_id": attempt.start}
        )
        session.rollout(prompts[attempt.first], initial_observation=observation, max_steps=attempt.switch)
        frames, success, steps = [], False, attempt.switch
        if not last["ended"]:  # still under way: now the meant task's instruction, recorded

            def recorder(observation, action, _frames=frames, _task=prompts[attempt.meant], **_):
                processed = session.env_preprocessor(preprocess_observation(observation))
                _frames.append(dataset_frame(processed, action, _task))

            result = session.rollout(
                prompts[attempt.meant],
                recorder=recorder,
                initial_observation=last["observation"],
                max_steps=args.max_steps - attempt.switch,
            )
            success, steps = bool(result.success), attempt.switch + result.steps
        if success:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
        log.append({**dataclasses.asdict(attempt), "success": success, "steps": steps, "frames": len(frames)})
        kept = sum(entry["success"] for entry in log)
        print(f"{i + 1}/{len(plan)} {log[-1]}  kept {kept}", flush=True)
        (out.parent / f"{out.name}_attempts.json").write_text(
            json.dumps({"prompts": prompts, "log": log}, indent=1)
        )
    dataset.finalize()
    session.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="a study file: the policy, the suite and the prompt")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", required=True, help="the dataset's folder (must not exist)")
    p.add_argument(
        "--tasks", type=int, nargs="+", default=[1, 4, 5, 6, 7, 8], help="the tasks to redirect among"
    )
    p.add_argument("--switch-steps", type=int, nargs="+", default=[10, 20, 30, 40, 60, 80])
    p.add_argument("--starts", type=int, default=8, help="initial states per meant task")
    p.add_argument(
        "--held-out", type=int, default=25, help="each task's first states of the study's, left out"
    )
    p.add_argument("--max-steps", type=int, default=300, help="episode length, the switch included")
    p.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("INDEX", "COUNT"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not 0 <= args.shard[0] < args.shard[1]:
        p.error("--shard INDEX COUNT needs 0 <= INDEX < COUNT")
    return args


if __name__ == "__main__":
    main(parse_args())
