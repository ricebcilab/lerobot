#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run a shared-autonomy user study on LIBERO with pi0.5.

Reads an experiment description from YAML (see configs/experiment/), then runs
`n_trials` trials. Each trial draws a task from the configured scene, shows
*you* the task to accomplish, and lets you attempt it with the VLA in the
shared-autonomy mode from the config. The VLA itself only ever receives the
config's `prompt` (e.g. "do something"), so the language instruction and the
human intent can be dissociated on purpose.

    ./run.sh experiment --config <condition>.yaml [--set KEY=VALUE ...] [--sweep KEY=V1,V2,...] [--dry-run]

`--config` is looked up as given, then under configs/experiment/, so a bare
condition name works from anywhere.

`--set` overrides one YAML key (dotted path, YAML value); `--sweep` runs one block of
`n_trials` per listed value with a single policy load. Every block is written to
`<output_dir>/<timestamp>_<config stem>[_<key>-<value>...]/`:

    config.yaml     the resolved configuration, schedule and the matrices in force
    trials.jsonl    one JSON record per completed trial
    trial_XXX.npz   per-step arrays (operator input, executed action, robot state)
    trial_XXX.mp4   the rollout video
"""

import argparse
import datetime as dt
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from config import (
    CONFIG_DIR,
    MODES,
    PROMPT_FROM_TASK,
    ExperimentSettings,
    load_experiment_settings,
    set_label,
)
from session import VIDEO_FPS, Session
from teleop import TeleopChain

from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import init_logging

EXPERIMENT_CONFIG_DIR = CONFIG_DIR / "experiment"


def build_schedule(task_ids: list[int], n_trials: int, order: str, seed: int) -> list[int]:
    """The task shown in each trial, decided up front so a run is reproducible."""
    rng = random.Random(seed)
    if order == "sequential":
        return [task_ids[i % len(task_ids)] for i in range(n_trials)]
    if order == "random":
        return [rng.choice(task_ids) for _ in range(n_trials)]
    schedule: list[int] = []  # shuffled: permute, cycling through the pool
    while len(schedule) < n_trials:
        block = list(task_ids)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:n_trials]


class TrialRecorder:
    """Collects one trial's per-step data and writes it as a .npz."""

    def __init__(self, chain: TeleopChain):
        self._chain = chain
        self._reads = chain.served.reads
        self.reads_at_start = chain.served.reads  # the counter is session-wide, so baseline it
        self.rows: list[dict] = []
        self.start = time.time()

    @property
    def total_reads(self) -> int:
        """How many times the teleop input was sampled during *this* trial."""
        return self._chain.served.reads - self.reads_at_start

    def __call__(self, step, observation, action, reward, terminated, truncated, info):
        state = observation.get("robot_state", {}) if isinstance(observation, dict) else {}
        eef = state.get("eef", {})
        gripper = state.get("gripper", {})
        joints = state.get("joints", {})
        reads = self._chain.served.reads
        self.rows.append(
            {
                "t": time.time() - self.start,
                "action": np.asarray(action, dtype=np.float32)[0],
                "user_translation": np.asarray(self._chain.served.last_translation, dtype=np.float32),
                "user_translation_raw": np.asarray(self._chain.raw.last_translation, dtype=np.float32),
                "user_gripper": float(self._chain.served.last_gripper),
                "user_reads": reads - self._reads,  # 0 = the input was not consulted this step
                "eef_pos": _first(eef.get("pos"), 3),
                "eef_quat": _first(eef.get("quat"), 4),
                "gripper_qpos": _first(gripper.get("qpos"), 2),
                "joint_pos": _first(joints.get("pos"), 7),
                "reward": float(np.asarray(reward).reshape(-1)[0]),
                "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
                "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
            }
        )
        self._reads = reads

    def save(self, path: Path, **scalars) -> None:
        arrays = {}
        if self.rows:
            for key in self.rows[0]:
                arrays[key] = np.stack([np.asarray(row[key]) for row in self.rows])
        np.savez_compressed(path, **arrays, **{k: np.asarray(v) for k, v in scalars.items()})


def _first(value, size: int) -> np.ndarray:
    """First sub-env's vector from a batched observation entry (NaNs if absent)."""
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array[0]
    return array.reshape(-1)[:size]


def prompt_for(prompt_setting: str, task_description: str) -> str:
    return task_description if prompt_setting == PROMPT_FROM_TASK else prompt_setting


@dataclass
class Block:
    """One run: a condition file plus the `--set` items in force (a sweep expands to several)."""

    name: str
    sets: list[str]


def expand_blocks(stem: str, sets: list[str], sweep: str | None) -> list[Block]:
    """The blocks a command line describes: one per sweep value, or a single block without --sweep."""
    if sweep is None:
        return [Block(stem + (f"_{set_label(sets)}" if sets else ""), list(sets))]
    key, sep, values = sweep.partition("=")
    values = [v.strip() for v in values.split(",") if v.strip()]
    if not sep or not key.strip() or not values:
        raise ValueError(f"--sweep expects KEY=v1,v2,... such as control.n_guided_steps=2,4,8, got {sweep!r}")
    blocks = []
    for value in values:
        block_sets = [*sets, f"{key.strip()}={value}"]
        blocks.append(Block(f"{stem}_{set_label(block_sets)}", block_sets))
    return blocks


def parse_args() -> tuple[argparse.Namespace, list[Block]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH", help="condition YAML (bare name = shipped)"
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one YAML key by dotted path, value parsed as YAML (repeatable), "
        "e.g. --set control.n_guided_steps=4 --set control.corruption=null",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        metavar="KEY=V1,V2,...",
        help="run one block of n_trials per value of KEY, in this process with a single policy load, "
        "e.g. --sweep control.n_reversal_steps=2,4,6,8,10",
    )
    parser.add_argument(
        "--n-trials", dest="n_trials", type=int, default=None, help="= --set experiment.n_trials"
    )
    parser.add_argument("--seed", type=int, default=None, help="= --set experiment.seed")
    parser.add_argument("--mode", default=None, choices=MODES, help="= --set control.mode")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="= --set experiment.output_dir")
    parser.add_argument("--port", type=int, default=None, help="= --set server.port")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the config, print each block's schedule and exit"
    )
    args = parser.parse_args()
    if not Path(args.config).exists() and (EXPERIMENT_CONFIG_DIR / args.config).exists():
        args.config = str(EXPERIMENT_CONFIG_DIR / args.config)  # a bare condition name
    if not Path(args.config).exists():
        found = sorted(p.name for p in EXPERIMENT_CONFIG_DIR.glob("*.yaml") if p.name != "base.yaml")
        parser.error(
            f"--config: {args.config} not found. Available in {EXPERIMENT_CONFIG_DIR}: {', '.join(found)}"
        )
    try:
        blocks = expand_blocks(Path(args.config).stem, args.sets, args.sweep)
    except ValueError as e:
        parser.error(str(e))
    return args, blocks


def _load_block(args: argparse.Namespace, block: Block) -> ExperimentSettings:
    overrides = {k: getattr(args, k) for k in ("n_trials", "seed", "mode", "output_dir", "port")}
    settings = load_experiment_settings(args.config, overrides, sets=block.sets)
    settings.name = block.name
    return settings


def _resolve_task_ids(settings: ExperimentSettings, where: str) -> list[int]:
    n_tasks = len(Session.list_tasks(settings.session.suite))  # also validates the suite name
    if settings.task_ids is None:
        return list(range(n_tasks))
    for task_id in settings.task_ids:
        if not 0 <= task_id < n_tasks:
            raise ValueError(
                f"{where}: scene.task_ids: {task_id!r} is not a task of {settings.session.suite}"
            )
    return settings.task_ids


def _provenance(
    settings: ExperimentSettings, block: Block, schedule: list[int], session: Session | None
) -> dict:
    control = settings.session.control
    resolved = {
        "name": settings.name,
        "overrides": list(block.sets),
        "n_trials": settings.n_trials,
        "seed": settings.seed,
        "task_order": settings.task_order,
        "output_dir": str(settings.output_dir),
        "suite": settings.session.suite,
        "task_ids": settings.task_ids,
        "prompt": settings.prompt,
        "policy_path": settings.session.policy_path,
        "n_action_steps": settings.session.n_action_steps,
        "compile": settings.session.compile,
        "mode": control.mode,
        "n_guided_steps": control.n_guided_steps,
        "n_reversal_steps": control.n_reversal_steps,
        "input_noise": control.input_noise,
        "max_steps": control.max_steps,
        "corruption": control.corruption,
        "reversal_adapter": control.reversal_adapter,
        "port": settings.session.port,
        "schedule": schedule,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if session is not None:
        resolved.update(session.resolved_matrices())
    return resolved


_FIXED_PER_PROCESS = ("policy_path", "n_action_steps", "compile", "port")


def _print_block_header(index: int, total: int, settings: ExperimentSettings, session: Session) -> None:
    print("\n" + "#" * 78)
    print(
        f"BLOCK {index + 1}/{total}: {settings.name}: {settings.n_trials} trials, mode {session.mode_label()}"
    )
    print(
        f'VLA prompt: "{settings.prompt}"'
        + ("  (the scene's own instruction)" if settings.prompt == PROMPT_FROM_TASK else "")
    )
    if session.chain.corruption.matrix is not None:
        print(session.chain.corruption.describe())
    if session.adapter.matrix is not None:
        print(session.adapter.describe())
    text = session.announce_mode()
    if text:
        print(text)
    print("#" * 78)


def run_block(
    session: Session, settings: ExperimentSettings, schedule: list[int], run_dir: Path
) -> list[dict]:
    """Run one block's trials in `session`, writing trials.jsonl and per-trial files under run_dir.

    Returns the trial records. Raises StopIteration-free: `q` ends the block early and returns
    what was recorded; a KeyboardInterrupt propagates after the partial results are on disk.
    """
    control = settings.session.control
    mode = session.mode
    results: list[dict] = []
    trials_path = run_dir / "trials.jsonl"
    for trial, task_id in enumerate(schedule):
        if (settings.session.suite, task_id) != (session.suite, session.task_id):
            session.set_scene(settings.session.suite, task_id)  # a new scene needs its own env + processors
        task_description = session.task_description
        vla_prompt = prompt_for(settings.prompt, task_description)

        print("\n" + "=" * 78)
        print(f"TRIAL {trial + 1}/{settings.n_trials}   {settings.session.suite} task {task_id}")
        print(f"YOUR TASK: {task_description}")
        print(f'VLA prompt: "{vla_prompt}"   mode: {session.mode_label()}')
        print("=" * 78)
        session.show_scene()
        # `task` stays up for the whole trial; rollout only overwrites `prompt`,
        # which is already the VLA's, so nothing visibly changes when it starts.
        session.view.stream.set_status(
            task=f"TRIAL {trial + 1}/{settings.n_trials}: {task_description}",
            prompt=vla_prompt,
            state="get ready",
            mode=session.mode_label(),
        )
        try:
            answer = input("Press Enter to start (s = skip this trial, q = end this block): ")
        except EOFError:
            answer = "q"
        if answer.strip().lower() == "q":
            print("Ending the block early.")
            break
        if answer.strip().lower() == "s":
            print("Skipped.")
            continue

        recorder = TrialRecorder(session.chain)
        started = time.time()
        result = session.rollout(vla_prompt, recorder=recorder, max_steps=control.max_steps)
        duration = time.time() - started

        video_name, steps_name = f"trial_{trial:03d}.mp4", f"trial_{trial:03d}.npz"
        write_video(run_dir / video_name, result.frames, fps=VIDEO_FPS)
        recorder.save(
            run_dir / steps_name,
            task_description=task_description,
            vla_prompt=vla_prompt,
            success=result.success,
            mode=mode,
            task_id=task_id,
        )
        record = {
            "trial": trial,
            "suite": settings.session.suite,
            "task_id": task_id,
            "task_description": task_description,
            "vla_prompt": vla_prompt,
            "mode": mode,
            "n_guided_steps": session.n_guided_steps if mode == "shared_flow_control" else None,
            "input_noise": control.input_noise,
            "corruption": session.chain.corruption.label,
            "reversal_adapter": session.adapter.label,
            "success": bool(result.success),
            "steps": int(result.steps),
            "duration_s": round(duration, 2),
            "user_reads": recorder.total_reads,
            "video": video_name,
            "steps_file": steps_name,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            **result.metrics,
            "n_reversal_steps": (
                result.metrics.get("n_reversal_steps") if mode == "shared_flow_reversal_steering" else None
            ),
        }
        results.append(record)
        with open(trials_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"{'SUCCESS' if result.success else 'no success'} after {result.steps} steps ({duration:.0f}s)")
        stats = session.stats_line()
        if stats:
            print(stats)
        print(f"saved {steps_name} + {video_name}")
    if results:
        wins = sum(r["success"] for r in results)
        print(
            f"\n{settings.name}: {len(results)} trials, {wins} success ({wins / len(results):.0%}), "
            f"mean {np.mean([r['steps'] for r in results]):.0f} steps"
        )
    print(f"Data: {run_dir}")
    return results


def main():
    args, blocks = parse_args()
    try:
        plan = []
        for block in blocks:
            settings = _load_block(args, block)
            settings.task_ids = _resolve_task_ids(settings, args.config)
            schedule = build_schedule(
                settings.task_ids, settings.n_trials, settings.task_order, settings.seed
            )
            plan.append((block, settings, schedule))
    except (OSError, ValueError) as e:
        raise SystemExit(f"--config: {e}") from e
    first = plan[0][1].session
    for _, settings, _ in plan[1:]:
        for key in _FIXED_PER_PROCESS:
            if getattr(settings.session, key) != getattr(first, key):
                raise SystemExit(f"--sweep/--set cannot change {key} between blocks (one policy per process)")

    if args.dry_run:
        for _block, settings, schedule in plan:
            control = settings.session.control
            tasks = Session.list_tasks(settings.session.suite)
            print(
                f"Config OK ({args.config}). Block {settings.name}: schedule ({settings.task_order}, seed {settings.seed}):"
            )
            for i, task_id in enumerate(schedule):
                print(f"  trial {i:03d}: {settings.session.suite} task {task_id}: {tasks[task_id]}")
            print(
                f'VLA prompt: "{settings.prompt}"   mode: {control.mode}   '
                f"n_guided_steps: {control.n_guided_steps}   n_reversal_steps: {control.n_reversal_steps}"
            )
            if control.corruption_matrix is not None:
                print(f"corruption M = {np.array2string(control.corruption_matrix, precision=3)}")
            if control.reversal_adapter_matrix is not None:
                print(f"reversal adapter F = {np.array2string(control.reversal_adapter_matrix, precision=3)}")
            print()
        return

    init_logging()
    first_settings = plan[0][1]
    first_settings.session.task_id = plan[0][2][0]
    session = Session(first_settings.session)
    all_results: dict[str, list[dict]] = {}
    try:
        for index, (block, settings, schedule) in enumerate(plan):
            session.apply_control(settings.session.control)
            run_dir = settings.output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{settings.name}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(_provenance(settings, block, schedule, session), sort_keys=False)
            )
            _print_block_header(index, len(plan), settings, session)
            all_results[settings.name] = run_block(session, settings, schedule, run_dir)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        session.close()

    if len(all_results) > 1:
        print("\nSummary:")
        for name, results in all_results.items():
            wins = sum(r["success"] for r in results)
            rate = f"{wins}/{len(results)} = {wins / len(results):.0%}" if results else "no trials"
            print(f"  {name}: {rate}")


if __name__ == "__main__":
    main()
