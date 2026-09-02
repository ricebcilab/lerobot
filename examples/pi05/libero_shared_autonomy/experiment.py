#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run a shared-autonomy user study on LIBERO with pi0.5.

Reads an experiment description from YAML (see config_experiment.yaml), then
runs `n_trials` trials. Each trial draws a task from the configured scene,
shows *you* the task to accomplish, and lets you attempt it with the VLA in the
shared-autonomy mode from the config -- the VLA itself only ever receives the
config's `prompt` (e.g. "do something"), so the language instruction and the
human intent can be dissociated on purpose.

    ./examples/pi05_libero/experiment.sh [--config config_experiment.yaml] [flags]

Every trial is written to `<output_dir>/<timestamp>_<name>/`:

    config.yaml     the resolved configuration, including the matrices in use
    trials.jsonl    one JSON record per trial (task, prompt, outcome, summary)
    trial_XXX.npz   per-step arrays (user input, executed action, robot state)
    trial_XXX.mp4   the rollout video

Teleop input, the live view and the control modes are exactly those of
interactive.py; see README_experiment.md.
"""

import argparse
import datetime as dt
import json
import random
import threading
import time
from contextlib import nullcontext
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml
from interactive import (
    MODES,
    VIDEO_FPS,
    FlowAdapter,
    FrameStream,
    ModeRunner,
    announce_mode,
    build_env,
    close_envs,
    flatten_config,
    list_tasks,
    make_handler,
    mode_display,
    read_yaml_config,
    run_rollout,
    write_video,
)
from teleop import KeyboardReader, TeleopChain, build_corruption, read_matrix_spec

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.steering import build_reversal_adapter
from lerobot.utils.utils import init_logging

DEFAULT_EXPERIMENT_CONFIG_FILE = Path(__file__).resolve().parent / "config_experiment.yaml"
TASK_ORDERS = ("random", "shuffled", "sequential")
PROMPT_FROM_TASK = "task"  # `prompt: task` gives the VLA the scene's own instruction

# yaml key -> flat config key
EXPERIMENT_CONFIG_KEYS = {
    "experiment": {
        "name": "name",
        "n_trials": "n_trials",
        "seed": "seed",
        "task_order": "task_order",
        "output_dir": "output_dir",
    },
    "scene": {"suite": "suite", "task_ids": "task_ids"},
    "prompt": "prompt",
    "policy": {"path": "policy_path", "n_action_steps": "n_action_steps", "compile": "compile"},
    "control": {
        "mode": "mode",
        "tau": "tau",
        "n_reverse_steps": "n_reverse_steps",
        "input_noise": "input_noise",
        "deterministic_corruption": "deterministic_corruption",
        "flow_reversal_adapter": "flow_reversal_adapter",
        "max_steps": "max_steps",
    },
    "server": {"port": "port"},
}

EXPERIMENT_DEFAULTS = {
    "name": "experiment",
    "n_trials": 10,
    "seed": 0,
    "task_order": "random",
    "output_dir": "outputs/pi05_libero_experiments",
    "suite": "libero_spatial",
    "task_ids": None,  # None = every task in the suite
    "prompt": "do something",
    "policy_path": "lerobot/pi05_libero_finetuned",
    "n_action_steps": 10,
    "compile": False,
    "mode": "shared_reverse_flow_steering",
    "tau": 5,
    "n_reverse_steps": None,  # None = reverse all the way to noise
    "input_noise": 0.0,
    "deterministic_corruption": None,  # off unless a path is given
    "flow_reversal_adapter": None,  # off unless a path is given
    "max_steps": None,  # None = the suite's own episode length
    "port": 8765,
}


def load_experiment_config(path: str | Path) -> dict:
    """Read + validate config_experiment.yaml into a flat dict of settings."""
    flat = {**EXPERIMENT_DEFAULTS}
    flat.update(flatten_config(read_yaml_config(path), EXPERIMENT_CONFIG_KEYS, str(path)))
    return validate_experiment_config(flat, str(path))


def validate_experiment_config(cfg: dict, where: str) -> dict:
    """Check ranges and enums so a typo fails before the model is loaded."""

    def fail(message):
        raise ValueError(f"{where}: {message}")

    if not isinstance(cfg["n_trials"], int) or cfg["n_trials"] < 1:
        fail(f"experiment.n_trials must be a positive integer, got {cfg['n_trials']!r}")
    if cfg["mode"] not in MODES:
        fail(f"control.mode must be one of {', '.join(MODES)}, got {cfg['mode']!r}")
    if cfg["task_order"] not in TASK_ORDERS:
        fail(f"experiment.task_order must be one of {', '.join(TASK_ORDERS)}, got {cfg['task_order']!r}")
    if cfg["input_noise"] < 0:
        fail("control.input_noise must be >= 0")
    if not isinstance(cfg["tau"], int) or cfg["tau"] < 0:
        fail("control.tau must be a non-negative integer")
    if cfg["n_reverse_steps"] is not None and (
        not isinstance(cfg["n_reverse_steps"], int) or cfg["n_reverse_steps"] < 1
    ):
        fail("control.n_reverse_steps must be a positive integer or null (null = all the way to noise)")
    if cfg["max_steps"] is not None and (not isinstance(cfg["max_steps"], int) or cfg["max_steps"] < 1):
        fail("control.max_steps must be a positive integer or null")
    if not isinstance(cfg["prompt"], str) or not cfg["prompt"].strip():
        fail("prompt must be a non-empty string (or `task` to use the scene's own instruction)")
    n_tasks = len(list_tasks(cfg["suite"]))  # also validates the suite name
    if cfg["task_ids"] is None:
        cfg["task_ids"] = list(range(n_tasks))
    if not isinstance(cfg["task_ids"], list) or not cfg["task_ids"]:
        fail("scene.task_ids must be a non-empty list of task ids (or null for all of them)")
    for task_id in cfg["task_ids"]:
        if not isinstance(task_id, int) or not 0 <= task_id < n_tasks:
            fail(f"scene.task_ids: {task_id!r} is not a task of {cfg['suite']} (0..{n_tasks - 1})")
    return cfg


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


def prompt_for(cfg: dict, task_description: str) -> str:
    return task_description if cfg["prompt"] == PROMPT_FROM_TASK else cfg["prompt"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EXPERIMENT_CONFIG_FILE),
        metavar="PATH",
        help=f"experiment YAML (default: {DEFAULT_EXPERIMENT_CONFIG_FILE.name} next to this script)",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="override experiment.n_trials")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument("--mode", default=None, choices=MODES, help="override control.mode")
    parser.add_argument("--output-dir", default=None, help="override experiment.output_dir")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the config, print the trial schedule and exit (no model, no env)",
    )
    args = parser.parse_args()
    if not Path(args.config).exists():
        found = sorted(p.name for p in DEFAULT_EXPERIMENT_CONFIG_FILE.parent.glob("config_experiment*.yaml"))
        hint = f" Available next to experiment.py: {', '.join(found)}." if found else ""
        parser.error(f"--config: {args.config} not found.{hint}")
    try:
        cfg = load_experiment_config(args.config)
    except (OSError, ValueError) as e:
        parser.error(f"--config: {e}")
    for key in ("n_trials", "seed", "mode", "output_dir", "port"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    return args, validate_experiment_config(cfg, str(args.config))


def main():
    args, cfg = parse_args()
    schedule = build_schedule(cfg["task_ids"], cfg["n_trials"], cfg["task_order"], cfg["seed"])
    tasks = list_tasks(cfg["suite"])

    if args.dry_run:
        print(f"Config OK ({args.config}).\nTrial schedule ({cfg['task_order']}, seed {cfg['seed']}):")
        for i, task_id in enumerate(schedule):
            print(f"  trial {i:03d}: {cfg['suite']} task {task_id} — {tasks[task_id]}")
        print(f'\nVLA prompt: "{cfg["prompt"]}"   mode: {cfg["mode"]}')
        return

    init_logging()
    run_dir = Path(cfg["output_dir"]) / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{cfg['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- teleop input
    stream = FrameStream()
    keyboard = KeyboardReader()
    chain = TeleopChain(keyboard, input_noise=cfg["input_noise"])
    flow_adapter = FlowAdapter()
    if cfg["deterministic_corruption"]:
        path = Path(cfg["deterministic_corruption"])
        chain.corruption.matrix = build_corruption(read_matrix_spec(path))
        chain.corruption.label = path.name
    if cfg["flow_reversal_adapter"]:
        path = Path(cfg["flow_reversal_adapter"])
        flow_adapter.matrix = build_reversal_adapter(read_matrix_spec(path))
        flow_adapter.label = path.name
    server = ThreadingHTTPServer(
        ("127.0.0.1", cfg["port"]),
        make_handler(stream, keyboard, chain.noisy, chain.corruption, flow_adapter),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\nLive view: http://localhost:{cfg['port']}  (VSCode should auto-forward the port)\n")

    # ---------------------------------------------------------------- policy
    print(f"Loading policy {cfg['policy_path']} ...")
    policy_cfg = PreTrainedConfig.from_pretrained(cfg["policy_path"])
    policy_cfg.pretrained_path = cfg["policy_path"]
    policy_cfg.n_action_steps = cfg["n_action_steps"]
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = cfg["compile"]

    env_cfg, envs_dict, vec_env = build_env(cfg["suite"], schedule[0])
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    autocast_ctx = (
        torch.autocast(device_type=torch.device(policy.config.device).type)
        if policy_cfg.use_amp
        else nullcontext()
    )
    runner = ModeRunner(
        policy,
        chain.reader,
        preprocessor,
        postprocessor,
        env_preprocessor,
        env_postprocessor,
        flow_adapter=flow_adapter,
    )

    # ---------------------------------------------------------------- provenance
    resolved = {
        **cfg,
        "schedule": schedule,
        "corruption_matrix": None if chain.corruption.matrix is None else chain.corruption.matrix.tolist(),
        "flow_adapter_matrix": None if flow_adapter.matrix is None else flow_adapter.matrix.tolist(),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    mode, tau, n_reverse = cfg["mode"], cfg["tau"], cfg["n_reverse_steps"]
    runner.n_reverse_steps = n_reverse
    print(f"\nExperiment {cfg['name']}: {cfg['n_trials']} trials, mode {mode}, output {run_dir}")
    print(
        f'VLA prompt: "{cfg["prompt"]}"'
        + ("  (the scene's own instruction)" if cfg["prompt"] == PROMPT_FROM_TASK else "")
    )
    if chain.corruption.matrix is not None:
        print(chain.corruption.describe())
    if flow_adapter.matrix is not None:
        print(flow_adapter.describe())
    if mode != "policy":
        chain.attach_spacemouse()
    announce_mode(mode, tau, policy, flow_adapter, n_reverse)

    results = []
    trials_path = run_dir / "trials.jsonl"
    try:
        current_task_id = schedule[0]
        for trial, task_id in enumerate(schedule):
            if task_id != current_task_id:  # a new scene needs its own env + processors
                close_envs(envs_dict)
                env_cfg, envs_dict, vec_env = build_env(cfg["suite"], task_id)
                runner.env_preprocessor, runner.env_postprocessor = make_env_pre_post_processors(
                    env_cfg=env_cfg, policy_cfg=policy_cfg
                )
                current_task_id = task_id
            task_description = vec_env.envs[0].task_description
            vla_prompt = prompt_for(cfg, task_description)

            print("\n" + "=" * 78)
            print(f"TRIAL {trial + 1}/{cfg['n_trials']}   {cfg['suite']} task {task_id}")
            print(f"YOUR TASK: {task_description}")
            print(f'VLA prompt: "{vla_prompt}"   mode: {mode}')
            print("=" * 78)
            vec_env.reset()
            stream.publish(vec_env.envs[0].render())
            # `task` stays up for the whole trial; run_rollout only overwrites `prompt`,
            # which is already the VLA's, so nothing visibly changes when the rollout starts
            stream.set_status(
                task=f"TRIAL {trial + 1}/{cfg['n_trials']}: {task_description}",
                prompt=vla_prompt,
                state="get ready",
                mode=mode,
            )
            try:
                answer = input("Press Enter to start (s = skip this trial, q = end the experiment): ")
            except EOFError:
                answer = "q"
            if answer.strip().lower() == "q":
                print("Ending the experiment early.")
                break
            if answer.strip().lower() == "s":
                print("Skipped.")
                continue

            recorder = TrialRecorder(chain)
            mode_label = mode_display(mode, tau, n_reverse, policy.config.num_inference_steps)
            stream.set_status(mode=mode_label)
            started = time.time()
            success, steps, frames = run_rollout(
                vec_env,
                prompt=vla_prompt,
                stream=stream,
                autocast_ctx=autocast_ctx,
                recorder=recorder,
                **runner.kwargs(mode, tau, n_reverse),
                **({"max_steps_override": cfg["max_steps"]} if cfg["max_steps"] else {}),
            )
            duration = time.time() - started

            video_name = f"trial_{trial:03d}.mp4"
            steps_name = f"trial_{trial:03d}.npz"
            write_video(run_dir / video_name, frames, fps=VIDEO_FPS)
            recorder.save(
                run_dir / steps_name,
                task_description=task_description,
                vla_prompt=vla_prompt,
                success=success,
                mode=mode,
                task_id=task_id,
            )
            record = {
                "trial": trial,
                "suite": cfg["suite"],
                "task_id": task_id,
                "task_description": task_description,
                "vla_prompt": vla_prompt,
                "mode": mode,
                "tau": tau if mode == "shared_flow_control" else None,
                "n_reverse_steps": (
                    (n_reverse or policy.config.num_inference_steps)
                    if mode == "shared_reverse_flow_steering"
                    else None
                ),
                "input_noise": cfg["input_noise"],
                "deterministic_corruption": chain.corruption.label,
                "flow_reversal_adapter": flow_adapter.label,
                "success": bool(success),
                "steps": int(steps),
                "duration_s": round(duration, 2),
                "user_reads": recorder.total_reads,
                "video": video_name,
                "steps_file": steps_name,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                **runner.metrics(mode),
            }
            results.append(record)
            with open(trials_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            stats = runner.stats_line(mode)
            print(f"{'SUCCESS' if success else 'no success'} after {steps} steps ({duration:.0f}s)")
            if stats:
                print(stats)
            print(f"saved {steps_name} + {video_name}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        close_envs(envs_dict)
        server.shutdown()

    if results:
        wins = sum(r["success"] for r in results)
        print(
            f"\n{len(results)} trials — {wins} success ({wins / len(results):.0%}), "
            f"mean {np.mean([r['steps'] for r in results]):.0f} steps"
        )
    print(f"Data: {run_dir}")


if __name__ == "__main__":
    main()
