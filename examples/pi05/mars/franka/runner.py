#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run the MARS arms on the Franka: the composition root, the trial loop, and the command line.

The runner is the only module here that knows all the ports at once. `build` wires them from factories so a
test drives a fake station, a fake steering policy and a linear Jacobian without a checkpoint, c3po or an arm;
`main` passes the real ones and nothing heavy is imported before it runs. The pieces it wires live next door:
`batches.py` turns an observation into the policy's batch, `record.py` writes a trial's arrays and identity,
and `arms.py` is what `mars_fixed` adds to the base config.

A station tick is not an action. The checkpoint plans at `ACTION_RATE`, the station integrates at its own
`control_rate`, so one action is sent `ceil(control_rate / action_rate)` ticks in a row
(`actions.ticks_per_action`) and the operator's command is read once per action, not once per tick.

The trial is the LIBERO session's, in the Franka's units: the same `trials.jsonl` and `trial_XXX.npz` keys, and
a flat `config.yaml` whose `mode`, `tau`, `lambda` and `reversal_adapter` are the keys
`../libero/notebooks/analyze.py` reads. `mode` is therefore LIBERO's label (`shared_flow_control`,
`shared_flow_reversal_steering`, `policy`, `teleop`), and the arm itself is recorded separately as `arm`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import batches
import numpy as np
import yaml
from actions import clamp_joint_velocity, command_from_qdot, ticks_per_action
from arms import _with_fixed_adapter
from batches import CameraMap, camera_map, observation_batch, to_action
from config import MODE_MAP, ExperimentConfig
from interfaces import (
    ACTION_DIMS,
    ARM_DIMS,
    GRIPPER_DIM,
    ArmSpec,
    JacobianPort,
    RobotPort,
    StationManifest,
)
from jacobian import PandaJacobian
from record import (
    TrialRecorder,
    flat_settings,
    initial_state_hash,
    nan_matrix,
    pair_id_for,
    provenance,
    scene_digest,
)
from station import IMAGE_SIZE, C3poStation, FakeStation, action_dict, build_arm_spec

EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXAMPLE_DIR / "configs"
DEFAULT_STATS = EXAMPLE_DIR / "assets" / "droid" / "norm_stats.json"
OUTPUT_ROOT = Path("outputs") / "pi05_mars_franka"

DROID_RESET_Q = np.array(
    [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483]
)
"""The DROID reset pose: the arm's home, and the fake station's stored q. Used when no start pose is configured."""

ACTION_RATE = 15.0
"""The checkpoint's action rate in Hz: one chunk step is one action, and holds for the ticks below it."""

RESET_TOLERANCE_RAD = 0.05
"""How far a reset may sit from the configured start pose before a trial refuses to begin."""

FAKE_CONTROL_RATE = ACTION_RATE
"""The fake station's tick rate: ten actions a second, so a dry run's first ten lines are ten actions."""


# ---------------------------------------------------------------- the wired objects


class PolicyBundle(NamedTuple):
    """A steering policy and the two pipelines its checkpoint was loaded with."""

    policy: Any
    preprocessor: Any
    postprocessor: Any
    state_dim: int = batches.STATE_DIM


@dataclass(frozen=True)
class Wired:
    """Everything one run needs, built once and shared by every trial."""

    config: ExperimentConfig
    policy: Any  # None on the teleop arm, which has no policy
    preprocessor: Any
    postprocessor: Any
    station: RobotPort
    chain: Any
    jacobian: JacobianPort
    arm: ArmSpec
    manifest: StationManifest
    cameras: CameraMap
    now: Callable[[], dt.datetime]
    state_dim: int = batches.STATE_DIM


@dataclass
class TrialResult:
    """What a trial produced: its JSON record and the per-step rows behind its .npz."""

    record: dict
    rows: list[dict]


# ---------------------------------------------------------------- building


def build(
    config: ExperimentConfig,
    *,
    stats_path: str | Path | None,
    station_factory: Callable[[ExperimentConfig], RobotPort],
    policy_factory: Callable[[ExperimentConfig, str | Path | None, dict[str, str]], PolicyBundle],
    now: Callable[[], dt.datetime],
    headless: bool = False,
    headless_operator: bool | None = None,
    start_pose: np.ndarray | None = None,
    jacobian_factory: Callable[[], JacobianPort] | None = None,
) -> Wired:
    """Wire one run: station, arm, Jacobian, the policy's pipelines, the operator's chain and the camera map.

    The station is built first because the manifest names its arm and cameras; the policy cannot be loaded
    before the rename map exists. `mars_fixed` is then the one arm whose config changes here: its null
    adapter matrix is `J+ M J` at the start pose, which needs the Jacobian and therefore the arm. The teleop
    arm loads no policy at all, since its action is the operator's command.
    """
    import readout_chain

    station = station_factory(config)
    manifest = station.manifest
    joint_names = getattr(getattr(station, "robot", None), "joint_names", None)
    arm = build_arm_spec(manifest, config.arm_overrides(), joint_names=joint_names)
    jacobian = (jacobian_factory or PandaJacobian)()
    start = DROID_RESET_Q if start_pose is None else np.asarray(start_pose, dtype=np.float64)
    if config.steering.mode == "mars_fixed":
        config = _with_fixed_adapter(config, jacobian, start)
    config.check_adapters()
    cameras = camera_map(config, manifest)
    state_dim = batches.STATE_DIM
    if config.steering.mode == "teleop":
        policy = preprocessor = postprocessor = None
    else:
        bundle = policy_factory(config, stats_path, cameras.rename_map())
        policy, preprocessor, postprocessor, state_dim = bundle
    chain = readout_chain.build_chain(
        {"corruption": config.readout.corruption, "input_noise": config.readout.input_noise},
        headless=headless if headless_operator is None else headless_operator,
    )
    return Wired(
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        station=station,
        chain=chain,
        jacobian=jacobian,
        arm=arm,
        manifest=manifest,
        cameras=cameras,
        now=now,
        state_dim=state_dim,
    )


PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"
"""The tokenizer a pi05 preprocessor instantiates, and a gated Google repository.

The checkpoint ships no tokenizer: `policy_preprocessor.json` names this one and `TokenizerProcessorStep` loads
it. Without an accepted licence and a token on the machine, the load fails with a `GatedRepoError` wrapped in a
hundred lines of transformers traceback, so it is turned into two lines here.
"""


def explain_pipeline_failure(error: Exception, policy_path: str) -> SystemExit:
    """The runner's exit for a pipeline that could not be built, naming the one thing to fix when it is access.

    Only the gated case is special: everything else keeps its own message, so a real bug is not hidden behind a
    licence hint.
    """
    text = f"{type(error).__name__}: {error}"
    if "gated" not in text.lower():
        return SystemExit(f"the policy's processors could not be built from {policy_path}: {text}")
    return SystemExit(
        f"the pi0.5 preprocessor needs the gated tokenizer {PALIGEMMA_TOKENIZER}, "
        "which this machine cannot read.\n"
        f"  1. accept the licence at https://huggingface.co/{PALIGEMMA_TOKENIZER} "
        "with the account this machine uses\n"
        "  2. log that machine in: `hf auth login` (or export HF_TOKEN=...), then run the same command again\n"
        "  offline instead: copy ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224 from a machine "
        "that has it, and set HF_HOME to the cache's parent"
    )


def build_policy(
    config: ExperimentConfig, stats_path: str | Path | None, rename_map: dict[str, str]
) -> PolicyBundle:
    """pi05_droid with openpi's normalization statistics installed as pipeline overrides.

    The published checkpoint carries no statistics (`stats.py`), so the two processors are built with the
    override's stats and the station's camera names. The caller's `stats_path` is that file; a missing one is
    named here rather than failing half-way through loading the weights.
    """
    import stats as checkpoint_stats

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.steering import SteeringConfig, SteeringPolicy

    if stats_path is None or not Path(stats_path).exists():
        raise FileNotFoundError(
            f"the checkpoint's statistics are missing: {stats_path} (pass --stats <openpi norm_stats.json>)"
        )
    base = PI05Policy.from_pretrained(config.policy.path)
    statistics = checkpoint_stats.read_openpi_stats(stats_path)
    width = checkpoint_stats.real_width(statistics)
    if width != ACTION_DIMS:
        raise SystemExit(
            f"--stats: {stats_path} describes {width} action dims, and this example drives {ACTION_DIMS} "
            "(seven joint velocities and a gripper position); it is not this checkpoint's statistics"
        )
    pre_overrides, post_overrides = checkpoint_stats.overrides(statistics, rename_map)
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=base.config,
            pretrained_path=config.policy.path,
            preprocessor_overrides=pre_overrides,
            postprocessor_overrides=post_overrides,
        )
    except Exception as error:  # the pipelines need assets the checkpoint does not carry
        raise explain_pipeline_failure(error, config.policy.path) from error
    policy = SteeringPolicy(
        SteeringConfig(**config.steering_config()),
        base_policy=base,
        postprocessor=postprocessor,
    )
    return PolicyBundle(policy, preprocessor, postprocessor, int(base.config.max_state_dim))


# ---------------------------------------------------------------- the action and the reset


def clamp_action(action: np.ndarray, q: np.ndarray, arm: ArmSpec, dt: float) -> np.ndarray:
    """One policy action as the arm can execute it: the velocity clamp, then the gripper's travel.

    The gripper is not binarized. The station takes a position and the policy's continuous value is the
    intent, so clipping to the station's range preserves the arm the policy chose; snapping it to an end is
    the teleop arm's business, where the operator has a button.
    """
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (GRIPPER_DIM + 1,):
        raise ValueError(f"one action must be ({GRIPPER_DIM + 1},), got {action.shape}")
    velocity = clamp_joint_velocity(
        action[:ARM_DIMS],
        np.asarray(q, dtype=np.float64),
        arm.qdot_cap,
        np.asarray(arm.q_min, dtype=np.float64),
        np.asarray(arm.q_max, dtype=np.float64),
        dt,
    )
    lo, hi = arm.gripper_range
    gripper = float(np.clip(action[GRIPPER_DIM], lo, hi))
    return np.concatenate([velocity, [gripper]])


def button_gripper(grip: float, arm: ArmSpec) -> float:
    """The gripper position a teleop button asks for: the closed end, or the other one."""
    lo, hi = arm.gripper_range
    closed = arm.gripper_closed_at
    closed_end = hi if hi - closed <= closed - lo else lo  # the nearer end, as `binarize_gripper` reads it
    return closed_end if grip > 0 else (lo if closed_end == hi else hi)


def require_start_pose(q: np.ndarray, start_pose: np.ndarray | None, tolerance_rad: float) -> None:
    """Raise unless `q` is within `tolerance_rad` of the configured start pose.

    A trial is only comparable to another arm's when it began from the same joint configuration, so a reset
    that landed elsewhere is refused rather than recorded. No configured pose means no check: the station's
    own reset is the reference then, and the reset hash still records what it was.
    """
    if start_pose is None:
        return
    q = np.asarray(q, dtype=np.float64)
    start = np.asarray(start_pose, dtype=np.float64)
    if q.shape != start.shape:
        raise ValueError(f"the start pose must have the reset's shape {q.shape}, got {start.shape}")
    if not tolerance_rad > 0:
        raise ValueError(f"the reset tolerance must be positive, got {tolerance_rad!r}")
    offset = float(np.max(np.abs(q - start)))
    if offset > tolerance_rad:
        raise ValueError(
            f"the station reset to a pose {offset:.4f} rad from the configured start "
            f"(tolerance {tolerance_rad} rad); refusing to start"
        )


# ---------------------------------------------------------------- one trial


def operator_activity(rows: list[dict], deadband: float = 0.05) -> tuple[int, float]:
    """How often the operator actually pushed, and how hard.

    A silent run and a working run look the same in a success rate: both say "no success". This is the number
    that tells them apart, so it is printed with every trial and a zero is a broken stick, not a hard task.
    """
    if not rows:
        return 0, 0.0
    pushes = [float(np.abs(np.asarray(row["user_translation"])).max()) for row in rows]
    return sum(1 for push in pushes if push >= deadband), max(pushes, default=0.0)


def run_trial(
    wired: Wired,
    *,
    trial: int,
    prompt: str,
    task_description: str,
    start_pose: np.ndarray | None,
    tolerance: float,
    max_steps: int,
    seed: int = 0,
    run_dir: Path | None = None,
    stop_reader: Any = None,
    brake: bool = True,
) -> TrialResult:
    """One trial: reset, check the pose, then one action held over the station's ticks.

    The command is read at an action boundary, turned into a joint velocity by the Jacobian, and steered
    through the policy; the postprocessor unnormalizes the action, `actions.clamp_joint_velocity` makes it
    executable and the gripper is clipped (not binarized) to the station's travel. Between boundaries the
    same eight numbers go out again, which is what keeps the policy's action rate and the station's clock
    aligned. `stop_reader.poll(tick)` may return ``"success"`` (the operator saw the task done) or
    ``"abort"`` (the run stops after this trial); it is polled once per station tick.
    """
    import torch

    config = wired.config
    station, chain, jacobian, arm = wired.station, wired.chain, wired.jacobian, wired.arm
    arm_name = wired.manifest.arm()
    obs = station.reset()
    q = np.asarray(obs[f"{arm_name}/joint_position"], dtype=np.float64)
    state_hash = initial_state_hash(obs, arm_name, wired.manifest.cameras)
    scene = scene_digest(obs, wired.manifest.cameras)
    require_start_pose(q, start_pose, tolerance)

    if wired.policy is not None:
        wired.policy.reset()
    chain.reset()
    ticks = ticks_per_action(wired.manifest.control_rate, ACTION_RATE)
    n_actions = int(config.policy.n_action_steps or 15)
    dt = 1.0 / wired.manifest.control_rate
    recorder = TrialRecorder()
    served = raw = np.zeros(3)
    plan = np.full(3, np.nan)
    action8: np.ndarray | None = None
    gripper = 0.0
    success = False
    aborted = False
    served_from_chunk = 0
    chunk_done = False  # the next action would compute a chunk: brake first (see below)

    for tick in range(max_steps):
        chain.advance()
        if stop_reader is not None:
            pressed = stop_reader.poll(tick)
            if pressed == "success":
                success = True
                break
            if pressed == "abort":
                aborted = True
                break

        if action8 is None or tick % ticks == 0:
            if brake and chunk_done:
                # The station holds the last velocity while the policy computes, so at 15 Hz a 300 ms chunk
                # pause drifts the arm 0.15 rad. One tick of zero velocity before each replan turns that
                # drift into a hold; it costs one tick per chunk and nothing else.
                chunk_done = False
                served_from_chunk = 0  # the next compute starts a new chunk
                served = np.asarray(chain.translation, dtype=np.float64).reshape(-1)
                brake_action = clamp_action(np.concatenate([np.zeros(ARM_DIMS), [gripper]]), q, arm, dt)
                recorder.add(
                    q=q, action=brake_action, plan=plan, served=served, raw=raw, gripper=gripper, reads=0
                )
                obs = station.step(action_dict(brake_action, arm_name))
                q = np.asarray(obs[f"{arm_name}/joint_position"], dtype=np.float64)
                continue
            served = np.asarray(chain.translation, dtype=np.float64).reshape(-1)
            raw = np.asarray(getattr(chain, "_raw", served), dtype=np.float64).reshape(-1)
            gripper = chain.grip()
            twist = np.zeros(6)
            twist[:3] = served * config.command.twist_full
            qdot_desired = jacobian.joint_velocity(q, twist)
            command = command_from_qdot(qdot_desired, config.command.qdot_full)
            if wired.policy is None:  # teleop: the operator's command is the action
                action8 = clamp_action(
                    np.concatenate([qdot_desired, [button_gripper(gripper, arm)]]), q, arm, dt
                )
                plan = np.full(3, np.nan)
            else:
                batch = wired.preprocessor(
                    observation_batch(obs, arm_name, wired.cameras, prompt, wired.state_dim)
                )
                batch["command"] = torch.as_tensor(command, dtype=torch.float32)
                with torch.inference_mode():
                    action = wired.policy.select_action(batch)
                    proposed = wired.policy.last_policy_action
                action8 = clamp_action(to_action(wired.postprocessor, action), q, arm, dt)
                plan = (
                    np.full(3, np.nan) if proposed is None else to_action(wired.postprocessor, proposed)[:3]
                )
            reads = 1
            served_from_chunk += 1
            if wired.policy is not None and served_from_chunk >= n_actions:
                chunk_done = True
        else:
            reads = 0

        recorder.add(q=q, action=action8, plan=plan, served=served, raw=raw, gripper=gripper, reads=reads)
        obs = station.step(action_dict(action8, arm_name))
        q = np.asarray(obs[f"{arm_name}/joint_position"], dtype=np.float64)

    duration = recorder.elapsed
    settings = flat_settings(
        wired,
        prompt=prompt,
        task_description=task_description,
        seed=seed,
        start_pose=start_pose,
        tolerance=tolerance,
        max_steps=max_steps,
    )
    steps_file = f"trial_{trial:03d}.npz"
    record = {
        "pair_id": pair_id_for(seed, trial),
        "initial_state_hash": state_hash,
        "scene_digest": scene,
        "trial": int(trial),
        **settings,
        "success": bool(success),
        "aborted": bool(aborted),
        "steps": len(recorder.rows),
        "duration_s": round(duration, 2),
        "user_reads": recorder.total_reads,
        "steps_file": steps_file,
        "finished_at": wired.now().isoformat(timespec="seconds"),
    }
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        recorder.save(
            run_dir / steps_file,
            task_description=task_description,
            vla_prompt=prompt,
            mode=settings["mode"],
            arm=settings["arm"],
            corruption_matrix=nan_matrix(settings["corruption_matrix"], 3),
            reversal_adapter_matrix=nan_matrix(settings["reversal_adapter_matrix"], 7),
            pair_id=record["pair_id"],
            initial_state_hash=state_hash,
            scene_digest=scene,
            trial=int(trial),
            success=bool(success),
        )
        with open(run_dir / "trials.jsonl", "a") as lines:
            lines.write(json.dumps(record) + "\n")
    return TrialResult(record=record, rows=recorder.rows)


# ---------------------------------------------------------------- the run


def run(
    wired: Wired,
    *,
    prompt: str,
    task_description: str,
    seed: int,
    start_pose: np.ndarray | None,
    tolerance: float,
    run_dir: Path,
    stop_reader: Any = None,
) -> list[dict]:
    """Every trial of one run, written under `run_dir`; an abort stops after the trial it ended."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            provenance(
                wired,
                prompt=prompt,
                task_description=task_description,
                seed=seed,
                start_pose=start_pose,
                tolerance=tolerance,
            ),
            sort_keys=False,
        )
    )
    max_steps = wired.config.trial.max_steps
    records: list[dict] = []
    for trial in range(wired.config.trial.trials):
        result = run_trial(
            wired,
            trial=trial,
            prompt=prompt,
            task_description=task_description,
            start_pose=start_pose,
            tolerance=tolerance,
            max_steps=max_steps,
            seed=seed,
            run_dir=run_dir,
            stop_reader=stop_reader,
        )
        record = result.record
        records.append(record)
        pushed, peak = operator_activity(result.rows)
        print(
            f"trial {trial + 1}/{wired.config.trial.trials}: "
            f"{'SUCCESS' if record['success'] else 'no success'} after {record['steps']} steps "
            f"({record['duration_s']:.1f}s), operator pushed on {pushed}/{len(result.rows)} ticks "
            f"(peak {peak:.2f})" + ("  [aborted]" if record["aborted"] else "")
        )
        if record["aborted"]:
            break
    wins = sum(record["success"] for record in records)
    print(
        f"\n{wired.config.steering.mode}: {len(records)} trials, {wins} success"
        + (f" ({wins / len(records):.0%})" if records else "")
    )
    print(f"Data: {run_dir}")
    return records


# ---------------------------------------------------------------- the command line


def _start_pose(text: str) -> np.ndarray:
    """Seven joint angles from a comma- or space-separated argument."""
    values = [float(value) for value in text.replace(",", " ").split()]
    if len(values) != ARM_DIMS:
        raise argparse.ArgumentTypeError(f"a start pose is {ARM_DIMS} numbers, got {len(values)}")
    return np.asarray(values, dtype=np.float64)


def _config_path(args: argparse.Namespace) -> Path:
    """The per-arm file: `--config` as given or under configs/, else configs/<arm>.yaml."""
    if args.config is not None:
        candidate = Path(args.config)
        if not candidate.exists():
            candidate = CONFIG_DIR / candidate
        if not candidate.exists():
            raise SystemExit(f"--config: {args.config} is not a file, and not under {CONFIG_DIR}")
        return candidate
    return CONFIG_DIR / f"{args.arm}.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The runner's flags."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=sorted(MODE_MAP), default="frs", help="condition file in configs/")
    parser.add_argument("--config", metavar="PATH", help="a condition file, as given or under configs/")
    parser.add_argument("--trials", type=int, default=None, help="override trial.trials")
    parser.add_argument(
        "--max-steps", dest="max_steps", type=int, default=None, help="override trial.max_steps"
    )
    parser.add_argument("--prompt", default=None, help="the VLA's prompt (default: the config's)")
    parser.add_argument("--task", default=None, help="the operator's task description (default: --prompt)")
    parser.add_argument("--stats", default=None, help=f"openpi's norm_stats.json (default: {DEFAULT_STATS})")
    parser.add_argument("--out", default=None, metavar="DIR", help="run directory (default: under outputs/)")
    parser.add_argument(
        "--dry-run", action="store_true", help="one fake-station trial; print actions, write nothing"
    )
    parser.add_argument("--fake-station", action="store_true", help="drive station.FakeStation")
    parser.add_argument("--seed", type=int, default=0, help="pairing seed")
    parser.add_argument(
        "--corruption",
        default=None,
        help="a readout corruption spec in YAML, overriding readout.corruption ('{rotation_z_deg: 20}')",
    )
    parser.add_argument(
        "--adapter", default=None, help="a user.pt for mars_learned, overriding steering.adapter_user"
    )
    parser.add_argument(
        "--start-pose", dest="start_pose", type=_start_pose, default=None, help="seven joint angles"
    )
    parser.add_argument(
        "--reset-tolerance",
        dest="reset_tolerance",
        type=float,
        default=RESET_TOLERANCE_RAD,
        help=f"radians a reset may differ from --start-pose (default: {RESET_TOLERANCE_RAD})",
    )
    parser.add_argument(
        "--stop-on", choices=("keypress",), default=None, help="end a trial on n, stop the run on q"
    )
    return parser.parse_args(argv)


class KeypressReader:
    """A tiny non-blocking stop reader: `n` succeeds the trial, `q` aborts the run.

    Unix only: `select` on stdin with cbreak so one key is read without Enter. Where `termios` is missing or
    stdin is not a terminal the reader is a no-op, which is what the tests and a redirected run need.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdin
        self._fd: int | None = None
        self._saved = None
        try:
            import select
            import termios
            import tty

            self._select = select
            fd = self._stream.fileno()
            if os.isatty(fd):
                self._saved = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                self._fd = fd
            self._termios = termios
        except (AttributeError, ImportError, OSError, ValueError):
            self._fd = None

    def poll(self, tick: int) -> str | None:
        """The key waiting on stdin, as ``"success"``, ``"abort"`` or None."""
        if self._fd is None:
            return None
        readable, _, _ = self._select.select([self._fd], [], [], 0)
        if not readable:
            return None
        key = os.read(self._fd, 1).decode(errors="ignore").lower()
        if key == "n":
            return "success"
        if key == "q":
            return "abort"
        return None

    def close(self) -> None:
        """Restore the terminal, if this reader changed it."""
        if self._saved is not None and self._fd is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)
            self._saved = None


class NullReader:
    """The headless stop reader: nothing is ever pressed."""

    def poll(self, tick: int) -> str | None:
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- main


def fake_observations(config: ExperimentConfig, q: np.ndarray | None = None) -> list[dict]:
    """One fake-station observation: a stored q, an open gripper and two zero frames."""
    arm = config.station.arm or "follower"
    q = DROID_RESET_Q if q is None else np.asarray(q, dtype=np.float64)
    obs = {f"{arm}/joint_position": q.copy(), f"{arm}/gripper_position": np.array([0.0])}
    for name in (config.cameras.exterior or "exterior", config.cameras.wrist or "wrist"):
        obs[name] = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return [obs]


def main(argv: Sequence[str] | None = None) -> int:
    """Build the real run, then either dry-run one trial or write the trials under `--out`."""
    args = parse_args(argv)
    try:
        config = ExperimentConfig.from_yaml(_config_path(args))
    except (OSError, ValueError) as error:
        raise SystemExit(f"--config: {error}") from error
    config = replace(
        config,
        trial=replace(
            config.trial,
            trials=args.trials if args.trials is not None else config.trial.trials,
            max_steps=args.max_steps if args.max_steps is not None else config.trial.max_steps,
        ),
    )
    if args.corruption is not None:
        config = replace(config, readout=replace(config.readout, corruption=yaml.safe_load(args.corruption)))
    if args.adapter is not None:
        config = replace(config, steering=replace(config.steering, adapter_user=args.adapter))
    stats_path = Path(args.stats) if args.stats is not None else DEFAULT_STATS
    if config.steering.mode != "teleop" and not stats_path.exists():
        raise SystemExit(f"--stats: {stats_path} does not exist (the checkpoint ships no statistics)")
    prompt = args.prompt or args.task or config.policy.prompt_generic
    task_description = args.task or args.prompt or config.policy.prompt_generic
    start_pose = args.start_pose
    fake = args.fake_station or args.dry_run
    if fake:
        station_factory = lambda cfg: FakeStation(  # noqa: E731 - one closure, no name worth giving it
            fake_observations(cfg, start_pose if start_pose is not None else DROID_RESET_Q),
            control_rate=FAKE_CONTROL_RATE,
        )
    else:
        station_factory = lambda cfg: C3poStation(cfg.station.host, cfg.station.port)  # noqa: E731

    wired = build(
        config,
        stats_path=stats_path,
        station_factory=station_factory,
        policy_factory=build_policy,
        now=dt.datetime.now,
        headless=fake,
        headless_operator=False,  # a dry run is a real stick test: only the station is fake
        start_pose=start_pose,
    )
    print(f"operator: {wired.chain.describe()}")
    reader = KeypressReader() if args.stop_on == "keypress" else NullReader()
    try:
        if args.dry_run:
            result = run_trial(
                wired,
                trial=0,
                prompt=prompt,
                task_description=task_description,
                start_pose=start_pose,
                tolerance=args.reset_tolerance,
                max_steps=min(10, wired.config.trial.max_steps),
                seed=args.seed,
                run_dir=None,
            )
            print(f"dry run: {len(result.rows)} steps, {wired.config.steering.mode}")
            for row in result.rows[:10]:
                print("  " + " ".join(f"{value:+.3f}" for value in row["action"]))
            print("nothing written")
            return 0
        run_dir = (
            Path(args.out)
            if args.out is not None
            else OUTPUT_ROOT / f"{wired.now():%Y%m%d_%H%M%S}_{wired.config.steering.mode}"
        )
        run(
            wired,
            prompt=prompt,
            task_description=task_description,
            seed=args.seed,
            start_pose=start_pose,
            tolerance=args.reset_tolerance,
            run_dir=run_dir,
            stop_reader=reader,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        reader.close()
        wired.station.close()


if __name__ == "__main__":
    raise SystemExit(main())
