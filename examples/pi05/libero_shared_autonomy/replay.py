#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Add `policy_translation` to trials recorded before it was recorded live.

    run.sh python replay.py outputs/human_pilot/p01 [--force] [--tolerance 1e-4]

`policy_translation` is what the policy would have executed on its own at each
step (see README, "What gets recorded"). experiment.py now records it during the
trial; for older runs this script regenerates it: the simulator is reset to the
trial's recorded initial state (checked against `initial_state_hash`), stepped
with the recorded actions so it passes through the same states, and at every
chunk boundary where the operator was pushing the plain policy is queried under
the trial's VLA prompt with the seed the live run used (`policy_seed + chunk`),
so the plan starts from the same Gaussian sample the steered chunk did. Where the
operator was not pushing the plan is the executed action itself.

The replayed end-effector trajectory must match the recorded one within
`--tolerance` (m); a trial that drifts is reported and left untouched. The array
is written into the existing `trial_XXX.npz` (atomically; every other array is
kept as is). Trials that already have it are skipped unless `--force`.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ControlSettings, SessionSettings  # noqa: E402
from reproducibility import TrialSpec  # noqa: E402
from session import Session  # noqa: E402

from lerobot.policies.pi05.steering import DEADBAND  # noqa: E402

STEERING_MODES = ("shared_flow_control", "shared_flow_reversal_steering")


def replay_trial(session: Session, record: dict, z, config: dict) -> tuple[np.ndarray, float]:
    """(policy_translation (T, 3), max end-effector drift in m) for one recorded trial."""
    spec = TrialSpec(**{k: record[k] for k in TrialSpec.__dataclass_fields__})
    if (spec.suite, spec.task_id) != (session.suite, session.task_id):
        session.set_scene(spec.suite, spec.task_id)
    observation = session.show_scene(spec)
    if session.initial_state_hash() != record["initial_state_hash"]:
        raise RuntimeError("reset did not reproduce the recorded initial state")

    actions, served, eef = z["action"], z["user_translation"], z["eef_pos"]
    n = config["n_action_steps"]
    steers = config["mode"] in STEERING_MODES and (
        config["mode"] != "shared_flow_control" or config["n_guided_steps"] > 0
    )
    plan = actions[:, :3].astype(np.float32).copy()  # the plan is the action wherever nothing steered it
    drift = 0.0
    for step in range(len(actions)):
        if step % n == 0 and steers and np.max(np.abs(served[step])) >= DEADBAND:
            torch.manual_seed((spec.policy_seed + step // n) % 2**32)
            chunk = session.env_actions(session.policy_chunk(observation, record["vla_prompt"]))
            plan[step : step + n] = chunk[: min(n, len(actions) - step), :3]
        pos = np.asarray(observation["robot_state"]["eef"]["pos"], dtype=np.float64).reshape(-1)[:3]
        drift = max(drift, float(np.max(np.abs(pos - eef[step]))))
        observation, *_ = session.vec_env.step(actions[step][None])
    return plan, drift


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="a run directory, or a directory of run directories")
    parser.add_argument("--force", action="store_true", help="recompute trials that already have the array")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max end-effector drift accepted (m)")
    parser.add_argument("--port", type=int, default=8790, help="live view port (the replay shows on it)")
    args = parser.parse_args()

    runs = (
        [args.root]
        if (args.root / "trials.jsonl").exists()
        else sorted(p for p in args.root.iterdir() if (p / "trials.jsonl").exists())
    )
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    os.environ.pop("DISPLAY", None)  # a replay needs no browser tab
    session: Session | None = None
    done = skipped = drifted = 0
    for run in runs:
        config = yaml.safe_load((run / "config.yaml").read_text())
        records = [
            json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines() if line.strip()
        ]
        print(f"== {run.name}: {len(records)} trials")
        for record in records:
            path = run / record["steps_file"]
            with np.load(path) as z:
                if "policy_translation" in z.files and not args.force:
                    skipped += 1
                    continue
                if session is None:
                    session = Session(
                        SessionSettings(
                            policy_path=config["policy_path"],
                            n_action_steps=config["n_action_steps"],
                            compile=config.get("compile", False),
                            suite=record["suite"],
                            task_id=record["task_id"],
                            port=args.port,
                            control=ControlSettings(mode="policy"),
                        )
                    )
                arrays = {key: z[key] for key in z.files}
                plan, drift = replay_trial(session, record, z, config)
            if drift > args.tolerance:
                drifted += 1
                print(
                    f"  {path.name}: end-effector drift {drift:.2e} m > {args.tolerance:.0e}, left untouched"
                )
                continue
            arrays["policy_translation"] = plan
            with tempfile.NamedTemporaryFile(dir=run, suffix=".npz", delete=False) as tmp:
                np.savez_compressed(tmp, **arrays)
            os.replace(tmp.name, path)
            done += 1
            print(f"  {path.name}: done (drift {drift:.1e} m)")
    if session is not None:
        session.close()
    print(f"replayed {done}, skipped {skipped} already done, {drifted} drifted")


if __name__ == "__main__":
    main()
