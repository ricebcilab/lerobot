# `experiment.sh` — shared-autonomy user study on LIBERO

`experiment.py` runs a scripted sequence of trials with the same policy, live
view and control modes as [`interactive.sh`](README_interactive.md), and
records every trial to disk. Each trial draws a task from the configured
scene, shows **you** the task to accomplish, and lets you attempt it with the
VLA in the configured shared-autonomy mode.

```bash
./examples/pi05_libero/experiment.sh --config my_study.yaml
./examples/pi05_libero/experiment.sh --config my_study.yaml --dry-run   # validate + print the schedule, run nothing
./examples/pi05_libero/experiment.sh                                    # only if a config_experiment.yaml sits next to the script
```

`--config` defaults to `config_experiment.yaml` next to `experiment.py`. Keeping
one config per condition (`config_experiment_<condition>.yaml`) and always
passing `--config` is usually clearer; if the file you name is missing, the
error lists the `config_experiment*.yaml` files it found.

The launcher exports the same environment variables as `interactive.sh`
(`LIBERO_CONFIG_PATH`, `MUJOCO_GL`, `HF_HUB_CACHE`, `MPLCONFIGDIR`) and
forwards its arguments to `experiment.py`. First-time setup is
`./examples/pi05_libero/setup.sh`.

## The prompt vs. the task

The **human** is always shown the scene's real instruction ("pick up the black
bowl…"). The **VLA** only ever receives the config's `prompt`, which defaults
to the deliberately uninformative `"do something"` — the policy contributes
manipulation priors while you supply the intent through the shared-autonomy
channel. Set `prompt: task` to give the VLA the scene's own instruction
instead, i.e. to run the conventional setup where both know the goal.

## Configuration

Everything lives in the YAML file you pass to `--config`. Unknown keys are
rejected and a key set to `null` keeps the built-in default, so a typo fails
immediately instead of silently doing nothing. Only the keys you want to
change need to be present.

| Key                                | Default                           | Meaning                                                                                                                                                                               |
| ---------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.name`                  | `experiment`                      | Goes into the output directory name                                                                                                                                                   |
| `experiment.n_trials`              | `10`                              | Number of trials                                                                                                                                                                      |
| `experiment.seed`                  | `0`                               | Seeds the task schedule — same seed, same tasks                                                                                                                                       |
| `experiment.task_order`            | `random`                          | `random` (uniform, with replacement), `shuffled` (permuted blocks, each task once per block), `sequential` (cycle in order)                                                           |
| `experiment.output_dir`            | `outputs/pi05_libero_experiments` | Parent of the run directory                                                                                                                                                           |
| `scene.suite`                      | `libero_spatial`                  | LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`)                                                                               |
| `scene.task_ids`                   | `null`                            | Task pool to draw from; `null` = every task in the suite                                                                                                                              |
| `prompt`                           | `"do something"`                  | Instruction handed to the VLA; the literal `task` uses the scene's own instruction                                                                                                    |
| `policy.path`                      | `lerobot/pi05_libero_finetuned`   | Hub id or local checkpoint directory                                                                                                                                                  |
| `policy.n_action_steps`            | `10`                              | Actions executed per predicted chunk                                                                                                                                                  |
| `policy.compile`                   | `false`                           | `torch.compile` the model (slow first trial, faster after)                                                                                                                            |
| `control.mode`                     | `shared_reverse_flow_steering`    | `policy`, `teleop`, `shared_override`, `shared_flow_control`, `shared_reverse_flow_steering` — see [the mode table](README_interactive.md#modes)                                      |
| `control.tau`                      | `5`                               | `shared_flow_control` only                                                                                                                                                            |
| `control.n_reverse_steps`          | `null`                            | `shared_reverse_flow_steering` only: denoising steps the reference is reversed through; `null` = all the way to noise. See [Partial reversal](README_interactive.md#partial-reversal) |
| `control.input_noise`              | `0.0`                             | Std of the Gaussian noise added to your x/y/z command                                                                                                                                 |
| `control.max_steps`                | `null`                            | Rollout length; `null` = the suite's own episode length                                                                                                                               |
| `control.deterministic_corruption` | `null`                            | Path to a 3×3 `M` YAML — **off unless a path is given**                                                                                                                               |
| `control.flow_reversal_adapter`    | `null`                            | Path to a 7×7 `F` YAML — **off unless a path is given**                                                                                                                               |
| `server.port`                      | `8765`                            | Live view port                                                                                                                                                                        |

A few keys can be overridden per run without editing the file:
`--n-trials`, `--seed`, `--mode`, `--output-dir`, `--port`.

## Running a session

For each trial the terminal prints the trial number, the scene, **your** task
and the VLA prompt. The live view shows the reset scene with your task on its
own line and the VLA prompt beneath it; both stay up for the whole trial, so
nothing changes on screen when the rollout actually begins. Then:

```
Press Enter to start (s = skip this trial, q = end the experiment):
```

`Enter` starts the rollout, `s` skips the trial without recording it, `q` ends
the experiment (everything recorded so far is kept). Drive with the SpaceMouse
and/or the keyboard exactly as in the interactive runner — click the live view
first so it captures your keys. The rollout ends on success, on the episode
limit, or at `control.max_steps`.

## What gets recorded

Each run creates `<output_dir>/<YYYYmmdd_HHMMSS>_<name>/` containing:

| File            | Contents                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`   | The resolved configuration, the trial schedule, and the corruption / adapter **matrices themselves** — the run is self-describing even if you later edit those YAML files |
| `trials.jsonl`  | One JSON object per trial, appended as it finishes                                                                                                                        |
| `trial_XXX.npz` | Per-step arrays for trial `XXX`                                                                                                                                           |
| `trial_XXX.mp4` | The rollout video (30 fps)                                                                                                                                                |

`trials.jsonl` fields: `trial`, `suite`, `task_id`, `task_description`,
`vla_prompt`, `mode`, `tau`, `n_reverse_steps`, `input_noise`, `deterministic_corruption`,
`flow_reversal_adapter`, `success`, `steps`, `duration_s`, `user_reads`
(how many times the teleop input was sampled during that trial),
`video`, `steps_file`, `finished_at`, plus the mode's steering statistics
(`guided_denoising_steps`, or `steered_chunks` and
`reconstruction_error_mean`).

`trial_XXX.npz` arrays, one row per control step:

| Array                                                          | Shape              | Contents                                                                                                                        |
| -------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `t`                                                            | `(T,)`             | Seconds since the rollout started                                                                                               |
| `action`                                                       | `(T, 7)`           | The env action actually executed: `Δx Δy Δz Δroll Δpitch Δyaw gripper`                                                          |
| `user_translation`                                             | `(T, 3)`           | The operator command **as consumed** — after the corruption matrix and the input noise                                          |
| `user_translation_raw`                                         | `(T, 3)`           | The same command **before** corruption and noise, i.e. the operator's true intent                                               |
| `user_gripper`                                                 | `(T,)`             | Operator gripper command (−1 open, +1 close)                                                                                    |
| `user_reads`                                                   | `(T,)`             | How many times the input was sampled during that step; `0` = the policy did not consult it (the flow modes read once per chunk) |
| `eef_pos`, `eef_quat`                                          | `(T, 3)`, `(T, 4)` | End-effector pose the policy saw at that step                                                                                   |
| `gripper_qpos`                                                 | `(T, 2)`           | Gripper joint positions                                                                                                         |
| `joint_pos`                                                    | `(T, 7)`           | Arm joint positions                                                                                                             |
| `reward`, `terminated`, `truncated`                            | `(T,)`             | Environment response                                                                                                            |
| `task_description`, `vla_prompt`, `mode`, `task_id`, `success` | scalars            | Trial identity, so a single `.npz` stands alone                                                                                 |

Missing robot-state fields are recorded as `NaN` rather than aborting a trial.
Loading a trial is just:

```python
import json, numpy as np
trials = [json.loads(line) for line in open(run_dir / "trials.jsonl")]
z = np.load(run_dir / "trial_000.npz")
z["user_translation_raw"], z["action"], z["eef_pos"], bool(z["success"])
```

## Analysing a session

[`analyze_experiments.ipynb`](analyze_experiments.ipynb) loads the two most
recent run directories and reports success rate with Wilson 95 % intervals, a
paired per-trial comparison (runs that share a suite, seed and `task_order`
see the same task schedule, so trial _i_ is the same task in both), a per-task
breakdown, step counts split by outcome, and how much the operator actually
steered. Point `RUN_DIRS` at other directories to compare anything else.

Jupyter is not part of the `pi`/`libero` extras; run it without touching the
project environment with:

```bash
uv run --extra pi --extra libero --with jupyterlab jupyter lab examples/pi05_libero/analyze_experiments.ipynb
```

## Notes

- The task schedule is computed up front from `seed`, so a run is reproducible
  and is written into `config.yaml`.
- The environment is rebuilt only when the scene changes; repeated tasks just
  reset.
- Skipped trials are not recorded; trial numbering follows the schedule index,
  so gaps in the file names are expected after a skip.
