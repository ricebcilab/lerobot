# pi0.5 + LIBERO shared autonomy

This folder provides a reproducible local setup for LeRobot's Pi0.5 policy and
the vanilla LIBERO benchmark. It uses the published
[`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
checkpoint by default.

## Install

From the repository root:

```bash
./examples/pi05/libero_shared_autonomy/setup.sh
```

The setup script:

- installs the locked `pi` and `libero` extras into `.venv`;
- writes a non-interactive LIBERO config under `.cache/libero`;
- downloads `lerobot/libero-assets` there; and
- links the installed `hf-libero` package to those project-local assets.

The evaluation launcher also keeps Hugging Face model downloads and Matplotlib
cache data under the project's ignored `.cache` directory. It leaves `HF_HOME`
unchanged so an existing `hf auth login` credential remains available for the
gated PaliGemma tokenizer used by Pi0.5.

Both `.venv` and `.cache` are ignored by git. Re-running the script is safe.
Every launcher in this folder sources `env.sh` first, which sets
`LIBERO_CONFIG_PATH`, `MUJOCO_GL`, `HF_HUB_CACHE` and `MPLCONFIGDIR` to paths
under the repo's `.cache/` directory (only if you have not already set them).

## Smoke evaluation

The default launcher evaluates task 0 of LIBERO-Spatial for one episode:

```bash
./examples/pi05/libero_shared_autonomy/eval.sh
```

This downloads the Pi0.5 checkpoint on first use. A CUDA GPU with a working
driver is strongly recommended; Pi0.5 is too large for a practical CPU rollout.
The PaliGemma tokenizer is gated: first accept its Hugging Face license and run
`hf auth login`. The initial rollout may spend several minutes compiling the
model; set `PI05_COMPILE_MODEL=false` to disable compilation.

The launcher is configured with environment variables. For example, run the
four standard suites with 10 episodes per task:

```bash
LIBERO_TASKS=libero_spatial,libero_object,libero_goal,libero_10 \
LIBERO_TASK_IDS= \
LIBERO_EPISODES=10 \
./examples/pi05/libero_shared_autonomy/eval.sh
```

Useful overrides:

| Variable             | Default                         | Meaning                                      |
| -------------------- | ------------------------------- | -------------------------------------------- |
| `PI05_MODEL_ID`      | `lerobot/pi05_libero_finetuned` | Local checkpoint or Hub model ID             |
| `LIBERO_TASKS`       | `libero_spatial`                | Comma-separated LIBERO suites                |
| `LIBERO_TASK_IDS`    | `[0]`                           | Task IDs; set empty to run every task        |
| `LIBERO_EPISODES`    | `1`                             | Episodes per task                            |
| `LIBERO_BATCH_SIZE`  | `1`                             | Parallel evaluation environments             |
| `PI05_ACTION_STEPS`  | `10`                            | Actions executed from each predicted chunk   |
| `PI05_COMPILE_MODEL` | `true`                          | Enable the checkpoint's `torch.compile` path |
| `LIBERO_OUTPUT_DIR`  | `outputs/pi05_libero_eval`      | Evaluation output directory                  |

For headless execution the launcher sets `MUJOCO_GL=egl`. Override it (for
example, with `MUJOCO_GL=osmesa`) only if the machine's rendering setup requires
another MuJoCo backend.

## Interactive prompting

To type free-text instructions and watch the policy react, use the interactive
launcher instead of the batch evaluation:

```bash
./examples/pi05/libero_shared_autonomy/interactive.sh
```

It loads the policy once, opens a live view at `http://localhost:8765` (VSCode
forwards the port automatically) showing the camera stream plus the policy's
per-step action vector (labeled end-effector deltas and gripper command), and
drops into a REPL: type any instruction
(or press Enter for the scene's built-in one) to run a rollout in the current
LIBERO scene. Each rollout is also saved as an MP4 under
`outputs/pi05_libero_interactive/`. Use `tasks` to list the current suite's
scenes, `task <suite> <id>` to switch scene, and `quit` to exit.

Besides the default `mode policy` (model-only control), four teleop modes
let you drive or steer the arm with a 3Dconnexion SpaceMouse and/or the
keyboard:

- `mode teleop` — you drive the arm (x/y/z and the gripper); the model is
  not involved.
- `mode shared_override` — pi0.5 drives, but your x/y/z replaces the
  model's translation in the executed action.
- `mode shared_flow_control [tau]` — pi0.5 drives; while you are pushing, your
  translation (normalized to the model's action space) is written into dims
  0-2 of `x_t` for the first `tau` of the 10 flow-matching denoising steps of
  each action chunk, and the remaining steps denoise freely. The executed
  action is entirely the model's output, steered through the early flow. Idle
  input = pure policy.
- `mode shared_reverse_flow_steering [n]` — Flow Reversal Steering
  ([Tang et al. 2026](https://arxiv.org/abs/2606.13675)): while you are
  pushing, a reference chunk that servos in your direction at uniform velocity
  (rotation zero, gripper held at the model's last command) is integrated
  _backward_ through the policy's own velocity field for the same 10 steps to
  find the latent noise that maps to it; the normal forward flow then runs from
  that noise instead of random noise. The executed action is entirely the
  model's output — the reference only picks the starting noise, so you get the
  generalist action mode nearest your intent. Idle input = pure policy. Costs
  one extra denoising pass per steered chunk. After each rollout the REPL
  prints how many chunks were steered and how far the executed translation
  landed from the reference (in action-std units). By default the reference is
  reversed through all 10 denoising steps, i.e. all the way to noise;
  `--n-reverse-steps n` (or `mode shared_reverse_flow_steering n`) stops the
  reversal after `n` of them instead, so the chunk still carries part of your
  reference and the forward flow finishes it off — smaller `n` keeps more of
  your intent, larger `n` gives the policy more freedom (up to `n = 10`, the
  default), and any `n < 10` costs the same 10 velocity evaluations as an
  unsteered chunk (`n` reverse + `10 - n` forward) against 20 for the full
  reversal, roughly twice as fast per chunk.

Input sources (both active at once; the SpaceMouse wins while deflected):

- **SpaceMouse** — read directly from `/dev/hidraw`, no extra install; x/y/z
  from the stick, any button toggles the gripper. Connected the first time
  you switch to a teleop mode; if none is plugged in, the mode still works
  with the keyboard alone.
- **Keyboard** — captured by the live-view page (click it to give it focus):
  <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> move forward/back/left/right,
  <kbd>PgUp</kbd>/<kbd>PgDn</kbd> (or <kbd>W</kbd>/<kbd>S</kbd>) move up/down,
  <kbd>Space</kbd> toggles the gripper, hold <kbd>Shift</kbd> for full speed
  (keys move at half speed by default). Arrow directions follow the SpaceMouse
  axis tuning in `teleop.py`, so `↑` moves the arm like pushing the stick
  forward. The page shows the held keys and gripper state under the action bars.

The REPL command reference is `interactive.py --help`; settings can also come
from `configs/interactive.yaml` (`interactive.sh --config`).

## Perturbing the operator

Three ways to make the operator's command less than perfect, useful for
studying how much steering the policy tolerates:

- **Input noise** — `--input-noise STD` (or `noise STD` in the REPL) adds
  independent Gaussian noise to Δx, Δy and Δz of your command at every control
  step while you are commanding; `0` (the default) is off.
- **Command corruption** — `control.corruption` in a config (or
  `--corruption FILE` / `corruption FILE` in the REPL) left-multiplies your
  Δx/Δy/Δz by a fixed 3×3 matrix `M` before it reaches the policy:

  ```yaml
  corruption: null                      # off
  corruption: {rotation_z_deg: 20}      # rotate the commanded direction about z
  corruption: {scale: [1, 1, 1]}        # per-axis gain
  corruption: {M: [[..3 rows..]]}       # literal
  ```

- **Reversal adapter** — `control.reversal_adapter` (or `--reversal-adapter FILE`
  / `adapter FILE` in the REPL) only affects `shared_reverse_flow_steering`: it
  left-multiplies the velocity field used by the reverse integration with a
  fixed 7×7 matrix `F`, so only the noise that the forward flow starts from
  changes — the executed action still comes from the policy's own field:

  ```yaml
  reversal_adapter: null
  reversal_adapter:
    translation: corruption             # copy M from control.corruption (error if that is null)
    orientation: zero                   # identity | zero
    gripper: zero                       # identity | zero
  reversal_adapter:
    translation: {rotation_z_deg: 20}   # or {scale: [...]}, or a 3x3 literal
    orientation: identity
    gripper: identity
  reversal_adapter: {F: [[..7 rows..]]} # literal
  ```

A file loaded through `--corruption FILE` / `--reversal-adapter FILE` (or live,
with `corruption FILE` / `adapter FILE`) holds the same spec forms at its top
level.

## Experiments

To run scripted, recorded trials from a YAML config instead of the REPL
(shared-autonomy user studies), use `experiment.sh` with one of the shipped
conditions:

```bash
./examples/pi05/libero_shared_autonomy/experiment.sh --config configs/experiment/reverse_flow_full.yaml
./examples/pi05/libero_shared_autonomy/experiment.sh --config configs/experiment/reverse_flow_full.yaml --dry-run   # validate + print the schedule, run nothing
```

`experiment.py` runs a scripted sequence of trials with the same policy, live
view and control modes as the interactive runner above, and records every
trial to disk. Each trial draws a task from the configured scene, shows
**you** the task to accomplish, and lets you attempt it with the VLA in the
configured shared-autonomy mode.

### The prompt vs. the task

The **human** is always shown the scene's real instruction ("pick up the black
bowl…"). The **VLA** only ever receives the config's `prompt`, which defaults
to the deliberately uninformative `"do something"` — the policy contributes
manipulation priors while you supply the intent through the shared-autonomy
channel. Set `prompt: task` to give the VLA the scene's own instruction
instead, i.e. to run the conventional setup where both know the goal.

### Conditions

Five condition files ship in `configs/experiment/`; each has `extends: base.yaml`
and overrides only what it changes:

- `flow_control_tau8.yaml` — shared flow control, `tau = 8`, clean operator command.
- `flow_control_tau8_rotz20.yaml` — shared flow control, `tau = 8`, operator
  command rotated 20° about z.
- `reverse_flow_full.yaml` — reverse flow steering, full reversal to noise,
  clean operator command.
- `reverse_flow_full_rotz20.yaml` — reverse flow steering, full reversal,
  command rotated 20° about z; the reversal adapter applies the same rotation
  to the velocity's translation block and freezes orientation and gripper
  during the reversal.
- `reverse_flow_5steps_rotz20.yaml` — reverse flow steering stopped after 5 of
  10 steps (t = 0.5), same rotation and adapter as `reverse_flow_full_rotz20`.

### Configuration

Everything lives in the YAML file you pass to `--config`. Unknown keys are
rejected and a key set to `null` keeps the built-in default, so a typo fails
immediately instead of silently doing nothing. Only the keys you want to
change need to be present; the run's name is the config file's stem (no
`experiment.name` key).

| Key                        | Default                           | Meaning                                                                                                                                                                |
| -------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.n_trials`      | `10`                              | Number of trials                                                                                                                                                       |
| `experiment.seed`          | `0`                               | Seeds the task schedule — same seed, same tasks                                                                                                                        |
| `experiment.task_order`    | `random`                          | `random` (uniform, with replacement), `shuffled` (permuted blocks, each task once per block), `sequential` (cycle in order)                                            |
| `experiment.output_dir`    | `outputs/pi05_libero_experiments` | Parent of the run directory                                                                                                                                            |
| `scene.suite`              | `libero_goal`                     | LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`)                                                                |
| `scene.task_ids`           | `null`                            | Task pool to draw from; `null` = every task in the suite                                                                                                               |
| `prompt`                   | `"do something"`                  | Instruction handed to the VLA; the literal `task` uses the scene's own instruction                                                                                     |
| `policy.path`              | `lerobot/pi05_libero_finetuned`   | Hub id or local checkpoint directory                                                                                                                                   |
| `policy.n_action_steps`    | `10`                              | Actions executed per predicted chunk                                                                                                                                   |
| `policy.compile`           | `false`                           | `torch.compile` the model (slow first trial, faster after)                                                                                                             |
| `control.mode`             | `shared_reverse_flow_steering`    | `policy`, `teleop`, `shared_override`, `shared_flow_control`, `shared_reverse_flow_steering` — see the mode list under [Interactive prompting](#interactive-prompting) |
| `control.tau`              | `5`                               | `shared_flow_control` only                                                                                                                                             |
| `control.n_reverse_steps`  | `null`                            | `shared_reverse_flow_steering` only: denoising steps the reference is reversed through; `null` = all the way to noise                                                  |
| `control.input_noise`      | `0.0`                             | Std of the Gaussian noise added to your x/y/z command                                                                                                                  |
| `control.max_steps`        | `null`                            | Rollout length; `null` = the suite's own episode length                                                                                                                |
| `control.corruption`       | `null`                            | Matrix spec applied to the operator's command — see [Perturbing the operator](#perturbing-the-operator); off unless set                                                |
| `control.reversal_adapter` | `null`                            | Adapter spec for `shared_reverse_flow_steering`'s reverse integration — see [Perturbing the operator](#perturbing-the-operator); off unless set                        |
| `server.port`              | `8765`                            | Live view port                                                                                                                                                         |

A few keys can be overridden per run without editing the file:
`--n-trials`, `--seed`, `--mode`, `--output-dir`, `--port`.

### Running a session

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

### What gets recorded

Each run creates `<output_dir>/<YYYYmmdd_HHMMSS>_<config stem>/` containing:

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
`reconstruction_error_mean`). `deterministic_corruption` and
`flow_reversal_adapter` hold a short label of the configured spec (e.g.
`rotation_z_deg=20`) or the loaded file's name, and `null` when off.

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

## Analysis

[`notebooks/analyze_experiments.ipynb`](notebooks/analyze_experiments.ipynb)
imports [`notebooks/analyze.py`](notebooks/analyze.py), which loads every run
directory that contains a `trials.jsonl` and reports success rate with Wilson 95%
intervals, a paired per-trial comparison, a per-task breakdown, step counts
split by outcome, and how much the operator actually steered. Point `RUN_DIRS`
at other directories to compare anything else. Its functions:

- `find_runs(root)` — run directories under `root` that recorded at least one
  trial, oldest first.
- `load_runs(run_dirs)` — loads `trials.jsonl` and `config.yaml` for each run
  into one combined `DataFrame` plus a dict of configs.
- `success_table(trials, configs)` — per-run trial and success counts, success
  rate, and a Wilson 95% confidence interval.
- `compare` / `paired_comparisons` — a paired per-trial comparison between two
  (or every pair of) runs via McNemar's exact test, using trials that share
  the same task schedule (same suite, seed and `task_order`).
- `input_activity(trials)` — per-trial operator engagement (fraction of steps
  commanding, mean speed and corruption shift while commanding, gripper-closed
  fraction, teleop reads), derived from the `.npz` step arrays.

Jupyter is not part of the `pi`/`libero` extras; run it without touching the
project environment with:

```bash
uv run --extra pi --extra libero --with jupyterlab jupyter lab examples/pi05/libero_shared_autonomy/notebooks/analyze_experiments.ipynb
```

The notebook's outputs are stripped before commit by the `nbstripout`
pre-commit hook, so diffs stay to the code cells.

## Fine-tuning

A basic Pi0.5 LIBERO fine-tuning job can be launched with:

```bash
uv run --locked --extra pi --extra libero lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.device=cuda \
  --output_dir=outputs/pi05_libero_train \
  --job_name=pi05_libero_train \
  --steps=6000 \
  --batch_size=1
```

Increase the batch size or use distributed training according to available
VRAM. The published LeRobot result used an effective batch size of 256 on eight
H100 GPUs; the command above deliberately starts with a safer per-process batch
size of 1.

## Troubleshooting

- **"SpaceMouse unavailable"** — not plugged in, or `/dev/hidraw*` is not
  readable; the mode still works with the keyboard.
- **Keys do nothing** — the browser page needs focus (it shows an orange hint
  until you click it), and a deflected SpaceMouse overrides the keyboard.
- **The arm wanders or goes the wrong way while I hold a direction** — check
  `noise` and `corruption` in the REPL (both are shown in the live view when
  active); `noise 0` and `corruption off` disable them.
- **`shared_reverse_flow_steering` behaves oddly** — check `adapter`; a
  non-identity `F` changes the reversal. `adapter off` restores the plain
  method.
- **Port already in use** — pass `--port`.
- **First rollout very slow with `--compile`** — expected; compilation takes a
  few minutes and the `No valid triton configs` messages are harmless.
- **Rendering errors** — try `MUJOCO_GL=osmesa` or make sure EGL drivers are
  installed.
