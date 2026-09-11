# pi0.5 + LIBERO shared autonomy

This folder provides a reproducible local setup for LeRobot's Pi0.5 policy and
the vanilla LIBERO benchmark. It uses the published
[`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
checkpoint by default.

## Install

From the repository root:

```bash
./examples/pi05/libero_shared_autonomy/run.sh setup
```

`run.sh` is the only launcher in this folder: `run.sh setup` once, then
`run.sh interactive`, `run.sh experiment`, or `run.sh <any command>` to run
something else (such as `lerobot-eval` or `jupyter`) inside the example's
environment. Every form sources `env.sh`, which sets `LIBERO_CONFIG_PATH`,
`MUJOCO_GL`, `HF_HUB_CACHE` and `MPLCONFIGDIR` to paths under the repo's
ignored `.cache/` directory (only if you have not already set them), and runs
from the repo root under `uv run` with the `pi` and `libero` extras.

The setup step:

- installs the locked `pi` and `libero` extras into `.venv`;
- writes a non-interactive LIBERO config under `.cache/libero`;
- downloads `lerobot/libero-assets` there; and
- links the installed `hf-libero` package to those project-local assets.

`env.sh` leaves `HF_HOME` unchanged so an existing `hf auth login` credential
remains available for the gated PaliGemma tokenizer used by Pi0.5. Both
`.venv` and `.cache` are ignored by git. Re-running setup is safe.

## Smoke evaluation

Batch evaluation is plain `lerobot-eval` run through `run.sh`. This evaluates
task 0 of LIBERO-Spatial for one episode:

```bash
./examples/pi05/libero_shared_autonomy/run.sh lerobot-eval \
  --output_dir=outputs/pi05_libero_eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --policy.compile_model=true \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1
```

This downloads the Pi0.5 checkpoint on first use. A CUDA GPU with a working
driver is strongly recommended; Pi0.5 is too large for a practical CPU rollout.
The PaliGemma tokenizer is gated: first accept its Hugging Face license and run
`hf auth login`. The initial rollout may spend several minutes compiling the
model; pass `--policy.compile_model=false` to disable compilation.

To run the four standard suites with 10 episodes per task, drop the
`--env.task_ids` line and change `--env.task` and `--eval.n_episodes`:

```bash
./examples/pi05/libero_shared_autonomy/run.sh lerobot-eval \
  --output_dir=outputs/pi05_libero_eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```

`--policy.path` accepts a local checkpoint directory as well as a Hub id.

For headless execution `env.sh` sets `MUJOCO_GL=egl`. Override it (for
example, with `MUJOCO_GL=osmesa`) only if the machine's rendering setup requires
another MuJoCo backend.

## Interactive prompting

To type free-text instructions and watch the policy react, use the interactive
launcher instead of the batch evaluation:

```bash
./examples/pi05/libero_shared_autonomy/run.sh interactive
```

It loads the policy once, opens a live view at `http://localhost:8765` in your browser when a display is available (VSCode
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
- `mode shared_flow_control [n_guided_steps]` — pi0.5 drives; while you are pushing, your
  translation (normalized to the model's action space) is written into dims
  0-2 of `x_t` for the first `n_guided_steps` of the 10 flow-matching denoising steps of
  each action chunk, and the remaining steps denoise freely. The executed
  action is entirely the model's output, steered through the early flow. Idle
  input = pure policy.
- `mode shared_flow_reversal_steering [n]` — Flow Reversal Steering
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
  `--n-reversal-steps n` (or `mode shared_flow_reversal_steering n`) stops the
  reversal after `n` of them instead, before delegation and forward integration
  back to actions. Depth's effect on intent preservation must be measured
  (`n = 10` is the full-reversal default). Reversal and forward integration both use step size 0.1: `n`
  reverse steps are followed by `n` forward steps. With the delegation below,
  a steered chunk costs **10 + n velocity evaluations** (`n` reverse,
  `10 - n` to obtain the unsteered state, and `n` forward), against 10 for FC
  or an unsteered chunk and 20 for full reversal. These counts exclude the
  synthetic operator's separate policy query. Only what you command is kept
  from the reference: after the reversal, the latent of the rotation and
  gripper dims, of the padding dims and of every step beyond the executed
  prefix is replaced by the unsteered latent (fresh Gaussian noise for a full
  reversal), the noise-space in-painting of Tang et al., App. D, so those are
  the policy's to decide. `control.operator_dims` (experiments) changes which
  dims count as commanded.

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
from `configs/interactive.yaml` (`run.sh interactive --config`).

## Perturbing the operator

Three ways to make the operator's command less than perfect, useful for
studying how much steering the policy tolerates:

- **Input noise** — `--input-noise STD` (or `noise STD` in the REPL) adds
  independent Gaussian noise to Δx, Δy and Δz whenever the command is sampled
  while you are commanding (once per chunk in FC/FRS, once per control step in
  teleop/override); `0` (the default) is off.
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
  / `adapter FILE` in the REPL) only affects `shared_flow_reversal_steering`: it
  left-multiplies the velocity field used by the reverse integration with a
  fixed 7×7 **environment-space** matrix `F`, so only the noise that the forward flow starts from
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

  Before applying `F` to normalized velocities, the wrapper converts it to
  `S⁻¹ F S`, where `S` is the diagonal matrix of action standard deviations.
  This applies to literal matrices as well as block specifications. Velocity
  transforms have no mean offset. Both matrices are saved in the run config.

A file loaded through `--corruption FILE` / `--reversal-adapter FILE` (or live,
with `corruption FILE` / `adapter FILE`) holds the same spec forms at its top
level.

## Experiments

To run scripted, recorded trials from a YAML config instead of the REPL
(shared-autonomy user studies), use `run.sh experiment` with one of the
shipped arms (a bare file name is looked up under `configs/experiment/`):

```bash
R=./examples/pi05/libero_shared_autonomy/run.sh
$R experiment --config flow_reversal_rotz20.yaml --dry-run   # validate + print the schedule, run nothing
$R experiment --config flow_reversal_rotz20.yaml             # one block of n_trials
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

### Arms, overrides and sweeps

`configs/experiment/base_rotz20.yaml` holds everything the study shares, including the
known corruption (`corruption: {rotation_z_deg: 20}`: the operator's command is
rotated 20° about z before the policy sees it). Three arm files extend it:

- `flow_control_rotz20.yaml` — shared flow control; the operator's command is
  written into the first `n_guided_steps` denoising steps of every chunk.
- `flow_reversal_rotz20.yaml` — native Flow Reversal Steering; the reference
  chunk is reversed `n_reversal_steps` of the way to noise (`null` = all the way).
- `flow_reversal_rotz20_adapted.yaml` — Flow Reversal Steering with the
  reversal adapter, our addition: the velocity field used by the reversal is
  reshaped by `F`, whose translation block is the known corruption and whose
  orientation and gripper blocks are identity, so the only difference from
  the native arm is the rotation applied to the reversal's translation.

The depth of each method is a sweep variable rather than a file. `--set KEY=VALUE`
overrides any YAML key (dotted path, value parsed as YAML) and `--sweep KEY=V1,V2,...`
runs one block of `n_trials` per value with a single policy load, waiting for you
to click **Start trial** before each trial:

```bash
R=./examples/pi05/libero_shared_autonomy/run.sh
$R experiment --config flow_control_rotz20.yaml           --sweep control.n_guided_steps=2,4,6,8,10
$R experiment --config flow_reversal_rotz20.yaml          --sweep control.n_reversal_steps=2,4,6,8,10
$R experiment --config flow_reversal_rotz20_adapted.yaml  --sweep control.n_reversal_steps=2,4,6,8,10

$R experiment --config flow_control_rotz20.yaml --set control.corruption=null      # a clean run
$R experiment --config flow_reversal_rotz20.yaml --set control.n_reversal_steps=5 --set experiment.seed=1
```

Each block gets its own run directory named after the file and the overrides in
force, e.g. `20260903_101500_flow_reversal_rotz20_n_reversal_steps-4/`, and its
`config.yaml` lists the overrides under `overrides`. Keys that would need a
different policy (`policy.*`, `server.port`) cannot change between blocks.

### Configuration

Everything lives in the YAML file you pass to `--config`. Unknown keys are
rejected and a key set to `null` keeps the built-in default, so a typo fails
immediately instead of silently doing nothing. Only the keys you want to
change need to be present; the run's name is the config file's stem plus any
`--set`/`--sweep` values (no `experiment.name` key).

| Key                        | Default                           | Meaning                                                                                                                                                                 |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.n_trials`      | `10`                              | Number of trials                                                                                                                                                        |
| `experiment.seed`          | `0`                               | Seeds task order, per-task initial-state permutations, and independent environment, policy, synthetic-operator and input-noise streams |
| `experiment.task_order`    | `random`                          | `random` (uniform, with replacement), `shuffled` (permuted blocks, each task once per block), `sequential` (cycle in order)                                             |
| `experiment.output_dir`    | `outputs/pi05_libero_experiments` | Parent of the run directory                                                                                                                                             |
| `scene.suite`              | `libero_goal`                     | LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`)                                                                 |
| `scene.task_ids`           | `null`                            | Task pool to draw from; `null` = every task in the suite                                                                                                                |
| `prompt`                   | `"do something"`                  | Instruction handed to the VLA; the literal `task` uses the scene's own instruction                                                                                      |
| `policy.path`              | `lerobot/pi05_libero_finetuned`   | Hub id or local checkpoint directory                                                                                                                                    |
| `policy.n_action_steps`    | `10`                              | Actions executed per predicted chunk                                                                                                                                    |
| `policy.compile`           | `false`                           | `torch.compile` the model (slow first trial, faster after)                                                                                                              |
| `control.mode`             | `shared_flow_reversal_steering`   | `policy`, `teleop`, `shared_override`, `shared_flow_control`, `shared_flow_reversal_steering` — see the mode list under [Interactive prompting](#interactive-prompting) |
| `control.n_guided_steps`   | `8` in `base_rotz20.yaml` (built-in `5`) | `shared_flow_control` only                                                                                                                                              |
| `control.n_reversal_steps` | `null`                            | `shared_flow_reversal_steering` only: denoising steps the reference is reversed through; `null` = all the way to noise                                                  |
| `control.input_noise`      | `0.0`                             | Std of the Gaussian noise added to your x/y/z command                                                                                                                   |
| `control.max_steps`        | `null`                            | Rollout length; `null` = the suite's own episode length                                                                                                                 |
| `control.corruption`       | `null`                            | Matrix spec applied to the operator's command — see [Perturbing the operator](#perturbing-the-operator); off unless set                                                 |
| `control.reversal_adapter` | `null`                            | Adapter spec for `shared_flow_reversal_steering`'s reverse integration — see [Perturbing the operator](#perturbing-the-operator); off unless set                        |
| `control.operator_dims`    | `[translation]`                   | Which dim groups the operator commands (`translation`, `rotation`, `gripper`); the rest of the reversed reference is delegated to the policy                            |
| `operator`                 | `human`                           | `human` (SpaceMouse / keyboard) or `policy`: the task-prompted policy drives, corrupted exactly like a person — see [Synthetic operator](#synthetic-operator-and-the-oracle-sweep) |
| `policy_operator.update_every_steps` | `null` | Environment steps between synthetic intent queries; null uses `policy.n_action_steps`. Requires `operator: policy` |
| `policy_operator.delay_steps` | `0` | Environment steps from observing the scene to delivering that synthetic command |
| `server.port`              | `8765`                            | Live view port                                                                                                                                                          |

A few keys can be overridden per run without editing the file:
`--n-trials`, `--seed`, `--mode`, `--output-dir`, `--port`.

### Running a session

Experiments reset once per trial, using its explicit initial-state ID and
environment seed. The preview and rollout use the same reset. Assignments are
independent of depth-block order and previous successes. Inference noise is
seeded separately for the operator and executing policy at each chunk boundary.
FC and FRS both consume one corrupted/noisy command per action chunk.
The single `seed` controls reproducibility: each task draws from a seeded
permutation of its available initial states, reshuffling after a complete pass.
Different seeds can reuse states; they do not automatically define disjoint
development/evaluation sets.

For each trial the terminal prints the trial number, the scene, **your** task
and the VLA prompt. The live view shows the reset scene with your task on its
own line and the VLA prompt beneath it; both stay up for the whole trial, so
nothing changes on screen when the rollout actually begins. Two buttons appear
under the status line while the run waits for you: **Start trial** starts the
rollout and **Skip trial** skips it without recording anything. The buttons are
only shown while a trial is waiting, so a click at any other time does nothing.
The terminal takes no input; `Ctrl+C` there ends the run, keeping everything
recorded so far. Drive with the SpaceMouse and/or the keyboard exactly as in
the interactive runner — clicking a button also gives the page keyboard focus.
The rollout ends on success, on the episode limit, or at `control.max_steps`.

### What gets recorded

Each run creates `<output_dir>/<YYYYmmdd_HHMMSS>_<config stem>/` containing:

| File            | Contents                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`   | The resolved configuration, the trial schedule, and the corruption / adapter **matrices themselves** — the run is self-describing even if you later edit those YAML files |
| `trials.jsonl`  | One JSON object per trial, appended as it finishes                                                                                                                        |
| `trial_XXX.npz` | Per-step arrays for trial `XXX`                                                                                                                                           |
| `trial_XXX.mp4` | The rollout video (30 fps)                                                                                                                                                |

`trials.jsonl` fields: `trial`, `suite`, `task_id`, `task_description`,
`vla_prompt`, `operator`, `operator_dims`, `mode`, `n_guided_steps`, `n_reversal_steps`,
`input_noise`, `corruption`, `reversal_adapter`, `success`, `mode_expressed`
(the scene element that moved most during the trial — an object name or an
articulation joint such as `wooden_cabinet_1_middle_level` — or `null` when
nothing moved; the behaviour the policy actually expressed, whether or not it
was the task), `moved` (every element that moved, with its displacement in m
or joint units), `steps`, `duration_s`, `user_reads`
(how many times the teleop input was sampled during that trial),
`video`, `steps_file`, `finished_at`, plus the mode's steering statistics
(`guided_steps`, or `steered_chunks` and
`reconstruction_error_mean`). `corruption` and
`reversal_adapter` hold a short label of the configured spec (e.g.
`rotation_z_deg=20`) or the loaded file's name, and `null` when off.

Corrected runs also record `init_state_id`, `env_seed`, `policy_seed`,
`operator_seed`, `input_noise_seed`, `pair_id` and `initial_state_hash` in
both the trial record and `.npz`. The hash fingerprints settled simulator
positions, velocities and actuator state, rounded to eight decimal places.
`steered_chunk_velocity_evaluations` reports the executing policy's cost per
steered chunk, excluding the operator. Run configs carry `implementation_version: 2`.

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
| `object_names`, `object_pos`                                   | `(K,)`, `(T, K, 3)` | Every scene object / fixture and its position before each step (the state the action was applied to)                          |
| `articulation_names`, `articulation_qpos`                      | `(J,)`, `(T, J)`   | Every non-robot slide / hinge joint (drawers, knobs, doors) and its value before each step                                      |
| `task_description`, `vla_prompt`, `mode`, `task_id`, `success` | scalars            | Trial identity, so a single `.npz` stands alone                                                                                 |

Missing robot-state fields are recorded as `NaN` rather than aborting a trial.
Loading a trial is just:

```python
import json, numpy as np

trials = [json.loads(line) for line in open(run_dir / "trials.jsonl")]
z = np.load(run_dir / "trial_000.npz")
z["user_translation_raw"], z["action"], z["eef_pos"], bool(z["success"])
```

### Synthetic operator and the oracle sweep

`operator: policy` (or `--set operator=policy`) replaces the SpaceMouse and
keyboard with the policy itself: by default, at every chunk boundary the same checkpoint is
queried with the scene's **real** instruction, the translation of its planned
chunk (mean over the executed prefix, clipped like a stick) becomes the
operator's command, and that command goes through the same corruption, noise
and recorders as a person's. The executing policy still only sees the arm's
prompt. Trials start without waiting for the button and the 20 Hz pacing is
dropped, so a block runs unattended. `user_translation_raw` is the consumed
operator command before corruption/noise and `user_translation` what the policy got.

`oracle_sweep.sh` runs the whole grid this way — the three arms swept over
depth on every LIBERO-Goal task, plus two policy-only anchors per task (the
arm's prompt = floor, the real task = ceiling):

```bash
./examples/pi05/libero_shared_autonomy/oracle_sweep.sh                         # everything, ~a few hours
TASKS="0 2" DEPTHS=4,10 N_TRIALS=5 ./examples/pi05/libero_shared_autonomy/oracle_sweep.sh
```

Corrected runs land in `outputs/pi05_libero_oracle_sweep_v2/`;
[`notebooks/analyze_sweep.ipynb`](notebooks/analyze_sweep.ipynb) compares the
methods on them (success vs. controlled steps, per-task breakdown, whether the
expressed behaviour was the task's, and how much of the intent got through).
Set the notebook's `ROOT` to that new directory and rerun its cells. Historical
notebook outputs describe the old implementation and must be regenerated.
`load_runs` rejects pooling corrected and legacy runs, and `paired_test` rejects
legacy runs without verified pairing metadata.

### SpaceMouse-like timing: the next experiment

The corrected oracle results select FC depth 10 and FRS depth 1.
One hypothesis is that these settings work well because the current operator is
a task-aware policy giving precise continuous commands with no reaction delay.
The results do not establish that corruption severity is the limiting factor.
Test the operator's timing before increasing corruption or tuning another adaptor.

The synthetic operator now supports continuous direction and magnitude with
slower intent updates and delayed delivery:

```yaml
operator: policy
policy_operator:
  update_every_steps: 20
  delay_steps: 4
```

At 20 environment steps/second, this proposes new intent once per second and
delivers it 0.2 seconds after observing the scene. It holds the preceding command
until delivery, starts idle with the gripper open, and resets its queue every
trial. Both translation and gripper are delayed; the current xyz-only arms
consume only translation. Directions are continuous, without axis quantization.
This models **intent update timing**, not the hardware's sampling rate, and the
numbers are experimental assumptions rather than a validated human model. The
underlying policy still supplies privileged task knowledge and precise directions.

FC and FRS consume commands every 10 environment steps by default. A command
generated at step 0 and delivered at step 4 is first consumed at step 10; the
robot therefore sees a 0.5-second delay. Delays 1 through 10 have the same
consumption schedule when queries are aligned to chunk boundaries. Avoid a dense
sub-chunk delay sweep: it repeats equivalent conditions. Changing the action
chunk length is a separate experiment, since it also changes policy feedback.

Use a factorial timing ablation, keeping the +20-degree corruption and F@20 fixed:

| Profile | Update steps | Delay steps | Purpose |
| --- | ---: | ---: | --- |
| Immediate oracle | 10 | 0 | Existing reference |
| Slower intent | 20 | 0 | Isolate held/stale intent |
| Delayed intent | 10 | 4 | Isolate reaction delay |
| Slower + delayed | 20 | 4 | Initial SpaceMouse-like profile |

For each new profile, compare FC, native FRS, and FRS+F@20 at depths 2/4/6/8/10.
Depth 1 is excluded from the next experiment; its oracle optimum need not carry
over to human input. Start with 5 paired trials on each of all ten
tasks, seed 0. This is 850 trials per profile including the two anchors. Existing
reference runs contain those first five seeded trials per task; select that same
subset when comparing timing profiles. Do not select an input impairment merely
because it produces an FRS win.

```bash
S=./examples/pi05/libero_shared_autonomy/oracle_sweep.sh
FC_DEPTHS=2,4,6,8,10 FRS_DEPTHS=2,4,6,8,10 N_TRIALS=5 SEED=0 \
  OUT=outputs/pi05_libero_spacemouse_u20_d4 "$S" \
  --set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4 --dry-run
# Remove --dry-run to collect. Repeat with (20,0) and (10,4), each in its own OUT.
```

Keep profiles in separate output roots; `load_runs` rejects pooling different
synthetic timing profiles and adds an `operator_profile` column. The resolved
timing settings are stored in `config.yaml` and each JSON trial record. New NPZ
arrays support diagnostics:

- `oracle_translation`: latest computed task-policy command, even if still awaiting
  delivery. This is not a fresh oracle query on every recorded frame.
- `operator_translation`: currently delivered source command before corruption.
- `operator_command_age_steps`: age since that delivered command was generated,
  or -1 before the first delivery. Age is evaluated before the recorded action.
- `operator_query_count`: cumulative queries in the trial.

The delivered source can change between policy chunk boundaries. Use
`user_translation_raw` and `user_reads` to identify what was actually consumed;
the source diagnostics alone do not identify the command behind a queued action.

Compare success, completion steps, task breakdowns, query counts, and videos at
grasp/contact transitions. If FC's preferred depth decreases or FRS's increases,
that supports the excessive-oracle-authority hypothesis; neither shift nor an
FRS win is guaranteed. A static rotation adaptor cannot directly repair stale
intent, so timing robustness and adaptor benefit must be measured separately.
After the pilot, check these timing assumptions against recorded SpaceMouse use,
then freeze each method's best configuration and confirm on new seeds with paired
resets. New seeds can revisit the same physical states; account for that in uncertainty.

### Experiments after the implementation corrections

The baseline, angle ablation, and confirmation below have now been collected;
in the corrected output directories. These commands document that study. The timing
ablation above is the next priority.

First rerun all ten tasks under the original 20-degree corruption, with a fresh
depth sweep. The old best depth may change now that forward integration uses
the correct step size. The full grid has 2,100 trials (21 conditions × 10 tasks × 10 resets):

```bash
S=./examples/pi05/libero_shared_autonomy/oracle_sweep.sh
FC_DEPTHS=2,4,6,8,10 FRS_DEPTHS=1,2,3,4,6,8,10 N_TRIALS=10 SEED=0 \
  OUT=outputs/pi05_libero_oracle_sweep_v2 "$S" --dry-run
# Run the same command without --dry-run to collect the corrected baseline.
```

`FC_DEPTHS` and `FRS_DEPTHS` independently choose method depths; `DEPTHS` remains
a fallback for both. `SEED` controls initial-state selection and stochastic
replicates. Every method receives the same assignments. Additional
arguments are passed to `experiment`, so `--dry-run` validates without loading a policy.

Next, select a shallow FRS depth on these development states and test adaptor
angles `-20, 0, 10, 20, 30, 40` with the corruption held at +20 degrees. For example,
for depth 2 (replace it if the corrected sweep selects a different depth):

```bash
R=./examples/pi05/libero_shared_autonomy/run.sh
"$R" experiment --config flow_reversal_rotz20_adapted.yaml \
  --set operator=policy --set experiment.task_order=sequential --n-trials=100 \
  --set control.n_reversal_steps=2 --seed=0 \
  --sweep control.reversal_adapter.translation.rotation_z_deg=-20,0,10,20,30,40 \
  --output-dir=outputs/pi05_libero_adapter_angles_v2
```

Keep angle variants separate in analysis: group by `run` and use
`paired_test(trials, "run", run_a, run_b)`; pooling every angle under `FRS+F`
would obscure which configuration is being evaluated. Add a clean-command control
for every arm (`--set control.corruption=null`; adapted FRS also needs
`--set control.reversal_adapter.translation=identity`).

Then freeze the selected FC depth, FRS depth and adaptor angle. Compare the
selected configurations with multiple new seeds and 20 trials per task.
For the original +20-degree adaptor, the confirmation grid can be run as:

```bash
# Substitute the depths selected on development states before running.
FC_DEPTHS=10 FRS_DEPTHS=2 N_TRIALS=20 SEED=1 \
  OUT=outputs/pi05_libero_confirm_v2 "$S"
```

Repeat with other seeds, leaving the selected configurations fixed. For a tuned
adaptor angle, invoke its selected `experiment` command separately with the same
seed and per-task trial counts. Compare best FC against best FRS+F using
`paired_test(selected_trials, "method", "FC", "FRS+F")`, which pairs by `pair_id`.
With multiple seeds, do not pass only `(task_id, trial)` as pairing keys. Seeds
can reuse initial states: also report uncertainty grouped by task/reset, rather than
treating every stochastic replicate as a new independent scene.

After establishing these corrected baselines, test guidance horizons of 10, 20
and 50 steps and coherent policy-based reference chunks. These are algorithm
ablations requiring additional configuration/code; the fixes preserve FRS's
existing executed-prefix delegation. Review failures on tasks 0, 2, 3 and 9,
while retaining all ten tasks in the final evaluation.

## Analysis

[`notebooks/analyze_experiments.ipynb`](notebooks/analyze_experiments.ipynb)
imports [`notebooks/analyze.py`](notebooks/analyze.py), which loads every run
directory that contains a `trials.jsonl` and reports success rate with Wilson 95%
intervals, a paired per-trial comparison, a per-task breakdown, step counts
split by outcome, and how much the operator actually steered. Point `RUN_DIRS`
at other directories to compare anything else. Its functions:

- `find_runs(root)` / `load_runs(run_dirs)` — the run directories under
  `root` that recorded a trial, and their `trials.jsonl` + `config.yaml` as one
  `DataFrame` (each trial tagged with the run's method, depth and operator)
  plus a dict of configs.
- `rate_table(trials, by)` — trial and success counts, rate and Wilson 95%
  interval per group of columns (`["run"]`, `["method", "depth"]`, ...);
  `success_table(trials, configs)` is the per-run view with labels and median
  steps, `best_depth` and `per_task_table` the per-method views.
- `paired_test(trials, by, a, b, pair_on=None)` / `paired_comparisons` — McNemar's
  exact test on matching `pair_id` values and verified identical initial-state
  hashes. Explicit keys are also verified; duplicate or unverified pairs fail.
- `task_targets` / `with_mode_accuracy` — which scene element each task is
  about (from the ceiling anchor) and whether each trial expressed it.
- `step_metrics(trials)` — per-trial operator engagement and intent
  transmission from the `.npz` step arrays (fraction of steps commanding,
  speed, corruption shift, gripper, reads, and the cosine between the executed
  translation and the raw vs. the corrupted command).
- `plot_performance_vs_depth(trials, ax, metric)` — one line per method,
  rate of `metric` against controlled denoising steps.

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
- **`shared_flow_reversal_steering` behaves oddly** — check `adapter`; a
  non-identity `F` changes the reversal. `adapter off` restores the plain
  method.
- **Port already in use** — pass `--port`.
- **First rollout very slow with `--compile`** — expected; compilation takes a
  few minutes and the `No valid triton configs` messages are harmless.
- **Rendering errors** — try `MUJOCO_GL=osmesa` or make sure EGL drivers are
  installed.
