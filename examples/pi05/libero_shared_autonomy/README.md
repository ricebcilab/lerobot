# pi0.5 + LIBERO shared autonomy

A reproducible local setup for LeRobot's Pi0.5 policy on the vanilla LIBERO
benchmark, plus a shared-autonomy study: a person (or a synthetic operator)
steers the policy through its flow-matching denoising, and the study compares
how much of the operator's intent each steering method lets through. It uses
the published
[`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
checkpoint by default.

```text
env.sh                               shared environment (sourced by run.sh)
run.sh                               shared launcher: setup | interactive | experiment | <command>
configure_libero.py                  one-time LIBERO configuration (run by `run.sh setup`)
config.py                            YAML/CLI settings for both runners
session.py                           one loaded policy + LIBERO scene + operator input + live view
teleop.py                            SpaceMouse/keyboard readers, corruption, noise, the synthetic operator
live_view.py                         the browser page: camera stream, status, keyboard, trial buttons
reproducibility.py                   per-trial seeds and initial states
interactive.py                       the REPL
experiment.py                        the scripted trial runner
replay.py                            adds policy_translation to runs recorded before it was recorded live
configs/interactive.yaml             REPL settings
configs/experiment/                  the study's arms (see Experiments)
experiments/arms.sh                  the arms (fc, frs, frs_ra, floor, ceiling) and how each is launched
experiments/oracle_sweep.sh          synthetic-operator study
experiments/task_audit.sh            picks the human-pilot task (oracle_sweep.sh with the SpaceMouse-like operator)
experiments/human_pilot.sh           the human study
notebooks/analyze.py                 loading, tables, paired tests, per-trial metrics, plots
notebooks/analyze_human_pilot.ipynb  the human study's figures
```

The steering methods themselves live in
[`lerobot.policies.pi05.steering`](../../../src/lerobot/policies/pi05/steering.py).

## Install

From the repository root:

```bash
./examples/pi05/libero_shared_autonomy/run.sh setup
```

This installs the locked `pi` and `libero` extras into `.venv`, writes a
non-interactive LIBERO config under `.cache/libero`, downloads
`lerobot/libero-assets` there and links the installed `hf-libero` package to
those project-local assets. Both `.venv` and `.cache` are ignored by git;
re-running setup is safe.

`run.sh` is the launcher for everything else: `run.sh interactive`,
`run.sh experiment`, or `run.sh <any command>` (such as `lerobot-eval` or
`jupyter`) inside the example's environment. Every form sources `env.sh`, which
sets `LIBERO_CONFIG_PATH`, `MUJOCO_GL=egl`, `HF_HUB_CACHE` and `MPLCONFIGDIR`
to paths under `.cache/` unless you have set them yourself, and runs from the
repo root under `uv run` with the `pi` and `libero` extras. `HF_HOME` is left
alone so your `hf auth login` credential applies.

Pi0.5 needs a CUDA GPU with a working driver; it is too large for a practical
CPU rollout. Its PaliGemma tokenizer is gated: accept the license on the Hub
and run `hf auth login` once. Every process re-checks the tokenizer against the
Hub at start-up, so the token must be valid — an expired one fails with a 401 —
or, with everything cached, set `HF_HUB_OFFLINE=1` to skip the check.

## Smoke evaluation

Batch evaluation is plain `lerobot-eval` run through `run.sh`. This evaluates
task 0 of LIBERO-Spatial for one episode and downloads the checkpoint on first
use:

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

The first rollout may spend several minutes compiling the model; pass
`--policy.compile_model=false` to skip that. For the four standard suites with
10 episodes per task, drop `--env.task_ids` and set
`--env.task=libero_spatial,libero_object,libero_goal,libero_10 --eval.n_episodes=10`.
`--policy.path` accepts a local checkpoint directory as well as a Hub id.

## Interactive prompting

```bash
./examples/pi05/libero_shared_autonomy/run.sh interactive
```

loads the policy once, opens the live view (`http://localhost:8765`) in your
browser when a display is available — VSCode forwards the port — and drops into
a REPL. The page shows the camera stream, the policy's per-step action vector
(end-effector deltas and gripper command) and, in the teleop modes, the held
keys and gripper state. A tab already polling the port is reused rather than
opened again, and it reconnects by itself when the next run on that port
starts, so a chain of runs keeps a single tab.

Type an instruction (or press Enter for the scene's built-in one) to run a
rollout in the current scene; each rollout is also saved as an MP4 under
`outputs/pi05_libero_interactive/`. `tasks` lists the suite's scenes,
`task <suite> <id>` switches scene, `quit` exits. Settings can also come from
`configs/interactive.yaml` (`run.sh interactive --config`); the command
reference is `interactive.py --help`.

Besides the default `mode policy` (model-only control), four teleop modes let
you drive or steer the arm with a 3Dconnexion SpaceMouse and/or the keyboard.
The policy denoises each action chunk in 10 flow-matching steps; the two
shared-autonomy modes differ in where your command enters that schedule.

- `mode teleop` — you drive x/y/z and the gripper; the model is not involved.
- `mode shared_override` — pi0.5 drives, but your x/y/z replaces the model's
  translation in the executed action.
- `mode shared_flow_control [tau] [lambda]` (**FC**, shared flow control, Kao
  et al. 2026) — while you are pushing, your translation, scaled by `ic_gain`
  into the noise prior's units (a full push writes `ic_gain` sigma; `action`
  writes it in the checkpoint's action units instead), is written into the
  translation dims of the first `tau` of the chunk's 50 steps of `x_t` at the
  start of each of the first `lambda` of the 10 denoising steps; the flow itself
  runs unmodified. `lambda` 1 (the default) writes the initial condition only,
  Kao et al.'s method; 0 is the policy alone; more re-writes as the flow
  proceeds, which is what this example's earlier FC did with its `n_guided_steps`
  (that name is still accepted for `lambda`). `tau` null writes the whole chunk;
  with `lambda` 1 and `ic_gain: action` that is the arm the human pilot ran. The
  executed action is entirely the model's output.
- `mode shared_flow_reversal_steering [n]` (**FRS**, Flow Reversal Steering,
  [Tang et al. 2026](https://arxiv.org/abs/2606.13675)) — while you are
  pushing, a reference chunk that servos in your direction at uniform velocity
  is integrated _backward_ through the policy's own velocity field for `n` of
  the 10 steps; the forward flow then runs from that latent instead of random
  noise. With all 10 steps (the default) this is the paper's method: the
  reference goes all the way to noise and you get the policy's action mode
  nearest your intent. Any `n < 10` is a **partial reversal**, this example's
  own variant that the paper does not have: the reference is taken to
  `t = n/10` and denoised back in `n` steps, so most of what is executed is the
  reference itself (about 85% of the command at `n = 1`). Only what you command
  survives the reversal: the rotation and gripper dims, the padding dims and
  every step beyond the executed prefix are delegated to the policy — fresh
  noise for a full reversal, as in Tang et al., App. D, and the unsteered flow's
  state at `t = n/10` for a partial one.
  A steered chunk costs `10 + n` velocity evaluations against 10 for FC or an
  unsteered chunk. After each rollout the REPL prints how many chunks were
  steered and how far the executed translation landed from the reference (in
  action-std units). The mechanics are documented in the
  [steering module](../../../src/lerobot/policies/pi05/steering.py).

Idle input means pure policy in every shared mode. Input sources (both active
at once; the SpaceMouse wins while deflected):

- **SpaceMouse** — read directly from `/dev/hidraw`, no extra install; x/y/z
  from the stick, any button toggles the gripper. Connected the first time you
  switch to a teleop mode; without one, the mode works with the keyboard alone.
- **Keyboard** — captured by the live-view page (click it to give it focus):
  <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> move forward/back/left/right,
  <kbd>PgUp</kbd>/<kbd>PgDn</kbd> (or <kbd>W</kbd>/<kbd>S</kbd>) move up/down,
  <kbd>Space</kbd> toggles the gripper, hold <kbd>Shift</kbd> for full speed
  (keys move at half speed by default). Arrow directions follow the SpaceMouse
  axis tuning in `teleop.py`, so `↑` moves the arm like pushing the stick
  forward.

## Perturbing the operator

Three ways to make the operator's command less than perfect, and one to keep
what is left of it undistorted:

- **Input noise** — `--input-noise STD` (or `noise STD` in the REPL) adds
  independent Gaussian noise to Δx, Δy and Δz whenever the command is sampled
  while you are commanding (once per chunk in FC/FRS, once per control step in
  teleop/override); `0` (the default) is off.
- **Command corruption** — `control.corruption` in a config (or
  `--corruption FILE` / `corruption FILE` in the REPL) left-multiplies your
  Δx/Δy/Δz by a fixed 3×3 matrix `M` before it reaches the policy, or runs it
  through a curl field:

  ```yaml
  corruption: null                      # off
  corruption: {rotation_z_deg: 20}      # rotate the commanded direction about z
  corruption: {scale: [1, 1, 1]}        # per-axis gain
  corruption: {M: [[..3 rows..]]}       # literal (bare rows work too)
  corruption: {random: {rotation_deg: [30, 90], scale: [1, 4]}}   # experiments only: a new M on every trial
  corruption: {curl: 20, tau: 0.1}      # a curl field (1/s, + = clockwise in x-y) on a command with inertia tau (s)
  ```

  The curl field is the cursor project's dynamic corruption: the served command
  is a velocity driven by your push, `dv/dt = (u − v)/tau + curl·R·v`, stepped
  once per control step (`dt`, default 0.05 s) from rest at every trial. A held
  push comes out turned clockwise by `atan(tau·curl)` and slowed by
  `1/sqrt(1 + (tau·curl)^2)`; the field keeps evolving while you are idle, but an
  idle push is served as is. A curl never lengthens a command, so `input_gain:
auto` is 1.

  The `random` form draws `M = R · diag(gains)` afresh for every trial: each
  operator axis gets its own gain, log-uniform in `scale`, and the result is
  rotated by an angle uniform in `rotation_deg` about an axis uniform in 3D
  (`[0, 0]` or `[1, 1]` switches a part off). The draw depends on the trial's
  reset alone, so every arm and depth sees the same `M` on the same trial, and
  an adapter with `translation: corruption` is rebuilt from it, keeping
  `F = M` trial by trial. The participant is not shown the draw; `--dry-run`
  lists it per trial.

- **Reversal adapter (MARS)** — `control.reversal_adapter` (or
  `--reversal-adapter FILE` / `adapter FILE` in the REPL) only affects
  `shared_flow_reversal_steering`; FRS with an adapter is **MARS** (Modular
  Adapters for Reversal Steering, the method this example used to call FRS-RA).
  An adapter has up to two modules: a **user module**, which multiplies the
  velocity field of the reverse integration only (`x_t += h · U · v`), so only
  the latent the forward flow starts from changes, and a **task module**, an
  affine map of the velocity in both directions, which reshapes every chunk the
  policy produces. The user module is a fixed 7×7 **environment-space** matrix
  `F` or a learned `U(x_t, t)` (see [Training a MARS user module](#training-a-mars-user-module)).
  Nothing is inverted by default: `F` is not `M⁻¹`, and a rotation adapter
  rotates the reversal field in the stated sense; `corruption_inverse` is there
  for the curl field, whose steady-state map stands in for `M`.

  ```yaml
  reversal_adapter: null                              # plain FRS
  reversal_adapter:                                   # MARS-U, fixed F
    translation: corruption                           # copy M from control.corruption (error if that is null)
    orientation: zero                                 # identity | zero (missing blocks are identity)
    gripper: zero
  reversal_adapter: {translation: corruption_inverse} # M^-1 (a curl's steady-state map inverted)
  reversal_adapter:
    translation: {rotation_z_deg: 40}                 # or {scale: [...]}, or a 3x3 literal
  reversal_adapter: {F: [[..7 rows..]]}               # literal
  reversal_adapter: {learned: outputs/user_modules/rotz20/user.pt}      # MARS-U, learned
  reversal_adapter: {user: {learned: u.pt}, task: t.pt}                 # MARS: user + task module
  reversal_adapter: {user: {translation: corruption}, task: t.pt}       # a fixed F with a task module
  ```

  Before applying `F` to normalized velocities the wrapper converts it to
  `S⁻¹ F S`, where `S` is the diagonal matrix of action standard deviations
  (velocities have no mean offset); learned modules work in the normalized space
  directly. The matrices and the module paths are saved in the run config.

- **Input gain** — `control.input_gain` turns the operator's command down so
  that the corruption can never push it into the env's `[-1, 1]` clip. The
  command is first limited to unit length, keeping its direction (a stick
  pushed on several axes at once can otherwise reach √3), and then multiplied
  by the gain. `auto` picks the largest gain that cannot clip, 1 / the most the
  corruption can lengthen a command (its largest singular value, or the top of
  a random family's `scale`); a number in `(0, 1]` is accepted only if it
  cannot clip either. `null` (the built-in default) leaves the command as it
  is, clip included; every shipped config sets `auto`, and in the REPL an
  `auto` gain follows each `corruption` command. The price is speed: with gains up to 4 the gain is 0.25, so an
  unscaled axis tops out at a quarter of full speed, and the idle threshold
  (0.05 in env units) then sits at a fifth of the stick's travel. The recorded
  raw command is the one after the gain.

A file loaded with `--corruption FILE` / `--reversal-adapter FILE` (or live in
the REPL) holds the same spec forms at its top level (the `random` form needs
trials, so the REPL rejects it).

## Experiments

`run.sh experiment` runs scripted, recorded trials from a YAML config with the
same policy, live view and modes as the REPL. Each trial draws a task from the
configured scene, shows **you** the task, and lets you attempt it with the VLA
in the configured shared-autonomy mode. A bare file name is looked up under
`configs/experiment/`:

```bash
R=./examples/pi05/libero_shared_autonomy/run.sh
$R experiment --config flow_reversal_rotz20.yaml --dry-run   # validate + print the schedule, run nothing
$R experiment --config flow_reversal_rotz20.yaml             # one block of n_trials
```

### The prompt vs. the task

The **human** is always shown the scene's real instruction ("put the bowl on
top of the cabinet"). The **VLA** only ever receives the config's `prompt`,
which defaults to the deliberately uninformative `"do something"` — the policy
contributes manipulation priors while the operator supplies the intent through
the steering channel. `prompt: task` gives the VLA the scene's own instruction
instead, the conventional setup where both know the goal.

### Arms, overrides and sweeps

`configs/experiment/base_rotz20.yaml` holds everything the study shares,
including the known corruption (`corruption: {rotation_z_deg: 20}`: the
operator's command is rotated 20° about z before the policy sees it). The arm
files extend it:

- `flow_control_rotz20.yaml` — **FC**, shared flow control (Kao et al.): `tau`
  null, `lambda` 1, `ic_gain: action`. Its depth is `tau`, in chunk steps.
- `flow_reversal_rotz20.yaml` — **FRS**, Flow Reversal Steering without an
  adapter. Swept over `n_reversal_steps`: only `n = 10` is the published method,
  every `n < 10` cell is a partial reversal (see [Interactive prompting](#interactive-prompting)).
- `mars_rotz20.yaml` — **MARS** with a fixed user module (MARS-U, formerly
  FRS-RA): `F`'s translation block is the known corruption `M` and its other
  blocks are identity, so the only difference from FRS is `M` applied to the
  reversal's translation. The study scripts always run it at `F = M`.
- `mars_learned_rotz20.yaml` — **MARS** with a learned user module
  (`{learned: outputs/user_modules/rotz20/user.pt}`, trained as below); the
  file is a template until that checkpoint exists.
- `policy_anchor.yaml` — the policy alone: with the arm's prompt the **floor**
  of every comparison, with `--set prompt=task` its **ceiling**.
- `base_scale.yaml` and `flow_control_scale.yaml`, `flow_reversal_scale.yaml`,
  `mars_scale.yaml` — the same three arms under a per-axis **gain**
  instead of a rotation (`corruption: {scale: [2, 1, 1]}`: the operator's x
  command is twice as sensitive as y and z). A separate family is needed because
  a `--set` on `control.corruption` merges into the file's mapping, so a scale
  cannot replace a rotation from the command line; the gain vector itself can be
  overridden (`--set 'control.corruption={scale: [1, 2, 1]}'`). The study
  scripts pick this family with `CORRUPTION=scale` (see `experiments/arms.sh`).
- `base_random.yaml` and `flow_control_random.yaml`, `flow_reversal_random.yaml`,
  `mars_random.yaml` — the same three arms under a corruption drawn
  **per trial** (`corruption: {random: {rotation_deg: [30, 90], scale: [1, 4]}}`)
  so nothing is ever clipped. `CORRUPTION=random` selects it; the ranges can be overridden
  (`--set 'control.corruption={random: {scale: [1, 2]}}'`).

The depth of each method (FC's `tau`, the reversal arms' `n_reversal_steps`) is a sweep variable rather than a file. `--set KEY=VALUE`
overrides any YAML key (dotted path, value parsed as YAML) and `--sweep KEY=V1,V2,...`
runs one block of `n_trials` per value with a single policy load:

```bash
$R experiment --config flow_control_rotz20.yaml           --sweep control.tau=5,10,25,50
$R experiment --config flow_control_rotz20.yaml           --sweep control.ic_gain=1,2,3,5    # Kao's gain, once per condition
$R experiment --config flow_reversal_rotz20.yaml          --sweep control.n_reversal_steps=1,2,4,6,8
$R experiment --config mars_rotz20.yaml                   --sweep control.n_reversal_steps=1,2,4,6,8
$R experiment --config flow_control_rotz20.yaml --set control.corruption=null      # a clean run
```

`experiments/arms.sh` packages exactly these launches as
`run_arm fc|frs|mars|mars_learned|floor|ceiling <depths> [flags]` (`frs_ra` is
still accepted for `mars`), so the study scripts
share one arm → config → sweep-key mapping. Each block gets its own run
directory named after the file and the overrides in force, e.g.
`20260903_101500_flow_reversal_rotz20_n_reversal_steps-4/`, and its
`config.yaml` lists the overrides under `overrides`. Keys that would need a
different policy (`policy.*`, `server.port`, `operator`) cannot change between
blocks.

### Configuration

Everything lives in the YAML file you pass to `--config`. Unknown keys are
rejected and a key set to `null` keeps the built-in default, so a typo fails
immediately instead of silently doing nothing. The run's name is the config
file's stem plus any `--set`/`--sweep` values.

| Key                                  | Default                           | Meaning                                                                                                                                                                                      |
| ------------------------------------ | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.n_trials`                | `10`                              | Number of trials                                                                                                                                                                             |
| `experiment.seed`                    | `0`                               | Seeds task order, per-task initial-state permutations, and independent environment, policy, synthetic-operator and input-noise streams                                                       |
| `experiment.task_order`              | `random`                          | `random` (uniform, with replacement), `shuffled` (permuted blocks, each task once per block), `sequential` (cycle in order)                                                                  |
| `experiment.output_dir`              | `outputs/pi05_libero_experiments` | Parent of the run directory                                                                                                                                                                  |
| `scene.suite`                        | `libero_goal`                     | LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`)                                                                                      |
| `scene.task_ids`                     | `null`                            | Task pool to draw from; `null` = every task in the suite                                                                                                                                     |
| `prompt`                             | `"do something"`                  | Instruction handed to the VLA; the literal `task` uses the scene's own instruction                                                                                                           |
| `policy.path`                        | `lerobot/pi05_libero_finetuned`   | Hub id or local checkpoint directory                                                                                                                                                         |
| `policy.n_action_steps`              | `10`                              | Actions executed per predicted chunk                                                                                                                                                         |
| `policy.compile`                     | `false`                           | `torch.compile` the model (slow first trial, faster after)                                                                                                                                   |
| `control.mode`                       | `shared_flow_reversal_steering`   | `policy`, `teleop`, `shared_override`, `shared_flow_control`, `shared_flow_reversal_steering` — see [Interactive prompting](#interactive-prompting)                                          |
| `control.tau`                        | `null`                            | `shared_flow_control`: chunk steps (of 50) the command is written into; `null` = the whole chunk                                                                                             |
| `control.lambda`                     | `1`                               | `shared_flow_control`: denoising steps at whose start it is written (1 = the initial condition, Kao et al.; 0 = policy alone). `n_guided_steps` is accepted as the old name                  |
| `control.ic_gain`                    | `action`                          | `shared_flow_control`: the command's scale in the noise prior (a full push writes `ic_gain` sigma), or `action` for the checkpoint's action units                                            |
| `control.n_reversal_steps`           | `null`                            | `shared_flow_reversal_steering` only: denoising steps the reference is reversed through; `null` = all the way to noise                                                                       |
| `control.input_noise`                | `0.0`                             | Std of the Gaussian noise added to the x/y/z command                                                                                                                                         |
| `control.input_gain`                 | `null`                            | Scale on the operator's command that keeps the corruption out of the clip: `null` (off), `auto` or a number in `(0, 1]` — see [Perturbing the operator](#perturbing-the-operator)            |
| `control.max_steps`                  | `null`                            | Rollout length; `null` = the suite's own episode length                                                                                                                                      |
| `control.corruption`                 | `null`                            | Matrix spec applied to the operator's command, `{curl: b, tau: s}` for a curl field, or `{random: ...}` for a new matrix per trial — see [Perturbing the operator](#perturbing-the-operator) |
| `control.reversal_adapter`           | `null`                            | MARS adapter spec for `shared_flow_reversal_steering`: a fixed `F`, `{learned: path}` or `{user: ..., task: ...}` — see [Perturbing the operator](#perturbing-the-operator)                  |
| `control.operator_dims`              | `[translation]`                   | Which dim groups the operator commands (`translation`, `rotation`, `gripper`); the rest of the reversed reference is delegated to the policy                                                 |
| `operator`                           | `human`                           | `human` (SpaceMouse / keyboard) or `policy` — see [Synthetic operator](#synthetic-operator)                                                                                                  |
| `policy_operator.update_every_steps` | `null`                            | Environment steps between synthetic intent queries; `null` uses `policy.n_action_steps`. Requires `operator: policy`                                                                         |
| `policy_operator.delay_steps`        | `0`                               | Environment steps from observing the scene to delivering that synthetic command                                                                                                              |
| `server.port`                        | `8765`                            | Live view port                                                                                                                                                                               |

`--n-trials`, `--seed`, `--output-dir` and `--port` are aliases for the
corresponding `--set`.

### Running a session

Experiments reset once per trial, to an explicit initial-state ID and
environment seed: the single `seed` draws each task's initial states from a
seeded permutation (reshuffling after a complete pass) and seeds the policy,
operator and input-noise streams separately, so the assignments are the same
in every block and at every depth. Different seeds can reuse states; they do
not define disjoint development/evaluation sets by themselves.

For each trial the terminal prints the trial number, the scene, **your** task
and the VLA prompt; the live view shows the reset scene with both. While the
run waits for you, two buttons appear under the status line: **Start trial**
starts the rollout and **Skip trial** skips it without recording anything. The
terminal takes no input; `Ctrl+C` there ends the run, keeping everything
recorded so far. Drive with the SpaceMouse and/or the keyboard exactly as in
the REPL — clicking a button also gives the page keyboard focus. The rollout
ends on success, on the episode limit, or at `control.max_steps`.

### What gets recorded

Each run creates `<output_dir>/<YYYYmmdd_HHMMSS>_<config stem>/` containing:

| File            | Contents                                                                                                                                                                                                                                             |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`   | The resolved configuration, the trial schedule and seeds, and the corruption / adapter **matrices themselves** (with a random corruption: `trial_corruptions`, the draw of every trial) — the run is self-describing even if you later edit the YAML |
| `trials.jsonl`  | One JSON object per trial, appended as it finishes                                                                                                                                                                                                   |
| `trial_XXX.npz` | Per-step arrays for trial `XXX`                                                                                                                                                                                                                      |
| `trial_XXX.mp4` | The rollout video (30 fps)                                                                                                                                                                                                                           |

`trials.jsonl` fields: `trial`, `suite`, `task_id`, `task_description`,
`vla_prompt`, `operator`, `policy_operator`, `operator_dims`, `mode`,
`tau`, `lambda`, `ic_gain` (older runs: `n_guided_steps`), `input_noise`, `input_gain` (the gain in force, `null` when
off), `corruption`, `reversal_adapter` (a short label
of the spec, e.g. `rotation_z_deg=20`, or the loaded file's name; `null` when
off), `corruption_matrix` and `reversal_adapter_matrix` (the 3×3 `M` and 7×7 `F`
in force on that trial, `null` when off), `corruption_parameters` (a random
draw's `rotation_axis`, `rotation_deg` and `scale`; `null` for a fixed
corruption), `success`, `mode_expressed` (the scene element that moved most — an object
or an articulation joint such as `wooden_cabinet_1_middle_level` — or `null`
when nothing moved: the behaviour the policy actually expressed, whether or not
it was the task), `moved` (every element that moved, with its displacement in m
or joint units), `steps`, `duration_s`, `user_reads` (how many times the
operator input was sampled), `video`, `steps_file`, `finished_at`, the reset
identity (`init_state_id`, `env_seed`, `policy_seed`, `operator_seed`,
`input_noise_seed`, `pair_id`, `initial_state_hash` — the hash fingerprints the
settled simulator positions, velocities and actuator state), and the mode's
steering statistics (`guided_steps`, or `n_reversal_steps`, `steered_chunks` and
`reconstruction_error_mean`).

`trial_XXX.npz` arrays, one row per control step:

| Array                                                                                 | Shape                      | Contents                                                                                                                                                                           |
| ------------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `t`                                                                                   | `(T,)`                     | Seconds since the rollout started                                                                                                                                                  |
| `action`                                                                              | `(T, 7)`                   | The env action actually executed: `Δx Δy Δz Δroll Δpitch Δyaw gripper`                                                                                                             |
| `user_translation`                                                                    | `(T, 3)`                   | The operator command **as consumed** — after the corruption matrix and the input noise                                                                                             |
| `user_translation_raw`                                                                | `(T, 3)`                   | The same command **before** corruption and noise, i.e. the operator's true intent                                                                                                  |
| `user_gripper`                                                                        | `(T,)`                     | Operator gripper command (−1 open, +1 close)                                                                                                                                       |
| `user_reads`                                                                          | `(T,)`                     | How many times the input was sampled during that step; `0` = the policy did not consult it (the flow modes read once per chunk)                                                    |
| `policy_translation`                                                                  | `(T, 3)`                   | What the policy would have executed on its own: for a steered chunk, the chunk it produced from the same observation and noise without the operator; otherwise the executed action |
| `oracle_translation`, `operator_translation`                                          | `(T, 3)`                   | Synthetic operator only: its latest computed command, and the one currently delivered (NaN for a human)                                                                            |
| `operator_command_age_steps`, `operator_query_count`                                  | `(T,)`                     | Synthetic operator only: age of the delivered command (−1 before the first delivery) and cumulative queries                                                                        |
| `eef_pos`, `eef_quat`                                                                 | `(T, 3)`, `(T, 4)`         | End-effector pose the policy saw at that step                                                                                                                                      |
| `gripper_qpos`                                                                        | `(T, 2)`                   | Gripper joint positions                                                                                                                                                            |
| `joint_pos`                                                                           | `(T, 7)`                   | Arm joint positions                                                                                                                                                                |
| `reward`, `terminated`, `truncated`                                                   | `(T,)`                     | Environment response                                                                                                                                                               |
| `object_names`, `object_pos`                                                          | `(K,)`, `(T, K, 3)`        | Every scene object / fixture and its position before each step (the state the action was applied to)                                                                               |
| `articulation_names`, `articulation_qpos`                                             | `(J,)`, `(T, J)`           | Every non-robot slide / hinge joint (drawers, knobs, doors) and its value before each step                                                                                         |
| `corruption_matrix`, `reversal_adapter_matrix`, `input_gain`                          | `(3, 3)`, `(7, 7)`, scalar | The `M`, `F` and input gain in force on this trial; NaN when off                                                                                                                   |
| `task_description`, `vla_prompt`, `mode`, `task_id`, `success` and the reset identity | scalars                    | Trial identity, so a single `.npz` stands alone                                                                                                                                    |

Missing robot-state fields are recorded as `NaN` rather than aborting a trial.
Loading a trial is just:

```python
import json, numpy as np

trials = [json.loads(line) for line in open(run_dir / "trials.jsonl")]
z = np.load(run_dir / "trial_000.npz")
z["user_translation_raw"], z["action"], z["eef_pos"], bool(z["success"])
```

### Synthetic operator

`operator: policy` (or `--set operator=policy`) replaces the SpaceMouse and
keyboard with the policy itself: the same checkpoint is queried with the
scene's **real** instruction, the translation of its planned chunk (mean over
the executed prefix, clipped like a stick) becomes the operator's command, and
that command goes through the same corruption, noise and recorders as a
person's. The executing policy still only sees the arm's prompt. Trials start
without waiting for the button and the 20 Hz pacing is dropped, so a block
runs unattended.

By default the operator is queried at every chunk boundary and delivers
immediately — the _immediate oracle_: task-aware, precise, no reaction delay.
`policy_operator` slows it down to something closer to a person on a
SpaceMouse:

```yaml
operator: policy
policy_operator:
  update_every_steps: 20 # new intent once per second at 20 steps/s
  delay_steps: 4 # delivered 0.2 s after observing the scene
```

It holds the preceding command until delivery, starts idle with the gripper
open, and resets its queue every trial. This models intent update timing, not
the hardware's sampling rate; the numbers are assumptions, not a validated
human model, and the underlying policy still supplies privileged task
knowledge. FC and FRS consume one command per 10-step chunk, so a command
generated at step 0 and delivered at step 4 is first consumed at step 10:
delays 1 through 10 give the same consumption schedule, and a dense sub-chunk
delay sweep repeats equivalent conditions. Profiles must not be pooled:
`load_runs` refuses to mix synthetic timing profiles and tags every trial with
`operator_profile`. The `oracle_translation`, `operator_translation`,
`operator_command_age_steps` and `operator_query_count` arrays (above) show
what the operator computed versus what the policy actually consumed
(`user_translation_raw` with `user_reads`).

### The oracle sweep

`experiments/oracle_sweep.sh` runs the grid with the synthetic operator: the
arms in `ARMS` swept over their depths on every LIBERO-Goal task, matched
resets across arms:

```bash
S=./examples/pi05/libero_shared_autonomy/experiments/oracle_sweep.sh
"$S"                                                  # 10 tasks, 10 trials per block, ~a few hours
TASKS="0 2" DEPTHS=4,10 N_TRIALS=5 "$S"
FC_DEPTHS=8,10 FRS_DEPTHS=2,4 ARMS="fc frs frs_ra ceiling" "$S"                  # no floor
"$S" --set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4  # flags reach every run
"$S" --dry-run                                        # validate without loading a policy
```

`FC_DEPTHS`, `FRS_DEPTHS` and `FRS_RA_DEPTHS` set the depths per arm (`DEPTHS`
sets all three), `SEED` the reset assignment, `OUT` the output root (default
`outputs/pi05_libero_oracle_sweep_v2`).

### Human pilot

`experiments/human_pilot.sh` runs one task for a person on the SpaceMouse: ten
matched trials per method and depth, FC at depths 2/4/6/8/10 and the reversal
arms at 1/2/4/6/8 — 150 trials, +20° corruption, MARS (then called MARS) at `F = M`, a 300-step
cap (LIBERO-Goal's own episode length), no synthetic delay:

```bash
P=./examples/pi05/libero_shared_autonomy/experiments/human_pilot.sh
"$P" --dry-run
OUT=outputs/human_pilot/p01 ORDER="fc frs frs_ra" "$P"
OUT=outputs/human_pilot/p02 ORDER="frs_ra fc frs" "$P"     # a new OUT and arm order per participant/session
"$P" --set 'control.corruption={rotation_z_deg: 40}'                  # another rotation
CORRUPTION=scale "$P" --set 'control.corruption={scale: [4, 1, 1]}'   # an x gain instead
CORRUPTION=random FC_DEPTHS=10 FRS_DEPTHS=1 "$P"                      # a new corruption every trial, never clipped
```

The first arm opens the live view (`http://localhost:8773`) and the same tab
follows the next arms. `TASK`, `N_TRIALS`, `SEED`, `FC_DEPTHS`, `FRS_DEPTHS`
(or `DEPTHS` for both), `ORDER`, `CORRUPTION`, `MAX_STEPS`, `PORT` and `OUT` are
environment variables; vary the arm order across participants to spread
practice and fatigue. The pilots recorded so far (`human_pilot/p01`,
`human_pilot_rotz40/p01`, `human_pilot_scalex4/p01`) ran with `MAX_STEPS=600`
and before the input gain existed (`--set control.input_gain=null` restores
that). Afterwards set `ROOT` in
[`analyze_human_pilot.ipynb`](notebooks/analyze_human_pilot.ipynb) to that
`OUT` and Run All: it checks counts and reset pairing, labels incomplete data
as partial, and exports the figures and CSVs under `<OUT>/figures/`
(`outcomes_vs_authority`, `best_vs_best`, `success_over_time`).

#### Why task 4 and these depths

Task 1 ("put the bowl on the stove") saturated in a first human session: FC
reached 100% by depth 6, leaving nothing to separate. `experiments/task_audit.sh`
picked the replacement by running `oracle_sweep.sh` with the SpaceMouse-like
operator (`update_every_steps: 20, delay_steps: 4`) on tasks 4-8, 20 matched
trials per cell, seed 0, FC over depths 4-10, FRS over 1-4, MARS
(F@40) over 1-6, plus the ceiling. Tasks 0/2/3/9 were left out: the oracle
sweep scores them ~0 for every method (the translation-only channel cannot
express them). Success at each method's best depth:

| Task                           | Ceiling | FC      | FRS    | MARS   | MARS − FC (McNemar, 20 pairs) |
| ------------------------------ | ------: | ------- | ------ | ------ | ----------------------------- |
| **4** bowl on top of cabinet   |     100 | d4: 20  | d1: 45 | d2: 60 | **+40, p = 0.021**            |
| 5 push plate to front of stove |      95 | d8: 30  | d4: 25 | d2: 25 | −4, p = 1.0                   |
| 6 cream cheese in bowl         |     100 | d8: 15  | d2: 25 | d1: 25 | +10, p = 0.69                 |
| 7 turn on stove                |     100 | d10: 85 | d1: 85 | d1: 85 | 0, p = 1.0                    |
| 8 bowl on plate                |     100 | d6: 55  | d4: 80 | d4: 65 | +9, p = 0.69                  |

Task 4 is the only one where MARS beats FC at both methods' best depths,
with a 100% ceiling and FC far from it (MARS vs. FRS is +15 points
but not significant at 20 pairs, p = 0.55). The timing profile is what opens
the gap: under the immediate oracle the same task gives FC d10 95% and FRS d1
100%. FC gains from depth while the reversal arms fall off by depth 4 on most
tasks, so the pilot sweeps FC over 2-10 and the reversal arms over 1-8. One
seed and one operator profile: the pilot is also a test of whether a person
behaves like (20, 4).

#### Study record

What has been run, in order, and what it found (all under `outputs/`). The
task audit above and `human_pilot/p01` ran MARS with a fixed 40° rotation
adapter (F@40) through a script override that has since been removed, and the
v2 sweeps compared several adapter angles; the scripts now always run `F = M`.
To repeat an F@40 cell, add
`--set 'control.reversal_adapter={translation: {rotation_z_deg: 40}}'` to a
`flow_reversal_ra_rotz20.yaml` run.

- `pi05_libero_oracle_sweep_v2` — the immediate oracle, all ten tasks, three
  arms over depth plus anchors, seed 0; confirmation on seeds 1-2 in
  `pi05_libero_confirm_v2`; adapter angles −20…40 at FRS depth 2 in
  `pi05_libero_adapter_angles_v2`. FC peaks deep (d10, 58%) and FRS shallow
  (d1, 51%); the adapter lifts FRS to parity with FC (F@40: 60%) but not past
  it, because the +20° corruption barely costs FC anything at FRS's best depth.
  F = M is not special: the optimum sits past it. Tasks 0/2/3/9 are at the
  floor for every method.
- `pi05_libero_task_audit_u20d4` — the audit above. The (20, 4) timing
  profile collapses FC (task 4: 95% → 20%) while shallow reversal keeps about
  half the trials.
- `human_pilot/p01` — the first pilot on task 4. At their best depths FC d10
  (90%), FRS d1 and MARS d2 (100%) tie on ten resets. The separation is at
  low measured authority (about 0.3, the shallow settings), where the reversal
  arms sit at 70-100% and FC at 40%: FC does not get less of the operator's
  push there, it converts the same push into fewer successes.

### Training a MARS user module

`train_user_module.py` calibrates a learned user module for one operator channel,
the cursor project's recipe ported to pi0.5. It rolls out the task-prompted
policy (the synthetic operator) to collect states at chunk boundaries, labels
each with the operator's chunk `L` under the real task, the push that chunk
implies (the `PolicyOperator` rule) through the arm's channel (its corruption or
curl, then input noise of std `sigma ~ U(sigma_min, sigma_max)`), the reference
chunk `a_bci` FRS builds from that served command, and `z_L`, `L` reversed through
the goal-free field (the arm's prompt). The module `U(x_t, t)` then minimizes the
field regression `U(x_t, t) P(x_t, t) = z_L − a_bci` along `x_t = (1 − t) a_bci +
t z_L`, so that a full reversal of what the channel delivered lands on the latent
of what the operator meant. The log reports, on held-out states, the heading
cosine between the regenerated chunk and `L` with and without the module.

```bash
python examples/pi05/libero_shared_autonomy/train_user_module.py \
    --config configs/experiment/mars_learned_rotz20.yaml \
    --out outputs/user_modules/rotz20 --episodes 20 --steps 3000      # ~1-2 h on one GPU
$R experiment --config mars_learned_rotz20.yaml --sweep control.n_reversal_steps=2,4,6,8,10
```

The checkpoint carries its kind (`user`), config and meta (corruption, policy,
prompt). Task modules (`kind: task`, `LearnedTaskModule`) have the slot, the
hooks and the loader, but no LIBERO trainer yet: the cursor project's task
module reads obstacle tokens the scene provides, which LIBERO does not.

**A note on partial reversals.** With `n_reversal_steps` < 10 this example runs
the forward flow from `t = n/10` in `n` steps of the policy's own step size
(since commit `b199239a`); the cursor project still runs it in `10 − n` steps,
which its notes flag as the reason `n = 9` fails there. Tuned depths do not
transfer between the two schedules.

## Analysis

[`notebooks/analyze.py`](notebooks/analyze.py) loads every run directory
that contains a `trials.jsonl` and tags each trial with the run's method
(`FC`, `FRS`, `MARS`, `policy (floor)`, `policy (ceiling)`), depth and
operator:

- `find_runs(root)` / `load_runs(run_dirs)` — the runs under `root` that
  recorded a trial, and their `trials.jsonl` + `config.yaml` as one `DataFrame`
  plus a dict of configs.
- `summary_table(trials, by, metric)` — per group of columns: `n`, the value
  and a 95% interval — the rate with a Wilson interval for a binary metric, the
  mean with a t interval otherwise. `best_depth` is the per-method view.
- `paired_test(trials, by, a, b, pair_on=None, metric)` / `paired_comparisons`
  — a paired test on matching `pair_id` values with verified identical
  initial-state hashes (duplicate or unverified pairs fail): McNemar's exact
  test for a binary metric, the Wilcoxon signed-rank test for a continuous
  one; both report `p`.
- `best_vs_best(trials)` — each steering method at its best depth on the given
  trials, the paired tests between them, and those trials (selection and test
  on the same data: optimistic for every method alike);
  `plot_best_vs_best(selected, ax, metric)` draws any metric on them as a bar
  per method with a significance-starred bracket per pair.
- `task_targets` / `task_goals` — which scene element each task is about (what
  the ceiling anchor's successes move most) and where it ends up.
- `trial_metrics(trials)` — per-trial metrics from the `.npz` arrays, one read
  per trial: operator engagement (fraction of steps commanding, speed,
  corruption shift, gripper, reads), intent agreement (the cosine between the
  executed translation and the raw vs. the corrupted command), path length and
  efficiency, time to success, task progress against the goal set, and the
  measured authority below. `success_over_time` gives the fraction of trials
  done by step _k_, the censoring-safe view of trial time.
- `step_authority(action, policy, command)` — **user authority** per step:
  with `a` the executed translation, `p` the policy's own plan
  (`policy_translation`) and `r` the operator's command, the barycentric
  coordinate of `a` on the segment from `p` to `r`, `⟨a − p, r − p⟩ / ‖r − p‖²`
  clipped to [0, 1]; policy authority is its complement (swapping `p` and `r`
  gives exactly `1 −` it). 0 whenever the operator is idle or the policy runs
  alone, 1 under teleop or override. `chunk_authority` takes it per action
  chunk over the steps where the operator pushed, and `trial_metrics` reports
  the mean over a trial's pushing chunks as `user_authority` — what the
  channel delivered while in use (`user_authority_all_steps` is the per-step
  mean over every step, which idle time dilutes). Depth is the knob; this is what it delivered, and
  it is the x-axis of every comparison — equal depths are not equal authority
  across methods (FC's counts steps the operator controls, FRS's steps the
  policy denoises freely).
- `plot_metric(trials, ax, metric, x="user_authority")` — distinct markers per method:
  any per-trial metric against the measured authority (each method/depth cell
  at its mean, with faint 95% intervals on both axes) or against `depth`
  (connected in depth order). Use `legend=False` for a shared figure legend;
  `plot_success_over_time` draws the time curves. Every plot colours a method
  the same way (`METHOD_COLORS`, a colourblind-safe triple for the steered
  arms, neutral for the anchors) and follows `paper.mplstyle`.

[`analyze_human_pilot.ipynb`](notebooks/analyze_human_pilot.ipynb) uses these
for the human study; the same calls work on any oracle-sweep directory. Jupyter
is not part of the `pi`/`libero` extras; run it without touching the project
environment with:

```bash
uv run --extra pi --extra libero --with jupyterlab jupyter lab examples/pi05/libero_shared_autonomy/notebooks/analyze_human_pilot.ipynb
```

Notebook outputs are stripped before commit by the `nbstripout` pre-commit
hook, so diffs stay to the code cells.

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

- **401 at the PaliGemma tokenizer** — the `hf auth login` token expired;
  log in again, or set `HF_HUB_OFFLINE=1` when everything is cached.
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
- **A pause every ten steps** — that is the policy's denoising pass at each
  chunk boundary (~200 ms for FC, more for FRS, which adds the reversal), not
  per-step overhead; the steps in between are paced to 20 Hz.
  `policy.compile: true` roughly halves it (measured 351 → 179 ms with the
  synthetic operator's two passes) at the cost of a few minutes of compilation
  in the first trial of each process; `compile` is recorded in the run config.
- **Port already in use** — pass `--port`.
- **First rollout very slow with `--compile`** — expected; compilation takes a
  few minutes and the `No valid triton configs` messages are harmless.
- **Rendering errors** — `env.sh` selects `MUJOCO_GL=egl`; try
  `MUJOCO_GL=osmesa` or make sure EGL drivers are installed.
