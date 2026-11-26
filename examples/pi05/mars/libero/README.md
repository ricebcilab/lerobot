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
teleop.py                            SpaceMouse/keyboard readers, the readout (corruption, noise), the synthetic operator
live_view.py                         the browser page: camera stream, status, keyboard, trial buttons
reproducibility.py                   per-trial seeds and initial states
interactive.py                       the REPL
experiment.py                        the scripted trial runner
replay.py                            adds policy_translation to runs recorded before it was recorded live
configs/interactive.yaml             REPL settings
configs/*.yaml                       base.yaml and the study's conditions (see Experiments)
notebooks/analyze.py                 loading, tables, paired tests, per-trial metrics, plots
```

The steering methods themselves live in
[`lerobot.policies.steering`](../../../../src/lerobot/policies/steering/): the
two shared modes are its `SteeringPolicy` over pi0.5 (the policy the cursor
example runs too), given the operator's command in the batch once per chunk.
The corruption and the noise are the shared operator readout
([`lerobot.teleoperators.readout`](../../../../src/lerobot/teleoperators/readout.py),
the cursor's `Readout` in 3-D).

## Install

From the repository root:

```bash
./examples/pi05/mars/libero/run.sh setup
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
./examples/pi05/mars/libero/run.sh lerobot-eval \
  --output_dir=examples/pi05/mars/results/libero_sim/eval \
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
./examples/pi05/mars/libero/run.sh interactive
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
`examples/pi05/mars/results/libero_study/interactive/`. `tasks` lists the suite's scenes,
`task <suite> <id>` switches scene, `quit` exits. Settings can also come from
`configs/interactive.yaml` (`run.sh interactive --config`); the command
reference is `interactive.py --help`.

Besides the default `mode policy` (model-only control), four teleop modes let
you drive or steer the arm with a 3Dconnexion SpaceMouse and/or the keyboard.
The policy denoises each action chunk in 10 flow-matching steps; the two
shared-autonomy modes differ in where your command enters that schedule.

- `mode teleop` — you drive x/y/z and the gripper; the model is not involved.
  A synthetic operator turns the hand too (its planned rotation); the devices
  give no rotation.
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
  every step beyond `pinned_steps` (default: the executed prefix) are delegated
  to the policy — fresh noise for a full reversal, the unsteered flow's state at
  `t = n/10` for a partial one. This is Tang et al.'s noise-space in-painting
  (App. D.1), but the paper pins fewer steps than it executes: a 5-step
  reference (`action_repeats`) in a 15-step chunk, 10 of them executed, so the
  policy fills in half of what runs. Our default pins every executed step;
  `control.pinned_steps` sets another prefix (5 is that interface's ratio; in the
  paper's own LIBERO runs the reference covers its whole 10-step chunk, all of it
  executed, and rotation and gripper stay in the reference instead of being
  handed back: `control.operator_dims=[translation,rotation,gripper]` keeps them,
  and `control.gripper_reference=0` holds the gripper halfway between open and
  closed as they do, where the default holds the model's last gripper command).
  A steered chunk costs `10 + n` velocity evaluations against 10 for FC or an
  unsteered chunk. After each rollout the REPL prints how many chunks were
  steered and how far the executed translation landed from the reference (in
  action-std units). The mechanics are documented in the
  [steering module](../../../../src/lerobot/policies/steering/).

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

- **Translation noise** — `--translation-noise STD` (or `noise STD` in the REPL) adds
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
  corruption: {curl: 20, tau: 0.1}      # a curl field (1/s, + = clockwise in x-y) on a command with inertia tau (s)
  corruption: {quantize: axes}          # a keypad: the command snapped to the axis of its largest component, at its own length
  ```

  The curl field is the cursor project's dynamic corruption: the served command
  is a velocity driven by your push, `dv/dt = (u − v)/tau + curl·R·v`, stepped
  once per control step (`dt`, default 0.05 s) from rest at every trial. A held
  push comes out turned clockwise by `atan(tau·curl)` and slowed by
  `1/sqrt(1 + (tau·curl)^2)`; the field keeps evolving while you are idle, but an
  idle push is served as is. A curl never lengthens a command, so `translation_gain:
auto` is 1.

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
  reversal_adapter: {learned: examples/pi05/mars/checkpoints/adapters/libero_sim/<user>/d<k>/user.pt}      # MARS-U, learned
  reversal_adapter: {user: {learned: u.pt}, task: t.pt}                 # MARS: user + task module
  reversal_adapter: {user: {translation: corruption}, task: t.pt}       # a fixed F with a task module
  ```

  Before applying `F` to normalized velocities the steering policy converts it to
  `S⁻¹ F S`, where `S` is the diagonal matrix of action standard deviations
  (velocities have no mean offset); learned modules work in the normalized space
  directly. The matrices and the module paths are saved in the run config.

- **Translation gain** — `control.translation_gain` turns the operator's command down so
  that the corruption can never push it into the env's `[-1, 1]` clip. The
  command is first limited to unit length, keeping its direction (a stick
  pushed on several axes at once can otherwise reach √3), and then multiplied
  by the gain. `auto` picks the largest gain that cannot clip, 1 / the most the
  corruption can lengthen a command (its largest singular value); a number in
  `(0, 1]` is accepted only if it
  cannot clip either. `null` (the built-in default) leaves the command as it
  is, clip included; every shipped config sets `auto`, and in the REPL an
  `auto` gain follows each `corruption` command. The price is speed: with gains up to 4 the gain is 0.25, so an
  unscaled axis tops out at a quarter of full speed, and the idle threshold
  (0.05 in env units) then sits at a fifth of the stick's travel. The recorded
  raw command is the one after the gain.

A file loaded with `--corruption FILE` / `--reversal-adapter FILE` (or live in
the REPL) holds the same spec forms at its top level.

## Experiments

`run.sh experiment` runs scripted, recorded trials from a YAML config with the
same policy, live view and modes as the REPL. Each trial draws a task from the
configured scene, shows **you** the task, and lets you attempt it with the VLA
in the configured shared-autonomy mode. A bare file name is looked up under
`configs/`:

```bash
R=./examples/pi05/mars/libero/run.sh
$R experiment --config base.yaml --dry-run   # validate + print the schedule, run nothing
$R experiment --config base.yaml             # one block of n_trials
```

### The prompt vs. the task

The **human** is always shown the scene's real instruction ("put the bowl on
top of the cabinet"). The **VLA** only ever receives the config's `prompt`,
which defaults to the deliberately uninformative `"do something"` — the policy
contributes manipulation priors while the operator supplies the intent through
the steering command. `prompt: task` gives the VLA the scene's own instruction
instead, the conventional setup where both know the goal.

### Arms, overrides and sweeps

`configs/base.yaml` holds every setting the experiments share, each
with its default (FRS on LIBERO-Goal, the prompt "do something", no
corruption); the study files (`study_*.yaml`) extend it. An arm is the base
plus a few settings, given with `--set`:

- **FC**, shared flow control (Kao et al.): `control.mode=shared_flow_control`,
  with `tau` null, `lambda` 1 and `ic_gain: action` from the base. Its depth is
  `tau`, in chunk steps.
- **FRS**, Flow Reversal Steering without an adapter: the base's own mode.
  Swept over `n_reversal_steps`: only `n = 10` is the published method, every
  `n < 10` cell is a partial reversal (see [Interactive prompting](#interactive-prompting)).
- **MARS** with a fixed user module (MARS-U, formerly FRS-RA):
  `control.reversal_adapter={translation: corruption, orientation: identity, gripper: identity}`.
  `F`'s translation block is the known corruption `M` and its other blocks are
  identity, so the only difference from FRS is `M` applied to the reversal's
  translation.
- **MARS** with a learned user module:
  `control.reversal_adapter={learned: examples/pi05/mars/checkpoints/adapters/libero_sim/<user>/d<k>/user.pt}`
  (trained as below).
- The policy alone: `control.mode=policy`, with the arm's prompt the **floor**
  of every comparison, with `--set prompt=task` its **ceiling**.
- A corruption: `control.corruption={rotation_z_deg: 20}` (the operator's
  command rotated 20° about z before the policy sees it) or
  `{scale: [2, 1, 1]}` (x twice as sensitive as y and z). A `--set` replaces
  the base's `null` whole, and a later `--set` an earlier one.

The depth of each method (FC's `tau`, the reversal arms' `n_reversal_steps`) is a sweep variable rather than a file. `--set KEY=VALUE`
overrides any YAML key (dotted path, value parsed as YAML) and `--sweep KEY=V1,V2,...`
runs one block of `n_trials` per value with a single policy load:

```bash
$R experiment --config base.yaml --set control.mode=shared_flow_control --sweep control.tau=5,10,25,50
$R experiment --config base.yaml --set control.mode=shared_flow_control --sweep control.ic_gain=1,2,3,5    # Kao's gain, once per condition
$R experiment --config base.yaml --set 'control.corruption={rotation_z_deg: 20}' --sweep control.n_reversal_steps=1,2,4,6,8
$R experiment --config study_goal_rotation.yaml --set control.mode=shared_flow_control --sweep control.tau=5,10,25,50   # a study user
```

Each block gets its own run
directory named after the file and the overrides in force, e.g.
`20260903_101500_study_goal_n_reversal_steps-4/`, and its
`config.yaml` lists the overrides under `overrides`. Keys that would need a
different policy (`policy.*`, `server.port`, `operator`) cannot change between
blocks.

### Configuration

Everything lives in the YAML file you pass to `--config`. Unknown keys are
rejected and a key set to `null` keeps the built-in default, so a typo fails
immediately instead of silently doing nothing. The run's name is the config
file's stem plus any `--set`/`--sweep` values.

| Key                                  | Default                                             | Meaning                                                                                                                                                                                              |
| ------------------------------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.n_trials`                | `10`                                                | Number of trials                                                                                                                                                                                     |
| `experiment.seed`                    | `0`                                                 | Seeds task order, per-task initial-state permutations, and independent environment, policy, synthetic-operator and noise streams                                                                     |
| `experiment.task_order`              | `random`                                            | `random` (uniform, with replacement), `shuffled` (permuted blocks, each task once per block), `sequential` (cycle in order)                                                                          |
| `experiment.first_occurrence`        | `0`                                                 | Each task's trials start at this occurrence of its initial states: `5` continues where a 5-per-task run stopped, so a development and a test split share no state within a pass                      |
| `experiment.output_dir`              | `examples/pi05/mars/results/libero_sim/experiments` | Parent of the run directory                                                                                                                                                                          |
| `scene.suite`                        | `libero_goal`                                       | LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`)                                                                                              |
| `scene.task_ids`                     | `null`                                              | Task pool to draw from; `null` = every task in the suite                                                                                                                                             |
| `prompt`                             | `"do something"`                                    | Instruction handed to the VLA; the literal `task` uses the scene's own instruction                                                                                                                   |
| `policy.path`                        | `lerobot/pi05_libero_finetuned`                     | Hub id or local checkpoint directory                                                                                                                                                                 |
| `policy.n_action_steps`              | `10`                                                | Actions executed per predicted chunk                                                                                                                                                                 |
| `policy.compile`                     | `false`                                             | `torch.compile` the model (slow first trial, faster after)                                                                                                                                           |
| `control.mode`                       | `shared_flow_reversal_steering`                     | `policy`, `teleop`, `shared_override`, `shared_flow_control`, `shared_flow_reversal_steering` — see [Interactive prompting](#interactive-prompting)                                                  |
| `control.tau`                        | `null`                                              | `shared_flow_control`: chunk steps (of 50) the command is written into; `null` = the whole chunk                                                                                                     |
| `control.lambda`                     | `1`                                                 | `shared_flow_control`: denoising steps at whose start it is written (1 = the initial condition, Kao et al.; 0 = policy alone). `n_guided_steps` is accepted as the old name                          |
| `control.ic_gain`                    | `action`                                            | `shared_flow_control`: the command's scale in the noise prior (a full push writes `ic_gain` sigma), or `action` for the checkpoint's action units                                                    |
| `control.zeroed_dims`                | `[]`                                                | `shared_flow_control`: dim groups (`rotation`, `gripper`) set to 0 in the steps the command is written into; Kao et al. on LIBERO zero rotation                                                      |
| `control.n_reversal_steps`           | `null`                                              | `shared_flow_reversal_steering` only: denoising steps the reference is reversed through; `null` = all the way to noise                                                                               |
| `control.pinned_steps`               | `null`                                              | `shared_flow_reversal_steering` only: leading chunk steps the reference pins, the rest is the policy's; `null` = the executed steps                                                                  |
| `control.translation_noise`          | `0.0`                                               | Std of the Gaussian noise added to the x/y/z command at every read, the operator resting or not                                                                                                      |
| `control.rotation_noise`             | `0.0`                                               | Std of the Gaussian noise added to each axis of the rotation command; only teleop mode executes that command                                                                                         |
| `control.gripper_noise`              | `0.0`                                               | Std of the Gaussian noise added to the gripper command, which is then served as its sign (open or close); only teleop mode executes it                                                               |
| `control.translation_gain`           | `null`                                              | Scale on the operator's command that keeps the corruption out of the clip: `null` (off), `auto` or a number in `(0, 1]` — see [Perturbing the operator](#perturbing-the-operator)                    |
| `control.max_steps`                  | `null`                                              | Rollout length; `null` = the suite's own episode length                                                                                                                                              |
| `control.corruption`                 | `null`                                              | Matrix spec applied to the operator's command, `{curl: b, tau: s}` for a curl field or `{quantize: axes}` for a keypad — see [Perturbing the operator](#perturbing-the-operator)                     |
| `control.reversal_adapter`           | `null`                                              | MARS adapter spec for `shared_flow_reversal_steering`: a fixed `F`, `{learned: path}` or `{user: ..., task: ...}` — see [Perturbing the operator](#perturbing-the-operator)                          |
| `control.operator_dims`              | `[translation]`                                     | Which dim groups the operator commands (`translation`, `rotation`, `gripper`); the rest of the reversed reference is delegated to the policy                                                         |
| `operator`                           | `human`                                             | `human` (SpaceMouse / keyboard) or `policy` — see [Synthetic operator](#synthetic-operator)                                                                                                          |
| `policy_operator.update_every_steps` | `null`                                              | Environment steps between synthetic intent queries; `null` uses `policy.n_action_steps`. Requires `operator: policy`                                                                                 |
| `policy_operator.delay_steps`        | `0`                                                 | Environment steps from observing the scene to delivering that synthetic command                                                                                                                      |
| `policy_operator.intent`             | `plan`                                              | Where the synthetic operator's intent comes from: `plan` the task-prompted policy's own plan, `target` a straight push at the task's object and then its destination (see Synthetic operator)        |
| `policy_operator.let_go_cosine`      | `null`                                              | Under a steering arm (FC, FRS, MARS) the operator rests (zero command) for the next update while the robot's executed motion heads with its intent at this cosine or more; `null` = it always pushes |
| `server.port`                        | `8765`                                              | Live view port                                                                                                                                                                                       |
| `server.headless`                    | `false`                                             | No live view: no page, no frame stream, no browser tab; for runs of the synthetic operator (the study files set it), refused with a human operator                                                   |

`--n-trials`, `--seed`, `--output-dir` and `--port` are aliases for the
corresponding `--set`.

### Running a session

Experiments reset once per trial, to an explicit initial-state ID and
environment seed: the single `seed` draws each task's initial states from a
seeded permutation (reshuffling after a complete pass) and seeds the policy,
operator and noise streams separately, so the assignments are the same
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

| File            | Contents                                                                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.yaml`   | The resolved configuration, the trial schedule and seeds, and the corruption / adapter **matrices themselves** — the run is self-describing even if you later edit the YAML |
| `trials.jsonl`  | One JSON object per trial, appended as it finishes                                                                                                                          |
| `trial_XXX.npz` | Per-step arrays for trial `XXX`                                                                                                                                             |
| `trial_XXX.mp4` | The rollout video (30 fps)                                                                                                                                                  |

`trials.jsonl` fields: `trial`, `suite`, `task_id`, `task_description`,
`vla_prompt`, `operator`, `policy_operator`, `operator_dims`, `mode`,
`tau`, `lambda`, `ic_gain` (older runs: `n_guided_steps`), `translation_noise`, `translation_gain` (`input_noise` and `input_gain` in runs recorded before 2026-11-16, which `analyze.py` loads under the new names; the gain in force, `null` when
off), `corruption`, `reversal_adapter` (a short label
of the spec, e.g. `rotation_z_deg=20`, or the loaded file's name; `null` when
off), `corruption_matrix` and `reversal_adapter_matrix` (the 3×3 `M` and 7×7 `F`
in force on that trial, `null` when off), `success`, `mode_expressed` (the scene element that moved most — an object
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
| `user_translation`                                                                    | `(T, 3)`                   | The operator command **as consumed** — after the corruption matrix and the translation noise                                                                                       |
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
| `corruption_matrix`, `reversal_adapter_matrix`, `translation_gain`                    | `(3, 3)`, `(7, 7)`, scalar | The `M`, `F` and translation gain in force on this trial; NaN when off                                                                                                             |
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

`policy_operator.intent: target` gives the operator a person's pointing in
place of the policy's plan (`PolicyOperator.target_push`, from the simulator's
positions of the gripper, the task's object and its destination): it pushes
straight at the object until the gripper is over it (within 6 cm sideways and
10 cm above), then stops and leaves the grasp to the policy; once the object
has moved 3 cm and is with the gripper it pushes toward a point 15 cm above
the destination (level with the gripper for an object pushed along the table)
and stops when the object is within 6 cm of the destination sideways, leaving
the release to the policy; a dropped object is approached again, and a button
has no destination. The push has unit length and shrinks inside 12 cm of its
point. It handles goals of one `on`, `in` or `turnon` predicate; rotation and
gripper are the policy's, so it is for the steering arms, not teleop. The
timing and the let-go rule below apply to it as to the plan.

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
open, and resets its queue every trial. With `let_go_cosine` (the study files
set 0.5) it watches what the robot executes: when the robot's mean executed
translation since its last update heads with its new intent at that cosine or
more, it lets go for the next update (zero translation and rotation, its
gripper kept), as a person stops pushing while the robot does the right thing.
The readout still serves its noise then, so the steering acts on a noisy rest.
Under teleop it never lets go, since nothing else moves the robot. This models intent update timing, not
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

### User study

The study of the MARS user module on LIBERO-Goal
([design](../../../../docs/superpowers/specs/2026-11-14-mars-libero-user-study-design.md)),
round 1 since 2026-11-20: the prompt-dropout checkpoint
(`examples/pi05/mars/checkpoints/policies/libero_generic_prompt/checkpoints/006000/pretrained_model`) is
told "do something", the synthetic operator's command says which goal is meant,
and three users corrupt it (the study dropped a fourth, a six-key pad, on 2026-11-25). LIBERO-Goal's other four
tasks (0, 2, 3, 9) are not played: an earlier synthetic-operator sweep scored them about 0 for every method (a
translation command cannot express them), so they serve the user modules' calibration instead.

| Condition file             | User                                                                   |
| -------------------------- | ---------------------------------------------------------------------- |
| `study_goal.yaml`          | clean readout (the reference)                                          |
| `study_goal_rotation.yaml` | decoding error: one random 3-D rotation of 60°, drawn from seed 0      |
| `study_goal_lowdof.yaml`   | low-DOF control: the vertical component dropped (`{scale: [1, 1, 0]}`) |
| `study_goal_lag.yaml`      | misperception and delay: a lag-and-turn field (`{curl: 2, tau: 0.5}`)  |

They run LIBERO-Goal's six tasks a translation command can express (1, 4, 5,
6, 7, 8; `study_object*.yaml` are the same users on LIBERO-Object). Every file
runs the immediate synthetic operator (a new intent at every chunk, consumed at
once), which lets go under a steering arm while the robot heads where it means
(`policy_operator.let_go_cosine: 0.5`), 5 trials per task in task order: the
development split. A test split takes the next states with
`--set experiment.first_occurrence=5`.

```bash
U=./examples/pi05/mars/libero/experiments/user_study.sh
P="--set policy.path=$PWD/examples/pi05/mars/checkpoints/policies/libero_generic_prompt/checkpoints/006000/pretrained_model"
N="SUITES=goal TRANSLATION_NOISE=0.2 ROTATION_NOISE=0.016 GRIPPER_NOISE=0.4"
env $N OUT=examples/pi05/mars/results/libero_sim/round1/gain "$U" gain $P      # FC's gain, once: 1.5, 3, 5, 8, 12 at tau 10 and 50, clean readout
env $N SPLIT=1 ARMS="fc frs teleop floor ceiling" FC_GAIN=3 OUT=examples/pi05/mars/results/libero_sim/round1/stage1 "$U" 1 $P   # the baselines and anchors
env $N MODULES=examples/pi05/mars/results/libero_sim/round1/modules "$U" modules $P   # one user module per user, calibrated by watching the policy
env $N "$U" 2 $P --dry-run                  # validate a stage's runs without loading a policy
```

`ARMS` picks the arms of stage 1, `USERS` the users, `SPLIT=1` gives every
knob value its own run, and any extra flag reaches every run.

Stage 2 trains one user module per user (`modules`: `train_user_module.py` in
the LIBERO-Goal scene on the four tasks outside Goal-6 (0, 2, 3, 9), five start
states each, into `MODULES/<user>/`), so a module has seen the scene but none of
the six evaluation tasks. Its calibration watches the policy under "do
something" for 20 episodes, unsteered, while the synthetic operator pushes where
the robot heads wrong and rests where it heads right (the let-go rule), so a
pushed state is one where the command matters and a rest teaches the module to
leave the policy alone. The noise, at the study's level, is drawn once per state.
MARS then runs with the module on the development split over
the reversal depth and the pinned prefix, as on the cursor grid.

The baselines are their papers' methods, each with its one knob tuned per user
on the development split, as on the cursor. FRS (Tang et al.) keeps the full
reversal and tunes the pinned prefix (`control.pinned_steps` over 5, 10, 25,
50). FC (Kao et al.) writes the initial condition only (`lambda` 1) with
rotation set to 0 in the steered steps (`control.zeroed_dims: [rotation]`, their
LIBERO recipe) and tunes `tau` (5 to 50); its gain is chosen once on the clean
readout, as the cursor's was. Teleop, as on the cursor, is the operator alone
(`control.mode: teleop`): its translation through the user's readout, its
rotation and gripper, and no policy. A user's corruption acts on the
translation; its noise reaches every part of the command at every read, at 0.2
on the translation, 0.016 on the rotation and 0.4 on the gripper. What differs
from the papers' own LIBERO runs: the checkpoint (lerobot's conversion,
fine-tuned, with 50-step chunks and mean/std action statistics, against
openpi's 10-step chunks and quantiles), rotation and gripper handed back to
the policy in FRS, the command's own length written in FC (theirs is a unit
direction), and a policy that knows the tasks but is not told which.

The launcher runs several experiments at a time across the GPUs (a worker
takes the next run as it finishes one). Results go to `OUT` (default
`examples/pi05/mars/results/libero_sim/stage<N>/`); each launch keeps its runs' logs
and the code state (`HEAD` and the working-tree diff) under `logs/<launch
time>/`, and a dry run writes nothing there.

### Training a MARS user module

`train_user_module.py` calibrates a learned user module for one operator readout,
the cursor project's recipe ported to pi0.5. It collects states at chunk
boundaries and labels each with the operator's chunk `L` under the real task,
the push that chunk implies (the `PolicyOperator` rule) through the arm's
readout (its corruption or curl, then noise at every read), the reference chunk
`a_bci` FRS builds from that served command, and `z_L`, `L` reversed through the
goal-free field (the arm's prompt). The module `U(x_t, t)` then minimizes the
field regression `U(x_t, t) P(x_t, t) = z_L − a_bci` along `x_t = (1 − t) a_bci +
t z_L`, so that a full reversal of what the readout delivered lands on the latent
of what the operator meant.

The states come from two kinds of rollout. `--episodes N` rolls out the
task-prompted policy. `--open-episodes N` rolls out the policy under the arm's
own prompt, unsteered, with the synthetic operator watching it and pushing or
resting by the arm's let-go rule (`--open-steps` keeps that many control steps
of each); a state's label is then the chunk the operator planned there and its
push the one it would deliver, zero where it rests. Task-prompted states alone
teach a module to discard the command, because there the policy's own plan
already is the label; with the let-go rule the operator pushes only where the
robot heads wrong. Commands get noise of std `sigma ~ U(--sigma)` through the
arm's readout, drawn anew at every batch or once per state with
`--frozen-noise` (as a recorded session has it). `--states <file>` reuses an
earlier run's states for any arm: the rollouts and the frozen draws do not
depend on the user, and the commands are served again through the arm's own
readout.

The log reports, on held-out states, what the rollouts' steering would execute
(`FlowReversalSteering`, the executed prefix pinned) at each of `--val-depths`:
the cosine between the executed steps' mean translation and the operator's
push, for the policy's own plan from the same noise (`plan`), FRS on the
user's command (`none`), the same through the module (`user`), FRS on a
faithful readout's command with the same noise draw (`clean`) and on the push
itself (`ideal`). It splits the states by whether the plan disagrees with the
push (`d<depth>_off_*`), agrees (`d<depth>_on_*`) or the operator rests,
scored there against what it means (`d<depth>_rest_*`), and reports each
group's mean executed length per step (`d<depth>_<group>_len_*`). A module
repairs the command if `off_user` moves from `off_none` toward `off_clean`
without a shorter `len_user`; one that only weakens the steering raises
`on_user` alone.

```bash
python examples/pi05/mars/libero/train_user_module.py \
    --config configs/base.yaml --set 'control.corruption={rotation_z_deg: 20}' \
    --out examples/pi05/mars/checkpoints/adapters/libero_sim/<user>/d<k> --episodes 20 --steps 3000      # ~45 min on one GPU
$R experiment --config base.yaml --set 'control.corruption={rotation_z_deg: 20}' \
    --set 'control.reversal_adapter={learned: examples/pi05/mars/checkpoints/adapters/libero_sim/<user>/d<k>/user.pt}' --sweep control.n_reversal_steps=2,4,6,8,10
```

The trainer takes the arm's readout and ignores its reversal adapter, so it starts
from the arm file that names the module it is about to write. `--set KEY=VALUE`
overrides the file as in `experiment.py`; calibrating on other scenes than the
arm evaluates on is `--set scene.suite=libero_spatial` (and `scene.task_ids`).
The episodes cycle through the tasks, each pass from a task's next initial
state. Calibration commands are served by the rollouts' rule: noise at every
read, an idle command included.

**The module's ranks.** Three options set how flexible `U` is; their defaults
are the module as it was, and a fresh module is the identity under every setting:

- `--step-rank r` (default: unset, every chunk step's 7×7 matrix free): every
  step's matrix is `I + Σ_k c_{s,k} B_k` over `r` 7×7 matrices shared by the
  chunk, the coefficients per step (`r = 49` spans every matrix).
- `--cross-step-rank q` (default 0: none): a rank-`q` coupling between the
  chunk's steps on top of the per-step matrices,
  `ṽ_s = A_s v_s + U_s Σ_s′ V_s′ᵀ v_s′`, `U` zero-initialized. Its head on the module's hidden layer has
  `2 · H · n_c · q` outputs (`n_c` the coupled dims): at pi0.5's `H = 50` with all
  7 dims, **700 · q** outputs, i.e. 700 · q · (256 + 1) weights at the default
  width (180 k at q = 1).
- `--cross-step-dims` (default: all 7): the dims the coupling covers, names from
  `ACTION_DIMS` or indices, e.g. `dx,dy,dz` or `0,1,2` (`n_c = 3`: 300 · q
  outputs); the other dims get no cross-step term. Only with `--cross-step-rank`.

The options are saved in `user.pt`'s config (only when they differ from the
defaults), and every place that loads a user module for steering
(`reversal_adapter: {learned: ...}` or `{user: ...}`, `AdapterModules.load_module`)
rebuilds it from that config: there is nothing to set at run time.

The checkpoint carries its kind (`user`), config and meta (corruption, policy,
prompt). The clean-operator visual task pilot below trains a task module from the
executing LIBERO policy's external and wrist camera features.

**A note on partial reversals.** With `n_reversal_steps` < 10 this example runs
the forward flow from `t = n/10` in `n` steps of the policy's own step size
(since commit `b199239a`); the cursor project still runs it in `10 − n` steps,
which its notes flag as the reason `n = 9` fails there. Tuned depths do not
transfer between the two schedules.

## Clean-operator visual task pilot

Run from the repository root with the existing LIBERO installation. Each stage reuses
the same output directory (default `outputs/pi05_libero_visual_task`). To use another
directory, pass the same `--out` to each invocation.

```bash
source examples/pi05/mars/libero/env.sh
PILOT=examples/pi05/mars/libero/train_visual_task.py
"${UV_RUN[@]}" python "$PILOT" probe
"${UV_RUN[@]}" python "$PILOT" collect
"${UV_RUN[@]}" python "$PILOT" train
"${UV_RUN[@]}" python "$PILOT" evaluate
```

The marked disk is a visual-only MuJoCo geom. Scoring checks whether the bowl centre
crosses its vertical projection between consecutive control steps, including the
terminal step. The disk is sampled on either the bowl-to-stove or bowl-to-plate
route for both tasks. The immediate clean operator receives the destination prompt.
The executing policy receives `do something`.

`probe` tests the waypoint teacher and plain FRS on development states, saves
representative two-camera videos and selects depth and prefix by joint success.
Inspect the videos and `development.json` before collecting. The fixed split has
20 training, 10 development and 20 held-out initial states per destination.
`collect` saves 100 successful, violation-free episodes balanced across destination
and marker route, recording failed attempts in the same directory. Each archive
holds both oriented RGB views and robot state at replans, the stride and all
executed actions. Completed trials can be resumed.

`train` fits a 64-wide task head on frozen SigLIP patch features from the pinned
`lerobot/pi05_libero_finetuned` snapshot. It also fits the same head with only its
null token. Both see identical data draws per training seed. Targets are the XYZ
components of noise minus normalized actions on valid steps; the base velocity
uses the full padded chunk. Load a trained head in the regular LIBERO runner with
`control.reversal_adapter: {task: path/to/task.pt}`. The checkpoint is fitted to
this policy's field.

`evaluate` compares clean FRS, masked T, visual T and the visual checkpoint given
an image of the same robot pose with the marker on the other route. Only the adapter
receives those counterfactual tokens. `report.md` contains per-arm placement,
violation and joint success, replan latency and paired intervals clustered by
initial state. Regenerate it with the `report` stage. The default training seeds
are 0, 1 and 2. For smaller development runs, use a separate output directory with
`--states` (per split and destination), `--demos` (multiple of four), `--steps`,
`--batch` and `--seeds`.

### Solid utensil holder and pickup diagnostics

Run the same stages with `--obstacle holder` to use a fixed ceramic crock with two
utensils. The holder run defaults to `outputs/pi05_libero_utensil_holder` and the
executing prompt `place the black bowl somewhere`. The operator still sees the
actual destination. `--prompt` changes the execution and adapter-training prompt
together. Use a separate output directory for each prompt.

```bash
source examples/pi05/mars/libero/env.sh
PILOT=examples/pi05/mars/libero/train_visual_task.py
for stage in probe prompt-check collect train evaluate; do
    "${UV_RUN[@]}" python "$PILOT" "$stage" --obstacle holder || break
done
```

The holder is 18.4 cm tall including the utensils. Reset sampling preserves the
original bowl and destination positions and rejects overlapping placements.
A blocked layout intersects the approximate swept width of a bowl on the direct
route. Joint success requires correct placement and zero physical bowl or robot
contacts with the holder, including transient contacts during physics substeps.

`prompt-check` compares the bowl prompt and `do something` on the same
development resets with both plain FRS and the unsteered frozen policy. Held-out
evaluation uses the main prompt for every arm. A verified pickup requires both
finger pads to contact the same object while it is at least 2.5 cm above its
starting height. Trial JSONs include the picked objects, maximum lift per object,
final LIBERO destination predicates and first holder contact.

To reproduce the original disk failure audit with separate output files:

```bash
"${UV_RUN[@]}" python "$PILOT" audit --out outputs/pi05_libero_disk_pick_audit
"${UV_RUN[@]}" python "$PILOT" audit-report --out outputs/pi05_libero_disk_pick_audit \
    --reference outputs/pi05_libero_visual_task
```

### Camera features before and after image–text fusion

New LIBERO holder experiments use post-fusion T. To reproduce the original comparison
with pre-fusion T, run the following from the repository root with the holder demonstrations available.

```bash
source examples/pi05/mars/libero/env.sh
"${UV_RUN[@]}" python examples/pi05/mars/libero/train_visual_task.py camera-pilot
```

Both heads use `place the black bowl somewhere, avoiding obstacles along the way`
during training and execution. The pre-fusion head reads the projected external and
wrist patches. The post-fusion head reads the same 512 camera positions after the
frozen language model processes the image and text prefix. Both reuse the prefix
pass that produces the action expert’s cache. The adapter receives camera features,
flow state, time, and base velocity. The teacher uses simulator geometry to produce
clean demonstrations, and the evaluator uses it to score contact.

`--reference` selects the source holder run with its demonstrations, checkpoint
revision and development-selected steering settings. The default output directory
is `outputs/pi05_libero_holder_camera_stages`. It contains the dataset digest,
six trained heads, and paired test results for plain FRS and each head with the
real and alternate holder views. Rerunning the command resumes completed heads and
trials. The report gives placement, contact and joint success by destination and layout.

### Gripper geometry and corrective demonstrations

Run the three post-fusion conditions against the archived real-view baseline from
`camera-pilot`. The collector uses the baseline heads on training resets. Evaluation
loads the archived baseline rows for the paired comparisons.

```bash
source examples/pi05/mars/libero/env.sh
"${UV_RUN[@]}" python examples/pi05/mars/libero/holder_factorial.py experiment
```

`--baseline` defaults to `outputs/pi05_libero_holder_camera_stages`. The new
checkpoints, recovery attempt logs, projection overlay, and report go under
`outputs/pi05_libero_holder_factorial`. Use the `collect` stage to run just the
resumable recovery collection. The geometry head adds external patch offsets from
the projected end effector. The mixed head samples one original chunk and one
clean teacher-recovery chunk per update. The combined head uses both changes.
Recovery quotas are 16 blocked-plate and 16 other episodes, with a minimum of
eight in each group and a limit of 200 attempts per group. Below the minimum,
the experiment runs geometry only.

To continue collection and train just the mixed head, use `holder_factorial.py mixed`.
It reuses completed trials from `--recovery-source` (the factorial run by default),
extends the collection limit to 2,000 attempts per pool, and writes to
`outputs/pi05_libero_holder_mixed`. It targets 16 recoveries per pool and uses
the same minimum of eight and the same held-out reset schedule.

### Replaying the approach to holder contact

The clean 10/10 intervention pilot starts from the seed-0 post-fusion checkpoint in
`outputs/pi05_libero_holder_camera_stages`. Run from the repository root, with the
same cached π0.5 snapshot as that experiment.

```bash
source examples/pi05/mars/libero/env.sh
PILOT=examples/pi05/mars/libero/holder_interventions.py
"${UV_RUN[@]}" python "$PILOT" collect
```

Collection replays the two replans before first contact and checks the original contact
step and geometry. Each branch changes XYZ for one ten-step prefix and then resumes
the same controller. A rescue requires correct placement without any holder contact.
The `pairs/` JSON files record failed attempts as well as rescues. Accepted archives
also store the camera views, input command, Gaussian sample and gripper reference.

The initial screen uses 20 blocked-plate contact episodes. Five rescues trigger
further collection to 32 rescues including at least 16 on blocked plate, capped at
200 contacting episodes. With those quotas met, the `train` stage compares
matched corrective-data continuations with and without the executed-prefix loss,
then selects a head on development layouts. If neither A nor B meets the development criterion,
the training stage fits a two-output contact predictor with reset-grouped cross-validation.
It must rank the rescuing prefix safer for at least two thirds of the robot-first pairs and
two thirds of the bowl-first pairs. Only pairs with contact within the executed prefix
enter this ranking check. Passing enables a third continuation with the frozen predictor
penalizing predicted contact. When the replay screen falls short and
gripper contacts dominate, the `wrist` stage tests one additional roll or pitch
correction on the gripper-first failures. Outputs default to
`outputs/pi05_libero_holder_interventions`.

### Single-seed rotation grid

With the holder mixed-T run available, run the paired rotation-user comparison:

```bash
source examples/pi05/mars/libero/env.sh
"${UV_RUN[@]}" python examples/pi05/mars/libero/holder_rotation_grid.py
```

This calibrates a learned U for the fixed 60° rotation on Goal tasks 0, 2, 3 and 9,
then pairs FRS, U, seed-0 mixed T and UT on the first five scheduled held-out resets
per destination and holder layout. The output under `outputs/pi05_libero_holder_rotation_grid`
includes the known-command check, checkpoint digests, reset schedule, per-trial JSON,
videos and paired gains and losses. Completed checkpoints and rollouts are reused.

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
- `wrong_target(trials, targets=None)` — per trial, whether what moved most was
  a scene element other than the task's target (the wrong object in the basket);
  NaN where the target is unknown. Give a frame without the ceiling anchor the
  targets of the stage that ran it (`suite_task_targets`). It cannot see the
  right element put in the wrong place, so it under-counts on LIBERO-Goal, whose
  tasks share the bowl.
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
  readout delivered while in use (`user_authority_all_steps` is the per-step
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
  --output_dir=examples/pi05/mars/checkpoints/policies/libero_train \
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

### Paired goals with two physical obstacles

Run `paired_obstacles.py --layouts 2` in the LIBERO environment for a four-rollout
smoke test. It uses the post-fusion condition C checkpoint by default. A fixed
utensil holder and a fixed wooden pepper mill occupy the two direct bowl-to-goal
routes. Their assignments alternate across layouts. Each layout is replayed for
both intended destinations on the same underlying LIBERO reset. Only the
goal-aware operator receives the destination instruction. The evaluator switches
the native placement predicate and verifies identical initial images, physical
state and obstacle positions across each pair.

A contact between either designated obstacle and the robot or bowl disqualifies
joint success. Intentional contacts with the bowl, table and placement surface are
handled by the ordinary task physics. Trial JSON records the first obstacle hit,
and the output directory includes initial external camera views.

Run `two_obstacle_pilot.py` for the clean-input training comparison. The pilot
freezes 16 paired evaluation layouts and runs holder-trained condition C on both
intentions. On disjoint training reset IDs it screens up to four balanced passes
of 20 layouts, seeking eight verified rescues per first obstacle hit and
destination. Each branch must finish at the requested destination without
contact with either obstacle. Successful source episodes supply preservation
prefixes. The predictor must rank the rescuing prefix safer on at least two
thirds of eligible held-out pairs for each obstacle. The run records a shortfall
before training if either collection or ranking falls short. Otherwise it
continues the original post-fusion checkpoint with the condition C loss and
evaluates the new checkpoint against the frozen baseline schedule. Results and
the advancement decision are in `outputs/pi05_libero_two_obstacle_pilot`.
