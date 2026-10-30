# Cursor shared autonomy with a flow-matching policy

A 2D cursor task in which a flow-matching policy (a small transformer, pi0-style
action chunks) reaches targets, and a person's or a simulated agent's command
steers it: teleop, shared flow control, Flow Reversal Steering (FRS) and FRS with
a reversal adapter (MARS, fixed or learned). The steering methods are lerobot's
pi0.5 steering ([`src/lerobot/policies/pi05/steering.py`](../../../../src/lerobot/policies/pi05/steering.py)),
which the cursor example binds, with the same names and knobs; the LIBERO example's README ([../libero/README.md](../libero/README.md))
describes each mode and knob on pi0.5.

Every experiment, with a person or with a simulated agent, starts from the
same portal and runs through the same mechanism; the only difference is the
agent. Write-ups: `reports/cursor_flow_policy_report.pdf` (each built locally by its `reports/build_*_report.py`; the PDFs are not committed)
(policy and steering simulations), `reports/mars_adapter_report.pdf`
(the learned adapter, with the math) and `reports/grid_report.pdf` (the
composed user × task adapter on the users × tasks grid, every method side by side).

## Quick start

Everything runs in the lerobot environment with the `mars` extra (torch, matplotlib, pygame, reportlab),
from the repo root. Day one, on a fresh clone:

```bash
uv sync --extra mars        # or: pip install -e ".[mars]"
uv run --extra mars python examples/pi05/mars/cursor/hub.py download
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1
uv run --extra mars python examples/pi05/mars/cursor/analyze_results.py --participants p1
```

`hub.py download` fetches the checkpoints the study and the grid use (the folders listed in
`hub_manifest.yaml`, 126 folders, about 97 MB, from the public Hub repo `ricebcilab/mars-cursor`; no login) into
`examples/pi05/mars/checkpoints/`. The session needs the SpaceMouse (see [../README.md](../README.md#spacemouse)
for the udev rule) and a display; `--participant sim` runs the same protocol with the expert agent, headless,
with neither. `analyze_results.py` prints the per-arm table and writes `results/analysis/all/`.

More:

```bash
uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor -q   # ~5 min

uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1          # a person on the SpaceMouse
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1 --resume # continue an interrupted visit
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1 --session 2   # a second visit, new layouts
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant sim         # the expert agent, headless (20 s)
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant sim --watch # the expert, rendered
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --arm flow_control_gain05y.yaml                    # free play with one arm
uv run --extra mars python examples/pi05/mars/cursor/analyze_results.py                                               # tables and figures for every run
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1          # a participant's study report (PDF)
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1 p2 p3 p4 # the group's
uv run --extra mars python examples/pi05/mars/cursor/reports/build_grid_report.py                             # the grid's report (PDF)
uv run --extra mars python examples/pi05/mars/figures/build_figures.py --device cpu                          # the paper's figures (examples/pi05/mars/figures/*.pdf, *.png, stats.json)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep study_condition                             # tune depths, test, pick the condition (25 min)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep task_space                                  # search the task parameters for the largest lead (~3 h)
```

If no window opens (e.g. over SSH), set `DISPLAY=:1`. A study run needs the SpaceMouse (the
block screens wait for it); free play also takes the arrow keys.

`hub.py manifest --check` checks that `hub_manifest.yaml` still lists every checkpoint folder the configs
reference (`configs/study.yaml`'s arms, `configs/sweeps/grid.yaml`'s methods in every cell, and the grid's seeds
1–4, each seed-0 module's `_s0` swapped for `_s<k>`); `hub.py manifest` rewrites it. `hub.py upload` (maintainers, on command only)
creates the public repo if it is missing and uploads each folder's converted files.

`tests/policies/pi0_pi05` (lerobot's pi0.5 tests, not this example's) loads the gated
`google/paligemma-3b-pt-224`: without a Hugging Face token that has accepted its license, those tests fail
with 401 errors.

## Layout

```
cursor_sa/            the library: env, policy, demos (+ the training sampler), steering (modes, reversal
                      adapters), devices, channel, agents, config, session (the experiment mechanism),
                      renderer, records (trial records and their summary), analysis (figure style,
                      kinematics, goal-choice statistics), paths
run_experiment.py     the portal: a protocol for a participant (human or expert), or free play
hub.py                the checkpoints on the Hugging Face Hub: download, manifest (--check), upload
hub_manifest.yaml     the checkpoint folders the study and the grid use (written by hub.py manifest)
convert_checkpoints.py  converts trainer checkpoints (policy.pt, adapter.pt) in place to lerobot's formats
run_sweep.py          tune-then-test sweeps of the steering methods with a simulated agent
analyze_results.py    per-participant and pooled analysis of any runs
train/                train_policy.py, train_adapter.py
research/             policy validation and diagnostics, not experiments: evaluate_policy_goal_given.py,
                      evaluate_policy_goal_hidden.py, design_checks.py, adapter_diagnostics.py, adapter_ranks_*.py,
                      steering_states.py (why steering fails through a retrained policy); each writes
                      results/research/<name>/ with its own log.txt
reports/              the PDFs and the scripts that build them; build_study_report.py --participants p<N> ... writes
                      study_p<N>.pdf for one participant or study_group.pdf for several (paper-style figures)
../figures/           (examples/pi05/mars/figures/, beside this folder) the paper's figures, PDF + PNG, and build_figures.py (one script; `--only fig1 fig2 fig3`):
                      fig1_method (MARS illustrated), fig2_study (the human study), fig3_grid (the user x task
                      grid over its five training seeds); stats.json holds the numbers the captions quote (Fig 1's
                      reproduce with `--device cpu`, the device it was built on: CUDA draws different policy noise). MARS
                      (Modular Adapters for Reversal Steering) is mars's display name: MARS = user + task modules,
                      MARS-U = the user module alone, MARS-T = the task module alone; the code's tokens are unchanged
configs/              arms (one control condition each), condition_*.yaml, study.yaml (the protocol),
                      sweeps/*.yaml (sweep definitions)
../checkpoints/       IGNORED  (examples/pi05/mars/checkpoints/) policies/<name>/, adapters/<name>/ (weights +
                      training logs; none tracked; hub.py download fetches the study's and the grid's)
../results/           IGNORED  (examples/pi05/mars/results/) p<N>/ and sim/ (runs), sim/sweeps/<name>/
                      (run_sweep.py), research/, analysis/
```

The tests live in lerobot's `tests/`: `tests/examples/pi05_mars_cursor/` (this example's), and
`tests/policies/cursor_flow/` and `tests/teleoperators/test_spacemouse.py` (its `src/` pieces). Paths below
such as `checkpoints/` and `results/` are under `examples/pi05/mars/`.

## How an experiment runs

- **Participant.** `p<N>` is a person, `sim` a simulated agent: `expert` pushes straight at the goal
  and never adapts (with obstacles it plans a route around them that keeps clear of the other targets
  and passes on the task's side, and pushes toward the farthest point of the route it can see with a
  margin, so it never cuts a corner; pushed off the route it re-plans from where the cursor is);
  `expert_chunk` means the
  goal-given policy's own chunk, step by step (the BCI
  model the learned adapter was trained on; used by the learned-adapter sweeps). All go through the
  same input channel (corruption, noise), the same controller and the same records.
- **Arm.** One control condition, a config under `configs/` (policy, mode and depth, channel, task).
  The four study arms extend `condition_study.yaml`: gain diag(1, 0.25) on the command plus input
  noise 1.0, 4 targets per trial at a reach of 0.35–0.5, each method at the depth tuned for that
  setting.
- **Protocol** (`configs/study.yaml`): the arms, trials and practice trials per block, the block
  order rule, trial timing. One block per arm.
- **Blocks.** The order is a row of a balanced Latin square chosen by the participant number and
  advanced by one row per visit (p1 → row 1 on visit 1, row 2 on visit 2, …; p2 starts at row 2;
  sim at row 0), so repeated visits of one person are counterbalanced too (since 2026-10-19; p1's
  five visits of 2026-10-18 all used row 1). Within a visit every arm sees the **same target layouts**
  (fixed from the participant and session), shuffled per block; practice layouts are a separate
  small set. The noise is drawn from its own generator, so the layouts never depend on it.
- **Trial.** The layout is shown; the trial starts once the stick has been idle for 0.3 s; the
  person has 5 s (`task.time_limit_s`) to move onto the gold target and hold for 0.3 s
  (`task.dwell_s`); 0.6 s of feedback. The task geometry and timing are arm parameters (`task:`),
  and every arm of a protocol must agree on the layout ones (targets, radius, spacing, reach),
  since the arms share layouts. Outcomes are
  `hit`, `wrong_target`, `collision` (the cursor touched an obstacle), `timeout` and `aborted` (Esc). A person sees only block and trial numbers
  (`display.blind`). A simulated agent runs a block's trials in one batch, headless; `--watch`
  renders it.
- **Records.** `results/<participant>/<protocol>_s<session>/manifest.json` (order, layout seed,
  resolved arms with their matrices, checkpoint SHA-256s, git commit, environment) and one
  `block<k>_<arm>.jsonl` per block, one record per trial: block, arm, method, trial, practice,
  layout id, start and end times, duration, outcome, targets, path, raw and served commands,
  per-frame times (real-time runs), and the arm's settings. Free play logs to
  `results/<participant>/free/<timestamp>.jsonl`.
- **Resume.** Re-running the same participant and session with `--resume` continues from the first
  unfinished trial; without it the portal refuses to overwrite.
- **Analysis.** `analyze_results.py` writes `results/analysis/<name>/`: `trials.csv`, `summary.csv`,
  `summary.md` (per participant × arm, and the mean over people), `success_by_arm.png`,
  `learning_curves.png` (hit rate against trial index, per arm) and `paths_<participant>.png`.
  `reports/build_study_report.py --participants p<N> ...` pools a participant's
  sessions into `reports/study_p<N>.pdf`: hit rates with bootstrap CIs and the MARS lead paired
  by layout, outcomes, per-session rates, cumulative hits against time and reach-aligned paths. Several
  participants give `reports/study_group.pdf`, with the participant as the unit (each one's sessions
  pooled, then mean ± s.e.m.; the lead per participant). Figures go to `results/analysis/study_<p|group>/`.

## Sweeps

`run_sweep.py --sweep <name>` runs the tune-then-test protocol with a simulated agent on the same
trial mechanism: for every grid cell (a corruption level × an input noise, × the task axes if any)
each method's depth (`tau`, `n_reverse_steps`) is tuned on the tuning trials, then every
method is tested at its tuned depth on new trials, with a paired bootstrap 95% CI on the focus
method's lead. A sweep is a file under `configs/sweeps/` naming the agent, the base arm, the methods
(arm overrides), the depths (one list, or a list per mode, since flow control's `tau` counts chunk
steps of 16 and the other depths Euler steps), the grid (`family: gain | rotation | curl | none`, `levels`, `noises`,
and optionally `axes`: arm parameters such as `targets`, `min_target_sep`, `reach` or `dwell_s`,
each with a list of values, every combination a cell), the trial counts and the focus. Cells whose
layouts cannot be sampled (too many targets for the spacing) are skipped and listed in `test.json`.
Outputs go to `results/sim/sweeps/<name>/`: `tuning.json`, `test.json`, `depths.yaml` (the tuned
depths, to copy into arm configs), `ranking.csv` (every cell, best lead first by the lower end of
its CI), `log.txt` (everything the sweep printed) and `figs/` (depth curves, the test map or lines, or `axes.png` with the lead against every
varied parameter, the showcase cell or, for a users × tasks grid, the grid's maps and summary, and
center-out `traces.png` when the sweep asks for them).
Results are saved after every group of cells; `--resume` continues an interrupted sweep. Every cell
draws its layouts, noise and bootstrap from seeds of its own, so `--cells` can run parts of a sweep in
parallel processes (each with its own `--out`) and `--merge` combines them, cell for cell the same as
one run: worth it when the expert's route planning makes a cell CPU-bound (the grid's obstacle cells).
A sweep can also run a **users × tasks grid** (`grid.users`: named corruption profiles with their
user-module checkpoint; `grid.tasks`: named obstacle specs with their task-module checkpoint and,
if a task needs its own, a `time_limit_s`; `grid.train`: the cells trained on). A method's
`reversal_adapter` may say `cell` in its user or task slot, resolved per cell (`{user: cell, task:
cell}` is the composed adapter), and a method's `model` or adapter slot may be given per user or per
task, nested in either order (`model: {T1: …, T2: …}`, `user: {T2: {U1: …, U2: …}, …}`: the grid's
retrained policy, one for each task with user modules calibrated on it). Cells are keyed `UxT|noise`; each
stores the focus method's paired lead over every other method, and gets `figs/grid.png` (success,
lead over the best of `against` and collisions over users × tasks, train cells outlined),
`figs/summary.png` (every method: success averaged over the cells with and without an obstacle, then
per cell the best of `against` beside every other method, with 95% CIs; methods that play the same arm
on a task, e.g. a task-module arm where the task has none, are named below it) and `grid_summary.md`
(by user, by task, trained-on against held-out).

| Sweep | Agent | Grid | Takes |
|---|---|---|---|
| `gain` | expert | vertical gain {1, 0.75, 0.5, 0.25, 0.1} × noise {0.75 … 2.5} | 63 min |
| `rotation` | expert | rotation {10°, 20°, 45°, 90°} × noise {0.25 … 1.75} | 38 min |
| `curl` | expert | curl {0, 5, 10, 20, 40}/s × noise {0, 0.3, 0.75}, with traces | 39 min |
| `study_condition` | expert | gain 0.25 × noise {0.25 … 1.75}, with the learned adapter | 23 min |
| `flow_control_gain` | expert | flow control's `ic_gain` {0.5, 1, 2, 3, 5, 8} × tau at the study condition, beside teleop and the study's own setting (action units, lambda 1 and 2); fixes `ic_gain` for every arm | 3 min |
| `learned_adapter_rot90`, `learned_adapter_gain025y` | expert_chunk | one corruption × noise {0, 0.5, 1}, all adapters | 27 min each |
| `task_space` | expert | gain 0.25 × noise {1, 1.25, 1.5} × targets {2, 4, 8, 12} × spacing {0.1, 0.12, 0.16, 0.24} × reach {0.2–0.35, 0.35–0.5, 0.5–0.75} × dwell {0.3, 0.5 s}, with the learned adapter; 100 tuning and 200 test trials, depths {1, 2, 3, 4, 6, 8, 10} | 113 min |
| `task_space_refine` | expert | gain 0.25 × noise {1, 1.25} × targets {3, 4, 5, 6} × reach (as above) × dwell {0.3, 0.5 s}; the full protocol, new seed | 50 min |
| `grid_expert_check` | expert | the expert alone (teleop, clean channel) and the frozen policy alone on the grid's tasks T1–T5 | 10 min |
| `grid_task_check` | expert_chunk | the goal-given policy's own chunks executed, with and without each task module, and each task's retrained policy alone (T2–T5) | 2 min on a free GPU |
| `grid` | expert | users {U1 … U4} × tasks {T1 … T5} at noise 1.0: the composed user × task adapter, each module alone, each task's retrained policy with its user modules, teleop, flow control and FRS (research notes) | ~2 h in 20 parallel parts, a cell each (`--cells`, then `--merge <parts> --out grid_s0` for the grid of record; without `--out` it writes `sweeps/grid/`); the cells add up to ~18 h |

Times are on one RTX 6000 Ada. Depths only need re-tuning when something that can move them changes:
the condition, a retrained adapter or a retrained policy.

## Conventions

Names are unified across scripts, configs, log fields, folders and figures:

- Methods: `teleop`, `policy`, `flow_control`, `frs`; `frs` with a `reversal_adapter` is `mars` (Modular
  Adapters for Reversal Steering). MARS with learned modules is the default and carries no qualifier:
  arm configs `mars_<condition>.yaml`, `study_mars.yaml`, sweep methods `mars` (user + task modules),
  `mars_user` (MARS-U), `mars_task` (MARS-T), `retrained_mars_user`, `mars_free` / `mars_goal` (the
  learned module's training pairs). MARS with the closed-form adapter F = M is `mars_fixed`
  (`mars_fixed_<condition>.yaml`). Runs recorded before 2026-10-26 name these `frs_ra` and
  `frs_ra_learned`; `records.ARM_ALIASES` maps the recorded sessions' arm names when they are loaded.
  `flow_control` is Kao et al.'s flow control with three knobs: the command, scaled by `ic_gain` to the
  noise prior's units, is written into the first `tau` of the chunk's 16 steps of x_t at the start of
  each of the first `lambda` Euler steps, and the flow itself runs unmodified. `lambda` 1 writes the
  initial condition only (Kao et al.'s method, the default), 0 writes nothing (the policy alone), more
  re-writes as the flow proceeds (lerobot's pi0.5 `shared_flow_control`, whose own "tau" is this lambda). The
  human study ran that implementation at its depth 1, which is `tau` 16, `lambda` 1 and the command
  in the policy's action units (`ic_gain` 2.0843, `steering.action_gain`): Kao et al.'s method with the
  whole chunk written. Its records, and every sweep result before git `029b3ed6`'s successor, say
  `mode: flow_control, tau: 1` for that (no `lambda`, no `ic_gain` field).
  An adapter spec is `{translation: corruption}` (fixed F = M), `{F: rows}`, `{learned: <name>}` (one
  checkpoint) or the composite `{user: <name | null>, task: <name | null>}`; checkpoints
  live in `checkpoints/adapters/<name>/` with a `kind` (`user` or `task`; the first ones, saved without
  a kind, are user modules).
  LeRobot's `shared_flow_control` and `shared_flow_reversal_steering` are accepted as aliases.
- Depth: `tau` (flow control: chunk steps, of 16) and `n_reverse_steps` (FRS, MARS), as the sweeps tune
  them; "depth" for both. Flow control's `lambda` and `ic_gain` are fixed by the arm.
- Agents: `human`, `expert`, `expert_chunk`. Participants: `p<N>`, `sim`.
- Policy goal: `goal_given` / `goal_hidden` in outputs; "goal dropout" is the training technique.
- Corruptions in names give the axis and the value: `gain025y` is a gain of 0.25 on the vertical axis
  (diag(1, 0.25)), `gain025x` on the horizontal one, `gain05y` 0.5 vertical, `rot60` and `rot90` a
  rotation in degrees, `curl20` a curl of 20/s. These replaced `gain025`, `gain` and `rot` on Oct 24;
  participant and study-session records from before keep the names they were recorded with (the
  study's adapter `gain025y_free` is the file their manifests hash as `gain025_free`).
- Channel: `corruption` (a matrix spec) and `input_noise` (a std); condition files are
  `condition_<what>.yaml`; sweep cells are keyed `<level>|<noise>`, plus `|<axis>=<value or name>` per
  task axis (an axis given as a mapping `name: value` is keyed by its names).
- Scripts are `verb_object.py` (`run_experiment.py`, `build_policy_report.py`); library modules are
  nouns. Record fields: `t_start`, `t_end`, `duration_s`, `outcome`.

## Configs

`configs/base.yaml` holds every arm parameter with a comment. Each arm says `extends: base.yaml` (or
another arm) and overrides only what it changes; a bare name is looked up in `configs/`; unknown or
missing keys are errors. In free play any arm parameter can be overridden on the command line
(`--speed-gain 0.3`, `--tau 8`, `--corruption null`).

| Arm | Method | Depth | Condition |
|---|---|---|---|
| `teleop.yaml`, `flow_control.yaml`, `frs.yaml` | teleop, flow_control (tau 8 of 16, Kao et al.'s default), frs | | none |
| `teleop_gain05y.yaml`, `flow_control_gain05y.yaml`, `frs_gain05y.yaml`, `mars_fixed_gain05y.yaml` | … , mars (F = M) | tau 8 (to re-tune), full reversal | `condition_gain05y.yaml`: gain diag(1, 0.5), noise 0.75 |
| `teleop_rot90.yaml`, `flow_control_rot90.yaml`, `frs_rot90.yaml`, `mars_fixed_rot90.yaml` | … , mars (F = M) | tau 8 (to re-tune), n 9, n 6 | `condition_rot90.yaml`: 90° rotation, noise 0.25 |
| `teleop_gain025y.yaml`, `frs_gain025y.yaml`, `mars_fixed_gain025y.yaml`, `mars_gain025y.yaml` | … , mars (F = M), mars (learned) | full reversal | `condition_gain025y_noise1.yaml`: gain diag(1, 0.25), noise 1.0 |
| `study_teleop.yaml`, `study_flow_control.yaml`, `study_frs.yaml`, `study_mars.yaml` | the study arms (flow control as the sessions ran it: tau 16, lambda 1, the command in action units) | tau 16 / gain 2.08, full reversal, n 8 (tuned) | `condition_study.yaml`: gain diag(1, 0.25), noise 1.0, 4 targets, reach 0.35–0.5 |
| `teleop_curl.yaml` | teleop | | curl 10/s on the cursor, inertia 0.1 s |
| (sweep base) | | | `condition_grid.yaml` (the grid): noise 1.0, 4 targets, reach 0.35–0.5, 8 s per trial (T1: 5 s, set by the sweep), speed gain by time |

Main parameters (`control:` and `task:` sections):

- `speed_gain` (default 0.5, half the demos' speed) scales the cursor's speed in every mode, policy
  and agent alike; `gain_mode` says how. `step` (the default and the study) scales every executed step
  and gives the policy's velocity history the unscaled displacement, so it keeps planning as if it
  moved at its normal speed while its position advances at half the rate (feeding it the slowed
  displacement instead made the policy alone stall: a target selected in 56% of trials against 100%).
  `time` (the grid) plays every planned step over 1/gain control steps, so the cursor follows the
  planned path at half speed and the history holds one entry per planned step, the motion the policy
  was trained on; chunks, and so input reads, then come half as often per second. On the open field the
  two are alike for the policy alone; on a route around an obstacle only `time` keeps the policy on it
  (research notes, the grid's checks).
- `tau` and `ic_gain` are flow control's knobs: `tau` chunk steps (of 16) take the command, at `ic_gain`
  sigma for a full push, as their initial condition (Kao et al.: 8 steps; their gain put the median
  command near 0.5 sigma). `ic_gain` is 2.0, chosen once on the study condition like Kao et al.'s gain
  (`flow_control_gain` sweep: 0.5 / 1 / 2 / 3 / 5 / 8 give 39 / 44 / 73 / 63 / 62 / 12% at the tuned tau;
  at 1 the policy mostly ignores the command and picks a wrong target, at 5 and above the chunk is pure
  command and stalls); `tau` is tuned per sweep cell. `lambda` is the Euler steps at whose start the
  command is written: 1 (the default) is Kao et al.'s initial condition, 0 the policy alone, more the
  lab's implementation, which re-writes x_t as the flow proceeds. The study's arm is tau 16, lambda 1 and
  the command in action units (`ic_gain` 2.0843, 2.1 sigma for a full push); at the study condition that
  scale reaches 69–73% (tau tuned to 12, lambda 1 or 2), within noise of `ic_gain` 2 at tuned tau (73%).
  `n_reverse_steps` is FRS's depth.
- `input_noise` and `corruption` act on the command on its way to the controller, as in LeRobot:
  raw → M·raw (clipped to [-1, 1]) → + Gaussian noise. Both apply only while pushing, so idle input
  stays idle. Corruption specs: `{scale: [sx, sy]}`, `{rotation_deg: d}`, `{M: rows}`, or `{curl: b,
  tau: s}`: a curl field on the command (the dynamics of the cursor curl below, acting on what the
  controller gets instead of on the cursor: the served command is a velocity driven by the push,
  turned clockwise by atan(τb) and slowed by 1/√(1 + (τb)²) at steady state, from rest each trial).
- `reversal_adapter`: a `ReversalAdapter` (`cursor_sa/steering.py`) with up to two modules. The
  **user module** (a fixed F = M, or a trained `UserModule` U(x_t, t)) multiplies the policy's
  velocity in the backward pass of the reversal only (frs): it undoes the user's command corruption.
  The **task module** (`TaskModule`: A(x_t, t, v, obstacles)·v + b, from obstacle tokens seen from
  the cursor) acts in both directions, so it reshapes every chunk the controller produces — in every
  mode, and the `expert_chunk` agent's intent — toward a task the frozen policy never saw. The
  forward field is task(v), the backward field user(task(v)); the two are trained apart and composed
  here (`{user: U, task: T}`).
- `execute` (default 2): the input is re-read each time the policy replans. LeRobot replans every
  chunk, but here that made steering much worse than direct control: with the expert agent at input
  noise 0.6, replanning every 8 steps gave 34% success (66% timeouts, from failing to hold still on
  the target) against 100% for teleop, while every 2 steps gave 97% (`research/design_checks.py`).
- `velocity_tau` and `curl`: cursor inertia and a clockwise curl force field on the cursor (see the
  curl section below).
- The task geometry and timing: `targets` [min, max] per trial, `target_radius` (0.05),
  `min_target_sep` (0.12, center to center), `reach` [min, max] from the cursor start to the goal
  ([0.2, null]: no maximum), `dwell_s` (0.3) and `time_limit_s` (5). The defaults are the task the
  policy was trained on. In free play: `--targets 4 4 --reach '[0.3, 0.6]' --dwell-s 0.5`.
- `obstacle` (null): static discs placed relative to the reach, the line from the cursor's start to
  the goal; touching one ends the trial with outcome `collision`. `{kind: disc, radius, along, offset,
  side}` puts one disc `along` the reach (a fraction) and `offset` across it; `{kind: wall, along,
  gap, gap_offset, radius: 0.03, extent: 0.25}` a row of touching discs across the reach with a gap;
  `{kind: ring, around: goal | start, ring_radius: 0.14, radius: 0.02, gap: 0.16, gap_angle: 0}` an
  arc of touching discs around the goal or the start, open by `gap` on the side facing away from the
  reach and turned by `gap_angle` degrees. `side` (random | fixed) draws per trial the side the offset,
  the gap or the turn goes to; a disc's `offset` and a ring's `gap_angle` may be a range [lo, hi],
  drawn per trial. Layouts keep obstacles clear of the targets and the start and passable on screen
  (an impossible spec is a "cannot place obstacles" error, which a sweep skips). The expert routes
  around obstacles (`env.plan_route`, a visibility graph over the discs inflated by a clearance), and
  demonstrations follow the expert's path, timed as one human-like movement; records carry each
  trial's discs (`obstacles`), the arm its spec. The study's policy does not see obstacles; a policy
  trained with `--obstacle` reads them as tokens (`flow_goaldrop_<task>`, the grid's ceiling).

## Training

From the repo root, as the quick start:

```bash
uv run --extra mars python examples/pi05/mars/cursor/train/train_policy.py --mode flow --name flow_goaldrop \
    --goal_dropout 0.25   # ~10 min on one RTX 6000 Ada
# a user module from 100 open-field demonstrations, then a task module from 300 of the task's demonstrations with
# that user's module in the loop (the grid)
uv run --extra mars python examples/pi05/mars/cursor/train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' \
    --name user_gain025y_expert_n100_s0 --n-demos 100 --steps 500 --frozen-noise   # under a minute
uv run --extra mars python examples/pi05/mars/cursor/train/train_adapter.py --kind task \
    --obstacle '{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024]}' \
    --user-module user_gain025y_expert_n100_s0 --corruption '{scale: [1.0, 0.25]}' \
    --name task_disc_large_n300_s0_with_gain025y --n-demos 300 --frozen-noise   # ~13 min
# the grid's retrained policy: an obstacle head, trained from scratch on 20k of one task's demonstrations (a quarter
# without the obstacle; --init flow_goaldrop fine-tunes the study's policy instead), then a user module calibrated
# on it
uv run --extra mars python examples/pi05/mars/cursor/train/train_policy.py --name flow_goaldrop_disc_large \
    --goal_dropout 0.25 --targets 4 4 --reach 0.35 0.5 --obstacle '{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024]}'
uv run --extra mars python examples/pi05/mars/cursor/train/train_adapter.py --kind user --model flow_goaldrop_disc_large \
    --corruption '{scale: [1.0, 0.25]}' --name user_gain025y_expert_n100_s0_on_disc_large --n-demos 100 --steps 500 --frozen-noise
# then convert what they wrote to lerobot's formats (below)
uv run --extra mars python examples/pi05/mars/cursor/convert_checkpoints.py
```

`train_adapter.py` trains one module of the reversal adapter, on the study's geometry (4 targets,
reach 0.35–0.5) at input noise 1.0 unless told otherwise. A **user module** learns to undo one
command corruption, calibrated open loop by an expert that knows the goal: the label is the
demonstration's next chunk, the command the expert's push through the user's channel held over the
chunk, and the field regression pulls the reversal of that command onto the label's latent. With
`--closed-loop <initial modules>` the module is recalibrated ReFIT-style: the initial module runs closed loop on
the open field at the grid's noise (`--closed-loop-trials` per module, 2000 in 10 s), every visited state joins
the data with the expert's push there and a straight min-jerk reach to the goal as its label, and the batches
mix open- and closed-loop states (`--closed-loop-mix`, 0.5); several initial modules pool their trials
(DAgger-style rounds). A **task module** learns, by flow matching
through the adapted field with the policy frozen, to reshape the chunk distribution onto
demonstrations around the task's obstacles (a share `--p-free` of layouts without them, where it
should stay the identity; `--lambda-id` pulls A, b toward the identity), with a user's frozen user
module in the loop (`--user-module` and that user's `--corruption`; `--lambda-field` weighs the field
regression through both modules on that user's commands). Its log reports the chunk-collision rate
of the adapted and the plain policy on held-out states, and of the chunks regenerated from the
user's commands. The checkpoint carries its `kind`; the first adapters (`gain025y_*`, `rot90_*`, no
kind) are user modules. The trainer as it was at git commit 5c2167cd also trained user modules from
the policy's own chunks (`gain025y_*`, `rot90_*` and grid 1's) and task modules without a user.

`--n-demos` sets how many demonstrations a module trains on (default 20 000, as many as the policy's own;
one demonstration is one reach, about a second). With few, the two modules want opposite budgets (the
data-efficiency note under the composed adapter): a user module over-fits the default 20 000 steps and
should train about five steps per demonstration (`--steps`, at least 100), while a task module needs the
full 20 000 steps whatever its data. `--frozen-noise` draws the channel's noise once per demonstration
step, as a recorded calibration has it, instead of anew in every batch.

Checkpoints go to `examples/pi05/mars/checkpoints/policies/<name>/` and `.../adapters/<name>/`, which git
ignores. The ones the study and the grid use (the study's policy and MARS adapter, the grid's policies and
modules of all five training seeds, each named with `_s<k>` after its demonstration count) are on the Hub
(`ricebcilab/mars-cursor`, listed in `hub_manifest.yaml`); `hub.py download` fetches them. The others named in
the research notes below (the 20 000-demonstration modules, grid 1's, the data-efficiency runs, `rot90_*`)
are not distributed.

The trainers still write the cursor project's formats (`policy.pt`, `adapter.pt`), and the loaders
(`FlowPolicy.load`, `ReversalAdapter.load`) read both those and lerobot's, so a fresh checkpoint works before it
is converted. After training, rerun `convert_checkpoints.py` to get the lerobot format: a policy's folder
becomes a `cursor_flow` policy (`config.json` + `model.safetensors`, loaded by
`CursorFlowPolicy.from_pretrained`), and an `adapter.pt` takes the `save_adapter_module` format (kind, config,
weights, meta, and a task module's `arch` written out). The originals stay beside them as `*.orig`, and running
it again changes nothing. When a policy folder holds both formats (a retrained `policy.pt` beside an older
converted `model.safetensors`), `policy.pt` wins, with a warning, until you convert again. `hub.py upload`
refuses unconverted folders.

## Research notes

Everything below was produced on the portal's mechanism (2026-10-15) by the sweeps above and the
`research/` scripts; the two reports carry the full tables and figures.

### The policy

**Task.** Screen coordinates are [0, 1]² at 50 Hz (20 ms steps). Each trial has a random cursor start
and 1–N random targets, one of which is the goal. Holding the cursor on any target for 0.3 s selects
it and ends the trial; selecting a non-goal target is a failure; nothing selected after 5 s is a
timeout. Movement rules match `CursorControlTask` in the lab's shared-autonomy benchmark (not part of this repo).

**Demonstrations** (`cursor_sa/demos.py`). Each reach is built from standard models of human reaching:
a minimum-jerk position profile, a duration set by Fitts' law with random jitter, a random sideways
bow of up to ±25% of the reach distance, and on 40% of reaches a missed endpoint fixed by an
overlapping corrective sub-movement. After arriving, the cursor holds still on the goal until it is
selected. Reaches between the same start and goal therefore vary, which gives the flow head a real
distribution to learn.

**Policy** (`cursor_sa/policy.py`). A transformer over one cursor token (position and its last 4
displacements), one token per target (position, offset from the cursor, goal flag) and 16 action
tokens (that step's noisy action and the flow time). Targets get no order embedding and padded
targets are masked out, so any number of targets works and the output is identical however they are
ordered (tested). As in pi0, cursor and target tokens do not attend to the action tokens. Flow
matching: loss ‖v_θ(x_t, t | o) − (a − z)‖² with x_t = (1 − t)z + t·a; sampling runs 10 Euler steps
from t = 0 (noise) to t = 1 (action chunk); `FlowPolicy.integrate` can run the flow backward too.
0.9M parameters (d_model 128, 4 layers, 4 heads). The regression baseline is the same network with
a mean-squared-error loss.

**Results (goal given).** Both models trained with 20k demos and 1–8 targets per trial for 40k steps;
500 new layouts per target count (`research/evaluate_policy_goal_given.py`,
`results/research/policy_goal_given/`).

| Targets | Method | Success | Median time to select (s) | Path efficiency | Sideways bow sd |
|---|---|---|---|---|---|
| 4 | demos | 100% | 0.90 | 0.975 | 0.14 |
| 4 | flow | 100% | 0.90 | 0.978 | 0.13 |
| 4 | regress | 100% | 0.89 | 1.000 | 0.03 |
| 16 | demos | 100% | 0.92 | 0.967 | 0.15 |
| 16 | flow | 100% | 0.91 | 0.979 | 0.13 |
| 16 | regress | 100% | 0.90 | 0.999 | 0.04 |

Both policies succeed on 100% of trials at every count, including 12 and 16 targets (more than
training ever showed), and time to target does not change with the number of distractors. The flow
head learns the distribution of reaches (paths bow left and right like the demos; bow sd 0.13 against
0.15 for the demos and 0.03 for regression), though it under-produces the most curved reaches. With
the goal given, the average path is already a good path, so regression does just as well on success.

### Goal-hidden policy (goal dropout)

Two models were trained the same way but with 25% of training samples' goal flag cleared:
`flow_goaldrop` and `regress_goaldrop`. In the training data the goal is a random target at least
0.2 from the start, so without a flag an ideal policy picks one of those targets at random and commits
to it. `research/evaluate_policy_goal_hidden.py` runs 200 layouts × 20 runs per target count with
every goal flag cleared (`results/research/policy_goal_hidden/`, `policy_goal_hidden_nodropout/`).

| Targets | Model | Selected a target | Spread of choices (ideal) | Picked a target closer than 0.2 | Median time (s) |
|---|---|---|---|---|---|
| 4 | flow, goal dropout | 100% | 0.99 (0.95) | 3.7% | 0.94 |
| 4 | regress, goal dropout | 99.5% | 0.00 (0.95) | 4.5% | 1.02 |
| 4 | flow, no dropout | 7.4% | 0.20 (0.95) | 20% | 2.80 |
| 8 | flow, goal dropout | 100% | 0.93 (0.90) | 7.5% | 0.94 |
| 8 | regress, goal dropout | 99.0% | 0.00 (0.90) | 15% | 1.02 |
| 8 | flow, no dropout | 17% | 0.37 (0.90) | 28% | 2.92 |

Spread of choices is the entropy of each layout's 20 choices divided by log(number of eligible
targets); the ideal is the same statistic for a uniform random choice with 20 runs. The flow policy
picks a target and commits to it, as fast as with the goal given, and spreads its choices across
runs about as evenly as the ideal; the flow head's noise decides the target and the velocity history
keeps the policy on it after replanning. Regression also commits, but to the same target every run.
Without dropout training, hiding the goal breaks the policy (it wanders). Reaching with the goal
given is barely affected (99.2–100% success; `policy_goal_given_dropout/`). This goal-hidden flow
policy (`flow_goaldrop`) is the one every steering experiment uses: without a goal its output already
holds one option per target, so the agent's command only has to choose among them.

### Steering simulations

All with the expert agent, 2–8 targets, speed gain 0.5, replanning every 2 steps. The tuned sweeps
below were re-run on 2026-10-26 with Kao et al.'s flow control (`ic_gain` 2, tau tuned); the first
table, the design checks, ran the lab's implementation (tau 16, lambda 5, action units). A first check with
the gain corruption and little noise (`research/design_checks.py`, 300 trials per cell, depths as
shipped):

| Config | Success, noise 0 / 0.3 | Wrong target, noise 0 / 0.3 |
|---|---|---|
| `teleop`, `flow_control`, `frs` (no corruption) | 100% / 100% | 0% / 0% |
| `teleop_gain05y` | 97.3% / 97.0% | 2.7% / 3.0% |
| `flow_control_gain05y` | 99.7% / 98.7% | 0.3% / 1.3% |
| `frs_gain05y` | 96.0% / 97.0% | 4.0% / 3.0% |
| `mars_fixed_gain05y` | 99.3% / 99.3% | 0.7% / 0.7% |

**Tuned sweeps** (`run_sweep.py`, 200 tuning and 500 test trials per cell; tuned depth in brackets).
Gain corruption diag(1, g) × input noise:

| g | Noise | Teleop | Flow control | FRS | MARS |
|---|---|---|---|---|---|
| 0.5 | 0.75 | 95.4% | 86.4% (14) | 92.4% (1) | 91.4% (7) |
| 0.5 | 1.5 | 67.0% | 68.8% (10) | 41.2% (5) | 50.2% (10) |
| 1.0 | 2.0 | 77.6% | 71.4% (10) | 47.0% (6) | 47.0% (6) |
| 0.25 | 1.0 | 51.4% | 68.0% (16) | 41.8% (5) | 61.0% (10) |
| 0.1 | 0.75 | 29.6% | 61.4% (16) | 25.6% (5) | 64.0% (10) |
| 0.1 | 1.0 | 26.4% | 56.8% (14) | 20.6% (4) | 49.6% (10) |

MARS beats every other method in none of the 25 settings beyond noise (its best lead over the best
other method is +5.0 [−0.2, +10.2] points, at g 0.25, noise 0.75). With weak or no gain corruption
(g ≥ 0.5) teleop is best; with strong corruption (g ≤ 0.25) shared control beats teleop and the
adapter adds 19–38 points over plain FRS, but flow control, at tau 14–16 of 16, is level with it or
ahead.
`n_reverse_steps` 9 performs badly everywhere: with LeRobot's schedule, a partial reversal of n of
N steps finishes in N − n forward steps, so n = 9 covers 90% of the flow in one Euler step.

The same protocol with a rotation of the command (`--sweep rotation`):

| Rotation | Noise | Teleop | Flow control | FRS | MARS | MARS lead [95% CI] |
|---|---|---|---|---|---|---|
| 45° | 0.75 | 99.6% | 68.8% (16) | 95.6% (2) | 97.4% (3) | −2.2 [−3.8, −0.8] |
| 45° | 1.25 | 88.6% | 64.4% (14) | 70.2% (6) | 74.2% (6) | −14.4 [−18.8, −10.0] |
| 90° | 0.25 | 2.0% | 33.8% (6) | 6.8% (9) | **92.2%** (6) | +58.4 [+54.0, +63.0] |
| 90° | 0.75 | 3.0% | 32.4% (14) | 4.4% (6) | **74.6%** (6) | +42.2 [+36.8, +47.6] |
| 90° | 1.25 | 2.6% | 31.0% (8) | 2.0% (5) | **48.6%** (6) | +17.6 [+11.8, +23.4] |
| 90° | 1.75 | 2.2% | 26.4% (14) | 2.2% (8) | **32.8%** (7) | +6.4 [+1.0, +11.8] |

At 90°, tuned MARS beats every other method at every noise level: an agent pushing straight at the
goal makes the cursor circle it, so teleop fails, plain FRS steers toward the rotated command, and
flow control's best is about chance; MARS's adapter (the known rotation) recovers the intent. At
45° or less teleop wins. Caveats: the adapter uses the known corruption, and the expert never
compensates for the rotation; a person would within a few trials.

**Curl force field.** `cursor_sa/env.py` can give the cursor inertia (`velocity_tau`) and a
clockwise curl field (`curl`, b in 1/s): dv/dt = (u − v)/τ + b·R·v. At steady state this turns the
command clockwise by atan(τb) and slows it by 1/√(1+(τb)²); reaches start straight and curve as the
cursor speeds up. Unlike the gain and rotation, the curl acts on the cursor itself, so it also bends
the policy's own chunks. `--sweep curl` (τ = 0.1 s; `figs/traces.png` shows center-out reaches of
every method at its tuned depth):

| Curl b | Noise | Teleop | Flow control | FRS | MARS (F = S⁻¹) | MARS lead [95% CI] |
|---|---|---|---|---|---|---|
| 10 | 0 | 99.8% | 0.0% (2) | 100.0% (1) | 99.0% (1) | −1.0 [−2.0, −0.2] |
| 10 | 0.75 | 88.6% | 0.0% (2) | 78.4% (9) | 82.6% (1) | −6.0 [−9.2, −2.6] |
| 20 | 0 | 44.6% | 0.0% (2) | 44.2% (9) | **81.6%** (2) | +37.0 [+32.2, +41.6] |
| 20 | 0.75 | 23.6% | 0.0% (2) | 25.0% (9) | **50.2%** (3) | +25.2 [+20.8, +29.6] |
| 40 | 0 | 0.0% | 0.0% (2) | 0.0% (1) | **63.2%** (2) | +63.2 [+59.0, +67.4] |
| 40 | 0.75 | 0.2% | 0.0% (2) | 0.0% (8) | **22.8%** (3) | +22.6 [+19.0, +26.2] |

FRS tracks teleop. Flow control fails outright under the cursor's inertia: the command enters only as
the chunk's initial condition, the policy's prior sets the speed from the lagging history, and the
cursor crawls (median step a third of FRS's at b = 0, 63% there) and times out (100% from b = 5 up;
the lab's implementation, lambda 10, managed 44–61% at b = 10). MARS needs the inverse of the curl's steady-state map
as its adapter (with the map itself it failed at every depth) and then plans counter-curved chunks,
like an internal model of the field. Why F = S⁻¹ works here while F = M worked for the rotation is
not understood yet.

### Learned reversal adapter (MARS)

`train/train_adapter.py` learns the reversal adapter instead of fixing F = M. The policy turns noise
a_noise into a chunk a_policy; the BCI delivers a corrupted, noisy version a_bci = f(a_policy). Along
x_t = (1 − t)·a_noise + t·a_bci, an adapter H (one 2×2 matrix per chunk step, from x_t and t,
initialized to I) is trained so that H(x_t, t)·P(x_t, t) = a_bci − a_noise, with P the frozen
goal-hidden policy field. Training pairs come from the goal-given flow (`--data goal`) or the
goal-hidden flow (`--data free`), with the trainer as it was at git commit 5c2167cd. The `learned_adapter_*` sweeps run the `expert_chunk` agent (the
operator means the goal-given policy's chunk; the channel corrupts every step) against every method;
at the full reversal (300 trials):

| Corruption, BCI noise | User alone | FRS | MARS, F = M | Learned H (goal pairs) | Learned H (free pairs) | Policy alone |
|---|---|---|---|---|---|---|
| rot 90°, 0 | 0.0% | 0.0% | **47.3%** | 25.0% | 27.0% | 26.3% |
| rot 90°, 1.0 | 0.0% | 0.3% | **32.7%** | 19.7% | 20.7% | 22.0% |
| gain [1, 0.25], 0 | 20.3% | 47.3% | **72.3%** | **72.3%** | 71.3% | 26.3% |
| gain [1, 0.25], 1.0 | 11.7% | 10.3% | 25.7% | 44.0% | **45.0%** | 22.0% |

Under the gain corruption the learned adapter matches F = M at low noise and beats it at high noise;
under the 90° rotation it fails, because the regression's least-squares solution shrinks the adapter
(restricted to a constant matrix it learns 0.2 × the rotation; `research/adapter_diagnostics.py`).
With depths tuned (the sweeps' test trials), stopping the learned reversal early helps under the
gain with noise (61.3% at n = 3 for σ = 0.5, 54.3% at n = 4 for σ = 1.0) and then beats Kao et al.'s
flow control on the BCI chunk (47.0% at tau 16, 38.3% at tau 12; the lab's implementation, lambda tuned,
had given 66.3% and 64.3%); under the rotation the learned adapter never beats the policy alone. The full write-up with the math is `reports/mars_adapter_report.pdf`.

### The study condition

`run_sweep.py --sweep study_condition` searched the live setting (the expert pushing at the goal,
constant reference chunks, as the portal runs it) for where MARS leads the best of teleop,
flow control and FRS most, with every method's depth tuned on 200 trials and tested on 500 new ones
(`results/sim/sweeps/study_condition/`). Under gain diag(1, 0.25):

| Input noise | Teleop | Flow control (tau) | FRS (n) | MARS, F = M (n) | MARS (n) | Lead [95% CI] |
|---|---|---|---|---|---|---|
| 0.25 | 86.4% | 70.4% (12) | 86.0% (2) | 93.6% (5) | 93.8% (4) | +7.4 [+4.6, +10.4] |
| 0.5 | 77.6% | 76.8% (16) | 79.6% (4) | 93.2% (10) | 93.0% (6) | +13.4 [+10.2, +16.8] |
| 0.75 | 68.2% | 76.6% (16) | 61.0% (5) | 81.8% (10) | 88.6% (10) | +12.0 [+7.6, +16.4] |
| **1.0** | 55.6% | 71.2% (14) | 46.0% (5) | 68.0% (10) | **79.0% (8)** | **+7.8 [+2.6, +13.0]** |
| 1.25 | 48.0% | 61.0% (12) | 37.0% (8) | 53.2% (10) | 71.6% (8) | +10.6 [+5.6, +15.6] |
| 1.5 | 36.6% | 60.0% (10) | 32.2% (7) | 40.6% (10) | 62.6% (8) | +2.6 [−3.0, +8.0] |
| 1.75 | 30.0% | 55.6% (12) | 21.6% (1) | 31.8% (8) | 56.0% (8) | +0.4 [−5.6, +6.4] |

Flow control here is Kao et al.'s (`ic_gain` 2, tau in chunk steps of 16; re-run 2026-10-25); the other
columns did not move. The lab's implementation (action units, lambda tuned) gave 94.6% (lambda 6) at noise
0.5, 72.8% (lambda 2) at 1.0, 64.0 / 55.0 / 47.2% (lambda 1) at 1.25 / 1.5 / 1.75: Kao's method is weaker
at low noise and holds up better at high noise, so the learned adapter's lead is significant up to
noise 1.25 and not beyond. Until 2026-10-16 the study used 1.25 with the depths from that sweep's
`depths.yaml` (flow control's lambda 1, n 8, n 8); the same protocol through the portal with the expert
(`run_experiment.py --protocol study.yaml --participant sim`, 40 trials per arm) gave teleop 37.5%,
flow control 70.0%, FRS 27.5%, MARS 65.0%, consistent with the sweep (`results/old/sim/study_s1`
is that run). The task parameter search below then moved the study to 4 targets, reach 0.35–0.5, noise 1.0.

### The task parameter space

`run_sweep.py --sweep task_space` asked which task makes MARS's lead over the best of
teleop, flow control and FRS largest, with the channel fixed at the study's gain diag(1, 0.25)
(re-run on 2026-10-26 under Kao et al.'s flow control; the condition was chosen on the earlier run,
which differed in the flow-control column only):
input noise {1, 1.25, 1.5} × targets per trial {2, 4, 8, 12} (fixed within a cell) × target spacing
{0.10, 0.12, 0.16, 0.24} × reach from the start to the goal {0.2–0.35, 0.35–0.5, 0.5–0.75} × dwell
{0.3, 0.5 s}; 270 cells (12 targets 0.24 apart do not fit), 100 tuning and 200 test trials each,
depths {1, 2, 3, 4, 6, 8, 10} (`results/sim/sweeps/task_space/`, `ranking.csv`, `figs/axes.png`).
The lead of each cell averaged over the other axes, and the share of cells where MARS beats all
three (paired bootstrap CI above 0):

| Axis | Value | Mean lead | Beats all | MARS success | Best other method |
|---|---|---|---|---|---|
| targets | 2 / **4** / 8 / 12 | −1.2 / **+2.6** / −2.6 / −5.2 | 18% / **31%** / 14% / 6% | 87 / 70 / 53 / 45% | flow control / flow control / teleop / teleop |
| spacing | 0.10 / 0.12 / 0.16 / 0.24 | −1.2 / −0.8 / −2.0 / −1.4 | 14–22% | 63–70% | |
| reach | 0.2–0.35 / **0.35–0.5** / 0.5–0.75 | −4.0 / **+0.8** / −0.8 | 20% / 21% / 12% | 72 / 64 / 58% | |
| dwell | **0.3** / 0.5 s | **+2.2** / −4.9 | 26% / 10% | 71 / 59% | |
| noise | **1.0** / 1.25 / 1.5 | **+3.0** / −1.3 / −5.7 | 34% / 14% / 4% | 77 / 65 / 53% | |

The number of targets decides most, and not in the direction one might expect: with 2 targets
every method is near its ceiling and flow control is the one to beat; with 8 or 12 the policy-based
methods pick wrong targets while teleop, which does not care how many there are, holds its 50% and
wins at short reach (8 targets, reach 0.2–0.35: teleop 72%, MARS 61%). Four targets is the peak.
The reach matters because teleop barely feels the gain over a short distance (71% at 0.2–0.35
against 32% at 0.5–0.75), while a long reach costs MARS too; the middle is best. The spacing does
nothing; a longer dwell and more noise lower every method and the lead with them. Where MARS leads,
teleop and FRS lose to timeouts and flow control to wrong targets: the gap is about finishing a medium
reach under the gain, not about choosing among targets. At 4 targets, dwell 0.3, averaged over the
spacings:

| Reach | Noise 1.0 | Noise 1.25 | Noise 1.5 |
|---|---|---|---|
| 0.2–0.35 | +4.8 (MARS 90%) | +11.2 (87%) | +9.0 (76%) |
| **0.35–0.5** | **+11.6 (85%)** | +9.2 (78%) | +4.2 (68%) |
| 0.5–0.75 | +1.9 (79%) | +0.6 (68%) | −5.6 (55%) |

The cells nearest the study condition (spacing 0.12, dwell 0.3, noise 1.25) but with 4 targets fixed
give +17.5 [+10.0, +25.0] at reach 0.2–0.35 (MARS 89%, teleop 72%, flow control 72%, FRS 63%) and
+9.0 [+2.0, +16.5] at reach 0.35–0.5 (76 / 56 / 67 / 42%), against +10.6 for the study condition as
run (2–8 targets, any reach ≥ 0.2): the 8-target and short-reach trials dilute it. The single best
cell (+18.5 [+9.5, +27.5]: 12 targets, spacing 0.1, reach 0.5–0.75, dwell 0.3, noise 1.0) is
selection noise, since its dwell-0.5 twin gives +3.5 and 12 targets lose 5 points on average.

**Refine** (`--sweep task_space_refine`, `results/sim/sweeps/task_space_refine/`): 3–6 targets ×
reach × dwell × noise {1, 1.25}, spacing 0.12, with 200 tuning and 500 test trials, depths 1–10 and a
new seed. The reach effect holds: at 0.35–0.5 the mean lead is +5.4 and MARS beats all three
methods in 9 of 16 cells, against +2.8 (8/16) at 0.2–0.35 and +1.1 (4/16) at 0.5–0.75. The number
of targets is flat from 3 to 6 (mean lead +2.9, +2.9, +3.5, +3.1; the collapse is at 8 and above), so
within that range it only sets how easy the task is (MARS 83% at 3 targets, 66% at 6). Dwell 0.5
takes 4.6 points of lead and 11 points of success from every method; noise 1.25 takes 4 points of
lead and 11 of success. Cells repeated from the screen agree to within a few points, with single cells
wobbling by up to ±7, so the reach effect is the finding and the best single cell is not. Dwell 0.3,
reach 0.35–0.5, lead [95% CI] and MARS success:

| Targets | Noise 1.0 | Noise 1.25 |
|---|---|---|
| 3 | +3.8 [−0.0, +7.8] (88%) | +7.0 [+2.6, +11.4] (86%) |
| **4** | **+11.8 [+7.0, +16.6] (86%)** | +6.4 [+1.2, +11.4] (76%) |
| 5 | +8.4 [+3.4, +13.2] (79%) | +11.6 [+6.2, +17.0] (74%) |
| 6 | +11.4 [+6.4, +16.2] (76%) | +7.2 [+1.4, +13.2] (65%) |

**The study condition since 2026-10-16** (`condition_study.yaml`) is therefore 4 targets, reach
0.35–0.5, dwell 0.3, noise 1.0, gain diag(1, 0.25) as before, with the arm depths from that cell of
the refine's `depths.yaml`: MARS 85.6% (n 8), flow control 73.8% (tau 16; the study arm's
own setting, lambda 1 in action units, gave 73.0% on the earlier run), teleop 67.4%, FRS 55.8% (full
reversal); lead +11.8 [+7.0, +16.6] over flow control, +18.2 over teleop, +29.8 over FRS. Against the
previous condition (+10.6 [+5.6, +15.6], MARS 71.6%) the lead is at least as large, every method
succeeds more often, and the noise is back inside the adapter's training range (0–1). Four fixed
targets and a reach of 0.35–0.5 are inside the policy's training distribution (1–8 targets, reach
≥ 0.2), so nothing was retrained. The methods fail as before: flow control commits to wrong targets
(23%), teleop and FRS time out (28%, 39%), MARS rarely does either (11%, 3%). The study protocol
through the portal with the expert (`run_experiment.py --protocol study.yaml --participant sim`,
40 trials per arm; `results/sim/study_s1`) gives MARS 85.0%, flow control 80.0%, teleop
62.5%, FRS 57.5%, consistent with the sweep.

### Composed user × task adapter

The reversal adapter has two modules, trained apart and composed only at test time
(`ReversalAdapter(user, task)`, `cursor_sa/steering.py`). The **user module** U(x_t, t) undoes one
user's command corruption, in the backward pass of the reversal only. The **task module** (A·v + b
from obstacle tokens) steers the frozen open-field policy's chunks onto an obstacle task, in the
forward and the backward pass. The questions: does a user module calibrated on the open field carry
over to obstacle tasks, and does a task module trained with one user carry over to users it never met?

**Lessons from a first grid** (grid 1, Oct 22: three users × four disc and wall tasks at noise 0.75,
user modules from the policy's own chunks, task modules trained without a user; its configs, code
paths and results were removed on Oct 23, the code and configs are in git at commit 5c2167cd):

- The field regression is the user-module objective that works in closed loop. Reversing,
  regenerating and matching through both flows sent the cursor to wrong targets (U1 on the open
  field: 50% at full reversal against 92%).
- Learned user modules trust the policy's course: pushed toward another target than the one the
  cursor moves to, the field module followed the push 12% of the time, against 38% with no module.
  The study's `gain025y_free` adapter behaves the same way.
- A disc centered on the reach splits the demonstrations between the two sides, and the task module
  learns neither (19–52% hits, against 54–86% with the disc off-center).
- Task modules trained without a user carried over to the two gain users (+5 to +14 points in all
  eight cells), not to the rotation user. Their chunks collided 0.1% of the time from demonstration
  states but 12–46% in closed loop.

**The grid** (`configs/condition_grid.yaml`, `configs/sweeps/grid*.yaml`). The study geometry (4
targets, reach 0.35–0.5) at input noise 1.0 for every user; 5 s per trial on T1, as in the study, and
8 s on the obstacle tasks, whose detours take longer. The cursor runs at the study's half speed by
slower time (`gain_mode: time`), not by smaller steps as in the study: the policy then sees the motion
it was trained on, which following a route around an obstacle needs (the checks below); the open-field
study keeps its controller.

- Users: U1 gain diag(1, 0.25); U2 gain diag(0.25, 1); U3 rotation 60°; U4 a curl field of 20/s on the
  command (τ 0.1 s: at steady state turned 63° clockwise and slowed to 45%).
- Tasks: T1 no obstacle; T2 a disc of radius 0.08 whose center sits 15–30% of its radius off the
  reach; T3 a wall of small discs (r 0.015) across the reach with a 0.16 gap 0.14 to one side; T4 a ring
  of discs (radius 0.14) around the goal, its gap turned 15–30° from the far side; T5 the same ring
  around the start, its gap turned 15–30° from the side facing away from the goal. Every offset,
  gap and turn goes to a random side, so the side to pass is part of the scene: one way round is
  shorter, the expert takes it, and both sides occur. `test.json` keeps every trial with the side it
  passed on (`trials`) and summarizes each cell by side (`by_side`).
- Why the side matters: a user whose channel turns pushes passes easily on one side only. Teleop on
  the wall (gap on a random side, 500 trials): U3, whose rotation bends pushes left, hits 58.0% with
  the gap on the left and 0.8% on the right; U4, whose curl bends them right, 0.0% and 23.7%; U1 42.0%
  and 49.4%. A fixed side would favor one of them on every trial.
- T1 keeps the study's 5 s because it is the study's task, and on the open field the policy methods'
  advantage over teleop is speed: an earlier run of this grid gave every task 8 s, and teleop then
  rarely ran out of time (4% timeouts for U1).
- The simulated user is the expert, the same planner whose paths the demonstrations follow: it plans
  a route and pushes toward the farthest point of the route it can see, through the user's channel
  (the demonstrations were recorded without one). On a clean channel it hits 100% of 1000 trials on
  every task with no collision,
  while the frozen policy alone hits 28% on T1 and 0–7% on T2–T5 (`grid_expert_check`).

**Training** (user modules on T1 with the identity task module, then one task module per user with
that user's module frozen, then the unseen combinations):

1. **User modules**, on T1. Open-loop calibration by the expert: the label is the demonstration's next
   chunk, the command the expert's push through the user's channel at noise 1.0, held over the chunk as
   the controller holds it; the field regression pulls the reversal of that command onto the label's
   latent. 100 demonstrations (about 1.5 min of reaching), one noise draw per demonstration step, 500
   steps (`user_<user>_expert_n100_s0`, under a minute each; seed k's modules carry `_s<k>` in its place).
2. **Task modules**: T2 with U1's module frozen in the loop, T3 with U2's, T4 with U3's, T5 with U4's.
   Flow matching on the task's demonstrations (which follow the expert's path), plus 0.03 × the field
   regression through both modules on that user's commands around the obstacles. 300 demonstrations (5–6
   min, a quarter of them without the obstacle), 20 000 steps (`task_<task>_n300_s0_with_<user>`, 13 min
   each). Until Oct 27 both kinds of module had 20 000 demonstrations and 20 000 steps (that grid:
   `results/old/sim/sweeps/grid_20000demos/`); the data-efficiency note below set these budgets.
3. **Retrained policies**, for comparison: for each obstacle task a policy with an obstacle head (one
   token per disc), trained from scratch on 20 000 of that task's demonstrations (a quarter without the
   obstacle; `flow_goaldrop_<task>`, 15–60 min each), and four user modules on each, calibrated as in step 1
   (`user_<user>_expert_n100_s0_on_<task>`, 16 in all). On T1 there is nothing to retrain: there the arm is the
   frozen policy and the user module.
4. **Test** on all 20 cells: 8 seen (T1 for every user, and the 4 training pairs), 12 unseen.
   Methods: teleop, flow control and FRS (the baselines), the user module alone, the task module alone,
   U + T, and the retrained policy with its user module. Depths tuned on 200 trials, tested on 500 new
   ones; seed 800, so every method in a cell sees the same layouts.

**Checks before the grid.**

- *User modules, open loop* (held-out demonstration states, full reversal): the regenerated chunk's
  heading agrees with the expert's label at cos 0.97, against 0.19–0.36 with no module (the
  20 000-demonstration modules: 0.90–0.91; grid 1's field modules 0.51–0.72).
- *User modules, closed loop on T1* (the 20 000-demonstration modules; 300 trials at 8 s under the study's
  controller; an ad-hoc sweep against grid 1's modules): the open-loop gain does not carry over. The grid's
  module then against grid 1's field module: U1 88.7% vs
  91.0%, U2 90.7% vs 93.3%, U3 91.3% (best at depth 1) vs 97.3%; teleop 92.3%, 90.7%, 98.0%. The new
  modules go to wrong targets more often (8–9% against 5–7%): expert demonstrations always head for
  the goal, so the calibration never shows the off-course states closed loop produces. For the curl
  user the module wins clearly: 69.3% against flow control 58.0%, teleop 53.3%, FRS 40.3%.
- *Target switches in the calibration* (dropped): demonstrations that switch target mid-reach made the
  module follow a course-changing push less, not more (6% against grid 1's 13.5% at noise 0.5),
  because a minimum-jerk demonstration keeps its old course for a while after the switch.
- *Field-loss weight* (pilots on the wall and a ring, 5000 steps): with the real user modules,
  weights 0.01–0.1 kept the task module's own chunk collisions at flow matching's level, and 0.03
  gave the fewest collisions of the chunks regenerated from the user's commands on both tasks (the
  rule fixed before the runs); a weight of 1 multiplied the chunk collisions about tenfold.
- *Task modules, go/no-go* (`grid_task_check`: the goal-given policy's own chunks executed, no user;
  gate ≥ 80% hits): T2 99.6%, T3 95.2% and T4 92.8% pass, from 0–19% without the module; T5's, from 300
  demonstrations, misses it at 74.2% (65% with the gap on the left, 84% on the right; the
  20 000-demonstration module had 85.2%). Under the study's controller the 20 000-demonstration ring
  modules missed it too (66.6% and 42.6%): the speed gain (next check), which is why the grid's controller
  differs from the study's.
- *The speed gain* (why the grid runs `gain_mode: time`). The study's controller (`step`, gain 0.5)
  halves every executed step but still moves one chunk step per control step, and divides the policy's
  history by the gain, so the cursor advances at half the rate its history says. A policy that must
  follow a curved route breaks: alone with the goal given, the 20 000-demonstration task modules hit 97%,
  99%, 67%, 43% on T2–T5 under it and the retrained policies 56%, 93%, 51%, 8%, against 99%, 99%, 94%, 85% and 90%, 99%,
  96%, 97% with each planned step played over two control steps (`time`: the same speed on screen, the
  demonstrations' history; `grid_task_check` before and after); executing 8 chunk steps instead of 2
  under `step` collapses both (2–18% on T2). On the open field the policy alone is unaffected (100%
  hits either way, goal given) and the study's steering arms move because their cadence does: with
  depths re-tuned, flow control −20, FRS −17 and MARS −10 points against teleop's 68.8% unchanged
  (1000 trials), about what re-reading the input half as often costs under `step` (execute 4: −15, −1,
  −14). The study keeps the controller its sessions were played on; the grid, a simulation, takes `time`.
- *Retrained policies, alone* (`grid_task_check`, goal given / hidden): T2 90.4% / 57.0%, T3 99.2% /
  97.8%, T4 96.4% / 95.4%, T5 96.6% / 36.8%, against the frozen policy with the task module's 99.6% /
  58.8%, 95.2% / 60.8%, 92.8% / 76.6% and 74.2% / 53.0%: a little ahead on the wall and the ring around
  the goal, well ahead on the ring around the start, behind on the disc. (With the goal hidden, the wall
  and the goal ring give the goal away, since obstacles sit on the reach; the retrained policies read that
  better than the task module does.) From held-out demonstration states their chunks collide 0.0–0.1% of
  the time, the task modules' 0.5–6.6%. Their user modules fit less well open loop (heading cos
  0.66–0.93, against 0.97). Grid 1's policy trained on all of its tasks (`flow_goaldrop_obstacles`, in git at
  5c2167cd) did worse under the study's controller: 25–38% on T2 and T3, 1% on the rings it never saw.
- *Fine-tuning instead of training from scratch* (`train_policy.py --init flow_goaldrop`: the study's
  policy with a new obstacle head, on the same demonstrations; checked under the study's controller
  only) did worse there: 6%, 18%, 4%, 2% alone with the goal given, 80–97% collisions, while still
  98.5% on T1. At the start of a trial it begins the
  detour on the route's side more often than the task module (65% of first chunks against 60%), then
  cuts it too tight: off the demonstrated route the study's policy pulls it straight to the goal.
  Replanning every 8 or 16 steps instead of 2 changes nothing, and training twice as long at three
  times the learning rate (validation loss 0.024 → 0.016) leaves the disc at 4%.

**Results** (`results/sim/sweeps/grid_s0/`, `figs/summary.png`): test success with collisions in
brackets, every method at its tuned depth, with user modules from 100 demonstrations and task modules from
300, training seed 0 (five seeds after the tables; the grid with 20 000-demonstration modules, before Oct 27:
`results/old/sim/sweeps/grid_20000demos/`). The starred task is the one that user's module was trained
with; T1 is seen by every user module. T1 has no task module and no retrained policy, so there the task
module row is FRS and the U + T and retrained rows are the user module (the same arms, so the same
trials).

| U1 | T1 | T2* | T3 | T4 | T5 |
|---|---|---|---|---|---|
| teleop | 67.6% | 20.2% (78) | 45.2% (48) | 3.0% (93) | 2.6% (95) |
| flow control | 50.6% | 27.4% (46) | 16.0% (60) | 3.0% (67) | 4.6% (78) |
| FRS | 40.6% | 12.2% (81) | 24.4% (58) | 2.6% (90) | 2.6% (91) |
| user module | 63.4% | 31.4% (53) | 32.4% (53) | 4.6% (73) | 8.6% (81) |
| task module | 40.6% | 18.4% (60) | 25.6% (55) | 2.4% (89) | 5.2% (86) |
| **U + T** | **63.4%** | **84.2% (1)** | **70.2% (19)** | **83.8% (4)** | **61.8% (10)** |
| retrained policy + U | 63.4% | 17.0% (79) | 67.6% (29) | 66.0% (33) | 36.6% (31) |

| U2 | T1 | T2 | T3* | T4 | T5 |
|---|---|---|---|---|---|
| teleop | 68.6% | 19.8% (77) | 48.6% (44) | 2.8% (92) | 3.2% (95) |
| flow control | 53.6% | 22.2% (53) | 19.6% (56) | 4.0% (73) | 6.4% (80) |
| FRS | 42.8% | 11.4% (82) | 28.4% (54) | 1.4% (90) | 4.0% (89) |
| user module | 62.6% | 22.6% (67) | 33.2% (49) | 3.0% (84) | 9.4% (78) |
| task module | 42.8% | 11.0% (79) | 29.0% (51) | 2.6% (92) | 5.2% (85) |
| **U + T** | **62.6%** | **82.8% (0)** | **77.2% (8)** | **88.4% (4)** | **63.2% (7)** |
| retrained policy + U | 62.6% | 13.2% (85) | 66.2% (29) | 61.4% (38) | 32.4% (32) |

| U3 | T1 | T2 | T3 | T4* | T5 |
|---|---|---|---|---|---|
| teleop | 90.4% | 42.8% (53) | 27.6% (68) | 12.8% (74) | 34.6% (54) |
| flow control | 39.0% | 19.8% (54) | 14.0% (66) | 3.2% (48) | 5.2% (83) |
| FRS | 45.0% | 26.8% (60) | 21.0% (70) | 7.2% (85) | 18.2% (64) |
| user module | 60.8% | 29.0% (45) | 19.2% (62) | 8.0% (76) | 14.0% (68) |
| task module | 45.0% | 31.2% (54) | 25.2% (64) | 8.0% (66) | 16.8% (66) |
| **U + T** | **60.8%** | **79.6% (2)** | **68.8% (11)** | **86.6% (5)** | **35.4% (30)** |
| retrained policy + U | 60.8% | 18.6% (74) | 64.6% (31) | 68.2% (31) | 35.6% (40) |

| U4 | T1 | T2 | T3 | T4 | T5* |
|---|---|---|---|---|---|
| teleop | 17.0% | 22.2% (50) | 11.0% (71) | 2.4% (62) | 2.0% (60) |
| flow control | 30.0% | 17.2% (42) | 10.8% (55) | 2.2% (61) | 0.2% (90) |
| FRS | 8.0% | 8.8% (54) | 6.4% (76) | 1.0% (65) | 1.2% (83) |
| user module | 38.0% | 19.4% (50) | 12.0% (62) | 4.2% (70) | 2.6% (87) |
| task module | 8.0% | 10.4% (52) | 9.8% (61) | 2.0% (74) | 0.8% (82) |
| **U + T** | **38.0%** | **64.8% (0)** | **65.0% (12)** | **75.4% (4)** | **61.8% (3)** |
| retrained policy + U | 38.0% | 17.8% (78) | 63.6% (33) | 67.8% (31) | 22.0% (39) |

Paired effects on T2, T3, T4, T5 (points; * where the bootstrap 95% CI excludes 0; (s) the training pair):

| Effect | U1 | U2 | U3 | U4 |
|---|---|---|---|---|
| U + T over the best of teleop, flow control and FRS | +57\*, +25\*, +81\*, +57\* | +61\*, +29\*, +84\*, +57\* | +37\*, +41\*, +74\*, +1 | +43\*, +54\*, +73\*, +60\* |
| adding the task module to the user module | +53\* (s), +38\*, +79\*, +53\* | +60\*, +44\* (s), +85\*, +54\* | +51\*, +50\*, +79\* (s), +21\* | +45\*, +53\*, +71\*, +59\* (s) |
| adding the user module to the task module | +66\* (s), +45\*, +81\*, +57\* | +72\*, +48\* (s), +86\*, +58\* | +48\*, +44\*, +79\* (s), +19\* | +54\*, +55\*, +73\*, +61\* (s) |
| U + T over the retrained policy + U | +67\*, +3, +18\*, +25\* | +70\*, +11\*, +27\*, +31\* | +61\*, +4, +18\*, 0 | +47\*, +1, +8\*, +40\* |

Success by the side passed, for the users whose channel turns pushes (teleop / user module / task
module / U + T):

| | T2 left | T2 right | T3 left | T3 right | T4 left | T4 right | T5 left | T5 right |
|---|---|---|---|---|---|---|---|---|
| U3, bends left | 89 / 34 / 60 / 80 | 0 / 24 / 5 / 79 | 54 / 31 / 49 / 74 | 0 / 7 / 1 / 63 | 24 / 15 / 15 / 90 | 0 / 0 / 0 / 83 | 72 / 29 / 35 / 31 | 0 / 0 / 0 / 39 |
| U4, bends right | 0 / 13 / 3 / 69 | 42 / 25 / 18 / 61 | 0 / 9 / 3 / 64 | 22 / 16 / 17 / 66 | 0 / 0 / 0 / 77 | 5 / 9 / 4 / 73 | 0 / 0 / 0 / 62 | 4 / 5 / 2 / 62 |

For U1 and U2 the two sides differ by at most 8 points (U + T on U2's T5), mostly far less
(`grid_summary.md`). On T1 (5 s) the user module against the best baseline: U1 −4, U2 −6\*, U3 −30\*,
U4 +8\*; against teleop −4, −6\*, −30\*, +21\*; against plain FRS +23\*, +20\*, +16\*, +30\*.

**Five training seeds** (Oct 28). The tables above and the bullets below are seed 0; Fig 3 shows all five
seeds (`figures/stats.json` has its numbers: means over the seeds; panel d's error bars are 95% t CIs over
the four users, the user the unit as the participant in Fig 2b; panel c gives the best method per cell with
its lead over the second, a split cell where that lead's 95% CI, resampling seeds, then trials, includes 0;
its rollouts take 8 trials from each seed's modules). Seeds 1–4
retrain every module on fresh draws of its 100 or 300 demonstrations (the user modules, the retrained arm's
user modules and the task modules; the retrained policies themselves stay one seed) and rerun the module
methods on the same trials. MARS success, mean ± SD over the five seeds, with the retrained arm's in brackets:

| 5 seeds | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|
| U1 | 65.8 ± 1.8 | 79.6 ± 4.3 (17.2 ± 0.9) | 70.8 ± 1.5 (67.5 ± 1.5) | 87.1 ± 2.3 (66.3 ± 0.6) | 59.7 ± 1.8 (34.1 ± 2.1) |
| U2 | 64.8 ± 1.6 | 81.6 ± 3.3 (14.1 ± 0.9) | 75.3 ± 1.9 (66.9 ± 0.9) | 86.8 ± 3.6 (61.4 ± 1.6) | 60.2 ± 4.1 (30.1 ± 1.6) |
| U3 | 61.7 ± 3.1 | 71.4 ± 7.6 (18.2 ± 1.3) | 66.2 ± 2.2 (65.3 ± 1.1) | 85.5 ± 2.7 (68.5 ± 0.7) | 35.1 ± 4.9 (33.4 ± 2.5) |
| U4 | 39.7 ± 1.4 | 58.4 ± 5.5 (17.3 ± 1.0) | 64.3 ± 2.3 (62.7 ± 1.0) | 74.8 ± 1.4 (68.2 ± 0.8) | 60.9 ± 1.6 (19.5 ± 1.5) |

Over the 16 obstacle cells MARS averages 69.9% ± 1.7 across seeds (seed 0 71.8%), the retrained arm 44.4% ±
0.5, the user module alone 15.5% and the task module alone 12.1%. Per cell MARS's seed SD is 1–8 points,
largest on the disc (U3: 62.6–79.6%), where seed 0 sits 5–8 points above the mean for U1, U3 and U4, so the
seed-0 disc numbers are optimistic. Pooling the seeds (95% CIs resampling seeds, then trials), MARS beats the
best baseline in 15 of 16 obstacle cells (U3 on the start ring is level with teleop, +0.5 [−3.8, +5.5]) and
the retrained arm in 13 (level on the wall for U3 and U4, +1.0 and +1.6, and on U3's start ring, +1.7). On T1
the user module is level with teleop for U1 (−1.8 [−4.7, +1.1]), behind for U2 (−3.8 [−6.6, −1.1]) and U3
(−28.7), ahead for U4 (+22.7). Alone with the goal given, the ring-around-start task module reaches 72–82%
over the seeds (2 of 5 pass the 80% gate), the other three 91–100% in every seed. Per-cell statistics:
`results/research/grid_seeds/seed_stats.json`; runs `results/sim/sweeps/grid_s<k>/` (seed 0 the grid of record,
seeds 1–4 their parts and `combined/`); checkpoints `checkpoints/adapters/*_n100_s<k>*` and `*_n300_s<k>*`, all five
seeds side by side.

- **Task modules carry over to users they never met.** In the 12 unseen cells the task module adds
  21–85 points to the user module alone, about as much as for the user it was trained with: T2 +60, +51,
  +45 (+53 for its own), T3 +38, +50, +53 (+44), T4 +79, +85, +71 (+79), T5 +53, +54, +21 (+59), all
  beyond noise. The smallest is the ring around the start with U3 (+21): that module, trained with U4's
  curl, carries over to the rotation user only in part.
- **U + T beats teleop, flow control and FRS in 15 of the 16 obstacle cells**, by 25–84 points: 65–84% on
  the disc, 65–77% on the wall, 75–88% on the ring around the goal and 62–63% on the ring around the start,
  with 0–19% collisions where the baselines and each module alone collide on 42–95% of trials. In the
  16th, U3 on the ring around the start, it is level with teleop (35.4% against 34.6%, +1 [−5, +7]) and
  collides on 30% of trials; with the 20 000-demonstration modules it had 60.4%.
- **Neither module does it alone.** On the obstacle tasks the user module alone is 21 points below to 4
  above the best baseline (above it beyond noise only on the ring around the start for U1 and U2, by 3–4
  points) and the task module alone 20 below to 1 above, never above it beyond noise; they beat plain FRS
  beyond noise in 8 and 5 of the 16 cells, by at most 19 and 6 points, and both leave collisions at the
  baselines' level (45–92%). Adding the user module to the task module gains 19–86 points, about what
  adding the task module to the user module gains (21–85).
- **A policy retrained for the task trails U + T, most on the disc and the ring around the start.**
  Retrained on 20 000 of the task's demonstrations (the task module has 300) and steered through its own
  user module, it hits 13–19% on the disc, 64–68% on the wall, 61–68% on the ring around the goal and
  22–37% on the ring around the start, with 29–85% collisions. U + T leads it by 47–70 points on the disc,
  8–27 on the goal ring and 25–40 on the start ring except for U3 (0), beyond noise in 12 of the 16 cells;
  on the wall the two are level for U1, U3 and U4 (+1 to +4) and U + T leads by 11 for U2. With the
  20 000-demonstration user modules of the previous grid the same policies hit only 8–41%: those modules,
  trained 20 000 steps, fitted the retrained fields poorly (heading cos 0.64–0.86, now 0.66–0.93, against
  0.97 on the frozen policy), so part of what looked like the retrained policies failing was their user
  modules. On the tuning trials the retrained arm's lowest collision rate over the depths is 24–78% per
  cell and U + T's 0–12% (34% for U3 on the start ring); on U1's goal ring U + T's collisions fall from 92%
  at depth 1 to 2–4% at depths 6–8, the retrained arm's never below 32%. Why it still collides
  (`research/steering_states.py`, U1 on the goal ring): open loop from demonstration states the retrained
  field is as good as the frozen policy with the task module, even on corrupted or misdirected pushes; the
  difference is the states each arm drives itself into. From the states the U + T arm visits, the frozen
  policy with the task module collides in 0.1% of the executed steps per replan and the retrained field in
  0.3–0.4%; the retrained arm drifts closer to the discs (4.9% of its live states within 0.02 of a disc,
  against 1.6% for U + T and none in the demonstrations), where its hazard is 1.0–1.1% per replan.
  Compounded over a trial's replans, U + T ends 3% of its trials in a collision and the retrained arm 24%.
  Both learned from the expert's routes, the task module from far fewer of them; it holds up off the routes
  because it learns only a correction on top of the frozen policy, which reaches a target from anywhere,
  while the retrained policy had to learn reaching and avoiding together from routes only.
- **With a channel that turns pushes, the two modules together rescue the side the channel cannot
  take.** U3's rotation bends pushes left: on the right side teleop, U3's own module and the task module
  alone hit 0–24%, U + T 63–83% on T2–T4 and 39% on T5. U4's curl bends them right: on the left they hit
  0–13%, U + T 62–77%. On the side the channel already favors, U + T trails teleop on U3's disc (80%
  against 89%), leads on the wall and the goal ring (74% against 54%, 90% against 24%) and trails on U3's
  start ring (31% against 72%), where the task module overrides a detour the channel was making on its own.
- **The task modules bound U + T.** Alone with the goal given three of the four pass their gate (99.6%,
  95.2%, 92.8%) and the ring around the start's reaches 74.2%; the ring around the goal gives the most
  (75–88%), the ring around the start the least (35–63%).
- **On the open field, at the study's 5 s,** the user module is behind teleop by 4 points for U1 and 6
  for U2, behind for U3 (−30) and ahead for U4 (+21): it times out less (5–12% against 7–71%) and goes to
  wrong targets more (25–57% against 2–12%). It beats plain FRS by 16–30 points and flow control by 8–22.
  The 20 000-demonstration modules were 1–4 points better (−3, −3, −26, +24 against teleop).
- Caveats: simulated users (a lag-free expert); five training seeds per module (above) but one per retrained
  policy; the previous grid's 20 000-demonstration user modules were no
  better than grid 1's in closed loop on T1 (checks above, under the study's controller).
- **Recalibrating the user modules closed loop (ReFIT-style, Oct 26) repairs part of the open-field gap
  and no more.** The trainer's `--closed-loop` runs the open-loop module on T1 at noise 1.0, keeps every
  visited state with the expert's push there, relabels it with a straight reach to the goal and trains a
  fresh module on mixed batches. On T1 at the grid's seed (500 trials; the open-loop module and teleop
  reproduce the grid) the recalibrated module gets U1 68.6 (from 64.4; teleop 67.6), U2 65.2 (65.6; 68.6),
  U3 77.4 (60.8; 90.4) and U4 45.4 (41.2; 17.0); at 2000 trials U1 is level with teleop (69.0 against 69.7,
  −0.7 [−3.3, +2.1]) and U3 stays behind (76.3 against 90.2, −13.9 [−16.2, −11.7]) with 22% wrong targets
  (from 36%). More closed-loop trials (10 000), a second DAgger round and closed-loop states alone change
  nothing or hurt (states alone: 40%). The limit is the signal: on the states the module visits, one served
  command at noise 1.0 has cosine 0.24 with the goal direction, 0.19 (U1) and 0.46 (U3) even with the
  corruption undone exactly, and the module's regenerated heading agrees with the relabel at 0.50 on
  held-out closed-loop states (0.85 on demonstration states); teleop wins the open field by integrating that
  noise over every 20 ms step, where the controller re-reads one sample every two. The recalibrated U does
  not compose with the task modules trained on the open-loop U: U3 × T4 falls from 89.2 to 65.6 (collisions
  6.6% → 25.4%), U1 × T2 from 81.0 to 79.4, so adopting it means retraining the four task modules and the
  grid. The grid keeps the open-loop modules; the four recalibrated checkpoints are `user_*_refit`.
- **How few demonstrations the modules need (Oct 27).** Until this check every grid module, like the policy,
  was trained on 20 000 demonstrations (one reach each, 5–7.5 h in all); the grid now runs the 100- and
  300-demonstration modules it led to (results above). Retrained on 10–1000 (one noise draw per
  demonstration step, `--frozen-noise`) and run on the grid's trials (seed 800; the 20 000-demonstration
  modules and teleop reproduce that grid trial for trial), the two modules part ways. A **user module needs tens**, trained about
  five steps per demonstration: T1 52.8 / 55.1 / 56.7 / 57.0 / 60.4% at 10 / 30 / 100 / 300 / 1000
  demonstrations (three draws of the demonstrations up to 100) against 58.8% with 20 000 (−6.1 [−8.2, −3.9],
  −3.8, −2.1 [−4.3, +0.1], −1.8, +1.5 points; FRS without a module 34.1), and a 10-demonstration user module
  composes with the 20 000-demonstration task module at full strength (U1 × T2 84.8 against 81.0); trained the default
  20 000 steps, 10 demonstrations over-fit (U1 × T1 44.0). A **task module needs about 300, and the full
  20 000 steps**: over the four trained cells it gets 76.9% at 300 (+1.3 [−0.9, +3.5] against 75.6) and 59.8%
  at 100 (−15.8 [−18.4, −13.4]); at 10 and 30 (T2, T4, T5) 24.3 and 31.6%, still 13 and 20 points over teleop.
  Trained five steps per demonstration it needs about 1000 (64.3%), and the 20 000-demonstration task module trained
  5000 steps loses 8 points on T2 (73.0). On small data the task module's held-out flow-matching loss rises
  after a few hundred steps while closed-loop success keeps climbing, so its training length cannot be chosen
  open loop; the user module's held-out heading cosine falls as it over-fits, and T1 with it. One draw of
  the demonstrations above 100 and for every 20 000-step task module; not run: T3 at 10 and 30 with 20 000
  steps, and user modules trained 20 000 steps beyond U1. Checkpoints `data_efficiency/*`, results
  `results/sim/sweeps/data_efficiency/`, configs, scripts and logs `results/research/data_efficiency/`.
- **The user module's ranks** (Oct 29; `train_adapter.py --step-rank r --cross-step-rank q`). Step rank r writes
  every step's matrix as I + Σ_k c_{s,k} B_k over r shared 2×2 bases (the default, unset, frees every step's
  matrix, which spans what r = 4 does); cross-step rank q adds a low-rank coupling across the chunk,
  ṽ_s = A_s v_s + U_s Σ_s′ V_s′ᵀ v_s′, with U zero-initialised so the module starts at the identity. On the grid's
  seed-0 trials and task modules, MARS against the grid's own (step rank 4, cross-step rank 0, 100 demonstrations;
  T1 56.2, obstacle cells 71.8, U4's obstacle cells 66.8%): cross-step rank 2 / 8 / 32 gains +4.2 / +3.5 / +3.4
  points on T1, +2.7 / +2.9 / +2.4 on the obstacle cells and +5.0 / +5.7 / +4.7 on U4's (every CI above 0;
  U3 × T5 35 → 60%), though a few T2 and T4 cells lose 1–7 points; step rank 1 or 2 at 100 demonstrations is within
  a point on the obstacle cells, and at 10 demonstrations every step rank loses (obstacle cells −0.5 to −1.2,
  U4's −3.6 to −4.9, T1 −1.4 at step rank 4 and −5.5 to −5.8 at 1 and 2). One training seed; not yet run over
  the grid's five. `research/adapter_ranks_sched.py`, `research/adapter_ranks_analysis.py`; checkpoints
  `user_<user>_expert_<variant>_s0` (e.g. `user_curl20_expert_cr2_n100_s0`; variants `sr1_n10`, `sr2_n10`, `sr4_n10`,
  `sr1_n100`, `sr2_n100`, `cr2_n100`, `cr8_n100`, `cr32_n100`; local only, not on the Hub), results
  `results/sim/sweeps/adapter_ranks/`, logs and stats `results/research/adapter_ranks/`. `--cross-step-dims 0`
  (a comma list of indices; default both dims) restricts the coupling to those dims; not run here. The knobs are the
  core's `LearnedUserModule` options (`step_rank`, `cross_step_rank`, `cross_step_dims`), saved in the module's config
  only when set, so every checkpoint without them loads as before.

## Next steps

1. **Run participants** with `study.yaml`; pilot data from Oct 14 (old log format, previous depths)
   is under `results/p0/pilot/`.
2. **Changes of course** are outside the deployed task (one goal per trial), and the user modules
   are calibrated without them. Grid 1's modules trusted the policy's course; calibrating on
   demonstrations that switch target mid-reach did not fix that (Oct 23 check: the module then followed
   a course-changing push less, because a demonstration's next chunk turns gradually after a switch).
   Also classifier-free guidance between the goal-given and goal-hidden outputs.
3. **Port the design checks** (`research/design_checks.py`) onto the sweep mechanism; they are the
   last script with their own rollout loop.
4. **The controller's gain mode.** Under the study's `step` the ring task modules miss their gate and
   the retrained policies fail (checks), so the grid runs `time`, which a human study on the obstacle
   tasks should use from the start. Should `step` be needed on obstacle tasks, the fix is in training:
   demonstrations replayed as `step` executes them, or training on closed-loop rollouts (DAgger-style).
5. **User modules recalibrated closed loop** (ReFIT-style): tried on Oct 26 (`--closed-loop`, results
   above): it recovers a third to two thirds of the open-field gap (U3 61 → 77%, teleop 90%) and does not
   compose with the task modules trained on the open-loop modules. To adopt it, retrain the task modules
   with the recalibrated user modules in the loop and re-run the grid; to close the rest of the gap the
   controller would have to integrate the served command over time as teleop does, rather than read one
   sample per replan.
6. **A person on the obstacle tasks.**
7. **Why steering needs both modules.** Driven by the policy alone the task module avoids the obstacles
   (`grid_task_check`), but steering a corrupted command without the user module leaves the collisions
   at FRS's level (45–92%). The retrained policies' remaining gap is measured (covariate shift that
   compounds over a trial; `research/steering_states.py`, one cell so far); calibrated with 100-demonstration
   user modules they reach 61–68% on the wall and the goal ring, so if a retrained policy is wanted as a
   contender, training it on its own closed-loop rollouts is the fix to try, and a fair comparison would
   give it the task module's 300 demonstrations rather than 20 000.
8. **Task modules from tens of demonstrations.** Trained the full 20 000 steps a task module matches a
   20 000-demonstration one at 300 demonstrations, which the grid now uses, while 10–30 recover a fifth to a
   third of its gain over teleop (research notes, data efficiency). For robot demonstrations, where tens are
   what one collects, the candidates are augmenting the demonstrations by the task's symmetries (a mirrored
   layout swaps the side to pass), one task module pretrained across the obstacle tasks and fine-tuned on the
   new one, and DAgger-style rounds that label the states the composed adapter visits.
