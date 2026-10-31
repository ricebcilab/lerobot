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
agent. The study's report is built locally by `reports/build_study_report.py` (the PDFs are not committed); the
paper's figures by `../figures/build_figures.py`.

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
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --arm study_mars.yaml                            # free play with one arm
uv run --extra mars python examples/pi05/mars/cursor/analyze_results.py                                               # tables and figures for every run
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1          # a participant's study report (PDF)
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1 p2 p3 p4 # the group's
uv run --extra mars python examples/pi05/mars/figures/build_figures.py --device cpu                          # the paper's figures (examples/pi05/mars/figures/*.pdf, *.png, stats.json)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep study_tuning                                # tune the study arms' settings, test (~15 min)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep grid --cells U1xT1 --out grid_part_U1xT1   # a part of the grid (see Sweeps)
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
reports/              build_study_report.py --participants p<N> ... writes study_p<N>.pdf for one participant or
                      study_group.pdf for several (paper-style figures)
../figures/           (examples/pi05/mars/figures/, beside this folder) the paper's figures, PDF + PNG, and build_figures.py (one script; `--only fig1 fig2 fig3`):
                      fig1_method (MARS illustrated), fig2_study (the human study), fig3_grid (the user x task
                      grid over its five training seeds); stats.json holds the numbers the captions quote. MARS
                      (Modular Adapters for Reversal Steering) is mars's display name: MARS = user + task modules,
                      MARS-U = the user module alone, MARS-T = the task module alone; the code's tokens are unchanged
configs/              arms (one control condition each), condition_*.yaml, study.yaml (the protocol),
                      sweeps/*.yaml (sweep definitions)
../checkpoints/       IGNORED  (examples/pi05/mars/checkpoints/) policies/<name>/, adapters/<name>/ (weights +
                      training logs; none tracked; hub.py download fetches the study's and the grid's)
../results/           IGNORED  (examples/pi05/mars/results/) p<N>/ and sim/ (runs), sim/sweeps/<name>/
                      (run_sweep.py), analysis/, old/ (superseded results, by date)
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
  goal-given policy's own chunk, step by step (used by `grid_task_check`). All go through the
  same input channel (corruption, noise), the same controller and the same records.
- **Arm.** One control condition, a config under `configs/` (policy, mode and its settings, channel, task).
  The four study arms extend `condition_study.yaml`: gain diag(1, 0.25) on the command plus input
  noise 1.0, 4 targets per trial at a reach of 0.35–0.5, each method at the settings `study_tuning`
  tuned for that condition.
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
each method's setting is tuned on the tuning trials, then every method is tested at its tuned
setting on new trials, with a paired bootstrap 95% CI on the focus method's lead. A sweep is a file
under `configs/sweeps/` naming the agent, the base arm, the methods (arm overrides), the settings to
tune per mode (`tune`: flow control's `tau`, `lambda` and `ic_gain`, FRS's `n_reverse_steps` and
`pinned_steps`, each with a list of values; every combination is a setting, recorded as
`{knob: value}`), the grid (`family: gain | rotation | curl | none`, `levels`, `noises`, and
optionally `axes`: arm parameters such as `targets`, `min_target_sep`, `reach` or `dwell_s`, each
with a list of values, every combination a cell), the trial counts and the focus. Cells whose
layouts cannot be sampled (too many targets for the spacing) are skipped and listed in `test.json`.
Outputs go to `results/sim/sweeps/<name>/`: `tuning.json`, `test.json`, `settings.yaml` (the tuned
settings, to copy into arm configs), `ranking.csv` (every cell, best lead first by the lower end of
its CI), `log.txt` (everything the sweep printed) and `figs/` (tuning curves, the test map or lines,
or `axes.png` with the lead against every varied parameter, the showcase cell or, for a users × tasks
grid, the grid's maps and summary, and center-out `traces.png` when the sweep asks for them).
`--fixed-settings <settings.yaml>` skips tuning and tests at the given settings (the same test trials
a tuned run would use); `--module-seed k` swaps every `_n<demos>_s0` checkpoint for training seed k's.
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
| `study_tuning` | expert | the study's condition: flow control over tau (lambda 1), FRS over pinned_steps (the full reversal), MARS over n_reverse_steps × pinned_steps; writes the study arms' settings | ~15 min |
| `grid_expert_check` | expert | the expert alone (teleop, clean channel) and the frozen policy alone on the grid's tasks T1–T5 | 10 min |
| `grid_task_check` | expert_chunk | the goal-given policy's own chunks executed, with and without each task module, and each task's retrained policy alone (T2–T5) | 2 min on a free GPU |
| `grid` | expert | users {U1 … U4} × tasks {T1 … T5} at noise 1.0: the composed user × task adapter, each module alone, each task's retrained policy with its user modules, teleop, flow control and FRS; per cell flow control tunes tau (lambda 1), FRS pinned_steps (the full reversal), every MARS variant n_reverse_steps × pinned_steps | seed 0 tuned in parallel parts (`--cells`, then `--merge <parts> --out grid_s0`); seeds 1–4 at seed 0's settings (`--module-seed k --fixed-settings .../grid_s0/settings.yaml --out grid_s<k>`) |

Times are on one RTX 6000 Ada. Settings only need re-tuning when something that can move them changes:
the condition, a retrained module or a retrained policy.

## Conventions

Names are unified across scripts, configs, log fields, folders and figures:

- Methods: `teleop`, `policy`, `flow_control`, `frs`; `frs` with a `reversal_adapter` is `mars` (Modular
  Adapters for Reversal Steering). MARS with learned modules is the default and carries no qualifier:
  `study_mars.yaml`, sweep methods `mars` (user + task modules), `mars_user` (MARS-U), `mars_task`
  (MARS-T), `retrained_mars_user`. MARS with the closed-form adapter F = M is `mars_fixed`. Runs
  recorded before 2026-10-26 name these `frs_ra` and `frs_ra_learned`; `records.ARM_ALIASES` maps the
  recorded sessions' arm names when they are loaded.
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
- Tuned settings: flow control's `tau` (chunk steps, of 16) and `lambda` (Euler steps the command is
  written at), FRS's and MARS's `n_reverse_steps` (Euler steps reversed, of 10) and `pinned_steps`
  (chunk steps the reference steers; Tang et al.'s `action_repeats`), as the sweeps tune them (`tune`).
  `ic_gain` is fixed by the arm.
- Agents: `human`, `expert`, `expert_chunk`. Participants: `p<N>`, `sim`.
- Policy goal: `goal_given` / `goal_hidden` in outputs; "goal dropout" is the training technique.
- Corruptions in names give the axis and the value: `gain025y` is a gain of 0.25 on the vertical axis
  (diag(1, 0.25)), `gain025x` on the horizontal one, `gain05y` 0.5 vertical, `rot60` and `rot90` a
  rotation in degrees, `curl20` a curl of 20/s. These replaced `gain025`, `gain` and `rot` on Oct 24;
  participant and study-session records from before keep the names they were recorded with (the
  study's adapter `learned_adapter/gain025y_free` is the file their manifests hash as `gain025_free`).
- Channel: `corruption` (a matrix spec) and `input_noise` (a std); condition files are
  `condition_<what>.yaml`; sweep cells are keyed `<level>|<noise>`, plus `|<axis>=<value or name>` per
  task axis (an axis given as a mapping `name: value` is keyed by its names).
- Scripts are `verb_object.py` (`run_experiment.py`, `build_study_report.py`); library modules are
  nouns. Record fields: `t_start`, `t_end`, `duration_s`, `outcome`.

## Configs

`configs/base.yaml` holds every arm parameter with a comment. Each arm says `extends: base.yaml` (or
another arm) and overrides only what it changes; a bare name is looked up in `configs/`; unknown or
missing keys are errors. In free play any arm parameter can be overridden on the command line
(`--speed-gain 0.3`, `--tau 8`, `--corruption null`).

| Arm | Method | Settings | Condition |
|---|---|---|---|
| `teleop.yaml`, `flow_control.yaml`, `frs.yaml` | teleop, flow_control (tau 8 of 16, Kao et al.'s default), frs | | none |
| `study_teleop.yaml`, `study_flow_control.yaml`, `study_frs.yaml`, `study_mars.yaml` | the study arms (flow control with the command in action units; MARS with `user_gain025y_expert_n100_s0`) | tuned by `study_tuning` (each file says its settings) | `condition_study.yaml`: gain diag(1, 0.25), noise 1.0, 4 targets, reach 0.35–0.5 |
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
  (the grid's checks).
- `tau` and `ic_gain` are flow control's knobs: `tau` chunk steps (of 16) take the command, at `ic_gain`
  sigma for a full push, as their initial condition (Kao et al.: 8 steps; their gain put the median
  command near 0.5 sigma). `ic_gain` is 2.0, chosen once on the study condition like Kao et al.'s gain
  (a sweep of 0.5 / 1 / 2 / 3 / 5 / 8 gave 39 / 44 / 73 / 63 / 62 / 12% at the tuned tau; at 1 the policy
  mostly ignores the command and picks a wrong target, at 5 and above the chunk is pure command and
  stalls; results under `results/old/2026-10-30/`). `lambda` is the Euler steps at whose start the
  command is written: 1 is Kao et al.'s initial condition, 0 the policy alone, more the lab's
  implementation, which re-writes x_t as the flow proceeds. The study's arm writes the command in action
  units (`ic_gain` 2.0843, 2.1 sigma for a full push). The sweeps tune `tau` and keep `lambda` 1, Kao et al.'s
  method (their Appendix D.2 perturbs later Euler steps only as an extension).
- `n_reverse_steps` and `pinned_steps` are FRS's knobs (MARS's too): how many of the flow's 10 Euler
  steps the reference is reversed (null: all, to noise), and how many leading chunk steps it steers
  (null: the whole chunk); the rest of the latent is handed back to the policy (Tang et al., App. D).
  FRS keeps the full reversal, Tang et al.'s method, and tunes `pinned_steps` (their `action_repeats`, fixed at
  5 of 15 there); MARS tunes both, since how far the reversal goes through the user module's field is part of the
  method (the 100-demonstration module fails at the full reversal).
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
  the target) against 100% for teleop, while every 2 steps gave 97% (a design check; results under
  `results/old/2026-10-30/research/design_checks/`). It is fixed, not tuned, and the same for every method.
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
user's commands. The checkpoint carries its `kind`; the first adapters (`learned_adapter/*`, no
kind) are user modules. The trainer as it was at git commit 5c2167cd also trained user modules from
the policy's own chunks (`learned_adapter/*`) and task modules without a user.

`--n-demos` sets how many demonstrations a module trains on (default 20 000, as many as the policy's own;
one demonstration is one reach, about a second). With few, the two modules want opposite budgets (the
data-efficiency notes, in git history): a user module over-fits the default 20 000 steps and
should train about five steps per demonstration (`--steps`, at least 100), while a task module needs the
full 20 000 steps whatever its data. `--frozen-noise` draws the channel's noise once per demonstration
step, as a recorded calibration has it, instead of anew in every batch.

Checkpoints go to `examples/pi05/mars/checkpoints/policies/<name>/` and `.../adapters/<name>/`, which git
ignores. The ones the study and the grid use (the study's policy and MARS adapter, the grid's policies and
modules of all five training seeds, each named with `_s<k>` after its demonstration count) are on the Hub
(`ricebcilab/mars-cursor`, listed in `hub_manifest.yaml`); `hub.py download` fetches them. Earlier experiments'
checkpoints are grouped by experiment under `adapters/learned_adapter/`, `adapters/grid_20000demos/`,
`adapters/refit/` and `adapters/data_efficiency/`, and are not distributed.

The trainers still write the cursor project's formats (`policy.pt`, `adapter.pt`), and the loaders
(`FlowPolicy.load`, `ReversalAdapter.load`) read both those and lerobot's, so a fresh checkpoint works before it
is converted. After training, rerun `convert_checkpoints.py` to get the lerobot format: a policy's folder
becomes a `cursor_flow` policy (`config.json` + `model.safetensors`, loaded by
`CursorFlowPolicy.from_pretrained`), and an `adapter.pt` takes the `save_adapter_module` format (kind, config,
weights, meta). The originals stay beside them as `*.orig`, and running
it again changes nothing. When a policy folder holds both formats (a retrained `policy.pt` beside an older
converted `model.safetensors`), `policy.pt` wins, with a warning, until you convert again. `hub.py upload`
refuses unconverted folders.

## Research notes

The notes on the experiments up to 2026-10-30 (the policy's checks, the steering simulations, the learned
adapter, the search for the study's condition and task parameters, the composed user × task adapter's
development, data efficiency, closed-loop recalibration and the user module's ranks) are in git history
(`git show b02c1551:examples/pi05/mars/cursor/README.md`); their results are under `results/old/2026-10-30/`
and their checkpoints under `checkpoints/adapters/{learned_adapter,grid_20000demos,refit,data_efficiency}/`.
On 2026-10-30 the study's and the grid's settings were re-tuned with the baselines as their papers define them
(flow control's tau at lambda 1; FRS's pinned prefix at the full reversal) and MARS over its reversal depth ×
pinned prefix, and the study's MARS arm moved to the grid's 100-demonstration user module.

## Next steps

1. **Run participants** with `study.yaml` at the re-tuned settings.
2. **Changes of course** are outside the deployed task (one goal per trial), and the user modules
   are calibrated without them; calibrating on demonstrations that switch target mid-reach did not fix
   that (Oct 23). Also classifier-free guidance between the goal-given and goal-hidden outputs.
3. **The controller's gain mode.** Under the study's `step` the ring task modules miss their gate and
   the retrained policies fail (checks), so the grid runs `time`, which a human study on the obstacle
   tasks should use from the start. Should `step` be needed on obstacle tasks, the fix is in training:
   demonstrations replayed as `step` executes them, or training on closed-loop rollouts (DAgger-style).
4. **User modules recalibrated closed loop** (ReFIT-style, `--closed-loop`, tried Oct 26): it recovers part
   of the open-field gap but does not compose with task modules trained on the open-loop modules; to adopt
   it, retrain the task modules with the recalibrated user modules in the loop and re-run the grid.
5. **A person on the obstacle tasks.**
6. **Task modules from tens of demonstrations**, for robot demonstrations: augmenting by the task's
   symmetries, one task module pretrained across the obstacle tasks and fine-tuned on the new one, and
   DAgger-style rounds that label the states the composed adapter visits.
