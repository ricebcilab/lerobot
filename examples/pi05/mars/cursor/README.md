# Cursor shared autonomy with a flow-matching policy

A 2D cursor task in which a flow-matching policy (a small transformer, pi0-style
action chunks) reaches targets, and a person's or a simulated agent's command
steers it: teleop, shared flow control, Flow Reversal Steering (FRS) and FRS with
a reversal adapter (MARS, fixed or learned). The steering methods are lerobot's
LeRobot's steering package ([`src/lerobot/policies/steering/`](../../../../src/lerobot/policies/steering/)),
which the cursor example binds, with the same names and knobs; the LIBERO example's README ([../libero/README.md](../libero/README.md))
describes each mode and knob on pi0.5.

Every experiment, with a person or with a simulated agent, starts from the
same portal and runs through the same mechanism; the only difference is the
agent. The October 2026 study's report is built locally by `reports/build_study_report.py` (the PDFs are not committed); the
paper's figures by `../paper/figures/build_figures.py`.

## Quick start

Everything runs in the lerobot environment with the `mars` extra (torch, matplotlib, pygame, reportlab),
from the repo root. Day one, on a fresh clone:

```bash
uv sync --extra mars        # or: pip install -e ".[mars]"
uv run --extra mars python examples/pi05/mars/cursor/tools/hub.py download
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant p1
uv run --extra mars python examples/pi05/mars/cursor/tools/analyze_results.py --study study_short --participants p1
```

`hub.py download` fetches the checkpoints the study and the grid use (the folders listed in
`hub_manifest.yaml`, 137 folders, about 150 MB, from the public Hub repo `ricebcilab/mars-cursor`; no login) into
`examples/pi05/mars/checkpoints/`. The session needs the SpaceMouse (see [../README.md](../README.md#spacemouse)
for the udev rule) and a display; `--participant sim` runs the same protocol with the expert agent, headless,
with neither. The runs go to `results/study_short/<participant>/` (the long study's to `results/study_long/<participant>/`);
`analyze_results.py` prints the per-arm table and writes `results/analysis/study_short/`.

More:

```bash
uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor -q   # ~5 min

uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant p1             # a person on the SpaceMouse
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant p1 --resume    # continue an interrupted visit
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U3 --participant p1 --session 2 # a second visit, new layouts, another user
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant sim            # the expert agent, headless (20 s)
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant sim --watch    # the expert, rendered
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant p1 --calibrate # MARS on the person's own user module (calibration first, +2.5 min)
uv run --extra mars python examples/pi05/mars/cursor/tools/analyze_results.py --study study_long                            # tables and figures for every run of a study
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1          # the October study's report for p1 (PDF)
uv run --extra mars python examples/pi05/mars/cursor/reports/build_study_report.py --participants p1 p2 p3 p4 # its group report
uv run --extra mars python examples/pi05/mars/paper/figures/build_figures.py --device cpu                          # the paper's figures (examples/pi05/mars/figures/*.pdf, *.png, stats.json)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep study_short_U1           # tune the short study's arms for one user with the expert, test (~2 min)
uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep grid --cells U1xT1 --out grid_part_U1xT1   # a part of the grid (see Sweeps)
```

If no window opens (e.g. over SSH), set `DISPLAY=:1`. A study run needs the SpaceMouse (the
block screens wait for it); `--allow-keyboard` lets the arrow keys drive the cursor too.

`hub.py manifest --check` checks that `hub_manifest.yaml` still lists every checkpoint folder the configs
reference (the study protocols' and `configs/sweeps/grid.yaml`'s methods in every cell, and the grid's seeds
1–4, each seed-0 module's `_s0` swapped for `_s<k>`); `hub.py manifest` rewrites it. `hub.py upload` (maintainers, on command only)
creates the public repo if it is missing and uploads each folder's converted files.

`tests/policies/pi0_pi05` (lerobot's pi0.5 tests, not this example's) loads the gated
`google/paligemma-3b-pt-224`: without a Hugging Face token that has accepted its license, those tests fail
with 401 errors.

## Layout

```
cursor_sa/            the library: env, policy, demos (+ the training sampler), steering (modes, reversal
                      adapters), devices, agents, config, session (the experiment mechanism),
                      renderer, records (trial records and their summary), analysis (figure style,
                      kinematics, goal-choice statistics), paths
run_experiment.py     the portal: a protocol for a participant (human or expert)
hub_manifest.yaml     the checkpoint folders the study and the grid use (written by hub.py manifest)
run_sweep.py          tune-then-test sweeps of the steering methods with a simulated agent
tools/                hub.py (the checkpoints on the Hugging Face Hub: download, manifest (--check), upload),
                      convert_checkpoints.py (trainer checkpoints to lerobot's formats, in place),
                      analyze_results.py (per-participant and pooled analysis of a study's runs),
                      diag_layouts.py (an image task module on train, fresh and moved layouts)
                      Obstacle specs take `decoy: true` (the obstacle on the reach to a random non-goal target; the goal-leak sweep
                      `grid_task_check_rel_leak`), see the design addendum of 2026-11-05 in docs/superpowers/specs, and `decoy: any`
                      (about a target drawn uniformly, the goal among them: training data without a goal cue; the pilot plan of 2026-11-09)
train/                train_policy.py, train_adapter.py
reports/              report_grid_image.py (the image task modules' gate verdicts, scan, recipe choice and width check), report_grid_vision.py (the vision grid and its decoy control);
                      build_study_report.py --participants p<N> ... (the October study) writes study_p<N>.pdf for one participant or
                      study_group.pdf for several (paper-style figures)
../paper/figures/     (examples/pi05/mars/paper/figures/) the paper's figures, PDF + PNG, and build_figures.py (one script; `--only fig1 fig2 fig3`):
                      fig1_method (MARS illustrated), fig2_study (the human study), fig3_grid (the user x task
                      grid over its five training seeds); stats.json holds the numbers the captions quote. MARS
                      (Modular Adapters for Reversal Steering) is mars's display name: MARS = user + task modules,
                      MARS-U = the user module alone, MARS-T = the task module alone; the code's tokens are unchanged
configs/              base.yaml and condition_*.yaml (arms), study_short_U1-U4.yaml and study_long_U1-U4.yaml (the
                      protocols, one per study and user, each its own sweep), sweeps/*.yaml (sweep definitions)
../checkpoints/       IGNORED  (examples/pi05/mars/checkpoints/) policies/<name>/, adapters/<name>/ (weights +
                      training logs; none tracked; hub.py download fetches the study's and the grid's);
                      adapters/participants/ holds the user modules of calibrated visits (never uploaded)
../results/           IGNORED  (examples/pi05/mars/results/) study_short/ and study_long/ (a study's runs, by
                      participant: p<N>/, sim/), sim/sweeps/<name>/ (run_sweep.py), analysis/, old/ (superseded
                      results, by date; old/2026-11-23/ the October study: p1-p7, sim/study_s1, analysis/)
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
  same readout (corruption, noise), the same controller and the same records.
- **Arm.** One control condition, a config under `configs/` (policy, mode and its settings, readout, task).
  A sweep builds one per method for every cell of its grid (`run_sweep.method_arm`: a base arm such as
  `condition_study.yaml` with the cell's user, task, policy and modules), and a study plays those.
- **Protocol** (`configs/study_short_<user>.yaml`, `configs/study_long_<user>.yaml`; `--protocol study_short --user U3`
  plays `study_short_U3.yaml`): the cells a participant plays
  (one user x the tasks, as a sweep of their own, with every cell's tuned settings), trials and practice trials
  per block, the block order rule, trial timing. One task section per task and, in it, one block per arm.
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
  renders it. A rendered trial's frames keep deadlines 20 ms apart from its start (`Renderer.frame`): a frame
  that runs late, as one with a new MARS chunk and a vision task module does (about 33 ms), is followed by a
  shorter one, so every arm plays at 50 Hz in wall-clock time as in steps; the run summary prints each arm's
  mean, median and 95th-percentile frame time.
- **Records.** `results/<study>/<participant>/<protocol>_s<session>_<task>/manifest.json` (order, layout seed,
  resolved arms with their matrices, checkpoint SHA-256s, git commit, environment) and one
  `block<k>_<arm>.jsonl` per block, one record per trial: block, arm, method, trial, practice,
  layout id, start and end times, duration, outcome, targets, path, raw and served commands,
  per-frame times (real-time runs), and the arm's settings.
- **Resume.** Re-running the same participant and session with `--resume` continues from the first
  unfinished trial; without it the portal refuses to overwrite. A resumed run must have the arms it began
  with (their settings are compared with the manifest's).

### The renovated study: a short and a long version (2026-11-23)

The study recorded in October 2026 (seven participants, now in `results/old/2026-11-23/p1` to `p7`) ran a protocol of four arm files at
noise 1.0, applied only while pushing. That protocol, its arm files and its tuning sweep are gone from `configs/`
(git history has them, and every record keeps its arm's settings). Two protocols replace it, both at input noise
0.5 served at every read, with the condition hidden
and a window that shows the cursor, the targets and the obstacles and nothing of the command or the plan
(`display.cues: false`):

| | `study_short_<user>.yaml` | `study_long_<user>.yaml` |
| --- | --- | --- |
| User | one per file: U1, U2, U3 or U4 | one per file: U1, U2, U3 or U4 |
| Tasks | T1, the open field | T1, then T2 to T5 |
| Arms | teleop, flow control, FRS, MARS with the user module | the same on T1; on T2 to T5 flow control and FRS on the task's fine-tuned policy, MARS with the user and the task module |
| Condition | the recorded study's (`step` gain mode, 5 s) | the grid's (`time` gain mode, 5 s on T1, 8 s on T2 to T5) |
| Blocks | 4 of 5 practice + 40 trials | 20, in 5 task sections |
| Session | about 14 min (20 at most) | about 1 h 30 to 1 h 45 min (2 h 40 at most) |

```bash
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_short --user U1 --participant p1
HF_HUB_OFFLINE=1 uv run --extra mars --extra pi python examples/pi05/mars/cursor/run_experiment.py --protocol study_long --user U1 --participant p1 --prepare
uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study_long --user U1 --participant p1
```

- **Participant and user on the command line.** `--participant p<N>` names the person (the folder under
  `results/<study>/`, and with `--session` the rows of the block and task orders and the layouts); `--user U<k>`
  names the user profile, the readout the person plays through (U1 vertical gain 0.25, U2 horizontal gain 0.25, U3
  a 60 degree rotation, U4 a curl field), and picks the study's protocol file for it (`study_short_U3.yaml`; a file
  named whole with `--protocol` needs no `--user`). A study named without its user is an error that lists them. The
  run folder carries the protocol's name, so one person's visits under several users sit side by side; give each
  its own `--session`, since the layouts follow the participant and the session.

- **A protocol is its own sweep.** A protocol's `cells:` is a sweep config (its base arm, the four methods with
  their modules and tuning grids, one user x the tasks played, the noise, the seed), with `settings`, what
  `run_sweep.py --sweep <protocol>` tuned for each cell (its `settings.yaml`), and `task_order` beside it. Every
  arm is the one that sweep builds for the user's cell, at its tuned setting and the sweep's noise, so a
  participant plays exactly what the expert was tuned on. A study's four protocols differ only in their user and
  its settings (a test checks it). A protocol's `study` names its folder under `results/`: the short study's runs go
  to `results/study_short/<participant>/`, every long protocol's to `results/study_long/<participant>/`.
- **Task sections.** A task is a section with its own layouts, block order and run folder
  (`results/<study>/<participant>/<protocol>_s<session>_<task>/`, a normal run, so the analysis reads it unchanged; its
  records also name the `task` and the `user`). T1 comes first and T2 to T5 in a balanced Latin square row by
  participant number and visit; each section takes the next row of the arms' square. A visit can stop after any
  block and `--resume`; finished sections are left as they are.
- **Modules.** The user modules are `study/user_<tag>_expert_n100_s0`: the grid's recipe (100 simulated expert
  reaches, 500 steps, frozen noise) at noise 0.5, with the loss over every read, a rest's noise included. Every
  participant shares them unless the visit is calibrated (next). The task modules are the grid's vision modules.
- **Calibration (`--calibrate`, either study).** The user module is trained in the session, on the participant's
  own commands. Before the first block the cursor plays demonstration reaches by itself on the open field (the
  protocol's `calibration`: 50 reaches after 5 practice ones) while the participant pushes as if steering it,
  through the user's readout. The reaches go straight to the goal with no miss and correction, each recorded step
  is played over `slow` frames (2, half speed), and the whole path is drawn ahead of the cursor, so the person can
  follow it. That is the shared modules' recipe with the person in the simulated expert's place: every frame keeps
  what the readout served, and a user module is trained on those commands toward the demonstration at the policy's
  own speed (`train_adapter.train_calibrated_user`, each step's command drawn from its frames, 500 steps, about 15 s
  on a GPU, a waiting screen meanwhile). It takes the shared module's place in MARS for the whole visit; in the long
  study the task modules compose with it as with any user's. MARS keeps the depth and prefix tuned with the shared
  module. About 2.5 minutes in all (about 71 s of reaches, as the dwell on the goal is not slowed, the gates between
  them, the training). The reaches are a run of their own,
  `<protocol>_s<session>_calibration/` (mode `playback`), and the module is
  `checkpoints/adapters/participants/<study>/<participant>/<protocol>_s<session>/`, named in every manifest and
  record of the visit; its meta holds the calibration and a closing check (how well a regenerated chunk heads
  where the demonstration did, with no module and with it, on the calibration's own steps). `--resume` continues
  an interrupted calibration and keeps a finished one, and a visit begun with `--calibrate` is resumed with it.
  `--participant sim --calibrate` plays the same block with the expert, the shared modules' recipe on new draws.
- **Scene tokens.** A vision task module reads pi0.5's tokens of a picture of the scene. `--prepare` makes them
  for the visit's layouts (which depend only on the participant, the visit and the task) and caches them under
  `checkpoints/scene/<protocol>/`, so the session itself never loads pi0.5; without it they are made when the
  session starts (about a minute).
- **Tuning.** `run_sweep.py --sweep study_short_<user>` (2 min) and `--sweep study_long_<user>` (about 40 min a user on
  two GPUs, in parts by `--cells`, then `--merge`) tune each arm with the simulated expert, as the grid does.
- **Analysis.** `analyze_results.py --study <study>` writes `results/analysis/<study>/`: `trials.csv`, `summary.csv`,
  `summary.md` (per participant × arm, and the mean over people), `success_by_arm.png`,
  `learning_curves.png` (hit rate against trial index, per arm) and `paths_<participant>.png`.
  For the October study, `reports/build_study_report.py --participants p<N> ...` (reading `results/old/2026-11-23/`) pools a participant's
  sessions into `reports/study_p<N>.pdf`: hit rates with bootstrap CIs and the MARS lead paired
  by layout, outcomes, per-session rates, cumulative hits against time and reach-aligned paths. Several
  participants give `reports/study_group.pdf`, with the participant as the unit (each one's sessions
  pooled, then mean ± s.e.m.; the lead per participant). Figures go to `results/old/2026-11-23/analysis/study_<p|group>/`.

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
a tuned run would use); `--module-seed k` swaps every `_n<demos>_s0` (or a time budget's `_t<seconds>_s0`) checkpoint
for training seed k's.
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
| `study_short_<user>` (the protocols `configs/study_short_U1.yaml` to `U4`) | expert | the short study's cell for one user (T1 under `condition_study.yaml`, noise 0.5): flow control over tau (lambda 1), FRS over pinned_steps (the full reversal), MARS over n_reverse_steps × pinned_steps; its `settings.yaml` goes into the protocol's `cells.settings` | ~2 min |
| `study_long_<user>` (the protocols `configs/study_long_U1.yaml` to `U4`) | expert | the long study's cells for one user (T1 to T5 under `condition_grid.yaml`, noise 0.5; flow control and FRS on the fine-tuned policies, MARS with the vision task modules), tuned the same way | ~40 min a user on two GPUs |
| `grid_expert_check` | expert | the expert alone (teleop, clean readout) and the frozen policy alone on the grid's tasks T1–T5 | 10 min |
| `grid_task_check` | expert_chunk | the goal-given policy's own chunks executed, with and without each task module, and each task's retrained policy alone (T2–T5) | 2 min on a free GPU |
| `grid` | expert | users {U1 … U4} × tasks {T1 … T5} at noise 1.0: the composed user × task adapter, each module alone, each task's retrained policy with its user modules, teleop, flow control and FRS; per cell flow control tunes tau (lambda 1), FRS pinned_steps (the full reversal), every MARS variant n_reverse_steps × pinned_steps | seed 0 tuned in parallel parts (`--cells`, then `--merge <parts> --out grid_s0`); seeds 1–4 at seed 0's settings (`--module-seed k --fixed-settings .../grid_s0/settings.yaml --out grid_s<k>`) |
| `grid_ft` | expert | the grid's 20 cells for its fine-tuned arm (Figure 3's in place of the retrained one): on each obstacle task the frozen policy fine-tuned on 300 demonstrations, as many as the task's module, MARS-U on it with each user's module refitted; on T1 the frozen policy and the user module; tuned as the grid's MARS variants, on the grid's trials | a part per cell, seed 0 tuned, seeds 1–4 at its settings; `results/research/grid_ft/pipeline.sh` trains and runs all five |
| `grid_vision` | expert | the grid's 20 cells with the vision task module (the rigid recipe at 300 demonstrations, rank 256 through the disk store) alone and with the user module, and flow control and FRS on each obstacle task's fine-tuned policy; Figure 3d's arms with MARS-U from `grid` and the fine-tuned arm from `grid_ft` | ~55 min per obstacle cell tuned |
| `grid_vision_decoy` | expert | the decoy control: the grid's four trained-on obstacle cells with the obstacle placed about a non-goal target, nine arms at the grid's tuned settings (`results/research/grid_vision/merge_settings.py`); not a Figure 3 panel | minutes |
| `demo_time_check` | expert_chunk | the demonstration-time curve's gate: on T2–T5, alone with the goal given and hidden, the frozen policy, with the grid's task module, retrained from scratch, and fine-tuned on 5 and 180 min of demonstrations | 3 min |
| `demo_time_mars`, `demo_time_ft` | expert | the grid's 16 obstacle cells against the time spent demonstrating each task (0.5, 1, 2, 5, 10, 20, 60 and 180 min): MARS with a task module per budget, and MARS-U on the frozen policy fine-tuned on the same demonstrations; every method tuned as the grid's MARS variants, on the grid's trials | a part per cell (`--cells`, then `--merge`), seed 0 tuned, seeds 1–2 at its settings; `results/research/demo_time/pipeline.sh <seed>` runs a seed end to end |
| `demo_time_vision`, `demo_time_baselines` | expert | the demonstration-time curve with the vision task module (Figure 3e, rederived 2026-11-09): `demo_time_mars`'s cells and budgets with the grid's rigid vision module per budget (`demo_time/image/...`), and flow control and FRS on each budget's fine-tuned policy (`demo_time_ft`'s models); driver `results/research/demo_time_vision/pipeline.sh` | ~2 days per seed on two GPUs |
| `grid_vision_any_pilot`, `grid_vision_any_pilot_decoy` | expert | the decoy-robust pilot (2026-11-09): U1xT2 with the vision module, the fine-tuned policy and its user refit trained on `decoy: any` layouts, on the grid's and the decoy layouts at the grid's tuned settings (`results/research/decoy_any/`); appendix material | minutes |

Times are on one RTX 6000 Ada. Settings only need re-tuning when something that can move them changes:
the condition, a retrained module or a retrained policy.

## Conventions

Names are unified across scripts, configs, log fields, folders and figures:

- Methods: `teleop`, `policy`, `flow_control`, `frs`; `frs` with a `reversal_adapter` is `mars` (Modular
  Adapters for Reversal Steering). MARS with learned modules is the default and carries no qualifier:
  the study's arm `mars`, sweep methods `mars` (user + task modules), `mars_user` (MARS-U), `mars_task`
  (MARS-T), `retrained_mars_user`. MARS with the closed-form adapter F = M is `mars_fixed`. Runs
  recorded before 2026-10-26 name these `frs_ra` and `frs_ra_learned`; `records.ARM_ALIASES` maps the
  recorded sessions' arm names when they are loaded.
  `flow_control` is Kao et al.'s flow control with three knobs: the command, scaled by `ic_gain` to the
  noise prior's units, is written into the first `tau` of the chunk's 16 steps of x_t at the start of
  each of the first `lambda` Euler steps, and the flow itself runs unmodified. `lambda` 1 writes the
  initial condition only (Kao et al.'s method, the default), 0 writes nothing (the policy alone), more
  re-writes as the flow proceeds (lerobot's pi0.5 `shared_flow_control`, whose own "tau" is this lambda). The
  human study ran that implementation at its depth 1, which is `tau` 16, `lambda` 1 and the command
  in the policy's action units (`ic_gain` 2.0843, `steering.to_action` of a full push): Kao et al.'s method with the
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
- Readout: `corruption` (a matrix spec) and `input_noise` (a std); condition files are
  `condition_<what>.yaml`; sweep cells are keyed `<level>|<noise>`, plus `|<axis>=<value or name>` per
  task axis (an axis given as a mapping `name: value` is keyed by its names).
- Scripts are `verb_object.py` (`run_experiment.py`, `build_study_report.py`); library modules are
  nouns. Record fields: `t_start`, `t_end`, `duration_s`, `outcome`.

## Configs

`configs/base.yaml` holds every arm parameter with a comment. Each arm says `extends: base.yaml` (or
another arm) and overrides only what it changes; a bare name is looked up in `configs/`; unknown or
missing keys are errors. A sweep names one of the two condition files below as its base and builds
every arm from it, with its method's mode and settings and its cell's user and task on top.

| Condition | Used by | What it sets |
|---|---|---|
| `condition_study.yaml` | the short study (`study_short_<user>.yaml`) | 4 targets, reach 0.35–0.5, 5 s per trial, speed gain by step |
| `condition_grid.yaml` | the grid, the long study and the other grid sweeps | noise 1.0 (a sweep's noise replaces it), 4 targets, reach 0.35–0.5, 8 s per trial (T1: 5 s, set by the sweep), speed gain by time |

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
  (null: the whole chunk); the rest of the latent is handed back to the policy (Tang et al., App. D). The forward
  pass then runs n steps of the policy's own 1/N from where the reversal stopped. The cursor and the robot run one
  implementation of each method: the controller (`steering.SharedController`), the sweeps and the `expert_chunk`
  agent take every chunk from lerobot's `SteeringPolicy` (mode `none` for the policy alone) over the cursor policy
  (`CursorFlowPolicy`, which offers pi0.5's sampler interface), so from the `FlowControlSteering` and
  `FlowReversalSteering` cores pi0.5 uses, unnormalized by the policy's postprocessor; and one operator readout
  (`lerobot.teleoperators.readout`: corruption, curl, noise, deadband). Until 2026-10-31 the cursor's forward pass took N − n steps instead, coarse at
  large depths, so earlier partial-reversal results are not comparable.
  FRS keeps the full reversal, Tang et al.'s method, and tunes `pinned_steps` (their `action_repeats`, fixed at
  5 of 15 there); MARS tunes both, since how far the reversal goes through the user module's field is part of the
  method (the 100-demonstration module fails at the full reversal).
- `input_noise` and `corruption` act on the command on its way to the controller, as in LeRobot:
  raw → M·raw (clipped to [-1, 1]) → + Gaussian noise. The corruption applies only while pushing, the noise
  at every step, so a resting user's command is the noise itself (since 2026-11-20; the grid, the study and the
  demonstration-time runs before then had noise only while pushing). Corruption specs (`lerobot.teleoperators.readout.corruption_spec`): `{scale: [sx, sy]}`,
  `{rotation_deg: d}`, `{M: rows}`, or `{curl: b, tau: s}`: a curl field on the command (the dynamics of the
  cursor curl below, acting on what the controller gets instead of on the cursor: the served command is a velocity driven by the push,
  turned clockwise by atan(τb) and slowed by 1/√(1 + (τb)²) at steady state, from rest each trial).
- `reversal_adapter`: lerobot's `AdapterModules` (composed in `cursor_sa/session.py`) with up to two modules. The
  **user module** (a fixed F = M, or a trained `LearnedUserModule` U(x_t, t)) multiplies the policy's
  velocity in the backward pass of the reversal only (frs): it undoes the user's command corruption.
  The **task module** (`LearnedTaskModule`: A(x_t, t, v, obstacles)·v + b, from obstacle tokens seen from
  the cursor) — the tokens come from the disc parameters or, with `token_source` `vlm_image` (π0.5's frozen vision tokens) or `cnn` (a small encoder trained with the module), from a rendered image of the scene — acts in both directions, so it reshapes every chunk the controller produces — in every
  mode, and the `expert_chunk` agent's intent — toward a task the frozen policy never saw. The
  forward field is task(v), the backward field user(task(v)); the two are trained apart and composed
  here (`{user: U, task: T}`). The spec forms are lerobot's adapter grammar in the cursor's two dims
  (`SteeringConfig.from_spec(spec, corruption, n_dims=2)`): `{translation: corruption}` (F = M), `{translation:
  <matrix spec>}`, `{F: <matrix spec>}`, `{learned: name}`, or `{user: name, task: name}` (checkpoint names).
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
  policy was trained on.
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

Policy folders written at this head (trained or converted) are not readable by lerobot before 21b5ee8c: their
`config.json` carries the sampler fields `num_inference_steps`, `n_action_steps`, `noise_time` and `max_action_dim`.

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
# then convert the policy it wrote to lerobot's format (below; adapters are written in it already)
uv run --extra mars python examples/pi05/mars/cursor/tools/convert_checkpoints.py
```

`train_adapter.py` trains one module of the reversal adapter, on the study's geometry (4 targets,
reach 0.35–0.5) at input noise 1.0 unless told otherwise. A **user module** learns to undo one
command corruption, calibrated open loop by an expert that knows the goal: the label is the
demonstration's next chunk, the command the expert's push through the user's readout held over the
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

`--token-source` (`discs`, the default; `vlm_image`; `cnn`) picks where a task module reads the obstacles: the disc
parameters, or the scene rendered at `--render-size` pixels (a multiple of 16 for `cnn`), through π0.5's frozen vision
tokens (`--vlm`, default `lerobot/pi05_base`) or a CNN trained with the module. `--pooled <entries file>` trains one
module over several contexts, a YAML `entries: [{obstacle, user_module, corruption}, ...]` in place of `--obstacle`,
`--user-module` and `--corruption`. Image modules are checkpointed under `image/` (`image/task_<ctx>_n300_s0_with_<user>_<source>`;
the pooled `image/task_all_n1200_s0_<source>`). Sweeps attach the scene to each cell's layouts and cache the arrays
in `checkpoints/scene/<sweep>/` (`--scene-cache` moves it); `run_sweep.py --precompute-scene` fills the cache for every cell in one
process and exits. A layout batch's scene array is `render_scene_array` (lerobot's `cursor_flow` processors: the render,
encoded for `vlm_*`); a steering policy's pre-processor gets the cursor's render step before its scene-tokens step
(`cursor_sa.steering.cursor_steering_pre_post_processors`, the pipelines `convert_checkpoints.py --steering` saves).
The `vlm_*` steps need `uv run --extra pi` and `HF_HUB_OFFLINE=1` with the π0.5 weights cached.

`--n-demos` sets how many demonstrations a module trains on (default 20 000, as many as the policy's own;
one demonstration is one reach, about a second). With few, the two modules want opposite budgets (the
data-efficiency notes, in git history): a user module over-fits the default 20 000 steps and
should train about five steps per demonstration (`--steps`, at least 100), while a task module needs the
full 20 000 steps whatever its data. `--frozen-noise` draws the readout's noise once per demonstration
step, as a recorded calibration has it, instead of anew in every batch.

`--demo-seconds` (the policy trainer's `--demo_seconds`) sets the budget in time instead: as many of the task's
demonstrations as fit in that much demonstrating, each taking its duration up to the goal's selection (0.95 s on
the disc and the wall, 1.25 s on the rings), and the two trainers draw the same ones for one budget and seed, so a
task module and a policy fine-tuned on that budget (`train_policy.py --init flow_goaldrop`) see the same
demonstrations (the demonstration-time sweeps; checkpoints under `demo_time/`, named with `_t<seconds>`).
`train_policy.py --eval_steps` picks how long a policy trains: at those steps and the last it scores the EMA
weights alone with the goal given on 200 layouts of its task under the grid's controller (seed 900, apart from the
sweeps' trials) and keeps the best, every score in its `train_log.json`.

Checkpoints go to `examples/pi05/mars/checkpoints/policies/<name>/` and `.../adapters/<name>/`, which git
ignores. The ones the study and the grid use (the study's policy and MARS adapter, the grid's policies and
modules of all five training seeds, each named with `_s<k>` after its demonstration count) are on the Hub
(`ricebcilab/mars-cursor`, listed in `hub_manifest.yaml`); `hub.py download` fetches them. Earlier experiments'
checkpoints are grouped by experiment under `adapters/learned_adapter/`, `adapters/grid_20000demos/`,
`adapters/refit/` and `adapters/data_efficiency/`, and are not distributed.

`train_adapter.py` writes `adapter.pt` in lerobot's `AdapterModules.save_module` format (kind, config, weights, meta),
which `AdapterModules.load` reads; `train_policy.py` writes a `cursor_flow` policy folder (`config.json` +
`model.safetensors`, `CursorFlowPolicy.save_pretrained`, and the policy's two processor pipelines, whose
postprocessor holds the action statistics a steering policy loading the folder by path reads), which
`cursor_sa.policy.load_policy` reads. The cursor project's own formats are `convert_checkpoints.py`'s alone: it turns
an older trainer's `policy.pt` into that folder (the loader refuses a folder holding a `policy.pt` until it is
converted), gives a policy folder converted before the pipelines its pipelines, and rewrites any `adapter.pt` still in
the cursor's old format. The originals stay beside them as `*.orig`, and running it again changes nothing.
`hub.py upload` refuses unconverted folders, and uploads (and `hub.py download` fetches) a policy's pipelines with it.

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

1. **Run participants** with `study_short_<user>.yaml` and `study_long_<user>.yaml`, after a pilot of each for the session length and the frame times.
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
