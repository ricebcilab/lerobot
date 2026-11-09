# Figure 3 on the vision task module: panel d rewired, panel e rederived

**Date:** 2026-11-09. **Branch:** `mars-image-task-module`. **Decided by the author (2026-11-09):** the leak-free
(`decoy: any`) line goes to the appendix only if reviewers ask (its pilot: `results/research/decoy_any/pilot.md`);
Figure 3d shows the vision grid (`grid_vision_s0..s4`, done 2026-11-09); Figure 3e is rederived with the vision task
module now, under the paper's placement rule (the obstacle about the goal).

## Facts

- `paper/figures/build_figures.py` still builds panel d from `grid_s<k>` (teleop, frozen flow_control, frozen frs,
  mars_task, mars_user, mars: the parametric module) joined with `grid_ft_s<k>` (ft_mars_user), and panel e from
  `demo_time_mars_s<k>` (mars_t<b>, parametric) and `demo_time_ft_s<k>` (ft_mars_user_t<b>). Nothing reads
  `grid_vision_s<k>`. The author noticed (2026-11-09).
- The vision grid's arms (configs/sweeps/grid_vision.yaml): teleop, ft_flow_control and ft_frs (each paper's baseline on
  the task's fine-tuned policy, replacing the frozen-policy flow_control and frs), mars_task and mars (vision module);
  MARS-U from `grid_s<k>`, Fine-tuned + MARS-U from `grid_ft_s<k>`; five seeds; teleop equal to `grid_s0`'s, the check.
- The demonstration-time curve (configs/sweeps/demo*time*{mars,ft}.yaml, `results/research/demo_time/`): budgets 30, 60,
  120, 300, 600, 1200, 3600, 10800 s on the grid's 16 obstacle cells, seeds 0-2 complete for the parametric module;
  per seed 32 fine-tuned policies (`demo_time/flow_goaldrop_<task>_ft_t<b>_s<k>`) and 128 user refits exist and are
  reused; the 32 parametric task modules per seed are not reused.
- Costs measured on the vision grid: a rigid vision module ~1 h on one GPU (20 000 steps whatever the budget; it holds
  ~30 GB), the tuned vision sweep of 20 cells with two vision arms and FC/FRS 7.6 h on two GPUs, a fixed-settings
  vision sweep 55 min (a sweep holds ~20 GB: a trainer and a sweep never share a GPU). Both GPUs (49 GB) are free.
- The scorer fix (7ea0e168, `--eval_obstacle`) does not touch these: every fine-tune here trains and scores on the
  paper's placement.

## The work

### Task 1: panel d from the vision grid (figure code, no GPU)

`build_figures.grid_data` reads `grid_vision_s<k>` as the run of record (teleop, ft*flow_control, ft_frs, mars_task,
mars), joins `grid_s<k>`'s mars_user and `grid_ft_s<k>`'s ft_mars_user trial for trial (teleop outcomes equal, the
existing check). `GRID_METHODS` = teleop, ft_flow_control, ft_frs, mars_task, mars_user, ft_mars_user, mars;
labels "Fine-tuned + FC", "Fine-tuned + FRS" (the frozen flow_control and frs leave the panel; on T1 ft*\* run the
frozen policy, as grid_vision.yaml maps T1); colours: the old flow_control and frs colours carry to the fine-tuned
pair; panel c (`fig3_schematic`) and the method legend unchanged in form. `tests/examples/pi05_mars_cursor/
test_figures.py` builds panel d from synthetic grid_vision/grid/grid_ft runs. Stats JSON keys that name the arms
follow. Panel e keeps reading the parametric runs until Task 3 (the figure must build throughout).

### Task 2: the 3e vision run (GPU; `results/research/demo_time_vision/`)

Configs (committed): `demo_time_vision.yaml` = demo*time_mars.yaml with the task modules
`demo_time/image/task*<task>_t<b>\_s0_with_<user>\_vlm_image_rel_rigid_r256`; `demo_time_baselines.yaml` = the
fine-tuned policies per budget under flow control (`ft_flow_control_t<b>`, tau tuned) and FRS (`ft_frs_t<b>`,
pinned_steps tuned), the grid_vision arms per budget. Fine-tuned + MARS-U per budget stays `demo_time_ft_s<k>`.
Training: `jobs_vision_s<k>.txt`, 32 vision modules per seed, the grid's recipe with `--demo-seconds <b>` (the same
draw as the budget's fine-tuned policy). Pipeline per seed, resumable, from a worktree pinned at the launch commit
with checkpoints and results symlinked to the main tree (the grid's R-G4; nothing edits the pinned code):
baselines sweeps (light, 16 cells in parallel) start at once; modules train one per GPU; the vision sweep runs its
cells 8 per GPU, in sequence; seed 0 tunes, seeds 1-2 test at seed 0's settings. Expected: ~16 h training + ~24 h tuned
sweep for seed 0, ~16 h + ~4 h per further seed: 3-4 days for three seeds.

### Task 3: panel e from the vision run (figure code, after seed 0 at least)

`DEMO_ARMS`: mars from `demo_time_vision`, ft (Fine-tuned + MARS-U) from `demo_time_ft`, ft_flow_control and ft_frs
from `demo_time_baselines`; the teleop check against `grid_s0` stays; the curve draws the four arms with panel d's
colours; stats keys per arm. Tests as Task 1's. The figure rebuilds with `--only fig3`; the paper's Figure 3 text is the
author's.

## Rulings

- Panel e carries no MARS-T line (panel d has it; the curve is MARS-UT against the fine-tuned arms). Cost if wrong: a
  vision sweep with a second tuned arm per budget, about a day.
- The vision 3e budgets and seeds are the parametric curve's (8 budgets, seeds 0-2). Cost: none; seeds 3-4 later if asked.
- The baselines sweeps tune per budget and cell as the grid's did (tau over 8 values, pinned_steps over 4). Cost: hours.
- Results and checkpoints of the parametric curve stay where they are; new names only.
