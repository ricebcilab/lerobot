# MARS grid rederived with the vision task module — run plan

> **For agentic workers:** a run plan, not a build: the code is in place (round 4c's store and rank, round 4d's package).
> Executed by `examples/pi05/mars/results/research/grid_vision/drive_grid.sh` from a worktree pinned at the post-Task-F head.

**Goal:** Figure 3d's grid with the vision task module in place of the parametric one, flow control and FRS on the
fine-tuned policy in place of the frozen-policy arms, and a decoy control (not a Figure 3 panel) that measures how much
of an obstacle-reading arm's gain is the obstacle-placement prior.

**Decisions (the user's, 2026-11-08):** FC and FRS run on the fine-tuned policy like Fine-tuned + MARS-U and only those
are reported; the grid and the Fig 3e curve are rederived with the vision adapter (the rename is waived); the decoy
control runs for one cell per obstacle task, is kept out of Figure 3, and its table goes to the user for placement;
the grid runs without a further go once Task F reports, in parallel with the refactor (altitude items 1-6) from a pinned
worktree; the Fig 3e curve still needs a go.

**Spec of the module:** `docs/superpowers/specs/2026-11-05-mars-image-relative-augmentation-design.md` (+ addendum);
the store and rank: `docs/superpowers/plans/2026-11-06-mars-round4c-token-store-and-width-check.md`.

## Facts

- Every Figure 3 arm hides the goal flag from the policy (`CursorController.act` builds the observation with
  `goal_given=False`); the picture draws all targets alike; disc tokens carry no target. The structural leak is the
  obstacle on the start->goal line, shared by both module families (decoy-reach sweep: 82/74 parametric, 83/74 image,
  44 bare, wrong-target % with the goal hidden, T2/T4).
- Existing: user modules U1-U4 seeds 0-4; parametric task modules 4 tasks x seeds 0-4; fine-tuned policies 4 tasks x
  seeds 0-4 with their 80 refitted user modules; vision rank-256 modules for T2 and T4 seed 0 (round 4c's width check:
  rank 256 trails full width by 1.8 / 2.0 points, inside the 5-point rule).
- Missing: vision rank-256 modules for T3 and T5 seed 0, and all four tasks seeds 1-4 (18 trainings; ~62 min each
  through the store; new layouts encode at ~45 pictures/s, 9,600 per module, ~3.5 min; the store grows ~1 MB per
  picture, ~180 GB for the 18, on 1.3 TB free).
- Sweep cost (from the parametric grid and the fine-tuned grid): a tuned T3-T5 cell with one 40-setting method ~24 min
  (fine-tuned policy), so grid_vision_s0 (two 40-setting methods + 12 baseline settings) ~55 min per obstacle cell,
  ~12 GPU-h, ~6.5 h wall on two GPUs; a fixed-settings seed ~40 min; the decoy sweeps minutes each.
- Wall estimate: trainings 2 + 16 at one per GPU (1 h + 8.5 h), s0 sweep 6.5 h, seeds 2.5 h, decoy 0.5 h: ~19 h.

## Rulings

- R-G1. The grid's vision module is the rank-256 store recipe, the same the Fig 3e curve must use (the full-width store
  would need ~3 TB for the curve); one recipe across 3d and 3e. Costs if wrong: 18 trainings + the sweeps at full
  width (~1 day).
- R-G2. The new sweep `grid_vision` carries teleop, ft_flow_control, ft_frs, mars_task (vision), mars (vision);
  MARS-U stays grid_s<k>'s mars_user and the fine-tuned arm grid_ft_s<k>'s ft_mars_user (neither involves a task
  module). The figure builder's method->run mapping changes after the runs (a code task, CPU).
- R-G3. The decoy control (`grid_vision_decoy`) runs the four trained-on cells (U1xT2, U2xT3, U3xT4, U4xT5) with
  `decoy: true`, nine arms at the grid's tuned settings (merged by `merge_settings.py`), seeds 0-4, 500 test trials.
- R-G4. GPU work runs from a worktree pinned at the post-Task-F head with `examples/pi05/mars/{checkpoints,results}`
  symlinked to the main tree; the refactor's acceptance check re-runs one grid cell (U1xT2, fixed settings) on the
  refactored code and must reproduce the pinned numbers.

## Files

- `examples/pi05/mars/cursor/configs/sweeps/grid_vision.yaml`, `grid_vision_decoy.yaml` (committed).
- Run folder `examples/pi05/mars/results/research/grid_vision/`: `jobs_vision_s0.txt` (2), `jobs_vision_seeds.txt` (16),
  `run_jobs.py` (REPO env pins the checkout), `merge_settings.py`, `drive_grid.sh`, `pipeline.log`, `logs/`.
- Outputs: `results/sim/sweeps/grid_vision_s{0..4}`, `grid_vision_decoy_s{0..4}`, `grid_vision_decoy_settings_s0.yaml`;
  modules `checkpoints/adapters/image/task_<task>_n300_s<k>_with_<user>_vlm_image_rel_rigid_r256`.

## Steps

1. Task F reports -> commit the two configs -> `git worktree add ../lerobot-grid HEAD`, symlink checkpoints and results,
   `uv sync --locked --extra mars --extra pi` there.
2. `REPO=/home/user/Projects/lerobot-grid nohup bash drive_grid.sh` (detached). Progress: `pipeline.log`.
3. Meanwhile (CPU): `reports/report_grid_vision.py` — per task x method means over users and seeds for the grid (with
   mars_user and ft_mars_user read from their runs) and the decoy table (cell x arm, hits / collisions / wrong target,
   decoy against the grid's own cell).
4. On DONE_grid: the two tables to the user; the figure builder's mapping and Fig 3d text follow the user's reading.
