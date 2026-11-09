# MARS decoy-robust obstacle rule: the pilot

**Date:** 2026-11-09. **Branch:** `mars-image-task-module` (b3b7b113). **Approved:** the author chose the controller's
recommendation of 2026-11-09 ("I'll go with your recommendation"): a pilot before any two-GPU-day retrain.

## Facts (gathered before anything runs)

- The layout sampler (`lerobot.envs.cursor._layout`) places every obstacle task's obstacle relative to the reach from
  the start to one target, `toward`: the goal by default, a random non-goal target under `decoy: true`. The grid's
  tasks draw four targets (`--targets 4 4`), so a target drawn uniformly is the goal one time in four.
- The decoy table (results/sim/reports/grid_vision/report.md) shows both module families send the goal-hidden policy to
  the target the obstacle points at: on U1xT2 the vision MARS-UT arm's wrong-target share is 44.4% on decoy layouts
  against 15.2% on the grid's (success 49.4 against 83.1); the parametric arm 23.2% against 12.2%.
- The fine-tuned policy reads obstacle tokens (centre, offset, radius), not pixels; it trains on the same placement
  rule with a free share of 0.25 (`--p_free`), as the task module does (`--p-free`). Nothing in either training
  decorrelates the obstacle's position from the goal.
- The grid's U1xT2 jobs: the vision module ~1 h on one GPU (20 000 steps, 300 demos, 32 rigid copies, rank-256 store),
  the fine-tuned policy 50 min (40 000 steps), the user refit on it minutes (500 steps), a fixed-settings sweep cell
  minutes. Both GPUs are free (2026-11-09 13:00).
- The sweep's test layouts come from `seed + 10` and the task spec alone (run_sweep.py:1029), not from the cell's
  name or the checkpoints, so a new sweep config with the pinned seed replays the pinned trials; teleop's outcome is
  the check (20.2% on U1xT2).

## Design

One new value of the obstacle spec's existing key: `decoy: any` places the obstacle about a target drawn uniformly
over all the targets, the goal among them. Training data drawn so carries no goal cue in the obstacle's position; the
only learnable rule is avoidance relative to the motion. Evaluation keeps the grid's placement (obstacle about the
goal: every trial an obstruction trial, the hard case, a subset of the training layouts) and the decoy control
(`decoy: true`) as the proof that the prior is gone. Both evaluation configs exist; only the training placement changes.

## The pilot (U1xT2, seed 0)

Checkpoints, named so `run_sweep.py --module-seed k` can address later seeds (`_n300_s0` kept):

1. `image/task_disc_large_any_n300_s0_with_gain025y_vlm_image_rel_rigid_r256`: the grid's T2 vision command with
   `decoy: any` in `--obstacle` (GPU 0, ~1 h).
2. `flow_goaldrop_disc_large_any_ft_n300_s0`: the grid's T2 fine-tune command with `decoy: any` (GPU 1, ~50 min).
3. `user_gain025y_expert_n100_s0_on_disc_large_any_ft_n300`: the grid's refit command on policy 2 (GPU 1, minutes).

Sweeps, each one cell at the pinned settings (no tuning): `grid_vision_any_pilot` (the grid's placement) and
`grid_vision_any_pilot_decoy` (`decoy: true`), arms teleop, ft_flow_control, ft_frs, ft_mars_user, mars_task, mars;
settings file `results/research/decoy_any/settings_U1xT2.yaml` (U1xT2's entries of grid_vision_s0/settings.yaml and
grid_vision_decoy_settings_s0.yaml). Outputs `results/sim/sweeps/grid_vision_any_pilot_s0`, `..._decoy_s0`.

Reading (`results/research/decoy_any/compare.py` writes `results/research/decoy_any/pilot.md`): per arm, success,
collision and wrong target on the grid layouts and on the decoy layouts, the pilot beside the pinned seed-0 runs.

- Go for the full retrain if MARS-UT (vision) holds its grid success near the pinned 83% (a drop of a few points is
  the leak's share, acceptable) and its decoy wrong-target share falls to the order of teleop's (4.8%).
- Stop if its grid success collapses toward MARS-T's level (15%): at 300 demonstrations the avoidance signal (about a
  quarter of the demonstrations have the obstacle in the way) is then too thin, and the appendix decoy table is the
  fallback; raising copies, not the goal share, is the only honest lever.

## Rulings

- `decoy: any` extends the existing key rather than adding a `toward` key: two committed configs, a report and the
  README spell `decoy: true`, and one key for one choice (the target the obstacle is placed for) is the smaller grammar.
  Cost if wrong: a rename.
- The free share stays 0.25 in both trainings, so the pilot differs from the grid in one thing. Cost: ~19% of the
  demonstrations have the obstacle in the way (0.75 x 0.25); if the pilot stops on thin signal, `--p-free 0` is the
  first thing to try before more copies.
- The pilot reuses the pinned settings (tuned for the leaky modules). Cost: the new module may prefer other reverse
  steps; a full run re-tunes at seed 0 as the grid did.
- Nothing under `results/sim/sweeps/grid_vision*` or the grid's checkpoints is touched; the pilot writes new names only.

## Not in the pilot

Seeds 1-4, tasks T3-T5, the Figure 3 report, any paper text: all after the author reads `pilot.md`.
