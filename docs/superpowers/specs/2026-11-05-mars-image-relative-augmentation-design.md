# MARS image task module, round 4: augmenting 300 demonstrations under the relative recipe

Date: 2026-11-05. Branch `mars-image-task-module`. Follows the round-3 design
(`2026-11-05-mars-image-two-view-demo-scan-design.md`) and its result.

## Why

Round 3 replaced the relative token geometry with two camera views and no explicit geometry. At 300 demonstrations, on
the same 500 fresh layouts per context, that module sat under round 2's relative module (`rel300`) on both contexts
(T2 29.8 / 69.0 against 55.2 / 38.2, T4 24.8 / 72.6 against 35.6 / 62.0, hits / collisions in %), and under the
fine-tuned policy. Open loop, on held-out demonstration frames, the two modules were level, so the whole gap is on states
the demonstrations never visit: the patch-minus-cursor offsets generalise across cursor positions by construction, a
picture does not. The user decided (2026-11-05, evening) to keep the round-3 code, move the experiment back to the
relative recipe, and add augmentation that could lift the relative module at 300 demonstrations. Relative offsets on a
real robot (a calibrated external camera plus forward kinematics) are to be worked out later and are not part of this
round.

`rel300` was never diagnosed: round 1's absolute module memorised its training layouts (train 95 %, fresh 28 % on T2);
whether `rel300` still does, or fails by closed-loop drift from its demonstration paths, decides which augmentation can
help. Round 3's one data-side lesson stands: the 10,000-scene teacher arm passes with the relative recipe, so the
architecture is settled and only the data is in question.

## Goals

1. Diagnose `rel300`: its hits on its own 225 obstacle training layouts, on 225 fresh layouts, and on the training
   layouts under a random rigid motion each, against the parametric module and the bare policy on the same layouts.
2. Measure, at 300 demonstrations on T2 and T4 with the relative recipe otherwise unchanged, three arms against
   `rel300`: state jitter with recovery labels, rigid motion of the scene, and both; on the gate layouts every arm so
   far has used (sweep seed 820, 500 fresh layouts per context).
3. Decide the recipe: of the three measured arms, the one with the highest mean hits over the two contexts among those
   that beat `rel300`'s mean by more than 3 points (`ADOPT`) is adopted; none clears it, the recipe stays `rel300`; an
   exact tie goes to the arm with fewer changes (`jit`, then `rigid`, then `both`). `both` is measured, so no untested
   combination is ever adopted (this replaces round 3's "adopted changes are combined"). Level and in band against the
   parametric module are reported as before. Rigid motion can be adopted if it wins; the robot conditions below are
   then part of the report (the user: the robot implementation is worked out later).
4. Stage 2, after the decision: the demonstration scan at 1,000 and 3,000 with the chosen recipe, if the 300-point
   is still short of the parametric module. Its job lists and sweep configs are written when the recipe is known. My
   call for a recipe with rigid motion: the sampler holds every scene's tokens on the GPU (about 20 GB per 12,000
   scenes; `K = 32` at 1,000 demonstrations would be 58 GB), so stage 2 holds the scene count at the teacher arm's
   about 10,000 (`K = 9` at 1,000, `K = 2` at 3,000) and the report says so.

## Non-goals and anti-goals

- No teacher data, closed-loop (expert-labelled) data, or any demonstration beyond the 300 in a training set. The
  teacher arm stays a comparison ceiling.
- No change to what any existing command computes: every new flag defaults to off, and a run with today's flags draws
  the same random numbers and saves the same weights.
- No change to the two-view code; `vlm_views` refuses the new flags.
- Nothing is deleted. Results and logs that stop being current move to `old/`.
- Jitter is the method expected to carry to a robot as it is; rigid motion carries the robot conditions listed below,
  which the report states whenever a rigid arm is adopted or quoted.
- The diagnostic does not gate the trainings: all six run; the diagnostic's readings are reported next to them.

## The relative recipe (unchanged)

`rel300`'s command, per context, from round 2's job list (`jobs_r2_rel.txt`):

    train_adapter.py --kind task --obstacle <ctx> --user-module <user> --corruption <spec> --token-source vlm_image
      --token-geometry relative --frozen-noise --seed 0 --n-demos 300 --name image/task_<ctx>_n300_s0_with_<user>_vlm_image_rel

Defaults it relies on: 20,000 steps, batch 512, `--val-demos 2000`, `--p-free 0.25` (75 of the 300 layouts have no
obstacle), one 224-px overhead picture per layout without the cursor, encoded once (256 tokens x 2048, held on the GPU
by the sampler), `--lambda-field 0.03`. A run takes about 54 minutes on one GPU. The new arms add one flag each and the
suffixes `_jit`, `_rigid`, `_both` to the module name.

## Augmentation 1: state jitter with recovery labels (the method)

What it does. For every training sample the sampler draws an offset `delta ~ N(0, sigma^2 I)` in screen units (the
screen is the unit square), displaces the observed cursor position by it, and corrects the label chunk so the cursor
rejoins the demonstrated path: the first `m` steps of the chunk each lose `delta / m`. Everything else in the sample
(history, goal, targets, scene tokens, the expert's push used by the field term) stays the demonstration step's. The
relative offsets are computed by the module from the displaced position, so the module sees the obstacle from a state
beside the path and learns the push back onto it. The validation split is never jittered.

Robot reading: perturb the recorded proprioception, keep the recorded position targets. Valid in any action space;
needs no assumption about the frozen policy; changes nothing in the image.

Values (the user's and mine): `sigma = 0.02` (a quarter of the T2 disc's radius, a third of a token patch), `m = 4`
(80 ms at 50 Hz; the correction is at most about half a typical reach speed). Flags `--jitter 0.02` (default 0 = off)
and `--jitter-steps 4`. Refused with `--token-source vlm_views` (the picture would show the cursor elsewhere). Allowed
with `discs`, `vlm_image`, `cnn`, teacher demonstrations and pooled runs.

Approximations, accepted: the cursor's velocity lag makes the recovery label approximate rather than exact; the expert's
push for the field term is the undisplaced step's (at sigma 0.02 the direction to the goal moves by at most a few
degrees). Randomness: the offsets are drawn from torch's global generator on the device, only when `--jitter > 0`, so a
run without the flag draws exactly as today.

How many: online, a fresh offset per sample, so every one of the run's 10.2 million samples (20,000 x 512) is a new
state within about 0.02 of the 300 paths. `rel300` has 14,500 distinct states (300 demonstrations of about 48 steps),
each seen about 700 times. Jitter adds no scene and no new obstacle-to-reach geometry.

## Augmentation 2: rigid motion of the scene (a diagnostic of layout diversity)

What it does. Each training demonstration gets `K` extra copies, each under one random rigid motion of the unit
screen: a rotation about the screen's centre by an angle uniform in [0, 2 pi), a shift uniform in [-0.25, 0.25]^2,
and a mirror about the screen's vertical midline with probability one half; the start, every target, every obstacle
disc's centre, and the action sequence (displacements) move together; radii do not change. A copy is accepted when
everything stays on screen (the start, the real targets and the discs with their radii inside
`[SCREEN_EDGE, 1 - SCREEN_EDGE]` for the discs, and the start and the real targets at least `TaskConfig.margin` (0.05, where the layout sampler keeps them) from the edge; every path point inside the screen) and replaying its actions from its start
succeeds; up to `20 K` draws per demonstration, fewer copies kept when they run out (the count is printed). The copies
are rendered from their moved layouts (exact, no picture resampling) and encoded like any layout. The free layouts
(no obstacle) are moved too. The expert's pushes of the field term are NOT recomputed on a copy: the builder check
measured that the route planner's side choice is not invariant under rigid motion (on moved copies the mean cosine
between a recomputed push and the moved action is 0.97 on T2 and 0.94 on T4, with 2.8 % and 8.4 % of copies under 0.5
and the planned route taking the other side on 4 % and 10 %), which would feed wrong-way commands to the rigid arms
alone. A copy's pushes are its original's pushes moved by the same motion (`move_actions` on the push sequence), so
push and action agree on a copy exactly as on its original; `ExpertData` accepts given pushes. The user's corruption
stays in the screen frame, as it is in reality. With `--frozen-noise` a copy draws its own noise track (one per
demonstration step, as every demonstration does; the teacher arm's 10,000 demonstrations had their own too); reusing
the original's track would need a copy-to-original map and is not done. The validation split is not augmented.

Why it is a diagnostic here. A moved copy is a new picture and a new set of offsets to the module, but the same
obstacle-to-reach geometry; so at equal scene count the comparison with the 10,000-scene teacher arm says whether the
module needs new geometries or just more views of the same 225. On a real robot the method has three conditions the
cursor meets for free, stated here so the report can say so: the frozen policy's velocity field must be made
equivariant by construction (evaluated on the original observation at the back-rotated point, never on a rotated
image), the arm in the external picture appears from directions it never comes from, and the correction must live in
the table plane (end-effector x, y), not in joint space. Mirroring also assumes no handedness in the task.

Values: `K = 32` (`--rigid-motions 32`, default 0 = off): 300 x 33 = 9,900 scenes and about 480,000 states, the teacher
arm's counts. Measured on the seed-0 demonstrations (round 3's feasibility check): the median demonstration stays on
screen at every rotation, the worst tenth at about half; a quarter of the screen of shift per axis is typical. Cost: 9,900
pictures encode in about two minutes; their tokens on the GPU take what the teacher run's did (about 20 GB with the
2,000 validation scenes), one job per GPU. Refused with `--token-source vlm_views` and with a curl corruption (the
curl's handedness is not mirror-invariant). Own generator `default_rng([seed, 11])`, so no other draw changes.

## The diagnostic script

`examples/pi05/mars/cursor/diag_layouts.py`: rebuilds a run's 300 training layouts exactly as the trainer drew them
(same generator and order: `layouts_for` with `default_rng(seed)`), keeps the 225 with an obstacle, draws 225 fresh
layouts of the task from another seed, and moves the training layouts by one random rigid motion each (the primitive
above). On each set it runs the gate's oracle arm (`expert_chunk`, teleop, readout off, 10 reverse steps) for the
named image module, the parametric module and the bare policy, with the image tokens computed as the sweep computes
them, and prints hits and collisions per set. The parametric module's equality across the three sets is the built-in
check that the moved layouts are valid (its disc tokens are cursor-relative). About ten minutes on a GPU for `rel300`
on one context. Output: a Markdown table on stdout and a JSON file under `results/sim/reports/diag_layouts/`.

Reading it (the readings are not exclusive; the report quotes the numbers and the readings that hold): train far above
fresh = memorisation of pictures; train about equal to fresh and both low = closed-loop drift; train about equal to
moved = the module already generalises across orientation, so rigid motion adds little; train far above moved = the
module is tied to the orientation it saw, which rigid motion attacks directly. It runs before the trainings only
because its encoder must not share a GPU with a trainer's encoding phase; it decides nothing about which trainings run.

## Sweeps and report

- `configs/sweeps/grid_task_check_rel_aug.yaml` (committed; seed 820; T2, T4; 500 test trials): rows `oracle`,
  `oracle_task`, `oracle_ft`, `oracle_task_rel` (the existing `rel300` modules), `oracle_task_rel_jit`,
  `oracle_task_rel_rigid`, `oracle_task_rel_both`, and the goal-hidden `policy`, `policy_task`, `policy_task_rel*`
  rows for the flag. `focus: oracle_task_rel_both`, `against: [oracle_task, oracle_ft]`.
- `report_grid_image.py --check` already reports every `oracle_task_<x>` arm. `--select` gains `--prefix` (default
  `views`, here `rel`) and `--base` (default `base`, here `rel`) so the adoption rule reads
  `oracle_task_<prefix>_<arm>` rows with the base arm named; `select.json` keeps its shape. The existing `--select`
  call computes exactly as today.
- Stage 2 (after the decision): `grid_task_check_rel_n1000.yaml`, `grid_task_check_rel_n3000.yaml` and a `--scan`
  over them, following the round-3 scan files, with the chosen suffix in the module names; `--scan` gains the same
  `--prefix`.

## Where the code goes

- `src/lerobot/envs/cursor.py` (mypy-strict): `move_layouts(layouts, angle, shift, mirror) -> Layouts` and
  `move_actions(actions, angle, mirror) -> np.ndarray`, vectorised over the batch (angle `[B]`, shift `[B, 2]`,
  mirror `[B]` bool); rotation about (0.5, 0.5), mirror as `x -> 1 - x` applied before the rotation; `scene` is
  dropped (the caller re-renders). `layouts_on_screen(layouts, edge=SCREEN_EDGE, margin=None) -> np.ndarray[bool]` (`margin`, default `edge`, bounds the start and targets) for the
  acceptance test.
- `examples/pi05/mars/cursor/cursor_sa/demos.py`: `augment_demos(rng, layouts, actions, lengths, k, task, push=None,
shift=0.25, tries=20) -> (layouts, actions, lengths, kept, push)` (the originals first, then the accepted copies;
  `kept` = copies per original; `push` `[B, L, 2]` sequences moved like the actions, `None` when not given);
  `DemoSampler(..., jitter=0.0, jitter_steps=4)` applying the jitter in `sample`.
- `examples/pi05/mars/cursor/train/train_adapter.py`: flags `--jitter`, `--jitter-steps`, `--rigid-motions`, their
  refusals, `ExpertData(..., push=None, jitter=0.0, jitter_steps=4)` (given pushes replace `expert_pushes`), the
  augmentation call in `expert_splits` (train split only, after its demonstrations and the originals' pushes, before
  rendering), the jitter handed to the train split only, both in the checkpoint `meta` (`jitter`, `jitter_steps`,
  `rigid_motions`, and `rigid_kept`: the copies actually kept), and a printed line with the kept count.
- `examples/pi05/mars/cursor/diag_layouts.py` (new) and `report_grid_image.py` (`--prefix`, `--base`).
- Tests beside the round-3 ones: `tests/policies/...` is not touched; env tests in `tests/envs/` (if the cursor env has
  tests there; else beside the existing cursor tests), the rest in `tests/examples/pi05_mars_cursor/`.

## Runs (controller, after the code is reviewed)

1. Diagnostic: `diag_layouts.py` for `rel300` on T2 and T4 (one GPU each, about ten minutes).
2. Six trainings, one job per GPU, run-folder job list `jobs_r4_aug.txt`: `_jit`, `_rigid`, `_both` on T2 and T4,
   about three hours in total.
3. Gate `grid_task_check_rel_aug` (both cells in parallel, about 15 minutes), `--check` and `--select --prefix rel
--base rel`.
4. Report to the user: the diagnostic's table, the gate table, the adoption decision by the rule in goal 3, costs, and
   the rulings made. Stage 2 waits for the user's go.

## Left to the builder

The exact form of the jitter's chunk correction when the chunk is shorter than `m` (correct over what there is); how
`augment_demos` batches `replay` (500 rows at a time, like `make_demos`); the diagnostic's arm construction (reuse
`run_sweep.py`'s loaders and `run_trials`); test values.

## Addendum (2026-11-06): stage 2, the goal-leak check and the clean-up

The 300-point gate (seed 820, 500 fresh layouts) gave rel300 55.2 / 38.2 and 35.6 / 62.0, jitter 23.6 / 69.0 and 14.6 /
85.4, rigid 98.4 / 1.0 and 88.2 / 11.6, both 87.2 / 9.0 and 76.8 / 21.4 (hits / collisions %, T2 and T4; parametric
99.6 / 0.0 and 92.8 / 3.4). Rigid is the recipe (level on T2, in band on T4). The user then asked for stage 2, seeds
1-2, the goal-leak check and the clean-up, plus an archive of abandoned artifacts and a rename of the kept ones.

**Stage 2 and seeds (controller).** The rigid recipe at 1,000 (9 copies) and 3,000 (2 copies) demonstrations, scenes
held near 10,000; seeds 1 and 2 at 300 with 32 copies (user modules, parametric references and fine-tuned policies at
those seeds exist; the sweep's `--module-seed` swaps every `_n300_s0` name). Sweeps `grid_task_check_rel_n{300,1000,
3000}` and `grid_task_check_rel_rigid` (committed); reports `--scan --prefix rel` and `--check`.

**The goal-leak check (decoy reach).** The flag on T2 (goal-hidden hits 74.4 with the rigid module against the
parametric module's 59.0) suggests the picture leaks the goal: every obstacle is placed relative to the reach from the
start to the goal, so where the obstacle is says where the goal is. The check places the obstacle relative to a decoy
reach instead: the obstacle spec gains `decoy: false` (default; `true` places the obstacle on the line from the start
to one real non-goal target drawn uniformly, with one extra draw from the layout generator only when `decoy` is set,
so every existing layout is unchanged); the passability check stays about the true goal; demonstrations and routes
still go to the true goal. A sweep `grid_task_check_rel_leak` plays T2 and T4 with `decoy: true` for the bare policy,
the parametric module and the rigid module, goal given and goal hidden. The gate table gains a `wrong target` column.
Reading: with the goal hidden, a module that infers the goal from the obstacle sends the policy to the decoy (wrong
target well above the parametric module's); with the goal given, hits on decoy layouts say whether the module still
avoids an obstacle that is not on its reach.

**Clean-up (user).** Scripts: only the two entry points stay at the top of `examples/pi05/mars/cursor`;
`analyze_results.py`, `convert_checkpoints.py`, `hub.py`, `diag_layouts.py` move to `tools/`, `report_grid_image.py`
to `reports/`; the test path setup, the README and the run-folder scripts follow. Tests: cut to about half the lines
and under ten minutes on an idle CPU, one behaviour test per feature, one tiny training per end-to-end path, the
per-round files folded into the existing ones; the equivalence tests and their reference files stay (they pin
agreement with the earlier cursor code; my call, reversible). Archive: artifacts of abandoned branches (round 1's
absolute and cnn modules, round 3's two-view modules and sweeps, round 4's jitter arms, and their run-folder scripts,
reports and scene caches) move to `old/<round>/` folders beside where they were, with an index; nothing is deleted.
Rename of the kept artifacts to a simpler convention: proposed to the user and done only on their approval.
