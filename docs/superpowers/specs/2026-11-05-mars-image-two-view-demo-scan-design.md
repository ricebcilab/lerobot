# MARS image task module, round 3: two robot-style views and a demo-count scan

Follows `2026-11-04-mars-image-round2-teacher-relative-design.md`. Round 2's code and results
stand; this document replaces its "round 3" paragraph (image-space augmentation with
transformed actions), which the user ruled out on 2026-11-05.

## Why

Round 2's quick gate (500 fresh layouts, hits / collisions in %):

| arm                                                      | T2          | T4          |
| -------------------------------------------------------- | ----------- | ----------- |
| parametric module, 300 demos (`oracle_task`)             | 99.6 / 0.0  | 92.8 / 3.4  |
| fine-tuned policy, 300 demos (`oracle_ft`)               | 41.8 / 58.2 | 34.6 / 63.4 |
| `rel300`: relative geometry, 300 demos                   | 55.2 / 38.2 | 35.6 / 62.0 |
| `teach10k_rel`: relative geometry, 10,000 teacher scenes | 99.6 / 0.0  | 95.8 / 2.8  |

The image module can do the task when it has enough scenes, and 300 demonstrations of one
fixed overhead picture each are not enough. Two things in rounds 1–2 do not exist on the
user's robot (a wrist camera and an overhead camera, pi0.5 as the policy):

- **The explicit offset of each patch from the effector** (`token_geometry: relative`). On an
  overhead camera it needs the camera calibrated against the robot. pi0.5 itself has no such
  term: SigLIP gives each patch a learned position inside its image, the cameras are told apart
  by their order in the prefix, and the robot's state enters as tokens.
- **Whole-scene mirrors, rotations and slides with transformed actions.** They are valid only
  for a top-down view.

The user's decisions (2026-11-05): the design must fully translate to the robot; geometry only
if a robot can supply it, else none; augmentation only if it can be applied to recorded robot
data, which includes small shifts and rotations of the image; the cursor gets both a
cursor-centred view and an overhead view, as the robot has; and the experiment is a scan over
the number of demonstrations (300, 1,000 and 3,000; not 10,000), to find how many the
vision-based module needs to perform as the parametric one does.

## Goals, non-goals, anti-goals

Goal: one number per context, the demonstrations the two-view image module needs to be level
with the parametric module trained on 300, measured on the round-2 gate's rollouts; and the
curve it comes from.

Non-goals this round: T3 and T5, the pooled arm, the grid, seeds 1–2 (they follow at the
matching count, on the user's go); counts above 3,000; the `cnn` and `vlm_prefix` sources;
photometric augmentation; early stopping; the detector branch (the VLM marks the obstacle and
its box feeds the parametric module), which stays the fallback if the scan ends without a match.

Anti-goals (each would make the result useless for the robot even if the numbers were good):

- anything derived from the cursor's position entering the scene tokens (offsets, crops chosen
  by a projection that a robot would need calibration for; the wrist view is exempt, it is a
  camera);
- obstacle parameters reaching the module's input;
- demonstrations other than the scripted expert's (no teacher rollouts in the scan's arms);
- transforms of a scene with transformed actions.

Rendering the two views from the layout is the simulator's camera, not privileged input.

## The input: two views, encoded as the cursor moves

Both views are 224 px pictures drawn as `render_layouts` draws them (obstacles, targets on top,
no goal mark), plus the cursor as a small disc in a colour the palette does not use:

- **overhead**: the unit screen, as today, with the cursor at its current position (a real
  overhead camera sees the arm);
- **wrist**: a square window of side `wrist_span` screens centred on the cursor, what lies
  outside the screen drawn as background; the cursor mark sits at its centre.

Each picture goes through pi0.5's `image_tokens` (the `vlm_image` path of `SceneEncoder`:
SigLIP and the projector, 256 tokens of width 2048, frozen). The module reads the overhead
tokens followed by the wrist tokens, 512 scene tokens, as a new token source (`vlm_views`).

**Geometry.** A learned position per scene token (512 of them, so it carries the camera and the
patch), which is the existing `absolute` geometry over a longer token list. `relative` is
refused for the new source. The query keeps the cursor's position as it has it today: that is
the robot's own state, which pi0.5 also receives.

**In the demonstrations** a frame (both views) is encoded every `view_stride` = 2 planned
steps, the gate arm's replan period. 300 demonstrations are 290 s (T2) and 371 s (T4) of
demonstrating at 50 Hz, so about 24 and 31 frames per demonstration. Training and validation
samples of this source are drawn only at the steps that have a frame, so a sample's views are
always drawn at its own cursor position: the rollouts build an observation only at a replan,
with views at the current position, and never see an older frame.

**In the rollouts** the tokens are made at every observation: the two views are rendered at the
trials' current cursor positions and encoded, each time the controller or the oracle agent
asks for an observation. Only unfinished trials are encoded, and two observations built at one
position (the goal-hidden policy rows do this) are encoded once. Nothing is cached on disk for
this source. The span and the token reduction (below) belong to the module: its checkpoint
carries them and a rollout makes each module's tokens from them, so one sweep can hold modules
with different spans.

The existing sources, their checkpoints and their sweeps are untouched.

## Token storage

A frame is two images of 256 × 2048 float16 tokens, 2.1 MB. At full width 3,000
demonstrations are 150–195 GB per run, and tokens kept in RAM cost about 1 GB of transfer per
training step.

So the tokens are reduced once, before storage, by a fixed linear map to `scene_rank` = 256
dims: the mean and the top principal directions of the run's own training tokens (a sample of
them), saved with the checkpoint and applied to the rollouts' tokens; the module's
`scene_in` then reads width 256. The run prints the share of variance kept.

| demonstrations | images to encode (T2 / T4) | reduced tokens (T2 / T4) |
| -------------- | -------------------------- | ------------------------ |
| 300            | 14,400 / 18,600            | 1.9 / 2.4 GB             |
| 1,000          | 48,000 / 62,000            | 6.3 / 8.1 GB             |
| 3,000          | 144,000 / 186,000          | 19 / 24 GB               |

**Check that the reduction is harmless,** in stage A, before any larger run: one T4 run of the
base recipe at 300 demonstrations with full-width tokens (about 39 GB with its validation
split, kept in RAM), gated beside the base recipe's reduced T4 run. If the reduced run's hits
are more than 5 points under it, the chosen recipe is retrained at 300 on both contexts at rank
512 and its T4 run compared with the same full-width run; if that is still more than 5 points
under, the scan runs at full width with the tokens in RAM. Stage A's choice of recipe is kept
either way (it is assumed not to depend on the rank).

The validation split is 300 scripted demonstrations for every run (the default 2,000 would
dominate the encoding).

## Augmentation

Two methods, both valid on recorded robot data, both applied to the training split only
(validation and rollouts see clean pictures and every token):

- **Scene-token masking.** Each training sample hides a random 30% of the 512 scene tokens
  from the attention (never the null token), in the flow-matching loss only. The field term
  through the user module (its reverse flow and its loss) sees every token, so its target and
  its prediction come from the same field.
- **Small shift and rotation of the picture, actions unchanged.** The trainer's own bilinear
  resample (`move_pictures`, not torchvision's transform) at lerobot's default `RandomAffine`
  magnitudes: up to 5° about the image centre and up to 5% of the image side.
  Because the tokens are precomputed, the draw is frozen: each stored picture (each view of
  each frame, independently) is moved with probability 0.5 by one random draw before it is
  encoded, and keeps that draw for the run. This costs no extra encoding or storage, and a
  demonstration's 24–31 frames give it as many different draws.

No photometric augmentation: the cursor's flat render has no lighting to vary.

## The scan

Scripted demonstrations (the stand-in for human ones, as in every arm so far), contexts T2 and
T4, seed 0, 20,000 steps, everything else as `rel300`'s recipe. Points: 300, 1,000, 3,000.

1. **Stage A, the recipe, at 300 demonstrations.** The base recipe (`wrist_span` 0.5, masking
   and the affine on) and three single changes to it: `wrist_span` 1.0; no masking; no affine.
   Four recipes on both contexts, eight runs, plus the full-width T4 run of the base recipe
   (Token storage), gated on 500 selection layouts of their own (the gate's arm at seed 821),
   so the choice is not made on the scan's test layouts. A change is adopted when its mean hits
   over T2 and T4 beat the base recipe's by more than 3 points; adopted changes are combined,
   untested as a combination. The rank is settled here too, before stage B starts.
2. **Stage B, the scan.** The chosen recipe at 300 (stage A's checkpoints when the recipe is
   one stage A trained), 1,000 and 3,000, both contexts, gated on the round-2 test layouts
   (seed 820) after each count.

**Level with the parametric module** (my reading of "the same performance"): hits at least
parametric's minus 3 points and collisions at most parametric's plus 3 points, parametric
being `oracle_task` at 300 demonstrations on the same layouts. The scan stops at the first
count that is level on both contexts. The report gives, per context: the curve; the first
count that is level; the first count inside round 2's 10-point pass band; a log-linear
estimate of where the curve crosses each; and the goal-hidden flag per point (hidden hits more
than 10 points over parametric's), which is reported and does not stop the scan. If 3,000 is
not level, the answer is "more than 3,000", any crossing beyond it is marked as an
extrapolation, and the detector branch is the next proposal.

**Gate:** round 2's arm unchanged (`expert_chunk` oracle, goal given, readout off, 500 fresh
layouts, seed 820, and the goal-hidden policy rows). Reference rows in the same sweep:
`oracle_task`, `oracle_ft`, `rel300`, `teach10k_rel`. The earlier parametric modules at other
counts (`checkpoints/adapters/data_efficiency/`) are not added: they were trained with
count-matched user modules, not the grid's, and T4 has none at 1,000.

One seed: the count found is an estimate from one draw. Seeds 1–2 at the matching count are a
follow-up on the user's go.

## Costs (estimates; the first runs replace them with measurements)

- Encoding: the round-2 trainer rendered and encoded 30–60 scenes per second. At that rate a
  3,000-demonstration run would spend 0.7–1.7 h encoding, so rendering is parallelised and the
  encoder batched before stage B; the target is 300 images per second or better.
- Training: 20,000 steps took 50–57 min with 256 tokens; 512 tokens and per-frame lookups
  should land at 1–1.5 h per run, one run per GPU (two for the 300-demo runs if the measured
  peak allows).
- Gate: about 30,000–60,000 images per method and context, minutes each (a batch runs until
  its slowest trial ends, so without skipping finished trials it would be up to 200,000).
- Stage A: nine runs, 3–5 h on the two GPUs. Stage B: four runs (two more if the chosen
  recipe is a combination or the rank changes), about 4 h. With the gates, 8–10 h in all.
- Build: comparable to round 2 (render, trainer data path, rollout observation, sweep, report).

## Where the code goes

- `src/lerobot/envs/cursor.py`: drawing the two views at given cursor positions (typed; the
  package is mypy-strict).
- `src/lerobot/policies/steering/` (`configuration_steering.py`, `modeling_steering.py`): the
  `vlm_views` source and its spec, token masking in `LearnedTaskModule.tokens`, the scene
  encoding of views and the token reduction, next to `scene_for` and `SceneEncoder`.
- `examples/pi05/mars/cursor/cursor_sa/`: per-frame scene lookup in `DemoSampler`; a scene
  provider called by `observation` / `SharedController` in rollouts.
- `examples/pi05/mars/cursor/train/train_adapter.py`: the flags, the frame table and the
  frozen affine draw.
- `examples/pi05/mars/cursor/run_sweep.py`, sweep configs for the selection and the scan,
  `report_grid_image.py` for the curve.
- Job lists and the pipeline stay in the git-ignored run folder
  (`examples/pi05/mars/results/research/image/`).
- The affine is the trainer's own bilinear resample (`move_pictures` in `train_adapter.py`, the
  uncovered border in the background's colour) at lerobot's default `RandomAffine` magnitudes
  (`src/lerobot/transforms/transforms.py`); nothing new goes into that package, whose
  `ImageTransforms` run inside `LeRobotDataset`, which the cursor demonstrations are not.

## Calls made here for the user to check

- "Level" means within 3 points on hits and collisions on both contexts.
- T2 and T4 at seed 0 only; the scan stops at the first level count.
- The recipe is chosen at 300 demonstrations among four variants, on separate layouts, by a
  3-point rule on mean hits.
- Frames every 2 planned steps, and training samples only at those steps; tokens reduced to
  256 dims, checked against full width once, with 512 and full width as the fallbacks.
- The affine is a frozen draw per stored picture with probability 0.5, at lerobot's default
  magnitudes; token masking is 30% and applies to the flow-matching loss only.
- The parametric module at 300 demonstrations is the only parametric reference row.
- The overhead view shows the cursor; the cursor's position stays in the module's query.

## Left to the builder

Flag and method names; the cursor mark's size and colour; what fills the border a moved
picture leaves (lerobot's default is black); how the sampler indexes frames and whether a
run's tokens live on the GPU or in RAM; how rendering is parallelised and the encoder batched;
the sample the reduction is fitted on; how the checkpoint carries the span and the reduction
so a rollout needs nothing else; the sweeps' layout (one per count or one for all); tests.
