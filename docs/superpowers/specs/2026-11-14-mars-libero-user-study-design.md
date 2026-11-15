# MARS on LIBERO, study 1: the user module picks the right object

**Date:** 2026-11-14. **Branch:** `main` (uncommitted). **Status:** approved to proceed on 2026-11-14 (build and stage 0;
each later stage on the author's go at its gate). Implementation plan:
`docs/superpowers/plans/2026-11-14-mars-libero-user-study.md`.

## Intent

The cursor grid shows that a user module calibrated from a few demonstrations undoes a user's readout where flow
control (FC) and Flow Reversal Steering (FRS) pass the corrupted command through. This study is the same claim on a
robot benchmark: pi0.5 on LIBERO is told only "do something", the operator's 3-D translation command says which
object or goal is meant, and the command reaches the policy through a corrupted readout. The claim to test:

> With a corrupted readout, FC and FRS take the wrong object or fail; MARS with a learned user module (MARS-U),
> calibrated from a few minutes of demonstration on other tasks, recovers most of what a clean readout achieves.

It is the robot analogue of the grid's open-field column (users x methods), and the cursor's "wrong target"
failure becomes a wrong object in the basket. The task module is out of scope here (study 2, held-out tasks).

## What is known

From the repository and earlier runs:

- **Harness.** `examples/pi05/mars/libero` runs policy, teleop, override, FC, FRS and MARS (fixed or learned user
  module) on any LIBERO suite, with a synthetic operator (the task-prompted policy's own plan as the command),
  matched resets across arms, per-trial records and paired tests. Since 2026-11-13 the shared modes run
  `SteeringPolicy` and the shared 3-D `Readout`, verified bit-identical to the old code on fifteen arms.
- **Checkpoint.** `lerobot/pi05_libero_finetuned` (the Hub now serves it as `lerobot/pi05_libero_finetuned_v044`):
  lerobot's reproduction of pi0.5 on LIBERO, Physical Intelligence's `pi05_libero` base fine-tuned for 6 000 more
  steps on `HuggingFaceVLA/libero`, which holds all four suites (Spatial, Object, Goal, Long). It is cached locally
  (7 GB). The same weights play both parts: the executing policy under "do something", and the synthetic operator
  under the task's instruction. It has seen every evaluation task with its instruction; that is what this study
  needs (a policy that can do each task once told which), and why study 2 needs a different one.
- **Earlier LIBERO-Goal runs (README study record).** A 20 degree rotation barely costs FC anything; the immediate
  oracle saturates (task 4: FC 95%, FRS 100%); the SpaceMouse-like timing profile (a new command every 20 steps,
  delivered 4 steps late) opens the gap (task 4: FC 20%, FRS 45%, MARS with a fixed adapter 60%). Tasks 0, 2, 3 and
  9 sit at the floor for every method: a translation-only command cannot express them.
- **LIBERO-Object scenes.** Ten tasks, "pick up the X and place it in the basket". Every scene holds the target, five
  other graspable objects and the basket, so "which object" is the operator's to say and a wrong object is
  observable (`mode_expressed`, the element that moved most, is already recorded per trial).
- **Cost per trial.** A steered rollout runs at about 0.08 s per control step on this machine (40 steps in 3.1 to
  3.5 s), so a full 280-step Object trial is under 25 s and a success ends sooner. Two 49 GB GPUs are free; the
  policy takes about 7 GB, so several runs fit per GPU.
- **Cursor grid users.** U1 and U2 a gain of 0.25 on one axis, U3 a 60 degree rotation, U4 a curl field (20 rad/s,
  time constant 0.1 s); user modules calibrated from 100 reaches in 500 gradient steps.

From today's smoke run of `train_user_module.py` (60 degree rotation, four Object scenes, 4 episodes, 300 steps):

- **It had never run.** It crashed at the first training step (the preprocessed batches hold fields that could
  not be stacked); fixed. The README's command also cannot start: it names `mars_learned_rotz20.yaml`, whose
  learned module is the file the command is meant to create (not fixed; item 1 below).
- **Cost.** Four episodes gave 59 chunk-boundary states in about 2.5 minutes including labelling; training ran at
  0.76 s per step at batch 8, so the README's 3 000 steps is about 40 minutes per user.
- **Signal.** On the six held-out states the regenerated chunk's heading cosine with the label was 0.25 to 0.63
  without the module and 0.75 to 0.95 with it across the six log points. Six states and fresh noise at every
  point: this shows the trainer learns something, not how good the module is.
- **The Object suite runs** in the harness, and the SpaceMouse attached through the shared reader.

Not known, and decisive:

1. What the policy does in an Object scene under "do something": which object it takes, and how often.
2. Whether a clean command steers it to the named object (the headroom every method shares).
3. Whether a learned user module of pi0.5's size (50 steps x 7 dims) trains to a useful correction from a few
   hundred states. The trainer had never run end to end before today.

## Design

**Setting.** pi0.5 with the prompt "do something"; the operator commands translation only (rotation and gripper
are the policy's); 10 actions executed per 50-step chunk; one command read per chunk.

**Suites.** LIBERO-Object is the main suite (all ten tasks). LIBERO-Goal's six expressible tasks (1, 4, 5, 6, 7, 8)
are the secondary suite, where earlier numbers exist.

**Users.** The four scenarios of the paper's Fig. 1a, as readouts of a 3-D translation command (x, y horizontal,
z vertical):

| User | Fig. 1a scenario        | Readout in 3-D                                                                                                                                                           | Spec                                        |
| ---- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| U0   | (reference)             | clean                                                                                                                                                                    | `null`                                      |
| U1   | decoding error          | a random 3-D rotation, fixed for the user: the axis drawn uniformly on the sphere, the angle 60 degrees, one draw from seed 0                                            | `{M: rows}` (the drawn matrix, written out) |
| U2   | low-DOF control         | a two-axis device: the vertical component is dropped, the reach projected onto the table plane                                                                           | `{scale: [1, 1, 0]}`                        |
| U3   | misperception and delay | a lag-and-turn field: the served command trails the push by about one chunk (0.5 s) and settles 45 degrees off in the horizontal plane at 0.7 of its size; z a plain lag | `{curl: 2, tau: 0.5}`                       |
| U4   | discrete input          | a six-key pad: the command snapped to the axis of its largest component, at full deflection, so a diagonal reach becomes a staircase                                     | `{quantize: axes}` (new; item 7 below)      |

U1's rotation is drawn once per user, not once per trial: a user module is calibrated against one readout, so a
rotation that changed every trial (the harness's existing `{random: ...}` family) would leave it nothing to learn.
The existing sampler (`RandomCorruption` with `rotation_deg: [60, 60]`) draws the matrices; they are written into
the condition files as literals with their axis and seed, so no new code is needed. A rotation leaves pushes along
its own axis untouched and turns pushes across it by the full 60 degrees (48 degrees on average over axes), so how
much a draw hurts depends on where its axis falls relative to the reach. The author chose one draw at 60 degrees
(2026-11-14); more draws would show the result does not hang on that axis. The angle is fixed rather than drawn from all rotations, most of which exceed 90 degrees
and reverse the command outright.

All four carry the same input noise, one level set in the preparation step before stage 1: the smallest of 0.1,
0.2, 0.3 and 0.5 at which clean FRS at depth 2 loses ten points or more on Object's development split (0.5 if none
does); the user modules train over a range around it. U2 and U4 have no inverse in command space (a dropped axis, a snap), and U3
is a dynamical system: these are the paper's argument for correcting in the flow. The strengths above are starting
points; stage 1 raises any that leaves the baselines unharmed. The cursor grid's users map onto these (its
rotation is U1 about the vertical, its curl is U3 at the cursor's time scale); its weak-axis gain, the human
study's user, has no panel in Fig. 1a and is left out unless wanted for continuity.

**The operator's true intent.** The scene's own task: on Object, which of the six objects goes in the basket.
It is carried by the synthetic operator, the same checkpoint prompted with the task's instruction ("pick up the
ketchup and place it in the basket"). At every update it plans a chunk from the current observation, and its
intended command `u` is that chunk's mean translation over the 10 steps about to be executed, clipped to [-1, 1]:
the direction a task-aware hand would move now. The readout turns `u` into what the controller gets; the
executing policy never sees the instruction. So the intent is exact and recorded at every step
(`user_translation_raw` beside the served `user_translation`), the calibration label is the operator's own chunk,
and a trial succeeds only if the benchmark's check for the intended task passes. The intent is a 3-D translation
only: orientation and the gripper are the policy's to supply.

**Operator timing.** The immediate operator (a new intent at every chunk, consumed at once) is the main condition,
by the author's decision after stage 0: under the SpaceMouse-like profile (`update_every_steps: 20`,
`delay_steps: 4`) every command consumed is 10 to 20 steps old and no method beats the floor on Object even with a
clean readout, so there would be nothing for a user module to recover. Delay stays in the study as the lag user; the
SpaceMouse-like profile is a robustness check on the final cells. The operator replans from the current state but
does not adapt to its readout, as the grid's scripted expert does not.

**The user module's input.** One served command per chunk, as on the cursor: no command history. An offline
analysis on stage 0's intent sequences (2026-11-14) found that the last four served commands would add information
for the lag and dropped-axis users and little for the others; the author chose to proceed without it.

**Arms.** Per user: FC, FRS, MARS-U (the learned user module) and override (the corrupted command pasted over
the policy's translation; the direct-control reference). Once per suite: the floor (policy, "do something"), the
ceiling (policy, the task's own instruction), and FC and FRS under U0, the reference for what a correction can
recover. There is no fixed-inverse arm.

**Calibration.** One user module per user, trained by `train_user_module.py` from the task-prompted policy's
rollouts through that user's readout. Proposed: calibrate on LIBERO-Spatial (20 episodes, two per task, about
three minutes of demonstration) and evaluate on Object and Goal, so that the module has never seen an evaluation
task: the modularity claim. Fallback if that does not transfer: calibrate on five Object tasks, evaluate on the
other five, and report both.

**Tuning and test split.** Every method's depth is tuned per user on a development split and tested on a disjoint
one, as on the grid: FRS and MARS over `n_reversal_steps` in {1, 2, 4, 6} (stage 0: depth 2 is best and deeper
reversals fall off); FC over the three best settings of its preparation sweep (`lambda` in {2, 4, 6, 8, 10} at
`tau` 10, and `ic_gain` in {2, 3, 5} at `lambda` 1: stage 0 showed that `tau` alone, at `lambda` 1 and the action
gain, does not steer), pooled over both suites; per suite. Development: each task's first 5 initial states (50 trials per cell on
Object, 30 on Goal-6). Test: its next 20 (200 and 120 trials per cell), same resets for every arm.

**Measures.** Success (the benchmark's own check); wrong-object rate (a trial whose most-moved element is an
object other than the target; on LIBERO-Object only, since LIBERO-Goal's tasks share the bowl and a wrong goal
with the right object would not count); time to success as success-over-time curves; measured user authority. Paired by
reset: McNemar's exact test for success and wrong object, Wilson intervals. Primary comparison per user: MARS-U
against the better of FC and FRS on Object's test split, pooled over tasks. With 200 pairs a difference of about
twelve points is detectable.

## Stages and gates

Each stage ends in a short report with numbers; the next starts on the author's go.

| Stage                  | What runs                                                                                                                                                                                                                                 | Trials                     | Gate to continue                                                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Anchors             | Object and Goal-6, 5 trials per task: floor, ceiling, FC and FRS under U0 at three depths, both operator profiles                                                                                                                         | about 1 100                | ceiling at least 90%; floor leaves room (wrong object common); clean steering reaches the named object in at least 60% of trials at some depth                |
| 1. Corrupted baselines | first a preparation step on the clean readout (FC's `lambda` and gain sweeps, the noise level, FRS at depths 1 and 4: about 1 100 trials); then FC, FRS and override under U0 to U4 at that noise on the development split of both suites | about 4 300                | for at least two users the best baseline loses 20 points or more to U0                                                                                        |
| 2. User modules        | train one module per user; open-loop check (heading cosine with and without the module); MARS-U on the development split                                                                                                                  | about 1 300, plus training | MARS-U beats the best baseline by 15 points or more for at least two users and recovers at least half of the gap to U0; at most two rounds of module settings |
| 3. Test                | settings frozen; every arm on the test split, Object (200 per cell) then Goal-6 (120); immediate-oracle check on Object's final cells                                                                                                     | about 7 600                | the result, whatever it is                                                                                                                                    |
| 4. Later (April)       | calibration-size curve (5, 20, 50 episodes); a human pilot on the SpaceMouse; more seeds                                                                                                                                                  |                            |                                                                                                                                                               |

Compute: about 12 500 trials at 15 to 30 s each is 50 to 100 GPU-hours: two to three days of wall time with three
runs on each of the two GPUs, plus about 40 minutes of training per user module (see the smoke run above).

If stage 0 fails on Object (the policy ignores the command, or always takes one object), the study moves to
Goal-6 alone, where steering is known to work, and Object becomes a note. If stage 1 shows no gap for a user, its
strength goes up before anything is trained (the lesson of the 20 degree runs).

## What has to be built

Small, all in the LIBERO example and its tests:

1. Trainer: the documented command names an arm file whose learned module does not exist yet, so the session
   refuses to start; the trainer should build its session without the arm's adapter.
2. Trainer: serve calibration commands by the rule the rollouts use. Today it noises a command whenever the raw
   push is active; the rollouts only when the corrupted command is.
3. Experiment runner: a first-occurrence offset per task, so development and test splits take disjoint initial
   states under one seed.
4. Condition files for Object and the users (a base per user, since `--set` merges into a corruption mapping).
5. `notebooks/analyze.py`: the wrong-object measure (the target from the ceiling's successes, as `task_targets`
   does), and the users x methods figure.
6. A study script in `experiments/` (users x arms x depths, several runs per GPU), in the form of
   `oracle_sweep.sh`.
7. The discrete-input readout: a `quantize` corruption in the shared `Readout` (the command snapped to the axis
   of its largest component, in any number of dims), with its spec; and a check that a singular matrix (U2) is
   accepted end to end, including by the automatic input gain.

One behaviour test per new piece, per the lean-test rule. Item 7 is the only change to shared infrastructure.

## Risks

- **Grasping, not steering, may decide success.** Steering matters in the reach. Wrong-object rate and time are
  co-primary so that a correct reach followed by a failed grasp still counts for the method.
- **The policy's own preference.** Under a generic prompt pi0.5 may favour the nearest or a familiar object
  strongly; a shallow reversal may not overcome it and a deep one may lose the policy's skill. Stage 0 measures
  this before anything else is spent.
- **The learned user module on pi0.5 is new.** On the cursor the rotation user needed more than 100 reaches
  (MARS-U trailed teleoperation by 22 points). Stage 2 has the clean readout as the reference and a bounded number
  of rounds. U2 and U4 lose information outright: there the module can only learn when to defer to the policy.
- **A synthetic operator does not adapt.** A person partly compensates for a rotation or a gain. The numbers are
  an upper bound on what the correction is worth; the human pilot (stage 4) is the check.
- **Trial counts.** 200 per cell against the grid's 500; small differences between baselines will not resolve.

## Decisions for the author

Settled on 2026-11-14 after stage 0: the immediate operator is the main condition; no command history in the user
module; no goal-aware wrong-target measure (wrong object is reported on LIBERO-Object only); stage 1 to proceed.

1. **Deadline.** Stages 0 to 3 as written are about two weeks including compute. For the 4 December abstract, a
   reduced stage 3 (Object only, 100 test trials per cell) fits; the full version is the April paper's.
2. **Users.** Settled 2026-11-14: the four Fig. 1a readouts as specified, U1 one random rotation at 60 degrees.
3. **Calibration.** On LIBERO-Spatial (the transfer claim), taken as the default when the author said to proceed;
   the held-out-Object fallback stays as written.
4. **Human pilot.** Left for April by the same default.
