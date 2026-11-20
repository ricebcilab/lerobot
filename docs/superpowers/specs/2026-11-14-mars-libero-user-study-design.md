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

| User | Fig. 1a scenario        | Readout in 3-D                                                                                                                                                                            | Spec                                        |
| ---- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| U0   | (reference)             | clean                                                                                                                                                                                     | `null`                                      |
| U1   | decoding error          | a random 3-D rotation, fixed for the user: the axis drawn uniformly on the sphere, the angle 60 degrees, one draw from seed 0                                                             | `{M: rows}` (the drawn matrix, written out) |
| U2   | low-DOF control         | a two-axis device: the vertical component is dropped, the reach projected onto the table plane                                                                                            | `{scale: [1, 1, 0]}`                        |
| U3   | misperception and delay | a lag-and-turn field: the served command trails the push by about one chunk (0.5 s) and settles 45 degrees off in the horizontal plane at 0.7 of its size; z a plain lag                  | `{curl: 2, tau: 0.5}`                       |
| U4   | discrete input          | a six-key pad: the command snapped to the axis of its largest component, its length kept (the author, 2026-11-16: a user only loses information), so a diagonal reach becomes a staircase | `{quantize: axes}` (new; item 7 below)      |

U1's rotation is drawn once per user, not once per trial: a user module is calibrated against one readout, so a
rotation that changed every trial (the harness's existing `{random: ...}` family) would leave it nothing to learn.
The existing sampler (`RandomCorruption` with `rotation_deg: [60, 60]`) draws the matrices; they are written into
the condition files as literals with their axis and seed, so no new code is needed. A rotation leaves pushes along
its own axis untouched and turns pushes across it by the full 60 degrees (48 degrees on average over axes), so how
much a draw hurts depends on where its axis falls relative to the reach. The author chose one draw at 60 degrees
(2026-11-14); more draws would show the result does not hang on that axis. The angle is fixed rather than drawn from all rotations, most of which exceed 90 degrees
and reverse the command outright.

All four, and the clean readout, carry translation noise 0.5 (the author, 2026-11-15; the cursor study's conditions run
at 1.0). An earlier rule of mine set it to 0.2 from how much noise clean FRS tolerates; the author rejected choosing
a study condition from a baseline's score, and that stage 1 (at 0.2) was discarded. The user modules train over a
range around 0.5. U2 and U4 have no inverse in command space (a dropped axis, a snap), and U3
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

**Arms.** Per user: FC, FRS, MARS-U (the learned user module) and teleop (the operator's corrupted command
alone moves the arm, rotation and gripper included; the direct-control reference, named as on the cursor and run
by the harness's mode `teleop`). Once per suite: the floor (policy, "do something"), the
ceiling (policy, the task's own instruction), and FC and FRS under U0, the reference for what a correction can
recover. There is no fixed-inverse arm.

**Calibration.** One user module per user, trained by `train_user_module.py` from the task-prompted policy's
rollouts through that user's readout. Proposed: calibrate on LIBERO-Spatial (20 episodes, two per task, about
three minutes of demonstration) and evaluate on Object and Goal, so that the module has never seen an evaluation
task: the modularity claim. Fallback if that does not transfer: calibrate on five Object tasks, evaluate on the
other five, and report both.

Stage 2 as run (the author's go, 2026-11-17): round 1 takes the trainer's defaults (every chunk step's matrix
free, no cross-step coupling, 3 000 steps at batch 8, command noise drawn from 0 to 1); the two episodes of a task
start from its initial states 0 and 1; the open-loop check is the heading cosine with and without the module over
every held-out state at noise 0.5. MARS then runs on the development split of both suites over the reversal depth
(1, 2, 4, 6, 10 of the policy's 10 steps) x the pinned prefix (5, 10, 25, 50): 20 settings per user and suite,
6 400 trials, Goal-6 first since stage 1 found the gap to close there (on Object no baseline beats the floor even
under the clean readout).

Round 2 (the author's go, 2026-11-18; Goal-6 only, on the prompt-dropout checkpoint
`outputs/pi05_libero_generic_prompt/checkpoints/006000`). Round 1's modules discarded the command: a task-prompted
rollout only visits states where the policy's own plan is already the label. Round 2 calibrates in the LIBERO-Goal
scene on the four tasks outside Goal-6 (0, 2, 3, 9; five start states each) from 20 task-prompted episodes and 20
under "do something" (their first 120 steps), every state labelled with the scene's task
(`user_study.sh modules`). Outcome: the modules react to the command but do not repair it. On the executed steps of
the chunk the steering computes, where the policy's plan disagrees with the operator's push, MARS-U is level with
FRS without a module at the full reversal and at depth 4 for all four users (differences -0.03 to +0.02 in cosine
with the push); at the full reversal a module only makes the motion follow the plan more, and at depth 4 it changes
nothing. The MARS grid was not run with them. Two facts from the same day bound what a user module can show on
this checkpoint at noise 0.5: a module that restored the clean readout exactly would gain about 12, 8, 18 and 1
points over full-reversal FRS (rotation, low-DOF, lag, keypad), so stage 2's gate of 15 points over the best
baseline cannot be met by repair alone; and the served command's cosine with the operator's push is 0.56 under the
clean readout against 0.29 to 0.49 under the users, so the noise is the larger part of every user's error. The
whole-chunk heading cosine of the trainer's log does not test a module (it rewards weakening the steering);
proposed in its place, not yet agreed: the executed steps' direction against the push, split by whether the plan
from the same noise agrees with it, with FRS on the clean readout as the target. An FRS control at MARS's depths
(the same loop without a module, `CONTROL=1`) is set up and not yet run; a MARS result at a partial depth is read
against it. Numbers: the ledger (`.superpowers/sdd/2026-11-14-mars-libero-user-study/progress.md`, 2026-11-18)
and `outputs/pi05_libero_user_study/generic_prompt/module_check/`.

Round 3 (the author's go, 2026-11-18 evening: the trainer change, a pilot on rotation, the remaining modules; FRS
stays at the full reversal, only MARS tunes its depth). The label changes from the latent of the operator's whole
chunk to where steering takes a faithful command: the plain reversal of the reference FRS builds from the
operator's push (`train_user_module.py --target command`); the regression runs at the reversal's own steps on the
translation dims, and the correction is complete by reversal step 4 (`--correct-by 4`), since a correction spread
over the whole reversal gains nothing at the depths where steering is strong (pilot). Calibration noise fixed at
0.5. On all calibration states at depth 4, where the policy's plan disagrees with the push, the executed steps'
cosine with the push is 0.52 / 0.38 / 0.39 / 0.50 with the module (rotation, low-DOF, lag, keypad) against 0.34 /
0.26 / 0.29 / 0.38 for FRS at that depth, 0.46 for FRS on the clean readout, and about 0 for full-reversal FRS.
A rotation module trained without noise undoes the rotation fully and then only reaches the clean readout; at
noise 0.5 the modules gain mostly by handling noise, and repair nothing for lag, low-DOF and keypad. Not yet run:
closed loop. Numbers: the ledger and `outputs/pi05_libero_user_study/generic_prompt/user_modules_r3/check/`.

**Tuning and test split.** As on the cursor grid, the baselines are their papers' methods, each with its one knob
tuned per user on a development split and tested on a disjoint one (revised 2026-11-15 after checking both papers;
my first plan tuned FRS's reversal depth and FC's `lambda`, knobs the papers do not have, and its results were
discarded).

- **FRS** (Tang et al., arXiv 2606.13675): the full reversal, 10 Euler steps each way; tuned over the pinned prefix,
  `pinned_steps` in {5, 10, 25, 50} of the 50-step chunk (10 are executed). 5 is the ratio of their real-robot
  keyboard interface (App. D.1: a 5-step reference in a 15-step chunk, 10 executed); in their own LIBERO runs the
  reference covers the whole 10-step chunk, which our 10 (every executed step) and 50 (the whole chunk) each match
  in one sense. The paper fixes this value; tuning it is the cursor study's choice.
- **FC** (Kao et al., arXiv 2606.10180): the command written into the initial condition only (`lambda` 1) of the
  first `tau` chunk steps, rotation set to 0 there (`zeroed_dims: [rotation]`, their LIBERO recipe, App. G.1);
  tuned over `tau` in {5, 10, 20, 30, 40, 50}. The gain is chosen once, as the cursor's was: on the clean readout at
  noise 0.5, gains 1.5 (their LIBERO value for a unit push), 3, 5, 8, 12 at `tau` 10 and 50; the gain with the
  highest success pooled over both suites at its better `tau` is taken, the smaller on a tie (the author,
  2026-11-15: stage 0 showed the paper's recipe near the floor here). **Result (2026-11-15, 800 trials): gain 3**,
  41 of 80 at `tau` 50, against 38 (gain 5, `tau` 50), 36 (1.5, `tau` 10), 35 (8, `tau` 10) and 29 (12, `tau` 10);
  the floor is 34 of 80. No gain steers well: Object stays at its floor (62%) up to gain 3 and falls above it (14%
  at gain 12 over the whole chunk), Goal-6 rises from 10% to 30% at gain 3 and 40% at gain 12.
- **MARS** tunes the reversal depth x the pinned prefix, as on the grid (the grid for LIBERO is set at stage 2).
- **Teleop** has no knob. It is the operator alone, as on the cursor (the author, 2026-11-16: full control,
  rotation and gripper included): the synthetic operator's translation through the user's readout, and its
  rotation and gripper (the mean planned turn of the executed steps; close iff the plan mostly closes); no policy
  acts. A user's corruption acts on the translation only; its noise reaches all three parts (the author,
  2026-11-16): 0.5 on the translation, 0.04 per axis on the rotation (the same ratio to the operator's commands,
  whose per-axis spread is 0.39 in translation and 0.032 in rotation) and 1.0 on the gripper, served as the sign
  of the noisy command (wrong on 16% of reads). Teleop reads the command at every control step, so its noise is
  redrawn 10 times a chunk, as on the cursor; the steering arms act on one read per chunk. The first stage 1 ran shared override in this
  place (the command over the policy's translation, the policy's own rotation and gripper); those runs are set
  aside.

Kept differences from the papers' own LIBERO runs, all agreed with the author on 2026-11-15: the checkpoint
(lerobot's conversion of `pi05_libero`, fine-tuned 6 000 steps, 50-step chunks and mean/std action statistics;
theirs is openpi's with 10-step chunks and quantile statistics); in FRS, rotation and gripper noise resampled after
the reversal instead of kept (MARS runs the same code, so the two differ only by the user module); in FC, the
command's own length written instead of a unit direction, and replanning every 10 steps instead of 5; the setting
(a policy that knows the tasks and is not told which, where the papers' policies are told the task and are weak at
it). No user involves the gripper: every arm leaves it to the policy.

Development: each task's first 5 initial states (50 trials per cell on
Object, 30 on Goal-6). Test: its next 20 (200 and 120 trials per cell), same resets for every arm.

**Measures.** Success (the benchmark's own check); wrong-object rate (a trial whose most-moved element is an
object other than the target; on LIBERO-Object only, since LIBERO-Goal's tasks share the bowl and a wrong goal
with the right object would not count); time to success as success-over-time curves; measured user authority. Paired by
reset: McNemar's exact test for success and wrong object, Wilson intervals. Primary comparison per user: MARS-U
against the better of FC and FRS on Object's test split, pooled over tasks. With 200 pairs a difference of about
twelve points is detectable.

## Stages and gates

Each stage ends in a short report with numbers; the next starts on the author's go.

| Stage                  | What runs                                                                                                                                                                                                 | Trials                     | Gate to continue                                                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Anchors             | Object and Goal-6, 5 trials per task: floor, ceiling, FC and FRS under U0 at three depths, both operator profiles                                                                                         | about 1 100                | ceiling at least 90%; floor leaves room (wrong object common); clean steering reaches the named object in at least 60% of trials at some depth                |
| 1. Corrupted baselines | first FC's gain on the clean readout (5 gains x 2 `tau`: about 800 trials); then FRS over the pinned prefix, FC over `tau` and teleop under U0 to U4 at noise 0.5 on the development split of both suites | about 5 200                | for at least two users the best baseline loses 20 points or more to U0                                                                                        |
| 2. User modules        | train one module per user; open-loop check (heading cosine with and without the module); MARS-U on the development split, depth x pinned prefix                                                           | about 6 400, plus training | MARS-U beats the best baseline by 15 points or more for at least two users and recovers at least half of the gap to U0; at most two rounds of module settings |
| 3. Test                | settings frozen; every arm on the test split, Object (200 per cell) then Goal-6 (120); immediate-oracle check on Object's final cells                                                                     | about 7 600                | the result, whatever it is                                                                                                                                    |
| 4. Later (April)       | calibration-size curve (5, 20, 50 episodes); a human pilot on the SpaceMouse; more seeds                                                                                                                  |                            |                                                                                                                                                               |

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
   of its largest component at its own length, in any number of dims), with its spec; and a check that a singular matrix (U2) is
   accepted end to end, including by the automatic translation gain.

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
