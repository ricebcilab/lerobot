# MARS on LIBERO, study 1: a learned user module undoes the user's readout

**Date:** 2026-11-14, rewritten 2026-11-20 for round 1. **Branch:** `main`. **Status:** round 1 started on
2026-11-20 (the author's go); every later stage starts on the author's go at its gate. The earlier iterations are
condensed in "Before round 1" below; their outputs were deleted on the author's word.

## Intent

The cursor grid shows that a user module calibrated from a few demonstrations undoes a user's readout where flow
control (FC) and Flow Reversal Steering (FRS) pass the corrupted command through. This study makes the same claim on
a robot benchmark. pi0.5 on LIBERO is told only "do something", the operator's 3-D translation command says which
goal is meant, and the command reaches the policy through a corrupted, noisy readout. The claim to test:

> With a corrupted readout, FC and FRS fail more often; MARS with a learned user module (MARS-U), calibrated from a
> few minutes of demonstration, recovers most of what a clean readout achieves.

It is the robot analogue of the grid's open-field column (users x methods). The task module is out of scope here
(study 2, held-out tasks).

## Round 1 design

**Policy.** The prompt-dropout fine-tune `outputs/pi05_libero_generic_prompt/checkpoints/006000/pretrained_model`
(`examples/pi05/mars/libero/train_generic_prompt.py`: lerobot-train from `lerobot/pi05_libero_finetuned` on all four
suites of `HuggingFaceVLA/libero`, the instruction replaced by "do something" with probability 0.5, action expert
only, 6 000 steps). Under "do something" it spreads over the scene's goals where the released checkpoint picked one,
and it still follows every task sentence. The same weights play the executing policy ("do something") and the
synthetic operator (the task's instruction). 10 actions are executed per 50-step chunk, one command read per chunk.

**Suite.** LIBERO-Goal's six tasks a translation command can express (1, 4, 5, 6, 7, 8): Goal-6. One scene holds
several goals the policy can do, so the command has something to choose. Development split: the first 5 of each
task's seeded permutation of its initial states (30 trials per cell); test split: the next 20 (120 per cell,
`experiment.first_occurrence: 5`); the same resets for every arm.

**Users.** The four scenarios of the paper's Fig. 1a, as readouts of a 3-D translation command (x, y horizontal,
z vertical):

| User | Fig. 1a scenario        | Readout in 3-D                                                                                                                                 | Spec                  |
| ---- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| U0   | (reference)             | clean                                                                                                                                          | `null`                |
| U1   | decoding error          | one random 3-D rotation at 60 degrees, fixed for the user (seed 0; its axis 16 degrees off vertical)                                           | `{M: rows}`           |
| U2   | low-DOF control         | a two-axis device: the vertical component dropped                                                                                              | `{scale: [1, 1, 0]}`  |
| U3   | misperception and delay | a lag-and-turn field: the served command trails the push by about one chunk and settles 45 degrees off in the horizontal plane at 0.7 its size | `{curl: 2, tau: 0.5}` |
| U4   | discrete input          | a six-key pad: the command snapped to the axis of its largest component, its length kept                                                       | `{quantize: axes}`    |

U2 and U4 have no inverse in command space and U3 is a dynamical system, which is the paper's argument for
correcting inside the flow.

**Noise.** Gaussian noise at every read, a rest included (a decoder's noise does not know the user rests; the
shared `lerobot.teleoperators.readout` has no other path since 2026-11-20). Translation 0.2 per axis, rotation 0.016,
gripper 0.4 (the author, 2026-11-20); the corruptions act on the translation only, and only teleop executes the
rotation and the gripper. The level comes from the readout alone, with no arm run (see "Before round 1"): at 0.2
the clean readout's served command agrees with the push at cosine 0.79 against 0.40 to 0.68 under the four users,
the room a perfect user module could recover is near its largest, and one read tells a rest from a push 78% of the
time. Rotation and gripper noise keep the ratios to translation noise set on 2026-11-16.

**The operator.** The intent is the scene's task. At every chunk the synthetic operator plans a chunk under the
task's instruction from the current observation, and its push is that chunk's mean translation over the 10 steps
about to execute, clipped to [-1, 1]; the intent is recorded at every step beside the served command. It is
immediate (a new intent at every chunk, consumed at once); the SpaceMouse-like timing is a robustness check. Under
a steering arm (FC, FRS, MARS) it lets go: when the robot's mean executed translation over the last chunk heads
with its new intent at cosine 0.5 or more, it rests (zero translation and rotation) for the next chunk, and pushes
again when the robot goes off course (`policy_operator.let_go_cosine: 0.5`; each step records `operator_resting`).
It judges what it sees, not the policy's plan. Under teleop it never lets go, since nothing else moves the robot.
A rest reaches the steering as noise. No method is told when the user rests (that flag exists only in the
simulator), so recognizing a rest is part of what a user module can learn.

**Arms.** Per user: FC, FRS, MARS-U and teleop. Once: the floor (the policy under "do something") and the ceiling
(the policy under the task's instruction); neither involves the user.

- **FRS** (Tang et al., arXiv 2606.13675): the full reversal, 10 Euler steps each way, rotation and gripper handed
  back to the policy; tuned over the pinned prefix in {5, 10, 25, 50} of the 50-step chunk. The paper fixes this
  value; tuning it is the cursor study's choice. Only MARS tunes the reversal depth.
- **FC** (Kao et al., arXiv 2606.10180): `lambda` 1, the command's own length written into the first `tau` chunk
  steps of the initial noise, rotation set to 0 there (their LIBERO recipe, App. G.1); tuned over `tau` in {5, 10,
  20, 30, 40, 50}. The gain is chosen once on the study condition: clean readout, gains 1.5 (their LIBERO value), 3,
  5, 8, 12 at `tau` 10 and 50, the gain with the highest success at its better `tau`, the smaller on a tie; if 12
  wins, the sweep's edge is reported before going on.
- **MARS-U**: FRS with the user's learned module in the backward pass; tuned over reversal depth x pinned prefix.
- **Teleop**: the operator alone, translation, rotation and gripper through the user's readout, read at every
  control step; no knob.

Kept differences from the papers' own LIBERO runs (agreed 2026-11-15): lerobot's conversion of `pi05_libero`
(50-step chunks, mean/std action statistics) where both papers ran openpi's (10-step chunks, quantile statistics);
FRS resamples rotation and gripper noise after the reversal; FC writes the command's own length and replans every
10 steps; the policy knows the tasks and is not told which.

**User modules.** The cursor's loss: the label is the latent of what the operator means (the operator's chunk
reversed through the goal-free field), regressed along the straight path from the served command. The calibration
watches the policy under "do something", unsteered, while the synthetic operator pushes or rests by the let-go
rule (the author, 2026-11-20: open loop). So a pushed state is one where the robot heads wrong and the command
carries what the plan lacks, and a rest, served as noise, teaches the module to leave the policy alone. A closed
loop under FRS was considered and dropped: at the full reversal FRS barely moves the robot off its plan, so its
states would look much like these, and these do not depend on the user. Every module is checked before any closed
loop on the executed steps of the chunk the steering computes, split by whether the policy's plan agrees with the
push, with FRS on the clean readout as the target, and on the executed length.

Recipe (built on a copy, installed at stage 1's gate on the author's go): per user, 20 rollouts in the LIBERO-Goal
scene on the four tasks outside Goal-6 (0, 2, 3, 9; start states 0 to 4) under "do something", the operator
watching; every chunk boundary a state, labelled with the chunk the operator planned there (about 500 states, a
fifth held out); the user's readout with noise 0.2 drawn once per state (the cursor's frozen noise). The cursor's
loss over the 7 action dims, rests included, 3 000 steps at batch 8, the module's default ranks. About an hour for
the four modules, two per GPU. Pass rule before any closed loop, on the held-out states at the full reversal: where
the plan disagrees with the push, MARS-U closes at least half the gap from FRS on the user's command to FRS on the
clean readout; its executed length is not more than 10% below FRS's in any group; and where the operator rests, the
executed motion heads with what it means at least as well as FRS's on the same noise.

**Measures.** Success (the benchmark's check), time to success, the robot's executed speed, the share of chunks in
which the user rests. Paired by reset: McNemar's exact test, Wilson intervals. Primary comparison per user: MARS-U
against the better of FC and FRS on the test split.

## Stages and gates

| Stage                  | What runs                                                                           | Trials (runs)   | Gate                                                                           |
| ---------------------- | ----------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------ |
| 1a. FC gain            | clean readout, 5 gains x `tau` 10 and 50                                            | 300 (10)        | the rule above; stop if 12 wins                                                |
| 1b. Baselines, anchors | FC over `tau`, FRS over the prefix and teleop under U0 to U4; floor and ceiling     | 1 710 (57)      | report; for at least two users the best baseline loses 20 points or more to U0 |
| 2. User modules        | one module per user, calibrated in closed loop; the executed-steps and length check | training        | the author's go on the recipe and on the check                                 |
| 3. MARS-U              | depth x pinned prefix on the development split                                      | set at the gate | set against stage 1's headroom                                                 |
| 4. Test                | settings frozen, every arm on the test split                                        | 120 per cell    | the result, whatever it is                                                     |

Outputs go to `outputs/pi05_libero_user_study/round1/` (`gain`, `stage1`, ...), launched by
`examples/pi05/mars/libero/experiments/user_study.sh` with `SUITES=goal TRANSLATION_NOISE=0.2 ROTATION_NOISE=0.016
GRIPPER_NOISE=0.4` and the checkpoint's `--set policy.path=...`. Every study run is headless (`server.headless`: no live view, no browser tab; videos are kept). Six trials run at a time (three per GPU), about
10 trials a minute.

## Round 1 results

**Stage 1a (2026-11-20).** FC's gain on the clean readout, success of 30 at `tau` 10 / 50: gain 1.5 7 / 17, 3 18 / 10,
5 16 / 15, 8 6 / 7, 12 8 / 7. The rule takes gain 3.

**Stage 1b (2026-11-20; 57 runs, 1 710 trials).** Success (%), clean / rotation / low-DOF / lag / keypad, pooled over
each arm's knob: FC 41.7 / 29.4 / 40.0 / 21.7 / 32.2, FRS 24.2 / 21.7 / 17.5 / 12.5 / 21.7, teleop 96.7 / 0 / 0 / 6.7
/ 36.7; floor 6.7, ceiling 100. Best cells: FC 60 / 40 / 56.7 / 33.3 / 36.7, FRS 36.7 / 26.7 / 23.3 / 16.7 / 23.3.
Paired against the clean readout, FC loses 12.2 (p 0.004), 1.7 (0.76), 20.0 (< 0.0001) and 9.4 (0.014) points and
FRS 2.5 (0.63), 6.7 (0.08), 11.7 (0.0005) and 2.5 (0.61). The operator rests in 42 to 50% of chunks under FC and
FRS. FRS moves the robot slower than the policy alone (0.39 to 0.45 against 0.48 executed translation per step),
FC faster (0.77 to 0.87, the ceiling 0.77). The gate passes for every user, set by teleop on the clean readout.
FRS on the clean readout reaches only 24%, so a module that only restored the clean readout would not reach FC;
MARS-U's label aims at the intended chunk itself, whose ceiling is near 100.

**Stage 2, user modules (2026-11-20).** Calibrated by the recipe above (580 states from 20 watched rollouts, 263
of them rests; one states file for the four users). On held-out states at the full reversal, where the plan
disagrees with the push, the executed motion's cosine with the push for FRS / MARS-U / FRS on the clean readout / FRS
on the exact push: low-DOF +0.02 / +0.12 / +0.18 / +0.25, lag +0.08 / +0.17 / +0.18 / +0.25, keypad +0.08 / +0.19 /
+0.18 / +0.25, rotation -0.03 / -0.01 / +0.01 / +0.14 (rotation's held-out split differs). Where the operator rests,
MARS-U heads with what it means at 0.51 against FRS's 0.43 (the plan 0.65), and no module shortens the motion
(1.07 to 1.20 times FRS's length). So the modules undo most of what the readout costs FRS, except for rotation, and
hand back part of the way at rests; they do not take the full reversal beyond what FRS reaches with an exact push.
At depth 4 they repair 11 to 31% of the gap, and there rest noise drives every arm off a correct course (0.15
against the plan's 0.65). All four pass the stated rule at the full reversal (rotation's on a gap of 0.04). The
MARS-U grid runs depths 10, 8 and 6 (the author's go; depths 4 and 2 left out on this check).

**Stage 3, MARS-U on the development split (2026-11-21; 48 runs, 1 440 trials, headless).** Success (%), pooled
over the pinned prefix, at depth 10 / 8 / 6: rotation 24.2 / 31.7 / 21.7, low-DOF 26.7 / 30.0 / 24.2, lag 13.3 /
17.5 / 14.2, keypad 16.7 / 25.0 / 26.7. Paired by reset and prefix against full-reversal FRS (21.7, 17.5, 12.5,
21.7): at depth 8 rotation +10.0 (p 0.004), low-DOF +12.5 (p 0.0007), lag +5.0 (p 0.21), keypad +3.3 (p 0.56); at
the full reversal, where only the module differs from FRS, low-DOF +9.2 (p 0.027) and the others within 5 points.
FC, pooled, stays ahead under every user but rotation (29.4, 40.0, 21.7, 32.2), and the best MARS-U cell (all at
depth 8: 40.0, 40.0, 23.3, 33.3) does not beat FC's best (40.0, 56.7, 33.3, 36.7). So MARS-U lifts the method it
builds on, above FRS on the clean readout for rotation and low-DOF, and the gap to FC remains; how much of the
depth-8 gain is the module's and how much the shallower reversal's needs FRS at depth 8 without a module.

**What limits the user module (2026-11-21; open-loop checks on held-out calibration states).** The measure is the
executed motion's cosine with what the operator means, on states where the policy's plan disagrees with it.

- Capacity and data are not the limit. A module of 22 000 parameters (translation only, width 64) scores the same
  as the 800 000-parameter one, and so does one fitted on 1 648 states from 60 rollouts in place of 580 (within 0.03
  for every user). The larger module's lower training loss was memorization.
- The reversal's depth is. FRS with the meant chunk itself as its reference and no module is where a perfect
  module would land. Plan disagrees / rests (the plan 0.77): depth 10 +0.22 / 0.87, depth 8 +0.27 / 0.88, depth 4
  +0.84 / 0.97. At depths 8 and 10 nothing a module does can turn the robot far, because the reversal keeps only
  the executed steps' translation and little of it survives.
- Depth-matched training at depth 4 (`--depth 4`, the label is the meant chunk reversed 4 steps) does not reach
  that ceiling. Depth 4, plan disagrees, FRS on the served command / small module (width 64) / width 256 / width
  256 with an additive term (`--additive`) / FRS on the clean readout / FRS on the exact push: rotation +0.45 /
  +0.59 / +0.48 / +0.41 / +0.60 / +0.77; low-DOF +0.33 / +0.32 / +0.26 / +0.18; lag +0.45 / +0.45 / +0.37 / +0.28;
  keypad +0.50 / +0.50 / +0.43 / +0.31 (the four share the clean and exact-push values). The training
  loss falls with capacity (2.2, 1.9, 1.4) while the held-out score falls too, so the extra fit is memorization.
  The module sees the noisy latent and the time, not the scene, and the meant chunk depends on the scene. At rests
  every module stays near FRS (0.40 to 0.48 against the plan's 0.66) and halves the motion (0.23 against 0.44).
  No depth-4 closed loop was run.
- The policy commits. Under "do something" it chooses its goal at the first chunk (8 samples spread, mean pairwise
  cosine 0.68, 70% follow the meant task) and from the second chunk on every sample heads one way (0.95 to 0.98).
  Told the new task in words after 40 or 80 steps on another task's path, it completes it in 30% and 23% of
  episodes (`round1/policy_switch_probe.json`). A state on the way to one goal occurs only in that goal's
  demonstrations, so neither a command nor a sentence finds a second goal there.

**Redirect data (2026-11-21 and 22, `make_redirect_data.py`, `outputs/pi05_libero_redirect/shard_0` to `shard_5`).**
The policy under task A's instruction for 10 to 80 steps, then under task B's, from 8 start states per task the
study does not use; the segment after the switch is kept, labelled with B's instruction, when B's check passes. Of
1 440 attempts (30 ordered pairs, 6 switch steps, 8 starts) 728 were kept (51%; 73 743 frames, mean segment 101
steps). By switch step 10 / 20 / 30 / 40 / 60 / 80 it kept 96 / 72 / 38 / 36 / 34 / 27%. By meant task 1 / 4 / 5 /
6 / 7 / 8 it kept 142 / 137 / 154 / 98 / 60 / 137 episodes of 240. The bowl tasks redirect among themselves in 39
to 48 of 48 attempts; into "turn on the stove" only 8 to 16 of 48 succeed. The training set holds 266 episodes
(27 638 frames) of these six tasks among 1 693 (273 465 frames), so merged the redirect data is 21% of the frames.

**Redirect fine-tune (2026-11-22 and 23, the author's go; `outputs/pi05_libero_redirect/finetune.sh`).** From
checkpoint 006000 on the merged set (`outputs/pi05_libero_redirect/merged`, 2 421 episodes, 347 208 frames, the
original's statistics so the action units stay), the first fine-tune's recipe (prompt dropout 0.5, action expert
only, batch 32 on each of 2 GPUs, 6 000 steps, peak learning rate 2.5e-5), into
`outputs/pi05_libero_redirect_finetune` (7 h 37 min). Three things found on the way.

- The library merge loses data. HuggingFaceVLA/libero's episode index names 69 data files and its data folder holds
  377, and `merge_datasets` reads the files the index names, so it wrote an index for 2 421 episodes over 121 112 of
  347 208 frames. The merge is done by hand (`merge.py`) and checked frame by frame (`verify_merged.py`).
- Checkpoint 006000's loss on the redirect episodes is 0.52 against 0.03 on the original ones (task sentence).
  The conventions are the same (position change per unit action, lag, gripper sign, image orientation). The
  policy's own samples from a redirect state differ from each other as much as they differ from the label (0.38
  and 0.37 action standard deviations, against 0.11 and 0.12 on the original paths), so the labels are faithful
  samples of a policy that is undecided off the demonstrated paths, and a fifth of the training frames carry that
  spread.
- A redirect episode ends at the instant of success, in mid-motion, and a chunk's padded tail repeats the last
  action. The first launch was stopped at step 393 and the run made with `--policy.mask_padded_actions=true`, the
  one change to the recipe (redirect loss by phase 0.37 / 0.45 / 0.32 with the mask against 0.37 / 0.50 / 0.66).

Training loss went from 0.159 over the first 200 steps to 0.133 at step 6 000.

## Round 2 results

Round 2 (2026-11-23) is round 1's condition on the redirect fine-tune's checkpoint 006000, run to answer the
author's question, whether FRS on the clean readout without the readout's noise exceeds FC on the corrupted
commands. Runs and reports are in `outputs/pi05_libero_user_study/round2/` (`eval.sh`, `gain_report.txt`,
`compare_report.txt`, `noise_check.txt`); 30 trials a cell, one seed.

- Anchors. Ceiling 30 of 30, floor 4 of 30 (round 1: 30 and 2), so step 6 000 is kept.
- FC's gain on the clean readout, success of 30 at `tau` 10 / 50: gain 1.5 16 / 17, 3 17 / 14, 5 11 / 12, 8 5 / 5,
  12 7 / 2. Gains 1.5 and 3 tie and the rule takes the smaller, 1.5 (round 1 took 3).
- Success (%), pooled over each arm's knob, with the best cell in brackets.

  | Arm                           | Round 1's checkpoint (FC gain 3) | Redirect fine-tune (FC gain 1.5) |
  | ----------------------------- | -------------------------------- | -------------------------------- |
  | FRS, clean readout, no noise  | 24.2 (30.0)                      | 21.7 (26.7)                      |
  | FRS, clean readout, noise 0.2 | 24.2 (36.7)                      | 33.3 (40.0)                      |
  | FC, rotation                  | 29.4 (40.0)                      | 35.6 (46.7)                      |
  | FC, low-DOF                   | 40.0 (56.7)                      | 39.4 (50.0)                      |
  | FC, lag                       | 21.7 (33.3)                      | 26.7 (40.0)                      |
  | FC, keypad                    | 32.2 (36.7)                      | 37.2 (53.3)                      |

- The answer is no. Without the noise, FRS on the clean readout is below FC under every user on the new
  checkpoint (pooled -13.9, -17.8, -5.0, -15.6 points for rotation, low-DOF, lag, keypad; Fisher p 0.010, 0.002,
  0.34, 0.005) and below FC's best cell everywhere on both. With the noise it is within -6.1 to +6.7 points of
  FC on the new checkpoint (no difference significant) and still below or level at the best cells.
- The noise-free arm is not a ceiling in closed loop. Paired by reset and prefix, removing the noise changes FRS by
  0.0 points on round 1's checkpoint and by -11.7 on the new one (p 0.007, 120 pairs). Without noise a rest is an
  exactly zero command, inside the deadband, so in the half of the chunks where the operator rests the unprompted
  policy runs alone, and with noise the steering stays on. Per chunk the executed motion heads with what the
  operator means more often without noise (pushes +0.71 against +0.64, rests +0.60 against +0.55), so the
  per-chunk direction does not decide success. Why this costs success on the new checkpoint only is not settled.
- What the fine-tune changed, paired by reset and knob. FRS on the clean readout with noise +9.2 points (p 0.027),
  without noise -2.5 (p 0.61), the two together +3.3 (p 0.24); FC under the four users together +3.9 (30.8 to
  34.7, p 0.054, its gain 3 against 1.5).
- The policy itself, before and after, on each task's first two development states (`round2/probes.sh`; outputs
  `round1/policy_probe_report_{old,new}.txt`, `round1/policy_switch_probe_report_old_new.txt`). Told task B in
  words after 40 or 80 steps under task A's instruction, it completes B in 34.2% of 120 episodes before and 40.0%
  after (+5.8, p 0.30, paired); after 40 steps 45.0 against 41.7%, after 80 steps 23.3 against 38.3% (+15.0, p
  0.049). Into "turn on the stove" 1 of 20 became 9; into task 5 14 became 8. Under "do something" along the meant
  task's path it is as committed as before (chunks 1 and 2: 98% of samples follow the meant task on both, mean
  pairwise cosine 0.97 before and 0.91 after; first chunk 73 and 75%, 0.79 and 0.77). So the fine-tune taught some
  late redirection by words and left the unprompted policy's commitment where it was.
- Where trials are decided (`round2/first_touch.py`, from the step records; an element counts as disturbed at 2 cm
  or 0.02 of joint travel, and a task's own element is the one the ceiling run disturbs first). The first element
  is disturbed around step 50 to 70. Share of trials whose first disturbed element is the task's own / success
  given that / success otherwise: ceiling 97 / 100 / -; floor 30 / 22 / 10; FRS clean with noise 43 / 75 / 2; FRS
  clean without noise 41 / 47 / 4; FC under the four users 38 to 49 / 52 to 77 / 10 to 13; FC clean (the gain runs
  at 1.5, 60 trials) 55 / 85 / 19. So a trial is decided by which object the robot reaches first, no steering arm
  chooses it much better than the unprompted policy, and after a right first reach FRS completes as often as FC.
  FRS's right first reach by task 1 / 4 / 5 / 6 / 7 / 8 is 50 / 65 / 0 / 10 / 40 / 95%. The operator pushes in 30
  to 40% of the chunks before the first reach. Without noise FRS loses after the right first reach on the bowl
  tasks 1 and 4 (18 and 8% against 90 and 31%), where the bowl's destination is the second choice.

Stage 1's run folders were deleted on the author's word after every trial was exported to
`outputs/pi05_libero_user_study/round1/baseline_trials.csv`, which the paired comparisons use.

## Before round 1

What the earlier iterations (2026-11-14 to 2026-11-20) established, all under the old user (always pushing, noise
only while pushing) and, unless said, translation noise 0.5:

- **The released checkpoint does not need the command.** `lerobot/pi05_libero_finetuned` with the clean, noise-free,
  immediate operator: ceiling 100% on Object and Goal-6, floor 62% on Object (the layout names the object) and 10%
  on Goal-6, FRS at depth 2 94% and 77%. Under "do something" it is near-deterministic given the state (0 of 258
  early states with two samples apart). Under the SpaceMouse-like timing no method beat the floor on Object. So the
  operator is immediate, Object was dropped and Goal-6 kept.
- **Prompt dropout makes the policy steerable by FC, not by FRS.** After the fine-tune, the spread of the
  "do something" samples on Goal relative to the ten sentences went from 0.06 to 0.46 (0.9 would be even), early
  states with samples heading apart from 0 of 128 to 65 of 122, task sentences still followed 8 of 8. Clean operator
  on Goal-6, released / fine-tuned: floor 10 / 7, ceiling 100 / 100, teleop 43 / 43, FC pooled 21 / 24 (best cell
  30 / 40), FRS pooled 37 / 29 (best cell 50 / 33).
- **Baselines on the fine-tuned checkpoint** (Goal-6, 30 trials per cell, pooled over each arm's knob, clean /
  rotation / low-DOF / lag / keypad): FC 24 / 12 / 16 / 12 / 22, FRS 29 / 18 / 22 / 12 / 28, teleop 43 / 0 / 0 / 3 /
  10; floor 7, ceiling 100. FC's gain was 3, chosen on the released checkpoint. The served command's cosine with the
  intent was 0.56 under the clean readout and 0.29 to 0.49 under the users, so the noise was the larger part of
  every user's error, and a module that restored the clean readout exactly could gain only about 12 / 8 / 18 / 1
  points over FRS.
- **Where FRS acts.** At the full reversal FRS barely turns the policy where its plan disagrees with the push, even
  for the exact push (executed cosine with the push 0.25); at depth 4 it executes the push almost exactly (0.88).
- **The cursor's loss, on calibration states where the plan is the label, ignores the command.** On the released
  checkpoint a module gave the same chunk for a command and its opposite (cosine 0.97 to 0.99). On the fine-tuned
  checkpoint, with half the states from "do something" rollouts, it reacted to the command (0.18 to 0.30) but did
  not repair it: where the plan disagreed, level with FRS without a module (-0.03 to +0.02) at the full reversal and
  at depth 4. The plan equalled the label in 83 to 86% of calibration draws, and the 50-step label's heading agreed
  with the 10-step command's at 0.50 mean cosine.
- **A command-space label repairs direction in open loop and loses in closed loop.** Label: the plain reversal path
  of the reference FRS builds from the faithful push, translation dims, the correction complete by reversal step 4.
  At depth 4 where the plan disagreed, the executed steps' cosine with the push was 0.52 / 0.38 / 0.39 / 0.50
  (rotation, low-DOF, lag, keypad) against 0.34 / 0.26 / 0.29 / 0.38 for FRS at that depth and 0.46 for FRS on the
  clean readout. In closed loop MARS-U was below full-reversal FRS for every user at depths 2 to 6 (pooled over
  prefixes: rotation 9 to 13 against 17.5, low-DOF 10 to 14 against 21.7, lag 5 to 8 against 11.7, keypad 13 to 17
  against 28.3); at depths 10 and 8 it was level for rotation (18.3, 16.7) and lag (14.2, 12.5), below for low-DOF
  (15.0, 16.7) and keypad (9.2, 14.2; p < 0.01). Cause: squared error toward the faithful command at this noise
  shrinks the command (by least squares about 0.3 of its length), so the robot moves slower (executed translation
  0.45 to 0.50 per step against FRS's 0.59 to 0.67; the policy alone 0.48) and runs out of its 300 steps. A rotation
  module trained without noise undid 99% of the rotation and did not shrink; trained at 0.5, 31 to 37%. Cued
  calibration (directions drawn evenly, four per state) undid 31 to 35% and still shrank the command.
- **Noise level for round 1, from the readout alone** (558 calibration pushes, mean length 0.58; served command's
  cosine with the push, clean / rotation / low-DOF / lag / keypad; the noise's share of a served command's energy;
  how often one read's length tells a rest from a push):

  | Noise | Clean | U1   | U2   | U3   | U4   | Noise share | Rest vs push |
  | ----- | ----- | ---- | ---- | ---- | ---- | ----------- | ------------ |
  | 0.10  | 0.92  | 0.70 | 0.53 | 0.72 | 0.79 | 7%          | 90%          |
  | 0.20  | 0.79  | 0.60 | 0.40 | 0.59 | 0.68 | 22%         | 78%          |
  | 0.50  | 0.49  | 0.39 | 0.25 | 0.36 | 0.43 | 64%         | 60%          |

**Lessons carried into round 1.**

- Check a user module on the executed steps of the steering's own chunk, split by whether the policy's plan agrees
  with the push, with FRS on the clean readout as the target, and check the executed length. A whole-chunk heading
  cosine rewards weakening the steering, and an open-loop direction gain is necessary, not sufficient; the closed
  loop decides.
- Check a gate against the headroom the baselines leave before writing it.
- A study condition is never chosen from a baseline's or a method's score; the noise level came from the readout.
- No simulator-only knob in a method (a noise-free calibration was rejected for that reason).
- Calibration states where the policy's plan already is the label teach a module to ignore the command.

## Risks

- **Grasping, not steering, may decide success.** Steering matters in the reach; time and executed speed are
  reported beside success.
- **The policy's own preference.** Under "do something" the fine-tuned policy still favours some goals (two Goal-6
  tasks stayed near zero under FRS on the clean readout at 0.5); steering has to overcome it.
- **A synthetic operator does not adapt.** A person partly compensates for a rotation or a gain, so the numbers
  bound what a correction is worth; a human pilot is for April.
- **Trial counts.** 30 per development cell and 120 per test cell; small differences between baselines will not
  resolve.

## Settled with the author

The immediate operator; no command history in the user module; the four Fig. 1a readouts with U1 one rotation at 60
degrees; the baselines as their papers define them, FRS at the full reversal (only MARS tunes depth); teleop as the
operator's full control; noise on every part of the command at every read; the let-go operator; translation noise
0.2 with rotation and gripper scaled alike; FC's gain re-swept on the study condition; Goal-6 only.
