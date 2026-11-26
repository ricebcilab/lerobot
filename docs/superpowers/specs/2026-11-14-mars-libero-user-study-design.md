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

Round 1's and round 2's folders (`outputs/pi05_libero_user_study/round1` and `round2`: the run folders, the trial
export `baseline_trials.csv`, the calibrated user modules, the probes, the analysis scripts and their reports, 24 GB
in all) were deleted on 2026-11-23 on the author's word. The numbers in this section and the next are the record; the
paths they name under `round1/` and `round2/` no longer exist. The checkpoints they ran on are kept
(`outputs/pi05_libero_generic_prompt`, `outputs/pi05_libero_redirect_finetune`), as is the redirect data
(`outputs/pi05_libero_redirect`).

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

**Redirect data (2026-11-21 and 22, `make_redirect_data.py` (removed from the tree on 2026-11-23, in git at
`dd938d17`), `outputs/pi05_libero_redirect/shard_0` to `shard_5`).**
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

- **Control of what FRS does with rotation and gripper (2026-11-23; `round2/control.sh`, `control_report.txt`).**
  Tang et al.'s LIBERO recipe (App. D.2) keeps rotation, at 0, and the gripper, halfway between open and closed, in
  the reference; the study's FRS redraws their noise after the reversal. On the clean readout at noise 0.2, the
  redirect fine-tune's checkpoint and the same resets, successes of 30 at prefix 5 / 10 / 25 / 50: redrawn 7 / 10 /
  12 / 11 (33.3%), rotation reversed 7 / 11 / 10 / 11 (32.5%; -0.8 points, p 1), rotation and gripper reversed
  (`control.gripper_reference=0`) 5 / 7 / 9 / 11 (26.7%; -6.7, p 0.13). The right first reach is 43.3 / 46.7 / 44.2%
  (p 0.34 and 1) and success after it 75 / 64 / 58%. The gripper closes in at least 95% of trials in every arm. So
  the difference kept from the paper does not cost FRS, the three are equal at prefix 50 (the paper's own case),
  and holding rotation and gripper to a filler does not move the first choice.
- **Checks for a completion module (2026-11-23 evening; `round2/oracle_completion.py`, `oracle_report.txt`,
  `decode_check.txt`).** The proposal was a scene-conditioned module that writes the latent FRS hands back to the
  policy (rotation, gripper, the steps past the prefix: 320 of 350 numbers at prefix 10). Two checks, before
  building it, on the clean readout at noise 0.2 and round 2's resets:
  - An oracle that fills the handed-back latent from the meant chunk's (the task-prompted policy's chunk reversed
    through the same field; it knows the task, so it bounds any such module): successes of 30 at prefix 5 / 10 /
    25 / 50, 15 / 12 / 11 / 8 (38.3% pooled; FRS 33.3%, McNemar p 0.33; prefix 5 50.0 against 23.3%, p 0.02);
    right first reach 52.5 against 43.3% (p 0.043). Level with FC under the four users (pooled 26.7-39.4%, best
    cells 40.0-53.3%), not above it.
  - The same oracle filling every element (the policy regenerates the meant chunk under "do something"): 19 of 30
    (63.3%; the ceiling 100%), right first reach 70%; "push the plate" and "turn on the stove" stay mostly out of
    reach. So the "do something" prior caps FRS-based methods on this checkpoint at about 63% even with perfect
    intent.
  - Whether the meant first object can be read from what a module would see (a softmax regression on the
    commands so far and the scene, each task's reset held out; the majority class, the bowl, is 50%): 63% at the
    first chunk, 50-55% at chunks 1 to 7. Without the readout's noise 59-73%, from the operator's intent at every
    chunk 63-91%; an operator that never rests (estimated on the same trajectories) 57-69%. The scene features
    add nothing over the commands.
  - Conclusion: in this setting (a policy told "do something" on tasks it masters) no FRS-based method was found
    that would beat FC; the prior and the operator's signal both limit it. The author chose to move LIBERO to the
    FRS paper's own regime: a policy told the task in words, on tasks it was not trained on (the paper's Goal base
    is a pi0.5 trained only on LIBERO-90), with the released Goal checkpoint, told the task, as the synthetic
    operator. LIBERO then tests correcting a corrupted command for an instructed policy, the cursor tests goal
    inference, and the paper claims no intent inference from LIBERO. Step 0 (the agent's base rates) started
    21:57 in `outputs/pi05_libero_user_study/instructed/`.
- **Step 0 of the instructed setting failed its gate (2026-11-23 22:08).** The agent was the community
  `SongMugeon/Pi05_libero90_finetuning_pi05_base_20k` (pi05_base fine-tuned 30 000 steps on LIBERO-90 only, per its
  train_config). Told the task, it did 15 of 60 on six of its own LIBERO-90 tasks (only "close the top drawer", 10 of
  10; every pick-and-place 0 to 3 of 10) and 1 of 30 on Goal-6, against a gate of 60% and 5-60%. Its dataset
  (`SongMugeon/Lerobot_libero90`) matches ours in features, action and state statistics and the orientation of both
  cameras, so the checkpoint itself is weak; it is not used.
- **Why MARS worked on the EMG feeding task (the author's figure, 2026-11-23).** There FRS on noisy EMG already
  reaches 73% right first grasps and MARS 95%: the prior suffices and the corrupted command is the loss, which the
  user module repairs. Feeding is one decision (which object), the rest shared. LIBERO-Goal's six tasks differ in
  object, destination and skill, and FRS on the clean command reaches 33%. LIBERO-Object has the feeding structure
  (in every scene six groceries, each some task's target, and one destination, the basket), but each layout belongs
  to one task in training, so the unprompted released policy reads the target off the layout (62% with no command,
  2026-11-14). The proposal: data with the target drawn evenly among the objects of each layout, made by the
  released policy told each object, and a goal-blind fine-tune on it (the cursor's and the feeding notes' recipe).
- **Two checks launched 22:52 (`outputs/pi05_libero_user_study/checks_1123/run.sh`).** The released
  `lerobot/pi05_libero_finetuned` alone, seed 0. Check 1: in a LIBERO-Object scene, told to pick up another task's
  object (pairs drawn before any run with numpy `default_rng(0)`: tomato sauce in task 0's scene, ketchup in 2's and
  3's, salad dressing in 4's, orange juice in 6's and 8's; 5 trials each); success is the told object lifted and
  ending in the basket; go on if 60% or more. Check 2: the same checkpoint told the task on LIBERO-90 tasks 3, 11,
  37, 41, 66 and 85 (drawn with `default_rng(1)` from the paper's 62 tasks under 40%; 10 trials each), the agent of
  the instructed setting; keep tasks at 5-60%.
- **Both checks fail their gates (2026-11-23 23:07; `checks_1123/object_check.txt`, `libero90_check.txt`).** Check 1:
  the told object in the basket in 10 of 30 trials (33%, gate 60%), all or nothing by pair: ketchup in task 2's
  scene 5 of 5 and orange juice in task 8's 5 of 5; tomato sauce in task 0's, ketchup in 3's, salad dressing in 4's
  and orange juice in 6's 0 of 5 each, and in those trials nothing was lifted at all, neither the told object nor
  the layout's usual target (LIBERO's own task done in 0 of 30). So words that contradict the layout stall the
  released policy in most scenes rather than send it to the layout's task. Check 2: 1 of 60 (task 11, "open the top
  drawer", 1 of 10; the other five 0 of 10): on LIBERO-90 the released checkpoint has almost none of the skills, so
  the instructed setting with it as the agent leaves FRS nothing to refine, and no working LIBERO-90 operator exists
  either.

## Round 3: the target user (2026-11-23 and 24)

Rounds 1 and 2's synthetic operator took its intent from the task-prompted policy's own plan (the mean of its
chunk's executed prefix). That command is the policy's local motion, not a pointing at the goal: choosing the goal's
object by the direction nearest the mean command is right 27 to 48% of the time, below always guessing the bowl,
and with the let-go rule it is a rest in about half the chunks. The author's own teleoperation on LIBERO-Goal under
FRS and FC found that once the gripper is near the right object the policy, if it has the skill, finishes alone,
which the first-reach numbers above agree with. So round 3 gives the operator a person's pointing
(`policy_operator.intent: target`, `PolicyOperator.target_push`): from the simulator's positions it pushes straight
at the task's object (unit length, shrinking inside 12 cm) until the gripper is over it (within 6 cm sideways and
under 10 cm above), then stops and leaves the grasp to the policy; once the object has moved 3 cm and is within
12 cm of the gripper it pushes toward a point 15 cm above the destination (level with the gripper for an object
pushed along the table) and stops when the object is within 6 cm of the destination sideways; a dropped object is
approached again; a button has no destination. Rotation and gripper stay the policy's; the timing, the let-go rule
and the noise at every read are unchanged. The constants were fixed before any result.

- **Gate (`outputs/pi05_libero_user_study/round3/gate.sh`, `gate_report.txt`; 10 runs, 300 trials).** Round 2's
  cells without a corruption, with the target user: the redirect checkpoint 006000, Goal-6, 30 trials a cell,
  translation noise 0.2 at every read, seed 0.
  Stated before the run: FRS pooled at 60% or more. Result: FRS 3 / 12 / 15 / 15 of 30 at prefix 5 / 10 / 25 / 50,
  37.5% pooled (95% 29-46; round 2: 33.3%), best cell 50.0%; right first reach 51.7% (43%), success after it
  66.1%. FC at gain 1.5, tau 5 / 10 / 20 / 30 / 40 / 50: 7 / 16 / 20 / 19 / 22 / 22, 58.9% pooled (52-66), best
  73.3% (round 2 at tau 10 and 50: 16 and 17); right first reach 66.1%, success after it 82.4%. The gate fails.
- **By task** (successes of 20 under FRS, of 30 under FC; right first reach in brackets): bowl on the stove 13 (80%)
  and 23 (80%); bowl on the cabinet 2 (70%) and 20 (90%); push the plate 1 (0%) and 11 (27%); cream cheese in the
  bowl 0 (0%) and 7 (27%); turn on the stove 14 (75%) and 20 (77%); bowl on the plate 15 (85%) and 25 (97%). The
  plate and the cream cheese lie beyond or beside the bowl as seen from the gripper's start, and the policy takes
  the bowl on the way; under FRS it also rarely carries the bowl up onto the cabinet.
- **Reading.** A pointing user lifts both arms, FC more than FRS, and with no corruption and noise 0.2 the full
  reversal stays below FC.
- **Depth check (`round3/depth.sh`, `depth_report.txt`; 8 runs, 240 trials, 2026-11-24 00:33 to 00:57).** The gate's
  condition (no corruption, noise 0.2, the let-go rule, the target user, seed 0) with FRS at a partial reversal, the
  depth MARS tunes, and no module. Stated before the run: go on only if some depth reaches 70% pooled over prefix 10
  and 25. Successes of 30 at prefix 10 / 25, pooled, right first reach, success after it:

  | Depth (of 10)   | Prefix 10 | Prefix 25 | Pooled (95%)  | Right first reach | Success after it |
  | --------------- | --------- | --------- | ------------- | ----------------- | ---------------- |
  | 2               | 8         | 7         | 25.0% (16-37) | 85.0%             | 29.4%            |
  | 4               | 14        | 8         | 36.7% (26-49) | 76.7%             | 45.7%            |
  | 6               | 12        | 14        | 43.3% (32-56) | 66.7%             | 62.5%            |
  | 8               | 12        | 15        | 45.0% (33-58) | 55.0%             | 72.7%            |
  | 10 (the gate's) | 12        | 15        | 45.0% (33-58) | 55.0%             | 75.8%            |

  The bar is not reached at any depth; no depth exceeds the full reversal, and depth 2 is below it on the same
  resets (4 trials gained, 16 lost, exact McNemar p 0.01). Depth trades the first reach against the finish. A
  shallow reversal follows the pointing (right first reach 85% at depth 2 against 55% at the full reversal) and
  loses the policy's own skill afterwards. On the four pick-and-place tasks (40 trials a depth) the task's object
  is lifted about as often at every depth (29 / 31 / 33 / 28 / 30 at depth 2 / 4 / 6 / 8 / 10), and a lift ends in
  success 7 of 29 times at depth 2, 12 of 31 at depth 4, 16 of 33 at depth 6, 17 of 28 at depth 8 and 18 of 30 at
  the full reversal, so the loss is in the carry and the placing. Even depth 2's first reach with the full
  reversal's finish would give about 0.85 x 0.76 = 64% (an estimate assuming the two are independent), below the
  bar and level with FC (58.9% pooled, 73.3% at its best tau). Pushing the plate and the cream cheese in the bowl
  stay at 0 to 2 of 10 at every depth. With this checkpoint and a goal-blind prompt, FRS on a command with no
  corruption does not reach FC at any depth, so MARS, which adds a user module to FRS at a tuned depth, has no
  room to lead here; the goal-blind LIBERO-Goal path is closed.

## Round 4: scenes that leave the object open (2026-11-24)

After the depth check the author asked for a more suitable benchmark and said that in the feeding deployment the user
commanded 3-D translation only while the policy supplied the grasp orientation. So what feeding has and LIBERO-Goal
lacks is one scene with several objects, each demonstrated as the target from that scene, and a finish that follows
from the object. LIBERO-90 has such scenes, and the Flow Control paper's LIBERO experiment (arXiv 2606.10180, Table 6) uses two of their tasks with pi0.5-LIBERO: task 60 (unsteered 28 of 50, scripted steering 47) and task 12 (22 and
32); adding the bowl's location to the prompt gave 28 and 24. Kitchen scene 2 holds three identical black bowls
and tasks 12 / 13 / 14 (the back / front / middle bowl onto the plate); living-room scene 4 holds two black bowls, a
chocolate pudding and a salad dressing and tasks 60 / 61 / 62 (the left bowl / pudding / dressing into the tray).
Others looked at: LIBERO-Spatial (two identical bowls, every task ends on the plate, but the layout names the bowl
except tasks 9 and 10), VLABench's `select_*` tasks (an official pi0.5 checkpoint at 40.6%, a new harness), RoboCasa
pick-and-place (low base skill), CALVIN (not integrated).

- **First check (`outputs/pi05_libero_user_study/round4/check.sh`, `check_report.txt`, `check_user.txt`; 10 runs, 300
  trials, 01:58 to 02:31).** The released `lerobot/pi05_libero_finetuned` (trained on the four suites, not
  LIBERO-90), seed 0, 10 trials a task. The policy alone told the task, the policy alone under the scene's sentence
  without the object ("put the black bowl on the plate", "pick up an object and put it in the tray", fixed before the
  run), and under that prompt with the target user, no corruption, translation noise 0.2 at every read, the let-go
  rule: FC (gain 1.5, tau 30) and FRS at the full reversal (prefix 10 and 25). Stated before the run: go on if FRS
  reaches 70% pooled and is at least level with FC, the policy alone under the generic prompt well below.

  | Arm            | Kitchen 2 (12 / 13 / 14) | Living room 4 (60 / 61 / 62) | Both scenes    |
  | -------------- | ------------------------ | ---------------------------- | -------------- |
  | Told the task  | 5 / 3 / 3 = 36.7%        | 9 / 10 / 0 = 63.3%           | 50.0% (30/60)  |
  | Generic prompt | 6 / 1 / 4 = 36.7%        | 7 / 3 / 0 = 33.3%            | 35.0% (21/60)  |
  | FC             | 7 / 2 / 5 = 46.7%        | 7 / 4 / 0 = 36.7%            | 41.7% (25/60)  |
  | FRS, prefix 10 | 9 / 0 / 2 = 36.7%        | 9 / 5 / 0 = 46.7%            | 44.2% (53/120) |
  | FRS, prefix 25 | 7 / 3 / 2 = 40.0%        | 9 / 7 / 0 = 53.3%            | with prefix 10 |

  The bar is not reached.

- **The finish is intact and the choice is open.** In the kitchen a bowl ended on the plate in 30 of 30 trials of
  the policy alone under either prompt, and success after a right first reach was 75 to 100% in every arm of both
  scenes. Under the generic prompt the first bowl moved was the back one 14 times, the middle 14 and the front 2;
  told which bowl, 13 / 12 / 4, so the sentence does not choose (as the Flow Control paper found). The salad
  dressing is never taken, told or not, and is outside this checkpoint's skill.
- **What fails is the selection.** FC leaves the choice where the policy alone had it (first moved 14 / 14 / 2). FRS
  pulls it to the nearest bowl (the back one 22 and 24 of 30) and leaves 11 of 30 kitchen trials and 12 of 30
  living-room trials with nothing delivered. In the living room FRS does move the choice (pudding 3 of 10 under the
  generic prompt, 5 and 7 under FRS).
- **The scripted user cannot name a bowl in this geometry (from the records, `check_user.txt`).** The gripper
  starts 27 cm above the table, so straight pushes at the three bowls have cosine 0.83 to 0.96 with each other
  (0.37 to 0.86 in the living room), about the size of the noise's turn at 0.2. The first object moves after about
  six chunks. In the trials that reached a wrong object first, the user had let go in 38 to 51% of those chunks,
  because the robot heading for a wrong bowl still moves at cosine 0.5 or more with the user's intent (mean 0.46 to
  0.67). Seen from above, the back bowl also lies on the line from the gripper to the middle bowl.
- **On the Flow Control paper's own two tasks** (12 and 60, the bowls the policy prefers anyway): the policy alone
  under the generic prompt 13 of 20, FC 14 of 20, FRS 34 of 40. The four other tasks carry the selection, and they
  are where every arm fails.
- **Reading.** Unlike Goal, the policy here has what steering needs (a finish that works for any choice, a choice
  the scene and the sentence leave open). The check fails on the user model: a 3-D straight push with a 3-D let-go
  test does not tell close targets apart. Whether a user that points in the table's plane, or a person, selects
  the bowl is not run.

## The September record (audit of 2026-11-24)

The author recalled MARS with a fixed adapter beating FC and FRS on simulated LIBERO and suspected a silent break.
The record is the study record of the old README (`git show 3c6966e0:examples/pi05/libero_shared_autonomy/README.md`,
lines 543 to 597); its run folders are no longer on this machine.

- **Condition then.** `lerobot/pi05_libero_finetuned` under "do something", the told policy's plan as the operator
  (always pushing, noise-free), a 20 degree rotation about z, each method at its best depth. FC was the example's
  own wrapper (the command written into the whole chunk at each of n denoising steps), FRS and MARS a partial
  reversal at depth 1 or 2, the adapter a fixed 40 degree rotation matrix.
- **Numbers then.** Ten Goal tasks with the operator acting at once, seeds 0 to 2: FC 58%, FRS 51%, MARS 60%
  ("to parity with FC but not past it"). Task audit with the delayed operator (a new intent every 20 steps, 4 late),
  20 trials a cell, seed 0, FC / FRS / MARS: task 4 20 / 45 / 60 (MARS against FC +40, p 0.021; against FRS +15,
  p 0.55), task 5 30 / 25 / 25, task 6 15 / 25 / 25, task 7 85 / 85 / 85, task 8 55 / 80 / 65. Human pilot on task 4,
  ten resets: 90 / 100 / 100.
- **What differs in rounds 1 to 4.** FC is Kao et al.'s method with a tuned gain; FRS runs the full reversal and
  MARS-U ran at depths 10 / 8 / 6 (2 to 6 before round 1); the adapter is a learned module calibrated under noise;
  the command carries noise 0.2 at every read; the operator lets go (rounds 1 and 2) or pushes straight at the
  object (rounds 3 and 4); the corruptions are the four users or none; the checkpoints are the two fine-tunes
  except in round 4.
- **Checks made, no run.** The fixed matrix still acts in the reversal only (`AdaptedField.adapt`); 247 unit tests
  pass; the round 3 gate (before the 11-24 trim) and the depth check (after it) agree bit for bit on the reset, the
  operator's first intent and the first served command in 120 matched trials (`round3/trim_identity.txt`); the
  rewritten code gave FRS at depth 2 77% on Goal-6 with the noise-free operator on the released checkpoint in
  mid-November (first bullet of "Before round 1").
- **The task 4 cell on today's code** (2026-11-24 03:50 to 04:02, the author's "Run the test";
  `outputs/pi05_libero_user_study/september/cell.sh`, `cell_report.txt`; stated before: unbroken if within about 22
  points of September). Successes of 20, today against September at each method's best depth then:

  | Cell                                      | Today | September |
  | ----------------------------------------- | ----- | --------- |
  | FRS, depth 1                              | 11    | 9         |
  | FRS, depth 2                              | 7     |           |
  | MARS, fixed 40 degree adapter, depth 1    | 14    |           |
  | MARS, fixed 40 degree adapter, depth 2    | 9     | 12        |
  | FC as then (written at 4 denoising steps) | 3     | 4         |
  | Policy told the task                      | 19    | 20        |

  Every cell is in range, so the path is not broken, and the order is September's: at their best depths MARS 70%,
  FRS 55%, FC 15%. Paired on the shared resets: MARS depth 1 against FC +11 / -0 (p 0.001), FRS depth 1 against FC
  +10 / -2 (p 0.039), MARS against FRS at depth 1 +6 / -3 (p 0.51) and at depth 2 +4 / -2 (p 0.69). The adapter
  acts, by a little: the executed translation sits 12.7 degrees off the operator's intent under FRS at depth 1 and
  10.3 with the adapter (14.5 and 7.9 at depth 2), of the corruption's 20.

- **Reading.** The lead over FC in this cell is the shallow reversal's (FRS alone has it); the adapter's own part
  is not resolved at 20 trials, as in September (p 0.55 then). What the study changed afterwards (the paper's FC,
  the full reversal for FRS, noise at every read, a learned module, the let-go rule, other corruptions and
  checkpoints) is where the lead went, and no run isolates which change.
- **The walk's first step** (2026-11-24 04:28 to 04:50, the author's "Run the walk"; `september/walk.sh`,
  `walk_report.txt`, `walk_noise.txt`): the reproduced cell with one change each, 20 trials a cell on the same
  resets.

  | Change                              | Cell                                 | Successes of 20 |
  | ----------------------------------- | ------------------------------------ | --------------- |
  | none (the reproduced cell)          | MARS depth 1 / FRS depth 1 / FC then | 14 / 11 / 3     |
  | the papers' baselines, no noise     | FC, gain 1.5 and 3, tau 10 and 30    | 4, 3, 4, 4      |
  |                                     | FRS, full reversal, prefix 10 and 25 | 1, 2            |
  | translation noise 0.2 at every read | MARS depth 1 / FRS depth 1 / FC then | 6 / 3 / 3       |

  The papers' baselines do not take the lead away: FC 18.8% pooled (15 of 80), FRS at the full reversal 7.5% (3 of
  40), and MARS at depth 1 beats each of the six cells without losing a trial (+10 to +13 / -0, p 0.002 or less).
  The noise does: MARS depth 1 falls from 14 to 6 (+3 / -11, p 0.057), FRS depth 1 from 11 to 3 (+1 / -9, p 0.021),
  FC as then stays at 3; under the noise no pair is resolved (MARS against FRS +4 / -1, against FC +5 / -2). The
  papers' baselines were not run under the noise.

- **How the noise gets through.** The flow modes read the readout once per chunk and hold that read for the 10
  executed steps (half a second); teleop reads every step. At depth 1 the executed translation follows the served
  command at cosine 0.98, against 0.34 to 0.53 for FC and the full reversal, where the policy's prior decides and
  the task is not done. So the shallow reversal is the only arm here that does what the operator means, and one
  noise draw steers it for half a second: the served command's cosine with the intent goes from 0.98 to 0.90 and
  the steps within 30 degrees of the intent from 98% to 77%.
- **The walk's second step** (2026-11-24 05:31 to 05:41, the author's "Do it"; `september/window.sh`, the same
  reports, `adapter_pooled.txt`): the shallow cells at noise 0.063 per read, the level the mean of the window's 10
  reads at 0.2 would leave (a stand-in that needs no code change), and the papers' baselines at noise 0.2.
  Successes of 20 on the same resets:

  | Cell                             | No noise | Noise 0.063 | Noise 0.2 |
  | -------------------------------- | -------- | ----------- | --------- |
  | MARS, depth 1, fixed adapter     | 14       | 11          | 6         |
  | FRS, depth 1                     | 11       | 8           | 3         |
  | FC per the paper, gain 3, tau 30 | 4        |             | 2         |
  | FRS, full reversal, prefix 25    | 2        |             | 3         |

  At the window's noise level most of the lead is back: MARS at depth 1 against FC at noise 0.2 +10 / -1 (p 0.012)
  and against the full reversal at noise 0.2 +9 / -1 (p 0.021); the baselines were not run at 0.063, and their
  cells at no noise and at 0.2 (2 to 4 of 20) bracket it. At noise 0.2 on every arm nothing is resolved (MARS
  against FC +6 / -2, p 0.29). The adapter adds three trials at each noise level; pooled over the three (a pooling
  chosen after seeing them) +14 / -5, p 0.064.

- **The walk's third step, the six tasks** (2026-11-24 05:48 to 06:22, the author's "Run it on the six tasks";
  `september/goal6.sh`, `goal6_report.txt`, `goal6_where.txt`; stated before: MARS at depth 1 ahead of both papers'
  baselines by 15 points or more pooled, each paired p below 0.05; NOT reached). The task 4 cell's condition on
  Goal-6, 10 trials a task, every steered arm at noise 0.063 per read. Successes of 10 per task and of 60:

  | Arm                              | t1  | t4  | t5  | t6  | t7  | t8  | Pooled     |
  | -------------------------------- | --- | --- | --- | --- | --- | --- | ---------- |
  | MARS, depth 1, fixed adapter     | 2   | 5   | 0   | 1   | 7   | 0   | 15 (25.0%) |
  | FRS, depth 1                     | 5   | 2   | 0   | 0   | 6   | 0   | 13 (21.7%) |
  | FC per the paper, gain 3, tau 30 | 3   | 0   | 0   | 0   | 0   | 3   | 6 (10.0%)  |
  | FRS, full reversal, prefix 25    | 2   | 1   | 2   | 1   | 1   | 7   | 14 (23.3%) |
  | Policy alone, "do something"     | 3   | 1   | 0   | 0   | 0   | 4   | 8 (13.3%)  |
  | Policy told the task             | 10  | 10  | 9   | 10  | 10  | 10  | 59 (98.3%) |

  MARS against FC +15 / -6 (15.0 points, p 0.078), against the full reversal +13 / -12 (1.7 points, p 1.0), against
  FRS at depth 1 +7 / -5 (p 0.77). The shallow reversal wins tasks 4 and 7 (the cabinet top and the stove's knob,
  which the policy alone never does) and loses task 8 (the bowl onto the plate, the policy's own default, 0 against
  7). It reaches and lifts but does not place: on the cream cheese 7 to 8 right first reaches and 6 to 9 lifts for
  0 to 1 successes; on task 8, 4 to 5 lifts for no success.

- **Reading.** Task 4's lead does not carry to the six tasks under September's cell. What the cell keeps and the
  study does not is the delayed operator (a new intent every 20 steps, 4 late): at depth 1 the arm executes a
  command up to 24 steps old, while the release and the wrist stay with a policy that is not told the goal. The
  record of the immediate operator on this checkpoint is 77% for FRS at depth 2 on Goal-6 (no corruption, no noise;
  first bullet of "Before round 1"), against 21.7% here at depth 1.
- **The walk's fourth step, the immediate operator** (2026-11-24 15:18 to 15:58, the author's "Run it with the
  immediate operator"; `september/goal6_immediate.sh`, `goal6_immediate_report.txt`, `goal6_immediate_where.txt`).
  The third step with one change: the told policy's plan is delivered at every chunk and consumed at once (the
  study's operator timing; still always pushing). Stated before: a MARS depth ahead of both papers' baselines by 15
  points or more pooled, each paired p below 0.05, read at each of the three depths, a depth that reaches it counting
  only once a new seed confirms it. REACHED at depths 1 and 2, the new seed still owed. Successes of 10 per task and
  of 60 (20 degree rotation, every steered arm at noise 0.063 per read, seed 0, the resets of the third step):

  | Arm                              | t1  | t4  | t5  | t6  | t7  | t8  | Pooled     |
  | -------------------------------- | --- | --- | --- | --- | --- | --- | ---------- |
  | MARS, depth 1, fixed adapter     | 8   | 10  | 10  | 8   | 10  | 10  | 56 (93.3%) |
  | MARS, depth 2, fixed adapter     | 8   | 9   | 5   | 1   | 9   | 10  | 42 (70.0%) |
  | MARS, depth 4, fixed adapter     | 7   | 1   | 0   | 1   | 2   | 8   | 19 (31.7%) |
  | FRS, depth 2                     | 7   | 7   | 8   | 1   | 9   | 9   | 41 (68.3%) |
  | FC per the paper, gain 3, tau 30 | 3   | 0   | 0   | 0   | 0   | 6   | 9 (15.0%)  |
  | FRS, full reversal, prefix 25    | 7   | 0   | 1   | 2   | 3   | 9   | 22 (36.7%) |
  | Policy alone, "do something"     | 3   | 1   | 0   | 0   | 0   | 4   | 8 (13.3%)  |
  | Policy told the task             | 10  | 10  | 9   | 10  | 10  | 10  | 59 (98.3%) |

  MARS at depth 1 against FC +47 / -0 (78.3 points), against the full reversal +35 / -1 (56.7 points); at depth 2
  +33 / -0 and +22 / -2 (55.0 and 33.3 points); at depth 4 +10 / -0 against FC (16.7 points, p 0.002) and +3 / -6
  against the full reversal (-5.0 points, p 0.51). Depth 2 against depth 1 +0 / -14, depth 4 against depth 2
  +1 / -24. MARS against FRS at depth 2 +5 / -4 (p 1.0). The operator's timing, the same arm immediate against
  delayed: MARS depth 1 +42 / -1, FC +3 / -0 (p 0.25), the full reversal +11 / -3 (p 0.057). The executed
  translation's cosine to the delivered command: 0.97 at depth 1, 0.95 at depth 2, 0.77 at depth 4, 0.74 FC, 0.78
  the full reversal. At depth 1 every task is reached and finished (right first reach 7 to 10 of 10, the cream
  cheese 10 reaches, 10 lifts, 8 successes); at depth 2 the cream cheese fails at the reach (3 right first reaches,
  1 success).

- **Reading.** The suspect of the third step was the cause: with the plan delivered at every chunk, the shallow
  reversal leads both papers' baselines on all six tasks by 57 to 78 points, and the lead falls with depth (93, 70,
  32%). What the run does not show: (1) any gain from the fixed adapter (level with FRS at depth 2; FRS at depth 1
  was not run with this operator); (2) a teleop arm, which this cell needs as its reference, since depth 1 executes
  the delivered command almost as given and the operator's command is the told policy's own plan; (3) the level at
  noise 0.2 per read (the 0.063 is the stand-in for a mean over the window's reads, which is not built); (4) a
  second seed (the bar's own condition for the depth picked here); (5) the study's users or a learned module.
- **Not run.** A new seed at depth 1 with FRS at depth 1 and teleop beside it; noise 0.2 per read under this
  operator; the window's mean itself in place of the stand-in; the study's users and a learned module.

## The replication: a trained module and tuned baselines (2026-11-24)

**Kept for the paper (2026-11-25).** The study is the final round's three users (rotation, lowdof, lag) with MARS,
FRS without a module at MARS's setting, FC and FRS at the full reversal; the keypad user was discarded, and the
teleop arms, the 20 degree rotation test, the earlier rounds' runs and the keypad's runs were deleted. What remains:
`examples/pi05/mars/results/libero_sim/replication/` (`users.sh`, `users_report.py`, the three users' runs).

**Files moved (2026-11-25).** `outputs/pi05_libero_user_study/` is now
`examples/pi05/mars/results/libero_sim/`, and its user modules (`replication/modules/`,
`replication/users/<user>/modules/`) are now `examples/pi05/mars/checkpoints/adapters/libero_sim/<user>/d<k>`
(the 20 degree rotation's under `rotz20/`). The paths below are as they were when the runs were made.

On the author's word ("switch to trained adaptor and well tuned FC and FRS settings like the one we ran recently to
see if the gain can be replicated", then "add in a fixed adaptor arm in final test" and "add in depth 10 (full
reverse) module"). The fourth step's cell, kept: Goal-6, the released checkpoint under "do something", the told
policy's plan as the operator at every chunk and always pushing, a 20 degree rotation of the command about z,
translation noise 0.063 at every read, seed 0. Tuning on each task's first 5 start states of the schedule (30 trials
a run), the test on its 11th to 20th (60 trials an arm), which share no start state with the tuning or the walk.
Launchers, reports and runs are in `outputs/pi05_libero_user_study/replication/` (`run_stage.sh`,
`train_module.sh`, `tune_report.txt`, `module_check.txt`, `test_report.txt`).

- **FC's gain.** Without the corruption, successes of 30 at `tau` 10 / 50: gain 1.5 5 / 6, 3 4 / 5, 5 8 / 7, 8 8 / 8,
  12 7 / 15. Gain 12 won, the sweep's edge, and the author ruled it out ("12 already seems like too large a gain as
  per what gain does in FC paper", then "Just do 1.5"). The paper (Sec. 3.4) keeps the written command "comparable
  in scale to the samples of N(0,I) ... without being so large that it is improbable under this distribution" (their
  inputs: median 0.46 sigma, 96% within 2 sigma; LIBERO 1.5 sigma). Measured on these runs, gain 1.5 writes a median
  of 0.9 sigma with every axis within 2 sigma; gain 12 a median of 8 sigma with 40% of axes within 2 sigma, and at
  `tau` 50 the executed translation follows the command at cosine 0.91 with 16% of its axes at the clip (0.54 and 0%
  at gain 1.5): the command passed through, not a steer inside the prior. FC runs at gain 1.5 from here on, and its
  gain is not swept past the paper's scale.
- **Baselines, tuned** (20 degrees, noise 0.063, successes of 30). FC at gain 1.5 over `tau` 5 / 10 / 20 / 30 / 40 /
  50: 4 / 5 / 3 / 5 / 5 / 5, every success on task 8, the policy's own default; `tau` 50 by the tie rule. FRS at the
  full reversal over the prefix 5 / 10 / 25 / 50: 5 / 9 / 13 / 12; prefix 25.
- **Modules.** One calibration of 20 watched rollouts on tasks 0, 2, 3, 9 (600 states, a fifth held out, noise 0.063
  drawn once per state), 3 000 steps each. Open-loop check on the held-out states where the plan disagrees with the
  push (cosine of the executed translation with the push; FRS on the user's command, on the uncorrupted readout, the
  module): the depth-1 module at depth 1 0.807 / 0.865 / 0.845 (65% of the gap, pass); the depth-2 module at depth 2
  0.679 / 0.730 / 0.771 (pass); the depth-4 module at depth 4 0.520 / 0.581 / 0.588 (pass); the module trained at
  the full reversal at depth 10 0.163 / 0.201 / 0.116 (fail; FRS on the push itself reaches only 0.234 there, so
  the full reversal leaves a module 0.04 to close). No executed length falls more than 3% below FRS's.
- **MARS, tuned** (successes of 30 at prefix 5 / 10 / 25 / 50). Depth 1: 7 / 23 / 27 / 25. Depth 2: 6 / 20 / 23 / 24.
  Depth 4: 7 / 11 / 17 / 15. Depth 10 with the full-reversal module (on the author's word, against its check): 4 /
  7 / 9 / 10, at or below FRS without a module. Pick: depth 1, prefix 25.
- **Test** (successes of 60, 95% Wilson interval; t1 t4 t5 t6 t7 t8):

  | Arm                                         | Successes          | By task          |
  | ------------------------------------------- | ------------------ | ---------------- |
  | MARS, trained module, depth 1, prefix 25    | 49, 81.7% (70-89)  | 10 10 4 7 8 10   |
  | MARS, fixed 40 degree adapter, same setting | 55, 91.7% (82-96)  | 10 10 8 7 10 10  |
  | FRS without an adapter, same setting        | 50, 83.3% (72-91)  | 9 10 7 5 10 9    |
  | MARS at the full reversal, prefix 50        | 28, 46.7% (35-59)  | 7 3 2 2 5 9      |
  | FRS, full reversal, prefix 25               | 23, 38.3% (27-51)  | 6 1 2 1 4 9      |
  | FC, gain 1.5, tau 50                        | 11, 18.3% (11-30)  | 1 2 0 0 2 6      |
  | Teleop                                      | 44, 73.3% (61-83)  | 8 10 7 4 7 8     |
  | The policy alone                            | 12, 20.0% (12-32)  | 1 2 0 0 2 7      |
  | The policy told the task                    | 59, 98.3% (91-100) | 10 9 10 10 10 10 |

  Paired on the shared resets (gained / lost, exact McNemar): MARS against FC +38 / -0 and against the full
  reversal +29 / -3 (both p < 0.001), so the bar stated before the run (15 points over both tuned baselines, each
  p below 0.05) is reached at +63.3 and +43.3 points. MARS against FRS at the same depth +8 / -9 (p 1.0); against
  the fixed adapter +3 / -9 (p 0.15; task 5 +1 / -5); the fixed adapter against FRS at the same depth +6 / -1
  (p 0.125); MARS against teleop +13 / -8 (p 0.38); MARS at the full reversal against FRS there +8 / -3 (p 0.23);
  FC against the policy alone +1 / -2.

- **Reading.** The fourth step's lead holds on unseen start states against baselines tuned on this cell, and it is
  the shallow reversal's: FRS at depth 1 without any adapter scores the same as MARS with the trained module, and
  the trained module is not ahead of teleop. The fixed 40 degree matrix is 8 points above both, unresolved at 60
  trials. On a 20 degree rotation at noise 0.063 the corruption leaves a module little to correct at depth 1 (0.81
  against 0.87 open loop). FC at the paper's gain does not move the released checkpoint off its default.
- **The study's four users** (the author's go: "We can run other users at the same noise level after this", "Leave
  out depth 4 and 10"; `replication/users.sh`, 2026-11-24 23:35 to 2026-11-25 08:35, `users_report.txt`). The same
  cell with each user in place of the 20 degree rotation, noise 0.063, FC at gain 1.5, MARS over depth 1 and 2 with
  a module trained per depth on one calibration per user, picks by the same rules, the same 60 test start states;
  the policy alone (20.0%) and told the task (98.3%) are the replication's. Modules were used whatever their
  open-loop check said, the run being unattended. One arm (rotation, MARS) failed at mkdir on a 273-character run
  folder name and was rerun from a copy of base.yaml with the matrix written in.

  | User                   | MARS (depth, prefix) | FRS, same depth, no module | FC    | FRS, full | Teleop | Bar         |
  | ---------------------- | -------------------- | -------------------------- | ----- | --------- | ------ | ----------- |
  | U1, 60 degree rotation | 80.0% (1, 50)        | 1.7%                       | 18.3% | 18.3%     | 0.0%   | reached     |
  | U2, vertical dropped   | 66.7% (2, 50)        | 55.0%                      | 20.0% | 38.3%     | 0.0%   | reached     |
  | U3, lag and turn       | 48.3% (1, 10)        | 41.7%                      | 21.7% | 21.7%     | 20.0%  | reached     |
  | U4, six-key pad        | 21.7% (2, 5)         | 20.0%                      | 16.7% | 35.0%     | 28.3%  | not reached |

  MARS against FRS at its own depth and prefix (the module's gain): U1 +47 / -0 (p < 0.001), U2 +9 / -2 (p 0.065;
  the bowl onto the cabinet, which needs the vertical motion this user cannot send, +7 / -0), U3 +7 / -3 (p 0.34),
  U4 +2 / -1. Against the full reversal: U1 +39 / -2, U2 +21 / -4, U3 +17 / -1 (each p <= 0.001), U4 +0 / -8
  (p 0.008). Open-loop checks at the depth used: U1 83%, U3 83%, U4 112% of the gap (pass), U2 37% (fail). The
  check did not predict closed loop on U2 (failed, yet ahead) or U4 (passed, at the floor: every MARS tuning cell
  1 to 5 of 30).

- **Reading.** On LIBERO the shallow reversal carries the lead over both papers' baselines, and the learned user
  module is what makes the shallow reversal usable when the corruption is large: under a 60 degree rotation FRS at
  depth 1 follows the corrupted command into failure (1.7%) and the module restores it (80.0%); under the 20 degree
  rotation it was not needed (83.3% without it). The keypad is the one user where a shallow reversal loses to the
  full one, with or without the module.

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
