# Running MARS on the Franka

This folder runs π0.5-DROID on an N-Droids Franka station through `c3po`, steered by a 3Dconnexion
SpaceMouse. The policy runs on `chopper` (the GPU machine, a 5090); the arm, the two ZED cameras and
the Robotiq gripper are on `bb8`, which runs the r2d2 server that c3po talks to at `10.42.0.1:9090`.
Phase A can do six things: `teleop` (the operator alone, and the demo-recording path), `autonomous`
(the policy alone), and `fc`, `frs`, `mars-f` (the three methods whose numbers already exist), plus
`probe` (the read-only bring-up report). Phase B adds the learned user module: `--record-states`
collects the states a session visits, `train-user` calibrates the module on them, `validate-user`
compares it offline against the fixed methods, and `mars-u` runs it. Phase C adds the learned task
module: `teleop --record` records demonstrations, `train-task` trains the module on them, and `mars-t`
(FRS with the task module) and the composed `mars` (FRS with both learned modules) run it; S2's
verdict is task success on the robot, and the design of that block is in the Phase C section.
Phase D runs that verdict: `study` is a block of trials over the config's arms, with a scene reset and
one y/n outcome per trial, and `analyse` reads the block's records and prints the per-arm table and
the paired tests. `DESIGN.md` is the interface contract; this file is the runbook.

Everything below runs from the lerobot checkout on `chopper`.

## Install

```bash
uv sync --extra pi
uv pip install -e ../c3po
```

`c3po` is not on PyPI: clone it beside the lerobot checkout and install it editable. The shorthand
`uv run mars-franka <command>` works after that `uv sync --extra pi`; every command in this folder
passes `--extra pi` explicitly, the form that does not depend on which extras the environment was last
synced with.

The checkpoint needs openpi's DROID statistics, which it does not ship. Download them once, exactly as
[`assets/README.md`](assets/README.md) records:

```bash
curl -sS --create-dirs -o examples/pi05/mars/franka/assets/droid/norm_stats.json \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json
```

The SpaceMouse is read straight from `/dev/hidraw`, so the node must be readable by your user. Drop
this rule in `/etc/udev/rules.d/99-spacemouse.rules`:

```
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c6*", MODE="0666"
```

and reload it: `sudo udevadm control --reload-rules && sudo udevadm trigger`, then replug the device.
The group form and the Bluetooth alternative are in
[`../README.md#spacemouse`](../README.md#spacemouse).

The PaliGemma tokenizer is gated. Accept the license on the Hub for `google/paligemma-3b-pt-224` and
run `hf auth login` once, or put a token in `HF_TOKEN`. With every file already cached,
`HF_HUB_OFFLINE=1` skips the check. A 401 at start-up is that token, not the arm.

Sanity check the install without hardware: a dry run builds a fake station, a fake source and a fake
bundle, prints every action and never imports c3po, torch or a checkpoint.

```bash
uv run --extra pi mars-franka teleop --dry-run --steps 4
```

Four trace lines and an exit status of 0 mean the composition is wired. If this fails, the problem is
the install; the arm is not involved.

## Probe first: read the hardware before it moves

The probe loads the checkpoint on the GPU and reads the station and the SpaceMouse. It builds no
action, so nothing in this folder can move the arm. Keep the e-stop in reach anyway: `reset()` is the
station's own episode signal, and whatever r2d2 does on reset is r2d2's doing.

```bash
mkdir -p examples/pi05/mars/franka/outputs/mars_franka
uv run --extra pi mars-franka probe --config examples/pi05/mars/franka/configs/franka.toml \
  2>&1 | tee examples/pi05/mars/franka/outputs/mars_franka/probe-session.txt
```

The report has four sections, each ending with a `status:` line. A section that cannot run prints
`[UNAVAILABLE]` or `[FAIL]` with the reason and the next section still runs. The exit status is 0
only when all four passed; `tee` hides that status, so read the four `status:` lines and the
`probe: n/4 sections passed` line instead of `$?`.

The SpaceMouse section needs you. It prints its instruction before the five-second read window: push
each translation axis to its stop in turn (+x right, +y toward you, +z down), twist each rotation
axis to its stop in turn (rx, ry, rz), then release the stick and hold it still for the half-second
rest read. Push every axis fully: a peak short of the stop is a lower bound on the scale.

One 3Dconnexion device can expose several hidraw nodes and only one carries the reports: the session
saw five nodes for one mouse and translation reports on `/dev/hidraw8` only, so the source searches
for the node that reports instead of taking the first by name. The section prints, per 3Dconnexion
node, its `HID_NAME` and report-descriptor size, then tallies every `(report id, byte length)` pair
and the max `|int16|` of the translation triple at offset 1 and — for a payload of 12 bytes or more —
the rotation triple at offset 7. The verdict accepts both device layouts: rotation is present when
id 2 reports arrive (a SpaceMouse Pro and similar) or when the id 1 six-axis report's second triple
moved above a few counts (a SpaceMouse Wireless through its Universal Receiver sends all six axes in
one 13-byte report and no id 2 at all, which is what the station measured). The section prints the
translation and rotation peaks to copy as the two full scales. Finally it prints which node the real
source picks, the rest six-vector it captures and whether it tared.

The first probe run downloads and loads the checkpoint, so the policy section can take minutes. Its
`chunk time` (the autonomous chunk) and the `steered chunk time fc`/`frs` lines after it (the same
prepare plus each mode's chunk-boundary act) are the numbers that decide whether the chunk pause fits
the control loop, so record them.

Copy the report's hardware facts into the config while the report is fresh:

| fact | what the probe prints | where it goes |
| --- | --- | --- |
| station native rate | `control rate: the loop asks for X Hz; the station's native rate is Y Hz` | `[station] rate`, if the rate check failed |
| arms and action keys | `arms:`, `action keys:` | nothing, unless the action-contract checks fail: that is a plumbing bug |
| camera feeds | `cameras:` | `[cameras] exterior`, `wrist`, and `third` when a third feed is wired |
| SpaceMouse node and HID id | `hidraw /dev/hidrawN: HID_ID ... (3Dconnexion), HID_NAME ..., report descriptor N bytes` and `source would pick: /dev/hidrawN` | `[spacemouse] path` and `hid_id` in the session record; leave `""` to search (the source keeps the node that reports) |
| translation full scale | `copy the measured translation peak into [spacemouse] full_scale = N` | `[spacemouse] full_scale` |
| rotation full scale | `copy the measured rotation peak into [spacemouse] rotation_full_scale = N` | `[spacemouse] rotation_full_scale` |
| source and sign | `configured source [...] and sign [...]` | `[spacemouse] source`/`sign`, after a jog confirms a direction |
| rest six-vector | `rest six-vector [x y z rx ry rz]` and `at rest after tare` | the session record; the captured offset is a device property (the session measured `x = +0.154`), so `[spacemouse] tare = true` subtracts it |
| reset pose | `reset pose q (...)` | `[command] q0` is pinned to the measured session pose; `[]` would take the station's reset pose each session |
| the lift and the gain | `q0:`, `F7 (7x7, ...)`, `gain:` | the session record; these are the readout the shared modes run |
| noise sigma | `noise sigma (8 dims: 7 command dims + gripper; the loop's noise is the first 7)` | check it against `[readout] noise`; the loop's command is 7 wide, the gripper is not noised |
| gripper ends | by hand at the first teleop: which `gripper_position` value closes the jaws | the session record; `1.0` is the closed end in `station.action_dict` |
| third image | `third camera: omitted ...` or `present` | `[cameras] third` |
| checkpoint facts | `type:`, `chunk_size:`, `n_action_steps:`, `num_inference_steps:`, `dims`, `dtype`, `use_relative_actions`, `image keys` | the session record; the config overrides `n_action_steps` and `dtype` |

After writing the SpaceMouse scales and, if you confirmed a direction, the sign map, run the probe
once more. The second report is the session's; copy it to the last table.

## Bring-up order

Do these in order, with the e-stop in reach. The first command that moves the arm has the most
runbook risk, so its gain is low and its readout clean.

### 1. Teleop at a low gain

Copy the shipped config rather than editing it, so the committed study config stays the record:

```bash
cp examples/pi05/mars/franka/configs/franka.toml /tmp/mars-franka-bringup.toml
```

Set `[command] translation_gain = 0.02` and `rotation_gain = 0.05` in the copy, then run plain teleop,
which uses the **clean** readout: no corruption and no noise.

```bash
uv run --extra pi mars-franka teleop --config /tmp/mars-franka-bringup.toml
```

The arm should follow the stick gently and stop when you let go. The gripper toggles on a button
press. The header's `source:` line names the hidraw node the source picked and its tare; if no
3Dconnexion node delivers a translation report the run stops with the list of nodes it tried (each
device exposes several nodes and only one reports, so a silent node is a wrong node, not a dead
device). If an axis moves the wrong way, fix `[spacemouse] source` or `sign` and run it again; if the arm jerks, saturates or runs away, stop: that is plumbing (the station rate, the
action keys or the action units), not the policy, and no later step can fix it. Ctrl-C ends the run
and closes the devices.

### 2. Autonomous

```bash
uv run --extra pi mars-franka autonomous --prompt "put the red block in the bowl" --trace
```

The policy alone, no operator. It should move toward the object and pause once per chunk; the probe's
`chunk time` says how long that pause is. An arm that barely moves, or moves somewhere unrelated,
has a prompt, a camera or a state-vector problem, not a steering one: check the camera frames and the
probe's brightness numbers before changing anything.

### 3. Teleop with the study readout

```bash
uv run --extra pi mars-franka teleop --config /tmp/mars-franka-bringup.toml --corrupted
```

`--corrupted` applies `[readout]` (the 20° command rotation and the noise) that the shared modes use.
The run header prints the resolved corruption and the gain. The arm stays controllable at the low
gain; this step confirms the readout does not break teleoperation.

### 4. Flow control

```bash
uv run --extra pi mars-franka fc --prompt "put the red block in the bowl" --trace
```

While you push, the command is written into the chunk's initial condition; release and the policy
acts alone. The trace's `cmd[...]` column is the served command in action units and `act[...]` the
action. Expect a soft pull, weaker than FRS.

### 5. Flow reversal steering

```bash
uv run --extra pi mars-franka frs --prompt "put the red block in the bowl" --trace
```

Stronger authority than FC: the executed prefix is pinned to the reversal of your command for eight
actions, then the policy resumes. Release the stick and the arm continues on the policy.

### 6. The fixed-adapter arm

```bash
uv run --extra pi mars-franka mars-f --prompt "put the red block in the bowl" --trace
```

FRS with the session's corruption lifted into the 7×7 `F` and inverted as the adapter. Push the same
directions as in step 5 and compare: the corrupted command should be pulled back toward what you
intended. This is the Phase A method arm; the learned user module (`mars-u`) is Phase B, below, and
the task module (`mars-t` and the composed `mars`) is Phase C.

### 7. Record demonstrations

```bash
uv run --extra pi mars-franka teleop --record pilot
```

Plain teleop is the clean-readout path, so the demonstrations are the operator's own commands.
c3po's recording keys apply: `n` ends and keeps the episode, `r` discards it for a re-record, `q`
stops and finalizes the dataset. The station records a LeRobot v3.0 dataset on `bb8`; c3po downloads
it when the recording context exits, and the command prints `dataset: <path>`. `--record` works for
every session command; the probe refuses it because it records nothing. These are Phase C's
demonstrations: the dataset lands in `[recording] dir` and `[task] demos` points at it, so Phase C,
below, starts here.

## Phase B: train and validate the user module

`mars-u` steers with a *learned* user module instead of the fixed lift: it corrects the velocity field
of FRS's reversal, so the readout's corruption is undone for the operator's own commands. The module is
calibrated from cues -- random Cartesian pushes at recorded states, through the same bridge the operator
uses -- never from human labels. It is also calibrated in the study's prompt-conditioned field:
`train-user` and `validate-user` take the same `--prompt` a session command does, so a checkpoint
trained under one prompt is refused under another. Do these in order, and do not run `mars-u` in a study
until `validate-user` shows the module moving the executed heading toward the faithful command.

### 1. Collect the states

`--record-states NAME` writes one row per chunk boundary -- the observation the policy's chunk starts
from, the joints, the operator's deflection and the served command -- to
`assets/states/NAME/states.npz`, alongside whatever the station records. It changes no behaviour, and
it composes with `--record`.

Collect from the study scene, under the study's prompt, with the methods whose field the module will
correct: a few `frs`/`fc` sessions with the operator pushing, plus an `autonomous` session or two for
states the policy visits on its own (an operator cannot push into a state the policy never reaches).
A few hundred states are enough; each one is about 0.3 MB in the file (two 224x224 uint8 images plus
small arrays), so 300 states are about 90 MB.

```bash
uv run --extra pi mars-franka frs --prompt "put the red block in the bowl" --record-states pilot
uv run --extra pi mars-franka autonomous --prompt "put the red block in the bowl" --record-states pilot
du -sh examples/pi05/mars/franka/assets/states
```

The config that collected the states must have the same `[readout]` as the config that trains on them:
the trainer replays the config's corruption and noise at each row's pose and episode, and refuses a
file whose `states.npz` meta records another corruption, noise level or seed (or a `q0` that a
configured `[command] q0` does not reproduce). Collect with the study config, not the low-gain bring-up
copy; a variant file for the study that changes `[readout]` makes its states a different condition.

### 2. Train

`train-user` needs what a session needs -- the checkpoint, openpi's statistics, CUDA, the gated
tokenizer -- plus the states file. It loads pi0.5 once and runs the study's own velocity field.

```bash
uv run --extra pi mars-franka train-user --config examples/pi05/mars/franka/configs/franka.toml \
  --prompt "put the red block in the bowl"
```

It writes `assets/user/rotz20/user.pt` (the module `[modes.mars-u]` and `[user] checkpoint` name) and
`train_log.json` (the split, every logged train/held-out loss, and the provenance the checkpoint is
reproduced against: states, readout, prompt, module shape, seed). The asset is gitignored. Budget tens
of minutes on the 5090: the first pass encodes every state and reverses `[user] cues` cues through its
field (a few thousand velocity evaluations), then `[training] steps` batches run. Time the first run
and record it in the hardware facts table; `steps` is a config key, not a flag.

### 3. Validate, offline

```bash
uv run --extra pi mars-franka validate-user --config examples/pi05/mars/franka/configs/franka.toml \
  --prompt "put the red block in the bowl"
```

No arm, no station, no env: it reads `[user] states` and `[user] checkpoint`, evaluates the trainer's
own held-out split (the same seed and `val_frac`, so the module never saw those states) with one fresh
cue per held-out state, at each reversal depth in `[training] val_depths`, and writes
`assets/user/rotz20/validation.json`. Six arms are compared, all from the same flow noise:

| arm | what it steers |
| --- | --- |
| `plan` | the policy's own chunk: no command, the same noise |
| `none` | FRS on the corrupted command the readout serves |
| `clean` | FRS on a faithful readout's command, same noise draw (the corruption is the only difference) |
| `ideal` | FRS on the cue itself: no readout noise |
| `f` | FRS with the fixed lift `F`: the `mars-f` arm |
| `user` | FRS with the trained module |

The number is the mean cosine between the executed translation and the cue's intent twist: the
executed chunk is unnormalized, mapped through `J(q)` to a Cartesian twist and averaged over the
executed steps, and the cosine is against the intent twist's linear part. `+1` means the executed
motion heads where the operator meant, `0` perpendicular, `-1` away. The table prints one row per arm
and one column per depth:

```
arm           d10        d4
plan      +0.02     +0.02
none      +0.45     +0.44
clean     +0.71     +0.70
ideal     +0.93     +0.92
f         +0.55     +0.54
user      +0.68     +0.66
```

(the numbers above are illustrative of the shape, not a measurement: run it on chopper and keep the
JSON). The go/no-go is the reader's and `validate-user` always exits 0 once it has evaluated the
module: it is a measurement, not a pass/fail. `user` must sit between `none` and `clean` and closer to
`clean`, i.e. `|user - clean| < |none - clean|` at every depth; the footer prints exactly that
comparison. A module that does not correct the heading sits at `none` or falls back toward `plan`; one
that corrects the whole readout lands near `clean`, and `ideal` is the ceiling (`user` above `clean` at
a depth means the module is not the only difference between the arms -- check the checkpoint and the
states). If `user` is not closer to `clean` than `none` is, fix the data or the calibration and retrain
-- do not tune the module on the held-out states.

### 4. Then `mars-u` in a study

```bash
uv run --extra pi mars-franka mars-u --prompt "put the red block in the bowl" --trace
```

`[modes.mars-u] adapter_user` is the checkpoint `[user] checkpoint` names, and the command builds the
same MARS-U arm the LIBERO sessions do: FRS with the learned module in the reversal. Push the same
directions as in `frs` and compare: the corrupted command should be pulled back toward what you meant.
The module is a correction of the readout, not a task policy -- if a push still heads wrong, validate
first rather than adjusting the module by feel.

## Phase C: the task module

`mars-t` adds the *task* module: a learned adaptation of the policy's velocity field, conditioned on
the task and the scene and fit to demonstrations of the task. The task module is not a correction of
the operator's readout; it reshapes every chunk the frozen policy produces, in both flow directions,
so it is a policy-level adaptation and it is always attached to a base policy, never a method of its
own. `[task]` names the demonstrations, the output and the module's shape; `[modes.mars-t]` and
`[modes.mars]` name the checkpoint the trainer writes. This section makes that checkpoint and runs it.

### 1. Record demonstrations

The demonstrations are plain teleop: the clean readout, the operator's own commands, the station's own
cameras (step 7 of the bring-up order).

```bash
uv run --extra pi mars-franka teleop --record pilot
```

`n` ends and keeps an episode, `r` discards the take that just ended for a re-record, and `q` finishes
the session. The station records a LeRobot v3.0 dataset on `bb8`; when the recording context exits,
c3po downloads it and the command prints `dataset: <path>`. The dataset lands in `[recording] dir`, so
the shipped config's download is `examples/pi05/mars/franka/outputs/mars_franka/pilot/`.

The task module learns the grasp timing, so the recorded action has to carry the gripper. The shipped
station writes `{arm}/joint_velocity` (7) and `{arm}/gripper_position` (1), and the loader accepts the
single 8-wide `action` column too; it refuses a 7-wide action or a feature it cannot map by name
(Troubleshooting). Record with the shipped station and config: the dataset's schema is r2d2's, and the
first recording is where it is verified.

### 2. Point `[task]` at the dataset, and train

`[task] demos` has to name the downloaded dataset. Move the download under the example's gitignored
assets (the shipped value), or point `[task] demos` at wherever it sits:

```bash
mkdir -p examples/pi05/mars/franka/assets/demos
mv examples/pi05/mars/franka/outputs/mars_franka/pilot examples/pi05/mars/franka/assets/demos/pilot
```

Then train, with the study's prompt:

```bash
uv run --extra pi mars-franka train-task --config examples/pi05/mars/franka/configs/franka.toml \
  --prompt "put the red block in the bowl"
```

`train-task` reads `[task] demos`, the `[training]` loop and `[task]`'s module shape. It writes
`assets/task/pilot/task.pt` -- the checkpoint `[modes.mars-t]`, `[modes.mars]` and `[task] checkpoint`
name -- and `train_log.json` beside it: the dataset's provenance and digest, its held-out split, the
module's recorded scene (`token_source`, `render_size`, `scene_rank`), the checkpoint facts the module
was built for, and every logged loss. The prompt is part of the field: the checkpoint records it, and
`train-task` refuses to replace a module trained under a different prompt.

The cost is the scene. The task module reads the frozen VLM's image tokens of the scene camera, so
`train-task` needs `lerobot/pi05_base` (about 14 GB) for its `SceneEncoder`: a **second VLM resident in
the process**, beside `lerobot/pi05_droid` on the same GPU. Each training step encodes its minibatch's
scene pictures, one VLM encode per sample per step (only the last pictures are cached), then runs the
field; the same second VLM is resident at inference whenever `mars-t` or `mars` runs, so both the
training time and the card's memory carry it. If the weights are not cached, the first call refuses to
encode rather than use a randomly initialised VLM.

Read the held-out loss in `train_log.json`: `log` holds one entry every `[training] log_every` steps
and at the last, with `step`, `train_loss`, `val_loss` and `skipped`. `val_loss` is the flow-matching
loss on the demonstrations the module never saw (whole episodes when the dataset has several, else a
seeded row split, per `[training] val_frac` and `seed`), with the noise and flow times redrawn from
the seed so the curve is the module's. It is a fit diagnostic, not the study's answer: see the S2
design below.

Budget tens of minutes to an hour on the 5090 for the first run, and put the wall time and the final
held-out loss in the hardware facts table. `[training] steps` is a config key, not a flag.

### 3. Then `mars-t`

```bash
uv run --extra pi mars-franka mars-t --prompt "put the red block in the bowl" --trace
```

`[modes.mars-t] adapter_task` is the checkpoint just trained. The command hands it to
`policy.load_bundle(config, task=...)`, so the bundle carries the checkpoint's recorded scene: the
loop adds the ZED2 frame (letterboxed to `[task] render_size`) to every chunk's batch, the policy's
scene step encodes it into tokens, and the task module reads them. FRS + the task module: the executed
prefix is pinned to the reversed command, and every chunk the field produces is reshaped by the
module. The composed method is `mars`, FRS with the user module in the reversal *and* the task module
in the field (MARS is `user ∘ task`):

```bash
uv run --extra pi mars-franka mars --prompt "put the red block in the bowl" --trace
```

`[modes.mars]` names both checkpoints; a run names the first one that is missing.

### The S2 design

The task module is a policy-level adaptation, so the study block that measures it holds the module
constant across its arms and compares each arm with the same arm without it:

| arm with the task module | the same arm without it |
| --- | --- |
| `autonomous+task` | `autonomous` |
| `frs+task`, i.e. `mars-t` | `frs` |
| `mars` (FRS + user + task) | `mars-u` (FRS + user) |

The point of the pairing is that the only difference inside a row is the task module: the base policy,
the operator's readout and the prompt stay fixed. `mars-t` and `mars` are shipped commands.
`autonomous+task` is the task module on the base-policy anchor, and `frs+task` is the `mars-t`
composition under the block's name. Phase D, below, runs the whole block: `[study]` names its arms,
`study` runs the trials and the success entry, and `analyse` prints the tables.

**S2's verdict is task success on the robot.** There is no simulation-free proxy for it: the question
is whether the arm performs the task more often with the task module than without it. The offline
number is the held-out flow-matching loss in `train_log.json` and nothing else -- it says the module
fit the held-out demonstrations, not that the policy solves the task, so it gates a run (a module that
did not fit is not worth a block) but it is not the verdict.

## Phase D: running a study

A **study** is a block of trials. A trial is one **arm** (a command name: `autonomous`, `teleop`,
`fc`, `frs`, `mars-f`, `mars-u`, `mars-t`, `mars`, or any `[modes.<name>]` table) on one scene setup.
Rounds organise the block: round `r` runs every arm once, so the round number is the pairing key. The
operator resets the scene between trials and ends the trial; after each trial the operator answers one
question on the terminal -- did the task succeed? That y/n entry is the only human label the study
needs. The numbers are the apparatus check first and a study second: see "How to read the tables".

### The two blocks

`[study]` in the shipped config names S2's first row (`arms = ["frs", "mars-t"]`). S1 and the other S2
rows are variant configs -- copy the shipped file and edit `[study] arms`:

| block | arms | what a row compares |
| --- | --- | --- |
| S1: de-corruption | `frs`, `mars-f`, `mars-u` and the anchors (`autonomous`, `teleop`) | the same corrupted readout handled three ways: plain FRS, the fixed lift `F`, the learned user module |
| S2: task module, row 1 | `frs`, `mars-t` | FRS with and without the task module; the shipped block |
| S2: task module, row 2 | `mars-u`, `mars` | FRS+user with and without the task module |
| S2: the policy anchor | `autonomous`, and a `[modes.<name>]` table with `mode = "none"` and `adapter_task` | the base policy's own chunk with and without the task module; the table is how "policy + task module, no steering" runs without new code |

A `[modes.<name>]` arm is composed exactly as the command of that name would be: the same
`[readout]`, the same mode's adapter strings, and the `mars-f` table's `adapter_matrix` resolved by
the loop. Nothing about an arm is study-specific; the block only decides who runs when.

### Running one

The command takes the config and the task text; everything else is the `[study]` table's:

```bash
uv run --extra pi mars-franka study --config examples/pi05/mars/franka/configs/franka.toml \
  --prompt "put the red block in the bowl"
```

Before the first trial it loads each arm's checkpoint once (so a two-arm block holds two modules
resident) and opens the station and the SpaceMouse once. Then, for each trial in the round order:

1. it prints `trial k/n: arm <arm>, pair <r>`;
2. it resets the station and digests the reset observation;
3. the arm runs for `[study] max_steps` control steps under the prompt;
4. it asks `success? [y = yes, n/blank = no, q = stop]`.

`y`/`yes` records a success, anything else -- a blank line included -- a failure. `q`, Ctrl-C or
Ctrl-D stops the block: the trials whose outcome was entered stay recorded, the trial that was running
or unanswered is not, and the command exits 130. There is no `--steps`, `--record`, `--trace`,
`--dry-run` or `--corrupted`: the block comes from `[study]` and a study runs the configured sessions.

`round` is 0-based (`pair = 0` is the first round). `order = "shuffled"` draws each round's arm order
from `[study] seed`; `order = "blocked"` runs every trial of one arm before the next arm -- useful
for the apparatus check, where the operator wants to change one thing at a time.

**The scene resets between trials, not between arms.** The station's `reset()` is the episode signal,
and whether it physically restores the objects is a probe fact, not a promise: on a station whose
reset leaves the scene where the last trial put it, two trials that should pair do not. That is why
every trial records a scene digest and the analysis pairs only equal digests.

### What the records are

The block writes one run directory, `<[study] out>/<timestamp>_<config stem>/`:

```
outputs/mars_franka/study/20261125T101500Z_franka/
  trials.jsonl            one line per completed trial
  trials/frs-0.npz        one npz per trial: executed (T,8), plan (T,8), served (T,7), q (T,7)
  config.toml             the block's config file, byte for byte
  run.json                git HEAD and dirty flag, timestamp, arms, rounds, order, seed,
                          the config path and SHA-256, and at the end the recorded/aborted counts
```

Each `trials.jsonl` line has `run` (the run directory's name), `arm`, `pair`, `prompt`, `success`,
`steps`, `duration_s`, `authority`, `engaged_fraction`, `mean_r`, `q0` (the reset pose), `scene_hash`,
`config`, `config_sha256` and `timestamp`. A number the mode cannot measure is JSON `null`: a teleop
trial has no policy plan, so its authority is null; an autonomous trial has no command, so its
`mean_r` and authority are null, and its `engaged_fraction` is 0. In the npz, a value the mode does not
have is NaN in its array -- the shapes stay `(T, 8)/(T, 7)`, so a re-analysis can tell "not measured"
from a measured zero.

**Authority** is the study's one continuous metric: with `a` the executed action, `p` the base
policy's own chunk step for that tick and `r` the served command, all in action units,
`clip(<a-p, r-p> / ||r-p||^2, 0, 1)` on dims 0-6, over the steps where the command is above
`[command] deadband` (the operator was pushing). It is the fraction of the distance from the policy's
plan to the operator's command that the executed action actually travelled; 1 means the method
delivered the operator's command, 0 means it stayed with the policy. `engaged_fraction` is the share
of steps the operator was above the deadband, and `mean_r` the mean command magnitude over those
steps.

**Pairing** uses `(pair, scene_hash)`: two trials of different arms pair only if they ran the same
round and their reset observations digested the same. The digest is a SHA-256 of the rounded reset
joint angles and a 32x32 uint8 downsample of the exterior camera's frame, so a scene the station did
not restore is a different digest. The analysis reports the trials it could not match instead of
pairing them by round alone; that unmatched count is the honesty of the test, not a nuisance to hide.

### How to read the tables

```bash
uv run --extra pi mars-franka analyse --run examples/pi05/mars/franka/outputs/mars_franka/study/<run>
```

`--run` is repeatable and also takes a directory of runs (`--run .../study` reads every run under it);
the CSVs (`per_arm.csv`, `paired.csv`) land beside the runs unless `--out DIR` says otherwise.

The per-arm table is one row per (run, arm):

| column | meaning |
| --- | --- |
| `n`, `successes`, `success_rate` | the trials of that arm, how many succeeded, and the rate |
| `wilson_low`, `wilson_high` | the **Wilson score 95% interval** for that rate. Not the textbook normal interval: Wilson stays inside [0, 1] and does not collapse when an arm wins (or loses) every trial, which small blocks do |
| `n_authority`, `mean_authority`, `authority_se` | the mean authority over the trials with an engaged step and its standard error; `n_authority` says how many trials that mean is over |

The paired table compares every pair of arms **within a run**, on the matched `(pair, scene_hash)`
trials only, and says its `n_matched` and `unmatched`:

| column | meaning |
| --- | --- |
| `n_matched`, `unmatched` | how many trials paired (both arms started from the same round and scene) and how many did not; every other number in the row is over the matched trials |
| `success_a`, `success_b` | the two arms' success rates over the matched trials |
| `a_only`, `b_only`, `discordant` | the 2x2 table's discordant cells: trials where `a` succeeded and `b` did not, and the reverse. Only these carry information about a difference |
| `mcnemar_p` | **McNemar's exact two-sided test** (the binomial on the discordant pairs; no normal approximation, no scipy). `p = 2 * P(X <= min(a_only, b_only))` under the fair coin, conditioned on the discordant count; capped at 1, and no discordant pair is `p = 1` |
| `authority_diff` | mean paired difference (`b - a`) over the matched trials where both arms have an authority |
| `wilcoxon_p` | the **Wilcoxon signed-rank test** on those paired authorities: exact for at most 20 nonzero differences, otherwise the normal approximation with a continuity and a tie correction |

Neither test is a verdict. A Wilson interval at `n = 10` spans most of [0, 1], and McNemar only sees
the pairs where the arms disagreed, so a block's table tells you whether a difference is worth a
bigger block, not whether MARS works. Every number is **one operator's** -- the operator drives the
study arms and enters the outcomes -- and a handful of trials: read the block as an **apparatus check
first** (does the arm move at all, is the scene reproducible, are the digests pairing) and a study
second. A block where the anchors behave as expected and the digests pair is worth pooling into a
larger one; a block with a broken reset or a failed arm is a bring-up fact, not a result.

## Commands

```bash
uv run --extra pi mars-franka <command> [--config PATH] [--prompt TEXT] [--record NAME]
                                      [--record-states NAME] [--steps N] [--trace] [--dry-run]
                                      [--corrupted]
uv run --extra pi mars-franka analyse --run DIR [--run DIR ...] [--out DIR]
```

| command | what it runs |
| --- | --- |
| `teleop` | the operator alone: stick and gripper button, no policy. Plain teleop is the clean readout; `--corrupted` applies `[readout]` |
| `autonomous` | the policy alone: no operator device, the prompt is the task |
| `fc` | flow control (Kao et al.): the command sets the chunk's initial condition |
| `frs` | flow reversal steering (Tang et al.): the executed prefix is pinned to the reversed command |
| `mars-f` | FRS with the fixed corruption adapter (`[modes.mars-f] adapter_matrix = "corruption"`) |
| `mars-u` | FRS with the learned user module (`[modes.mars-u] adapter_user`, written by `train-user`); a run names the file while it does not exist |
| `mars-t` | FRS with the learned task module (`[modes.mars-t] adapter_task`, written by `train-task`); a run names the file while it does not exist |
| `mars` | the composed method, MARS = `user ∘ task`: FRS with both learned modules (`[modes.mars] adapter_user` and `adapter_task`) |
| `train-user` | Phase B's user-module trainer: reads `[user] states`, writes `[user] out/user.pt` and `train_log.json`; not a session |
| `validate-user` | Phase B's offline validator: evaluates `[user] checkpoint` on the held-out states and writes `[user] out/validation.json`; no arm, no station |
| `train-task` | Phase C's task-module trainer: reads `[task] demos`, writes `[task] out/task.pt` and `train_log.json`; not a session |
| `study` | Phase D's block runner: the `[study]` arms x rounds, the scene resets, the y/n outcome, and the run's records |
| `analyse` | Phase D's analysis: reads the runs' `trials.jsonl` and prints and writes the per-arm and paired tables; no config, no station |
| `probe` | the bring-up report; moves nothing and records nothing |

| flag | meaning |
| --- | --- |
| `--config PATH` | the config file; default `examples/pi05/mars/franka/configs/franka.toml` |
| `--prompt TEXT` | the task text; required by every policy command, optional for `teleop` and `probe` |
| `--record NAME` | record episodes and print the downloaded dataset path; refused by `probe` |
| `--record-states NAME` | keep the chunk-boundary states in `<config.states.dir>/NAME/states.npz`; every session command takes it |
| `--steps N` | stop after N control steps; non-recording runs only |
| `--trace` | print one line per tick: tick, mode, served command, action, the joints before it, gripper |
| `--dry-run` | fake station, source and bundle: no c3po, no torch, no checkpoint, no device |
| `--corrupted` | `teleop` only; the shared commands always apply `[readout]` |
| `--run DIR` | `analyse` only, repeatable: a run directory, or a directory holding run directories |
| `--out DIR` | `analyse` only: where `per_arm.csv` and `paired.csv` go (default: beside the runs) |

Ctrl-C ends a run and closes the station and the source; the exit status is 130 (`study` also returns
130 when the operator stops a block, with the finished trials recorded). A user error (config, missing
device, missing asset, missing c3po) prints one line to stderr and exits 2. `study`, `analyse` and the
trainers are not sessions: they refuse `--steps`, `--record`, `--record-states`, `--trace`, `--dry-run`
and `--corrupted`, and `analyse` also refuses `--config` and `--prompt` because it reads only the
records.

## Configuration

One config file per session: unknown keys are rejected, an absent key keeps the built-in default,
and the explicit offs are the strings `corruption = "none"` and `mode = "none"` (TOML has no null).
A variant file overrides individual keys, so a study condition is a file the repository keeps.

| section | knobs that matter, and where the number comes from |
| --- | --- |
| `[station]` | `host`/`port` (`10.42.0.1:9090`), `arm` (must be the single arm the probe prints), `rate = 15.0` |
| `[cameras]` | `exterior`/`wrist`/`third`: the feed names the probe's manifest prints. Absent `third` omits the image key, and the model fills it with a zero mask |
| `[policy]` | `path = "lerobot/pi05_droid"`, `stats = "assets/droid/norm_stats.json"`, `device = "cuda"`, `n_action_steps = 8`, `dtype = "bfloat16"` |
| `[command]` | `translation_gain = 0.11`, `rotation_gain = 0.5`, `twist_length = 0.2`, `deadband = 0.02`, `gain = "auto"`, `q0 = []` |
| `[readout]` | `corruption = {rotation_z_deg = 20}`, `noise = 0.35`, `seed = 0` |
| `[steering]` | `n_reversal_steps = 10`, `pinned_steps = 8` |
| `[modes.fc]` | `mode = "flow_control"`, `tau = 8`, `lam = 1`, `ic_gain = 15.0` |
| `[modes.frs]` | `mode = "frs"` |
| `[modes.mars-f]` | `mode = "frs"`, `adapter_matrix = "corruption"` (or `"corruption_inverse"`, or 7×7 rows) |
| `[modes.mars-u]` | `mode = "frs"`, `adapter_user`: the user module `train-user` writes; absent until then |
| `[modes.mars-t]` | `mode = "frs"`, `adapter_task`: the task module `train-task` writes; absent until then |
| `[modes.mars]` | `mode = "frs"`, `adapter_user` and `adapter_task`: the composed method, both modules; each absent until its trainer has run |
| `[states]` | `dir = "assets/states"`: a session's `--record-states NAME` lands at `<dir>/NAME/states.npz` |
| `[user]` | `states` (the file or the session directory), `out` (`user.pt`, `train_log.json`, `validation.json`), `checkpoint` (`validate-user` and the study read it), `cues`, `sigma`, and the module's shape (`horizon`, `d`, `tdim`, `time_embedding`, `step_rank`, `cross_step_rank`, `additive`, `correct_by`) |
| `[task]` | `demos` (the downloaded `teleop --record` dataset), `out` (`task.pt`, `train_log.json`), `checkpoint` (`[modes.mars-t]`/`[modes.mars]` read it), `token_source = "vlm_image"`, `render_size = 224`, `scene_rank`, `d`, `tdim`, `lambda_id` |
| `[training]` | `steps`, `batch`, `lr`, `val_frac`, `seed`, `log_every`, `val_depths`, `device`: the trainer's loop, its held-out split and the depths the validator runs |
| `[study]` | Phase D's block: `arms` (command names and `[modes.<name>]` tables, each must resolve), `trials` (the rounds: every arm once per round), `order = "shuffled"` or `"blocked"`, `max_steps` (a trial's cap, then the outcome question), `out` (the run-directory parent), `seed` (the shuffled order's) |
| `[spacemouse]` | `path`, `hid_id`, `full_scale`, `rotation_full_scale`, `source`, `sign`, `tare` |
| `[recording]` | `dir = "outputs/mars_franka"`, relative to this folder |

The provenance of the numbers, because the runbook is also the record:

- `[station] rate = 15.0` is the checkpoint's action rate [R]: π0.5-DROID was trained at 15 Hz, and
  one served action per control step is one trained step. The probe compares it with the station's
  native rate, and a native rate below 15 Hz is a failure of the loop's assumption, not a knob.
- `[command] translation_gain = 0.11` is FC's Appendix E value of about 11 cm/s at full deflection
  [R]. `rotation_gain = 0.5` is a feel knob: nothing measured it. `twist_length = 0.2` is the plant's
  0.2 rad per action unit [V], measured from the previous project's hardware logs and recorded in
  `flow-steering/docs/plant.md` (beside the lerobot checkout, not in this repository).
- `[command] deadband = 0.02` action units is the study's default. It is not measured.
- `[command] gain = "auto"` is `1 / ||F||_2` when the corruption is on, so the corrupted command
  cannot clip; clipping would break the lift that makes MARS's learning problem well-posed.
- `[readout] corruption = {rotation_z_deg = 20}` is the study's known corruption, written in LIBERO's
  command-corruption grammar. It is the condition `mars-f` inverts; it is not a perturbation measured
  on this arm.
- `[readout] noise = 0.35` is per joint in action-standard-deviation units (the statistics file
  supplies the std). The study sets it; no paper does.
- `[steering] n_reversal_steps = 10` is the full reversal and `pinned_steps = 8` the executed prefix,
  8/10 being Tang et al.'s pin-executed ratio [R].
- `[modes.fc] tau = 8`, `lam = 1`, `ic_gain = 15.0` are Kao et al.'s DROID numbers [R].
- `[policy] n_action_steps = 8` is the chunk prefix executed before the policy is recomputed. Nothing
  in the papers sets 8; it is a study choice, and the probe measures the pause it causes.
- `[spacemouse] full_scale` and `rotation_full_scale` ship as unverified 350.0 defaults. The probe
  replaces them with measured counts; until it does, the stick's deflection is a guess.
- `[spacemouse] tare = true` subtracts the device's rest six-vector, captured at start after the
  node's reports arrive and the stick settles. It is on because the session measured `x = +0.154` at
  rest: without the offset a released stick commands a slow drift. The offset is the sensor's, not
  the session's, so it should repeat; the probe prints it.
- `[user] cues = 3` and `sigma = [0.2, 0.5]` are the calibration's: three cue draws per state, their
  readout noise uniform in that action-std range. `validate-user` fixes the noise at the midpoint and
  draws one fresh cue per held-out state; these are study choices, nothing measures them.
- `[task] token_source = "vlm_image"` is the one scene source this phase implements: the frozen VLM's
  image tokens of the scene camera (the ZED2). `render_size = 224` is what the camera frame is
  letterboxed to before the encoder; the VLM resizes it to its own resolution, so the number is the
  batch's side, not the model's. `lambda_id = 1e-3` is main's pull of the module's `A, b` toward the
  identity (a fresh, zero-head module is plain FRS). `d`/`tdim` are the module's width and time
  embedding; `[task] checkpoint` is what the study's task arms load.
- `[training] val_frac = 0.1`, `seed = 0` and `val_depths = [10, 4]` are the split and the two reversal
  depths the validator reports: the full reversal (`n_reversal_steps`) and a partial one. The
  validator reuses the trainer's split, so the module never saw the evaluated states; `train-task`
  holds out the demonstrations with the same rule (whole episodes when there are several).
- `[training] steps = 3000` is a study choice; nothing in the papers sets it. The first training run's
  wall time belongs in the hardware facts table below.
- `[study] trials = 10`, `max_steps = 600` and the S1/S2 arm lists are study choices; nothing measures
  them. `order = "shuffled"` with `seed = 0` fixes the block's run order (each round is the seed's
  permutation of the arms), so a variant config re-running a block has the same order; `"blocked"`
  runs arm by arm for the apparatus check. The authority metric is DESIGN.md's formula, computed by
  the study from the executed actions, the policy's plans and the served commands it records -- it is
  a measurement of the delivered command, not a label.
- **Every study number is one operator's and a small block's.** The operator drives every arm and
  enters every outcome, so the per-arm rates are that person's; the Wilson intervals, McNemar and
  Wilcoxon are there to keep a small n honest, not to make it large. Read a block as an apparatus
  check first (is the scene reproducible, do the digests pair, do the anchors move) and a study
  second.

`[command] q0` is the session's measured reset pose, pinned so the corruption lift `F7`, the
`mars-f` adapter and the trainer's replay are the same 7×7 across sessions; `[]` would capture the
station's reset pose at connect, and a station whose reset pose drifts would move `F7` between
sessions. The probe prints both the pose and the 7×7 `F` it produces.

## Troubleshooting

- **`ModuleNotFoundError: c3po`** — install it editable from the lerobot checkout:
  `uv pip install -e ../c3po`. It is not on PyPI, so a fresh machine needs the clone.
- **The station is unreachable** — c3po raises `RobotNotFoundError` (or the probe's station section
  prints `[UNAVAILABLE] ImportError`/`RobotNotFoundError`). Check the Ethernet cable to `bb8`, that
  r2d2 is running, and `ping 10.42.0.1`. The probe's station section is safe to run until it passes.
- **A 401 from the tokenizer** — the PaliGemma tokenizer is gated: accept the license for
  `google/paligemma-3b-pt-224` on the Hub and run `hf auth login` (or set `HF_TOKEN`). Everything
  cached? Set `HF_HUB_OFFLINE=1` to skip the check.
- **`--dry-run` passes but the real run fails** — the dry run builds fakes: no c3po, no SpaceMouse, no
  checkpoint, no station. A real run needs all of them, plus the statistics file when the readout has
  noise. The dry run also runs `mars-t`/`mars` without their task checkpoints (they are gitignored), so
  it checks the composition, not the hardware or the assets.
- **`lerobot/pi05_base` is missing (the scene tokens)** — the task module's scene reads the frozen
  VLM's image tokens, and its `SceneEncoder` loads `lerobot/pi05_base` (about 14 GB) on first use, at
  `train-task` and at `mars-t`/`mars`. Without the weights it refuses to encode rather than run a
  randomly initialised VLM (`no model.safetensors for ...`). The repo needs the same Hub auth as
  `pi05_droid`; a cache already populated on `chopper` works offline. It is a second VLM in the
  process, so a CUDA out-of-memory on a run that used to fit is this, not the policy.
- **`train-task` refuses the demonstrations dataset** — the loader discovers the dataset's feature
  names and refuses what it cannot map, naming the feature or role: it needs one `<arm>/joint_position`
  observation, that arm's `gripper_position`, and an action that is either one 8-wide float `action`
  or `<prefix>/joint_velocity` plus its `gripper_position` sibling; each configured camera role
  (`[cameras] exterior`/`wrist`) must match an image or video feature. The schema is r2d2's and is
  verified on the first recording; a refusal naming the action schema means a dataset from a
  different station build or a recording made with a 7-wide action (the task module learns the grasp
  timing, so the gripper dim is required). Re-record with the shipped station and config.
- **`mars-t`/`mars` names a missing task checkpoint** — the tables name `[task] checkpoint`
  (`assets/task/pilot/task.pt`), which is gitignored and absent until `train-task --prompt "<task>"`
  has run (with `[task] demos` pointed at the recorded dataset). The error names the file and the
  command that writes it; a dry run does not need it.
- **`train-task` refuses a checkpoint trained under another prompt** — a task module shapes one
  prompt-conditioned field, so the checkpoint records the prompt it was trained with and `train-task`
  refuses to overwrite a module recorded under another one. Point `[task] out`/`checkpoint` elsewhere
  for a different task, or move the module.
- **`study` refuses a session flag** — `study` takes `--config` and `--prompt` only: the block's
  arms, rounds, order, cap and output parent come from `[study]`. If the flag you want is a trial
  count, set `[study] trials`; if it is a step cap, `[study] max_steps`.
- **`analyse` refuses a record field** — the analysis reads the Phase D record and refuses a legacy
  or foreign one by name rather than dropping the column. A `trials.jsonl` written by an older
  protocol (the LIBERO study's fields, or a draft's `mean_|r|`) is not this block's; point `--run` at
  the run directory `study` wrote.
- **The analysis reports unmatched trials** — the two trials of a pair did not start from the same
  scene: `pair` matches (the round) but `scene_hash` does not, which is `reset()` not restoring the
  objects, a moved camera, or a scene the operator changed between trials. Fix the scene reset or
  read the matched tables only; do not pair by round alone.
- **`study` shows the GPU out of memory at the second arm** — each arm's checkpoint is loaded once
  for the whole block and held, so a two-arm block with the task module resident (S2) holds two
  models. A 5090 fits the shipped blocks; if one does not, run the arms as two one-arm blocks and
  analyse the two run directories together (`--run` is repeatable), accepting that they no longer
  pair.
- **`teleop --corrupted` refuses to start with a missing statistics file** — the readout's noise is
  in action-standard-deviation units and the std comes from `assets/droid/norm_stats.json`. Plain
  `teleop` is clean and does not need it; `--corrupted` and every shared command do. Download the
  file as above.
- **The SpaceMouse is not found, or its axes are wrong** — the probe's SpaceMouse section lists every
  hidraw node with its `HID_ID` and marks the 3Dconnexion ones: no node means the device or the udev
  rule, a silent node means the stick or the wrong interface. Rotation is one check over both device
  layouts, so a `[FAIL] id 2 (rotation) reports` on the combined layout means neither the id 2 path
  nor the id 1 six-axis report's second triple (offset 7) moved. The probe names the node the source
  picks (`source would pick:`) and tallies `(id, byte length)` pairs: a six-axis single report is
  13 bytes, and its second triple is the rotation. Fix the direction with `[spacemouse] source`/`sign`
  after a low-gain jog; the probe cannot see a sign because that needs motion.
- **A pause every chunk** — expected in Phase A. The probe's `chunk time` is how long the
  pre-processor plus one denoising pass takes; when it exceeds one control period you get a pause
  every `n_action_steps` actions. The station measured **640 ms** (prepare 6 + act 634) for the
  autonomous chunk on the 5090, and at `n_action_steps = 8` the window is only 533 ms of actions, so
  the pause is already ~107 ms per chunk. The probe now also prints `steered chunk time fc` and
  `steered chunk time frs` after the autonomous chunk: each is the same prepare plus the mode's
  chunk-boundary act, and FRS's reversal does roughly twice the velocity evaluations, so those are
  the numbers that decide whether the 15 Hz loop stalls. The loop does not hide the pause (no
  prefetch, no compile, no real-time chunking in Phase A); it is a measured session fact, so compare
  a run's behaviour with the probe's numbers before looking for a fault. If the pause matters, raise
  `[policy] n_action_steps` toward the checkpoint's 15: the same chunk cost then spans 15 control
  periods instead of 8 (1000 ms instead of 533 ms), at the cost of a longer executed prefix before
  the operator's next command is read; `[modes.*] pinned_steps` (FRS's executed reversal prefix)
  follows it up so the same prefix is pinned.
- **`teleop --corrupted` refuses to start with a missing statistics file** — the readout's noise is
  in action-standard-deviation units and the std comes from `assets/droid/norm_stats.json`. Plain
  `teleop` is clean and does not need it; `--corrupted` and every shared command do. Download the
  file as above.

## Hardware facts this session ran under

Fill this in after the probe and the first teleop, and keep it with the run outputs. These are the
facts the code cannot know.

| fact | this session |
| --- | --- |
| date, host, station | |
| station native control rate (Hz) | |
| arms, action keys | |
| camera feeds (exterior / wrist / third) | |
| SpaceMouse node and HID id | |
| translation full scale (`[spacemouse] full_scale`) | |
| rotation full scale (`[spacemouse] rotation_full_scale`) | |
| `source` / `sign` | |
| rest-state six-vector | |
| gripper ends: the `gripper_position` value that closes the jaws | |
| reset pose `q0` | |
| `||F7||2`, auto gain, noise sigma | |
| chunk time (ms) at `n_action_steps`: autonomous / `fc` / `frs` | |
| third camera convention | |
| user module: states file, scene, collecting runs | |
| `train-user` wall time, final held-out loss | |
| `validate-user`: per-arm cosines at each depth (paste the table) | |
| `mars-u` session check (date, what the push did) | |
| demonstrations: `[task] demos` dataset, episode count, action schema it recorded | |
| `lerobot/pi05_base` cache location and size (the scene encoder) | |
| `train-task` wall time, final held-out flow-matching loss (`train_log.json`) | |
| `mars-t` / `mars` session check (date, what the task module changed) | |
| study block: date, arms, rounds, order, the run directory (`run.json`'s `run`, `git_head`, `git_dirty`) | |
| study block: operator (who drove every arm and entered every outcome; the numbers are per-operator) | |
| study block: scene reset -- does `reset()` restore the objects? (matched vs unmatched counts per pair) | |
| study block: what the per-arm and paired tables said (paste the two tables or the CSVs' paths) | |
| probe output | `examples/pi05/mars/franka/outputs/mars_franka/probe-session.txt` |
