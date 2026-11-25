# Running MARS on the Franka

This folder runs π0.5-DROID on an N-Droids Franka station through `c3po`, steered by a 3Dconnexion
SpaceMouse. The policy runs on `chopper` (the GPU machine, a 5090); the arm, the two ZED cameras and
the Robotiq gripper are on `bb8`, which runs the r2d2 server that c3po talks to at `10.42.0.1:9090`.
Phase A can do six things: `teleop` (the operator alone, and the demo-recording path), `autonomous`
(the policy alone), and `fc`, `frs`, `mars-f` (the three methods whose numbers already exist), plus
`probe` (the read-only bring-up report). It cannot train yet: the user module (`mars-u`), the task
module (`mars-t`) and the composed `mars` arm need checkpoints from Phases B and C, there is no study
protocol, success entry or analysis, and `configs/franka.toml` deliberately has no table for the
three missing arms so a run names the missing table instead of loading nothing. `DESIGN.md` is the
interface contract; this file is the runbook.

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

The first probe run downloads and loads the checkpoint, so the policy section can take minutes. Its
`chunk time` is the number that decides whether the chunk pause fits the control loop, so record it.

Copy the report's hardware facts into the config while the report is fresh:

| fact | what the probe prints | where it goes |
| --- | --- | --- |
| station native rate | `control rate: the loop asks for X Hz; the station's native rate is Y Hz` | `[station] rate`, if the rate check failed |
| arms and action keys | `arms:`, `action keys:` | nothing, unless the action-contract checks fail: that is a plumbing bug |
| camera feeds | `cameras:` | `[cameras] exterior`, `wrist`, and `third` when a third feed is wired |
| SpaceMouse node and HID id | `hidraw /dev/hidrawN: HID_ID ... (3Dconnexion)` | `[spacemouse] path` and `hid_id` in the session record; leave `""` to search |
| translation full scale | `copy the measured translation peak into [spacemouse] full_scale = N` | `[spacemouse] full_scale` |
| rotation full scale | `copy the measured rotation peak into [spacemouse] rotation_full_scale = N` | `[spacemouse] rotation_full_scale` |
| source and sign | `configured source [...] and sign [...]` | `[spacemouse] source`/`sign`, after a jog confirms a direction |
| rest-state six-vector | `rest-state six-vector [x y z rx ry rz]` | the session record; a non-zero axis is a held stick or a device that is not re-zeroing |
| reset pose | `reset pose q (...)` | the session record; `[command] q0` pins it for later runs |
| the lift and the gain | `q0:`, `F7 (7x7, ...)`, `gain:` | the session record; these are the readout the shared modes run |
| noise sigma | `noise sigma (7, action units)` | check it against `[readout] noise` |
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
press. If an axis moves the wrong way, fix `[spacemouse] source` or `sign` and run it again; if the arm jerks, saturates or runs away, stop: that is plumbing (the station rate, the
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
intended. This is the Phase A method arm; the learned user module and task module are Phase B/C.

### 7. Record demonstrations

```bash
uv run --extra pi mars-franka teleop --record pilot
```

Plain teleop is the clean-readout path, so the demonstrations are the operator's own commands.
c3po's recording keys apply: `n` ends and keeps the episode, `r` discards it for a re-record, `q`
stops and finalizes the dataset. The station records a LeRobot v3.0 dataset on `bb8`; c3po downloads
it when the recording context exits, and the command prints `dataset: <path>`. `--record` works for
every session command; the probe refuses it because it records nothing.

## Commands

```bash
uv run --extra pi mars-franka <command> [--config PATH] [--prompt TEXT] [--record NAME]
                                      [--steps N] [--trace] [--dry-run] [--corrupted]
```

| command | what it runs |
| --- | --- |
| `teleop` | the operator alone: stick and gripper button, no policy. Plain teleop is the clean readout; `--corrupted` applies `[readout]` |
| `autonomous` | the policy alone: no operator device, the prompt is the task |
| `fc` | flow control (Kao et al.): the command sets the chunk's initial condition |
| `frs` | flow reversal steering (Tang et al.): the executed prefix is pinned to the reversed command |
| `mars-f` | FRS with the fixed corruption adapter (`[modes.mars-f] adapter_matrix = "corruption"`) |
| `mars-u`, `mars-t`, `mars` | not built: Phase B/C adapters; the command names the missing `[modes.*]` table |
| `probe` | the bring-up report; moves nothing and records nothing |

| flag | meaning |
| --- | --- |
| `--config PATH` | the config file; default `examples/pi05/mars/franka/configs/franka.toml` |
| `--prompt TEXT` | the task text; required by every policy command, optional for `teleop` and `probe` |
| `--record NAME` | record episodes and print the downloaded dataset path; refused by `probe` |
| `--steps N` | stop after N control steps; non-recording runs only |
| `--trace` | print one line per tick: tick, mode, served command, action, the joints before it, gripper |
| `--dry-run` | fake station, source and bundle: no c3po, no torch, no checkpoint, no device |
| `--corrupted` | `teleop` only; the shared commands always apply `[readout]` |

Ctrl-C ends a run and closes the station and the source; the exit status is 130. A user error (config,
missing device, missing asset, missing c3po) prints one line to stderr and exits 2.

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
| `[spacemouse]` | `path`, `hid_id`, `full_scale`, `rotation_full_scale`, `source`, `sign` |
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

`[command] q0 = []` means the station's reset pose, captured once at connect. The probe prints that
pose and the 7×7 `F` it produces; set `q0` to seven joint angles to pin the lift when the station's
reset pose must not be trusted.

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
  noise. The dry run checks the composition, not the hardware.
- **The SpaceMouse is not found, or its axes are wrong** — the probe's SpaceMouse section lists every
  hidraw node with its `HID_ID` and marks the 3Dconnexion ones: no node means the device or the udev
  rule, a silent node means the stick or the wrong interface, and a missing id 2 means rotation is
  not arriving. Fix the direction with `[spacemouse] source`/`sign` after a low-gain jog; the probe
  cannot see a sign because that needs motion.
- **A pause every chunk** — expected in Phase A. The probe's `chunk time` is how long the
  pre-processor plus one denoising pass takes; when it exceeds one control period you get a pause
  every `n_action_steps` actions. The loop does not hide it (no prefetch, no compile, no real-time
  chunking in Phase A); it is a measured session fact, so compare a run's behaviour with the probe's
  number before looking for a fault.
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
| chunk time (ms) at `n_action_steps` | |
| third camera convention | |
| probe output | `examples/pi05/mars/franka/outputs/mars_franka/probe-session.txt` |
