# MARS on a Franka arm

Shared autonomy for a π0.5-DROID checkpoint on an N-Droids Franka, driven by a 3Dconnexion SpaceMouse through
c3po. Four methods share one path: the policy alone, flow control (Kao et al.), flow reversal steering (Tang
et al.), and MARS — FRS with a reversal adapter, either a fixed `J⁺ M J` or a learned user module.

Nothing here is method code. The methods, the adapter holder and the readout live in
`lerobot.policies.steering` and `lerobot.teleoperators.readout`; this folder adds the ports a real station
needs: the transport, the arm's kinematics, the operator's device, and the trial record.

`DESIGN.md` records the decisions this folder implements. `DECISION.md` is the record a session fills in from
the hardware, and `../../../../franka-mars-plan.pdf` (in the workspace, not the repository) carries the
reasoning.

## Before the first run

- A station reachable over the network (c3po's default `10.42.0.1:9090`), a Franka on it, and the SpaceMouse
  readable. The udev rule for the SpaceMouse is in [`../README.md`](../README.md#spacemouse).
- The environment: `uv sync --extra pi --extra mars` (add `--extra dataset` to train a user module).
- c3po, which is not on PyPI: `pip install -e path/to/c3po`. Every command except `run.sh verify` imports it.
- The checkpoint downloads on first use. Its statistics do not ship with it: copy OpenPI's DROID
  `norm_stats.json` into `assets/droid/` ([`assets/README.md`](assets/README.md)).

## Slice 0: read the hardware before moving it

```bash
./run.sh probe --policy lerobot/pi05_droid --station 10.42.0.1
```

The probe loads no weights and moves nothing. It prints the checkpoint's action width, chunk, inference steps
and image keys; the station's arms, cameras, action keys, control rate, joint limits and whether `step(None)`
holds; and the SpaceMouse it found. Each half prints `unavailable: <reason>` rather than dying, so one missing
piece does not hide the other.

Paste the block into `DECISION.md` and answer the rest by hand — in particular the jog test, which is the only
way to know the stick's axis map and its full-scale count.

## The first contact: teleoperation only

```bash
./run.sh teleop --trials 1 --dry-run            # wires everything, prints actions, writes nothing
./run.sh teleop --trials 2 --max-steps 100      # moves the arm; the stick drives it
```

`teleop` has no policy in the loop: the stick's command goes through the same readout (corruption, gain,
noise) and the same clamps as every other arm, and the arm moves at `qdot_full` per full deflection. Press
`n` to end a trial, `q` to stop the run. Keep the e-stop in reach; this is the slice that proves the transport,
the pacing and the clamps before a learned component can move the arm.

## The arms

```bash
./run.sh floor     --trials 5 --task "put the block in the bowl"
./run.sh ceiling   --trials 5 --task "put the block in the bowl" --prompt "put the block in the bowl"
./run.sh fc        --trials 5 --task "..."
./run.sh frs       --trials 5 --task "..."
./run.sh mars_fixed    --trials 5 --task "..." --corruption '{rotation_z_deg: 20}'
./run.sh mars_learned  --trials 5 --task "..." --adapter outputs/user_modules/<name>/user.pt
```

| arm | what it is | what it isolates |
| --- | --- | --- |
| `teleop` | the stick alone | the interface's ceiling, and the transport |
| `floor` | the policy alone, prompt `do something` | the prior with no task knowledge |
| `ceiling` | the policy alone, the task's own prompt | the model's own competence |
| `fc` | flow control: the command in the initial condition | a soft write, no inversion |
| `frs` | the full or partial reversal, no adapter | the latent choice |
| `mars_fixed` | FRS with `F = J⁺ M J` at the start pose | a linear correction, wrong away from that pose |
| `mars_learned` | FRS with a trained `U(x_t, t)` | the correction that can follow the pose |

Knobs live in `configs/<arm>.yaml`, so a session's settings are a file the repository keeps. `--config` runs
any condition file; the runner writes the effective settings into the run's `config.yaml`. Two overrides are on
the command line because a block varies them: `--corruption` (anything the readout's grammar accepts) and
`--adapter` (a `user.pt` for `mars_learned`). `--trials`, `--max-steps`, `--prompt`, `--task`, `--start-pose`,
`--reset-tolerance` and `--seed` are the rest.

`mars_fixed` leaves `adapter_matrix: null` in its file on purpose: the runner fills it with `J⁺ M J` at the
start pose, from the same damped solve the arm uses, and records it in the trial. Its corruption comes from
`--corruption`, in the same grammar as the readout (`{rotation_z_deg: 20}`, `{scale: [2, 1, 1]}`, `{M: rows}`).

## The trial

The arm starts from the same joint configuration every trial (`--start-pose`, seven angles; `--reset-tolerance`
is how far the station's reset may sit from it). The scene is placed by hand; press the start key, and the
runner reads the stick once per action, holds that action for `ceil(control_rate / 15)` station ticks, and
records every station tick.

An action is the policy's own chunk step: seven joint velocities and a gripper position. The station
integrates the velocities, so the runner sends one action per action and repeats it for
`ceil(control_rate / 15)` station ticks, rather than sending position targets. The gripper is the policy's; the
operator's button only drives `teleop`.

A station that runs at the checkpoint's own rate (15 Hz) runs one action per tick, and the station keeps
applying the last velocity while the policy computes the next chunk. A 300 ms pause at 0.5 rad/s would drift
the arm 0.15 rad, so the runner sends one tick of zero velocity before every replan: the arm holds its
position through the pause instead of continuing. It costs one tick per chunk.

Success is labelled on the keyboard (`n`) at the moment it happens; `q` abandons a trial. Both are recorded, so
a session can be re-read without trusting a video.

## Records

Each run writes one directory, `outputs/pi05_mars_franka/<stamp>_<arm>/`:

- `config.yaml`: the effective settings, flat, with `mode`, `tau`, `lambda`, `n_reversal_steps`,
  `reversal_adapter` and `prompt` where [`../libero/notebooks/analyze.py`](../libero/notebooks/analyze.py)
  reads them.
- `trials.jsonl`: one line per trial — identity (`pair_id`, `initial_state_hash`), the arm's settings, the
  matrices in force, and the outcome (`success`, `aborted`, `steps`, `duration_s`, `user_reads`).
- `trial_XXX.npz`: per station tick — `action` (8), `policy_translation` (the policy's own plan, postprocessed),
  `user_translation` (the served command) and `user_translation_raw`, `user_gripper`, `user_reads`, `joint_pos`,
  `t`, plus the trial's scalars.

## Analysis

```bash
./run.sh report outputs/pi05_mars_franka/<run> [<run> ...]
```

One line per arm with its trials, successes, Wilson 95% interval and mean duration; the mean measured
authority over the pushing steps; and, with `--pair A B`, the paired difference between two arms on `pair_id`
where the two resets hash equal. A pair whose resets differ was not the same trial and is reported as
skipped.

Authority is read off the arrays rather than scheduled: with `a` the executed joint velocity, `p` the policy's
own plan and `r` the served command, it is the barycentric coordinate of `a` on the segment `p -> r`, clipped
to [0, 1]. It is the honest x-axis for comparing arms, because equal knob settings are not equal authority.

## Training a user module, offline

```bash
./run.sh train --dataset <recorded dataset> --policy lerobot/pi05_droid \
    --stats assets/droid/norm_stats.json \
    --prompt "put the block in the bowl" --deployment-prompt "do something" \
    --out outputs/user_modules/block --steps 3000
```

The trainer reads states from a dataset c3po recorded, labels each with the policy's chunk under the **real**
task prompt, converts the label to the operator's space through the Jacobian, serves the command through the
arm's readout, reverses the label under the **deployment** prompt, and fits `U(x_t, t) P(x_t, t) = z_L − û`.
No robot time. Set `--step-rank 3` first: the stick spans three Cartesian dims, so the correction should not
need more ([`DESIGN.md`](DESIGN.md)).

## What this folder does not do

- No task module: no π0.5 trainer for one exists, and both tasks here are pick-and-place.
- No `lerobot.robots` class: c3po is the transport, and its contract is joint positions and velocities.
- No ROS, no pinocchio, no asynchronous inference. The chunk pause is measured before any of those is added.

## Tests

```bash
./run.sh verify     # tests/examples/pi05_mars_franka and tests/policies/steering, no hardware
```

## Troubleshooting

- **`ModuleNotFoundError: c3po`** — install it from its source tree; it is not on PyPI.
- **"No SpaceMouse ... found"** — the udev rule, or plug it in; the runner still runs with a null source, which
  steers nothing. Every run prints `operator: <node>` or `operator: none (<why>)` before its first trial, and
  each trial prints `operator pushed on N/M ticks`, so a silent stick is visible without opening a file.
- **The stick is found but nothing moves** — a 3Dconnexion device has several hidraw interfaces and only one
  carries the translations. `run.sh probe … --stick 5` reads each candidate for five seconds: the node with
  `id 1` reports and a peak near 1.00 is the one. Pin it with `SPACEMOUSE_HIDRAW=/dev/hidrawN`.
- **`--stats: ... does not exist`** — copy OpenPI's `norm_stats.json` into `assets/droid/`; see above.
- **401 at the tokenizer** — `hf auth login`; the checkpoint's tokenizer is gated.
- **The arm stutters once per chunk** — the policy's pause is the denoising pass, and a `trial_XXX.npz`'s `t`
  column shows it: the gap between two actions longer than one action period is the chunk being computed. If it
  exceeds `n_action_steps / 15` seconds, lower `n_reversal_steps` or enable `compile` before adding a prefetch
  thread.
