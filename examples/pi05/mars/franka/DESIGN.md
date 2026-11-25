# MARS on a Franka arm — design (v2)

Status: interface freeze for Phase A. Owner: lead. Implementers: subagents, one slice each, on their own
branches. This file is the contract; a slice that must change it reports the change instead of making it.

This example runs π0.5-DROID on an N-Droids Franka station through `c3po`, under five methods plus teleop.
The methods are `lerobot.policies.steering`'s; this folder is plumbing. Replaces the `mars-franka` branch's
example, which drove its own clock and was never brought up.

## Non-goals (Phase A)

- No user-module training (Phase B), no task module, no scene tokens (Phase C).
- No prefetch, no `torch.compile`, no real-time chunking. A chunk pause every `n_action_steps` is expected
  and is measured by `probe`, not hidden.
- No trial protocol, success entry, or analysis. Phase A records nothing but the station's own dataset.
- No sim, no fake policy, no fake robot outside tests and `--dry-run`.

## The control model

**c3po owns the clock.** `robot.step(action)` sends the action and blocks until the station's next
observation; that is the control cycle. The loop never calls `time.sleep`, never counts ticks per action, and
never resends an action. The station is asked for the policy's action rate (`station.rate = 15.0`, one action
per control step), which is what the checkpoint was trained at.

One loop iteration:
```
obs (already in hand) -> command source -> method -> action8 -> obs = station.step(action8)
```
The first action of an episode is computed from the `reset()` observation.

## Units and the signal chain

| quantity | unit |
|---|---|
| SpaceMouse axes | deflection in [-1, 1] (raw counts / full scale) |
| twist | 3 linear (m/s) + 3 angular (rad/s) |
| joint velocity `qdot` | rad/s |
| **action unit** | the checkpoint's action scale: one unit = 0.2 rad per control step at 15 Hz = **3 rad/s** |
| normalized action | `(action - mean) / std`, the checkpoint's internal space (steering cores only) |

```
SpaceMouse 6 axes  (translation: command.translation_gain m/s at full deflection,
                    rotation:    command.rotation_gain rad/s at full deflection)
  -> twist6                        (m/s, rad/s)
  -> intent7 = IK(twist6, q)       damped least squares on the 6x7 geometric Jacobian,
                                   divided by ACTION_UNIT        [action units]
  -> served7 = clip(gain * F @ intent7 + sigma * eps, -1, 1)     [action units]
        F      = I + J+(q0) (M6 - I) J(q0), the fixed 7x7 corruption lift; I when corruption is off
        sigma  = noise * action_std, per joint                       (`readout.noise`, std units)
  -> method:
       teleop      action8 = [served7, gripper]                   (no policy)
       autonomous  action8 = policy.select_action(batch)          (no command)
       fc/frs/mars* command = served7 into SteeringPolicy.select_action(batch)
  -> action8 (8,) -> {arm}/joint_velocity = clip(action8[:7], -1, 1),
                      {arm}/gripper_position = 1.0 if action8[7] > 0.5 else 0.0
```

- **Why the corruption is lifted through `J(q0)`.** The command space is 7 joint velocities, but the
  operator only commands a 6-D twist whose image is a 6-D subspace that rotates with `q`. A 3x3 or 6x6
  matrix applied in the twist frame is pose-dependent in joint space, and the MARS user module cannot see
  `q`. Applying `M6` once, in the nominal commanded subspace at `q0`, makes the distortion a **fixed
  invertible 7x7** on the command: `F` is `M6` on `range(J+(q0))` and the identity on its orthogonal
  complement, so its exact inverse is `I + J+(q0)(M6^-1 - I)J(q0)`. The learning problem is then
  well-posed, which is the whole reason we chose command-space corruption.
- `J(q0)`'s linear block is rank 3 and the full 6-row block rank 6 (the arm is 7-DoF, one redundant dim).
- **`gain`** (`command.gain = "auto" | float`): `auto` is `1 / ||F||_2` when corruption is on, else 1.0, so
  the corrupted command cannot clip. Clipping is a nonlinearity that would break the lifting argument.
- **deadband** is in action units (`command.deadband`, default 0.02), not deflections: below it the operator
  counts as idle and the row gets the policy's own chunk (`SteeringConfig.deadband`).
- **`q0`** is `command.q0` (7 joint angles) when set, else the station's `reset()` pose, captured once at
  connect. `probe` prints it, and the session record names it.
- The **gripper** is the operator's in teleop (button: open/close) and **the policy's** in every shared mode
  (it is dim 7 of an 8-wide action; the command covers dims 0-6 only and the steering core delegates 7+).

## Files

```
examples/pi05/mars/franka/
  DESIGN.md          this file
  README.md          runbook: exact commands, config, bring-up order, hardware facts (slice A6)
  config.py          TOML -> frozen dataclasses; unknown keys rejected; `null` does not exist in TOML,
                     so absent = default and `corruption = "none"` / `mode = "none"` are the explicit offs
  bridge.py          pure numpy: Jacobian, IK, units, corruption lift, readout   (no c3po, no torch)
  station.py         c3po port + fake: reset/step/close/record/download, obs -> raw batch dict, action dict
  policy.py          pi0.5 loading, normalization-statistics install, SteeringPolicy per mode, teleop action
  source.py          the operator's device: the DeflectionSource port, the SpaceMouse and a fake
  loop.py            the one control loop, the record loop, the per-tick trace line
  cli.py             `mars-franka <command>`: argument parsing, config load, composition root
  probe.py           prints what the station and the checkpoint actually are; moves nothing
  configs/franka.toml  the study config (station, cameras, policy, command, readout, steering, [modes.*])
  assets/droid/norm_stats.json   gitignored (assets/.gitignore); from openpi-assets (see assets/README.md)
  assets/user/, assets/task/     Phase B/C checkpoints (gitignored)
src/lerobot/scripts/mars_franka.py   console-script shim (examples/ is not an installed package)
tests/examples/pi05_mars_franka/     offline tests, one file per module above
```

## Module interfaces (frozen)

### `bridge.py` — pure numpy, no I/O, no c3po, no torch

```python
ARM_DIMS = 7
TWIST_DIMS = 6
ACTION_UNIT = 3.0  # rad/s per action unit: 0.2 rad per control step at 15 Hz

def geometric_jacobian(q: NDArray) -> NDArray          # (6, 7), linear rows then angular
def flange_pose(q: NDArray) -> NDArray                 # (3,), the Jacobian's reference point
def joint_velocity(twist: NDArray, q: NDArray, length: float = 0.2) -> NDArray  # (7,) rad/s
def action_units(qdot: NDArray) -> NDArray             # (7,), rad/s -> action units
def operator_command(twist: NDArray, q: NDArray, length: float = 0.2) -> NDArray  # (7,) action units
def corruption_lift(matrix6: NDArray, q0: NDArray, length: float = 0.2) -> NDArray  # (7, 7)
def lift_inverse(matrix6: NDArray, q0: NDArray, length: float = 0.2) -> NDArray    # (7, 7)
def matrix6_from_spec(spec, rng=None) -> NDArray       # (6, 6); the three families, block-diag with I3
def readout(intent7: NDArray, matrix7: NDArray | None, noise_std7: NDArray | None,
            gain: float = 1.0, rng=None) -> NDArray    # (7,) served, clipped to [-1, 1]
def nominal_pose(obs: dict, arm: str) -> NDArray       # (7,)
```
`length` is the mixed-unit scale of the 6-row solve: the angular rows of `J` *and* the angular part of the
twist are scaled by `length` (a symmetric weighting, `min ||S(J qdot - twist)||`), so metres and radians enter
the least-squares with the same weight and `J qdot = twist` is unchanged. Damping (`lambda = 1e-2`) is used in
`joint_velocity` only; `corruption_lift`/`lift_inverse` use the undamped Moore-Penrose right inverse of the
same scaled Jacobian, because the exact-inverse property below needs `J J+ = I6`.
`matrix6_from_spec` accepts the repo's corruption grammar on the 3x3 translation block — `lerobot.teleoperators.readout.matrix_spec(spec, 3)`'s
forms: `{rotation_z_deg: d}`, `{scale: [sx, sy, sz]}`, `{M: rows}` (or bare rows), and `{random: {...}}` drawn
per trial from `rng` with LIBERO's semantics (`R(axis, angle) @ diag(gains)`, axis uniform on the sphere,
angle uniform in `rotation_deg`, gains log-uniform in `scale`). It lifts the block to 6x6 as `blkdiag(M3, I3)`
and rejects anything else (a curl or a quantize corruption has no matrix). Names follow LIBERO's: `M` is the
literal key, `rotation_z_deg` the 3-D rotation (`rotation_deg` is the 2-D plane's).
The Jacobian is copied from the hardware-validated implementation in `examples-from-unrelated-project/`
(mujoco_menagerie `panda.xml`, hinge about the *child* frame origin). Do not re-derive it.

### `station.py`

```python
@dataclass(frozen=True)
class StationManifest: arms: tuple[str, ...]; cameras: tuple[str, ...]; action_keys: tuple[str, ...]
                       control_rate: float

class Station(Protocol):
    manifest: StationManifest
    arm: str                                          # the single arm, validated
    def reset(self) -> dict[str, NDArray]
    def step(self, action8: NDArray) -> dict[str, NDArray]
    def close(self) -> None
    def cancel_episode(self) -> None                  # the recording loop's re-record
class RecordingStation(Station, Protocol):            # C3poStation; FakeStation is not one
    def record(self, name: str)                       # c3po's recording context manager
    def download(self, name: str, dest: Path) -> Path


class C3poStation:      # lazily imports c3po in __init__, so tests never need it
    def __init__(self, host: str, port: int, rate: float)
    # Station, plus:
    def record(self, name: str)          # c3po robot.recording(name) context manager
    def download(self, name: str, dest: Path) -> Path   # c3po robot.download_dataset, called before close()

class FakeStation:      # deterministic, for tests and --dry-run; records the actions it was sent
def action_dict(action8: NDArray, arm: str) -> dict[str, NDArray]
def observation_batch(obs: dict, *, arm: str, cameras: dict, task: str) -> dict
    # {"observation.images.base_0_rgb": float32 (3,224,224) in [0,1], exterior letterboxed,
    #  "observation.images.left_wrist_0_rgb": ... wrist ...,
    #  "observation.state": float32 (32,) = concat(joint_position (7,), gripper (1,)) then zeros,
    #  "task": task}
    # "observation.images.right_wrist_0_rgb" is OMITTED unless cameras["third"] names a feed.
def resize_with_pad(image: NDArray, size: int = 224) -> NDArray   # openpi's letterbox, PIL bilinear
```

### `policy.py` and `stats.py`

```python
@dataclass(frozen=True)
class Bundle:
    policy; preprocessor; postprocessor          # the real objects on the configured device
    mean: NDArray | None; std: NDArray | None    # action statistics, 8-wide raw units; None without a stats
                                                 # file (fake_bundle carries a stand-in std so dry runs exercise noise)
    deadband: float                              # the operator's deadband, action units
    def prepare(self, raw_batch: dict) -> dict   # the pre-processor: once per chunk, the expensive half
    def act(self, batch: dict, command7: NDArray | None) -> NDArray   # (8,) action units, no preprocessing
    def action(self, raw_batch, command7) -> NDArray                  # act(prepare(...)), the one-shot form
    def reset(self) -> None                      # a new episode: the base policy and the steering policy
def load_bundle(cfg) -> Bundle                   # checkpoint, processors, statistics; device and bf16 from cfg
def make_steering(bundle: Bundle, steering) -> SteeringPolicy   # once per session; knob-checked on build
def steering_batch(bundle: Bundle, raw_batch: dict) -> dict     # preprocessor(raw_batch); the command is added after
def teleop_action(served7: NDArray, gripper_closed: bool) -> NDArray  # (8,)
def fake_bundle() -> Bundle                      # numpy-only stand-in for --dry-run
```
- `stats.py` holds `install_stats(preprocessor, postprocessor, stats_path, dims=8)`: openpi's
  `norm_stats.json` -> the `q01`/`q99` statistics the checkpoint's `QUANTILES` mapping reads, on the
  pre-processor's normalizer for `observation.state` and `action` and the post-processor's unnormalizer for
  `action`, with the `features` entries the mixin needs to normalize an observation key; returns the action
  `(mean, std)` 8-wide via `FlowControlSteering.action_mean_std`.
- The **8 meaningful dims are sliced after** the post-processor: the unnormalizer's per-dim quantiles are what
  turn the model's normalized 32-wide tensor into station units, the gripper (dim 7) included.
- `n_dims = 7` (the command width) and `reference_step = 1.0` (one action unit per control step) are
  properties of this action space, not knobs — they are constants in `policy.py` with their provenance, not
  config fields. `deadband` comes from `[command]`.
- The steering policy is built once by `make_steering` and cached on the bundle; `Bundle.action` reuses it.

### `loop.py`

```python
class DeflectionSource(Protocol):
    def deflection(self) -> NDArray   # (6,) in [-1, 1], device axes [x, y, z, rx, ry, rz]
    def gripper(self) -> bool         # True = closed (a latched toggle)
    def reset(self) -> None           # a new episode: clear the latch
    def close(self) -> None
class SpaceMouseSource:               # wraps SpaceMouseTwist; the lerobot import is inside __init__
class FakeSource:                     # tests and --dry-run
class Loop:
    def __init__(self, *, config, station, mode: str, source=None, bundle=None, steering=None,
                 adapter_from_corruption: str | None = None)   # mode: teleop | autonomous | steered
    def run(self, *, prompt: str, max_steps: int | None = None, trace: bool = False) -> None
    def run_recording(self, *, name: str, dest: Path, prompt: str, trace: bool = False) -> Path
```
The loop owns the readout, not the caller: it resolves the corruption spec and `q0` at the first episode,
applies gain/noise, and redraws a per-trial `random` family at every episode — including the `mars-f` adapter
matrix, because that matrix *is* the session's corruption.
- The command is read **once per chunk** in the shared modes (`tick % n_action_steps == 0`) and **every
  tick** in teleop; the steering core reads the batch's command only when its queue is empty, so a
  per-tick read would change nothing. The whole served command — IK at the boundary pose, the corruption,
  the gain and **one noise draw** — is computed at that boundary and held for the chunk, so a seeded run's
  noise sequence does not depend on `n_action_steps`.
- `run_recording` follows c3po's documented `recording()` pattern (`n`/`r`/`q` flags, `cancel_episode`),
  takes a `RecordingStation`, downloads the dataset with its `download` after the recording context exits and
  before `close`, and returns its local path.
- The loop owns the readout and the adapter sync: it resolves the corruption spec and `q0` at the first
  episode, applies `gain = 1/||F7||2` (`auto`) and `sigma = readout.noise * action_std[:7]`, and redraws a
  per-trial `random` family (F7, gain and the `mars-f` adapter matrix) at every episode. It prints the resolved
  corruption, `q0`, `F7` and gain once, when it first resolves them.
- No sleeps, no counters of station ticks. `trace` prints one line per tick: tick, mode, the command in force
  for that chunk, the action, the joint positions before it, and the operator's gripper bit (a dash in the
  shared modes, where the gripper is the policy's).

### `cli.py`

```
mars-franka <command> [--config PATH] [--prompt TEXT] [--record NAME] [--steps N] [--trace] [--dry-run]
commands: autonomous, teleop, fc, frs, mars-f, mars-u, mars-t, mars, probe
```
`teleop` without `--corrupted` uses the **clean** readout (no corruption, no noise): that is the
demonstration-collection path. `teleop --corrupted` applies `[readout]`. Every shared mode applies
`[readout]`. `--dry-run` uses `FakeStation` and never imports c3po or torch, and prints every action.
`--config` defaults to `examples/pi05/mars/franka/configs/franka.toml`.

### What the checkpoint's own code settles (read from the artifact, [V])

These were checked against `lerobot/pi05_droid`'s `config.json`, the cached pre/post-processor JSONs, and
`modeling_pi05.py`; a slice may not re-open them by assumption:

- **Images**: the model does `img = img * 2.0 - 1.0` for SigLIP and then resizes with padding, so a batch
  image must be **float in [0,1]** (uint8 would be multiplied as 0-255 and destroyed). It accepts either
  channel order, and a client-side letterbox to 224 skips its resize. We send float32 `(3,224,224)`.
- **A missing image key is legal**: `PI05Policy._preprocess_images` fills absent `config.image_features` keys with
  an image of -1 and a **zero mask**. That is why the third slot is *omitted*, not filled with zeros — a
  zeros image would carry mask 1 and be a different input. `cameras.third` can name a feed to include it
  instead; the probe prints which convention a session ran.
- **State**: `observation.state` is declared `(32,)` and is **not** padded by the model, while the prompt's
  state tokens are the discretized values of every dim of it. So the station sends the 32-wide vector
  (8 real dims, zeros beyond), and the state statistics must be installed 32 wide.
- **Statistics**: `assets/droid/norm_stats.json` has **both** `state` and `actions`, each 32 wide with
  `q01`/`q99`/`mean`/`std`, meaningful in dims 0-7 and zero beyond. The checkpoint's `normalization_mapping`
  is `QUANTILES` for STATE and ACTION, and `NormalizerProcessorStep`/`UnnormalizerProcessorStep` read the
  stats named by that mode, so the install puts `q01` and `q99` on the pre-processor's normalizer and on the
  post-processor's unnormalizer for `action` and `observation.state` — not `mean`/`std`. `action_mean_std`
  (which the steering cores use) converts quantiles to its affine form itself.
- Actions are padded by the model (`pad_vector(batch[ACTION], max_action_dim)`), so the station sends 8 wide.

## Config schema (`configs/franka.toml`)

```toml
[station]   host = "10.42.0.1" ; port = 9090 ; arm = "follower" ; rate = 15.0
[cameras]   exterior = "scene_zed" ; wrist = "wrist_zed"        # third: absent = omit the third image key
[policy]    path = "lerobot/pi05_droid" ; stats = "assets/droid/norm_stats.json" ; device = "cuda"
            n_action_steps = 8 ; dtype = "bfloat16"
[command]   translation_gain = 0.11 ; rotation_gain = 0.5 ; twist_length = 0.2
            deadband = 0.02 ; gain = "auto" ; q0 = []
[readout]   corruption = {rotation_z_deg = 20} ; noise = 0.35 ; seed = 0
            # built-in defaults are neutral: corruption off, noise 0.0, seed 0. The values above are the
            # shipped study config's, written out; a variant file overrides them and writes "none" to clear.
[steering]  n_reversal_steps = 10 ; pinned_steps = 8
[modes.fc]      mode = "flow_control" ; tau = 8 ; lam = 1 ; ic_gain = 15.0
[modes.frs]     mode = "frs"
[modes.mars-f]  mode = "frs" ; adapter_matrix = "corruption"     # or "corruption_inverse" or 7x7 rows
[modes.mars-u]  mode = "frs" ; adapter_user = "assets/user/<condition>/user.pt"
[modes.mars-t]  mode = "frs" ; adapter_task = "assets/task/<task>/task.pt"
[modes.mars]    mode = "frs" ; adapter_user = ... ; adapter_task = ...
[spacemouse] path = "" ; hid_id = "" ; full_scale = 350.0 ; rotation_full_scale = 350.0
             source = [0, 1, 2, 3, 4, 5] ; sign = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
             # rotation_full_scale is a guess until the probe measures it; empty path/hid_id = search
[recording] dir = "outputs/mars_franka"
```
Unknown keys are rejected. Values are the method's own defaults unless the file sets them: FC's `tau=8`,
`ic_gain=15.0` are Kao et al.'s DROID numbers (flow-steering's constants); FRS's `n_reversal_steps=10` is
the full reversal and `pinned_steps=8` the executed prefix (Tang et al.'s pin-executed ratio).

### The station's action contract [V-R]

The action keys are `{arm}/joint_velocity` (7,) and `{arm}/gripper_position` (1,), the gripper a position
binarized at 0.5. That is not a guess: the previous project's client
(`examples-from-unrelated-project/droid.py`) sends exactly these two keys to this station, was run on it, and its
logs are the source of the plant measurements in `flow-steering/docs/plant.md` (bb8 integrates a joint-velocity
unit as `q_d = q + 0.2 v`). The probe prints the manifest's own `action_keys` on every session, so a station that
disagrees is named before anything moves. The camera feed names (`[cameras]`), the gripper's physical ends and
whether `reset()` restores the scene remain probe facts.

## The one `src/` change

`src/lerobot/policies/steering/`, default-preserving, so the LIBERO and cursor suites are bit-unchanged:

1. `SteeringConfig.reference_step: float | None = None`; `SteeringPolicy.core` passes
   `cfg.reference_step` when set, else today's expression.
2. `FlowControlSteering`'s written dims, when `operator_dims is None`, become `slice(0, cfg.n_dims)`
   instead of `slice(None)`. A no-op for the cursor (`max_action_dim == n_dims == 2`) and for LIBERO
   (`operator_dims = ["translation"]`).
3. `FlowReversalSteering` gains `command_dims: int | None = None` and `delegation_mask` honours it: when
   `operator_dims is None`, the dims from `command_dims` on are **delegated to the policy**. `core` passes
   `cfg.n_dims`. Today's mask with `operator_dims=None` is all-False — every dim pinned, which in a 32-wide
   action space would pin the gripper (dim 7) and the padding to the reference's zeros instead of letting the
   policy choose them. The 7-D operator space hides this because its command is the whole action width. A
   no-op for the cursor (its `mask[:, 2:]` is empty) and for LIBERO (the named-groups branch).

That is all. FRS with a 7-wide command, `reference_step = 1.0` and `n_dims = 7` then builds
`(command - mean) / std` over dims 0-6 on a 32-wide action space, pins those dims for the executed prefix,
and leaves the gripper (dim 7) and the padding to the policy.

## Slices and acceptance (Phase A)

| slice | files | acceptance |
|---|---|---|
| A1 steering patch | `src/lerobot/policies/steering/{configuration,modeling}_steering.py`, `tests/policies/steering/test_steering_action_space.py` | new tests: FRS reference from a 7-wide command on 32-wide stats equals `(command-mean)/std` in dims 0-6 and zero beyond, with `reference_step`; `n_dims=7` delegates dims 7+ and pins dims 0-6 for the executed prefix; FC writes dims 0..6 of `x_t`; cursor and LIBERO masks bit-identical; defaults unchanged (existing suites green) |
| A2 SpaceMouse rotation | `src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py`, `tests/teleoperators/test_spacemouse.py` | report id 2 parses to `(rx, ry, rz)`; `SpaceMouseReader.rotation`; 6-axis mapping class; additive (existing tests untouched and green) |
| A3 config + bridge | `examples/.../config.py`, `bridge.py`, `tests/.../test_config.py`, `test_bridge.py` | config parses the schema from a TOML written in the test, rejects unknown keys, and has neutral defaults; Jacobian agrees with a finite difference of `flange_pose` and with a pinned DROID-reset pose; `lift_inverse(F) @ F == I` for the three families; the three matrix specs equal `readout.matrix_spec`'s; IK round-trips a small twist; readout clips and seeds |
| A4 station | `examples/.../station.py`, `tests/.../test_station.py` | fake station round-trip; `observation_batch` keys/shapes/dtypes and the 32-wide state padding; `action_dict` clip/binarize; c3po never imported at module level |
| A5 policy | `examples/.../policy.py`, `tests/.../test_policy.py` | `install_stats` on the real `norm_stats.json` installs 32-wide `q01`/`q99` for `action` and `observation.state` on the right steps and returns the 8-wide action mean/std; a real `NormalizerProcessorStep`/`UnnormalizerProcessorStep` round-trip an 8-wide action through them; the full end-to-end pipeline test is skipped when the gated PaliGemma tokenizer is not cached (it runs on chopper) |
| A6 integration | `examples/.../loop.py`, `cli.py`, `probe.py`, `configs/franka.toml`, `README.md`, `src/lerobot/scripts/mars_franka.py`, `pyproject.toml` | loop drives a fake station and a fake sampler with exactly one `step` per action and no sleeps; every command and every mode reaches the right composition; `--dry-run` runs offline |

## Verification gates

- **Offline (me, before every merge):** `pytest tests/examples/pi05_mars_franka tests/policies/steering
  tests/teleoperators/test_spacemouse.py -q`, `ruff check`, `ruff format --check`, and `mars-franka <each
  shipped command> --dry-run --steps 4 --prompt ...`. [V]
- **Hardware (you, chopper), in this order:** `probe` -> `autonomous --prompt ...` -> `teleop` (6-DoF, low
  gain) -> `teleop --corrupted` -> `fc` -> `frs` -> `mars-f`; then `teleop --record <name>` for demos.
  The probe prints the station manifest, `q0`, `F`, the checkpoint facts, the action statistics, the chunk
  time on the 5090, and the third-camera convention, so the session record names what it ran under.

## Decisions record

1. c3po owns the clock: one action per `step`, station asked for 15 Hz, no sleeps, no held actions.
2. The command is 7 joint velocities in action units (IK of a 6-D twist); the gripper is the policy's in
   shared modes and the button's in teleop.
3. Corruption is applied in command space and lifted once through `J(q0)` to a fixed invertible 7x7, so
   MARS's user module has a well-posed target. Stick-space corruption was rejected as pose-dependent.
4. Study defaults: translation-only operator command (LIBERO-comparable); 6-DoF for teleop and demos.
5. Arms: `fc`, `frs`, `mars-f`, `mars-u`, `mars-t`, `mars` (composed), plus `autonomous` and `teleop`
   anchors; two study blocks (S1 de-corruption, S2 task module).
6. Phase B: cue-based target-command calibration on states from pilot rollouts; loss on all 7 dims.
7. Phase C: task module on teleop demonstrations, `n_dims = 8` (the gripper is dim 7), `vlm_image` scene
   tokens first.
8. Branch `mars-franka-v2` off `origin/main`; `origin/mars-franka` stays as the archive of the old attempt.
