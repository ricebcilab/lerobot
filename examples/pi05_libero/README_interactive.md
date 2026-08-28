# `interactive.sh` — pi0.5 LIBERO interactive runner

`interactive.sh` loads the pi0.5 policy once, streams a LIBERO scene live to
your browser, and drops you into a terminal REPL where each line you type runs
one rollout. Besides model-only control it offers four teleop / shared-control
modes driven by a 3Dconnexion SpaceMouse and/or the keyboard, optionally with
a deterministic corruption (a fixed 3×3 matrix) and/or Gaussian noise applied to
your command to emulate an imperfect operator.

```bash
./examples/pi05_libero/interactive.sh [flags forwarded to interactive.py]
```

Run it from anywhere; the script `cd`s to the repo root itself. First-time
setup (dependencies, LIBERO config, model download) is `./examples/pi05_libero/setup.sh`.

## What the launcher does

`interactive.sh` is a thin wrapper around `interactive.py`:

1. Exports the environment variables below (only if you have not set them).
2. `cd`s to the repository root.
3. Runs `uv run --locked --extra pi --extra libero python interactive.py "$@"` —
   every argument you pass is forwarded unchanged.

| Variable             | Default                         | Purpose                                               |
| -------------------- | ------------------------------- | ----------------------------------------------------- |
| `LIBERO_CONFIG_PATH` | `<repo>/.cache/libero`          | LIBERO config/asset directory (written by `setup.sh`) |
| `MUJOCO_GL`          | `egl`                           | Headless MuJoCo rendering (no display needed)         |
| `HF_HUB_CACHE`       | `<repo>/.cache/huggingface/hub` | Where the checkpoint is cached                        |
| `MPLCONFIGDIR`       | `<repo>/.cache/matplotlib`      | Keeps matplotlib from writing to `$HOME`              |

Override any of them by exporting before launching, e.g.
`MUJOCO_GL=osmesa ./examples/pi05_libero/interactive.sh`.

## Command-line flags

All flags are optional. `./examples/pi05_libero/interactive.sh --help` prints
the same list.

| Flag                                | Default                           | Meaning                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--config [PATH]`                   | off                               | Read the settings below from a YAML file; any flag you also pass on the command line wins. Without `PATH` the shipped [`config_interactive.yaml`](config_interactive.yaml) is used (see [Config file](#config-file))                                                                                                                                                         |
| `--policy-path PATH`                | `lerobot/pi05_libero_finetuned`   | Hub id or local directory of the pi0.5 checkpoint                                                                                                                                                                                                                                                                                                                            |
| `--suite NAME`                      | `libero_spatial`                  | Starting LIBERO suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_100`                                                                                                                                                                                                                                                              |
| `--task-id N`                       | `0`                               | Starting scene index within the suite (`tasks` in the REPL lists them)                                                                                                                                                                                                                                                                                                       |
| `--mode NAME`                       | `policy`                          | Control mode to start in (changeable live with `mode`)                                                                                                                                                                                                                                                                                                                       |
| `--tau N`                           | `5`                               | `shared_flow_control`: leading denoising steps your input steers                                                                                                                                                                                                                                                                                                             |
| `--n-action-steps N`                | `10`                              | Actions executed per predicted chunk before the model re-plans (chunk size is 50)                                                                                                                                                                                                                                                                                            |
| `--port N`                          | `8765`                            | Port of the live view, `http://localhost:<port>` (bound to 127.0.0.1)                                                                                                                                                                                                                                                                                                        |
| `--deterministic-corruption [PATH]` | off                               | Multiply Δx/Δy/Δz of your SpaceMouse/keyboard command by a fixed 3×3 matrix `M` read from a YAML file at every control step (`x → M·x`). Without `PATH` the shipped [`deterministic_corruption.yaml`](deterministic_corruption.yaml) is used. Can be changed live with `corruption` (see [Deterministic corruption](#deterministic-corruption))                              |
| `--flow-reversal-adapter [PATH]`    | off                               | Adapt the velocity field used by `shared_reverse_flow_steering`'s _reverse_ integration with a 7×7 matrix `F` read from a YAML file: `x_t += h · (F · v)` instead of `x_t += h · v`. Without `PATH` the shipped [`flow_reversal_adapter.yaml`](flow_reversal_adapter.yaml) is used. Can be changed live with `adapter` (see [Flow-reversal adapter](#flow-reversal-adapter)) |
| `--input-noise STD`                 | `0`                               | Std of the isotropic Gaussian noise added independently to Δx, Δy and Δz of your SpaceMouse/keyboard command at every control step (translation units, full deflection = 1). `0` = off. Can be changed live with `noise` (see [Input noise](#input-noise))                                                                                                                   |
| `--compile`                         | off                               | `torch.compile` the model as `eval.sh` does: the first rollout takes several minutes to compile, later ones are faster                                                                                                                                                                                                                                                       |
| `--output-dir DIR`                  | `outputs/pi05_libero_interactive` | Where each rollout's MP4 is written                                                                                                                                                                                                                                                                                                                                          |

Examples:

```bash
# Different scene and port
./examples/pi05_libero/interactive.sh --suite libero_object --task-id 2 --port 9000

# Your own checkpoint, compiled
./examples/pi05_libero/interactive.sh --policy-path outputs/train/pi05_v1/checkpoints/last/pretrained_model --compile

# Perturb your teleop command with noise of std 0.1 (10% of full deflection)
./examples/pi05_libero/interactive.sh --input-noise 0.1

# Corrupt your teleop command with the shipped matrix, or with your own file
./examples/pi05_libero/interactive.sh --deterministic-corruption
./examples/pi05_libero/interactive.sh --deterministic-corruption my_matrix.yaml --input-noise 0.05

# Adapt the flow reversal with your own 7x7 F
./examples/pi05_libero/interactive.sh --flow-reversal-adapter my_F.yaml
```

## Config file

Instead of passing flags you can keep the settings in a YAML file and start
with `--config [PATH]`:

```bash
./examples/pi05_libero/interactive.sh --config                    # config_interactive.yaml
./examples/pi05_libero/interactive.sh --config my_setup.yaml --port 9000
```

The shipped [`config_interactive.yaml`](config_interactive.yaml) lists every
supported key with its default. Precedence is **command line > config file >
built-in default**, a key set to `null` keeps the built-in default, and
unknown keys are rejected so a typo cannot silently do nothing. Both
corruption files stay off unless the config gives a path (or you load one in
the REPL).

For running scripted trials from a config instead of a REPL, see
[README_experiment.md](README_experiment.md).

## The live view

Open the printed URL (`http://localhost:8765` by default; VSCode forwards the
port automatically). The page shows:

- the agent-view camera stream;
- the current instruction, `step i / max`, rollout state
  (`running` / `SUCCESS` / `no success`) and the active mode. During an
  `experiment.py` run the human-facing task is shown on its own line with the
  VLA prompt beneath it, and both stay up for the whole trial;
- the executed action vector as signed bars: `Δx Δy Δz Δroll Δpitch Δyaw gripper`
  (env range −1…1; gripper −1 = open, +1 = close);
- the keyboard legend, the keys currently held, the keyboard's gripper state,
  the loaded corruption file, the input-noise σ and the loaded flow-adapter
  file when they are active, and a warning if the page does not have focus.

Keyboard control is captured **by this page**, so click it once before
driving. Keys are released automatically when the tab loses focus.

## The REPL

The prompt shows the current mode, e.g. `policy>`. Lines are interpreted as:

| Input                    | Effect                                                                                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| any text                 | Run one rollout with that text as the instruction to pi0.5                                                                                                                               |
| empty line               | Run one rollout with the scene's built-in instruction (in `teleop` mode: start driving)                                                                                                  |
| `tasks`                  | List the task ids and instructions of the current suite                                                                                                                                  |
| `task <suite> <id>`      | Switch scene, e.g. `task libero_spatial 3` (rebuilds the env, no model reload)                                                                                                           |
| `mode <name> [tau]`      | Switch control mode (see below); `tau` only applies to `shared_flow_control`                                                                                                             |
| `noise [std]`            | Show (no argument) or set the std of the noise added to your x/y/z command, e.g. `noise 0.1`; `noise 0` turns it off. Takes effect from the next rollout step                            |
| `corruption [path\|off]` | Show (no argument) the deterministic corruption matrix, load one from a YAML file (`corruption my_matrix.yaml`), or clear it (`corruption off`). Takes effect from the next rollout step |
| `adapter [path\|off]`    | Show (no argument) the flow-reversal adapter `F`, load one from a YAML file (`adapter my_F.yaml`), or clear it (`adapter off`). Takes effect from the next predicted chunk               |
| `quit`, `exit`, Ctrl-D   | Exit (env and server are shut down)                                                                                                                                                      |

Each rollout prints `step i/max` while running, then the outcome
(`SUCCESS` / `no success`), duration, video path, and mode-specific
statistics. Rollout length is the suite's default (≈14 s) except in `teleop`
mode, which allows 900 steps (45 s at the 20 Hz control rate).

### Modes

| Mode                           | Who drives     | What your input does                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `policy` (default)             | pi0.5          | Nothing — model-only rollout with your instruction                                                                                                                                                                                                                                                                                                                                                                |
| `teleop`                       | you            | Your x/y/z and gripper are executed directly (`ZeroPolicy`, the model is not loaded into the loop). Press Enter to start                                                                                                                                                                                                                                                                                          |
| `shared_override`              | pi0.5 + you    | Your Δx/Δy/Δz replace the model's translation in the executed action; rotation and gripper stay the model's                                                                                                                                                                                                                                                                                                       |
| `shared_flow_control [tau]`    | pi0.5, steered | Your Δx/Δy/Δz (normalized to the model's action space) are written into dims 0-2 of `x_t` for the first `tau` of the 10 flow-matching denoising steps (default `tau=5`, allowed 0…10); the rest denoise freely. The executed action is the model's own                                                                                                                                                            |
| `shared_reverse_flow_steering` | pi0.5, steered | Flow Reversal Steering ([Tang et al. 2026](https://arxiv.org/abs/2606.13675)): a reference chunk that servos in your direction at uniform velocity is integrated _backward_ through the model's velocity field for 10 steps to find its latent noise; the forward flow then starts from that noise instead of random noise. The executed action is the model's own, snapped to the nearest generalist action mode |

In every shared mode, idle input (inside the 0.05 deadband) means pure
policy — you can let go at any time. Switching to a non-`policy` mode
connects the SpaceMouse the first time (if one is plugged in) and prints the
keyboard legend.

After a rollout the shared flow modes print how much steering happened:
`shared_flow_control` reports the number of guided denoising steps;
`shared_reverse_flow_steering` reports the number of steered chunks and the
mean distance (in action-std units) between the executed translation and your
reference.

### Input sources

Both are active at once; the SpaceMouse wins whenever it is deflected.

**SpaceMouse** (3Dconnexion SpaceMouse Pro, read directly from `/dev/hidraw`,
no extra install): stick x/y/z → Δx/Δy/Δz, any button toggles the gripper. If
no device is found the mode still switches, keyboard-only.

**Keyboard** (in the browser page):

| Key                           | Action                                                                             |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `↑` `↓` `←` `→`               | forward / back / left / right (same directions as pushing the stick)               |
| `PgUp` / `PgDn`, or `W` / `S` | up / down                                                                          |
| `Space`                       | toggle gripper (used in `teleop`; the shared modes leave the gripper to the model) |
| hold `Shift`                  | full speed (keys move at half speed otherwise)                                     |

### Deterministic corruption

`--deterministic-corruption [PATH]` (or `corruption PATH` in the REPL) applies
a fixed 3×3 matrix `M` to the merged SpaceMouse/keyboard command at every
control step: `[Δx, Δy, Δz] → M · [Δx, Δy, Δz]`. It affects every mode that
uses your input (`teleop`, `shared_override`, `shared_flow_control`,
`shared_reverse_flow_steering`).

The YAML file holds `M` as three rows of three numbers (rows = output axes,
columns = input axes; `matrix:` is accepted as the key too, as is a bare list
of rows):

```yaml
# rotate the commanded direction 30° about z, halve the z speed
M:
  - [0.866, -0.5, 0.0]
  - [0.5, 0.866, 0.0]
  - [0.0, 0.0, 0.5]
```

The shipped [`deterministic_corruption.yaml`](deterministic_corruption.yaml)
is what `--deterministic-corruption` uses when no `PATH` is given. It keeps
both that rotation and the identity (a no-op) in the file with one of them
commented out — whichever rows are uncommented are the ones applied. Edit it
or point the flag at your own file. A malformed file (not 3×3, non-numeric,
non-finite) is rejected with an error.

- The product is clipped to the env's ±1 action range.
- Like the noise below, the matrix is applied only while you are actually
  commanding (clean input beyond the 0.05 deadband), so idle input stays
  exactly zero whatever `M` is.
- The gripper and the model's rotation are never affected.
- When both are active the corruption is applied first and the noise is added
  on top: `M · x + ε`.

The loaded file name is shown in the live view next to the keyboard legend
and by `corruption` in the REPL (which also prints `M`).

### Input noise

`--input-noise STD` (or `noise STD` in the REPL) perturbs the merged
SpaceMouse/keyboard command (after the deterministic corruption, if any), so
it affects every mode that uses your input (`teleop`, `shared_override`,
`shared_flow_control`, `shared_reverse_flow_steering`):

- At every control step (20 Hz in `teleop` / `shared_override`; once per
  predicted chunk in the two flow modes, which read your command when they
  re-plan) an independent sample from N(0, STD²) is added to each of Δx, Δy
  and Δz — isotropic, i.e. the same std on all three axes, fresh at every
  step.
- Units are those of the translation action: full stick deflection or a held
  key with `Shift` is 1.0, a held key without `Shift` is 0.5, so `0.1` is 10 %
  of full scale. The result is clipped to the env's ±1 range.
- Noise is added only while you are actually commanding (clean input beyond
  the 0.05 deadband). Idle input stays exactly zero, so letting go still hands
  control back to the policy in the shared modes and holds the arm still in
  `teleop`.
- The gripper and the model's rotation are never perturbed.

The current σ is shown in the live view next to the keyboard legend and by
`noise` in the REPL.

### Flow-reversal adapter

`--flow-reversal-adapter [PATH]` (or `adapter PATH` in the REPL) only affects
`shared_reverse_flow_steering`. That mode integrates your reference chunk
_backward_ through the policy's velocity field to find the noise it comes
from; the adapter left-multiplies that field with a fixed 7×7 matrix `F` at
every reversal step:

```
x_t  ←  x_t + h · (F · v(x_t, t))          instead of   x_t + h · v(x_t, t)
```

- Rows and columns are the env action dimensions in the policy's _normalized_
  action space — `Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper` (rows = output
  components of the adapted velocity, columns = input components). The
  padding dimensions (7…31) are never adapted.
- Only the **reversal** is adapted. The forward flow that produces the
  executed action still uses the policy's own field, so the executed action
  remains a genuine pi0.5 sample — `F` only changes which noise it starts from.
- `F = I` is an exact no-op, bit-identical to running without the flag.

The YAML file holds `F` as seven rows of seven numbers under an `F:` key
(`matrix:` and a bare list of rows are accepted too). The shipped
[`flow_reversal_adapter.yaml`](flow_reversal_adapter.yaml) is the identity
with commented suggestions, e.g.:

| `F`                                | Effect on the reversal                                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `diag(1.5, 1.5, 1.5, 1, 1, 1, 1)`  | Reverses translation faster — noise further from the policy's modes, stronger but less in-distribution steering |
| `diag(1, 1, 1, 1, 1, 1, 0)`        | Freezes the gripper dimension during reversal                                                                   |
| identity with rows 0 and 1 swapped | Inverts the reversal's x/y coupling                                                                             |

The loaded file name is shown in the live view and by `adapter` in the REPL
(which also prints `F` and flags the identity).

## Outputs

- One MP4 per rollout in `--output-dir`, named
  `YYYYMMDD_HHMMSS_<first-words-of-instruction>.mp4` (30 fps).
- Nothing else is written; the live view is not recorded separately.

## Tuning knobs (edit the source)

| Constant                   | File                                | Meaning                                                |
| -------------------------- | ----------------------------------- | ------------------------------------------------------ |
| `RATE_HZ = 20`             | `interactive.py`                    | Control loop cap; LIBERO's native rate                 |
| `TELEOP_MAX_STEPS = 900`   | `interactive.py`                    | Rollout length in `teleop` mode (must stay < 1000)     |
| `DEADBAND = 0.05`          | `interactive.py`, `teleop_input.py` | Below this the input counts as idle                    |
| `DEFAULT_SPEED = 0.5`      | `teleop_input.py`                   | Deflection produced by a held key without `Shift`      |
| `AXIS_SOURCE`, `AXIS_SIGN` | `spacemouse.py`                     | Stick-axis → action-axis mapping (keyboard follows it) |

## Troubleshooting

- **"SpaceMouse unavailable"** — not plugged in, or `/dev/hidraw*` is not
  readable; the mode still works with the keyboard.
- **Keys do nothing** — the browser page needs focus (it shows an orange hint
  until you click it), and a deflected SpaceMouse overrides the keyboard.
- **The arm wanders or goes the wrong way while I hold a direction** — check
  `noise` and `corruption` in the REPL (both are shown in the live view when
  active); `noise 0` and `corruption off` disable them.
- **`shared_reverse_flow_steering` behaves oddly** — check `adapter`; a
  non-identity `F` changes the reversal. `adapter off` restores the plain
  method.
- **Port already in use** — pass `--port`.
- **First rollout very slow with `--compile`** — expected; compilation takes a
  few minutes and the `No valid triton configs` messages are harmless.
- **Rendering errors** — try `MUJOCO_GL=osmesa` or make sure EGL drivers are
  installed.
