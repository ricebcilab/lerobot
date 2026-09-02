# pi0.5 examples reorganization: merge the shared-autonomy branch into main

**Date:** 2026-09-01
**Status:** approved design, awaiting implementation plan
**Branch being merged:** `origin/pi05-libero-shared-autonomy-experiments` (3 commits on top of `main` at `b1962750`, fast-forwardable)

## 1. Goal

Merge the LIBERO shared-autonomy work into `main` and reorganize both pi0.5 pipelines
(feeding fine-tuning, LIBERO shared autonomy) under one base folder with modules cut by
responsibility, configs that state intent instead of numbers, no duplicated code between
entry points, and unit tests for the parts that can run without a GPU.

Behavior of the pipelines does not change. Every control mode, REPL command, config key,
launcher, and output file keeps its meaning. Only where things live and how they are
named and loaded changes.

## 2. Target layout

```
examples/pi05/
  README.md                        index: one paragraph + one command per pipeline
  feeding_finetune/                git mv of examples/feeding_pi05
    README.md
    convert_nwb_to_lerobot.py
    build_dataset.py               was build_dataset_parallel.py
    train.sh                       was train_feeding.sh
  libero_shared_autonomy/          git mv of examples/pi05_libero
    README.md                      setup, eval, interactive, experiment, analysis
    env.sh                         LIBERO/MuJoCo/HF/Matplotlib cache environment, sourced by launchers
    setup.sh  eval.sh  interactive.sh  experiment.sh
    configure_libero.py            unchanged
    teleop.py                      operator input (devices, merge, corruption, noise, recording)
    live_view.py                   browser window (frame stream, status, keyboard intake)
    session.py                     Session: policy + scene + teleop + live view + mode; rollout()
    config.py                      YAML -> settings (read, merge, schema, paths, matrix specs)
    interactive.py                 REPL over a Session
    experiment.py                  scheduled trials over a Session
    notebooks/
      analyze.py                   run dirs -> per-trial table + success statistics
      analyze_experiments.ipynb    outputs cleared; imports analyze.py and plots
    configs/
      interactive.yaml
      experiment/
        base.yaml
        flow_control_tau8.yaml
        flow_control_tau8_rotz20.yaml
        reverse_flow_full.yaml
        reverse_flow_full_rotz20.yaml
        reverse_flow_5steps_rotz20.yaml

src/lerobot/policies/pi05/steering.py       FlowControlPolicy, ReverseFlowSteeringPolicy,
                                            reverse_flow, ReversalAdapter, build_reversal_adapter,
                                            get_action_mean_std, TeleopSource protocol
src/lerobot/policies/pi05/modeling_pi05.py  the branch's noise_fn / flow_start_time /
                                            num_forward_steps hooks, unchanged

tests/policies/pi0_pi05/test_pi05_steering.py
tests/policies/pi0_pi05/test_pi05.py        + cases for the sampler hooks
tests/examples/pi05_libero_shared_autonomy/
  conftest.py                               adds the example folder to sys.path
  test_config.py  test_teleop.py  test_experiment.py  test_analyze.py
```

Removed: `examples/pi05_libero/README_interactive.md`, `README_experiment.md`,
`teleop_input.py`, `spacemouse.py`, `deterministic_corruption.yaml`,
`flow_reversal_adapter.yaml`, `config_interactive.yaml`, the five
`config_experiment_*.yaml` files, `analyze_experiments.ipynb` (replaced by the stripped copy
under `notebooks/`).

## 3. Naming rules

- Folders under `examples/pi05/` name the pipeline by what it does: `feeding_finetune`,
  `libero_shared_autonomy`.
- Modules name one responsibility: `teleop`, `live_view`, `session`, `config`.
- Entry points are verbs or nouns for what you run: `interactive.py`, `experiment.py`,
  `build_dataset.py`, `train.sh`, `eval.sh`, `setup.sh`.
- Experiment condition configs are `<mode>_<variant>[_<corruption>].yaml`. Modes are
  `flow_control` and `reverse_flow`; variants are `tau8`, `full`, `5steps`; the corruption
  tag is `rotz20`.
- A run directory is `<output_dir>/<YYYYmmdd_HHMMSS>_<config stem>`. The `experiment.name`
  key is removed; the file name is the name.
- Video and step files inside a run keep their names: `trial_XXX.mp4`, `trial_XXX.npz`,
  `trials.jsonl`, `config.yaml`.

## 4. Library code in `src/lerobot/policies/pi05/steering.py`

Moved from the example without behavioral change:

- `get_action_mean_std(postprocessor)`.
- `FlowControlPolicy(policy, source, tau, postprocessor)`.
- `reverse_flow(x, velocity, num_steps, adapter=None, n_reverse_steps=None)`.
- `ReverseFlowSteeringPolicy(policy, source, postprocessor, adapter=None, n_reverse_steps=None)`.
- The 7x7 adapter holder, renamed `ReversalAdapter` (was `FlowAdapter`), with `matrix`,
  `label`, and `describe()`.

New in this module:

- `class TeleopSource(Protocol)` with `translation -> np.ndarray` (shape 3, env units in
  [-1, 1]) and `gripper -> float` (-1 open, +1 close). The wrappers type their `source`
  argument with it, so the module has no import from `examples/`.
- `ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")` and
  `GRIPPER_DIM = 6`, replacing the example's `ACTION_LABELS` for the algorithm's needs.
  The display labels with Greek deltas stay in `live_view.py`.
- `DEADBAND = 0.05` as the single definition; `teleop.py` imports it.
- `build_reversal_adapter(spec, corruption=None) -> np.ndarray` (see section 6).

Not moved: anything that reads a config, prints, or knows about LIBERO.

The sampler hooks added by the branch in `modeling_pi05.py` are kept as they are. The
duplicated `adapter` paragraph in the old `interactive.py` docstring is dropped with the
rest of that docstring.

## 5. Example modules

### 5.1 `teleop.py`

Everything between the operator's hands and the policy. Merges the old `spacemouse.py`
and `teleop_input.py` plus `TeleopChain`, `MatrixReader`, and the corruption helpers
from `interactive.py`.

Contents, in order:

- SpaceMouse: `find_spacemouse_device`, `parse_report`, `normalize`, `SpaceMouseReader`,
  the axis tuning constants, `GRIPPER_OPEN`, `GRIPPER_CLOSE`.
- Keyboard: `KEY_PUSH`, `KeyboardReader`.
- `CombinedReader`.
- `CommandCorruption` (was `MatrixReader`): wraps a source, holds a 3x3 `matrix` and
  `label`, `describe()`. Applies `M @ translation` only while the source is beyond
  `DEADBAND`, clips to [-1, 1].
- `build_corruption(spec) -> np.ndarray` (see section 6).
- `NoisyReader`, `RecordingReader`.
- `TeleopChain(keyboard, input_noise=0.0)` with the same chain as today:
  `sources -> raw -> corruption -> noise -> served`. Exposes `reader` (what modes
  consume), `raw`, `served`, `corruption`, `noisy`, `attach_spacemouse()`.

### 5.2 `live_view.py`

The browser window. Contents: `PAGE` (HTML/JS, unchanged), `ACTION_LABELS` for display,
`FrameStream`, and `LiveView`:

```python
class LiveView:
    def __init__(self, port: int, keyboard: KeyboardReader, status_extra: Callable[[], dict])
    stream: FrameStream
    def start(self) -> None        # ThreadingHTTPServer on 127.0.0.1:port, daemon thread
    def close(self) -> None
```

`status_extra` is called on every `/status` request and merged into the status JSON; the
Session passes a callable returning `keys`, `keyboard_gripper`, `input_noise`,
`corruption`, and `flow_adapter`, so `live_view` does not import `teleop` beyond the
keyboard type or know about matrices.

### 5.3 `session.py`

The single object both entry points drive.

```python
@dataclass
class RolloutResult:
    success: bool
    steps: int
    frames: list[np.ndarray]
    metrics: dict          # steering counters for the mode that ran

class Session:
    @classmethod
    def from_settings(cls, s: SessionSettings) -> "Session"
        # loads policy + processors + autocast, builds TeleopChain, ReversalAdapter,
        # LiveView, and the first scene; applies s.control (mode, tau, n_reverse_steps,
        # input_noise, corruption, reversal_adapter)
    def set_scene(self, suite: str, task_id: int) -> None
    def set_mode(self, mode: str, tau: int | None = None, n_reverse_steps: int | None = None) -> None
    def rollout(self, prompt: str, recorder=None, max_steps: int | None = None) -> RolloutResult
    def announce_mode(self) -> str        # the text the REPL and driver print
    def mode_label(self) -> str           # short label for terminal and live view
    def resolved_matrices(self) -> dict   # for provenance
    def close(self) -> None
    # attributes: policy, chain, adapter, view, suite, task_id, task_description, mode, tau,
    # n_reverse_steps, list_tasks(suite)
```

`rollout` is the old `run_rollout` plus the old `ModeRunner.kwargs` dispatch. The
teleop-only path (`ZeroPolicy`, identity processors, `TELEOP_MAX_STEPS`) stays inside
`Session` as private helpers. `RATE_HZ`, `VIDEO_FPS`, `MODES` live here.

### 5.4 `config.py`

Pure functions, no torch or LIBERO import.

- `read_yaml(path) -> dict`.
- `deep_merge(base, overlay) -> dict` (overlay wins; a `null` in the overlay resets the
  key to the built-in default, matching today's "null keeps the default" rule).
- `flatten(data, schema, where) -> dict`, rejecting unknown keys as today.
- No path resolution is needed: after matrices moved inline, no config key names a
  file. `policy.path` may be a Hub id and is passed through untouched.
- `load_interactive_settings(path, cli_overrides) -> SessionSettings`.
- `load_experiment_settings(path, cli_overrides) -> ExperimentSettings`: reads
  `base.yaml` from the same directory as the condition file when the condition file has
  `extends: base.yaml` (the five shipped conditions do; a standalone file without
  `extends` still works).
- `SessionSettings` and `ExperimentSettings` are dataclasses with the same keys the two
  YAML schemas have today. `ExperimentSettings` embeds a `SessionSettings`.
- Range and enum validation from today's `validate_experiment_config`, minus the LIBERO
  task-count check, which needs the `libero` package. That check moves to
  `experiment.py` where the suite is listed anyway.
- Matrix specs (section 6) are parsed here into arrays before they reach `Session`.

### 5.5 `interactive.py`

Module docstring holds the REPL command reference (single source; the README links to
`--help`). `main()` parses flags, builds `SessionSettings`, constructs a `Session`, and
runs the REPL. Each REPL command becomes a small function taking the session and the
token list. Target size about 200 lines.

### 5.6 `experiment.py`

Module docstring describes the run directory format. Contents: `TASK_ORDERS`,
`PROMPT_FROM_TASK`, `build_schedule`, `TrialRecorder`, `prompt_for`, `parse_args`
(`--config`, `--n-trials`, `--seed`, `--mode`, `--output-dir`, `--port`, `--dry-run`),
`main()`. The trial loop is today's loop with the setup replaced by `Session.from_settings`
and each rollout by `session.rollout(...)`. The per-trial record keeps every field it has
today.

### 5.7 `notebooks/analyze.py` and the notebook

`analyze.py` exposes `load_runs(paths) -> pandas.DataFrame` (one row per trial from
`trials.jsonl`, with per-trial user reads derived from the `.npz` step arrays when the
jsonl predates the per-trial counter), `success_table(df)` (success rate with a Wilson
interval per run), and `compare(df, a, b)` (two-proportion test between two runs). The
notebook imports these and plots; its outputs are cleared before commit and the
`nbstripout` hook is added to `.pre-commit-config.yaml` scoped to `examples/**/*.ipynb`.

## 6. Matrix specification

Two matrices exist. They are named for what they do and specified by intent.

**Corruption** `M`, 3x3, env translation units: what the robot receives is
`M @ what the operator commanded`. Applied in `teleop.CommandCorruption`.

**Reversal adapter** `F`, 7x7, the policy's normalized action space: during reverse
integration only, `x_t += h * (F @ v)`. Applied in `steering.reverse_flow`.

YAML forms accepted for `control.corruption`:

```yaml
corruption: null                      # off
corruption: {rotation_z_deg: 20}      # rotate the commanded direction about z
corruption: {scale: [1, 1, 1]}        # per-axis gain
corruption: {M: [[..3 rows..]]}       # literal
```

YAML forms accepted for `control.reversal_adapter`:

```yaml
reversal_adapter: null
reversal_adapter:
  translation: corruption             # copy M from control.corruption (error if that is null)
  orientation: zero                   # identity | zero
  gripper: zero                       # identity | zero
reversal_adapter:
  translation: {rotation_z_deg: 20}   # or {scale: [...]}, or a 3x3 literal
  orientation: identity
  gripper: identity
reversal_adapter: {F: [[..7 rows..]]} # literal
```

`build_corruption(spec)` lives in `teleop.py`; `build_reversal_adapter(spec, corruption)`
lives in `steering.py`; the `translation: corruption` cross-reference is resolved by
`config.py`, which calls both. Both builders raise `ValueError` with the offending key in
the message. `rotation_z_deg` produces the exact `cos/sin` values rather than the rounded
three-decimal constants in the old files.

The REPL commands `corruption <file.yaml|off>` and `adapter <file.yaml|off>` accept a file
whose top level is one of the forms above. The old bare `M:` / `F:` files keep working
through the literal form.

Provenance: `config.yaml` in each run directory records the spec as written and the
resolved matrices as lists, as today.

## 7. Experiment configs

`configs/experiment/base.yaml`:

```yaml
experiment:
  n_trials: 10
  seed: 0
  task_order: random
  output_dir: outputs/pi05_libero_experiments
scene:
  suite: libero_goal
  task_ids: null
prompt: "do something"
policy:
  path: lerobot/pi05_libero_finetuned
  n_action_steps: 10
  compile: false
control:
  mode: shared_reverse_flow_steering
  tau: 5
  n_reverse_steps: null
  input_noise: 0.0
  max_steps: null
  corruption: null
  reversal_adapter: null
server:
  port: 8765
```

Each condition file is `extends: base.yaml` plus a `control:` block. For example
`reverse_flow_5steps_rotz20.yaml`:

```yaml
extends: base.yaml
control:
  mode: shared_reverse_flow_steering
  n_reverse_steps: 5
  corruption: { rotation_z_deg: 20 }
  reversal_adapter:
    { translation: corruption, orientation: zero, gripper: zero }
```

The five conditions reproduce the five files on the branch exactly (the branch's
corruption files carry 20 degrees; the same value is used here). `configs/interactive.yaml`
has the same `policy`, `scene` (with `task_id`), `control`, `server`, and `output_dir`
keys as today, in the new matrix form.

## 8. Launchers and docs

`env.sh` exports `LIBERO_CONFIG_PATH`, `MUJOCO_GL`, `HF_HUB_CACHE`, `MPLCONFIGDIR` with
the same defaults as today and creates the cache directories. `setup.sh`, `eval.sh`,
`interactive.sh`, `experiment.sh` source it and keep their current behavior and variables.
Paths inside `feeding_finetune/` scripts and README are updated to the new folder.

Docs: `examples/pi05/README.md` (index), `feeding_finetune/README.md` (moved, paths
fixed), `libero_shared_autonomy/README.md` (the current README with the interactive and
experiment sections merged in from the two READMEs being removed, trimmed of the REPL
reference, which now lives in `interactive.py --help`). `AGENT_GUIDE.md` and `CLAUDE.md`
are checked for references to the old paths and updated if any exist.

## 9. Tests

`tests/policies/pi0_pi05/test_pi05_steering.py` (CPU, no checkpoint):

- `reverse_flow` with a linear velocity field `v = a * x`: reversing then integrating
  forward with the same step count returns the input to float tolerance; with
  `n_reverse_steps = k` the result equals the closed-form partial integration.
- The adapter is applied only to the first 7 dims; identity adapter equals no adapter.
- `build_reversal_adapter` for each YAML form, including the error cases.
- `FlowControlPolicy`: with a fake policy whose `predict_action_chunk` records hook calls,
  the hook writes normalized translation into dims 0-2 for steps `< tau` only and skips
  idle input.
- `ReverseFlowSteeringPolicy`: with a fake policy, `flow_start_time` and
  `num_forward_steps` are passed only when steering, `n_reverse_steps` validation, gripper
  reference tracking after a chunk.

`tests/policies/pi0_pi05/test_pi05.py` gains cases for the sampler: `noise_fn` receives a
callable velocity and its return replaces the starting noise; `flow_start_time` and
`num_forward_steps` change the step size so the schedule lands at `t=0`; `num_forward_steps
< 1` raises. These use whatever small-config pi0.5 fixture the file already has; if none is
CPU-runnable, the tests are marked with the file's existing GPU skip.

`tests/examples/pi05_libero_shared_autonomy/`:

- `test_config.py`: deep merge, unknown-key rejection, `extends`, path resolution relative
  to the config file, every shipped condition file loads and matches the expected resolved
  `control` block, `interactive.yaml` loads.
- `test_teleop.py`: `KeyboardReader` staleness and toggle parity, `CombinedReader`
  priority, `CommandCorruption` deadband and clipping, `build_corruption` forms,
  `NoisyReader` seeded, `RecordingReader` counts, `TeleopChain` ordering.
- `test_experiment.py`: `build_schedule` for the three orders, `TrialRecorder` row shape
  and per-trial read counting, `prompt_for`.
- `test_analyze.py`: a synthetic run directory with two trials round-trips through
  `load_runs` and `success_table`.

Runtime verification, done by hand on the GPU box after the code lands:
`experiment.py --dry-run` on all five conditions, one short interactive rollout in each of
the five modes, and one two-trial experiment with `reverse_flow_5steps_rotz20.yaml`.
`pre-commit run --all-files` passes.

## 10. Merge and history

1. Merge `origin/pi05-libero-shared-autonomy-experiments` into `main` (a regular merge;
   `main` gained the spec commit after the branch point).
2. Create branch `reorg-pi05-examples` (in place; the checkout had no venv to duplicate).
3. Reorganize in a series of commits, each leaving tests green:
   move feeding; move libero folder and add `env.sh`; extract `steering.py` with tests;
   `teleop.py`; `live_view.py` and `session.py`; `config.py` and configs; rewrite entry
   points; notebooks; docs. Use `git mv` so blame follows.
4. Merge `reorg-pi05-examples` into `main` after the runtime verification.

## 11. Out of scope

Switching the CLI to draccus, changing any control mode's mathematics, touching the
feeding pipeline's behavior, the sibling repos referenced by the feeding README, and
adding new experiment conditions.
