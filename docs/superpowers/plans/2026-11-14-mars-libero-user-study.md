# MARS LIBERO User Study (build and stage 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build what study 1 needs in the LIBERO example and run its stage 0 (anchors), ending in a report at the stage 0 gate.

**Architecture:** The study is condition files plus a launcher over the existing `experiment.py`; the only new method code is one readout stage (a `quantize` corruption) in the shared `Readout`. The runner gains a per-task state offset for disjoint splits, the analysis a wrong-target measure, and the user-module trainer three fixes it needs before stage 2.

**Tech Stack:** Python 3.12+, PyTorch, numpy, pandas, bash; `uv run`; LIBERO through `lerobot.envs.libero`; pi0.5 `lerobot/pi05_libero_finetuned`.

**Spec:** `docs/superpowers/specs/2026-11-14-mars-libero-user-study-design.md`

## Global Constraints

- Work on `main`; no commit, no push, no branch or worktree (the author's standing instruction). Tasks end with tests and hooks, not a commit.
- Nothing computes differently for existing conditions: without the new keys every existing config, record and seed stream is unchanged.
- Tests on the CPU: `CUDA_VISIBLE_DEVICES="" uv run pytest <files> -q -p no:cacheprovider`. One behaviour test per new piece.
- Hooks on every touched file: `uv run pre-commit run --files <files>`.
- `lerobot.teleoperators` is mypy-strict: annotate what is added there.
- Users (spec): U1 one random 3-D rotation, 60 degrees, axis uniform on the sphere, seed 0, fixed for the user; U2 `{scale: [1, 1, 0]}`; U3 `{curl: 2, tau: 0.5}`; U4 `{quantize: axes}`. No fixed-inverse arm.
- Operator: `operator: policy`, `update_every_steps: 20`, `delay_steps: 4`; the immediate oracle is `update_every_steps: null`, `delay_steps: 0`.
- Suites: `libero_object` all ten tasks; `libero_goal` tasks `[1, 4, 5, 6, 7, 8]`.
- Stage 0 only is run by this plan. Stages 1 to 3 wait for the author's go at each gate.
- No code under `src/lerobot` or `examples/pi05/mars/libero` is edited while a study run is in progress (each block is a new process importing the working tree).

## Review Focus

- A push whose two largest components tie: `quantize` must pick one axis deterministically (the first), not split.
- An idle push through `quantize`: stays exactly as it is, as every other readout stage leaves it.
- `first_occurrence` larger than a task's number of initial states: wraps through reshuffled passes exactly as a long run does, never an index error.
- A condition with `{scale: [1, 1, 0]}` and `input_gain: auto`: loads, and a purely vertical push is served idle.
- A study launch whose run fails (out of memory, port in use): the launcher reports the failed job by name and the others continue.

---

### Task 1: The `quantize` readout stage

**Files:**

- Modify: `src/lerobot/teleoperators/readout.py`
- Test: `tests/teleoperators/test_readout.py`

**Interfaces:**

- Produces: `quantize(raw, deadband: float = DEADBAND) -> np.ndarray`; `quantize_spec(spec: Any, where: str = "control.corruption") -> bool`; `Readout(..., quantize: bool = False)`. `corruption_spec` raises `ValueError` on a spec with a `quantize` key (it has no matrix or curl to return).

- [ ] **Step 1: Failing test** `test_quantize_snaps_an_active_push_to_its_largest_axis`:

```python
def test_quantize_snaps_an_active_push_to_its_largest_axis():
    raw = np.array([[0.3, -0.7, 0.2], [0.5, 0.5, 0.1], [0.01, 0.02, 0.0]])
    np.testing.assert_array_equal(readout.quantize(raw), [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.01, 0.02, 0.0]])
    assert readout.quantize_spec({"quantize": "axes"}) and not readout.quantize_spec({"scale": [1, 1, 0]})
    with pytest.raises(ValueError, match="quantize"):
        readout.quantize_spec({"quantize": "octants"})
    with pytest.raises(ValueError, match="quantize"):
        corruption_spec({"quantize": "axes"}, n_dims=3)
    arm = Readout(quantize=True, n_dims=3, noise_gate="command")
    np.testing.assert_array_equal(arm(raw[0]), [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(arm.core(raw[None]), readout.quantize(raw)[None])
    with pytest.raises(ValueError, match="not both"):
        Readout(np.eye(3), quantize=True, n_dims=3)
```

- [ ] **Step 2:** Run it; expect failure (`quantize` undefined).
- [ ] **Step 3:** Implement. `quantize`: where a row is active, a vector with the sign of the largest-magnitude component (first on a tie) at that index and zeros elsewhere; idle rows unchanged. `Readout`: `quantize` excludes a matrix and a curl; `read` and `core` apply `quantize` in place of the corruption (noise after it, as for the others); `chunk` raises for it. Update the module and class docstrings.
- [ ] **Step 4:** Run `tests/teleoperators/test_readout.py`; all pass. Hooks.

### Task 2: The LIBERO harness accepts `{quantize: axes}` and a singular matrix

**Files:**

- Modify: `examples/pi05/mars/libero/config.py` (`ControlSettings.corruption_quantize`, `_RESOLVED`, `build_control`, `largest_gain`), `teleop.py` (`ReadoutReader.quantize`), `session.py` (`apply_control`, `resolved_matrices`), `README.md` (the corruption forms)
- Test: `tests/examples/pi05_mars_libero/test_config.py`, `test_teleop.py`

**Interfaces:**

- Consumes: Task 1.
- Produces: `ControlSettings.corruption_quantize: bool`; `ReadoutReader(..., quantize: bool = False)` with a `quantize` property (a setter that rebuilds the shared readout as `curl`'s does, and refuses a matrix or curl alongside).

- [ ] **Step 1: Failing tests.** In `test_config.py`, `test_quantize_and_singular_corruptions_resolve`: a condition with `control.corruption: {quantize: axes}` gives `corruption_quantize is True`, `corruption_matrix is None`, `corruption_curl is None`, `resolved_input_gain == 1.0` under `input_gain: auto`; one with `{scale: [1, 1, 0]}` gives a matrix `diag(1, 1, 0)` and gain 1.0. In `test_teleop.py`, `test_a_keypad_readout_snaps_and_a_dropped_axis_goes_idle`: a `ReadoutReader(Source((0.3, -0.7, 0.2)), quantize=True).translation == [0, -1, 0]`; a reader with `matrix=diag(1, 1, 0)` over `Source((0.0, 0.0, 0.8))` serves a command inside `DEADBAND`.
- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement: `build_control` reads `readout.quantize_spec` first (then no matrix, curl or family; the adapter basis is the identity); `largest_gain` is 1.0 for it; `apply_control` puts it in force; provenance records `corruption_quantize`.
- [ ] **Step 4:** Run the two files; hooks.

### Task 3: Disjoint splits (`experiment.first_occurrence`)

**Files:**

- Modify: `examples/pi05/mars/libero/reproducibility.py` (`trial_specs`), `config.py` (`ExperimentSettings.first_occurrence: int = 0`, schema, validation), `experiment.py` (`build_trial_specs`, `_provenance`), `README.md` (the configuration table)
- Test: `tests/examples/pi05_mars_libero/test_experiment.py`

**Interfaces:**

- Produces: `trial_specs(schedule, suite, seed, state_counts, first_occurrence: int = 0)`: each task's kth trial is its occurrence `first_occurrence + k`; occurrence `j` has the same state and seeds whatever `first_occurrence` it is reached from.

- [ ] **Step 1: Failing test** `test_first_occurrence_continues_a_tasks_states_where_a_shorter_run_stopped`: with `counts = {0: 7, 1: 7}`, `specs(schedule of 24 alternating tasks)` sliced to each task's occurrences 5 to 11 equals `trial_specs(schedule of 14, ..., first_occurrence=5)` field for field (so past a reshuffle at occurrence 7); and the default is unchanged.
- [ ] **Step 2:** Run; expect a `TypeError`.
- [ ] **Step 3:** Implement by counting each task's occurrences from zero and emitting from `first_occurrence` on (the permutation stream must be drawn through the skipped passes). Validate a non-negative integer.
- [ ] **Step 4:** Run the file; hooks.

### Task 4: The wrong-target measure

**Files:**

- Modify: `examples/pi05/mars/libero/notebooks/analyze.py`, `README.md` (analysis list)
- Test: `tests/examples/pi05_mars_libero/test_analyze.py`

**Interfaces:**

- Produces: `wrong_target(trials: pd.DataFrame, targets: dict[int, str | None] | None = None) -> pd.Series` (bool, indexed as `trials`): True where a trial's `mode_expressed` is a scene element other than its task's target (`task_targets(trials)` when `targets` is None); False where nothing moved or the task has no known target.

- [ ] **Step 1: Failing test** `test_wrong_target_is_another_element_moving_most`: three trials of a task whose ceiling success moved `ketchup_1`: `mode_expressed` of `ketchup_1`, `milk_1`, `None` give `[False, True, False]`.
- [ ] **Step 2 to 4:** Run, implement, run, hooks.

### Task 5: The trainer starts, serves as the rollouts do, and calibrates elsewhere

**Files:**

- Modify: `examples/pi05/mars/libero/train_user_module.py`, `README.md` (the training section's command)
- Test: `tests/examples/pi05_mars_libero/test_train_user_module.py`

**Interfaces:**

- Consumes: Task 1 (`Readout(quantize=...)`), Task 2 (`control.corruption_quantize`).
- Produces: `readout_core(command, matrix=None, curl=None, n_steps=10, quantize=False)`; `serve(core, sigma, rng)` (noise only where the core is itself active, the rollouts' rule; idle cores served as they are); CLI `--set KEY=VALUE` (repeatable, as `experiment.py`'s) so a run can name its calibration scenes.

- [ ] **Step 1: Failing tests.** `test_serve_noises_only_an_active_core` (a core inside the deadband under an active raw push is served unchanged; an active core is noised and clipped); `test_readout_core_matrix_curl_and_idle` gains a `quantize=True` line; `test_the_session_is_built_without_the_arms_adapter` (`calibration_control(control)` returns a copy with `reversal_adapter`, its matrix and both module paths cleared and the corruption fields kept).
- [ ] **Step 2 to 4:** Run, implement (`calibration_control(control) -> ControlSettings`; `train` builds its `Session` from it; the batch's `active` weight is `readout.is_active(served)`), run, hooks.
- [ ] **Step 5:** End to end on the GPU, from `examples/pi05/mars/libero`: `python train_user_module.py --config configs/experiment/mars_learned_rotz20.yaml --out <scratch> --episodes 1 --steps 20 --batch 4 --log-every 10` exits 0 and writes `user.pt`.

### Task 6: Condition files and the launcher

**Files:**

- Create: `examples/pi05/mars/libero/configs/experiment/study_object.yaml`, `study_goal.yaml`, and `study_{object,goal}_{rotation,lowdof,lag,keypad}.yaml` (eight files, each `extends:` its suite's base and sets `control.corruption`)
- Create: `examples/pi05/mars/libero/experiments/user_study.sh`
- Modify: `examples/pi05/mars/libero/README.md` (a "User study" subsection: the files, the stages, the launcher)
- Test: `tests/examples/pi05_mars_libero/test_config.py` (`CONDITIONS` picks the new files up; add `test_the_study_rotation_is_the_recorded_draw`)

**Interfaces:**

- Produces: `study_object.yaml`: `extends: base_rotz20.yaml`, `experiment: {n_trials: 50, task_order: sequential, output_dir: outputs/pi05_libero_user_study}`, `scene: {suite: libero_object, task_ids: null}`, `operator: policy`, `policy_operator: {update_every_steps: 20, delay_steps: 4}`, `control: {corruption: null, input_gain: auto}`. `study_goal.yaml`: `extends: study_object.yaml`, `scene: {suite: libero_goal, task_ids: [1, 4, 5, 6, 7, 8]}`, `experiment.n_trials: 30`. The rotation file's `M` is `RandomCorruption(rotation_deg=(60, 60)).sample(np.random.default_rng(0)).matrix`, written as rows with the axis in a comment; the test recomputes it.
- Produces: `user_study.sh STAGE` with `STAGE=0` implemented: per suite the jobs floor, ceiling, FC (`control.mode=shared_flow_control`, `--sweep control.tau=10,25,50`) and FRS (`--sweep control.n_reversal_steps=2,6,10`), the last two under both operator profiles. Environment: `OUT` (default `outputs/pi05_libero_user_study/stage0`), `GPUS` (default `"0 1"`), `PER_GPU` (default 3), `BASE_PORT` (default 8800), `SEED` (default 0); `--dry-run` passes through. Each worker slot takes every `(len(GPUS) * PER_GPU)`th job, sets `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` to its GPU and its own port, logs to `$OUT/logs/<job>.log`, and prints `FAILED <job>` without stopping the others. Before launching it writes `$OUT/code_state.txt` (`git rev-parse HEAD` and `git diff HEAD`).

- [ ] **Step 1:** Failing test for the rotation draw; write the ten files; `test_every_shipped_condition_loads` passes over them.
- [ ] **Step 2:** Write the launcher; `user_study.sh 0 --dry-run` validates all twelve jobs without loading a policy.
- [ ] **Step 3:** Hooks; the whole `tests/examples/pi05_mars_libero` folder passes.

### Task 7: Stage 0

- [ ] **Step 1:** Full MARS suites pass on the CPU; hooks clean on every changed file.
- [ ] **Step 2:** One-block smoke on the GPU: `study_object.yaml` with `--n-trials 2 --set control.max_steps=40` in FRS writes two trial records.
- [ ] **Step 3:** `examples/pi05/mars/libero/experiments/user_study.sh 0` in the background; no code edits until it finishes.
- [ ] **Step 4:** Report against the gate (spec, "Stages and gates"): per suite and operator profile, success and wrong-target rate with Wilson intervals for floor, ceiling and each FC and FRS depth; what the floor moves (the `mode_expressed` counts per task); time per trial. State plainly whether each gate condition holds: ceiling at least 90%; the floor leaves room; clean steering reaches the named object in at least 60% of trials at some depth. Stop there for the author's go.
