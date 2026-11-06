# MARS round 4d: the cursor example keeps only what is the cursor's — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The generic parts of `cursor_sa/readout.py` and `cursor_sa/steering.py` move into the lerobot packages they belong to, the re-export shim `cursor_sa/env.py` goes away, and every number the example computes stays the same.

**Architecture:** Three moves, each pinned by the existing equivalence and reference tests. The readout's stateful class and its open-loop curl integration join the numpy readout package; its torch twins join the steering package. The composed adapter, its coercion helper and the flow wrappers join the steering package under a name that does not collide with the package's existing `ReversalAdapter` holder. The cursor example imports from the packages and keeps its units, observation, controller and sampler.

**Tech Stack:** Python 3.12, numpy, PyTorch, the MARS cursor example, `lerobot.teleoperators.readout`, `lerobot.policies.steering`.

**Spec:** the user's request of 2026-11-06 (afternoon): "Do it alongside with removing the now empty `cursor_sa/env.py`, and moving anything generic in `cursor_sa/steering.py` out of cursor example as well", following the readout answer (the `Readout` class, `curl_commands` and the torch twins are generic; the YAML spec parsing is the example's).

## Global Constraints

- Every number stays: the equivalence tests (`test_ranks_equivalence.py`, `test_steering_equivalence.py`, `test_spacemouse_equivalence.py`, `test_steering_core_cursor.py`) and the reference files are untouched and must pass; a tiny task-module training before and after the move saves identical weights (the test from round 4c's store work, or the existing default-off identity checks, re-run).
- Existing checkpoints load unchanged: `ReversalAdapter.load` (the example's name) keeps working as a thin alias, and `save_adapter_module` / `load_adapter_module` formats do not change.
- `lerobot.teleoperators.readout` stays numpy-only (it imports no torch today); anything torch goes to `lerobot.policies.steering`.
- `lerobot.policies.steering`'s public `__all__` gains the moved names; the package's existing `ReversalAdapter` (the `MatrixHolder` subclass consumed by `FlowReversalSteeringPolicy`) keeps its name and behaviour; the moved composed module takes a different name (see Task C).
- No import of `cursor_sa` from anything under `src/lerobot/`.
- mypy: `lerobot.envs` is strict (unchanged here); the steering and teleoperators packages are gradual — annotate what you move as far as the surrounding code is.
- Tests on the CPU only; pre-commit on every touched file; commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `git add` names exactly the task's files; no push; nothing under `examples/pi05/mars/results/` or `checkpoints/` touched by a build task. Jobs may be training from this checkout: never leave a file syntactically broken on disk, and do not edit `train/train_adapter.py` or `cursor_sa/demos.py` beyond import lines.

## Review Focus

- A readout with a curl corruption must integrate the same velocity state after the move (the curl equivalence numbers in `test_steering_equivalence.py` and the steady-state check). Pinned by the frozen tests; Task A re-runs them.
- The composed adapter applied through `FlowReversalSteeringPolicy` and through the example's `steered_chunk` must give the same chunk as before the move for a user module, a task module, a matrix and their composition. Pinned by the equivalence tests; Task C adds one direct test on the moved class if none exists.
- The removal of the env shim must not change which `Layouts` class a test sees (`cursor_sa.env.Layouts is lerobot.envs.cursor.Layouts` today, by star import); `_layout` is imported by one test directly from the package after the move.
- Scripts that import by bare name through the test path (`run_sweep.py`, `tools/`, `reports/`) and the run-folder shell scripts must still start: Task B runs each `--help`.

---

### Task A: The readout's generic parts move to the packages

**Files:**
- Modify: `src/lerobot/teleoperators/readout.py` (gains `Readout` and `curl_commands`), `src/lerobot/policies/steering/modeling_steering.py` or a small new module `src/lerobot/policies/steering/readout_torch.py` (gains `serve_command`, `corrupt_chunk`), `src/lerobot/policies/steering/__init__.py` (exports), `examples/pi05/mars/cursor/cursor_sa/readout.py` (keeps `matrix_spec`, `corruption_spec`, `adapter_spec`; re-imports the rest so every existing import line keeps working)
- Test: `tests/policies/steering/test_steering_shared.py` or a new `tests/policies/steering/test_readout.py` (the moved functions under their package names: one `Readout` batch step with a matrix, one with a curl against `curl_discretization`, `curl_commands` against a step loop, the torch twins against the numpy originals); the cursor example's tests unchanged and re-run.

**Interfaces:**
- Consumes: today's `cursor_sa/readout.py` (`Readout.__init__(matrix, sigma, rng, curl=None)` and its methods — read them), `lerobot.teleoperators.readout` (`DEADBAND`, `corrupt`, `curl_step`, `curl_serve`, `add_noise`, `is_active`), `lerobot.envs.cursor.curl_discretization`.
- Produces: `lerobot.teleoperators.readout.Readout` (same signature and behaviour; numpy only; the curl discretisation imported from `lerobot.envs.cursor` or moved beside `curl_discretization`'s existing numpy core in the readout package if that is where it lives), `lerobot.teleoperators.readout.curl_commands(raw, curl, dt=0.02)`; `lerobot.policies.steering.serve_command(core, raw, sigma, noise=None)`, `lerobot.policies.steering.corrupt_chunk(command, corruption, sigma)` (torch). `cursor_sa/readout.py` imports all four and keeps exporting them.

- [ ] Step 1: write the failing package-level tests; Step 2: run them; Step 3: move (cut from the example, paste into the package, adjust imports; the example re-imports); Step 4: run the new tests, `tests/examples/pi05_mars_cursor` whole (about 7 minutes) and `tests/policies/steering`; Step 5: pre-commit, commit `MARS: the readout's stateful class, its curl integration and its torch twins move into lerobot`.

---

### Task B: The env shim goes

**Files:**
- Delete: `examples/pi05/mars/cursor/cursor_sa/env.py` (`git rm`)
- Modify: every `from .env import ...` under `examples/pi05/mars/cursor/cursor_sa/` and every `from cursor_sa.env import ...` under `examples/pi05/mars/cursor/` and `tests/` (22 files) to import from `lerobot.envs.cursor` (one test imports `_layout`: import it from the package too); the run-folder shell scripts need no change (they call scripts, not modules).

**Interfaces:** none new. The example modules that re-exported env names to others (`cursor_sa/demos.py` re-exports? check `__all__`s and star imports: nothing else may rely on `cursor_sa.env` as a namespace, grep `cursor_sa.env` and `env\.` attribute access on the module object).

- [ ] Step 1: a test (in `test_tools_layout.py`) that `cursor_sa` has no `env` module and that `cursor_sa.demos.Layouts is lerobot.envs.cursor.Layouts`; Step 2: run it (fails: the shim exists); Step 3: the removal and the import rewrites; Step 4: `tests/examples/pi05_mars_cursor` whole plus every `--help` of `run_sweep.py`, `run_experiment.py`, `tools/*.py`, `reports/report_grid_image.py` with `CUDA_VISIBLE_DEVICES=""`; Step 5: pre-commit, commit `MARS: the cursor example imports its env from lerobot.envs.cursor directly`.

---

### Task C: The steering module's generic parts move to lerobot.policies.steering

**Files:**
- Modify: `src/lerobot/policies/steering/modeling_steering.py` (or a new `adapters.py` in the package, re-exported), `src/lerobot/policies/steering/__init__.py`, `examples/pi05/mars/cursor/cursor_sa/steering.py`
- Test: `tests/policies/steering/test_steering_core.py` or a new `tests/policies/steering/test_composed_adapter.py`; the example's tests re-run.

**What moves (generic):** the composed adapter class (today `cursor_sa.steering.ReversalAdapter(nn.Module)`: `user`, `task`, `matrix`; `kind`; `field(obs)` building the core's `AdaptedField` with the task module bound to `obs`; `adapt`; `load` from `load_adapter_module`), `as_adapter` (coercion of None / a task module / a callable user part / a matrix), `_field`, and the flow wrappers `flow`, `reverse_flow`, `forward_flow` IF they are thin forms of the package's `integrate` / `reverse_flow` / `forward_flow` that can be unified without changing numbers (read both; if the package's versions take a sampler where the example's take a policy and an obs, keep the example's as thin cursor adapters and say so in the report). The `UserModule` factory stays in the example (it is "the core's `LearnedUserModule` with the cursor's dims and defaults") unless it is only a partial application — then it becomes a one-line alias.
**What stays (cursor):** `MODES`, `GAIN_MODES`, `MAX_SPEED`, `method_name`, `to_action`, `to_deflection`, `bci_chunk`, `observation`, `view_scene_fn`, `hide_goal`, `action_gain`, `CursorSampler`, `steering_wrapper` and `steered_chunk` (they carry the cursor's units through `to_action`; if the units can be injected as a callable, move the wrapper and keep a cursor partial), `_chunk`, `SharedController`.

**Interfaces:**
- Produces: `lerobot.policies.steering.AdapterModules(nn.Module)` — the moved composed adapter, with `n_dims` a constructor argument (default 2) instead of the hard-coded `n_dims=2` in `field`; `lerobot.policies.steering.as_adapter(adapter, device="cpu", n_dims=2)`; the example keeps `ReversalAdapter = AdapterModules` (alias, so `ReversalAdapter.load` and every `isinstance` keep working) and `TaskModule = LearnedTaskModule` as today. Checkpoint format unchanged.

- [ ] Step 1: package-level tests for `AdapterModules` (kind per combination; `field` binds a task module and puts the matrix on the obs device; `load` of a saved user module and of a saved task module; `as_adapter` coercions) and for the moved flows if moved; Step 2: run them; Step 3: move, alias in the example, update the package `__all__` and docstrings; Step 4: run `tests/policies/steering` and `tests/examples/pi05_mars_cursor` whole; Step 5: pre-commit, commit `MARS: the composed adapter and its helpers move into lerobot.policies.steering`.

---

### Task D (controller): final review and the user's hand-over

- [ ] Whole-range review of the three commits on the most capable model, with the equivalence tests as the bar; one fix wave if needed.
- [ ] Report: what moved where (a table), what stayed and why, the test counts and runtime, and the rulings made.
