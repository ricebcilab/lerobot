# MARS round 4d: the cursor example keeps only what is the cursor's — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The generic parts of `cursor_sa/readout.py` and `cursor_sa/steering.py` move into the lerobot packages they belong to, the re-export shim `cursor_sa/env.py` goes away, and every number the example computes stays the same.

**Architecture:** Three moves, each pinned by the existing equivalence and reference tests. The readout's stateful class and its open-loop curl integration join the numpy readout package; its torch twins join the steering package. The composed adapter, its coercion helper and the flow wrappers join the steering package under a name that does not collide with the package's existing `ReversalAdapter` holder. The cursor example imports from the packages and keeps its units, observation, controller and sampler.

**Tech Stack:** Python 3.12, numpy, PyTorch, the MARS cursor example, `lerobot.teleoperators.readout`, `lerobot.policies.steering`.

**Spec:** the user's request of 2026-11-06 (afternoon): "Do it alongside with removing the now empty `cursor_sa/env.py`, and moving anything generic in `cursor_sa/steering.py` out of cursor example as well", following the readout answer (the `Readout` class, `curl_commands` and the torch twins are generic; the YAML spec parsing is the example's).

## Global Constraints

- Every number stays: the equivalence tests (`test_ranks_equivalence.py`, `test_steering_equivalence.py`, `test_spacemouse_equivalence.py`, `test_steering_core_cursor.py`) must pass; the reference files change only in the import line and call site of `corrupt_chunk` (Task A) and `cursor_sa.env` (Task B), never in a number; a tiny task-module training before and after the move saves identical weights (the test from round 4c's store work, or the existing default-off identity checks, re-run).
- Existing checkpoints load unchanged: `ReversalAdapter.load` (the example's name) keeps working as a thin alias, and `save_adapter_module` / `load_adapter_module` formats do not change.
- The readout's rule has one home, `lerobot.teleoperators.readout`: its torch twins move there too (torch is a base dependency of lerobot; import it at module level). Nothing of the readout goes to the steering package.
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
- Modify: `src/lerobot/teleoperators/readout.py` (gains `Readout`, `curl_commands`, `serve_command`, `corrupt_chunk`), `examples/pi05/mars/cursor/cursor_sa/readout.py` (keeps `matrix_spec`, `corruption_spec`, `adapter_spec`; re-imports the rest so every existing import line keeps working)
- Test: a new `tests/policies/steering/test_readout.py` (`Readout.__call__` with a matrix and with a curl against `curl_discretization`; `core` against a step loop of `__call__` from rest for both branches; `chunk` and `serve` against `__call__` on the same numbers, the twins check); the cursor example's tests re-run with their call sites rewritten (the frozen reference files `_reference_steering.py` and `_reference_ranks.py` import `corrupt_chunk`: rewrite that import and call to `Readout(...).chunk(...)` only, nothing else in them).

**Interfaces:**
- Consumes: today's `cursor_sa/readout.py` (`Readout.__init__(matrix, sigma, rng, curl=None)` and its methods — read them), `lerobot.teleoperators.readout` (`DEADBAND`, `corrupt`, `curl_step`, `curl_serve`, `add_noise`, `is_active`), `lerobot.envs.cursor.curl_discretization`.
- Produces: `lerobot.teleoperators.readout.Readout`, one class for the rule in both forms, no standalone twins:
  - `Readout(corruption=None, input_noise=0.0, rng=None, curl=None, dt=0.02)`, `reset(n_trials)`, `__call__(raw)`: as today (numpy, closed loop, the curl's velocity state per trial).
  - `core(raw)` (numpy, `[B, L, 2]` -> `[B, L, 2]`): what the readout serves before noise for a whole sequence from rest — the matrix branch (`corrupt` per step, clipped) or the curl branch (`curl_step` from rest, clipped). It replaces both the example's `curl_commands` and the trainer's `readout_core`; the trainer calls `Readout(corruption, curl=curl).core(push)`.
  - `chunk(command, sigma)` (torch, `[B, H, 2]`, `sigma` `[B]`): the rule on a chunk per trial for training — today's `corrupt_chunk` with the instance's matrix (`None` = no corruption); with a curl it raises as `__call__` does for chunks. The noise draws stay `torch.randn_like` on torch's global generator, so training draws do not change.
  - `serve(core, raw, sigma, noise=None)` (torch, static): the noise stage alone — today's `serve_command`, unchanged.
  The curl discretisation stays importable from `lerobot.envs.cursor` (it already delegates to the readout package's numpy core). The example's `cursor_sa/readout.py` keeps only the YAML spec parsing and re-exports `Readout`, `DEADBAND`, `is_active`; callers of `curl_commands`, `serve_command`, `corrupt_chunk` and `readout_core` (the trainer, `cursor_sa/steering.py`'s `bci_chunk`, the tests and the two frozen reference files, which may only have their import lines and call sites rewritten, not their numbers) move to the methods. `cursor_sa/readout.py` imports all four and keeps exporting them.

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

---

### Task E: The steering package itself, functions folded into classes along the package's concerns

**Why (user, 2026-11-06 afternoon):** "consolidation to the steering utils, by cleanly folding functions into classes, following the examples of other implementations in `src/lerobot/policies`." Today `modeling_steering.py` is 1,734 lines with about thirty module-level functions around five concerns; the sibling policies keep a `configuration_<name>.py` (a config dataclass and specs), a `modeling_<name>.py` (the `nn.Module` cores and the policy class, with few module-level helpers, all pure math), and a `processor_<name>.py` (processing steps as classes).

**The map (revised 2026-11-06 16:15 after the user's correction: the siblings keep exactly `configuration_<name>.py`, `modeling_<name>.py` and `processor_<name>.py`, so no new files beyond a `processor_steering.py`):**

| today (module-level) | becomes | in |
|---|---|---|
| `validate_matrix`, `rotation_about_z`, `translation_matrix`, `format_matrix`, `build_reversal_adapter`, `parse_adapter_spec` | `AdapterSpec.parse(...)`, `AdapterSpec.build(...)` (class methods), `AdapterSpec.matrix(...)` / `.rotation(...)` / `.validate(...)` (statics), `MatrixHolder.format()` | `configuration_steering.py` |
| `apply_user_module`, `user_field_loss` | `LearnedUserModule.apply(x_t, t, v)`, `LearnedUserModule.field_loss(policy, ...)` (the loss a module's `forward` would return in a lerobot policy) | `modeling_steering.py` |
| `_BoundTaskModule`, `reduce_tokens`, `task_field_loss` | `LearnedTaskModule.bind(obs)` (already), `LearnedTaskModule.reduce(tokens)` (static), `LearnedTaskModule.field_loss(policy, ...)` | `modeling_steering.py` |
| `save_adapter_module`, `load_adapter_module`, `convert_cursor_module`, `_MODULE_KINDS` | `AdapterModules.save(path, kind, meta)`, `AdapterModules.load(path, device)`, `AdapterModules.from_cursor_checkpoint(ckpt)` on the composed adapter moved in Task C | `modeling_steering.py` |
| `AdaptedField`, `_adapt_velocity`, `integrate`, `reverse_flow`, `forward_flow`, `frs_schedule`, `frs_latent`, `uniform_reference`, `regenerate`, `paired_chunks`, `heading_cosine` | `AdaptedField.adapt` (already) and a `Flow(sampler, field)` class: `integrate(x, t0, h, n)`, `reverse(x, n, steps)`, `forward(x, t0, t1, steps)`, `latent(...)`, `regenerate(...)`, `paired_chunks(...)`, statics `schedule(...)`, `uniform_reference(...)`, `heading_cosine(...)` | `modeling_steering.py` |
| `flow_control_write`, `delegation_mask`, `task_velocity_hook`, `get_action_mean_std` | methods of `_ChunkSteering` / `FlowControlPolicy` / `FlowReversalSteeringPolicy` (`write`, `delegation_mask`, `velocity_hook`, `action_mean_std`) | `modeling_steering.py` |
| `train_module`, `step_if_finite`, `_rng_state`, `_set_rng_state` | leave the package: they move into the example's trainer `examples/pi05/mars/cursor/train/train_adapter.py` (their only caller), as lerobot keeps training loops in scripts, not in policy packages; `step_if_finite` and the generator save/restore go with the loop as private helpers there. The package keeps no training loop. | `train/train_adapter.py` (the example) |
| `scene_for`, `encode_views`, `fit_token_reduction`, `SCENE_VLM_PATH`, `SCENE_PROMPT`, `SceneEncoder` | `SceneEncoder` (moved) with `scene_for(layouts, source, size)`, `encode_views(images, batch)`, `fit_reduction(batches, rank, center)`; the `discs` / `cnn` branches of `scene_for` reachable without an encoder as a static method — images to tokens is pre-processing, which is what `processor_<name>.py` holds in the siblings | `processor_steering.py` (new, the package's third file) |
| `sinusoidal_time_embedding`, `check_steps`, `scene_spec`, the constants | stay module-level (pure helpers and specs, as the siblings keep theirs) | where they are |

**Final formulation (2026-11-06 16:40, the user: the package must match the siblings' convention).** What a sibling is: a registered `PreTrainedConfig` dataclass in `configuration_<name>.py`; a `PreTrainedPolicy` subclass in `modeling_<name>.py` that wraps the core `nn.Module`s and implements `forward` (the training loss), `predict_action_chunk`, `select_action`, `reset`, `get_optim_params`; `ProcessorStep` classes and a `make_<name>_pre_post_processors(config, dataset_stats)` factory in `processor_<name>.py`; an entry in `factory.py`; `__init__.py` exporting the config, the policy and the cores (`cursor_flow` is the precedent for a policy with its own observation dict and no dataset features: `validate_features` is a no-op, the delta indices are `None`, the optimizer preset is AdamW). The steering package has none of these today. After Task E:

- `configuration_steering.py`: the constants, `TeleopSource`, `scene_spec`, `check_steps` stay module-level; `AdapterSpec` gains `parse` and `build` (class methods; today's `parse_adapter_spec`, `build_reversal_adapter`) and the matrix helpers as statics; `MatrixHolder.format()`; NEW `SteeringConfig(PreTrainedConfig)` registered as `"steering"`: `mode` (`flow_control` | `frs`), the mode's knobs (`tau`, `lam`, `ic_gain`, `n_reversal_steps`, `deadband`, `operator_dims`), `adapter: AdapterSpec | None` (the fixed matrix, the user module path, the task module path), `n_dims`, `horizon`, the base policy reference (`base_policy_type`, `base_policy_path`: the policy whose chunks are steered), and the task module's scene settings (`scene_source`, `render_size`, `token_geometry`, `scene_rank`, `vlm_path`); `get_optimizer_preset` returns AdamW (adapter training), `validate_features` is a no-op, the delta indices are `None`, as in `CursorFlowConfig`.
- `modeling_steering.py`: the cores stay `nn.Module`s with their helpers folded in (the map above): `LearnedUserModule`, `LearnedTaskModule`, `AdapterModules`, `AdaptedField`, `Flow`, and the two mode cores, renamed `FlowControlSteering` and `FlowReversalSteering` (today's `FlowControlPolicy` / `FlowReversalSteeringPolicy`; the old names stay as aliases for one round) so that only the wrapper carries the word Policy, as `FlowPolicy` sits inside `CursorFlowPolicy`. NEW `SteeringPolicy(PreTrainedPolicy)`: `config_class = SteeringConfig`, `name = "steering"`; `__init__(config, base_policy=None)` builds the `AdapterModules` from the spec (loading module checkpoints with `AdapterModules.load`) and the mode core over the base policy's sampler (a `TeleopSource`); `forward(batch)` returns the adapter's field loss (`batch["kind"]` user or task; the chunk, the command, the scene tokens as the example's trainer already assembles them); `predict_action_chunk(batch)` returns the steered chunk for `batch["command"]`; `select_action` its first step; `reset()` clears the core's queues and the adapter's state; `get_optim_params()` the adapter modules' parameters; `save_pretrained` / `from_pretrained` carry the adapter weights and the config in lerobot's format (the base policy is passed in or loaded from the config's reference by the factory). `tools/convert_checkpoints.py` gains the conversion of an `adapter.pt` folder to that format, as it did for the cursor policy; `adapter.pt` stays loadable by `AdapterModules.load`.
- `processor_steering.py` (new): `SceneEncoder` moves here with `scene_for`, `encode_views`, `fit_reduction`; `SteeringSceneTokensProcessorStep(ProcessorStep)` registered with `ProcessorStepRegistry` (images or layouts to scene tokens, with the module's reduction when it carries one); `SteeringCommandProcessorStep` (the command's deadband and units into the policy's action units); `make_steering_pre_post_processors(config, dataset_stats=None)` returning the pre and post `PolicyProcessorPipeline`s with the same signature as `make_pi05_pre_post_processors`.
- `factory.py`: `"steering"` in `get_policy_class` and `make_policy_config`; `__init__.py`: `SteeringConfig`, `SteeringPolicy`, `make_steering_pre_post_processors`, the cores and the module classes, plus the deprecated aliases.
- The training loop leaves the package for the example's trainer (above). The example's `cursor_sa/steering.py` keeps the cursor's units, observation, `CursorSampler` (the cursor policy as a `TeleopSource`), `SharedController` and thin `steered_chunk` wrappers that call the package.
- Numbers: unchanged, by the equivalence and reference tests and by one new test that `SteeringPolicy.predict_action_chunk` equals the example's `steered_chunk` for each mode on the same inputs.

`__init__.py` exports the classes and keeps every name the example, the tests and the π0.5 steering harness import today as aliases for this round, each marked deprecated in its docstring; a later round removes the aliases once every caller is on the methods. No number changes: every equivalence and reference test must pass unchanged; `save`/`load` keep the checkpoint format; the π0.5 harness tests (`tests/policies/steering/test_steering_wrappers.py`, `test_steering_core.py`) must pass unchanged except for import lines.

**Files:** create `src/lerobot/policies/steering/processor_steering.py`; modify `modeling_steering.py` (the classes above; no other new file), `configuration_steering.py`, `__init__.py` (drops `train_module`, `step_if_finite`); `examples/pi05/mars/cursor/train/train_adapter.py` gains the training loop (moved verbatim with its helpers; the trainer's own `train_task`/`train_user` call it as before); the rest of the example and the tests only in import lines and call sites (the tests of the loop, if any live in `tests/policies/steering`, move to `tests/examples/pi05_mars_cursor/test_training.py`). **Tests:** the package's existing tests re-run; one new test per folded class asserting the method equals the alias on the same inputs.

- [ ] Step 1: the alias-equality tests; Step 2: fold concern by concern (processor, flow, adapters, losses, policies, configuration) and move the training loop to the example, running `tests/policies/steering` after each; Step 3: the example's call sites, `tests/examples/pi05_mars_cursor` whole; Step 4: pre-commit, one commit per concern, subjects `MARS: the steering package's <concern> as a class`.
