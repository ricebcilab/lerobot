# MARS altitude round: one path from the base policy to the steered chunk

**Date:** 2026-11-08. **Branch:** `mars-image-task-module`. **Status:** approved in outline by the author
(items 1-6 of the controller's recommendation of 2026-11-08; item 7, a scene-source registry, deferred).

## Intent

Round 4d made `lerobot.policies.steering` a sibling of the other policies on the surface. Underneath, the four
cleanup reviews of Task F found the same rules written twice at a deeper level: the steering policy drives a cursor
base through one path (`SampleChunkSampler`, `_cursor_base`, an `action_scale` read off the model) and a pi0.5 base
through another; the cursor example rebuilds the cores' wiring instead of running the policy that ships; two adapter
holders (`ReversalAdapter` for the cores, `AdapterModules` for the policy) and two spec grammars (the example's
2-D forms, the package's 7-D blocks) describe one thing; the cursor-format checkpoint migration lives in the package;
the package imports the cursor environment to render scenes; and `SceneEncoder` re-derives pi0.5's prefix pass.
This round removes those duplicates so that what the paper measures is the path that ships, and so that the LIBERO
and robot work starts from one contract rather than from the cursor's special case.

## Goals

1. **One sampler contract.** A base policy offers pi0.5's sampler interface and the steering policy knows no other:
   `config.num_inference_steps`, `config.chunk_size`, `config.max_action_dim`, `config.n_action_steps`,
   `config.noise_time` (the flow time of pure noise: 0.0 for the cursor's convention, 1.0 for pi0.5's);
   `predict_action_chunk(batch, **hooks) -> normalized chunk (B, H, n)` taking pi0.5's `ActionSelectKwargs`
   (`noise`, `noise_fn`, `x_t_hook`, `velocity_hook`, `flow_start_time`, `num_forward_steps`); `reset()`;
   `velocity_field(batch) -> velocity(x_t, t)` for inversion and adapter training; and a postprocessor whose
   unnormalizer step carries the action statistics. `CursorFlowPolicy` implements it; `SampleChunkSampler`,
   `_cursor_base`, `_flow_model` and `_action_scale` leave `SteeringPolicy`.
2. **The example runs `SteeringPolicy`.** `SharedController`, the sweep and the expert-chunk agent obtain their chunks
   from a `SteeringPolicy` over a `CursorFlowPolicy`; `steering_wrapper`, `_chunk` and the hand-built cores go. The
   example's `steered_chunk(policy, obs, command, mode, ...)` keeps its signature as a thin call into the policy: it
   is the equivalence suites' oracle.
3. **One adapter holder and one spec grammar.** `AdapterModules` is the holder the cores, the policy and the LIBERO
   session share (`ReversalAdapter` and `MatrixHolder` go); `SteeringConfig` parses every matrix, corruption and
   adapter spec for any `n_dims` (the example's `matrix_spec` / `adapter_spec` go; `corruption_spec` becomes the
   readout's, in `lerobot.teleoperators.readout`).
4. **One checkpoint format.** Trainers write lerobot's formats directly (`AdapterModules.save_module` for modules,
   `save_pretrained` for the cursor policy); the cursor-format migrations (`from_cursor_checkpoint`, `policy.pt`
   reading) live only in `tools/convert_checkpoints.py`.
5. **The package independent of the cursor environment.** `lerobot.policies.steering` imports nothing from
   `lerobot.envs.cursor`: the scene-tokens step reads pictures (`observation.scene_images`); rendering layouts to
   pictures is a cursor-side processor step.
6. **pi0.5 exposes its prefix encoding.** `PI05Pytorch.encode_prefix(...)` returns the prefix's hidden state with
   the pad masks and the KV cache; `sample_actions`, `velocity_field` and `SceneEncoder.prefix_tokens` call it.

## Non-goals

- A scene-source registry (item 7): deferred until a depth or second-camera source is designed.
- Any change to the numbers: no experiment is re-derived by this round. The grid running from the pinned worktree
  (`/home/user/Projects/lerobot-grid`, 3a6215b7) is unaffected; its results are compared against one cell re-run on
  the refactored code at the end.
- The LIBERO session's own wiring beyond what goals 3 and 6 touch (it keeps building cores directly).
- Renaming or re-deriving any result, report or paper figure.

## Anti-goals

- A new abstraction layer over the base policy (a registry of base kinds, a strategy object): the contract is the
  sibling's existing interface, nothing more.
- Keeping compatibility shims or deprecated aliases for the names that go.

## Constraints (every task's hard bars)

- **Numbers never change.** The equivalence and reference suites (`tests/examples/pi05_mars_cursor/
test_*equivalence*.py`, `test_steering_core_cursor.py`, `test_steering_policy_cursor.py`, `_reference_*.py`) and the
  pi0.5 harness tests (`tests/policies/steering/test_steering_core.py`, `test_steering_wrappers.py`,
  `test_steering_shared.py`) pass with import-line and name-only edits (a changed expected value or a weakened
  assertion is a defect); a tiny adapter training saves the same `adapter.pt` as at the round's base; the scene
  digests (`run_sweep.scene_digest`, `cursor_sa.dataset.layout_digests`, `encoder_digest`) stay byte-identical (the
  pinned digests test); and at the end one grid cell (U1xT2 of `grid_vision`, fixed settings) re-run on the
  refactored code reproduces the pinned run's trial outcomes.
- **Sibling layout and public contracts untouched:** `configuration_steering.py` holds `SteeringConfig` alone;
  `modeling_steering.py`, `processor_steering.py`; `PreTrainedConfig`, `PreTrainedPolicy`, `ProcessorStep`, the
  factory entries. `cursor_flow` gains a `processor_cursor_flow.py`, as the siblings have.
- **Floating-point care where a rule moves:** when the same value is computed by a moved expression, the order of
  operations is kept (documented per task); a postprocessor with `mean = 0` and a uniform `std` must leave the
  cursor's chunks and references bit-identical.
- CPU only for implementers (`CUDA_VISIBLE_DEVICES=""`); nothing under `examples/pi05/mars/results` or
  `examples/pi05/mars/checkpoints`; the worktree `/home/user/Projects/lerobot-grid` is never touched; `mypy` strict
  modules (`lerobot.envs`, `lerobot.configs`) keep their annotations.
- Task F's fix round (I1, I2, M1-M6 of its review) lands before Task 1 starts (same files).

## Decisions the author made or approved

- Items 1-6 in the order 1, 2, 3, 4, 5, 6 (the controller's recommendation, approved 2026-11-08 ~03:30); item 7
  (registry) not now.
- The refactor proceeds on the branch while the grid runs from a pinned worktree; one re-run grid cell is the
  acceptance check (R-G4 of the grid plan).

## Decisions left to the controller (made here)

- **Units of the fixed adapter matrix F:** the operator's (environment) units everywhere, normalized at use by the
  postprocessor's action statistics (`normalized_matrix(std)`, the pi0.5 cores' existing rule). A uniform `std`
  (the cursor's scalar `action_scale`) short-circuits to the matrix itself, so the cursor's numbers do not move.
- **The cursor policy's postprocessor:** lerobot's `UnnormalizerProcessorStep` with `mean = 0` and `std =
action_scale` on both dims; `predict_action_chunk` returns the normalized chunk, as pi0.5's does.
- **A mode for the base policy's own chunk:** `SteeringConfig.mode = "none"` (the task module's velocity hook, no
  core), so the example's `policy` and `policy_task` arms are `SteeringPolicy` arms too.
- **The matrix grammar lives with the readout:** `lerobot.teleoperators.readout.matrix_spec(spec, n_dims, where)`
  and `corruption_spec(spec, where)` parse every matrix and corruption (the readout applies them); `SteeringConfig`
  keeps the adapter grammar (`build_adapter`, `from_spec`, both taking `n_dims`) and calls the readout's for the
  matrices. The rotation key names its axis where there is one: `rotation_deg` in the plane (2 dims),
  `rotation_z_deg` about z (3 dims); both are one rule in one parser, so no config and no pinned test changes.
- **Scene size:** the package defines `SCENE_IMAGE_SIZE = 224`; `lerobot.envs.cursor.SCENE_SIZE` keeps its value and
  a test asserts the two agree.

## What is left to the builder

Internal names of private helpers, the exact split of tests between files, and the order of edits inside a task,
within the plan's interfaces.
