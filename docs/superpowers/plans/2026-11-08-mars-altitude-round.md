# MARS Altitude Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One path from any base policy to the steered chunk: the cursor policy offers pi0.5's sampler contract, the
cursor example runs `SteeringPolicy`, one adapter holder, one spec grammar, one checkpoint format, a steering package
that knows no cursor environment, and a pi0.5 that exposes its prefix encoding.

**Architecture:** `CursorFlowPolicy` gains the sampler contract and a processor pipeline (`processor_cursor_flow.py`),
so `SteeringPolicy` drives it exactly as it drives pi0.5 and its cursor-only branches go. The example's controller,
sweep and agents then obtain chunks from a `SteeringPolicy`. `AdapterModules` absorbs `ReversalAdapter`; the readout
module owns the matrix grammar and `SteeringConfig` the adapter grammar, both dimension-aware; the cursor-format
migrations move into the converter; the scene-tokens step reads pictures and a cursor-side step renders them;
`PI05Pytorch.encode_prefix` serves `sample_actions`, `velocity_field` and `SceneEncoder`.

**Tech Stack:** Python 3.12, PyTorch, lerobot (`PreTrainedPolicy`, `ProcessorStep`, processor pipelines), pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-mars-altitude-round-design.md`

## Global Constraints

- Numbers never change. Pinned suites pass with import-line and name-only edits: `tests/examples/pi05_mars_cursor/
test_steering_equivalence.py`, `test_ranks_equivalence.py`, `test_steering_core_cursor.py`,
  `test_steering_policy_cursor.py`, `test_spacemouse_equivalence.py`, `_reference_*.py`; `tests/policies/steering/
test_steering_core.py`, `test_steering_wrappers.py`, `test_steering_shared.py`. The only further edit allowed is the
  one each task names (a fixture built on a deleted class rebuilt on its replacement, asserts untouched). A changed
  expected value or a weakened assertion is a defect.
- Before every commit: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/policies/steering
tests/policies/cursor_flow tests/examples/pi05_mars_cursor tests/teleoperators -q`; the tiny adapter training of
  `tests/examples/pi05_mars_cursor/test_training.py` saves an `adapter.pt` whose sha256 equals the round's base (record
  the base hash in the ledger before Task 1); the digests test (`test_dataset.py`'s pinned digests) passes unchanged.
- Floating point: an expression that moves keeps its order of operations; the plan says where. A postprocessor with
  `mean = 0` and a uniform `std` leaves every cursor chunk and reference bit-identical (`torch.equal`).
- Sibling layout and public contracts untouched (`configuration_steering.py` holds `SteeringConfig` alone;
  `PreTrainedConfig`, `PreTrainedPolicy`, `ProcessorStep`, the factory entries). New processor code lives in
  `processor_<policy>.py` files, as the siblings'.
- CPU only (`CUDA_VISIBLE_DEVICES=""`); nothing under `examples/pi05/mars/results` or `examples/pi05/mars/checkpoints`;
  `/home/user/Projects/lerobot-grid` untouched; no subagents from implementers; `pre-commit run --files <touched>`
  clean; commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; no push.
- `lerobot.envs` and `lerobot.configs` are mypy-strict: annotate what you add there.
- Task F's fix round (task-F-review.md: I1, I2, M1-M6) is committed before Task 1 starts.

## Review Focus

1. A pi0.5-style base (no `sample_chunk`, pi0.5's `noise_time = 1.0`) driven by `SteeringPolicy` after Task 1 must
   produce the same chunk as before Task 1 for the same stub: Task 1's `test_steering_policy.py` pins a stub base's
   chunk before and after (the stub records the kwargs it receives; `x_t_hook` and `velocity_hook` arrive unchanged).
2. A `SteeringConfig` with `mode = "none"` and a command in the batch must ignore the command (the base's own chunk
   through the task module), not raise: Task 1's test.
3. A fixed adapter matrix on a pi0.5 base after Task 3 must be normalized exactly as `ReversalAdapter.normalized_matrix`
   did (`matrix * std[None, :] / std[:, None]`, that order), and a uniform `std` must return the matrix object's values
   unchanged: Task 3's test compares both.
4. A corruption spec with `rotation_deg` for 3 dims, or `rotation_z_deg` for 2, must raise naming the key for that
   dimension, not silently rotate: Task 4's test.
5. A scene-tokens step given pictures of the wrong size for the task module must raise before encoding: Task 6's test
   (`LearnedTaskModule.spec`'s size check reached from `observation.scene_images`).

---

### Task 1: The cursor policy offers the sampler contract; the steering policy has one path

**Files:**

- Modify: `src/lerobot/policies/cursor_flow/configuration_cursor_flow.py`, `modeling_cursor_flow.py`
- Create: `src/lerobot/policies/cursor_flow/processor_cursor_flow.py`
- Modify: `src/lerobot/policies/factory.py` (the `make_pre_post_processors` entry for `cursor_flow`),
  `src/lerobot/policies/steering/configuration_steering.py` (`STEERING_MODES`), `modeling_steering.py`
  (`SteeringPolicy`), `src/lerobot/policies/steering/__init__.py`, `README.md` of both packages
- Test: `tests/policies/cursor_flow/test_cursor_flow.py`, `tests/policies/steering/test_steering_policy.py`

**Interfaces:**

- Produces: `CursorFlowConfig` fields `num_inference_steps: int = 10` (replaces the module constant `FLOW_STEPS`),
  `n_action_steps: int = 8`, `noise_time: float = 0.0`, `max_action_dim: int = 2`, and a read-only property
  `chunk_size -> int` returning `horizon`.
- Produces: `CursorFlowPolicy.predict_action_chunk(batch, **hooks) -> Tensor` the NORMALIZED chunk `(B, H, 2)`:
  `self.model.sample_chunk(self.observation(batch), self.config.num_inference_steps, **hooks)`, `hooks` being pi0.5's
  `noise`, `noise_fn`, `x_t_hook`, `velocity_hook`, `flow_start_time`, `num_forward_steps`;
  `CursorFlowPolicy.observation(batch) -> dict` (keys with `observation.` stripped, any `observation.scene_tokens`
  left to the caller, a `pos` of shape `(2,)` unsqueezed with every tensor to a batch of one);
  `CursorFlowPolicy.select_action(batch) -> Tensor` the first `n_action_steps` steps served one per call from a queue,
  normalized (the postprocessor unnormalizes), `reset()` clearing it; `CursorFlowPolicy.velocity_field(batch) ->
Callable[[Tensor, Tensor | float], Tensor]` = `lambda x, t: self.model(obs, x, t)` on `obs = self.observation(batch)`;
  properties `horizon`, `hist`, `action_scale` mirroring the model's.
- Produces: `make_cursor_flow_pre_post_processors(config: CursorFlowConfig) -> tuple[PolicyProcessorPipeline,
PolicyProcessorPipeline]`: pre = `[RenameObservationsProcessorStep(rename_map={})]`; post =
  `[UnnormalizerProcessorStep(features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))}, norm_map={FeatureType.ACTION:
NormalizationMode.MEAN_STD}, stats={"action": {"mean": [0.0, 0.0], "std": [config.action_scale, config.action_scale]}}),
DeviceProcessorStep(device="cpu")]`, names `POLICY_PREPROCESSOR_DEFAULT_NAME` / `POLICY_POSTPROCESSOR_DEFAULT_NAME`;
  the factory's `make_pre_post_processors` returns it for a `CursorFlowConfig`.
- Produces: `SteeringConfig.mode` accepts `"none"` (`STEERING_MODES = ("none", "flow_control", "frs")`): the base's own
  chunk through the task module's velocity hook, no core; `SteeringPolicy.core` raises `ValueError("mode 'none' has no
steering core")`.
- Produces: `SteeringPolicy` without `SampleChunkSampler` use, `_cursor_base`, `_flow_model`, `_action_scale`,
  `_reference`: the core is built on `self.base_policy` directly; FRS's reference is the core's own from the
  postprocessor's statistics; `observation(batch)` has one rule: drop `STEERING_BATCH_KEYS`, and when
  `observation.scene_tokens` is present replace it by `scene = {config.scene_source: tokens}` (raising when
  `scene_source` is None); `_velocity(obs, batch)` is `self.base_policy.velocity_field(batch)` (a base without it raises
  `TypeError` naming the contract); `predict_action_chunk` returns the normalized chunk (the postprocessor's job to
  unnormalize) for every base.
- `SampleChunkSampler` stays in the package until Task 2 (the example still uses it).
- Task 1's one allowed pinned edit: in `tests/examples/pi05_mars_cursor/test_steering_policy_cursor.py` the policy's
  outputs (`predict_action_chunk`, `select_action`) pass through the cursor postprocessor
  (`make_cursor_flow_pre_post_processors(policy.config)[1]`) before the existing asserts, which do not change
  (`torch.equal` against the example's `steered_chunk` in screen units).

- [ ] **Step 1: Write the failing tests** in `tests/policies/cursor_flow/test_cursor_flow.py`:

```python
def test_predict_action_chunk_is_normalized_and_passes_hooks():
    policy = CursorFlowPolicy(CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123))
    obs = _obs()
    z = torch.randn(3, policy.config.horizon, 2)
    seen = {}
    hook = lambda step, t, x: seen.setdefault("steps", []).append(step) or x
    chunk = policy.predict_action_chunk(obs, noise=z, x_t_hook=hook)
    assert torch.equal(chunk, policy.model.sample_chunk(obs, 10, noise=z))
    assert seen["steps"] == list(range(10))
    assert policy.config.chunk_size == policy.horizon and policy.config.noise_time == 0.0

def test_the_postprocessor_restores_screen_units_exactly():
    policy = CursorFlowPolicy(CursorFlowConfig(d_model=32, layers=2, heads=2, d_ff=64, action_scale=0.0123))
    _, post = make_cursor_flow_pre_post_processors(policy.config)
    torch.manual_seed(0); chunk = policy.predict_action_chunk(_obs())
    torch.manual_seed(0); want = policy.model.sample(_obs(), steps=10)
    assert torch.equal(post(chunk), want)

def test_prefixed_and_unbatched_observations():
    obs = _obs(b=1)
    one = {k: v[0] for k, v in obs.items()}                        # (2,) pos
    prefixed = {f"observation.{k}": v for k, v in obs.items()}
    torch.manual_seed(0); a = policy.predict_action_chunk(obs)
    torch.manual_seed(0); b = policy.predict_action_chunk(prefixed)
    torch.manual_seed(0); c = policy.predict_action_chunk(one)
    assert torch.equal(a, b) and torch.equal(a, c)

def test_select_action_serves_n_action_steps_then_replans(): ...   # 8 calls -> one chunk's first 8 steps, the 9th a new chunk
def test_velocity_field_is_the_model(): ...                         # velocity_field(obs)(x, t) == model(obs, x, t)
```

and in `tests/policies/steering/test_steering_policy.py`:

```python
def test_a_pi05_style_stub_is_driven_unchanged():       # the stub from the existing tests: same chunk and kwargs as before
def test_the_cursor_base_goes_through_the_same_path():  # no attribute of SteeringPolicy named _cursor_base/_flow_model/_action_scale
def test_mode_none_ignores_the_command():               # SteeringConfig(mode="none"): predict_action_chunk(batch with command) == base chunk with the task hook
def test_observation_has_one_rule():                    # a prefixed cursor batch and a pi0.5 batch both lose only STEERING_BATCH_KEYS; scene_tokens -> scene[source]
```

- [ ] **Step 2: Run them**: `uv run --extra mars pytest tests/policies/cursor_flow tests/policies/steering/test_steering_policy.py -q` — FAIL (missing fields, `FLOW_STEPS`, no processor module).

- [ ] **Step 3: Implement** the interfaces above. `FLOW_STEPS` goes (the config field replaces it; `test_predict_action_chunk_is_the_plain_policy_chunk` becomes the normalized assertion plus the postprocessor one). In `SteeringPolicy`, `core` builds `FlowControlSteering(self.base_policy, None, self._postprocessor, ...)` / `FlowReversalSteering(self.base_policy, None, self._postprocessor, n_reversal_steps=..., operator_dims=..., pinned_steps=..., deadband=...)` with no `reference_fn`; the FRS core's default reference computes `(per_step * step - mean) / std` in that order (`per_step` the held command, `step = max_speed * dt`), which for `mean = 0` equals `Flow.held_reference`'s `per_step * step / action_scale` bit for bit; keep `Flow.held_reference` (the example and the equivalence suites use it). A postprocessor is required for FRS on every base (the cursor's comes from Task 1's factory; `SteeringPolicy.base_policy` loads it from the folder as today).

- [ ] **Step 4: Run the suites** (Global Constraints) — PASS; the pinned `test_steering_policy_cursor.py` unchanged.

- [ ] **Step 5: Commit** `MARS: the cursor policy offers the sampler contract; SteeringPolicy has one path`.

### Task 2: The cursor example runs SteeringPolicy

**Files:**

- Modify: `examples/pi05/mars/cursor/cursor_sa/policy.py`, `cursor_sa/steering.py`, `cursor_sa/session.py`,
  `cursor_sa/agents.py`, `run_sweep.py`, `train/train_adapter.py`, `train/train_policy.py`, `tools/hub.py`,
  `tools/convert_checkpoints.py`, `README.md`; `src/lerobot/policies/steering/modeling_steering.py` (delete
  `SampleChunkSampler`), `__init__.py`, `README.md`
- Test: `tests/examples/pi05_mars_cursor/test_steering_equivalence.py` (import lines only),
  `tests/policies/steering/test_steering_shared.py` (the one allowed fixture edit), `test_session.py`, `test_sweep.py`,
  `test_steering_policy_cursor.py` (unchanged)

**Interfaces:**

- Consumes: Task 1's `CursorFlowPolicy` contract, `make_cursor_flow_pre_post_processors`, `SteeringConfig.mode = "none"`.
- Produces: `cursor_sa.policy.load_policy(path, device="cpu") -> CursorFlowPolicy` (a lerobot policy directory only, in
  eval mode on `device`; a `policy.pt` is the converter's business); `cursor_sa.policy.as_policy(model_or_policy) ->
CursorFlowPolicy` (a `FlowPolicy` wrapped with a config read from its attributes `horizon, hist, d_model, layers,
heads, d_ff, mode, action_scale, obstacle_tokens`, sharing the module, not a copy; a `CursorFlowPolicy` as is).
- Produces: `SharedController.__init__(self, policy, layouts, mode, ...)` with the same parameters as today; `policy`
  a `CursorFlowPolicy` (or a `FlowPolicy`, wrapped by `as_policy`); it builds
  `self.steering = SteeringPolicy(self.steering_config(mode, tau, n_reverse_steps, steps, execute, dt, max_speed,
ic_gain, lam, pinned_steps), base_policy=policy, postprocessor=make_cursor_flow_pre_post_processors(policy.config)[1])`
  and sets `self.steering.adapter = as_adapter(adapter, device)` (`None` -> `AdapterModules()`); mode `policy` maps to
  `SteeringConfig.mode = "none"`, `teleop` builds no steering policy. `act` obtains the chunk as
  `self.steering.predict_action_chunk({**obs, "command": commands})` unnormalized by the postprocessor;
  `sample_chunk(obs, z=None)` is `self.steering.predict_action_chunk(obs, noise=z)` (no command: the plain chunk
  through the task hook, as `_chunk(sampler, None, ...)` gave). `steering_config(...) -> SteeringConfig` is a
  `@staticmethod` on the controller: `SteeringConfig(mode=..., tau=tau, lam=lam, ic_gain=ic_gain,
n_reversal_steps=n_reverse_steps, pinned_steps=pinned_steps, deadband=DEADBAND, n_dims=2, horizon=policy.horizon,
num_inference_steps=steps, n_action_steps=execute, max_speed=max_speed, dt=dt)`.
- Produces: `steered_chunk(policy, obs, command, mode, tau=5, n_reverse_steps=None, steps=10, dt=0.02, max_speed=MAX_SPEED,
adapter=None, reference=None, ic_gain=1.0, lam=1, pinned_steps=None) -> Tensor` keeps its signature and its return
  (screen units per step), implemented as one `SteeringPolicy` call on `as_policy(policy)`.
- Deleted: `steering_wrapper`, `_chunk`, `_field`, `action_gain` from `cursor_sa/steering.py`; `SampleChunkSampler`
  from the package; `cursor_sa.policy.FlowPolicy.load`'s dual-format reading and `load_checkpoint` (Task 5 gives the
  converter the `policy.pt` reader). Kept: `observation`, `hide_goal`, `view_scene_fn`, `to_action`, `to_deflection`,
  `bci_chunk`, `flow`, `reverse_flow`, `forward_flow`, `method_name`, `steered_chunk`, `SharedController`.
- `train/train_policy.py` saves with `CursorFlowPolicy.save_pretrained(policy_dir)` (no `policy.pt`); the trainer
  `train_adapter.py` loads `load_policy(...).model` (its velocity calls stay on `FlowPolicy`); `load_checkpoints`
  returns `CursorFlowPolicy` objects and `run_cell_batch` / `run_trials` pass them to the controller.

- [ ] **Step 1: Write the failing tests** in `tests/examples/pi05_mars_cursor/test_session.py`:

```python
def test_the_controller_runs_a_steering_policy(small_policy, layouts):
    ctl = SharedController(small_policy, layouts, "frs", n_reverse_steps=4, device="cpu")
    assert isinstance(ctl.steering, SteeringPolicy) and ctl.steering.config.mode == "frs"
    assert ctl.steering.base_policy is as_policy(small_policy) or ctl.steering.base_policy.model is small_policy.model

def test_policy_mode_is_mode_none():
    ctl = SharedController(small_policy, layouts, "policy", device="cpu")
    assert ctl.steering.config.mode == "none"

def test_steered_chunk_matches_the_controller_bit_for_bit(): ...   # steered_chunk(...) == ctl.act's chunk * 1 under one seed
```

and `test_sweep.py`: `load_checkpoints` returns `CursorFlowPolicy` instances and `steering_wrapper` no longer exists
(`not hasattr(cursor_sa.steering, "steering_wrapper")`).

- [ ] **Step 2: Run them** — FAIL (`ctl.steering` missing).

- [ ] **Step 3: Implement** the interfaces. In `tests/policies/steering/test_steering_shared.py` the fixture class
      `CursorSampler(SampleChunkSampler)` becomes `as_policy(FlowPolicy(...))` (Task 2's one allowed pinned edit: the
      fixture only). Grep `SampleChunkSampler`, `steering_wrapper`, `_chunk(`, `FlowPolicy.load`, `load_checkpoint`
      across `src/`, `examples/`, `tests/`: nothing left but the example's `flow` / `reverse_flow` / `forward_flow`.

- [ ] **Step 4: Run the suites** — PASS; the pinned `test_steering_equivalence.py` changes import lines only (list
      them in the report); the tiny `adapter.pt` hash equals the base's.

- [ ] **Step 5: Commit** `MARS: the cursor example runs SteeringPolicy`.

### Task 3: One adapter holder

**Files:**

- Modify: `src/lerobot/policies/steering/modeling_steering.py` (`AdapterModules`, `AdaptedField`, `Flow.adapt_velocity`,
  `FlowReversalSteering`, `FlowControlSteering`), `__init__.py`, `README.md`; `examples/pi05/mars/libero/session.py`,
  `experiment.py`, `interactive.py` (every `ReversalAdapter` / `MatrixHolder` use)
- Test: `tests/policies/steering/test_adapter_modules.py`, `test_steering_core.py` and `test_steering_wrappers.py`
  (rename `ReversalAdapter` -> `AdapterModules`, `field(std, like)` -> `field(std=std, like=like)`; nothing else)

**Interfaces:**

- Produces: `AdapterModules.__init__(self, user=None, task=None, matrix=None, n_dims: int | None = 2, label: str | None =
None)`; `matrix` a property holding `np.ndarray | None` in the operator's (environment) units, its setter validating
  with `SteeringConfig.checked_matrix(value, self.n_dims or N_ACTION_DIMS, "adapter matrix")` (a `(B, n, n)` batch of
  matrices also accepted, as today's tensor was); attributes `label`, `user_label`, `task_label`; `kind` as today;
  `describe() -> str` (ReversalAdapter's text, "MARS adapter: off" when none); `format() -> str` and
  `format_rows(matrix) -> str` (MatrixHolder's); `load_modules(user_path=None, task_path=None, device="cpu") -> None`
  (ReversalAdapter's: kinds checked, labels set); `normalized_matrix(std: np.ndarray) -> np.ndarray | None` =
  `self.matrix * std[np.newaxis, :] / std[:, np.newaxis]` in that order, returning `self.matrix` itself when
  `np.all(std == std.flat[0])`; `field(obs=None, *, std=None, like=None) -> AdaptedField | None`: None when `kind ==
"none"`; the matrix (normalized when `std` is given) as a tensor of `like.dtype` and `like.device` when `like` is
  given, else float32 on the observation's device; the task module bound to `obs` when `obs` is given; `adapt(obs, x,
t, v, backward)` as today.
- Produces: `FlowReversalSteering(..., adapter: AdapterModules | None = None)` and `FlowControlSteering` read
  `adapter.field(std=self._std, like=x_t)`; `Flow.adapt_velocity` accepts a tensor, an `AdaptedField` or an
  `AdapterModules` (its `field(like=x)` with no std: the holder must be normalized already, or have no fixed matrix —
  raise `ValueError` naming `field(std=...)` otherwise, as today's `ReversalAdapter` branch does).
- Deleted: `ReversalAdapter`, `MatrixHolder`, their `__all__` entries. `SteeringConfig.adapter_matrix`'s doc: the
  operator's units, normalized at use by the postprocessor's statistics.
- `SteeringPolicy._field(obs)` passes `std` from the postprocessor's action statistics when a fixed matrix is set
  (`FlowControlSteering.action_mean_std(self._postprocessor)[1]`); the cursor's uniform std returns the matrix unchanged.

- [ ] **Step 1: Write the failing tests** in `tests/policies/steering/test_adapter_modules.py`:

```python
def test_normalized_matrix_is_the_pinned_expression_and_uniform_std_is_identity():
    m = np.array([[1.0, 0.3], [0.1, 0.9]]); std = np.array([2.0, 0.5])
    holder = AdapterModules(matrix=m)
    assert np.array_equal(holder.normalized_matrix(std), m * std[np.newaxis, :] / std[:, np.newaxis])
    assert holder.normalized_matrix(np.array([0.0123, 0.0123])) is holder.matrix

def test_field_follows_like_and_std():      # dtype/device of `like`; std given -> normalized; no std -> as is
def test_describe_load_modules_and_labels(tmp_path):   # ReversalAdapter's behaviours on AdapterModules
def test_the_old_holders_are_gone():        # no ReversalAdapter / MatrixHolder in the package
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** The LIBERO session's `self.adapter = ReversalAdapter()` becomes
      `AdapterModules(n_dims=N_ACTION_DIMS)`; its `.matrix = ...`, `.label`, `load_modules`, `normalized_matrix`, `kind`,
      `user_label` / `task_label` reads stay by name.

- [ ] **Step 4: Run the suites** — PASS; `test_steering_core.py` / `test_steering_wrappers.py` edits are the renames
      named above (list every line).

- [ ] **Step 5: Commit** `MARS: one adapter holder`.

### Task 4: One spec grammar

**Files:**

- Modify: `src/lerobot/teleoperators/readout.py` (gains `matrix_spec`, `corruption_spec`),
  `src/lerobot/policies/steering/configuration_steering.py` (`build_adapter`, `from_spec` take `n_dims`;
  `translation`, `rotation` go), `examples/pi05/mars/cursor/cursor_sa/config.py` and every importer of
  `cursor_sa.readout` (`train_adapter.py`, `session.py`, `agents.py`, tests), `examples/pi05/mars/libero/config.py`
- Delete: `examples/pi05/mars/cursor/cursor_sa/readout.py`
- Test: `tests/teleoperators/test_readout.py`, `tests/policies/steering/test_steering_wrappers.py` (name-only:
  `SteeringConfig.translation(x)` -> `matrix_spec(x, 3)`, `SteeringConfig.rotation(20)` -> `matrix_spec({"rotation_z_deg":
20}, 3)`), `tests/examples/pi05_mars_cursor/test_cursor_flow.py` (import lines)

**Interfaces:**

- Produces: `lerobot.teleoperators.readout.matrix_spec(spec, n_dims: int, where: str = "matrix") -> np.ndarray | None`:
  `None` for `None`; `{scale: [s_1..s_n]}` (exactly `n_dims` entries) the diagonal; `{rotation_deg: d}` (n_dims == 2)
  the plane rotation `[[c, -s], [s, c]]` with `c, s = cos(radians(d)), sin(radians(d))`; `{rotation_z_deg: d}`
  (n_dims == 3) the rotation about z (`SteeringConfig.rotation`'s block, same expressions); `{M: rows}` the literal;
  bare rows the literal; any other key raises `ValueError(f"{where}: expected exactly one of {keys} or M, got ...")`
  where `keys` names the rotation key for that dimension (so `rotation_deg` with 3 dims raises and names
  `rotation_z_deg`); checked by `SteeringConfig.checked_matrix(m, n_dims, where)`.
- Produces: `readout.corruption_spec(spec, where="control.corruption") -> tuple[np.ndarray | None, dict | None]` (the
  example's, for 2 dims: `matrix_spec(spec, 2, where)` or `{curl, tau}`).
- Produces: `SteeringConfig.build_adapter(spec, corruption=None, where="reversal_adapter", n_dims=N_ACTION_DIMS)`: with
  `n_dims == 2` the only block is `translation` (`corruption`, `corruption_inverse`, `identity`, `zero` or a matrix spec
  for 2 dims; `orientation` / `gripper` keys raise), `{F: rows}` literal for `n_dims`; `from_spec(spec, corruption=None,
where=..., n_dims=N_ACTION_DIMS, **fields)` passes `n_dims` down and sets `n_dims` on the config.
- The example's `adapter_spec(spec, corruption)` -> `SteeringConfig.from_spec(spec, corruption, n_dims=2).adapter_matrix`
  (as a `np.ndarray` or None) at its callers; `matrix_spec` / `corruption_spec` imports move to the readout's.
  `configuration_steering.py` keeps importing `readout.DEADBAND` (the dependency direction is readout <- steering).

- [ ] **Step 1: Write the failing tests** (`tests/teleoperators/test_readout.py`):

```python
def test_matrix_spec_forms_per_dimension():
    assert np.array_equal(matrix_spec({"scale": [1.0, 0.25]}, 2), np.diag([1.0, 0.25]))
    assert np.allclose(matrix_spec({"rotation_deg": 90}, 2), [[0, -1], [1, 0]])
    c, s = np.cos(np.radians(20)), np.sin(np.radians(20))
    assert np.array_equal(matrix_spec({"rotation_z_deg": 20}, 3), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="rotation_z_deg"):
        matrix_spec({"rotation_deg": 20}, 3)
    with pytest.raises(ValueError, match="rotation_deg"):
        matrix_spec({"rotation_z_deg": 20}, 2)

def test_corruption_spec_is_the_examples():   # {curl, tau} -> (None, dict); a matrix form -> (M, None); bad curl raises

def test_build_adapter_two_dims_refuses_blocks_it_has_no_dims_for():  # orientation key with n_dims=2 raises; translation: corruption copies the 2x2
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; delete `cursor_sa/readout.py`; grep `cursor_sa.readout`,
      `SteeringConfig.translation`, `SteeringConfig.rotation`: nothing left.

- [ ] **Step 4: Run the suites** — PASS. **Step 5: Commit** `MARS: one spec grammar, the matrices' with the readout`.

### Task 5: One checkpoint format

**Files:**

- Modify: `examples/pi05/mars/cursor/tools/convert_checkpoints.py` (gains `from_cursor_checkpoint(ckpt) -> dict`,
  moved verbatim from `AdapterModules`, and `read_policy_pt(path) -> dict` from `cursor_sa.policy.load_checkpoint`),
  `src/lerobot/policies/steering/modeling_steering.py` (delete `AdapterModules.from_cursor_checkpoint`),
  `tools/hub.py`, `README.md`
- Test: the pinned suites' import lines (`tests/policies/steering/test_steering_core.py:408-418`,
  `tests/examples/pi05_mars_cursor/test_ranks_equivalence.py`, `test_steering_equivalence.py:312`,
  `test_steering_core_cursor.py`) load it from the converter module through the existing `_converter()` loader of
  `tests/policies/cursor_flow/test_cursor_flow.py` (move that loader to `tests/examples/pi05_mars_cursor/conftest.py` as
  a fixture `converter`); `tests/policies/cursor_flow/test_cursor_flow.py` (the conversion tests unchanged in asserts)

**Interfaces:**

- Produces: `convert_checkpoints.from_cursor_checkpoint(ckpt: dict) -> dict` (the deleted static method, same body);
  `convert_checkpoints.read_policy_pt(path) -> dict` (`{config, state_dict}` of a trainer's `policy.pt`);
  `convert_policy(path)` reads `policy.pt` through it. `AdapterModules.read_checkpoint`'s arch refusal stays (a
  validation, not a migration).

- [ ] **Step 1: Write the failing test** (`tests/policies/cursor_flow/test_cursor_flow.py`):
      `test_the_package_has_no_cursor_migration()`: `not hasattr(AdapterModules, "from_cursor_checkpoint")` and the
      converter's `from_cursor_checkpoint` converts the fixture checkpoint as before (the existing conversion asserts).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Suites** — PASS (pinned edits: import lines only, listed).
- [ ] **Step 5: Commit** `MARS: the cursor-format migrations live in the converter`.

### Task 6: The package independent of the cursor environment

**Files:**

- Modify: `src/lerobot/policies/steering/processor_steering.py` (no `lerobot.envs.cursor` import; `SCENE_IMAGE_SIZE =
224`; `SteeringSceneTokensProcessorStep` reads `observation.scene_images` only; `SceneEncoder.scene_array` goes),
  `modeling_steering.py` (`SCENE_SIZE` -> `SCENE_IMAGE_SIZE`), `src/lerobot/policies/cursor_flow/processor_cursor_flow.py`
  (gains `CursorSceneRenderProcessorStep` and `render_scene_array`), `src/lerobot/envs/cursor.py` (`SCENE_SIZE` stays),
  `examples/pi05/mars/cursor/run_sweep.py` (`attach_scene`), `cursor_sa/dataset.py`, `train/train_adapter.py`,
  `tools/convert_checkpoints.py` (prepends the render step to a converted steering folder's pre-processor), READMEs
- Test: `tests/policies/steering/test_processor_steering.py`, `test_scene_tokens.py`, `test_scene_encoder.py`,
  `tests/policies/cursor_flow/test_cursor_flow.py`, `tests/examples/pi05_mars_cursor/test_dataset.py` (digests pinned,
  unchanged)

**Interfaces:**

- Produces: `processor_steering.SCENE_IMAGE_SIZE: int = 224` (the default `render_size` / `image_size` everywhere the
  package had `SCENE_SIZE`); a test asserts `SCENE_IMAGE_SIZE == lerobot.envs.cursor.SCENE_SIZE`.
- Produces: `SteeringSceneTokensProcessorStep(source, render_size, vlm_path, batch, device, reduction, encoder)` reading
  `observation.scene_images` (uint8 `[B, 3, S, S]`, `S == render_size` else `ValueError` before encoding) and, when
  present, `observation.scene_digest` (a `str`) as the memo key, else `hashlib.sha256(pictures.tobytes()).hexdigest()`;
  `OBS_LAYOUTS` and the layouts branch go; `OBS_SCENE_DIGEST = f"{OBS_PREFIX}scene_digest"`.
- Produces (`processor_cursor_flow.py`): `@ProcessorStepRegistry.register(name="cursor_scene_render_processor_step")
@dataclass class CursorSceneRenderProcessorStep(ProcessorStep)` with `render_size: int = SCENE_SIZE`: reads
  `observation.layouts` (a `Layouts`), writes `observation.scene_images = render_layouts(layouts, render_size)` and
  `observation.scene_digest` (the sha256 over `start, targets, mask, obstacles, obstacle_mask` that the steering step
  computed, moved verbatim), memoizing the last layouts' render; `render_scene_array(layouts, source, size, encoder,
batch) -> np.ndarray` (the old `SceneEncoder.scene_array`, same body, importing `SceneEncoder._encode`).
- The steering factory `make_steering_pre_post_processors` is unchanged (it adds the tokens step); the example's
  pipelines and the converter's folders prepend `CursorSceneRenderProcessorStep(render_size)` when layouts are the
  input (one helper `cursor_steering_pre_post_processors(config)` in `cursor_sa/steering.py`).
- `run_sweep.attach_scene` and the dataset's encoding call `render_scene_array`; `run_sweep.scene_digest` is unchanged.

- [ ] **Step 1: Write the failing tests**:

```python
def test_the_package_imports_no_cursor_env():          # grep-free: importlib finds no "lerobot.envs.cursor" in the three modules' source
def test_tokens_step_reads_pictures_and_memoizes_on_the_digest():   # same pictures + digest -> encoder called once; a new digest -> twice
def test_tokens_step_refuses_the_wrong_picture_size():  # S != render_size raises before encode (Review Focus 5)
def test_render_step_writes_pictures_and_digest():      # equals render_layouts(layouts, size) and the moved digest
def test_scene_sizes_agree(): assert SCENE_IMAGE_SIZE == SCENE_SIZE
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Suites** — PASS; the digests test unchanged.
- [ ] **Step 5: Commit** `MARS: the steering package knows no cursor environment; the cursor renders its scenes`.

### Task 7: pi0.5 exposes its prefix encoding

**Files:**

- Modify: `src/lerobot/policies/pi05/modeling_pi05.py` (`_encode_prefix` -> public `encode_prefix` returning the
  prefix hidden state too), `src/lerobot/policies/steering/processor_steering.py` (`SceneEncoder.prefix_tokens`)
- Test: `tests/policies/steering/test_scene_encoder.py` (a stub `PI05Pytorch`-like model with `encode_prefix`),
  `tests/policies/pi05/` if a stub-level test of `sample_actions` exists (unchanged asserts)

**Interfaces:**

- Produces: `PI05Pytorch.encode_prefix(images, img_masks, tokens, masks) -> tuple[Tensor, Tensor, Any]` = `(hidden,
prefix_pad_masks, past_key_values)`, `hidden` the prefix output embeddings `(B, L, 2048)` (the first element of
  `paligemma_with_expert.forward(...)`'s output that `_encode_prefix` discarded), the rest as today including the
  eager-attention setting; `sample_actions` and `velocity_field` call it and ignore `hidden`.
- `SceneEncoder.prefix_tokens` becomes: `hidden, _, _ = model.encode_prefix([pixels], [ones], tokens.expand(b, -1),
masks.expand(b, -1))`; `out.append(hidden[:, :n_image].to(torch.float16).cpu())` with `n_image = hidden.shape[1] -
tokens.shape[1]`; its own mask/position/forward code goes.

- [ ] **Step 1: Write the failing test** (`test_scene_encoder.py`): a stub model whose `encode_prefix` records its
      arguments and returns a known `hidden`; `prefix_tokens` returns `hidden[:, :n_image]` as float16 numpy and called
      it once per batch; `not hasattr(PI05Pytorch, "_encode_prefix")`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Suites** — PASS.
- [ ] **Step 5: Commit** `pi0.5 exposes encode_prefix; the scene encoder uses it`.

### Task 8 (controller, GPU): acceptance against the pinned grid

- [ ] Record before Task 1: the tiny adapter's sha256 at the round's base and the base commit.
- [ ] After Task 7: the full suites; `tools/convert_checkpoints.py --help`, `run_sweep.py --help`, `train_adapter.py
--help` exit 0; then on a free GPU, from the main tree, `run_sweep.py --sweep grid_vision --cells U1xT2
--fixed-settings ../results/sim/sweeps/grid_vision_s0/settings.yaml --out grid_vision_s0_check_U1xT2 --results
<scratch>`; its `test.json["test"]["U1xT2|1.0"]["trials"]["outcome"]` equals `grid_vision_s0`'s for every method
      (`==` on the lists). A difference stops the round until explained.
- [ ] Ledger: rulings, the pinned-test edit list per task, the hashes.
