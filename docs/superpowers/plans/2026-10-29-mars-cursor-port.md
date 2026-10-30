# MARS cursor port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the cursor-control shared-autonomy project into this repo as `examples/pi05/mars/`, with one
task-generic MARS implementation in `src/lerobot/policies/pi05/steering.py` used by both the cursor and
LIBERO examples, and the same grid numbers as before.

**Architecture:** First copy the cursor project in unchanged and get its tests green in lerobot's `.venv`.
Then move the method into `steering.py` behind a velocity-callable interface, the cursor policy into
`policies/cursor_flow/`, and the SpaceMouse into `teleoperators/spacemouse/`. The cursor example becomes a
thin layer over them. Finally, prove equivalence twice: unit tests against a frozen copy of the old cursor
steering, then a seed-0 re-run of the grid's test stage.

**Tech Stack:** Python 3, PyTorch 2.11 (lerobot `.venv`, run through `uv run`), pytest, pygame, matplotlib,
reportlab, huggingface_hub.

**Spec:** `docs/superpowers/specs/2026-10-29-mars-cursor-port-design.md` (read it first; this plan argues from it).

## Global Constraints

- Source of the copy: ricebcirig commit `7a3fddb6`, folder `analysis/nishal/cursor_control_shared_autonomy`
  (call it `$SRC`). It is the last commit without the rank knobs, so nothing rank-related comes over.
  Copy it with `git -C <ricebcirig clone> archive 7a3fddb6 analysis/nishal/cursor_control_shared_autonomy | tar -x -C /tmp/...`.
  The working copy at `bd2a48f2` is only for reading `checkpoints/` and `results/` (gitignored, not in the archive).
- Work on a branch `mars-cursor-port` off `main` in the lerobot checkout.
- **No `git commit`, `git push` or Hub upload unless Nishal commands it at that time.** Each task's last step
  stages its files; it commits only if Nishal has said to.
- Code in `src/` never imports from `examples/`.
- No method behavior or default value changes. LIBERO's existing tests pass unchanged except for import paths.
- No machine paths (`/home/user`, `t15`, `brand-rice`, `miniconda`) in committed files.
- GPU work only on GPU 1 (`CUDA_VISIBLE_DEVICES=1`), at most 3 trainings and 6 processes at once.
- `src/` files: Apache license header, type hints, ruff clean (`uv run ruff check src tests`).
  Example code keeps the cursor project's style.
- `examples/pi05/mars/results/` and `examples/pi05/mars/checkpoints/` are gitignored.
- Test command: `uv run --extra mars pytest <path> -q` from the repo root.

## Review Focus

1. **Per-trial fixed matrices.** The cursor passes F as `[B, 2, 2]` (one per trial). `AdaptedField` today
   reads `n = matrix.shape[0]`, which would be B. The test is in Task 3.
2. **Rows inside the deadband.** An idle operator's rows must get the plain policy chunk drawn from the same
   noise as mode "policy". The equivalence test in Task 4 includes idle rows.
3. **Time conventions.** The cursor's flow time goes 0 = noise → 1 = action; pi0.5's goes 1 = noise → 0 = action.
   A sign slip reverses the flow silently. Task 3 tests both conventions against the old functions with `torch.equal`.
4. **Old checkpoints.** Cursor policy and adapter files without `kind` / `n_dims` (the first adapters were
   saved without a kind; they are user modules) must convert, and must load identically. Tests are in Tasks 3 and 5.
5. **SpaceMouse absent or a different model.** No device means a clear error naming the HID ids it looked
   for. A `256F` vendor id is found. Tests are in Task 6.

---

### Task 1: Branch, extra, copy the cursor project in, tests green as is

**Files:**

- Create: `examples/pi05/mars/cursor/` (the whole `$SRC` tree except `figures/` and `README.md`), `examples/pi05/mars/figures/` (from `$SRC/figures/`), `examples/pi05/mars/cursor/README.md` (from `$SRC/README.md`)
- Create: `tests/examples/pi05_mars_cursor/conftest.py`, then move `$SRC/tests/*.py` into `tests/examples/pi05_mars_cursor/`
- Modify: `pyproject.toml` (new `mars` extra), `uv.lock`, `.gitignore`
- Modify: `examples/pi05/mars/cursor/cursor_sa/paths.py`, `examples/pi05/mars/figures/build_figures.py` (its `sys.path` insert), every `reports/build_*.py` docstring that names the `t15` venv

**Interfaces:**

- Produces: `cursor_sa.paths.ROOT == examples/pi05/mars/cursor`, `RESULTS == ROOT.parent / "results"`,
  `CHECKPOINTS == ROOT.parent / "checkpoints"`; `CONFIGS` and `REPORTS` stay under `ROOT`.
  The example's import root is `examples/pi05/mars/cursor` (scripts do `from cursor_sa import ...`).

- [ ] **Step 1:** `git switch -c mars-cursor-port`. Extract the archive into the scratchpad and copy the tree into the layout above. Do not copy `results/` or `checkpoints/`.
- [ ] **Step 2:** Add the extra `mars = ["lerobot[pygame-dep]", "lerobot[matplotlib-dep]", "lerobot[scipy-dep]", "reportlab>=4,<5", "pillow>=10", "pyyaml>=6"]`. Pin reportlab to whatever range `uv lock` resolves; match the pin style used elsewhere in the file. Then run `uv lock`.
- [ ] **Step 3:** Add `examples/pi05/mars/results/` and `examples/pi05/mars/checkpoints/` to `.gitignore`. Point `paths.py` as in Interfaces. `build_figures.py` inserts `Path(__file__).resolve().parents[1] / "cursor"` into `sys.path`. Report docstrings say `python reports/build_*.py`.
- [ ] **Step 4:** Write `conftest.py` the way `tests/examples/pi05_libero_shared_autonomy/conftest.py` does: it inserts `examples/pi05/mars/cursor` and `examples/pi05/mars` (for `figures`) into `sys.path`.
- [ ] **Step 5:** Copy the needed artifacts from the `bd2a48f2` working copy:
  - `results/` → `examples/pi05/mars/results/`, without `sim/sweeps/adapter_ranks` or `research/adapter_ranks`
  - `checkpoints/` → `examples/pi05/mars/checkpoints/`, without `adapters/adapter_ranks`

  Use `rsync -a --exclude adapter_ranks`.

- [ ] **Step 6:** Run `uv run --extra mars pytest tests/examples/pi05_mars_cursor -q`. Expected: everything passes, including the grid- and figure-dependent tests that skip when results are missing (they must run here, not skip). Fix only import paths. Any other failure is a porting bug to report, not to patch.
- [ ] **Step 7:** Run `grep -rnE "/home/user|t15|brand-rice|miniconda" examples/pi05/mars --include=*.py --include=*.md --include=*.yaml`. Expected: no hits. A README that names the old ricebcirig location as history is fine if it carries no absolute path.
- [ ] **Step 8:** Stage. Commit only on Nishal's command (message: "MARS: copy the cursor-control project in as examples/pi05/mars").

### Task 2: Move the LIBERO example under `mars/`

**Files:**

- Move: `examples/pi05/libero_shared_autonomy/` → `examples/pi05/mars/libero/` (`git mv`)
- Move: `tests/examples/pi05_libero_shared_autonomy/` → `tests/examples/pi05_mars_libero/` (`git mv`); fix `EXAMPLE_DIR` in its `conftest.py`
- Modify: `examples/pi05/README.md`; `examples/pi05/mars/libero/README.md`, `env.sh` (`REPO_ROOT` is now four levels up), `run.sh`, and the notebook's paths; `tests/examples/pi05_mars_libero/test_analyze.py`
- Create: `examples/pi05/mars/README.md`

**Interfaces:**

- Produces: the LIBERO example at `examples/pi05/mars/libero/`, otherwise unchanged.

- [ ] **Step 1:** Do the `git mv`s. Then `grep -rn "libero_shared_autonomy" examples tests docs/README* src` and fix every hit outside `docs/superpowers/`. The old specs and plans stay as written.
- [ ] **Step 2:** In `env.sh`, `REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"`. Check it with `bash -c 'source examples/pi05/mars/libero/env.sh && echo $LIBERO_CONFIG_PATH'`. Expected: `<repo>/.cache/libero`.
- [ ] **Step 3:** Write `examples/pi05/mars/README.md`. It covers:
  - one paragraph on MARS and the two tasks
  - a day-one quick start: install, `hub.py download`, a study session, the metrics (exact commands from Task 8)
  - links to `cursor/README.md`, `libero/README.md` and `figures/`

  Replace the libero_shared_autonomy entry in `examples/pi05/README.md` with a one-paragraph `mars/` entry.

- [ ] **Step 4:** Run `uv run --extra pi --extra libero pytest tests/examples/pi05_mars_libero tests/policies/pi0_pi05/test_pi05_steering.py -q`. Expected: the same pass/skip counts as on `main` before the move. Record both counts in the task report.
- [ ] **Step 5:** Stage; commit only on command ("MARS: move the LIBERO shared-autonomy example under examples/pi05/mars/libero").

### Task 3: `steering.py` core: one integrator, per-trial matrices, the obstacle task module

**Files:**

- Modify: `src/lerobot/policies/pi05/steering.py`
- Create: `tests/examples/pi05_mars_cursor/_reference_steering.py`. It is the `7a3fddb6` cursor `steering.py`, frozen, with its relative imports (`.channel`, `.policy`) rewritten to `cursor_sa.channel` / `cursor_sa.policy`. Tasks 3 and 4 compare against it.
- Test: `tests/policies/pi0_pi05/test_steering_core.py` (new; pi0.5 and generic tests), `tests/examples/pi05_mars_cursor/test_steering_core_cursor.py` (new; the tests that need the cursor reference)

**Interfaces:**

- Produces:
  - `integrate(x: Tensor, velocity: Callable[[Tensor, float | Tensor], Tensor], t0: float, h: float, n: int, adapter=None, backward: bool = False, grad: bool = False) -> Tensor`. It runs `n` Euler steps `x ← x + h · adapt(v(x, t))` with `t = t0 + i·h`. When `grad` is False it runs under `torch.no_grad()`.
  - `reverse_flow(...)` and `forward_flow(...)` keep their signatures and results and become one-line calls of `integrate`. pi0.5's forward pass is `t0=1, h=-1/N`; its reverse is `t0=0, h=+1/N, backward=True`.
  - `AdaptedField(matrix, user_module=None, task_module=None, n_dims: int | None = None)`. A matrix may be `(n, n)` or per-trial `(B, n, n)`, and `n` is read from `shape[-1]`. When `n_dims` is given it overrides each module's `n_dims` attribute; the default stays `N_ACTION_DIMS`.
  - `ObstacleTaskModule(horizon=16, d=128, tdim=64, heads=4, layers=2, scale=0.2, n_dims=2)` is the cursor `TaskModule`, moved as is. `forward(obs, x, t, v) -> (A, b)`, and `bind(obs) -> Callable[[x, t, v], (A, b)]`, whose result carries `n_dims`. It uses `sinusoidal_time_embedding` and a local `_gelu_mlp(d_in, d)` (Linear, GELU, Linear, the cursor's `mlp`).
  - Module checkpoints gain `"arch"`: `"mlp"` (the default, `LearnedTaskModule`) or `"obstacle"` (`ObstacleTaskModule`), for kind `task` only. `load_adapter_module` reads `arch`, defaulting to `"mlp"`, so old LIBERO files load.
  - `convert_cursor_module(ckpt: dict) -> dict`. The cursor format is `{kind?, config, state_dict, args, obstacle}`; the output is the `save_adapter_module` format.
    - A missing kind means `"user"`, built as `LearnedUserModule(horizon, n_dims=2, d, tdim)`.
    - The state-dict keys `net.0/2/4.*` are already identical.
    - Task modules get `arch="obstacle"`.
    - `args` and `obstacle` go into `meta`.

- [ ] **Step 1: Write the failing tests.**
  - `test_integrate_reproduces_pi05_flows_exactly`: on a random linear field and random `x`, `torch.equal(integrate(x, f, 0.0, 1/10, 10, backward=True), old_reverse_flow(x, f, 10))`. Same for `forward_flow` with `stop_at=4`. Before editing, copy the old two functions into the test as `old_*`.
  - `test_integrate_reproduces_cursor_flows_exactly`: the cursor's convention is reverse `t0=1, h=-1/N, backward=True` and forward `t0=1-n/N, h=(n/N)/m`. This test goes in `test_steering_core_cursor.py`. Assert `torch.equal` against `_reference_steering.reverse_flow` / `forward_flow`, with `policy(obs, x, t)` wrapped as `lambda x, t: policy(obs, x, t)` and `t` a `(B,)` tensor.
  - `test_adapted_field_takes_per_trial_matrices`: with B=3, n=2 and a per-trial `F` of shape `[3, 2, 2]`, `adapt(..., backward=True)` equals `einsum("bij,bhj->bhi", F, v)`. A 7-dim `v` with `F` `(2, 2)` changes only the first two dims.
  - `test_obstacle_task_module_starts_at_identity_and_round_trips`: with a fresh module, `A == I` and `b == 0` for obstacles present and absent. After `save_adapter_module(..., kind="task")` with `arch` set and then `load_adapter_module`, the outputs are identical.
  - `test_cursor_user_module_converts_and_matches`: this test goes in `test_steering_core_cursor.py`. Build `_reference_steering.UserModule(16)` and perturb its weights. Convert its checkpoint dict and load it. Assert `allclose(new(x, t), old(x, t), atol=1e-5)`. It is not exact, because the time embeddings compute the same frequencies by `exp` vs `10**`.
- [ ] **Step 2:** Run `uv run --extra mars --extra pi pytest tests/policies/pi0_pi05/test_steering_core.py tests/examples/pi05_mars_cursor/test_steering_core_cursor.py -q`. Expected: FAIL (the names are not defined).
- [ ] **Step 3:** Implement the Interfaces in `steering.py`. Keep the existing `reverse_flow` / `forward_flow` docstrings. In the module docstring, add one paragraph saying the core takes any velocity field and either time convention.
- [ ] **Step 4:** Run the new tests plus `tests/policies/pi0_pi05/test_pi05_steering.py` and `tests/examples/pi05_mars_libero`. Expected: all pass; the pass counts for the last two match Task 2's.
- [ ] **Step 5:** Stage; commit only on command ("steering: one Euler integrator for either flow-time convention, per-trial matrices, the obstacle task module").

### Task 4: Chunk functions and losses in `steering.py`; the cursor example on top

**Files:**

- Modify: `src/lerobot/policies/pi05/steering.py`
- Modify: `examples/pi05/mars/cursor/cursor_sa/steering.py`, `examples/pi05/mars/cursor/train/train_adapter.py`
- Test: `tests/examples/pi05_mars_cursor/test_steering_equivalence.py` (new)

**Interfaces:**

- Consumes: Task 3's `integrate`, `AdaptedField`, `ObstacleTaskModule.bind`, `LearnedUserModule`.
- Produces (in `steering.py`). All of these are in the cursor convention (t = 0 is noise), and each docstring says so:
  - `flow_control_chunk(velocity, z, written, steer, tau, lam, steps, adapter=None) -> Tensor`. `written` is the already-scaled reference `[B, H, n]`.
  - `frs_chunk(velocity, z, ref, steer, n_reverse_steps, steps, adapter=None) -> Tensor`.
  - `steered_chunk(velocity, command, mode, *, horizon, to_reference, tau=5, n_reverse_steps=None, steps=10, adapter=None, reference=None, ic_gain=1.0, lam=1, deadband=DEADBAND) -> Tensor`. It returns the normalized chunk. `to_reference(deflection) -> action-unit chunk` is the caller's conversion (the cursor passes its `to_action`). The noise draw `z = torch.randn(B, horizon, n)` stays first, as today, so random streams line up.
  - `user_field_loss(user, velocity, latent, target, t, task=None, weight=None) -> Tensor` and `task_field_loss(task, velocity, chunk, z, t, lambda_id, n_dims) -> (Tensor, dict)`. These are the bodies of `train_adapter.py`'s `field_loss` and `task_loss`, written over a velocity callable and a bound task callable.
- Produces (in the cursor example): `cursor_sa/steering.py` keeps its public names (`steered_chunk(policy, obs, command, mode, ...)`, `SharedController`, `ReversalAdapter`, `FixedAdapter`, `UserModule`, `TaskModule`, `as_adapter`, `bci_chunk`, `to_action`, `to_deflection`, `observation`, `action_gain`) so the rest of the example is untouched.
  - The method parts become wrappers. Each binds `obs` (`velocity = lambda x, t: policy(obs, x, t)`, `task.bind(obs)`), builds an `AdaptedField(..., n_dims=2)`, applies `policy.action_scale`, and calls the core.
  - `UserModule` is `LearnedUserModule` with `n_dims=2`, and `TaskModule` is `ObstacleTaskModule`; aliases are fine.
  - `ReversalAdapter.load` reads both the old and the converted checkpoint formats.

- [ ] **Step 1: Write `test_steering_equivalence.py`.** For each mode in `policy`, `teleop`-free shared modes (`flow_control` with `lam` 1 and 3, `frs` with `n_reverse_steps` None and 4):
  - Run with a fixed-seed small `FlowPolicy` (random weights), a batch of 6 commands where 2 are inside the deadband, and an adapter that is each of: none, a per-trial F `[6, 2, 2]`, a perturbed user module, a perturbed obstacle task module with obstacles, and user plus task.
  - Assert `allclose(new, reference, atol=1e-5)` for chunks from `cursor_sa.steering.steered_chunk` versus `_reference_steering.steered_chunk`, under `torch.manual_seed(0)` before each.
  - Also assert that `user_field_loss` / `task_field_loss` equal the reference `field_loss` / `task_loss` under the same seed.
- [ ] **Step 2:** Run it. Expected: it passes trivially, because nothing has moved yet. It is the guard for Step 3.
- [ ] **Step 3:** Implement the Interfaces. Remove the moved bodies from the cursor files; `train_adapter.py` calls the steering losses.
- [ ] **Step 4:** Run `uv run --extra mars --extra pi pytest tests/examples/pi05_mars_cursor tests/policies/pi0_pi05 tests/examples/pi05_mars_libero -q`. Expected: all pass, with the equivalence test included.
- [ ] **Step 5:** Run `grep -n "cursor_sa\|examples" src/lerobot/policies/pi05/steering.py`. Expected: no imports (docstring mentions only).
- [ ] **Step 6:** Stage; commit only on command ("steering: the cursor's steered chunk, flow control, FRS and MARS losses on the shared core").

### Task 5: The cursor policy as `policies/cursor_flow/`, the env as `envs/cursor.py`, and the checkpoint converter

The env moves here too: `cursor_sa/env.py` imports only numpy, heapq and dataclasses. It moves verbatim to
`src/lerobot/envs/cursor.py`, and `cursor_sa/env.py` becomes `from lerobot.envs.cursor import *` (plus any
private names its 23 importers use), so nothing else in the example changes. It is not registered as an
`EnvConfig` (the builder's call in the spec; nothing needs it yet). Test: the example's
`test_obstacles.py` and `test_session.py` pass unchanged after the move.

**Files:**

- Create: `src/lerobot/envs/cursor.py` (from `cursor_sa/env.py`, license header added); Modify: `examples/pi05/mars/cursor/cursor_sa/env.py` (re-export)
- Create: `src/lerobot/policies/cursor_flow/__init__.py`, `configuration_cursor_flow.py`, `modeling_cursor_flow.py`, `README.md`
- Modify: `src/lerobot/policies/factory.py` (`get_policy_class`, `make_policy_config`), `src/lerobot/policies/__init__.py` if other configs are imported there
- Modify: `examples/pi05/mars/cursor/cursor_sa/policy.py` (re-exports `FlowPolicy`, `time_embedding`, `mlp`; `load` reads either format), `cursor_sa/paths.py`
- Create: `examples/pi05/mars/cursor/convert_checkpoints.py`
- Test: `tests/policies/cursor_flow/test_cursor_flow.py`

**Interfaces:**

- Produces:
  - `CursorFlowConfig(PreTrainedConfig)`, registered as `"cursor_flow"`, with fields mirroring `FlowPolicy`'s config (`horizon=16, hist=4, d_model=128, layers=4, heads=4, d_ff=512, mode="flow", action_scale=0.01, obstacle_tokens=False`). The abstract presets return lerobot's defaults (AdamW, no scheduler). `validate_features` is a no-op, and the delta indices are None.
  - `CursorFlowPolicy(PreTrainedPolicy)`, `name = "cursor_flow"`, holding `self.model: FlowPolicy`. The cursor code uses `.model` directly.
    - `predict_action_chunk(batch)` returns a 10-step forward-flow chunk (`steering.integrate`, cursor convention) times `action_scale`.
    - `select_action` returns its first step.
    - `forward` raises `NotImplementedError("trained by examples/pi05/mars/cursor/train/train_policy.py")`.
    - `get_optim_params` returns `self.parameters()`, and `reset` is a no-op.
  - `FlowPolicy` (with `Block`, `time_embedding`, `mlp`) moves verbatim into `modeling_cursor_flow.py`.
  - `convert_checkpoints.py [--checkpoints DIR]` converts every `policies/<name>/policy.pt` to `policies/<name>/` holding `config.json` + `model.safetensors` (`save_pretrained`). It converts every `adapters/<name>/adapter.pt` in place to the `save_adapter_module` format (`convert_cursor_module`), keeping `adapter.pt.orig`. Re-running it is a no-op.
  - `cursor_sa.paths.policy_path(name)` returns the converted directory. `FlowPolicy.load(path)` (the example's) loads a directory through `CursorFlowPolicy.from_pretrained(path).model`, and an old `.pt` as before.

- [ ] **Step 1: Write the failing tests.**
  - `test_round_trip_is_exact`: a small random `FlowPolicy`, saved in the old format to `tmp_path`, converted and loaded by `from_pretrained`, gives `torch.equal` velocities on a random obs dict (with and without obstacles, `obstacle_tokens` True and False).
  - `test_factory_knows_cursor_flow`: `get_policy_class("cursor_flow") is CursorFlowPolicy`.
  - `test_converter_is_idempotent`: two runs over a tmp checkpoints tree, where the second changes no file's mtime.
- [ ] **Step 2:** Run `uv run --extra mars pytest tests/policies/cursor_flow -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run --extra mars python examples/pi05/mars/cursor/convert_checkpoints.py` on the copied checkpoints. Then run `tests/policies/cursor_flow`, `tests/examples/pi05_mars_cursor` and `tests/policies/pi0_pi05`. Expected: all pass.
- [ ] **Step 5:** Stage; commit only on command ("policies: cursor_flow, the cursor task's flow-matching policy; checkpoint converter").

### Task 6: SpaceMouse teleoperator; both examples use it

**Files:**

- Create: `src/lerobot/teleoperators/spacemouse/__init__.py`, `configuration_spacemouse.py`, `teleop_spacemouse.py`
- Modify: `src/lerobot/teleoperators/__init__.py` and `utils.py` (register it, as `gamepad` is registered), `pyproject.toml` (a `spacemouse = []` extra only if the gamepad pattern needs one; hidraw needs no package)
- Modify: `examples/pi05/mars/cursor/cursor_sa/devices.py`, `examples/pi05/mars/libero/teleop.py`
- Test: `tests/teleoperators/test_spacemouse.py`

**Interfaces:**

- Produces:
  - `SPACEMOUSE_VENDORS = ("046D", "256F")`.
  - `find_spacemouse(hid_id: str | None = None, sysfs: Path = Path("/sys/class/hidraw")) -> str`. With `hid_id` None it returns the first hidraw node whose uevent `HID_ID` has a 3Dconnexion vendor. It raises `FileNotFoundError` naming the ids searched.
  - `parse_report(data: bytes) -> tuple[str, tuple[int, int, int] | int | None]`: raw counts, as the LIBERO reader returns them today.
  - `SpaceMouseReader(path: str, full_scale: float = 350.0)`, a background thread. `.axes` is the device channels `(x, y, z)` divided by `full_scale` and clipped to [-1, 1]. `.buttons` is the bitmask. `take_press() -> bool` is true once per new press. It also has `.connected` and `close()`.
  - `SpaceMouseConfig(TeleoperatorConfig)`, registered as `"spacemouse"`, with `hid_id: str | None = None` and `full_scale: float = 350.0`. `SpaceMouseTeleop(Teleoperator)`: `get_action()` returns `{"x": float, "y": float, "z": float, "buttons": int}`. `is_calibrated` is True and `calibrate` / `configure` are no-ops.
  - Each example keeps its own axis mapping. The cursor maps to `[x, -y]`; LIBERO uses `AXIS_SOURCE` / `AXIS_SIGN`. The cursor's `SpaceMouse` and LIBERO's `SpaceMouseReader` keep their public attributes as thin wrappers.
  - When `hid_id` is None, the default search still finds today's `0000046D:0000C62B` device.

- [ ] **Step 1: Write the failing tests.**
  - `test_parse_report`: bytes `b"\x01" + struct.pack("<hhh", 350, -175, 0)` give `("translation", (350, -175, 0))`, and `b"\x03\x01\x00\x00\x00"` gives `("buttons", 1)`.
  - `test_find_by_vendor`: a fake sysfs tree in `tmp_path` holds `hidraw3/device/uevent` with `HID_ID=0003:0000256F:0000C635` and returns `/dev/hidraw3`.
  - `test_missing_device_names_the_ids`: an empty tree raises with `"046D"` and `"256F"` in the message.
  - `test_full_scale`: a reader fed reports through a pipe or temp file with `full_scale=700` gives `axes == (0.5, -0.25, 0)` for counts `(350, -175, 0)`.
  - Keep `tests/examples/pi05_mars_libero/test_teleop.py` green.
- [ ] **Step 2:** Run `uv run --extra mars pytest tests/teleoperators/test_spacemouse.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Point both examples' readers at it.
- [ ] **Step 4:** Run `uv run --extra mars --extra pi --extra libero pytest tests/teleoperators/test_spacemouse.py tests/examples/pi05_mars_cursor tests/examples/pi05_mars_libero -q`. Expected: all pass.
- [ ] **Step 5:** With the lab's SpaceMouse plugged in, run `uv run --extra mars python -c "from lerobot.teleoperators.spacemouse.teleop_spacemouse import find_spacemouse; print(find_spacemouse())"`. Expected: a `/dev/hidrawN` path. If no device is attached in the session, say so in the report instead.
- [ ] **Step 6:** In `examples/pi05/mars/README.md`, document the udev rule (`KERNEL=="hidraw*", ATTRS{idVendor}=="046d|256f", MODE="0666"` or the group form) and how to check `full_scale` (a full push should read about 1.0 on its axis).
- [ ] **Step 7:** Stage; commit only on command ("teleoperators: SpaceMouse over hidraw, any 3Dconnexion device; examples use it").

### Task 7: Seed-0 grid re-run: the same numbers

**Files:**

- Modify: `examples/pi05/mars/cursor/run_sweep.py` (new `--fixed-depths PATH`)
- Create: `examples/pi05/mars/cursor/research/port_check.py`
- Test: `tests/examples/pi05_mars_cursor/test_sweep.py` (extend)

**Interfaces:**

- Produces:
  - `run_sweep.py --fixed-depths <depths.yaml>` skips tuning and tests every method at the depths in the file (the format `depths.yaml` already has). Its `tuning.json` is empty and its `depths.yaml` is a copy of the input.
  - `port_check.py --old <test.json> --new <test.json>` prints, per cell and method, the old success, the new success, the old run's 95% Wilson interval, and whether the new value lies inside it. It writes `results/research/port_check/port_check.json` and exits 1 if any cell is outside.

- [ ] **Step 1:** Write the test `test_fixed_depths_skips_tuning`. A tiny sweep (one cell, two methods, tune 4 / test 4 trials, CPU) run with `--fixed-depths` gives `test.json` whose methods ran at the given depths, and an empty `tuning.json`. Run it; expected FAIL.
- [ ] **Step 2:** Implement `--fixed-depths`. Run the test; expected PASS.
- [ ] **Step 3:** Implement `port_check.py`.
- [ ] **Step 4:** Run the 20 cells on GPU 1 in parallel, at most 6 processes:
      `CUDA_VISIBLE_DEVICES=1 uv run --extra mars python examples/pi05/mars/cursor/run_sweep.py --sweep grid --fixed-depths examples/pi05/mars/results/sim/sweeps/grid/depths.yaml --cells <cell> --out port_check_<cell>`
      Then run `--merge` into `port_check`. Expect about 1–2 h.
- [ ] **Step 5:** Run `uv run --extra mars python examples/pi05/mars/cursor/research/port_check.py --old examples/pi05/mars/results/sim/sweeps/grid/test.json --new examples/pi05/mars/results/sim/sweeps/port_check/test.json`. Expected: exit 0, with all 140 cell × method values inside their intervals. Report the largest absolute difference and how many values match exactly. If any cell is outside, stop and report; do not loosen the check.
- [ ] **Step 6:** Run `uv run --extra mars python examples/pi05/mars/figures/build_figures.py --out <scratch>`, then compare its `stats.json` with `examples/pi05/mars/figures/stats.json`. Expected:
  - Fig 2 and Fig 3 values are identical (they are read from copied results).
  - Fig 1 values are within 1e-3. Fig 1 is computed live; the tolerance covers the time-embedding rounding.
- [ ] **Step 7:** Stage the code; results stay ignored. Commit only on command ("MARS: fixed-depth sweeps and the port check; seed-0 grid reproduced").

### Task 8: Hub weights and the day-one path

**Files:**

- Create: `examples/pi05/mars/cursor/hub.py`, `examples/pi05/mars/cursor/hub_manifest.yaml`
- Modify: `examples/pi05/mars/README.md`, `examples/pi05/mars/cursor/README.md` (quick start rewritten for this repo)
- Test: `tests/examples/pi05_mars_cursor/test_hub.py`

**Interfaces:**

- Produces:
  - `hub_manifest.yaml` lists every checkpoint folder under `checkpoints/` that these configs reference: `configs/study.yaml` and its arms, `configs/sweeps/grid.yaml` (including the `retrained_mars_user` policies and user modules), and the grid's seeds 1–4 modules (`adapters/grid_seeds/`). `hub.py manifest --check` rebuilds the list from the configs and fails if the file is out of date.
  - `hub.py download [--repo ricebcilab/mars-cursor]` fetches those folders into `examples/pi05/mars/checkpoints/` with `huggingface_hub.snapshot_download`.
  - `hub.py upload [--repo ricebcilab/mars-cursor]` creates the public model repo if it is missing and uploads the manifest's converted folders. **Run it only on Nishal's command.**
- Day-one commands, which go in both READMEs:

  ```
  uv sync --extra mars
  uv run --extra mars python examples/pi05/mars/cursor/hub.py download
  uv run --extra mars python examples/pi05/mars/cursor/run_experiment.py --protocol study.yaml --participant p1
  uv run --extra mars python examples/pi05/mars/cursor/analyze_results.py --participants p1
  ```

- [ ] **Step 1:** Write the tests.
  - `test_manifest_matches_configs`: `hub.py manifest --check` exits 0.
  - `test_download_targets`: with `snapshot_download` monkeypatched, `download` asks for exactly the manifest's folders and the target directory.
    Run them; expected FAIL.
- [ ] **Step 2:** Implement `hub.py` and the manifest. Run the tests; expected PASS.
- [ ] **Step 3:** Run a fresh-clone check without the Hub:
  1. `git clone` the branch into the scratchpad, then `uv sync --extra mars`.
  2. Copy the manifest's checkpoint folders in, standing in for `download`.
  3. Run `run_experiment.py --protocol study.yaml --participant sim` (headless).
  4. Run `analyze_results.py --participants sim`.

  Expected: the session completes and the per-arm table prints, with MARS above teleop, flow control and FRS as in `examples/pi05/mars/results/sim/`.

- [ ] **Step 4:** Ask Nishal to run the SpaceMouse session (`--participant p0`) on the fresh clone, and to say when to run `hub.py upload`. After the upload, re-run Step 3 with the real `download`.
- [ ] **Step 5:** Run the whole suite: `uv run --extra mars --extra pi --extra libero pytest tests/examples/pi05_mars_cursor tests/examples/pi05_mars_libero tests/policies/pi0_pi05 tests/policies/cursor_flow tests/teleoperators/test_spacemouse.py -q` and `uv run ruff check src tests`. Expected: all pass and ruff is clean.
- [ ] **Step 6:** Stage; commit only on command ("MARS: Hub weights manifest and download, day-one quick start").
