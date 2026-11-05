# MARS image task module, round 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the image task module cursor-relative token geometry, add a teacher-rollout demonstration source to the adapter trainer, and run the two-arm quick check on T2 and T4 under the user's new gate rule.

**Architecture:** `LearnedTaskModule` gains `token_geometry="relative"`: each VLM patch (or CNN cell) token is the projected content plus the disc-token MLP applied to the patch centre's offset from the cursor, with no learned absolute position. `train_adapter.py` gains `--token-geometry` and `--teacher`, the latter replacing the scripted demonstrations with the parametric teacher's `oracle_task` rollouts (hits only, resampled to the policy's step rate). Two gate sweeps and a three-verdict gate report (pass / marginal / FAIL against the parametric module and the fine-tuned policy) complete the check.

**Tech Stack:** Python 3.12, PyTorch, numpy, PyYAML, pytest; `uv run --extra mars` (`--extra pi` for the VLM), `HF_HUB_OFFLINE=1`.

**Spec:** `docs/superpowers/specs/2026-11-04-mars-image-round2-teacher-relative-design.md` (amends `docs/superpowers/specs/2026-11-03-mars-image-task-module-design.md`).

## Global Constraints

- Python commands run through `uv run --extra mars` (add `--extra pi --extra test` where the VLM or pytest is needed); real-VLM work needs `HF_HUB_OFFLINE=1` and a `CUDA_VISIBLE_DEVICES`.
- The `discs` branch of `LearnedTaskModule` stays bitwise unchanged: its config, state dict and outputs. `tests/policies/steering/test_steering_equivalence*.py` and the existing `test_scene_tokens.py` tests must stay green.
- A round-1 image checkpoint (config without `token_geometry`) loads as `absolute` and gives the same tokens as before.
- Results are never deleted; superseded results move to an `old/` folder. Round-1 checkpoints under `checkpoints/adapters/image/` stay.
- Checkpoint names: `image/task_<ctx>_n300_s0_with_<user>_vlm_image_rel`, `image/task_<ctx>_n10000_s0_with_<user>_vlm_image_teacher_rel` (the `_n<N>_s0` tag is what `--module-seed` substitutes).
- The gate rule (spec): per context, *pass* = hits ≥ parametric hits − 10 points and collisions ≤ parametric collisions + 10 points; *floor* = hits > fine-tuned policy's hits (`oracle_ft`); above the floor, within the collision rule, but not a pass = *marginal*; otherwise *FAIL*. Goal-hidden flag unchanged (`policy_task_<x>` hits > `policy_task` hits + 10 points).
- No push, no merge, without the user's go. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A relative module at test time: `run_sweep` binds the module to the sweep's observation, which carries `pos`; the bound call must equal the unbound call for `relative` (Task 1 test).
2. A round-1 `vlm_image` checkpoint keeps its `scene_pos` and absolute behaviour after this change (Task 1 test: a config without the key constructs an absolute module whose state dict has `scene_pos`).
3. A teacher rollout whose selection step is odd ends on half a planned step; `planned_steps` must drop it rather than invent a step (Task 3 test on synthetic paths of odd and even length).
4. A gate run without an `oracle_ft` row (the round-1 run) must still render: no marginal verdict is possible, `ft hits` shows `n/a` (Task 5 test).
5. `--teacher` naming an image module would fail deep inside the rollout for want of scene arrays; refuse it up front with a message (Task 3 test).

---

### Task 1: Relative token geometry in `LearnedTaskModule`

**Files:**
- Modify: `src/lerobot/policies/steering/configuration_steering.py` (constants near `SCENE_SOURCES`)
- Modify: `src/lerobot/policies/steering/modeling_steering.py` (`LearnedTaskModule.__init__`, `tokens`, class docstring)
- Modify: `src/lerobot/policies/steering/__init__.py` (export `TOKEN_GEOMETRIES`)
- Test: `tests/policies/steering/test_scene_tokens.py`

**Interfaces:**
- Consumes: `LearnedTaskModule(horizon, d, tdim, heads, layers, scale, n_dims, token_source, scene_dim, scene_tokens, image_size)`; `tokens(obs, b) -> (tokens, pad)`; `self.token_in` (the 4-feature MLP); `self.scene_in`; `self.cnn`; `CNN_PATCH`.
- Produces: `TOKEN_GEOMETRIES: tuple[str, ...] = ("absolute", "relative")` in `configuration_steering.py`; `LearnedTaskModule(..., token_geometry: str = "absolute")`; attribute `token_geometry`; non-persistent buffer `token_centres` `(scene_tokens, 2)` and float attribute `token_half` on a relative module; config key `token_geometry` present only when `"relative"`.

- [ ] **Step 1: Write the failing tests**

Change the helper and add tests in `tests/policies/steering/test_scene_tokens.py`:

```python
from lerobot.policies.steering import TOKEN_GEOMETRIES

def _module(src, geometry="absolute"):
    return LearnedTaskModule(
        horizon=16, token_source=src, token_geometry=geometry,
        scene_dim=32 if src.startswith("vlm") else None,
        scene_tokens=16 if src.startswith("vlm") else None,
        image_size=64 if src == "cnn" else None,
    )

def _centres(g):  # token k = (row, col) = divmod(k, g); x right (col), y down (row)
    rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([(cols.flatten() + 0.5) / g, (rows.flatten() + 0.5) / g], -1)


@pytest.mark.parametrize("src", ["vlm_image", "cnn"])
def test_relative_geometry_round_trips_and_absolute_stays_the_default_without_a_key(src, tmp_path):
    assert TOKEN_GEOMETRIES == ("absolute", "relative")
    rel, ab = _module(src, "relative"), _module(src)
    assert rel.config["token_geometry"] == "relative" and "token_geometry" not in ab.config
    assert "scene_pos" not in rel.state_dict() and "scene_pos" in ab.state_dict()
    assert LearnedTaskModule(**ab.config).token_geometry == "absolute"      # a round-1 config
    obs, (x, v, t) = _scene(), _xvt()
    save_adapter_module(rel, tmp_path / "rel.pt", "task")
    back, kind, _ = load_adapter_module(tmp_path / "rel.pt")
    assert kind == "task" and back.token_geometry == "relative"
    a, b = rel(x, t, v, obs), back(x, t, v, obs)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


@pytest.mark.parametrize("src", ["vlm_image", "cnn"])
def test_relative_tokens_are_the_projection_plus_the_cursor_relative_geometry(src):
    m, obs = _module(src, "relative"), _scene()
    x = obs["scene"][src]
    rel = (_centres(4)[None] - obs["pos"][:, None]) / m.scale
    feats = torch.cat([rel, torch.full_like(rel[..., :1], (0.5 / 4) / m.scale), rel.norm(dim=-1, keepdim=True)], -1)
    content = m.scene_in(x.float()) if src == "vlm_image" else m.cnn(x.float() / 255).flatten(2).transpose(1, 2)
    tokens, pad = m.tokens(obs, 3)
    assert pad is None and tokens.shape == (3, 17, 128)
    assert torch.allclose(tokens[:, 1:], content + m.token_in(feats), atol=1e-6)
    assert torch.equal(m.token_centres, _centres(4)) and m.token_half == 0.5 / 4


def test_relative_geometry_is_refused_where_tokens_have_no_screen_position():
    for src in ("discs", "vlm_prefix"):
        with pytest.raises(ValueError, match="relative"):
            _module(src, "relative")
    with pytest.raises(ValueError, match="square"):
        LearnedTaskModule(token_source="vlm_image", token_geometry="relative", scene_dim=32, scene_tokens=15)
    with pytest.raises(ValueError, match="token_geometry"):
        _module("vlm_image", "polar")
    obs = _scene()
    del obs["pos"]
    with pytest.raises(ValueError, match="pos"):
        _module("vlm_image", "relative").tokens(obs, 3)


def test_relative_tokens_shift_with_the_cursor_by_whole_tokens():
    # vlm_image only: its content can be made identical per token; the CNN's zero-padded borders cannot. The centres
    # buffer is shared, so this pins the row/col convention for both sources.
    src = "vlm_image"
    m, obs = _module(src, "relative"), _scene()
    x = obs["scene"][src]
    obs["scene"][src] = x[:, :1].expand_as(x).clone()                                   # every token the same
    p = torch.tensor([[0.3, 0.4]]).expand(3, -1)
    tp = m.tokens({**obs, "pos": p}, 3)[0][:, 1:].view(3, 4, 4, -1)                 # [b, row, col, d]
    tx = m.tokens({**obs, "pos": p + torch.tensor([0.25, 0.0])}, 3)[0][:, 1:].view(3, 4, 4, -1)
    ty = m.tokens({**obs, "pos": p + torch.tensor([0.0, 0.25])}, 3)[0][:, 1:].view(3, 4, 4, -1)
    assert torch.allclose(tx[:, :, 1:], tp[:, :, :-1], atol=1e-5)   # a cursor one column right sees the same offsets one column right
    assert torch.allclose(ty[:, 1:, :], tp[:, :-1, :], atol=1e-5)   # one row down likewise
```

Change `test_a_bound_call_equals_the_unbound_call` to parametrize over `("discs", "absolute"), ("vlm_image", "absolute"), ("vlm_prefix", "absolute"), ("cnn", "absolute"), ("vlm_image", "relative"), ("cnn", "relative")` and build with `_module(src, geometry)`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/policies/steering/test_scene_tokens.py -q`
Expected: the new tests fail with `TypeError: ... unexpected keyword argument 'token_geometry'` / `ImportError: TOKEN_GEOMETRIES`.

- [ ] **Step 3: Implement**

In `configuration_steering.py`, next to `SCENE_SOURCES`: `TOKEN_GEOMETRIES: tuple[str, ...] = ("absolute", "relative")` with a one-line comment (absolute: a learned position per token, the round-1 module; relative: the token centre's offset from the cursor through the disc-token MLP). Export it from `__init__.py` next to `SCENE_SOURCES`.

In `LearnedTaskModule.__init__(..., token_geometry: str = "absolute")`:
- `token_geometry not in TOKEN_GEOMETRIES` → `ValueError` naming `token_geometry`; `relative` with `token_source` not in `("vlm_image", "cnn")` → `ValueError` mentioning "relative".
- `self.config["token_geometry"] = token_geometry` only when it is `"relative"` (so `discs` and round-1 configs are unchanged).
- Relative: `g = isqrt(scene_tokens)`; `g * g != scene_tokens` → `ValueError` mentioning "square". Build the centres as `_centres(g)` above (row-major, x = col, y = row), `register_buffer("token_centres", centres, persistent=False)`, `self.token_half = 0.5 / g`. Create `scene_pos` only for absolute image modules.
- Everything else as today.

In `tokens()`, image branch: after `scene = ...`, if `self.token_geometry == "relative"`: `pos = obs.get("pos")`; `None` → `ValueError` mentioning `pos`; `rel = (self.token_centres[None] - pos[:, None]) / self.scale`; `feats = cat([rel, full(half / scale), rel.norm(dim=-1, keepdim=True)], -1)`; `scene = scene + self.token_in(feats)`; else `scene = scene + self.scene_pos`. Return `cat([null, scene], 1), None`.

Update the class docstring (one sentence on `token_geometry`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra mars --extra test pytest tests/policies/steering/ -q`
Expected: all pass, including the equivalence tests.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/steering/configuration_steering.py src/lerobot/policies/steering/modeling_steering.py src/lerobot/policies/steering/__init__.py tests/policies/steering/test_scene_tokens.py
git commit -m "MARS: the image task module can read each token's offset from the cursor instead of a learned position"
```

---

### Task 2: `--token-geometry` in the adapter trainer

**Files:**
- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (`train_task`, `save`, `parse_args`, module docstring's TASK MODULE paragraph)
- Test: `tests/examples/pi05_mars_cursor/test_training.py`

**Interfaces:**
- Consumes: `LearnedTaskModule(..., token_geometry=)` (Task 1), `TOKEN_GEOMETRIES`.
- Produces: `--token-geometry {absolute,relative}` (default `absolute`); `args.token_geometry`; the checkpoint's `meta["token_geometry"]`.

- [ ] **Step 1: Write the failing test**

```python
def test_train_a_relative_geometry_image_task_module(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_rel",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "cnn",
                                                  "--render-size", "64", "--token-geometry", "relative", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t_rel" / "adapter.pt")
    assert ckpt["config"]["token_geometry"] == "relative" and ckpt["meta"]["token_geometry"] == "relative"
    assert "scene_pos" not in ckpt["state_dict"]
    assert ReversalAdapter.load(ckpt_dir / "t_rel" / "adapter.pt").task.token_geometry == "relative"
    with pytest.raises(SystemExit):
        train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "x", "--user-module", "u",
                                  "--corruption", "{rotation_deg: 60}", "--token-geometry", "relative"])   # discs have no screen tokens
```

Also assert in the existing `test_train_an_image_task_module_from_the_render` that `ckpt["meta"]["token_geometry"] == "absolute"` and `"token_geometry" not in ckpt["config"]`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q -k "relative_geometry or from_the_render"`
Expected: FAIL (`unrecognized arguments: --token-geometry`).

- [ ] **Step 3: Implement**

`parse_args`: `p.add_argument("--token-geometry", choices=TOKEN_GEOMETRIES, default="absolute", help="task: relative puts each scene token's offset from the cursor into the token (vlm_image, cnn) instead of a learned position")`; `p.error` when `relative` and `token_source in ("discs", "vlm_prefix")`. `train_task`: `TaskModule(policy.horizon, d=args.d, layers=args.layers, token_geometry=args.token_geometry, **scene_spec(...))`. `save`: `meta` gains `token_geometry=args.token_geometry`. Import `TOKEN_GEOMETRIES` from `lerobot.policies.steering`. One sentence in the docstring's TASK MODULE paragraph.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/train/train_adapter.py tests/examples/pi05_mars_cursor/test_training.py
git commit -m "MARS: the adapter trainer takes --token-geometry"
```

---

### Task 3: Teacher demonstrations (`--teacher`)

**Files:**
- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (new `planned_steps`, `teacher_demos`; `expert_splits`; `save`; `parse_args`; module docstring, a TEACHER paragraph)
- Test: `tests/examples/pi05_mars_cursor/test_training.py`

**Interfaces:**
- Consumes: `run_trials(arm, layouts, agent, readout, policy, adapters, device)`, `readout_for(arm, rng, off=True)`, `config.load_arm(name, over)`, `arm_task(arm)`, `ExpertChunkAgent` (`cursor_sa.agents`), `ReversalAdapter.load(adapter_path(name), device)`, `layouts_for(args, obstacle, rng, n, p_free) -> (layouts, task)`, `Layouts.concat(parts)`, `ExpertData(layouts, actions, lengths, policy, device, corruption, curl, frozen_noise)`.
- Produces:
  - `planned_steps(path: np.ndarray) -> np.ndarray`: `np.diff(np.asarray(path)[::2], axis=0)`, the planned steps a `gain_mode: time, speed_gain: 0.5` rollout executed (two control steps per planned step; an odd final control step is half a planned step inside the hold and is dropped).
  - `teacher_demos(args, obstacle, policy, device, rng, n, p_free=0.0) -> (layouts, task, actions, lengths, info)`: `n` hits of the teacher's `oracle_task` rollouts as demonstrations; `actions [n, L, 2]` zero after each demo's length, `lengths [n]` = number of planned steps to selection, `info = dict(drawn=int, kept=int, outcomes=dict[str, int], records=list)`.
  - `--teacher NAME` (task only), `--teacher-chunk INT` (layouts per rollout batch, default 500); `meta["teacher"]`, `meta["teacher_demos"] = {"train": {drawn, kept, outcomes}, "val": {...}}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_planned_steps_resample_a_half_speed_path_and_drop_an_odd_final_control_step():
    path = np.cumsum(np.repeat([[0.01, 0.0]], 6, axis=0), axis=0)          # 6 positions: 5 control steps
    assert train_adapter.planned_steps(path).shape == (2, 2)                # control steps 0-1 and 2-3; the 5th is half a step
    assert np.allclose(train_adapter.planned_steps(path), [[0.02, 0.0], [0.02, 0.0]])
    seven = np.vstack([path, path[-1:] + [0.01, 0.0]])                      # 7 positions: 6 control steps, 3 planned
    assert train_adapter.planned_steps(seven).shape == (3, 2)


def _tiny_teacher(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.03, along: 0.5}", "--name", "t",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", *COMMON]))


def test_teacher_demonstrations_are_the_teachers_hits_at_the_policys_step_rate(ckpt_dir):
    _tiny_teacher(ckpt_dir)
    args = train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.03, along: 0.5}", "--name", "s", "--user-module", "u",
                                     "--corruption", "{rotation_deg: 60}", "--teacher", "t", "--teacher-chunk", "8", *COMMON])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = FlowPolicy.load(train_adapter.policy_path(train_adapter.config.load_arm("base.yaml")["model"]), device)
    layouts, task, actions, lengths, info = train_adapter.teacher_demos(args, args.obstacle, policy, device, np.random.default_rng(0), 4, args.p_free)
    assert len(layouts) == 4 and info["kept"] == 4 and info["drawn"] >= 4 and info["drawn"] % 8 == 0
    assert all(r["outcome"] == "hit" for r in info["records"]) and sum(info["outcomes"].values()) == info["drawn"]
    for i, r in enumerate(info["records"]):
        path = np.asarray(r["path"])
        assert lengths[i] == (len(path) - 1) // 2 and np.allclose(actions[i, :lengths[i]], train_adapter.planned_steps(path))
        assert np.allclose(np.abs(np.diff(path[::2], axis=0)).sum(), np.abs(np.diff(path, axis=0)).sum(), atol=2e-3)  # same travel, half the steps
    assert np.all(actions[np.arange(4)[:, None], np.arange(actions.shape[1])[None] >= lengths[:, None]] == 0)
    assert replay(actions, layouts, replace(task, cursor_noise=0.0)).success.all()        # a valid demonstration at the policy's rate


def test_train_a_task_module_on_teacher_demonstrations(ckpt_dir):
    _tiny_teacher(ckpt_dir)
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.03, along: 0.5}", "--name", "s",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", *COMMON,
                                                  "--teacher", "t", "--teacher-chunk", "8", "--n-demos", "4", "--val-demos", "2"]))
    ckpt = torch.load(ckpt_dir / "s" / "adapter.pt")
    assert ckpt["meta"]["teacher"] == "t" and ckpt["meta"]["teacher_demos"]["train"]["kept"] == 4 and ckpt["meta"]["teacher_demos"]["val"]["kept"] == 2
    assert {"val_fm", "collide_composed"} <= set(json.loads((ckpt_dir / "s" / "train_log.json").read_text())["log"][-1])
    base = ["--kind", "task", "--obstacle", "{kind: disc, radius: 0.03, along: 0.5}", "--name", "x", "--user-module", "u", "--corruption", "{rotation_deg: 60}"]
    for bad in ([*base, "--teacher", "t", "--demo-seconds", "10"],                                  # a time budget is the scripted expert's
                ["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "x", "--teacher", "t"],   # user modules calibrate on the open field
                ["--kind", "task", "--pooled", "p.yaml", "--name", "x", "--teacher", "t"]):           # no pooled teacher run
        with pytest.raises(SystemExit):
            train_adapter.parse_args(bad)


def test_an_image_module_is_refused_as_a_teacher(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_cnn",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "cnn", "--render-size", "64", *COMMON]))
    with pytest.raises(ValueError, match="discs"):
        train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "s",
                                                      "--user-module", "u", "--corruption", "{rotation_deg: 60}", *COMMON, "--teacher", "t_cnn", "--n-demos", "2", "--val-demos", "2"]))
```

Imports the test file needs: `from dataclasses import replace`, `from cursor_sa.demos import replay`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q -k "teacher or planned_steps"`
Expected: FAIL (`AttributeError: planned_steps`, `unrecognized arguments: --teacher`).

- [ ] **Step 3: Implement**

`planned_steps(path)` as specified.

`teacher_demos(args, obstacle, policy, device, rng, n, p_free=0.0)`:
- `adapter = ReversalAdapter.load(adapter_path(args.teacher), device)`; if `adapter.task is None or adapter.task.token_source != "discs"` → `ValueError("--teacher must be a parametric (discs) task module: ...")`.
- The arm: `config.load_arm("condition_grid.yaml", dict(model=args.model or config.load_arm("base.yaml")["model"], mode="teleop", reversal_adapter={"user": None, "task": args.teacher}, obstacle=args.obstacle, corruption=None, targets=list(args.targets), reach=list(args.reach)))` — the gate's `oracle_task` arm: check `arm["mode"] == "teleop"`, `arm["gain_mode"] == "time"`, `arm["speed_gain"] == 0.5`, `arm["execute"] == 2` with an assertion and a message saying the resampling below assumes two control steps per planned step (`planned_steps` is only right under that pacing). Follow `closed_loop_splits` for the `load_arm` call shape; pass `obstacle` as the spec `args.obstacle` carries (what `method_arm` passes is the raw cell spec; `load_arm` normalises).
- Loop: `layouts_for(args, obstacle, rng, args.teacher_chunk, p_free)` → `run_trials(arm, layouts, ExpertChunkAgent(), readout_for(arm, rng, off=True), policy, {args.teacher: adapter}, device)`; keep rows with `outcome == "hit"` (layouts by index, records) until `n` are kept; truncate to `n`; count every outcome in `info["outcomes"]`, `info["drawn"]` the layouts rolled out.
- `layouts = Layouts.concat(kept parts)` (or the single part); `actions` zero-padded `[n, max length, 2]` from `planned_steps(record["path"])`, `lengths[i] = len(steps)`; `task` from `layouts_for`.
- Print one line: kept / drawn, the outcome counts, seconds.

`expert_splits`: when `args.teacher`, both splits use `teacher_demos(args, obstacle, policy, device, rng, n, p_free)` instead of `layouts_for` + `make_demos`; keep the scene attach and `ExpertData` as they are; collect `info` minus `records` per split into `args.teacher_demos = {"train": ..., "val": ...}` (so `save`'s `vars(args)` and a `meta["teacher_demos"]` entry carry them). `save`: `meta` gains `teacher=args.teacher` and `teacher_demos=getattr(args, "teacher_demos", None)`.

`parse_args`: `--teacher` (help: "task: demonstrations are this parametric task module's oracle_task rollouts (hits only, resampled to the policy's step rate) instead of the scripted expert's"), `--teacher-chunk` (int, default 500). Errors: `--teacher` with `--kind user`, with `--pooled`, with `--demo-seconds`.

Docstring: a TEACHER paragraph (five lines: what the rollouts are, hits only, the resampling, why).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/train/train_adapter.py tests/examples/pi05_mars_cursor/test_training.py
git commit -m "MARS: the adapter trainer can take a parametric teacher's rollouts as demonstrations"
```

---

### Task 4: Round-2 gate sweeps with a fine-tuned reference row

**Files:**
- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image2_quick.yaml`
- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image2.yaml`
- Test: `tests/examples/pi05_mars_cursor/test_sweep.py` (`test_image_sweep_configs_resolve_every_arm_per_cell`)

**Interfaces:**
- Consumes: `load_sweep`, `method_arm` (a method's `model:` given per task resolves through `per_cell`; `reversal_adapter: {task: {T2: ..}}` through `cell_adapter`), `grid_task_check_image.yaml` as the template.
- Produces: the two sweep names used by Task 6's pipeline and the report (`--check grid_task_check_image2_quick`).

- [ ] **Step 1: Write the failing test**

Append to `test_image_sweep_configs_resolve_every_arm_per_cell`:

```python
    quick = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2_quick.yaml")
    assert set(quick["grid"]["tasks"]) == {"T2", "T4"} and quick["focus"] == "oracle_task_rel" and quick["against"] == ["oracle_task", "oracle_ft"]
    assert set(quick["methods"]) == {"oracle", "oracle_task", "oracle_ft", "oracle_task_rel", "oracle_task_teach_rel", "policy", "policy_task", "policy_task_rel", "policy_task_teach_rel"}
    assert method_arm(quick, "oracle_ft", "U0xT4")["model"] == "flow_goaldrop_ring_goal_ft_n300_s0" and method_arm(quick, "oracle_ft", "U0xT4")["learned_adapter"] is None
    assert method_arm(quick, "oracle_task_rel", "U0xT2")["learned_adapter"]["task"] == "image/task_disc_large_n300_s0_with_gain025y_vlm_image_rel"
    assert method_arm(quick, "oracle_task_teach_rel", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n10000_s0_with_rot60_vlm_image_teacher_rel"
    assert quick["seed"] == check["seed"] and quick["trials"] == check["trials"] and quick["arm"] == check["arm"] and quick["agent"] == check["agent"]
    full = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2.yaml")
    assert set(full["grid"]["tasks"]) == {"T2", "T3", "T4", "T5"} and set(full["methods"]) == {"oracle", "oracle_task", "oracle_ft", "oracle_task_rel", "oracle_task_rel_all", "policy", "policy_task", "policy_task_rel", "policy_task_rel_all"}
    assert method_arm(full, "oracle_ft", "U0xT3")["model"] == "flow_goaldrop_wall_gap_ft_n300_s0"
    assert method_arm(full, "oracle_task_rel", "U0xT5")["learned_adapter"]["task"] == "image/task_ring_start_n300_s0_with_curl20_vlm_image_rel"
    assert method_arm(full, "oracle_task_rel_all", "U0xT3")["learned_adapter"]["task"] == "image/task_all_n1200_s0_vlm_image_rel"
```

If `learned_adapter` is not the key `method_arm` gives an arm without a module, use whatever the `oracle` arm of `grid_task_check_image.yaml` has, and assert equality with that.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q -k image_sweep_configs`
Expected: FAIL (file not found).

- [ ] **Step 3: Write the two YAMLs**

Copy `grid_task_check_image.yaml`'s structure (same `agent`, `arm`, `tune`, `grid.users`, `noises`, `trials`, `seed`). Header comment: what the round-2 gate tests, the three verdicts, and the two commands (`--precompute-scene`, then the sweep). `oracle_ft: {mode: teleop, readout: off, model: {T2: flow_goaldrop_disc_large_ft_n300_s0, T3: flow_goaldrop_wall_gap_ft_n300_s0, T4: flow_goaldrop_ring_goal_ft_n300_s0, T5: flow_goaldrop_ring_start_ft_n300_s0}}` (the quick file lists T2 and T4 only). `oracle_task_rel` / `policy_task_rel` map each task to `image/task_<ctx>_n300_s0_with_<user>_vlm_image_rel` (disc_large/gain025y, wall_gap/gain025x, ring_goal/rot60, ring_start/curl20); `oracle_task_teach_rel` / `policy_task_teach_rel` (quick only) to `image/task_<ctx>_n10000_s0_with_<user>_vlm_image_teacher_rel`; `oracle_task_rel_all` / `policy_task_rel_all` (full only) to `image/task_all_n1200_s0_vlm_image_rel`. `focus: oracle_task_rel`, `against: [oracle_task, oracle_ft]`. The quick file's `grid.tasks` holds T2 and T4 only.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image2_quick.yaml examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image2.yaml tests/examples/pi05_mars_cursor/test_sweep.py
git commit -m "MARS: round-2 gate sweeps with a fine-tuned reference row"
```

---

### Task 5: Three-verdict gate report

**Files:**
- Modify: `examples/pi05/mars/cursor/report_grid_image.py` (`gate_table`, the gate table rendering in `main`, module docstring, drop `HIT_MIN`)
- Test: `tests/examples/pi05_mars_cursor/test_report_grid_image.py`

**Interfaces:**
- Consumes: the check run's `test[cell]["metrics"][method]` with `success` and `collision` fractions.
- Produces: `gate_table(check, image_arms) -> list[dict]` rows with keys `context, arm, hits, ref_hits, ft_hits, collision, ref_collision, verdict, pass, hidden_hits, ref_hidden_hits, flag`; `verdict in ("pass", "marginal", "FAIL")`; `pass == (verdict == "pass")`. Markdown gate header: `context, arm, hits, ref hits, ft hits, collision, ref collision, verdict, hidden hits, ref hidden, flag`.

- [ ] **Step 1: Write the failing test**

Replace `test_gate_rule_and_flag` with:

```python
def test_gate_verdicts_pass_marginal_fail_and_flag():
    def cell(ref, arm_hits, arm_col=0.10, ft=0.5, hidden=0.25):
        m = {"oracle": dict(success=0.9, collision=0.5), "oracle_task": dict(success=ref, collision=0.05),
             "oracle_task_rel": dict(success=arm_hits, collision=arm_col), "policy_task": dict(success=0.2, collision=0.1),
             "policy_task_rel": dict(success=hidden, collision=0.1)}
        if ft is not None:
            m["oracle_ft"] = dict(success=ft, collision=0.2)
        return dict(metrics=m)
    cells = {"U0xT2|0.0": cell(0.95, 0.86),                 # within 10 points of parametric: pass
             "U0xT3|0.0": cell(0.95, 0.80),                 # above the fine-tuned floor, more than 10 under parametric: marginal
             "U0xT4|0.0": cell(0.95, 0.50, hidden=0.5),     # at the floor: FAIL, and goal-hidden flag
             "U0xT5|0.0": cell(0.742, 0.65),                # T5: parametric 74.2, so 65 passes
             "U0xT6|0.0": cell(0.95, 0.90, arm_col=0.30),   # hits fine, collisions 25 points over: FAIL
             "U0xT7|0.0": cell(0.95, 0.80, ft=None)}        # no fine-tuned row: no marginal possible
    rows = {r["context"]: r for r in gate_table(dict(test=cells), ["rel"])}
    assert [rows[f"T{k}"]["verdict"] for k in (2, 3, 4, 5, 6, 7)] == ["pass", "marginal", "FAIL", "pass", "FAIL", "FAIL"]
    assert rows["T2"]["pass"] and not rows["T3"]["pass"] and rows["T2"]["ref_hits"] == 0.95 and rows["T2"]["ft_hits"] == 0.5
    assert rows["T4"]["flag"] and not rows["T2"]["flag"] and rows["T7"]["ft_hits"] is None
```

In the `main` rendering test(s), update the expected gate header and verdict strings (`pass`/`marginal`/`FAIL`) and assert a row with no `oracle_ft` renders `n/a` under `ft hits`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py -q`
Expected: FAIL (`KeyError: 'verdict'`).

- [ ] **Step 3: Implement**

In `gate_table`: `ref_hits = met.get("oracle_task", {}).get("success")`, `ft_hits = met.get("oracle_ft", {}).get("success")`. `collisions_ok = ref_c is not None and col <= ref_c + MARGIN`; `passes = ref_hits is not None and hits >= ref_hits - MARGIN and collisions_ok`; `above_floor = ft_hits is not None and hits > ft_hits`; `verdict = "pass" if passes else "marginal" if (above_floor and collisions_ok) else "FAIL"`; `pass = verdict == "pass"`. Drop `HIT_MIN`; rewrite the docstring to the rule above (fractions). In `main`, the gate table header and row as in Interfaces, `_fmt` for `n/a` on `None`. Module docstring: the three verdicts in one sentence.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/report_grid_image.py tests/examples/pi05_mars_cursor/test_report_grid_image.py
git commit -m "MARS: the gate report gives pass, marginal or FAIL against the parametric module and the fine-tuned policy"
```

---

### Task 6: Round-2 jobs and pipeline stages (controller; the run folder is git-ignored)

**Files:**
- Create: `examples/pi05/mars/results/research/image/make_jobs2.py` → `jobs_r2_quick.txt` (four lines, name TAB command)
- Modify: `examples/pi05/mars/results/research/image/pipeline.sh` (stages `train2`, `gate2`)

- [ ] **Step 1: The four jobs** (`T2` = disc_large / gain025y / `{scale: [1.0, 0.25]}`; `T4` = ring_goal / rot60 / `{rotation_deg: 60}`; obstacle specs as `make_jobs.py`'s `TASKS`):

```
task_<ctx>_n300_s0_with_<user>_vlm_image_rel            HF_HUB_OFFLINE=1 uv run --extra mars --extra pi python examples/pi05/mars/cursor/train/train_adapter.py --kind task --obstacle '<spec>' --user-module user_<user>_expert_n100_s0 --corruption '<corr>' --name image/task_<ctx>_n300_s0_with_<user>_vlm_image_rel --token-source vlm_image --token-geometry relative --n-demos 300 --frozen-noise --seed 0
task_<ctx>_n10000_s0_with_<user>_vlm_image_teacher_rel  ... --name image/task_<ctx>_n10000_s0_with_<user>_vlm_image_teacher_rel --token-source vlm_image --token-geometry relative --teacher task_<ctx>_n300_s0_with_<user> --n-demos 10000 --val-demos 500 --frozen-noise --seed 0
```

Order: the two teacher jobs first (heavier), then the two `rel300` jobs, so `run_jobs.py --per-gpu 2 --gpus 0 1` puts one teacher job on each GPU.

- [ ] **Step 2: Pipeline stages.** `train2`: `run_jobs.py jobs_r2_quick.txt --per-gpu 2 --gpus 0 1` → `runner_r2.log`, `DONE_train_r2`. `gate2`: precompute `grid_task_check_image2_quick` on GPU 1 (`--extra pi`), the sweep, then `report_grid_image.py --check grid_task_check_image2_quick --results ../results --out ../results/sim/reports/grid_image2_quick` → `DONE_gate2`. Logs under `logs/`.

- [ ] **Step 3: Dry-run** `make_jobs2.py` and `bash -n pipeline.sh`; check each job command parses: `uv run --extra mars python examples/pi05/mars/cursor/train/train_adapter.py --help` and a `parse_args` call on each job's argument list (from the repo root, in a Python one-liner).

---

### Task 7: Run the quick check and report (controller)

- [ ] **Step 1:** `nohup bash pipeline.sh train2 ...` detached; monitor `runner_r2.log` and `logs/`. Expected: four checkpoints under `checkpoints/adapters/image/`, each with `train_log.json`; the teacher runs' `meta["teacher_demos"]` show kept = 10,000 / 500.
- [ ] **Step 2:** `bash pipeline.sh gate2`. Expected: `results/sim/sweeps/grid_task_check_image2_quick/test.json`, `results/sim/reports/grid_image2_quick/report.md`.
- [ ] **Step 3:** Post the gate table (hits, collisions, verdicts for `rel`, `teach_rel`; `oracle_task` and `oracle_ft` reference rows; the flag column) to the user and **stop**: stage 2 (T3/T5, pooled, grid) only on the user's go and only if `rel300` passed both contexts.
