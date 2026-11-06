# MARS round 4: augmenting 300 demonstrations under the relative recipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two augmentations of the relative image task module's 300 demonstrations (state jitter with recovery labels; rigid motion of the scene), a diagnostic of why `rel300` fails, and the sweep and report pieces that gate the three new arms.

**Architecture:** Rigid motion is a vectorised primitive on `Layouts` and action sequences in the cursor env, applied once by the trainer to the train split before rendering (copies are real demonstrations: rendered, encoded, pushed like any other). Jitter is an online perturbation in `DemoSampler.sample`, train split only. A diagnostic script reuses the sweep's loaders and trial runner on three layout sets. The report's recipe selection gains an arm prefix.

**Tech Stack:** Python 3.12, numpy, PyTorch, the MARS cursor example (`examples/pi05/mars/cursor`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-mars-image-relative-augmentation-design.md`

## Global Constraints

- No change to what any existing command computes: every new flag defaults to off; a run without the new flags draws the same random numbers and saves the same weights as at `f714e5b7`. New draws come from torch's global generator only when `--jitter > 0`, and rigid motions from `np.random.default_rng([seed, 11])` only when `--rigid-motions > 0`.
- `src/lerobot/envs/cursor.py` is mypy-strict (`lerobot.envs`): full annotations on everything added.
- `--jitter` and `--rigid-motions` are refused (`p.error`) with `--token-source vlm_views`; `--rigid-motions` is also refused with a curl corruption and with `--teacher`.
- Rotation is about the screen centre `(0.5, 0.5)`; mirror is `x -> 1 - x` applied before the rotation; shifts are uniform in `[-0.25, 0.25]^2`; disc radii never change; `Layouts.scene` is dropped by `move_layouts`.
- A rigid copy is accepted only when `layouts_on_screen` holds, every path point lies in `[0, 1]^2`, and `replay` of its actions succeeds; at most `tries * k` draws per demonstration; demonstrations of length 0 get no copies.
- Jitter: `delta ~ N(0, sigma^2 I)` per sample in screen units; `obs["pos"] += delta`; the first `m = min(jitter_steps, horizon)` chunk steps each lose `delta / m`; nothing else in the sample changes; never on the validation split.
- Defaults: `--jitter 0.0`, `--jitter-steps 4`, `--rigid-motions 0`; the runs use `--jitter 0.02` and `--rigid-motions 32`.
- A copy's field-term pushes are its original's pushes moved by the copy's motion (`move_actions`), never recomputed by the expert on the moved layout; `ExpertData` takes them as `push=`. With `--frozen-noise` a copy draws its own noise track.
- Checkpoint `meta` gains `jitter`, `jitter_steps`, `rigid_motions`, `rigid_kept` (copies kept, summed over the train split; `None` for user modules).
- Decision rule (controller): the adopted recipe is the measured arm with the highest mean hits among those beating `rel300`'s mean by more than `ADOPT` (3 points); none, the recipe stays `rel300`; a tie goes to `jit`, then `rigid`, then `both`.
- Tests run on the CPU only: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest ...`. Pre-commit on every touched file: `uv run --extra dev pre-commit run --files <files>`. Nothing under `examples/pi05/mars/results/` or `checkpoints/` is touched.
- Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `git add` names exactly the task's files.

## Review Focus

- A demonstration none of whose rigid copies fits the screen (a reach spanning the screen): fewer or zero copies, a printed count, no error, and the original rows unchanged bit for bit. Test in Task 2.
- A jittered sample at a hold step (`k` at or past the demonstration's end, label all zeros) and a chunk shorter than `m`: the correction still returns the cursor to the goal over the steps there are. Test in Task 2.
- A free layout (no obstacle) under rigid motion: replay succeeds and the copy is kept. Test in Task 2.
- A context whose user corruption is anisotropic (`gain025y`, `rot60`): the field term's pushes are recomputed on the moved layouts and passed through the unchanged corruption, so nothing in `readout_core` needs a change; a curl corruption is refused. Test in Task 3.
- The diagnostic's rebuilt training layouts must equal the trainer's: any draw from the seed's generator before `layouts_for` breaks it. Test in Task 4 against a recorded tiny run.

---

### Task 1: Rigid-motion primitives in the cursor env

**Files:**
- Modify: `src/lerobot/envs/cursor.py` (after the `Layouts` class and its helpers)
- Test: `tests/examples/pi05_mars_cursor/test_rigid_motion.py` (new; import through `cursor_sa.env`, as `test_obstacles.py` does)

**Interfaces:**
- Consumes: `Layouts` (`start [B, 2]`, `targets [B, N, 2]`, `mask`, `goal`, `obstacles [B, K, 3]`, `obstacle_mask`, `scene`), `SCREEN_EDGE`.
- Produces:
  - `move_layouts(layouts: Layouts, angle: np.ndarray, shift: np.ndarray, mirror: np.ndarray) -> Layouts` — `angle [B]` radians, `shift [B, 2]`, `mirror [B]` bool; every point `p` becomes `R(angle) (M(p) - c) + c + shift` with `c = (0.5, 0.5)`, `M` the mirror `x -> 1 - x` when `mirror`, `R` the rotation `[[cos, -sin], [sin, cos]]`; applied to `start`, `targets` (padding rows too), disc centres; radii, `mask`, `goal`, `obstacle_mask` copied; `scene=None`.
  - `move_actions(actions: np.ndarray, angle: np.ndarray, mirror: np.ndarray) -> np.ndarray` — `[B, L, 2]` displacements: mirror flips the x component, then the rotation is applied (no shift).
  - `layouts_on_screen(layouts: Layouts, edge: float = SCREEN_EDGE) -> np.ndarray` — bool `[B]`: `start` and every real target inside `[edge, 1 - edge]^2`, and every real disc's centre plus and minus its radius inside it.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_zero_motion_is_the_identity_and_drops_the_scene():
    lay = _lay(); lay.scene = {"x": np.zeros((len(lay), 2))}
    out = move_layouts(lay, np.zeros(len(lay)), np.zeros((len(lay), 2)), np.zeros(len(lay), bool))
    for a, b in [(out.start, lay.start), (out.targets, lay.targets), (out.obstacles, lay.obstacles)]:
        assert np.allclose(a, b)
    assert (out.mask == lay.mask).all() and (out.goal == lay.goal).all() and (out.obstacle_mask == lay.obstacle_mask).all()
    assert out.scene is None

def test_a_quarter_turn_about_the_centre_and_a_mirror():
    lay = _one(start=(0.7, 0.5), goal=(0.5, 0.8))         # helper as in test_obstacles.py
    out = move_layouts(lay, np.array([np.pi / 2]), np.zeros((1, 2)), np.array([False]))
    assert np.allclose(out.start[0], (0.5, 0.7)) and np.allclose(out.goal_pos[0], (0.2, 0.5))
    out = move_layouts(lay, np.zeros(1), np.array([[0.1, -0.1]]), np.array([True]))
    assert np.allclose(out.start[0], (0.4, 0.4))

def test_a_motion_is_rigid_and_keeps_radii():
    # pairwise distances among start, real targets and real disc centres are preserved; radii equal
    ...

def test_moved_actions_follow_the_moved_start():
    # start' + cumsum(move_actions(a)) == the moved path (move_layouts applied to start + cumsum(a) point by point)
    ...

def test_on_screen_needs_the_discs_and_targets_inside_the_edge():
    # a disc at (0.05, 0.5) radius 0.08 -> False; moved inward by 0.1 -> True; a target at (0.995, 0.5) -> False
    ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_rigid_motion.py -q`
Expected: FAIL with ImportError on `move_layouts`.

- [ ] **Step 3: Implement the three functions in `src/lerobot/envs/cursor.py`** (vectorised; `np.einsum` or `@` with a `[B, 2, 2]` rotation stack; the padding target rows move too so no mask logic is needed there).

- [ ] **Step 4: Run the tests, mypy and pre-commit**

Run: the pytest command above; `uv run --extra dev mypy src/lerobot/envs/cursor.py`; `uv run --extra dev pre-commit run --files src/lerobot/envs/cursor.py tests/examples/pi05_mars_cursor/test_rigid_motion.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/envs/cursor.py tests/examples/pi05_mars_cursor/test_rigid_motion.py
git commit -m "MARS round 4: rigid motions of a cursor layout and its actions"
```

---

### Task 2: Demonstration copies under rigid motion, and jitter in the sampler

**Files:**
- Modify: `examples/pi05/mars/cursor/cursor_sa/demos.py` (`DemoSampler.__init__` / `sample`; new `augment_demos` after `make_demos`)
- Test: `tests/examples/pi05_mars_cursor/test_augmentation.py` (new)

**Interfaces:**
- Consumes: Task 1's three functions (import from `lerobot.envs.cursor` as `demos.py` imports the env); `replay(actions, layouts, task)` (`.success`, `.steps`); `Layouts.concat`, `Layouts.__getitem__`.
- Produces:
  - `augment_demos(rng: np.random.Generator, layouts, actions, lengths, k: int, task, push=None, shift: float = 0.25, tries: int = 20) -> tuple[Layouts, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]` — `(layouts_out, actions_out, lengths_out, kept, push_out)`: rows `0..B-1` are the originals unchanged (layouts, actions, lengths, push bit for bit), then the accepted copies grouped by original in original order; `kept [B]` int = copies per original (`0` for a demonstration of length 0); `push_out` = `push` (`[B, L, 2]`) with each copy's rows `move_actions(push[i], angle, mirror)` of its motion, `None` when `push` is `None`. Draws per round, for every original still short of `k`: `angle ~ U[0, 2 pi)`, `shift ~ U[-shift, shift]^2`, `mirror ~ Bernoulli(1/2)`, in that order from `rng`; a copy is accepted by the Global Constraints' rule, its `lengths` from `replay(...).steps`; at most `tries * k` rounds. `replay` runs 500 rows at a time.
  - `DemoSampler(layouts, actions, lengths, horizon, hist, device, frames=None, jitter: float = 0.0, jitter_steps: int = 4)` — `sample` applies the jitter of the Global Constraints when `jitter > 0` (the returned chunk is a fresh tensor; `obs["hist"]` is untouched).

- [ ] **Step 1: Write the failing tests**

```python
def test_copies_are_rigid_images_of_their_originals_and_the_originals_come_first():
    lay = sample_layouts(rng, 6, 4, task); actions, lengths = make_demos(rng, lay, task)
    out, acts, lens, kept = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, task)
    assert len(out) == 6 + kept.sum() and (kept <= 2).all()
    assert np.array_equal(out.start[:6], lay.start) and np.array_equal(acts[:6], actions) and np.array_equal(lens[:6], lengths)
    # every copy: on screen, replay succeeds and ends on its goal, distances start-goal-discs preserved
    env = replay(acts[6:], out[6:], task); assert env.success.all()
    assert layouts_on_screen(out[6:]).all()

def test_a_failed_demonstration_gets_no_copies():
    # lengths[i] = 0 -> kept[i] == 0 and no row for it

def test_copies_run_out_on_a_screen_spanning_reach():
    # a start at (0.02, 0.5) and goal at (0.98, 0.5) with tries=1: kept[0] < k (most draws leave the screen), no error

def test_free_layouts_are_copied_too():
    # task without obstacle: copies kept and replays succeed

def test_pushes_move_with_their_copies():
    push = np.random.default_rng(2).normal(size=actions.shape)
    out, acts, lens, kept, moved = augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, task, push=push)
    assert np.array_equal(moved[:6], push) and moved.shape == acts.shape
    # a rigid motion preserves dot products: per step, moved push . moved action == push . action of the original
    origin = np.repeat(np.arange(6), kept)
    assert np.allclose((moved[6:] * acts[6:]).sum(-1), (push[origin] * actions[origin]).sum(-1), atol=1e-9)
    assert augment_demos(np.random.default_rng(1), lay, actions, lengths, 2, task)[4] is None

def test_jitter_zero_draws_nothing():
    torch.manual_seed(0); a = sampler.sample(8); torch.manual_seed(0); b = DemoSampler(..., jitter=0.0).sample(8)
    assert torch.equal(a[0]["pos"], b[0]["pos"]) and torch.equal(a[1], b[1])

def test_jitter_displaces_the_position_and_the_chunk_returns_to_the_path():
    s = DemoSampler(lay, actions, lengths, horizon=8, hist=2, device="cpu", jitter=0.02, jitter_steps=4)
    torch.manual_seed(0); obs, chunk, i, k = s.sample(4096, indexed=True)
    ref = DemoSampler(lay, actions, lengths, 8, 2, "cpu")
    delta = obs["pos"] - ref.pos[i, k]
    assert abs(delta.std() - 0.02) < 0.002
    base = ref.actions[i[:, None], k[:, None] + ref.window][:, 2:]
    assert torch.allclose(chunk[:, :4].sum(1), base[:, :4].sum(1) - delta, atol=1e-6)
    assert torch.equal(chunk[:, 4:], base[:, 4:])
    assert torch.equal(obs["hist"], ref.actions[i[:, None], k[:, None] + ref.window][:, :2])   # hist untouched

def test_jitter_with_a_short_chunk_corrects_over_what_there_is():
    # horizon=2, jitter_steps=4: the whole chunk's sum equals base - delta
```

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_augmentation.py -q`
Expected: FAIL (ImportError / TypeError on the new keyword).

- [ ] **Step 3: Implement `augment_demos` and the sampler's jitter in `demos.py`**

- [ ] **Step 4: Run the new tests and the sampler's existing ones**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_augmentation.py tests/examples/pi05_mars_cursor/test_obstacles.py tests/examples/pi05_mars_cursor/test_training.py::test_frozen_noise_serves_one_draw_per_demo_step -q`
Expected: all pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
git add examples/pi05/mars/cursor/cursor_sa/demos.py tests/examples/pi05_mars_cursor/test_augmentation.py
git commit -m "MARS round 4: rigid copies of the demonstrations and a jittered sampler"
```

---

### Task 3: The trainer's flags

**Files:**
- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (`ExpertData.__init__`, `expert_splits`, the task `meta`, `parse_args`, the module docstring's TASK MODULE paragraph)
- Test: `tests/examples/pi05_mars_cursor/test_training.py`

**Interfaces:**
- Consumes: Task 2's `augment_demos` and the sampler's `jitter`, `jitter_steps`; `corruption_spec` (its curl); `_record_splits` (existing test helper that captures every split's `ExpertData`).
- Produces: flags `--jitter FLOAT` (default 0.0), `--jitter-steps INT` (default 4), `--rigid-motions INT` (default 0); `ExpertData(layouts, actions, lengths, policy, device, corruption, curl, frozen_noise=False, frames=None, push=None, jitter=0.0, jitter_steps=4)` — `push` (`[B, L, 2]`, given) replaces `expert_pushes(layouts, pos)`, `jitter` and `jitter_steps` go to its sampler; in `expert_splits`, for the train split only, after its demonstrations are drawn and before any rendering or frame table, when `args.rigid_motions`: the originals' pushes `expert_pushes(layouts, pos)` (pos as `ExpertData` computes it), then `augment_demos(np.random.default_rng([args.seed, 11]), layouts, actions, lengths, args.rigid_motions, task, push=push)`, the moved pushes handed to that split's `ExpertData` as `push=`, a printed line `train: <B> demonstrations and <kept.sum()> moved copies (<min>-<max> per demonstration)`, and `args.rigid_kept` accumulated over entries; `meta["jitter"]`, `meta["jitter_steps"]`, `meta["rigid_motions"]`, `meta["rigid_kept"]` (`None` for user modules; `rigid_kept` `None` when `--rigid-motions 0`). Without `--rigid-motions` the pushes are computed inside `ExpertData` as today (no extra call).

- [ ] **Step 1: Write the failing tests** (beside `test_train_a_relative_geometry_image_task_module`, same `COMMON` and `cnn --render-size 64 --token-geometry relative` setup)

```python
def test_train_with_jitter_and_rigid_motions_records_them(ckpt_dir, monkeypatch):
    splits = _record_splits(monkeypatch)   # the existing helper
    ... train task with "--jitter", "0.02", "--jitter-steps", "3", "--rigid-motions", "2" ...
    meta = torch.load(ckpt_dir / "t_aug" / "adapter.pt")["meta"]
    assert meta["jitter"] == 0.02 and meta["jitter_steps"] == 3 and meta["rigid_motions"] == 2
    assert 0 < meta["rigid_kept"] <= 2 * 60
    train, val = splits[-1]["train"], splits[-1]["val"]   # _record_splits appends every call's {"train", "val"} dict
    assert train.sampler.jitter == 0.02 and val.sampler.jitter == 0.0
    assert len(train.sampler.lengths) == 60 + meta["rigid_kept"] and len(val.sampler.lengths) == 40
    # a copy's pushes are its original's moved: push . action per step equal between copy and original (as in Task 2's test),
    # and they differ from what the expert would recompute on the moved layout for at least one copy row

def test_the_pushes_of_a_copy_are_not_recomputed(ckpt_dir, monkeypatch):
    # monkeypatch train_adapter.expert_pushes to count calls and record the batch size: with --rigid-motions 2 it is
    # called once for the train split with the 60 originals (and once for the val split), never with 60 + kept rows

def test_the_new_flags_default_off_and_leave_the_meta_explicit(ckpt_dir):
    # a run without the flags: meta jitter 0.0, jitter_steps 4, rigid_motions 0, rigid_kept None; a user module: all None

def test_jitter_and_rigid_motions_are_refused_where_they_make_no_sense():
    # SystemExit for: --token-source vlm_views with --jitter 0.02; with --rigid-motions 2;
    # --rigid-motions 2 with --corruption '{curl: 2.0}'; --rigid-motions 2 with --teacher <name>
```

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q -k "jitter or rigid or new_flags"`
Expected: FAIL (unrecognised arguments).

- [ ] **Step 3: Implement the flags, the refusals, the split wiring, the meta and the docstring paragraph**

- [ ] **Step 4: Run the whole training test file**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q`
Expected: all pass (run it in two halves with `-k` if CPU contention makes it slow, and say so).

- [ ] **Step 5: Pre-commit and commit**

```bash
git add examples/pi05/mars/cursor/train/train_adapter.py tests/examples/pi05_mars_cursor/test_training.py
git commit -m "MARS round 4: the trainer jitters its samples and copies its demonstrations under rigid motions"
```

---

### Task 4: The layout diagnostic

**Files:**
- Create: `examples/pi05/mars/cursor/diag_layouts.py`
- Test: `tests/examples/pi05_mars_cursor/test_diag_layouts.py` (new)

**Interfaces:**
- Consumes: `train_adapter.parse_args`, `train_adapter.layouts_for(args, obstacle, rng, n, p_free)`, `obstacle_spec`; `run_sweep.load_sweep`, `method_arm(sweep, method, level)`, `scene_sources(adapters)`, `attach_scene(layouts, sources, cache_dir, key, encoder_factory, vlm)`, `run_cell_batch(sweep, arm, method, layouts, noises, agent, policy, adapters, device, seed)`; `cursor_sa.session.load_checkpoints(arms, device)`; `cursor_sa.agents.make_agent`; Task 1's `move_layouts`, `layouts_on_screen`.
- Produces:
  - `rebuild_layouts(seed: int, n: int, p_free: float, obstacle, targets=(4, 4), reach=(0.35, 0.5)) -> Layouts` — the trainer's train-split layouts for those arguments (`layouts_for` with `np.random.default_rng(seed)`, the first draws of that stream; the defaults are the trainer's `--targets` and `--reach`).
  - `moved_layouts(layouts, rng, shift=0.25, tries=50) -> Layouts` — one accepted random rigid motion per row (redrawn until `layouts_on_screen`; a row that never fits keeps its original, counted and printed).
  - `main(argv)` with `--sweep NAME --context T2|T4 --method oracle_task_rel --seed 0 --n-demos 300 --p-free 0.25 --fresh-seed 820 --trials N --out DIR [--cpu]`: the sets `train` (the rebuilt layouts' obstacle rows), `fresh` (`layouts_for` with `default_rng(fresh_seed)`, the same count, `p_free 0`) and `moved` (`moved_layouts(train, default_rng([seed, 12]))`, a stream disjoint from the trainer's rigid copies), capped at `--trials` rows each; for each set and each of `oracle`, `oracle_task`, `--method`: `attach_scene`, `run_cell_batch` with the sweep's seed and noise 0.0; prints a Markdown table `set x arm` of `hits / collisions` in % and writes `<out>/<context>_<method>.json` (`{set: {arm: {"hits": ..., "collision": ..., "n": ...}}}`).

- [ ] **Step 1: Write the failing tests**

```python
def test_rebuilt_layouts_are_the_trainers(ckpt_dir, monkeypatch):
    splits = _record_splits(monkeypatch)     # from test_training.py (import it)
    ... a tiny task run with COMMON (60 demos, p_free 0.25) ...
    lay = rebuild_layouts(0, 60, 0.25, obstacle_spec({...}))
    train = splits[-1]["train"].sampler
    assert np.allclose(lay.start, train.pos[:, 0].numpy()) and np.allclose(lay.obstacles, train.obstacles.numpy())

def test_moved_layouts_stay_on_screen_and_are_rigid():
    ...

def test_the_diagnostic_runs_end_to_end_on_a_tiny_sweep(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch, geometry="relative")   # from test_sweep.py
    ... a tiny sweep yaml with oracle, oracle_task (adapter: null is fine for the tiny grid), oracle_task_rel -> name ...
    main(["--sweep", str(yaml), "--context", "T2", "--method", "oracle_task_rel", "--n-demos", "6", "--trials", "3",
          "--out", str(tmp_path / "diag"), "--scene-cache", str(tmp_path / "scene"), "--cpu"])
    out = json.loads((tmp_path / "diag" / "T2_oracle_task_rel.json").read_text())
    assert set(out) == {"train", "fresh", "moved"} and all(0 <= v["hits"] <= 1 for s in out.values() for v in s.values())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_diag_layouts.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write `diag_layouts.py`** (a script like `report_grid_image.py`: module docstring with the command, `main(argv=None)`, `if __name__ == "__main__": sys.exit(main())`).

- [ ] **Step 4: Run the tests**

Expected: all pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
git add examples/pi05/mars/cursor/diag_layouts.py tests/examples/pi05_mars_cursor/test_diag_layouts.py
git commit -m "MARS round 4: a diagnostic of an image module on its training, fresh and moved layouts"
```

---

### Task 5: The gate sweep and the report's arm prefix

**Files:**
- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_rel_aug.yaml`
- Modify: `examples/pi05/mars/cursor/report_grid_image.py` (`_arm_means`, `select_recipe`, `width_check` untouched, `scan_rows` and the `--scan` count regex, `parse_args`)
- Test: `tests/examples/pi05_mars_cursor/test_report_grid_image.py`, `tests/examples/pi05_mars_cursor/test_sweep.py` (`test_image_sweep_configs_resolve_every_arm_per_cell`)

**Interfaces:**
- Consumes: the existing `report_grid_image.py` functions and their tests' synthetic check dicts.
- Produces:
  - The sweep config, following `grid_task_check_views_n300.yaml`: seed 820, T2 and T4 as there, `trials: {tune: 20, test: 500}`, methods `oracle`, `oracle_task`, `oracle_ft` (the same fine-tuned models), `oracle_task_rel` (`image/task_disc_large_n300_s0_with_gain025y_vlm_image_rel`, `image/task_ring_goal_n300_s0_with_rot60_vlm_image_rel`), `oracle_task_rel_jit`, `oracle_task_rel_rigid`, `oracle_task_rel_both` (the same names with `_jit`, `_rigid`, `_both` appended), `policy`, `policy_task`, and `policy_task_rel`, `policy_task_rel_jit`, `policy_task_rel_rigid`, `policy_task_rel_both`; `focus: oracle_task_rel_both`; `against: [oracle_task, oracle_ft]`.
  - `_arm_means(check, arms, prefix="views")` reading `oracle_task_<prefix>_<arm>`; `select_recipe(check, base="base", changes=None, prefix="views")`; `scan_rows(runs, prefix="views")` and the `--scan` discovery regex `oracle_task_<prefix>_n(\d+)$`; CLI `--prefix` (default `views`) and `--base` (default `base`) on `--select` and `--prefix` on `--scan`. Every existing call and test computes as before.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_reads_a_prefix_and_a_base_arm():
    check = _run({"T2": {"oracle_task_rel": _m(0.55), "oracle_task_rel_jit": _m(0.59), "oracle_task_rel_rigid": _m(0.57)},
                  "T4": {"oracle_task_rel": _m(0.35), "oracle_task_rel_jit": _m(0.39), "oracle_task_rel_rigid": _m(0.37)}})   # the file's helpers
    out = select_recipe(check, base="rel", changes=["jit", "rigid"], prefix="rel")
    assert out["adopted"] == ["jit"]   # jit 4 points over the base, rigid 2 over

def test_scan_reads_a_prefix():
    ... runs with oracle_task_rel_n300 rows; scan_rows(runs, prefix="rel") gives the same rows as the views case

def test_the_select_cli_passes_prefix_and_base(tmp_path):
    ... main(["--select", run, "--prefix", "rel", "--base", "rel", "--results", ..., "--out", ...]) writes select.json
```
and add `grid_task_check_rel_aug` to the list in `test_image_sweep_configs_resolve_every_arm_per_cell`.

- [ ] **Step 2: Run them to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py tests/examples/pi05_mars_cursor/test_sweep.py::test_image_sweep_configs_resolve_every_arm_per_cell -q`
Expected: FAIL.

- [ ] **Step 3: Write the config and the report changes**

- [ ] **Step 4: Run the two test files**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py tests/examples/pi05_mars_cursor/test_sweep.py -q`
Expected: all pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
git add examples/pi05/mars/cursor/configs/sweeps/grid_task_check_rel_aug.yaml examples/pi05/mars/cursor/report_grid_image.py tests/examples/pi05_mars_cursor/test_report_grid_image.py tests/examples/pi05_mars_cursor/test_sweep.py
git commit -m "MARS round 4: the augmentation gate sweep and an arm prefix in the report"
```

---

### Task 6 (controller, not a build task): the runs

- [ ] Job list `examples/pi05/mars/results/research/image/jobs_r4_aug.txt`: `rel300`'s two commands with `--jitter 0.02` (`_jit`), `--rigid-motions 32` (`_rigid`), both (`_both`), `--name image/..._vlm_image_rel_<suffix>`; `run_jobs.py jobs_r4_aug.txt --per-gpu 1 --gpus 0 1`.
- [ ] Diagnostic first: `diag_layouts.py --sweep grid_task_check_rel_aug --context T2 --method oracle_task_rel --trials 225 --out ../results/sim/reports/diag_layouts` and the same for T4, one per GPU, before the trainings (the trainers' encoding would collide with the diagnostic's encoder).
- [ ] The six trainings; then `run_sweep.py --sweep grid_task_check_rel_aug` as `pipeline.sh`'s `views_sweep` runs a sweep (one part per cell, two GPUs, merged); then `report_grid_image.py --check grid_task_check_rel_aug` and `--select grid_task_check_rel_aug --prefix rel --base rel --out ../results/sim/reports/grid_rel_aug`.
- [ ] Decide by the Global Constraints' rule; report to the user (diagnostic table, gate table, decision, costs, rulings); stage 2 (the scan with the chosen recipe; with rigid motion in it, the scene count held at about 10,000: `K = 9` at 1,000, `K = 2` at 3,000) waits for their go.
