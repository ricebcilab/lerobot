# MARS round 4c: a disk token store and the width check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The task-module trainer keeps scene tokens in a disk store shared across runs and moves each batch to the GPU, so a run's size is bounded by disk, not GPU memory; the relative image module can read reduced-width tokens; the width check says whether the reduction costs accuracy.

**Architecture:** A `TokenStore` (float16 memory-mapped array plus a digest index) keyed per layout, so any run whose layouts were already encoded reuses their rows; the sampler keeps row indices and gathers a batch from the store per step through a pinned host buffer. The library's `scene_rank` reduction, today `vlm_views`-only, becomes a property of any token source. The width check trains the rigid recipe at full width and at rank 256 from the same code and gates both on the same layouts.

**Tech Stack:** Python 3.12, numpy (memmap), PyTorch, the MARS cursor example.

**Spec:** this plan's "Why" section is the spec (the user's decision of 2026-11-06 afternoon: dataset on disk like the π0.5 fine-tune, then the width check; stop before the grid and the curve).

## Why

Today's trainer holds every scene's tokens on the GPU (float16, 1 MB per picture): 300 demonstrations with 32 rigid copies plus 2,000 validation scenes are 12 GB; 1,000 with 32 copies are 35 GB and 3,000 are 101 GB, beyond a 49 GB card, and Figure 3e's 180-minute budget (11,203 demonstrations) with 32 copies is 372 GB. The scan therefore held the scene count near 10,000 (ruling R18), which changed what it measured. Encoding runs at 82 pictures a second, so re-encoding per batch (the π0.5 fine-tune's way, which trains its encoder) would cost six seconds a step; the frozen encoder's tokens must be precomputed and stored. A reduced width (PCA to rank 256, about 98.5 % of the variance, eight times smaller) makes the curve's stores fit comfortably, if the relative module loses nothing worth noting: the width check, with the 5-point rule from round 3.

## Global Constraints

- Without the new flags every existing command computes exactly as today: same draws, same losses, same saved weights (the store is an implementation of where the tokens live; a run with `--scene-store` set and the rest unchanged must produce the same module as the GPU-resident path, checked by a test on a tiny run with a stub encoder).
- Store format: `<dir>/<source>_<size>_<encoder digest>/tokens.f16` (a raw float16 array `[rows, 256, 2048]` opened with `np.memmap`, grown by appending), `index.json` (`{layout digest: row}`), `meta.json` (shape, dtype, the encoder path and digest). A layout's digest is the sha256 of its `start`, `targets`, `mask`, `goal`, `obstacles`, `obstacle_mask` arrays as bytes plus the render spec (size), 16 hex characters. Appends are atomic per batch (write rows, then update the index); a crashed append leaves the index consistent with the rows it names.
- The sampler never holds full-width tokens on the GPU for a stored source: per batch it gathers `rows[i]` from the memmap into a pinned float16 host buffer and copies it to the device (non-blocking); the returned `obs["scene"][source]` is the same tensor the GPU-resident path returns, bit for bit.
- Reduced width: `--scene-rank R` for `vlm_image` (and `vlm_prefix`) fits the reduction on 2,000 rows sampled from the train split's rows (centred unless `--scene-center no`), stores it in the module (`set_reduction`), and the sampler feeds reduced tokens (`reduce_tokens` on the gathered batch, on the device); the module's `tokens()` already accepts full-width or reduced input for a module carrying a basis, so sweeps need no change.
- `src/lerobot/policies/steering/*` keeps its public names; `scene_rank` becomes legal for every source except `discs` (config key `scene_rank` written when set, for any source); existing checkpoints load unchanged.
- Width-check rule: the reduced module may trail the full-width one by at most 5 points of hits on each context (the round-3 `WIDTH_DROP`), both trained from this code on the same demonstrations and copies and gated on the same 500 layouts.
- Tests on the CPU only (`CUDA_VISIBLE_DEVICES=""`); pre-commit on every touched file; commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; `git add` names exactly the task's files; no push; nothing under `examples/pi05/mars/results/` or `checkpoints/` touched by a build task (tests use `tmp_path` stores).

## Review Focus

- Two runs that share layouts (the same seed and count, different flags) must share rows: the second run encodes nothing for them. Test in Task A.
- A store from another encoder path or render size must not be read: its folder name differs. Test in Task A.
- A row gathered for a copy must be the copy's own picture, not its original's (the digest includes the moved arrays). Test in Task A.
- `--scene-rank` on `vlm_image` with `--token-geometry relative`: the offsets are added after `scene_in`, so the reduction must not touch them. Test in Task A (the relative module's tokens equal projection-plus-geometry on reduced input).
- The budget-nesting assumption for the curve's shared stores: a smaller `--demo-seconds` draw is a prefix of a larger one at the same seed. Test in Task A (`budget_demos` at 30 and 60 seconds with the same seed).

---

### Task A: The token store, the sampler's batch path, reduced width for the image sources

**Files:**
- Create: `examples/pi05/mars/cursor/cursor_sa/scene_store.py`
- Modify: `examples/pi05/mars/cursor/cursor_sa/demos.py` (`DemoSampler`), `examples/pi05/mars/cursor/train/train_adapter.py` (`expert_splits`, flags, meta, docstring), `src/lerobot/policies/steering/modeling_steering.py` (`LearnedTaskModule.__init__` validation and config of `scene_rank`; `scene_spec`), `src/lerobot/policies/steering/configuration_steering.py` if `scene_spec` lives there, `examples/pi05/mars/cursor/reports/report_grid_image.py` (`width_check` and `--width` take `--prefix`, default `views`)
- Test: `tests/examples/pi05_mars_cursor/test_scene_store.py` (new), `tests/examples/pi05_mars_cursor/test_training.py` (one added assertion block in the existing image run, one new tiny run), `tests/policies/steering/test_scene_tokens.py` (the `scene_rank` validation and the relative-plus-reduced check), `tests/examples/pi05_mars_cursor/test_report_grid_image.py` (`--width --prefix rel`)

**Interfaces:**
- Consumes: `scene_for(layouts, source, size, encoder) -> np.ndarray` (float16 `[B, 256, 2048]` for the vlm sources); `fit_token_reduction(batches, rank, center=True) -> (mean, basis, kept)`; `reduce_tokens(tokens, mean, basis)`; `LearnedTaskModule.set_reduction(mean, basis)`; `DemoSampler(layouts, actions, lengths, horizon, hist, device, frames=None, jitter=0.0, jitter_steps=4)`; `ExpertData(..., push=None, jitter=0.0, jitter_steps=4)`; `budget_demos(args, obstacle, rng, seconds, p_free)`.
- Produces:
  - `class TokenStore(dir: Path, source: str, size: int, encoder_digest: str)` with `rows_for(layouts, encoder) -> np.ndarray[int64]` (the row of every layout; layouts absent from the index are rendered, encoded with `scene_for` in batches of 256 and appended, in index order), `gather(rows: np.ndarray, out: torch.Tensor | None = None) -> torch.Tensor` (float16 `[len(rows), 256, 2048]` on the CPU, into a pinned buffer when given), `sample_rows(rows, n, rng) -> np.ndarray` (for the reduction fit), properties `n_rows`, `path`; `layout_digests(layouts, size) -> list[str]`; `encoder_digest(vlm_path) -> str` (the same 12-hex digest the sweep's `scene_digest` derives from the encoder path, reused or factored out of `run_sweep.py`). The store is a plain object; the encoder is passed in by the caller and may be `None` when every row exists.
  - `DemoSampler(..., store: tuple[TokenStore, np.ndarray, str] | None = None, reduction: tuple[torch.Tensor, torch.Tensor] | None = None)` — `(store, rows_per_demo, source)`; `sample` gathers `rows[i]` through a pinned buffer sized for the largest `n` seen, moves it to the device, applies `reduce_tokens(batch, mean, basis)` when `reduction` is set, and puts it in `obs["scene"][source]`; with `store` set, `layouts.scene` is not read.
  - Trainer flags: `--scene-store DIR` (default `None` = today's GPU-resident path; the controller will pass `checkpoints/scene/store`), `--scene-rank R` now accepted for `vlm_image` and `vlm_prefix` (refused for `discs` and `cnn`; `--scene-center` applies), the fit on 2,000 rows sampled with `np.random.default_rng([args.seed, 13])` from the train split's rows (centred by default); meta gains `scene_store` (the folder name or `None`), `scene_rank`, `scene_center` for these sources; a printed line `train: <n> scenes, <m> newly encoded (<s> s), store <path>` per split.
  - `report_grid_image.py`: `width_check(check, reduced="base", full="full", context="T4", prefix="views")`; `--width` takes `--prefix` and `--context`; everything existing computes as before.

- [ ] **Step 1: Write the failing tests**

```python
# test_scene_store.py (a stub encoder whose image_tokens returns a deterministic function of the picture)
def test_rows_are_shared_between_two_draws_of_the_same_layouts(tmp_path):
    # rows_for on layouts A encodes len(A) rows; rows_for on A again (new store object, same dir) encodes 0 and returns the same rows
def test_a_copy_has_its_own_row_and_a_different_encoder_its_own_store(tmp_path):
    # move_layouts(A, ...) rows differ from A's; a store with another encoder_digest has a different path and empty index
def test_gather_returns_the_stored_tokens_bit_for_bit_and_fills_a_pinned_buffer(tmp_path):
def test_a_crashed_append_leaves_the_index_consistent(tmp_path):
    # simulate: write rows, raise before the index update; reopening finds n_rows == len(index)
def test_budgets_are_nested_at_one_seed():
    # budget_demos at 30 s and at 60 s with default_rng(0): the 30 s layouts equal the first rows of the 60 s layouts
# test_training.py
def test_a_stored_run_equals_the_resident_run(ckpt_dir, tmp_path, monkeypatch):
    # tiny vlm_image run with the ViewStub encoder, once resident, once with --scene-store tmp_path/store: identical adapter.pt state_dict and train_log
def test_a_reduced_image_module_trains_from_the_store_and_carries_its_basis(ckpt_dir, tmp_path, monkeypatch):
    # --token-source vlm_image --token-geometry relative --scene-rank 8 --scene-store ...: meta scene_rank 8, scene_basis non-zero, scene_in.in_features 8; --scene-rank with discs/cnn -> SystemExit
# test_scene_tokens.py
def test_scene_rank_is_legal_for_the_image_sources_and_relative_tokens_add_the_geometry_after_the_projection():
# test_report_grid_image.py
def test_width_reads_a_prefix(): ...
```

- [ ] **Step 2: Run them to verify they fail**
- [ ] **Step 3: Implement** (`scene_store.py`; the sampler path; the trainer's store wiring in `expert_splits` for the train and val splits, the fit, the meta, the docstring; the library validation; the report prefix).
- [ ] **Step 4: Run** `tests/examples/pi05_mars_cursor/test_scene_store.py test_training.py test_augmentation.py test_report_grid_image.py tests/policies/steering/test_scene_tokens.py` and `mypy src/lerobot/envs/cursor.py` (unchanged) — the steering package is not mypy-strict.
- [ ] **Step 5: Pre-commit and commit** — `MARS round 4c: a disk token store, the sampler's batch path and reduced width for the image sources`

---

### Task B (controller): the width check

- [ ] Job list `jobs_r4c_width.txt`: the rigid recipe at 300 demonstrations on T2 and T4 twice, `--scene-store checkpoints/scene/store` in both, once at full width (`..._vlm_image_rel_rigid_full`) and once with `--scene-rank 256` (`..._vlm_image_rel_rigid_r256`); one job per GPU (the second pair reuses the first pair's rows; the fit adds under a minute).
- [ ] Sweep `grid_task_check_rel_width.yaml` (seed 820, 500 trials, T2 and T4): rows `oracle_task`, `oracle_task_rel_full`, `oracle_task_rel_r256`; then `report_grid_image.py --width grid_task_check_rel_width --prefix rel --reduced r256 --full full --context T2` and `--context T4`.
- [ ] Report: the two numbers, the rule's verdict per context, the store's size on disk and the gather time per step at both widths; stop before the grid and the curve.
