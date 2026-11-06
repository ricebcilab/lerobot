# MARS round 4b: the goal-leak check and the clean-up — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A decoy-reach obstacle placement and its sweep (does the image module infer the goal from the obstacle?), the cursor example's scripts moved into folders by role, and the two test folders cut to about half with one behaviour test per feature.

**Architecture:** One optional key in the obstacle spec, drawn only when set, so every existing layout is unchanged; a sweep config and one report column. The moves are mechanical with the test path setup, README and run-folder scripts following. The test cut keeps every behaviour pinned once and runs a tiny training only per end-to-end path.

**Tech Stack:** Python 3.12, numpy, PyTorch, the MARS cursor example (`examples/pi05/mars/cursor`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-mars-image-relative-augmentation-design.md`, section "Addendum (2026-11-06)".

## Global Constraints

- Without `decoy: true` every layout, demonstration and run computes exactly as today: no new draw from any generator, no changed output. Existing checkpoints load unchanged.
- `src/lerobot/envs/cursor.py` is mypy-strict.
- Tests on the CPU only: `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest <files> -q`; GPUs are busy with training jobs and must not be used. Nothing under `examples/pi05/mars/results/` or `examples/pi05/mars/checkpoints/` is touched by a build task (the archive is the controller's).
- Training and sweep jobs run from this checkout: never leave a file syntactically broken on disk; `run_sweep.py` and `train/train_adapter.py` keep their paths and public names.
- Pre-commit on every touched file; commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` (exactly that name); `git add` names exactly the task's files; no push.
- Test-cut targets (both folders together): at most 180 tests and 4,800 lines (from 327 and 8,785), the whole of `tests/examples/pi05_mars_cursor` plus `tests/policies/steering` under 10 minutes on an idle CPU; `test_training.py` at most 9 tiny training runs. The equivalence tests (`test_ranks_equivalence.py`, `test_steering_equivalence.py`, `test_spacemouse_equivalence.py`, `test_steering_core_cursor.py`) and the `_reference_*.py` files stay as they are.

## Review Focus

- A decoy layout whose decoy target coincides with the goal direction (a target near the goal): the obstacle may still block the true reach; the passability check about the true goal must still hold and demonstrations must still succeed. Test in Task A.
- A layout with only one real target (no decoy available): `decoy: true` must refuse clearly, not index out of range. Test in Task A.
- After the moves, every command in the README and every run-folder script that names a moved script must still run (`python tools/diag_layouts.py --help` etc.). Test in Task B1 (a test that imports each moved module through the test path setup).
- Every test dropped in Task B2 must be listed with its disposition (merged into which test, or redundant with which, or why it pinned nothing); a behaviour pinned only by a dropped test is a defect. Reviewed in Task B2.
- The whole-suite runtime claim must be measured, not estimated. Task B2 reports the measured wall time.

---

### Task A: Decoy-reach obstacles, the leak sweep and the wrong-target column

**Files:**
- Modify: `src/lerobot/envs/cursor.py` (`obstacle_spec` defaults/validation; `sample_layouts`), `examples/pi05/mars/cursor/report_grid_image.py` (`gate_table` and its Markdown: a `wrong target` column from the `wrong_target` outcome share, next to `collision`; `n/a` when absent)
- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_rel_leak.yaml`
- Test: `tests/examples/pi05_mars_cursor/test_obstacles.py`, `tests/examples/pi05_mars_cursor/test_report_grid_image.py`, the sweep-config resolution test in `tests/examples/pi05_mars_cursor/test_sweep.py`

**Interfaces:**
- Consumes: `obstacle_spec(spec)` (defaults per kind; unknown keys refused), `sample_layouts(rng, n_trials, n_targets, cfg)` (draw order: targets, then starts, then `place_obstacles(rng, s, pts[goal], cfg.obstacle)`), `place_obstacles`, `_passable(obstacles, mask, s, pts, goal, sign, cfg)`.
- Produces: obstacle specs accept `decoy: bool` (default `False`) for every kind. With `decoy` true, `sample_layouts` draws `d = rng.integers(n - 1)` right before `place_obstacles` and places relative to `pts[d if d < goal else d + 1]` (a real non-goal target); `_passable` and everything after stay about the true goal. `n_targets` of 1 with `decoy` raises `ValueError("decoy needs at least two targets")`. `gate_table` rows gain `wrong_target` (share of `wrong_target` outcomes of the `oracle_task_<x>` arm) and the gate Markdown a `wrong target` column; `report.json` carries it.
- The sweep config `grid_task_check_rel_leak.yaml`: like `grid_task_check_rel_rigid.yaml` (seed 820, 500 test trials, `focus: oracle_task_rel_rigid`, `against: [oracle_task]`) with `decoy: true` added to both cells' obstacle specs (`{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024], decoy: true}`, `{kind: ring, around: goal, gap_angle: [15, 30], decoy: true}`), the same `adapter:` references (the parametric modules trained on reach-relative obstacles), and the rows `oracle`, `oracle_task`, `oracle_task_rel_rigid`, `policy`, `policy_task`, `policy_task_rel_rigid` (no `oracle_ft`).

- [ ] **Step 1: Write the failing tests**

```python
def test_a_decoy_obstacle_sits_on_the_reach_to_a_non_goal_target():
    spec = obstacle_spec({"kind": "disc", "radius": 0.05, "along": 0.5, "decoy": True})
    lay = sample_layouts(np.random.default_rng(0), 40, 4, TaskConfig(obstacle=spec))
    for b in range(40):
        start, goal, disc = lay.start[b], lay.goal_pos[b], lay.real_obstacles(b)[0, :2]
        others = lay.targets[b][lay.mask[b] & ~lay.goal_mask()[b]]
        # the disc centre is at `along` of the reach to exactly one non-goal target (offset 0 by default), never to the goal
        on = [np.allclose(disc, start + 0.5 * (t - start), atol=1e-6) for t in others]
        assert sum(on) == 1 and not np.allclose(disc, start + 0.5 * (goal - start), atol=1e-6)
    assert np.allclose(lay.goal_pos, lay.goal_pos)  # the goal is still the goal

def test_layouts_without_decoy_are_bit_identical_to_before():
    # the same seed with and without the key explicitly False gives equal layouts; and a spec without the key equals one with decoy False

def test_decoy_needs_two_targets():
    with pytest.raises(ValueError, match="decoy"): sample_layouts(np.random.default_rng(0), 2, 1, TaskConfig(obstacle=obstacle_spec({..., "decoy": True})))

def test_decoy_layouts_still_admit_a_demonstration():
    # make_demos on 20 decoy layouts (disc and ring) succeeds on all (lengths > 0)

def test_the_gate_table_reports_wrong_targets():
    # a synthetic check with outcomes: wrong_target share appears in the row and the Markdown has the column; n/a when the records lack it
```
and add `grid_task_check_rel_leak` to the config-resolution test's list.

- [ ] **Step 2: Run them to verify they fail**
- [ ] **Step 3: Implement** (`obstacle_spec`: `"decoy": False` in every kind's defaults, validated as a bool; `sample_layouts`: the decoy draw and target; `gate_table`/Markdown; the config).
- [ ] **Step 4: Run the three test files and mypy on `cursor.py`**
- [ ] **Step 5: Pre-commit and commit** — `MARS round 4b: decoy-reach obstacles, the goal-leak sweep and a wrong-target column`

---

### Task B1: The example's scripts move into folders by role

**Files:**
- Move (git mv): `examples/pi05/mars/cursor/{analyze_results,convert_checkpoints,hub,diag_layouts}.py` → `examples/pi05/mars/cursor/tools/`; `examples/pi05/mars/cursor/report_grid_image.py` → `examples/pi05/mars/cursor/reports/`
- Modify: `tests/examples/pi05_mars_cursor/conftest.py` (add `cursor/tools` and `cursor/reports` to `sys.path`), `examples/pi05/mars/cursor/README.md` (every command naming a moved script), `examples/pi05/mars/cursor/hub_manifest.yaml` if it names `hub.py`, the module docstrings of the moved scripts (their own command lines), `examples/pi05/mars/cursor/reports/report_grid_image.py` and `tools/diag_layouts.py` where they locate `configs/` or `results/` relative to their own file (one level deeper now: check `cursor_paths` use and any `Path(__file__).parent` arithmetic), and the git-ignored run-folder scripts under `examples/pi05/mars/results/research/image/` that name `report_grid_image.py` or `diag_layouts.py` (`pipeline.sh`, `gate4.sh`, `drive3.sh`, `sweep_one.sh`, `smoke3_sweep.sh`): these are not committed; edit them in place.
- Test: `tests/examples/pi05_mars_cursor/test_tools_layout.py` (new, tiny): importing `diag_layouts`, `report_grid_image`, `hub`, `convert_checkpoints`, `analyze_results` through the conftest path works, and `tools/` and `reports/` contain exactly those files plus `build_study_report.py`.

**Interfaces:** Consumes nothing new. Produces the new paths; every import name stays the same (the folders are on the test path, not packages), so `from report_grid_image import ...` and `from diag_layouts import ...` keep working.

- [ ] **Step 1: Write the failing layout test** (it fails because the folders do not exist)
- [ ] **Step 2: `git mv` the five scripts; fix path arithmetic inside them; update conftest, README, manifest, run-folder scripts**
- [ ] **Step 3: Run** `CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_tools_layout.py tests/examples/pi05_mars_cursor/test_report_grid_image.py tests/examples/pi05_mars_cursor/test_diag_layouts.py tests/examples/pi05_mars_cursor/test_hub.py -q` and `uv run --extra mars python examples/pi05/mars/cursor/tools/diag_layouts.py --help`, `.../reports/report_grid_image.py --help`
- [ ] **Step 4: Pre-commit and commit** — `MARS: the cursor example's tools and reports move into folders by role`

---

### Task B2: The test cut

**Files:**
- Modify or delete under `tests/examples/pi05_mars_cursor/` and `tests/policies/steering/` (never the four equivalence test files, the `_reference_*.py` files, `_view_stub.py` or `conftest.py` beyond what B1 did).

**Interfaces:** none. The code under test does not change in this task.

Principles (binding): one behaviour test per feature; a tiny training run only once per end-to-end path (user module; task module with disc tokens; image task module, cnn relative; jitter + rigid; pooled; teacher; two views; fine-tune on a budget; the user-then-task pair counts as one path) — fold the other assertions of that path into that one run; refusals of a flag family in one parametrised parse-only test; no test whose failure cannot be caused by the behaviour it names; formatting of a report pinned once; per-round files folded into the file of the feature (`test_rigid_motion.py` and `test_diag_layouts.py` into `test_augmentation.py`; `test_scene.py` stays for the env's views, the token/encoder tests stay in `tests/policies/steering`); example-level duplicates of library-level steering tests dropped in favour of the library test.

Per-file targets (tests): `test_training.py` ≤ 14 (≤ 9 tiny runs); `test_sweep.py` ≤ 14; `test_report_grid_image.py` ≤ 10; `test_cursor_flow.py` ≤ 18; `test_steering_wrappers.py` ≤ 24; `test_steering_core.py` ≤ 14; `test_scene_tokens.py` ≤ 9; `test_augmentation.py` (merged) ≤ 12; `test_scene.py` ≤ 6; `test_obstacles.py` ≤ 8; `test_session.py` ≤ 7; `test_figures.py` ≤ 7; `test_hub.py` ≤ 5; `test_adapter.py` ≤ 7; `test_grid.py` + `test_grid_sweep.py` ≤ 6 (one file); `test_checkpoint_formats.py` ≤ 4; `test_scene_encoder.py` ≤ 6; `test_steering_shared.py` 4. Whole-suite: ≤ 180 tests, ≤ 4,800 lines, under 10 minutes measured on the idle CPU (`time` the two folders once at the end; the training file alone must be under 7 minutes).

- [ ] **Step 1: Inventory** — write `tests-cut-ledger.md` in the SDD workspace: every current test with its disposition (keep / merge into `<test>` / drop as redundant with `<test>` / drop because it cannot fail on its claim), before touching a file.
- [ ] **Step 2: Cut and merge file by file**, running each file after its cut.
- [ ] **Step 3: Run both folders once, timed** (`time CUDA_VISIBLE_DEVICES="" uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor tests/policies/steering -q`), and record the counts (`pytest --collect-only -q | tail -1`, `wc -l`).
- [ ] **Step 4: Pre-commit and commit** — `MARS: the cursor and steering test suites cut to one behaviour test per feature`

---

### Task C (controller): archive the abandoned branches' artifacts

After the stage-2 driver has finished (nothing running references them): move, never delete, with `old/<round>/` folders beside the originals and an index `examples/pi05/mars/results/research/image/ARCHIVE.md` (what moved where, the headline numbers, the ledger paths):
- Round 1 (absolute and cnn modules): `checkpoints/adapters/image/{*_cnn, *_vlm_image (not *_rel*), task_all_*, task_ring_start_*, task_wall_gap_*}`; sweeps `grid_task_check_image`; reports `grid_image_gate`; scene cache `grid_task_check_image`; run folder `jobs_cnn_s*.txt`, `jobs_vlm_s*.txt`, `jobs_s*.txt`, `jobs_diag_*`, `pooled_s*.yaml`, `make_jobs.py`, `diag/`, `pipeline_*.out`, the round-1 `DONE_*` files.
- Round 3 (two views): `checkpoints/adapters/image/*views*` and `image_smoke/`; sweeps `grid_views_*`, `grid_task_check_views_n300(_parts)`; reports `grid_views_n300`, `grid_views_scan`; scene cache `grid_task_check_views_n300`; run folder `drive3.sh`, `make_jobs3.py`, `jobs_r3_*`, `grid_views_*.yaml`, `smoke3*`, `sweep_one.sh`, `DONE_gate3b_n300`.
- Round 4 abandoned arms: `checkpoints/adapters/image/*_rel_jit`, `*_rel_both` (the gate sweep that measured them stays).
Kept in place: round 2's `rel` and `teacher_rel` modules and `grid_task_check_image2_quick`; round 4's rigid modules, `grid_task_check_rel_*` sweeps and reports, `diag_layouts`; the run folder's `pipeline.sh`, `run_jobs.py`, `jobs_r2_*`, `jobs_r4_*`, `drive4.sh`, `gate4.sh`, `logs/`.

### Task D (controller, on the user's approval): rename the kept artifacts

Proposal to the user (not done until approved): module names drop the token-source infix, `image/task_<ctx>_n<N>_s<k>_with_<user>_rel`, `..._rel_rigid`, `..._teacher_rel`; sweep and report names drop `grid_task_check_` for `gate_` (`gate_rel_aug`, `gate_rel_n300`, `gate_rel_rigid_s1`, `gate_rel_leak`; round 2's `grid_task_check_image2_quick` becomes `gate_rel_round2`); the committed sweep configs, the README, the hub manifest and the gate/driver scripts follow; ledgers and the design docs keep their historical names.
