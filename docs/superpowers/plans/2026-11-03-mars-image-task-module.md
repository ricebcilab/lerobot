# MARS Image Task Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A MARS task module that reads the cursor task's context from a rendered image through π0.5's frozen vision tokens (or a from-scratch CNN as control), trained with the existing trainer at 300 demos per context (per-task and pooled over T2–T5), gated and evaluated on the grid's 16 obstacle cells against the parametric module and the controls.

**Architecture:** `Layouts` grows a `scene` dict (per-layout arrays: a rendered image for the CNN, VLM tokens for `vlm_*`) that rides through indexing/tile/concat into the policy observation, so `bind(obs)` and the flows are untouched. `LearnedTaskModule` grows a `token_source` that swaps where its cross-attention keys come from (disc tokens today; projected VLM tokens or CNN tokens with a learned 2D positional embedding). A small `scene.py` renders layouts headlessly and, lazily loading `lerobot/pi05_base`, turns renders into π0.5 image tokens. The trainer and sweep runner attach scenes where they draw layouts; the sweep caches them per cell on disk.

**Tech Stack:** Python 3.12+, PyTorch, numpy, Pillow (in the `mars` extra), π0.5 via `lerobot.policies.pi05` (`transformers`, in the `pi` extra), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-11-03-mars-image-task-module-design.md`

## Global Constraints

- Run everything from the repo root with `uv run --extra mars`; steps that touch π0.5 (VLM scene tokens, the real-weights smoke test) add `--extra pi`. `lerobot.policies.steering` must import without `transformers` (lazy imports inside functions only).
- `lerobot.envs` is mypy-strict: annotate every new function in `src/lerobot/envs/cursor.py`.
- Existing `discs` checkpoints must load unchanged: a module with `token_source="discs"` saves exactly the config it saves today (the new keys appear in `config` only for image sources, the `obstacle_tokens` pattern of `cursor_flow`).
- The render never marks the goal; it draws obstacle discs and target circles only (no cursor, HUD, plan or command), in the high-contrast palette `RENDER_PALETTE` (Task 1), at `SCENE_SIZE = 224` for every real run.
- Training recipe of every new task module = the grid's: `--n-demos 300 --frozen-noise --steps 20000` (pooled: `--steps 80000`), `--p-free 0.25 --goal-dropout 0.25 --lambda-id 1e-3 --lambda-field 0.03 --batch 512 --lr 1e-3`, `--seed k`, and the grid's user pairing: T2 `{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024]}` with `user_gain025y_expert_n100_s<k>` / `{scale: [1.0, 0.25]}`; T3 `{kind: wall, along: 0.5, gap: 0.16, gap_offset: 0.14, radius: 0.015, extent: 0.4}` with `user_gain025x_expert_n100_s<k>` / `{scale: [0.25, 1.0]}`; T4 `{kind: ring, around: goal, gap_angle: [15, 30]}` with `user_rot60_expert_n100_s<k>` / `{rotation_deg: 60}`; T5 `{kind: ring, around: start, gap_angle: [15, 30]}` with `user_curl20_expert_n100_s<k>` / `{curl: 20, tau: 0.1}`.
- Checkpoint names: `image/task_<disc_large|wall_gap|ring_goal|ring_start>_n300_s<k>_with_<gain025y|gain025x|rot60|curl20>_<vlm_image|cnn>`; pooled `image/task_all_n1200_s<k>_<source>`. Reference parametric modules: `task_disc_large_n300_s0_with_gain025y`, `task_wall_gap_n300_s0_with_gain025x`, `task_ring_goal_n300_s0_with_rot60`, `task_ring_start_n300_s0_with_curl20`.
- Sweeps reuse `condition_grid.yaml`, the grid's `seed`, `trials: {tune: 200, test: 500}`, users U1–U4, tasks T2–T5 and `train: [[U1, T2], [U2, T3], [U3, T4], [U4, T5]]` copied from `configs/sweeps/grid.yaml`; MARS variants tune `{n_reverse_steps: [1..10], pinned_steps: [2, 4, 8, 16]}`.
- Gate (spec): per context, `oracle_task_<x>` hits ≥ 80% and collision ≤ parametric `oracle_task` collision + 10 points; goal-hidden `policy_task_<x>` hits > `policy_task` hits + 10 points ⇒ FLAG (reported, not discarded).
- Commits: message style `MARS: <what changed, one sentence>`, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never commit `checkpoints/` or `results/` (git-ignored). Run `pre-commit run --files <changed files>` before each commit.

## Review Focus

1. `Layouts.concat` of parts where some carry a scene and some do not (practice rows next to study rows) — must raise a clear `ValueError`, never silently drop the scene. Test in Task 1.
2. The wall's 0.015-radius discs after anti-aliased downsampling — must stay visible (darker than background along the wall) or the gate blames the module for the render. Test in Task 1.
3. A module with an image `token_source` given an observation that has no matching scene (a sweep whose layouts were never attached, e.g. `grid.yaml` by mistake) — must raise naming the source, not run as the identity; `obs=None` (the LIBERO call) still runs. Test in Task 2.
4. fp16 cached tokens fed to float32 projection weights — must work and match float32 input within tolerance. Test in Task 2.
5. The sweep's scene cache key — must include cell, split, source and image size and must NOT include `--module-seed` (the layouts are the sweep's, not the module's), so seeds 1–2 reuse seed 0's cache and a different render size misses it. Test in Task 7.

---

### Task 1: `Layouts.scene` and the headless render

**Files:**
- Modify: `src/lerobot/envs/cursor.py:74-153` (`Layouts`), add `render_layouts` after `sample_layouts`
- Test: `tests/examples/pi05_mars_cursor/test_scene.py` (new)

**Interfaces:**
- Consumes: `Layouts` (dataclass: `start, targets, mask, goal, obstacles, obstacle_mask`), `Layouts.real_obstacles(b)`, `TaskConfig.target_radius`.
- Produces: `Layouts.scene: dict[str, np.ndarray] | None = None` (7th field, default None; every array `[B, ...]`), carried by `__getitem__`, `tile(k)` and `concat(parts)` (all-None → None; all dicts with equal keys → per-key concatenate; mixed → `ValueError("scene: some parts carry one and some do not")`). `RENDER_PALETTE: dict[str, tuple[int, int, int]] = {"bg": (255, 255, 255), "obstacle": (40, 40, 40), "obstacle_edge": (0, 0, 0), "target_fill": (205, 218, 242), "target_edge": (60, 100, 200)}`, `SCENE_SIZE: int = 224`, `render_layouts(layouts: Layouts, size: int = SCENE_SIZE, target_radius: float = 0.05, supersample: int = 2) -> np.ndarray` returning uint8 `[B, 3, size, size]`: the unit screen mapped to the square (x right, y down as `renderer.to_px` maps it), drawn with Pillow at `size * supersample` then resized with `Image.LANCZOS`; disc radius in pixels clamped to ≥ 1.5 px at `size`; targets as filled circles with a 2 px edge; no goal mark, no cursor.

- [ ] **Step 1: Write the failing tests**

```python
# tests/examples/pi05_mars_cursor/test_scene.py
import numpy as np, pytest
from cursor_sa.env import Layouts, RENDER_PALETTE, SCENE_SIZE, TaskConfig, obstacle_spec, render_layouts, sample_layouts

def _layouts(n=3, spec=None, seed=0):
    cfg = TaskConfig(min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec(spec))
    return sample_layouts(np.random.default_rng(seed), n, 4, cfg)

def test_scene_rides_through_indexing_tile_and_concat():
    a = _layouts(3); a.scene = {"cnn": np.arange(3 * 2 * 4 * 4, dtype=np.uint8).reshape(3, 2, 4, 4), "vlm_image": np.ones((3, 5, 6), np.float16)}
    assert a[[1]].scene["cnn"].shape == (1, 2, 4, 4) and np.array_equal(a[[1]].scene["cnn"][0], a.scene["cnn"][1])
    perm = np.array([2, 0, 1]); assert np.array_equal(a[perm].scene["vlm_image"], a.scene["vlm_image"][perm])
    assert a.tile(2).scene["cnn"].shape == (6, 2, 4, 4) and np.array_equal(a.tile(2).scene["cnn"][3:], a.scene["cnn"])
    both = Layouts.concat([a, a[[0]]]); assert both.scene["vlm_image"].shape == (4, 5, 6) and set(both.scene) == {"cnn", "vlm_image"}
    plain = _layouts(2); assert plain.scene is None and Layouts.concat([plain, plain]).scene is None and plain.tile(2).scene is None
    with pytest.raises(ValueError, match="scene"):
        Layouts.concat([a, plain])

def test_render_draws_discs_and_targets_and_never_the_goal():
    lay = _layouts(2, {"kind": "disc", "radius": 0.08, "along": 0.5})
    img = render_layouts(lay, size=SCENE_SIZE)
    assert img.shape == (2, 3, SCENE_SIZE, SCENE_SIZE) and img.dtype == np.uint8
    x, y, r = lay.real_obstacles(0)[0]; px, py = int(x * SCENE_SIZE), int(y * SCENE_SIZE)
    assert tuple(img[0, :, py, px]) == RENDER_PALETTE["obstacle"]                      # disc center is obstacle colour
    assert tuple(img[0, :, 2, 2]) == RENDER_PALETTE["bg"]                               # a corner is background
    gx, gy = (lay.goal_pos[0] * SCENE_SIZE).astype(int); ox, oy = (lay.targets[0, (lay.goal[0] + 1) % 4] * SCENE_SIZE).astype(int)
    assert tuple(img[0, :, gy, gx]) == tuple(img[0, :, oy, ox]) == RENDER_PALETTE["target_fill"]   # goal looks like any target
    no_obstacle = render_layouts(_layouts(1))
    assert not (no_obstacle == np.array(RENDER_PALETTE["obstacle"])[None, :, None, None]).all(1).any()   # no obstacle pixels

def test_render_keeps_the_wall_visible():
    lay = _layouts(1, {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14, "radius": 0.015, "extent": 0.4})
    img = render_layouts(lay).astype(int)
    discs = lay.real_obstacles(0)
    for x, y, _ in discs[::3]:  # along the wall every disc center is well darker than the background
        assert img[0, :, int(y * SCENE_SIZE), int(x * SCENE_SIZE)].mean() < 128
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py -q`
Expected: FAIL — `ImportError: cannot import name 'RENDER_PALETTE'`.

- [ ] **Step 3: Implement `Layouts.scene` (field, `__getitem__`, `tile`, `concat`) and `render_layouts` in `src/lerobot/envs/cursor.py`**

`tile` tiles each array with `np.tile(v, (k,) + (1,) * (v.ndim - 1))`; `concat` pads nothing (scene arrays share trailing shapes by construction) and raises on a mix. `render_layouts` builds one `PIL.Image` per layout from `real_obstacles(b)` and `targets[b, mask[b]]` (import Pillow inside the function so `lerobot.envs.cursor` keeps importing without it), returns `np.stack` of `np.asarray(img).transpose(2, 0, 1)`.

- [ ] **Step 4: Run the tests and the existing env/obstacle tests**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py tests/examples/pi05_mars_cursor/test_obstacles.py -q && uv run --extra dev mypy src/lerobot/envs/cursor.py`
Expected: all PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/envs/cursor.py tests/examples/pi05_mars_cursor/test_scene.py
git commit -m "MARS: layouts carry a scene and render headlessly" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `LearnedTaskModule.token_source`

**Files:**
- Modify: `src/lerobot/policies/steering/modeling_steering.py:218-310` (`LearnedTaskModule`)
- Test: `tests/policies/steering/test_scene_tokens.py` (new)

**Interfaces:**
- Consumes: `LearnedTaskModule(horizon, d, tdim, heads, layers, scale, n_dims)`, `save_adapter_module`, `load_adapter_module`, `obs["scene"]` as Task 1/4 shape it (a dict source → tensor; `cnn`: uint8 `[B, 3, S, S]`; `vlm_*`: `[B, N, D]` float16/32).
- Produces: `LearnedTaskModule(..., token_source: str = "discs", scene_dim: int | None = None, scene_tokens: int | None = None, image_size: int | None = None)`; `SCENE_SOURCES = ("discs", "vlm_image", "vlm_prefix", "cnn")`; `token_source` attribute. Rules in `__init__`: `vlm_*` needs `scene_dim` and `scene_tokens`; `cnn` needs `image_size % 16 == 0` and sets `scene_tokens = (image_size // 16) ** 2`; `config` gains `token_source, scene_dim, scene_tokens, image_size` only when `token_source != "discs"`. New submodules: `scene_in = nn.Linear(scene_dim, d)` (vlm), `cnn = nn.Sequential` of four `nn.Conv2d(k=3, stride=2, padding=1)` 3→32→64→128→d with GELU between (cnn), `scene_pos = nn.Parameter(0.02 * torch.randn(scene_tokens, d))` (both). `tokens(obs, b)`: `discs` as today; `obs is None` → null token only; image source with `obs.get("scene")` missing the source → `ValueError(f"task module with token_source {src!r} needs obs['scene'][{src!r}]")`; else tokens = `[null, scene_in(x.float()) + scene_pos]` or `[null, cnn(x.float() / 255).flatten(2).transpose(1, 2) + scene_pos]`, mask None. The head stays zero-initialized (identity at init for every source).

- [ ] **Step 1: Write the failing tests**

```python
# tests/policies/steering/test_scene_tokens.py
import pytest, torch
from lerobot.policies.steering import LearnedTaskModule, load_adapter_module, save_adapter_module
from lerobot.policies.steering.modeling_steering import SCENE_SOURCES

def _xvt(b=3, h=16): return torch.randn(b, h, 2), torch.randn(b, h, 2), torch.rand(b)
def _scene(b=3): return {"pos": torch.rand(b, 2), "scene": {"vlm_image": torch.randn(b, 16, 32, dtype=torch.float16), "cnn": torch.randint(0, 256, (b, 3, 64, 64), dtype=torch.uint8)}}
def _module(src): return LearnedTaskModule(horizon=16, token_source=src, scene_dim=32 if src.startswith("vlm") else None, scene_tokens=16 if src.startswith("vlm") else None, image_size=64 if src == "cnn" else None)

@pytest.mark.parametrize("src", ["vlm_image", "vlm_prefix", "cnn"])
def test_image_sources_start_at_identity_and_round_trip(src, tmp_path):
    torch.manual_seed(0); m = _module(src).eval(); x, v, t = _xvt(); obs = _scene()
    a, b = m(x, t, v, obs); assert torch.equal(a, torch.eye(2).expand(3, 16, 2, 2)) and torch.equal(b, torch.zeros(3, 16, 2))
    a, b = m(x, t, v, None); assert torch.equal(a, torch.eye(2).expand(3, 16, 2, 2))          # the LIBERO call still runs
    assert m.config["token_source"] == src and m.config["scene_tokens"] == 16
    with torch.no_grad():
        for p in m.parameters(): p.add_(0.1 * torch.randn_like(p))
    m.requires_grad_(False); path = save_adapter_module(m, tmp_path / "t.pt", kind="task")
    loaded, kind, _ = load_adapter_module(path)
    assert kind == "task" and loaded.token_source == src
    a0, b0 = m(x, t, v, obs); a1, b1 = loaded(x, t, v, obs)
    assert torch.equal(a0, a1) and torch.equal(b0, b1) and not torch.equal(a0, torch.eye(2).expand(3, 16, 2, 2))

def test_discs_config_is_unchanged_and_an_image_module_needs_its_scene():
    assert set(LearnedTaskModule(horizon=16).config) == {"horizon", "d", "tdim", "heads", "layers", "scale", "n_dims"}
    m = _module("vlm_image"); x, v, t = _xvt()
    with pytest.raises(ValueError, match="vlm_image"):
        m(x, t, v, {"pos": torch.rand(3, 2)})
    with pytest.raises(ValueError, match="vlm_image"):
        m(x, t, v, {"pos": torch.rand(3, 2), "scene": {"cnn": torch.zeros(3, 3, 64, 64, dtype=torch.uint8)}})
    with pytest.raises(ValueError): LearnedTaskModule(horizon=16, token_source="vlm_image")              # no scene_dim
    with pytest.raises(ValueError): LearnedTaskModule(horizon=16, token_source="cnn", image_size=50)   # not a multiple of 16
    with pytest.raises(ValueError): LearnedTaskModule(horizon=16, token_source="nope")

def test_fp16_tokens_match_fp32_within_tolerance():
    torch.manual_seed(0); m = _module("vlm_image").eval()
    with torch.no_grad(): torch.nn.init.normal_(m.out.weight, std=0.3)
    x, v, t = _xvt(); obs = _scene(); half = obs["scene"]["vlm_image"]
    a16, b16 = m(x, t, v, obs); a32, b32 = m(x, t, v, {**obs, "scene": {"vlm_image": half.float()}})
    torch.testing.assert_close(a16, a32, atol=1e-3, rtol=1e-3); torch.testing.assert_close(b16, b32, atol=1e-3, rtol=1e-3)
    assert a16.dtype == torch.float32

def test_cnn_tokens_change_with_the_image_and_vlm_tokens_with_the_scene():
    torch.manual_seed(0); x, v, t = _xvt(); obs = _scene()
    for src in ("cnn", "vlm_image"):
        m = _module(src).eval()
        with torch.no_grad(): torch.nn.init.normal_(m.out.weight, std=0.3)
        other = {**obs, "scene": {src: torch.flip(obs["scene"][src], [0])}}   # another layout's scene
        assert not torch.equal(m(x, t, v, obs)[0], m(x, t, v, other)[0])
    assert SCENE_SOURCES == ("discs", "vlm_image", "vlm_prefix", "cnn")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra test pytest tests/policies/steering/test_scene_tokens.py -q`
Expected: FAIL — `ImportError: cannot import name 'SCENE_SOURCES'`.

- [ ] **Step 3: Implement the token sources in `LearnedTaskModule`**

Keep `tokens()` the single dispatch point; `forward` is unchanged. Document the sources in the class docstring (one line each).

- [ ] **Step 4: Run the new and the existing steering tests**

Run: `uv run --extra mars --extra test pytest tests/policies/steering tests/examples/pi05_mars_cursor/test_adapter.py tests/examples/pi05_mars_cursor/test_checkpoint_formats.py -q`
Expected: all PASS (the `discs` reference checks in `_reference_steering` unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/steering/modeling_steering.py tests/policies/steering/test_scene_tokens.py
git commit -m "MARS: the task module's scene tokens come from a pluggable source" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `scene.py` — scene arrays for a source, with π0.5's vision tokens

**Files:**
- Create: `src/lerobot/policies/steering/scene.py`
- Modify: `src/lerobot/policies/steering/__init__.py` (export `SceneEncoder`, `scene_for`, `scene_spec`, `SCENE_SOURCES`), `src/lerobot/policies/steering/README.md` (a paragraph on the sources)
- Test: `tests/policies/steering/test_scene.py` (new)

**Interfaces:**
- Consumes: `render_layouts`, `SCENE_SIZE` (Task 1); `SCENE_SOURCES` (Task 2); π0.5: `PI05Policy.from_pretrained(path)`, `policy.model.paligemma_with_expert.embed_image(img)` (float32 `[B, 3, 224, 224]` in [-1, 1] → `[B, 256, 2048]`), `policy.model.embed_prefix(images, img_masks, tokens, masks)` and `policy.model.paligemma_with_expert.forward(attention_mask, position_ids, inputs_embeds=[prefix, None], use_cache=False)` (→ `[prefix_hidden, None], _`), `resize_with_pad_torch`.
- Produces:
  - `scene_spec(source: str, size: int = SCENE_SIZE) -> dict` → `{"token_source", "scene_dim", "scene_tokens", "image_size"}` for `LearnedTaskModule(**spec)`: `vlm_*` → `scene_dim 2048, scene_tokens 256, image_size size` (the encoder always resizes the render to 224 for SigLIP, so the token count never depends on the render size; `size` is only what is drawn and cached); `cnn` → `scene_dim None, scene_tokens (size // 16) ** 2, image_size size` (size % 16 == 0 required); `discs` → `{"token_source": "discs"}`.
  - `class SceneEncoder(path: str = "lerobot/pi05_base", device: str = "cuda", prompt: str = "Move the cursor to the target.")`, lazy: nothing π0.5-related is imported until `image_tokens`/`prefix_tokens` first run; `image_tokens(images: np.ndarray, batch: int = 64) -> np.ndarray` (uint8 `[B, 3, S, S]` → float16 `[B, 256, 2048]`; resize-with-pad to 224 when S ≠ 224; `/255`, `* 2 - 1`, `embed_image`, no grad); `prefix_tokens(images, batch=64) -> np.ndarray` (float16 `[B, 256, 2048]`: the image-token slice of the prefix's last hidden states with the fixed prompt tokenized by the policy's tokenizer, no state tokens); `close()` frees the policy.
  - `scene_for(layouts, source: str, size: int = SCENE_SIZE, encoder=None) -> np.ndarray`: `cnn` → `render_layouts(layouts, size)`; `vlm_image` → `encoder.image_tokens(render)`; `vlm_prefix` → `encoder.prefix_tokens(render)`; `discs` → `ValueError`. `encoder` is anything with those two methods (tests pass a stub).

- [ ] **Step 1: Write the failing tests**

```python
# tests/policies/steering/test_scene.py
import importlib, os, sys, numpy as np, pytest, torch
from lerobot.envs.cursor import SCENE_SIZE, TaskConfig, obstacle_spec, sample_layouts
from lerobot.policies.steering import LearnedTaskModule, SceneEncoder, scene_for, scene_spec

class Stub:
    def __init__(self): self.calls = []
    def image_tokens(self, images): self.calls.append(images.shape); return np.zeros((len(images), 256, 2048), np.float16) + images.mean()
    def prefix_tokens(self, images): return self.image_tokens(images) + 1

def _layouts(n=4): return sample_layouts(np.random.default_rng(0), n, 4, TaskConfig(min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})))

def test_scene_spec_builds_a_module_that_accepts_scene_for():
    lay = _layouts(); stub = Stub()
    for src in ("vlm_image", "vlm_prefix", "cnn"):
        spec = scene_spec(src, 112); m = LearnedTaskModule(horizon=16, **spec)
        arr = scene_for(lay, src, 112, encoder=stub)
        assert arr.shape[0] == 4 and (arr.dtype == np.uint8 if src == "cnn" else arr.dtype == np.float16)
        obs = {"pos": torch.rand(4, 2), "scene": {src: torch.as_tensor(arr)}}
        a, _ = m(torch.randn(4, 16, 2), torch.rand(4), torch.randn(4, 16, 2), obs); assert a.shape == (4, 16, 2, 2)
    assert stub.calls[0] == (4, 3, 112, 112) and scene_spec("vlm_image")["scene_tokens"] == 256 and scene_spec("cnn")["scene_tokens"] == 196
    assert scene_spec("vlm_image", 112)["scene_tokens"] == 256 and scene_spec("cnn", 112)["scene_tokens"] == 49
    with pytest.raises(ValueError): scene_for(lay, "discs")
    with pytest.raises(ValueError): scene_spec("cnn", 50)

def test_steering_imports_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)   # import of transformers would now raise ImportError
    for name in ("lerobot.policies.steering", "lerobot.policies.steering.scene"):
        importlib.reload(importlib.import_module(name))
    enc = SceneEncoder(device="cpu"); assert enc is not None     # lazy: constructing it loads nothing

@pytest.mark.skipif(os.environ.get("MARS_SCENE_REAL") != "1", reason="loads lerobot/pi05_base (14 GB); MARS_SCENE_REAL=1 to run")
def test_real_pi05_image_tokens_have_the_documented_shape():
    enc = SceneEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    tok = scene_for(_layouts(2), "vlm_image", SCENE_SIZE, encoder=enc); pre = scene_for(_layouts(2), "vlm_prefix", SCENE_SIZE, encoder=enc)
    assert tok.shape == (2, 256, 2048) and tok.dtype == np.float16 and np.isfinite(tok.astype(np.float32)).all()
    assert pre.shape == (2, 256, 2048) and not np.array_equal(tok, pre); enc.close()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/policies/steering/test_scene.py -q`
Expected: FAIL — `ImportError: cannot import name 'SceneEncoder'`.

- [ ] **Step 3: Implement `scene.py` and the exports**

`image_tokens` runs `policy.model.paligemma_with_expert.embed_image` under `torch.no_grad()` in batches on `device`; `prefix_tokens` builds the prefix as `PI05Pytorch.forward` does (`embed_prefix` → `make_att_2d_masks` → `position_ids = cumsum(pad) - 1`) and takes `[:, :256]` of the language model's last hidden state. Both cast to float16 numpy on CPU.

- [ ] **Step 4: Run the fast tests, then the real one once**

Run: `uv run --extra mars --extra test pytest tests/policies/steering/test_scene.py -q`
Expected: 2 PASS, 1 skipped.
Run: `CUDA_VISIBLE_DEVICES=1 MARS_SCENE_REAL=1 uv run --extra mars --extra pi --extra test pytest tests/policies/steering/test_scene.py -q -k real`
Expected: PASS (loads `lerobot/pi05_base` offline from the HF cache; ~1–2 min).

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/policies/steering/scene.py src/lerobot/policies/steering/__init__.py src/lerobot/policies/steering/README.md tests/policies/steering/test_scene.py
git commit -m "MARS: scene arrays for the task module's sources, with pi0.5's image tokens" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The scene reaches the cursor observation and the demo sampler

**Files:**
- Modify: `examples/pi05/mars/cursor/cursor_sa/steering.py:101-109` (`observation`), `examples/pi05/mars/cursor/cursor_sa/demos.py:170-199` (`DemoSampler`)
- Test: `tests/examples/pi05_mars_cursor/test_scene.py` (extend)

**Interfaces:**
- Consumes: `Layouts.scene` (Task 1).
- Produces: `observation(layouts, pos, hist, device, goal_given)` adds `obs["scene"] = {k: torch.as_tensor(v, device=device)}` when `layouts.scene is not None` (dtypes kept: uint8 / float16); `DemoSampler` keeps `self.scene` the same way and `sample()` adds `obs["scene"] = {k: v[i]}`. `hide_goal` needs no change (shallow copy).

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/examples/pi05_mars_cursor/test_scene.py
import torch
from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.steering import SharedController, hide_goal, observation
from cursor_sa.policy import FlowPolicy

def test_observation_and_sampler_carry_the_scene():
    lay = _layouts(3, {"kind": "disc", "radius": 0.08, "along": 0.5}); lay.scene = {"cnn": render_layouts(lay, 64)}
    obs = observation(lay, lay.start, np.zeros((3, 4, 2)))
    assert obs["scene"]["cnn"].dtype == torch.uint8 and obs["scene"]["cnn"].shape == (3, 3, 64, 64) and "scene" in hide_goal(obs)
    assert "scene" not in observation(_layouts(2), np.zeros((2, 2)), np.zeros((2, 4, 2)))
    policy = FlowPolicy().eval(); ctl = SharedController(policy, lay, "policy")
    assert torch.equal(ctl.observation()["scene"]["cnn"], obs["scene"]["cnn"])
    cfg = TaskConfig(min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5}))
    sampler = DemoSampler(lay, *make_demos(np.random.default_rng(0), lay, cfg), policy.horizon, policy.hist, "cpu")
    o, _, i, _ = sampler.sample(5, indexed=True)
    assert torch.equal(o["scene"]["cnn"], torch.as_tensor(lay.scene["cnn"])[i])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py -q -k observation_and_sampler`
Expected: FAIL — `KeyError: 'scene'`.

- [ ] **Step 3: Implement in `observation()` and `DemoSampler`**

- [ ] **Step 4: Run the cursor suite**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor -q -x --ignore=tests/examples/pi05_mars_cursor/test_figures.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/cursor_sa/steering.py examples/pi05/mars/cursor/cursor_sa/demos.py tests/examples/pi05_mars_cursor/test_scene.py
git commit -m "MARS: the cursor observation and demo sampler carry the scene" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Trainer — `--token-source` for a per-task image module

**Files:**
- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (`expert_splits` 162-180, `train_task` 337-375, `save` 409-417, `parse_args` 437-end, module docstring)
- Test: `tests/examples/pi05_mars_cursor/test_training.py` (extend)

**Interfaces:**
- Consumes: `scene_for`, `scene_spec`, `SceneEncoder` (Task 3); `TaskModule = LearnedTaskModule`.
- Produces: arguments `--token-source {discs,vlm_image,vlm_prefix,cnn}` (default `discs`), `--render-size` (default `SCENE_SIZE`), `--vlm` (default `lerobot/pi05_base`); `expert_splits(args, obstacle, policy, device, rng, corruption, curl, p_free=0.0, encoder=None)` attaches `layouts.scene = {args.token_source: scene_for(layouts, args.token_source, args.render_size, encoder)}` to both splits when the source is not `discs` (after `budget_demos`/`layouts_for`, before `ExpertData`); `train_task` builds `TaskModule(policy.horizon, d=args.d, layers=args.layers, **scene_spec(args.token_source, args.render_size))`, creates one `SceneEncoder(args.vlm, device)` for `vlm_*` and `close()`s it after the splits are built; `save` meta gains `token_source`, `render_size`. The `cnn` encoder trains with the module (it is a submodule, so `train_module` already optimizes it).

- [ ] **Step 1: Write the failing test**

```python
# appended to tests/examples/pi05_mars_cursor/test_training.py (COMMON and ckpt_dir as the file defines them)
def test_train_an_image_task_module_from_the_render(ckpt_dir):
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_cnn",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "cnn",
                                                  "--render-size", "64", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t_cnn" / "adapter.pt")
    assert ckpt["config"]["token_source"] == "cnn" and ckpt["config"]["image_size"] == 64 and ckpt["meta"]["token_source"] == "cnn"
    loaded = ReversalAdapter.load(ckpt_dir / "t_cnn" / "adapter.pt"); assert loaded.task.token_source == "cnn"
    log = json.loads((ckpt_dir / "t_cnn" / "train_log.json").read_text())["log"]
    assert {"val_fm", "chunk_collision_hidden", "collide_composed"} <= set(log[-1])

def test_vlm_sources_use_the_encoder_once_per_split(ckpt_dir, monkeypatch):
    calls = []
    class Stub:
        def __init__(self, *a, **k): pass
        def image_tokens(self, images): calls.append(len(images)); return np.zeros((len(images), 256, 2048), np.float16)
        def close(self): calls.append("closed")
    monkeypatch.setattr(train_adapter, "SceneEncoder", Stub)
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--name", "t_vlm",
                                                  "--user-module", "u", "--corruption", "{rotation_deg: 60}", "--token-source", "vlm_image",
                                                  "--render-size", "56", *COMMON]))
    assert calls[-1] == "closed" and len(calls) == 3                      # train split, val split, closed
    assert torch.load(ckpt_dir / "t_vlm" / "adapter.pt")["config"]["scene_dim"] == 2048
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q -k "image_task_module or vlm_sources"`
Expected: FAIL — `error: unrecognized arguments: --token-source`.

- [ ] **Step 3: Implement the arguments, the attach in `expert_splits`, the module construction and the meta**

Import `SceneEncoder, scene_for, scene_spec` from `lerobot.policies.steering` at module level (they are lazy). Extend the docstring's TASK MODULE paragraph with two sentences on the image sources.

- [ ] **Step 4: Run the training tests**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/train/train_adapter.py tests/examples/pi05_mars_cursor/test_training.py
git commit -m "MARS: train a task module on the rendered scene" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Trainer — one module pooled over several contexts

**Files:**
- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (`train_task`, `parse_args`, `save`, docstring)
- Test: `tests/examples/pi05_mars_cursor/test_training.py` (extend)

**Interfaces:**
- Consumes: Task 5's trainer.
- Produces: `--pooled FILE` (YAML `entries: [{obstacle: <spec>, user_module: <name>, corruption: <spec>}, ...]`; with it `--obstacle/--user-module/--corruption` are refused by `parse_args`, `args.obstacle = None`); `train_task` builds one `expert_splits` and one frozen user module per entry (each entry `--n-demos` demos, its own `obstacle_spec`/`corruption_spec`), one module; `batch_loss(step)` uses entry `step % len(entries)` (its data for the flow-matching loss, its user module for λ_field); `validate()` reports the per-entry numbers suffixed `_<j>` (`val_fm_0`, `chunk_collision_hidden_0`, …) plus their means under the unsuffixed names; `save` meta gains `pooled: entries` (and `obstacle: None`).

- [ ] **Step 1: Write the failing test**

```python
# appended to tests/examples/pi05_mars_cursor/test_training.py
def test_train_a_pooled_task_module_over_two_contexts(ckpt_dir, tmp_path):
    for name, corr in (("u_rot", "{rotation_deg: 60}"), ("u_gain", "{scale: [1.0, 0.25]}")):
        train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", corr, "--name", name, *COMMON]))
    (tmp_path / "pooled.yaml").write_text(
        "entries:\n  - {obstacle: {kind: disc, radius: 0.08, along: 0.5}, user_module: u_rot, corruption: {rotation_deg: 60}}\n"
        "  - {obstacle: {kind: ring, around: start, gap_angle: [15, 30]}, user_module: u_gain, corruption: {scale: [1.0, 0.25]}}\n")
    train_adapter.train(train_adapter.parse_args(["--kind", "task", "--pooled", str(tmp_path / "pooled.yaml"), "--name", "t_all",
                                                  "--token-source", "cnn", "--render-size", "64", *COMMON]))
    ckpt = torch.load(ckpt_dir / "t_all" / "adapter.pt")
    assert ckpt["meta"]["obstacle"] is None and [e["user_module"] for e in ckpt["meta"]["pooled"]] == ["u_rot", "u_gain"]
    log = json.loads((ckpt_dir / "t_all" / "train_log.json").read_text())["log"]
    assert {"val_fm", "val_fm_0", "val_fm_1", "chunk_collision_hidden_1"} <= set(log[-1])
    with pytest.raises(SystemExit):   # --pooled and --obstacle together
        train_adapter.parse_args(["--kind", "task", "--pooled", str(tmp_path / "pooled.yaml"), "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}",
                                  "--name", "x", "--corruption", "{rotation_deg: 60}"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q -k pooled`
Expected: FAIL — `error: unrecognized arguments: --pooled`.

- [ ] **Step 3: Implement the pooled mode**

Refactor `train_task` so the single-context path is the one-entry case of the pooled path (one list of `(data, user)` pairs); do not duplicate the loss code.

- [ ] **Step 4: Run the training tests**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/train/train_adapter.py tests/examples/pi05_mars_cursor/test_training.py
git commit -m "MARS: one task module trained over several contexts" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Sweep runner — attach and cache scenes per cell

**Files:**
- Modify: `examples/pi05/mars/cursor/run_sweep.py` (after `load_checkpoints` at 848-851; the cell loop at 926-935; `main` arguments at 803-816; module docstring)
- Test: `tests/examples/pi05_mars_cursor/test_sweep.py` (extend)

**Interfaces:**
- Consumes: `scene_for`, `SceneEncoder` (Task 3); `Layouts.scene` (Task 1); loaded `adapters` (`ReversalAdapter.task.token_source`, `.image_size`).
- Produces: `scene_sources(adapters) -> dict[str, int]` (source → image size over every loaded task module with an image source); `attach_scene(layouts, sources: dict[str, int], cache_dir: Path, key: str, encoder_factory) -> None` (per source: `cache_dir / f"{key}_{source}_{size}.npy"` loaded if present, else computed with `scene_for` and saved; `encoder_factory()` called at most once per process, only for `vlm_*`); in the cell loop, `attach_scene(tune_layouts, ..., key=f"{cell_key(level, noises[0], axes)}_tune")` and `_test` likewise with `cache_dir = CHECKPOINTS / "scene" / sweep["name"]`; `--precompute-scene` (attach for every cell and exit before tuning); `--scene-cache DIR` override (default above). `--module-seed` does not enter the key.

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/examples/pi05_mars_cursor/test_sweep.py
from cursor_sa import paths as cursor_paths
from lerobot.policies.steering import LearnedTaskModule, save_adapter_module
from run_sweep import attach_scene, scene_sources

def _image_adapter(tmp_path, monkeypatch, name="image/t_cnn"):
    monkeypatch.setattr(cursor_paths, "CHECKPOINTS", tmp_path / "ckpt")
    import cursor_sa.session as session_mod, run_sweep as sweep_mod
    monkeypatch.setattr(session_mod, "adapter_path", lambda n: tmp_path / "ckpt" / "adapters" / n / "adapter.pt")
    save_adapter_module(LearnedTaskModule(horizon=16, token_source="cnn", image_size=64), tmp_path / "ckpt" / "adapters" / name / "adapter.pt", "task")
    return name

def test_scene_cache_key_is_cell_split_source_and_size_not_the_module_seed(tmp_path):
    lay = sample_layouts(np.random.default_rng(0), 3, 4, TaskConfig(min_reach=0.3, max_reach=0.5))
    calls = []
    attach_scene(lay, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=lambda: calls.append(1))
    assert lay.scene["cnn"].shape == (3, 3, 64, 64) and (tmp_path / "U1xT2|1.0_test_cnn_64.npy").exists() and calls == []
    again = sample_layouts(np.random.default_rng(0), 3, 4, TaskConfig(min_reach=0.3, max_reach=0.5))
    attach_scene(again, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=None); assert np.array_equal(again.scene["cnn"], lay.scene["cnn"])
    attach_scene(again, {"cnn": 32}, tmp_path, "U1xT2|1.0_test", encoder_factory=None); assert (tmp_path / "U1xT2|1.0_test_cnn_32.npy").exists()

def test_tiny_sweep_attaches_the_scene_for_an_image_task_module(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch)
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_image\nagent: expert\narm: condition_grid.yaml\nmethods:\n  teleop: {mode: teleop}\n"
        f"  mars_cnn: {{mode: frs, reversal_adapter: {{task: {name}}}, tune: {{n_reverse_steps: [2]}}}}\n"
        "grid:\n  users: {U1: {corruption: {scale: [1.0, 0.25]}, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [1.0]\nfocus: mars_cnn\nseed: 3\n")
    sweep_main(["--sweep", str(tmp_path / "tiny.yaml"), "--tune-trials", "2", "--test-trials", "2", "--results", str(tmp_path),
                "--out", "tiny_image", "--scene-cache", str(tmp_path / "scene")])
    assert {p.name for p in (tmp_path / "scene").glob("*.npy")} == {"U1xT2|1.0_tune_cnn_64.npy", "U1xT2|1.0_test_cnn_64.npy"}
    assert set(json.loads((tmp_path / "sim" / "sweeps" / "tiny_image" / "test.json").read_text())["test"]["U1xT2|1.0"]["metrics"]) == {"teleop", "mars_cnn"}
    sweep_main(["--sweep", str(tmp_path / "tiny.yaml"), "--results", str(tmp_path), "--out", "pre", "--scene-cache", str(tmp_path / "scene2"), "--precompute-scene"])
    assert len(list((tmp_path / "scene2").glob("*.npy"))) == 2 and not (tmp_path / "sim" / "sweeps" / "pre" / "test.json").exists()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q -k scene`
Expected: FAIL — `ImportError: cannot import name 'attach_scene'`.

- [ ] **Step 3: Implement `scene_sources`, `attach_scene`, the loop hook and the two arguments**

The `encoder_factory` in `main` is `lambda: SceneEncoder(device=device)`, created on first `vlm_*` need and `close()`d at exit. With `--precompute-scene` the layouts are still drawn exactly as the loop draws them (same code path, no duplication).

- [ ] **Step 4: Run the sweep tests**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/run_sweep.py tests/examples/pi05_mars_cursor/test_sweep.py
git commit -m "MARS: sweeps attach and cache the scene the image task modules read" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: The gate and grid sweep configs, and the README rows

**Files:**
- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image.yaml`, `examples/pi05/mars/cursor/configs/sweeps/grid_image.yaml`
- Modify: `examples/pi05/mars/cursor/README.md:183-193` (two table rows), `:286-300` (task module paragraph: the sources), `:350-396` (training: `--token-source`, `--pooled`)
- Test: `tests/examples/pi05_mars_cursor/test_sweep.py` (extend `test_sweep_configs_load_and_methods_resolve` or add one)

**Interfaces:**
- Consumes: `load_sweep`, `method_arm`, `cell_adapter` (per-task dict or one string resolves per cell).
- Produces:
  - `grid_task_check_image.yaml`: as `grid_task_check.yaml` (`agent: expert_chunk`, `arm: condition_grid.yaml`, `U0`, tasks T2–T5 with the parametric adapters, `noises: [0.0]`, `trials: {tune: 20, test: 500}`, `seed: 820`, `tune: {frs: {n_reverse_steps: [10]}}`) with methods `oracle`, `oracle_task` (`{task: cell}`), `oracle_task_vlm`, `oracle_task_vlm_all`, `oracle_task_cnn`, `oracle_task_cnn_all`, `policy`, `policy_task`, `policy_task_vlm`, `policy_task_vlm_all`, `policy_task_cnn`, `policy_task_cnn_all`; per-task slots name the `image/task_<ctx>_n300_s0_with_<user>_<source>` checkpoints, `_all` slots the one `image/task_all_n1200_s0_<source>`; `focus: oracle_task_vlm`, `against: [oracle_task]`.
  - `grid_image.yaml`: `agent: expert`, `arm: condition_grid.yaml`, the grid's users/tasks T2–T5 (parametric adapters in `tasks.*.adapter`)/`train`/`seed`/`trials`; methods `teleop`, `mars` (`{user: cell, task: cell}`), `mars_vlm`, `mars_vlm_all`, `mars_cnn`, `mars_cnn_all` (each `{user: cell, task: <per-task dict | one name>}`, `tune: *mars`); `focus: mars_vlm`, `against: [mars, mars_cnn]`; a header comment with the part/merge/module-seed commands as `grid_ft.yaml` has.

- [ ] **Step 1: Write the failing test**

```python
# appended to tests/examples/pi05_mars_cursor/test_sweep.py
def test_image_sweep_configs_resolve_every_arm_per_cell():
    check = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image.yaml")
    assert check["focus"] == "oracle_task_vlm" and method_arm(check, "oracle_task_vlm", "U0xT3")["learned_adapter"]["task"] == "image/task_wall_gap_n300_s0_with_gain025x_vlm_image"
    assert method_arm(check, "oracle_task_cnn_all", "U0xT5")["learned_adapter"]["task"] == "image/task_all_n1200_s0_cnn"
    grid = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_image.yaml")
    assert set(grid["methods"]) == {"teleop", "mars", "mars_vlm", "mars_vlm_all", "mars_cnn", "mars_cnn_all"} and set(grid["grid"]["tasks"]) == {"T2", "T3", "T4", "T5"}
    arm = method_arm(grid, "mars_vlm", "U2xT4")
    assert arm["learned_adapter"] == {"user": "user_gain025x_expert_n100_s0", "task": "image/task_ring_goal_n300_s0_with_rot60_vlm_image"}
    assert method_arm(grid, "mars", "U2xT4")["learned_adapter"]["task"] == "task_ring_goal_n300_s0_with_rot60"
    seeded = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_image.yaml", module_seed=2)
    assert method_arm(seeded, "mars_vlm_all", "U1xT2")["learned_adapter"]["task"] == "image/task_all_n1200_s2_vlm_image"
    base = load_sweep(config.CONFIG_DIR / "sweeps" / "grid.yaml")
    assert grid["seed"] == base["seed"] and grid["trials"] == base["trials"] and grid["grid"]["train"] == base["grid"]["train"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q -k image_sweep_configs`
Expected: FAIL — `FileNotFoundError: ... grid_task_check_image.yaml`.

- [ ] **Step 3: Write the two configs and the README changes**

- [ ] **Step 4: Run the sweep tests and the README's config loading test**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_sweep.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/configs/sweeps/grid_task_check_image.yaml examples/pi05/mars/cursor/configs/sweeps/grid_image.yaml examples/pi05/mars/cursor/README.md tests/examples/pi05_mars_cursor/test_sweep.py
git commit -m "MARS: the image task modules' gate and grid sweeps" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: The report: gate verdicts, grid tables, pooled statistic

**Files:**
- Create: `examples/pi05/mars/cursor/report_grid_image.py`
- Test: `tests/examples/pi05_mars_cursor/test_report_grid_image.py` (new)

**Interfaces:**
- Consumes: `results/sim/sweeps/<run>/test.json` layout (`test[cell_key]["metrics"][method]{success, collision}`, `["trials"]["outcome"][method]` list of `"hit"|...`, `["trials"]["side"]`, `["train"]`), `pooled_success` (run_sweep), the join and trial-identity check of `build_figures.grid_data` (reimplemented here for arbitrary runs, not imported from the paper code).
- Produces: CLI `report_grid_image.py --check grid_task_check_image --grid grid_image_s0 [grid_image_s1 ...] --ft grid_ft_s0 [grid_ft_s1 ...] --results <dir> --out <dir>`; functions `gate_table(check: dict, image_arms: list[str]) -> list[dict]` (per context T2–T5 and arm: `hits`, `collision`, `ref_collision`, `pass` = hits ≥ 80 and collision ≤ ref + 10; `hidden_hits`, `ref_hidden_hits`, `flag` = hidden_hits > ref + 10), `join_ft(grid_run: dict, ft_run: dict, arm: str = "ft_mars_user") -> dict` (raises `ValueError` unless `trials.side` and the `teleop` outcomes agree cell for cell), `pooled_over_seeds(runs: list[dict], method: str) -> dict(mean, lo, hi, per_seed)` (per seed the mean over the 16 cells' success; 95% t-CI over seeds, as Fig. 3e), writes `report.md` (gate table, per-cell success per arm, per-context means over users, pooled table) and `report.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/examples/pi05_mars_cursor/test_report_grid_image.py
import json, numpy as np, pytest
from report_grid_image import gate_table, join_ft, pooled_over_seeds, main as report_main

def _cell(methods, hits, side=None, n=10):
    return dict(metrics={m: dict(success=h, collision=0.1 * i) for i, (m, h) in enumerate(zip(methods, hits))},
                trials=dict(outcome={m: ["hit"] * int(h * n) + ["timeout"] * (n - int(h * n)) for m, h in zip(methods, hits)}, side=side or [1] * n), train=False)

def test_gate_rule_and_flag():
    cells = {f"U0xT{k}|0.0": dict(metrics={"oracle": dict(success=0.9, collision=0.5), "oracle_task": dict(success=0.95, collision=0.05),
             "oracle_task_vlm": dict(success=0.85, collision=0.10 if k != 5 else 0.3), "policy_task": dict(success=0.2, collision=0.1),
             "policy_task_vlm": dict(success=0.25 if k != 4 else 0.5, collision=0.1)}) for k in (2, 3, 4, 5)}
    rows = {(r["context"], r["arm"]): r for r in gate_table(dict(test=cells), ["vlm"])}
    assert rows[("T2", "vlm")]["pass"] and not rows[("T5", "vlm")]["pass"] and rows[("T4", "vlm")]["flag"] and not rows[("T2", "vlm")]["flag"]

def test_join_ft_checks_the_trials_then_pools_over_seeds(tmp_path):
    ms = ["teleop", "mars", "mars_vlm"]; grid = dict(test={f"U{u}xT{t}|1.0": _cell(ms, [0.2, 0.7, 0.6 + 0.01 * u]) for u in (1, 2) for t in (2, 3)})
    ft = dict(test={k: _cell(["teleop", "ft_mars_user"], [0.2, 0.4]) for k in grid["test"]})
    joined = join_ft(grid, ft); assert joined["test"]["U1xT2|1.0"]["metrics"]["ft_mars_user"]["success"] == 0.4
    bad = dict(test={k: _cell(["teleop", "ft_mars_user"], [0.3, 0.4]) for k in grid["test"]})
    with pytest.raises(ValueError, match="trials"): join_ft(grid, bad)
    pooled = pooled_over_seeds([joined, joined], "mars_vlm")
    assert pooled["mean"] == pytest.approx(61.5) and len(pooled["per_seed"]) == 2 and pooled["lo"] <= pooled["mean"] <= pooled["hi"]
    sweeps = tmp_path / "sim" / "sweeps"
    for name, run in (("grid_task_check_image", dict(test={})), ("grid_image_s0", grid), ("grid_ft_s0", ft)):
        (sweeps / name).mkdir(parents=True); (sweeps / name / "test.json").write_text(json.dumps(run))
    report_main(["--check", "grid_task_check_image", "--grid", "grid_image_s0", "--ft", "grid_ft_s0", "--results", str(tmp_path), "--out", str(tmp_path / "rep")])
    assert (tmp_path / "rep" / "report.md").exists() and "mars_vlm" in (tmp_path / "rep" / "report.md").read_text()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'report_grid_image'`.

- [ ] **Step 3: Implement `report_grid_image.py`**

An image arm `x` maps to methods `oracle_task_<x>` / `policy_task_<x>` in the check run and `mars_<x>` in the grid run; `gate_table` takes the context from the cell key (`U0xT<k>`). Markdown tables only; no figures.

- [ ] **Step 4: Run the test**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pi05/mars/cursor/report_grid_image.py tests/examples/pi05_mars_cursor/test_report_grid_image.py
git commit -m "MARS: a report for the image task modules' gate and grid" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Seed 0 — train, gate, sweep

**Files:**
- Create (git-ignored, the demo-time convention): `examples/pi05/mars/results/research/image/pipeline.sh`, `jobs_task_s<k>.txt`, a copy of `results/research/demo_time/run_jobs.py`, `pooled_s<k>.yaml`
- Produces: checkpoints under `checkpoints/adapters/image/`, scene caches under `checkpoints/scene/`, runs `results/sim/sweeps/grid_task_check_image`, `grid_image_s0_parts/*`, `grid_image_s0`.

- [ ] **Step 1: Write the job list for seed 0** — 10 lines, one per module, from the repo root, e.g.

```
CUDA_VISIBLE_DEVICES=1 uv run --extra mars --extra pi python examples/pi05/mars/cursor/train/train_adapter.py --kind task --obstacle '{kind: disc, radius: 0.08, along: 0.5, offset: [0.012, 0.024]}' --user-module user_gain025y_expert_n100_s0 --corruption '{scale: [1.0, 0.25]}' --name image/task_disc_large_n300_s0_with_gain025y_vlm_image --token-source vlm_image --n-demos 300 --frozen-noise --seed 0
```

(T3/T4/T5 with their user and corruption; the same four with `--token-source cnn` and `_cnn` names, no `--extra pi`; the two pooled: `--pooled pooled_s0.yaml --name image/task_all_n1200_s0_<source> --steps 80000`, the YAML's four entries = the four per-task lines' obstacle/user_module/corruption.) `pipeline.sh <seed>` modeled on `demo_time/pipeline.sh`: `vlm_*` jobs one per GPU at a time (each loads 14 GB), `cnn` jobs four per GPU; then `run_sweep.py --sweep grid_image --precompute-scene` (GPU, once), then the gate, then the parts, then the merge.

- [ ] **Step 2: Train seed 0's ten modules**

Run: `bash examples/pi05/mars/results/research/image/pipeline.sh 0` (or its training stage alone).
Expected: ten `adapter.pt` under `checkpoints/adapters/image/`; each `train_log.json` ends with finite `val_fm` and `chunk_collision_hidden` below `chunk_collision_policy_hidden`.

- [ ] **Step 3: Run the gate**

Run: `cd examples/pi05/mars/cursor && CUDA_VISIBLE_DEVICES=1 uv run --extra mars --extra pi python run_sweep.py --sweep grid_task_check_image --precompute-scene && uv run --extra mars python run_sweep.py --sweep grid_task_check_image`
Then: `uv run --extra mars python report_grid_image.py --check grid_task_check_image --grid grid_image_s0 --results ../results --out ../results/sim/reports/grid_image` is for later; for the gate alone read `results/sim/sweeps/grid_task_check_image/grid_summary.md`.
Expected: per context, every `oracle_task_<x>` row hits ≥ 80% with collision ≤ `oracle_task` + 10. A failing per-task module goes back to Step 2 with the render ablation (display palette, `--render-size 448`), then the spec's agent-centered fallback — stop and report to the user before building the fallback.

- [ ] **Step 4: Run the grid at seed 0**

Run: `CUDA_VISIBLE_DEVICES=1 uv run --extra mars --extra pi python run_sweep.py --sweep grid_image --precompute-scene`, then the 16 parts `run_sweep.py --sweep grid_image --cells <cell> --out grid_image_s0_parts/<cell>` in the background (two GPUs alternating, as the pipeline does), then `--merge grid_image_s0_parts/* --out grid_image_s0`.
Expected: `results/sim/sweeps/grid_image_s0/test.json` with 16 cells and the six methods; `teleop` outcomes identical to `grid_s0`'s cell for cell (the report's join checks it).

- [ ] **Step 5: Record** — in the pipeline log, the wall time per stage; nothing to commit (results are git-ignored). Post the gate table and the per-context means to the user before starting seeds 1–2.

---

### Task 11: Seeds 1–2, the report, the hand-off

- [ ] **Step 1: Job lists for seeds 1 and 2** — as seed 0's with `_s1`/`_s2` in every name, `--seed k`, and the seed-k user modules (`user_<user>_expert_n100_s<k>`; confirm they exist under `checkpoints/adapters/`, else `hub.py download`).

- [ ] **Step 2: Train and sweep at seed 0's settings**

Run: `bash .../image/pipeline.sh 1` and `... 2` (their parts pass `--module-seed k --fixed-settings ../results/sim/sweeps/grid_image_s0/settings.yaml`; the scene cache is shared).
Expected: `grid_image_s1`, `grid_image_s2` merged.

- [ ] **Step 3: Build the report**

Run: `cd examples/pi05/mars/cursor && uv run --extra mars python report_grid_image.py --check grid_task_check_image --grid grid_image_s0 grid_image_s1 grid_image_s2 --ft grid_ft_s0 grid_ft_s1 grid_ft_s2 --results ../results --out ../results/sim/reports/grid_image`
Expected: `report.md` with the gate table (pass/flag per context and arm), per-cell and per-context success for `mars`, `mars_vlm`, `mars_vlm_all`, `mars_cnn`, `mars_cnn_all`, `ft_mars_user`, and the pooled-over-16-cells table with 95% t-CIs over the three seeds; reference numbers reproduce: `mars` ≈ 73.3%, `ft_mars_user` ≈ 46.0%.

- [ ] **Step 4: Hand off** — give the user the report path and the three numbers that decide the spec's outcome (pooled `mars_vlm`, `mars_vlm_all`, `mars_cnn` against `mars` and `ft_mars_user`), with any FLAG rows called out. Save a memory note with the results and the run names.

---

## Self-review

- **Spec coverage.** Scene tokens as a pluggable source — Task 2; the render and its palette, no goal mark — Task 1; per-layout precomputed features riding on `Layouts`, the sweep cache keyed by layout draw and render spec — Tasks 1, 4, 7; vision tower from `lerobot/pi05_base`, `vlm_prefix` with a fixed prompt without state — Task 3; the pooled arm's data (each context with its own user, `_n1200_s<k>`) — Tasks 6, 8; controls (parametric `mars`, `teleop`, `cnn`, `grid_ft` joined) — Tasks 8, 9; the gate's numeric rule and the goal-hidden flag — Tasks 8, 9, 10; seeds 0 tuned then 1–2 with seed-k user modules — Tasks 10, 11; the deliverable report, Fig. 3 untouched — Task 9; `--extra pi` only for VLM steps, one `LearnedTaskModule` class — Tasks 2, 3, 5; tests stub the VLM — Tasks 3, 5, 7. The agent-centered fallback is a stop-and-report, not a task (the spec triggers it only from a gate failure). `convert_checkpoints.py` needs no change: it rebuilds `LearnedTaskModule(**config)` and the config carries the source (covered by Task 2's round trip).
- **Step scan.** Every code step names its file, signature and pinned values; bodies are left to the implementer except the three places a choice is not determined (concat's mixed-scene rule, the render's drawing order, the pooled trainer's round-robin), which are stated.
- **Type consistency.** `Layouts.scene: dict[str, np.ndarray] | None` (Task 1) is what `observation`/`DemoSampler` (Task 4) tensorize into `obs["scene"]: dict[str, Tensor]`, which `LearnedTaskModule.tokens` (Task 2) reads by `token_source`; `scene_for` (Task 3) returns the per-source array that `attach_scene` (Task 7) and `expert_splits` (Task 5) store under that source; `scene_spec` (Task 3) produces exactly the keyword arguments Task 2's `__init__` takes; `image_size` is the cache key's size (Task 7) and `scene_spec`'s (Task 3).
- **Review Focus.** Items 1–2 → Task 1 tests; 3–4 → Task 2 tests; 5 → Task 7's first test.
- **Proportion.** The plan is about twice the spec's length; code blocks are tests and commands, not implementations.
