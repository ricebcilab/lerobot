# MARS image task module, round 3 (two views, demo-count scan) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the image task module the robot's two camera views (overhead and cursor-centred), encoded as the cursor moves, with robot-valid augmentation, and scan 300 / 1,000 / 3,000 scripted demonstrations to find how many it needs to be level with the parametric module.

**Architecture:** A new token source `vlm_views`: the env renders two 224 px views at the cursor's position, pi0.5's vision tokens of both (512 tokens) feed `LearnedTaskModule` with a learned position per token and nothing derived from the cursor. The trainer encodes a frame every 2 planned steps into a frame table (tokens reduced to 256 dims by a PCA the module carries), the sampler reads each sample's own frame, and rollouts encode at every observation through a scene provider on the controller. Small sweeps and three rules in the report (recipe choice, width check, scan) run the experiment.

**Tech Stack:** Python 3.12, PyTorch, numpy (the affine: the trainer's own bilinear resample `move_pictures`, at lerobot's default `RandomAffine` magnitudes), Pillow, PyYAML, pytest; `uv run --extra mars` (`--extra pi` for the VLM, `--extra test` for pytest, `--extra dev` for pre-commit), `HF_HUB_OFFLINE=1`.

**Spec:** `docs/superpowers/specs/2026-11-05-mars-image-two-view-demo-scan-design.md` (follows `2026-11-04-mars-image-round2-teacher-relative-design.md`).

## Global Constraints

- Python commands run through `uv run --extra mars` (add `--extra pi --extra test` where the VLM or pytest is needed); real-VLM work needs `HF_HUB_OFFLINE=1` and a `CUDA_VISIBLE_DEVICES`.
- The existing sources (`discs`, `vlm_image`, `vlm_prefix`, `cnn`) stay bitwise unchanged: module configs, state dicts, tokens, the trainer's draws. `RENDER_PALETTE` and `render_layouts`' pixels do not change (the scene cache hashes them). Round-1 and round-2 checkpoints load and behave as before; every existing test stays green.
- A run without the new flags consumes the numpy and torch random streams exactly as before. The views pipeline (frame sample, affine) draws only from its own `np.random.default_rng([seed, 7])`.
- Nothing derived from the cursor's position enters the scene tokens except through the rendered pictures; `token_geometry: relative` is refused for `vlm_views`; no obstacle parameter enters an image module's input; the scan's arms use scripted demonstrations only.
- Values (spec): base recipe `wrist_span` 0.5, token masking 0.3, affine probability 0.5 at 5° and 5% (`VIEW_AFFINE_DEGREES = 5.0`, `VIEW_AFFINE_TRANSLATE = 0.05`), `view_stride` 2, `scene_rank` 256, validation 300 demonstrations, 20,000 steps, seed 0. Rules: level = hits ≥ parametric − 3 points and collisions ≤ parametric + 3 points on both contexts; a recipe change is adopted when its mean hits beat the base's by more than 3 points; the reduced run may trail the full-width run by at most 5 points. Selection sweeps use seed 821, scan sweeps seed 820.
- Checkpoint names: stage A `image/task_<ctx>_n300_s0_with_<user>_views_<recipe>` with recipe in `base`, `span1`, `nomask`, `noaffine`, `full`; the scan `image/task_<ctx>_n<N>_s0_with_<user>_views`. Contexts: `disc_large` with `gain025y` (T2), `ring_goal` with `rot60` (T4).
- `lerobot.envs` is mypy-strict: new functions there are fully annotated. Pre-commit (ruff, typos, bandit, mypy) passes on every touched file; comparisons of a `token_*` variable with a string literal carry `# nosec B105` as the file already does.
- Results are never deleted; superseded results move to an `old/` folder. No push, no merge, without the user's go. Commits end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. A rollout batch in which some trials have finished: their rows get no pictures and zero tokens, and the running rows' tokens equal those of a batch of only those rows (Task 6 test).
2. Two observations built at one cursor position (the goal-hidden rows build two per replan) encode once; a moved cursor encodes again (Task 6 test).
3. A demonstration of odd length or of length 0: `ceil(n / 2)` frames, none for an empty one, and no sample reads another demonstration's frame (Task 4 tests).
4. A reduced module fed full-width tokens (a rollout) and stored reduced tokens (training) gives identical tokens; any other width raises (Task 2 test).
5. One sweep holding two view modules with different spans (stage A does this): each is rendered at its own span and nothing is written to the scene cache (Task 6 test).

---

### Task 1: The two views in the cursor env

**Files:**

- Modify: `src/lerobot/envs/cursor.py` (next to `render_layouts`)
- Test: `tests/examples/pi05_mars_cursor/test_scene.py`

**Interfaces:**

- Consumes: `Layouts`, `render_layouts`, `RENDER_PALETTE`, `RENDER_TARGET_RADIUS`, `RENDER_SUPERSAMPLE`, `RENDER_MIN_RADIUS`, `SCENE_SIZE`.
- Produces:
  - `RENDER_CURSOR: tuple[int, int, int] = (230, 60, 30)` and `RENDER_CURSOR_RADIUS: float = 0.02` (screens). Not a `RENDER_PALETTE` entry.
  - `VIEWS: tuple[str, ...] = ("overhead", "wrist")`.
  - `render_views(layouts: Layouts, pos: np.ndarray, wrist_span: float, size: int = SCENE_SIZE, target_radius: float = RENDER_TARGET_RADIUS, supersample: int = RENDER_SUPERSAMPLE) -> np.ndarray`: uint8 `[B, 2, 3, size, size]`. View 0 is the unit screen exactly as `render_layouts` draws it plus the cursor mark at `pos[b]`. View 1 is the square window of side `wrist_span` centred on `pos[b]`: a screen point `p` lands at pixel `(p - pos + wrist_span / 2) / wrist_span * size`, every radius is divided by `wrist_span`, what lies outside the screen is background, the mark sits at the centre. The mark is drawn last. `ValueError` for `wrist_span <= 0` or `pos.shape != (B, 2)`.

- [ ] **Step 1: Write the failing tests** in `tests/examples/pi05_mars_cursor/test_scene.py` (import `RENDER_CURSOR`, `RENDER_CURSOR_RADIUS`, `VIEWS`, `render_views` from `cursor_sa.env`):

```python
def _near(pixel, colour, tol=3):
    return np.abs(pixel.astype(int) - np.array(colour)).max() <= tol


def test_views_draw_the_cursor_and_centre_the_wrist_window_on_it():
    task = TaskConfig(min_reach=0.35, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5}))
    lay = sample_layouts(np.random.default_rng(0), 3, 4, task)
    disc = lay.obstacles[:, 0, :2]
    pos = disc + np.where(disc > 0.5, -1.0, 1.0) * np.array([0.1, 0.05])   # the cursor beside the disc, on the screen-centre side
    views = render_views(lay, pos, 0.5)
    assert VIEWS == ("overhead", "wrist") and views.shape == (3, 2, 3, 224, 224) and views.dtype == np.uint8
    base = render_layouts(lay)
    reach = int(np.ceil(RENDER_CURSOR_RADIUS * 224)) + 5
    for b in range(3):
        cx, cy = (pos[b] * 224).astype(int)
        assert _near(views[b, 0, :, cy, cx], RENDER_CURSOR) and _near(views[b, 1, :, 112, 112], RENDER_CURSOR)
        far = np.ones((224, 224), bool)
        far[max(cy - reach, 0):cy + reach + 1, max(cx - reach, 0):cx + reach + 1] = False
        assert np.array_equal(views[b, 0][:, far], base[b][:, far])   # away from the mark the overhead view is render_layouts'
        col, row = ((disc[b] - pos[b]) / 0.5 + 0.5) * 224             # where the window puts the disc's centre
        assert _near(views[b, 1, :, int(row), int(col)], RENDER_PALETTE["obstacle"])


def test_a_wrist_window_as_wide_as_the_screen_at_its_centre_is_the_overhead_view():
    lay = sample_layouts(np.random.default_rng(1), 2, 4, TaskConfig(min_reach=0.35, max_reach=0.5))
    views = render_views(lay, np.full((2, 2), 0.5), 1.0)
    assert np.array_equal(views[:, 1], views[:, 0])
    corner = render_views(lay, np.full((2, 2), 0.05), 0.5)             # the window reaches 0.2 outside the screen
    assert (corner[:, 1, :, 0, 0] == 255).all()
    for bad in (lambda: render_views(lay, np.full((2, 2), 0.5), 0.0), lambda: render_views(lay, np.full((1, 2), 0.5), 0.5)):
        with pytest.raises(ValueError):
            bad()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py -k "views or wrist" -v`
Expected: FAIL (ImportError: `render_views`).

- [ ] **Step 3: Implement** the constants and `render_views` in `src/lerobot/envs/cursor.py`. One drawing routine parameterised by the window (origin, side) serves both views; `render_layouts`' output must stay pixel-identical (leave its body alone or express it through the routine with origin 0, side 1, no mark).

- [ ] **Step 4: Run** the two tests and the file's existing ones. Expected: PASS; `test_render_draws_discs_and_targets_and_never_the_goal` and `test_render_keeps_the_wall_visible` still pass.

- [ ] **Step 5: Commit** — `MARS round 3: the cursor env draws the overhead and wrist views`.

---

### Task 2: The `vlm_views` token source in `LearnedTaskModule`

**Files:**

- Modify: `src/lerobot/policies/steering/configuration_steering.py` (`SCENE_SOURCES`, `scene_spec`)
- Modify: `src/lerobot/policies/steering/modeling_steering.py` (`LearnedTaskModule`, `reduce_tokens`, class docstring)
- Modify: `src/lerobot/policies/steering/__init__.py` (export `reduce_tokens`, `VLM_VIEWS_TOKENS`)
- Test: `tests/policies/steering/test_scene_tokens.py`

**Interfaces:**

- Consumes: `LearnedTaskModule(horizon, d, tdim, heads, layers, scale, n_dims, token_source, scene_dim, scene_tokens, image_size, token_geometry)`; `tokens(obs, b) -> (tokens, pad)`; `scene_in`, `scene_pos`; `VLM_SCENE_TOKENS`, `VLM_SCENE_DIM`.
- Produces:
  - `SCENE_SOURCES = ("discs", "vlm_image", "vlm_prefix", "cnn", "vlm_views")`; `VLM_VIEWS_TOKENS = 2 * VLM_SCENE_TOKENS`.
  - `scene_spec(source, size=SCENE_SIZE, *, wrist_span: float | None = None, scene_rank: int | None = None) -> dict`: for `vlm_views` exactly `{"token_source": "vlm_views", "scene_dim": 2048, "scene_tokens": 512, "image_size": size, "wrist_span": wrist_span, "scene_rank": scene_rank}`; `ValueError` without a positive `wrist_span`, for a `scene_rank` outside `1..2047`, and for either keyword with another source. The other sources' dicts are unchanged.
  - `LearnedTaskModule(..., wrist_span: float | None = None, scene_rank: int | None = None)`; attributes `wrist_span`, `scene_rank`. `vlm_views` needs `wrist_span > 0` (`ValueError` naming `wrist_span`); `scene_rank` with another source raises (naming `scene_rank`); `relative` geometry stays refused. Config: `wrist_span` always for `vlm_views`, `scene_rank` only when set. `scene_in` reads `scene_rank or scene_dim`.
  - With `scene_rank`: persistent buffers `scene_mean` `(scene_dim,)` and `scene_basis` `(scene_dim, scene_rank)`, zeros until `set_reduction(mean: torch.Tensor, basis: torch.Tensor) -> None` copies them in (`ValueError` on a wrong shape).
  - `reduce_tokens(tokens: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor`: `((tokens.float() - mean) @ basis)` as float16. `tokens()` applies it when a reduced module is handed width `scene_dim`, takes width `scene_rank` as is, and raises `ValueError` naming `width` otherwise.
  - `obs["scene_keep"]` (bool `(B, scene_tokens)`, optional, any image source): the tokens the attention may see. `tokens()` then returns the padding mask `cat([False, ~keep])` (the null token is never hidden); a wrong shape raises `ValueError` naming `scene_keep`.

- [ ] **Step 1: Write the failing tests** in `tests/policies/steering/test_scene_tokens.py` (import `reduce_tokens`, `scene_spec`):

```python
def _views(rank=None, **kw):
    return LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=32, scene_tokens=8, wrist_span=0.5, scene_rank=rank, **kw)


def _reduction(dim=32, rank=4):
    return torch.randn(dim), torch.linalg.qr(torch.randn(dim, rank))[0]


def test_a_views_module_keeps_its_span_and_reduction_through_a_checkpoint(tmp_path):
    assert "vlm_views" in SCENE_SOURCES
    full, red = _views(), _views(rank=4)
    assert full.config["wrist_span"] == 0.5 and "scene_rank" not in full.config and "scene_mean" not in full.state_dict()
    assert red.config["scene_rank"] == 4 and red.scene_in.in_features == 4 and full.scene_in.in_features == 32
    assert red.state_dict()["scene_basis"].shape == (32, 4) and red.state_dict()["scene_pos"].shape[0] == 8
    mean, basis = _reduction()
    red.set_reduction(mean, basis)
    save_adapter_module(red, tmp_path / "v.pt", "task")
    back, kind, _ = load_adapter_module(tmp_path / "v.pt")
    assert kind == "task" and back.wrist_span == 0.5 and back.scene_rank == 4
    torch.testing.assert_close(back.scene_basis, basis)
    torch.testing.assert_close(back.scene_mean, mean)
    with pytest.raises(ValueError):
        red.set_reduction(mean, basis[:, :3])
    assert set(_module("vlm_image").config) == {"horizon", "d", "tdim", "heads", "layers", "scale", "n_dims",
                                               "token_source", "scene_dim", "scene_tokens", "image_size"}   # no new keys


def test_a_reduced_module_reads_full_width_and_reduced_tokens_alike():
    red = _views(rank=4)
    red.set_reduction(*_reduction())
    raw, obs = torch.randn(3, 8, 32).to(torch.float16), dict(pos=torch.rand(3, 2))
    stored = reduce_tokens(raw, red.scene_mean, red.scene_basis)
    assert stored.dtype == torch.float16 and stored.shape == (3, 8, 4)
    a = red.tokens({**obs, "scene": {"vlm_views": raw}}, 3)[0]
    b = red.tokens({**obs, "scene": {"vlm_views": stored}}, 3)[0]
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="width"):
        red.tokens({**obs, "scene": {"vlm_views": torch.randn(3, 8, 5)}}, 3)


def test_scene_keep_hides_tokens_from_the_attention_and_never_the_null_token():
    m = _views()
    torch.nn.init.normal_(m.out.weight)                                   # a fresh head is zero: the output would not depend on tokens
    scene = torch.randn(2, 8, 32)
    obs = dict(pos=torch.rand(2, 2), scene={"vlm_views": scene})
    keep = torch.tensor([[True] * 4 + [False] * 4, [False] * 8])
    _, pad = m.tokens({**obs, "scene_keep": keep}, 2)
    assert pad.shape == (2, 9) and not pad[:, 0].any() and torch.equal(pad[:, 1:], ~keep)
    assert m.tokens(obs, 2)[1] is None
    x, v, t = torch.randn(2, 16, 2), torch.randn(2, 16, 2), torch.rand(2)
    changed = scene.clone()
    changed[0, 4:] += 1.0                                                 # only hidden tokens change
    a = m(x, t, v, {**obs, "scene_keep": keep})
    b = m(x, t, v, {"pos": obs["pos"], "scene": {"vlm_views": changed}, "scene_keep": keep})
    torch.testing.assert_close(a[0][0], b[0][0])
    assert not torch.allclose(a[0][0], m(x, t, v, obs)[0][0])              # and hiding them matters
    with pytest.raises(ValueError, match="scene_keep"):
        m.tokens({**obs, "scene_keep": keep[:, :5]}, 2)


def test_views_spec_and_refusals():
    assert scene_spec("vlm_views", 224, wrist_span=0.5, scene_rank=256) == {
        "token_source": "vlm_views", "scene_dim": 2048, "scene_tokens": 512, "image_size": 224, "wrist_span": 0.5, "scene_rank": 256}
    assert scene_spec("vlm_views", 224, wrist_span=1.0)["scene_rank"] is None
    assert set(scene_spec("vlm_image")) == {"token_source", "scene_dim", "scene_tokens", "image_size"}
    for bad in ({}, {"wrist_span": 0.0}, {"wrist_span": 0.5, "scene_rank": 2048}, {"wrist_span": 0.5, "scene_rank": 0}):
        with pytest.raises(ValueError):
            scene_spec("vlm_views", 224, **bad)
    with pytest.raises(ValueError):
        scene_spec("vlm_image", 224, wrist_span=0.5)
    with pytest.raises(ValueError, match="relative"):
        _views(token_geometry="relative")
    with pytest.raises(ValueError, match="wrist_span"):
        LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=32, scene_tokens=8)
    with pytest.raises(ValueError, match="scene_rank"):
        LearnedTaskModule(horizon=16, token_source="vlm_image", scene_dim=32, scene_tokens=16, scene_rank=4)
```

Also add `("vlm_views", "absolute")` to the file's parametrized bound-equals-unbound test (its scene tensor for this source is `(B, 8, 32)` with the module from `_views()`).

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/policies/steering/test_scene_tokens.py -v`. Expected: the new tests FAIL (`vlm_views` not a source), the old ones pass.

- [ ] **Step 3: Implement** in the two steering files and export from `__init__.py`. The new constructor branches come after the existing ones, so the existing sources' parameter creation order (and with it their initial weights under a seed) does not move.

- [ ] **Step 4: Run** `uv run --extra mars --extra test pytest tests/policies/steering -v`. Expected: PASS, including the steering equivalence tests.

- [ ] **Step 5: Commit** — `MARS round 3: a two-view token source with a carried token reduction and token masking`.

---

### Task 3: Encoding views and fitting the token reduction

**Files:**

- Modify: `src/lerobot/policies/steering/modeling_steering.py` (`SceneEncoder`, next to `scene_for`)
- Modify: `src/lerobot/policies/steering/__init__.py` (export `encode_views`, `fit_token_reduction`)
- Test: `tests/policies/steering/test_scene_encoder.py`

**Interfaces:**

- Consumes: `SceneEncoder.image_tokens`, `_pixels`, `embed_image`; `reduce_tokens` (Task 2).
- Produces:
  - `SceneEncoder.encode(self, images: np.ndarray, batch: int = 256) -> torch.Tensor`: uint8 `[B, 3, S, S]` to float16 `[B, 256, 2048]` on the encoder's device. `image_tokens` returns the same values as before (it becomes `encode(images, batch).cpu().numpy()`).
  - `encode_views(images: np.ndarray, encoder, batch: int = 256) -> torch.Tensor`: uint8 `[B, V, 3, S, S]` to float16 `[B, V * 256, 2048]` on the encoder's device, view 0's tokens first. `encoder` is anything with `encode`.
  - `fit_token_reduction(batches: Iterable[torch.Tensor], rank: int) -> tuple[torch.Tensor, torch.Tensor, float]`: token batches `[..., W]` (leading dims flattened, sums kept in float64) to `mean (W,)` float32, `basis (W, rank)` float32 with orthonormal columns, the largest-variance direction first, and the share of variance kept. `ValueError` unless `0 < rank < W`.

- [ ] **Step 1: Write the failing tests** in `tests/policies/steering/test_scene_encoder.py` (import `encode_views`, `fit_token_reduction`):

```python
class _PatchEncoder:
    """Tokens that depend on the picture: token j's first three dims are patch j's mean colour / 255."""

    def encode(self, images, batch=256):
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.nn.functional.pad(pooled, (0, 2045)).to(torch.float16)


def test_encode_views_keeps_the_view_order():
    images = np.zeros((2, 2, 3, 32, 32), np.uint8)
    images[:, 1] = 255
    tokens = encode_views(images, _PatchEncoder())
    assert tokens.shape == (2, 512, 2048) and tokens.dtype == torch.float16
    assert float(tokens[:, :256, :3].max()) == 0.0 and float(tokens[:, 256:, :3].min()) == 1.0


def test_token_reduction_keeps_the_top_directions():
    g = torch.Generator().manual_seed(0)
    true = torch.linalg.qr(torch.randn(16, 3, generator=g))[0]
    x = (torch.randn(4000, 3, generator=g) * torch.tensor([5.0, 2.0, 1.0])) @ true.T + 0.7
    mean, basis, kept = fit_token_reduction(iter(x.split(500)), 3)
    assert mean.shape == (16,) and basis.shape == (16, 3) and mean.dtype == basis.dtype == torch.float32 and kept > 0.999
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5)
    torch.testing.assert_close(mean, x.mean(0), atol=1e-4)
    assert float(((x - mean) @ basis @ basis.T + mean - x).abs().max()) < 1e-3
    assert abs(float(basis[:, 0] @ true[:, 0])) > 0.99
    assert 0.9 < fit_token_reduction(iter(x.split(500)), 2)[2] < 0.999                    # 29 of 30 parts of the variance
    torch.testing.assert_close(fit_token_reduction(iter(x.reshape(8, 500, 16)), 3)[0], mean, atol=1e-5)
    for bad in (16, 0):
        with pytest.raises(ValueError):
            fit_token_reduction(iter(x.split(500)), bad)
```

In the file's real-pi0.5 test (`test_real_pi05_image_tokens_have_the_documented_shape`, skipped without the weights) add: `encoder.encode(images).cpu().numpy()` equals `encoder.image_tokens(images)` exactly.

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/policies/steering/test_scene_encoder.py -v`. Expected: ImportError.

- [ ] **Step 3: Implement** the three functions. `fit_token_reduction` accumulates the sum and the `W x W` second moment on the first batch's device and takes the top eigenvectors of the covariance.

- [ ] **Step 4: Run** the file. Expected: PASS (the real-pi0.5 test passes or skips as before).

- [ ] **Step 5: Commit** — `MARS round 3: view encoding on the encoder's device and a PCA token reduction`.

---

### Task 4: Per-frame scene lookup in the demo sampler

**Files:**

- Modify: `examples/pi05/mars/cursor/cursor_sa/demos.py` (`DemoSampler`, new `frame_steps`)
- Test: `tests/examples/pi05_mars_cursor/test_scene.py`

**Interfaces:**

- Consumes: `DemoSampler(layouts, actions, lengths, horizon, hist, device)`, `sample(n, goal_dropout, indexed)`.
- Produces:
  - `frame_steps(lengths, stride) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `(demo [F], step [F], start [B])`, int64. Demonstration `i` has frames at steps `0, stride, ... < lengths[i]`, stored from row `start[i]`; an empty demonstration has none.
  - `DemoSampler(..., frames=None)` with `frames = {source: (tokens, start, stride)}`: `tokens` a tensor `[F, ...]` on any device, `start` `[B]`. With frames, `sample` floors every `k` to a multiple of the stride before anything is read at `k`, and `obs["scene"][source] = tokens[start[i] + k // stride]` on the sampler's device; sources in `layouts.scene` stay per layout. Two frame sources with different strides raise `ValueError`. Without frames nothing changes, including the random numbers drawn.

- [ ] **Step 1: Write the failing tests** in `tests/examples/pi05_mars_cursor/test_scene.py` (import `frame_steps`):

```python
def test_frame_steps_cover_every_demonstration_at_the_stride():
    demo, step, start = frame_steps(np.array([5, 0, 4, 1]), 2)
    assert demo.tolist() == [0, 0, 0, 2, 2, 3] and step.tolist() == [0, 2, 4, 0, 2, 0] and start.tolist() == [0, 3, 3, 5]
    assert frame_steps(np.array([3]), 1)[1].tolist() == [0, 1, 2]


def test_the_sampler_reads_each_samples_own_frame():
    rng, task, policy = np.random.default_rng(0), TaskConfig(min_reach=0.35, max_reach=0.5), FlowPolicy()
    lay = sample_layouts(rng, 6, 4, task)
    actions, lengths = make_demos(rng, lay, task)
    demo, step, start = frame_steps(lengths, 2)
    tokens = torch.stack([torch.as_tensor(demo), torch.as_tensor(step)], -1).float()[:, None]   # each frame names its demo and step
    sampler = DemoSampler(lay, actions, lengths, policy.horizon, policy.hist, "cpu", frames={"vlm_views": (tokens, start, 2)})
    obs, _, i, k = sampler.sample(512, indexed=True)
    assert (k % 2 == 0).all() and (k < sampler.lengths[i]).all() and set(k.tolist()) >= {0, 2, 4}
    got = obs["scene"]["vlm_views"][:, 0]
    assert torch.equal(got[:, 0].long(), i) and torch.equal(got[:, 1].long(), k)
    torch.testing.assert_close(obs["pos"], sampler.pos[i, k])
    plain = DemoSampler(lay, actions, lengths, policy.horizon, policy.hist, "cpu")
    assert "scene" not in plain.sample(8)[0]
    torch.manual_seed(5)
    a = plain.sample(64, indexed=True)
    torch.manual_seed(5)
    b = DemoSampler(lay, actions, lengths, policy.horizon, policy.hist, "cpu", frames=None).sample(64, indexed=True)
    assert torch.equal(a[3], b[3]) and (a[3] % 2 == 1).any()                                   # no frames: every step, the same draws
    with pytest.raises(ValueError):
        DemoSampler(lay, actions, lengths, policy.horizon, policy.hist, "cpu",
                    frames={"a": (tokens, start, 2), "b": (tokens, start, 4)})
```

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py -k "frame or sampler" -v`. Expected: ImportError.

- [ ] **Step 3: Implement** `frame_steps` and the `frames` argument in `cursor_sa/demos.py`.

- [ ] **Step 4: Run** the file. Expected: PASS, with `test_observation_and_sampler_carry_the_scene` unchanged.

- [ ] **Step 5: Commit** — `MARS round 3: the demo sampler reads each sample's own frame`.

---

### Task 5: The trainer's two-view data path, augmentation and flags

**Files:**

- Modify: `examples/pi05/mars/cursor/train/train_adapter.py` (docstring paragraph VIEWS, `move_pictures`, `view_frames`, `ExpertData`, `expert_splits`, `train_task`, `save`, `parse_args`)
- Test: `tests/examples/pi05_mars_cursor/test_training.py`

**Interfaces:**

- Consumes: `render_views` (Task 1); `scene_spec(..., wrist_span, scene_rank)`, `set_reduction`, `obs["scene_keep"]`, `reduce_tokens` (Task 2); `SceneEncoder.encode`, `encode_views`, `fit_token_reduction` (Task 3); `frame_steps`, `DemoSampler(frames=)` (Task 4).
- Produces:
  - Flags: `--token-source vlm_views`; `--wrist-span` (float, 0.5); `--view-stride` (int, 2); `--scene-rank` (int, 256; 0 = full width); `--view-affine` (float, 0.0: the probability a stored training picture is moved); `--token-drop` (float, 0.0); `--render-workers` (int, 16; 0 renders in-process).
  - Refusals (`p.error`, each before any file is opened): `vlm_views` with `--token-geometry relative`, with `--teacher`, with `--pooled`; `--token-drop > 0` or `--view-affine > 0` with `--token-source discs`; `--view-affine > 0` with a source other than `vlm_views`; `--wrist-span <= 0`; `--view-stride < 1`; `--scene-rank` outside `0..2047`; `--token-drop` outside `[0, 1)`; `--view-affine` outside `[0, 1]`.
  - Constants `VIEW_AFFINE_DEGREES = 5.0`, `VIEW_AFFINE_TRANSLATE = 0.05`, `REDUCTION_SAMPLE = 2000` (frames the reduction is fitted on), `VIEW_CHUNK = 256` (frames rendered and encoded at a time), `VIEW_GPU_BYTES = 12 * 2**30` (a split's tokens above this stay in RAM).
  - `move_pictures(images: np.ndarray, rng: np.random.Generator, p: float) -> tuple[np.ndarray, np.ndarray]`: uint8 `[N, 3, S, S]`; each picture independently, with probability `p`, is rotated about its centre by an angle in ±5° and shifted by up to 5% of `S` per axis (one uniform draw each), bilinear, the uncovered border filled with `RENDER_PALETTE["bg"]`. Returns the pictures and the bool mask of the moved ones. (Background fill, not torchvision's default black: an obstacle here is a dark disc.)
  - `view_frames(args, layouts, actions, lengths, encoder, device, reduction=None, affine_rng=None) -> tuple[torch.Tensor, np.ndarray, dict]`: the frame table of one split. Frames at `frame_steps(lengths, args.view_stride)`, each rendered by `render_views` at the demonstration's position at that step; with `affine_rng`, `move_pictures(..., args.view_affine)` on every picture (the two views independently); `encode_views`; with `args.scene_rank`, reduced by `reduction` or, when it is `None`, by `fit_token_reduction` on up to `REDUCTION_SAMPLE` frames chosen with `affine_rng` (or a `default_rng([args.seed, 7])` when there is none). Returns float16 tokens `[F, 512, rank or 2048]` (on `device` when at most `VIEW_GPU_BYTES`, else on the CPU), `start`, and `info = dict(frames, moved, kept, reduction, seconds)` (`reduction` the `(mean, basis)` used or `None`, `kept` the variance share when fitted here).
  - `ExpertData(..., frames=None)` passes `frames` to its `DemoSampler`.
  - `expert_splits` for `vlm_views`: the demonstrations are drawn exactly as for any source; then the train split's `view_frames` fits the reduction and moves pictures with `np.random.default_rng([args.seed, 7])`, and the val split reuses that reduction with no moving. It prints the frames, the pictures moved and the variance kept, and sets `args.view_info = {"train": {...}, "val": {...}}` (each `info` without `reduction`) and `args.view_reduction`.
  - `train_task`: the module is built from `scene_spec(args.token_source, args.render_size, wrist_span=args.wrist_span, scene_rank=args.scene_rank or None)` for `vlm_views` and gets `set_reduction(*args.view_reduction)` when reduced. With `--token-drop p > 0`, `batch_loss` gives `task_loss` the observation plus `scene_keep = torch.rand(batch, task.scene_tokens, device=device) >= p`; the field term (`reverse_flow`, `field_loss`) and validation get the observation without it.
  - Checkpoint meta gains `wrist_span`, `view_stride`, `scene_rank`, `view_affine`, `token_drop`, `views` (`args.view_info`, `None` for other sources).

- [ ] **Step 1: Write the failing tests** in `tests/examples/pi05_mars_cursor/test_training.py`:

```python
class _ViewStub:
    """Stands in for SceneEncoder: token j's first three dims are patch j's mean colour / 255."""

    def __init__(self, *a, **k):
        pass

    def encode(self, images, batch=256):
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.nn.functional.pad(pooled, (0, 2045)).to(torch.float16)

    def close(self):
        pass


VIEW_TASK = ["--kind", "task", "--obstacle", "{kind: disc, radius: 0.08, along: 0.5}", "--user-module", "u", "--corruption", "{rotation_deg: 60}"]
VIEW_FLAGS = ["--token-source", "vlm_views", "--render-size", "56", "--scene-rank", "2", "--view-affine", "0.5", "--token-drop", "0.3",
         "--frozen-noise", "--render-workers", "0"]


def test_move_pictures_moves_a_share_and_fills_with_the_background():
    images = np.full((200, 3, 32, 32), 255, np.uint8)
    images[:, :, 12:20, 12:20] = 0                                         # a dark square on the picture's centre
    out, moved = train_adapter.move_pictures(images, np.random.default_rng(0), 0.5)
    assert out.shape == images.shape and out.dtype == np.uint8 and 60 < moved.sum() < 140
    assert np.array_equal(out[~moved], images[~moved]) and not np.array_equal(out[moved], images[moved])
    assert (out[:, :, 0, 0] == 255).all() and (out[:, :, -1, -1] == 255).all()   # the uncovered border is background
    ys, xs = np.mgrid[:32, :32]
    for img in out[moved][:20]:
        w = 255.0 - img[0]
        assert abs((w * ys).sum() / w.sum() - 15.5) <= 2.2 and abs((w * xs).sum() / w.sum() - 15.5) <= 2.2   # 5% of 32 px, and some blur
    again, moved2 = train_adapter.move_pictures(images, np.random.default_rng(0), 0.5)
    assert np.array_equal(again, out) and np.array_equal(moved2, moved)
    assert not train_adapter.move_pictures(images, np.random.default_rng(0), 0.0)[1].any()


def test_train_a_two_view_task_module_on_the_scripted_draws(ckpt_dir, monkeypatch):
    monkeypatch.setattr(train_adapter, "SceneEncoder", _ViewStub)
    monkeypatch.setattr(train_adapter, "VIEW_CHUNK", 32)
    monkeypatch.setattr(train_adapter, "REDUCTION_SAMPLE", 64)
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    splits = _record_splits(monkeypatch)
    train_adapter.train(train_adapter.parse_args([*VIEW_TASK, "--name", "t_views", *VIEW_FLAGS, *COMMON]))
    (views,) = splits
    ckpt = torch.load(ckpt_dir / "t_views" / "adapter.pt")
    cfg, meta = ckpt["config"], ckpt["meta"]
    assert (cfg["token_source"], cfg["wrist_span"], cfg["scene_rank"], cfg["scene_tokens"]) == ("vlm_views", 0.5, 2, 512)
    assert ckpt["state_dict"]["scene_basis"].abs().sum() > 0               # the fitted reduction, not the zeros it starts as
    assert (meta["view_stride"], meta["token_drop"], meta["view_affine"], meta["wrist_span"], meta["scene_rank"]) == (2, 0.3, 0.5, 0.5, 2)
    lengths = views["train"].sampler.lengths.cpu().numpy()
    info = meta["views"]
    assert info["train"]["frames"] == int(np.ceil(lengths / 2).sum()) and 0 < info["train"]["moved"] < 2 * info["train"]["frames"]
    assert info["val"]["moved"] == 0 and 0 < info["train"]["kept"] <= 1 and info["val"]["kept"] is None
    assert views["train"].sampler.sample(4)[0]["scene"]["vlm_views"].shape == (4, 512, 2)
    log = json.loads((ckpt_dir / "t_views" / "train_log.json").read_text())["log"]
    assert {"val_fm", "chunk_collision_hidden", "collide_composed"} <= set(log[-1])
    assert ReversalAdapter.load(ckpt_dir / "t_views" / "adapter.pt").task.wrist_span == 0.5
    splits.clear()                                                         # the same seed with another source: the same demonstrations
    train_adapter.train(train_adapter.parse_args([*VIEW_TASK, "--name", "t_cnn", "--token-source", "cnn", "--render-size", "64", "--frozen-noise", *COMMON]))
    (cnn,) = splits
    for split in ("train", "val"):
        for field in ("obstacles", "targets", "pos", "lengths"):
            torch.testing.assert_close(getattr(views[split].sampler, field), getattr(cnn[split].sampler, field))
    torch.testing.assert_close(views["train"].noise, cnn["train"].noise)


def test_token_masking_reaches_the_flow_matching_loss_only(ckpt_dir, monkeypatch):
    monkeypatch.setattr(train_adapter, "SceneEncoder", _ViewStub)
    monkeypatch.setattr(train_adapter, "VIEW_CHUNK", 32)
    monkeypatch.setattr(train_adapter, "REDUCTION_SAMPLE", 64)
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    seen, real_fm, real_field = {"fm": [], "field": []}, train_adapter.task_loss, train_adapter.field_loss

    def fm(policy, task, obs, *a):
        seen["fm"].append(obs.get("scene_keep"))
        return real_fm(policy, task, obs, *a)

    def field(policy, user, task, free, *a):
        seen["field"].append("scene_keep" in free)
        return real_field(policy, user, task, free, *a)

    monkeypatch.setattr(train_adapter, "task_loss", fm)
    monkeypatch.setattr(train_adapter, "field_loss", field)
    train_adapter.train(train_adapter.parse_args([*VIEW_TASK, "--name", "t_mask", *VIEW_FLAGS, *COMMON]))
    keeps = [k for k in seen["fm"] if k is not None]
    assert len(keeps) == 4 and all(k.shape == (32, 512) and k.dtype == torch.bool for k in keeps)   # COMMON: 4 steps of batch 32
    assert 0.6 < torch.stack(keeps).float().mean() < 0.8
    assert any(k is None for k in seen["fm"])                              # validation sees every token
    assert seen["field"] and not any(seen["field"])


def test_full_width_views_and_the_flags_refusals(ckpt_dir, monkeypatch):
    monkeypatch.setattr(train_adapter, "SceneEncoder", _ViewStub)
    train_adapter.train(train_adapter.parse_args(["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]))
    train_adapter.train(train_adapter.parse_args([*VIEW_TASK, "--name", "t_full", "--token-source", "vlm_views", "--render-size", "56", "--scene-rank", "0",
                                                  "--view-stride", "8", "--render-workers", "0", *COMMON, "--n-demos", "6", "--val-demos", "4"]))
    ckpt = torch.load(ckpt_dir / "t_full" / "adapter.pt")
    assert "scene_rank" not in ckpt["config"] and "scene_basis" not in ckpt["state_dict"] and ckpt["meta"]["views"]["train"]["kept"] is None
    assert ckpt["state_dict"]["scene_in.weight"].shape[1] == 2048 and ckpt["meta"]["views"]["train"]["moved"] == 0
    v = ["--token-source", "vlm_views"]
    for bad in ([*VIEW_TASK, "--name", "x", *v, "--token-geometry", "relative"], [*VIEW_TASK, "--name", "x", *v, "--teacher", "t"],
                ["--kind", "task", "--pooled", "missing.yaml", "--name", "x", *v], [*VIEW_TASK, "--name", "x", "--token-drop", "0.3"],
                [*VIEW_TASK, "--name", "x", "--token-source", "cnn", "--view-affine", "0.5"], [*VIEW_TASK, "--name", "x", *v, "--wrist-span", "0"],
                [*VIEW_TASK, "--name", "x", *v, "--view-stride", "0"], [*VIEW_TASK, "--name", "x", *v, "--scene-rank", "2048"],
                [*VIEW_TASK, "--name", "x", *v, "--token-drop", "1.0"], [*VIEW_TASK, "--name", "x", *v, "--view-affine", "1.5"]):
        with pytest.raises(SystemExit):
            train_adapter.parse_args(bad)
```

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -k "views or masking or move_pictures" -v`. Expected: FAIL (`move_pictures` missing, `vlm_views` not a choice).

- [ ] **Step 3: Implement** in `train_adapter.py`. Rendering goes through a process pool of `--render-workers` over chunks of `VIEW_CHUNK` frames (in-process at 0). Nothing in this path draws from the trainer's main `rng` or from torch's global generator. Add a `VIEWS:` paragraph to the module docstring beside `TEACHER:` stating the two views, the stride and the sampling at frame steps, the frozen affine, the masking's scope, and the reduction.

- [ ] **Step 4: Run** `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_training.py -v`. Expected: PASS, all earlier trainer tests included.

- [ ] **Step 5: Commit** — `MARS round 3: the adapter trainer encodes two views per frame, with a frozen affine, token masking and a token reduction`.

---

### Task 6: Views in the rollouts

**Files:**

- Modify: `examples/pi05/mars/cursor/cursor_sa/steering.py` (`observation`, `SharedController`, new `view_scene_fn`)
- Modify: `examples/pi05/mars/cursor/cursor_sa/session.py` (`run_trials`)
- Modify: `examples/pi05/mars/cursor/run_sweep.py` (`scene_sources`, `run_cell_batch` and its callers, the docstring's scene paragraph)
- Test: `tests/examples/pi05_mars_cursor/test_scene.py`, `tests/examples/pi05_mars_cursor/test_sweep.py`

**Interfaces:**

- Consumes: `render_views` (Task 1); a `vlm_views` module's `wrist_span`, `image_size` (Task 2); `encode_views` (Task 3).
- Produces:
  - `observation(layouts, pos, hist, device="cpu", goal_given=False, scene=None)`: `scene` (source to tensor) is added to `obs["scene"]` beside the layouts' own, moved to `device`.
  - `SharedController(..., scene_fn=None)` with attributes `scene_fn` and `active` (`None`, or a bool `[B]` of the trials still running). `observation()` calls `scene_fn(self.layouts, self.pos, self.active)` and reuses the result while both the positions and `active` are unchanged.
  - `view_scene_fn(module, encoder) -> Callable[[Layouts, np.ndarray, np.ndarray | None], dict[str, torch.Tensor]]`: `{"vlm_views": tokens}`, float16 `[B, 512, 2048]` on the encoder's device: `encode_views(render_views(rows' layouts, rows' pos, module.wrist_span, module.image_size), encoder)` for the active rows, zeros for the others, nothing rendered when none is active. It calls `render_views` through the module global `cursor_sa.steering.render_views`.
  - `run_trials(..., scene_encoder=None)`: when the arm's task module reads `vlm_views`, `scene_encoder` (an encoder, or a zero-argument factory, called once) is required (`ValueError` naming `scene_encoder`) and becomes the controller's `scene_fn` through `view_scene_fn`; `ctl.active = ~env.done` is set before every `agent.step`.
  - `run_sweep`: `scene_sources` skips `vlm_views` (nothing is cached for it); `run_cell_batch(..., scene_encoder=None)` hands the run's `encoder_factory` to `run_trials`, so the encoder is built only if a view module plays; `--precompute-scene` builds none for view modules.

- [ ] **Step 1: Write the failing tests.** In `tests/examples/pi05_mars_cursor/test_scene.py` (import `view_scene_fn`, `encode_views`, `LearnedTaskModule`; copy `_PatchEncoder` from Task 3 with a counter `self.images += len(images)` set to 0 in `__init__`):

```python
def test_a_controller_asks_its_scene_provider_once_per_position():
    lay, calls = sample_layouts(np.random.default_rng(0), 4, 4, TaskConfig(min_reach=0.35, max_reach=0.5)), []

    def scene_fn(layouts, pos, active):
        calls.append(None if active is None else active.copy())
        return {"vlm_views": torch.full((len(layouts), 2, 3), float(len(calls)))}

    ctl = SharedController(FlowPolicy(), lay, "policy", scene_fn=scene_fn)
    a, b = ctl.observation(), ctl.observation(goal_given=True)
    assert len(calls) == 1 and torch.equal(a["scene"]["vlm_views"], b["scene"]["vlm_views"]) and calls[0] is None
    ctl.pos = ctl.pos + 0.01
    assert float(ctl.observation()["scene"]["vlm_views"][0, 0, 0]) == 2.0 and len(calls) == 2
    ctl.active = np.array([True, False, True, True])
    ctl.observation()
    assert len(calls) == 3 and calls[2].tolist() == [True, False, True, True]
    assert "scene" not in SharedController(FlowPolicy(), lay, "policy").observation()


def test_view_scene_fn_encodes_the_running_trials_views_at_their_positions():
    lay = sample_layouts(np.random.default_rng(0), 4, 4, TaskConfig(min_reach=0.35, max_reach=0.5))
    module = LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=2048, scene_tokens=512, image_size=56, wrist_span=0.5)
    enc, pos = _PatchEncoder(), lay.start + 0.05
    fn = view_scene_fn(module, enc)
    full = fn(lay, pos, None)["vlm_views"]
    assert full.shape == (4, 512, 2048) and full.dtype == torch.float16 and enc.images == 8
    assert torch.equal(full, encode_views(render_views(lay, pos, 0.5, 56), _PatchEncoder()))
    active = np.array([True, False, True, False])
    part = fn(lay, pos, active)["vlm_views"]
    assert enc.images == 12 and torch.equal(part[active], full[active]) and not part[~active].any()
    assert fn(lay, pos, np.zeros(4, bool))["vlm_views"].shape == (4, 512, 2048) and enc.images == 12
```

In `tests/examples/pi05_mars_cursor/test_sweep.py` (reuse `_image_adapter`'s `adapter_path` patch; `_ViewStub` as in Task 5 with a class counter `made` incremented in `__init__`):

```python
def test_tiny_sweep_runs_two_view_modules_at_their_own_spans(tmp_path, monkeypatch):
    import cursor_sa.session as session_mod
    import cursor_sa.steering as steering_mod

    monkeypatch.setattr(session_mod, "adapter_path", lambda n: tmp_path / "ckpt" / "adapters" / n / "adapter.pt")
    for name, span in (("image/v05", 0.5), ("image/v10", 1.0)):
        save_adapter_module(LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=2048, scene_tokens=512, image_size=56,
                                              wrist_span=span, scene_rank=2), tmp_path / "ckpt" / "adapters" / name / "adapter.pt", "task")
    rendered, real, _ViewStub.made = [], steering_mod.render_views, 0

    def spy(layouts, pos, span, size, **k):
        rendered.append((span, size))
        return real(layouts, pos, span, size, **k)

    monkeypatch.setattr(steering_mod, "render_views", spy)
    monkeypatch.setattr(sweep_mod, "SceneEncoder", _ViewStub)
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_views\nagent: expert_chunk\narm: condition_grid.yaml\nmethods:\n"
        "  oracle_a: {mode: teleop, readout: off, reversal_adapter: {task: image/v05}}\n"
        "  policy_b: {mode: policy, reversal_adapter: {task: image/v10}}\n"
        "tune: {frs: {n_reverse_steps: [10]}}\ngrid:\n  users: {U0: {corruption: null, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [0.0]\nfocus: oracle_a\nseed: 3\n"
    )
    args = ["--sweep", str(tmp_path / "tiny.yaml"), "--results", str(tmp_path), "--scene-cache", str(tmp_path / "scene")]
    sweep_main([*args, "--out", "pre", "--precompute-scene"])
    assert _ViewStub.made == 0 and not rendered                             # nothing to precompute for per-step views
    sweep_main([*args, "--out", "tiny_views", "--tune-trials", "2", "--test-trials", "3"])
    assert {span for span, _ in rendered} == {0.5, 1.0} and {size for _, size in rendered} == {56} and _ViewStub.made == 1
    assert not list((tmp_path / "scene").glob("*.npy"))
    test_json = tmp_path / "sim" / "sweeps" / "tiny_views" / "test.json"
    assert set(json.loads(test_json.read_text())["test"]["U0xT2|0.0"]["metrics"]) == {"oracle_a", "policy_b"}
```

(`sweep_mod` is `run_sweep` as the test file imports it; match its existing import style.) Add to the same file a direct call of `run_trials` on a `vlm_views` arm without `scene_encoder`, expecting `ValueError` matching `scene_encoder`; build the arm with `method_arm` on the tiny sweep above.

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_scene.py tests/examples/pi05_mars_cursor/test_sweep.py -k "provider or view" -v`. Expected: FAIL (`scene_fn` unexpected, `view_scene_fn` missing).

- [ ] **Step 3: Implement** in the three files. `run_experiment.py` and `fig_traces` keep calling `run_trials` without an encoder: a view module there fails with the `scene_encoder` message.

- [ ] **Step 4: Run** `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor -v`. Expected: PASS, the session, sweep and steering-equivalence tests included.

- [ ] **Step 5: Commit** — `MARS round 3: rollouts encode the two views at every observation, running trials only`.

---

### Task 7: The sweeps and the report's three rules

**Files:**

- Create: `examples/pi05/mars/cursor/configs/sweeps/grid_task_check_views_select.yaml`, `grid_task_check_views_width.yaml`, `grid_task_check_views_n300.yaml`, `grid_task_check_views_n1000.yaml`, `grid_task_check_views_n3000.yaml`
- Modify: `examples/pi05/mars/cursor/report_grid_image.py`
- Test: `tests/examples/pi05_mars_cursor/test_sweep.py`, `tests/examples/pi05_mars_cursor/test_report_grid_image.py`

**Interfaces:**

- Consumes: `grid_task_check_image2_quick.yaml` as the template (agent, arm, `tune`, the `U0` user, the T2 and T4 task entries, `noises: [0.0]`, `trials: {tune: 20, test: 500}`, `against: [oracle_task, oracle_ft]`); `gate_table`, `_within`, `_context`, `_load`, `_table`, `_fmt`.
- Produces, sweeps (each header states its stage and rule, as the template's does):
  - `..._views_select` (seed 821, T2 and T4): `oracle`, `oracle_task`, `oracle_ft`, and `oracle_task_views_<r>` for `r` in `base`, `span1`, `nomask`, `noaffine`, each naming `image/task_disc_large_n300_s0_with_gain025y_views_<r>` (T2) and `image/task_ring_goal_n300_s0_with_rot60_views_<r>` (T4). `focus: oracle_task_views_base`. No goal-hidden rows.
  - `..._views_width` (seed 821, T4 only): `oracle_task`, `oracle_task_views_base`, `oracle_task_views_full`. `focus: oracle_task_views_base`, `against: [oracle_task]`.
  - `..._views_n<N>` for N in 300, 1000, 3000 (seed 820, T2 and T4): `oracle`, `oracle_task`, `oracle_ft`, `oracle_task_views_n<N>`, `policy`, `policy_task`, `policy_task_views_n<N>`, the view rows naming `image/task_disc_large_n<N>_s0_with_gain025y_views` and `image/task_ring_goal_n<N>_s0_with_rot60_views`; the n300 file also carries the template's `oracle_task_rel`, `oracle_task_teach_rel`, `policy_task_rel`, `policy_task_teach_rel` rows. `focus: oracle_task_views_n<N>`.
- Produces, report:
  - `LEVEL = 0.03`, `ADOPT = 0.03`, `WIDTH_DROP = 0.05`.
  - `scan_rows(runs: dict[int, dict]) -> list[dict]`: one row per context and count, sorted by context then count, keys `context, demos, hits, collision, ref_hits, ref_collision, level, in_band, hidden_hits, ref_hidden_hits, flag`. `level` = hits ≥ ref − `LEVEL` and collision ≤ ref + `LEVEL`; `in_band` the same with `MARGIN`; both through `_within`; `flag` as `gate_table`'s. `ValueError` naming `layouts` when a context's `oracle_task` metrics differ between runs.
  - `first_count(rows: list[dict], key: str) -> dict[str, int | None]`: per context, the smallest count whose row has `key` true.
  - `crossing(counts: list[int], values: list[float], threshold: float) -> tuple[float | None, bool]`: where the curve, linear in `log10(count)` between points, first reaches `threshold`; the first count when it starts there; past the last count, along the last segment when that rises, marked `True` (an extrapolation); `(None, False)` when it never gets there and the last segment does not rise.
  - `select_recipe(check: dict, base: str = "base", changes: tuple[str, ...] = ("span1", "nomask", "noaffine")) -> dict`: `{"means": {arm: mean hits of oracle_task_views_<arm> over the contexts}, "adopted": [changes whose mean exceeds the base's by more than ADOPT]}`.
  - `width_check(check: dict, reduced: str = "base", full: str = "full", context: str = "T4") -> dict`: `{"reduced_hits", "full_hits", "ok"}`, `ok` when reduced ≥ full − `WIDTH_DROP` (the exact boundary passes).
  - CLI: one of `--check`, `--select RUN`, `--width RUN`, `--scan RUN [RUN ...]` (their counts read from the `oracle_task_views_n<N>` methods). `--select` and `--width` print their dict and write `select.json` / `width.json` to `--out`; `--scan` writes `report.md` (a `## Scan` table of the rows in percent, then per context the first level count, the first in-band count and the two crossings, an extrapolated one marked as such, and `more than <largest count>` when none is level) and `report.json` (`scan`, `first_level`, `first_in_band`, `crossings`).

- [ ] **Step 1: Write the failing tests.** In `tests/examples/pi05_mars_cursor/test_sweep.py`:

```python
def test_view_sweep_configs_resolve_every_arm():
    sel = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_views_select.yaml")
    assert sel["seed"] == 821 and set(sel["grid"]["tasks"]) == {"T2", "T4"} and sel["focus"] == "oracle_task_views_base"
    assert set(sel["methods"]) == {"oracle", "oracle_task", "oracle_ft", *(f"oracle_task_views_{r}" for r in ("base", "span1", "nomask", "noaffine"))}
    assert method_arm(sel, "oracle_task_views_span1", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n300_s0_with_rot60_views_span1"
    width = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_views_width.yaml")
    assert width["seed"] == 821 and set(width["grid"]["tasks"]) == {"T4"}
    assert method_arm(width, "oracle_task_views_full", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n300_s0_with_rot60_views_full"
    quick = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2_quick.yaml")
    for n in (300, 1000, 3000):
        scan = load_sweep(config.CONFIG_DIR / "sweeps" / f"grid_task_check_views_n{n}.yaml")
        assert scan["seed"] == quick["seed"] == 820 and scan["trials"] == quick["trials"] and scan["arm"] == quick["arm"] and scan["agent"] == quick["agent"]
        assert {"oracle", "oracle_task", "oracle_ft", "policy", "policy_task", f"oracle_task_views_n{n}", f"policy_task_views_n{n}"} <= set(scan["methods"])
        assert method_arm(scan, f"oracle_task_views_n{n}", "U0xT2")["learned_adapter"]["task"] == f"image/task_disc_large_n{n}_s0_with_gain025y_views"
        assert method_arm(scan, f"policy_task_views_n{n}", "U0xT4")["learned_adapter"]["task"] == f"image/task_ring_goal_n{n}_s0_with_rot60_views"
        assert ("oracle_task_teach_rel" in scan["methods"]) == (n == 300)
```

In `tests/examples/pi05_mars_cursor/test_report_grid_image.py` (import the new names):

```python
def _m(hits, col=0.0):
    return {"success": hits, "collision": col}


def _run(by_ctx):
    return {"test": {f"U0x{ctx}|0.0": {"metrics": m} for ctx, m in by_ctx.items()}}


def test_scan_rows_level_band_flag_and_first_counts():
    ref = {"oracle_task": _m(0.928, 0.034), "policy_task": _m(0.764)}
    runs = {300: _run({"T4": {**ref, "oracle_task_views_n300": _m(0.50, 0.45), "policy_task_views_n300": _m(0.40)}}),
            1000: _run({"T4": {**ref, "oracle_task_views_n1000": _m(0.83, 0.13), "policy_task_views_n1000": _m(0.87)}}),
            3000: _run({"T4": {**ref, "oracle_task_views_n3000": _m(0.898, 0.064), "policy_task_views_n3000": _m(0.70)}})}
    rows = scan_rows(runs)
    assert [(r["demos"], r["level"], r["in_band"], r["flag"]) for r in rows] == [
        (300, False, False, False), (1000, False, True, True), (3000, True, True, False)]   # 3000 sits exactly 3 points off on both
    assert first_count(rows, "level") == {"T4": 3000} and first_count(rows, "in_band") == {"T4": 1000}
    assert first_count(rows[:1], "level") == {"T4": None}
    runs[1000]["test"]["U0xT4|0.0"]["metrics"]["oracle_task"] = _m(0.90, 0.034)
    with pytest.raises(ValueError, match="layouts"):
        scan_rows(runs)


def test_crossing_is_log_linear_and_marks_an_extrapolation():
    n, extra = crossing([300, 3000], [0.5, 0.9], 0.7)
    assert abs(n - 10 ** (np.log10(300) + 0.5)) < 1e-6 and not extra                       # halfway in log10
    n, extra = crossing([300, 1000, 3000], [0.2, 0.4, 0.6], 0.8)
    assert extra and abs(n - 3000 * (3000 / 1000)) < 1e-6                                   # one more step of the last rise
    assert crossing([300, 1000, 3000], [0.6, 0.5, 0.4], 0.8) == (None, False)
    assert crossing([300, 1000], [0.9, 0.95], 0.8) == (300.0, False)


def test_select_recipe_and_width_check_apply_their_margins():
    check = _run({"T2": {f"oracle_task_views_{r}": _m(h) for r, h in (("base", 0.50), ("span1", 0.56), ("nomask", 0.53), ("noaffine", 0.40))},
                  "T4": {f"oracle_task_views_{r}": _m(h) for r, h in (("base", 0.30), ("span1", 0.32), ("nomask", 0.33), ("noaffine", 0.36))}})
    out = select_recipe(check)
    assert out["adopted"] == ["span1"] and abs(out["means"]["base"] - 0.40) < 1e-9          # +4; nomask is +3 exactly; noaffine -2
    width = _run({"T4": {"oracle_task_views_base": _m(0.35), "oracle_task_views_full": _m(0.40)}})
    assert width_check(width)["ok"]                                                        # exactly 5 points under
    width["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_base"] = _m(0.348)
    assert not width_check(width)["ok"]
```

Add a test that runs `main(["--scan", ...])` on three small `test.json` files written to `tmp_path` (the first test's `runs`) and asserts `report.md` contains `## Scan`, `3000` on the level line for T4, and that `report.json` has the keys `scan`, `first_level`, `first_in_band`, `crossings`; and one where no count is level, asserting `more than 3000`.

- [ ] **Step 2: Run them to see them fail** — `uv run --extra mars --extra test pytest tests/examples/pi05_mars_cursor/test_report_grid_image.py tests/examples/pi05_mars_cursor/test_sweep.py -k "scan or crossing or select or view_sweep" -v`. Expected: FAIL (files and names missing).

- [ ] **Step 3: Write** the five sweep files and the report's functions and CLI modes. `--check` keeps working as it does.

- [ ] **Step 4: Run** the two test files. Expected: PASS, the existing gate-verdict tests included.

- [ ] **Step 5: Commit** — `MARS round 3: the recipe, width and scan sweeps and their rules in the report`.

---

### Task 8: Jobs, pipeline stages and a real-encoder smoke run (controller; the run folder is git-ignored)

**Files:**

- Create: `examples/pi05/mars/results/research/image/make_jobs3.py`
- Modify: `examples/pi05/mars/results/research/image/pipeline.sh` (by rename, not in place, if any pipeline is running)

- [ ] **Step 1: `make_jobs3.py`.** Common to every job: `HF_HUB_OFFLINE=1 uv run --extra mars --extra pi python examples/pi05/mars/cursor/train/train_adapter.py --kind task --obstacle '<spec>' --user-module user_<u>_expert_n100_s0 --corruption '<c>' --token-source vlm_views --view-stride 2 --val-demos 300 --frozen-noise --seed 0` (`TASKS`, `USERS` from `make_jobs.py`; `disc_large`, `ring_goal`). Recipes: base `--wrist-span 0.5 --token-drop 0.3 --view-affine 0.5 --scene-rank 256`; `span1` sets `--wrist-span 1.0`; `nomask` `--token-drop 0`; `noaffine` `--view-affine 0`; `full` (ring_goal only) `--scene-rank 0`. `python make_jobs3.py a` writes `jobs_r3_a.txt` (nine jobs, `--n-demos 300`, names per Global Constraints). `python make_jobs3.py b <N> --recipe '<flags>'` writes `jobs_r3_n<N>.txt` (two jobs, `--n-demos <N>`, names `..._views`).
- [ ] **Step 2: Pipeline stages.** `train3a` (runs `jobs_r3_a.txt`), `gate3a` (the select and width sweeps, then `report_grid_image.py --select` and `--width` into `results/sim/reports/grid_views_select`), `train3b <N>`, `gate3b <N>` (the `n<N>` sweep, its `--check` report, then `--scan` over every scan sweep run so far into `results/sim/reports/grid_views_scan`). Sweeps run with `--extra pi`; the two contexts of a sweep go to the two GPUs with `--cells` and are merged.
- [ ] **Step 3: Smoke run with the real encoder,** one GPU: a base-recipe ring_goal job at `--n-demos 20 --val-demos 10 --steps 200 --name image_smoke/views`, then a `--test-trials 20` run of a one-method copy of the select sweep on it. Record in the ledger: pictures encoded per second (render and encode), the variance kept at rank 256, peak GPU memory, seconds per training step, pictures per rollout row. Rulings if needed: under the spec's target of 300 pictures per second, raise the render workers and the encode batch before stage A and record what was reached; variance kept under 0.95, raise the rank to 512 for stage A and note it.
- [ ] **Step 4:** Decide one or two jobs per GPU for the 300-demo runs from the measured peak (two only if two peaks plus 4 GB fit in 49 GB); the `full` job runs alone on its GPU.

---

### Task 9: Run stage A, choose the recipe, run the scan, report (controller)

- [ ] **Step 1: Stage A.** `train3a`, then `gate3a`. Read `select.json` and `width.json`.
- [ ] **Step 2: Settle the recipe and the rank** (spec rules, logged in the ledger with the numbers). Recipe = base plus every adopted change. If `width.json` is not ok: retrain the chosen recipe at 300 on both contexts at `--scene-rank 512`, gate its T4 run beside the same full-width run, and if it is still more than 5 points under, run the scan at `--scene-rank 0`.
- [ ] **Step 3: The scan's 300 point.** When the chosen recipe and rank are ones stage A trained, copy those two checkpoint folders to the `..._n300_s0_..._views` names (a copy, not a move; log it); otherwise `make_jobs3.py b 300` and train. `gate3b 300`.
- [ ] **Step 4: 1,000, then 3,000.** `make_jobs3.py b 1000`, `train3b 1000`, `gate3b 1000`. Stop if the scan report shows 1,000 level on both contexts; otherwise the same for 3,000.
- [ ] **Step 5: Report to the user and stop.** The scan table per context (hits, collisions, level, in band, flag), the first level count, the crossing estimates with any extrapolation marked, stage A's table and the recipe chosen, the width check, the measured costs, and the rulings made with their cost if wrong. Seeds 1–2, T3 and T5 and the detector branch wait for the user's go.
