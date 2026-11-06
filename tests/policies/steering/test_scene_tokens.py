import pytest
import torch

from lerobot.policies.steering import (
    TOKEN_GEOMETRIES,
    LearnedTaskModule,
    load_adapter_module,
    reduce_tokens,
    save_adapter_module,
    scene_spec,
)
from lerobot.policies.steering.modeling_steering import SCENE_SOURCES


def _xvt(b=3, h=16):
    return torch.randn(b, h, 2), torch.randn(b, h, 2), torch.rand(b)


def _scene(b=3):
    return {
        "pos": torch.rand(b, 2),
        "scene": {
            "vlm_image": torch.randn(b, 16, 32, dtype=torch.float16),
            "cnn": torch.randint(0, 256, (b, 3, 64, 64), dtype=torch.uint8),
        },
    }


def _module(src, geometry="absolute"):
    return LearnedTaskModule(
        horizon=16,
        token_source=src,
        token_geometry=geometry,
        scene_dim=32 if src.startswith("vlm") else None,
        scene_tokens=16 if src.startswith("vlm") else None,
        image_size=64 if src == "cnn" else None,
    )


@pytest.mark.parametrize("src", ["vlm_image", "vlm_prefix", "cnn"])
def test_image_sources_start_at_identity_round_trip_and_read_their_scene(src, tmp_path):
    torch.manual_seed(0)
    m = _module(src).eval()
    x, v, t = _xvt()
    obs = _scene()
    if src == "vlm_prefix":
        obs["scene"] = {"vlm_prefix": obs["scene"]["vlm_image"]}
    eye = torch.eye(2).expand(3, 16, 2, 2)
    a, b = m(x, t, v, obs)
    assert torch.equal(a, eye) and torch.equal(b, torch.zeros(3, 16, 2))
    assert torch.equal(m(x, t, v, None)[0], eye)  # the LIBERO call still runs
    keys = {"horizon", "d", "tdim", "heads", "layers", "scale", "n_dims"}
    assert set(LearnedTaskModule(horizon=16).config) == keys  # the discs' config as it was
    assert set(m.config) == keys | {"token_source", "scene_dim", "scene_tokens", "image_size"}  # no new keys
    assert m.config["token_source"] == src and m.config["scene_tokens"] == 16
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m.requires_grad_(False)
    loaded, kind, _ = load_adapter_module(save_adapter_module(m, tmp_path / "t.pt", kind="task"))
    assert kind == "task" and loaded.token_source == src
    a0, b0 = m(x, t, v, obs)
    a1, b1 = loaded(x, t, v, obs)
    assert torch.equal(a0, a1) and torch.equal(b0, b1) and not torch.equal(a0, eye)
    other = {
        **obs,
        "scene": {k: torch.flip(s, [0]) for k, s in obs["scene"].items()},
    }  # another layout's scene
    assert not torch.equal(m(x, t, v, other)[0], a0)
    if src == "vlm_image":  # fp16 tokens give what fp32 ones do, to a tolerance
        a32, b32 = m(x, t, v, {**obs, "scene": {src: obs["scene"][src].float()}})
        torch.testing.assert_close(a0, a32, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(b0, b32, atol=1e-3, rtol=1e-3)
        assert a0.dtype == torch.float32


def _live(src, geometry="absolute"):
    """A module with a non-zero head (so its output depends on the tokens) and its observation."""
    torch.manual_seed(0)
    m = (_views() if src == "vlm_views" else _module(src, geometry)).eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.out.weight, std=0.3)
    obs = _scene()
    if src == "vlm_views":
        obs["scene"] = {"vlm_views": torch.randn(3, 8, 32, dtype=torch.float16)}
    if src == "vlm_prefix":
        obs["scene"] = {"vlm_prefix": obs["scene"]["vlm_image"]}
    if src == "discs":
        obs["obstacles"] = torch.rand(3, 2, 3) * 0.2
        obs["obstacle_mask"] = torch.tensor([[True, True], [True, False], [False, False]])
    return m, obs


@pytest.mark.parametrize(
    "src, geometry",
    [
        ("discs", "absolute"),
        ("vlm_image", "absolute"),
        ("vlm_prefix", "absolute"),
        ("cnn", "absolute"),
        ("vlm_views", "absolute"),
        ("vlm_image", "relative"),
        ("cnn", "relative"),
    ],
)
def test_a_bound_call_equals_the_unbound_call(src, geometry):
    m, obs = _live(src, geometry)
    x, v, t = _xvt()
    bound = m.bind(obs)
    for _ in range(3):  # the first call computes the tokens, the others reuse them
        a, b = bound(x, t, v)
        a0, b0 = m(x, t, v, obs)
        assert torch.equal(a, a0) and torch.equal(b, b0)
        assert not torch.equal(a, torch.eye(2).expand(3, 16, 2, 2))
        x, v, t = _xvt()


def test_a_bound_module_runs_the_cnn_once_and_gradients_reach_it():
    m, obs = _live("cnn")
    runs = []
    m.cnn.register_forward_hook(lambda *_: runs.append(1))
    bound = m.bind(obs)
    x, v, t = _xvt()
    for _ in range(10):
        bound(x, t, v)
    assert len(runs) == 1
    m(x, t, v, obs)  # the unbound call runs it every time
    assert len(runs) == 2
    half = {"pos": obs["pos"][:2], "scene": {"cnn": obs["scene"]["cnn"][:2]}}
    bound.obs = half  # another batch size: the tokens are made again, for it
    x, v, t = _xvt(b=2)
    a, b = bound(x, t, v)
    assert len(runs) == 3
    a0, b0 = m(x, t, v, half)
    assert torch.equal(a, a0) and torch.equal(b, b0)
    bound(x, t, v)
    assert len(runs) == 4  # the unbound call above, and nothing more for the bound one
    bound.obs = obs
    x3, v3, t3 = _xvt()
    bound(x3, t3, v3)
    assert len(runs) == 5
    # gradients reach the CNN through the cached tokens
    m, obs = _live("cnn")
    m.train().requires_grad_(True)
    runs = []
    m.cnn.register_forward_hook(lambda *_: runs.append(1))
    bound = m.bind(obs)
    x, v, t = _xvt()
    with (
        torch.no_grad()
    ):  # a first call without gradients must not leave tokens a later call with them would reuse
        bound(x, t, v)
    loss = 0.0
    for _ in range(3):
        a, b = bound(x, t, v)
        loss = loss + (a**2).sum() + (b**2).sum()
    loss.backward()
    assert len(runs) == 2  # once without gradients, once with
    grads = [p.grad for p in m.cnn.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    reference, obs_ref = _live("cnn")
    reference.train().requires_grad_(True)
    ref_loss = 0.0
    for _ in range(3):
        a, b = reference(x, t, v, obs_ref)
        ref_loss = ref_loss + (a**2).sum() + (b**2).sum()
    ref_loss.backward()
    for p, q in zip(m.cnn.parameters(), reference.cnn.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-3, atol=1e-3)  # summed in another order


def _centres(g):  # token k = (row, col) = divmod(k, g); x right (col), y down (row)
    rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([(cols.flatten() + 0.5) / g, (rows.flatten() + 0.5) / g], -1)


@pytest.mark.parametrize("src", ["vlm_image", "cnn"])
def test_relative_tokens_are_the_projection_plus_the_cursor_relative_geometry(src, tmp_path):
    assert TOKEN_GEOMETRIES == ("absolute", "relative")
    m, ab, obs = _module(src, "relative"), _module(src), _scene()
    assert m.config["token_geometry"] == "relative" and "token_geometry" not in ab.config
    assert "scene_pos" not in m.state_dict() and "scene_pos" in ab.state_dict()
    assert LearnedTaskModule(**ab.config).token_geometry == "absolute"  # a round-1 config
    back, kind, _ = load_adapter_module(save_adapter_module(m, tmp_path / "rel.pt", "task"))
    x, v, t = _xvt()
    assert kind == "task" and back.token_geometry == "relative"
    assert all(torch.allclose(a, b) for a, b in zip(m(x, t, v, obs), back(x, t, v, obs), strict=True))
    rel = (_centres(4)[None] - obs["pos"][:, None]) / m.scale
    feats = torch.cat(
        [rel, torch.full_like(rel[..., :1], (0.5 / 4) / m.scale), rel.norm(dim=-1, keepdim=True)], -1
    )
    x = obs["scene"][src]
    content = (
        m.scene_in(x.float()) if src == "vlm_image" else m.cnn(x.float() / 255).flatten(2).transpose(1, 2)
    )
    tokens, pad = m.tokens(obs, 3)
    assert pad is None and tokens.shape == (3, 17, 128)
    assert torch.allclose(tokens[:, 1:], content + m.token_in(feats), atol=1e-6)
    assert torch.equal(m.token_centres, _centres(4)) and m.token_half == 0.5 / 4
    if src == "cnn":  # the CNN's zero-padded borders keep its content from being the same per token
        return
    # every token the same content: a cursor one column right, or one row down, sees the same offsets shifted by one
    # token (the centres buffer is shared, so this pins the row/col convention for both sources)
    obs["scene"][src] = x[:, :1].expand_as(x).clone()
    p = torch.tensor([[0.3, 0.4]]).expand(3, -1)
    tp, tx, ty = (
        m.tokens({**obs, "pos": p + d}, 3)[0][:, 1:].view(3, 4, 4, -1)  # [b, row, col, d]
        for d in (torch.zeros(2), torch.tensor([0.25, 0.0]), torch.tensor([0.0, 0.25]))
    )
    assert torch.allclose(tx[:, :, 1:], tp[:, :, :-1], atol=1e-5) and torch.allclose(
        ty[:, 1:, :], tp[:, :-1, :], atol=1e-5
    )


def _views(rank=None, **kw):
    return LearnedTaskModule(
        horizon=16,
        token_source="vlm_views",
        scene_dim=32,
        scene_tokens=8,
        wrist_span=0.5,
        scene_rank=rank,
        **kw,
    )


def _reduction(dim=32, rank=4):
    return torch.randn(dim), torch.linalg.qr(torch.randn(dim, rank))[0]


def test_a_views_module_keeps_its_span_and_reduction_through_a_checkpoint(tmp_path):
    assert "vlm_views" in SCENE_SOURCES
    assert scene_spec("vlm_views", 224, wrist_span=0.5, scene_rank=256) == {
        "token_source": "vlm_views",
        "scene_dim": 2048,
        "scene_tokens": 512,
        "image_size": 224,
        "wrist_span": 0.5,
        "scene_rank": 256,
    }
    assert scene_spec("vlm_views", 224, wrist_span=1.0)["scene_rank"] is None
    assert set(scene_spec("vlm_image")) == {"token_source", "scene_dim", "scene_tokens", "image_size"}
    full, red = _views(), _views(rank=4)
    assert (
        full.config["wrist_span"] == 0.5
        and "scene_rank" not in full.config
        and "scene_mean" not in full.state_dict()
    )
    assert red.config["scene_rank"] == 4 and red.scene_in.in_features == 4 and full.scene_in.in_features == 32
    assert red.state_dict()["scene_basis"].shape == (32, 4) and red.state_dict()["scene_pos"].shape[0] == 8
    mean, basis = _reduction()
    red.set_reduction(mean, basis)
    back, kind, _ = load_adapter_module(save_adapter_module(red, tmp_path / "v.pt", "task"))
    assert kind == "task" and back.wrist_span == 0.5 and back.scene_rank == 4
    torch.testing.assert_close(back.scene_basis, basis)
    torch.testing.assert_close(back.scene_mean, mean)
    # a reduced module reads full-width and reduced tokens alike
    raw, obs = torch.randn(3, 8, 32).to(torch.float16), {"pos": torch.rand(3, 2)}
    stored = reduce_tokens(raw, red.scene_mean, red.scene_basis)
    assert stored.dtype == torch.float16 and stored.shape == (3, 8, 4)
    a = red.tokens({**obs, "scene": {"vlm_views": raw}}, 3)[0]
    assert torch.equal(a, red.tokens({**obs, "scene": {"vlm_views": stored}}, 3)[0])


def test_scene_keep_hides_tokens_from_the_attention_and_never_the_null_token():
    m = _views()
    torch.nn.init.normal_(m.out.weight)  # a fresh head is zero: the output would not depend on tokens
    scene = torch.randn(2, 8, 32)
    obs = {"pos": torch.rand(2, 2), "scene": {"vlm_views": scene}}
    keep = torch.tensor([[True] * 4 + [False] * 4, [False] * 8])
    _, pad = m.tokens({**obs, "scene_keep": keep}, 2)
    assert pad.shape == (2, 9) and not pad[:, 0].any() and torch.equal(pad[:, 1:], ~keep)
    assert m.tokens(obs, 2)[1] is None
    x, v, t = torch.randn(2, 16, 2), torch.randn(2, 16, 2), torch.rand(2)
    changed = scene.clone()
    changed[0, 4:] += 1.0  # only hidden tokens change
    a = m(x, t, v, {**obs, "scene_keep": keep})
    b = m(x, t, v, {"pos": obs["pos"], "scene": {"vlm_views": changed}, "scene_keep": keep})
    torch.testing.assert_close(a[0][0], b[0][0])
    assert not torch.allclose(a[0][0], m(x, t, v, obs)[0][0])  # and hiding them matters
    with pytest.raises(ValueError, match="scene_keep"):
        m.tokens({**obs, "scene_keep": keep[:, :5]}, 2)


def test_the_modules_refuse_what_they_cannot_read():
    x, v, t = _xvt()
    m = _module("vlm_image")
    for scene in (
        {},
        {"cnn": torch.zeros(3, 3, 64, 64, dtype=torch.uint8)},
    ):  # an image module needs its scene
        with pytest.raises(ValueError, match="vlm_image"):
            m(x, t, v, {"pos": torch.rand(3, 2), **({"scene": scene} if scene else {})})
    obs = _scene()
    del obs["pos"]
    with pytest.raises(ValueError, match="pos"):
        _module("vlm_image", "relative").tokens(obs, 3)
    red = _views(rank=4)
    red.set_reduction(*_reduction())
    with pytest.raises(ValueError, match="width"):
        red.tokens({"pos": torch.rand(3, 2), "scene": {"vlm_views": torch.randn(3, 8, 5)}}, 3)
    with pytest.raises(ValueError):
        red.set_reduction(*_reduction(rank=3))
    for build, match in (
        (lambda: LearnedTaskModule(horizon=16, token_source="vlm_image"), "scene_dim"),  # no scene_dim
        (
            lambda: LearnedTaskModule(horizon=16, token_source="cnn", image_size=50),
            None,
        ),  # not a multiple of 16
        (lambda: LearnedTaskModule(horizon=16, token_source="nope"), None),
        (lambda: _module("discs", "relative"), "relative"),  # tokens without a screen position
        (lambda: _module("vlm_prefix", "relative"), "relative"),
        (
            lambda: LearnedTaskModule(
                token_source="vlm_image", token_geometry="relative", scene_dim=32, scene_tokens=15
            ),
            "square",
        ),
        (lambda: _module("vlm_image", "polar"), "token_geometry"),
        (lambda: _views(token_geometry="relative"), "relative"),
        (
            lambda: LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=32, scene_tokens=8),
            "wrist_span",
        ),
        (
            lambda: LearnedTaskModule(
                horizon=16, token_source="vlm_image", scene_dim=32, scene_tokens=16, scene_rank=4
            ),
            "scene_rank",
        ),
        (lambda: scene_spec("vlm_image", 224, wrist_span=0.5), None),
    ):
        with pytest.raises(ValueError, match=match):
            build()
    for bad in (
        {},
        {"wrist_span": 0.0},
        {"wrist_span": 0.5, "scene_rank": 2048},
        {"wrist_span": 0.5, "scene_rank": 0},
    ):
        with pytest.raises(ValueError):
            scene_spec("vlm_views", 224, **bad)
