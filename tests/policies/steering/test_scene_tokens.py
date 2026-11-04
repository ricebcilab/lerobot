import pytest
import torch

from lerobot.policies.steering import LearnedTaskModule, load_adapter_module, save_adapter_module
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


def _module(src):
    return LearnedTaskModule(
        horizon=16,
        token_source=src,
        scene_dim=32 if src.startswith("vlm") else None,
        scene_tokens=16 if src.startswith("vlm") else None,
        image_size=64 if src == "cnn" else None,
    )


@pytest.mark.parametrize("src", ["vlm_image", "vlm_prefix", "cnn"])
def test_image_sources_start_at_identity_and_round_trip(src, tmp_path):
    torch.manual_seed(0)
    m = _module(src).eval()
    x, v, t = _xvt()
    obs = _scene()
    if src == "vlm_prefix":
        obs["scene"] = {"vlm_prefix": obs["scene"]["vlm_image"]}
    a, b = m(x, t, v, obs)
    assert torch.equal(a, torch.eye(2).expand(3, 16, 2, 2)) and torch.equal(b, torch.zeros(3, 16, 2))
    a, b = m(x, t, v, None)  # the LIBERO call still runs
    assert torch.equal(a, torch.eye(2).expand(3, 16, 2, 2))
    assert m.config["token_source"] == src and m.config["scene_tokens"] == 16
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    m.requires_grad_(False)
    path = save_adapter_module(m, tmp_path / "t.pt", kind="task")
    loaded, kind, _ = load_adapter_module(path)
    assert kind == "task" and loaded.token_source == src
    a0, b0 = m(x, t, v, obs)
    a1, b1 = loaded(x, t, v, obs)
    assert torch.equal(a0, a1) and torch.equal(b0, b1)
    assert not torch.equal(a0, torch.eye(2).expand(3, 16, 2, 2))


def test_discs_config_is_unchanged_and_an_image_module_needs_its_scene():
    assert set(LearnedTaskModule(horizon=16).config) == {
        "horizon",
        "d",
        "tdim",
        "heads",
        "layers",
        "scale",
        "n_dims",
    }
    m = _module("vlm_image")
    x, v, t = _xvt()
    with pytest.raises(ValueError, match="vlm_image"):
        m(x, t, v, {"pos": torch.rand(3, 2)})
    with pytest.raises(ValueError, match="vlm_image"):
        m(x, t, v, {"pos": torch.rand(3, 2), "scene": {"cnn": torch.zeros(3, 3, 64, 64, dtype=torch.uint8)}})
    with pytest.raises(ValueError):
        LearnedTaskModule(horizon=16, token_source="vlm_image")  # no scene_dim
    with pytest.raises(ValueError):
        LearnedTaskModule(horizon=16, token_source="cnn", image_size=50)  # not a multiple of 16
    with pytest.raises(ValueError):
        LearnedTaskModule(horizon=16, token_source="nope")


def test_fp16_tokens_match_fp32_within_tolerance():
    torch.manual_seed(0)
    m = _module("vlm_image").eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.out.weight, std=0.3)
    x, v, t = _xvt()
    obs = _scene()
    half = obs["scene"]["vlm_image"]
    a16, b16 = m(x, t, v, obs)
    a32, b32 = m(x, t, v, {**obs, "scene": {"vlm_image": half.float()}})
    torch.testing.assert_close(a16, a32, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(b16, b32, atol=1e-3, rtol=1e-3)
    assert a16.dtype == torch.float32


def test_cnn_tokens_change_with_the_image_and_vlm_tokens_with_the_scene():
    torch.manual_seed(0)
    x, v, t = _xvt()
    obs = _scene()
    for src in ("cnn", "vlm_image"):
        m = _module(src).eval()
        with torch.no_grad():
            torch.nn.init.normal_(m.out.weight, std=0.3)
        other = {**obs, "scene": {src: torch.flip(obs["scene"][src], [0])}}  # another layout's scene
        assert not torch.equal(m(x, t, v, obs)[0], m(x, t, v, other)[0])
    assert SCENE_SOURCES == ("discs", "vlm_image", "vlm_prefix", "cnn")
