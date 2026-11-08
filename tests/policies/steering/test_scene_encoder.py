#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scene arrays for the task module's token sources: the spec of a module for a source, the per-layout array
it reads, and pi0.5's vision tokens (loaded lazily)."""

import os
import subprocess  # nosec B404
import sys

import numpy as np
import pytest
import torch

from lerobot.envs.cursor import SCENE_SIZE, TaskConfig, obstacle_spec, render_layouts, sample_layouts
from lerobot.policies.steering import (
    LearnedTaskModule,
    SceneEncoder,
)


class Stub:
    def __init__(self):
        self.calls = []

    def image_tokens(self, images):
        self.calls.append(images.shape)
        return np.zeros((len(images), 256, 2048), np.float16) + images.mean()

    def prefix_tokens(self, images):
        return self.image_tokens(images) + 1


def _layouts(n=4, seed=0, obstacle=None):
    return sample_layouts(
        np.random.default_rng(seed),
        n,
        4,
        TaskConfig(
            min_reach=0.3,
            max_reach=0.5,
            obstacle=obstacle_spec(obstacle or {"kind": "disc", "radius": 0.08, "along": 0.5}),
        ),
    )


def test_scene_spec_builds_a_module_that_accepts_scene_array():
    lay = _layouts()
    stub = Stub()
    for src in ("vlm_image", "vlm_prefix", "cnn"):
        spec = LearnedTaskModule.spec(src, 112)
        m = LearnedTaskModule(horizon=16, **spec)
        arr = SceneEncoder.scene_array(lay, src, 112, encoder=stub)
        assert arr.shape[0] == 4 and (arr.dtype == np.uint8 if src == "cnn" else arr.dtype == np.float16)
        obs = {"pos": torch.rand(4, 2), "scene": {src: torch.as_tensor(arr)}}
        a, _ = m(torch.randn(4, 16, 2), torch.rand(4), torch.randn(4, 16, 2), obs)
        assert a.shape == (4, 16, 2, 2)
    assert (
        stub.calls[0] == (4, 3, 112, 112)
        and LearnedTaskModule.spec("vlm_image")["scene_tokens"] == 256
        and LearnedTaskModule.spec("cnn")["scene_tokens"] == 196
    )
    assert (
        LearnedTaskModule.spec("vlm_image", 112)["scene_tokens"] == 256
        and LearnedTaskModule.spec("cnn", 112)["scene_tokens"] == 49
    )
    with pytest.raises(ValueError):
        SceneEncoder.scene_array(lay, "discs")
    with pytest.raises(ValueError):
        LearnedTaskModule.spec("cnn", 50)


def test_steering_imports_without_transformers():
    # In a fresh interpreter, so the reload does not leave duplicate classes behind for later tests. With
    # transformers unavailable (its import raises ImportError), both modules import and constructing the
    # encoder loads nothing.
    code = (
        "import sys; sys.modules['transformers'] = None; "
        "import lerobot.policies.steering, lerobot.policies.steering.modeling_steering, "
        "lerobot.policies.steering.configuration_steering; "
        "from lerobot.policies.steering import SceneEncoder; SceneEncoder(device='cpu')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # nosec B603


@pytest.mark.parametrize("where", ["dir", "repo"])
def test_encoder_without_weights_raises_before_building_a_policy(where, tmp_path):
    # PI05Policy.from_pretrained returns a randomly initialised policy when model.safetensors is missing; the
    # encoder must refuse instead of producing tokens from random weights.
    if where == "dir":
        (tmp_path / "config.json").write_text("{}")
        path = str(tmp_path)
    else:
        path = "lerobot/no_such_checkpoint"  # not in the local HF cache
    enc = SceneEncoder(path=path, device="cpu")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        enc.image_tokens(np.zeros((1, 3, 224, 224), np.uint8))
    assert enc._policy is None


@pytest.mark.skipif(
    os.environ.get("MARS_SCENE_REAL") != "1",
    reason="loads lerobot/pi05_base (14 GB); MARS_SCENE_REAL=1 to run",
)
def test_real_pi05_image_tokens_have_the_documented_shape():
    enc = SceneEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    tok = SceneEncoder.scene_array(_layouts(2), "vlm_image", SCENE_SIZE, enc)
    pre = SceneEncoder.scene_array(_layouts(2), "vlm_prefix", SCENE_SIZE, enc)
    assert (
        tok.shape == (2, 256, 2048) and tok.dtype == np.float16 and np.isfinite(tok.astype(np.float32)).all()
    )
    assert pre.shape == (2, 256, 2048) and not np.array_equal(tok, pre)
    images = render_layouts(_layouts(2), SCENE_SIZE)
    assert np.array_equal(enc.encode(images).cpu().numpy(), enc.image_tokens(images))
    assert np.isfinite(pre.astype(np.float32)).all()
    # the tokens carry the scene: other layouts, another obstacle task, give other tokens; the same layouts the same
    other = SceneEncoder.scene_array(_layouts(2, seed=1), "vlm_image", SCENE_SIZE, enc)
    # T3 wall against T2 disc at one seed: the same targets and an unrendered start, so only the obstacles differ
    wall_spec = {
        "kind": "wall",
        "along": 0.5,
        "gap": 0.16,
        "gap_offset": 0.14,
        "radius": 0.015,
        "extent": 0.4,
    }
    wall_layouts = _layouts(2, obstacle=wall_spec)
    assert np.array_equal(wall_layouts.targets, _layouts(2).targets)
    wall = SceneEncoder.scene_array(wall_layouts, "vlm_image", SCENE_SIZE, enc)
    again = SceneEncoder.scene_array(_layouts(2), "vlm_image", SCENE_SIZE, enc)
    assert not np.array_equal(tok, other) and not np.array_equal(tok, wall)
    assert not np.array_equal(tok[0], tok[1])  # the two layouts of one batch differ
    assert np.array_equal(tok, again)
    enc.close()


class _PatchEncoder:
    """Tokens that depend on the picture: token j's first three dims are patch j's mean colour / 255."""

    def encode(self, images, batch=256):
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.nn.functional.pad(pooled, (0, 2045)).to(torch.float16)


def test_the_encoder_batches_and_keeps_the_view_order():
    images = np.zeros((2, 2, 3, 32, 32), np.uint8)
    images[:, 1] = 255
    tokens = SceneEncoder.view_tokens(images, _PatchEncoder())
    assert tokens.shape == (2, 512, 2048) and tokens.dtype == torch.float16
    assert float(tokens[:, :256, :3].max()) == 0.0 and float(tokens[:, 256:, :3].min()) == 1.0

    class Recording(SceneEncoder):  # image_tokens encodes batch by batch and frees each on the CPU
        def __init__(self):
            super().__init__(device="cpu")
            self.sizes = []

        def encode(self, images, batch=256):
            self.sizes.append(len(images))
            return torch.full((len(images), 256, 2048), float(images.mean()), dtype=torch.float16)

    enc = Recording()
    out = enc.image_tokens(
        np.arange(5, dtype=np.uint8)[:, None, None, None] * np.ones((1, 3, 4, 4), np.uint8), batch=2
    )
    assert enc.sizes == [2, 2, 1] and isinstance(out, np.ndarray) and out.dtype == np.float16
    np.testing.assert_array_equal(out[:, 0, 0], np.array([0.5, 0.5, 2.5, 2.5, 4.0], np.float16))


def test_token_reduction_keeps_the_top_directions_centred_or_not():
    g = torch.Generator().manual_seed(0)
    true = torch.linalg.qr(torch.randn(16, 3, generator=g))[0]
    x = (torch.randn(4000, 3, generator=g) * torch.tensor([5.0, 2.0, 1.0])) @ true.T + 0.7
    mean, basis, kept = SceneEncoder.fit_reduction(iter(x.split(500)), 3)
    assert (
        mean.shape == (16,)
        and basis.shape == (16, 3)
        and mean.dtype == basis.dtype == torch.float32
        and kept > 0.999
    )
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(mean, x.mean(0), atol=1e-4, rtol=0)
    assert float(((x - mean) @ basis @ basis.T + mean - x).abs().max()) < 1e-3
    assert abs(float(basis[:, 0] @ true[:, 0])) > 0.99
    assert (
        0.9 < SceneEncoder.fit_reduction(iter(x.split(500)), 2)[2] < 0.999
    )  # 29 of 30 parts of the variance
    torch.testing.assert_close(
        SceneEncoder.fit_reduction(iter(x.reshape(8, 500, 16)), 3)[0], mean, atol=1e-5, rtol=0
    )
    for bad in (16, 0):
        with pytest.raises(ValueError):
            SceneEncoder.fit_reduction(iter(x.split(500)), bad)
    # uncentred: the second moment's top directions, the large constant part (as pi0.5's tokens have) first
    g = torch.Generator().manual_seed(0)
    offset = torch.linalg.qr(torch.randn(16, 1, generator=g))[0][:, 0]
    x = torch.randn(4000, 16, generator=g) + 50.0 * offset  # a large constant part, as pi0.5's tokens have
    mean, basis, kept = SceneEncoder.fit_reduction(iter(x.split(500)), 3, center=False)
    assert mean.shape == (16,) and mean.dtype == basis.dtype == torch.float32 and not mean.any()
    assert basis.shape == (16, 3) and abs(float(basis[:, 0] @ offset)) > 0.99
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)
    second = (x.double().T @ x.double()) / len(x)
    values = torch.linalg.eigvalsh(second)
    assert abs(kept - float(values[-3:].sum() / values.sum())) < 1e-6 and kept > 0.99
    centred, default = (
        SceneEncoder.fit_reduction(iter(x.split(500)), 3, center=True),
        SceneEncoder.fit_reduction(iter(x.split(500)), 3),
    )
    assert (
        all(torch.equal(a, b) for a, b in zip(centred[:2], default[:2], strict=True))
        and centred[2] == default[2]
    )
    torch.testing.assert_close(centred[0], x.mean(0), atol=1e-4, rtol=0)
