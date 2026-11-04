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

from lerobot.envs.cursor import SCENE_SIZE, TaskConfig, obstacle_spec, sample_layouts
from lerobot.policies.steering import LearnedTaskModule, SceneEncoder, scene_for, scene_spec


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


def test_scene_spec_builds_a_module_that_accepts_scene_for():
    lay = _layouts()
    stub = Stub()
    for src in ("vlm_image", "vlm_prefix", "cnn"):
        spec = scene_spec(src, 112)
        m = LearnedTaskModule(horizon=16, **spec)
        arr = scene_for(lay, src, 112, encoder=stub)
        assert arr.shape[0] == 4 and (arr.dtype == np.uint8 if src == "cnn" else arr.dtype == np.float16)
        obs = {"pos": torch.rand(4, 2), "scene": {src: torch.as_tensor(arr)}}
        a, _ = m(torch.randn(4, 16, 2), torch.rand(4), torch.randn(4, 16, 2), obs)
        assert a.shape == (4, 16, 2, 2)
    assert (
        stub.calls[0] == (4, 3, 112, 112)
        and scene_spec("vlm_image")["scene_tokens"] == 256
        and scene_spec("cnn")["scene_tokens"] == 196
    )
    assert (
        scene_spec("vlm_image", 112)["scene_tokens"] == 256 and scene_spec("cnn", 112)["scene_tokens"] == 49
    )
    with pytest.raises(ValueError):
        scene_for(lay, "discs")
    with pytest.raises(ValueError):
        scene_spec("cnn", 50)


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
    tok = scene_for(_layouts(2), "vlm_image", SCENE_SIZE, encoder=enc)
    pre = scene_for(_layouts(2), "vlm_prefix", SCENE_SIZE, encoder=enc)
    assert (
        tok.shape == (2, 256, 2048) and tok.dtype == np.float16 and np.isfinite(tok.astype(np.float32)).all()
    )
    assert pre.shape == (2, 256, 2048) and not np.array_equal(tok, pre)
    assert np.isfinite(pre.astype(np.float32)).all()
    # the tokens carry the scene: other layouts, another obstacle task, give other tokens; the same layouts the same
    other = scene_for(_layouts(2, seed=1), "vlm_image", SCENE_SIZE, encoder=enc)
    ring = scene_for(
        _layouts(2, obstacle={"kind": "ring", "around": "goal", "gap_angle": [15, 30]}),
        "vlm_image",
        SCENE_SIZE,
        encoder=enc,
    )
    again = scene_for(_layouts(2), "vlm_image", SCENE_SIZE, encoder=enc)
    assert not np.array_equal(tok, other) and not np.array_equal(tok, ring)
    assert not np.array_equal(tok[0], tok[1])  # the two layouts of one batch differ
    assert np.array_equal(tok, again)
    enc.close()
