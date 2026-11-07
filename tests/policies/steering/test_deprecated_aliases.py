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
"""The package's deprecated aliases (kept for one round): each returns what the method that replaced it returns on
the same inputs, and says it is deprecated."""

import numpy as np
import torch

import lerobot.policies.steering as steering
from lerobot.envs.cursor import TaskConfig, obstacle_spec, sample_layouts
from lerobot.policies.steering import SceneEncoder


def _layouts(n=3, seed=0):
    task = TaskConfig(
        min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    return sample_layouts(np.random.default_rng(seed), n, 4, task)


class _Encoder(SceneEncoder):
    """A SceneEncoder whose tokens are a fixed function of the pictures (nothing loaded)."""

    def __init__(self):
        super().__init__(device="cpu")

    def encode(self, images, batch=256):
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.nn.functional.pad(pooled, (0, 2045)).to(torch.float16)

    def image_tokens(self, images, batch=64):
        return self.encode(images).numpy()

    def prefix_tokens(self, images, batch=64):
        return -self.image_tokens(images)


def test_every_alias_is_marked_deprecated():
    for name in ("scene_for", "encode_views", "fit_token_reduction"):
        assert getattr(steering, name).__doc__.startswith("Deprecated alias of"), name


def test_scene_encoder_aliases_equal_its_methods():
    lay, enc = _layouts(), _Encoder()
    for source in ("cnn", "vlm_image", "vlm_prefix"):
        want = SceneEncoder.scene_array(lay, source, 32, enc)
        assert np.array_equal(steering.scene_for(lay, source, 32, enc), want)
        if source != "cnn":
            assert np.array_equal(enc.scene_for(lay, source, 32), want)
    views = np.random.default_rng(0).integers(0, 255, (2, 2, 3, 32, 32), dtype=np.uint8)
    want = SceneEncoder.view_tokens(views, enc)
    assert torch.equal(steering.encode_views(views, enc), want) and torch.equal(enc.encode_views(views), want)
    x = torch.randn(600, 16, generator=torch.Generator().manual_seed(0))
    for center in (True, False):
        new = SceneEncoder.fit_reduction(iter(x.split(200)), 3, center)
        old = steering.fit_token_reduction(iter(x.split(200)), 3, center=center)
        assert torch.equal(new[0], old[0]) and torch.equal(new[1], old[1]) and new[2] == old[2]
