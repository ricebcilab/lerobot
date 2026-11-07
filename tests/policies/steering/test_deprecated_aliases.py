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


# ---------------------------------------------------------------- Flow


def _field():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(7, 7, generator=g) * 0.3

    def velocity(x, t):
        t = torch.as_tensor(t, dtype=x.dtype).reshape(-1, 1, 1) if torch.is_tensor(t) else t
        return torch.tanh(x @ w) * (1 + t)

    return velocity, torch.randn(3, 6, 7, generator=g), torch.eye(7) + 0.1 * torch.randn(7, 7, generator=g)


def test_flow_aliases_equal_its_methods():
    from lerobot.policies.steering import Flow

    velocity, x, f = _field()
    for batch_time in (False, True):
        for backward in (False, True):
            assert torch.equal(
                steering.integrate(x, velocity, 0.2, 0.1, 4, f, backward=backward, batch_time=batch_time),
                Flow(velocity, f, batch_time=batch_time).integrate(x, 0.2, 0.1, 4, backward=backward),
            )
    for n in (None, 3):
        assert torch.equal(steering.reverse_flow(x, velocity, 5, f, n), Flow(velocity, f).reverse(x, 5, n))
        assert torch.equal(steering.forward_flow(x, velocity, 5, n, f), Flow(velocity, f).forward(x, 5, n))
        for noise_time in (1.0, 0.0):
            assert steering.frs_schedule(5, n, noise_time) == Flow.schedule(5, n, noise_time)
            delegated = torch.zeros(6, 7, dtype=torch.bool)
            delegated[:, 3:] = True
            noise = torch.randn(3, 6, 7, generator=torch.Generator().manual_seed(4))
            assert torch.equal(
                steering.frs_latent(velocity, x, noise, 5, n, f, delegated, noise_time, batch_time=True),
                Flow(velocity, f, noise_time, batch_time=True).latent(x, noise, 5, n, delegated),
            )
    for noise_time in (1.0, 0.0):
        assert torch.equal(
            steering.regenerate(velocity, x, 4, f, noise_time), Flow(velocity, f, noise_time).regenerate(x, 4)
        )
    assert torch.equal(steering.heading_cosine(x, x.flip(0)), Flow.heading_cosine(x, x.flip(0)))
    args = (np.array([[0.3, -0.2, 0.1]]), 0.5, np.zeros(7), np.ones(7) * 2, 6, 9)
    assert torch.equal(steering.uniform_reference(*args), Flow.uniform_reference(*args))
    v = velocity(x, 0.3)
    assert torch.equal(
        steering.modeling_steering._adapt_velocity(f, x, 0.3, v, True),
        Flow.adapt_velocity(f, x, 0.3, v, True),
    )


def test_a_cursor_convention_flow_reverses_and_runs_forward_from_t_one():
    from lerobot.policies.steering import Flow

    velocity, x, f = _field()
    flow = Flow(velocity, f, noise_time=0.0, batch_time=True)
    assert torch.equal(flow.reverse(x, 5, 3), flow.integrate(x, 1.0, -1.0 / 5, 3, backward=True))
    assert torch.equal(flow.forward(x, 5, 2), flow.integrate(x, 0.0, 1.0 / 5, 3))


def test_paired_chunks_alias_restores_the_generator_as_the_method():
    from lerobot.policies.steering import Flow

    class Sampler:
        def predict_action_chunk(self, batch, **kw):
            return torch.randn(2, 3) + kw.get("shift", 0.0)

    for fn in (steering.paired_chunks, Flow.paired_chunks):
        torch.manual_seed(0)
        steered, plain = fn(Sampler(), None, True, None, shift=1.0)
        after = torch.randn(1)
        torch.manual_seed(0)
        assert torch.equal(steered - 1.0, plain) and torch.equal(torch.randn(2, 3), plain)
        assert torch.equal(torch.randn(1), after)
    for name in (
        "integrate",
        "reverse_flow",
        "forward_flow",
        "frs_latent",
        "frs_schedule",
        "regenerate",
        "uniform_reference",
        "heading_cosine",
        "paired_chunks",
    ):
        assert getattr(steering, name).__doc__.startswith("Deprecated alias of"), name
