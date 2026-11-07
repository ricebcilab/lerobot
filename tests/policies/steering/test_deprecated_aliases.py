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
import pytest
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


# ---------------------------------------------------------------- the adapter modules


def _perturbed(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g))
    return module


def test_module_method_aliases_equal_the_methods():
    from lerobot.policies.steering import LearnedTaskModule, LearnedUserModule

    g = torch.Generator().manual_seed(5)
    user = _perturbed(LearnedUserModule(6, n_dims=2, d=16, tdim=8, cross_step_rank=1), 1)
    task = _perturbed(LearnedTaskModule(horizon=6, d=16, tdim=8, heads=2, layers=1), 2)
    x, v, t = torch.randn(3, 6, 2, generator=g), torch.randn(3, 6, 2, generator=g), torch.rand(3, generator=g)
    matrices = lambda x_t, t_: torch.eye(2).expand(3, 6, 2, 2) * 1.5  # noqa: E731 (a callable without velocity)
    for u in (user, matrices):
        assert torch.equal(
            steering.apply_user_module(u, x, t, v), LearnedUserModule.apply_correction(u, x, t, v)
        )
    velocity = lambda x_t, t_: torch.tanh(x_t)  # noqa: E731
    latent, target, weight = (
        torch.randn(3, 6, 2, generator=g),
        torch.randn(3, 6, 2, generator=g),
        torch.ones(3),
    )
    for kw in ({}, {"task": task, "weight": weight}):
        assert torch.equal(
            steering.user_field_loss(user, velocity, latent, target, t, **kw),
            LearnedUserModule.field_loss(user, velocity, latent, target, t, **kw),
        )
    old, new = (
        fn(task, velocity, target, latent, t, 0.1, 2)
        for fn in (steering.task_field_loss, LearnedTaskModule.field_loss)
    )
    assert torch.equal(old[0], new[0]) and old[1] == new[1]
    tokens, mean, basis = (
        torch.randn(2, 4, 16, generator=g),
        torch.randn(16, generator=g),
        torch.randn(16, 3, generator=g),
    )
    assert torch.equal(
        steering.reduce_tokens(tokens, mean, basis), LearnedTaskModule.reduce(tokens, mean, basis)
    )
    for name in (
        "apply_user_module",
        "user_field_loss",
        "task_field_loss",
        "reduce_tokens",
        "save_adapter_module",
        "load_adapter_module",
        "convert_cursor_module",
    ):
        assert getattr(steering, name).__doc__.startswith("Deprecated alias of"), name


def test_checkpoint_aliases_write_and_read_the_same_files(tmp_path):
    from lerobot.policies.steering import AdapterModules, LearnedTaskModule, LearnedUserModule

    user = _perturbed(LearnedUserModule(6, n_dims=2, d=16, tdim=8), 3)
    a = steering.save_adapter_module(user, tmp_path / "a.pt", "user", meta={"k": 1})
    b = AdapterModules.save_module(user, tmp_path / "b.pt", "user", meta={"k": 1})
    c = AdapterModules(user=user).save(tmp_path / "c.pt", meta={"k": 1})  # the one learned module's slot
    saved = [
        torch.load(p, weights_only=True) for p in (a, b, c)
    ]  # the zip records each file's name: compare contents
    for ckpt in saved[1:]:
        assert {k: v for k, v in ckpt.items() if k != "state_dict"} == {
            k: v for k, v in saved[0].items() if k != "state_dict"
        }
        assert all(torch.equal(v, ckpt["state_dict"][k]) for k, v in saved[0]["state_dict"].items())
    old, new = steering.load_adapter_module(a), AdapterModules.load_module(a)
    assert old[1:] == new[1:] == ("user", {"k": 1})
    assert all(torch.equal(v, new[0].state_dict()[k]) for k, v in old[0].state_dict().items())
    loaded = AdapterModules.load(c)
    assert loaded.kind == "user" and loaded.meta == {"k": 1} and not loaded.training
    with pytest.raises(ValueError, match="kind must be one of"):
        AdapterModules(user=user).save(tmp_path / "d.pt", "task")
    with pytest.raises(ValueError, match="needs a kind"):
        AdapterModules(user=user, task=LearnedTaskModule(horizon=6)).save(tmp_path / "e.pt")
    cursor = {"config": {"horizon": 6, "d": 16, "tdim": 8}, "state_dict": user.state_dict(), "args": {"x": 1}}
    assert (
        steering.convert_cursor_module(cursor).keys() == AdapterModules.from_cursor_checkpoint(cursor).keys()
    )
    assert (
        steering.convert_cursor_module(cursor)["config"]
        == AdapterModules.from_cursor_checkpoint(cursor)["config"]
    )


# ---------------------------------------------------------------- the steering cores


def test_core_aliases_equal_the_methods():
    from types import SimpleNamespace

    from lerobot.policies.steering import AdaptedField, FlowControlSteering, FlowReversalSteering

    assert steering.FlowControlPolicy is FlowControlSteering
    assert steering.FlowReversalSteeringPolicy is FlowReversalSteering
    g = torch.Generator().manual_seed(6)
    x, written = torch.randn(3, 6, 7, generator=g), torch.randn(3, 6, 3, generator=g)
    steer = torch.tensor([True, False, True])[:, None, None]
    for step in (0, 2):
        old = steering.flow_control_write(x.clone(), step, written, 4, 2, slice(0, 3), steer)
        assert torch.equal(old, FlowControlSteering.write(x.clone(), step, written, 4, 2, slice(0, 3), steer))
    for dims, pinned in ((("translation",), 2), (None, None), (("translation", "gripper"), None)):
        assert torch.equal(
            steering.delegation_mask(6, 9, dims, pinned),
            FlowReversalSteering.delegation_mask(6, 9, dims, pinned),
        )
    stats = {
        "action": {"q01": np.arange(7.0), "q99": np.arange(7.0) + 2, "mean": np.zeros(7), "std": np.ones(7)}
    }
    for mode in (None, "QUANTILES"):
        post = SimpleNamespace(steps=[SimpleNamespace(stats=stats, norm_map={"ACTION": mode})])
        old, new = steering.get_action_mean_std(post), FlowControlSteering.action_mean_std(post)
        assert np.array_equal(old[0], new[0]) and np.array_equal(old[1], new[1])
    field = AdaptedField(
        None, task_module=lambda x_t, t, v: (torch.eye(7).expand(3, 6, 7, 7), torch.ones_like(v))
    )
    v = torch.randn(3, 6, 7, generator=g)
    assert torch.equal(
        steering.task_velocity_hook(field)(0, 0.5, x, v),
        FlowControlSteering.velocity_hook(field)(0, 0.5, x, v),
    )
    assert (
        steering.task_velocity_hook(None) is None
        and FlowReversalSteering.velocity_hook(AdaptedField(None)) is None
    )
    for name in ("flow_control_write", "delegation_mask", "task_velocity_hook", "get_action_mean_std"):
        assert getattr(steering, name).__doc__.startswith("Deprecated alias of"), name
