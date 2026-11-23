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
"""The shared operator readout (`lerobot.teleoperators.readout`) against the rules the cursor and the robot
examples each implemented before it (frozen copies below): the same numbers, bit for bit, for the same draws."""

import numpy as np
import pytest
import torch

from lerobot.policies.steering import SteeringConfig
from lerobot.teleoperators import readout
from lerobot.teleoperators.readout import corruption_spec, matrix_spec

DEADBAND = 0.05
ROT = np.array([[0.0, 1.0], [-1.0, 0.0]])


# ---------------------------------------------------------------- the rules as each example had them


def old_cursor_curl(tau, curl, dt):
    eye = np.eye(2)
    decay = np.exp(-dt / tau) * (np.cos(curl * dt) * eye + np.sin(curl * dt) * ROT)
    m_inv = (-eye / tau - curl * ROT) / (1 / tau**2 + curl**2)
    return decay, m_inv @ (decay - eye) / tau


def old_robot_curl(tau, curl, dt):
    decay2, drive2 = old_cursor_curl(tau, curl, dt)
    lag = np.exp(-dt / tau)
    decay, drive = np.eye(3), np.eye(3)
    decay[:2, :2], decay[2, 2] = decay2, lag
    drive[:2, :2], drive[2, 2] = drive2, 1.0 - lag
    return decay, drive


def old_cursor_readout(raw, matrix, sigma, rng):
    """The cursor example's Readout (a matrix corruption) before the shared core."""
    out = np.clip(np.einsum("...ij,...j->...i", matrix, raw), -1.0, 1.0)
    out = np.clip(out + rng.normal(size=out.shape) * sigma, -1.0, 1.0)
    return np.where((np.abs(raw).max(-1) >= DEADBAND)[..., None], out, raw)


# ---------------------------------------------------------------- the checks


@pytest.mark.parametrize("tau,curl", [(0.1, 20.0), (0.1, -5.0), (0.3, 0.0)])
def test_curl_is_both_examples_curl(tau, curl):
    for got, old in zip(
        readout.curl_discretization(tau, curl, 0.02, 2), old_cursor_curl(tau, curl, 0.02), strict=True
    ):
        np.testing.assert_array_equal(got, old)
    for got, old in zip(
        readout.curl_discretization(tau, curl, 0.05, 3), old_robot_curl(tau, curl, 0.05), strict=True
    ):
        np.testing.assert_array_equal(got, old)
    k = tau * curl
    np.testing.assert_allclose(readout.curl_steady_state(tau, curl, 2), (np.eye(2) + k * ROT) / (1 + k**2))
    # held long enough, the stepped field reaches its steady state
    decay, drive = readout.curl_discretization(tau, curl, 0.02, 3)
    v, u = np.zeros(3), np.array([0.5, 0.2, -0.3])
    for _ in range(2000):
        v = readout.curl_step(v, u, decay, drive)
    np.testing.assert_allclose(v, readout.curl_steady_state(tau, curl, 3) @ u, atol=1e-9)


def test_matrix_corruption_and_noise_are_the_cursor_readout_on_active_pushes():
    """The frozen copy noised active pushes only; on those the shared core is it (idle ones: the next section)."""
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    raw = np.random.default_rng(1).uniform(-1, 1, (5, 2))
    m = np.array([[1.0, 0.0], [0.0, 0.25]])
    sigma = np.array([0.0, 0.5, 1.0, 1.5, 2.0])[:, None]
    old = old_cursor_readout(raw, m, sigma, rng_a)
    new = readout.add_noise(readout.corrupt(raw, m), sigma[:, 0], rng_b.normal(size=raw.shape))
    np.testing.assert_array_equal(new, old)


def test_a_3d_curl_readout_steps_and_reads_apart():
    """The robot reads once per chunk and steps once per control step: ``step`` then ``read`` is a call, and the
    third axis is a plain lag (``old_robot_curl``); the spec's ``dt`` is the readout's."""
    matrix, curl = corruption_spec({"curl": 20, "tau": 0.1, "dt": 0.05}, n_dims=3)
    assert matrix is None and curl == {"curl": 20.0, "tau": 0.1, "dt": 0.05}
    decay, drive = old_robot_curl(0.1, 20.0, 0.05)
    live, called = readout.Readout(curl=curl, n_dims=3), readout.Readout(curl=curl, n_dims=3)
    live.reset(None)
    push, vel = np.array([0.6, -0.3, 0.8]), np.zeros(3)
    np.testing.assert_array_equal(live.read(push), np.zeros(3))  # at rest until the field is stepped
    for _ in range(10):
        live.step(push)
        vel = decay @ vel + drive @ push
        served = called(push)
    np.testing.assert_allclose(live.read(push), np.clip(vel, -1.0, 1.0), atol=1e-15)
    np.testing.assert_array_equal(live.read(push), served)
    assert np.array_equal(corruption_spec({"rotation_z_deg": 90}, n_dims=3)[0], readout.rotation(90))


def test_corruption_is_per_row_and_leaves_idle_input_alone():
    raw = np.array([[0.5, 0.5], [0.02, 0.03]])
    per_row = np.stack([np.eye(2), 3 * np.eye(2)])
    np.testing.assert_array_equal(readout.corrupt(raw, per_row), [[0.5, 0.5], [0.02, 0.03]])
    np.testing.assert_array_equal(readout.corrupt(np.array([0.5, 0.5]), 3 * np.eye(2)), [1.0, 1.0])  # clipped
    np.testing.assert_array_equal(
        readout.curl_serve(np.array([[2.0, -2.0], [0.5, 0.5]]), raw), [[1.0, -1.0], raw[1]]
    )
    with pytest.raises(ValueError, match="curl"):
        readout.curl_discretization(0.0, 1.0, 0.02)


# ---------------------------------------------------------------- the stateful readout, its curl and its torch forms


class _Draws:
    """An rng whose normal draws are given, so the numpy readout and its torch twins spend the same numbers."""

    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=np.float64)

    def normal(self, size):
        assert tuple(size) == self.draws.shape
        return self.draws


def test_readout_with_a_matrix_is_corrupt_then_noise():
    g = np.random.default_rng(2)
    raw = g.uniform(-1, 1, (5, 2))
    raw[3] = 0.01  # idle: left as is by the corruption, then noised
    m, sigma = np.array([[0.0, 1.0], [-1.0, 0.0]]) * 1.5, g.uniform(0, 1, 5)
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    served = readout.Readout(m, sigma, rng_a)(raw)
    np.testing.assert_array_equal(
        served, readout.add_noise(readout.corrupt(raw, m), sigma, rng_b.normal(size=(5, 2)))
    )
    per_trial = np.stack([np.eye(2), 0.5 * np.eye(2)])
    chunk = g.uniform(-1, 1, (2, 4, 2))
    np.testing.assert_array_equal(
        readout.Readout(per_trial)(chunk), readout.corrupt(chunk, per_trial[:, None])
    )


def test_readout_with_a_curl_steps_the_field_from_rest():
    curl, dt = {"curl": 20.0, "tau": 0.1}, 0.02
    decay, drive = readout.curl_discretization(curl["tau"], curl["curl"], dt)
    pushes = np.random.default_rng(0).uniform(-1, 1, (3, 25, 2))
    pushes[1, 4] = 0.0  # an idle step: the push served, the field still driven
    live = readout.Readout(curl=curl, dt=dt)
    live.reset(3)
    vel = np.zeros((3, 2))
    for k in range(25):
        vel = readout.curl_step(vel, pushes[:, k], decay, drive)
        np.testing.assert_array_equal(live(pushes[:, k]), readout.curl_serve(vel, pushes[:, k]))
    with pytest.raises(ValueError, match="one command per trial"):
        live(pushes[:, :4])
    with pytest.raises(ValueError, match="not both"):
        readout.Readout(np.eye(2), curl=curl)


@pytest.mark.parametrize("kind", ["matrix", "curl"])
def test_core_is_the_readout_before_noise_from_rest(kind):
    corruption = np.array([[1.2, 0.3], [-0.4, 0.9]]) if kind == "matrix" else None
    curl = {"curl": -8.0, "tau": 0.2} if kind == "curl" else None
    pushes = np.random.default_rng(4).uniform(-1, 1, (3, 12, 2))
    pushes[0, 2] = 0.01
    core = readout.Readout(corruption, curl=curl).core(pushes)
    assert core.shape == pushes.shape
    live = readout.Readout(corruption, curl=curl)
    live.reset(3)
    stepped = np.stack([live(pushes[:, k]) for k in range(12)], 1)
    active = readout.is_active(pushes)
    np.testing.assert_allclose(
        core[active], stepped[active], atol=1e-12
    )  # idle steps: core is the rule's, served raw
    if kind == "none":
        assert core is pushes
    if kind == "matrix":
        np.testing.assert_array_equal(core, np.clip(pushes @ corruption.T, -1.0, 1.0))
    with pytest.raises(ValueError, match="one matrix"):
        readout.Readout(np.stack([np.eye(2)] * 3)).core(pushes)


def test_noise_applies_at_every_read_an_idle_push_included():
    """A decoder's noise does not know the operator rests: an idle push, and a push the corruption shrank to idle,
    are served as noise, from the same draws an active push gets (numpy and torch alike); with no noise an idle
    push stays idle."""
    raw = np.array([[0.4, -0.2, 0.9], [0.1, 0.0, -0.02], [0.0, 0.0, 0.0]])  # active, shrunk to idle, idle
    scale = np.diag([0.25, 1.0, 1.0])
    draws = np.random.default_rng(3).normal(size=raw.shape)
    served = readout.Readout(scale, 0.6, _Draws(draws), n_dims=3)(raw)
    np.testing.assert_allclose(served, np.clip(readout.corrupt(raw, scale) + 0.6 * draws, -1.0, 1.0))
    core = torch.tensor(readout.corrupt(raw, scale))
    np.testing.assert_allclose(
        readout.Readout.serve(core, torch.full((3,), 0.6), torch.tensor(draws)).numpy(), served, atol=1e-12
    )
    np.testing.assert_array_equal(readout.Readout(scale, 0.0, n_dims=3)(raw)[2], 0.0)


def test_serve_is_the_numpy_last_stage_for_the_same_draws():
    g = np.random.default_rng(5)
    raw = g.uniform(-1, 1, (6, 2))
    raw[1] = 0.02  # idle
    m, sigma, noise = np.diag([1.0, 0.25]), g.uniform(0, 2, 6), g.normal(size=(6, 2))
    served = readout.Readout(m, sigma, _Draws(noise))(raw)
    core = readout.Readout(m).core(raw[:, None])[:, 0]
    out = readout.Readout.serve(torch.tensor(core), torch.tensor(sigma), torch.tensor(noise))
    np.testing.assert_allclose(out.numpy(), served, atol=1e-12)
    fresh = readout.Readout.serve(
        torch.tensor(core), torch.zeros(6)
    )  # no noise: the core, the idle row's too
    np.testing.assert_allclose(fresh.numpy(), core, atol=1e-12)


@pytest.mark.parametrize("matrix", [None, "tensor"])
def test_chunk_is_the_readout_on_a_chunk_for_the_same_draws(matrix):
    g = np.random.default_rng(6)
    chunk = g.uniform(-1, 1, (4, 5, 2))
    chunk[2, 3] = 0.01  # an idle step
    m = np.array([[0.8, -0.6], [0.6, 0.8]]) * 1.3
    corruption = None if matrix is None else (m if matrix == "numpy" else torch.tensor(m))
    sigma = g.uniform(0, 1, 4)
    torch.manual_seed(11)
    out = readout.Readout(corruption).chunk(torch.tensor(chunk), torch.tensor(sigma))
    torch.manual_seed(11)
    draws = torch.randn(4, 5, 2, dtype=torch.float64).numpy()
    served = readout.Readout(None if matrix is None else m, sigma, _Draws(draws))(chunk)
    np.testing.assert_allclose(out.numpy(), served, atol=1e-12)  # the idle step noised alike
    with pytest.raises(ValueError, match="curl"):
        readout.Readout(curl={"curl": 20.0, "tau": 0.1}).chunk(torch.tensor(chunk), torch.tensor(sigma))


# ---------------------------------------------------------------- the matrix and corruption specs


def test_matrix_spec_forms_per_dimension():
    assert np.array_equal(matrix_spec({"scale": [1.0, 0.25]}, 2), np.diag([1.0, 0.25]))
    assert np.allclose(matrix_spec({"rotation_deg": 90}, 2), [[0, -1], [1, 0]])
    c, s = np.cos(np.radians(20)), np.sin(np.radians(20))
    assert np.array_equal(
        matrix_spec({"rotation_z_deg": 20}, 3), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    )
    with pytest.raises(ValueError, match="rotation_z_deg"):
        matrix_spec({"rotation_deg": 20}, 3)
    with pytest.raises(ValueError, match="rotation_deg"):
        matrix_spec({"rotation_z_deg": 20}, 2)
    assert matrix_spec(None, 2) is None
    assert np.array_equal(matrix_spec({"M": [[0, 1], [1, 0]]}, 2), matrix_spec([[0, 1], [1, 0]], 2))
    with pytest.raises(ValueError, match="scale"):
        matrix_spec({"scale": [1.0, 1.0, 1.0]}, 2)
    with pytest.raises(ValueError, match="2x2"):
        matrix_spec([[1, 0, 0]], 2)


def test_corruption_spec_is_the_examples():
    assert corruption_spec({"curl": 20, "tau": 0.1}) == (None, {"curl": 20.0, "tau": 0.1})
    matrix, curl = corruption_spec({"scale": [1.0, 0.25]})
    assert np.array_equal(matrix, np.diag([1.0, 0.25])) and curl is None
    assert corruption_spec(None) == (None, None)
    for bad in (
        {"curl": 20},
        {"curl": 20, "tau": 0},
        {"curl": float("nan"), "tau": 0.1},
        {"rotation_z_deg": 20},
    ):
        with pytest.raises(ValueError, match="control.corruption"):
            corruption_spec(bad)


def test_build_adapter_two_dims_refuses_blocks_it_has_no_dims_for():
    m = matrix_spec({"scale": [2.0, 0.5]}, 2)
    assert np.array_equal(SteeringConfig.build_adapter({"translation": "corruption"}, m, n_dims=2), m)
    for key in ("orientation", "gripper"):
        with pytest.raises(ValueError, match=key):
            SteeringConfig.build_adapter({"translation": "corruption", key: "identity"}, m, n_dims=2)
    assert np.array_equal(SteeringConfig.build_adapter({"F": {"scale": [3, 3]}}, n_dims=2), 3 * np.eye(2))
    fixed = SteeringConfig.from_spec({"translation": "corruption"}, m, n_dims=2, device="cpu")
    assert fixed.n_dims == 2 and np.array_equal(fixed.fixed_matrix, m)
    # a string user is a checkpoint, {learned: <path>}'s shorthand (the cursor's composite of names)
    named = SteeringConfig.from_spec({"user": "u", "task": "t"}, n_dims=2, device="cpu")
    assert (named.adapter_user, named.adapter_task, named.fixed_matrix) == ("u", "t", None)


def test_quantize_snaps_an_active_push_to_its_largest_axis_at_its_own_length():
    raw = np.array([[0.3, -0.7, 0.2], [0.5, 0.5, 0.1], [0.01, 0.02, 0.0], [0.9, 0.9, 0.9]])
    # the largest component's axis (the first on a tie) at the push's own length, so only its direction is lost;
    # an idle push untouched; a push longer than full deflection capped there
    np.testing.assert_allclose(
        readout.quantize(raw),
        [[0.0, -np.sqrt(0.62), 0.0], [np.sqrt(0.51), 0.0, 0.0], [0.01, 0.02, 0.0], [1.0, 0.0, 0.0]],
    )
    assert readout.quantize_spec({"quantize": "axes"}) and not readout.quantize_spec({"scale": [1, 1, 0]})
    with pytest.raises(ValueError, match="quantize"):
        readout.quantize_spec({"quantize": "octants"})
    with pytest.raises(ValueError, match="quantize"):
        corruption_spec({"quantize": "axes"}, n_dims=3)
    arm = readout.Readout(quantize=True, n_dims=3)
    np.testing.assert_allclose(arm(raw[0]), [0.0, -np.sqrt(0.62), 0.0])
    np.testing.assert_array_equal(arm.core(raw[None]), readout.quantize(raw)[None])
    with pytest.raises(ValueError, match="not both"):
        readout.Readout(np.eye(3), quantize=True, n_dims=3)
