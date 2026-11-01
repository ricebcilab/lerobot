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

from lerobot.teleoperators import readout

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
    """cursor_sa.readout.Readout (a matrix corruption) before the shared core."""
    out = np.clip(np.einsum("...ij,...j->...i", matrix, raw), -1.0, 1.0)
    out = np.clip(out + rng.normal(size=out.shape) * sigma, -1.0, 1.0)
    return np.where((np.abs(raw).max(-1) >= DEADBAND)[..., None], out, raw)


def old_robot_noisy(clean, std, rng):
    """libero/teleop.py NoisyReader before the shared core."""
    if std == 0.0 or np.max(np.abs(clean)) < DEADBAND:
        return clean
    return np.clip(clean + rng.normal(0.0, std, size=3), -1.0, 1.0)


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


def test_matrix_corruption_and_noise_are_the_cursor_readout():
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    raw = np.random.default_rng(1).uniform(-1, 1, (5, 2))
    raw[2] = 0.01  # idle: served as is
    m = np.array([[1.0, 0.0], [0.0, 0.25]])
    sigma = np.array([0.0, 0.5, 1.0, 1.5, 2.0])[:, None]
    old = old_cursor_readout(raw, m, sigma, rng_a)
    new = readout.add_noise(readout.corrupt(raw, m), raw, sigma[:, 0], rng_b.normal(size=raw.shape))
    np.testing.assert_array_equal(new, old)
    np.testing.assert_array_equal(new[2], raw[2])


def test_noise_is_the_robot_noisy_reader():
    for clean in (np.array([0.4, -0.2, 0.9]), np.array([0.01, 0.0, -0.02])):
        rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
        old = old_robot_noisy(clean, 0.6, rng_a)
        new = (
            clean
            if not readout.is_active(clean)
            else readout.add_noise(clean, clean, 0.6, rng_b.normal(size=3))
        )
        np.testing.assert_array_equal(new, old)
        assert rng_a.random() == rng_b.random()  # the same draws spent


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
