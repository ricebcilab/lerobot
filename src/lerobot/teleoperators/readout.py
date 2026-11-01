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
"""The operator's readout: what a corruption and input noise do to a teleop command on its way to the
controller, shared by the robot (pi0.5 / LIBERO) and the cursor shared-autonomy examples.

A command is a deflection in [-1, 1] per axis; inside ``DEADBAND`` (max |component|) the operator counts as idle,
and idle input stays exactly idle at every stage. The stages, in order:

- a matrix corruption: ``clip(M @ raw)``;
- or a curl field on the command: the served command is a velocity v driven by the push u through
  dv/dt = (u - v) / tau + curl * R v (R a clockwise quarter turn in the first two axes, any further axis a plain
  lag), stepped exactly over each control step (``curl_discretization``, ``curl_step``), then clipped;
- Gaussian noise of std sigma per axis, clipped (``add_noise``).

Everything is numpy and batched over leading axes: a command is (..., n), a matrix (n, n) or one per row
(..., n, n), sigma a scalar or one per row.
"""

import numpy as np

DEADBAND = 0.05  # max |component| below which the operator is idle

_ROT = np.array([[0.0, 1.0], [-1.0, 0.0]])  # a clockwise quarter turn in the first two axes


def is_active(command, deadband: float = DEADBAND) -> np.ndarray:
    """True where a command (..., n) is outside the deadband."""
    return np.max(np.abs(np.asarray(command, dtype=np.float64)), axis=-1) >= deadband


def corrupt(raw, matrix=None, deadband: float = DEADBAND) -> np.ndarray:
    """A matrix corruption: ``clip(M @ raw)`` where the raw command (..., n) is active, the raw command elsewhere;
    ``matrix`` (n, n), one per row (..., n, n), or None (the command unchanged)."""
    raw = np.asarray(raw, dtype=np.float64)
    if matrix is None:
        return raw.copy()
    out = np.clip(np.einsum("...ij,...j->...i", np.asarray(matrix, dtype=np.float64), raw), -1.0, 1.0)
    return np.where(is_active(raw, deadband)[..., None], out, raw)


def curl_discretization(tau: float, curl: float, dt: float, n_dims: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(decay, drive), (n_dims, n_dims): the exact one-step update v <- decay v + drive u of
    dv/dt = (u - v) / tau + curl * R v for a push u held over ``dt`` s; R turns the first two axes clockwise
    (+ curl = clockwise), any further axis is a plain first-order lag with time constant ``tau``."""
    if not (np.isfinite(curl) and tau > 0 and dt > 0 and n_dims >= 2):
        raise ValueError(
            f"a curl needs a finite curl, positive tau and dt, and n_dims >= 2; got {curl}, {tau}, {dt}, {n_dims}"
        )
    eye = np.eye(2)
    decay2 = np.exp(-dt / tau) * (np.cos(curl * dt) * eye + np.sin(curl * dt) * _ROT)
    m_inv = (-eye / tau - curl * _ROT) / (1 / tau**2 + curl**2)
    drive2 = m_inv @ (decay2 - eye) / tau
    if n_dims == 2:
        return decay2, drive2
    lag = np.exp(-dt / tau)
    decay, drive = np.eye(n_dims) * lag, np.eye(n_dims) * (1.0 - lag)
    decay[:2, :2], drive[:2, :2] = decay2, drive2
    return decay, drive


def curl_steady_state(tau: float, curl: float, n_dims: int = 2) -> np.ndarray:
    """(n_dims, n_dims): the map from a held push to the steady-state served velocity, a clockwise turn of the
    first two axes by atan(tau * curl) and a slowdown by 1 / sqrt(1 + (tau * curl)^2); further axes unchanged."""
    k = tau * curl
    s = np.eye(n_dims)
    s[:2, :2] = (np.eye(2) + k * _ROT) / (1 + k**2)
    return s


def curl_step(vel, raw, decay: np.ndarray, drive: np.ndarray) -> np.ndarray:
    """One control step of the curl field: v <- decay v + drive u, for velocities and pushes (..., n). Every push
    drives it, idle ones too (the field keeps evolving while the operator rests)."""
    return np.asarray(vel, dtype=np.float64) @ decay.T + np.asarray(raw, dtype=np.float64) @ drive.T


def curl_serve(vel, raw, deadband: float = DEADBAND) -> np.ndarray:
    """What a curl readout serves: the field's velocity, clipped, where the raw push is active; the push elsewhere."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.where(
        is_active(raw, deadband)[..., None], np.clip(np.asarray(vel, dtype=np.float64), -1.0, 1.0), raw
    )


def add_noise(core, raw, sigma, noise, deadband: float = DEADBAND) -> np.ndarray:
    """The readout's last stage: ``core`` (..., n) (the corrupted command) plus ``sigma`` times the standard normal
    draws ``noise`` (..., n) per axis, clipped, where the raw push is active; the raw push elsewhere. ``sigma`` a
    scalar or one per row (...,). The caller draws ``noise``, so each keeps its own random stream (the robot draws
    only while the operator pushes, the cursor on every step)."""
    core, raw = np.asarray(core, dtype=np.float64), np.asarray(raw, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    noisy = np.clip(
        core + np.asarray(noise, dtype=np.float64) * (sigma[..., None] if sigma.ndim else sigma), -1.0, 1.0
    )
    return np.where(is_active(raw, deadband)[..., None], noisy, raw)
