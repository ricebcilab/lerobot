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

The functions are numpy and batched over leading axes: a command is (..., n), a matrix (n, n) or one per row
(..., n, n), sigma a scalar or one per row. ``Readout`` is the stateful 2D readout of the cursor example built on
them (numpy, closed loop, a curl velocity per trial), with the same rule over a whole sequence from rest
(``Readout.core``) and its torch forms for training (``Readout.chunk``, ``Readout.serve``).
"""

import numpy as np
import torch

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


class Readout:
    """What happens to a command between the operator and the controller:
    raw -> M @ raw (corruption, clipped to [-1, 1]) -> + N(0, input_noise^2) per
    axis (clipped again). Both apply only while the operator is commanding
    (outside DEADBAND), so idle input stays exactly idle.

    With `curl` ({curl, tau}) the corruption is a curl field instead of M: the
    readout keeps a velocity per trial, driven by every raw push (idle ones too)
    and reset by `reset(n_trials)` at the start of a batch of trials; the
    controller gets that velocity (clipped), then the noise. Steps of `dt` s.

    Batched: commands are [B, 2], or [B, H, 2] for a whole chunk per trial (not
    with a curl). `corruption` is one 2x2 matrix or one per trial [B, 2, 2];
    `input_noise` is one std or one per trial [B].
    """

    def __init__(self, corruption=None, input_noise=0.0, rng=None, curl=None, dt=0.02):
        self.corruption, self.input_noise = corruption, np.asarray(input_noise, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.curl, self.vel = curl, None
        self.dt = dt
        if curl is not None:
            if corruption is not None:
                raise ValueError("a readout has a matrix corruption or a curl, not both")
            self._decay, self._drive = curl_discretization(curl["tau"], curl["curl"], dt)

    def reset(self, n_trials=1):
        """A new batch of trials: the curl's velocity starts at rest."""
        self.vel = None if self.curl is None else np.zeros((n_trials, 2))

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=float)
        extra = (1,) * (raw.ndim - 2)  # per-trial parameters broadcast over a chunk's steps
        out = raw
        if self.curl is not None:
            if raw.ndim != 2:
                raise ValueError("a curl readout takes one command per trial and step, [B, 2]")
            if self.vel is None or self.vel.shape != raw.shape:
                self.vel = np.zeros_like(raw)
            self.vel = curl_step(self.vel, raw, self._decay, self._drive)
            out = np.clip(self.vel, -1.0, 1.0)
        if self.corruption is not None:
            m = self.corruption
            out = corrupt(raw, m.reshape(m.shape[:1] + extra + m.shape[1:]) if m.ndim == 3 else m)
        if np.any(self.input_noise):  # a draw per step and axis while the noise is on, active or not
            sigma = (
                self.input_noise.reshape(self.input_noise.shape + extra)
                if self.input_noise.ndim
                else self.input_noise
            )
            return add_noise(out, raw, sigma, self.rng.normal(size=out.shape))
        return np.where(is_active(raw)[..., None], out, raw)

    def core(self, raw):
        """What the readout serves before noise for raw pushes [B, L, 2] given step by step from rest (numpy),
        every step's rule applied whether the push is active or not: ``clip(raw @ M.T)`` per step with one 2x2
        matrix M, a curl's velocity after every step clipped to [-1, 1], or the pushes themselves with no
        corruption. The readout's own curl velocity is left alone (this one integrates from rest)."""
        if self.curl is not None:
            decay, drive = curl_discretization(self.curl["tau"], self.curl["curl"], self.dt)
            vel, out = np.zeros(raw.shape[::2]), np.empty_like(raw, dtype=float)
            for k in range(raw.shape[1]):
                vel = curl_step(vel, raw[:, k], decay, drive)
                out[:, k] = vel
            return np.clip(out, -1.0, 1.0)
        if self.corruption is None:
            return raw
        m = np.asarray(self.corruption)
        if m.shape != (2, 2):
            raise ValueError(f"a readout's core takes one matrix corruption (2, 2), got {m.shape}")
        return np.clip(raw @ m.T, -1.0, 1.0)

    def chunk(self, command, sigma):
        """The readout in torch, for training: deflection chunks `command` [B, H, 2] and a noise std per trial
        `sigma` [B]; the corruption (one 2x2 matrix or None) as a tensor on the command's device. The same rule,
        step by step: M @ raw, clip, + noise, clip, only where the step is active. A curl keeps a velocity over
        steps, so it takes no chunks."""
        if self.curl is not None:
            raise ValueError("a curl readout takes one command per trial and step, not a chunk")
        if self.corruption is None:
            out = command
        else:
            m = torch.as_tensor(self.corruption, dtype=command.dtype, device=command.device)
            out = torch.clamp(command @ m.T, -1.0, 1.0)
        out = torch.clamp(out + sigma[:, None, None] * torch.randn_like(out), -1.0, 1.0)
        active = command.abs().amax(-1, keepdim=True) >= DEADBAND
        return torch.where(active, out, command)

    @staticmethod
    def serve(core, raw, sigma, noise=None):
        """The readout's last stage in torch, for training: `core` [B, 2] (M @ raw clipped, or a curl's
        velocity) plus noise of std `sigma` [B], clipped, only where the raw push [B, 2] is active.
        `noise` [B, 2]: the standard normal draws to scale by sigma (default: fresh ones)."""
        noise = torch.randn_like(core) if noise is None else noise
        out = torch.clamp(core + sigma[:, None] * noise, -1.0, 1.0)
        return torch.where(raw.abs().amax(-1, keepdim=True) >= DEADBAND, out, raw)
