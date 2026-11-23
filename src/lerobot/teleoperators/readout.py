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
and the corruption stages leave idle input idle. The noise does not: a decoder's noise does not know the operator
rests, so it is added at every read, an idle push included. The stages, in order:

- a matrix corruption: ``clip(M @ raw)``;
- or a curl field on the command: the served command is a velocity v driven by the push u through
  dv/dt = (u - v) / tau + curl * R v (R a clockwise quarter turn in the first two axes, any further axis a plain
  lag), stepped exactly over each control step (``curl_discretization``, ``curl_step``), then clipped;
- or a discrete input (``quantize``): the command snapped to the axis of its largest component at its own length,
  so only its direction is lost: the nearest of the 2n key directions;
- Gaussian noise of std sigma per axis, clipped (``add_noise``), at every read.

The functions are numpy and batched over leading axes: a command is (..., n), a matrix (n, n) or one per row
(..., n, n), sigma a scalar or one per row. ``Readout`` is the stateful readout both examples run, built on them
(numpy, closed loop, a curl velocity per trial; ``n_dims`` 2 on the cursor, 3 for the robot's translation), with the
same rule over a whole sequence from rest (``Readout.core``) and its torch forms for training (``Readout.chunk``,
``Readout.serve``). Both examples share the noise rule: a draw per read and axis while the noise is on.

The specs that configure a readout are parsed here too, for any number of dims: ``matrix_spec`` builds every matrix
of a config (a corruption M, and the fixed adapter F's blocks that ``SteeringConfig.build_adapter`` reads),
``corruption_spec`` a corruption (a matrix or a curl), ``quantize_spec`` says whether a spec is the discrete input,
``checked_matrix`` checks a literal, ``rotation`` is the rotation about z.
"""

from typing import Any

import numpy as np
import torch

DEADBAND = 0.05  # max |component| below which the operator is idle
# A matrix spec's rotation key per dimension: in the plane (2 dims), about z (3 dims); no rotation key otherwise.
ROTATION_KEYS = {2: "rotation_deg", 3: "rotation_z_deg"}

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


def quantize(raw, deadband: float = DEADBAND) -> np.ndarray:
    """A discrete input: an active command (..., n) snapped to the axis of its largest component (the first on a
    tie), with that component's sign and the command's own length, so only its direction is lost (a length beyond
    full deflection is capped at 1, as every stage's output is); an idle command unchanged."""
    raw = np.asarray(raw, dtype=np.float64)
    axis = np.argmax(np.abs(raw), axis=-1)[..., None]
    length = np.minimum(np.linalg.norm(raw, axis=-1, keepdims=True), 1.0)
    snapped = np.zeros_like(raw)
    np.put_along_axis(snapped, axis, np.sign(np.take_along_axis(raw, axis, axis=-1)) * length, axis=-1)
    return np.where(is_active(raw, deadband)[..., None], snapped, raw)


def add_noise(core, sigma, noise) -> np.ndarray:
    """The readout's last stage: ``core`` (..., n) (the corrupted command, idle or not) plus ``sigma`` times the
    standard normal draws ``noise`` (..., n) per axis, clipped. ``sigma`` a scalar or one per row (...,). The caller
    draws ``noise``, so each keeps its own random stream."""
    core, sigma = np.asarray(core, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    return np.clip(
        core + np.asarray(noise, dtype=np.float64) * (sigma[..., None] if sigma.ndim else sigma), -1.0, 1.0
    )


# ---------------------------------------------------------------- the specs: matrices and corruptions


def checked_matrix(matrix: Any, size: int, where: str) -> np.ndarray:
    """Coerce `matrix` to a finite (size, size) float array or raise ValueError."""
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: entries must be numbers ({e})") from e
    if m.shape != (size, size):
        raise ValueError(f"{where}: expected a {size}x{size} matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{where}: entries must be finite")
    return m


def rotation(degrees: float) -> np.ndarray:
    """3x3 rotation of a translation command about the z axis."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_spec(spec: Any, n_dims: int, where: str = "matrix") -> np.ndarray | None:
    """The (n_dims, n_dims) matrix of a spec, or None for a null spec. Forms: ``{scale: [s_1, ..., s_n]}`` (exactly
    ``n_dims`` gains, the diagonal); the rotation by ``d`` degrees, ``{rotation_deg: d}`` in the plane (2 dims) or
    ``{rotation_z_deg: d}`` about z (3 dims; ``rotation``, whose top-left block is the plane's); ``{M: rows}`` or bare
    rows (the literal). Checked by ``checked_matrix``; any other key, the other dimension's rotation key included,
    raises naming the forms of this dimension."""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        return checked_matrix(spec, n_dims, where)
    keys, rotation_key = set(spec), ROTATION_KEYS.get(n_dims)
    if keys == {"scale"}:
        scale = np.asarray(spec["scale"], dtype=np.float64)
        if scale.shape != (n_dims,) or not np.all(np.isfinite(scale)):
            raise ValueError(f"{where}: scale must be {n_dims} finite numbers, got {spec['scale']!r}")
        matrix = np.diag(scale)
    elif rotation_key is not None and keys == {rotation_key}:
        matrix = rotation(float(spec[rotation_key]))
        if n_dims == 2:
            matrix = np.ascontiguousarray(matrix[:2, :2])
    elif keys == {"M"}:
        matrix = spec["M"]
    else:
        forms = ", ".join([rotation_key, "scale"] if rotation_key else ["scale"])
        raise ValueError(f"{where}: expected exactly one of {forms} or M, got {sorted(keys)}")
    return checked_matrix(matrix, n_dims, where)


def corruption_spec(
    spec: Any, where: str = "control.corruption", n_dims: int = 2
) -> tuple[np.ndarray | None, dict[str, float] | None]:
    """(matrix M or None, curl {curl, tau[, dt]} or None) from a corruption spec in ``n_dims`` dims (2: the
    cursor's; 3: the robot's translation): a matrix form (``matrix_spec``), or ``{curl: b, tau: s}`` for a curl
    field on the command (b in 1/s, + = clockwise; tau the command's time constant in s), optionally with ``dt``,
    the control step the field is stepped at in s (the readout's own when absent)."""
    if isinstance(spec, dict) and "quantize" in spec:
        raise ValueError(f"{where}: a quantize corruption has no matrix or curl; read it with quantize_spec")
    if isinstance(spec, dict) and "curl" in spec:
        if not {"curl", "tau"} <= set(spec) <= {"curl", "tau", "dt"}:
            raise ValueError(
                f"{where}: a curl corruption takes curl and tau (and optionally dt), got {sorted(spec)}"
            )
        try:
            curl = {key: float(spec[key]) for key in ("curl", "tau", "dt") if key in spec}
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: curl, tau and dt must be numbers, got {spec!r}") from e
        if not (np.isfinite(curl["curl"]) and curl["tau"] > 0 and curl.get("dt", 1.0) > 0):
            raise ValueError(f"{where}: curl must be finite and tau (and dt) positive, got {spec!r}")
        return None, curl
    return matrix_spec(spec, n_dims, where), None


def quantize_spec(spec: Any, where: str = "control.corruption") -> bool:
    """Whether a corruption spec is the discrete input, ``{quantize: axes}`` (``quantize``); False for any other
    spec. A ``quantize`` key with another value, or beside other keys, raises."""
    if not isinstance(spec, dict) or "quantize" not in spec:
        return False
    if set(spec) != {"quantize"} or spec["quantize"] != "axes":
        raise ValueError(f"{where}: a quantize corruption is exactly {{quantize: axes}}, got {spec!r}")
    return True


class Readout:
    """What happens to a command between the operator and the controller:
    raw -> M @ raw (corruption, clipped to [-1, 1]) -> + N(0, input_noise^2) per
    axis (clipped again). The corruption applies only while the operator is
    commanding (outside DEADBAND); the noise at every read, so a resting
    operator's command is the noise itself.

    With `curl` ({curl, tau[, dt]}) the corruption is a curl field instead of M: the
    readout keeps a velocity per trial, driven by every raw push (idle ones too)
    and reset by `reset(n_trials)` at the start of a batch of trials; the
    controller gets that velocity (clipped), then the noise. Steps of `dt` s (the
    curl's own `dt` when it has one).

    With `quantize` the corruption is the discrete input instead: the command
    snapped to the axis of its largest component at its own length (`quantize`),
    then the noise.

    Batched: commands are [B, n], or [B, H, n] for a whole chunk per trial (not
    with a curl), or [n] for a single unbatched trial; n = `n_dims` (2: the cursor;
    3: the robot's translation, where the curl turns x-y and z is a plain lag).
    `corruption` is one n x n matrix or one per trial [B, n, n]; `input_noise` is
    one std or one per trial [B].

    A call is `step(raw)` (the curl's velocity advances one control step) then
    `read(raw)` (what is served for that push); a controller that reads less often
    than it steps (the robot's, once per chunk) calls them apart. A read draws
    one normal per row and axis while the noise is on.
    """

    def __init__(
        self,
        corruption=None,
        input_noise=0.0,
        rng=None,
        curl=None,
        dt=0.02,
        n_dims=2,
        quantize=False,
    ):
        self.corruption, self.input_noise = corruption, np.asarray(input_noise, dtype=float)
        self.quantize = bool(quantize)
        if self.quantize and (corruption is not None or curl is not None):
            raise ValueError("a readout has a matrix corruption, a curl or a quantize, not both")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.curl, self.vel = curl, None
        self.dt = dt if curl is None else curl.get("dt", dt)
        self.n_dims = n_dims
        if curl is not None:
            if corruption is not None:
                raise ValueError("a readout has a matrix corruption or a curl, not both")
            self._decay, self._drive = curl_discretization(curl["tau"], curl["curl"], self.dt, n_dims)

    def reset(self, n_trials=1):
        """A new batch of trials: the curl's velocity starts at rest (`n_trials` None: one unbatched trial)."""
        shape = (self.n_dims,) if n_trials is None else (n_trials, self.n_dims)
        self.vel = None if self.curl is None else np.zeros(shape)

    def step(self, raw):
        """One control step of the curl field under the push `raw` ([B, n], or [n]); nothing without a curl."""
        if self.curl is None:
            return
        raw = np.asarray(raw, dtype=float)
        if raw.ndim > 2:
            raise ValueError(f"a curl readout takes one command per trial and step, [B, {self.n_dims}]")
        if self.vel is None or self.vel.shape != raw.shape:
            self.vel = np.zeros_like(raw)
        self.vel = curl_step(self.vel, raw, self._decay, self._drive)

    def read(self, raw):
        """What the readout serves for the push `raw` as it stands (the curl's velocity is not advanced)."""
        raw = np.asarray(raw, dtype=float)
        extra = (1,) * (raw.ndim - 2)  # per-trial parameters broadcast over a chunk's steps
        if self.curl is not None:
            if raw.ndim > 2:
                raise ValueError(f"a curl readout takes one command per trial and step, [B, {self.n_dims}]")
            out = curl_serve(np.zeros_like(raw) if self.vel is None else self.vel, raw)
        elif self.quantize:
            out = quantize(raw)
        elif self.corruption is not None:
            m = self.corruption
            out = corrupt(raw, m.reshape(m.shape[:1] + extra + m.shape[1:]) if m.ndim == 3 else m)
        else:
            out = raw.copy()
        if not np.any(self.input_noise):
            return out
        sigma = (
            self.input_noise.reshape(self.input_noise.shape + extra)
            if self.input_noise.ndim
            else self.input_noise
        )
        return add_noise(out, sigma, self.rng.normal(size=out.shape))

    def __call__(self, raw):
        self.step(raw)
        return self.read(raw)

    def core(self, raw):
        """What the readout serves before noise for raw pushes [B, L, n] given step by step from rest (numpy),
        every step's rule applied whether the push is active or not: ``clip(raw @ M.T)`` per step with one n x n
        matrix M, a curl's velocity after every step clipped to [-1, 1], or the pushes themselves with no
        corruption. The readout's own curl velocity is left alone (this one integrates from rest)."""
        if self.curl is not None:
            decay, drive = self._decay, self._drive
            vel, out = np.zeros(raw.shape[::2]), np.empty_like(raw, dtype=float)
            for k in range(raw.shape[1]):
                vel = curl_step(vel, raw[:, k], decay, drive)
                out[:, k] = vel
            return np.clip(out, -1.0, 1.0)
        if self.quantize:  # an idle step has no largest axis to speak of: it stays as it is
            return quantize(raw)
        if self.corruption is None:
            return raw
        m = np.asarray(self.corruption)
        if m.shape != (self.n_dims, self.n_dims):
            raise ValueError(
                f"a readout's core takes one matrix corruption ({self.n_dims}, {self.n_dims}), got {m.shape}"
            )
        return np.clip(raw @ m.T, -1.0, 1.0)

    def chunk(self, command, sigma):
        """The readout in torch, for training: deflection chunks `command` [B, H, n] and a noise std per trial
        `sigma` [B]; the corruption (one n x n matrix or None) as a tensor on the command's device. The same rule,
        step by step: M @ raw where the step is active, clip, + noise at every step, clip. A curl keeps a velocity
        over steps, so it takes no chunks."""
        if self.curl is not None:
            raise ValueError("a curl readout takes one command per trial and step, not a chunk")
        if self.quantize:
            raise ValueError("a quantize readout has no torch form: serve its numpy core")
        out = command
        if self.corruption is not None:
            m = torch.as_tensor(self.corruption, dtype=command.dtype, device=command.device)
            active = command.abs().amax(-1, keepdim=True) >= DEADBAND
            out = torch.where(active, torch.clamp(command @ m.T, -1.0, 1.0), command)
        return torch.clamp(out + sigma[:, None, None] * torch.randn_like(out), -1.0, 1.0)

    @staticmethod
    def serve(core, sigma, noise=None):
        """The readout's last stage in torch, for training: `core` [B, n] (M @ raw clipped, or a curl's
        velocity; idle or not) plus noise of std `sigma` [B], clipped. `noise` [B, n]: the standard normal draws
        to scale by sigma (default: fresh ones)."""
        noise = torch.randn_like(core) if noise is None else noise
        return torch.clamp(core + sigma[:, None] * noise, -1.0, 1.0)
