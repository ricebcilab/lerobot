#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Shared-autonomy steering of pi0.5's flow-matching sampler.

Two ways for an operator's translation command to steer the policy without
touching the executed action, which is always the model's own output:

- ``FlowControlPolicy``: shared flow control (Kao et al. 2026): the command,
  scaled by ``ic_gain`` into the noise prior's units, is written into the
  translation dims of the first ``tau`` chunk steps of ``x_t`` at the start of
  each of the first ``lam`` Euler steps (``lam`` 1 = the initial condition
  only); the flow itself runs unmodified.
- ``FlowReversalSteeringPolicy``: a uniform-velocity reference chunk built from
  the command is integrated *backward* through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator ``TeleopSource`` and the checkpoint's action statistics
from the postprocessor. ``reverse_flow`` is the integrator. ``ReversalAdapter``
is MARS (Modular Adapters for Reversal Steering, the method formerly called
FRS-RA): a *user module* that adapts the velocity field of the reversal only
(a fixed 7x7 matrix F, typically the known command corruption, or a learned
``LearnedUserModule`` U(x_t, t)) and/or a *task module* (A(x_t, t, v) v + b)
that adapts the field in both directions, so it reshapes every chunk the
policy produces. The forward field is task(v) and the backward field
user(task(v)); the modules are trained apart and composed here. FRS with a
user module alone is MARS-U, with a task module alone MARS-T.

Elements of the reference the operator does not command (``operator_dims``
selects which of translation / rotation / gripper they do), the padding dims and
the steps beyond ``pinned_steps`` are *delegated* to the policy: after the
reversal their latent is replaced by the unsteered one (fresh Gaussian noise for
a full reversal). This is the noise-space in-painting of Tang et al., App. D.1,
whose human interface steers a 5-step reference (``action_repeats``) in a
15-step chunk of which 10 steps are executed, setting the other 10 latent steps
to N(0, I): the reference covers half the executed steps, and the policy fills in
the rest. The wrapper's default pins the executed steps instead; set
``pinned_steps`` to match the paper's ratio.

Under the pi0.5 wrappers sits a task-generic core that takes any velocity field
``velocity(x, t)`` and any number of action dims: ``integrate`` (one Euler
integrator for either flow-time convention; the sign of ``h`` sets the
direction of flow time, ``backward`` whether the user part of the adapter acts):

- pi0.5 (t=1 noise, t=0 action): the reversal (action toward noise) runs
  ``t0=0, h=+1/N``, the forward pass (noise toward action) ``t0=1, h=-1/N``;
- the cursor (t=0 noise, t=1 action): the reversal runs ``t0=1, h=-1/N``, the
  forward pass from the reversal's landing point ``t0=1-n/N`` with ``h>0``
  (``t0=0, h=+1/N`` for a whole chunk from noise).

Also ``AdaptedField`` (fixed or per-trial matrices, user and task modules; a user
module acts through ``apply_user_module``, i.e. its ``velocity``), the
module classes (``LearnedUserModule`` with its rank knobs, ``LearnedTaskModule``, which
reads obstacles when it is given them) and their checkpoints, and, in the
cursor convention, the chunk functions ``flow_control_chunk``, ``frs_chunk`` and
``steered_chunk`` and the module training losses ``user_field_loss`` and
``task_field_loss``.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

# LIBERO's OSC_POSE action: end-effector position deltas, axis-angle deltas, gripper.
ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
N_ACTION_DIMS = len(ACTION_DIMS)
GRIPPER_DIM = ACTION_DIMS.index("gripper")
GRIPPER_OPEN = -1.0
# Groups of action dims an operator can command; anything else is the policy's.
OPERATOR_DIM_GROUPS = {
    "translation": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(GRIPPER_DIM, GRIPPER_DIM + 1),
}
GRIPPER_CLOSE = 1.0
DEADBAND = 0.05  # default `deadband` of the wrappers: below this (env units) the operator counts as idle


class TeleopSource(Protocol):
    """What the steering wrappers read: the operator's current command."""

    @property
    def translation(self) -> np.ndarray:  # shape (3,), env units in [-1, 1]
        ...

    @property
    def gripper(self) -> float:  # GRIPPER_OPEN or GRIPPER_CLOSE
        ...


def check_steps(name: str, value, low: int, high: int, unit: str, optional: bool = False) -> None:
    """Raise unless ``value`` is an integer (not a bool) in [``low``, ``high``] ``unit``, or None when
    ``optional``: the one check of the step knobs (tau, lambda, n_reversal_steps, pinned_steps)."""
    if optional and value is None:
        return
    if not (isinstance(value, int) and not isinstance(value, bool) and low <= value <= high):
        none = "None or " if optional else ""
        raise ValueError(f"{name} must be {none}an integer in [{low}, {high}] ({unit}), got {value!r}")


# ---------------------------------------------------------------- matrices


def validate_matrix(matrix, size: int, where: str) -> np.ndarray:
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


def rotation_about_z(degrees: float) -> np.ndarray:
    """3x3 rotation of a translation command about the z axis."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def translation_matrix(spec, where: str = "matrix", literal_key: str = "M") -> np.ndarray:
    """Build a 3x3 from a spec: {rotation_z_deg: d}, {scale: [sx, sy, sz]}, {M: rows} or bare rows."""
    if isinstance(spec, dict):
        keys = set(spec)
        if keys == {"rotation_z_deg"}:
            return rotation_about_z(float(spec["rotation_z_deg"]))
        if keys == {"scale"}:
            scale = np.asarray(spec["scale"], dtype=np.float64)
            if scale.shape != (3,) or not np.all(np.isfinite(scale)):
                raise ValueError(f"{where}: scale must be three finite numbers, got {spec['scale']!r}")
            return np.diag(scale)
        if keys == {literal_key}:
            return validate_matrix(spec[literal_key], 3, f"{where}.{literal_key}")
        raise ValueError(
            f"{where}: expected exactly one of rotation_z_deg, scale or {literal_key}, got {sorted(keys)}"
        )
    return validate_matrix(spec, 3, where)


_BLOCK_CHOICES = ("identity", "zero")


def build_reversal_adapter(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> np.ndarray | None:
    """Build the 7x7 reversal adapter F from a spec, or None when the spec is None.

    Forms: ``{F: rows}`` (literal), or blocks ``translation`` (``corruption`` to
    copy the 3x3 command corruption, ``corruption_inverse`` for its inverse, the
    adapter that recovers a curl field's steady state on the cursor task,
    ``identity``, ``zero``, or any ``translation_matrix`` spec), ``orientation``
    and ``gripper`` (``identity`` or ``zero``). Missing blocks are identity.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
    if "F" in spec:
        if set(spec) != {"F"}:
            raise ValueError(f"{where}: a literal F cannot be combined with block keys")
        return validate_matrix(spec["F"], N_ACTION_DIMS, f"{where}.F")
    unknown = set(spec) - {"translation", "orientation", "gripper"}
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)} (expected translation, orientation, gripper)"
        )

    f = np.eye(N_ACTION_DIMS)
    translation = spec.get("translation", "identity")
    if translation in ("corruption", "corruption_inverse"):
        if corruption is None:
            raise ValueError(f"{where}.translation: '{translation}' requested but control.corruption is null")
        f[:3, :3] = corruption if translation == "corruption" else np.linalg.inv(corruption)
    elif translation == "zero":
        f[:3, :3] = 0.0
    elif translation != "identity":
        f[:3, :3] = translation_matrix(translation, f"{where}.translation")
    for key, dims in (
        ("orientation", OPERATOR_DIM_GROUPS["rotation"]),
        ("gripper", OPERATOR_DIM_GROUPS["gripper"]),
    ):
        choice = spec.get(key, "identity")
        if choice not in _BLOCK_CHOICES:
            raise ValueError(f"{where}.{key}: expected identity or zero, got {choice!r}")
        if choice == "zero":
            f[dims, dims] = 0.0
    return f


def format_matrix(matrix: np.ndarray) -> str:
    """Rows of a small matrix on one line, for the terminal: `[+0.94 -0.34 +0.00; ...]`."""
    return "[" + "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in matrix) + "]"


class MatrixHolder:
    """Mutable holder for an optional square matrix (None = off) with a display label.

    Consumers read the holder on every use, so replacing ``matrix`` takes effect
    immediately; the setter validates the shape.
    """

    size: int
    where: str

    def __init__(self, matrix=None, label: str | None = None):
        self.matrix = matrix
        self.label = label

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix = None if value is None else validate_matrix(value, self.size, self.where)


# ---------------------------------------------------------------- MARS: modular reversal adapters


def sinusoidal_time_embedding(t, dim: int, device=None, dtype=None) -> torch.Tensor:
    """(B, dim) embedding of flow times t (a float or a (B,) tensor): [sin(1000 t w_k), cos(1000 t w_k)],
    w_k = 10^(-4k/(dim/2)); the adapters' time conditioning."""
    t = torch.as_tensor(t, dtype=dtype or torch.float32, device=device).reshape(-1)
    half = dim // 2
    omega = 10.0 ** (-4.0 * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    angles = 1000.0 * t[:, None] * omega[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class LearnedUserModule(nn.Module):
    """MARS user module U(x_t, t): one n x n matrix per chunk step, applied to the policy's velocity
    in the BACKWARD pass of the reversal; it undoes one user's command corruption. Works in the
    policy's normalized action space on the first ``n_dims`` dims of ``x_t``. Zero-initialized:
    a fresh module is the identity (plain FRS). ``time_embedding`` is ``"mars"``
    (``sinusoidal_time_embedding``) or ``"pi05"`` (pi0.5's own flow-time embedding, periods 0.004-4).

    Two knobs set its flexibility (n = ``n_dims``, H = ``horizon``); their defaults are the module
    as it always was, bit for bit, and a fresh module is the identity under every setting:

    - ``step_rank`` r (default None: every step's matrix free): each step's correction mixes r
      n x n matrices shared by the chunk, A_s = I + sum_k c_sk B_k, the coefficients c (per step)
      and the matrices B from (x_t, t). The coefficients start at zero, the matrices do not (else
      neither would get a gradient). r = n * n spans every matrix, as the default does.
    - ``cross_step_rank`` q (default 0: none): a rank-q coupling between the chunk's steps on top
      of the per-step matrices, U V^T on the stacked chunk: out[s, a] += sum_r U[s, a, r] m_r with
      m_r = sum_{s', a'} V[s', a', r] v[s', a'], U and V (H, n_c, q) from (x_t, t) through an extra
      head (``cross``) on the hidden layer. U starts at zero.
    - ``cross_step_dims`` (default None: all n dims): indices into the module's dims that the
      coupling reads and writes (unique, in range, non-empty); the other dims get no cross-step
      term. Only with ``cross_step_rank`` > 0. Names (e.g. LIBERO's ``ACTION_DIMS``) are mapped to
      indices where configs are parsed.

    The cross head has 2 * H * n_c * q outputs (n_c the coupled dims), so (d + 1) * 2 * H * n_c * q
    parameters: for LIBERO's H = 50 and n = 7, 700 * q outputs (179 900 * q parameters at d = 256).

    ``velocity(x_t, t, v)`` applies the whole correction (call it, not ``forward``, to apply the
    module: ``apply_user_module`` does); ``forward(x_t, t)`` gives the per-step matrices alone,
    (B, H, n, n), which is all of it when ``cross_step_rank`` is 0. The knobs enter ``config`` only
    when they differ from their defaults, so checkpoints without them load and save as before."""

    def __init__(
        self,
        horizon: int,
        n_dims: int = N_ACTION_DIMS,
        d: int = 256,
        tdim: int = 64,
        time_embedding: str = "mars",
        step_rank: int | None = None,
        cross_step_rank: int = 0,
        cross_step_dims: list[int] | None = None,
    ):
        super().__init__()
        if time_embedding not in ("mars", "pi05"):
            raise ValueError(f"time_embedding must be 'mars' or 'pi05', got {time_embedding!r}")
        if step_rank is not None and step_rank < 1:
            raise ValueError(f"step_rank must be None or >= 1, got {step_rank}")
        if cross_step_rank < 0:
            raise ValueError(f"cross_step_rank must be >= 0, got {cross_step_rank}")
        if cross_step_dims is not None:
            cross_step_dims = [int(i) for i in cross_step_dims]
            if not cross_step_rank:
                raise ValueError("cross_step_dims needs cross_step_rank > 0")
            if not cross_step_dims:
                raise ValueError("cross_step_dims must be None or a non-empty list of dim indices")
            if len(set(cross_step_dims)) != len(cross_step_dims):
                raise ValueError(f"cross_step_dims must be unique, got {cross_step_dims}")
            if not all(0 <= i < n_dims for i in cross_step_dims):
                raise ValueError(f"cross_step_dims must lie in [0, {n_dims}), got {cross_step_dims}")
        self.config: dict = {
            "horizon": horizon,
            "n_dims": n_dims,
            "d": d,
            "tdim": tdim,
            "time_embedding": time_embedding,
        }
        if step_rank is not None:
            self.config["step_rank"] = step_rank
        if cross_step_rank:
            self.config["cross_step_rank"] = cross_step_rank
        if cross_step_dims is not None:
            self.config["cross_step_dims"] = list(cross_step_dims)
        self.horizon, self.n_dims, self.tdim, self.time_embedding = horizon, n_dims, tdim, time_embedding
        self.step_rank, self.cross_step_rank, self.cross_step_dims = (
            step_rank,
            cross_step_rank,
            cross_step_dims,
        )
        n = n_dims
        out = horizon * n * n if step_rank is None else n * n * step_rank + horizon * step_rank
        self.net = nn.Sequential(
            nn.Linear(horizon * n + tdim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, out)
        )
        head = self.net[-1]
        first = 0 if step_rank is None else n * n * step_rank  # the coefficients start at zero, the basis not
        nn.init.zeros_(head.weight[first:])
        nn.init.zeros_(head.bias[first:])
        self.n_coupled = 0
        if cross_step_rank:
            self.n_coupled = n if cross_step_dims is None else len(cross_step_dims)
            n_u = horizon * self.n_coupled * cross_step_rank
            self.cross = nn.Linear(d, 2 * n_u)  # U then V
            nn.init.zeros_(self.cross.weight[:n_u])
            nn.init.zeros_(self.cross.bias[:n_u])

    def _hidden(self, x_t: torch.Tensor, t) -> torch.Tensor:
        b, h, n = x_t.shape[0], self.horizon, self.n_dims
        x = x_t[:, :h, :n].reshape(b, h * n).to(torch.float32)
        if self.time_embedding == "pi05":
            from .modeling_pi05 import create_sinusoidal_pos_embedding

            t = torch.as_tensor(t, dtype=torch.float32, device=x.device).reshape(-1)
            emb = create_sinusoidal_pos_embedding(t, self.tdim, 0.004, 4.0, x.device).float().expand(b, -1)
        else:
            emb = sinusoidal_time_embedding(t, self.tdim, device=x.device).expand(b, -1)
        return self.net[:-1](torch.cat([x, emb], dim=-1))

    def _matrices(self, hidden: torch.Tensor) -> torch.Tensor:
        b, h, n = hidden.shape[0], self.horizon, self.n_dims
        out = self.net[-1](hidden)
        eye = torch.eye(n, device=hidden.device, dtype=hidden.dtype)
        if self.step_rank is None:
            return eye + out.view(b, h, n, n)
        r = self.step_rank
        basis, coef = out[:, : n * n * r].view(b, r, n, n), out[:, n * n * r :].view(b, h, r)
        return eye + torch.einsum("bsk,bkij->bsij", coef, basis)

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        """x_t (B, H, >= n_dims), t a float or (B,) -> the per-step matrices (B, H, n, n)."""
        return self._matrices(self._hidden(x_t, t))

    def velocity(self, x_t: torch.Tensor, t, v: torch.Tensor) -> torch.Tensor:
        """The corrected velocity: x_t (B, H, >= n_dims), t a float or (B,), v (B, H, n) -> (B, H, n) in
        v's dtype; the per-step matrices, plus the cross-step coupling when ``cross_step_rank`` > 0."""
        hidden = self._hidden(x_t, t)
        out = torch.einsum("bhij,bhj->bhi", self._matrices(hidden).to(v.dtype), v)
        if self.cross_step_rank:
            b, h, q = hidden.shape[0], self.horizon, self.cross_step_rank
            uv = self.cross(hidden).view(b, 2, h, self.n_coupled, q)
            dims = self.cross_step_dims
            coupled = (v if dims is None else v[..., dims]).to(uv.dtype)
            m = torch.einsum("bsar,bsa->br", uv[:, 1], coupled)
            term = torch.einsum("bsar,br->bsa", uv[:, 0], m).to(v.dtype)
            if dims is None:
                out = out + term
            else:
                out = out.index_add(-1, torch.as_tensor(dims, device=out.device), term)
        return out


def apply_user_module(user, x_t: torch.Tensor, t, v: torch.Tensor) -> torch.Tensor:
    """A user module's correction of the velocity ``v`` (B, H, n): its own ``velocity`` when it has one (a
    ``LearnedUserModule``, with any cross-step coupling), else its per-step matrices, ``user(x_t, t)``
    (B, H, n, n) applied step by step (a fixed matrix holder or any such callable). In ``v``'s dtype."""
    if hasattr(user, "velocity"):
        return user.velocity(x_t, t, v)
    return torch.einsum("bhij,bhj->bhi", user(x_t, t).to(v.dtype), v)


def _gelu_mlp(d_in: int, d: int) -> nn.Sequential:
    """Linear, GELU, Linear: the cursor policy's ``mlp`` (the obstacle tokens' encoder)."""
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


class _BoundTaskModule:
    """A ``LearnedTaskModule`` with its observation fixed: the ``(x_t, t, v) -> (A, b)`` callable
    that ``AdaptedField`` takes as a task module."""

    def __init__(self, module: "LearnedTaskModule", obs: dict):
        self.module, self.obs, self.n_dims = module, obs, module.n_dims

    def __call__(self, x_t: torch.Tensor, t, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.module(x_t, t, v, self.obs)


class LearnedTaskModule(nn.Module):
    """MARS task module (the cursor example's ``TaskModule``): A(x_t, t, v[, obstacles]) v + b, one affine
    map of the policy's velocity per chunk step, in BOTH directions of the flow; it reshapes the frozen
    policy's chunk distribution toward a task. The chunk's steps attend (masked cross-attention) to a
    learned null token and, when the observation has them, one token per obstacle (position relative to
    the agent, radius, distance, in units of ``scale``). The head is zero-initialized: a fresh module is
    the identity. ``obs`` is optional: ``pos`` (B, 2, the agent's position in [0, 1]; zeros when absent),
    ``obstacles`` (B, K, 3: center, radius) and ``obstacle_mask`` (B, K, True for real obstacles). Without
    it (pi0.5, which sees the scene through its own field) the module is called as ``(x_t, t, v)``; with
    it, ``bind(obs)`` gives that callable."""

    def __init__(
        self,
        horizon: int = 16,
        d: int = 128,
        tdim: int = 64,
        heads: int = 4,
        layers: int = 2,
        scale: float = 0.2,
        n_dims: int = 2,
    ):
        super().__init__()
        self.config = {
            "horizon": horizon,
            "d": d,
            "tdim": tdim,
            "heads": heads,
            "layers": layers,
            "scale": scale,
            "n_dims": n_dims,
        }
        self.horizon, self.tdim, self.scale, self.n_dims = horizon, tdim, scale, n_dims
        self.token_in = _gelu_mlp(4, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.query_in = nn.Linear(n_dims + n_dims + tdim + 2, d)
        self.step_emb = nn.Parameter(0.02 * torch.randn(horizon, d))
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.norm_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.norm_k = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                for _ in range(layers)
            ]
        )
        self.out = nn.Linear(d, n_dims * n_dims + n_dims)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def tokens(self, obs: dict | None, b: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The obstacle tokens (B, 1 + K, d) with the null token first, and their padding mask."""
        null = self.null.expand(b, 1, -1)
        obstacles = None if obs is None else obs.get("obstacles")
        if obstacles is None or obstacles.shape[1] == 0:
            return null, None
        rel = (obstacles[..., :2] - obs["pos"][:, None]) / self.scale
        feats = torch.cat([rel, obstacles[..., 2:3] / self.scale, rel.norm(dim=-1, keepdim=True)], -1)
        tokens = torch.cat([null, self.token_in(feats)], 1)
        mask = obs.get("obstacle_mask")
        if mask is None:
            return tokens, None
        return tokens, torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=tokens.device), ~mask], 1)

    def forward(
        self, x_t: torch.Tensor, t, v: torch.Tensor, obs: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t (B, H, >= n), v (B, H, >= n), t a float or (B,), obs optional -> (A (B, H, n, n), b (B, H, n))."""
        b, h, n = x_t.shape[0], x_t.shape[1], self.n_dims
        tokens, pad = self.tokens(obs, b)
        temb = (
            sinusoidal_time_embedding(t, self.tdim, device=x_t.device)
            .expand(b, -1)[:, None]
            .expand(-1, h, -1)
        )
        pos = (
            torch.zeros(b, h, 2, device=x_t.device)
            if obs is None or obs.get("pos") is None
            else (2 * obs["pos"] - 1)[:, None].expand(-1, h, -1)
        )
        q = self.query_in(torch.cat([x_t[..., :n], v[..., :n], temb, pos], -1)) + self.step_emb
        for attn, norm_q, norm_k, ff in zip(self.attn, self.norm_q, self.norm_k, self.ff, strict=True):
            k = norm_k(tokens)
            q = q + attn(norm_q(q), k, k, key_padding_mask=pad, need_weights=False)[0]
            q = q + ff(q)
        out = self.out(q)
        a = torch.eye(n, device=x_t.device) + out[..., : n * n].view(b, h, n, n)
        return a, out[..., n * n :]

    def bind(self, obs: dict) -> _BoundTaskModule:
        """This module with ``obs`` fixed: a ``(x_t, t, v) -> (A, b)`` callable carrying ``n_dims``."""
        return _BoundTaskModule(self, obs)


_MODULE_KINDS = {"user": LearnedUserModule, "task": LearnedTaskModule}


def save_adapter_module(module: nn.Module, path, kind: str, meta: dict | None = None) -> Path:
    """Save a MARS module checkpoint: ``kind`` (user | task, matching the module's class), its config,
    weights and free-form meta."""
    if kind not in _MODULE_KINDS or type(module) is not _MODULE_KINDS[kind]:
        raise ValueError(f"kind must be one of {sorted(_MODULE_KINDS)} and match the module, got {kind!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"kind": kind, "config": module.config, "state_dict": module.state_dict(), "meta": meta or {}}, path
    )
    return path


def load_adapter_module(path, device="cpu") -> tuple[nn.Module, str, dict]:
    """Load a MARS module checkpoint: (module in eval mode, kind, meta). Task checkpoints written while
    there were two task module classes carry ``arch``: ``"obstacle"`` is this ``LearnedTaskModule``;
    ``"mlp"`` (and an MLP's weights with no arch, the first LIBERO files) is the removed MLP module."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)  # tensors, numbers, strings only
    kind = ckpt.get("kind")
    if kind not in _MODULE_KINDS:
        raise ValueError(f"{path}: unknown adapter module kind {kind!r} (format {ckpt.get('format')!r})")
    if kind == "task" and (
        ckpt.get("arch", "obstacle") != "obstacle" or "net.0.weight" in ckpt["state_dict"]
    ):
        raise ValueError(
            f"{path}: a task module of arch {ckpt.get('arch', 'mlp')!r}, the MLP task module, which was removed; "
            "retrain it as a LearnedTaskModule"
        )
    module = _MODULE_KINDS[kind](**ckpt["config"])
    module.load_state_dict(ckpt["state_dict"])
    return module.to(device).eval().requires_grad_(False), kind, ckpt.get("meta", {})


def convert_cursor_module(ckpt: dict) -> dict:
    """A cursor example module checkpoint (``{kind?, config, state_dict, args, obstacle}``; the first
    adapters, saved without ``kind``, are user modules) in the ``save_adapter_module`` format. A user
    module becomes a ``LearnedUserModule`` over 2 dims (same layers, same state-dict keys; its time
    embedding computes the same frequencies, so outputs agree to float rounding; its ``step_rank`` and
    ``cross_step_rank``, when set, carry over, and ``cross_step_dims``, which the cursor never had, stays
    None: the coupling covers both dims), a task module a ``LearnedTaskModule`` (over 2 dims).
    ``args`` and ``obstacle`` go into ``meta``."""
    kind = ckpt.get("kind", "user")
    config = dict(ckpt["config"])
    meta = {"args": ckpt.get("args"), "obstacle": ckpt.get("obstacle")}
    if kind == "user":
        ranks = {k: config[k] for k in ("step_rank", "cross_step_rank") if config.get(k)}  # only when set
        config = {
            "horizon": config["horizon"],
            "n_dims": 2,
            "d": config["d"],
            "tdim": config["tdim"],
            **ranks,
        }
        return {"kind": "user", "config": config, "state_dict": ckpt["state_dict"], "meta": meta}
    if kind == "task":
        config["n_dims"] = 2
        return {
            "kind": "task",
            "config": config,
            "state_dict": ckpt["state_dict"],
            "meta": meta,
        }
    raise ValueError(f"unknown cursor module kind {kind!r}; expected 'user' or 'task'")


@dataclass(frozen=True)
class AdapterSpec:
    """A parsed ``reversal_adapter`` spec: a fixed user matrix F (environment space) and/or the paths
    of learned modules. The session loads the modules; the matrix is rebuilt per trial when the
    corruption is random."""

    matrix: np.ndarray | None = None
    user_path: str | None = None
    task_path: str | None = None

    def __eq__(self, other):
        if not isinstance(other, AdapterSpec):
            return NotImplemented
        same_matrix = (self.matrix is None and other.matrix is None) or (
            self.matrix is not None and other.matrix is not None and np.array_equal(self.matrix, other.matrix)
        )
        return same_matrix and (self.user_path, self.task_path) == (other.user_path, other.task_path)


def _module_path(value, where: str) -> str:
    if not isinstance(value, str | Path) or not str(value):
        raise ValueError(f"{where}: expected a checkpoint path, got {value!r}")
    return str(value)


def parse_adapter_spec(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> AdapterSpec:
    """Parse a reversal-adapter spec into fixed matrix / module paths.

    Forms: ``null``; a fixed user matrix as ``build_reversal_adapter`` takes it (blocks or ``{F: rows}``);
    ``{learned: <path>}`` (a learned user module); or the composite ``{user: <fixed spec | {learned: path}
    | null>, task: <path | null>}`` (MARS = user + task; one slot alone is MARS-U or MARS-T).
    """
    if spec is None:
        return AdapterSpec()
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
    keys = set(spec)
    if keys == {"learned"}:
        return AdapterSpec(user_path=_module_path(spec["learned"], f"{where}.learned"))
    if keys and keys <= {"user", "task"}:
        user, task = spec.get("user"), spec.get("task")
        task_path = None if task is None else _module_path(task, f"{where}.task")
        if user is None:
            return AdapterSpec(task_path=task_path)
        if isinstance(user, dict) and "learned" in user:
            if set(user) != {"learned"}:
                raise ValueError(f"{where}.user: a learned module cannot be combined with a fixed matrix")
            return AdapterSpec(
                user_path=_module_path(user["learned"], f"{where}.user.learned"), task_path=task_path
            )
        return AdapterSpec(
            matrix=build_reversal_adapter(user, corruption, f"{where}.user"), task_path=task_path
        )
    if keys & {"learned", "user", "task"}:
        raise ValueError(
            f"{where}: expected null, a fixed matrix spec, {{learned: path}} or {{user: ..., task: ...}}, got {sorted(keys)}"
        )
    return AdapterSpec(matrix=build_reversal_adapter(spec, corruption, where))


class ReversalAdapter(MatrixHolder):
    """The MARS adapter of ``FlowReversalSteeringPolicy``: a user module (``matrix``, a fixed
    environment-space F, or ``user_module``, a learned U(x_t, t) or any (x_t, t) -> (B, H, n, n)
    callable) applied in the reversal only, and a ``task_module`` ((x_t, t, v) -> (A, b)) applied in
    both directions. All None = off (plain FRS). Consumers read the holder on every use."""

    size = N_ACTION_DIMS
    where = "reversal adapter"

    def __init__(self, matrix=None, label: str | None = None, user_module=None, task_module=None):
        super().__init__(matrix, label)
        self.user_module = user_module
        self.task_module = task_module
        self.user_label: str | None = None
        self.task_label: str | None = None

    @property
    def kind(self) -> str:
        user = self._matrix is not None or self.user_module is not None
        task = self.task_module is not None
        return {
            (True, True): "composed",
            (True, False): "user",
            (False, True): "task",
            (False, False): "none",
        }[(user, task)]

    def load_modules(self, user_path=None, task_path=None, device="cpu") -> None:
        """Load learned modules from checkpoints into the user and/or task slot (None leaves a slot alone)."""
        if user_path is not None:
            module, kind, _ = load_adapter_module(user_path, device)
            if kind != "user":
                raise ValueError(f"{user_path} holds a {kind} module, not a user module")
            self.user_module, self.user_label = module, str(user_path)
        if task_path is not None:
            module, kind, _ = load_adapter_module(task_path, device)
            if kind != "task":
                raise ValueError(f"{task_path} holds a {kind} module, not a task module")
            self.task_module, self.task_label = module, str(task_path)

    def describe(self) -> str:
        if self.kind == "none":
            return "Reversal adapter: off (plain FRS)."
        parts = []
        if self._matrix is not None:
            identity = " (identity, a no-op)" if np.allclose(self._matrix, np.eye(N_ACTION_DIMS)) else ""
            parts.append(f"fixed user matrix F = {self.label}{identity} = {format_matrix(self._matrix)}")
        if self.user_module is not None:
            parts.append(f"learned user module {self.user_label or type(self.user_module).__name__}")
        if self.task_module is not None:
            parts.append(f"task module {self.task_label or type(self.task_module).__name__}")
        name = {"user": "MARS-U", "task": "MARS-T", "composed": "MARS"}[self.kind]
        return f"Reversal adapter: {name} ({'; '.join(parts)})."

    def normalized_matrix(self, std: np.ndarray) -> np.ndarray | None:
        """Convert an environment-space velocity transform to normalized coordinates: S^-1 F S.

        Velocities have no mean offset. Apply this to all seven dimensions so literal
        matrices that mix translation, orientation and gripper use the same convention.
        """
        if self.matrix is None:
            return None
        return self.matrix * std[np.newaxis, :] / std[:, np.newaxis]

    def field(self, std: np.ndarray, like: torch.Tensor) -> "AdaptedField | None":
        """The adapter as applied to normalized velocities on `like`'s device, or None when off."""
        if self.kind == "none":
            return None
        matrix = self.normalized_matrix(std)
        tensor = None if matrix is None else torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
        return AdaptedField(tensor, self.user_module, self.task_module)


class AdaptedField:
    """Applies MARS modules to a velocity: task (both directions), then user (backward only, through
    ``apply_user_module``: a learned module's ``velocity``, any other callable's per-step matrices).
    Only the first n action dims are touched; the padding dims pass through. ``matrix`` is a fixed
    (n, n) F or one per trial, (B, n, n). A module acts on its ``n_dims`` attribute's dims
    (``N_ACTION_DIMS`` when it has none) unless ``n_dims`` is given, which overrides both."""

    def __init__(
        self, matrix: torch.Tensor | None, user_module=None, task_module=None, n_dims: int | None = None
    ):
        self.matrix, self.user_module, self.task_module = matrix, user_module, task_module
        self.n_dims = n_dims

    def _dims(self, module) -> int:
        return self.n_dims if self.n_dims is not None else getattr(module, "n_dims", N_ACTION_DIMS)

    def adapt(
        self, x_t: torch.Tensor, t: float | torch.Tensor, v: torch.Tensor, backward: bool
    ) -> torch.Tensor:
        result = v
        if self.task_module is not None:
            n = self._dims(self.task_module)
            head = result[..., :n]
            a, b = self.task_module(x_t, t, head)
            head = torch.einsum("bhij,bhj->bhi", a.to(head.dtype), head) + b.to(head.dtype)
            result = torch.cat([head, result[..., n:]], dim=-1)
        if backward:
            if self.matrix is not None:
                n = self.matrix.shape[-1]
                result = torch.cat([result[..., :n] @ self.matrix.transpose(-1, -2), result[..., n:]], dim=-1)
            if self.user_module is not None:
                n = self._dims(self.user_module)
                head = result[..., :n]
                head = apply_user_module(self.user_module, x_t, t, head)
                result = torch.cat([head, result[..., n:]], dim=-1)
        return result


def _adapt_velocity(adapter, x_t, t, v, backward: bool):
    """`adapter` is None, a normalized (n, n) or per-trial (B, n, n) matrix (the legacy fixed-F form), an
    AdaptedField, or a ReversalAdapter holder without a fixed matrix used directly (a holder's matrix is in
    environment units: convert the holder with ``field(std, like)`` first)."""
    if adapter is None:
        return v
    if isinstance(adapter, torch.Tensor):
        if not backward:
            return v
        n = adapter.shape[-1]
        return torch.cat([v[..., :n] @ adapter.transpose(-1, -2), v[..., n:]], dim=-1)
    if isinstance(adapter, ReversalAdapter):
        if adapter.matrix is not None:
            raise ValueError(
                "a ReversalAdapter's fixed matrix is in environment units; pass its field(std, like) instead"
            )
        if adapter.kind == "none":
            return v
        adapter = AdaptedField(None, adapter.user_module, adapter.task_module)
    return adapter.adapt(x_t, t, v, backward)


def integrate(
    x: torch.Tensor,
    velocity: Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor],
    t0: float,
    h: float,
    n: int,
    adapter=None,
    backward: bool = False,
    grad: bool = False,
    batch_time: bool = False,
) -> torch.Tensor:
    """``n`` Euler steps ``x <- x + h * adapt(v(x, t))`` at flow times ``t = t0 + i * h``, through any
    velocity field, in either time convention: the sign of ``h`` picks the direction of flow time,
    ``backward`` whether the adapter's user part (the fixed matrix F and/or the user module) acts
    (the reversal toward noise); a task module acts in both directions. ``adapter`` is
    anything ``_adapt_velocity`` takes. ``t`` is passed to ``velocity`` and the adapter as the float
    ``t0 + i * h``, or with ``batch_time`` as a (B,) float32 tensor of it on ``x``'s device (the
    cursor policy's form). With ``grad`` False the loop runs under ``torch.no_grad()``; with it True
    the caller's grad mode holds (differentiable when grad is enabled, for training)."""
    with torch.enable_grad() if grad and torch.is_grad_enabled() else torch.no_grad():
        for i in range(n):
            t = t0 + i * h
            if batch_time:
                t = torch.full((len(x),), t, device=x.device)
            x = x + h * _adapt_velocity(adapter, x, t, velocity(x, t), backward=backward)
    return x


# ---------------------------------------------------------------- chunks and losses, t = 0 noise convention
#
# Everything in this section is in the cursor's flow-time convention: t = 0 is noise, t = 1 the action,
# and flow times reach `velocity` and the adapter as (B,) float32 tensors (``integrate``'s ``batch_time``).

STEERED_CHUNK_MODES = ("policy", "flow_control", "frs")


@torch.no_grad()
def flow_control_chunk(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    written: torch.Tensor,
    steer: torch.Tensor,
    tau: int,
    lam: int,
    steps: int,
    adapter=None,
) -> torch.Tensor:
    """Flow control (Kao et al. 2026) in the t = 0 noise convention: the normalized chunk from noise ``z``
    (B, H, n). ``written`` (B, H, n) is the reference already scaled into the noise prior's units; its first
    ``tau`` chunk steps are written into the rows where ``steer`` (B, 1, 1, bool) holds at the start of each
    of the first ``lam`` (lambda) of the ``steps`` Euler steps (t = 0 to 1). The rest of the chunk keeps its
    noise and the flow runs unmodified (the adapter's task module in its field, as in every forward pass).
    lambda 1: the initial condition only, Kao et al.'s method; lambda 0: nothing written, the policy alone;
    lambda > 1: the lab's variant."""
    written = written[:, :tau]
    x, h = z.clone(), 1.0 / steps
    for i in range(steps):
        if i < lam:
            x[:, :tau] = torch.where(steer, written, x[:, :tau])
        x = integrate(x, velocity, i * h, h, 1, adapter, backward=False, batch_time=True)
    return x


@torch.no_grad()
def frs_chunk(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    ref: torch.Tensor,
    steer: torch.Tensor,
    n_reverse_steps: int | None,
    steps: int,
    adapter=None,
    pinned_steps: int | None = None,
) -> torch.Tensor:
    """Flow Reversal Steering (MARS with an adapter) in the t = 0 noise convention: the normalized chunk.
    The reference ``ref`` (B, H, n, normalized) is reversed n of the N = ``steps`` steps (t = 1 toward
    1 - n/N, the adapter's user part acting), then run forward in N - n steps, so a partial reversal costs
    the same N evaluations as a plain chunk; a full reversal (n = N, ``n_reverse_steps`` None) runs forward
    in N steps. Rows where ``steer`` (B, 1, 1, bool) is False get the plain chunk from ``z``.

    With ``pinned_steps`` only the reference's first ``pinned_steps`` chunk steps are steered: the latent of
    the later steps is replaced by the unsteered one at the reversal's landing time (``z`` itself for a full
    reversal, else ``z`` run forward to 1 - n/N, N - n more evaluations), for the policy to fill in -- Tang
    et al.'s noise-space in-painting (App. D: a 5-step reference in a 15-step chunk, the rest set to
    N(0, I)). None steers the whole chunk."""
    n = steps if n_reverse_steps is None else n_reverse_steps
    check_steps("n_reverse_steps", n, 1, steps, "Euler steps")
    latent = integrate(ref.clone(), velocity, 1.0, -1.0 / steps, n, adapter, backward=True, batch_time=True)
    t0, m = 1.0 - n / steps, steps if n == steps else steps - n
    if pinned_steps is not None and pinned_steps < latent.shape[1]:
        free = (
            z if n == steps else integrate(z, velocity, 0.0, 1.0 / steps, steps - n, adapter, batch_time=True)
        )
        latent = torch.cat([latent[:, :pinned_steps], free[:, pinned_steps:]], 1)
    steered = integrate(latent, velocity, t0, (1.0 - t0) / m, m, adapter, batch_time=True)
    plain = (
        steered if steer.all() else integrate(z, velocity, 0.0, 1.0 / steps, steps, adapter, batch_time=True)
    )
    return torch.where(steer, steered, plain)


@torch.no_grad()
def steered_chunk(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    command: torch.Tensor,
    mode: str,
    *,
    horizon: int,
    to_reference: Callable[[torch.Tensor], torch.Tensor],
    tau: int = 5,
    n_reverse_steps: int | None = None,
    steps: int = 10,
    adapter=None,
    reference: torch.Tensor | None = None,
    ic_gain: float = 1.0,
    lam: int = 1,
    deadband: float = DEADBAND,
    pinned_steps: int | None = None,
) -> torch.Tensor:
    """One normalized action chunk (B, ``horizon``, n) for commands (B, n), in the t = 0 noise convention.

    The reference the shared modes steer toward is the command held constant over the chunk, or
    ``reference`` (B, H, n, a command per step) when given: ``"flow_control"`` writes it, scaled by
    ``ic_gain``, into x_t (``flow_control_chunk``); ``"frs"`` converts it with the caller's
    ``to_reference`` (command units to the policy's normalized actions) and reverses it (``frs_chunk``),
    steering only its first ``pinned_steps`` chunk steps when given. ``"policy"`` is the plain forward
    flow. Rows whose command (or whole reference) is inside ``deadband`` (max |component| below it) get
    the plain chunk, from the same noise they would have had in mode
    ``"policy"``: the noise ``z`` is drawn first, so random streams line up across modes."""
    b, n = len(command), command.shape[-1]
    z = torch.randn(b, horizon, n, device=command.device)
    src = torch.as_tensor(command if reference is None else reference)
    active = src.abs().amax(-1) >= deadband
    steer = (active if reference is None else active.any(-1)).to(command.device)[:, None, None]
    if mode == "policy" or not steer.any():
        return integrate(z, velocity, 0.0, 1.0 / steps, steps, adapter, batch_time=True)
    deflection = command[:, None].expand(-1, horizon, -1) if reference is None else reference
    if mode == "flow_control":
        check_steps("tau", tau, 0, horizon, "chunk steps")
        check_steps("lambda", lam, 0, steps, "Euler steps")
        return flow_control_chunk(velocity, z, ic_gain * deflection, steer, tau, lam, steps, adapter)
    if mode == "frs":
        check_steps("pinned_steps", pinned_steps, 1, horizon, "chunk steps", optional=True)
        return frs_chunk(
            velocity, z, to_reference(deflection), steer, n_reverse_steps, steps, adapter, pinned_steps
        )
    raise ValueError(f"unknown mode {mode!r}; expected one of {STEERED_CHUNK_MODES}")


def user_field_loss(
    user: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latent: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    task: Callable | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """The user module's field regression: along the straight path x_t = (1 - t) latent + t target
    (t (B,), one flow time per sample),

        U(x_t, t) T(v) = target - latent,   v = velocity(x_t, t) (no grad, in float32),

    on the module's first ``n_dims`` dims (all of ``v``'s for a callable without them), T the bound task
    module ``task(x_t, t, v) -> (A, b)`` (A v + b) if any, else the identity; U(x_t, t) T(v) is
    ``apply_user_module`` (a ``LearnedUserModule``'s ``velocity``, cross-step coupling included). In the
    t = 0 noise convention ``latent`` is the noise and ``target`` the action; in pi0.5's (t = 1 noise) they
    swap. A full reversal of ``target`` through the backward field then lands on ``latent``. Gradients reach
    whichever module trains. ``weight`` (B,) (e.g. the active commands) averages the per-sample errors."""
    x = (1 - t[:, None, None]) * latent + t[:, None, None] * target
    with torch.no_grad():
        v = velocity(x, t).float()
    if task is not None:
        a, b = task(x, t, v)
        v = torch.einsum("bsij,bsj->bsi", a, v) + b
    n = getattr(user, "n_dims", v.shape[-1])
    pred = apply_user_module(user, x.float(), t, v[..., :n])
    err = ((pred - (target - latent)[..., :n].float()) ** 2).mean((1, 2))
    if weight is None:
        return err.mean()
    weight = weight.float()
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def task_field_loss(
    task: Callable,
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    chunk: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    lambda_id: float,
    n_dims: int,
) -> tuple[torch.Tensor, dict]:
    """The task module's flow matching through the adapted field, in the t = 0 noise convention: on the
    policy's interpolation x_t = (1 - t) z + t a (``chunk`` a, normalized (B, H, n); t (B,)),

        L = || A v + b - (a - z) ||^2 + lambda_id (||A - I||^2 + ||b||^2),   v = velocity(x_t, t) (no grad),

    ``task(x_t, t, v) -> (A, b)`` the bound task module, I the ``n_dims`` identity. Returns the loss and its
    two parts (``fm``, ``identity``) as floats."""
    x = (1 - t[:, None, None]) * z + t[:, None, None] * chunk
    with torch.no_grad():
        v = velocity(x, t)
    mat, b = task(x, t, v)
    fm = ((torch.einsum("bsij,bsj->bsi", mat, v) + b - (chunk - z)) ** 2).mean()
    identity = ((mat - torch.eye(n_dims, device=x.device)) ** 2).mean() + (b**2).mean()
    return fm + lambda_id * identity, {"fm": fm.item(), "identity": identity.item()}


def _rng_state() -> tuple:
    """CPU and CUDA generator states, so a second sampling pass can reuse the first one's noise."""
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return torch.get_rng_state(), cuda


def _set_rng_state(state: tuple) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def paired_chunks(
    policy, batch, steered: bool, common: dict | None = None, **steer_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """The steered chunk and the policy's own plan for the same observation and the same noise.

    Returns (steered, plain). When `steered` is False the two are the same tensor: no
    second pass. Otherwise the generator state is restored between the passes so the
    plain chunk starts from the Gaussian sample the steered one used, and the state
    after the steered pass is restored afterwards, so a run's random stream is exactly
    what it would be without the second pass. `common` kwargs go to both passes (a MARS
    task module shapes the policy's own plan too); `steer_kwargs` only to the steered one.
    """
    common = common or {}
    if not steered:
        chunk = policy.predict_action_chunk(batch, **common)
        return chunk, chunk
    before = _rng_state()
    chunk = policy.predict_action_chunk(batch, **common, **steer_kwargs)
    after = _rng_state()
    _set_rng_state(before)
    plain = policy.predict_action_chunk(batch, **common)
    _set_rng_state(after)
    return chunk, plain


# ---------------------------------------------------------------- FlowControlPolicy


def get_action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]:
    """The affine (mean, std) of the checkpoint's action normalization: normalized = (x - mean) / std.

    Read from the unnormalizer step's stats and its normalization mode: mean/std as is,
    quantiles and min/max as the equivalent affine map onto [-1, 1].
    """
    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if not stats or "action" not in stats:
            continue
        action = {k: np.asarray(v, dtype=np.float64) for k, v in stats["action"].items()}
        mode = getattr(getattr(step, "norm_map", None), "get", lambda *_: None)("ACTION")
        mode = getattr(mode, "value", mode) or "MEAN_STD"
        if mode == "QUANTILES":
            lo, hi = action["q01"], action["q99"]
        elif mode == "QUANTILE10":
            lo, hi = action["q10"], action["q90"]
        elif mode == "MIN_MAX":
            lo, hi = action["min"], action["max"]
        else:
            return action["mean"], np.maximum(action["std"], 1e-6)
        return (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    raise RuntimeError("No action stats in the postprocessor; the shared flow modes need them.")


class _ChunkSteering:
    """What both pi0.5 wrappers share with ``PI05Policy.select_action``'s queue: once per chunk the
    operator's command is read and held (inside ``deadband`` it counts as idle), ``_sample_chunk(batch)``
    returns the chunk and the policy's own plan for the same noise, and their first ``n_action_steps``
    steps are served one per call (``last_policy_action`` the plan's step)."""

    def __init__(self, policy, source: TeleopSource, deadband: float):
        self._policy = policy
        self._source = source
        self.deadband = deadband
        self._pending_command: np.ndarray | None = None
        self._queue: deque = deque()
        self._policy_queue: deque = deque()
        self.last_policy_action: torch.Tensor | None = (
            None  # the policy's own plan for the step just returned
        )

    def _read_command(self, read: bool = True) -> np.ndarray | None:
        """The operator's translation for the next chunk; None when not read or inside the deadband."""
        command = self._source.translation if read else None
        return command if command is not None and np.max(np.abs(command)) >= self.deadband else None

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self._policy_queue.clear()
        self._pending_command = None

    def select_action(self, batch) -> torch.Tensor:
        if len(self._queue) == 0:
            chunk, plain = self._sample_chunk(batch)
            n = self._policy.config.n_action_steps
            actions = list(chunk[:, :n].transpose(0, 1))
            self._queue.extend(actions)
            self._policy_queue.extend(actions if plain is chunk else plain[:, :n].transpose(0, 1))
        self.last_policy_action = self._policy_queue.popleft()
        return self._queue.popleft()


class FlowControlPolicy(_ChunkSteering):
    """pi0.5 wrapper implementing shared_flow_control (Kao et al. 2026).

    While the operator pushes, the command is written into ``x_t`` as the chunk's
    initial condition: scaled by ``ic_gain`` into the noise prior's units (a full
    push writes ``ic_gain`` sigma), into the translation dims of the first ``tau``
    chunk steps, at the start of each of the first ``lam`` (lambda) Euler steps of
    the chunk's denoising. The flow itself runs unmodified, so the executed action
    is entirely the model's output. ``lam`` 1 writes the initial condition only
    (Kao et al.'s method), 0 writes nothing (the policy alone), more keeps
    re-writing as the flow proceeds (the lab's earlier variant, whose
    ``n_guided_steps`` was this lambda, with the whole chunk written in action
    units). ``ic_gain="action"`` writes the command in the checkpoint's action
    units, ``(command - mean) / std``, which with ``tau`` = the whole chunk and
    ``lam`` 1 is the arm the recorded human study ran. Guidance is skipped while
    the input is inside ``deadband``, so idle input means pure policy.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource,
        postprocessor,
        tau: int | None = None,
        lam: int = 1,
        ic_gain: float | str = "action",
        deadband: float = DEADBAND,
    ):
        super().__init__(policy, source, deadband)
        self.tau = tau  # the setters validate against the policy's config
        self.lam = lam
        self.ic_gain = ic_gain
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:3], std[:3]
        self.guided_steps = 0  # per-rollout count of Euler steps the command was written at
        self._pending_target: torch.Tensor | None = None

    @property
    def tau(self) -> int:
        return self._tau

    @tau.setter
    def tau(self, value: int | None) -> None:
        chunk = self._policy.config.chunk_size
        check_steps("tau", value, 1, chunk, "chunk steps", optional=True)
        self._tau = chunk if value is None else value

    @property
    def lam(self) -> int:
        return self._lam

    @lam.setter
    def lam(self, value: int) -> None:
        check_steps("lam", value, 0, self._policy.config.num_inference_steps, "Euler steps")
        self._lam = value

    @property
    def ic_gain(self) -> float | str:
        return self._ic_gain

    @ic_gain.setter
    def ic_gain(self, value: float | str) -> None:
        if value == "action":
            self._ic_gain = "action"
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"ic_gain must be a positive number or 'action', got {value!r}")
        self._ic_gain = float(value)

    def reset(self) -> None:
        super().reset()
        self.guided_steps = 0

    def written_command(self, command: np.ndarray) -> np.ndarray:
        """What a command writes into ``x_t``'s translation dims: ``ic_gain * command`` in the noise
        prior's units, or ``(command - mean) / std`` in the checkpoint's action units."""
        if self._ic_gain == "action":
            return (np.asarray(command, dtype=np.float64) - self._mean) / self._std
        return self._ic_gain * np.asarray(command, dtype=np.float64)

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook uses the current chunk's teleop command and must stay eager.
        if step >= self._lam or self._pending_command is None:
            return x_t
        if self._pending_target is None:
            target = self.written_command(self._pending_command)
            self._pending_target = torch.as_tensor(target, dtype=x_t.dtype, device=x_t.device)
        x_t[:, : self._tau, :3] = self._pending_target
        self.guided_steps += 1
        return x_t

    def _sample_chunk(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Hold one consumed command (including its noise) for the whole chunk,
        # matching FRS and the synthetic operator's refresh cadence.
        self._pending_command = self._read_command(read=bool(self._lam))
        self._pending_target = None  # resolved on the device by the first write
        return paired_chunks(self._policy, batch, self._pending_command is not None, x_t_hook=self._x_t_hook)


def reverse_flow(
    x: torch.Tensor,
    velocity,
    num_steps: int,
    adapter=None,
    n_reversal_steps: int | None = None,
) -> torch.Tensor:
    """Integrate the flow backward (Euler, t: 0 -> 1) from a clean chunk toward its latent noise.

    Mirrors PI05Pytorch.sample_actions, which integrates x_{t+dt} = x_t + dt * v(x_t, t)
    with dt = -1/num_steps from t=1 (noise) to t=0 (action); here the same field is
    stepped with dt = +1/num_steps starting at t=0. This is the inversion of Flow
    Reversal Steering (Tang et al. 2026, arXiv:2606.13675).

    The step size is always 1/`num_steps`, i.e. the policy's own schedule.
    `n_reversal_steps` is how many of those steps to take: the default (None) runs
    the whole way to t=1, pure noise, while a smaller count stops early and
    returns a chunk at t = n_reversal_steps / num_steps, still carrying some of
    the reference. The caller must then start the forward flow at that time.

    `adapter` is None, an (n, n) matrix F already in normalized coordinates (the
    fixed user module: x_t += h * (F @ v) on the first n action dims, the env's 7,
    leaving the padding dims alone; F = I is an exact no-op) or an ``AdaptedField``
    (MARS: task module, then user module, since this is the backward pass).
    """
    steps = num_steps if n_reversal_steps is None else n_reversal_steps
    return integrate(x, velocity, 0.0, 1.0 / num_steps, steps, adapter, backward=True, grad=True)


def forward_flow(
    x: torch.Tensor, velocity, num_steps: int, stop_at: int | None = None, adapter=None
) -> torch.Tensor:
    """Integrate the flow forward (Euler) from noise at t=1 down to t = stop_at / num_steps.

    Mirrors PI05Pytorch.sample_actions without hooks: x_{t-h} = x_t - h * v(x_t, t). With
    ``stop_at`` None it runs all the way to t=0 (the unsteered action chunk); with
    ``stop_at = n`` it takes ``num_steps - n`` steps and returns the unsteered state at
    t = n / num_steps, i.e. where a reversal stopped after ``n`` steps would sit.
    ``adapter`` (an ``AdaptedField``) applies its task module only: the user module
    acts on the reversal alone.
    """
    steps = num_steps if stop_at is None else num_steps - stop_at
    return integrate(x, velocity, 1.0, -(1.0 / num_steps), steps, adapter, grad=True)


def uniform_reference(
    command, gripper_normalized, mean: np.ndarray, std: np.ndarray, chunk_size: int, max_action_dim: int
) -> torch.Tensor:
    """The reference chunk FRS steers toward: env-space translation ``command`` ((3,) or (B, 3)) held at
    uniform velocity over the chunk, rotation zero, the gripper at ``gripper_normalized`` (a float or
    (B,)), in the model's normalized padded space: (B, chunk_size, max_action_dim)."""
    command = np.atleast_2d(np.asarray(command, dtype=np.float64))
    env_action = np.zeros((command.shape[0], N_ACTION_DIMS))
    env_action[:, :3] = command
    normalized = (env_action - mean) / std
    normalized[:, GRIPPER_DIM] = np.asarray(gripper_normalized, dtype=np.float64)
    ref = torch.zeros(command.shape[0], chunk_size, max_action_dim, dtype=torch.float32)
    ref[..., :N_ACTION_DIMS] = torch.as_tensor(normalized, dtype=torch.float32)[:, None, :]
    return ref


def delegation_mask(
    chunk_size: int, max_action_dim: int, operator_dims: tuple[str, ...], pinned_steps: int | None
) -> torch.Tensor:
    """(chunk_size, max_action_dim) bool mask of the reference elements left to the policy.

    True = delegated: the dims outside ``operator_dims`` (names from OPERATOR_DIM_GROUPS),
    the padding dims beyond the env's 7, and every dim of the steps from ``pinned_steps``
    on (None = the whole chunk stays pinned).
    """
    unknown = [d for d in operator_dims if d not in OPERATOR_DIM_GROUPS]
    if unknown:
        raise ValueError(f"operator_dims: unknown {unknown} (expected from {', '.join(OPERATOR_DIM_GROUPS)})")
    mask = torch.ones(chunk_size, max_action_dim, dtype=torch.bool)
    for name in operator_dims:
        mask[:, OPERATOR_DIM_GROUPS[name]] = False
    if pinned_steps is not None:
        mask[pinned_steps:, :] = True
    return mask


class FlowReversalSteeringPolicy(_ChunkSteering):
    """pi0.5 wrapper implementing shared_flow_reversal_steering (Flow Reversal Steering).

    While the teleop input is deflected (beyond `deadband`), a reference chunk
    that servos in the commanded direction at uniform velocity (rotation zero,
    gripper held at the model's last executed command, `gripper_open` before
    the first chunk) is integrated *backward* through the
    policy's own velocity field for num_inference_steps to find the latent
    noise that maps to it; the normal forward flow then denoises from that
    noise instead of a random one. The executed action is entirely the
    model's output — the reference only picks the starting noise, so what
    comes out is the generalist action mode nearest your intent. Idle input
    = pure policy (random noise, as usual).

    An optional `ReversalAdapter` (MARS) adapts the velocity field: its user module
    (a fixed 7x7 matrix F or a learned U(x_t, t)) in the reversal only, its task
    module in both directions, including the forward flow that produces the executed
    chunk and the policy's own plan; see `reverse_flow`, `forward_flow` and `AdaptedField`.

    `n_reversal_steps` stops the reversal early instead of going all the way to
    noise: with n of the policy's N = num_inference_steps steps the reference is
    integrated up to t = n/N, and the forward flow then runs
    from there in n steps, maintaining the policy's step size in both directions.
    With delegation, a steered chunk costs N + n velocity evaluations: n reverse,
    N - n to obtain the unsteered state, and n forward. Depth changes how far
    inversion travels before delegation and reconstruction; its effect on
    intent preservation is empirical. n = N (or None) is the full reversal,
    the method of Tang et al.; a partial reversal (n < N) is this module's own
    variant, which the paper does not have, and at small n the executed chunk
    is mostly the reference itself.

    `operator_dims` names the dim groups the operator commands (default: only
    translation) and `pinned_steps` how many leading steps of the chunk the
    reference constrains (default: the policy's n_action_steps, i.e. the executed
    prefix; None = all of it). Everything else is delegated to the policy: after
    the reversal, those latent elements are replaced with the unsteered latent at
    the same time -- the chunk's own Gaussian sample for a full reversal, else the
    unsteered flow integrated down to t = n/N (N - n extra velocity evaluations).
    Tang et al. (App. D.1) pin fewer steps than they execute: a 5-step reference
    (`action_repeats`) in a 15-step chunk with 10 executed, so the policy fills in
    half of what runs. The default here pins every executed step; for the paper's
    setting, pin half of n_action_steps.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource,
        postprocessor,
        adapter: ReversalAdapter | None = None,
        n_reversal_steps: int | None = None,
        operator_dims: tuple[str, ...] = ("translation",),
        pinned_steps: int | None = -1,
        deadband: float = DEADBAND,
        gripper_open: float = GRIPPER_OPEN,
    ):
        super().__init__(policy, source, deadband)
        self._adapter = adapter
        self.gripper_open = gripper_open
        self.n_reversal_steps = n_reversal_steps  # validated by the setter below
        self._operator_dims = tuple(operator_dims)
        # The setter validates the pinned steps and builds the delegation mask (checking operator_dims)
        self.pinned_steps = policy.config.n_action_steps if pinned_steps == -1 else pinned_steps
        self._adapter_cache: tuple | None = None
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]
        self._gripper_ref = self._normalize_gripper(gripper_open)
        self._last_reference: torch.Tensor | None = None
        self.steered_chunks = 0  # per-rollout count of chunks started from inverted noise
        self.reconstruction_errors: list[float] = []  # per steered chunk, in std units

    @property
    def operator_dims(self) -> tuple[str, ...]:
        return self._operator_dims

    @operator_dims.setter
    def operator_dims(self, value) -> None:
        cfg = self._policy.config
        dims = tuple(value)
        self._delegated = delegation_mask(cfg.chunk_size, cfg.max_action_dim, dims, self._pinned_steps)
        self._operator_dims = dims

    @property
    def pinned_steps(self) -> int | None:
        return self._pinned_steps

    @pinned_steps.setter
    def pinned_steps(self, value: int | None) -> None:
        cfg = self._policy.config
        check_steps("pinned_steps", value, 1, cfg.chunk_size, "chunk steps", optional=True)
        self._delegated = delegation_mask(cfg.chunk_size, cfg.max_action_dim, self._operator_dims, value)
        self._pinned_steps = value

    @property
    def n_reversal_steps(self) -> int | None:
        return self._n_reversal_steps

    @n_reversal_steps.setter
    def n_reversal_steps(self, value: int | None) -> None:
        total = self._policy.config.num_inference_steps
        check_steps("n_reversal_steps", value, 1, total, "Euler steps", optional=True)
        self._n_reversal_steps = None if value == total else value

    def _flow_kwargs(self) -> dict:
        """Where the forward flow starts, once the reference has been reversed part-way."""
        if self._n_reversal_steps is None:
            return {}  # full reversal: the usual schedule from t=1 in N steps
        total = self._policy.config.num_inference_steps
        return {
            "flow_start_time": self._n_reversal_steps / total,
            "num_forward_steps": self._n_reversal_steps,
        }

    def _normalize_gripper(self, gripper: float) -> float:
        return float((gripper - self._mean[GRIPPER_DIM]) / self._std[GRIPPER_DIM])

    def reset(self) -> None:
        super().reset()
        self._gripper_ref = self._normalize_gripper(self.gripper_open)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, command: np.ndarray) -> torch.Tensor:
        """Uniform-velocity chunk for env-space translation `command`, in the model's normalized space."""
        cfg = self._policy.config
        return uniform_reference(
            command, self._gripper_ref, self._mean, self._std, cfg.chunk_size, cfg.max_action_dim
        )

    def _adapter_field(self, like: torch.Tensor) -> AdaptedField | None:
        """The adapter as applied to normalized velocities on `like`'s device; the fixed matrix is
        converted once and rebuilt only when the holder's matrix or modules change."""
        if self._adapter is None or self._adapter.kind == "none":
            self._adapter_cache = None
            return None
        key = (
            self._adapter.matrix,
            self._adapter.user_module,
            self._adapter.task_module,
            like.device,
            like.dtype,
        )
        cached = self._adapter_cache
        # The cache keeps the matrix itself, not its id: a matrix that replaces a freed one can
        # land on the same address, and would then be mistaken for the one it replaced.
        if (
            cached is None
            or any(a is not b for a, b in zip(cached[0][:3], key[:3], strict=True))
            or cached[0][3:] != key[3:]
        ):
            self._adapter_cache = (key, self._adapter.field(self._std, like))
        return self._adapter_cache[1]

    @torch.compiler.disable
    def _velocity_hook(self, step: int, time_: float, x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # The forward flow that produces the executed chunk: the task module only.
        field = self._adapter_field(x_t)
        return v if field is None else field.adapt(x_t, time_, v, backward=False)

    def _common_kwargs(self) -> dict:
        """Kwargs every chunk takes, steered or not: the task module's velocity hook."""
        if self._adapter is not None and self._adapter.task_module is not None:
            return {"velocity_hook": self._velocity_hook}
        return {}

    @torch.compiler.disable
    def _noise_fn(self, velocity, noise: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # this builds the reference from live teleop input and must stay eager.
        # `select_action` has already sampled the input (once per chunk) and put
        # it in _pending_command, so the forward schedule it chose matches this hook.
        command = self._pending_command
        if command is None:
            self._last_reference = None
            return noise
        reference = self.reference_chunk(command).to(device=noise.device, dtype=noise.dtype)
        self._last_reference = reference
        self.steered_chunks += 1
        total = self._policy.config.num_inference_steps
        field = self._adapter_field(noise)
        latent = reverse_flow(
            reference, velocity, total, adapter=field, n_reversal_steps=self._n_reversal_steps
        )
        delegated = self._delegated.to(noise.device)
        if not delegated.any():
            return latent
        # Delegated elements follow the unsteered flow: at t=1 that is the chunk's own
        # Gaussian sample (Tang et al., App. D: the freed noises are set to N(0, I)).
        unsteered = (
            noise
            if self._n_reversal_steps is None
            else forward_flow(noise, velocity, total, stop_at=self._n_reversal_steps, adapter=field)
        )
        return torch.where(delegated, unsteered, latent)

    def _sample_chunk(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample the operator once per chunk: the forward schedule depends on
        # whether we are steering, so it cannot be decided inside the hook.
        self._pending_command = self._read_command()
        steering = self._pending_command is not None
        flow_kwargs = self._flow_kwargs() if steering else {}
        if not steering:
            self._last_reference = None  # no reference to score this chunk against
        chunk, plain = paired_chunks(
            self._policy, batch, steering, self._common_kwargs(), noise_fn=self._noise_fn, **flow_kwargs
        )
        executed = chunk[0, : self._policy.config.n_action_steps].float().cpu()  # one device sync per chunk
        self._gripper_ref = float(executed[-1, GRIPPER_DIM])
        if self._last_reference is not None:
            target = self._last_reference[0, : executed.shape[0], :3].float().cpu()
            self.reconstruction_errors.append((executed[:, :3] - target).abs().mean().item())
        return chunk, plain
