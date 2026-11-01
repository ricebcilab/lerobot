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

"""Shared-autonomy steering of a flow-matching sampler: ``FlowControlPolicy`` (Kao et al. 2026) and
``FlowReversalSteeringPolicy`` (Tang et al. 2026) over any policy with pi0.5's sampler interface, the MARS
adapters (``LearnedUserModule``, ``LearnedTaskModule``, ``ReversalAdapter``) with their checkpoints, losses and
training loop, the one Euler integrator of either flow-time convention, and each method's core. The settings
and specs are ``configuration_steering``'s; this folder's README describes the methods."""

import math
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .configuration_steering import (
    DEADBAND,
    GRIPPER_DIM,
    GRIPPER_OPEN,
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    MatrixHolder,
    TeleopSource,
    check_steps,
    format_matrix,
)

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
            from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

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
        self.token_in = nn.Sequential(
            nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)
        )  # the obstacle tokens' encoder
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


# ---------------------------------------------------------------- the steering methods, one core each
#
# The steering wrappers below run these, on pi0.5 and on the cursor policy alike, so a method's mechanics (what is
# written or reversed, where, and the forward schedule) are decided in one place. Either flow-time convention:
# ``noise_time`` 1.0 is pi0.5's (t = 1 noise, t = 0 action), 0.0 the cursor's.


def flow_control_write(
    x_t: torch.Tensor,
    step: int,
    written: torch.Tensor,
    tau: int,
    lam: int,
    dims=slice(None),
    steer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flow control's one rule (Kao et al. 2026): at the start of each of the first ``lam`` (lambda) Euler steps
    of a chunk's denoising, the first ``tau`` chunk steps of ``x_t``'s ``dims`` take ``written``, the command
    already in the noise prior's units ((n,) for every row, or (B, H, n) per row and step), in the rows where
    ``steer`` (B, 1, 1, bool) holds (all rows without it); the flow itself runs unmodified. lambda 1 writes the
    initial condition only (Kao et al.'s method), 0 nothing, more re-writes as the flow proceeds (the lab's
    variant). ``x_t`` is written in place and returned."""
    if step >= lam:
        return x_t
    w = written[:, :tau] if written.dim() == 3 else written
    region = x_t[:, :tau, dims]
    x_t[:, :tau, dims] = w if steer is None else torch.where(steer, w.expand_as(region), region)
    return x_t


def frs_schedule(steps: int, n_reverse_steps: int | None, noise_time: float = 1.0) -> tuple[float, int]:
    """Where FRS's forward pass starts and how many Euler steps it takes once the reference has been reversed
    n of the policy's N = ``steps`` steps (None: all of them): from the reversal's landing time back to the
    action in n steps of the policy's own 1/N, mirroring the reversal (n = N: from pure noise in N steps). A
    steered chunk costs 2n velocity evaluations (n back, n forward), N + n when a partial reversal hands steps
    back to the policy (N - n more for the unsteered latent)."""
    if noise_time not in (0.0, 1.0):
        raise ValueError(
            f"noise_time must be 1.0 (pi0.5: t = 1 noise) or 0.0 (the cursor: t = 0 noise), got {noise_time}"
        )
    n = steps if n_reverse_steps is None else n_reverse_steps
    check_steps("n_reverse_steps", n, 1, steps, "Euler steps")
    action_time = 1.0 - noise_time
    return action_time + (noise_time - action_time) * n / steps, n


def frs_latent(
    velocity: Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor],
    reference: torch.Tensor,
    noise: torch.Tensor,
    steps: int,
    n_reverse_steps: int | None = None,
    adapter=None,
    delegated: torch.Tensor | None = None,
    noise_time: float = 1.0,
    grad: bool = False,
    batch_time: bool = False,
) -> torch.Tensor:
    """Flow Reversal Steering's (and MARS's) latent: the ``reference`` chunk integrated backward n of N =
    ``steps`` Euler steps of 1/N toward noise (the adapter's user part acting; None: all the way, Tang et al.'s
    method), then the elements ``delegated`` marks (a bool mask broadcastable to the chunk: the dims the
    operator does not command, padding, the steps past the pinned prefix) replaced by the unsteered latent at
    the same flow time -- the chunk's own ``noise`` for a full reversal, else ``noise`` run forward N - n
    steps -- for the policy to fill in (Tang et al.'s noise-space in-painting, App. D.1). The forward pass
    starts from it on ``frs_schedule``'s schedule."""
    _, n = frs_schedule(steps, n_reverse_steps, noise_time)
    action_time = 1.0 - noise_time
    h = (noise_time - action_time) / steps  # toward noise
    latent = integrate(
        reference, velocity, action_time, h, n, adapter, backward=True, grad=grad, batch_time=batch_time
    )
    if delegated is not None and bool(delegated.any()):
        unsteered = (
            noise
            if n == steps
            else integrate(
                noise, velocity, noise_time, -h, steps - n, adapter, grad=grad, batch_time=batch_time
            )
        )
        latent = torch.where(delegated.to(latent.device), unsteered, latent)
    return latent


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


def heading_cosine(chunk_a: torch.Tensor, chunk_b: torch.Tensor, n_dims: int = 3) -> torch.Tensor:
    """Cosine between two chunks' total displacements (the sum over steps of their first ``n_dims`` dims), per
    sample: how well a regenerated chunk heads where its label does."""
    return nn.functional.cosine_similarity(chunk_a[..., :n_dims].sum(1), chunk_b[..., :n_dims].sum(1), dim=-1)


@torch.no_grad()
def regenerate(
    velocity,
    chunk: torch.Tensor,
    num_steps: int,
    adapter=None,
    noise_time: float = 1.0,
    batch_time: bool = False,
) -> torch.Tensor:
    """FRS at full reversal: ``chunk`` reversed through the adapted backward field (task and user modules) all
    the way to noise, then regenerated through the forward field (the task module alone: ``AdaptedField``
    applies the user part backward only). Either time convention (``noise_time`` and ``batch_time`` as in
    ``integrate``). The trainers' validation: what the policy makes of the operator's command through the
    modules."""
    h = (2 * noise_time - 1) / num_steps  # toward noise: +1/N in pi0.5's convention, -1/N in the cursor's
    z = integrate(
        chunk, velocity, 1 - noise_time, h, num_steps, adapter, backward=True, batch_time=batch_time
    )
    return integrate(z, velocity, noise_time, -h, num_steps, adapter, batch_time=batch_time)


def step_if_finite(opt: torch.optim.Optimizer, loss: torch.Tensor, params, clip: float = 1.0) -> bool:
    """One optimizer step, unless the loss or the clipped gradient norm is not finite (a batch where a flow
    blew up): then the step is skipped and the weights stay whole. Returns whether a step was taken."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    norm = nn.utils.clip_grad_norm_(list(params), clip)
    if not torch.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        return False
    opt.step()
    return True


def train_module(
    module: nn.Module,
    steps: int,
    lr: float,
    batch_loss: Callable[[int], torch.Tensor],
    validate: Callable[[], dict] | None = None,
    log_every: int = 1000,
    weight_decay: float = 1e-4,
    clip: float = 1.0,
) -> list[dict]:
    """The MARS modules' training loop, shared by the cursor and the LIBERO trainers: AdamW with a cosine
    schedule over ``steps``, gradients clipped to ``clip``, non-finite batches skipped (``step_if_finite``).
    ``batch_loss(step)`` returns one batch's loss. Every ``log_every`` steps, and at the last, the module is put
    in eval mode for ``validate()`` (named numbers, or None) and {step, train_loss (the window's mean),
    skipped, **validation} is logged and printed. Returns the log."""
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    log: list[dict] = []
    running, skipped, tic = 0.0, 0, time.time()
    for step in range(1, steps + 1):
        loss = batch_loss(step)
        skipped += not step_if_finite(opt, loss, module.parameters(), clip)
        sched.step()
        running += loss.item() if torch.isfinite(loss) else 0.0
        if step % log_every == 0 or step == steps:
            module.eval()
            val = validate() if validate is not None else {}
            module.train()
            entry = {"step": step, "train_loss": running / log_every, "skipped": skipped, **val}
            log.append(entry)
            shown = "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in val.items())
            print(
                f"step {step:6d}  train {entry['train_loss']:.4f}  {shown}"
                + (f"  skipped {skipped}" if skipped else "")
                + f"  ({time.time() - tic:.0f}s)",
                flush=True,
            )
            running = 0.0
    return log


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
    """What both steering wrappers share. ``steered_chunk(batch, commands)`` turns a batch of operator commands,
    one per row (a row inside ``deadband`` counts as idle and gets the policy's own chunk), into the steered chunk
    and the policy's own plan for the same noise. As a policy (pi0.5's ``select_action``), once per chunk the
    operator's command is read from ``source`` and held, and the chunk's first ``n_action_steps`` steps are served
    one per call (``last_policy_action`` the plan's step).

    Any flow policy with pi0.5's sampler interface can be steered: a ``config`` with ``num_inference_steps``,
    ``chunk_size``, ``max_action_dim``, ``n_action_steps`` and optionally ``noise_time`` (the flow time of noise:
    1.0, pi0.5's convention, by default; the cursor's is 0.0), and ``predict_action_chunk(batch, **hooks)`` taking
    ``noise_fn``, ``flow_start_time``, ``num_forward_steps``, ``x_t_hook`` and ``velocity_hook``
    (``PI05Pytorch.sample_actions``; the cursor's ``FlowPolicy.sample_chunk``)."""

    def __init__(self, policy, source: TeleopSource | None, deadband: float):
        self._policy = policy
        self._source = source
        self.deadband = deadband
        self._noise_time = float(getattr(policy.config, "noise_time", 1.0))
        self._pending_command = None
        self._queue: deque = deque()
        self._policy_queue: deque = deque()
        self.last_policy_action: torch.Tensor | None = (
            None  # the policy's own plan for the step just returned
        )

    def _read_command(self, read: bool = True) -> np.ndarray | None:
        """The operator's translation for the next chunk; None when not read or inside the deadband."""
        command = self._source.translation if read else None
        return command if command is not None and np.max(np.abs(command)) >= self.deadband else None

    def _active(self, commands) -> torch.Tensor:
        """(B,) bool: the rows of ``commands`` ((B, n) commands, or (B, H, n) a command per chunk step, active when
        any step is) outside the deadband."""
        magnitude = (
            torch.as_tensor(np.asarray(commands) if not torch.is_tensor(commands) else commands)
            .abs()
            .amax(-1)
        )
        return (magnitude.amax(-1) if magnitude.dim() == 2 else magnitude) >= self.deadband

    @staticmethod
    def _per_row(chunk: torch.Tensor, plain: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        """The steered rows of ``chunk``, the policy's own plan in the idle ones (the same noise)."""
        return (
            chunk if bool(steer.all()) else torch.where(steer.to(chunk.device)[:, None, None], chunk, plain)
        )

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


def task_velocity_hook(field):
    """The velocity hook a field's task module needs in every forward pass (the policy's own plan too), or None."""
    if field is None or getattr(field, "task_module", None) is None:
        return None

    @torch.compiler.disable
    def hook(step: int, time_: float, x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return field.adapt(x_t, time_, v, backward=False)

    return hook


class FlowControlPolicy(_ChunkSteering):
    """Shared flow control (Kao et al. 2026) on any flow policy with pi0.5's sampler interface; the pi0.5 wrapper.

    While the operator pushes, the command is written into ``x_t`` as the chunk's initial condition
    (``flow_control_write``): scaled by ``ic_gain`` into the noise prior's units (a full push writes ``ic_gain``
    sigma), into the commanded ``dims`` (pi0.5's translation, the first 3, by default) of the first ``tau`` chunk
    steps, at the start of each of the first ``lam`` (lambda) Euler steps of the chunk's denoising. The flow itself
    runs unmodified, so the executed action is entirely the model's output. ``lam`` 1 writes the initial condition
    only (Kao et al.'s method), 0 writes nothing (the policy alone), more keeps re-writing as the flow proceeds (the
    lab's earlier variant, whose ``n_guided_steps`` was this lambda, with the whole chunk written in action units).
    ``ic_gain="action"`` writes the command in the checkpoint's action units, ``(command - mean) / std`` (needs the
    ``postprocessor``), which with ``tau`` = the whole chunk and ``lam`` 1 is the arm the recorded human study ran.
    Guidance is skipped while the input is inside ``deadband``, so idle input means pure policy. A command may also
    be a chunk of commands, one per step (B, H, n), written step by step.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource | None,
        postprocessor=None,
        tau: int | None = None,
        lam: int = 1,
        ic_gain: float | str = "action",
        deadband: float = DEADBAND,
        dims=slice(0, 3),
    ):
        super().__init__(policy, source, deadband)
        self._mean = self._std = None
        if postprocessor is not None:
            mean, std = get_action_mean_std(postprocessor)
            self._mean, self._std = mean[:3], std[:3]
        self.tau = tau  # the setters validate against the policy's config
        self.lam = lam
        self.ic_gain = ic_gain
        self._dims = dims
        self.guided_steps = 0  # per-rollout count of Euler steps the command was written at
        self._pending_target: torch.Tensor | None = None
        self._pending_steer: torch.Tensor | None = None

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
            if self._mean is None:
                raise ValueError("ic_gain='action' needs the postprocessor's action statistics")
            self._ic_gain = "action"
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"ic_gain must be a positive number or 'action', got {value!r}")
        self._ic_gain = float(value)

    def reset(self) -> None:
        super().reset()
        self.guided_steps = 0

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook uses the current chunk's teleop command and must stay eager.
        if step >= self._lam or self._pending_command is None:
            return x_t
        if self._pending_target is None:  # in the noise prior's units: ic_gain x command, or the action units
            command = self._pending_command
            if self._ic_gain == "action":
                written = (np.asarray(command, dtype=np.float64) - self._mean) / self._std
            else:
                written = self._ic_gain * (
                    command if torch.is_tensor(command) else np.asarray(command, dtype=np.float64)
                )
            target = torch.as_tensor(written, dtype=x_t.dtype, device=x_t.device)
            self._pending_target = (
                target[:, None] if target.dim() == 2 else target
            )  # (B, n): one per row, every step
        steer = None if self._pending_steer is None else self._pending_steer.to(x_t.device)
        self.guided_steps += 1
        return flow_control_write(
            x_t, step, self._pending_target, self._tau, self._lam, dims=self._dims, steer=steer
        )

    def steered_chunk(self, batch, commands, field=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The chunk for ``commands`` ((B, n), or (B, H, n) per step; None: every row idle) and the policy's own
        plan from the same noise. ``field`` (an ``AdaptedField``) adds its task module to every forward pass."""
        hook = task_velocity_hook(field)
        common = {} if hook is None else {"velocity_hook": hook}
        steer = None if commands is None else self._active(commands)
        if steer is None or not bool(steer.any()):
            self._pending_command = None
            chunk = self._policy.predict_action_chunk(batch, **common)
            return chunk, chunk
        self._pending_command = commands
        self._pending_target = None  # resolved on the device by the first write
        self._pending_steer = None if bool(steer.all()) else steer[:, None, None]
        chunk, plain = paired_chunks(self._policy, batch, True, common, x_t_hook=self._x_t_hook)
        return self._per_row(chunk, plain, steer), plain

    def _sample_chunk(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Hold one consumed command (including its noise) for the whole chunk,
        # matching FRS and the synthetic operator's refresh cadence.
        command = self._read_command(read=bool(self._lam))
        return self.steered_chunk(batch, None if command is None else np.asarray(command)[None])


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
    chunk_size: int, max_action_dim: int, operator_dims: tuple[str, ...] | None, pinned_steps: int | None
) -> torch.Tensor:
    """(chunk_size, max_action_dim) bool mask of the reference elements left to the policy.

    True = delegated: the dims outside ``operator_dims`` (names from OPERATOR_DIM_GROUPS; None = every dim is
    commanded, as on the cursor), the padding dims beyond the env's 7, and every dim of the steps from
    ``pinned_steps`` on (None = the whole chunk stays pinned).
    """
    if operator_dims is None:
        mask = torch.zeros(chunk_size, max_action_dim, dtype=torch.bool)
    else:
        unknown = [d for d in operator_dims if d not in OPERATOR_DIM_GROUPS]
        if unknown:
            raise ValueError(
                f"operator_dims: unknown {unknown} (expected from {', '.join(OPERATOR_DIM_GROUPS)})"
            )
        mask = torch.ones(chunk_size, max_action_dim, dtype=torch.bool)
        for name in operator_dims:
            mask[:, OPERATOR_DIM_GROUPS[name]] = False
    if pinned_steps is not None:
        mask[pinned_steps:, :] = True
    return mask


class FlowReversalSteeringPolicy(_ChunkSteering):
    """Flow Reversal Steering (Tang et al. 2026) on any flow policy with pi0.5's sampler interface; the pi0.5
    wrapper (shared_flow_reversal_steering).

    While the operator pushes, a reference chunk built from the command (by default a uniform-velocity chunk in the
    commanded direction, rotation zero, gripper held at the model's last executed command, `gripper_open` before the
    first chunk; any ``reference_fn(commands) -> (B, H, D)`` in the model's normalized space otherwise) is
    integrated *backward* through the policy's own velocity field to the latent noise that maps to it
    (``frs_latent``); the normal forward flow then denoises from that noise instead of a random one, on
    ``frs_schedule``'s schedule. The executed action is entirely the model's output -- the reference only picks the
    starting noise, so what comes out is the generalist action mode nearest your intent. Idle input = pure policy.

    An optional adapter (MARS) adapts the velocity field: a `ReversalAdapter` here (its user module, a fixed 7x7
    matrix F or a learned U(x_t, t), in the reversal only, its task module in both directions, including the forward
    flow that produces the executed chunk and the policy's own plan), or an ``AdaptedField`` passed per call to
    ``steered_chunk`` (the cursor binds one to each observation).

    `n_reversal_steps` stops the reversal early instead of going all the way to noise: with n of the policy's
    N = num_inference_steps steps the reference is integrated n steps of 1/N, and the forward flow then runs from
    there in n steps of 1/N. A steered chunk costs 2n velocity evaluations, N + n with delegation. n = N (or None)
    is the full reversal, the method of Tang et al.; a partial reversal (n < N) is this module's own variant, which
    the paper does not have, and at small n the executed chunk is mostly the reference itself.

    `operator_dims` names the dim groups the operator commands (default: only translation; None: every action dim,
    as on the cursor) and `pinned_steps` how many leading steps of the chunk the reference constrains (default: the
    policy's n_action_steps, i.e. the executed prefix; None = all of it). Everything else is delegated to the
    policy: after the reversal, those latent elements are replaced with the unsteered latent at the same time --
    the chunk's own Gaussian sample for a full reversal, else the unsteered flow integrated N - n steps.
    Tang et al. (App. D.1) pin fewer steps than they execute: a 5-step reference (`action_repeats`) in a 15-step
    chunk with 10 executed, so the policy fills in half of what runs. The default here pins every executed step;
    for the paper's setting, pin half of n_action_steps.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource | None,
        postprocessor=None,
        adapter: ReversalAdapter | None = None,
        n_reversal_steps: int | None = None,
        operator_dims: tuple[str, ...] | None = ("translation",),
        pinned_steps: int | None = -1,
        deadband: float = DEADBAND,
        gripper_open: float = GRIPPER_OPEN,
        reference_fn: Callable | None = None,
    ):
        super().__init__(policy, source, deadband)
        self._adapter = adapter
        self.gripper_open = gripper_open
        self._reference_fn = reference_fn
        self.n_reversal_steps = n_reversal_steps  # validated by the setter below
        self._operator_dims = None if operator_dims is None else tuple(operator_dims)
        # The setter validates the pinned steps and builds the delegation mask (checking operator_dims)
        self.pinned_steps = policy.config.n_action_steps if pinned_steps == -1 else pinned_steps
        self._adapter_cache: tuple | None = None
        self._mean = self._std = None
        if postprocessor is not None:
            mean, std = get_action_mean_std(postprocessor)
            self._mean, self._std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]
            self._gripper_ref = self._normalize_gripper(gripper_open)
        elif reference_fn is None:
            raise ValueError(
                "a FlowReversalSteeringPolicy needs the postprocessor's action statistics or a reference_fn"
            )
        self._pending_reference: torch.Tensor | None = None
        self._last_reference: torch.Tensor | None = None
        self.steered_chunks = 0  # per-rollout count of chunks started from inverted noise
        self.reconstruction_errors: list[float] = []  # per steered chunk, in std units

    @property
    def operator_dims(self) -> tuple[str, ...] | None:
        return self._operator_dims

    @operator_dims.setter
    def operator_dims(self, value) -> None:
        cfg = self._policy.config
        dims = None if value is None else tuple(value)
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

    def _normalize_gripper(self, gripper: float) -> float:
        return float((gripper - self._mean[GRIPPER_DIM]) / self._std[GRIPPER_DIM])

    def reset(self) -> None:
        super().reset()
        if self._mean is not None:
            self._gripper_ref = self._normalize_gripper(self.gripper_open)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, command) -> torch.Tensor:
        """The reference for ``command`` ((n,) or (B, n), or (B, H, n) per step with a ``reference_fn``), in the
        model's normalized space: the ``reference_fn``'s, else the uniform-velocity chunk of env-space translation."""
        if self._reference_fn is not None:
            return self._reference_fn(command)
        cfg = self._policy.config
        return uniform_reference(
            command, self._gripper_ref, self._mean, self._std, cfg.chunk_size, cfg.max_action_dim
        )

    def _adapter_field(self, like: torch.Tensor) -> AdaptedField | None:
        """The wrapper's adapter as applied to normalized velocities on `like`'s device; the fixed matrix is
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

    def steered_chunk(self, batch, commands, field=None) -> tuple[torch.Tensor, torch.Tensor]:
        """The chunk for ``commands`` ((B, n), or (B, H, n) per step; None: every row idle) and the policy's own
        plan from the same noise. ``field`` (an ``AdaptedField``) replaces the wrapper's adapter for this call."""
        cfg = self._policy.config

        def adapted(like: torch.Tensor) -> AdaptedField | None:
            return field if field is not None else self._adapter_field(like)

        @torch.compiler.disable
        def velocity_hook(step: int, time_: float, x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # every forward pass, the policy's own plan too: the task module only
            a = adapted(x_t)
            return v if a is None else a.adapt(x_t, time_, v, backward=False)

        @torch.compiler.disable
        def noise_fn(velocity, noise: torch.Tensor) -> torch.Tensor:
            # compiler.disable: with --compile, sample_actions is dynamo-traced; this reverses the live command's
            # reference and must stay eager. The shared core: the reversal, then the delegated elements on the
            # unsteered flow (from noise: the chunk's own Gaussian sample; Tang et al., App. D).
            reference = self._pending_reference.to(device=noise.device, dtype=noise.dtype)
            return frs_latent(
                velocity,
                reference,
                noise,
                cfg.num_inference_steps,
                self._n_reversal_steps,
                adapter=adapted(noise),
                delegated=self._delegated,
                noise_time=self._noise_time,
                grad=True,
            )

        holder = field if field is not None else self._adapter
        common = {"velocity_hook": velocity_hook} if getattr(holder, "task_module", None) is not None else {}
        steer = None if commands is None else self._active(commands)
        if steer is None or not bool(steer.any()):
            self._pending_reference = None
            chunk = self._policy.predict_action_chunk(batch, **common)
            return chunk, chunk
        self._pending_reference = self.reference_chunk(commands)
        schedule = {}  # the full reversal: the usual schedule from noise in N steps
        if self._n_reversal_steps is not None:  # part-way: from the reversal's landing time, n steps of 1/N
            start, n = frs_schedule(
                cfg.num_inference_steps, self._n_reversal_steps, noise_time=self._noise_time
            )
            schedule = {"flow_start_time": start, "num_forward_steps": n}
        chunk, plain = paired_chunks(self._policy, batch, True, common, noise_fn=noise_fn, **schedule)
        return self._per_row(chunk, plain, steer), plain

    def _sample_chunk(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample the operator once per chunk: the forward schedule depends on
        # whether we are steering, so it cannot be decided inside the hook.
        self._pending_command = self._read_command()
        steering = self._pending_command is not None
        chunk, plain = self.steered_chunk(
            batch, np.asarray(self._pending_command)[None] if steering else None
        )
        self._last_reference = self._pending_reference if steering else None
        if steering:
            self.steered_chunks += 1
        executed = chunk[0, : self._policy.config.n_action_steps].float().cpu()  # one device sync per chunk
        self._gripper_ref = float(executed[-1, GRIPPER_DIM])
        if self._last_reference is not None:
            target = self._last_reference[0, : executed.shape[0], :3].float().cpu()
            self.reconstruction_errors.append((executed[:, :3] - target).abs().mean().item())
        return chunk, plain
