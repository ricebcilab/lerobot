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

"""Shared-autonomy steering of a flow-matching sampler: ``SteeringPolicy``, the lerobot policy, and the cores it
wraps: ``FlowControlSteering`` (Kao et al. 2026) and ``FlowReversalSteering`` (Tang et al. 2026) over any policy with
pi0.5's sampler interface, the MARS adapters (``LearnedUserModule``, ``LearnedTaskModule`` and their one holder,
``AdapterModules``, with ``as_adapter``) with their checkpoints and losses, and ``Flow``, the one Euler integrator of
either flow-time convention. The settings and specs are
``configuration_steering``'s, the pre-processing ``processor_steering``'s; this folder's README describes the
methods."""

import math
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.envs.cursor import SCENE_SIZE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import batched_trial
from lerobot.teleoperators import readout
from lerobot.utils.constants import (
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_steering import (
    GRIPPER_DIM,
    N_ACTION_DIMS,
    OPERATOR_DIM_GROUPS,
    SCENE_SOURCES,
    TOKEN_GEOMETRIES,
    SteeringConfig,
)

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0
# the default `deadband` of the cores: below this (env units) the operator counts as idle
DEADBAND = readout.DEADBAND
# The task module's image token specs (``LearnedTaskModule.spec``); the scene encoder's are ``processor_steering``'s.
VLM_SCENE_TOKENS = 256  # SigLIP's patches of the 224 px input every render is resized to
VLM_VIEWS_TOKENS = 2 * VLM_SCENE_TOKENS  # vlm_views: the overhead view's tokens, then the wrist view's
VLM_SCENE_DIM = 2048  # the width of pi0.5's prefix (SigLIP after the multimodal projector, and Gemma's)
CNN_PATCH = 16  # the task module's CNN stem downsamples by 16

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
    module: ``apply_correction`` does); ``forward(x_t, t)`` gives the per-step matrices alone,
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

    @staticmethod
    def apply_correction(user, x_t: torch.Tensor, t, v: torch.Tensor) -> torch.Tensor:
        """A user module's correction of the velocity ``v`` (B, H, n): its own ``velocity`` when it has one (a
        ``LearnedUserModule``, with any cross-step coupling), else its per-step matrices, ``user(x_t, t)``
        (B, H, n, n) applied step by step (a fixed matrix holder or any such callable). In ``v``'s dtype."""
        if hasattr(user, "velocity"):
            return user.velocity(x_t, t, v)
        return torch.einsum("bhij,bhj->bhi", user(x_t, t).to(v.dtype), v)

    @staticmethod
    def field_loss(
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
        ``apply_correction`` (a ``LearnedUserModule``'s ``velocity``, cross-step coupling included). In the
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
        pred = LearnedUserModule.apply_correction(user, x.float(), t, v[..., :n])
        err = ((pred - (target - latent)[..., :n].float()) ** 2).mean((1, 2))
        if weight is None:
            return err.mean()
        weight = weight.float()
        return (err * weight).sum() / weight.sum().clamp_min(1.0)


class _BoundTaskModule:
    """A ``LearnedTaskModule`` with its observation fixed: the ``(x_t, t, v) -> (A, b)`` callable
    that ``AdaptedField`` takes as a task module. The scene tokens depend on the observation alone, so the
    first call makes them and the calls after it (a flow's Euler steps) reuse them, as long as the batch size
    and the gradient mode are the same; the result is the unbound call's."""

    def __init__(self, module: "LearnedTaskModule", obs: dict):
        self.module, self.obs, self.n_dims = module, obs, module.n_dims
        self._tokens: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._key: tuple[int, bool] | None = None  # batch size and grad mode the tokens were made for

    def __call__(self, x_t: torch.Tensor, t, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (x_t.shape[0], torch.is_grad_enabled())
        if self._tokens is None or self._key != key:
            self._tokens, self._key = self.module.tokens(self.obs, key[0]), key
        return self.module(x_t, t, v, self.obs, precomputed=self._tokens)


class LearnedTaskModule(nn.Module):
    """MARS task module (the cursor example's task module): A(x_t, t, v[, obstacles]) v + b, one affine
    map of the policy's velocity per chunk step, in BOTH directions of the flow; it reshapes the frozen
    policy's chunk distribution toward a task. The chunk's steps attend (masked cross-attention) to a
    learned null token and, when the observation has them, one token per obstacle (position relative to
    the agent, radius, distance, in units of ``scale``). The head is zero-initialized: a fresh module is
    the identity. ``obs`` is optional: ``pos`` (B, 2, the agent's position in [0, 1];
    required by the disc tokens and by relative geometry; the ``forward`` query, which has no observation, uses
    zeros),
    ``obstacles`` (B, K, 3: center, radius) and ``obstacle_mask`` (B, K, True for real obstacles). Without
    it (pi0.5, which sees the scene through its own field) the module is called as ``(x_t, t, v)``; with
    it, ``bind(obs)`` gives that callable.

    ``token_source`` picks where the scene tokens come from (the null token is always first):
    ``discs`` the obstacle discs above; ``vlm_image`` / ``vlm_prefix`` the frozen VLM's image / prefix
    tokens, ``obs["scene"][source]`` (B, ``scene_tokens``, ``scene_dim``), through a linear projection;
    ``cnn`` the rendered image, ``obs["scene"]["cnn"]`` (B, 3, S, S) uint8 with S = ``image_size``,
    through a small CNN trained with the module (stride 16: ``(S / 16) ** 2`` tokens). The image
    sources add a learned position per token (``token_geometry="absolute"``), or, with ``"relative"``
    (``vlm_image`` / ``cnn`` only), the offset of the token's patch centre from the cursor through the
    disc-token MLP and no learned position. For ``cnn`` the assumed cell centres are the stride grid's
    ``((c + 0.5) * 16) / S``, while a 4-layer k3/s2/p1 stem's receptive field is centred near ``16c`` px, about
    half a cell up and left: exact for ``vlm_image`` (SigLIP's patch grid), to be corrected before a ``cnn``
    relative run.

    ``vlm_views`` reads the frozen VLM's vision tokens of two camera views, ``obs["scene"]["vlm_views"]``
    (B, ``scene_tokens``, ``scene_dim``): the overhead view's tokens, then the wrist view's (512 in all);
    ``wrist_span`` (> 0, required) records how much of the scene the wrist view spans. Nothing derived from
    the cursor's position enters these tokens, so ``relative`` geometry is refused. ``wrist_span`` is a config
    key of ``vlm_views`` only.

    With ``scene_rank`` (any ``vlm_*`` source; not ``discs`` or ``cnn``, which have no 2048-wide tokens) the
    module carries a fixed linear reduction (``scene_mean`` (scene_dim,), ``scene_basis`` (scene_dim,
    scene_rank), zeros until ``set_reduction``; persistent, so they travel in the checkpoint) and
    ``scene_in`` reads ``scene_rank`` wide tokens: ``tokens`` reduces a full-width input with
    ``reduce`` and takes a ``scene_rank``-wide one as stored; any other width raises. Relative
    geometry is added after the projection, as at full width. ``scene_rank`` is a config key only when set,
    so a module without it has the config and weights it always had.

    ``obs["scene_keep"]`` (B, ``scene_tokens``, bool; any image source, optional) marks the scene tokens the
    attention may see; the others are padded out of it. The null token is never hidden. Without it every
    token is visible."""

    def __init__(  # nosec B107 (token_source is not a password)
        self,
        horizon: int = 16,
        d: int = 128,
        tdim: int = 64,
        heads: int = 4,
        layers: int = 2,
        scale: float = 0.2,
        n_dims: int = 2,
        token_source: str = "discs",
        scene_dim: int | None = None,
        scene_tokens: int | None = None,
        image_size: int | None = None,
        token_geometry: str = "absolute",
        wrist_span: float | None = None,
        scene_rank: int | None = None,
    ):
        super().__init__()
        if token_geometry not in TOKEN_GEOMETRIES:
            raise ValueError(f"token_geometry must be one of {TOKEN_GEOMETRIES}, got {token_geometry!r}")
        if token_geometry == "relative" and token_source not in ("vlm_image", "cnn"):  # nosec B105
            raise ValueError(
                f"relative token_geometry needs token_source vlm_image or cnn, got {token_source!r}"
            )
        self._check_scene(token_source, image_size, wrist_span, scene_rank, width=None)
        if token_source.startswith("vlm") and (scene_dim is None or scene_tokens is None):
            raise ValueError(f"token_source {token_source!r} needs scene_dim and scene_tokens")
        if scene_rank is not None and not 1 <= scene_rank < scene_dim:
            raise ValueError(f"scene_rank must be in 1..{scene_dim - 1}, got {scene_rank}")
        if token_source == "cnn":  # nosec B105
            scene_tokens = (image_size // CNN_PATCH) ** 2
        grid = math.isqrt(scene_tokens) if token_geometry == "relative" else 0  # nosec B105
        if token_geometry == "relative" and grid * grid != scene_tokens:  # nosec B105
            raise ValueError(f"relative token_geometry needs a square token grid, got {scene_tokens} tokens")
        self.config = {
            "horizon": horizon,
            "d": d,
            "tdim": tdim,
            "heads": heads,
            "layers": layers,
            "scale": scale,
            "n_dims": n_dims,
        }
        if token_source != "discs":  # nosec B105
            self.config |= {
                "token_source": token_source,
                "scene_dim": scene_dim,
                "scene_tokens": scene_tokens,
                "image_size": image_size,
            }
        if token_source == "vlm_views":  # nosec B105
            self.config["wrist_span"] = wrist_span
        if scene_rank is not None:
            self.config["scene_rank"] = scene_rank
        if token_geometry == "relative":  # nosec B105
            self.config["token_geometry"] = token_geometry
        self.wrist_span, self.scene_rank = wrist_span, scene_rank
        self.token_geometry = token_geometry
        self.horizon, self.tdim, self.scale, self.n_dims = horizon, tdim, scale, n_dims
        self.token_source, self.scene_dim, self.scene_tokens, self.image_size = (
            token_source,
            scene_dim,
            scene_tokens,
            image_size,
        )
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
        # after everything above, so a discs module's weights do not depend on the image sources
        if token_source.startswith("vlm"):
            self.scene_in = nn.Linear(scene_rank or scene_dim, d)
        elif token_source == "cnn":  # nosec B105
            chans = [3, 32, 64, 128, d]
            convs = []
            for i in range(4):
                convs.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1))
                if i < 3:
                    convs.append(nn.GELU())
            self.cnn = nn.Sequential(*convs)
        if token_geometry == "relative":  # nosec B105
            # token k = (row, col) = divmod(k, grid); x to the right (col), y down (row)
            rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
            centres = torch.stack([(cols.flatten() + 0.5) / grid, (rows.flatten() + 0.5) / grid], -1)
            self.register_buffer("token_centres", centres, persistent=False)
            self.token_half = 0.5 / grid
        elif token_source != "discs":  # nosec B105
            self.scene_pos = nn.Parameter(0.02 * torch.randn(scene_tokens, d))
        if scene_rank is not None:  # zeros until set_reduction; persistent, so a checkpoint carries them
            self.register_buffer("scene_mean", torch.zeros(scene_dim))
            self.register_buffer("scene_basis", torch.zeros(scene_dim, scene_rank))

    @staticmethod
    def _check_scene(
        source: str,
        size: int | None,
        wrist_span: float | None,
        scene_rank: int | None,
        width: int | None = VLM_SCENE_DIM,
    ) -> None:
        """The scene checks the module and ``spec`` share: ``source`` one of ``SCENE_SOURCES``; ``wrist_span`` (> 0)
        exactly for ``vlm_views``; ``scene_rank`` only for a ``vlm_*`` source and, with ``width``, in 1..width - 1;
        ``cnn`` renders a positive multiple of ``CNN_PATCH`` in ``size``."""
        if source not in SCENE_SOURCES:
            raise ValueError(f"token_source must be one of {SCENE_SOURCES}, got {source!r}")
        if source == "vlm_views" and (wrist_span is None or wrist_span <= 0):  # nosec B105
            raise ValueError(f"token_source 'vlm_views' needs a positive wrist_span, got {wrist_span}")
        if source != "vlm_views" and wrist_span is not None:  # nosec B105
            raise ValueError(f"wrist_span belongs to token_source 'vlm_views', not {source!r}")
        if scene_rank is not None:
            if not source.startswith("vlm_"):
                raise ValueError(
                    f"scene_rank reduces the vlm_* sources' tokens, not token_source {source!r}'s"
                )
            if width is not None and not 1 <= scene_rank < width:
                raise ValueError(f"scene_rank must be in 1..{width - 1}, got {scene_rank}")
        if source == "cnn" and (size is None or size <= 0 or size % CNN_PATCH):  # nosec B105
            raise ValueError(
                f"token_source 'cnn' needs a size (image_size) that is a multiple of {CNN_PATCH}, got {size}"
            )

    @classmethod
    def spec(
        cls,
        source: str,
        size: int = SCENE_SIZE,
        *,
        wrist_span: float | None = None,
        scene_rank: int | None = None,
    ) -> dict:
        """The ``LearnedTaskModule`` kwargs for ``source`` on size-``size`` renders: ``discs`` needs none beyond
        the source; ``vlm_*`` read 256 tokens of width 2048 whatever the size (the encoder resizes every render
        to 224 for SigLIP; ``size`` is only what is drawn and cached); ``cnn`` reads the render itself,
        ``(size // 16) ** 2`` tokens (size a multiple of 16); ``vlm_views`` reads both views' tokens (512 of
        width 2048) and needs the wrist view's ``wrist_span`` (> 0, ``vlm_views`` alone). Every ``vlm_*`` source takes
        an optional ``scene_rank`` (1..2047), the width of the carried token reduction; ``vlm_views`` always has the
        key (None: full width), the others only when it is set. ``discs`` and ``cnn`` have no 2048-wide tokens to
        reduce."""
        cls._check_scene(source, size, wrist_span, scene_rank)
        if source == "discs":  # nosec B105
            return {"token_source": "discs"}
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if source == "cnn":  # nosec B105
            return {
                "token_source": "cnn",
                "scene_dim": None,
                "scene_tokens": (size // CNN_PATCH) ** 2,
                "image_size": size,
            }
        if source == "vlm_views":  # nosec B105
            return {
                "token_source": source,
                "scene_dim": VLM_SCENE_DIM,
                "scene_tokens": VLM_VIEWS_TOKENS,
                "image_size": size,
                "wrist_span": wrist_span,
                "scene_rank": scene_rank,
            }
        return {
            "token_source": source,
            "scene_dim": VLM_SCENE_DIM,
            "scene_tokens": VLM_SCENE_TOKENS,
            "image_size": size,
            **({} if scene_rank is None else {"scene_rank": scene_rank}),
        }

    def set_reduction(self, mean: torch.Tensor, basis: torch.Tensor) -> None:
        """Set the fixed linear token reduction: ``mean`` (scene_dim,), ``basis`` (scene_dim, scene_rank)."""
        if self.scene_rank is None:
            raise ValueError("set_reduction needs a module built with scene_rank")
        if mean.shape != self.scene_mean.shape or basis.shape != self.scene_basis.shape:
            raise ValueError(
                f"reduction shapes must be {tuple(self.scene_mean.shape)} and {tuple(self.scene_basis.shape)}, "
                f"got {tuple(mean.shape)} and {tuple(basis.shape)}"
            )
        with torch.no_grad():
            self.scene_mean.copy_(mean)
            self.scene_basis.copy_(basis)

    def tokens(self, obs: dict | None, b: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The scene tokens (B, 1 + N, d) with the null token first, and their padding mask (None when
        every token is real). Without ``obs`` only the null token."""
        null = self.null.expand(b, 1, -1)
        src = self.token_source
        if src != "discs":
            if obs is None:
                return null, None
            x = (obs.get("scene") or {}).get(src)
            if x is None:
                raise ValueError(f"task module with token_source {src!r} needs obs['scene'][{src!r}]")
            if src == "cnn":
                scene = self.cnn(x.float() / 255).flatten(2).transpose(1, 2)
            else:
                if self.scene_rank is not None:
                    if x.shape[-1] == self.scene_dim:
                        x = LearnedTaskModule.reduce(x, self.scene_mean, self.scene_basis)
                    elif x.shape[-1] != self.scene_rank:
                        raise ValueError(
                            f"a reduced task module takes tokens of width {self.scene_dim} or "
                            f"{self.scene_rank}, got width {x.shape[-1]}"
                        )
                scene = self.scene_in(x.float())
            if self.token_geometry == "relative":  # nosec B105
                pos = obs.get("pos")
                if pos is None:
                    raise ValueError("task module with relative token_geometry needs obs['pos']")
                rel = (self.token_centres[None] - pos[:, None]) / self.scale
                half = torch.full_like(rel[..., :1], self.token_half / self.scale)
                feats = torch.cat([rel, half, rel.norm(dim=-1, keepdim=True)], -1)
                scene = scene + self.token_in(feats)
            else:
                scene = scene + self.scene_pos
            keep = obs.get("scene_keep")
            if keep is None:
                return torch.cat([null, scene], 1), None
            if tuple(keep.shape) != (b, scene.shape[1]):
                raise ValueError(
                    f"obs['scene_keep'] must have shape {(b, scene.shape[1])}, got {tuple(keep.shape)}"
                )
            pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=scene.device), ~keep.bool()], 1)
            return torch.cat([null, scene], 1), pad
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
        self,
        x_t: torch.Tensor,
        t,
        v: torch.Tensor,
        obs: dict | None = None,
        precomputed: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t (B, H, >= n), v (B, H, >= n), t a float or (B,), obs optional -> (A (B, H, n, n), b (B, H, n)).
        ``precomputed`` is ``self.tokens(obs, B)`` made earlier (what ``bind`` does once per observation); the
        tokens are made here without it."""
        b, h, n = x_t.shape[0], x_t.shape[1], self.n_dims
        tokens, pad = self.tokens(obs, b) if precomputed is None else precomputed
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

    @staticmethod
    def reduce(tokens: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """The fixed linear token reduction of a reduced ``vlm_*`` module: ``(tokens - mean) @ basis`` in float32,
        stored as float16. ``tokens`` (..., scene_dim), ``mean`` (scene_dim,), ``basis`` (scene_dim, rank)."""
        return ((tokens.float() - mean) @ basis).to(torch.float16)

    @staticmethod
    def field_loss(
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
        two parts (``fm``, ``identity``) as detached tensors (the caller converts them when it logs: no host sync
        per step here)."""
        x = (1 - t[:, None, None]) * z + t[:, None, None] * chunk
        with torch.no_grad():
            v = velocity(x, t)
        mat, b = task(x, t, v)
        fm = ((torch.einsum("bsij,bsj->bsi", mat, v) + b - (chunk - z)) ** 2).mean()
        identity = ((mat - torch.eye(n_dims, device=x.device)) ** 2).mean() + (b**2).mean()
        return fm + lambda_id * identity, {"fm": fm.detach(), "identity": identity.detach()}


def _adapter_kind(has_user: bool, has_task: bool) -> str:
    """An adapter's kind from which parts it has: composed, user, task or none."""
    return {
        (True, True): "composed",
        (True, False): "user",
        (False, True): "task",
        (False, False): "none",
    }[(has_user, has_task)]


class AdaptedField:
    """Applies MARS modules to a velocity: task (both directions), then user (backward only, through
    ``LearnedUserModule.apply_correction``: a learned module's ``velocity``, any other callable's per-step matrices).
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
                head = LearnedUserModule.apply_correction(self.user_module, x_t, t, head)
                result = torch.cat([head, result[..., n:]], dim=-1)
        return result


def _obs_device(obs: dict) -> torch.device | None:
    """The device of an observation: its first tensor's (depth first through nested dicts), or None."""
    for value in obs.values():
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict) and (device := _obs_device(value)) is not None:
            return device
    return None


class AdapterModules(nn.Module):
    """The MARS adapter holder of the cores, the policy and the LIBERO session: a user part (``matrix``, a fixed F,
    and/or ``user``, a ``LearnedUserModule`` or any (x_t, t) -> (B, H, n, n) callable) applied in the reversal only,
    and a ``task`` module (a ``LearnedTaskModule``, bound to each observation, or any (x_t, t, v) -> (A, b) callable)
    applied in both directions: forward task(v), backward user(task(v)). All None = off (plain FRS). Consumers read
    the holder on every use, so a replaced matrix or module takes effect at once.

    ``matrix`` is in the operator's (environment) units: one (n, n) or one per trial (B, n, n), a float64 array,
    checked on assignment (``readout.checked_matrix`` at ``n_dims``, or ``N_ACTION_DIMS`` when ``n_dims`` is
    None). ``normalized_matrix(std)`` is it in the policy's normalized coordinates, S^-1 F S, by the postprocessor's
    action statistics; ``field`` is the ``AdaptedField`` the flows apply, that matrix (normalized when given ``std``)
    as a tensor. The modules act on ``n_dims`` dims (2 on the cursor); ``n_dims=None`` lets each act on its own
    ``n_dims`` (``N_ACTION_DIMS`` for a callable without one) and checks the matrix at ``N_ACTION_DIMS``: the
    7-D holder of the pi0.5 cores and the LIBERO session (``AdapterModules(n_dims=None)``). ``label``, ``user_label``
    and ``task_label`` name the parts for display (``describe``); ``load_modules`` reads checkpoints into the slots,
    ``load`` one module into a new holder, ``save_module`` writes one."""

    KINDS: dict[str, type[nn.Module]] = {
        "user": LearnedUserModule,
        "task": LearnedTaskModule,
    }  # checkpoint kinds

    def __init__(self, user=None, task=None, matrix=None, n_dims: int | None = 2, label: str | None = None):
        super().__init__()
        self.user, self.task, self.n_dims = user, task, n_dims
        self.meta: dict = {}  # a loaded checkpoint's free-form meta
        self.matrix = matrix
        self.label = label
        self.user_label: str | None = None
        self.task_label: str | None = None

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        if value is None:
            self._matrix = None
            return
        if torch.is_tensor(value):
            value = value.detach().cpu()
        size, where = self.n_dims or N_ACTION_DIMS, "adapter matrix"
        try:
            per_trial = np.ndim(value) == 3
        except (TypeError, ValueError):  # ragged rows: the check below names them
            per_trial = False
        if per_trial:  # one per trial, each checked
            self._matrix = np.stack([readout.checked_matrix(one, size, where) for one in value])
        else:
            self._matrix = readout.checked_matrix(value, size, where)

    @property
    def kind(self) -> str:
        return _adapter_kind(self.user is not None or self._matrix is not None, self.task is not None)

    def format(self) -> str:
        """The holder's matrix on one line, for the terminal (``format_rows``; one per trial, side by side); "off"
        when it holds none."""
        if self._matrix is None:
            return "off"
        return " ".join(
            self.format_rows(m) for m in (self._matrix if self._matrix.ndim == 3 else [self._matrix])
        )

    @staticmethod
    def format_rows(matrix: np.ndarray) -> str:
        """Rows of a small matrix on one line, for the terminal: `[+0.94 -0.34 +0.00; ...]`."""
        return "[" + "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in matrix) + "]"

    def describe(self) -> str:
        """The adapter in force, for the terminal: its kind (MARS-U, MARS-T, MARS) and its parts' labels."""
        if self.kind == "none":
            return "MARS adapter: off"
        parts = []
        if self._matrix is not None:
            identity = (
                " (identity, a no-op)" if np.allclose(self._matrix, np.eye(self._matrix.shape[-1])) else ""
            )
            parts.append(f"fixed user matrix F = {self.label}{identity} = {self.format()}")
        if self.user is not None:
            parts.append(f"learned user module {self.user_label or type(self.user).__name__}")
        if self.task is not None:
            parts.append(f"task module {self.task_label or type(self.task).__name__}")
        name = {"user": "MARS-U", "task": "MARS-T", "composed": "MARS"}[self.kind]
        return f"MARS adapter: {name} ({'; '.join(parts)})."

    def load_modules(self, user_path=None, task_path=None, device="cpu") -> None:
        """Load learned modules from checkpoints into the user and/or task slot (None leaves a slot alone), each
        labelled with its path; raises when a checkpoint holds the other kind."""
        if user_path is not None:
            module, kind, _ = AdapterModules.load_module(user_path, device)
            if kind != "user":
                raise ValueError(f"{user_path} holds a {kind} module, not a user module")
            self.user, self.user_label = module, str(user_path)
        if task_path is not None:
            module, kind, _ = AdapterModules.load_module(task_path, device)
            if kind != "task":
                raise ValueError(f"{task_path} holds a {kind} module, not a task module")
            self.task, self.task_label = module, str(task_path)

    def normalized_matrix(self, std: np.ndarray) -> np.ndarray | None:
        """The fixed matrix, an environment-space velocity transform, in normalized coordinates: S^-1 F S (None
        when there is none). Velocities have no mean offset. Apply this to all of the matrix's dims, so literal
        matrices that mix translation, orientation and gripper use the same convention. A uniform ``std`` (the
        cursor's scalar action scale) returns the matrix itself, so its values are the held ones bit for bit."""
        if self._matrix is None or np.all(std == std.flat[0]):
            return self._matrix
        return self._matrix * std[np.newaxis, :] / std[:, np.newaxis]

    def field(
        self, obs: dict | None = None, *, std: np.ndarray | None = None, like=None
    ) -> AdaptedField | None:
        """The ``AdaptedField`` the flows apply, or None when the holder is off: the fixed matrix (normalized by
        ``std``, the postprocessor's action std over its dims, when given: ``normalized_matrix``) as a tensor of
        ``like``'s dtype on ``like``'s device, or without ``like`` float32 on the observation's device (its first
        tensor's); the user module; the task module bound to ``obs`` when given."""
        if self.kind == "none":
            return None
        matrix = self._matrix if std is None else self.normalized_matrix(std)
        tensor = None
        if matrix is not None:
            if like is not None:
                tensor = torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
            else:
                device = None if obs is None else _obs_device(obs)
                tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        task = self.task
        if obs is not None and task is not None and hasattr(task, "bind"):
            task = task.bind(obs)
        return AdaptedField(tensor, self.user, task, n_dims=self.n_dims)

    def adapt(self, obs: dict, x: torch.Tensor, t, v: torch.Tensor, backward: bool) -> torch.Tensor:
        field = self.field(obs)
        return v if field is None else field.adapt(x, t, v, backward)

    @classmethod
    def load(cls, path, device="cpu") -> "AdapterModules":
        """One module of a module checkpoint (``save_module``'s format) in its slot (``kind`` user or task), in
        eval mode; the checkpoint's free-form meta is ``meta``."""
        module, kind, meta = AdapterModules.load_module(path, device)
        adapter = cls(**{kind: module}).to(device).eval()
        adapter.meta = meta
        return adapter

    @staticmethod
    def save_module(module: nn.Module, path, kind: str, meta: dict | None = None) -> Path:
        """Save a MARS module checkpoint: ``kind`` (user | task, matching the module's class), its config,
        weights and free-form meta."""
        if kind not in AdapterModules.KINDS or type(module) is not AdapterModules.KINDS[kind]:
            raise ValueError(
                f"kind must be one of {sorted(AdapterModules.KINDS)} and match the module, got {kind!r}"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"kind": kind, "config": module.config, "state_dict": module.state_dict(), "meta": meta or {}},
            path,
        )
        return path

    @staticmethod
    def read_checkpoint(path, device="cpu", mmap: bool = False) -> dict:
        """A MARS module checkpoint as saved (``save_module``'s dict: kind, config, state_dict, meta), its kind
        checked. With ``mmap`` the tensors are memory-mapped, not read (a read of the config alone: only the small
        pickle comes off the disk). Task checkpoints written while there were two task module classes carry ``arch``:
        ``"obstacle"`` is this ``LearnedTaskModule``; ``"mlp"`` (and an MLP's weights with no arch, the first LIBERO
        files) is the removed MLP module, which raises."""
        ckpt = torch.load(
            Path(path), map_location=device, weights_only=True, mmap=mmap
        )  # tensors, numbers, strings only
        kind = ckpt.get("kind")
        if kind not in AdapterModules.KINDS:
            raise ValueError(f"{path}: unknown adapter module kind {kind!r} (format {ckpt.get('format')!r})")
        if kind == "task" and (
            ckpt.get("arch", "obstacle") != "obstacle" or "net.0.weight" in ckpt["state_dict"]
        ):
            raise ValueError(
                f"{path}: a task module of arch {ckpt.get('arch', 'mlp')!r}, the MLP task module, which was removed; "
                "retrain it as a LearnedTaskModule"
            )
        return ckpt

    @staticmethod
    def load_module(path, device="cpu") -> tuple[nn.Module, str, dict]:
        """Load a MARS module checkpoint (``read_checkpoint``): (module in eval mode, kind, meta)."""
        ckpt = AdapterModules.read_checkpoint(path, device)
        kind = ckpt["kind"]
        module = AdapterModules.KINDS[kind](**ckpt["config"])
        module.load_state_dict(ckpt["state_dict"])
        return module.to(device).eval().requires_grad_(False), kind, ckpt.get("meta", {})

    @staticmethod
    def from_cursor_checkpoint(ckpt: dict) -> dict:
        """A cursor example module checkpoint (``{kind?, config, state_dict, args, obstacle}``; the first
        adapters, saved without ``kind``, are user modules) in the ``save_module`` format. A user
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


def as_adapter(adapter, device="cpu", n_dims: int | None = 2) -> AdapterModules | None:
    """None or an ``AdapterModules`` as is; a ``LearnedTaskModule`` in the task slot; a user module or any
    (x_t, t) callable in the user slot; anything else a fixed matrix F ((n, n) or (B, n, n)), held as the operator's
    units are (``AdapterModules.matrix``). ``device`` does not apply to a matrix, which takes the device of each use
    (``AdapterModules.field``)."""
    if adapter is None or isinstance(adapter, AdapterModules):
        return adapter
    if isinstance(adapter, LearnedTaskModule):
        return AdapterModules(task=adapter, n_dims=n_dims)
    if callable(adapter):
        return AdapterModules(user=adapter, n_dims=n_dims)
    return AdapterModules(matrix=adapter, n_dims=n_dims)


class Flow:
    """A velocity field and an adapter in one flow-time convention: the one Euler integrator of the steering
    methods and the trainers, and what is built on it (FRS's latent and schedule, the validation's regeneration).

    ``velocity(x, t)`` is the policy's field; ``adapter`` is anything ``adapt_velocity`` takes (None, a normalized
    (n, n) or per-trial (B, n, n) matrix F, an ``AdaptedField``, or an ``AdapterModules`` holder without a fixed
    matrix). ``noise_time`` is the flow time of noise: 1.0 is pi0.5's convention (t = 1 noise, t = 0 the action),
    0.0 the cursor's. With ``batch_time`` the flow times reach ``velocity`` and the adapter as (B,) float32
    tensors (the cursor policy's form), else as floats. ``reverse`` and ``forward`` are the reversal toward noise
    and the unsteered forward pass at the policy's own step of 1/N; the statics ``Flow.schedule``,
    ``Flow.uniform_reference``, ``Flow.heading_cosine`` and ``Flow.paired_chunks`` need no field."""

    def __init__(
        self,
        velocity: Callable[[torch.Tensor, float | torch.Tensor], torch.Tensor],
        adapter=None,
        noise_time: float = 1.0,
        batch_time: bool = False,
    ):
        Flow.check_noise_time(noise_time)  # any other value would make toward_noise's step 0
        self.velocity, self.adapter, self.noise_time, self.batch_time = (
            velocity,
            adapter,
            noise_time,
            batch_time,
        )

    @staticmethod
    def check_noise_time(noise_time: float) -> None:
        """Raise unless ``noise_time`` is one of the two conventions: 1.0 (pi0.5) or 0.0 (the cursor)."""
        if noise_time not in (0.0, 1.0):
            raise ValueError(
                f"noise_time must be 1.0 (pi0.5: t = 1 noise) or 0.0 (the cursor: t = 0 noise), got {noise_time}"
            )

    @staticmethod
    def adapt_velocity(adapter, x_t, t, v, backward: bool):
        """`adapter` is None, a normalized (n, n) or per-trial (B, n, n) matrix (the legacy fixed-F form), an
        AdaptedField, or an AdapterModules holder without a fixed matrix used directly (its ``field(like=x_t)``; a
        holder's matrix is in the operator's units: convert the holder with ``field(std=..., like=...)`` first)."""
        if adapter is None:
            return v
        if isinstance(adapter, torch.Tensor):
            adapter = AdaptedField(adapter)
        elif isinstance(adapter, AdapterModules):
            if adapter.matrix is not None:
                raise ValueError(
                    "an AdapterModules' fixed matrix is in the operator's units; pass its field(std=..., like=...) "
                    "instead"
                )
            adapter = adapter.field(like=x_t)
            if adapter is None:
                return v
        return adapter.adapt(x_t, t, v, backward)

    def toward_noise(self, steps: int) -> float:
        """The signed Euler step of ``steps`` per unit of flow time, toward noise: +1/N in pi0.5's convention, -1/N in
        the cursor's (the reversal's step; the forward pass takes its negative)."""
        return (2 * self.noise_time - 1) / steps

    def integrate(
        self, x: torch.Tensor, t0: float, h: float, n: int, backward: bool = False, grad: bool = False
    ) -> torch.Tensor:
        """``n`` Euler steps ``x <- x + h * adapt(v(x, t))`` at flow times ``t = t0 + i * h``, in either time
        convention: the sign of ``h`` picks the direction of flow time, ``backward`` whether the adapter's user part
        (the fixed matrix F and/or the user module) acts (the reversal toward noise); a task module acts in both
        directions. ``t`` is passed to ``velocity`` and the adapter as the float ``t0 + i * h``, or with
        ``batch_time`` as a (B,) float32 tensor of it on ``x``'s device. With ``grad`` False the loop runs under
        ``torch.no_grad()``; with it True the caller's grad mode holds (differentiable when grad is enabled, for
        training)."""
        with torch.enable_grad() if grad and torch.is_grad_enabled() else torch.no_grad():
            for i in range(n):
                t = t0 + i * h
                if self.batch_time:
                    t = torch.full((len(x),), t, device=x.device)
                x = x + h * Flow.adapt_velocity(self.adapter, x, t, self.velocity(x, t), backward=backward)
        return x

    def reverse(self, x: torch.Tensor, num_steps: int, n_reversal_steps: int | None = None) -> torch.Tensor:
        """Integrate the flow backward (Euler) from a clean chunk toward its latent noise: in pi0.5's convention
        t: 0 -> 1 with dt = +1/num_steps (PI05Pytorch.sample_actions steps the same field with dt = -1/num_steps
        from t=1, noise, to t=0, the action), in the cursor's t: 1 -> 0. This is the inversion of Flow Reversal
        Steering (Tang et al. 2026, arXiv:2606.13675).

        The step size is always 1/`num_steps`, i.e. the policy's own schedule. `n_reversal_steps` is how many of
        those steps to take: the default (None) runs the whole way to pure noise, while a smaller count stops early
        and returns a chunk at n_reversal_steps / num_steps of the way, still carrying some of the reference. The
        caller must then start the forward flow at that time. The adapter acts as in the backward pass (MARS: task
        module, then user module; a fixed F, x_t += h * (F @ v) on the first n action dims, F = I an exact no-op).
        Differentiable when grad is enabled."""
        steps = num_steps if n_reversal_steps is None else n_reversal_steps
        return self.integrate(
            x, 1.0 - self.noise_time, self.toward_noise(num_steps), steps, backward=True, grad=True
        )

    def forward(self, x: torch.Tensor, num_steps: int, stop_at: int | None = None) -> torch.Tensor:
        """Integrate the flow forward (Euler) from noise toward the action, stopping at stop_at / num_steps of the
        reversal's way. Mirrors PI05Pytorch.sample_actions without hooks: x_{t-h} = x_t - h * v(x_t, t). With
        ``stop_at`` None it runs all the way to the action (the unsteered chunk); with ``stop_at = n`` it takes
        ``num_steps - n`` steps and returns the unsteered state where a reversal stopped after ``n`` steps would
        sit. The adapter applies its task module only: the user module acts on the reversal alone."""
        steps = num_steps if stop_at is None else num_steps - stop_at
        return self.integrate(x, self.noise_time, -self.toward_noise(num_steps), steps, grad=True)

    @staticmethod
    def schedule(steps: int, n_reverse_steps: int | None, noise_time: float = 1.0) -> tuple[float, int]:
        """Where FRS's forward pass starts and how many Euler steps it takes once the reference has been reversed
        n of the policy's N = ``steps`` steps (None: all of them): from the reversal's landing time back to the
        action in n steps of the policy's own 1/N, mirroring the reversal (n = N: from pure noise in N steps). A
        steered chunk costs 2n velocity evaluations (n back, n forward), N + n when a partial reversal hands steps
        back to the policy (N - n more for the unsteered latent)."""
        Flow.check_noise_time(noise_time)
        n = steps if n_reverse_steps is None else n_reverse_steps
        _ChunkSteering.check_steps("n_reverse_steps", n, 1, steps, "Euler steps")
        action_time = 1.0 - noise_time
        return action_time + (noise_time - action_time) * n / steps, n

    def latent(
        self,
        reference: torch.Tensor,
        noise: torch.Tensor,
        steps: int,
        n_reverse_steps: int | None = None,
        delegated: torch.Tensor | None = None,
        grad: bool = False,
    ) -> torch.Tensor:
        """Flow Reversal Steering's (and MARS's) latent: the ``reference`` chunk integrated backward n of N =
        ``steps`` Euler steps of 1/N toward noise (the adapter's user part acting; None: all the way, Tang et al.'s
        method), then the elements ``delegated`` marks (a bool mask broadcastable to the chunk: the dims the
        operator does not command, padding, the steps past the pinned prefix) replaced by the unsteered latent at
        the same flow time -- the chunk's own ``noise`` for a full reversal, else ``noise`` run forward N - n
        steps -- for the policy to fill in (Tang et al.'s noise-space in-painting, App. D.1). The forward pass
        starts from it on ``schedule``'s schedule."""
        _, n = Flow.schedule(steps, n_reverse_steps, self.noise_time)
        h = self.toward_noise(steps)
        latent = self.integrate(reference, 1.0 - self.noise_time, h, n, backward=True, grad=grad)
        if delegated is not None and bool(delegated.any()):
            unsteered = (
                noise if n == steps else self.integrate(noise, self.noise_time, -h, steps - n, grad=grad)
            )
            latent = torch.where(delegated.to(latent.device), unsteered, latent)
        return latent

    @torch.no_grad()
    def regenerate(self, chunk: torch.Tensor, num_steps: int) -> torch.Tensor:
        """FRS at full reversal: ``chunk`` reversed through the adapted backward field (task and user modules) all
        the way to noise, then regenerated through the forward field (the task module alone: ``AdaptedField``
        applies the user part backward only). The trainers' validation: what the policy makes of the operator's
        command through the modules."""
        return self.forward(self.reverse(chunk, num_steps), num_steps)

    @staticmethod
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

    @staticmethod
    def held_reference(commands: torch.Tensor, horizon: int, step: float, action_scale) -> torch.Tensor:
        """The reference FRS steers a ``sample_chunk`` policy (the cursor's) toward: ``commands`` in the operator's
        units ((B, n), held over the ``horizon`` chunk steps, or (B, H, n) one per step), a full deflection moving
        ``step`` action units per control step (max speed x dt), in the policy's normalized units (``action_scale``
        action units each): ``command * step / action_scale``."""
        per_step = commands if commands.dim() == 3 else commands[:, None].expand(-1, horizon, -1)
        return per_step * step / action_scale

    @staticmethod
    def heading_cosine(chunk_a: torch.Tensor, chunk_b: torch.Tensor, n_dims: int = 3) -> torch.Tensor:
        """Cosine between two chunks' total displacements (the sum over steps of their first ``n_dims`` dims), per
        sample: how well a regenerated chunk heads where its label does."""
        return nn.functional.cosine_similarity(
            chunk_a[..., :n_dims].sum(1), chunk_b[..., :n_dims].sum(1), dim=-1
        )

    @staticmethod
    def _rng_state() -> tuple:
        """CPU and CUDA generator states, so a second sampling pass can reuse the first one's noise."""
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return torch.get_rng_state(), cuda

    @staticmethod
    def _set_rng_state(state: tuple) -> None:
        cpu, cuda = state
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)

    @staticmethod
    def paired_chunks(
        policy, batch, common: dict | None = None, **steer_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The steered chunk and the policy's own plan for the same observation and the same noise.

        Returns (steered, plain), the steered pass first. The generator state is restored between the passes so the
        plain chunk starts from the Gaussian sample the steered one used, and the state after the steered pass is
        restored afterwards, so a run's random stream is exactly what it would be without the second pass. `common`
        kwargs go to both passes (a MARS task module shapes the policy's own plan too); `steer_kwargs` only to the
        steered one.
        """
        common = common or {}
        before = Flow._rng_state()
        chunk = policy.predict_action_chunk(batch, **common, **steer_kwargs)
        after = Flow._rng_state()
        Flow._set_rng_state(before)
        plain = policy.predict_action_chunk(batch, **common)
        Flow._set_rng_state(after)
        return chunk, plain


# ---------------------------------------------------------------- the steering methods, one core each
#
# The steering wrappers below run these, on pi0.5 and on the cursor policy alike, so a method's mechanics (what is
# written or reversed, where, and the forward schedule) are decided in one place. Either flow-time convention:
# ``noise_time`` 1.0 is pi0.5's (t = 1 noise, t = 0 action), 0.0 the cursor's.


class TeleopSource(Protocol):
    """What the steering cores read: the operator's current command."""

    @property
    def translation(self) -> np.ndarray:  # shape (3,), env units in [-1, 1]
        ...

    @property
    def gripper(self) -> float:  # GRIPPER_OPEN or GRIPPER_CLOSE
        ...


def active_rows(commands, deadband: float) -> torch.Tensor:
    """(B,) bool: the rows of ``commands`` ((B, n) commands, or (B, H, n) a command per chunk step, active when any
    step is; a tensor or anything ``np.asarray`` takes) whose largest deflection is at least ``deadband``."""
    magnitude = (
        torch.as_tensor(commands if torch.is_tensor(commands) else np.asarray(commands)).abs().amax(-1)
    )
    return (magnitude.amax(-1) if magnitude.dim() == 2 else magnitude) >= deadband


class _ChunkSteering:
    """What both steering wrappers share. ``steered_chunk(batch, commands)`` turns a batch of operator commands,
    one per row (a row inside ``deadband`` counts as idle and gets the policy's own chunk), into the steered chunk
    and the policy's own plan for the same noise; with ``paired=False`` the plan is None when every row steers (one
    pass, the same chunk and the same generator state). As a policy (pi0.5's ``select_action``), once per chunk the
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
        """(B,) bool: the rows of ``commands`` outside the deadband (``active_rows``)."""
        return active_rows(commands, self.deadband)

    @staticmethod
    def check_steps(name: str, value, low: int, high: int, unit: str, optional: bool = False) -> None:
        """Raise unless ``value`` is an integer (not a bool) in [``low``, ``high``] ``unit``, or None when
        ``optional``: the one check of the step knobs (tau, lambda, n_reversal_steps, pinned_steps)."""
        if optional and value is None:
            return
        if not (isinstance(value, int) and not isinstance(value, bool) and low <= value <= high):
            none = "None or " if optional else ""
            raise ValueError(f"{name} must be {none}an integer in [{low}, {high}] ({unit}), got {value!r}")

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

    @staticmethod
    def action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def velocity_hook(field):
        """The velocity hook a field's task module needs in every forward pass (the policy's own plan too), or None."""
        if field is None or getattr(field, "task_module", None) is None:
            return None

        @torch.compiler.disable
        def hook(step: int, time_: float, x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return field.adapt(x_t, time_, v, backward=False)

        return hook


class FlowControlSteering(_ChunkSteering):
    """Shared flow control (Kao et al. 2026) on any flow policy with pi0.5's sampler interface; the pi0.5 wrapper.

    While the operator pushes, the command is written into ``x_t`` as the chunk's initial condition
    (``write``): scaled by ``ic_gain`` into the noise prior's units (a full push writes ``ic_gain``
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
            mean, std = self.action_mean_std(postprocessor)
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
        self.check_steps("tau", value, 1, chunk, "chunk steps", optional=True)
        self._tau = chunk if value is None else value

    @property
    def lam(self) -> int:
        return self._lam

    @lam.setter
    def lam(self, value: int) -> None:
        self.check_steps("lam", value, 0, self._policy.config.num_inference_steps, "Euler steps")
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

    @staticmethod
    def write(
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
        return self.write(x_t, step, self._pending_target, self._tau, self._lam, dims=self._dims, steer=steer)

    def steered_chunk(
        self, batch, commands, field=None, paired: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The chunk for ``commands`` ((B, n), or (B, H, n) per step; None: every row idle) and the policy's own
        plan from the same noise (None when ``paired`` is False and every row steers: the plan is then not drawn).
        ``field`` (an ``AdaptedField``) adds its task module to every forward pass."""
        hook = self.velocity_hook(field)
        common = {} if hook is None else {"velocity_hook": hook}
        steer = None if commands is None else self._active(commands)
        if steer is None or not bool(steer.any()):
            self._pending_command = None
            chunk = self._policy.predict_action_chunk(batch, **common)
            return chunk, chunk
        self._pending_command = commands
        self._pending_target = None  # resolved on the device by the first write
        every = bool(steer.all())
        self._pending_steer = None if every else steer[:, None, None]
        if every and not paired:
            return self._policy.predict_action_chunk(batch, **common, x_t_hook=self._x_t_hook), None
        chunk, plain = Flow.paired_chunks(self._policy, batch, common, x_t_hook=self._x_t_hook)
        return self._per_row(chunk, plain, steer), plain

    def _sample_chunk(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # Hold one consumed command (including its noise) for the whole chunk,
        # matching FRS and the synthetic operator's refresh cadence.
        command = self._read_command(read=bool(self._lam))
        return self.steered_chunk(batch, None if command is None else np.asarray(command)[None])


class FlowReversalSteering(_ChunkSteering):
    """Flow Reversal Steering (Tang et al. 2026) on any flow policy with pi0.5's sampler interface; the pi0.5
    wrapper (shared_flow_reversal_steering).

    While the operator pushes, a reference chunk built from the command (``reference_chunk``: by default a
    uniform-velocity chunk in the commanded direction, rotation zero, gripper held at the model's last executed
    command, `gripper_open` before the first chunk; with ``reference_step`` the command held at that step per control
    step; any ``reference_fn(commands) -> (B, H, D)`` in the model's normalized space otherwise) is
    integrated *backward* through the policy's own velocity field to the latent noise that maps to it
    (``Flow.latent``); the normal forward flow then denoises from that noise instead of a random one, on
    ``Flow.schedule``'s schedule. The executed action is entirely the model's output -- the reference only picks the
    starting noise, so what comes out is the generalist action mode nearest your intent. Idle input = pure policy.

    An optional adapter (MARS) adapts the velocity field: an `AdapterModules` holder here (its user part, a fixed
    7x7 matrix F in the operator's units, normalized by the postprocessor's action std, or a learned U(x_t, t), in the
    reversal only, its task module in both directions, including the forward flow that produces the executed chunk and
    the policy's own plan; `AdapterModules(n_dims=None)`, the 7-D holder), or an ``AdaptedField`` passed per call to
    ``steered_chunk`` (the steering policy binds one to each observation).

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
        adapter: AdapterModules | None = None,
        n_reversal_steps: int | None = None,
        operator_dims: tuple[str, ...] | None = ("translation",),
        pinned_steps: int | None = -1,
        deadband: float = DEADBAND,
        gripper_open: float = GRIPPER_OPEN,
        reference_fn: Callable | None = None,
        reference_step: float | None = None,
    ):
        super().__init__(policy, source, deadband)
        self._adapter = adapter
        self.gripper_open = gripper_open
        self._reference_fn = reference_fn
        self._reference_step = reference_step
        self.n_reversal_steps = n_reversal_steps  # validated by the setter below
        self._operator_dims = None if operator_dims is None else tuple(operator_dims)
        # The setter validates the pinned steps and builds the delegation mask (checking operator_dims)
        self.pinned_steps = policy.config.n_action_steps if pinned_steps == -1 else pinned_steps
        self._adapter_cache: tuple | None = None
        self._mean = self._std = None
        if postprocessor is not None:
            mean, std = self.action_mean_std(postprocessor)
            self._mean, self._std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]
            if self._has_gripper:
                self._gripper_ref = self._normalize_gripper(gripper_open)
        elif reference_fn is None:
            raise ValueError(
                "a FlowReversalSteering needs the postprocessor's action statistics or a reference_fn"
            )
        self._pending_reference: torch.Tensor | None = None
        self._last_reference: torch.Tensor | None = None
        self.steered_chunks = 0  # per-rollout count of chunks started from inverted noise
        self.reconstruction_errors: list[float] = []  # per steered chunk, in std units

    @staticmethod
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
            SteeringConfig.check_operator_dims(operator_dims)
            mask = torch.ones(chunk_size, max_action_dim, dtype=torch.bool)
            for name in operator_dims:
                mask[:, OPERATOR_DIM_GROUPS[name]] = False
        if pinned_steps is not None:
            mask[pinned_steps:, :] = True
        return mask

    @property
    def operator_dims(self) -> tuple[str, ...] | None:
        return self._operator_dims

    @operator_dims.setter
    def operator_dims(self, value) -> None:
        cfg = self._policy.config
        dims = None if value is None else tuple(value)
        self._delegated = self.delegation_mask(cfg.chunk_size, cfg.max_action_dim, dims, self._pinned_steps)
        self._operator_dims = dims

    @property
    def pinned_steps(self) -> int | None:
        return self._pinned_steps

    @pinned_steps.setter
    def pinned_steps(self, value: int | None) -> None:
        cfg = self._policy.config
        self.check_steps("pinned_steps", value, 1, cfg.chunk_size, "chunk steps", optional=True)
        self._delegated = self.delegation_mask(cfg.chunk_size, cfg.max_action_dim, self._operator_dims, value)
        self._pinned_steps = value

    @property
    def n_reversal_steps(self) -> int | None:
        return self._n_reversal_steps

    @n_reversal_steps.setter
    def n_reversal_steps(self, value: int | None) -> None:
        total = self._policy.config.num_inference_steps
        self.check_steps("n_reversal_steps", value, 1, total, "Euler steps", optional=True)
        self._n_reversal_steps = None if value == total else value

    @staticmethod
    def has_gripper_dim(mean: np.ndarray | None) -> bool:
        """Whether action statistics (their ``mean``) span the 7-D operator space, its gripper dim included (pi0.5's;
        the cursor's two dims do not): the space whose reference rule is ``Flow.uniform_reference``."""
        return mean is not None and mean.shape[0] > GRIPPER_DIM

    @property
    def _has_gripper(self) -> bool:
        """Whether this core's action statistics have the gripper dim: only then is a gripper reference held."""
        return self.has_gripper_dim(self._mean)

    def _normalize_gripper(self, gripper: float) -> float:
        return float((gripper - self._mean[GRIPPER_DIM]) / self._std[GRIPPER_DIM])

    def reset(self) -> None:
        super().reset()
        if self._has_gripper:
            self._gripper_ref = self._normalize_gripper(self.gripper_open)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, command) -> torch.Tensor:
        """The reference for ``command`` ((n,) or (B, n), or (B, H, n) per step with a ``reference_fn`` or a
        ``reference_step``), in the model's normalized space: the ``reference_fn``'s; else, with a ``reference_step``,
        the command held over the chunk (or one per step), a full deflection moving ``reference_step`` action units
        per control step, ``(per_step * reference_step - mean) / std`` in that order, in torch and the command's dtype,
        over its n dims (zero padding beyond them); else the uniform-velocity chunk of env-space translation
        (``Flow.uniform_reference``).

        Which default applies is a property of the action space, not of the policy: ``uniform_reference`` is the
        7-D operator space's rule (translation, rotation, the held gripper dim), and an action space without that
        gripper dim (the cursor's two dims) has no such rule and takes the held command. With ``mean`` 0 (the
        cursor's statistics) the held form equals ``Flow.held_reference``'s ``per_step * step / action_scale`` bit
        for bit."""
        if self._reference_fn is not None:
            return self._reference_fn(command)
        cfg = self._policy.config
        if self._reference_step is None:
            if not self._has_gripper:
                raise ValueError(
                    "an action space without the gripper dim needs a reference_step or a reference_fn (its "
                    f"statistics have {self._mean.shape[0]} dims; Flow.uniform_reference is the 7-D operator space's)"
                )
            return Flow.uniform_reference(
                command, self._gripper_ref, self._mean, self._std, cfg.chunk_size, cfg.max_action_dim
            )
        command = torch.as_tensor(command if torch.is_tensor(command) else np.asarray(command))
        command = command[None] if command.dim() == 1 else command
        per_step = command if command.dim() == 3 else command[:, None].expand(-1, cfg.chunk_size, -1)
        n = per_step.shape[-1]
        mean, std = (
            torch.as_tensor(a[:n], dtype=per_step.dtype, device=per_step.device)
            for a in (self._mean, self._std)
        )
        reference = (per_step * self._reference_step - mean) / std
        return (
            nn.functional.pad(reference, (0, cfg.max_action_dim - n)) if n < cfg.max_action_dim else reference
        )

    def _adapter_field(self, like: torch.Tensor) -> AdaptedField | None:
        """The wrapper's adapter as applied to normalized velocities on `like`'s device (``field(std=..., like=...)``);
        the fixed matrix is converted once and rebuilt only when the holder's matrix or modules change. Raises
        ``ValueError`` for a fixed matrix when the core has no postprocessor (no std to normalize it by)."""
        if self._adapter is None or self._adapter.kind == "none":
            self._adapter_cache = None
            return None
        if self._adapter.matrix is not None and self._std is None:
            raise ValueError(
                "a fixed adapter matrix is in the operator's units and is normalized by the postprocessor's action "
                "std; pass the postprocessor, or a field(std=..., like=...) per call"
            )
        key = (
            self._adapter.matrix,
            self._adapter.user,
            self._adapter.task,
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
            self._adapter_cache = (key, self._adapter.field(std=self._std, like=like))
        return self._adapter_cache[1]

    def steered_chunk(
        self, batch, commands, field=None, paired: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The chunk for ``commands`` ((B, n), or (B, H, n) per step; None: every row idle) and the policy's own
        plan from the same noise (None when ``paired`` is False and every row steers: the plan is then not drawn).
        ``field`` (an ``AdaptedField``) replaces the wrapper's adapter for this call."""
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
            return Flow(velocity, adapted(noise), self._noise_time).latent(
                reference,
                noise,
                cfg.num_inference_steps,
                self._n_reversal_steps,
                delegated=self._delegated,
                grad=True,
            )

        task = field.task_module if field is not None else getattr(self._adapter, "task", None)
        common = {"velocity_hook": velocity_hook} if task is not None else {}
        steer = None if commands is None else self._active(commands)
        if steer is None or not bool(steer.any()):
            self._pending_reference = None
            chunk = self._policy.predict_action_chunk(batch, **common)
            return chunk, chunk
        self._pending_reference = self.reference_chunk(commands)
        schedule = {}  # the full reversal: the usual schedule from noise in N steps
        if self._n_reversal_steps is not None:  # part-way: from the reversal's landing time, n steps of 1/N
            start, n = Flow.schedule(
                cfg.num_inference_steps, self._n_reversal_steps, noise_time=self._noise_time
            )
            schedule = {"flow_start_time": start, "num_forward_steps": n}
        if not paired and bool(steer.all()):
            return self._policy.predict_action_chunk(batch, **common, noise_fn=noise_fn, **schedule), None
        chunk, plain = Flow.paired_chunks(self._policy, batch, common, noise_fn=noise_fn, **schedule)
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
        if self._has_gripper:
            self._gripper_ref = float(executed[-1, GRIPPER_DIM])
        if self._last_reference is not None:
            target = self._last_reference[0, : executed.shape[0], :3].float().cpu()
            self.reconstruction_errors.append((executed[:, :3] - target).abs().mean().item())
        return chunk, plain


# ---------------------------------------------------------------- the lerobot policy wrapper

# The steering policy's own batch keys, not the observation's: the operator's command, and a training batch's
# (``SteeringPolicy.forward``'s, which the pre-processor carries through untouched)
COMMAND = "command"
TRAINING_KEYS = ("kind", "chunk", "latent", "noise", "time", "weight", "bci_target")
STEERING_BATCH_KEYS = (COMMAND, *TRAINING_KEYS)
# The config fields of pi0.5's sampler interface a base policy offers (``noise_time`` optional: 1.0, pi0.5's)
SAMPLER_CONFIG_FIELDS = ("num_inference_steps", "chunk_size", "max_action_dim", "n_action_steps")


def _obs_rows(obs: dict) -> int | None:
    """The batch size of an observation: its first tensor's leading dim, or None."""
    for value in obs.values():
        if torch.is_tensor(value) and value.dim() > 0:
            return int(value.shape[0])
    return None


class SteeringPolicy(PreTrainedPolicy):
    """Shared-autonomy steering as a lerobot policy: a frozen base policy's chunks steered toward the operator's
    command by the mode's core (``FlowControlSteering`` or ``FlowReversalSteering``; mode ``"none"``: the base's own
    chunk) through the MARS adapter (``AdapterModules``, the policy's only weights); saving, loading and the Hub
    through ``PreTrainedPolicy``.

    The base policy (``base_policy``, or loaded from ``config.base_policy_path`` on first use, onto
    ``config.device``) is not part of this policy's weights, so ``SteeringPolicy.to`` does not move it: a passed-in
    base is the caller's (often shared with other code) to place, a loaded one follows ``config.device``. Every base is
    driven one way, through pi0.5's sampler interface (the contract): a ``config`` with ``num_inference_steps``,
    ``chunk_size``, ``max_action_dim``, ``n_action_steps`` and ``noise_time`` (1.0, pi0.5's, when absent),
    ``predict_action_chunk(batch, **hooks)`` returning the normalized chunk, ``reset()``, ``velocity_field(batch)``
    (``forward``'s), and a ``postprocessor`` whose unnormalizer step carries the action statistics (FRS's reference
    and ``forward``'s chunk read them). ``PI05Policy`` and ``CursorFlowPolicy`` offer it (``CursorFlowPolicy.wrapping``
    puts a bare ``FlowPolicy`` behind it, ``make_cursor_flow_pre_post_processors`` makes its postprocessor).

    A batch is the base policy's (the cursor's dict, plain or under ``observation.`` as the pre-processor leaves it;
    pi0.5's features), ``observation.scene_tokens`` the task module's scene, and ``command``, the operator's command
    per row ((B, n), or (n,) for a single unbatched trial) or per chunk step ((B, H, n)) in the operator's units
    (deflections in [-1, 1]). The base gets the batch untouched less this policy's own keys (``observation``), the
    adapter its own view of it (``_adapter_view``, where the scene tokens become the task module's scene). Chunks
    come out normalized, as the base's: its postprocessor unnormalizes them.

    ``forward`` takes the example trainer's batches (the pre-processor carries their keys through untouched, but
    does not build them): ``batch["kind"]`` ``"task"`` is the task module's flow matching on ``chunk`` (in action
    units, normalized by the postprocessor's statistics, which it needs), ``"user"`` the
    user module's field regression from ``latent`` to ``bci_target`` (the BCI's chunk, normalized: a regression
    target, not an operator command) through the task module if any; ``noise``, ``time`` and ``weight`` as the trainer
    draws them (drawn here when absent)."""

    config_class = SteeringConfig
    name = "steering"

    def __init__(self, config: SteeringConfig, base_policy=None, postprocessor=None, **kwargs: Any):
        super().__init__(config)
        user, task = self._module("user"), self._module("task")
        self.adapter = AdapterModules(user=user, task=task, matrix=config.fixed_matrix, n_dims=config.n_dims)
        object.__setattr__(
            self, "_base_policy", base_policy
        )  # not a submodule: its weights are not this policy's
        self._postprocessor = postprocessor
        self._base_preprocessor = None  # the base's own pre-processor, when it is loaded from its folder
        self._core: FlowControlSteering | FlowReversalSteering | None = None
        self._action_queue: deque = deque()

    @staticmethod
    def resolve_module(
        config: SteeringConfig, kind: str, build: bool = True
    ) -> tuple[nn.Module | None, dict | None]:
        """The ``kind`` module of ``config`` and its constructor config: built from the config recorded on ``config``
        (``user_module`` / ``task_module``; its weights come with the policy's), else read from its checkpoint
        (``adapter_user`` / ``adapter_task``; one ``torch.load``), else (None, None). With ``build`` False only the
        constructor config is resolved (the module is not built, so no generator is drawn from): the processor
        factory's form. Raises when the checkpoint holds the other kind of module."""
        arch, path = getattr(config, f"{kind}_module"), getattr(config, f"adapter_{kind}")
        if arch is not None:
            return (AdapterModules.KINDS[kind](**arch) if build else None), arch
        if path is None:
            return None, None
        if build:
            module, found, _ = AdapterModules.load_module(path)
            arch = module.config
        else:
            module, ckpt = None, AdapterModules.read_checkpoint(path, mmap=True)  # the config alone
            found, arch = ckpt["kind"], ckpt["config"]
        if found != kind:
            raise ValueError(f"{path} holds a {found} module, not a {kind} module")
        return module, arch

    def _module(self, kind: str) -> nn.Module | None:
        """The ``kind`` module (``resolve_module``), its config recorded when it was read from its checkpoint. A task
        module's scene settings fill the config's unset ones (``SteeringConfig.scene_settings``, which raises on a
        contradiction)."""
        recorded = getattr(self.config, f"{kind}_module") is not None
        module, arch = self.resolve_module(self.config, kind)
        if module is None:
            return None
        if not recorded:
            setattr(self.config, f"{kind}_module", dict(arch))
        if kind == "task":
            for key, value in self.config.scene_settings(module.config).items():
                setattr(self.config, key, value)
        return module

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, base_policy=None, postprocessor=None, **kwargs):
        """``PreTrainedPolicy.from_pretrained``, the base policy (and its postprocessor) passed in or loaded from the
        config's reference on first use. The adapter is served frozen, as ``AdapterModules.load_module`` serves a
        module checkpoint, because that is what reproduces the checkpoint's chunks bit for bit: in the task module's
        attention, ``torch.matmul``'s fold heuristic reads ``requires_grad`` on the transposed batch-first input and
        picks another evaluation order when the weights require grad, even under ``torch.no_grad`` (a last-bit
        difference, batch-size dependent: about 3e-7 at B = 2, bit-equal at other sizes). A loaded steering folder
        therefore has nothing trainable, a departure from the sibling policies that is harmless today (``forward``
        needs the example trainer's ``kind`` batches); ``policy.adapter.requires_grad_(True)`` to train it further."""
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        policy.adapter.requires_grad_(False)
        policy.set_base_policy(base_policy, postprocessor)
        return policy

    def set_base_policy(self, base_policy, postprocessor=None) -> None:
        """Steer ``base_policy`` (None: the config's reference, loaded on first use) from now on."""
        object.__setattr__(self, "_base_policy", base_policy)
        self._postprocessor = postprocessor
        self._core = None

    @property
    def base_policy(self):
        """The policy whose chunks are steered. Loaded from ``config.base_policy_path`` (a local folder or a Hub
        repo id) on first use, with its pre/post processors when the folder has them (``make_pre_post_processors``
        with the folder as ``pretrained_path``): a pi0.5-style base needs its postprocessor's action statistics."""
        if self._base_policy is None:
            path = self.config.base_policy_path
            if path is None:
                raise ValueError(
                    "SteeringPolicy needs a base policy: pass one, or set config.base_policy_path"
                )
            from huggingface_hub.errors import HFValidationError
            from huggingface_hub.utils import validate_repo_id

            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            if not Path(path).is_dir():
                try:
                    validate_repo_id(path)
                except HFValidationError:
                    raise FileNotFoundError(
                        f"base_policy_path {path!r} is neither a local folder nor a Hub repo id"
                    ) from None
            policy = get_policy_class(self.config.base_policy_type).from_pretrained(path)
            object.__setattr__(self, "_base_policy", policy.to(self.config.device).eval())
            pipelines = [
                Path(path) / f"{name}.json"
                for name in (POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME)
            ]
            if self._postprocessor is None and all(p.is_file() for p in pipelines):
                self._base_preprocessor, self._postprocessor = make_pre_post_processors(
                    policy.config, pretrained_path=str(path)
                )
        return self._base_policy

    @property
    def base_preprocessor(self):
        """The base policy's own pre-processor (a pi0.5 base's normalizer and tokenizer), loaded with it from
        ``config.base_policy_path``, or None. A pi0.5-style batch goes through it before this policy."""
        _ = self.base_policy
        return self._base_preprocessor

    def _sampler(self):
        """The base policy, checked to offer the sampler contract's ``predict_action_chunk``, ``reset`` and config
        fields; raises ``TypeError`` naming what it lacks."""
        base = self.base_policy
        config = getattr(base, "config", None)
        missing = [
            name for name in ("predict_action_chunk", "reset") if not callable(getattr(base, name, None))
        ]
        missing += [f"config.{name}" for name in SAMPLER_CONFIG_FIELDS if not hasattr(config, name)]
        if missing:
            raise TypeError(
                f"the base policy ({type(base).__name__}) does not offer pi0.5's sampler interface: no "
                f"{', '.join(missing)} (a FlowPolicy: CursorFlowPolicy.wrapping(model))"
            )
        return base

    def _action_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The base policy's action statistics (mean, std; ``action_mean_std``) from its postprocessor, or None."""
        _ = self.base_policy  # a base loaded from its folder brings its postprocessor
        return None if self._postprocessor is None else _ChunkSteering.action_mean_std(self._postprocessor)

    def _no_stats(self, what: str) -> ValueError:
        """The error for ``what`` needing the base policy's action statistics when there are none."""
        return ValueError(
            f"{what} from the base policy's postprocessor's action statistics, and there is none: none was passed "
            f"with the base, and its folder (config.base_policy_path = {self.config.base_policy_path!r}) has no "
            f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json / {POLICY_POSTPROCESSOR_DEFAULT_NAME}.json; pass it with "
            "set_base_policy(policy, postprocessor=...) (a cursor base's: "
            "make_cursor_flow_pre_post_processors(policy.config)[1])"
        )

    @property
    def core(self) -> "FlowControlSteering | FlowReversalSteering":
        """The mode's steering core over the base policy, built on first use. FRS's reference is the core's own, from
        the postprocessor's action statistics: ``Flow.uniform_reference`` in the 7-D operator space (pi0.5's), the
        command held at ``max_speed * dt`` per control step (``reference_step``) in an action space without its
        gripper dim (the cursor's), the rule that space has (``FlowReversalSteering.reference_chunk``)."""
        cfg = self.config
        if cfg.mode == "none":
            raise ValueError("mode 'none' has no steering core")
        if self._core is None:
            base = self._sampler()
            dims = None if cfg.operator_dims is None else tuple(cfg.operator_dims)
            if cfg.mode == "flow_control":
                if dims is not None and len(dims) != 1:
                    raise ValueError(
                        f"flow control writes one group of action dims, got operator_dims {dims}"
                    )
                written = slice(None) if dims is None else OPERATOR_DIM_GROUPS[dims[0]]
                self._core = FlowControlSteering(
                    base,
                    None,
                    self._postprocessor,
                    tau=cfg.tau,
                    lam=cfg.lam,
                    ic_gain=cfg.ic_gain,
                    deadband=cfg.deadband,
                    dims=written,
                )
            else:
                stats = self._action_stats()
                if stats is None:
                    raise self._no_stats("frs builds its reference")
                step = None if FlowReversalSteering.has_gripper_dim(stats[0]) else cfg.max_speed * cfg.dt
                self._core = FlowReversalSteering(
                    base,
                    None,
                    self._postprocessor,
                    n_reversal_steps=cfg.n_reversal_steps,
                    operator_dims=dims,
                    pinned_steps=cfg.pinned_steps,
                    deadband=cfg.deadband,
                    reference_step=step,
                )
        return self._core

    def observation(self, batch: dict) -> dict:
        """The base policy's batch: ``batch`` untouched but for the steering policy's own keys
        (``STEERING_BATCH_KEYS``), the cursor's dict under its prefix or not and pi0.5's features alike."""
        return {k: v for k, v in batch.items() if k not in STEERING_BATCH_KEYS}

    def _adapter_view(self, obs: dict) -> dict:
        """The adapter's view of the base's batch ``obs`` (``observation``'s), the task module's input contract (the
        cursor's dict: it reads ``pos``, ``obstacles``, ``obstacle_mask``, ``scene`` and ``scene_keep``): every key's
        ``observation.`` prefix stripped, the scene step's inputs (``layouts``, ``scene_images``) dropped, its output
        (``observation.scene_tokens``) put at ``scene[config.scene_source]``, and a single unbatched trial given a
        batch of one (``batched_trial``). The adapter's field binds to it, and a command's rows are counted on it.
        Raises ``ValueError`` when there are scene tokens and the config has no scene source."""
        view = {k.removeprefix(OBS_PREFIX): v for k, v in obs.items()}
        for key in ("layouts", "scene_images"):  # the scene step's inputs, not the task module's
            view.pop(key, None)
        tokens = view.pop("scene_tokens", None)
        if tokens is not None:
            if self.config.scene_source is None:
                raise ValueError("observation.scene_tokens needs a scene_source (a task module's)")
            view["scene"] = {**(view.get("scene") or {}), self.config.scene_source: tokens}
        return batched_trial(view)

    def _field(self, view: dict) -> AdaptedField | None:
        """The adapter's field bound to the adapter view ``view`` (None when the adapter is off). In mode ``"frs"`` a
        fixed matrix (the operator's units) is normalized by the base's postprocessor's action std over its dims
        (``normalized_matrix``; the cursor's uniform std leaves it as it is), and ``ValueError`` is raised without a
        postprocessor. The other modes never run the reversal, the only pass F acts in, so they take F as held and
        need no statistics."""
        if self.adapter.matrix is None or self.config.mode != "frs":
            return self.adapter.field(view)
        stats = self._action_stats()
        if stats is None:
            raise self._no_stats("a fixed adapter matrix is normalized")
        return self.adapter.field(view, std=stats[1][: self.adapter.matrix.shape[-1]])

    def _velocity(self, obs: dict):
        """The base policy's velocity field for its batch ``obs`` (the contract's ``velocity_field``)."""
        velocity_field = getattr(self.base_policy, "velocity_field", None)
        if velocity_field is None:
            raise TypeError(
                f"the base policy ({type(self.base_policy).__name__}) does not offer the sampler contract's "
                "velocity_field(batch) (pi0.5's sampler interface; a FlowPolicy: CursorFlowPolicy.wrapping(model))"
            )
        return velocity_field(obs)

    def _normalized_chunk(self, chunk: Tensor) -> Tensor:
        """A chunk in the base policy's action units, normalized by its postprocessor's statistics, ``(chunk - mean) /
        std`` over the chunk's n dims (the statistics' first n: a pi0.5 base's translation, the cursor's two). Raises
        ``ValueError`` without a postprocessor."""
        stats = self._action_stats()
        if stats is None:
            raise self._no_stats("a task batch's chunk (in action units) is normalized")
        n = chunk.shape[-1]
        mean, std = (torch.as_tensor(a[:n], dtype=chunk.dtype, device=chunk.device) for a in stats)
        return (chunk - mean) / std

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.adapter.parameters()

    def reset(self) -> None:
        """Clears the action queue and the core's queues (and the base policy's)."""
        self._action_queue.clear()
        if self._core is not None:
            self._core.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """The adapter's training loss for ``batch["kind"]`` (``"task"`` or ``"user"``) and its parts, on the
        example trainer's batches (see the class docstring; the pre-processor carries their keys through)."""
        obs, kind = self.observation(batch), batch.get("kind")
        view = self._adapter_view(obs)
        velocity = self._velocity(obs)
        if kind == "task":
            if self.adapter.task is None:
                raise ValueError("a task batch needs a task module")
            chunk = self._normalized_chunk(batch["chunk"])
            z = batch.get("noise")
            z = torch.randn_like(chunk) if z is None else z
            t = batch.get("time")
            t = torch.rand(len(chunk), device=chunk.device) if t is None else t
            task = self.adapter.task.bind(view)
            loss, parts = LearnedTaskModule.field_loss(
                task, velocity, chunk, z, t, self.config.lambda_id, self.config.n_dims
            )
            return loss, {k: v.item() for k, v in parts.items()}
        if kind == "user":
            if self.adapter.user is None:
                raise ValueError("a user batch needs a user module")
            latent = batch["latent"]
            t = batch.get("time")
            t = torch.rand(len(latent), device=latent.device) if t is None else t
            task = None if self.adapter.task is None else self.adapter.task.bind(view)
            loss = LearnedUserModule.field_loss(
                self.adapter.user, velocity, latent, batch["bci_target"], t, task, batch.get("weight")
            )
            return loss, {"field": loss.item()}
        raise ValueError(f"batch['kind'] must be 'task' or 'user', got {kind!r}")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """The steered chunk (B, H, n) for ``batch["command"]`` (None: the base's own chunk; rows inside the
        deadband: the policy's own chunk, from the same noise; in mode ``"none"`` ignored: the base's own chunk), the
        adapter bound to its view of the observation; normalized, as the base's (its postprocessor unnormalizes it).
        ``noise`` (pi0.5's keyword: the flow's starting noise, (B, H, n)) is the base's own chunk's (mode ``"none"``,
        or no command); a steered chunk draws its own, so a command with ``noise`` raises ``ValueError``."""
        obs = self.observation(batch)
        view = self._adapter_view(obs)
        command = self._command(batch.get(COMMAND), view)  # checked in every mode, mode "none" too
        field = self._field(view)
        noise = kwargs.get("noise")
        if self.config.mode == "none" or command is None:  # the base's own chunk, through the task module
            hooks = {"velocity_hook": FlowControlSteering.velocity_hook(field), "noise": noise}
            return self._sampler().predict_action_chunk(
                obs, **{k: v for k, v in hooks.items() if v is not None}
            )
        if noise is not None:
            raise ValueError(
                "noise starts the base policy's own chunk (mode 'none', or no command); a steered chunk draws its own"
            )
        # the steered chunk alone: the policy's own plan is not drawn when every row steers
        return self.core.steered_chunk(obs, command, field=field, paired=False)[0]

    @staticmethod
    def _command(command, view: dict) -> Tensor | None:
        """The operator's command with its leading batch dim: a 1-D command (a single unbatched trial's, as
        ``AddBatchDimensionProcessorStep`` gives lerobot's state one, which knows no command) becomes (1, n); a
        non-tensor one a float32 tensor. Raises ``ValueError`` naming the shapes when the command is not (B, n) or
        (B, H, n) for the adapter view's batch B."""
        if command is None:
            return None
        if not torch.is_tensor(command):
            command = torch.as_tensor(np.asarray(command), dtype=torch.float32)
        if command.dim() == 1:
            command = command[None]
        rows = _obs_rows(view)
        if command.dim() not in (2, 3) or (rows is not None and command.shape[0] != rows):
            raise ValueError(
                f"command of shape {tuple(command.shape)} for an observation batch of {rows}: the command must be "
                "(n,) for one unbatched trial, (B, n) per row or (B, H, n) per chunk step"
            )
        return command

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """One action (B, n) per call: a steered chunk for this batch's command when the queue is empty, its first
        ``n_action_steps`` steps (the base policy's, the contract's field) then served one per call, as the cores' own
        ``select_action`` and lerobot's chunking policies serve theirs. ``reset`` empties the queue."""
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            n = self.base_policy.config.n_action_steps
            self._action_queue.extend(chunk[:, :n].transpose(0, 1))
        return self._action_queue.popleft()
