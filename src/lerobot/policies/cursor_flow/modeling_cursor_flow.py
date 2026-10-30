#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The MARS cursor task's flow-matching policy.

`FlowPolicy` is the cursor example's transformer flow-matching policy, moved here verbatim
(``examples/pi05/mars/cursor/cursor_sa/policy.py`` re-exports it). `CursorFlowPolicy` is a thin
`PreTrainedPolicy` around it, so a trained cursor policy saves as ``config.json`` +
``model.safetensors`` and loads with ``from_pretrained``; the cursor code uses its ``.model``.

FlowPolicy's own description:

Tokens: one for the cursor (position and recent velocity), one per target
(position, offset from the cursor, goal flag), optionally one per obstacle
(center, offset from the cursor, radius; a policy built with obstacle_tokens),
and one per future step of the action chunk. Targets and obstacles carry no
order embedding and padded rows are masked out of attention, so the policy
takes any number of them in any order.
As in pi0, cursor and target tokens do not attend to the action tokens, so the
observation encoding does not depend on the noisy chunk being denoised.

The flow convention is the cursor's (see ``lerobot.policies.pi05.steering``): t=0 is
Gaussian noise, t=1 is the action chunk, and sampling Euler-integrates v_theta from 0
to 1. mode="regress" is the baseline: the same network predicts the chunk
directly with an MSE loss, so it averages over the variety in the demos.
"""

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pi05.steering import integrate
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_cursor_flow import CursorFlowConfig


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = 1000 * t[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], -1)


def mlp(d_in, d):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


class Block(nn.Module):
    """Pre-norm transformer block; `allow` is a boolean [B, 1, L, L] attention mask."""

    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x, allow):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        x = x + self.proj(att.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.norm2(x))


class FlowPolicy(nn.Module):
    """Observations are dicts of tensors:
        pos      [B, 2]     cursor position, screen units in [0, 1]
        hist     [B, h, 2]  last h per-step cursor displacements, oldest first
        targets  [B, N, 2]  target centers
        mask     [B, N]     True for real targets (False rows are padding)
        goal     [B, N]     True for the goal target
        obstacles      [B, K, 3]  disc centers and radii   (read only with obstacle_tokens)
        obstacle_mask  [B, K]     True for real discs
    Action chunks are [B, horizon, 2] per-step displacements in screen units.
    """

    def __init__(
        self,
        horizon=16,
        hist=4,
        d_model=128,
        layers=4,
        heads=4,
        d_ff=512,
        mode="flow",
        action_scale=0.01,
        obstacle_tokens=False,
    ):
        super().__init__()
        assert mode in ("flow", "regress")
        self.config = dict(
            horizon=horizon, hist=hist, d_model=d_model, layers=layers, heads=heads, d_ff=d_ff, mode=mode
        )
        if obstacle_tokens:  # only then, so a policy without them saves the config it always did
            self.config["obstacle_tokens"] = True
        self.horizon, self.hist, self.mode, self.obstacle_tokens = horizon, hist, mode, obstacle_tokens
        self.register_buffer("action_scale", torch.tensor(float(action_scale)))
        d = d_model
        self.cursor_in = mlp(2 + 2 * hist, d)
        self.target_in = mlp(5, d)
        self.type_emb = nn.Parameter(0.02 * torch.randn(3, d))  # cursor, target, action
        self.step_emb = nn.Parameter(0.02 * torch.randn(horizon, d))
        self.action_in = nn.Linear(2, d)
        self.time_in = mlp(d, d)
        self.action_time = mlp(2 * d, d)
        self.blocks = nn.ModuleList(Block(d, heads, d_ff) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        if obstacle_tokens:  # their own input layer and type embedding: type_emb keeps its 3 rows
            self.obstacle_in = mlp(5, d)
            self.obstacle_emb = nn.Parameter(0.02 * torch.randn(d))

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)  # torch>=2.7 default, made explicit
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()

    def forward(self, obs, x=None, t=None):
        """Flow mode: velocity v_theta(x_t, t | obs) for a normalized chunk x at
        flow time t [B]. Regress mode: the normalized chunk (x and t unused)."""
        pos = 2 * obs["pos"] - 1
        cursor = self.cursor_in(torch.cat([pos, (obs["hist"] / self.action_scale).flatten(1)], -1))
        tgt = 2 * obs["targets"] - 1
        feats = torch.cat([tgt, tgt - pos[:, None], obs["goal"][..., None].float()], -1)
        targets = self.target_in(feats)

        B, H = len(pos), self.horizon
        ones = torch.ones(B, 1, dtype=torch.bool, device=pos.device)
        context, keys = [cursor[:, None] + self.type_emb[0], targets + self.type_emb[1]], [ones, obs["mask"]]
        obstacles = obs.get("obstacles") if self.obstacle_tokens else None
        if obstacles is not None and obstacles.shape[1]:  # centers and radii in the targets' units
            c = 2 * obstacles[..., :2] - 1
            context.append(
                self.obstacle_in(torch.cat([c, c - pos[:, None], 2 * obstacles[..., 2:]], -1))
                + self.obstacle_emb
            )
            keys.append(obs["obstacle_mask"])
        if self.mode == "flow":
            temb = self.time_in(time_embedding(t, self.step_emb.shape[1]))
            act = self.action_time(torch.cat([self.action_in(x), temb[:, None].expand(B, H, -1)], -1))
        else:
            act = torch.zeros(B, H, self.step_emb.shape[1], device=pos.device)
        tokens = torch.cat(context + [act + self.step_emb + self.type_emb[2]], 1)

        L = tokens.shape[1]
        n_ctx = L - H
        keys = torch.cat(keys + [ones.expand(B, H)], 1)
        allow = keys[:, None, None, :].expand(B, 1, L, L).clone()
        allow[:, :, :n_ctx, n_ctx:] = False
        for block in self.blocks:
            tokens = block(tokens, allow)
        return self.out(tokens[:, n_ctx:])

    def loss(self, obs, chunk):
        a = chunk / self.action_scale
        if self.mode == "regress":
            return F.mse_loss(self(obs), a)
        z = torch.randn_like(a)
        t = torch.rand(len(a), device=a.device)
        x = (1 - t[:, None, None]) * z + t[:, None, None] * a
        return F.mse_loss(self(obs, x, t), a - z)

    @torch.no_grad()
    def integrate(self, obs, x, t0=0.0, t1=1.0, steps=10):
        """Euler-integrate the flow on normalized chunks x from time t0 to t1.
        0 -> 1 turns noise into a chunk (mu_theta in the steering README);
        1 -> 0 runs it backward, mapping a chunk to its latent noise."""
        dt = (t1 - t0) / steps
        for i in range(steps):
            t = torch.full((len(x),), t0 + i * dt, device=x.device)
            x = x + dt * self(obs, x, t)
        return x

    @torch.no_grad()
    def sample(self, obs, steps=10, z=None):
        """Action chunk [B, horizon, 2] in screen units per step."""
        if self.mode == "regress":
            return self(obs) * self.action_scale
        if z is None:
            z = torch.randn(len(obs["pos"]), self.horizon, 2, device=obs["pos"].device)
        return self.integrate(obs, z, steps=steps) * self.action_scale


# ---------------------------------------------------------------- the lerobot policy wrapper

FLOW_STEPS = 10  # Euler steps of a chunk, as the cursor example samples


class CursorFlowPolicy(PreTrainedPolicy):
    """A `FlowPolicy` as a lerobot policy: saving, loading and the Hub through `PreTrainedPolicy`.

    The batch is `FlowPolicy`'s dict observation (``pos``, ``hist``, ``targets``, ``mask``, ``goal`` and
    optionally ``obstacles`` / ``obstacle_mask``), not dataset features; there is no processor pipeline.
    Actions are per-step cursor displacements in screen units.
    """

    config_class = CursorFlowConfig
    name = "cursor_flow"

    def __init__(self, config: CursorFlowConfig, **kwargs: Any):
        super().__init__(config)
        self.model = FlowPolicy(
            horizon=config.horizon,
            hist=config.hist,
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
            d_ff=config.d_ff,
            mode=config.mode,
            action_scale=config.action_scale,
            obstacle_tokens=config.obstacle_tokens,
        )

    def get_optim_params(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def reset(self) -> None:
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("trained by examples/pi05/mars/cursor/train/train_policy.py")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """An action chunk (B, horizon, 2) in screen units per step. Flow mode: noise drawn as the cursor
        example draws it, Euler-integrated from t=0 (noise) to t=1 (action) in ``FLOW_STEPS`` steps
        (``steering.integrate``), the cursor example's plain policy chunk. Regress mode: the predicted chunk."""
        model = self.model
        if model.mode == "regress":
            return model(batch) * model.action_scale
        pos = batch["pos"]
        z = torch.randn(len(pos), model.horizon, 2, device=pos.device)
        chunk = integrate(
            z, lambda x, t: model(batch, x, t), 0.0, 1.0 / FLOW_STEPS, FLOW_STEPS, batch_time=True
        )
        return chunk * model.action_scale

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """The first step (B, 2) of a fresh action chunk."""
        return self.predict_action_chunk(batch)[:, 0]
