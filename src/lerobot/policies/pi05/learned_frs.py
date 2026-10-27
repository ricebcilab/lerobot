"""Compatibility for trained XYZ FRS-RA checkpoints and exact-noise collection."""

import hashlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .steering import FlowReversalSteeringPolicy, ReversalAdapter

PROMPT = "reach, grasp, and bring to mouth"


def sha256(path):
    """Stream a checkpoint or recording digest."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LearnedReversalAdapter(nn.Module):
    """Legacy environment-space XYZ matrices, applied to the trained action prefix."""

    n_dims = 3

    def __init__(
        self, chunk_size=50, hidden_dim=128, time_dim=32, pinned_steps=25, action_std=(1.0, 1.0, 1.0)
    ):
        super().__init__()
        if not 1 <= pinned_steps <= chunk_size or time_dim < 2 or time_dim % 2:
            raise ValueError("Invalid chunk prefix or timestep embedding")
        self.architecture = {
            "chunk_size": chunk_size,
            "hidden_dim": hidden_dim,
            "time_dim": time_dim,
            "pinned_steps": pinned_steps,
            "action_std": list(action_std),
        }
        self.horizon = chunk_size
        self.pinned_steps = pinned_steps
        self.net = nn.Sequential(
            nn.Linear(chunk_size * 3 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_size * 9),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("std", torch.as_tensor(action_std, dtype=torch.float32))
        if self.std.shape != (3,) or not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Expected three positive finite action scales")

    def forward(self, x, t):
        from .modeling_pi05 import create_sinusoidal_pos_embedding

        t = torch.as_tensor(t, device=x.device, dtype=torch.float32).reshape(-1).expand(x.shape[0])
        emb = create_sinusoidal_pos_embedding(t, self.architecture["time_dim"], 0.004, 4.0, x.device)
        features = torch.cat((x[..., :3].float().flatten(1), emb.float()), dim=1)
        delta = self.net(features).reshape(x.shape[0], self.horizon, 3, 3)
        return delta + torch.eye(3, device=x.device)

    def apply_velocity(self, x, t, velocity):
        """Preserve the original S^-1 H S calculation and prefix mask exactly."""
        delta = self(x, t) - torch.eye(3, device=x.device)
        env_v = velocity[..., :3].float() * self.std
        correction = torch.matmul(delta, env_v.unsqueeze(-1)).squeeze(-1) / self.std
        result = velocity.clone()
        n = self.pinned_steps
        result[:, :n, :3] = velocity[:, :n, :3] + correction[:, :n].to(velocity.dtype)
        return result

    def save(self, path, metadata):
        torch.save(
            {
                "format": "learned_frs_v1",
                "architecture": self.architecture,
                "state_dict": self.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path, expected=None, device="cpu"):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle.get("format") != "learned_frs_v1":
            raise ValueError("Unsupported learned adapter checkpoint")
        for key, value in (expected or {}).items():
            if bundle["metadata"].get(key) != value:
                raise ValueError(f"Learned adapter contract mismatch: {key}")
        model = cls(**bundle["architecture"])
        model.load_state_dict(bundle["state_dict"])
        model.legacy_metadata = bundle["metadata"]
        return model.to(device).eval(), bundle["metadata"]


class LearnedFRSPolicy(FlowReversalSteeringPolicy):
    """Legacy entry point backed by the current modular FRS sampler."""

    def __init__(self, *args, learned_adapter, **kwargs):
        if kwargs.get("adapter") is not None:
            raise ValueError("Legacy wrapper expects no separate adapter")
        kwargs["adapter"] = ReversalAdapter(user_module=learned_adapter)
        super().__init__(*args, **kwargs)
        if self.operator_dims != ("translation",) or learned_adapter.pinned_steps != self.pinned_steps:
            raise ValueError("Adapter prefix/dimensions differ from executed prefix")
        np.testing.assert_allclose(learned_adapter.std.cpu(), self._std[:3], rtol=1e-6)
        self.learned_adapter = learned_adapter


class RecordingPolicy:
    """Capture original noise, observations and clean chunks before output gains."""

    def __init__(self, policy, directory, contract):
        self.policy, self.directory, self.contract = policy, Path(directory), contract
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = deque()
        self.index = -1
        self.last_path = None

    def reset(self):
        self.queue.clear()
        self.policy.reset()

    def select_action(self, batch):
        if not self.queue:
            self.index += 1
            captured = {}

            def capture(velocity, noise):
                captured["noise"] = noise.detach().cpu()
                return noise

            clean = self.policy.predict_action_chunk(batch, noise_fn=capture)
            self.last_path = self.directory / f"chunk_{self.index:06d}.pt"
            torch.save(
                {
                    "batch": {k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)},
                    "noise": captured["noise"],
                    "clean": clean.detach().cpu(),
                    "contract": self.contract,
                },
                self.last_path,
            )
            self.queue.extend(clean[:, : self.policy.config.n_action_steps].transpose(0, 1))
        return self.queue.popleft()
