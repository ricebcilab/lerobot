"""Transformer flow-matching policy that outputs cursor action chunks.

The policy itself is lerobot's `cursor_flow` policy
(`lerobot.policies.cursor_flow.modeling_cursor_flow`, where its description lives):
`FlowPolicy`, `Block`, `time_embedding` and `mlp` are re-exported from there. A
trained policy is saved as a lerobot policy directory (config.json +
model.safetensors, `CursorFlowPolicy`); the trainer writes the cursor format
(policy.pt), which `convert_checkpoints.py` converts. `FlowPolicy.load` reads both.
"""

import logging
from pathlib import Path

import torch

from lerobot.policies.cursor_flow.modeling_cursor_flow import Block, CursorFlowPolicy, mlp, time_embedding  # noqa: F401
from lerobot.policies.cursor_flow.modeling_cursor_flow import FlowPolicy as _FlowPolicy

log = logging.getLogger(__name__)


class FlowPolicy(_FlowPolicy):
    """lerobot's `FlowPolicy`, whose `load` also reads a converted policy directory."""

    @classmethod
    def load(cls, path, device="cpu"):
        """A policy directory (config.json + model.safetensors; a policy.pt in it, a trainer's output not yet
        converted, is read first) or a policy.pt file, in eval mode on `device`."""
        path = Path(path)
        if path.is_dir() and (path / "policy.pt").is_file():
            if (path / "model.safetensors").is_file():
                log.warning(f"{path} holds both policy.pt and model.safetensors; loading policy.pt (a trainer's "
                            f"output not yet converted; run convert_checkpoints.py)")
            path = path / "policy.pt"
        if path.is_dir():
            return CursorFlowPolicy.from_pretrained(path, strict=True).model.to(device).eval()
        return super().load(path, device)


def load_checkpoint(path):
    """The cursor format's {config, state_dict} of a policy, from either format (see `FlowPolicy.load`)."""
    path = Path(path)
    if path.is_file():
        return torch.load(path, map_location="cpu")
    model = FlowPolicy.load(path)
    return dict(config=model.config, state_dict=model.state_dict())
