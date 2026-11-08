"""Transformer flow-matching policy that outputs cursor action chunks.

The policy itself is lerobot's `cursor_flow` policy
(`lerobot.policies.cursor_flow.modeling_cursor_flow`, where its description lives):
`FlowPolicy`, `Block`, `time_embedding` and `mlp` are re-exported from there. A
trained policy is a lerobot policy directory (config.json + model.safetensors and the
policy's processor pipelines, `CursorFlowPolicy`), which `load_policy` reads; a cursor
project's policy.pt is `convert_checkpoints.py`'s to convert. `as_policy` puts a bare
`FlowPolicy` behind the sampler contract the steering policy drives.
"""

from dataclasses import replace
from pathlib import Path

from lerobot.policies.cursor_flow.modeling_cursor_flow import (  # noqa: F401
    Block,
    CursorFlowPolicy,
    FlowPolicy,
    mlp,
    time_embedding,
)


def load_policy(path, device="cpu"):
    """The `CursorFlowPolicy` of a lerobot policy directory (config.json + model.safetensors), in eval mode on
    `device`. A folder holding a trainer's policy.pt (the cursor format) is refused: converting it is
    `tools/convert_checkpoints.py`'s business."""
    path = Path(path)
    if (path / "policy.pt").is_file():
        raise ValueError(f"{path} holds a policy.pt, the cursor format, not yet converted: run "
                         f"tools/convert_checkpoints.py")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} is not a lerobot policy directory (no config.json)")
    return CursorFlowPolicy.from_pretrained(path, strict=True).to(device).eval()


def as_policy(model_or_policy, *, num_inference_steps=None, n_action_steps=None):
    """A `CursorFlowPolicy` serving `model_or_policy`'s module (shared, not copied): a `FlowPolicy` wrapped with a
    config read from its attributes (`CursorFlowPolicy.wrapping`); a `CursorFlowPolicy` as is. `num_inference_steps`
    and `n_action_steps` (None: as they are) set the sampler's Euler steps and executed steps on a config of the
    wrapper's own: a passed policy whose values differ is wrapped anew around the same module, its config copied
    with them, never changed."""
    changes = {k: v for k, v in dict(num_inference_steps=num_inference_steps, n_action_steps=n_action_steps).items()
               if v is not None}
    if isinstance(model_or_policy, CursorFlowPolicy):
        policy = model_or_policy
        if all(getattr(policy.config, k) == v for k, v in changes.items()):
            return policy
        return CursorFlowPolicy(replace(policy.config, **changes), model=policy.model)
    policy = CursorFlowPolicy.wrapping(model_or_policy)
    for key, value in changes.items():  # the fresh wrapper's own config
        setattr(policy.config, key, value)
    return policy
