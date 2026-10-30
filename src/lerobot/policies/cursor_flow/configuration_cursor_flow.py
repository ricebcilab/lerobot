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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig


@PreTrainedConfig.register_subclass("cursor_flow")
@dataclass
class CursorFlowConfig(PreTrainedConfig):
    """Configuration of the cursor task's flow-matching policy (`FlowPolicy`, MARS cursor example).

    The fields mirror `FlowPolicy`'s constructor. The policy reads its own dict observation (cursor
    position, displacement history, targets, optional obstacles) rather than dataset features, so it has
    no input/output features, no processor pipeline and no delta indices. It is trained by the example's
    own trainer (``examples/pi05/mars/cursor/train/train_policy.py``), not by ``lerobot-train``.

    Args:
        horizon: Action chunk length, in control steps.
        hist: Number of past cursor displacements in the observation.
        d_model: Transformer width.
        layers: Number of transformer blocks.
        heads: Attention heads per block.
        d_ff: Feed-forward width.
        mode: ``"flow"`` (flow matching) or ``"regress"`` (direct regression baseline).
        action_scale: Screen units per normalized action unit (the RMS demo step). The trained value is
            also held in the weights as a buffer, which is what the model uses.
        obstacle_tokens: Whether the policy reads obstacles as tokens.
    """

    horizon: int = 16
    hist: int = 4
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    d_ff: int = 512
    mode: str = "flow"
    action_scale: float = 0.01
    obstacle_tokens: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in ("flow", "regress"):
            raise ValueError(f"mode must be 'flow' or 'regress', got {self.mode!r}")

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig()

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
