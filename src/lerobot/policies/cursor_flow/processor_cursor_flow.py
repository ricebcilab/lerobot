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

"""The cursor flow policy's processing: ``make_cursor_flow_pre_post_processors``, its pipelines, and the cursor's
scenes as pictures: ``CursorSceneRenderProcessorStep``, the trials' layouts rendered for a steering policy's
scene-tokens step, and ``render_scene_array``, a layout batch's scene array for a task module's source."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.envs.cursor import SCENE_SIZE, Layouts, render_layouts
from lerobot.policies.steering import LearnedTaskModule, SceneEncoder
from lerobot.policies.steering.processor_steering import OBS_SCENE_DIGEST, OBS_SCENE_IMAGES
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_cursor_flow import CursorFlowConfig

OBS_LAYOUTS = f"{OBS_PREFIX}layouts"  # the render step's input: the trials' layouts (a ``Layouts``)


def render_scene_array(
    layouts: Layouts,
    source: str,
    size: int = SCENE_SIZE,
    encoder: SceneEncoder | None = None,
    batch: int | None = None,
) -> np.ndarray:
    """The per-layout scene array for ``source``, ``[B, ...]``: ``cnn`` the render (uint8 ``[B, 3, size,
    size]``), ``vlm_image`` / ``vlm_prefix`` the encoder's ``image_tokens`` / ``prefix_tokens`` of it
    (``SceneEncoder._encode``). ``encoder`` is anything with those two methods, typically one ``SceneEncoder`` built
    once by the caller; ``batch`` (None: the encoder's default) goes to them. ``discs`` reads the layouts' obstacles
    directly and has no scene array."""
    if source == "discs":  # nosec B105
        raise ValueError("source 'discs' reads the layouts' obstacles; it has no scene array")
    LearnedTaskModule.spec(source, size)  # the source and size checks
    images = render_layouts(layouts, size)
    if source == "cnn":  # nosec B105
        return images
    if encoder is None:
        raise ValueError(f"source {source!r} needs an encoder (e.g. a SceneEncoder)")
    return SceneEncoder._encode(encoder, source, images, batch)


@ProcessorStepRegistry.register(name="cursor_scene_render_processor_step")
@dataclass
class CursorSceneRenderProcessorStep(ProcessorStep):
    """The cursor's scenes as pictures, for a steering policy's scene-tokens step.

    Reads the trials' layouts (``observation.layouts``, a ``Layouts``) from the observation and writes
    ``observation.scene_images``, their render (``render_layouts`` at ``render_size``: uint8 ``[B, 3, S, S]``), and
    ``observation.scene_digest``, a sha256 over what the render is drawn from, the key the tokens step
    (``SteeringSceneTokensProcessorStep``) keeps its tokens under. The layouts stay in the observation. The last
    layouts' render is kept, so a replan on the same trials draws nothing (``reset`` clears it)."""

    render_size: int = SCENE_SIZE

    _last: tuple[str, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )  # (layouts' digest, pictures)

    @staticmethod
    def _digest(layouts: Layouts) -> str:
        """A sha256 over what a render of ``layouts`` is drawn from (start, targets, mask, obstacles, their mask)."""
        h = hashlib.sha256()
        for a in (layouts.start, layouts.targets, layouts.mask, layouts.obstacles, layouts.obstacle_mask):
            a = np.ascontiguousarray(a)
            h.update(f"{a.dtype.str}{a.shape}".encode() + a.tobytes())
        return h.hexdigest()

    def reset(self) -> None:
        self._last = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("CursorSceneRenderProcessorStep needs an observation")
        layouts = observation.get(OBS_LAYOUTS)
        if layouts is None:
            raise ValueError(f"CursorSceneRenderProcessorStep needs {OBS_LAYOUTS!r} in the observation")
        digest = self._digest(layouts)
        if self._last is None or self._last[0] != digest:
            self._last = (digest, render_layouts(layouts, self.render_size))
        transition[TransitionKey.OBSERVATION] = {
            **observation,
            OBS_SCENE_IMAGES: self._last[1],
            OBS_SCENE_DIGEST: digest,
        }
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"render_size": self.render_size}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Adds ``observation.scene_images`` (3 x ``render_size`` x ``render_size`` pictures)."""
        observation = dict(features.get(PipelineFeatureType.OBSERVATION, {}))
        observation[OBS_SCENE_IMAGES] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, self.render_size, self.render_size)
        )
        return {**features, PipelineFeatureType.OBSERVATION: observation}


def make_cursor_flow_pre_post_processors(
    config: CursorFlowConfig,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the cursor flow policy.

    The policy reads its own dict observation (``CursorFlowPolicy.observation``), so the pre-processing pipeline
    only renames (nothing, by default). The post-processing pipeline unnormalizes the policy's normalized chunks to
    screen units per step, ``chunk * std + mean`` with mean 0 and std ``config.action_scale`` on both dims (the
    model's ``chunk * action_scale``, bit for bit), and moves them to the CPU. Its unnormalizer step carries the
    action statistics the steering cores read.

    Args:
        config: The cursor flow policy configuration.

    Returns:
        A tuple containing the pre-processor pipeline and the post-processor pipeline.
    """
    input_steps = [RenameObservationsProcessorStep(rename_map={})]
    scale = config.action_scale
    output_steps = [
        UnnormalizerProcessorStep(
            features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={ACTION: {"mean": [0.0, 0.0], "std": [scale, scale]}},
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
