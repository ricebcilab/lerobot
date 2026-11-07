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
"""The steering policy's processor: each step on a hand-built transition, its features, and the factory's
pipelines (modelled on tests/processor/test_act_processor.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.envs.cursor import TaskConfig, obstacle_spec, render_layouts, sample_layouts
from lerobot.policies.steering import (
    DEADBAND,
    LearnedTaskModule,
    SceneEncoder,
    SteeringCommandProcessorStep,
    SteeringSceneTokensProcessorStep,
    make_steering_pre_post_processors,
)
from lerobot.policies.steering.processor_steering import (
    COMMAND,
    OBS_LAYOUTS,
    OBS_SCENE_IMAGES,
    OBS_SCENE_TOKENS,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

SIZE = 32


class TokenStub:
    """Stands in for SceneEncoder's image_tokens / prefix_tokens: float16 [B, 256, 2048] from each picture's
    16 x 16 cells (a local generator's weights)."""

    def __init__(self):
        self.images = 0
        self.weight = torch.randn(3, 2048, generator=torch.Generator().manual_seed(0))

    def image_tokens(self, images):
        self.images += len(images)
        x = torch.as_tensor(images).float() / 255
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return torch.tanh(3 * pooled @ self.weight).to(torch.float16).numpy()

    def prefix_tokens(self, images):
        return -self.image_tokens(images)


def _layouts(n=3, seed=0):
    task = TaskConfig(
        min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    return sample_layouts(np.random.default_rng(seed), n, 4, task)


def _transition(observation=None, complementary=None):
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: complementary,
    }


def _config(task_module=None, scene_source="discs"):
    return SimpleNamespace(
        input_features={},
        output_features={},
        normalization_mapping={},
        device="cpu",
        task_module=task_module,
        scene_source=scene_source,
        render_size=SIZE,
        vlm_path="lerobot/pi05_base",
        deadband=DEADBAND,
        n_dims=2,
    )


# ---------------------------------------------------------------- the scene tokens step


@pytest.mark.parametrize("source", ["vlm_image", "vlm_prefix"])
def test_scene_tokens_from_layouts_and_from_pictures_are_the_encoders(source):
    lay = _layouts()
    step = SteeringSceneTokensProcessorStep(source=source, render_size=SIZE, encoder=TokenStub())
    out = step(_transition({OBS_LAYOUTS: lay, "observation.pos": torch.zeros(3, 2)}))
    tokens = out[TransitionKey.OBSERVATION][OBS_SCENE_TOKENS]
    want = torch.as_tensor(SceneEncoder.scene_array(lay, source, SIZE, TokenStub()))
    assert tokens.dtype == torch.float16 and tokens.shape == (3, 256, 2048)
    assert torch.equal(tokens, want)
    assert torch.equal(
        out[TransitionKey.OBSERVATION]["observation.pos"], torch.zeros(3, 2)
    )  # the rest passes through
    pictures = torch.as_tensor(render_layouts(lay, SIZE))
    from_pictures = step(_transition({OBS_SCENE_IMAGES: pictures}))[TransitionKey.OBSERVATION][
        OBS_SCENE_TOKENS
    ]
    assert torch.equal(from_pictures, want)


def test_scene_tokens_carry_a_task_modules_reduction():
    g = torch.Generator().manual_seed(1)
    mean, basis = torch.randn(2048, generator=g), torch.linalg.qr(torch.randn(2048, 8, generator=g))[0]
    lay = _layouts()
    step = SteeringSceneTokensProcessorStep(render_size=SIZE, encoder=TokenStub(), reduction=(mean, basis))
    tokens = step(_transition({OBS_LAYOUTS: lay}))[TransitionKey.OBSERVATION][OBS_SCENE_TOKENS]
    full = torch.as_tensor(SceneEncoder.scene_array(lay, "vlm_image", SIZE, TokenStub()))
    assert tokens.shape == (3, 256, 8) and torch.equal(tokens, LearnedTaskModule.reduce(full, mean, basis))
    state = step.state_dict()
    again = SteeringSceneTokensProcessorStep(**step.get_config(), encoder=TokenStub())
    again.load_state_dict(state)
    assert torch.equal(
        again(_transition({OBS_LAYOUTS: lay}))[TransitionKey.OBSERVATION][OBS_SCENE_TOKENS], tokens
    )


def test_scene_tokens_step_refuses_what_it_cannot_encode():
    step = SteeringSceneTokensProcessorStep(render_size=SIZE, encoder=TokenStub())
    with pytest.raises(ValueError, match="observation.layouts"):
        step(_transition({"observation.pos": torch.zeros(1, 2)}))
    with pytest.raises(ValueError, match="needs an observation"):
        step(_transition(None))
    for source in ("discs", "cnn", "vlm_views", "nope"):
        with pytest.raises(ValueError, match="source must be one of"):
            SteeringSceneTokensProcessorStep(source=source)


def test_scene_tokens_step_builds_its_encoder_on_first_use_only():
    step = SteeringSceneTokensProcessorStep(vlm_path="some/repo", device="cpu")
    assert step._encoder is None
    encoder = step.scene_encoder()
    assert isinstance(encoder, SceneEncoder) and encoder._policy is None  # constructing it loads nothing
    assert (encoder.path, encoder.device) == ("some/repo", "cpu") and step.scene_encoder() is encoder


def test_scene_tokens_step_declares_its_feature():
    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.pos": PolicyFeature(type=FeatureType.STATE, shape=(2,))
        }
    }
    out = SteeringSceneTokensProcessorStep(encoder=TokenStub()).transform_features(features)
    assert out[PipelineFeatureType.OBSERVATION][OBS_SCENE_TOKENS].shape == (256, 2048)
    assert out[PipelineFeatureType.OBSERVATION]["observation.pos"].shape == (2,)
    assert OBS_SCENE_TOKENS not in features[PipelineFeatureType.OBSERVATION]  # the input is not modified
    reduced = SteeringSceneTokensProcessorStep(
        encoder=TokenStub(), reduction=(torch.zeros(2048), torch.zeros(2048, 5))
    )
    assert reduced.transform_features({})[PipelineFeatureType.OBSERVATION][OBS_SCENE_TOKENS].shape == (256, 5)


# ---------------------------------------------------------------- the command step


def test_command_step_keeps_the_commanded_dims_and_zeroes_the_idle_rows():
    raw = np.array([[0.5, -0.2, 9.0], [DEADBAND / 2, 0.0, 9.0], [0.0, -DEADBAND, 9.0]])
    out = SteeringCommandProcessorStep()(_transition(complementary={COMMAND: raw, "task": "x"}))
    data = out[TransitionKey.COMPLEMENTARY_DATA]
    assert data["task"] == "x" and data[COMMAND].dtype == torch.float32
    torch.testing.assert_close(data[COMMAND], torch.tensor([[0.5, -0.2], [0.0, 0.0], [0.0, -DEADBAND]]))
    per_step = torch.zeros(2, 4, 2)
    per_step[0, 3] = torch.tensor([0.0, 0.3])  # one active step makes the whole row active
    per_step[1] = DEADBAND / 4
    chunk = SteeringCommandProcessorStep().command(per_step)
    assert torch.equal(chunk[0], per_step[0]) and not chunk[1].any()


def test_command_step_needs_a_command_and_declares_no_feature():
    with pytest.raises(ValueError, match="command"):
        SteeringCommandProcessorStep()(_transition(complementary={"task": "x"}))
    with pytest.raises(ValueError, match="command"):
        SteeringCommandProcessorStep()(_transition())
    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.pos": PolicyFeature(type=FeatureType.STATE, shape=(2,))
        }
    }
    assert SteeringCommandProcessorStep().transform_features(features) == features
    assert SteeringCommandProcessorStep(deadband=0.1, n_dims=3).get_config() == {"deadband": 0.1, "n_dims": 3}


# ---------------------------------------------------------------- the factory


def test_factory_assembles_the_steps_in_order():
    pre, post = make_steering_pre_post_processors(_config())
    assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME and post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
    assert [type(s) for s in pre.steps] == [
        RenameObservationsProcessorStep,
        AddBatchDimensionProcessorStep,
        NormalizerProcessorStep,
        SteeringCommandProcessorStep,
        DeviceProcessorStep,
    ]
    assert [type(s) for s in post.steps] == [UnnormalizerProcessorStep, DeviceProcessorStep]
    assert post.steps[1].device == "cpu"
    pre, _ = make_steering_pre_post_processors(_config(task_module="t.pt", scene_source="vlm_image"))
    assert [type(s) for s in pre.steps][3:5] == [
        SteeringSceneTokensProcessorStep,
        SteeringCommandProcessorStep,
    ]
    assert (pre.steps[3].source, pre.steps[3].render_size) == ("vlm_image", SIZE)
    for source in ("discs", "cnn"):  # no scene tokens to encode
        pre, _ = make_steering_pre_post_processors(_config(task_module="t.pt", scene_source=source))
        assert SteeringSceneTokensProcessorStep not in [type(s) for s in pre.steps]


def test_the_pre_pipeline_carries_the_observation_and_the_command():
    pre, post = make_steering_pre_post_processors(_config())
    pos = torch.rand(3, 2)
    batch = pre({"observation.pos": pos, COMMAND: np.array([[0.5, 0.0], [0.0, 0.0], [-0.3, 0.4]])})
    assert torch.equal(batch["observation.pos"], pos)
    torch.testing.assert_close(batch[COMMAND], torch.tensor([[0.5, 0.0], [0.0, 0.0], [-0.3, 0.4]]))
    action = torch.rand(3, 16, 2)
    assert torch.equal(post(action), action)
