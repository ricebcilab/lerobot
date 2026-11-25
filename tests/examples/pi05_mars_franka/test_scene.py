"""The scene a task checkpoint asks for: its read-back, its one-step pipeline and the scene camera's pictures.

The real ``SceneEncoder`` loads ``lerobot/pi05_base`` (about 14 GB), which is not cached in the offline
environment, so the step is exercised with an injected fake encoder (``Scene.step(encoder=...)``: the
step's own encoder is injectable for exactly this). The task checkpoints are written with
``AdapterModules.save_module`` (a built ``LearnedTaskModule`` and no training or weights worth keeping),
and ``from_checkpoint`` reads their recorded constructor config. No camera, no station, no VLM.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from scene import OUT_OF_SCOPE, TASK_SOURCE, Scene, from_checkpoint, scene_images
from station import resize_with_pad

from lerobot.policies.steering import (
    VLM_SCENE_DIM,
    VLM_SCENE_TOKENS,
    AdapterModules,
    LearnedTaskModule,
    SteeringSceneTokensProcessorStep,
)
from lerobot.policies.steering.processor_steering import OBS_SCENE_IMAGES, OBS_SCENE_TOKENS

SIZE = 128
"""The checkpoints' render size: small enough that saving one is instant (the VLM resizes it to 224)."""


def _task_checkpoint(path: Path, source: str = TASK_SOURCE) -> Path:
    """A task module checkpoint of ``source`` built with no training: its recorded config is the scene."""
    spec = LearnedTaskModule.spec(source, SIZE, wrist_span=0.5 if source == "vlm_views" else None)
    module = LearnedTaskModule(**spec)
    return AdapterModules.save_module(module, path, kind="task", meta={"test": "scene"})


def _discs_checkpoint(path: Path) -> Path:
    """A ``discs`` checkpoint: its config omits ``token_source`` (discs is the module's default)."""
    module = LearnedTaskModule(**LearnedTaskModule.spec("discs"))
    return AdapterModules.save_module(module, path, kind="task")


class _FakeEncoder:
    """The scene step's encoder contract without the VLM: float16 ``[B, 256, 2048]`` zeros per batch.

    Counts the pictures it encoded so a test can see the step's digest cache at work.
    """

    def __init__(self) -> None:
        self.pictures = 0

    def image_tokens(self, images, batch=64) -> torch.Tensor:
        self.pictures += len(images)
        return torch.zeros((len(images), VLM_SCENE_TOKENS, VLM_SCENE_DIM), dtype=torch.float16)

    def prefix_tokens(self, images, batch=64) -> torch.Tensor:
        raise AssertionError("a vlm_image step reads image_tokens, not prefix_tokens")


# ---------------------------------------------------------------- reading a checkpoint


def test_from_checkpoint_reads_the_task_modules_source_and_render_size(tmp_path):
    path = _task_checkpoint(tmp_path / "task.pt")
    scene = from_checkpoint(path)
    assert scene == Scene(source="vlm_image", render_size=SIZE)
    assert scene.source == TASK_SOURCE and scene.render_size == SIZE


@pytest.mark.parametrize("source", OUT_OF_SCOPE)
def test_from_checkpoint_refuses_a_source_this_phase_does_not_build(tmp_path, source):
    """DESIGN.md's Phase C scope: only ``vlm_image``. The other sources are named, not silently skipped."""
    path = (
        _discs_checkpoint(tmp_path / "task.pt")
        if source == "discs"
        else _task_checkpoint(tmp_path / "task.pt", source)
    )
    with pytest.raises(ValueError) as raised:
        from_checkpoint(path)
    message = str(raised.value)
    assert "out of scope" in message
    assert f"{source!r}" in message and TASK_SOURCE in message
    assert all(other in message for other in OUT_OF_SCOPE)


def test_a_scene_refuses_a_source_this_phase_does_not_build():
    with pytest.raises(ValueError, match="out of scope"):
        Scene(source="cnn", render_size=SIZE)
    with pytest.raises(ValueError, match="positive"):
        Scene(source=TASK_SOURCE, render_size=0)


# ---------------------------------------------------------------- the step


def test_the_step_is_the_steering_scene_tokens_step_alone():
    pipeline = Scene(TASK_SOURCE, SIZE).step(device="cpu", encoder=_FakeEncoder())
    assert len(pipeline.steps) == 1
    assert isinstance(pipeline.steps[0], SteeringSceneTokensProcessorStep)
    assert pipeline.steps[0].render_size == SIZE and pipeline.steps[0].source == TASK_SOURCE


def test_the_step_turns_scene_images_into_float16_tokens_and_keeps_the_batch():
    """``observation.scene_images`` -> ``observation.scene_tokens``: float16 ``[B, 256, 2048]``, the rest of
    the batch (the state, the prompt and a training batch's keys) untouched."""
    encoder = _FakeEncoder()
    pipeline = Scene(TASK_SOURCE, SIZE).step(device="cpu", encoder=encoder)
    pictures = np.zeros((2, 3, SIZE, SIZE), dtype=np.uint8)
    pictures[1] = 7
    batch = {
        OBS_SCENE_IMAGES: pictures,
        "observation.state": np.zeros((2, 32), dtype=np.float32),
        "task": "pick the cube",
        "kind": "task",
        "chunk": np.zeros((2, 8, 8), dtype=np.float32),
    }

    prepared = pipeline(batch)

    tokens = prepared[OBS_SCENE_TOKENS]
    assert torch.is_tensor(tokens)
    assert tokens.shape == (2, VLM_SCENE_TOKENS, VLM_SCENE_DIM)
    assert tokens.dtype == torch.float16
    np.testing.assert_array_equal(prepared["observation.state"], batch["observation.state"])
    np.testing.assert_array_equal(prepared["chunk"], batch["chunk"])
    assert prepared["task"] == "pick the cube"
    assert prepared["kind"] == "task"  # a training batch's keys go through, not dropped as metadata
    assert encoder.pictures == 2


def test_the_step_encodes_the_same_pictures_once():
    """The step keys its tokens on the pictures' digest: a chunk boundary replans on a still scene without
    re-running the VLM."""
    encoder = _FakeEncoder()
    pipeline = Scene(TASK_SOURCE, SIZE).step(device="cpu", encoder=encoder)
    pictures = np.full((1, 3, SIZE, SIZE), 3, dtype=np.uint8)

    first = pipeline({OBS_SCENE_IMAGES: pictures})[OBS_SCENE_TOKENS]
    again = pipeline({OBS_SCENE_IMAGES: pictures})[OBS_SCENE_TOKENS]

    assert encoder.pictures == 1
    assert torch.equal(first, again)
    pipeline({OBS_SCENE_IMAGES: np.full((1, 3, SIZE, SIZE), 4, dtype=np.uint8)})
    assert encoder.pictures == 2  # a new scene is a new encode


# ---------------------------------------------------------------- the letterbox


def test_scene_images_letterboxes_a_720x1280_frame_to_uint8_channel_first():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :640] = 255
    scene = scene_images({"scene_zed": frame}, "scene_zed", 224)
    assert scene.shape == (3, 224, 224)
    assert scene.dtype == np.uint8
    np.testing.assert_array_equal(scene, np.moveaxis(resize_with_pad(frame, 224), -1, 0))
    # the frame reaches both side edges (16:9 scaled to 224 wide), so the bars are top and bottom rows
    content = round(720 * 224 / 1280)
    top = (224 - content) // 2
    assert np.all(scene[:, :top, :] == 0) and np.all(scene[:, top + content :, :] == 0)
    assert np.all(scene[:, top : top + content, 0] == 255)  # the frame's white left half
    assert np.all(scene[:, top : top + content, -1] == 0)  # its black right half


def test_scene_images_needs_the_configured_feed_and_a_uint8_frame():
    with pytest.raises(ValueError, match="no scene camera feed 'scene_zed'"):
        scene_images({"other_zed": np.zeros((8, 8, 3), dtype=np.uint8)}, "scene_zed", 8)
    with pytest.raises(ValueError, match="uint8"):
        scene_images({"scene_zed": np.zeros((8, 8, 3), dtype=np.float32)}, "scene_zed", 8)
