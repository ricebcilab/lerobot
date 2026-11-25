"""The scene a MARS task checkpoint asks for: which pictures to make, and the step that tokenizes them.

Phase C conditions a ``LearnedTaskModule`` on a scene, and the module's checkpoint records which source
it reads (DESIGN.md, "Phase C: the MARS task module"). One source is in this phase's scope:
``vlm_image``, the frozen VLM's image tokens of one scene picture. The inference and training batches
therefore carry ``observation.scene_images`` -- the scene camera (ZED2), letterboxed to the module's
render size, uint8 channel-first -- and the steering package's ``SteeringSceneTokensProcessorStep``
turns them into ``observation.scene_tokens``, the tokens ``SteeringPolicy._adapter_view`` hands the task
module as its scene.

``Scene`` is the checkpoint's answer as a value (source, render size); ``from_checkpoint`` reads it from
a task module's recorded constructor config without building the module; ``Scene.step`` is the one-step
pipeline that encodes pictures into tokens (with the ``SceneEncoder`` injectable: the real one loads
``lerobot/pi05_base``, a second VLM in the process, so tests pass a fake); and ``scene_images`` is the
letterbox from a station observation to the step's picture, through ``station.resize_with_pad`` -- the
same openpi letterbox the policy's own images take, not a second derivation of it.

Import-light on purpose: ``loop`` imports this module on ``--dry-run``'s path, so the lerobot imports
are inside the two functions that need them and nothing here loads torch, a checkpoint or a VLM at
import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from station import resize_with_pad

if TYPE_CHECKING:
    from lerobot.processor import PolicyProcessorPipeline

TASK_SOURCE = "vlm_image"
"""The one token source this phase's scene reads: the frozen VLM's image tokens of one picture."""

OUT_OF_SCOPE = ("cnn", "discs", "vlm_prefix", "vlm_views")
"""The task module's other token sources: their camera renders, obstacle discs, prefix tokens and camera
views are not built here (DESIGN.md's Phase C scope)."""


@dataclass(frozen=True)
class Scene:
    """What a task checkpoint asks for: the token source it reads and the size it renders pictures at.

    ``source`` is ``vlm_image`` only: the other sources belong to later work (``OUT_OF_SCOPE``).
    ``render_size`` is the side of the square picture the scene camera is letterboxed to; the VLM resizes
    it to pi0.5's own resolution, so it is only what the batch carries.
    """

    source: str
    render_size: int

    def __post_init__(self) -> None:
        if self.source != TASK_SOURCE:
            _refuse_source(self.source, where="")
        if isinstance(self.render_size, bool) or not isinstance(self.render_size, int):
            raise ValueError(f"render_size must be an integer, got {self.render_size!r}")
        if self.render_size <= 0:
            raise ValueError(f"render_size must be positive, got {self.render_size}")

    def step(self, *, device: str, encoder: Any | None = None) -> PolicyProcessorPipeline:
        """This scene's one-step pipeline: a batch with ``observation.scene_images`` -> the same batch with
        ``observation.scene_tokens`` (float16 ``[B, 256, 2048]``).

        The step is the steering package's ``SteeringSceneTokensProcessorStep`` -- the same step the
        steering policy's pre-processor uses -- over the batch converter that carries a training batch's
        keys through untouched (``steering_batch_to_transition``). ``encoder`` is the step's
        ``SceneEncoder``, or any object with its ``image_tokens``/``prefix_tokens``: the real one loads
        ``lerobot/pi05_base`` on ``device`` on first use, which is what makes the step injectable in the
        offline tests.
        """
        from lerobot.policies.steering import SteeringSceneTokensProcessorStep
        from lerobot.policies.steering.processor_steering import steering_batch_to_transition
        from lerobot.processor import PolicyProcessorPipeline

        step = SteeringSceneTokensProcessorStep(
            source=self.source,
            render_size=self.render_size,
            device=device,
            encoder=encoder,
        )
        return PolicyProcessorPipeline(
            steps=[step],
            to_transition=steering_batch_to_transition,
        )


def from_checkpoint(path: Path | str) -> Scene:
    """The scene a task module's checkpoint asks for: its recorded ``token_source`` and ``image_size``.

    ``AdapterModules.read_checkpoint`` reads the checkpoint's recorded constructor config, so no module
    is built and no weights are needed beyond the file: the scene is known before the VLM it asks for is
    anywhere in sight. ``token_source`` must be ``vlm_image``; the other sources are refused by name as
    this phase's scope, not silently ignored. A checkpoint of the other kind (a user module) has no
    scene and is refused as such.
    """
    from lerobot.policies.steering import AdapterModules

    checkpoint = AdapterModules.read_checkpoint(path)
    if checkpoint["kind"] != "task":
        raise ValueError(
            f"{path}: a {checkpoint['kind']} module checkpoint; a scene is recorded by a task module"
        )
    config = checkpoint["config"]
    source = config.get("token_source", "discs")  # the module's config omits its default, discs
    if source != TASK_SOURCE:
        _refuse_source(source, where=f"{path}: ")
    size = config.get("image_size")
    if size is None:
        raise ValueError(
            f"{path}: the task checkpoint records no image_size, so its render size is unknown; "
            f"a {TASK_SOURCE!r} module records the size it was built with"
        )
    return Scene(source=source, render_size=int(size))


def scene_images(obs: Mapping[str, NDArray], camera: str, size: int) -> NDArray:
    """The scene camera's frame as the scene step's picture: uint8 ``(3, size, size)``, channel first.

    ``camera`` is the station's feed name (the config's ``[cameras] exterior``, the ZED2); the frame is a
    ``(H, W, 3)`` uint8 picture, letterboxed through ``station.resize_with_pad`` -- the openpi letterbox
    the policy's own images take, one scale for both axes and black bars, nothing stretched -- and moved
    channel first as the checkpoint's own images are.
    """
    if camera not in obs:
        raise ValueError(f"the observation has no scene camera feed {camera!r} (keys: {sorted(obs)})")
    frame = np.asarray(obs[camera])
    return np.moveaxis(resize_with_pad(frame, size), -1, 0)


def _refuse_source(source: str, *, where: str) -> None:
    """Raises this phase's scope: ``vlm_image`` is built, the module's other sources are not."""
    raise ValueError(
        f"{where}the task module's token_source is {source!r}; this phase implements {TASK_SOURCE!r} only, "
        f"the {', '.join(OUT_OF_SCOPE)} sources are out of scope"
    )
