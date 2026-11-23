#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""What turns a station observation into the policy's batch: the camera map and the tensors.

pi05_droid wants a proprioception state, three letterboxed images in its own feature names, and the task.
The station sends two cameras under its own names, so this module is the one place that knows both naming
schemes: `CameraMap` names the feeds, and the preprocessor's rename step (built with the same map) does the
renaming. Nothing here imports torch at module import time; the batch is a torch batch only when it is made.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from config import ExperimentConfig
from interfaces import ACTION_DIMS, StationManifest
from station import IMAGE_SIZE, resize_with_pad, state_from_obs

__all__ = [
    "CameraMap",
    "camera_map",
    "to_action",
    "observation_batch",
    "pad_state",
    "suggested_cameras",
    "unnormalized",
]

STATE_DIM = 32
"""The checkpoint's `max_state_dim`, and so the width of the state it was trained to read.

DROID's statistics are 32 wide and the state is written into the prompt as one number per dim, so a batch that
sends the arm's eight real dims alone fails twice: the normalizer's `(x - mean) / std` cannot broadcast 8
against 32, and the state token string would be 24 numbers shorter than the checkpoint's. Zero padding is what
the checkpoint's own client sends, and the padding dims' statistics are zero, so they normalize to zero.
"""

DROID_IMAGE_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
"""pi05_droid's three camera inputs, in the checkpoint's feature names (DESIGN.md)."""


@dataclass(frozen=True)
class CameraMap:
    """Which checkpoint image each station camera feeds, and the feeds the station does not have.

    `sources` is keyed by the checkpoint's feature name; a `None` value is a camera the checkpoint wants and
    this station does not supply, which the runner fills black (`DESIGN.md`: the checkpoint wants three, the
    station feeds two). The rename map is what the preprocessor's rename step takes.
    """

    sources: Mapping[str, str | None]

    def rename_map(self) -> dict[str, str]:
        """The station observation key to checkpoint key map the preprocessor's rename step takes."""
        return {
            f"observation.images.{source}": name
            for name, source in self.sources.items()
            if source is not None
        }

    def black(self) -> tuple[str, ...]:
        """The checkpoint image keys the station does not feed."""
        return tuple(name for name, source in self.sources.items() if source is None)


_EXTERIOR_ROLES = ("scene", "front", "base", "ext", "external", "overview")
_WRIST_ROLES = ("wrist", "hand", "gripper", "tool")

"""Name fragments that say which role a feed plays. `scene_zed` is the exterior view and `wrist_zed` the
wrist one; a station whose names say neither falls back to its own order. The config always wins, so this is
only what an unnamed feed defaults to."""


def _first(feeds: list[str], roles: tuple[str, ...]) -> str | None:
    """The first feed whose name carries one of `roles`, or None."""
    return next((name for name in feeds if any(role in name.lower() for role in roles)), None)


def _is_depth(name: str) -> bool:
    """Whether a station feed is a depth or mono image rather than an RGB frame.

    These stations send a depth map beside each ZED, and it is one channel: the checkpoint cannot read it, and
    picking it as the wrist feed would fail deep inside the batch with a shape error instead of here.
    """
    return any(token in name.lower() for token in ("depth", "mono", "ir_"))


def suggested_cameras(manifest: StationManifest) -> dict[str, str]:
    """Which station feeds to use when the config names none: the exterior view and the wrist one.

    Feeds whose names carry a role (`scene`, `front`, `base` for the exterior; `wrist`, `hand` for the wrist)
    win, and a depth or mono feed never qualifies. A station that names neither falls back to its own order.
    The config always overrides this, so it is a suggestion the probe prints and a default the runner takes.
    """
    feeds = [name for name in manifest.cameras if not _is_depth(name)]
    if len(feeds) < 2:
        raise ValueError(
            f"the station sends cameras {manifest.cameras}; the runner needs an exterior and a wrist feed, and "
            "neither can be a depth or mono feed (name them in `cameras:`)"
        )
    exterior = _first(feeds, _EXTERIOR_ROLES) or feeds[0]
    wrist = _first(feeds, _WRIST_ROLES) or next(name for name in feeds if name != exterior)
    return {"exterior": exterior, "wrist": wrist}


def camera_map(config: ExperimentConfig, manifest: StationManifest) -> CameraMap:
    """The checkpoint's three image feeds, from the config's two names or the station's own order.

    The per-arm file names the station's exterior and wrist feeds once the probe has filled them in; until
    then the manifest's order is the station's own and is used as-is. Every named feed must be one the
    station actually sends, and the checkpoint's third feed is black.
    """
    exterior, wrist = config.cameras.exterior, config.cameras.wrist
    if exterior is None or wrist is None:
        feeds = [name for name in manifest.cameras if not _is_depth(name)]
        if len(feeds) < 2:
            raise ValueError(
                f"the station sends cameras {manifest.cameras}; the runner needs an exterior and a wrist feed, "
                "and neither can be a depth or mono feed (name them in `cameras:`)"
            )
        roles = suggested_cameras(manifest)
        exterior = exterior or roles["exterior"]
        wrist = wrist or roles["wrist"]
    missing = [name for name in (exterior, wrist) if name not in manifest.cameras]
    if missing:
        raise ValueError(f"cameras {missing} are not among the station's {manifest.cameras}")
    return CameraMap({DROID_IMAGE_KEYS[0]: exterior, DROID_IMAGE_KEYS[1]: wrist, DROID_IMAGE_KEYS[2]: None})


def pad_state(state: np.ndarray, state_dim: int = STATE_DIM) -> np.ndarray:
    """The arm's state, zero-padded to the checkpoint's width (a no-op when it is already that wide)."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] > state_dim:
        raise ValueError(f"the state has {state.shape[0]} dims, more than the checkpoint's {state_dim}")
    return np.concatenate([state, np.zeros(state_dim - state.shape[0], dtype=np.float32)])


def observation_batch(
    obs: Mapping[str, np.ndarray], arm: str, cameras: CameraMap, prompt: str, state_dim: int = STATE_DIM
) -> dict[str, Any]:
    """One station observation as the preprocessor's input batch: state, three images, and the task.

    The images keep the station's keys, `observation.images.<feed>`, so the checkpoint's rename step is what
    maps them to its own feature names; the black feed is written under its checkpoint name directly, since
    there is no station key to rename from. Frames are resized with openpi's letterbox and scaled to [0, 1],
    the preparation the DROID client does.
    """
    import torch

    state = pad_state(state_from_obs(obs, arm), state_dim)
    batch: dict[str, Any] = {"observation.state": torch.from_numpy(state)}
    for name, source in cameras.sources.items():
        if source is None:
            frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            key = name
        else:
            frame = np.array(resize_with_pad(np.asarray(obs[source]), IMAGE_SIZE), copy=True)
            key = f"observation.images.{source}"
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        batch[key] = image
    batch["task"] = prompt
    return batch


def to_action(postprocessor, value, *, dims: int = ACTION_DIMS) -> np.ndarray:
    """A policy action as the ``(dims,)`` the arm can execute: unnormalized, then the checkpoint's padding cut.

    The DROID checkpoint's action feature is 32 wide (its statistics are too), while the arm has eight numbers'
    worth of action. The padding dims' statistics are zero, so unnormalizing them yields zero and dropping them
    loses nothing; keeping them would be a 32-long "action" that the clamps refuse.
    """
    action = unnormalized(postprocessor, value)
    if action.shape[0] < dims:
        raise ValueError(
            f"the checkpoint produced {action.shape[0]} action dims, fewer than the {dims} this arm executes"
        )
    return action[:dims]


def unnormalized(postprocessor, value) -> np.ndarray:
    """A policy action after the postprocessor, as one flat float array of its action dims.

    The pipeline's action form is a tensor; a caller holding a transition-shaped mapping is accepted too, so
    the runner does not depend on which of the two a checkpoint's postprocessor returns.
    """
    from lerobot.utils.constants import ACTION

    out = postprocessor(value)
    if isinstance(out, Mapping):
        out = out[ACTION]
    if hasattr(out, "detach"):
        out = out.detach().cpu()
    return np.asarray(out, dtype=np.float64).reshape(-1)
