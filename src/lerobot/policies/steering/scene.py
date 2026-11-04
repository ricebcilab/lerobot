#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""The scene a ``LearnedTaskModule`` reads, per token source: ``scene_spec`` (the module's kwargs for a source),
``scene_for`` (the per-layout array it reads from ``obs["scene"][source]``) and ``SceneEncoder`` (pi0.5's
frozen vision tokens of the rendered layouts). pi0.5 and ``transformers`` are imported only when the encoder
first runs, so this module imports without them."""

from pathlib import Path

import numpy as np
import torch

from lerobot.envs.cursor import SCENE_SIZE, Layouts, render_layouts

from .modeling_steering import SCENE_SOURCES

VLM_SCENE_TOKENS = 256  # SigLIP's patches of the 224 px input every render is resized to
VLM_SCENE_DIM = 2048  # the width of pi0.5's prefix (SigLIP after the multimodal projector, and Gemma's)
CNN_PATCH = 16  # the task module's CNN stem downsamples by 16
PI05_TOKENIZER = "google/paligemma-3b-pt-224"  # the tokenizer pi0.5's processor uses


def scene_spec(source: str, size: int = SCENE_SIZE) -> dict:
    """The ``LearnedTaskModule`` kwargs for ``source`` on size-``size`` renders: ``discs`` needs none beyond
    the source; ``vlm_*`` read 256 tokens of width 2048 whatever the size (the encoder resizes every render
    to 224 for SigLIP; ``size`` is only what is drawn and cached); ``cnn`` reads the render itself,
    ``(size // 16) ** 2`` tokens (size a multiple of 16)."""
    if source not in SCENE_SOURCES:
        raise ValueError(f"source must be one of {SCENE_SOURCES}, got {source!r}")
    if source == "discs":  # nosec B105
        return {"token_source": "discs"}
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if source == "cnn":  # nosec B105
        if size % CNN_PATCH:
            raise ValueError(f"source 'cnn' needs a size that is a multiple of {CNN_PATCH}, got {size}")
        return {
            "token_source": "cnn",
            "scene_dim": None,
            "scene_tokens": (size // CNN_PATCH) ** 2,
            "image_size": size,
        }
    return {
        "token_source": source,
        "scene_dim": VLM_SCENE_DIM,
        "scene_tokens": VLM_SCENE_TOKENS,
        "image_size": size,
    }


def scene_for(
    layouts: Layouts, source: str, size: int = SCENE_SIZE, encoder: "SceneEncoder | None" = None
) -> np.ndarray:
    """The per-layout scene array for ``source``, ``[B, ...]``: ``cnn`` the render (uint8 ``[B, 3, size,
    size]``), ``vlm_image`` / ``vlm_prefix`` the encoder's ``image_tokens`` / ``prefix_tokens`` of it (float16
    ``[B, 256, 2048]``, cast to float16 whatever the encoder returns). ``encoder`` is anything with those two
    methods, typically one ``SceneEncoder`` built once by the caller. ``discs`` reads the layouts' obstacles
    directly and has no scene array."""
    if source == "discs":  # nosec B105
        raise ValueError("source 'discs' reads the layouts' obstacles; it has no scene array")
    scene_spec(source, size)  # the source and size checks
    images = render_layouts(layouts, size)
    if source == "cnn":  # nosec B105
        return images
    if encoder is None:
        raise ValueError(f"source {source!r} needs an encoder (e.g. a SceneEncoder)")
    tokens = encoder.image_tokens(images) if source == "vlm_image" else encoder.prefix_tokens(images)  # nosec B105
    return np.asarray(tokens, dtype=np.float16)


class SceneEncoder:
    """pi0.5's frozen vision tokens of rendered scenes. ``image_tokens`` are SigLIP's patch tokens after the
    multimodal projector (``embed_image``), the tokens pi0.5's prefix starts with; ``prefix_tokens`` are the
    same positions in the last hidden state of pi0.5's language model run on the prefix, the image followed
    by the fixed ``prompt`` (tokenized as pi0.5's processor does, without state tokens, so the tokens depend
    on the layout alone). Both take uint8 ``[B, 3, S, S]``, resized with padding to pi0.5's resolution when S
    differs, and return float16 ``[B, 256, 2048]`` on the CPU.

    Lazy: constructing it loads nothing; the first call loads ``path`` (``PI05Policy.from_pretrained``) and
    imports pi0.5 and ``transformers``, after checking that its ``model.safetensors`` is on disk. ``close``
    frees the policy."""

    def __init__(
        self,
        path: str = "lerobot/pi05_base",
        device: str = "cuda",
        prompt: str = "Move the cursor to the target.",
    ):
        self.path, self.device, self.prompt = path, device, prompt
        self._policy = None
        self._prompt_ids: tuple[torch.Tensor, torch.Tensor] | None = None  # tokens, masks: [1, L]

    def _check_weights(self) -> None:
        """Raises unless ``path``'s ``model.safetensors`` is on disk (in the directory ``path``, or in the local
        Hugging Face cache for a repo id): ``PI05Policy.from_pretrained`` returns a randomly initialised policy
        when it cannot find the file, and its tokens would look like real ones."""
        from huggingface_hub import try_to_load_from_cache

        if Path(self.path).is_dir():
            found = (Path(self.path) / "model.safetensors").is_file()
        else:
            found = isinstance(try_to_load_from_cache(self.path, "model.safetensors"), str)
        if not found:
            raise FileNotFoundError(
                f"no model.safetensors for {self.path!r} (a directory, or a repo id in the local Hugging Face "
                "cache); refusing to encode with randomly initialised weights"
            )

    def _load(self):
        if self._policy is None:
            self._check_weights()
            from transformers import AutoTokenizer

            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            policy = PI05Policy.from_pretrained(self.path)
            policy.to(self.device).eval().requires_grad_(False)
            tokenizer = AutoTokenizer.from_pretrained(PI05_TOKENIZER)
            text = self.prompt.strip().replace("_", " ").replace("\n", " ")
            ids = tokenizer(
                f"Task: {text};\nAction: ",
                max_length=policy.config.tokenizer_max_length,
                truncation=True,
                padding="max_length",
                padding_side="right",
                return_tensors="pt",
            )
            self._prompt_ids = (
                ids["input_ids"].to(self.device),
                ids["attention_mask"].to(device=self.device, dtype=torch.bool),
            )
            self._policy = policy
        return self._policy

    def _pixels(self, images: np.ndarray, resolution: tuple[int, int]) -> torch.Tensor:
        """uint8 ``[B, 3, S, S]`` as pi0.5's ``_preprocess_images`` feeds SigLIP: float32 ``[B, 3, H, W]`` in
        [-1, 1], resized with padding to ``resolution`` when the size differs."""
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        img = torch.as_tensor(images, device=self.device).to(torch.float32) / 255.0
        img = img.permute(0, 2, 3, 1)  # BHWC, as resize_with_pad_torch takes it
        if tuple(img.shape[1:3]) != tuple(resolution):
            img = resize_with_pad_torch(img, *resolution)
        return img.permute(0, 3, 1, 2) * 2.0 - 1.0

    @torch.no_grad()
    def image_tokens(self, images: np.ndarray, batch: int = 64) -> np.ndarray:
        """SigLIP's projected patch tokens of uint8 ``[B, 3, S, S]`` images: float16 ``[B, 256, 2048]``."""
        policy = self._load()
        vlm = policy.model.paligemma_with_expert
        out = []
        for i in range(0, len(images), batch):
            pixels = self._pixels(images[i : i + batch], policy.config.image_resolution)
            out.append(vlm.embed_image(pixels).to(torch.float16).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, VLM_SCENE_TOKENS, VLM_SCENE_DIM), np.float16)

    @torch.no_grad()
    def prefix_tokens(self, images: np.ndarray, batch: int = 64) -> np.ndarray:
        """The image positions of the language model's last hidden state on the prefix (the image, then the
        fixed prompt), as ``PI05Pytorch`` encodes it: float16 ``[B, 256, 2048]``."""
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        policy = self._load()
        model = policy.model
        tokens, masks = self._prompt_ids
        out = []
        for i in range(0, len(images), batch):
            pixels = self._pixels(images[i : i + batch], policy.config.image_resolution)
            b = len(pixels)
            embs, pad_masks, att_masks = model.embed_prefix(
                [pixels],
                [torch.ones(b, dtype=torch.bool, device=self.device)],
                tokens.expand(b, -1),
                masks.expand(b, -1),
            )
            att_2d = model._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            (hidden, _), _ = model.paligemma_with_expert.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=False,
            )
            n_image = embs.shape[1] - tokens.shape[1]
            out.append(hidden[:, :n_image].to(torch.float16).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, VLM_SCENE_TOKENS, VLM_SCENE_DIM), np.float16)

    def close(self) -> None:
        """Frees the policy (and the GPU memory it held); a later call loads it again."""
        self._policy, self._prompt_ids = None, None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
