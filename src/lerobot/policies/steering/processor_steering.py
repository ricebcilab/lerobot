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

"""The steering policy's pre-processing: ``SceneEncoder``, pi0.5's frozen vision tokens of rendered scenes (the
engine of the scene step), the two steering steps (``SteeringSceneTokensProcessorStep``, the scene's tokens for a
MARS task module; ``SteeringCommandProcessorStep``, the operator's command), and
``make_steering_pre_post_processors``, the pipelines around them."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.envs.cursor import SCENE_SIZE, Layouts, render_layouts
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_steering import SteeringConfig
from .modeling_steering import DEADBAND, VLM_SCENE_DIM, VLM_SCENE_TOKENS, LearnedTaskModule

SCENE_VLM_PATH = "lerobot/pi05_base"  # the pi0.5 checkpoint SceneEncoder reads by default
SCENE_PROMPT = "Move the cursor to the target."  # the prompt prefix_tokens runs behind the image by default
PI05_TOKENIZER = "google/paligemma-3b-pt-224"  # the tokenizer pi0.5's processor uses

STORED_SCENE_SOURCES = ("vlm_image", "vlm_prefix")  # the sources with one token array per layout

OBS_LAYOUTS = "observation.layouts"  # the scene step's input: the trials' layouts (a ``Layouts``) ...
OBS_SCENE_IMAGES = "observation.scene_images"  # ... or their pictures, uint8 [B, 3, S, S]
OBS_SCENE_TOKENS = "observation.scene_tokens"  # the scene step's output, float16 [B, 256, width]
COMMAND = "command"  # the operator's command, in the complementary data


class SceneEncoder:
    """pi0.5's frozen vision tokens of rendered scenes. ``image_tokens`` are SigLIP's patch tokens after the
    multimodal projector (``embed_image``), the tokens pi0.5's prefix starts with; ``prefix_tokens`` are the
    same positions in the last hidden state of pi0.5's language model run on the prefix, the image followed
    by the fixed ``prompt`` (tokenized as pi0.5's processor does, without state tokens, so the tokens depend
    on the layout alone). Both take uint8 ``[B, 3, S, S]``, resized with padding to pi0.5's resolution when S
    differs, and return float16 ``[B, 256, 2048]`` on the CPU.

    ``SceneEncoder.scene_for(layouts, source, size)`` is a layout batch's scene array through this encoder and
    ``SceneEncoder.encode_views(images, batch)`` the tokens of several views per scene; their static forms,
    ``SceneEncoder.scene_array`` and ``SceneEncoder.view_tokens``, take any object with the encoder's methods
    (``scene_array`` none for the ``cnn`` render). ``SceneEncoder.fit_reduction`` is the PCA of token batches a
    reduced task module carries.

    Lazy: constructing it loads nothing; the first call loads ``path`` (``PI05Policy.from_pretrained``) and
    imports pi0.5 and ``transformers``, after checking that its ``model.safetensors`` is on disk. ``close``
    frees the policy."""

    def __init__(
        self,
        path: str = SCENE_VLM_PATH,
        device: str = "cuda",
        prompt: str = SCENE_PROMPT,
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
    def encode(self, images: np.ndarray, batch: int = 256) -> torch.Tensor:
        """SigLIP's projected patch tokens of uint8 ``[B, 3, S, S]`` images, kept on the encoder's device:
        float16 ``[B, 256, 2048]``."""
        policy = self._load()
        vlm = policy.model.paligemma_with_expert
        out = []
        for i in range(0, len(images), batch):
            pixels = self._pixels(images[i : i + batch], policy.config.image_resolution)
            out.append(vlm.embed_image(pixels).to(torch.float16))
        if not out:
            return torch.zeros((0, VLM_SCENE_TOKENS, VLM_SCENE_DIM), dtype=torch.float16, device=self.device)
        return torch.cat(out)

    def image_tokens(self, images: np.ndarray, batch: int = 64) -> np.ndarray:
        """``encode`` on the CPU as numpy: float16 ``[B, 256, 2048]``, ``batch`` pictures at a time, each batch's
        tokens moved to the CPU before the next is encoded (the device holds one batch, not all of them)."""
        out = [self.encode(images[i : i + batch], batch).cpu() for i in range(0, len(images), batch)]
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

    def scene_for(self, layouts: Layouts, source: str, size: int = SCENE_SIZE) -> np.ndarray:
        """The per-layout scene array for ``source`` through this encoder (``scene_array`` with it)."""
        return SceneEncoder.scene_array(layouts, source, size, self)

    def encode_views(self, images: np.ndarray, batch: int = 256) -> torch.Tensor:
        """The tokens of uint8 ``[B, V, 3, S, S]`` views through this encoder (``view_tokens`` with it)."""
        return SceneEncoder.view_tokens(images, self, batch)

    @staticmethod
    def scene_array(
        layouts: Layouts, source: str, size: int = SCENE_SIZE, encoder: "SceneEncoder | None" = None
    ) -> np.ndarray:
        """The per-layout scene array for ``source``, ``[B, ...]``: ``cnn`` the render (uint8 ``[B, 3, size,
        size]``), ``vlm_image`` / ``vlm_prefix`` the encoder's ``image_tokens`` / ``prefix_tokens`` of it (float16
        ``[B, 256, 2048]``, cast to float16 whatever the encoder returns). ``encoder`` is anything with those two
        methods, typically one ``SceneEncoder`` built once by the caller. ``discs`` reads the layouts' obstacles
        directly and has no scene array."""
        if source == "discs":  # nosec B105
            raise ValueError("source 'discs' reads the layouts' obstacles; it has no scene array")
        LearnedTaskModule.spec(source, size)  # the source and size checks
        images = render_layouts(layouts, size)
        if source == "cnn":  # nosec B105
            return images
        if encoder is None:
            raise ValueError(f"source {source!r} needs an encoder (e.g. a SceneEncoder)")
        tokens = encoder.image_tokens(images) if source == "vlm_image" else encoder.prefix_tokens(images)  # nosec B105
        return np.asarray(tokens, dtype=np.float16)

    @staticmethod
    def view_tokens(images: np.ndarray, encoder, batch: int = 256) -> torch.Tensor:
        """The tokens of uint8 ``[B, V, 3, S, S]`` views, ``encoder.encode`` of each view's pictures joined view
        by view (view 0's tokens first): float16 ``[B, V * 256, 2048]`` on the encoder's device."""
        b, v = images.shape[:2]
        tokens = encoder.encode(images.reshape(b * v, *images.shape[2:]), batch)
        return tokens.reshape(b, v * tokens.shape[1], tokens.shape[2])

    @staticmethod
    def fit_reduction(
        batches: Iterable[torch.Tensor], rank: int, center: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """A PCA of token batches ``[..., W]`` (leading dims flattened): the ``mean`` ``[W]``, the ``basis``
        ``[W, rank]`` of orthonormal columns (largest variance first), both float32, and the share of the variance
        the basis keeps. Sums are kept in float64 on the first batch's device. Raises ``ValueError`` unless
        ``0 < rank < W``. ``center=False`` keeps the tokens' constant part: ``mean`` is zeros and ``basis`` the top
        ``rank`` eigenvectors of the uncentred second moment (largest first), ``kept`` their share of the total
        second moment."""
        total = second = None
        count = 0
        for batch in batches:
            x = batch.reshape(-1, batch.shape[-1]).to(torch.float64)
            if total is None:
                width = x.shape[1]
                if not 0 < rank < width:
                    raise ValueError(f"rank must satisfy 0 < rank < {width}, got {rank}")
                total = torch.zeros(width, dtype=torch.float64, device=x.device)
                second = torch.zeros(width, width, dtype=torch.float64, device=x.device)
            total += x.sum(0)
            second += x.T @ x
            count += len(x)
        if total is None or second is None or count < 2:
            raise ValueError("fit_reduction needs at least two tokens")
        if not center:
            values, vectors = torch.linalg.eigh(second / count)  # ascending
            values = values.clamp_min(0)
            kept = float(values[-rank:].sum() / values.sum())
            zeros = torch.zeros(second.shape[0], dtype=torch.float32, device=second.device)
            return zeros, vectors[:, -rank:].flip(1).float().contiguous(), kept
        mean = total / count
        cov = (second - count * torch.outer(mean, mean)) / (count - 1)
        values, vectors = torch.linalg.eigh(cov)  # ascending
        values = values.clamp_min(0)
        kept = float(values[-rank:].sum() / values.sum())
        return mean.float(), vectors[:, -rank:].flip(1).float().contiguous(), kept


# ---------------------------------------------------------------- the steering steps


@ProcessorStepRegistry.register(name="steering_scene_tokens_processor_step")
@dataclass
class SteeringSceneTokensProcessorStep(ProcessorStep):
    """The scene's tokens for a MARS task module with an image source (``vlm_image`` / ``vlm_prefix``).

    Reads the trials' layouts (``observation.layouts``, a ``Layouts``, rendered at ``render_size``) or their
    pictures (``observation.scene_images``, uint8 ``[B, 3, S, S]``) from the observation and writes
    ``observation.scene_tokens``: float16 ``[B, 256, 2048]``, the ``SceneEncoder``'s tokens of ``source``, or
    ``[B, 256, rank]`` when the step holds a task module's ``reduction`` (``(mean, basis)``, applied as the module
    applies it). The encoder (``vlm_path`` on ``device``, ``batch`` pictures at a time) is built on first use, or
    given as ``encoder`` (any object with ``image_tokens`` / ``prefix_tokens``)."""

    source: str = "vlm_image"
    render_size: int = SCENE_SIZE
    vlm_path: str = SCENE_VLM_PATH
    batch: int = 64
    device: str = "cpu"
    reduction: tuple[torch.Tensor, torch.Tensor] | None = None
    encoder: Any | None = None

    _encoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source not in STORED_SCENE_SOURCES:
            raise ValueError(f"source must be one of {STORED_SCENE_SOURCES}, got {self.source!r}")
        LearnedTaskModule.spec(self.source, self.render_size)  # the size check
        self._encoder = self.encoder

    def scene_encoder(self) -> Any:
        """The encoder, built on first use."""
        if self._encoder is None:
            self._encoder = SceneEncoder(self.vlm_path, self.device)
        return self._encoder

    def tokens(self, observation: dict[str, Any]) -> torch.Tensor:
        """The scene tokens of an observation's layouts or pictures, float16, reduced when the step holds a
        reduction."""
        layouts, images = observation.get(OBS_LAYOUTS), observation.get(OBS_SCENE_IMAGES)
        if layouts is None and images is None:
            raise ValueError(
                f"SteeringSceneTokensProcessorStep needs {OBS_LAYOUTS!r} or {OBS_SCENE_IMAGES!r} in the observation"
            )
        encoder = self.scene_encoder()
        if layouts is not None:
            array = SceneEncoder.scene_array(layouts, self.source, self.render_size, encoder)
        else:
            pictures = np.asarray(images.cpu() if torch.is_tensor(images) else images, dtype=np.uint8)
            encode = encoder.image_tokens if self.source == "vlm_image" else encoder.prefix_tokens  # nosec B105
            array = np.asarray(encode(pictures), dtype=np.float16)
        tokens = torch.as_tensor(array)
        if self.reduction is not None:
            mean, basis = self.reduction
            tokens = LearnedTaskModule.reduce(tokens, mean.to(tokens.device), basis.to(tokens.device))
        return tokens

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("SteeringSceneTokensProcessorStep needs an observation")
        observation = dict(observation)
        observation[OBS_SCENE_TOKENS] = self.tokens(observation)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def width(self) -> int:
        """The tokens' width: the reduction's rank, else the VLM's 2048."""
        return VLM_SCENE_DIM if self.reduction is None else int(self.reduction[1].shape[1])

    def get_config(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "render_size": self.render_size,
            "vlm_path": self.vlm_path,
            "batch": self.batch,
            "device": self.device,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.reduction is None:
            return {}
        return {"scene_mean": self.reduction[0], "scene_basis": self.reduction[1]}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.reduction = (state["scene_mean"], state["scene_basis"]) if state else None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Adds ``observation.scene_tokens`` (256 tokens of the step's width)."""
        observation = dict(features.get(PipelineFeatureType.OBSERVATION, {}))
        observation[OBS_SCENE_TOKENS] = PolicyFeature(
            type=FeatureType.ENV, shape=(VLM_SCENE_TOKENS, self.width())
        )
        return {**features, PipelineFeatureType.OBSERVATION: observation}


@ProcessorStepRegistry.register(name="steering_command_processor_step")
@dataclass
class SteeringCommandProcessorStep(ProcessorStep):
    """The operator's command as the steering policy reads it.

    Reads ``command`` from the complementary data (one command per row, ``[B, >= n_dims]``, or one per chunk
    step, ``[B, H, >= n_dims]``, in the operator's units: a deflection in [-1, 1] per dim), keeps its first
    ``n_dims`` dims as float32 and zeroes the rows inside ``deadband`` (every entry below it in magnitude: the
    operator is idle and the row gets the policy's own chunk), and writes it back. The steering policy converts
    the deflection to the policy's action units itself, as each mode needs it (flow control writes it at
    ``ic_gain``; FRS's reference moves at ``max_speed``)."""

    deadband: float = DEADBAND
    n_dims: int = 2

    def command(self, command: Any) -> torch.Tensor:
        """The processed command: float32, the first ``n_dims`` dims, the idle rows zeroed."""
        command = torch.as_tensor(np.asarray(command) if not torch.is_tensor(command) else command)
        command = command[..., : self.n_dims].to(torch.float32)
        magnitude = command.abs().amax(-1)
        active = (magnitude.amax(-1) if magnitude.dim() == 2 else magnitude) >= self.deadband
        shape = (-1,) + (1,) * (command.dim() - 1)
        return torch.where(active.reshape(shape), command, torch.zeros_like(command))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if not data or data.get(COMMAND) is None:
            raise ValueError(f"SteeringCommandProcessorStep needs {COMMAND!r} in the complementary data")
        transition[TransitionKey.COMPLEMENTARY_DATA] = {**data, COMMAND: self.command(data[COMMAND])}
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"deadband": self.deadband, "n_dims": self.n_dims}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """The command is complementary data, not a feature: the features are unchanged."""
        return features


def steering_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """``batch_to_transition`` that also carries the operator's ``command`` in the complementary data."""
    transition = batch_to_transition(batch)
    if COMMAND in batch:
        data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        data[COMMAND] = batch[COMMAND]
        transition[TransitionKey.COMPLEMENTARY_DATA] = data
    return transition


def make_steering_pre_post_processors(
    config: SteeringConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    preprocessor_kwargs: dict[str, Any] | None = None,
    postprocessor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the steering policy.

    The pre-processing pipeline prepares input data for the policy by:
    1. Renaming features to match pretrained configurations.
    2. Adding a batch dimension.
    3. Normalizing input and output features based on dataset statistics (none for a base policy that reads
       its own observation dict, as the cursor's).
    4. Encoding the scene into tokens for a task module with an image source (``vlm_image`` / ``vlm_prefix``; the
       source and render size are the task module's, read from its checkpoint when the config has not recorded it).
    5. Reading the operator's command (the first ``n_dims`` dims, the idle rows zeroed).
    6. Moving all data to the specified device.

    The post-processing pipeline handles the policy's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving data to the CPU.

    Args:
        config: The configuration object for the steering policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    task = config.task_module
    if task is None and config.adapter_task is not None:
        task = torch.load(config.adapter_task, map_location="cpu", weights_only=True)["config"]
    scene = config.scene_settings(task)  # the task module's, where the config leaves them unset
    if task is not None and scene["scene_source"] in STORED_SCENE_SOURCES:
        input_steps.append(
            SteeringSceneTokensProcessorStep(
                source=scene["scene_source"],
                render_size=scene["render_size"] or SCENE_SIZE,
                vlm_path=config.vlm_path or SCENE_VLM_PATH,
                device=config.device,
            )
        )
    input_steps += [
        SteeringCommandProcessorStep(deadband=config.deadband, n_dims=config.n_dims),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=steering_batch_to_transition,
            **(preprocessor_kwargs or {}),
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            **(postprocessor_kwargs or {}),
        ),
    )
