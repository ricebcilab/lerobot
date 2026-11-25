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

"""The steering policy's configuration: ``SteeringConfig``, registered ``"steering"``, and the constants its validation
and its adapter specs read (the modes, the task module's scene sources and token geometries, the action layout the
fixed reversal adapter F is built in). ``modeling_steering`` holds the policy and the cores, ``processor_steering`` the
pre-processing; this folder's README describes the methods."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, LRSchedulerConfig, OptimizerConfig
from lerobot.teleoperators import readout

STEERING_MODES = ("none", "flow_control", "frs")
SCENE_SOURCES = ("discs", "vlm_image", "vlm_prefix", "cnn", "vlm_views")
# absolute: a learned position per image token (round 1); relative: the token centre's offset from the cursor
# through the disc-token MLP
TOKEN_GEOMETRIES: tuple[str, ...] = ("absolute", "relative")
# The end-effector action space the steering cores assume (LIBERO's OSC_POSE: three translation, three rotation and
# one gripper dim), in which a fixed reversal adapter F is built, and the groups of it an operator can command
# (anything else is the policy's).
ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
N_ACTION_DIMS = len(ACTION_DIMS)
GRIPPER_DIM = ACTION_DIMS.index("gripper")
OPERATOR_DIM_GROUPS = {
    "translation": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(GRIPPER_DIM, GRIPPER_DIM + 1),
}
_BLOCK_CHOICES = ("identity", "zero")  # an adapter spec's orientation and gripper blocks


@PreTrainedConfig.register_subclass("steering")
@dataclass
class SteeringConfig(PreTrainedConfig):
    """Configuration of the steering policy (``SteeringPolicy``): a frozen flow policy's chunks steered toward an
    operator's command by flow control or Flow Reversal Steering, through the MARS adapter modules.

    The base policy reads its own observation (the cursor's dict, or pi0.5's features), so this config has, like
    ``CursorFlowConfig``, no input/output features of its own and no delta indices; its optimizer preset (AdamW)
    trains the adapter modules. The flow's settings (Euler steps, chunk length, the steps executed per replan) are
    the base policy's own, its config's sampler fields: this config does not repeat them.

    The adapter is four flat fields, so the CLI sets each directly: ``from_spec`` reads a ``reversal_adapter`` spec
    (a fixed F and/or module checkpoint paths) into them, and ``build_adapter`` builds the fixed F of a spec, for any
    ``n_dims``. The matrices in a spec are the operator readout's grammar (``lerobot.teleoperators.readout``'s
    ``matrix_spec``, which parses a command corruption's too).

    Args:
        mode: ``"flow_control"`` (Kao et al. 2026), ``"frs"`` (Flow Reversal Steering, Tang et al. 2026; MARS
            with an adapter), or ``"none"`` (the base policy's own chunk through the task module's velocity hook, the
            command ignored; no steering core).
        tau: Flow control: the chunk steps the command is written into (None: the whole chunk).
        lam: Flow control: the Euler steps the command is written at (lambda; 1 = the initial condition only).
        ic_gain: Flow control: the command's scale in the noise prior (a full push writes ``ic_gain`` sigma), or
            ``"action"``: the command in the base policy's action units (needs its postprocessor's statistics).
        zeroed_dims: Flow control: the action dim groups set to 0 in the steps the command is written into (names
            of ``OPERATOR_DIM_GROUPS``; Kao et al. on LIBERO, App. G.1: rotation), or None: they keep their noise.
        n_reversal_steps: FRS: the Euler steps of the reversal (None: all the way to noise).
        pinned_steps: FRS: the leading chunk steps the reference pins (None: all of them).
        deadband: Below this (the operator's units) the operator counts as idle and the row gets the policy's own
            chunk.
        operator_dims: The action dim groups the operator commands (names of ``OPERATOR_DIM_GROUPS``; None: every
            dim, as on the cursor).
        gripper_reference: FRS: the gripper value every reference holds, in environment units (Tang et al. on
            LIBERO, App. D.2: 0, halfway between open and closed), or None: the model's last executed gripper
            command. Only in the 7-D operator space (pi0.5's).
        adapter_matrix: The MARS adapter's fixed user matrix F (rows, ``n_dims`` x ``n_dims``; 7 x 7 in a LIBERO
            spec), or None. In the operator's (environment) units, normalized at use by the base's postprocessor's
            action statistics (``AdapterModules.normalized_matrix``; the cursor's uniform std leaves it as it is).
        adapter_user: The learned user module's checkpoint (``AdapterModules.save_module``'s format), or None.
        adapter_task: The learned task module's checkpoint, or None. All three None: plain FRS.
        adapter_blocks: The fixed matrix's spec as given (blocks or ``{F: rows}``), which ``build_adapter`` rebuilds
            ``adapter_matrix`` from (for another corruption: a random one is drawn per trial), or None.
        user_module: The user module's constructor config, recorded when the module is built (so a saved policy
            rebuilds it without its checkpoint); None: read from ``adapter_user``.
        task_module: The task module's, likewise.
        lambda_id: The task module's pull of A, b toward the identity in ``forward``.
        n_dims: The action dims the adapter acts on, and the most dims an operator's command may have (the command
            step refuses a wider one): 2 on the cursor, 3 for a pi0.5 base's translation.
        max_speed: The command's speed at full deflection, in action units per second (FRS's reference in an action
            space without the 7-D operator space's gripper dim, the cursor's; pi0.5's uses the command as is).
        dt: The control period in seconds (that reference: a full push moves ``max_speed * dt`` per step).
        reference_step: FRS: the action units one full deflection commands per control step, or None: the action
            space's own rule (the cursor's ``max_speed * dt``, or ``Flow.uniform_reference`` in the 7-D operator
            space).
        base_policy_type: The registered type of the policy whose chunks are steered.
        base_policy_path: Its checkpoint (a directory or a Hub repo id), loaded when no base policy is passed.
        scene_source: The task module's token source (``SCENE_SOURCES``).
        render_size: The size the scene is rendered at for the scene tokens.
        token_geometry: The task module's token geometry (``TOKEN_GEOMETRIES``).
        scene_rank: The width of the task module's token reduction (None: full width).
            The four scene settings are the task module's: None (unset) takes the module's value when the policy
            records the module (``scene_settings``); an explicit value that contradicts the module's raises.
        vlm_path: The pi0.5 checkpoint whose vision tokens the ``vlm_*`` sources read (None: the scene encoder's
            default, ``processor_steering.SCENE_VLM_PATH``).
        normalization_mapping: The normalization of the base policy's features (none for the cursor's).
    """

    mode: str = "frs"
    tau: int | None = 5
    lam: int = 1
    ic_gain: float | str = 1.0
    zeroed_dims: list[str] | None = None
    n_reversal_steps: int | None = None
    pinned_steps: int | None = None
    deadband: float = readout.DEADBAND
    operator_dims: list[str] | None = None
    gripper_reference: float | None = None
    adapter_matrix: list[list[float]] | None = None
    adapter_user: str | None = None
    adapter_task: str | None = None
    adapter_blocks: dict | None = None
    user_module: dict | None = None
    task_module: dict | None = None
    lambda_id: float = 1e-3
    n_dims: int = 2
    max_speed: float = 1.2
    dt: float = 0.02
    reference_step: float | None = None
    base_policy_type: str = "cursor_flow"
    base_policy_path: str | None = None
    scene_source: str | None = None
    render_size: int | None = None
    token_geometry: str | None = None
    scene_rank: int | None = None
    vlm_path: str | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in STEERING_MODES:
            raise ValueError(f"mode must be one of {STEERING_MODES}, got {self.mode!r}")
        if self.scene_source is not None and self.scene_source not in SCENE_SOURCES:
            raise ValueError(f"scene_source must be one of {SCENE_SOURCES}, got {self.scene_source!r}")
        if self.token_geometry is not None and self.token_geometry not in TOKEN_GEOMETRIES:
            raise ValueError(f"token_geometry must be one of {TOKEN_GEOMETRIES}, got {self.token_geometry!r}")
        if self.reference_step is not None and not self.reference_step > 0:
            raise ValueError(
                f"reference_step must be > 0 (action units per control step), got {self.reference_step!r}"
            )
        if self.n_dims < 1:
            raise ValueError(
                f"n_dims must be >= 1 (the action dims the adapter acts on; flow control writes them, and FRS "
                f"delegates the ones beyond the command), got {self.n_dims!r}"
            )
        self.check_operator_dims(self.operator_dims or ())
        self.check_operator_dims(self.zeroed_dims or ())
        if (
            self.adapter_matrix is not None
        ):  # a square of numbers, kept as lists so the config compares and serializes
            rows = len(self.adapter_matrix)
            self.adapter_matrix = readout.checked_matrix(self.adapter_matrix, rows, "adapter_matrix").tolist()

    @property
    def fixed_matrix(self) -> np.ndarray | None:
        """``adapter_matrix`` as a float64 array, or None."""
        return None if self.adapter_matrix is None else np.asarray(self.adapter_matrix, dtype=np.float64)

    def scene_settings(self, task: dict | None = None) -> dict[str, Any]:
        """The scene settings (``scene_source``, ``render_size``, ``token_geometry``, ``scene_rank``): this config's,
        each unset one (None) taken from the task module's constructor config ``task`` (``LearnedTaskModule.config``:
        ``token_source``, ``image_size``, ``token_geometry``, ``scene_rank``, absent keys being the module's
        defaults). Raises ``ValueError`` naming both values when an explicit setting contradicts the module's (a
        ``render_size`` is not checked against a module without an image size, a ``discs`` one)."""
        settings = {
            "scene_source": self.scene_source,
            "render_size": self.render_size,
            "token_geometry": self.token_geometry,
            "scene_rank": self.scene_rank,
        }
        if task is None:
            return settings
        module = {
            "scene_source": task.get("token_source", "discs"),
            "render_size": task.get("image_size"),
            "token_geometry": task.get("token_geometry", "absolute"),
            "scene_rank": task.get("scene_rank"),
        }
        for key, value in module.items():
            if settings[key] is None:
                settings[key] = value
            elif settings[key] != value and (value is not None or key != "render_size"):
                raise ValueError(
                    f"{key}: the config says {settings[key]!r}, the task module was built with {value!r}"
                )
        return settings

    @staticmethod
    def check_operator_dims(operator_dims) -> None:
        """Raise unless every name in ``operator_dims`` is one of ``OPERATOR_DIM_GROUPS``."""
        unknown = [d for d in operator_dims if d not in OPERATOR_DIM_GROUPS]
        if unknown:
            raise ValueError(
                f"operator_dims: unknown {unknown} (expected from {', '.join(OPERATOR_DIM_GROUPS)})"
            )

    # ---------------------------------------------------------------- the adapter spec

    @staticmethod
    def _spec_matrix(spec, n_dims: int, where: str) -> np.ndarray:
        """``readout.matrix_spec``'s matrix of a spec that must give one (a null spec raises)."""
        matrix = readout.matrix_spec(spec, n_dims, where)
        if matrix is None:
            raise ValueError(f"{where}: expected a matrix spec, got null")
        return matrix

    @classmethod
    def build_adapter(
        cls,
        spec,
        corruption: np.ndarray | None = None,
        where: str = "reversal_adapter",
        n_dims: int = N_ACTION_DIMS,
    ) -> np.ndarray | None:
        """Build the (n_dims, n_dims) reversal adapter F from a spec, or None when the spec is None.

        Forms: ``{F: <matrix spec>}`` (rows, or any ``readout.matrix_spec`` form for ``n_dims``), or blocks
        ``translation`` (``corruption`` to copy the command corruption, ``corruption_inverse`` for its inverse, the
        adapter that recovers a curl field's steady state on the cursor task, ``identity``, ``zero``, or a matrix
        spec), ``orientation`` and ``gripper`` (``identity`` or ``zero``). Missing blocks are identity. The blocks
        are ``OPERATOR_DIM_GROUPS``' dims of the 7-D operator space (``N_ACTION_DIMS``, the default) that fit in
        ``n_dims``: ``translation`` the first min(3, n_dims) dims, so with 2 dims (the cursor's) the only block, and
        a block with no dims there raises.
        """
        if spec is None:
            return None
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
        if not spec:
            raise ValueError(f"{where}: an empty mapping is no adapter spec (null means none)")
        if "F" in spec:
            if set(spec) != {"F"}:
                raise ValueError(f"{where}: an F cannot be combined with block keys")
            return cls._spec_matrix(spec["F"], n_dims, f"{where}.F")
        translation_dims = OPERATOR_DIM_GROUPS["translation"]
        blocks = {
            "translation": slice(translation_dims.start, min(translation_dims.stop, n_dims)),
            "orientation": OPERATOR_DIM_GROUPS["rotation"],
            "gripper": OPERATOR_DIM_GROUPS["gripper"],
        }
        unknown = set(spec) - set(blocks)
        if unknown:
            raise ValueError(
                f"{where}: unknown keys {sorted(unknown)} (expected translation, orientation, gripper)"
            )
        for key in spec:
            if blocks[key].stop > n_dims:
                raise ValueError(
                    f"{where}.{key}: a {n_dims}-D action has no {key} block "
                    f"(dims {blocks[key].start}-{blocks[key].stop - 1} of the {N_ACTION_DIMS}-D operator space)"
                )

        f = np.eye(n_dims)
        dims = blocks["translation"]
        translation = spec.get("translation", "identity")
        if translation in ("corruption", "corruption_inverse"):
            if corruption is None:
                raise ValueError(
                    f"{where}.translation: '{translation}' needs a corruption to copy or invert, but "
                    "control.corruption is null (or a curl field)"
                )
            f[dims, dims] = corruption if translation == "corruption" else np.linalg.inv(corruption)
        elif translation == "zero":
            f[dims, dims] = 0.0
        elif translation != "identity":
            f[dims, dims] = cls._spec_matrix(translation, dims.stop - dims.start, f"{where}.translation")
        for key in ("orientation", "gripper"):
            choice = spec.get(key, "identity")
            if choice not in _BLOCK_CHOICES:
                raise ValueError(f"{where}.{key}: expected identity or zero, got {choice!r}")
            if choice == "zero":
                f[blocks[key], blocks[key]] = 0.0
        return f

    @staticmethod
    def _module_path(value, where: str) -> str:
        if not isinstance(value, str | Path) or not str(value):
            raise ValueError(f"{where}: expected a checkpoint path, got {value!r}")
        return str(value)

    @classmethod
    def from_spec(
        cls,
        spec,
        corruption: np.ndarray | None = None,
        where: str = "reversal_adapter",
        n_dims: int | None = None,
        **fields: Any,
    ) -> "SteeringConfig":
        """A config whose adapter fields are a reversal-adapter spec's (``fields`` set the others).

        Forms: ``null``; a fixed user matrix as ``build_adapter`` takes it (blocks or ``{F: ...}``);
        ``{learned: <path>}`` (a learned user module); or the composite ``{user: <fixed spec | {learned: path} |
        path | null>, task: <path | null>}`` (MARS = user + task; one slot alone is MARS-U or MARS-T; a path as the
        user is ``{learned: path}``, the cursor's checkpoint names).

        ``n_dims``: None builds a fixed F at ``N_ACTION_DIMS`` (the 7-D operator space) and leaves the config's own
        ``n_dims``; a number builds F at that size and sets the config's ``n_dims`` to it (the cursor passes 2).
        """
        if n_dims is not None:
            fields["n_dims"] = n_dims
        size = N_ACTION_DIMS if n_dims is None else n_dims
        if spec is None:
            return cls(**fields)
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")

        def fixed(blocks, at):
            matrix = cls.build_adapter(blocks, corruption, at, size)
            return {"adapter_matrix": matrix.tolist(), "adapter_blocks": blocks}

        keys = set(spec)
        if keys == {"learned"}:
            return cls(adapter_user=cls._module_path(spec["learned"], f"{where}.learned"), **fields)
        if keys and keys <= {"user", "task"}:
            user, task = spec.get("user"), spec.get("task")
            task_path = None if task is None else cls._module_path(task, f"{where}.task")
            if user is None:
                return cls(adapter_task=task_path, **fields)
            if isinstance(user, str | Path):  # {learned: path}'s shorthand
                return cls(
                    adapter_user=cls._module_path(user, f"{where}.user"), adapter_task=task_path, **fields
                )
            if isinstance(user, dict) and "learned" in user:
                if set(user) != {"learned"}:
                    raise ValueError(f"{where}.user: a learned module cannot be combined with a fixed matrix")
                user_path = cls._module_path(user["learned"], f"{where}.user.learned")
                return cls(adapter_user=user_path, adapter_task=task_path, **fields)
            return cls(adapter_task=task_path, **fixed(user, f"{where}.user"), **fields)
        if keys & {"learned", "user", "task"}:
            raise ValueError(
                f"{where}: expected null, a fixed matrix spec, {{learned: path}} or {{user: ..., task: ...}}, got {sorted(keys)}"
            )
        return cls(**fixed(spec, where), **fields)

    # ---------------------------------------------------------------- lerobot's policy config contract

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
