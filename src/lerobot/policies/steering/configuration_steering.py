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

"""Settings of the steering wrappers and the MARS adapters: the end-effector action space the wrappers assume
(LIBERO's OSC_POSE: three translation, three rotation and one gripper dim) and the operator's view of it
(``TeleopSource``, ``DEADBAND``, ``OPERATOR_DIM_GROUPS``), the one check of the step knobs, the matrix specs of a
command corruption or a fixed reversal adapter, and the ``reversal_adapter`` specs (a fixed F and/or the paths of
learned modules). ``modeling_steering`` holds the methods; this folder's README describes them."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lerobot.teleoperators import readout

ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
N_ACTION_DIMS = len(ACTION_DIMS)
GRIPPER_DIM = ACTION_DIMS.index("gripper")
GRIPPER_OPEN = -1.0
# Groups of action dims an operator can command; anything else is the policy's.
OPERATOR_DIM_GROUPS = {
    "translation": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(GRIPPER_DIM, GRIPPER_DIM + 1),
}
GRIPPER_CLOSE = 1.0
DEADBAND = (
    readout.DEADBAND
)  # default `deadband` of the wrappers: below this (env units) the operator counts as idle


class TeleopSource(Protocol):
    """What the steering wrappers read: the operator's current command."""

    @property
    def translation(self) -> np.ndarray:  # shape (3,), env units in [-1, 1]
        ...

    @property
    def gripper(self) -> float:  # GRIPPER_OPEN or GRIPPER_CLOSE
        ...


def check_steps(name: str, value, low: int, high: int, unit: str, optional: bool = False) -> None:
    """Raise unless ``value`` is an integer (not a bool) in [``low``, ``high``] ``unit``, or None when
    ``optional``: the one check of the step knobs (tau, lambda, n_reversal_steps, pinned_steps)."""
    if optional and value is None:
        return
    if not (isinstance(value, int) and not isinstance(value, bool) and low <= value <= high):
        none = "None or " if optional else ""
        raise ValueError(f"{name} must be {none}an integer in [{low}, {high}] ({unit}), got {value!r}")


# ---------------------------------------------------------------- matrices


def validate_matrix(matrix, size: int, where: str) -> np.ndarray:
    """Coerce `matrix` to a finite (size, size) float array or raise ValueError."""
    try:
        m = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: entries must be numbers ({e})") from e
    if m.shape != (size, size):
        raise ValueError(f"{where}: expected a {size}x{size} matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{where}: entries must be finite")
    return m


def rotation_about_z(degrees: float) -> np.ndarray:
    """3x3 rotation of a translation command about the z axis."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def translation_matrix(spec, where: str = "matrix", literal_key: str = "M") -> np.ndarray:
    """Build a 3x3 from a spec: {rotation_z_deg: d}, {scale: [sx, sy, sz]}, {M: rows} or bare rows."""
    if isinstance(spec, dict):
        keys = set(spec)
        if keys == {"rotation_z_deg"}:
            return rotation_about_z(float(spec["rotation_z_deg"]))
        if keys == {"scale"}:
            scale = np.asarray(spec["scale"], dtype=np.float64)
            if scale.shape != (3,) or not np.all(np.isfinite(scale)):
                raise ValueError(f"{where}: scale must be three finite numbers, got {spec['scale']!r}")
            return np.diag(scale)
        if keys == {literal_key}:
            return validate_matrix(spec[literal_key], 3, f"{where}.{literal_key}")
        raise ValueError(
            f"{where}: expected exactly one of rotation_z_deg, scale or {literal_key}, got {sorted(keys)}"
        )
    return validate_matrix(spec, 3, where)


_BLOCK_CHOICES = ("identity", "zero")


def build_reversal_adapter(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> np.ndarray | None:
    """Build the 7x7 reversal adapter F from a spec, or None when the spec is None.

    Forms: ``{F: rows}`` (literal), or blocks ``translation`` (``corruption`` to
    copy the 3x3 command corruption, ``corruption_inverse`` for its inverse, the
    adapter that recovers a curl field's steady state on the cursor task,
    ``identity``, ``zero``, or any ``translation_matrix`` spec), ``orientation``
    and ``gripper`` (``identity`` or ``zero``). Missing blocks are identity.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
    if "F" in spec:
        if set(spec) != {"F"}:
            raise ValueError(f"{where}: a literal F cannot be combined with block keys")
        return validate_matrix(spec["F"], N_ACTION_DIMS, f"{where}.F")
    unknown = set(spec) - {"translation", "orientation", "gripper"}
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)} (expected translation, orientation, gripper)"
        )

    f = np.eye(N_ACTION_DIMS)
    translation = spec.get("translation", "identity")
    if translation in ("corruption", "corruption_inverse"):
        if corruption is None:
            raise ValueError(f"{where}.translation: '{translation}' requested but control.corruption is null")
        f[:3, :3] = corruption if translation == "corruption" else np.linalg.inv(corruption)
    elif translation == "zero":
        f[:3, :3] = 0.0
    elif translation != "identity":
        f[:3, :3] = translation_matrix(translation, f"{where}.translation")
    for key, dims in (
        ("orientation", OPERATOR_DIM_GROUPS["rotation"]),
        ("gripper", OPERATOR_DIM_GROUPS["gripper"]),
    ):
        choice = spec.get(key, "identity")
        if choice not in _BLOCK_CHOICES:
            raise ValueError(f"{where}.{key}: expected identity or zero, got {choice!r}")
        if choice == "zero":
            f[dims, dims] = 0.0
    return f


def format_matrix(matrix: np.ndarray) -> str:
    """Rows of a small matrix on one line, for the terminal: `[+0.94 -0.34 +0.00; ...]`."""
    return "[" + "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in matrix) + "]"


class MatrixHolder:
    """Mutable holder for an optional square matrix (None = off) with a display label.

    Consumers read the holder on every use, so replacing ``matrix`` takes effect
    immediately; the setter validates the shape.
    """

    size: int
    where: str

    def __init__(self, matrix=None, label: str | None = None):
        self.matrix = matrix
        self.label = label

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix = None if value is None else validate_matrix(value, self.size, self.where)


# ---------------------------------------------------------------- reversal-adapter specs


@dataclass(frozen=True)
class AdapterSpec:
    """A parsed ``reversal_adapter`` spec: a fixed user matrix F (environment space) and/or the paths
    of learned modules. The session loads the modules; the matrix is rebuilt per trial when the
    corruption is random."""

    matrix: np.ndarray | None = None
    user_path: str | None = None
    task_path: str | None = None

    def __eq__(self, other):
        if not isinstance(other, AdapterSpec):
            return NotImplemented
        same_matrix = (self.matrix is None and other.matrix is None) or (
            self.matrix is not None and other.matrix is not None and np.array_equal(self.matrix, other.matrix)
        )
        return same_matrix and (self.user_path, self.task_path) == (other.user_path, other.task_path)


def _module_path(value, where: str) -> str:
    if not isinstance(value, str | Path) or not str(value):
        raise ValueError(f"{where}: expected a checkpoint path, got {value!r}")
    return str(value)


def parse_adapter_spec(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> AdapterSpec:
    """Parse a reversal-adapter spec into fixed matrix / module paths.

    Forms: ``null``; a fixed user matrix as ``build_reversal_adapter`` takes it (blocks or ``{F: rows}``);
    ``{learned: <path>}`` (a learned user module); or the composite ``{user: <fixed spec | {learned: path}
    | null>, task: <path | null>}`` (MARS = user + task; one slot alone is MARS-U or MARS-T).
    """
    if spec is None:
        return AdapterSpec()
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(spec).__name__}")
    keys = set(spec)
    if keys == {"learned"}:
        return AdapterSpec(user_path=_module_path(spec["learned"], f"{where}.learned"))
    if keys and keys <= {"user", "task"}:
        user, task = spec.get("user"), spec.get("task")
        task_path = None if task is None else _module_path(task, f"{where}.task")
        if user is None:
            return AdapterSpec(task_path=task_path)
        if isinstance(user, dict) and "learned" in user:
            if set(user) != {"learned"}:
                raise ValueError(f"{where}.user: a learned module cannot be combined with a fixed matrix")
            return AdapterSpec(
                user_path=_module_path(user["learned"], f"{where}.user.learned"), task_path=task_path
            )
        return AdapterSpec(
            matrix=build_reversal_adapter(user, corruption, f"{where}.user"), task_path=task_path
        )
    if keys & {"learned", "user", "task"}:
        raise ValueError(
            f"{where}: expected null, a fixed matrix spec, {{learned: path}} or {{user: ..., task: ...}}, got {sorted(keys)}"
        )
    return AdapterSpec(matrix=build_reversal_adapter(spec, corruption, where))
