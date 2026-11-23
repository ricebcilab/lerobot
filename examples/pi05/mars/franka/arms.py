#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""What an arm adds on top of the base config: the corruption, and the MARS `mars_fixed` adapter.

The corruption acts on the Cartesian twist the operator pushes with, and the fixed adapter is that corruption
expressed on the joint velocities the command is built from. This module is the algebra between the two,
numpy only, so a start pose and a Jacobian can be checked before an arm moves.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from config import ExperimentConfig
from interfaces import JacobianPort

__all__ = [
    "block_corruption",
    "corruption_matrix",
    "fixed_adapter_matrix",
    "_with_fixed_adapter",
]


def block_corruption(corruption: np.ndarray, twist_dims: int = 6) -> np.ndarray:
    """A three-axis command corruption as the twist-space operator `M = [M3, 0; 0, I3]`.

    The operator's stick pushes three translation axes, and the corruption `M3` acts there; the flange's
    rotation is not commanded by this example, so its block stays the identity. `twist_dims=2` is the toy
    Jacobian's, where the corruption is already the whole operator.
    """
    matrix = np.asarray(corruption, dtype=np.float64)
    if matrix.shape == (twist_dims, twist_dims):
        return matrix
    if twist_dims != 6 or matrix.shape != (3, 3):
        raise ValueError(f"a corruption must be ({twist_dims}, {twist_dims}) here, got {matrix.shape}")
    block = np.eye(6)
    block[:3, :3] = matrix
    return block


def fixed_adapter_matrix(jacobian: JacobianPort, q_start: np.ndarray, corruption: np.ndarray) -> np.ndarray:
    """The MARS fixed adapter `F = J+ M J` at `q_start`, in the joint-velocity space the command uses.

    The corruption acts on the Cartesian twist, not on the joints: a push `v` arrives at the arm as `M v`
    (three translation axes corrupted, three rotation axes left alone). The Jacobian carries a joint velocity
    to a twist, `v = J qdot`, so the same corruption expressed on joint velocities is `J+ M J`: map the joint
    command to a twist, corrupt it, and solve back for the joint command with the damped least-squares inverse
    the arm's Jacobian uses. That inverse is read from the port's `joint_velocity`, so a singular start pose
    degrades the way the arm's own solve does rather than by a second, different regularisation.

    `corruption` is the twist-space operator: a 3x3 command corruption is embedded as `[M3, 0; 0, I3]` when
    the Jacobian maps to six twist rows, and a square operator of the twist width is taken as given (the
    two-joint toy's test). `J` is assembled by driving each joint with a unit velocity and reading the twist,
    which is the port's definition of the Jacobian.
    """
    q = np.asarray(q_start, dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError(f"q_start must be a non-empty joint vector, got {q.shape}")
    joints = np.eye(q.size)
    twist_dims = int(np.asarray(jacobian.twist(q, joints[0])).size)
    matrix = block_corruption(corruption, twist_dims)
    jac = np.column_stack([np.asarray(jacobian.twist(q, joint), dtype=np.float64) for joint in joints])
    inverse = np.column_stack(
        [np.asarray(jacobian.joint_velocity(q, unit), dtype=np.float64) for unit in np.eye(twist_dims)]
    )
    return inverse @ matrix @ jac


def corruption_matrix(spec: Any) -> np.ndarray:
    """The 3x3 matrix a fixed adapter inverts, parsed by the readout's shared grammar.

    A null corruption is the identity: there is nothing to reverse, so `mars_fixed` reduces to the projection
    `J+ J` and stays a valid configuration. A curl field or a quantizer has no matrix and is refused, because
    the fixed adapter is built from a matrix, not from a readout.
    """
    if spec is None:
        return np.eye(3)
    from lerobot.teleoperators import readout

    if readout.quantize_spec(spec):
        raise ValueError(
            "steering.mode 'mars_fixed' needs a matrix corruption, but readout.corruption is a quantizer"
        )
    matrix, _curl = readout.corruption_spec(spec, n_dims=3)
    if matrix is None:
        raise ValueError(
            "steering.mode 'mars_fixed' needs a matrix corruption, but readout.corruption is a curl field"
        )
    return np.asarray(matrix, dtype=np.float64)


def _with_fixed_adapter(
    config: ExperimentConfig, jacobian: JacobianPort, q_start: np.ndarray
) -> ExperimentConfig:
    """The config a `mars_fixed` run uses, with its null matrix filled from the Jacobian at the start pose."""
    if config.steering.adapter_matrix is not None:
        return config
    matrix = fixed_adapter_matrix(jacobian, q_start, corruption_matrix(config.readout.corruption))
    return replace(
        config,
        steering=replace(config.steering, adapter_matrix=[[float(value) for value in row] for row in matrix]),
    )


def _matrix_list(spec: Any) -> list[list[float]] | None:
    """A readout matrix spec as rows, or None when it is absent or has no matrix (a curl, a quantizer)."""
    if spec is None:
        return None
    try:
        return corruption_matrix(spec).tolist()
    except ValueError:
        return None


def _adapter_label(config: ExperimentConfig) -> str | None:
    """The label LIBERO's `method_of` reads to tell MARS from plain FRS; None for every other arm."""
    if config.steering.mode == "mars_fixed":
        return "J+ M J"
    if config.steering.mode == "mars_learned":
        return config.steering.adapter_user
    return None
