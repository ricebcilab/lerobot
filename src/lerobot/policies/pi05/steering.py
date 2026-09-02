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

"""Shared-autonomy steering of pi0.5's flow-matching sampler.

Two ways for an operator's translation command to steer the policy without
touching the executed action, which is always the model's own output:

- ``FlowControlPolicy``: the command is written into dims 0-2 of ``x_t`` for the
  first ``tau`` denoising steps of each chunk (the rest denoise freely).
- ``ReverseFlowSteeringPolicy``: a uniform-velocity reference chunk built from
  the command is integrated *backward* through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator ``TeleopSource`` and the checkpoint's action statistics
from the postprocessor. ``reverse_flow`` is the integrator; ``ReversalAdapter``
holds an optional 7x7 matrix F that adapts the reversal's velocity field only.
"""

from collections import deque
from typing import Protocol

import numpy as np
import torch

# LIBERO's OSC_POSE action: end-effector position deltas, axis-angle deltas, gripper.
ACTION_DIMS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
N_ACTION_DIMS = len(ACTION_DIMS)
GRIPPER_DIM = ACTION_DIMS.index("gripper")
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0
DEADBAND = 0.05  # below this (env units, full deflection = 1) the operator counts as idle


class TeleopSource(Protocol):
    """What the steering wrappers read: the operator's current command."""

    @property
    def translation(self) -> np.ndarray:  # shape (3,), env units in [-1, 1]
        ...

    @property
    def gripper(self) -> float:  # GRIPPER_OPEN or GRIPPER_CLOSE
        ...


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
_ORIENTATION = slice(3, 6)
_GRIPPER = slice(GRIPPER_DIM, GRIPPER_DIM + 1)


def build_reversal_adapter(
    spec, corruption: np.ndarray | None = None, where: str = "reversal_adapter"
) -> np.ndarray | None:
    """Build the 7x7 reversal adapter F from a spec, or None when the spec is None.

    Forms: ``{F: rows}`` (literal), or blocks ``translation`` (``corruption`` to
    copy the 3x3 command corruption, ``identity``, ``zero``, or any
    ``translation_matrix`` spec), ``orientation`` and ``gripper`` (``identity``
    or ``zero``). Missing blocks are identity.
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
    if translation == "corruption":
        if corruption is None:
            raise ValueError(f"{where}.translation: 'corruption' requested but control.corruption is null")
        f[:3, :3] = corruption
    elif translation == "zero":
        f[:3, :3] = 0.0
    elif translation != "identity":
        f[:3, :3] = translation_matrix(translation, f"{where}.translation")
    for key, dims in (("orientation", _ORIENTATION), ("gripper", _GRIPPER)):
        choice = spec.get(key, "identity")
        if choice not in _BLOCK_CHOICES:
            raise ValueError(f"{where}.{key}: expected identity or zero, got {choice!r}")
        if choice == "zero":
            f[dims, dims] = 0.0
    return f


class ReversalAdapter:
    """Mutable holder for the reversal adapter F (None = off).

    ``ReverseFlowSteeringPolicy`` reads the holder on every chunk, so replacing
    ``matrix`` takes effect immediately. ``label`` is display-only.
    """

    def __init__(self, matrix=None, label: str | None = None):
        self.matrix = matrix
        self.label = label

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix = None if value is None else validate_matrix(value, N_ACTION_DIMS, "reversal adapter")

    def describe(self) -> str:
        if self._matrix is None:
            return "Reversal adapter: off."
        identity = " (identity, a no-op)" if np.allclose(self._matrix, np.eye(N_ACTION_DIMS)) else ""
        rows = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in self._matrix)
        return f"Reversal adapter: reverse integration uses x_t += h * F @ v, F = {self.label}{identity} = [{rows}]."


# ---------------------------------------------------------------- FlowControlPolicy


def get_action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension action mean/std from the checkpoint's action unnormalizer stats."""
    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats and "action" in stats:
            mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
            std = np.maximum(np.asarray(stats["action"]["std"], dtype=np.float64), 1e-6)
            return mean, std
    raise RuntimeError("No action stats in the postprocessor; the shared flow modes need them.")


class FlowControlPolicy:
    """pi0.5 wrapper implementing shared_flow_control.

    For the first `tau` of the chunk's denoising steps, the teleop
    translation (SpaceMouse or keyboard, normalized into the model's action
    space) is written into dims 0-2 of x_t across the whole chunk; the
    remaining steps denoise freely. The executed action is entirely the
    model's output — the operator steers only the early flow. Guidance is
    skipped while the input is inside DEADBAND, so idle input means pure
    policy.
    """

    def __init__(self, policy, source: TeleopSource, tau: int, postprocessor):
        self._policy = policy
        self._source = source
        self.tau = tau
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:3], std[:3]
        self._queue: deque = deque()
        self.hook_calls = 0  # per-rollout count of guided denoising steps

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self.hook_calls = 0

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook reads live teleop input and must stay eager.
        if step >= self.tau:
            return x_t
        cmd = self._source.translation
        if np.max(np.abs(cmd)) < DEADBAND:
            return x_t
        target = (cmd - self._mean) / self._std
        x_t[..., :3] = torch.as_tensor(target, dtype=x_t.dtype, device=x_t.device)
        self.hook_calls += 1
        return x_t

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the guidance hook.
        if len(self._queue) == 0:
            chunk = self._policy.predict_action_chunk(batch, x_t_hook=self._x_t_hook)
            actions = chunk[:, : self._policy.config.n_action_steps]
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()


def reverse_flow(
    x: torch.Tensor,
    velocity,
    num_steps: int,
    adapter: torch.Tensor | None = None,
    n_reverse_steps: int | None = None,
) -> torch.Tensor:
    """Integrate the flow backward (Euler, t: 0 -> 1) from a clean chunk toward its latent noise.

    Mirrors PI05Pytorch.sample_actions, which integrates x_{t+dt} = x_t + dt * v(x_t, t)
    with dt = -1/num_steps from t=1 (noise) to t=0 (action); here the same field is
    stepped with dt = +1/num_steps starting at t=0. This is the inversion of Flow
    Reversal Steering (Tang et al. 2026, arXiv:2606.13675).

    The step size is always 1/`num_steps`, i.e. the policy's own schedule.
    `n_reverse_steps` is how many of those steps to take: the default (None) runs
    the whole way to t=1, pure noise, while a smaller count stops early and
    returns a chunk at t = n_reverse_steps / num_steps, still carrying some of
    the reference. The caller must then start the forward flow at that time.

    `adapter` is an optional (n, n) matrix F that adapts the velocity field of the
    reversal, x_t += h * (F @ v): it is applied to the first n action dimensions of
    every velocity evaluation (the env's 7 here), leaving the padding dimensions
    alone. F = I is an exact no-op. Only the reversal is adapted; the forward flow
    that produces the executed action still uses the policy's own field.
    """
    h = 1.0 / num_steps
    steps = num_steps if n_reverse_steps is None else n_reverse_steps
    x_t = x
    for step in range(steps):
        v = velocity(x_t, step * h)
        if adapter is not None:
            n = adapter.shape[0]
            v = torch.cat([v[..., :n] @ adapter.transpose(-1, -2), v[..., n:]], dim=-1)
        x_t = x_t + h * v
    return x_t


class ReverseFlowSteeringPolicy:
    """pi0.5 wrapper implementing shared_reverse_flow_steering (Flow Reversal Steering).

    While the teleop input is deflected, a reference chunk that servos in the
    commanded direction at uniform velocity (rotation zero, gripper held at the
    model's last executed command) is integrated *backward* through the
    policy's own velocity field for num_inference_steps to find the latent
    noise that maps to it; the normal forward flow then denoises from that
    noise instead of a random one. The executed action is entirely the
    model's output — the reference only picks the starting noise, so what
    comes out is the generalist action mode nearest your intent. Idle input
    = pure policy (random noise, as usual).

    An optional `ReversalAdapter` holding a 7x7 matrix F adapts the velocity field
    of the reversal only (x_t += h * F @ v); see `reverse_flow`.

    `n_reverse_steps` stops the reversal early instead of going all the way to
    noise: with n of the policy's N = num_inference_steps steps the reference is
    only partially destroyed, landing at t = n/N, and the forward flow then runs
    from there in N - n steps (so a steered chunk costs the same N velocity
    evaluations as an unsteered one). Smaller n keeps more of the operator's
    reference in the result, larger n leaves the policy more freedom; n = N (or
    None) is the full reversal.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource,
        postprocessor,
        adapter: ReversalAdapter | None = None,
        n_reverse_steps: int | None = None,
    ):
        self._policy = policy
        self._source = source
        self._adapter = adapter
        self.n_reverse_steps = n_reverse_steps  # validated by the setter below
        self._pending_cmd: np.ndarray | None = None
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]
        self._queue: deque = deque()
        self._gripper_ref = self._normalize_gripper(GRIPPER_OPEN)
        self._last_reference: torch.Tensor | None = None
        self.steered_chunks = 0  # per-rollout count of chunks started from inverted noise
        self.reconstruction_errors: list[float] = []  # per steered chunk, in std units

    @property
    def n_reverse_steps(self) -> int | None:
        return self._n_reverse_steps

    @n_reverse_steps.setter
    def n_reverse_steps(self, value: int | None) -> None:
        total = self._policy.config.num_inference_steps
        if value is not None and not (isinstance(value, int) and 1 <= value <= total):
            raise ValueError(f"n_reverse_steps must be an integer in [1, {total}], got {value!r}")
        self._n_reverse_steps = None if value == total else value

    def _flow_kwargs(self) -> dict:
        """Where the forward flow starts, once the reference has been reversed part-way."""
        if self._n_reverse_steps is None:
            return {}  # full reversal: the usual schedule from t=1 in N steps
        total = self._policy.config.num_inference_steps
        return {
            "flow_start_time": self._n_reverse_steps / total,
            "num_forward_steps": total - self._n_reverse_steps,
        }

    def _normalize_gripper(self, gripper: float) -> float:
        return float((gripper - self._mean[GRIPPER_DIM]) / self._std[GRIPPER_DIM])

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self._gripper_ref = self._normalize_gripper(GRIPPER_OPEN)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, cmd: np.ndarray) -> torch.Tensor:
        """Uniform-velocity chunk for env-space translation `cmd`, in the model's normalized space."""
        env_action = np.zeros(N_ACTION_DIMS)
        env_action[:3] = cmd
        normalized = (env_action - self._mean) / self._std
        normalized[GRIPPER_DIM] = self._gripper_ref
        cfg = self._policy.config
        ref = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32)
        ref[..., : len(normalized)] = torch.as_tensor(normalized, dtype=torch.float32)
        return ref

    @torch.compiler.disable
    def _noise_fn(self, velocity, noise: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # this builds the reference from live teleop input and must stay eager.
        # `select_action` has already sampled the input (once per chunk) and put
        # it in _pending_cmd, so the forward schedule it chose matches this hook.
        cmd = self._pending_cmd
        if cmd is None:
            self._last_reference = None
            return noise
        reference = self.reference_chunk(cmd).to(device=noise.device, dtype=noise.dtype)
        self._last_reference = reference
        self.steered_chunks += 1
        adapter_matrix = (
            None if self._adapter is None or self._adapter.matrix is None else self._adapter.matrix
        )
        if adapter_matrix is not None:
            adapter_matrix = torch.as_tensor(adapter_matrix, dtype=noise.dtype, device=noise.device)
        return reverse_flow(
            reference,
            velocity,
            self._policy.config.num_inference_steps,
            adapter=adapter_matrix,
            n_reverse_steps=self._n_reverse_steps,
        )

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the noise hook.
        if len(self._queue) == 0:
            # Sample the operator once per chunk: the forward schedule depends on
            # whether we are steering, so it cannot be decided inside the hook.
            cmd = self._source.translation
            steering = np.max(np.abs(cmd)) >= DEADBAND
            self._pending_cmd = cmd if steering else None
            flow_kwargs = self._flow_kwargs() if steering else {}
            chunk = self._policy.predict_action_chunk(batch, noise_fn=self._noise_fn, **flow_kwargs)
            actions = chunk[:, : self._policy.config.n_action_steps]
            self._gripper_ref = float(actions[0, -1, GRIPPER_DIM])
            if self._last_reference is not None:
                executed = actions[0, :, :3].float().cpu()
                target = self._last_reference[0, : actions.shape[1], :3].float().cpu()
                self.reconstruction_errors.append((executed - target).abs().mean().item())
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()
