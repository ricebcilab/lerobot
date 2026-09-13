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
  first ``n_guided_steps`` denoising steps of each chunk (the rest denoise freely).
- ``FlowReversalSteeringPolicy``: a uniform-velocity reference chunk built from
  the command is integrated *backward* through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator ``TeleopSource`` and the checkpoint's action statistics
from the postprocessor. ``reverse_flow`` is the integrator; ``ReversalAdapter``
holds an optional 7x7 matrix F that adapts the reversal's velocity field only.

Elements of the reference the operator does not command (``operator_dims``
selects which of translation / rotation / gripper they do), the padding dims and
the steps beyond ``pinned_steps`` are *delegated* to the policy: after the
reversal their latent is replaced by the unsteered one (fresh Gaussian noise for
a full reversal), the noise-space in-painting of Tang et al., App. D.
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
# Groups of action dims an operator can command; anything else is the policy's.
OPERATOR_DIM_GROUPS = {
    "translation": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(GRIPPER_DIM, GRIPPER_DIM + 1),
}
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


class ReversalAdapter(MatrixHolder):
    """The environment-space reversal adapter F of ``FlowReversalSteeringPolicy`` (None = off)."""

    size = N_ACTION_DIMS
    where = "reversal adapter"

    def describe(self) -> str:
        if self._matrix is None:
            return "Reversal adapter: off."
        identity = " (identity, a no-op)" if np.allclose(self._matrix, np.eye(N_ACTION_DIMS)) else ""
        return (
            f"Reversal adapter: environment-space F = {self.label}{identity} = {format_matrix(self._matrix)}."
        )

    def normalized_matrix(self, std: np.ndarray) -> np.ndarray | None:
        """Convert an environment-space velocity transform to normalized coordinates: S^-1 F S.

        Velocities have no mean offset. Apply this to all seven dimensions so literal
        matrices that mix translation, orientation and gripper use the same convention.
        """
        if self.matrix is None:
            return None
        return self.matrix * std[np.newaxis, :] / std[:, np.newaxis]


def _rng_state() -> tuple:
    """CPU and CUDA generator states, so a second sampling pass can reuse the first one's noise."""
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return torch.get_rng_state(), cuda


def _set_rng_state(state: tuple) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def paired_chunks(policy, batch, steered: bool, **steer_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """The steered chunk and the policy's own plan for the same observation and the same noise.

    Returns (steered, plain). When `steered` is False the two are the same tensor: no
    second pass. Otherwise the generator state is restored between the passes so the
    plain chunk starts from the Gaussian sample the steered one used, and the state
    after the steered pass is restored afterwards, so a run's random stream is exactly
    what it would be without the second pass.
    """
    if not steered:
        chunk = policy.predict_action_chunk(batch)
        return chunk, chunk
    before = _rng_state()
    chunk = policy.predict_action_chunk(batch, **steer_kwargs)
    after = _rng_state()
    _set_rng_state(before)
    plain = policy.predict_action_chunk(batch)
    _set_rng_state(after)
    return chunk, plain


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

    For the first `n_guided_steps` of the chunk's denoising steps, the teleop
    translation (SpaceMouse or keyboard, normalized into the model's action
    space) is written into dims 0-2 of x_t across the whole chunk; the
    remaining steps denoise freely. The executed action is entirely the
    model's output — the operator steers only the early flow. Guidance is
    skipped while the input is inside DEADBAND, so idle input means pure
    policy.
    """

    def __init__(self, policy, source: TeleopSource, n_guided_steps: int, postprocessor):
        self._policy = policy
        self._source = source
        self.n_guided_steps = n_guided_steps
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:3], std[:3]
        self._queue: deque = deque()
        self.guided_steps = 0  # per-rollout count of guided denoising steps
        self._pending_command: np.ndarray | None = None
        self._pending_target: torch.Tensor | None = None
        self._policy_queue: deque = deque()
        self.last_policy_action: torch.Tensor | None = (
            None  # the policy's own plan for the step just returned
        )

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self._policy_queue.clear()
        self.guided_steps = 0
        self._pending_command = None

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook uses the current chunk's teleop command and must stay eager.
        if step >= self.n_guided_steps or self._pending_command is None:
            return x_t
        if self._pending_target is None:
            target = (self._pending_command - self._mean) / self._std
            self._pending_target = torch.as_tensor(target, dtype=x_t.dtype, device=x_t.device)
        x_t[..., :3] = self._pending_target
        self.guided_steps += 1
        return x_t

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the guidance hook.
        if len(self._queue) == 0:
            # Hold one consumed command (including its noise) for the whole chunk,
            # matching FRS and the synthetic operator's refresh cadence.
            command = self._source.translation if self.n_guided_steps else None
            self._pending_command = (
                command if command is not None and np.max(np.abs(command)) >= DEADBAND else None
            )
            self._pending_target = None  # normalized on the device by the first guided step
            chunk, plain = paired_chunks(
                self._policy, batch, self._pending_command is not None, x_t_hook=self._x_t_hook
            )
            n = self._policy.config.n_action_steps
            actions = list(chunk[:, :n].transpose(0, 1))
            self._queue.extend(actions)
            self._policy_queue.extend(actions if plain is chunk else plain[:, :n].transpose(0, 1))
        self.last_policy_action = self._policy_queue.popleft()
        return self._queue.popleft()


def reverse_flow(
    x: torch.Tensor,
    velocity,
    num_steps: int,
    adapter: torch.Tensor | None = None,
    n_reversal_steps: int | None = None,
) -> torch.Tensor:
    """Integrate the flow backward (Euler, t: 0 -> 1) from a clean chunk toward its latent noise.

    Mirrors PI05Pytorch.sample_actions, which integrates x_{t+dt} = x_t + dt * v(x_t, t)
    with dt = -1/num_steps from t=1 (noise) to t=0 (action); here the same field is
    stepped with dt = +1/num_steps starting at t=0. This is the inversion of Flow
    Reversal Steering (Tang et al. 2026, arXiv:2606.13675).

    The step size is always 1/`num_steps`, i.e. the policy's own schedule.
    `n_reversal_steps` is how many of those steps to take: the default (None) runs
    the whole way to t=1, pure noise, while a smaller count stops early and
    returns a chunk at t = n_reversal_steps / num_steps, still carrying some of
    the reference. The caller must then start the forward flow at that time.

    `adapter` is an optional (n, n) matrix F, already in normalized coordinates,
    that adapts the reversal's velocity field, x_t += h * (F @ v): it is applied
    to the first n action dimensions of
    every velocity evaluation (the env's 7 here), leaving the padding dimensions
    alone. F = I is an exact no-op. Only the reversal is adapted; the forward flow
    that produces the executed action still uses the policy's own field.
    """
    h = 1.0 / num_steps
    steps = num_steps if n_reversal_steps is None else n_reversal_steps
    x_t = x
    for step in range(steps):
        v = velocity(x_t, step * h)
        if adapter is not None:
            n = adapter.shape[0]
            v = torch.cat([v[..., :n] @ adapter.transpose(-1, -2), v[..., n:]], dim=-1)
        x_t = x_t + h * v
    return x_t


def forward_flow(x: torch.Tensor, velocity, num_steps: int, stop_at: int | None = None) -> torch.Tensor:
    """Integrate the flow forward (Euler) from noise at t=1 down to t = stop_at / num_steps.

    Mirrors PI05Pytorch.sample_actions without hooks: x_{t-h} = x_t - h * v(x_t, t). With
    ``stop_at`` None it runs all the way to t=0 (the unsteered action chunk); with
    ``stop_at = n`` it takes ``num_steps - n`` steps and returns the unsteered state at
    t = n / num_steps, i.e. where a reversal stopped after ``n`` steps would sit.
    """
    h = 1.0 / num_steps
    steps = num_steps if stop_at is None else num_steps - stop_at
    x_t = x
    for step in range(steps):
        x_t = x_t - h * velocity(x_t, 1.0 - step * h)
    return x_t


def delegation_mask(
    chunk_size: int, max_action_dim: int, operator_dims: tuple[str, ...], pinned_steps: int | None
) -> torch.Tensor:
    """(chunk_size, max_action_dim) bool mask of the reference elements left to the policy.

    True = delegated: the dims outside ``operator_dims`` (names from OPERATOR_DIM_GROUPS),
    the padding dims beyond the env's 7, and every dim of the steps from ``pinned_steps``
    on (None = the whole chunk stays pinned).
    """
    unknown = [d for d in operator_dims if d not in OPERATOR_DIM_GROUPS]
    if unknown:
        raise ValueError(f"operator_dims: unknown {unknown} (expected from {', '.join(OPERATOR_DIM_GROUPS)})")
    mask = torch.ones(chunk_size, max_action_dim, dtype=torch.bool)
    for name in operator_dims:
        mask[:, OPERATOR_DIM_GROUPS[name]] = False
    if pinned_steps is not None:
        mask[pinned_steps:, :] = True
    return mask


class FlowReversalSteeringPolicy:
    """pi0.5 wrapper implementing shared_flow_reversal_steering (Flow Reversal Steering).

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

    `n_reversal_steps` stops the reversal early instead of going all the way to
    noise: with n of the policy's N = num_inference_steps steps the reference is
    integrated up to t = n/N, and the forward flow then runs
    from there in n steps, maintaining the policy's step size in both directions.
    With delegation, a steered chunk costs N + n velocity evaluations: n reverse,
    N - n to obtain the unsteered state, and n forward. Depth changes how far
    inversion travels before delegation and reconstruction; its effect on
    intent preservation is empirical. n = N (or None) is the full reversal.

    `operator_dims` names the dim groups the operator commands (default: only
    translation) and `pinned_steps` how many leading steps of the chunk the
    reference constrains (default: the policy's n_action_steps, i.e. the executed
    prefix; None = all of it). Everything else is delegated to the policy: after
    the reversal, those latent elements are replaced with the unsteered latent at
    the same time -- the chunk's own Gaussian sample for a full reversal, else the
    unsteered flow integrated down to t = n/N (N - n extra velocity evaluations).
    """

    def __init__(
        self,
        policy,
        source: TeleopSource,
        postprocessor,
        adapter: ReversalAdapter | None = None,
        n_reversal_steps: int | None = None,
        operator_dims: tuple[str, ...] = ("translation",),
        pinned_steps: int | None = -1,
    ):
        self._policy = policy
        self._source = source
        self._adapter = adapter
        self.n_reversal_steps = n_reversal_steps  # validated by the setter below
        self.pinned_steps = policy.config.n_action_steps if pinned_steps == -1 else pinned_steps
        self.operator_dims = operator_dims  # the setter builds the delegation mask
        self._pending_command: np.ndarray | None = None
        self._adapter_cache: tuple | None = None
        self._policy_queue: deque = deque()
        self.last_policy_action: torch.Tensor | None = (
            None  # the policy's own plan for the step just returned
        )
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:N_ACTION_DIMS], std[:N_ACTION_DIMS]
        self._queue: deque = deque()
        self._gripper_ref = self._normalize_gripper(GRIPPER_OPEN)
        self._last_reference: torch.Tensor | None = None
        self.steered_chunks = 0  # per-rollout count of chunks started from inverted noise
        self.reconstruction_errors: list[float] = []  # per steered chunk, in std units

    @property
    def operator_dims(self) -> tuple[str, ...]:
        return self._operator_dims

    @operator_dims.setter
    def operator_dims(self, value) -> None:
        cfg = self._policy.config
        dims = tuple(value)
        self._delegated = delegation_mask(cfg.chunk_size, cfg.max_action_dim, dims, self.pinned_steps)
        self._operator_dims = dims

    @property
    def n_reversal_steps(self) -> int | None:
        return self._n_reversal_steps

    @n_reversal_steps.setter
    def n_reversal_steps(self, value: int | None) -> None:
        total = self._policy.config.num_inference_steps
        if value is not None and not (isinstance(value, int) and 1 <= value <= total):
            raise ValueError(f"n_reversal_steps must be an integer in [1, {total}], got {value!r}")
        self._n_reversal_steps = None if value == total else value

    def _flow_kwargs(self) -> dict:
        """Where the forward flow starts, once the reference has been reversed part-way."""
        if self._n_reversal_steps is None:
            return {}  # full reversal: the usual schedule from t=1 in N steps
        total = self._policy.config.num_inference_steps
        return {
            "flow_start_time": self._n_reversal_steps / total,
            "num_forward_steps": self._n_reversal_steps,
        }

    def _normalize_gripper(self, gripper: float) -> float:
        return float((gripper - self._mean[GRIPPER_DIM]) / self._std[GRIPPER_DIM])

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self._policy_queue.clear()
        self._gripper_ref = self._normalize_gripper(GRIPPER_OPEN)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, command: np.ndarray) -> torch.Tensor:
        """Uniform-velocity chunk for env-space translation `command`, in the model's normalized space."""
        env_action = np.zeros(N_ACTION_DIMS)
        env_action[:3] = command
        normalized = (env_action - self._mean) / self._std
        normalized[GRIPPER_DIM] = self._gripper_ref
        cfg = self._policy.config
        ref = torch.zeros(1, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32)
        ref[..., : len(normalized)] = torch.as_tensor(normalized, dtype=torch.float32)
        return ref

    def _adapter_tensor(self, like: torch.Tensor) -> torch.Tensor | None:
        """The adapter in normalized coordinates on `like`'s device, rebuilt only when the holder's matrix changes."""
        matrix = None if self._adapter is None else self._adapter.matrix
        if matrix is None:
            self._adapter_cache = None
            return None
        key = (id(matrix), like.device, like.dtype)
        if self._adapter_cache is None or self._adapter_cache[0] != key:
            normalized = self._adapter.normalized_matrix(self._std)
            self._adapter_cache = (key, torch.as_tensor(normalized, dtype=like.dtype, device=like.device))
        return self._adapter_cache[1]

    @torch.compiler.disable
    def _noise_fn(self, velocity, noise: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # this builds the reference from live teleop input and must stay eager.
        # `select_action` has already sampled the input (once per chunk) and put
        # it in _pending_command, so the forward schedule it chose matches this hook.
        command = self._pending_command
        if command is None:
            self._last_reference = None
            return noise
        reference = self.reference_chunk(command).to(device=noise.device, dtype=noise.dtype)
        self._last_reference = reference
        self.steered_chunks += 1
        total = self._policy.config.num_inference_steps
        latent = reverse_flow(
            reference,
            velocity,
            total,
            adapter=self._adapter_tensor(noise),
            n_reversal_steps=self._n_reversal_steps,
        )
        delegated = self._delegated.to(noise.device)
        if not delegated.any():
            return latent
        # Delegated elements follow the unsteered flow: at t=1 that is the chunk's own
        # Gaussian sample (Tang et al., App. D: the freed noises are set to N(0, I)).
        unsteered = (
            noise
            if self._n_reversal_steps is None
            else forward_flow(noise, velocity, total, stop_at=self._n_reversal_steps)
        )
        return torch.where(delegated, unsteered, latent)

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the noise hook.
        if len(self._queue) == 0:
            # Sample the operator once per chunk: the forward schedule depends on
            # whether we are steering, so it cannot be decided inside the hook.
            command = self._source.translation
            steering = np.max(np.abs(command)) >= DEADBAND
            self._pending_command = command if steering else None
            flow_kwargs = self._flow_kwargs() if steering else {}
            if not steering:
                self._last_reference = None  # no reference to score this chunk against
            chunk, plain = paired_chunks(
                self._policy, batch, steering, noise_fn=self._noise_fn, **flow_kwargs
            )
            actions = chunk[:, : self._policy.config.n_action_steps]
            executed = actions[0].float().cpu()  # one device sync per chunk
            self._gripper_ref = float(executed[-1, GRIPPER_DIM])
            if self._last_reference is not None:
                target = self._last_reference[0, : executed.shape[0], :3].float().cpu()
                self.reconstruction_errors.append((executed[:, :3] - target).abs().mean().item())
            steps = list(actions.transpose(0, 1))
            self._queue.extend(steps)
            self._policy_queue.extend(
                steps if plain is chunk else plain[:, : self._policy.config.n_action_steps].transpose(0, 1)
            )
        self.last_policy_action = self._policy_queue.popleft()
        return self._queue.popleft()
