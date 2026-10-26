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

- ``FlowControlPolicy``: shared flow control (Kao et al. 2026): the command,
  scaled by ``ic_gain`` into the noise prior's units, is written into the
  translation dims of the first ``tau`` chunk steps of ``x_t`` at the start of
  each of the first ``lam`` Euler steps (``lam`` 1 = the initial condition
  only); the flow itself runs unmodified.
- ``FlowReversalSteeringPolicy``: a uniform-velocity reference chunk built from
  the command is integrated *backward* through the policy's velocity field to
  the latent noise that maps to it (Flow Reversal Steering, Tang et al. 2026,
  arXiv:2606.13675); the forward flow then starts from that noise.

Both read an operator ``TeleopSource`` and the checkpoint's action statistics
from the postprocessor. ``reverse_flow`` is the integrator. ``ReversalAdapter``
is MARS (Modular Adapters for Reversal Steering, the method formerly called
FRS-RA): a *user module* that adapts the velocity field of the reversal only
(a fixed 7x7 matrix F, typically the known command corruption, or a learned
``LearnedUserModule`` U(x_t, t)) and/or a *task module* (A(x_t, t, v) v + b)
that adapts the field in both directions, so it reshapes every chunk the
policy produces. The forward field is task(v) and the backward field
user(task(v)); the modules are trained apart and composed here. FRS with a
user module alone is MARS-U, with a task module alone MARS-T.

Elements of the reference the operator does not command (``operator_dims``
selects which of translation / rotation / gripper they do), the padding dims and
the steps beyond ``pinned_steps`` are *delegated* to the policy: after the
reversal their latent is replaced by the unsteered one (fresh Gaussian noise for
a full reversal), the noise-space in-painting of Tang et al., App. D.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

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
DEADBAND = 0.05  # default `deadband` of the wrappers: below this (env units) the operator counts as idle


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


# ---------------------------------------------------------------- MARS: modular reversal adapters


def sinusoidal_time_embedding(t, dim: int, device=None, dtype=None) -> torch.Tensor:
    """(B, dim) embedding of flow times t (a float or a (B,) tensor): [sin(1000 t w_k), cos(1000 t w_k)],
    w_k = 10^(-4k/(dim/2)); the adapters' time conditioning."""
    t = torch.as_tensor(t, dtype=dtype or torch.float32, device=device).reshape(-1)
    half = dim // 2
    omega = 10.0 ** (-4.0 * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    angles = 1000.0 * t[:, None] * omega[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _mlp(d_in: int, d: int, d_out: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(d_in, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d_out))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class LearnedUserModule(nn.Module):
    """MARS user module U(x_t, t): one n x n matrix per chunk step, applied to the policy's velocity
    in the BACKWARD pass of the reversal; it undoes one user's command corruption. Works in the
    policy's normalized action space on the first ``n_dims`` dims of ``x_t``. Zero-initialized:
    a fresh module is the identity (plain FRS)."""

    def __init__(self, horizon: int, n_dims: int = N_ACTION_DIMS, d: int = 256, tdim: int = 64):
        super().__init__()
        self.config = {"horizon": horizon, "n_dims": n_dims, "d": d, "tdim": tdim}
        self.horizon, self.n_dims, self.tdim = horizon, n_dims, tdim
        self.net = _mlp(horizon * n_dims + tdim, d, horizon * n_dims * n_dims)

    def forward(self, x_t: torch.Tensor, t) -> torch.Tensor:
        """x_t (B, H, >= n_dims), t a float or (B,) -> (B, H, n, n)."""
        b, h, n = x_t.shape[0], self.horizon, self.n_dims
        x = x_t[:, :h, :n].reshape(b, h * n).to(torch.float32)
        emb = sinusoidal_time_embedding(t, self.tdim, device=x.device).expand(b, -1)
        delta = self.net(torch.cat([x, emb], dim=-1)).view(b, h, n, n)
        return torch.eye(n, device=x.device, dtype=x.dtype) + delta


class LearnedTaskModule(nn.Module):
    """MARS task module: an affine map of the policy's velocity per chunk step, A(x_t, t, v) v + b, in
    BOTH directions of the flow; it reshapes the frozen policy's chunk distribution toward a task.
    Conditioned on the chunk state and velocity only (the cursor version also reads obstacle
    tokens; pi0.5 sees the scene through its own field). Zero-initialized: the identity."""

    def __init__(self, horizon: int, n_dims: int = N_ACTION_DIMS, d: int = 256, tdim: int = 64):
        super().__init__()
        self.config = {"horizon": horizon, "n_dims": n_dims, "d": d, "tdim": tdim}
        self.horizon, self.n_dims, self.tdim = horizon, n_dims, tdim
        self.net = _mlp(2 * horizon * n_dims + tdim, d, horizon * (n_dims * n_dims + n_dims))

    def forward(self, x_t: torch.Tensor, t, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_t (B, H, >= n), v (B, H, n), t a float or (B,) -> (A (B, H, n, n), b (B, H, n))."""
        b, h, n = x_t.shape[0], self.horizon, self.n_dims
        x = x_t[:, :h, :n].reshape(b, h * n).to(torch.float32)
        vel = v[:, :h, :n].reshape(b, h * n).to(torch.float32)
        emb = sinusoidal_time_embedding(t, self.tdim, device=x.device).expand(b, -1)
        out = self.net(torch.cat([x, vel, emb], dim=-1)).view(b, h, n * n + n)
        a = torch.eye(n, device=x.device, dtype=x.dtype) + out[..., : n * n].view(b, h, n, n)
        return a, out[..., n * n :]


_MODULE_KINDS = {"user": LearnedUserModule, "task": LearnedTaskModule}


def save_adapter_module(module: nn.Module, path, kind: str, meta: dict | None = None) -> Path:
    """Save a MARS module checkpoint: ``kind`` (user | task), its config, weights and free-form meta."""
    if kind not in _MODULE_KINDS or not isinstance(module, _MODULE_KINDS[kind]):
        raise ValueError(f"kind must be one of {sorted(_MODULE_KINDS)} and match the module, got {kind!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"kind": kind, "config": module.config, "state_dict": module.state_dict(), "meta": meta or {}}, path
    )
    return path


def load_adapter_module(path, device="cpu") -> tuple[nn.Module, str, dict]:
    """Load a MARS module checkpoint: (module in eval mode, kind, meta)."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)  # tensors, numbers, strings only
    kind = ckpt.get("kind")
    if kind not in _MODULE_KINDS:
        raise ValueError(f"{path}: unknown adapter module kind {kind!r}")
    module = _MODULE_KINDS[kind](**ckpt["config"])
    module.load_state_dict(ckpt["state_dict"])
    return module.to(device).eval().requires_grad_(False), kind, ckpt.get("meta", {})


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


class ReversalAdapter(MatrixHolder):
    """The MARS adapter of ``FlowReversalSteeringPolicy``: a user module (``matrix``, a fixed
    environment-space F, or ``user_module``, a learned U(x_t, t) or any (x_t, t) -> (B, H, n, n)
    callable) applied in the reversal only, and a ``task_module`` ((x_t, t, v) -> (A, b)) applied in
    both directions. All None = off (plain FRS). Consumers read the holder on every use."""

    size = N_ACTION_DIMS
    where = "reversal adapter"

    def __init__(self, matrix=None, label: str | None = None, user_module=None, task_module=None):
        super().__init__(matrix, label)
        self.user_module = user_module
        self.task_module = task_module
        self.user_label: str | None = None
        self.task_label: str | None = None

    @property
    def kind(self) -> str:
        user = self._matrix is not None or self.user_module is not None
        task = self.task_module is not None
        return {
            (True, True): "composed",
            (True, False): "user",
            (False, True): "task",
            (False, False): "none",
        }[(user, task)]

    def load_modules(self, user_path=None, task_path=None, device="cpu") -> None:
        """Load learned modules from checkpoints into the user and/or task slot (None leaves a slot alone)."""
        if user_path is not None:
            module, kind, _ = load_adapter_module(user_path, device)
            if kind != "user":
                raise ValueError(f"{user_path} holds a {kind} module, not a user module")
            self.user_module, self.user_label = module, str(user_path)
        if task_path is not None:
            module, kind, _ = load_adapter_module(task_path, device)
            if kind != "task":
                raise ValueError(f"{task_path} holds a {kind} module, not a task module")
            self.task_module, self.task_label = module, str(task_path)

    def describe(self) -> str:
        if self.kind == "none":
            return "Reversal adapter: off (plain FRS)."
        parts = []
        if self._matrix is not None:
            identity = " (identity, a no-op)" if np.allclose(self._matrix, np.eye(N_ACTION_DIMS)) else ""
            parts.append(f"fixed user matrix F = {self.label}{identity} = {format_matrix(self._matrix)}")
        if self.user_module is not None:
            parts.append(f"learned user module {self.user_label or type(self.user_module).__name__}")
        if self.task_module is not None:
            parts.append(f"task module {self.task_label or type(self.task_module).__name__}")
        name = {"user": "MARS-U", "task": "MARS-T", "composed": "MARS"}[self.kind]
        return f"Reversal adapter: {name} ({'; '.join(parts)})."

    def normalized_matrix(self, std: np.ndarray) -> np.ndarray | None:
        """Convert an environment-space velocity transform to normalized coordinates: S^-1 F S.

        Velocities have no mean offset. Apply this to all seven dimensions so literal
        matrices that mix translation, orientation and gripper use the same convention.
        """
        if self.matrix is None:
            return None
        return self.matrix * std[np.newaxis, :] / std[:, np.newaxis]

    def field(self, std: np.ndarray, like: torch.Tensor) -> "AdaptedField | None":
        """The adapter as applied to normalized velocities on `like`'s device, or None when off."""
        if self.kind == "none":
            return None
        matrix = self.normalized_matrix(std)
        tensor = None if matrix is None else torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
        return AdaptedField(tensor, self.user_module, self.task_module)


class AdaptedField:
    """Applies MARS modules to a velocity: task (both directions), then user (backward only).
    Only the first n action dims are touched; the padding dims pass through."""

    def __init__(self, matrix: torch.Tensor | None, user_module=None, task_module=None):
        self.matrix, self.user_module, self.task_module = matrix, user_module, task_module

    def adapt(self, x_t: torch.Tensor, t: float, v: torch.Tensor, backward: bool) -> torch.Tensor:
        n = N_ACTION_DIMS
        if self.matrix is not None:
            n = self.matrix.shape[0]
        head = v[..., :n]
        if self.task_module is not None:
            a, b = self.task_module(x_t, t, head)
            head = torch.einsum("bhij,bhj->bhi", a.to(head.dtype), head) + b.to(head.dtype)
        if backward:
            if self.matrix is not None:
                head = head @ self.matrix.transpose(-1, -2)
            if self.user_module is not None:
                head = torch.einsum("bhij,bhj->bhi", self.user_module(x_t, t).to(head.dtype), head)
        return torch.cat([head, v[..., n:]], dim=-1)


def _adapt_velocity(adapter, x_t, t, v, backward: bool):
    """`adapter` is None, a normalized (n, n) matrix (the legacy fixed-F form), an AdaptedField, or a
    ReversalAdapter holder used directly (its matrix is then taken as already normalized)."""
    if adapter is None:
        return v
    if isinstance(adapter, torch.Tensor):
        if not backward:
            return v
        n = adapter.shape[0]
        return torch.cat([v[..., :n] @ adapter.transpose(-1, -2), v[..., n:]], dim=-1)
    if isinstance(adapter, ReversalAdapter):
        if adapter.kind == "none":
            return v
        matrix = (
            None
            if adapter.matrix is None
            else torch.as_tensor(adapter.matrix, dtype=v.dtype, device=v.device)
        )
        adapter = AdaptedField(matrix, adapter.user_module, adapter.task_module)
    return adapter.adapt(x_t, t, v, backward)


def _rng_state() -> tuple:
    """CPU and CUDA generator states, so a second sampling pass can reuse the first one's noise."""
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return torch.get_rng_state(), cuda


def _set_rng_state(state: tuple) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def paired_chunks(
    policy, batch, steered: bool, common: dict | None = None, **steer_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """The steered chunk and the policy's own plan for the same observation and the same noise.

    Returns (steered, plain). When `steered` is False the two are the same tensor: no
    second pass. Otherwise the generator state is restored between the passes so the
    plain chunk starts from the Gaussian sample the steered one used, and the state
    after the steered pass is restored afterwards, so a run's random stream is exactly
    what it would be without the second pass. `common` kwargs go to both passes (a MARS
    task module shapes the policy's own plan too); `steer_kwargs` only to the steered one.
    """
    common = common or {}
    if not steered:
        chunk = policy.predict_action_chunk(batch, **common)
        return chunk, chunk
    before = _rng_state()
    chunk = policy.predict_action_chunk(batch, **common, **steer_kwargs)
    after = _rng_state()
    _set_rng_state(before)
    plain = policy.predict_action_chunk(batch, **common)
    _set_rng_state(after)
    return chunk, plain


# ---------------------------------------------------------------- FlowControlPolicy


def get_action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]:
    """The affine (mean, std) of the checkpoint's action normalization: normalized = (x - mean) / std.

    Read from the unnormalizer step's stats and its normalization mode: mean/std as is,
    quantiles and min/max as the equivalent affine map onto [-1, 1].
    """
    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if not stats or "action" not in stats:
            continue
        action = {k: np.asarray(v, dtype=np.float64) for k, v in stats["action"].items()}
        mode = getattr(getattr(step, "norm_map", None), "get", lambda *_: None)("ACTION")
        mode = getattr(mode, "value", mode) or "MEAN_STD"
        if mode == "QUANTILES":
            lo, hi = action["q01"], action["q99"]
        elif mode == "QUANTILE10":
            lo, hi = action["q10"], action["q90"]
        elif mode == "MIN_MAX":
            lo, hi = action["min"], action["max"]
        else:
            return action["mean"], np.maximum(action["std"], 1e-6)
        return (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    raise RuntimeError("No action stats in the postprocessor; the shared flow modes need them.")


class FlowControlPolicy:
    """pi0.5 wrapper implementing shared_flow_control (Kao et al. 2026).

    While the operator pushes, the command is written into ``x_t`` as the chunk's
    initial condition: scaled by ``ic_gain`` into the noise prior's units (a full
    push writes ``ic_gain`` sigma), into the translation dims of the first ``tau``
    chunk steps, at the start of each of the first ``lam`` (lambda) Euler steps of
    the chunk's denoising. The flow itself runs unmodified, so the executed action
    is entirely the model's output. ``lam`` 1 writes the initial condition only
    (Kao et al.'s method), 0 writes nothing (the policy alone), more keeps
    re-writing as the flow proceeds (the lab's earlier variant, whose
    ``n_guided_steps`` was this lambda, with the whole chunk written in action
    units). ``ic_gain="action"`` writes the command in the checkpoint's action
    units, ``(command - mean) / std``, which with ``tau`` = the whole chunk and
    ``lam`` 1 is the arm the recorded human study ran. Guidance is skipped while
    the input is inside ``deadband``, so idle input means pure policy.
    """

    def __init__(
        self,
        policy,
        source: TeleopSource,
        postprocessor,
        tau: int | None = None,
        lam: int = 1,
        ic_gain: float | str = "action",
        deadband: float = DEADBAND,
    ):
        self._policy = policy
        self._source = source
        self.deadband = deadband
        self.tau = tau  # the setters validate against the policy's config
        self.lam = lam
        self.ic_gain = ic_gain
        mean, std = get_action_mean_std(postprocessor)
        self._mean, self._std = mean[:3], std[:3]
        self._queue: deque = deque()
        self.guided_steps = 0  # per-rollout count of Euler steps the command was written at
        self._pending_command: np.ndarray | None = None
        self._pending_target: torch.Tensor | None = None
        self._policy_queue: deque = deque()
        self.last_policy_action: torch.Tensor | None = (
            None  # the policy's own plan for the step just returned
        )

    @property
    def tau(self) -> int:
        return self._tau

    @tau.setter
    def tau(self, value: int | None) -> None:
        chunk = self._policy.config.chunk_size
        if value is None:
            value = chunk
        if not (isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= chunk):
            raise ValueError(f"tau must be an integer in [1, {chunk}] (chunk steps) or None, got {value!r}")
        self._tau = value

    @property
    def lam(self) -> int:
        return self._lam

    @lam.setter
    def lam(self, value: int) -> None:
        total = self._policy.config.num_inference_steps
        if not (isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= total):
            raise ValueError(f"lam must be an integer in [0, {total}] (Euler steps), got {value!r}")
        self._lam = value

    @property
    def n_guided_steps(self) -> int:
        """The old name of ``lam`` (the Euler steps the command is written at)."""
        return self._lam

    @n_guided_steps.setter
    def n_guided_steps(self, value: int) -> None:
        self.lam = value

    @property
    def ic_gain(self) -> float | str:
        return self._ic_gain

    @ic_gain.setter
    def ic_gain(self, value: float | str) -> None:
        if value == "action":
            self._ic_gain = "action"
            return
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"ic_gain must be a positive number or 'action', got {value!r}")
        self._ic_gain = float(value)

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self._policy_queue.clear()
        self.guided_steps = 0
        self._pending_command = None

    def written_command(self, command: np.ndarray) -> np.ndarray:
        """What a command writes into ``x_t``'s translation dims: ``ic_gain * command`` in the noise
        prior's units, or ``(command - mean) / std`` in the checkpoint's action units."""
        if self._ic_gain == "action":
            return (np.asarray(command, dtype=np.float64) - self._mean) / self._std
        return self._ic_gain * np.asarray(command, dtype=np.float64)

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook uses the current chunk's teleop command and must stay eager.
        if step >= self._lam or self._pending_command is None:
            return x_t
        if self._pending_target is None:
            target = self.written_command(self._pending_command)
            self._pending_target = torch.as_tensor(target, dtype=x_t.dtype, device=x_t.device)
        x_t[:, : self._tau, :3] = self._pending_target
        self.guided_steps += 1
        return x_t

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the guidance hook.
        if len(self._queue) == 0:
            # Hold one consumed command (including its noise) for the whole chunk,
            # matching FRS and the synthetic operator's refresh cadence.
            command = self._source.translation if self._lam else None
            self._pending_command = (
                command if command is not None and np.max(np.abs(command)) >= self.deadband else None
            )
            self._pending_target = None  # resolved on the device by the first write
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
    adapter=None,
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

    `adapter` is None, an (n, n) matrix F already in normalized coordinates (the
    fixed user module: x_t += h * (F @ v) on the first n action dims, the env's 7,
    leaving the padding dims alone; F = I is an exact no-op) or an ``AdaptedField``
    (MARS: task module, then user module, since this is the backward pass).
    """
    h = 1.0 / num_steps
    steps = num_steps if n_reversal_steps is None else n_reversal_steps
    x_t = x
    for step in range(steps):
        t = step * h
        v = _adapt_velocity(adapter, x_t, t, velocity(x_t, t), backward=True)
        x_t = x_t + h * v
    return x_t


def forward_flow(
    x: torch.Tensor, velocity, num_steps: int, stop_at: int | None = None, adapter=None
) -> torch.Tensor:
    """Integrate the flow forward (Euler) from noise at t=1 down to t = stop_at / num_steps.

    Mirrors PI05Pytorch.sample_actions without hooks: x_{t-h} = x_t - h * v(x_t, t). With
    ``stop_at`` None it runs all the way to t=0 (the unsteered action chunk); with
    ``stop_at = n`` it takes ``num_steps - n`` steps and returns the unsteered state at
    t = n / num_steps, i.e. where a reversal stopped after ``n`` steps would sit.
    ``adapter`` (an ``AdaptedField``) applies its task module only: the user module
    acts on the reversal alone.
    """
    h = 1.0 / num_steps
    steps = num_steps if stop_at is None else num_steps - stop_at
    x_t = x
    for step in range(steps):
        t = 1.0 - step * h
        x_t = x_t - h * _adapt_velocity(adapter, x_t, t, velocity(x_t, t), backward=False)
    return x_t


def uniform_reference(
    command, gripper_normalized, mean: np.ndarray, std: np.ndarray, chunk_size: int, max_action_dim: int
) -> torch.Tensor:
    """The reference chunk FRS steers toward: env-space translation ``command`` ((3,) or (B, 3)) held at
    uniform velocity over the chunk, rotation zero, the gripper at ``gripper_normalized`` (a float or
    (B,)), in the model's normalized padded space: (B, chunk_size, max_action_dim)."""
    command = np.atleast_2d(np.asarray(command, dtype=np.float64))
    env_action = np.zeros((command.shape[0], N_ACTION_DIMS))
    env_action[:, :3] = command
    normalized = (env_action - mean) / std
    normalized[:, GRIPPER_DIM] = np.asarray(gripper_normalized, dtype=np.float64)
    ref = torch.zeros(command.shape[0], chunk_size, max_action_dim, dtype=torch.float32)
    ref[..., :N_ACTION_DIMS] = torch.as_tensor(normalized, dtype=torch.float32)[:, None, :]
    return ref


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

    While the teleop input is deflected (beyond `deadband`), a reference chunk
    that servos in the commanded direction at uniform velocity (rotation zero,
    gripper held at the model's last executed command, `gripper_open` before
    the first chunk) is integrated *backward* through the
    policy's own velocity field for num_inference_steps to find the latent
    noise that maps to it; the normal forward flow then denoises from that
    noise instead of a random one. The executed action is entirely the
    model's output — the reference only picks the starting noise, so what
    comes out is the generalist action mode nearest your intent. Idle input
    = pure policy (random noise, as usual).

    An optional `ReversalAdapter` (MARS) adapts the velocity field: its user module
    (a fixed 7x7 matrix F or a learned U(x_t, t)) in the reversal only, its task
    module in both directions, including the forward flow that produces the executed
    chunk and the policy's own plan; see `reverse_flow`, `forward_flow` and `AdaptedField`.

    `n_reversal_steps` stops the reversal early instead of going all the way to
    noise: with n of the policy's N = num_inference_steps steps the reference is
    integrated up to t = n/N, and the forward flow then runs
    from there in n steps, maintaining the policy's step size in both directions.
    With delegation, a steered chunk costs N + n velocity evaluations: n reverse,
    N - n to obtain the unsteered state, and n forward. Depth changes how far
    inversion travels before delegation and reconstruction; its effect on
    intent preservation is empirical. n = N (or None) is the full reversal,
    the method of Tang et al.; a partial reversal (n < N) is this module's own
    variant, which the paper does not have, and at small n the executed chunk
    is mostly the reference itself.

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
        deadband: float = DEADBAND,
        gripper_open: float = GRIPPER_OPEN,
    ):
        self._policy = policy
        self._source = source
        self._adapter = adapter
        self.deadband = deadband
        self.gripper_open = gripper_open
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
        self._gripper_ref = self._normalize_gripper(gripper_open)
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
        self._gripper_ref = self._normalize_gripper(self.gripper_open)
        self._last_reference = None
        self.steered_chunks = 0
        self.reconstruction_errors = []

    def reference_chunk(self, command: np.ndarray) -> torch.Tensor:
        """Uniform-velocity chunk for env-space translation `command`, in the model's normalized space."""
        cfg = self._policy.config
        return uniform_reference(
            command, self._gripper_ref, self._mean, self._std, cfg.chunk_size, cfg.max_action_dim
        )

    def _adapter_field(self, like: torch.Tensor) -> AdaptedField | None:
        """The adapter as applied to normalized velocities on `like`'s device; the fixed matrix is
        converted once and rebuilt only when the holder's matrix or modules change."""
        if self._adapter is None or self._adapter.kind == "none":
            self._adapter_cache = None
            return None
        key = (
            self._adapter.matrix,
            self._adapter.user_module,
            self._adapter.task_module,
            like.device,
            like.dtype,
        )
        cached = self._adapter_cache
        # The cache keeps the matrix itself, not its id: a matrix that replaces a freed one can
        # land on the same address, and would then be mistaken for the one it replaced.
        if (
            cached is None
            or any(a is not b for a, b in zip(cached[0][:3], key[:3], strict=True))
            or cached[0][3:] != key[3:]
        ):
            self._adapter_cache = (key, self._adapter.field(self._std, like))
        return self._adapter_cache[1]

    def _adapter_tensor(self, like: torch.Tensor) -> torch.Tensor | None:
        """The fixed user matrix in normalized coordinates on `like`'s device (None without one)."""
        field = self._adapter_field(like)
        return None if field is None else field.matrix

    @torch.compiler.disable
    def _velocity_hook(self, step: int, time_: float, x_t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # The forward flow that produces the executed chunk: the task module only.
        field = self._adapter_field(x_t)
        return v if field is None else field.adapt(x_t, time_, v, backward=False)

    def _common_kwargs(self) -> dict:
        """Kwargs every chunk takes, steered or not: the task module's velocity hook."""
        if self._adapter is not None and self._adapter.task_module is not None:
            return {"velocity_hook": self._velocity_hook}
        return {}

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
        field = self._adapter_field(noise)
        latent = reverse_flow(
            reference, velocity, total, adapter=field, n_reversal_steps=self._n_reversal_steps
        )
        delegated = self._delegated.to(noise.device)
        if not delegated.any():
            return latent
        # Delegated elements follow the unsteered flow: at t=1 that is the chunk's own
        # Gaussian sample (Tang et al., App. D: the freed noises are set to N(0, I)).
        unsteered = (
            noise
            if self._n_reversal_steps is None
            else forward_flow(noise, velocity, total, stop_at=self._n_reversal_steps, adapter=field)
        )
        return torch.where(delegated, unsteered, latent)

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the noise hook.
        if len(self._queue) == 0:
            # Sample the operator once per chunk: the forward schedule depends on
            # whether we are steering, so it cannot be decided inside the hook.
            command = self._source.translation
            steering = np.max(np.abs(command)) >= self.deadband
            self._pending_command = command if steering else None
            flow_kwargs = self._flow_kwargs() if steering else {}
            if not steering:
                self._last_reference = None  # no reference to score this chunk against
            chunk, plain = paired_chunks(
                self._policy, batch, steering, self._common_kwargs(), noise_fn=self._noise_fn, **flow_kwargs
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
