"""The input channel between an agent and the controller.

A command is a deflection [x, y] in [-1, 1]; inside DEADBAND the agent counts as
idle. `InputChannel` perturbs the command on its way to the controller, as in
LeRobot's TeleopChain: raw -> M @ raw (corruption) -> + Gaussian noise. A
corruption is a 2x2 matrix M, or a curl field on the command: the command the
controller gets is a velocity v driven by the raw push through
dv/dt = (raw - v) / tau + curl * R v (R a clockwise quarter turn), the dynamics
`env.CursorEnv` gives the cursor, here acting on the command instead.
"""

import numpy as np
import torch

from .env import curl_discretization

DEADBAND = 0.05               # max |component| below which the operator is idle (as in LeRobot)


def is_active(command):
    """True where a command [..., 2] is outside the deadband."""
    return np.abs(command).max(-1) >= DEADBAND


def matrix_spec(spec, where="matrix"):
    """2x2 matrix from a spec, or None for a null spec. Forms, as in LeRobot:
    {scale: [sx, sy]} (per-axis gain), {rotation_deg: d}, {M: rows}, or bare rows."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        if set(spec) == {"scale"}:
            m = np.diag(np.asarray(spec["scale"], dtype=float))
        elif set(spec) == {"rotation_deg"}:
            c, s = np.cos(np.radians(spec["rotation_deg"])), np.sin(np.radians(spec["rotation_deg"]))
            m = np.array([[c, -s], [s, c]])
        elif set(spec) == {"M"}:
            m = np.asarray(spec["M"], dtype=float)
        else:
            raise ValueError(f"{where}: expected exactly one of scale, rotation_deg or M, got {sorted(spec)}")
    else:
        m = np.asarray(spec, dtype=float)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)):
        raise ValueError(f"{where}: expected a finite 2x2 matrix, got {spec!r}")
    return m


def corruption_spec(spec, where="control.corruption"):
    """(matrix M or None, curl {curl, tau} or None) from a corruption spec: a matrix form
    (`matrix_spec`), or {curl: b, tau: s} for a curl field on the command (b in 1/s, + = clockwise;
    tau the command's time constant in s)."""
    if isinstance(spec, dict) and "curl" in spec:
        if set(spec) != {"curl", "tau"}:
            raise ValueError(f"{where}: a curl corruption takes exactly curl and tau, got {sorted(spec)}")
        curl, tau = float(spec["curl"]), float(spec["tau"])
        if not (np.isfinite(curl) and tau > 0):
            raise ValueError(f"{where}: curl must be finite and tau positive, got {spec!r}")
        return None, {"curl": curl, "tau": tau}
    return matrix_spec(spec, where), None


def adapter_spec(spec, corruption, where="reversal_adapter"):
    """The fixed 2x2 user matrix F of a reversal adapter spec (MARS), or None. Forms, as in LeRobot:
    {translation: corruption} copies the corruption M; {translation: <matrix spec>}; {F: rows}. The
    cursor's action is translation only, so F is 2x2. {learned: <name>} is a learned checkpoint, and
    {user: <name | null>, task: <name | null>} a user and a task module from their checkpoints,
    composed; checkpoints are loaded by the caller (`config.adapter_names`), so those give no F."""
    if spec is None or (isinstance(spec, dict) and set(spec) == {"learned"}):
        return None
    if isinstance(spec, dict) and spec and set(spec) <= {"user", "task"}:
        for key, value in spec.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{where}: {key} must be a checkpoint name or null, got {value!r}")
        return None
    if not isinstance(spec, dict) or len(spec) != 1 or not set(spec) <= {"translation", "F"}:
        raise ValueError(f"{where}: expected {{translation: ...}}, {{F: rows}}, {{learned: name}} or "
                         f"{{user: ..., task: ...}}, got {spec!r}")
    if "F" in spec:
        return matrix_spec(spec["F"], f"{where}.F")
    if spec["translation"] == "corruption":
        if corruption is None:
            raise ValueError(f"{where}: translation: corruption needs a corruption to copy")
        return corruption.copy()
    return matrix_spec(spec["translation"], f"{where}.translation")


class InputChannel:
    """What happens to a command between the operator and the controller:
    raw -> M @ raw (corruption, clipped to [-1, 1]) -> + N(0, input_noise^2) per
    axis (clipped again). As in LeRobot, both apply only while the operator is
    commanding (outside DEADBAND), so idle input stays exactly idle.

    With `curl` ({curl, tau}) the corruption is a curl field instead of M: the
    channel keeps a velocity per trial, driven by every raw push (idle ones too)
    and reset by `reset(n_trials)` at the start of a batch of trials; the
    controller gets that velocity (clipped), then the noise. Steps of `dt` s.

    Batched: commands are [B, 2], or [B, H, 2] for a whole chunk per trial (not
    with a curl). `corruption` is one 2x2 matrix or one per trial [B, 2, 2];
    `input_noise` is one std or one per trial [B].
    """

    def __init__(self, corruption=None, input_noise=0.0, rng=None, curl=None, dt=0.02):
        self.corruption, self.input_noise = corruption, np.asarray(input_noise, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.curl, self.vel = curl, None
        if curl is not None:
            if corruption is not None:
                raise ValueError("a channel has a matrix corruption or a curl, not both")
            self._decay, self._drive = curl_discretization(curl["tau"], curl["curl"], dt)

    def reset(self, n_trials=1):
        """A new batch of trials: the curl's velocity starts at rest."""
        self.vel = None if self.curl is None else np.zeros((n_trials, 2))

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=float)
        extra = (1,) * (raw.ndim - 2)  # per-trial parameters broadcast over a chunk's steps
        out = raw
        if self.curl is not None:
            if raw.ndim != 2:
                raise ValueError("a curl channel takes one command per trial and step, [B, 2]")
            if self.vel is None or self.vel.shape != raw.shape:
                self.vel = np.zeros_like(raw)
            self.vel = self.vel @ self._decay.T + raw @ self._drive.T
            out = np.clip(self.vel, -1.0, 1.0)
        if self.corruption is not None:
            m = self.corruption
            if m.ndim == 3:
                m = m.reshape(m.shape[:1] + extra + m.shape[1:])
            out = np.clip(np.einsum("...ij,...j->...i", m, raw), -1.0, 1.0)
        if np.any(self.input_noise):
            sigma = self.input_noise.reshape(self.input_noise.shape + extra + (1,)) if self.input_noise.ndim else self.input_noise
            out = np.clip(out + self.rng.normal(size=out.shape) * sigma, -1.0, 1.0)
        return np.where(is_active(raw)[..., None], out, raw)


def curl_commands(raw, curl, dt=0.02):
    """What a curl channel ({curl, tau}) serves, before noise, for raw pushes [B, L, 2] given step by
    step from rest (numpy): the velocity after every step, clipped to [-1, 1]."""
    decay, drive = curl_discretization(curl["tau"], curl["curl"], dt)
    vel, out = np.zeros(raw.shape[::2]), np.empty_like(raw, dtype=float)
    for k in range(raw.shape[1]):
        vel = vel @ decay.T + raw[:, k] @ drive.T
        out[:, k] = vel
    return np.clip(out, -1.0, 1.0)


def serve_command(core, raw, sigma, noise=None):
    """The torch twin of the channel's last stage, for training: `core` [B, 2] (M @ raw clipped, or a
    curl's velocity) plus noise of std `sigma` [B], clipped, only where the raw push [B, 2] is active.
    `noise` [B, 2]: the standard normal draws to scale by sigma (default: fresh ones)."""
    noise = torch.randn_like(core) if noise is None else noise
    out = torch.clamp(core + sigma[:, None] * noise, -1.0, 1.0)
    return torch.where(raw.abs().amax(-1, keepdim=True) >= DEADBAND, out, raw)


def corrupt_chunk(command, corruption, sigma):
    """`InputChannel` in torch, for training: deflection chunks `command` [B, H, 2],
    one 2x2 `corruption` (or None) and a noise std per trial `sigma` [B]. The same
    rule, step by step: M @ raw, clip, + noise, clip, only where the step is active."""
    out = command if corruption is None else torch.clamp(command @ corruption.T, -1.0, 1.0)
    out = torch.clamp(out + sigma[:, None, None] * torch.randn_like(out), -1.0, 1.0)
    active = command.abs().amax(-1, keepdim=True) >= DEADBAND
    return torch.where(active, out, command)
