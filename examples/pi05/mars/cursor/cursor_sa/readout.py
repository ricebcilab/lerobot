"""The readout between an agent and the controller.

A command is a deflection [x, y] in [-1, 1]; inside DEADBAND the agent counts as
idle. `Readout` perturbs the command on its way to the controller, as in
LeRobot's TeleopChain: raw -> M @ raw (corruption) -> + Gaussian noise. A
corruption is a 2x2 matrix M, or a curl field on the command: the command the
controller gets is a velocity v driven by the raw push through
dv/dt = (raw - v) / tau + curl * R v (R a clockwise quarter turn), the dynamics
`env.CursorEnv` gives the cursor, here acting on the command instead. The rules and the stateful
`Readout` (with its torch forms for training, `Readout.chunk` and `Readout.serve`) are LeRobot's operator
readout (`lerobot.teleoperators.readout`), which the robot example's TeleopChain runs too; this module keeps
the specs that configure it and re-exports the rest.
"""

import numpy as np

from lerobot.teleoperators.readout import DEADBAND, Readout, add_noise, corrupt, curl_step, is_active  # noqa: F401


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
