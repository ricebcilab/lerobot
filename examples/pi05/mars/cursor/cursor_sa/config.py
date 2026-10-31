"""YAML configs: arms and protocols.

An **arm** is one control condition (configs/<arm>.yaml): the policy, the control
mode and its depth, the input channel (corruption, noise) and the task. An arm
file says `extends: base.yaml` (or another arm) and overrides only what it
changes. A **protocol** (configs/study.yaml) names the arms a participant plays
and how: trials and practice per block, block order, trial timing. Unknown or
missing keys are errors, so a typo cannot silently do nothing. Spec values
(corruption, reversal_adapter) are replaced whole, never merged.
"""

from pathlib import Path

import yaml

from .channel import adapter_spec, corruption_spec
from .env import obstacle_spec
from .paths import CONFIGS as CONFIG_DIR
from .steering import GAIN_MODES, MODES, method_name

ARM_KEYS = {  # dotted YAML key -> parameter name
    "policy.model": "model",
    "policy.flow_steps": "flow_steps",
    "policy.execute": "execute",
    "control.mode": "mode",
    "control.tau": "tau",
    "control.lambda": "lambda",  # `lam` only where it is a keyword argument (lambda is a Python keyword)
    "control.ic_gain": "ic_gain",
    "control.n_reverse_steps": "n_reverse_steps",
    "control.pinned_steps": "pinned_steps",
    "control.speed_gain": "speed_gain",
    "control.gain_mode": "gain_mode",
    "control.input_noise": "input_noise",
    "control.corruption": "corruption",
    "control.reversal_adapter": "reversal_adapter",
    "task.targets": "targets",
    "task.target_radius": "target_radius",
    "task.min_target_sep": "min_target_sep",
    "task.reach": "reach",
    "task.dwell_s": "dwell_s",
    "task.time_limit_s": "time_limit_s",
    "task.obstacle": "obstacle",
    "task.velocity_tau": "velocity_tau",
    "task.curl": "curl",
}
LAYOUT_KEYS = ("targets", "target_radius", "min_target_sep", "reach", "obstacle")  # the task parameters the layouts depend on
MODE_ALIASES = {"shared_flow_control": "flow_control", "shared_flow_reversal_steering": "frs"}  # LeRobot's names
CHUNK_STEPS = 16  # the policies' chunk length (FlowPolicy's horizon): flow_control's tau counts chunk steps
PROTOCOL_DEFAULTS = {
    "blocks": {"trials": 40, "practice": 5, "order": "latin_square"},
    "trial": {"ready_hold_s": 0.3, "feedback_s": 0.6},
    "expert": {"lag": 0, "slow_radius": 0.15, "clearance": 0.05, "margin": 0.03},
    "display": {"blind": True},
}


def resolve(path):
    """A config path, or a bare file name looked up in configs/."""
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"no config {path} (nor in {CONFIG_DIR})")
    return path


def _nested(path):
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    merged = _nested(resolve(path.parent / parent)) if parent else {}
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"{path}: section {section!r} must be a mapping")
        merged[section] = {**merged.get(section, {}), **values}
    return merged


# ---------------------------------------------------------------- arms


def load_arm(path, overrides=None):
    """An arm's parameters {name: value}, validated, with the derived fields
    corruption_matrix, corruption_curl, adapter_matrix, learned_adapter and method. `overrides`
    (parameter name -> value) win over the file, as command-line flags do."""
    path = resolve(path)
    params = {}
    for section, values in _nested(path).items():
        for key, value in values.items():
            if f"{section}.{key}" not in ARM_KEYS:
                raise ValueError(f"{path}: unknown key {section}.{key}")
            params[ARM_KEYS[f"{section}.{key}"]] = value
    missing = sorted(set(ARM_KEYS.values()) - set(params))
    if missing:
        raise ValueError(f"{path}: missing {missing} (does it extend base.yaml?)")
    params.update(overrides or {})
    params["name"] = path.stem
    return derive(validate_arm(params, path.name))


def validate_arm(params, where="arm"):
    params["mode"] = MODE_ALIASES.get(params["mode"], params["mode"])
    steps = params["flow_steps"]
    if params["mode"] not in MODES:
        raise ValueError(f"{where}: control.mode must be one of {MODES}, got {params['mode']!r}")
    if not 0 <= params["tau"] <= CHUNK_STEPS:  # Kao et al.: chunk steps whose initial condition carries the command
        raise ValueError(f"{where}: control.tau must be in [0, {CHUNK_STEPS}] chunk steps, got {params['tau']}")
    if not 0 <= params["lambda"] <= steps:  # the lab's variant: Euler steps the command's chunk is held for
        raise ValueError(f"{where}: control.lambda must be in [0, {steps}] Euler steps, got {params['lambda']}")
    if params["ic_gain"] <= 0:
        raise ValueError(f"{where}: control.ic_gain must be positive")
    if params["n_reverse_steps"] is not None and not 1 <= params["n_reverse_steps"] <= steps:
        raise ValueError(f"{where}: control.n_reverse_steps must be null or in [1, {steps}]")
    if params["pinned_steps"] is not None and not 1 <= params["pinned_steps"] <= CHUNK_STEPS:
        raise ValueError(f"{where}: control.pinned_steps must be null or in [1, {CHUNK_STEPS}] chunk steps")
    if params["speed_gain"] <= 0:
        raise ValueError(f"{where}: control.speed_gain must be positive")
    if params["gain_mode"] not in GAIN_MODES:
        raise ValueError(f"{where}: control.gain_mode must be one of {GAIN_MODES}, got {params['gain_mode']!r}")
    if params["curl"] and not params["velocity_tau"]:
        raise ValueError(f"{where}: task.curl needs task.velocity_tau > 0 (a force needs inertia to act on)")
    if not (isinstance(params["targets"], (list, tuple)) and len(params["targets"]) == 2):
        raise ValueError(f"{where}: task.targets must be [min, max]")
    reach = params["reach"]
    if not (isinstance(reach, (list, tuple)) and len(reach) == 2 and reach[0] >= 0
            and (reach[1] is None or reach[1] >= reach[0])):
        raise ValueError(f"{where}: task.reach must be [min, max] with max >= min or null, got {reach!r}")
    for key in ("target_radius", "min_target_sep", "dwell_s", "time_limit_s"):
        if not params[key] > 0:
            raise ValueError(f"{where}: task.{key} must be positive, got {params[key]!r}")
    params["obstacle"] = obstacle_spec(params["obstacle"], f"{where}: task.obstacle")
    return params


def derive(params):
    spec = params["reversal_adapter"]
    params["corruption_matrix"], params["corruption_curl"] = corruption_spec(params["corruption"], "control.corruption")
    params["adapter_matrix"] = adapter_spec(spec, params["corruption_matrix"], "control.reversal_adapter")
    # the checkpoints the spec names: {learned: name} -> the name; {user: ..., task: ...} -> a dict of names
    if isinstance(spec, dict) and "learned" in spec:
        params["learned_adapter"] = spec["learned"]
    elif isinstance(spec, dict) and set(spec) <= {"user", "task"}:
        names = {k: spec.get(k) for k in ("user", "task")}
        params["learned_adapter"] = names if any(names.values()) else None
    else:
        params["learned_adapter"] = None
    has_adapter = params["adapter_matrix"] is not None or params["learned_adapter"] is not None
    params["method"] = method_name(params["mode"], has_adapter)
    return params


def adapter_names(arm):
    """The adapter checkpoints an arm loads, in user-then-task order."""
    names = arm["learned_adapter"]
    if isinstance(names, str):
        return [names]
    return [n for n in (names or {}).values() if n] if isinstance(names, dict) else []


# ---------------------------------------------------------------- protocols


def load_protocol(path):
    """A protocol {name, arms, blocks, trial, expert, display}; arms maps arm name -> config file."""
    path = resolve(path)
    data = yaml.safe_load(path.read_text()) or {}
    unknown = set(data) - {"name", "arms"} - set(PROTOCOL_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    arms = data.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError(f"{path}: arms must map arm names to config files")
    for name, file in arms.items():
        resolve(file)
    protocol = dict(name=data.get("name", path.stem), arms=dict(arms))
    for section, defaults in PROTOCOL_DEFAULTS.items():
        given = data.get(section) or {}
        bad = set(given) - set(defaults)
        if bad:
            raise ValueError(f"{path}: unknown keys in {section}: {sorted(bad)}")
        protocol[section] = {**defaults, **given}
    if protocol["blocks"]["order"] not in ("latin_square", "listed"):
        raise ValueError(f"{path}: blocks.order must be latin_square or listed")
    return protocol
