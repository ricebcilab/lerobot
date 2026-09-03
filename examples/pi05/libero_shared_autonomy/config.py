"""YAML settings for the LIBERO shared-autonomy runners.

Both entry points read the same nested YAML shape (policy / scene / control /
server) into dataclasses. A file may say `extends: other.yaml` (relative to
itself) to overlay on another; a key set to null keeps the built-in default,
and unknown keys are rejected. Matrix specs under `control` are resolved here
into arrays (see `lerobot.policies.pi05.steering` for the accepted forms).
"""

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from teleop import build_corruption

from lerobot.policies.pi05.steering import build_reversal_adapter

MODES = ("policy", "shared_override", "shared_flow_control", "shared_flow_reversal_steering", "teleop")
TASK_ORDERS = ("random", "shuffled", "sequential")
PROMPT_FROM_TASK = "task"  # `prompt: task` gives the VLA the scene's own instruction

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_INTERACTIVE_CONFIG = CONFIG_DIR / "interactive.yaml"


@dataclass
class ControlSettings:
    mode: str = "policy"
    n_guided_steps: int = 5
    n_reversal_steps: int | None = None
    input_noise: float = 0.0
    max_steps: int | None = None
    corruption: dict | list | None = None  # spec as written
    reversal_adapter: dict | None = None  # spec as written
    corruption_matrix: np.ndarray | None = field(default=None, repr=False)
    reversal_adapter_matrix: np.ndarray | None = field(default=None, repr=False)


@dataclass
class SessionSettings:
    policy_path: str = "lerobot/pi05_libero_finetuned"
    n_action_steps: int = 10
    compile: bool = False
    suite: str = "libero_spatial"
    task_id: int = 0
    port: int = 8765
    output_dir: Path = Path("outputs/pi05_libero_interactive")
    control: ControlSettings = field(default_factory=ControlSettings)


@dataclass
class ExperimentSettings:
    name: str
    session: SessionSettings
    n_trials: int = 10
    seed: int = 0
    task_order: str = "random"
    output_dir: Path = Path("outputs/pi05_libero_experiments")
    task_ids: list[int] | None = None  # None = every task in the suite
    prompt: str = "do something"


_CONTROL_SCHEMA = {
    "mode": "mode",
    "n_guided_steps": "n_guided_steps",
    "n_reversal_steps": "n_reversal_steps",
    "input_noise": "input_noise",
    "max_steps": "max_steps",
    "corruption": "corruption",
    "reversal_adapter": "reversal_adapter",
}
_POLICY_SCHEMA = {"path": "policy_path", "n_action_steps": "n_action_steps", "compile": "compile"}

INTERACTIVE_SCHEMA = {
    "policy": _POLICY_SCHEMA,
    "scene": {"suite": "suite", "task_id": "task_id"},
    "control": _CONTROL_SCHEMA,
    "server": {"port": "port"},
    "output_dir": "output_dir",
}

EXPERIMENT_SCHEMA = {
    "experiment": {
        "n_trials": "n_trials",
        "seed": "seed",
        "task_order": "task_order",
        "output_dir": "output_dir",
    },
    "scene": {"suite": "suite", "task_ids": "task_ids"},
    "prompt": "prompt",
    "policy": _POLICY_SCHEMA,
    "control": _CONTROL_SCHEMA,
    "server": {"port": "port"},
}


# ---------------------------------------------------------------- YAML mechanics


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping (empty file -> {})."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return base updated by overlay, recursing into nested mappings. Neither input is mutated."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_extends(path: str | Path, _seen: tuple = ()) -> dict:
    """Read a YAML file, overlaying it on the file named by its `extends` key (relative to it)."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"{path}: circular extends chain")
    data = read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{path}: extends must be a file name, got {parent!r}")
    base = load_yaml_with_extends(path.parent / parent, (*_seen, path))
    return deep_merge(base, data)


def flatten(data: dict, schema: dict, where: str) -> dict:
    """Map a nested config dict onto flat setting names, rejecting unknown keys and skipping nulls."""
    flat = {}
    for key, value in data.items():
        if key not in schema:
            raise ValueError(f"{where}: unknown key '{key}' (expected one of {', '.join(schema)})")
        spec = schema[key]
        if isinstance(spec, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: '{key}' must be a mapping of {', '.join(spec)}")
            for sub, sub_value in value.items():
                if sub not in spec:
                    raise ValueError(
                        f"{where}: unknown key '{key}.{sub}' (expected one of {', '.join(spec)})"
                    )
                if sub_value is not None:
                    flat[spec[sub]] = sub_value
        elif value is not None:
            flat[spec] = value
    return flat


def spec_label(spec) -> str | None:
    """Short display form of a matrix spec, e.g. 'rotation_z_deg=20' (None for no spec)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        parts = []
        for key, value in spec.items():
            parts.append(f"{key}=literal" if isinstance(value, list) else f"{key}={value}")
        return ", ".join(parts)
    return "literal"


# ---------------------------------------------------------------- validation


def _fail(where: str, message: str):
    raise ValueError(f"{where}: {message}")


def build_control(flat: dict, where: str) -> ControlSettings:
    """ControlSettings from flat settings, with ranges checked and matrix specs resolved."""
    control = ControlSettings(**{k: flat[k] for k in ControlSettings.__dataclass_fields__ if k in flat})
    if control.mode not in MODES:
        _fail(where, f"control.mode must be one of {', '.join(MODES)}, got {control.mode!r}")
    if not isinstance(control.n_guided_steps, int) or control.n_guided_steps < 0:
        _fail(where, "control.n_guided_steps must be a non-negative integer")
    if control.n_reversal_steps is not None and (
        not isinstance(control.n_reversal_steps, int) or control.n_reversal_steps < 1
    ):
        _fail(
            where, "control.n_reversal_steps must be a positive integer or null (null = all the way to noise)"
        )
    if not isinstance(control.input_noise, int | float) or control.input_noise < 0:
        _fail(where, "control.input_noise must be >= 0")
    if control.max_steps is not None and (not isinstance(control.max_steps, int) or control.max_steps < 1):
        _fail(where, "control.max_steps must be a positive integer or null")
    control.corruption_matrix = build_corruption(control.corruption, f"{where}: control.corruption")
    control.reversal_adapter_matrix = build_reversal_adapter(
        control.reversal_adapter, control.corruption_matrix, f"{where}: control.reversal_adapter"
    )
    return control


def _session_from_flat(flat: dict, where: str, defaults: SessionSettings) -> SessionSettings:
    session = SessionSettings(
        policy_path=flat.get("policy_path", defaults.policy_path),
        n_action_steps=flat.get("n_action_steps", defaults.n_action_steps),
        compile=flat.get("compile", defaults.compile),
        suite=flat.get("suite", defaults.suite),
        task_id=flat.get("task_id", defaults.task_id),
        port=flat.get("port", defaults.port),
        output_dir=Path(flat.get("output_dir", defaults.output_dir)),
        control=build_control(flat, where),
    )
    if not isinstance(session.n_action_steps, int) or session.n_action_steps < 1:
        _fail(where, "policy.n_action_steps must be a positive integer")
    return session


def load_interactive_settings(path: str | Path | None, overrides: dict) -> SessionSettings:
    """Settings for interactive.py: file (optional) then CLI overrides (flat names, None = unset)."""
    where = str(path) if path is not None else "command line"
    flat = flatten(load_yaml_with_extends(path), INTERACTIVE_SCHEMA, where) if path is not None else {}
    flat.update({k: v for k, v in overrides.items() if v is not None})
    return _session_from_flat(flat, where, SessionSettings())


def parse_set(item: str) -> tuple[list[str], object]:
    """Parse a `--set KEY=VALUE` item: a dotted YAML path and a YAML-parsed value."""
    key, sep, raw = item.partition("=")
    key = key.strip()
    if not sep or not key or not all(part.isidentifier() for part in key.split(".")):
        raise ValueError(
            f"--set expects KEY=VALUE with a dotted key such as control.n_guided_steps=4, got {item!r}"
        )
    return key.split("."), yaml.safe_load(raw) if raw.strip() else None


def sets_to_tree(items: list[str]) -> dict:
    """Nested mapping from `--set` items, later items winning: ['control.n_guided_steps=4'] -> {'control': {...}}."""
    tree: dict = {}
    for item in items:
        parts, value = parse_set(item)
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {item}: '{part}' is not a section")
        node[parts[-1]] = value
    return tree


def set_label(items: list[str]) -> str:
    """Filesystem-safe suffix describing `--set` items: control.n_guided_steps=4 -> n_guided_steps-4."""
    parts = []
    for item in items:
        keys, value = parse_set(item)
        text = "null" if value is None else re.sub(r"[^A-Za-z0-9._=,+-]", "", str(value).replace(": ", "="))
        parts.append(f"{keys[-1]}-{text[:40]}")
    return "_".join(parts)


def load_experiment_settings(
    path: str | Path, overrides: dict, sets: list[str] | None = None
) -> ExperimentSettings:
    """Settings for experiment.py: the condition file (with extends), `--set` items, then flat CLI overrides."""
    path = Path(path)
    where = str(path)
    data = load_yaml_with_extends(path)
    if sets:
        data = deep_merge(data, sets_to_tree(sets))
    flat = flatten(data, EXPERIMENT_SCHEMA, where)
    flat.update({k: v for k, v in overrides.items() if v is not None})
    defaults = ExperimentSettings(name=path.stem, session=SessionSettings())
    session_defaults = SessionSettings(output_dir=defaults.output_dir)
    session_flat = {
        k: v for k, v in flat.items() if k not in ("n_trials", "seed", "task_order", "task_ids", "prompt")
    }
    settings = ExperimentSettings(
        name=path.stem,
        session=_session_from_flat(session_flat, where, session_defaults),
        n_trials=flat.get("n_trials", defaults.n_trials),
        seed=flat.get("seed", defaults.seed),
        task_order=flat.get("task_order", defaults.task_order),
        output_dir=Path(flat.get("output_dir", defaults.output_dir)),
        task_ids=flat.get("task_ids", defaults.task_ids),
        prompt=flat.get("prompt", defaults.prompt),
    )
    if not isinstance(settings.n_trials, int) or settings.n_trials < 1:
        _fail(where, f"experiment.n_trials must be a positive integer, got {settings.n_trials!r}")
    if settings.task_order not in TASK_ORDERS:
        _fail(
            where,
            f"experiment.task_order must be one of {', '.join(TASK_ORDERS)}, got {settings.task_order!r}",
        )
    if not isinstance(settings.prompt, str) or not settings.prompt.strip():
        _fail(where, "prompt must be a non-empty string (or `task` to use the scene's own instruction)")
    if settings.task_ids is not None and (
        not isinstance(settings.task_ids, list)
        or not settings.task_ids
        or not all(isinstance(t, int) for t in settings.task_ids)
    ):
        _fail(where, "scene.task_ids must be a non-empty list of task ids (or null for all of them)")
    return settings
