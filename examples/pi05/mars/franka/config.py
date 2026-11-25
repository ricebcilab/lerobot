"""``configs/franka.toml`` -> frozen dataclasses: what a run is, checked before anything is loaded.

The example is driven by a config file, so the file is the one place a station address, a camera
key, a checkpoint, the operator's gains and the mode knobs are read. This module parses it with
``tomllib`` (standard library), rejects unknown keys, resolves the example's own relative paths,
and refuses knob combinations a mode cannot read. It is torch-free and c3po-free so a config can be
checked before a checkpoint, a camera or an arm is anywhere in sight.

TOML has no ``null``: an absent key means "the built-in default", and the built-in defaults are the
harmless ones (corruption off, no noise). The explicit offs are the strings ``corruption = "none"``
(equivalent to the absent default, so a variant file can turn off what it inherits) and
``mode = "none"``. Relative paths (``policy.stats``, ``steering.adapter_user``/``adapter_task``,
``recording.dir``, ``states.dir``, ``user.states``/``out``/``checkpoint``) resolve against this
example's directory, not the caller's working directory. Phase B's ``[user]`` and ``[training]``
tables are the user-module trainer's knobs (DESIGN.md, "Phase B"); the ``null`` defaults there
(``horizon``, ``step_rank``, ``correct_by``) are the absent key, since TOML cannot spell a null.
Phase C's ``[task]`` table is the task-module trainer's (DESIGN.md, "Phase C"): the demonstrations
dataset, the output, and the module's scene and width knobs; ``[training]`` is shared with it.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from bridge import ARM_DIMS, TWIST_DIMS

EXAMPLE_DIR = Path(__file__).parent
"""A relative path in a config file means a path beside this file."""

STEERING_MODES = ("none", "flow_control", "frs")

_SECTIONS = frozenset(
    {
        "station",
        "cameras",
        "policy",
        "command",
        "readout",
        "steering",
        "modes",
        "spacemouse",
        "recording",
        "states",
        "user",
        "task",
        "training",
    }
)
"""The config file's sections: station, cameras, policy, command, readout, steering, modes,
spacemouse, recording, states, user, task, training."""

_MODE_KNOBS = {
    "none": frozenset(),
    "flow_control": frozenset({"tau", "lam", "ic_gain"}),
    "frs": frozenset({"n_reversal_steps", "pinned_steps", "adapter_matrix", "adapter_user", "adapter_task"}),
}
"""The knobs each steering mode reads; a mode table may not set another mode's knobs."""


# ---------------------------------------------------------------- the config tree


@dataclass(frozen=True)
class StationConfig:
    host: str = "10.42.0.1"
    port: int = 9090
    arm: str = "follower"
    rate: float = 15.0


@dataclass(frozen=True)
class CameraConfig:
    exterior: str = "scene_zed"
    wrist: str = "wrist_zed"
    third: str | None = None  # absent: the third image key is omitted, not filled with zeros


@dataclass(frozen=True)
class PolicyConfig:
    path: str = "lerobot/pi05_droid"
    stats: Path = EXAMPLE_DIR / "assets" / "droid" / "norm_stats.json"
    device: str = "cuda"
    n_action_steps: int = 8
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class CommandConfig:
    translation_gain: float = 0.11  # m/s at full deflection (FC App. E's ~11 cm/s)
    rotation_gain: float = 0.5  # rad/s at full deflection
    twist_length: float = 0.2  # m: the mixed-unit scale of bridge.joint_velocity
    deadband: float = 0.02  # action units below which the operator is idle
    gain: str | float = "auto"  # "auto" = 1 / ||F||_2 with corruption on, else 1.0
    q0: tuple[float, ...] = ()  # () = the station's reset pose


@dataclass(frozen=True)
class ReadoutConfig:
    corruption: dict[str, Any] | None = None  # None = off; the TOML string "none" also parses here
    noise: float = 0.0  # std per joint, in action-standard-deviation units
    seed: int = 0


@dataclass(frozen=True)
class SteeringConfig:
    """One `[steering]`/`[modes.*]` table: the knobs every mode may inherit and the ones it uses.

    ``mode`` is set on a ``[modes.*]`` table only; the ``[steering]`` defaults leave it "none".
    ``tau``/``lam``/``ic_gain`` are FC's (Kao et al.'s DROID numbers); ``n_reversal_steps`` is
    FRS's full reversal and ``pinned_steps`` its executed prefix (Tang et al.).
    """

    mode: str = "none"
    tau: int = 8
    lam: int = 1
    ic_gain: float = 15.0
    n_reversal_steps: int = 10
    pinned_steps: int = 8
    adapter_matrix: str | tuple[tuple[float, ...], ...] | None = None
    adapter_user: Path | None = None
    adapter_task: Path | None = None


@dataclass(frozen=True)
class RecordingConfig:
    dir: Path = EXAMPLE_DIR / "outputs" / "mars_franka"


@dataclass(frozen=True)
class StatesConfig:
    """The `[states]` table: where a session's chunk-boundary states land (DESIGN.md, Phase B)."""

    dir: Path = EXAMPLE_DIR / "assets" / "states"


@dataclass(frozen=True)
class UserConfig:
    """The `[user]` table: the user-module trainer's input, output and module shape (DESIGN.md, Phase B).

    ``states`` names a states file or the directory holding one (``<dir>/states.npz``, which is what
    ``--record-states NAME`` writes); ``train-user`` writes ``<out>/user.pt`` and ``<out>/train_log.json``;
    ``checkpoint`` is where ``validate-user`` and ``[modes.mars-u]`` read the trained module from.
    ``sigma`` is the readout noise range in action-standard-deviation units (the study's), drawn per cue;
    ``horizon``, ``step_rank`` and ``correct_by`` are None unless a variant config sets them.
    """

    states: Path = EXAMPLE_DIR / "assets" / "states" / "pilot"
    out: Path = EXAMPLE_DIR / "assets" / "user" / "rotz20"
    checkpoint: Path = EXAMPLE_DIR / "assets" / "user" / "rotz20" / "user.pt"
    cues: int = 3  # cue draws per state: the calibration session's length
    sigma: tuple[float, float] = (0.2, 0.5)  # the readout's noise range, in action-std units
    horizon: int | None = None  # None = the checkpoint's chunk_size
    d: int = 256  # the module's width
    tdim: int = 64  # its time embedding's width
    time_embedding: str = "mars"  # "mars" (sinusoidal) or "pi05" (the policy's own)
    step_rank: int | None = None  # None = every chunk step's matrix free
    cross_step_rank: int = 0  # 0 = no coupling between the chunk's steps
    additive: bool = False  # U v + b: the module supplies a velocity term of its own
    correct_by: int | None = None  # the reversal step the correction is complete by; None = the last


@dataclass(frozen=True)
class TaskConfig:
    """The `[task]` table: the task-module trainer's input, output and module shape (DESIGN.md, Phase C).

    ``demos`` names the downloaded ``teleop --record`` dataset the module is trained on; ``train-task``
    writes ``<out>/task.pt`` and ``<out>/train_log.json``; ``checkpoint`` is where ``[modes.mars-t]`` and
    the composed ``[modes.mars]`` read the trained module from (Phase C4). ``token_source`` is the scene
    source the module records (``vlm_image`` only in this phase), ``render_size`` the side of the square
    scene render the VLM's tokens come from, and ``scene_rank`` the width of the fixed token reduction
    (None: full width). ``d``/``tdim`` are the module's width and time embedding's width; ``lambda_id`` is
    the identity pull in main's field loss. ``horizon`` is never a knob: it is the checkpoint's chunk.
    """

    demos: Path = EXAMPLE_DIR / "assets" / "demos" / "pilot"
    out: Path = EXAMPLE_DIR / "assets" / "task" / "pilot"
    checkpoint: Path = EXAMPLE_DIR / "assets" / "task" / "pilot" / "task.pt"
    token_source: str = "vlm_image"  # the scene source the module records: vlm_image only in this phase
    render_size: int = 224  # the side of the square scene render the VLM's tokens come from
    scene_rank: int | None = None  # None = full-width (2048) VLM tokens
    d: int = 128  # the module's width
    tdim: int = 64  # its time embedding's width
    lambda_id: float = 1e-3  # the pull of A, b toward the identity, in main's field loss


@dataclass(frozen=True)
class TrainingConfig:
    """The `[training]` table: the trainer's loop, split and device (DESIGN.md, Phase B)."""

    steps: int = 3000
    batch: int = 8
    lr: float = 1e-3
    val_frac: float = 0.1
    seed: int = 0
    log_every: int = 100
    val_depths: tuple[int, ...] = (10, 4)  # the reversal depths validate-user runs
    device: str = "cuda"  # where the checkpoint and the module live while training


@dataclass(frozen=True)
class SpaceMouseConfig:
    """The operator's ``[spacemouse]`` table: how the device is found and its six channels mapped.

    ``path`` and ``hid_id`` are ``SpaceMouseTwist``'s; the empty string means search (the library's
    default, ``SPACEMOUSE_HID_ID`` when unset). ``source`` maps the device's six channels
    (0: x, 1: y, 2: z, 3: rx, 4: ry, 5: rz) onto the twist's axes and ``sign`` flips them. The
    scales are the device's counts per unit of full deflection. **The defaults are unverified on
    hardware**: ``rotation_full_scale`` in particular is a guess, and the probe measures the real
    scales and the channel mapping.
    """

    path: str = ""  # "" = search for the device
    hid_id: str = ""  # "" = search; otherwise a HID id such as 0000046D:0000C62B
    full_scale: float = 350.0
    rotation_full_scale: float = 350.0
    source: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # the device channels, in the twist's axis order
    sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Config:
    station: StationConfig = field(default_factory=StationConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    modes: Mapping[str, SteeringConfig] = field(default_factory=lambda: MappingProxyType({}))
    spacemouse: SpaceMouseConfig = field(default_factory=SpaceMouseConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    states: StatesConfig = field(default_factory=StatesConfig)
    user: UserConfig = field(default_factory=UserConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------- value checks


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where}: expected an integer of at least {minimum}, got {value!r}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _positive(value: Any, where: str) -> float:
    number = _number(value, where)
    if not number > 0:
        raise ValueError(f"{where}: expected a positive number, got {value!r}")
    return number


def _non_negative(value: Any, where: str) -> float:
    number = _number(value, where)
    if number < 0:
        raise ValueError(f"{where}: expected a non-negative number, got {value!r}")
    return number


def _fraction(value: Any, where: str) -> float:
    number = _number(value, where)
    if not 0 < number <= 1:
        raise ValueError(f"{where}: expected a number in (0, 1], got {value!r}")
    return number


def _flag(value: Any, where: str) -> bool:
    """A TOML boolean (``true``/``false``): 1 and 0 are integers, not flags, and are refused."""
    if not isinstance(value, bool):
        raise ValueError(f"{where}: expected true or false, got {value!r}")
    return value


def _pair(value: Any, where: str) -> tuple[float, float]:
    """A ``[low, high]`` range of finite non-negative numbers, low <= high."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{where}: expected a [low, high] pair, got {value!r}")
    low, high = (_non_negative(item, f"{where}[{index}]") for index, item in enumerate(value))
    if low > high:
        raise ValueError(f"{where}: expected low <= high, got {value!r}")
    return low, high


def _time_embedding(value: Any, where: str) -> str:
    if value not in ("mars", "pi05"):
        raise ValueError(f"{where}: expected 'mars' or 'pi05', got {value!r}")
    return value


def _depths(value: Any, where: str) -> tuple[int, ...]:
    """A non-empty list of reversal depths (the positive Euler-step counts validate-user runs)."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where}: expected a non-empty list of reversal depths, got {value!r}")
    return tuple(_integer(item, f"{where}[{index}]") for index, item in enumerate(value))


def _port(value: Any, where: str) -> int:
    port = _integer(value, where)
    if port > 65535:
        raise ValueError(f"{where}: expected a port number, got {value!r}")
    return port


def _path(value: Any, where: str) -> Path:
    """A path as written: relative ones resolve against the example directory."""
    path = Path(_text(value, where))
    return path if path.is_absolute() else EXAMPLE_DIR / path


def _gain(value: Any, where: str) -> str | float:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"{where}: expected 'auto' or a number in (0, 1], got {value!r}")
        return value
    return _fraction(value, where)


def _q0(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where}: expected a list of {ARM_DIMS} joint angles or [], got {value!r}")
    angles = tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))
    if len(angles) not in (0, ARM_DIMS):
        raise ValueError(
            f"{where}: expected {ARM_DIMS} joint angles or [] (the reset pose), got {len(angles)}"
        )
    return angles


def _corruption(value: Any, where: str) -> dict[str, Any] | None:
    """A matrix/random corruption spec, or None for the explicit off (TOML has no null)."""
    if isinstance(value, str):
        if value != "none":
            raise ValueError(f"{where}: the only string corruption is 'none', got {value!r}")
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: expected a corruption table or 'none', got {value!r}")
    return dict(value)


def _adapter_matrix(value: Any, where: str) -> str | tuple[tuple[float, ...], ...]:
    if isinstance(value, str):
        if value not in ("corruption", "corruption_inverse"):
            raise ValueError(
                f"{where}: expected 'corruption', 'corruption_inverse' or {ARM_DIMS} rows, got {value!r}"
            )
        return value
    shaped = isinstance(value, (list, tuple)) and len(value) == ARM_DIMS
    if not shaped or any(not isinstance(row, (list, tuple)) or len(row) != ARM_DIMS for row in value):
        raise ValueError(f"{where}: expected {ARM_DIMS} rows of {ARM_DIMS} numbers, got {value!r}")
    return tuple(tuple(_number(item, where) for item in row) for row in value)


def _search_text(value: Any, where: str) -> str:
    """A device path or HID id as written: the empty string means search for the device."""
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a string ('' = search), got {value!r}")
    return value


def _device_channels(value: Any, where: str) -> tuple[int, ...]:
    """The six device channels a twist reads, each an index into the device's six channels."""
    if not isinstance(value, (list, tuple)) or len(value) != TWIST_DIMS:
        raise ValueError(f"{where}: expected {TWIST_DIMS} device channel indexes, got {value!r}")
    channels = tuple(_integer(item, f"{where}[{index}]", 0) for index, item in enumerate(value))
    if any(channel >= TWIST_DIMS for channel in channels):
        raise ValueError(f"{where}: indexes the device channels 0-{TWIST_DIMS - 1}, got {value!r}")
    return channels


def _device_signs(value: Any, where: str) -> tuple[float, ...]:
    """One sign per device channel (a negative one flips that axis)."""
    if not isinstance(value, (list, tuple)) or len(value) != TWIST_DIMS:
        raise ValueError(f"{where}: expected {TWIST_DIMS} signs, got {value!r}")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))


def _reject_unknown(table: dict, known: set[str] | frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key {unknown[0]!r} (expected one of {', '.join(sorted(known))})")


# ---------------------------------------------------------------- the sections


def _build(cls, table: Any, conversions: Mapping[str, Any], where: str):
    """A section dataclass from a TOML table, converting only the keys the file sets."""
    if not isinstance(table, dict):
        raise ValueError(f"{where}: expected a table, got {type(table).__name__}")
    _reject_unknown(table, frozenset(conversions), where)
    return cls(
        **{key: convert(table[key], f"{where}.{key}") for key, convert in conversions.items() if key in table}
    )


def _station(table: Any) -> StationConfig:
    return _build(
        StationConfig, table, {"host": _text, "port": _port, "arm": _text, "rate": _positive}, "[station]"
    )


def _cameras(table: Any) -> CameraConfig:
    return _build(CameraConfig, table, {"exterior": _text, "wrist": _text, "third": _text}, "[cameras]")


def _policy(table: Any) -> PolicyConfig:
    return _build(
        PolicyConfig,
        table,
        {"path": _text, "stats": _path, "device": _text, "n_action_steps": _integer, "dtype": _text},
        "[policy]",
    )


def _command(table: Any) -> CommandConfig:
    return _build(
        CommandConfig,
        table,
        {
            "translation_gain": _non_negative,
            "rotation_gain": _non_negative,
            "twist_length": _positive,
            "deadband": _non_negative,
            "gain": _gain,
            "q0": _q0,
        },
        "[command]",
    )


def _readout(table: Any) -> ReadoutConfig:
    return _build(
        ReadoutConfig,
        table,
        {
            "corruption": _corruption,
            "noise": _non_negative,
            "seed": lambda value, where: _integer(value, where, 0),
        },
        "[readout]",
    )


def _spacemouse(table: Any) -> SpaceMouseConfig:
    return _build(
        SpaceMouseConfig,
        table,
        {
            "path": _search_text,
            "hid_id": _search_text,
            "full_scale": _positive,
            "rotation_full_scale": _positive,
            "source": _device_channels,
            "sign": _device_signs,
        },
        "[spacemouse]",
    )


def _user(table: Any) -> UserConfig:
    """The `[user]` trainer table: the input, the output and the module's shape and knobs."""
    return _build(
        UserConfig,
        table,
        {
            "states": _path,
            "out": _path,
            "checkpoint": _path,
            "cues": _integer,
            "sigma": _pair,
            "horizon": _integer,
            "d": _integer,
            "tdim": _integer,
            "time_embedding": _time_embedding,
            "step_rank": _integer,
            "cross_step_rank": lambda value, where: _integer(value, where, 0),
            "additive": _flag,
            "correct_by": _integer,
        },
        "[user]",
    )


def _task(table: Any) -> TaskConfig:
    """The `[task]` trainer table: the demonstrations, the output and the module's scene and shape."""
    return _build(
        TaskConfig,
        table,
        {
            "demos": _path,
            "out": _path,
            "checkpoint": _path,
            "token_source": _text,
            "render_size": _integer,
            "scene_rank": _integer,
            "d": _integer,
            "tdim": _integer,
            "lambda_id": _non_negative,
        },
        "[task]",
    )


def _training(table: Any) -> TrainingConfig:
    """The `[training]` table: the loop, the split and the device."""
    return _build(
        TrainingConfig,
        table,
        {
            "steps": _integer,
            "batch": _integer,
            "lr": _positive,
            "val_frac": _fraction,
            "seed": lambda value, where: _integer(value, where, 0),
            "log_every": _integer,
            "val_depths": _depths,
            "device": _text,
        },
        "[training]",
    )


_STEERING_KNOBS = {
    "tau": _integer,
    "lam": _integer,
    "ic_gain": _positive,
    "n_reversal_steps": _integer,
    "pinned_steps": _integer,
    "adapter_matrix": _adapter_matrix,
    "adapter_user": _path,
    "adapter_task": _path,
}
"""The knobs a `[steering]` or `[modes.*]` table may set, and how each is read."""


def _steering(table: Any) -> SteeringConfig:
    """The `[steering]` defaults: every knob, no ``mode`` (a mode is named by a mode table)."""
    return _build(SteeringConfig, table, _STEERING_KNOBS, "[steering]")


def _mode(value: Any, where: str) -> str:
    if value not in STEERING_MODES:
        raise ValueError(f"{where}: expected one of {STEERING_MODES}, got {value!r}")
    return value


def _modes(table: Any, base: SteeringConfig) -> Mapping[str, SteeringConfig]:
    """One resolved ``SteeringConfig`` per named mode: the `[steering]` defaults overridden by the table."""
    if not isinstance(table, dict):
        raise ValueError(f"[modes]: expected a table of mode names, got {type(table).__name__}")
    resolved = {}
    for name, override in table.items():
        where = f"[modes.{name}]"
        if not isinstance(override, dict):
            raise ValueError(f"{where}: expected a table, got {type(override).__name__}")
        if "mode" not in override:
            raise ValueError(f"{where}: a mode names one of {STEERING_MODES} in 'mode'")
        mode = _mode(override["mode"], f"{where}.mode")
        knobs = {key: value for key, value in override.items() if key != "mode"}
        _reject_unknown(knobs, frozenset(_STEERING_KNOBS), where)
        stray = sorted(set(knobs) - _MODE_KNOBS[mode])
        if stray:
            allowed = ", ".join(sorted(_MODE_KNOBS[mode])) or "only its own 'mode'"
            raise ValueError(f"{where}: mode {mode!r} does not read {stray[0]!r}; it reads {allowed}")
        knobs = {
            key: convert(knobs[key], f"{where}.{key}")
            for key, convert in _STEERING_KNOBS.items()
            if key in knobs
        }
        resolved[name] = replace(base, mode=mode, **knobs)
    return MappingProxyType(resolved)


# ---------------------------------------------------------------- reading a file


def load_config(path: str | Path) -> Config:
    """The config in ``path``: the built-in defaults for every absent key, unknown keys rejected."""
    path = Path(path)
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a table at the top level, got {type(data).__name__}")
    _reject_unknown(data, _SECTIONS, str(path))
    steering = _steering(data.get("steering", {}))
    return Config(
        station=_station(data.get("station", {})),
        cameras=_cameras(data.get("cameras", {})),
        policy=_policy(data.get("policy", {})),
        command=_command(data.get("command", {})),
        readout=_readout(data.get("readout", {})),
        steering=steering,
        modes=_modes(data.get("modes", {}), steering),
        spacemouse=_spacemouse(data.get("spacemouse", {})),
        recording=_build(RecordingConfig, data.get("recording", {}), {"dir": _path}, "[recording]"),
        states=_build(StatesConfig, data.get("states", {}), {"dir": _path}, "[states]"),
        user=_user(data.get("user", {})),
        task=_task(data.get("task", {})),
        training=_training(data.get("training", {})),
    )
