"""The Franka station's YAML settings, as frozen dataclasses.

What changes between one Franka and the next is a station address, two camera keys, a checkpoint, an arm's limits
and which MARS arm is being run. That belongs in a file, not in the code, and a per-arm file should be able to
say only what it changes: `extends: base.yaml` overlays it on a base file, section by section.

This module is torch-free on purpose. It hands the runner the kwargs for `lerobot.policies.steering.SteeringConfig`
as a plain dict, so a condition file can be checked before a checkpoint, a camera or an arm is anywhere in
sight. It also stops at the door of the hardware: it does not know the station's joint names or its camera keys.
The station builds an `ArmSpec` from its own names and `arm_overrides()`; the runner asks `camera_names()` for the
keys the probe has filled in; `steering_config()` supplies everything the policy needs and nothing it does not.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from interfaces import ARM_DIMS, GRIPPER_DIM

MODE_MAP = {
    "teleop": "none",  # the operator's stick alone, no policy
    "floor": "none",  # the generic-prompt policy, the command ignored
    "ceiling": "none",  # the task-prompted policy, the command ignored
    "fc": "flow_control",
    "frs": "frs",
    "mars_fixed": "frs",  # FRS with a fixed MARS adapter matrix
    "mars_learned": "frs",  # FRS with a learned user module
}
"""The arm each condition file names, and the `SteeringConfig.mode` it is."""


# ---------------------------------------------------------------- reading a file


def read_yaml(path: str | Path) -> dict:
    """Read a YAML mapping; an empty file is an empty mapping."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def deep_merge(base: dict, overlay: dict) -> dict:
    """`base` updated by `overlay`, section by section; neither input is mutated.

    A section in the overlay replaces only the keys it names, which is what makes a per-arm file short: it sets
    `steering.mode` and keeps the base's `tau`. A null in the overlay survives the merge so that the loader can
    read it as "use the built-in default".
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_with_extends(path: str | Path, _seen: tuple = ()) -> dict:
    """A YAML file overlaid on the file its `extends` key names, relative to itself."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"{path}: circular extends chain")
    data = read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{path}: extends must be a file name, got {parent!r}")
    return deep_merge(load_yaml_with_extends(path.parent / parent, (*_seen, path)), data)


# ---------------------------------------------------------------- value checks


def _text(value, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _number(value, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None


def _positive(value, key: str) -> float:
    """A positive number, naming the key whether the value is the wrong type or the wrong sign."""
    number = _number(value, key)
    if not number > 0:
        raise ValueError(f"{key} must be positive, got {value!r}")
    return number


def _non_negative(value, key: str) -> float:
    number = _number(value, key)
    if number < 0:
        raise ValueError(f"{key} must not be negative, got {value!r}")
    return number


def _count(value, key: str, minimum: int = 1) -> int:
    """An integer of at least `minimum` (a bool is an int in Python and is not a count here)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}, got {value!r}")
    return value


def _joint_values(values, key: str) -> tuple[float, ...]:
    """One number per joint; the length is checked so a six-joint arm cannot pass for a Franka."""
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be {ARM_DIMS} numbers, got {values!r}") from None
    if len(numbers) != ARM_DIMS:
        raise ValueError(f"{key} must name {ARM_DIMS} joints, got {len(numbers)}")
    return numbers


def _gripper_range(values) -> tuple[float, float]:
    try:
        low, high = (float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError(f"arm.gripper_range must be (low, high), got {values!r}") from None
    if not high > low:
        raise ValueError(f"arm.gripper_range must be (low, high) with high > low, got {values!r}")
    return low, high


def _matrix(value):
    """The fixed adapter matrix as given: a square of `ARM_DIMS`, which is the width the arm's adapter acts on."""
    try:
        rows = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError):
        raise ValueError(
            f"steering.adapter_matrix must be {ARM_DIMS} rows of {ARM_DIMS} numbers, got {value!r}"
        ) from None
    if len(rows) != ARM_DIMS or any(len(row) != ARM_DIMS for row in rows):
        raise ValueError(
            f"steering.adapter_matrix must be {ARM_DIMS} rows of {ARM_DIMS} numbers, got {len(rows)} rows"
        )
    return rows


# ---------------------------------------------------------------- the sections


@dataclass(frozen=True)
class StationSettings:
    """Where the station listens, and which arm to drive (`None`: the manifest's only one)."""

    host: str = "10.42.0.1"
    port: int = 9090
    arm: str | None = None
    rate: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _text(self.host, "station.host"))
        object.__setattr__(self, "port", _count(self.port, "station.port"))
        if self.rate is not None:
            object.__setattr__(self, "rate", _positive(self.rate, "station.rate"))
        if self.arm is not None:
            object.__setattr__(self, "arm", _text(self.arm, "station.arm"))


@dataclass(frozen=True)
class CameraSettings:
    """The two observation keys, null until the station's probe names them."""

    exterior: str | None = None
    wrist: str | None = None

    def __post_init__(self) -> None:
        for key in ("exterior", "wrist"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, _text(getattr(self, key), f"cameras.{key}"))


@dataclass(frozen=True)
class PolicySettings:
    """The checkpoint, its prompt for the floor arm, and the prefix of its chunk that is executed."""

    path: str = "lerobot/pi05_droid"
    prompt_generic: str = "do something"
    n_action_steps: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _text(self.path, "policy.path"))
        if self.device is not None:
            object.__setattr__(self, "device", _text(self.device, "policy.device"))
        object.__setattr__(self, "prompt_generic", _text(self.prompt_generic, "policy.prompt_generic"))
        if self.n_action_steps is not None:
            object.__setattr__(self, "n_action_steps", _count(self.n_action_steps, "policy.n_action_steps"))


@dataclass(frozen=True)
class CommandSettings:
    """The two speeds a full stick deflection asks for: a joint rate, and the flange speed before the Jacobian.

    Both must be positive: `command_from_qdot` divides by `qdot_full`, and a zero would make every command
    infinite rather than full deflection.
    """

    qdot_full: float = 0.5
    twist_full: float = 0.11

    def __post_init__(self) -> None:
        object.__setattr__(self, "qdot_full", _positive(self.qdot_full, "command.qdot_full"))
        object.__setattr__(self, "twist_full", _positive(self.twist_full, "command.twist_full"))


@dataclass(frozen=True)
class ArmSettings:
    """The arm's limits, and the gripper's travel.

    `q_min`/`q_max` are null when the station's own limits apply; the `qdot_max` cap, the gripper's range and the
    position a closed gripper takes are always the config's, because they are safety choices rather than
    kinematics. The cap must cover every joint, so a short list is a mistake rather than a default.
    """

    q_min: tuple[float, ...] | None = None
    q_max: tuple[float, ...] | None = None
    qdot_max: tuple[float, ...] = (0.5,) * ARM_DIMS
    gripper_range: tuple[float, float] = (0.0, 1.0)
    gripper_closed_at: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "qdot_max", _joint_values(self.qdot_max, "arm.qdot_max"))
        if any(cap <= 0 for cap in self.qdot_max):
            raise ValueError(f"arm.qdot_max must be positive for every joint, got {self.qdot_max!r}")
        object.__setattr__(self, "gripper_range", _gripper_range(self.gripper_range))
        object.__setattr__(
            self, "gripper_closed_at", _number(self.gripper_closed_at, "arm.gripper_closed_at")
        )
        low, high = self.gripper_range
        if not low <= self.gripper_closed_at <= high:
            raise ValueError(
                f"arm.gripper_closed_at must lie in arm.gripper_range {self.gripper_range}, "
                f"got {self.gripper_closed_at}"
            )
        if (self.q_min is None) != (self.q_max is None):
            raise ValueError(
                "arm.q_min and arm.q_max must be set together, or both null for the station's limits"
            )
        if self.q_min is not None:
            q_min = _joint_values(self.q_min, "arm.q_min")
            q_max = _joint_values(self.q_max, "arm.q_max")
            if any(high <= low for low, high in zip(q_min, q_max, strict=True)):
                raise ValueError("arm.q_max must exceed arm.q_min for every joint")
            object.__setattr__(self, "q_min", q_min)
            object.__setattr__(self, "q_max", q_max)


@dataclass(frozen=True)
class ReadoutSettings:
    """What corrupts the command on its way from the operator: a spec, and noise on every axis.

    The corruption is kept as written and resolved by the readout's own grammar; this module does not need to
    know its forms to hand it on.
    """

    corruption: dict | list | None = None
    input_noise: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_noise", _non_negative(self.input_noise, "readout.input_noise"))


@dataclass(frozen=True)
class SteeringSettings:
    """Which MARS arm is run and the knobs it takes.

    `tau` and `lam` are flow control's, `n_reversal_steps` and `pinned_steps` FRS's, and `ic_gain` flow control's
    noise-prior scale. Their ranges are the steering config's own business: this module carries them, so a
    condition file is checked by the code that knows what the numbers mean.
    """

    mode: str = "frs"
    tau: int = 14
    lam: int = 1
    ic_gain: float | str = 1.0
    n_reversal_steps: int = 5
    pinned_steps: int | None = None
    adapter_user: str | None = None
    adapter_matrix: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODE_MAP:
            raise ValueError(f"steering.mode must be one of {', '.join(MODE_MAP)}, got {self.mode!r}")
        if self.n_reversal_steps is not None:
            object.__setattr__(
                self,
                "n_reversal_steps",
                _count(self.n_reversal_steps, "steering.n_reversal_steps"),
            )
        if self.pinned_steps is not None:
            object.__setattr__(self, "pinned_steps", _count(self.pinned_steps, "steering.pinned_steps"))
        if self.adapter_user is not None:
            object.__setattr__(self, "adapter_user", _text(self.adapter_user, "steering.adapter_user"))
        if self.adapter_matrix is not None:
            object.__setattr__(self, "adapter_matrix", _matrix(self.adapter_matrix))


@dataclass(frozen=True)
class TrialSettings:
    """How many trials, and how long each may run before it counts as a failure."""

    trials: int = 20
    max_steps: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(self, "trials", _count(self.trials, "trial.trials"))
        object.__setattr__(self, "max_steps", _count(self.max_steps, "trial.max_steps"))


SECTIONS = {
    "station": StationSettings,
    "cameras": CameraSettings,
    "policy": PolicySettings,
    "command": CommandSettings,
    "arm": ArmSettings,
    "readout": ReadoutSettings,
    "steering": SteeringSettings,
    "trial": TrialSettings,
}
"""The YAML's sections and the settings each becomes; the order is the order of the documented file."""


def _section(cls, name: str, values: dict, where: str):
    """One YAML section as settings, rejecting unknown keys and reading null as "keep the default"."""
    known = cls.__dataclass_fields__
    given = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"{where}: unknown key '{name}.{key}' (expected one of {', '.join(known)})")
        if value is not None:
            given[key] = value
    try:
        return cls(**given)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from None


@dataclass(frozen=True)
class ExperimentConfig:
    """One condition file: what the runner, the station and the policy are told."""

    station: StationSettings = field(default_factory=StationSettings)
    cameras: CameraSettings = field(default_factory=CameraSettings)
    policy: PolicySettings = field(default_factory=PolicySettings)
    command: CommandSettings = field(default_factory=CommandSettings)
    arm: ArmSettings = field(default_factory=ArmSettings)
    readout: ReadoutSettings = field(default_factory=ReadoutSettings)
    steering: SteeringSettings = field(default_factory=SteeringSettings)
    trial: TrialSettings = field(default_factory=TrialSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """The config a condition file describes, with its `extends` chain resolved and unknown keys rejected."""
        path = Path(path)
        data = load_yaml_with_extends(path)
        where = str(path)
        unknown = [key for key in data if key not in SECTIONS]
        if unknown:
            raise ValueError(f"{where}: unknown key '{unknown[0]}' (expected one of {', '.join(SECTIONS)})")
        sections = {}
        for name, settings in SECTIONS.items():
            values = data.get(name)
            if values is None:
                sections[name] = settings()
                continue
            if not isinstance(values, dict):
                raise ValueError(
                    f"{where}: '{name}' must be a mapping of {', '.join(settings.__dataclass_fields__)}"
                )
            sections[name] = _section(settings, name, values, where)
        return cls(**sections)

    def steering_config(self) -> dict:
        """The kwargs for `SteeringConfig`, in this arm's action layout.

        The layout is the Franka's, not LIBERO's: all seven joints are commands and the gripper is the eighth
        action dim, left to the policy because the operator's gripper is a button. The reference step is
        `qdot_full`, so a full deflection is a full step of the FRS reference, and the pinned prefix is the
        executed one unless the file pins its own.
        """
        self.check_adapters()
        steering = self.steering
        return {
            "mode": MODE_MAP[steering.mode],
            "n_dims": ARM_DIMS,
            "gripper_dim": GRIPPER_DIM,
            "operator_dims": None,
            "reference_step": self.command.qdot_full,
            "pinned_steps": (
                self.steering.pinned_steps
                if self.steering.pinned_steps is not None
                else (self.policy.n_action_steps or None)
            ),
            "tau": steering.tau,
            "lam": steering.lam,
            "ic_gain": steering.ic_gain,
            "n_reversal_steps": steering.n_reversal_steps,
            "adapter_user": steering.adapter_user,
            "adapter_matrix": steering.adapter_matrix,
        }

    def check_adapters(self) -> None:
        """Raise unless the arm and its adapters agree.

        An arm that steers nothing has no adapter to use, and one of the two MARS arms is defined by its adapter,
        so a matrix without FRS or a `mars_*` arm without its module is a condition file that says two things at
        once. Both are cheap to catch here rather than at the policy's first step.
        """
        steering = self.steering
        mode = MODE_MAP[steering.mode]
        for key in ("adapter_user", "adapter_matrix"):
            if getattr(steering, key) is not None and mode != "frs":
                raise ValueError(
                    f"steering.{key}: the {steering.mode!r} arm steers nothing, so it has no adapter to use "
                    "(an adapter needs frs, mars_fixed or mars_learned)"
                )
        needs = {
            "mars_fixed": ("adapter_matrix", "the J+ M J matrix at the start pose"),
            "mars_learned": ("adapter_user", "a user.pt checkpoint"),
        }
        if steering.mode in needs:
            key, what = needs[steering.mode]
            if getattr(steering, key) is None:
                raise ValueError(f"steering.mode is {steering.mode!r}, which needs steering.{key} ({what})")

    def arm_overrides(self) -> dict:
        """The `ArmSpec` fields the config fixes, for the station to merge over the arm's own.

        The station knows the joint names and, unless the file says otherwise, the joint limits from its manifest.
        The velocity cap and the gripper's travel are always the config's here, because they are what keeps the
        operator's request inside the arm; `q_min`/`q_max` appear only when the file sets them.
        """
        overrides = {
            "qdot_max": self.arm.qdot_max,
            "gripper_range": self.arm.gripper_range,
            "gripper_closed_at": self.arm.gripper_closed_at,
        }
        if self.arm.q_min is not None:
            overrides["q_min"] = self.arm.q_min
            overrides["q_max"] = self.arm.q_max
        return overrides

    def camera_names(self) -> tuple[str, str]:
        """The `(exterior, wrist)` observation keys, once the station's probe has filled them in."""
        names = []
        for key in ("exterior", "wrist"):
            name = getattr(self.cameras, key)
            if name is None:
                raise ValueError(
                    f"cameras.{key} is null: the runner reads the camera names after the probe has named them"
                )
            names.append(name)
        return names[0], names[1]
