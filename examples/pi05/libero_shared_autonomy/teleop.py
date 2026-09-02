#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Operator input for the LIBERO shared-autonomy runners.

Where the operator's command comes from and what happens to it on the way to
the policy:

    SpaceMouse + keyboard -> CombinedReader -> RecordingReader (raw)
        -> CommandCorruption (M @ x) -> NoisyReader (+ noise)
        -> RecordingReader (served)

`TeleopChain` builds that pipeline; the modes consume `chain.reader`. Every
reader exposes `translation` (shape 3, env units in [-1, 1]) and `gripper`
(-1 open, +1 close), the `TeleopSource` protocol from
`lerobot.policies.pi05.steering`.
"""

import struct
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import yaml

from lerobot.policies.pi05.steering import (
    DEADBAND,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    translation_matrix,
    validate_matrix,
)

__all__ = ["DEADBAND", "GRIPPER_CLOSE", "GRIPPER_OPEN"]  # re-exported for readers of this module

SPACEMOUSE_HID_ID = "0000046D:0000C62B"  # vendor:product as it appears in /sys uevent
AXIS_SCALE = 350.0
# Map device channels (0=x: right push +, 1=y: pull-toward-you push +, 2=z:
# press-down +) onto the arm action dims [Δx, Δy, Δz]. Tuned by feel on the
# real device so each push moves the arm the same way: forward push → forward,
# left push → left, pull up → up. If a direction feels wrong, flip its sign;
# if two axes are crossed, swap their AXIS_SOURCE entries.
AXIS_SOURCE = np.array([1, 0, 2])  # device channel feeding each action dim
AXIS_SIGN = np.array([1.0, -1.0, -1.0])


def find_spacemouse_device(hid_id: str = SPACEMOUSE_HID_ID) -> str:
    """Locate the SpaceMouse /dev/hidraw node via /sys/class/hidraw."""
    for sysdir in sorted(Path("/sys/class/hidraw").glob("hidraw*")):
        uevent = sysdir / "device" / "uevent"
        try:
            if hid_id in uevent.read_text():
                return f"/dev/{sysdir.name}"
        except OSError:
            continue
    raise FileNotFoundError(
        f"No SpaceMouse (HID id {hid_id}) found under /sys/class/hidraw. Is it plugged in?"
    )


def parse_report(data: bytes) -> tuple[str, tuple | int | None]:
    """Parse one raw hidraw report (report id is the first byte).

    Returns ("translation", (x, y, z)) with raw int16 counts,
    ("buttons", bitmask), or ("other", None).
    """
    if len(data) >= 7 and data[0] == 1:
        return "translation", struct.unpack_from("<hhh", data, 1)
    if len(data) >= 2 and data[0] == 3:
        bits = int.from_bytes(data[1:5], "little")
        return "buttons", bits
    return "other", None


def normalize(raw_xyz: tuple[int, int, int]) -> np.ndarray:
    """Raw int16 counts → env-action range [-1, 1] via the AXIS_SOURCE/AXIS_SIGN map."""
    scaled = np.clip(np.array(raw_xyz, dtype=np.float64) / AXIS_SCALE, -1.0, 1.0)
    return scaled[AXIS_SOURCE] * AXIS_SIGN


class SpaceMouseReader:
    """Background thread reading translation + buttons from the SpaceMouse.

    `translation` is the current stick deflection in [-1, 1]^3 (zero at rest —
    the device sends a zeroing report on release). Any button press toggles
    `gripper` between open (-1) and close (+1).
    """

    def __init__(self, device_path: str | None = None):
        self.device_path = device_path or find_spacemouse_device()
        self._file = open(self.device_path, "rb", buffering=0)  # noqa: SIM115
        self._lock = threading.Lock()
        self._translation = np.zeros(3)
        self._gripper = GRIPPER_OPEN
        self._prev_buttons = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                data = self._file.read(64)
            except (OSError, ValueError):  # unplugged or closed
                break
            if not data:
                break
            self._handle(data)

    def _handle(self, data: bytes) -> None:
        kind, payload = parse_report(data)
        if kind == "translation":
            values = normalize(payload)
            with self._lock:
                self._translation = values
        elif kind == "buttons":
            with self._lock:
                pressed = payload & ~self._prev_buttons
                self._prev_buttons = payload
                if pressed:
                    self._gripper = GRIPPER_CLOSE if self._gripper == GRIPPER_OPEN else GRIPPER_OPEN

    @property
    def translation(self) -> np.ndarray:
        with self._lock:
            return self._translation.copy()

    @property
    def gripper(self) -> float:
        with self._lock:
            return self._gripper

    def close(self) -> None:
        self._file.close()


DEFAULT_SPEED = 0.5  # fraction of full-scale deflection a held key produces
FAST_KEY = "shift"  # hold for full-scale (1.0) deflection
STALE_AFTER = 1.0  # seconds without a page update before held keys are dropped

# Each key is a full-scale push on the SpaceMouse's *device* axes (x: right +,
# y: toward-you +, z: press-down +) and goes through normalize(), so
# the keyboard inherits the AXIS_SOURCE/AXIS_SIGN tuning of the physical stick:
# ArrowUp moves the arm exactly like pushing the stick forward.
KEY_PUSH = {
    "arrowup": (0, -1, 0),  # push forward
    "arrowdown": (0, 1, 0),  # pull toward you
    "arrowleft": (-1, 0, 0),
    "arrowright": (1, 0, 0),
    "pageup": (0, 0, -1),  # pull up
    "pagedown": (0, 0, 1),  # press down
    "w": (0, 0, -1),
    "s": (0, 0, 1),
}


class KeyboardReader:
    """Translation + gripper state driven by the set of keys held in the browser.

    `update(held, toggles)` is called by the HTTP server on every key event and
    heartbeat from the page: `held` is the full set of currently pressed keys
    (idempotent, so a lost event cannot stick a key) and `toggles` is the
    page's running count of gripper-toggle presses (Space). If the page stops
    reporting for `stale_after` seconds, translation falls back to zero; the
    gripper keeps its last commanded state.
    """

    def __init__(
        self,
        speed: float = DEFAULT_SPEED,
        stale_after: float = STALE_AFTER,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.speed = speed
        self.stale_after = stale_after
        self._clock = clock
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._translation = np.zeros(3)
        self._gripper = GRIPPER_OPEN
        self._toggles = 0
        self._updated = -np.inf

    def update(self, held: Iterable[str], toggles: int) -> None:
        held = set(held)
        keys = {k.lower() for k in held}
        push = np.zeros(3)
        for key in keys:
            if key in KEY_PUSH:
                push += KEY_PUSH[key]
        scale = 1.0 if FAST_KEY in keys else self.speed
        translation = normalize(tuple(push * AXIS_SCALE)) * scale
        with self._lock:
            self._held = held
            self._translation = translation
            self._updated = self._clock()
            if toggles < self._toggles:  # page reloaded: its counter restarted
                self._toggles = 0
            if (toggles - self._toggles) % 2:
                self._gripper = GRIPPER_CLOSE if self._gripper == GRIPPER_OPEN else GRIPPER_OPEN
            self._toggles = toggles

    def _fresh(self) -> bool:
        return self._clock() - self._updated <= self.stale_after

    @property
    def translation(self) -> np.ndarray:
        with self._lock:
            return self._translation.copy() if self._fresh() else np.zeros(3)

    @property
    def gripper(self) -> float:
        with self._lock:
            return self._gripper

    @property
    def held(self) -> set[str]:
        with self._lock:
            return set(self._held) if self._fresh() else set()


class CombinedReader:
    """Merge several teleop sources (SpaceMouse, keyboard, ...) into one.

    Translation comes from the first source in `sources` deflected beyond
    DEADBAND (so list the SpaceMouse first to give it priority), zero if all
    are idle. The gripper toggles whenever *any* source toggles: each source's
    state is the parity of its own presses, so the merged state is closed iff
    an odd number of sources report closed.
    """

    def __init__(self, sources: Iterable):
        self.sources = list(sources)

    @property
    def translation(self) -> np.ndarray:
        for source in self.sources:
            translation = np.asarray(source.translation, dtype=np.float64)
            if np.max(np.abs(translation)) >= DEADBAND:
                return translation
        return np.zeros(3)

    @property
    def gripper(self) -> float:
        n_closed = sum(source.gripper == GRIPPER_CLOSE for source in self.sources)
        return GRIPPER_CLOSE if n_closed % 2 else GRIPPER_OPEN


class CommandCorruption:
    """Deterministically corrupt a teleop source's translation: x -> M @ x.

    `matrix` is a 3x3 array-like (rows = output axes, columns = input axes) or
    None for identity/off; it can be changed at any time. The product is
    clipped to the env's [-1, 1] action range. Like `NoisyReader`, the
    transform is only applied while the source is commanding beyond DEADBAND,
    so an idle source stays exactly idle whatever M is. `label` is free-form
    (the file the matrix came from) and only used for display. The gripper is
    passed through unchanged.
    """

    def __init__(self, source, matrix=None, label: str | None = None):
        self.source = source
        self.matrix = matrix
        self.label = label

    @property
    def matrix(self) -> np.ndarray | None:
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix = None if value is None else validate_matrix(value, 3, "corruption matrix")

    @property
    def translation(self) -> np.ndarray:
        clean = np.asarray(self.source.translation, dtype=np.float64)
        if self._matrix is None or np.max(np.abs(clean)) < DEADBAND:
            return clean
        return np.clip(self._matrix @ clean, -1.0, 1.0)

    @property
    def gripper(self) -> float:
        return self.source.gripper

    def describe(self) -> str:
        if self._matrix is None:
            return "Command corruption: off."
        rows = "; ".join(" ".join(f"{v:+.2f}" for v in row) for row in self._matrix)
        return f"Command corruption: x/y/z -> M @ x/y/z while you command, M = {self.label} = [{rows}]."


def build_corruption(spec, where: str = "corruption") -> np.ndarray | None:
    """3x3 M from a spec ({rotation_z_deg: d}, {scale: [...]}, {M: rows}, bare rows) or None."""
    return None if spec is None else translation_matrix(spec, where)


def read_matrix_spec(path: str | Path):
    """Read a matrix spec from a YAML file (for the REPL's live `corruption` / `adapter` commands)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"{path}: empty file")
    return data


class NoisyReader:
    """Add isotropic Gaussian noise to a teleop source's translation.

    Every read of `translation` draws an independent N(0, std^2) sample for
    each of x, y and z — consecutive control steps therefore get independent
    noise — and clips the result to the env's [-1, 1] action range. Noise is
    only added while the source is actually commanding (deflected beyond
    DEADBAND): an idle source stays exactly idle, so the shared modes still
    hand control back to the policy when you let go. `std` can be changed at
    any time; 0 disables the noise. The gripper is passed through unchanged.
    """

    def __init__(self, source, std: float = 0.0, rng: np.random.Generator | None = None):
        self.source = source
        self.std = std
        self._rng = rng if rng is not None else np.random.default_rng()

    @property
    def std(self) -> float:
        return self._std

    @std.setter
    def std(self, value: float) -> None:
        value = float(value)
        if not value >= 0.0:  # also rejects NaN
            raise ValueError(f"noise std must be >= 0, got {value}")
        self._std = value

    @property
    def translation(self) -> np.ndarray:
        clean = np.asarray(self.source.translation, dtype=np.float64)
        if self._std == 0.0 or np.max(np.abs(clean)) < DEADBAND:
            return clean
        return np.clip(clean + self._rng.normal(0.0, self._std, size=3), -1.0, 1.0)

    @property
    def gripper(self) -> float:
        return self.source.gripper


class RecordingReader:
    """Pass-through wrapper that remembers the last command it served.

    Readers are sampled by whoever consumes them (the action hook once per
    control step, the shared flow policies once per denoising step or chunk),
    and `NoisyReader` draws fresh noise on every read — so logging a separate
    read would record a value that was never used. Wrapping the chain instead
    records exactly what was handed out: `last_translation` / `last_gripper`
    hold the most recent values and `reads` counts how many times the source
    has been sampled (log its delta to see when the input was actually
    consulted).
    """

    def __init__(self, source):
        self.source = source
        self.last_translation = np.zeros(3)
        self.last_gripper = GRIPPER_OPEN
        self.reads = 0

    @property
    def translation(self) -> np.ndarray:
        translation = np.asarray(self.source.translation, dtype=np.float64)
        self.last_translation = translation
        self.reads += 1
        return translation

    @property
    def gripper(self) -> float:
        self.last_gripper = self.source.gripper
        return self.last_gripper


class TeleopChain:
    """The teleop input pipeline shared by the runner and the experiment driver.

    Sources (SpaceMouse first, then keyboard) are merged, the raw command is
    recorded, then the deterministic corruption and the Gaussian noise are
    applied, and the value finally served is recorded too:

        SpaceMouse + keyboard -> raw -> M @ x -> + noise -> served

    `reader` is what the modes consume; `raw` and `served` are the recorders an
    experiment logs (`raw.last_translation` is the operator's true intent,
    `served.last_translation` what the policy actually got).
    """

    def __init__(self, keyboard: KeyboardReader, input_noise: float = 0.0):
        self.keyboard = keyboard
        self.spacemouse = None
        self.combined = CombinedReader([keyboard])
        self.raw = RecordingReader(self.combined)
        self.corruption = CommandCorruption(self.raw)
        self.noisy = NoisyReader(self.corruption, std=input_noise)
        self.served = RecordingReader(self.noisy)

    @property
    def reader(self) -> RecordingReader:
        return self.served

    def attach_spacemouse(self) -> None:
        """Connect a SpaceMouse and give it priority over the keyboard (no-op if already tried)."""
        if self.spacemouse is not None:
            return
        try:
            self.spacemouse = SpaceMouseReader()
            self.combined.sources.insert(0, self.spacemouse)
            print(f"SpaceMouse connected ({self.spacemouse.device_path}); any button toggles the gripper.")
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"SpaceMouse unavailable ({e}) — keyboard only.")
