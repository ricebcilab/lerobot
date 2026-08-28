#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Teleop input sources for the interactive LIBERO runner besides the SpaceMouse.

`KeyboardReader` is fed by the browser live view (keydown/keyup events are
POSTed to /keys by the page) and exposes the same `translation` / `gripper`
interface as `spacemouse.SpaceMouseReader`. `CombinedReader` merges several
such sources so a SpaceMouse and the keyboard can be used in the same session.
`MatrixReader` applies a fixed 3x3 matrix to the merged command (deterministic
corruption, loaded from YAML by `load_corruption_matrix`), `NoisyReader`
perturbs it with isotropic Gaussian noise, and `RecordingReader` remembers what
was served so an experiment can log exactly the command the policy consumed.
"""

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import yaml
from spacemouse import AXIS_SCALE, GRIPPER_CLOSE, GRIPPER_OPEN, normalize

DEFAULT_SPEED = 0.5  # fraction of full-scale deflection a held key produces
FAST_KEY = "shift"  # hold for full-scale (1.0) deflection
DEADBAND = 0.05  # below this a source counts as idle (matches FlowControlPolicy)
STALE_AFTER = 1.0  # seconds without a page update before held keys are dropped

# Each key is a full-scale push on the SpaceMouse's *device* axes (x: right +,
# y: toward-you +, z: press-down +) and goes through spacemouse.normalize(), so
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


def validate_matrix(matrix, size: int = 3, where: str = "matrix") -> np.ndarray:
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


def load_matrix(path: str | Path, size: int = 3, keys: Iterable[str] = ("M", "matrix")) -> np.ndarray:
    """Read a `size` x `size` matrix from a YAML file.

    Accepted layouts: a top-level key from `keys` holding `size` rows of
    `size` numbers, or the bare list of rows. Raises ValueError for anything
    else (and OSError if the file cannot be read).
    """
    path = Path(path)
    keys = tuple(keys)
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                data = data[key]
                break
        else:
            expected = " (or ".join(f"`{k}:`" for k in keys) + ")" * (len(keys) - 1)
            raise ValueError(f"{path}: expected a top-level {expected} key holding a {size}x{size} matrix")
    return validate_matrix(data, size, str(path))


def load_corruption_matrix(path: str | Path) -> np.ndarray:
    """Read the 3x3 teleop corruption matrix M from a YAML file."""
    return load_matrix(path, size=3, keys=("M", "matrix"))


class MatrixReader:
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
