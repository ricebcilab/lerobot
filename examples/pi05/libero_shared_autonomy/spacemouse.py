#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal SpaceMouse reader for the interactive LIBERO runner.

Reads a 3Dconnexion SpaceMouse Pro directly from its /dev/hidraw node with
plain file I/O — no hidapi dependency, no robosuite driver (whose default USB
ids target a different model). Requires the hidraw node to be readable by the
user (this machine has a udev rule making the SpaceMouse world-readable).

HID protocol (wired SpaceMouse Pro, USB id 046d:c62b):
  report id 1: translation — x, y, z as little-endian int16, range ~±350
  report id 2: rotation    — ignored here
  report id 3: buttons     — 32-bit little-endian bitmask
"""

import struct
import threading
from pathlib import Path

import numpy as np

SPACEMOUSE_HID_ID = "0000046D:0000C62B"  # vendor:product as it appears in /sys uevent
AXIS_SCALE = 350.0
# Map device channels (0=x: right push +, 1=y: pull-toward-you push +, 2=z:
# press-down +) onto the arm action dims [Δx, Δy, Δz]. Tuned by feel on the
# real device so each push moves the arm the same way: forward push → forward,
# left push → left, pull up → up. If a direction feels wrong, flip its sign;
# if two axes are crossed, swap their AXIS_SOURCE entries.
AXIS_SOURCE = np.array([1, 0, 2])  # device channel feeding each action dim
AXIS_SIGN = np.array([1.0, -1.0, -1.0])

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


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
