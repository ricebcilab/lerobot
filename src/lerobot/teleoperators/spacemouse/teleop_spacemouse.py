#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""A 3Dconnexion SpaceMouse read straight from Linux /dev/hidraw (no driver or package needed).

The device sends a translation report (id 1: three int16 counts x, y, z) whenever the stick
moves, including a zeroing report on release, and a button report (id 3: a bitmask) whenever a
button changes. Device channels: x is a push to the right (+), y a pull toward you (+), z a
press down (+). Rotation reports (id 2) are ignored.

`SpaceMouseReader` keeps the latest state in a background thread; `SpaceMouseTeleop` wraps it
as a LeRobot teleoperator. `SpaceMouseAxes` is a reader mapped onto an application's axes (which
device channel feeds each axis, and with which sign), found and scaled from the
`SPACEMOUSE_HID_ID` and `SPACEMOUSE_FULL_SCALE` environment variables unless told otherwise.
"""

import os
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_spacemouse import SpaceMouseConfig

SPACEMOUSE_VENDORS = ("046D", "256F")  # 3Dconnexion: under Logitech's vendor id, then its own
# Logitech's vendor id also covers its keyboards, mice and receivers; 3Dconnexion's products
# under it are 0xC6xx (C603 SpaceMouse Plus ... C62B SpaceMouse Pro, C640 NuLOOQ).
LOGITECH_3DCONNEXION_PRODUCT_PREFIX = "C6"

# A SpaceMouse Pro's HID id (vendor:product, as in /sys/class/hidraw/*/device/uevent): the form the
# SPACEMOUSE_HID_ID environment variable takes to pin one device.
SPACEMOUSE_PRO_HID_ID = "0000046D:0000C62B"
FULL_SCALE = 350.0  # raw counts at full deflection (SPACEMOUSE_FULL_SCALE's default)

REPORT_TRANSLATION = 1
REPORT_BUTTONS = 3
READ_SIZE = 64


def _is_3dconnexion(uevent: str) -> bool:
    """Whether a hidraw uevent's HID_ID (bus:vendor:product, 8 hex digits each) is a 3Dconnexion device."""
    for line in uevent.splitlines():
        if line.startswith("HID_ID="):
            parts = line[len("HID_ID=") :].strip().upper().split(":")
            if len(parts) != 3:
                return False
            vendor, product = parts[1][-4:], parts[2][-4:]
            if vendor == "256F":
                return True
            return vendor == "046D" and product.startswith(LOGITECH_3DCONNEXION_PRODUCT_PREFIX)
    return False


def find_spacemice(hid_id: str | None = None, sysfs: Path = Path("/sys/class/hidraw")) -> list[str]:
    """Every /dev/hidraw node of a SpaceMouse, sorted.

    With `hid_id` (e.g. "0000046D:0000C62B") the nodes whose uevent contains it; with None every
    node whose HID_ID has a 3Dconnexion vendor. Empty when there is none. A 3Dconnexion device
    usually exposes more than one interface, and only one of them carries the translation reports,
    so the list matters to a caller that did not get what it expected.
    """
    nodes = []
    for sysdir in sorted(Path(sysfs).glob("hidraw*"), key=lambda path: path.name):
        try:
            uevent = (sysdir / "device" / "uevent").read_text()
        except OSError:
            continue
        if (hid_id in uevent) if hid_id is not None else _is_3dconnexion(uevent):
            nodes.append(f"/dev/{sysdir.name}")
    return nodes


def find_spacemouse(hid_id: str | None = None, sysfs: Path = Path("/sys/class/hidraw")) -> str:
    """The first /dev/hidraw node of a SpaceMouse (`find_spacemice`'s order).

    Raises FileNotFoundError naming the ids searched; a device with several interfaces resolves to
    whichever node the kernel lists first, which is why `find_spacemice` exists for a caller that
    needs to try them all.
    """
    nodes = find_spacemice(hid_id, sysfs)
    if nodes:
        return nodes[0]
    wanted = (
        f"HID id {hid_id}" if hid_id is not None else f"3Dconnexion vendor {' or '.join(SPACEMOUSE_VENDORS)}"
    )
    raise FileNotFoundError(f"No SpaceMouse ({wanted}) found under {sysfs}. Is it plugged in?")


def parse_report(data: bytes) -> tuple[str, tuple[int, int, int] | int | None]:
    """Parse one raw hidraw report (report id is the first byte).

    Returns ("translation", (x, y, z)) with raw int16 counts, ("buttons", bitmask), or
    ("other", None).
    """
    if len(data) >= 7 and data[0] == REPORT_TRANSLATION:
        return "translation", struct.unpack_from("<hhh", data, 1)
    if len(data) >= 2 and data[0] == REPORT_BUTTONS:
        return "buttons", int.from_bytes(data[1:5], "little")
    return "other", None


class SpaceMouseReader:
    """Reads a SpaceMouse hidraw node in a background thread.

    `axes` is the latest device (x, y, z) divided by `full_scale` and clipped to [-1, 1];
    `buttons` the latest bitmask; `presses` counts new presses (a bit going from up to down) and
    `take_press()` is True once per press since the last call. When the device is unplugged or
    closed the thread ends and `connected` turns False; `axes` and `buttons` keep their last
    values. The node is only read, never written.
    """

    def __init__(self, path: str, full_scale: float = 350.0):
        self.path = path
        self.full_scale = float(full_scale)
        self._file = open(path, "rb", buffering=0)  # noqa: SIM115
        self._lock = threading.Lock()
        self._axes: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._buttons = 0
        self._presses = 0
        self._pressed = False
        self._motion_reports = 0
        self._reports = 0
        self.connected = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                data = self._file.read(READ_SIZE)
            except (OSError, ValueError):  # unplugged or closed
                break
            if not data:
                break
            self._handle(data)
        self.connected = False

    def _handle(self, data: bytes) -> None:
        kind, value = parse_report(data)
        with self._lock:
            if kind == "translation":
                scaled = np.clip(np.array(value, dtype=np.float64) / self.full_scale, -1.0, 1.0)
                self._axes = (float(scaled[0]), float(scaled[1]), float(scaled[2]))
                self._motion_reports += 1
            elif kind == "buttons":
                if value & ~self._buttons:  # a new press
                    self._presses += 1
                    self._pressed = True
                self._buttons = value
            self._reports += 1

    @property
    def axes(self) -> tuple[float, float, float]:
        with self._lock:
            return self._axes

    @property
    def buttons(self) -> int:
        with self._lock:
            return self._buttons

    @property
    def presses(self) -> int:
        with self._lock:
            return self._presses

    @property
    def motion_reports(self) -> int:
        """Translation reports handled so far (0 means `axes` is still the initial rest value)."""
        with self._lock:
            return self._motion_reports

    @property
    def reports_handled(self) -> int:
        """Reports of any kind handled so far."""
        with self._lock:
            return self._reports

    def take_press(self) -> bool:
        """True once per button press since the last call."""
        with self._lock:
            pressed, self._pressed = self._pressed, False
        return pressed

    def close(self) -> None:
        self._file.close()


def env_hid_id() -> str | None:
    """The `SPACEMOUSE_HID_ID` environment variable (a HID id such as `SPACEMOUSE_PRO_HID_ID`), or None when
    unset: then any 3Dconnexion device is taken."""
    return os.environ.get("SPACEMOUSE_HID_ID") or None


def env_full_scale() -> float:
    """The `SPACEMOUSE_FULL_SCALE` environment variable (raw counts at full deflection), default `FULL_SCALE`."""
    return float(os.environ.get("SPACEMOUSE_FULL_SCALE") or FULL_SCALE)


class SpaceMouseAxes:
    """A SpaceMouse mapped onto an application's axes: axis i is `sign[i]` times the device channel
    `source[i]` (0: x, a push to the right; 1: y, a pull toward you; 2: z, a press down), in [-1, 1].

    Without `path` the device is searched for by `hid_id` (default: the SPACEMOUSE_HID_ID environment
    variable, unset = any 3Dconnexion device); `full_scale` defaults to SPACEMOUSE_FULL_SCALE (unset =
    `FULL_SCALE`). `reader` is the `SpaceMouseReader` underneath (buttons, presses, report counts);
    like it, `translation` keeps its last value when the device is unplugged.
    """

    def __init__(
        self,
        source,
        sign,
        path: str | None = None,
        hid_id: str | None = None,
        full_scale: float | None = None,
    ):
        self.source = np.asarray(source, dtype=np.intp)
        self.sign = np.asarray(sign, dtype=np.float64)
        if self.source.shape != self.sign.shape or self.source.ndim != 1:
            raise ValueError(f"source and sign must be two lists of one length, got {source!r} and {sign!r}")
        self.path = path or find_spacemouse(hid_id if hid_id is not None else env_hid_id())
        self.reader = SpaceMouseReader(
            self.path, full_scale=full_scale if full_scale is not None else env_full_scale()
        )

    def map(self, device_xyz) -> np.ndarray:
        """Device (x, y, z) onto the application's axes."""
        return np.asarray(device_xyz, dtype=np.float64)[self.source] * self.sign

    @property
    def connected(self) -> bool:
        return self.reader.connected

    @property
    def translation(self) -> np.ndarray:
        return self.map(self.reader.axes)

    def close(self) -> None:
        self.reader.close()


class SpaceMouseTeleop(Teleoperator):
    """A 3Dconnexion SpaceMouse as a teleoperator: the device's x, y, z in [-1, 1] and its button bitmask.

    Linux only (/dev/hidraw); the node must be readable by the user (see the udev rule in
    examples/pi05/mars/README.md).
    """

    config_class = SpaceMouseConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseConfig):
        super().__init__(config)
        self.config = config
        self.reader: SpaceMouseReader | None = None

    @property
    def action_features(self) -> dict:
        return {"x": float, "y": float, "z": float, "buttons": int}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.reader is not None and self.reader.connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.reader is not None:  # unplugged since: release the old node first
            self.reader.close()
        path = find_spacemouse(self.config.hid_id)
        self.reader = SpaceMouseReader(path, full_scale=self.config.full_scale)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> dict[str, Any]:
        x, y, z = self.reader.axes
        return {"x": x, "y": y, "z": z, "buttons": self.reader.buttons}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None
