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

The device sends its motion in one of two layouts, and this module decodes both. In the pair layout
(the SpaceMouse Pro and similar) a translation report (id 1: three int16 counts x, y, z) arrives
whenever the stick moves, including a zeroing report on release, and a separate rotation report
(id 2: three int16 counts rx, ry, rz) whenever it is twisted. In the combined layout (the SpaceMouse
Wireless through its Universal Receiver, measured on our station 2026-11-25) one 13-byte report under
id 1 carries all six axes: the translation triple at offset 1 and the rotation triple at offset 7, so
id 2 never arrives and a reader that only waits for it loses the twist. A button report (id 3: a
bitmask) arrives whenever a button changes in both layouts. Device channels: x is a push to the right
(+), y a pull toward you (+), z a press down (+); the rotation channels are the device's own rx, ry, rz
order. The rotation full scale is not known yet (the probe measures it), so `rotation` divides by the
translation's full scale unless `SPACEMOUSE_ROTATION_FULL_SCALE` overrides it.

`SpaceMouseReader` keeps the latest state in a background thread; `SpaceMouseTeleop` wraps it
as a LeRobot teleoperator. `SpaceMouseAxes` is a reader mapped onto an application's axes (which
device channel feeds each axis, and with which sign), found and scaled from the
`SPACEMOUSE_HID_ID` and `SPACEMOUSE_FULL_SCALE` environment variables unless told otherwise.
`SpaceMouseTwist` is the same mapping over all six channels, the three translations first.
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
# Raw counts at full rotation: a guess until the probe measures the hardware, so the translation's
# full scale stands in until SPACEMOUSE_ROTATION_FULL_SCALE overrides it.
ROTATION_FULL_SCALE = FULL_SCALE

REPORT_TRANSLATION = 1
REPORT_ROTATION = 2
REPORT_BUTTONS = 3
READ_SIZE = 64
TRANSLATION_REPORT_SIZE = 7  # the report id, then the translation's three int16 counts
COMBINED_REPORT_SIZE = 13  # the report id, then six int16 counts: translation, then rotation

TWIST_DIMS = 6  # device channels in twist order: x, y, z (translation) then rx, ry, rz (rotation)


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


def find_spacemouse(hid_id: str | None = None, sysfs: Path = Path("/sys/class/hidraw")) -> str:
    """The /dev/hidraw node of a SpaceMouse.

    With `hid_id` (e.g. "0000046D:0000C62B") the first node whose uevent contains it; with None
    the first node whose HID_ID has a 3Dconnexion vendor. Raises FileNotFoundError naming the ids
    searched.
    """
    for sysdir in sorted(Path(sysfs).glob("hidraw*")):
        try:
            uevent = (sysdir / "device" / "uevent").read_text()
        except OSError:
            continue
        if (hid_id in uevent) if hid_id is not None else _is_3dconnexion(uevent):
            return f"/dev/{sysdir.name}"
    wanted = (
        f"HID id {hid_id}" if hid_id is not None else f"3Dconnexion vendor {' or '.join(SPACEMOUSE_VENDORS)}"
    )
    raise FileNotFoundError(f"No SpaceMouse ({wanted}) found under {sysfs}. Is it plugged in?")


def parse_report(data: bytes) -> tuple[str, tuple[int, int, int] | int | None]:
    """Parse one raw hidraw report (report id is the first byte).

    Returns ("translation", (x, y, z)) or ("rotation", (rx, ry, rz)) with raw int16 counts,
    ("buttons", bitmask), or ("other", None).
    """
    if len(data) >= TRANSLATION_REPORT_SIZE and data[0] == REPORT_TRANSLATION:
        return "translation", struct.unpack_from("<hhh", data, 1)
    if len(data) >= TRANSLATION_REPORT_SIZE and data[0] == REPORT_ROTATION:
        return "rotation", struct.unpack_from("<hhh", data, 1)
    if len(data) >= 2 and data[0] == REPORT_BUTTONS:
        return "buttons", int.from_bytes(data[1:5], "little")
    return "other", None


def parse_motion(data: bytes) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """A translation report's int16 triples: the translation at offset 1 and the rotation at offset 7.

    The two device layouts (module docstring): a 7-byte id 1 report carries the translation alone and
    the rotation half is None; a 13-byte id 1 report (the combined layout) carries both in one packet.
    ``data`` is a report `parse_report` accepted as "translation" (id 1, at least
    `TRANSLATION_REPORT_SIZE` bytes), which is what guarantees the first triple exists.
    """
    translation = struct.unpack_from("<hhh", data, 1)
    rotation = struct.unpack_from("<hhh", data, 7) if len(data) >= COMBINED_REPORT_SIZE else None
    return translation, rotation


def _scaled(counts, scale: float) -> tuple[float, float, float]:
    """Three raw int16 counts as floats in [-1, 1], clipped, at `scale` counts full deflection."""
    values = np.clip(np.array(counts, dtype=np.float64) / scale, -1.0, 1.0)
    return (float(values[0]), float(values[1]), float(values[2]))


class SpaceMouseReader:
    """Reads a SpaceMouse hidraw node in a background thread.

    `axes` is the latest device (x, y, z) divided by `full_scale` and clipped to [-1, 1];
    `rotation` the latest device (rx, ry, rz) divided by `rotation_full_scale` and clipped to
    [-1, 1]; `buttons` the latest bitmask; `presses` counts new presses (a bit going from up to
    down) and `take_press()` is True once per press since the last call. Both device layouts are
    decoded (module docstring): an id 1 report with 13 bytes sets `axes` and `rotation` together,
    a 7-byte id 1 report sets `axes` alone, and an id 2 report sets `rotation` alone. When the device
    is unplugged or closed the thread ends and `connected` turns False; `axes`, `rotation` and
    `buttons` keep their last values. The node is only read, never written.
    """

    def __init__(
        self,
        path: str,
        full_scale: float = 350.0,
        rotation_full_scale: float | None = None,
    ):
        self.path = path
        self.full_scale = float(full_scale)
        self.rotation_full_scale = (
            float(rotation_full_scale) if rotation_full_scale is not None else env_rotation_full_scale()
        )
        self._file = open(path, "rb", buffering=0)  # noqa: SIM115
        self._lock = threading.Lock()
        self._axes: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
        combined_rotation = None
        if kind == "translation":  # id 1: the translation, plus the rotation in the combined layout
            value, combined_rotation = parse_motion(data)
        with self._lock:
            if kind == "translation":
                self._axes = _scaled(value, self.full_scale)
                if combined_rotation is not None:
                    self._rotation = _scaled(combined_rotation, self.rotation_full_scale)
                self._motion_reports += 1
            elif kind == "rotation":
                self._rotation = _scaled(value, self.rotation_full_scale)
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
    def rotation(self) -> tuple[float, float, float]:
        with self._lock:
            return self._rotation

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


def env_rotation_full_scale() -> float:
    """The `SPACEMOUSE_ROTATION_FULL_SCALE` environment variable (raw counts at full rotation), falling back
    to `SPACEMOUSE_FULL_SCALE` (unset = `FULL_SCALE`).

    The explicit rotation variable wins. Falling back to the translation's full scale keeps a user who set
    only `SPACEMOUSE_FULL_SCALE` from getting a stale rotation scale; the device scales the two reports
    independently, so a measured rotation scale still overrides it. The rotation full scale has not been
    measured on the hardware yet, so the fallback is a guess either way.
    """
    return float(os.environ.get("SPACEMOUSE_ROTATION_FULL_SCALE") or env_full_scale())


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


class SpaceMouseTwist:
    """A SpaceMouse's six channels as one 6-D twist: the three translations, then the three rotations.

    Axis i of `twist` is `sign[i]` times device channel `source[i]`, over the concatenated device
    channels (0: x, 1: y, 2: z, 3: rx, 4: ry, 5: rz), each in [-1, 1]; `[x, y, z, rx, ry, rz]` with
    the translation axes first. The reader decodes both device layouts, so the rotation channels are
    filled whether the device sent an id 2 report or put the rotation in the id 1 report's second
    triple. The rotation full scale is a guess (see `SpaceMouseReader.rotation`).

    Without `path` the device is searched for by `hid_id` (default: the SPACEMOUSE_HID_ID environment
    variable, unset = any 3Dconnexion device); `full_scale` defaults to SPACEMOUSE_FULL_SCALE (unset =
    `FULL_SCALE`), and `rotation_full_scale` to SPACEMOUSE_ROTATION_FULL_SCALE (unset =
    SPACEMOUSE_FULL_SCALE, unset = `FULL_SCALE`). `reader` is the `SpaceMouseReader` underneath (buttons,
    presses, report counts); like it, `twist` keeps its last value when the device is unplugged.
    """

    def __init__(
        self,
        source,
        sign,
        path: str | None = None,
        hid_id: str | None = None,
        full_scale: float | None = None,
        rotation_full_scale: float | None = None,
    ):
        self.source = np.asarray(source, dtype=np.intp)
        self.sign = np.asarray(sign, dtype=np.float64)
        if self.source.shape != self.sign.shape or self.source.ndim != 1:
            raise ValueError(f"source and sign must be two lists of one length, got {source!r} and {sign!r}")
        if self.source.size != TWIST_DIMS:
            raise ValueError(f"a twist maps all {TWIST_DIMS} device channels, got {source!r}")
        if np.any((self.source < 0) | (self.source >= TWIST_DIMS)):
            raise ValueError(f"source indexes the device channels 0-{TWIST_DIMS - 1}, got {source!r}")
        self.path = path or find_spacemouse(hid_id if hid_id is not None else env_hid_id())
        self.reader = SpaceMouseReader(
            self.path,
            full_scale=full_scale if full_scale is not None else env_full_scale(),
            rotation_full_scale=rotation_full_scale
            if rotation_full_scale is not None
            else env_rotation_full_scale(),
        )

    def map(self, device_xyzrot) -> np.ndarray:
        """Device (x, y, z, rx, ry, rz) onto the twist's six axes."""
        return np.asarray(device_xyzrot, dtype=np.float64)[self.source] * self.sign

    @property
    def connected(self) -> bool:
        return self.reader.connected

    @property
    def twist(self) -> np.ndarray:
        """The six mapped channels in [-1, 1], the translations first: [x, y, z, rx, ry, rz]."""
        return self.map((*self.reader.axes, *self.reader.rotation))

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
