# Frozen reference for equivalence tests: a frozen copy of the original cursor project's
# cursor_sa/devices.py (before the port), unchanged. Do not edit; tests compare the cursor example's SpaceMouse wrapper
# (over lerobot.teleoperators.spacemouse) against it.

"""Input devices: the SpaceMouse and the keyboard.

Each exposes `translation`, the current command [x, y] in [-1, 1] (x right, y up
on screen). The command then goes through `channel.InputChannel`.

The SpaceMouse reader follows the lab's LeRobot reader
(examples/pi05/libero_shared_autonomy/teleop.py, now examples/pi05/mars/libero/teleop.py): same raw
/dev/hidraw reports and full-scale count, but only the stick's two horizontal
axes are used, so pushing the stick forward moves the cursor up the screen.
"""

import struct
import threading
from pathlib import Path

import numpy as np

SPACEMOUSE_HID_ID = "0000046D:0000C62B"
AXIS_SCALE = 350.0            # raw counts at full deflection


def find_spacemouse(hid_id=SPACEMOUSE_HID_ID):
    """The SpaceMouse's /dev/hidraw node, or None if it is not plugged in."""
    for sysdir in sorted(Path("/sys/class/hidraw").glob("hidraw*")):
        try:
            if hid_id in (sysdir / "device" / "uevent").read_text():
                return f"/dev/{sysdir.name}"
        except OSError:
            continue
    return None


def parse_report(report):
    """Raw hidraw report -> ("translation", [x, y]) | ("buttons", bitmask) | ("other", None).

    Device channels: x is a push to the right (+), y a pull toward you (+), so
    screen y is the negated device y.
    """
    if len(report) >= 7 and report[0] == 1:
        x, y, _ = struct.unpack_from("<hhh", report, 1)
        return "translation", np.clip(np.array([x, -y], dtype=float) / AXIS_SCALE, -1.0, 1.0)
    if len(report) >= 2 and report[0] == 3:
        return "buttons", int.from_bytes(report[1:5], "little")
    return "other", None


class SpaceMouse:
    """Reads the stick in a background thread. `connected` turns False if it is unplugged."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "rb", buffering=0)  # noqa: SIM115
        self._lock = threading.Lock()
        self._translation = np.zeros(2)
        self._button, self._buttons_down = False, 0
        self.connected = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                data = self._file.read(64)
            except (OSError, ValueError):  # unplugged or closed
                data = b""
            if not data:
                break
            kind, value = parse_report(data)
            with self._lock:
                if kind == "translation":
                    self._translation = value
                elif kind == "buttons":
                    if value & ~self._buttons_down:  # a new press
                        self._button = True
                    self._buttons_down = value
        with self._lock:
            self._translation = np.zeros(2)
        self.connected = False

    @property
    def translation(self):
        with self._lock:
            return self._translation.copy()

    def take_button(self):
        """True once per button press since the last call."""
        with self._lock:
            pressed, self._button = self._button, False
        return pressed

    def close(self):
        self._file.close()


class Keyboard:
    """Arrow keys: half deflection by default, full while Shift is held.
    The app calls `update` once per frame with the keys currently held."""

    def __init__(self, speed=0.5):
        self.speed = speed
        self._translation = np.zeros(2)

    def update(self, left, right, up, down, fast=False):
        push = np.array([right - left, up - down], dtype=float)
        norm = np.linalg.norm(push)
        self._translation = push / norm * (1.0 if fast else self.speed) if norm else np.zeros(2)

    @property
    def translation(self):
        return self._translation.copy()
