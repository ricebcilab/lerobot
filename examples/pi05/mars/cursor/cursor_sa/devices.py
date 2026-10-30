"""Input devices: the SpaceMouse and the keyboard.

Each exposes `translation`, the current command [x, y] in [-1, 1] (x right, y up
on screen). The command then goes through `channel.InputChannel`.

The SpaceMouse is read by lerobot's SpaceMouse teleoperator
(`lerobot.teleoperators.spacemouse`), which the LIBERO example
(examples/pi05/mars/libero/teleop.py) uses too: same raw /dev/hidraw reports and
full-scale count, but only the stick's two horizontal axes are used, so pushing
the stick forward moves the cursor up the screen.
"""

import os

import numpy as np

from lerobot.teleoperators.spacemouse import teleop_spacemouse as hid

SPACEMOUSE_HID_ID = "0000046D:0000C62B"  # a SpaceMouse Pro's HID id: the form the SPACEMOUSE_HID_ID env var takes
AXIS_SCALE = 350.0            # raw counts at full deflection (SPACEMOUSE_FULL_SCALE's default)


def spacemouse_hid_id():
    """The `SPACEMOUSE_HID_ID` environment variable (a HID id such as `SPACEMOUSE_HID_ID` above, as in
    /sys/class/hidraw/*/device/uevent), or None when unset: then any 3Dconnexion device is taken."""
    return os.environ.get("SPACEMOUSE_HID_ID") or None


def spacemouse_full_scale():
    """The `SPACEMOUSE_FULL_SCALE` environment variable (raw counts at full deflection), default AXIS_SCALE."""
    return float(os.environ.get("SPACEMOUSE_FULL_SCALE") or AXIS_SCALE)


def find_spacemouse(hid_id=None):
    """The SpaceMouse's /dev/hidraw node (any 3Dconnexion device unless `hid_id`), or None if it is not plugged in."""
    try:
        return hid.find_spacemouse(hid_id)
    except FileNotFoundError:
        return None


def _screen(x, y):
    """Device x, y (already in [-1, 1]) -> screen [x, y]; `+ 0.0` keeps a zero y at +0.0."""
    return np.array([x, -y], dtype=float) + 0.0


def parse_report(report):
    """Raw hidraw report -> ("translation", [x, y]) | ("buttons", bitmask) | ("other", None).

    Device channels: x is a push to the right (+), y a pull toward you (+), so
    screen y is the negated device y.
    """
    kind, value = hid.parse_report(report)
    if kind == "translation":
        x, y, _ = np.clip(np.array(value, dtype=float) / AXIS_SCALE, -1.0, 1.0)
        return kind, _screen(x, y)
    return kind, value


class SpaceMouse:
    """Reads the stick in a background thread. `connected` turns False if it is unplugged."""

    def __init__(self, path, full_scale=AXIS_SCALE):
        self.path = path
        self._reader = hid.SpaceMouseReader(path, full_scale=full_scale)

    @property
    def connected(self):
        return self._reader.connected

    @property
    def translation(self):
        if not self._reader.connected:  # unplugged: at rest
            return np.zeros(2)
        x, y, _ = self._reader.axes
        return _screen(x, y)

    def take_button(self):
        """True once per button press since the last call."""
        return self._reader.take_press()

    def close(self):
        self._reader.close()


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
