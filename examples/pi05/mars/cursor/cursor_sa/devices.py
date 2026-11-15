"""Input devices: the SpaceMouse and the keyboard.

Each exposes `translation`, the current command [x, y] in [-1, 1] (x right, y up
on screen). The command then goes through `readout.Readout`.

The SpaceMouse is lerobot's axis-mapped reader
(`lerobot.teleoperators.spacemouse.SpaceMouseAxes`), which the LIBERO example
(examples/pi05/mars/libero/teleop.py) uses too: same raw /dev/hidraw reports,
full-scale count and environment variables (SPACEMOUSE_HID_ID,
SPACEMOUSE_FULL_SCALE), but only the stick's two horizontal axes are used, so
pushing the stick forward moves the cursor up the screen.
"""

import numpy as np

from lerobot.teleoperators.keyboard import KeyboardAxes
from lerobot.teleoperators.spacemouse import teleop_spacemouse as hid

# The screen's axes from the device's: screen x is a push to the right (device x), screen y a push
# forward (device y is a pull toward you, so its negative).
SCREEN_SOURCE, SCREEN_SIGN = (0, 1), (1.0, -1.0)


def find_spacemouse(hid_id=None):
    """The SpaceMouse's /dev/hidraw node (any 3Dconnexion device unless `hid_id`), or None if it is not plugged in."""
    try:
        return hid.find_spacemouse(hid_id)
    except FileNotFoundError:
        return None


class SpaceMouse(hid.SpaceMouseAxes):
    """lerobot's axis-mapped SpaceMouse on the screen's two axes, read in a background thread.
    `connected` turns False if it is unplugged, and the command is then at rest."""

    def __init__(self, path, full_scale=None):
        super().__init__(SCREEN_SOURCE, SCREEN_SIGN, path, full_scale=full_scale)

    @property
    def translation(self):
        if not self.connected:  # unplugged: at rest
            return np.zeros(2)
        return super().translation + 0.0  # `+ 0.0` keeps a zero y at +0.0

    def take_button(self):
        """True once per button press since the last call."""
        return self.reader.take_press()


# Arrow keys as pushes on the screen's axes (x right, y up).
KEY_PUSHES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}


def Keyboard(speed=0.5):  # noqa: N802 (the device's name, like SpaceMouse)
    """The arrow keys through lerobot's held-keys reader (`lerobot.teleoperators.keyboard.KeyboardAxes`, the LIBERO
    example's too): half deflection by default, full while Shift is held, a diagonal no faster than an axis. The
    app calls `hold` once per frame with the names of the keys currently down ("left", "right", "up", "down",
    "shift")."""
    return KeyboardAxes(KEY_PUSHES, speed=speed, unit=True)
