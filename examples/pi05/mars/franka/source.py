"""The operator's device: the deflection port, the SpaceMouse adapter and the deterministic fake.

A :class:`DeflectionSource` is six deflection axes in [-1, 1] (device order ``[x, y, z, rx, ry, rz]``)
and a latched gripper button — position, not velocity: the loop multiplies the axes by the config's
translation/rotation gains to get a twist. :class:`SpaceMouseSource` wraps lerobot's six-channel
``SpaceMouseTwist`` and toggles the latch on the device's press edge; :class:`FakeSource` is the
deterministic stand-in the tests and ``--dry-run`` drive, with settable fields.

The sources are their own module (not part of ``loop.py``) because they are a port with two adapters:
a test can import and exercise them without the loop, c3po or torch. lerobot is imported inside
``SpaceMouseSource.__init__`` for that reason: its teleoperator package imports torch, and importing
this module must not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from bridge import TWIST_DIMS
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


class DeflectionSource(Protocol):
    """The operator's device: six deflection axes in [-1, 1] and a latched gripper button.

    The axes are the device's, ``[x, y, z, rx, ry, rz]``; the loop scales the first three to m/s and
    the last three to rad/s. ``gripper()`` is a latch, not a momentary press: it returns True while
    the operator has the gripper closed. ``reset()`` is the per-episode latch reset the loop performs
    at every episode start; a stateless source implements it as a no-op.
    """

    def deflection(self) -> NDArray: ...

    def gripper(self) -> bool: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class SpaceMouseSource:
    """A 3Dconnexion SpaceMouse as a :class:`DeflectionSource`: lerobot's six-channel twist, latched.

    ``path``/``hid_id``/``full_scale``/``rotation_full_scale`` are ``SpaceMouseTwist``'s (environment
    variables' defaults included); ``source``/``sign`` map the device's six channels onto the twist's
    axes, identity by default. A press of any button toggles the gripper latch; the device's button
    report is a bitmask, so ``reader.take_press()`` is the once-per-press edge.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        hid_id: str | None = None,
        full_scale: float | None = None,
        rotation_full_scale: float | None = None,
        source: Sequence[int] | None = None,
        sign: Sequence[float] | None = None,
    ) -> None:
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTwist

        self._twist = SpaceMouseTwist(
            source=(0, 1, 2, 3, 4, 5) if source is None else source,
            sign=(1.0,) * TWIST_DIMS if sign is None else sign,
            path=path,
            hid_id=hid_id,
            full_scale=full_scale,
            rotation_full_scale=rotation_full_scale,
        )
        self._closed = False

    def deflection(self) -> NDArray:
        return np.asarray(self._twist.twist, dtype=np.float64)

    def gripper(self) -> bool:
        """True while the gripper is latched closed; each button press toggles the latch."""
        if self._twist.reader.take_press():
            self._closed = not self._closed
        return self._closed

    def reset(self) -> None:
        """A new episode starts with the gripper open (the latch, not the device)."""
        self._closed = False

    def close(self) -> None:
        self._twist.close()


class FakeSource:
    """A deterministic :class:`DeflectionSource` for tests and ``--dry-run``: fields, not a device.

    ``deflection_vector`` and ``gripper_closed`` are settable between ticks (a test changes the stick
    mid-chunk); ``reset`` opens the gripper latch at each episode and is a no-op otherwise. ``reads``
    counts the deflection reads, which is how a test checks the command cadence.
    """

    def __init__(self, *, deflection: NDArray | None = None, gripper: bool = False) -> None:
        self.deflection_vector = np.zeros(TWIST_DIMS) if deflection is None else np.asarray(deflection)
        if self.deflection_vector.shape != (TWIST_DIMS,):
            raise ValueError(
                f"deflection must be ({TWIST_DIMS},) in [-1, 1], got shape {self.deflection_vector.shape}"
            )
        self.gripper_closed = bool(gripper)
        self.reads = 0

    def deflection(self) -> NDArray:
        self.reads += 1
        return np.asarray(self.deflection_vector, dtype=np.float64).copy()

    def gripper(self) -> bool:
        return self.gripper_closed

    def reset(self) -> None:
        self.gripper_closed = False

    def close(self) -> None:
        """Nothing to release: a fake source is a field."""
