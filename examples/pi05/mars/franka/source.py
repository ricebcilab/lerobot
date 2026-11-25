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

**Which node carries the motion.** A 3Dconnexion SpaceMouse shows up as several hidraw nodes for one
device (the session saw five: ``hidraw3``, ``hidraw5``-``hidraw8`` under vendor 256F), and on this base
only one of them delivered translation reports — ``/dev/hidraw8``; the first by name (``hidraw3``) was
silent. So the source does not take the first 3Dconnexion node it finds: with no ``path`` it opens
every candidate, waits at most ``MOTION_SECONDS`` for a translation report
(``SpaceMouseReader.motion_reports``), keeps the first node that delivers one, and raises
:class:`FileNotFoundError` naming every candidate when none does. ``path`` in ``[spacemouse]`` pins the
node and skips the search. The chosen node is ``SpaceMouseSource.path`` and the run header prints it.

**Tare.** A sensor has a rest offset: the session measured ``x = +0.154`` with the stick released, and
subtracting that offset is what makes "stick at rest" mean a zero command. With ``tare`` (the
``[spacemouse] tare`` default, ``True``) the source captures the six-vector once at start — after the
node has delivered its first translation report and after the mapped reading has stayed unchanged for
``SETTLE_SECONDS`` (the stick is at rest; a stick that keeps moving is captured where it is at the
deadline) — and subtracts that six-vector from every later reading, clipped to [-1, 1]. The offset is a
property of the device, not of the session: the probe prints it (``rest six-vector``) and it should be
stable across sessions; ``tare = false`` reads the device untared.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from bridge import TWIST_DIMS
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

MOTION_SECONDS = 1.0
"""How long the node search waits for a translation report: one second is far longer than the device's
report period, and short enough that a silent node is skipped without a noticeable start-up delay."""

SETTLE_SECONDS = 0.5
"""The tare capture: the mapped reading must stay unchanged this long before it is taken as the rest
offset (the operator releases the stick when the session header prints; half a second is enough)."""

TARE_SECONDS = 2.0
"""The whole tare capture's bound: a stick that never settles is captured at this deadline, where it is."""

POLL_SECONDS = 0.02
"""How often the node search and the tare capture re-read the reader's state."""

HIDRAW_SYSFS = Path("/sys/class/hidraw")
"""Where the hidraw uevents are (the same tree lerobot's ``find_spacemouse`` reads)."""


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


def _vector(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{float(value):+.3f}" for value in values) + "]"


def spacemouse_nodes(
    hid_id: str | None = None,
    *,
    module: object | None = None,
    sysfs: Path | str | None = None,
) -> list[str]:
    """The ``/dev/hidraw*`` nodes of a 3Dconnexion device, in sysfs order.

    ``hid_id`` (e.g. ``"0000046D:0000C62B"``) keeps only the nodes whose uevent names that HID id;
    ``None`` keeps every 3Dconnexion device, using the same vendor test as lerobot's
    ``find_spacemouse`` (``_is_3dconnexion``: vendor 256F, or 046D with a C6xx product). ``module`` is
    lerobot's ``teleop_spacemouse`` module (imported here when None); ``sysfs`` is the hidraw class
    directory (the real one when None). It does not open the nodes: the caller decides which one
    carries motion.
    """
    if module is None:
        module = importlib.import_module("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    root = HIDRAW_SYSFS if sysfs is None else Path(sysfs)
    nodes = []
    for sysdir in sorted(root.glob("hidraw*")):
        try:
            uevent = (sysdir / "device" / "uevent").read_text()
        except OSError:
            continue
        if (hid_id in uevent) if hid_id is not None else module._is_3dconnexion(uevent):  # noqa: SLF001
            nodes.append(f"/dev/{sysdir.name}")
    return nodes


class SpaceMouseSource:
    """A 3Dconnexion SpaceMouse as a :class:`DeflectionSource`: the node that carries motion, tared.

    ``path`` pins the hidraw node (``[spacemouse] path``); without it the nodes are searched in sysfs
    order and the first that delivers a translation report within ``MOTION_SECONDS`` is used, because
    one device can expose several nodes and only one of them reports. ``hid_id`` filters the search to
    one device (`SPACEMOUSE_HID_ID`'s form). If no node delivers motion the constructor raises
    :class:`FileNotFoundError` naming every candidate; it never silently takes a silent node.

    ``full_scale``/``rotation_full_scale``/``source``/``sign`` are ``SpaceMouseTwist``'s. With ``tare``
    the source captures the rest six-vector at start (``rest``) and subtracts it from every later
    reading, clipped to [-1, 1]; see the module docstring for the capture rule. ``describe()`` is the
    run header's line: the node it took and whether it tared.

    ``nodes``/``motion_seconds``/``settle_seconds``/``tare_seconds``/``poll_seconds`` are the search
    and capture knobs (the defaults are the hardware's; a test shortens them).
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
        tare: bool = True,
        nodes: Sequence[str] | None = None,
        motion_seconds: float = MOTION_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
        tare_seconds: float = TARE_SECONDS,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        mouse = importlib.import_module("lerobot.teleoperators.spacemouse.teleop_spacemouse")

        self._module = mouse
        self._source = (0, 1, 2, 3, 4, 5) if source is None else tuple(source)
        self._sign = (1.0,) * TWIST_DIMS if sign is None else tuple(float(value) for value in sign)
        self._full_scale = full_scale
        self._rotation_full_scale = rotation_full_scale
        self._tare = bool(tare)
        self._motion_seconds = float(motion_seconds)
        self._settle_seconds = float(settle_seconds)
        self._tare_seconds = float(tare_seconds)
        self._poll_seconds = float(poll_seconds)
        self._closed = False
        self.rest: NDArray | None = None

        if path is not None:
            self.path = str(path)
            try:
                self._twist = self._open(self.path)
            except OSError as error:
                raise FileNotFoundError(
                    f"[spacemouse] path {self.path!r} cannot be opened: {error}"
                ) from error
        else:
            candidates = (
                [str(node) for node in nodes]
                if nodes is not None
                else spacemouse_nodes(hid_id or None, module=mouse)
            )
            if not candidates:
                wanted = f" (HID id {hid_id})" if hid_id else ""
                raise FileNotFoundError(
                    f"No 3Dconnexion hidraw node found{wanted} under {HIDRAW_SYSFS}: is the SpaceMouse "
                    "plugged in, and is the udev rule that makes its nodes readable installed?"
                )
            self.path, self._twist = self._choose(candidates)
        if self._tare:
            self.rest = self._capture_rest()

    # ---------------------------------------------------------------- the node search

    def _open(self, path: str):
        """One candidate as lerobot's six-channel twist, with this source's mapping and scales."""
        return self._module.SpaceMouseTwist(
            source=self._source,
            sign=self._sign,
            path=path,
            full_scale=self._full_scale,
            rotation_full_scale=self._rotation_full_scale,
        )

    def _choose(self, candidates: Sequence[str]) -> tuple[str, object]:
        """The first candidate that delivers a translation report within the bounded wait.

        Every candidate is opened (one reader thread each), the nodes are polled in order until the
        deadline, and the node not taken are closed; if none reports, every candidate is named in the
        FileNotFoundError, because taking the first silent node is exactly the session's failure.
        """
        twists, unreadable = {}, {}
        for candidate in candidates:
            try:
                twists[candidate] = self._open(candidate)
            except OSError as error:  # not readable by this user: say so, keep trying the others
                unreadable[candidate] = error
        if not twists:
            details = "; ".join(f"{node}: {error}" for node, error in unreadable.items())
            raise FileNotFoundError(f"No 3Dconnexion node could be opened ({details})")
        chosen = None
        try:
            deadline = time.monotonic() + self._motion_seconds
            while chosen is None:
                for candidate in candidates:
                    twist = twists.get(candidate)
                    if twist is not None and twist.reader.motion_reports > 0:
                        chosen = candidate
                        break
                if chosen is not None or time.monotonic() >= deadline:
                    break
                time.sleep(self._poll_seconds)
            if chosen is None:
                details = ", ".join(candidates)
                if unreadable:
                    details += f" (unreadable: {', '.join(unreadable)})"
                raise FileNotFoundError(
                    f"No 3Dconnexion node delivered a translation report within "
                    f"{self._motion_seconds:g} s: tried {details}. Move the stick while the session "
                    "starts, or pin the node with [spacemouse] path."
                )
            return chosen, twists[chosen]
        finally:
            for candidate, twist in twists.items():
                if candidate != chosen:
                    twist.close()

    def _capture_rest(self) -> NDArray | None:
        """The rest six-vector: the mapped reading after reports arrive and the stick has settled.

        Waits up to ``tare_seconds`` for the node's first translation report (the search already saw
        one, so this is immediate; an explicit path may be silent), then requires the mapped
        six-vector to stay exactly unchanged for ``settle_seconds`` — a stick being held still is
        also unchanged, but the operator is told to release it — and captures it. A reading that never
        settles is captured where it is at the deadline; a silent node captures nothing (None).
        Note the capture is *after* the device's zeroing reports have stopped changing the reading:
        the rest offset is the sensor's, so it is what the stick reads when released, not zeros.
        """
        reader = self._twist.reader
        deadline = time.monotonic() + self._tare_seconds
        while reader.motion_reports == 0:
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._poll_seconds)
        rest = self._mapped()
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(self._poll_seconds)
            current = self._mapped()
            if np.array_equal(current, rest):
                if time.monotonic() - stable_since >= self._settle_seconds:
                    return rest
            else:
                rest = current
                stable_since = time.monotonic()
        return rest

    def _mapped(self) -> NDArray:
        """The reader's latest six channels in the twist's axes, in [-1, 1] per channel."""
        raw = np.asarray([*self._twist.reader.axes, *self._twist.reader.rotation], dtype=np.float64)
        return raw[np.asarray(self._source, dtype=np.intp)] * np.asarray(self._sign, dtype=np.float64)

    # ---------------------------------------------------------------- the port

    def deflection(self) -> NDArray:
        """The six axes in [-1, 1]; with a captured rest offset, that offset is subtracted first."""
        six = np.asarray(self._twist.twist, dtype=np.float64)
        if self.rest is not None:
            six = six - self.rest
        return np.clip(six, -1.0, 1.0)

    def gripper(self) -> bool:
        """True while the gripper is latched closed; each button press toggles the latch."""
        if self._twist.reader.take_press():
            self._closed = not self._closed
        return self._closed

    @property
    def tare(self) -> bool:
        """Whether the rest offset is captured and subtracted (the ``[spacemouse] tare`` flag)."""
        return self._tare

    def reset(self) -> None:
        """A new episode starts with the gripper open (the latch, not the device; the tare stays)."""
        self._closed = False

    def describe(self) -> str:
        """The run header's line: the node it took, and whether and where it tared."""
        if not self._tare:
            return f"{self.path} (tare off)"
        if self.rest is None:
            return f"{self.path} (tare on, rest offset not captured: the node reported no motion)"
        return f"{self.path} (tare on, rest offset {_vector(self.rest)})"

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

    def describe(self) -> str:
        """The run header's line for a dry run: there is no device to name."""
        return "fake (--dry-run: no device, no tare)"

    def close(self) -> None:
        """Nothing to release: a fake source is a field."""
