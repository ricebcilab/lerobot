"""The LIBERO example's SpaceMouseReader (a thin wrapper over lerobot.teleoperators.spacemouse) against
the frozen reader it replaced (`_reference_spacemouse`): identical raw reports give bit-identical
translations and the same gripper toggles."""

import queue
import struct
import threading

import pytest
import teleop

from lerobot.teleoperators.spacemouse import teleop_spacemouse as hid

from . import _reference_spacemouse as ref


def motion(x, y, z=0):
    return b"\x01" + struct.pack("<hhh", x, y, z)


def buttons(mask):
    return b"\x03" + mask.to_bytes(4, "little")


class ScriptedHid:
    """A hidraw node stand-in: each read() returns the next fed report (or raises a fed exception)."""

    def __init__(self):
        self._queue = queue.Queue()
        self._cond = threading.Condition()
        self.reads = 0
        self.fed = 0

    def read(self, n):
        with self._cond:
            self.reads += 1
            self._cond.notify_all()
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass

    def feed(self, item, timeout=5.0):
        """Hand the reader one report and return once it has handled it (is back in read())."""
        self.fed += 1
        self._queue.put(item)
        with self._cond:
            assert self._cond.wait_for(lambda: self.reads > self.fed, timeout)

    def end(self, item):
        self._queue.put(item)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


REPORTS = [
    motion(0, 0, 0),
    motion(350, -175, 12),
    motion(-350, 175, -350),
    motion(700, -700, 700),
    motion(32767, -32768, -32768),
    motion(1, -1, 0),
    buttons(1),
    buttons(1),
    buttons(3),
    buttons(0),
    buttons(2),
    b"\x02" + struct.pack("<hhh", 50, 50, 50),
    b"\x01\x10",
    b"\x03",
    buttons(6),
    motion(-17, 299, 40),
    buttons(0),
    buttons(1),
]


@pytest.mark.parametrize("ending", [OSError("unplugged")])
def test_reader_matches_the_reference(monkeypatch, ending):
    made = []

    def fake_open(path, mode="r", buffering=-1):
        assert mode == "rb", "the SpaceMouse must be opened read-only"
        made.append(ScriptedHid())
        return made[-1]

    monkeypatch.setattr(ref, "open", fake_open, raising=False)
    monkeypatch.setattr(hid, "open", fake_open, raising=False)
    old = ref.SpaceMouseReader("/dev/hidraw-test")
    new = teleop.SpaceMouseReader("/dev/hidraw-test")
    old_hid, new_hid = made
    assert old.device_path == new.device_path
    assert same(old.translation, new.translation) and old.gripper == new.gripper
    for report in REPORTS:
        old_hid.feed(report)
        new_hid.feed(report)
        assert same(old.translation, new.translation), report
        assert old.gripper == new.gripper, report
    old_hid.end(ending)  # unplugged while deflected: both keep the last translation
    new_hid.end(ending)
    old._thread.join(5.0)
    new.reader._thread.join(5.0)
    assert not old._thread.is_alive() and not new.connected
    assert same(old.translation, new.translation) and old.gripper == new.gripper
    assert new.translation.any()
    old.close()
    new.close()
