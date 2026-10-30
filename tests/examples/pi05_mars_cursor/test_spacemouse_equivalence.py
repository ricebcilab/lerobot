"""The cursor example's SpaceMouse (a thin wrapper over lerobot.teleoperators.spacemouse) against the
frozen cursor reader (`_reference_devices`): identical raw reports give bit-identical commands,
button presses and connection state."""

import functools
import queue
import struct
import threading
import time

import _reference_devices as ref
import pytest
from cursor_sa import devices

from lerobot.teleoperators.spacemouse import teleop_spacemouse as hid


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

    def end(self, item=b""):
        self._queue.put(item)


def open_scripted(monkeypatch):
    """Make both readers' open() hand out their own ScriptedHid; returns (old reader, its hid, new, its hid)."""
    made = []

    def fake_open(path, mode="r", buffering=-1):
        assert mode == "rb", "the SpaceMouse must be opened read-only"
        made.append(ScriptedHid())
        return made[-1]

    monkeypatch.setattr(ref, "open", fake_open, raising=False)
    monkeypatch.setattr(hid, "open", fake_open, raising=False)
    old = ref.SpaceMouse("/dev/hidraw-test")
    new = devices.SpaceMouse("/dev/hidraw-test")
    return old, made[0], new, made[1]


def wait_disconnected(*readers, timeout=5.0):
    deadline = time.monotonic() + timeout
    while any(r.connected for r in readers):
        assert time.monotonic() < deadline
        time.sleep(0.001)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


REPORTS = [
    motion(0, 0),
    motion(350, -175, 12),
    motion(-350, 175),
    motion(0, -1),
    motion(1, 0),
    motion(700, -700, -700),  # beyond full scale: clipped
    motion(32767, -32768, 5),
    motion(-32768, 32767),
    motion(123, 0),
    buttons(1),
    buttons(1),
    buttons(3),
    buttons(2),
    buttons(0),
    b"\x02" + struct.pack("<hhh", 50, 50, 50),  # rotation: ignored
    b"\x01\x10",  # too short for a translation
    b"\x03",  # too short for buttons
    b"\x03\x04",
    motion(-17, 299),
    buttons(0),
    buttons(8),
    motion(0, 0),
]


@pytest.mark.parametrize("ending", [b"", OSError("unplugged"), ValueError("closed")])
def test_reader_matches_the_reference(monkeypatch, ending):
    old, old_hid, new, new_hid = open_scripted(monkeypatch)
    assert same(old.translation, new.translation)
    assert old.path == new.path and old.connected == new.connected is True
    for i, report in enumerate(REPORTS):
        old_hid.feed(report)
        new_hid.feed(report)
        assert same(old.translation, new.translation), report
        assert old.connected == new.connected is True
        if i % 3 == 2:  # take the button latch only now and then, so presses pile up in between
            assert old.take_button() == new.take_button()
    assert old.take_button() == new.take_button()
    assert old.take_button() == new.take_button() is False

    old_hid.feed(buttons(0))
    new_hid.feed(buttons(0))
    old_hid.feed(buttons(16))  # a press left unread when the device goes away
    new_hid.feed(buttons(16))
    old_hid.feed(motion(200, 100))  # deflected when it goes away
    new_hid.feed(motion(200, 100))
    old_hid.end(ending)
    new_hid.end(ending)
    wait_disconnected(old, new)
    assert same(old.translation, new.translation)
    assert not new.translation.any()
    assert old.take_button() == new.take_button() is True
    old.close()
    new.close()


@pytest.mark.parametrize("report", REPORTS + [motion(0, 350), motion(-1, 1, 1)])
def test_parse_report_matches_the_reference(report):
    old_kind, old_value = ref.parse_report(report)
    new_kind, new_value = devices.parse_report(report)
    assert old_kind == new_kind
    if old_kind == "translation":
        assert same(old_value, new_value)
    else:
        assert old_value == new_value


def test_find_spacemouse_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(hid, "find_spacemouse", functools.partial(hid.find_spacemouse, sysfs=tmp_path))
    assert devices.find_spacemouse() is None
    (tmp_path / "hidraw3" / "device").mkdir(parents=True)
    (tmp_path / "hidraw3" / "device" / "uevent").write_text("HID_ID=0003:0000046D:0000C62B\n")
    assert devices.find_spacemouse() == "/dev/hidraw3"
    assert devices.find_spacemouse(devices.SPACEMOUSE_HID_ID) == "/dev/hidraw3"
    assert devices.find_spacemouse("0000256F:0000C635") is None


class RecordingReader:
    """Stands in for lerobot's SpaceMouseReader and records how it was built."""

    made = []

    def __init__(self, path, full_scale=350.0):
        self.path, self.full_scale, self.connected = path, full_scale, True
        RecordingReader.made.append(self)

    def close(self):
        pass


@pytest.mark.parametrize(
    "env, hid_id, full_scale",
    [
        ({}, None, 350.0),  # unset: any 3Dconnexion device, today's full-scale count
        ({"SPACEMOUSE_HID_ID": "0000256F:0000C635"}, "0000256F:0000C635", 350.0),
        ({"SPACEMOUSE_FULL_SCALE": "700"}, None, 700.0),
    ],
)
def test_spacemouse_env_vars_reach_the_reader(monkeypatch, env, hid_id, full_scale):
    from cursor_sa import agents

    monkeypatch.delenv("SPACEMOUSE_HID_ID", raising=False)
    monkeypatch.delenv("SPACEMOUSE_FULL_SCALE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    searched = []
    monkeypatch.setattr(hid, "find_spacemouse", lambda hid_id=None: searched.append(hid_id) or "/dev/hidraw7")
    monkeypatch.setattr(hid, "SpaceMouseReader", RecordingReader)
    RecordingReader.made.clear()
    agent = agents.HumanAgent()
    assert agent.poll()
    assert searched == [hid_id]
    assert [(r.path, r.full_scale) for r in RecordingReader.made] == [("/dev/hidraw7", full_scale)]
    agent.close()
