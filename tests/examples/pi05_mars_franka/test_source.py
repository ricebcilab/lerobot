"""The operator's device port: the fake, the SpaceMouse's node search and tare, its latched button.

No loop, no station, no c3po, and no torch: ``FakeSource`` is a set of fields, and
``SpaceMouseSource`` is exercised against a stand-in SpaceMouse module injected into ``sys.modules``
(lerobot's SpaceMouse module imports torch at module scope, which the lazy import exists to avoid).
The fakes have the same shape as lerobot's: a ``SpaceMouseTwist`` with a ``reader``
(``motion_reports``, ``axes``, ``rotation``, ``take_press``) and the mapping ``map``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import source as source_module
from bridge import TWIST_DIMS
from source import FakeSource, SpaceMouseSource, spacemouse_nodes

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"

FAST = {"motion_seconds": 0.02, "settle_seconds": 0.0, "tare_seconds": 0.2, "poll_seconds": 0.001}


class _FakeReader:
    """lerobot's ``SpaceMouseReader`` shape: the latest channels and the report counters."""

    def __init__(self, axes=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), motion=0.0):
        self.axes = tuple(axes)
        self.rotation = tuple(rotation)
        self.motion_reports = int(motion)
        self.presses = [False]

    def take_press(self) -> bool:
        return self.presses.pop(0) if self.presses else False


def _install(monkeypatch, specs: dict) -> list:
    """Install a fake ``lerobot...teleop_spacemouse``; ``specs[path]`` is its reader's state.

    Every twist built is appended to the returned list (path, reader, closed), so a test can assert
    which nodes the search opened, which it kept and which it closed.
    """
    built = []

    class FakeTwist:
        def __init__(self, *, source, sign, path=None, full_scale=None, rotation_full_scale=None):
            self.source = np.asarray(source)
            self.sign = np.asarray(sign)
            self.path = path
            self.reader = _FakeReader(**specs.get(path, {}))
            self.closed = False
            built.append(self)

        def map(self, six):
            return np.asarray(six, dtype=np.float64)[self.source] * self.sign

        @property
        def twist(self):
            return self.map((*self.reader.axes, *self.reader.rotation))

        def close(self):
            self.closed = True

    module = types.ModuleType("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    module.SpaceMouseTwist = FakeTwist
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.spacemouse.teleop_spacemouse", module)
    return built


def _spacemouse(*, nodes=None, **kwargs):
    return SpaceMouseSource(nodes=nodes, **{**FAST, **kwargs})


# ---------------------------------------------------------------- the fake source


def test_the_fake_source_is_settable_for_the_next_tick():
    """Tests and ``--dry-run`` move the stick between ticks and latch the gripper by assignment."""
    source = FakeSource(deflection=np.arange(TWIST_DIMS, dtype=np.float64) / 10.0, gripper=True)
    np.testing.assert_allclose(source.deflection(), np.arange(TWIST_DIMS) / 10.0)
    assert source.gripper() is True

    source.deflection_vector = np.full(TWIST_DIMS, 0.5)
    source.gripper_closed = False
    np.testing.assert_array_equal(source.deflection(), np.full(TWIST_DIMS, 0.5))
    assert source.gripper() is False
    assert source.reads == 2


def test_the_fake_source_resets_its_latch_at_each_episode():
    source = FakeSource(gripper=True)
    source.reset()
    assert source.gripper() is False


def test_a_deflection_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="deflection"):
        FakeSource(deflection=np.zeros(TWIST_DIMS - 1))


def test_the_fake_source_describes_itself_for_the_run_header():
    assert "fake" in FakeSource().describe()


# ---------------------------------------------------------------- the node search


def test_an_explicit_path_wins_over_the_search(monkeypatch):
    """``[spacemouse] path`` pins the node: no other candidate is opened and no motion is required."""
    built = _install(monkeypatch, {"/dev/hidraw3": {"motion": 0}})
    source = _spacemouse(path="/dev/hidraw3", nodes=["/dev/hidraw8"], tare=False)
    assert source.path == "/dev/hidraw3"
    assert [twist.path for twist in built] == ["/dev/hidraw3"]
    source.close()
    assert built[0].closed


def test_the_search_takes_the_node_that_delivers_motion_not_the_first_one(monkeypatch):
    """The session's shape: hidraw3 and hidraw5 are the same device but silent; hidraw8 reports.

    Taking the first 3Dconnexion node by name would read a node that carries no motion, which is
    exactly what the session found; the search keeps hidraw8 and closes the silent nodes.
    """
    built = _install(
        monkeypatch,
        {
            "/dev/hidraw3": {"motion": 0},
            "/dev/hidraw5": {"motion": 0},
            "/dev/hidraw8": {"motion": 295, "axes": (0.154, 0.0, 0.0)},
        },
    )
    source = _spacemouse(nodes=["/dev/hidraw3", "/dev/hidraw5", "/dev/hidraw8"])
    assert source.path == "/dev/hidraw8"
    by_path = {twist.path: twist for twist in built}
    assert by_path["/dev/hidraw8"].closed is False
    assert by_path["/dev/hidraw3"].closed and by_path["/dev/hidraw5"].closed
    source.close()


def test_no_node_with_motion_is_an_error_naming_every_candidate(monkeypatch):
    """Silence on all nodes is a start-up failure that says which nodes were tried — never a silent
    first pick."""
    built = _install(monkeypatch, {"/dev/hidraw3": {"motion": 0}, "/dev/hidraw5": {"motion": 0}})
    with pytest.raises(FileNotFoundError) as raised:
        _spacemouse(nodes=["/dev/hidraw3", "/dev/hidraw5"])
    message = str(raised.value)
    assert "/dev/hidraw3" in message and "/dev/hidraw5" in message
    assert "translation report" in message
    assert all(twist.closed for twist in built)


def test_an_unreadable_candidate_is_named_and_the_rest_still_searched(monkeypatch):
    """A node this user cannot open does not hide a readable one; a total failure names the errors."""

    class FakeTwist:
        def __init__(self, *, path=None, **_kwargs):
            if path == "/dev/hidraw3":
                raise PermissionError("permission denied")
            self.path, self.reader, self.closed = path, _FakeReader(motion=1), False

        def close(self):
            self.closed = True

    module = types.ModuleType("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    module.SpaceMouseTwist = FakeTwist
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.spacemouse.teleop_spacemouse", module)
    source = _spacemouse(nodes=["/dev/hidraw3", "/dev/hidraw8"], tare=False)
    assert source.path == "/dev/hidraw8"


def test_no_3dconnexion_node_at_all_is_an_error(monkeypatch):
    _install(monkeypatch, {})
    monkeypatch.setattr(source_module, "HIDRAW_SYSFS", Path("/nonexistent-hidraw"))
    with pytest.raises(FileNotFoundError, match="no 3Dconnexion node|No 3Dconnexion hidraw node"):
        SpaceMouseSource(tare=False, nodes=[])


def test_the_node_enumeration_uses_the_librarys_vendor_test_and_the_hid_id_filter(tmp_path):
    """The candidates are the sysfs hidraw nodes lerobot's ``_is_3dconnexion`` accepts, in order."""
    sysfs = tmp_path / "hidraw"
    entries = {
        "hidraw3": "0003:0000256F:0000C652",
        "hidraw4": "0003:000026CE:000001A2",
        "hidraw8": "0003:0000256F:0000C652",
    }
    for name, hid in entries.items():
        device = sysfs / name / "device"
        device.mkdir(parents=True)
        (device / "uevent").write_text(f"DRIVER=hid-generic\nHID_ID={hid}\nHID_NAME=3Dconnexion\n")

    module = types.SimpleNamespace(_is_3dconnexion=lambda uevent: "0000256F" in uevent)
    assert spacemouse_nodes(module=module, sysfs=sysfs) == ["/dev/hidraw3", "/dev/hidraw8"]
    assert spacemouse_nodes("0000256F:0000C652", module=module, sysfs=sysfs) == [
        "/dev/hidraw3",
        "/dev/hidraw8",
    ]
    assert spacemouse_nodes("0000C62B", module=module, sysfs=sysfs) == []


# ---------------------------------------------------------------- the tare


def test_the_tare_captures_the_rest_offset_and_subtracts_it(monkeypatch):
    """The session's rest state is x = +0.154, not zero: taring makes a released stick a zero command."""
    built = _install(
        monkeypatch,
        {"/dev/hidraw8": {"motion": 295, "axes": (0.154, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0)}},
    )
    source = _spacemouse(nodes=["/dev/hidraw8"])
    assert source.tare is True
    np.testing.assert_allclose(source.rest, [0.154, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(source.deflection(), np.zeros(TWIST_DIMS), atol=1e-12)

    reader = built[0].reader
    reader.axes = (1.0, 0.0, 0.0)  # a full push right: the rest offset comes off the reading
    np.testing.assert_allclose(source.deflection(), [0.846, 0.0, 0.0, 0.0, 0.0, 0.0])
    reader.axes = (-1.0, 0.0, 0.0)  # and the offset is clipped out of [-1, 1] after the subtract
    np.testing.assert_allclose(source.deflection(), [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    source.close()


def test_tare_off_reads_the_device_untared(monkeypatch):
    _install(monkeypatch, {"/dev/hidraw8": {"motion": 1, "axes": (0.154, 0.0, 0.0)}})
    source = _spacemouse(nodes=["/dev/hidraw8"], tare=False)
    assert source.rest is None and source.tare is False
    np.testing.assert_allclose(source.deflection(), [0.154, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert "tare off" in source.describe()
    source.close()


def test_a_silent_node_captures_no_tare(monkeypatch):
    """An explicit path is trusted even when it reports nothing: the tare is not invented from zeros."""
    _install(monkeypatch, {"/dev/hidraw3": {"motion": 0}})
    source = SpaceMouseSource(
        path="/dev/hidraw3",
        tare=True,
        motion_seconds=0.0,
        settle_seconds=0.0,
        tare_seconds=0.05,
        poll_seconds=0.001,
    )
    assert source.rest is None
    assert "not captured" in source.describe()
    source.close()


def test_the_tare_waits_for_the_stick_to_settle(monkeypatch):
    """A reading still changing is not taken as rest: the capture takes the settled value, not the
    first one it saw (a stick released from a deflection passes through intermediate readings)."""

    class ScriptedReader:
        """Hands out a scripted sequence of readings; the last one repeats forever."""

        def __init__(self):
            self.readings = [(0.0, 0.0, 0.0), (0.9, 0.0, 0.0), (0.4, 0.0, 0.0), (0.1, 0.0, 0.0)]
            self.read = 0
            self.motion_reports = 10
            self.rotation = (0.0, 0.0, 0.0)

        @property
        def axes(self):
            value = self.readings[min(self.read, len(self.readings) - 1)]
            self.read += 1
            return value

    class FakeTwist:
        def __init__(self, **kwargs):
            self.path = kwargs.get("path")
            self.reader = ScriptedReader()
            self.closed = False

        def map(self, six):
            return np.asarray(six, dtype=np.float64)

        def close(self):
            self.closed = True

    module = types.ModuleType("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    module.SpaceMouseTwist = FakeTwist
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.spacemouse.teleop_spacemouse", module)

    source = SpaceMouseSource(
        path="/dev/hidraw8",
        tare=True,
        motion_seconds=0.0,
        settle_seconds=0.01,
        tare_seconds=0.5,
        poll_seconds=0.005,
    )
    # The script settles at 0.1 after two changes; the 0.4 window's stability is what is captured.
    assert source.rest is not None
    assert source.rest[0] == pytest.approx(0.1)
    source.close()


def test_describe_names_the_node_and_the_tare(monkeypatch):
    _install(monkeypatch, {"/dev/hidraw8": {"motion": 1, "axes": (0.154, 0.0, 0.0)}})
    source = _spacemouse(nodes=["/dev/hidraw8"])
    description = source.describe()
    assert "/dev/hidraw8" in description and "tare on" in description and "+0.154" in description
    source.close()


# ---------------------------------------------------------------- the button


def test_the_spacemouse_button_latches_the_gripper_once_per_press(monkeypatch):
    """lerobot's reader is a bitmask with a ``take_press`` edge; a press toggles the latch, so the
    button is a switch and not a momentary "hold to close" (DESIGN's SpaceMouse contract)."""
    presses = [False, True, False, True, False]
    twist = types.SimpleNamespace(
        twist=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        reader=types.SimpleNamespace(take_press=lambda: presses.pop(0) if presses else False),
        close=lambda: None,
    )
    module = types.ModuleType("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    module.SpaceMouseTwist = lambda **kwargs: twist
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.spacemouse.teleop_spacemouse", module)

    source = SpaceMouseSource(path="/dev/hidraw0", tare=False)
    assert [source.gripper() for _ in range(5)] == [False, True, True, False, False]
    np.testing.assert_array_equal(source.deflection(), np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))
    source.reset()
    assert source.gripper() is False
    source.close()


def test_importing_the_sources_needs_neither_torch_nor_c3po():
    """The device port is its own module: it can be imported and tested without the loop and without
    the real SpaceMouse package (whose torch import lives inside ``SpaceMouseSource.__init__``)."""
    code = (
        "import sys, source; "
        "assert 'torch' not in sys.modules, 'source imported torch'; "
        "assert 'c3po' not in sys.modules, 'source imported c3po'; "
        "assert 'lerobot' not in sys.modules, 'source imported lerobot'; "
        "assert hasattr(source, 'DeflectionSource') and hasattr(source, 'SpaceMouseSource')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
