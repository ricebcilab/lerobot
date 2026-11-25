"""The operator's device port: the deterministic fake and the SpaceMouse's latched gripper button.

No loop, no station, no c3po, and no torch: ``FakeSource`` is a set of fields, and
``SpaceMouseSource`` is exercised against a stand-in ``SpaceMouseTwist`` injected into ``sys.modules``
(lerobot's SpaceMouse module imports torch at module scope, which the lazy import exists to avoid).
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from bridge import TWIST_DIMS
from source import FakeSource, SpaceMouseSource

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"


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


def test_the_spacemouse_button_latches_the_gripper_once_per_press(monkeypatch):
    """lerobot's reader is a bitmask with a ``take_press`` edge; a press toggles the latch, so the
    button is a switch and not a momentary "hold to close" (DESIGN's SpaceMouse contract)."""
    presses = [False, True, False, True, False]
    closed = []
    twist = types.SimpleNamespace(
        twist=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        reader=types.SimpleNamespace(take_press=lambda: presses.pop(0) if presses else False),
        close=lambda: closed.append(True),
    )
    module = types.ModuleType("lerobot.teleoperators.spacemouse.teleop_spacemouse")
    module.SpaceMouseTwist = lambda **kwargs: twist
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.spacemouse.teleop_spacemouse", module)

    source = SpaceMouseSource()
    assert [source.gripper() for _ in range(5)] == [False, True, True, False, False]
    np.testing.assert_array_equal(source.deflection(), np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))
    source.reset()
    assert source.gripper() is False
    source.close()
    assert closed == [True]


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
