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

import os
import struct
import time
from pathlib import Path

import pytest

from lerobot.teleoperators.spacemouse import SpaceMouseConfig, SpaceMouseTeleop, teleop_spacemouse
from lerobot.teleoperators.spacemouse.teleop_spacemouse import (
    SPACEMOUSE_VENDORS,
    SpaceMouseReader,
    find_spacemouse,
    parse_report,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.errors import DeviceNotConnectedError


def motion(x: int, y: int, z: int) -> bytes:
    return b"\x01" + struct.pack("<hhh", x, y, z)


def buttons(mask: int) -> bytes:
    return b"\x03" + mask.to_bytes(4, "little")


def fake_sysfs(root: Path, nodes: dict[str, str]) -> Path:
    """A /sys/class/hidraw stand-in: {"hidraw3": "0003:0000256F:0000C635"} -> hidraw3/device/uevent."""
    for name, hid_id in nodes.items():
        (root / name / "device").mkdir(parents=True)
        (root / name / "device" / "uevent").write_text(f"DRIVER=hid-generic\nHID_ID={hid_id}\nHID_NAME=x\n")
    return root


def wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        time.sleep(0.001)


def pipe_reader(reports: list[bytes], **kwargs) -> SpaceMouseReader:
    """A reader over a pipe that carries `reports` and then EOF; returns once it has read them all.

    A pipe does not keep report boundaries, so each report is written only after the previous one
    was handled.
    """
    r, w = os.pipe()
    try:
        reader = SpaceMouseReader(f"/dev/fd/{r}", **kwargs)
    finally:
        os.close(r)
    for i, report in enumerate(reports):
        os.write(w, report)
        wait_for(lambda i=i: reader.reports_handled > i)
    os.close(w)
    wait_for(lambda: not reader.connected)
    return reader


def test_parse_report():
    assert parse_report(b"\x01" + struct.pack("<hhh", 350, -175, 0)) == ("translation", (350, -175, 0))
    assert parse_report(b"\x03\x01\x00\x00\x00") == ("buttons", 1)
    assert parse_report(b"\x09") == ("other", None)
    assert parse_report(b"\x01\x00") == ("other", None)  # too short for a translation


def test_find_by_vendor(tmp_path):
    sysfs = fake_sysfs(tmp_path, {"hidraw3": "0003:0000256F:0000C635"})
    assert find_spacemouse(sysfs=sysfs) == "/dev/hidraw3"


def test_find_skips_other_logitech_devices(tmp_path):
    # A typical desk: a Logitech receiver (046D:C534) sorts before the SpaceMouse Pro (046D:C62B).
    sysfs = fake_sysfs(
        tmp_path,
        {
            "hidraw2": "0003:0000046D:0000C534",
            "hidraw3": "0003:0000046D:0000C62B",
            "hidraw5": "0003:0000046D:00004023",
        },
    )
    assert find_spacemouse(sysfs=sysfs) == "/dev/hidraw3"


def test_find_by_hid_id(tmp_path):
    sysfs = fake_sysfs(tmp_path, {"hidraw1": "0003:0000256F:0000C635", "hidraw4": "0003:0000046D:0000C62B"})
    assert find_spacemouse("0000046D:0000C62B", sysfs=sysfs) == "/dev/hidraw4"
    with pytest.raises(FileNotFoundError, match="0000046D:0000C631"):
        find_spacemouse("0000046D:0000C631", sysfs=sysfs)


def test_missing_device_names_the_ids(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        find_spacemouse(sysfs=tmp_path)
    assert "046D" in str(excinfo.value) and "256F" in str(excinfo.value)
    assert SPACEMOUSE_VENDORS == ("046D", "256F")


def test_full_scale():
    reader = pipe_reader([motion(350, -175, 0)], full_scale=700)
    assert reader.axes == (0.5, -0.25, 0)


def test_axes_clip_and_default_scale():
    reader = pipe_reader([motion(700, -350, 35)])
    assert reader.axes == (1.0, -1.0, 0.1)


def test_buttons_and_presses():
    reader = pipe_reader([buttons(1), buttons(3), buttons(2), buttons(0), b"\x09", buttons(4)])
    assert reader.buttons == 4
    assert reader.presses == 3  # 0->1, 1->3 (button 2 added), 0->4; releases do not count
    assert reader.take_press() is True
    assert reader.take_press() is False


def test_teleop_get_action(tmp_path, monkeypatch):
    r, w = os.pipe()
    monkeypatch.setattr(teleop_spacemouse, "find_spacemouse", lambda hid_id=None: f"/dev/fd/{r}")
    config = SpaceMouseConfig(calibration_dir=tmp_path, full_scale=700)
    teleop = make_teleoperator_from_config(config)
    assert isinstance(teleop, SpaceMouseTeleop)
    assert teleop.is_calibrated is True
    teleop.calibrate()
    teleop.configure()
    assert set(teleop.action_features) == {"x", "y", "z", "buttons"}
    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()

    teleop.connect()
    os.close(r)
    assert teleop.is_connected
    os.write(w, motion(350, -175, 0))
    wait_for(lambda: teleop.reader.reports_handled == 1)
    os.write(w, buttons(1))
    wait_for(lambda: teleop.reader.reports_handled == 2)
    assert teleop.get_action() == {"x": 0.5, "y": -0.25, "z": 0.0, "buttons": 1}

    os.close(w)  # unplugged
    wait_for(lambda: not teleop.is_connected)
    with pytest.raises(DeviceNotConnectedError):
        teleop.get_action()
    teleop.disconnect()
    assert teleop.reader is None
