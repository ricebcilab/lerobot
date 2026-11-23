"""The operator chain and the probe's formatters, without a SpaceMouse or a station.

The chain is the one piece of the example between a person and the policy, so these tests pin the three
rules an operator would notice: a corruption applies while the stick is pushed, an idle stick stays exactly
idle, and the gripper toggles once per press.
"""

import json
import struct
from types import SimpleNamespace

import numpy as np
import probe
import readout_chain
from interfaces import StationManifest

from lerobot.teleoperators import readout


class FakeReader:
    """A SpaceMouse reader: settable device axes and a queue of button presses."""

    def __init__(self, axes=(0.0, 0.0, 0.0)):
        self.axes = axes
        self.full_scale = 350.0
        self._pressed = False

    def press(self):
        self._pressed = True

    def take_press(self):
        pressed, self._pressed = self._pressed, False
        return pressed


def make_chain(corruption=None, input_noise=0.0, axes=(0.0, 0.0, 0.0)):
    """A chain over a fake stick, with the robot's noise gate (``noise_gate="command"``)."""
    reader = FakeReader(axes)
    served = readout.Readout(corruption=corruption, input_noise=input_noise, n_dims=3, noise_gate="command")
    return readout_chain.SpaceMouseChain(reader, served), reader


def device_axes_for(base):
    """The device (x, y, z) whose mapped command is ``base``: the inverse of the module's axis map."""
    device = np.empty(3)
    device[readout_chain.AXIS_SOURCE] = np.asarray(base, dtype=np.float64) / readout_chain.AXIS_SIGN
    return tuple(device)


def test_full_push_maps_through_the_corruption_and_is_clipped():
    # A base-frame push straight along +y through a matrix that doubles it: 2 is clipped to 1.
    corruption = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    chain, _ = make_chain(corruption, axes=device_axes_for((0.0, 1.0, 0.0)))

    np.testing.assert_array_equal(readout_chain.map_axes(chain.reader.axes), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(chain.translation, [1.0, 1.0, 0.0])


def test_push_inside_the_deadband_comes_out_exactly_as_the_raw_push():
    # A corruption ten times the push would be obvious; inside the deadband it must not run at all.
    chain, _ = make_chain(10.0 * np.eye(3), axes=device_axes_for((0.02, -0.03, 0.04)))
    raw = readout_chain.map_axes(chain.reader.axes)

    assert np.max(np.abs(raw)) < readout.DEADBAND
    np.testing.assert_array_equal(chain.translation, raw)


def test_noise_is_applied_only_when_the_corrupted_command_is_active():
    # Idle stick, noise configured: nothing is drawn, the command stays exactly zero.
    chain, _ = make_chain(input_noise=0.5)
    for _ in range(5):
        np.testing.assert_array_equal(chain.translation, np.zeros(3))

    # A full push the corruption zeroes is itself idle, so it is served without noise either.
    chain, _ = make_chain(np.zeros((3, 3)), input_noise=0.5, axes=device_axes_for((0.0, 1.0, 0.0)))
    for _ in range(5):
        np.testing.assert_array_equal(chain.translation, np.zeros(3))


def test_grip_toggles_once_per_press():
    chain, reader = make_chain()
    assert chain.grip() == -1.0  # open

    reader.press()
    assert chain.grip() == 1.0  # one press closes
    assert chain.grip() == 1.0  # and the press is consumed once

    reader.press()
    assert chain.grip() == -1.0  # the next press opens again


def test_null_chain_is_idle_and_open():
    chain = readout_chain.NullChain("headless")
    np.testing.assert_array_equal(chain.translation, np.zeros(3))
    assert chain.grip() == -1.0
    assert chain.why == "headless"
    chain.advance()
    chain.reset()
    np.testing.assert_array_equal(chain.translation, np.zeros(3))


def test_build_chain_degrades_to_null():
    chain = readout_chain.build_chain({"corruption": None, "input_noise": 0.0}, headless=True)
    assert isinstance(chain, readout_chain.NullChain)
    assert "headless" in chain.why


def test_build_chain_degrades_when_no_device_is_found(monkeypatch):
    monkeypatch.setattr(
        readout_chain, "find_spacemouse", lambda: (_ for _ in ()).throw(FileNotFoundError("no node"))
    )
    chain = readout_chain.build_chain({"corruption": None, "input_noise": 0.0}, headless=False)
    assert isinstance(chain, readout_chain.NullChain)
    assert "no node" in chain.why


def test_format_manifest_names_the_arm_cameras_and_actions():
    manifest = StationManifest(
        arms=("follower",),
        cameras=("front", "wrist"),
        action_keys=("follower/joint_velocity", "follower/gripper_position"),
        control_rate=50.0,
    )
    text = probe.format_manifest(manifest)

    for needle in (
        "follower",
        "front",
        "wrist",
        "follower/joint_velocity",
        "follower/gripper_position",
        "50",
    ):
        assert needle in text


def pi05_config(**overrides):
    """The fields of a pi05 config the probe reads, as the config loader hands them over."""
    fields = {
        "type": "pi05",
        "chunk_size": 15,
        "n_action_steps": 10,
        "num_inference_steps": 10,
        "max_action_dim": 32,
        "use_relative_actions": False,
        "input_features": {
            "observation.images.base_0_rgb": SimpleNamespace(type="VISUAL", shape=(3, 224, 224)),
            "observation.images.wrist_0_rgb": SimpleNamespace(type="VISUAL", shape=(3, 224, 224)),
            "observation.state": SimpleNamespace(type="STATE", shape=(32,)),
        },
        "output_features": {"action": SimpleNamespace(type="ACTION", shape=(32,))},
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_format_checkpoint_names_every_field():
    text = probe.format_checkpoint(pi05_config())

    for needle in (
        "type: pi05",
        "chunk_size: 15",
        "n_action_steps: 10",
        "num_inference_steps: 10",
        "max_action_dim: 32",
        "use_relative_actions: false",
        "observation.images.base_0_rgb",
        "observation.images.wrist_0_rgb",
        "(32,)",
    ):
        assert needle in text
    assert "observation.state" not in text  # only the image keys are named


def test_format_checkpoint_says_when_a_field_is_missing():
    config = pi05_config()
    del config.use_relative_actions

    text = probe.format_checkpoint(config)

    assert "use_relative_actions: unavailable" in text


def test_format_checkpoint_reads_the_postprocessor_widths(tmp_path):
    (tmp_path / "postprocessor.json").write_text(
        json.dumps({"action": {"mean": [0.0] * 8, "std": [1.0] * 8}})
    )
    text = probe.format_checkpoint(pi05_config(pretrained_path=str(tmp_path)))

    assert "action mean/std widths: 8 / 8" in text


def test_format_spacemouse_names_the_node_and_scale():
    text = probe.format_spacemouse("/dev/hidraw0", 350.0)
    assert "/dev/hidraw0" in text
    assert "350" in text

    assert "unavailable" in probe.format_spacemouse(None, None)


def test_the_node_can_be_pinned_for_a_device_with_several_interfaces(monkeypatch, tmp_path):
    """`SPACEMOUSE_HIDRAW` wins over discovery, because a device's HID id names all of its interfaces."""
    import readout_chain
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseReader

    node = tmp_path / "hidraw7"
    node.write_bytes(b"")

    def fake_reader(path, full_scale=350.0):
        reader = type("R", (), {})()
        reader.path, reader.axes, reader.take_press = str(path), (0.0, 0.0, 0.0), lambda: False
        return reader

    monkeypatch.setattr(readout_chain, "SpaceMouseReader", fake_reader)
    monkeypatch.setenv("SPACEMOUSE_HIDRAW", str(node))
    chain = readout_chain.build_chain({"corruption": None, "input_noise": 0.0}, headless=False)
    assert chain.describe() == f"SpaceMouse {node}"
    assert SpaceMouseReader is not None  # the real class is what the fake replaces


def test_the_live_stick_read_decodes_a_translation_report(tmp_path, monkeypatch, capsys):
    """A node that speaks must reach the printed line: the import that decodes it is as load-bearing as the read."""
    import lerobot.teleoperators.spacemouse as spacemouse_package
    import lerobot.teleoperators.spacemouse.teleop_spacemouse as spacemouse_module
    import probe

    node = tmp_path / "hidraw7"
    node.write_bytes(b"\x01" + struct.pack("<hhh", 120, -60, 0))
    # the probe imports the finder from the package and the scale from the module, so both are patched
    monkeypatch.setattr(spacemouse_package, "find_spacemice", lambda: [str(node)])
    monkeypatch.setattr(spacemouse_module, "env_full_scale", lambda: 350.0)

    summary = probe._stick_block(0.6)
    printed = capsys.readouterr().out
    assert "raw (+120, -60, +0)" in printed
    assert "raw (+120, -60, +0)" in summary
