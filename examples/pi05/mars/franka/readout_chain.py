#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The operator's input chain for the Franka: stick -> axis map -> unit limit / gain -> readout.

The runner needs one thing from the person at the bench: a command, seven numbers in ``[-1, 1]``, that
the steering policy steers toward. This module produces the three-dimensional translation part of it
(the grip is a button, and the task's gripper is delegated to the policy). The corruption and noise are
lerobot's shared operator readout, in the robot's 3-D configuration and with the robot's noise rule
(``noise_gate="command"``), so the Franka and the LIBERO session perturb a person the same way.

The chain is deliberately thin: the SpaceMouse reader is passed in, so tests use a fake stick, and
`build_chain` is the only place that touches a device. A headless run gets `NullChain`, which is idle and
open, so the rest of the example never branches on whether a SpaceMouse exists.
"""

from __future__ import annotations

import numpy as np

from lerobot.teleoperators import readout
from lerobot.teleoperators.spacemouse import SpaceMouseReader, find_spacemouse
from lerobot.teleoperators.spacemouse.teleop_spacemouse import env_full_scale

GRIPPER_CLOSE = 1.0
GRIPPER_OPEN = -1.0

# The SpaceMouse's device axes (x: right +, y: toward you +, z: down +; already scaled to [-1, 1] by
# SpaceMouseReader.axes) mapped onto the station's base frame (x forward, y left, z up). Unverified until
# the jog test: if a push moves the arm the wrong way, flip that axis's sign here, once, and record it in
# DECISION.md; if two axes are crossed, swap their AXIS_SOURCE entries.
AXIS_SOURCE = np.array([1, 0, 2], dtype=np.intp)  # device channel feeding each base axis
AXIS_SIGN = np.array([-1.0, -1.0, -1.0])


def map_axes(device_xyz) -> np.ndarray:
    """The device's (x, y, z) in the station's base frame, shape (3), device values in [-1, 1]."""
    return np.asarray(device_xyz, dtype=np.float64)[AXIS_SOURCE] * AXIS_SIGN


class SpaceMouseChain:
    """The operator's command, from a SpaceMouse reader through a gain and the shared readout.

    `translation` is what the policy steers toward: the stick's push, unit-limited and scaled by `gain`
    (None is unity), then served by `readout` (a corruption matrix, a curl field or a keypad, then noise).
    It stays in [-1, 1]. `advance()` steps the readout's curl clock once per control step, under the last
    push read, so a curl keeps its dynamics whether the caller reads every step or once per chunk.
    `grip()` is +1.0 while the gripper is closed and -1.0 while it is open; each button press toggles it.
    """

    def __init__(self, reader, readout, gain=None):
        if gain is not None and not 0.0 < float(gain) <= 1.0:
            raise ValueError(f"gain must be a number in (0, 1] or None, got {gain!r}")
        self.reader = reader
        self.readout = readout
        self.gain = None if gain is None else float(gain)
        self._raw = np.zeros(3)
        self._closed = False

    @property
    def translation(self) -> np.ndarray:
        """The served command, (3,) in [-1, 1]: the readout's answer for the stick's current push."""
        self._raw = self._command()
        return self.readout.read(self._raw)

    def _command(self) -> np.ndarray:
        """The base-frame push, unit-limited (a diagonal push is shortened) and gained."""
        command = map_axes(self.reader.axes)
        command = command / max(1.0, float(np.linalg.norm(command)))
        return command if self.gain is None else self.gain * command

    def advance(self) -> None:
        """One control step of the readout's curl clock; a matrix or keypad readout ignores it."""
        self.readout.step(self._raw)

    def grip(self) -> float:
        """+1.0 closed, -1.0 open; one press toggles, and a press is consumed once."""
        if self.reader.take_press():
            self._closed = not self._closed
        return GRIPPER_CLOSE if self._closed else GRIPPER_OPEN

    def reset(self, n_trials: int = 1) -> None:
        """A new trial: the curl's velocity starts from rest and the gripper opens.

        One trial is the unbatched form here, so the velocity matches the (3,) push `translation` reads;
        more trials keep the readout's batched width for a caller that reads stacks of them.
        """
        self.readout.reset(None if n_trials == 1 else n_trials)
        self._raw = np.zeros(3)
        self._closed = False


class NullChain:
    """The headless operator: an idle stick and an open gripper.

    It stands in for a SpaceMouse during tests and ``--dry-run``; `why` records which of the two reasons
    (headless, or no device) put it here, so the runner can print it once.
    """

    def __init__(self, why: str = "headless"):
        self.why = why

    @property
    def translation(self) -> np.ndarray:
        return np.zeros(3)

    def advance(self) -> None:
        pass

    def grip(self) -> float:
        return GRIPPER_OPEN

    def reset(self, n_trials: int = 1) -> None:
        pass


def _build_readout(settings) -> readout.Readout:
    """The shared readout in 3-D with the robot's noise rule, from a settings dict's corruption and noise.

    `settings` is ``{"corruption": <spec or None>, "input_noise": <std>}``; the corruption is parsed by
    lerobot's grammar (matrix, curl, quantize), so the Franka config reads like LIBERO's.
    """
    corruption = settings.get("corruption")
    quantize = readout.quantize_spec(corruption)
    matrix, curl = (None, None) if quantize else readout.corruption_spec(corruption, n_dims=3)
    noise = float(settings.get("input_noise", 0.0))
    if not np.isfinite(noise) or noise < 0:
        raise ValueError(f"input_noise must be a finite number >= 0, got {noise}")
    return readout.Readout(
        corruption=matrix,
        input_noise=noise,
        curl=curl,
        n_dims=3,
        noise_gate="command",
        quantize=quantize,
    )


def build_chain(settings, *, headless: bool) -> SpaceMouseChain | NullChain:
    """The operator source the runner reads: a SpaceMouse chain, or `NullChain` with the reason.

    The readout is built first, even headless, so a bad corruption spec fails at startup rather than
    mid-trial. A missing or unreadable device is not an error: the runner gets an idle operator and can
    still record, which is what ``--dry-run`` needs.
    """
    served = _build_readout(settings)
    if headless:
        return NullChain("headless")
    try:
        reader = SpaceMouseReader(find_spacemouse(), full_scale=env_full_scale())
    except OSError as e:  # FileNotFoundError, PermissionError: no device, or no access to it
        return NullChain(f"no SpaceMouse: {e}")
    return SpaceMouseChain(reader, served)


__all__ = [
    "AXIS_SIGN",
    "AXIS_SOURCE",
    "GRIPPER_CLOSE",
    "GRIPPER_OPEN",
    "NullChain",
    "SpaceMouseChain",
    "build_chain",
    "map_axes",
]
