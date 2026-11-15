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

import numpy as np
import pytest

from lerobot.teleoperators.keyboard import KeyboardAxes


def test_held_keys_become_a_command_on_the_applications_axes():
    now = [0.0]
    # pushes in a stick's channels, mapped like the LIBERO example's stick: axis i = sign[i] * channel[source[i]]
    keys = KeyboardAxes(
        {"ArrowUp": (0, -1, 0), "ArrowRight": (1, 0, 0), "w": (0, 0, -1), "PageUp": (0, 0, -1)},
        source=[1, 0, 2],
        sign=[1.0, -1.0, -1.0],
        stale_after=1.0,
        clock=lambda: now[0],
    )
    np.testing.assert_array_equal(keys.translation, 0)
    keys.hold({"ArrowUp", "ArrowRight"})
    np.testing.assert_array_equal(keys.translation, [-0.5, -0.5, 0.0])  # half deflection per axis
    keys.hold(
        {"arrowup", "Shift", "w", "PageUp", "x"}
    )  # names in any case; shift is fast; unknown keys ignored
    np.testing.assert_array_equal(
        keys.translation, [-1.0, 0.0, 1.0]
    )  # two keys on one axis clip at full scale
    assert keys.held == {"arrowup", "Shift", "w", "PageUp", "x"}
    now[0] = 2.0  # no report for longer than stale_after: the keys are dropped
    np.testing.assert_array_equal(keys.translation, 0)
    assert keys.held == set()


def test_a_unit_keyboard_is_no_faster_on_a_diagonal():
    keys = KeyboardAxes({"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}, unit=True)
    keys.hold(["right", "up"])
    np.testing.assert_allclose(keys.translation, np.array([1.0, 1.0]) / np.sqrt(2) * 0.5)
    keys.hold(["right", "left"])  # opposite keys cancel
    np.testing.assert_array_equal(keys.translation, [0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        KeyboardAxes({"a": (1, 0), "b": (1, 0, 0)})
