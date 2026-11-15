"""The cursor example's keyboard: lerobot's held-keys reader on the screen's two axes (the stick is pinned in
test_spacemouse_equivalence.py)."""

import numpy as np
from cursor_sa import devices


def test_the_arrow_keys_push_at_half_deflection_and_full_with_shift():
    keys = devices.Keyboard()
    keys.hold(["right"])
    np.testing.assert_array_equal(keys.translation, [0.5, 0.0])
    keys.hold(["right", "up", "shift"])
    np.testing.assert_allclose(keys.translation, np.array([1.0, 1.0]) / np.sqrt(2))  # a diagonal is no faster
    keys.hold([])
    np.testing.assert_array_equal(keys.translation, [0.0, 0.0])
