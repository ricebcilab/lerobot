"""Rigid motions of a cursor layout and of its actions."""

import numpy as np
from cursor_sa.env import Layouts, layouts_on_screen, move_actions, move_layouts


def _one(start, goal, discs=()):
    """A one-trial layout from a start, a goal and optional discs (x, y, radius)."""
    targets = np.array([goal], float)[None]
    discs = np.array(discs, float).reshape(1, -1, 3)
    return Layouts(
        np.array([start], float),
        targets,
        np.ones((1, 1), bool),
        np.array([0]),
        discs,
        np.ones(discs.shape[:2], bool),
    )


def _lay():
    rng = np.random.default_rng(0)
    return Layouts(
        rng.uniform(0.2, 0.8, (3, 2)),
        rng.uniform(0.2, 0.8, (3, 4, 2)),
        np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], bool),
        np.array([0, 1, 2]),
        np.concatenate([rng.uniform(0.2, 0.8, (3, 2, 2)), rng.uniform(0.02, 0.08, (3, 2, 1))], 2),
        np.array([[1, 1], [1, 0], [0, 0]], bool),
    )


def _points(lay):
    return np.concatenate([lay.start[:, None], lay.targets, lay.obstacles[..., :2]], 1)


def test_a_zero_motion_is_the_identity_and_drops_the_scene():
    lay = _lay()
    lay.scene = {"x": np.zeros((len(lay), 2))}
    out = move_layouts(lay, np.zeros(len(lay)), np.zeros((len(lay), 2)), np.zeros(len(lay), bool))
    for a, b in [(out.start, lay.start), (out.targets, lay.targets), (out.obstacles, lay.obstacles)]:
        assert np.allclose(a, b)
    assert (out.mask == lay.mask).all() and (out.goal == lay.goal).all()
    assert (out.obstacle_mask == lay.obstacle_mask).all()
    assert out.scene is None


def test_a_quarter_turn_about_the_centre_and_a_mirror():
    lay = _one(start=(0.7, 0.5), goal=(0.5, 0.8))
    out = move_layouts(lay, np.array([np.pi / 2]), np.zeros((1, 2)), np.array([False]))
    assert np.allclose(out.start[0], (0.5, 0.7)) and np.allclose(out.goal_pos[0], (0.2, 0.5))
    out = move_layouts(lay, np.zeros(1), np.array([[0.1, -0.1]]), np.array([True]))
    assert np.allclose(out.start[0], (0.4, 0.4))


def test_a_motion_is_rigid_and_keeps_radii():
    lay = _lay()
    rng = np.random.default_rng(1)
    out = move_layouts(
        lay, rng.uniform(-np.pi, np.pi, 3), rng.uniform(-0.1, 0.1, (3, 2)), np.array([True, False, True])
    )
    before, after = _points(lay), _points(out)
    d0 = np.linalg.norm(before[:, :, None] - before[:, None], axis=-1)
    d1 = np.linalg.norm(after[:, :, None] - after[:, None], axis=-1)
    assert np.allclose(d0, d1)
    assert (out.obstacles[..., 2] == lay.obstacles[..., 2]).all()


def test_moved_actions_follow_the_moved_start():
    rng = np.random.default_rng(2)
    angle, shift, mirror = (
        rng.uniform(-np.pi, np.pi, 3),
        rng.uniform(-0.1, 0.1, (3, 2)),
        np.array([True, False, True]),
    )
    start = rng.uniform(0.3, 0.7, (3, 2))
    actions = rng.uniform(-0.05, 0.05, (3, 6, 2))
    path = start[:, None] + np.cumsum(actions, 1)

    lay = Layouts(start, path, np.ones(path.shape[:2], bool), np.zeros(3, int))
    moved = move_layouts(lay, angle, shift, mirror)
    got = moved.start[:, None] + np.cumsum(move_actions(actions, angle, mirror), 1)
    assert np.allclose(got, moved.targets)


def test_on_screen_needs_the_discs_and_targets_inside_the_edge():
    assert not layouts_on_screen(_one((0.5, 0.5), (0.8, 0.5), [[0.05, 0.5, 0.08]]))[0]
    inward = move_layouts(
        _one((0.5, 0.5), (0.8, 0.5), [[0.05, 0.5, 0.08]]),
        np.zeros(1),
        np.array([[0.1, 0.0]]),
        np.zeros(1, bool),
    )
    assert layouts_on_screen(inward)[0]
    assert not layouts_on_screen(_one((0.5, 0.5), (0.995, 0.5)))[0]
    assert layouts_on_screen(_one((0.5, 0.5), (0.8, 0.5)))[0]
    # padding discs and padding targets do not count
    lay = _lay()
    lay.targets[1, 2:] = 5.0
    lay.obstacles[2] = 5.0
    assert layouts_on_screen(lay).all()
