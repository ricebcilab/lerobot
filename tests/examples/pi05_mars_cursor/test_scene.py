import hashlib

import numpy as np
import pytest
import torch
from cursor_sa.demos import DemoSampler, make_demos
from cursor_sa.env import (
    RENDER_CURSOR,
    RENDER_CURSOR_RADIUS,
    RENDER_PALETTE,
    SCENE_SIZE,
    VIEWS,
    Layouts,
    TaskConfig,
    obstacle_spec,
    render_layouts,
    render_views,
    sample_layouts,
)
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import SharedController, hide_goal, observation


def _layouts(n=3, spec=None, seed=4):
    cfg = TaskConfig(min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec(spec))
    return sample_layouts(np.random.default_rng(seed), n, 4, cfg)


def test_scene_rides_through_indexing_tile_and_concat():
    a = _layouts(3)
    a.scene = {
        "cnn": np.arange(3 * 2 * 4 * 4, dtype=np.uint8).reshape(3, 2, 4, 4),
        "vlm_image": np.ones((3, 5, 6), np.float16),
    }
    assert a[[1]].scene["cnn"].shape == (1, 2, 4, 4) and np.array_equal(
        a[[1]].scene["cnn"][0], a.scene["cnn"][1]
    )
    perm = np.array([2, 0, 1])
    assert np.array_equal(a[perm].scene["vlm_image"], a.scene["vlm_image"][perm])
    assert a.tile(2).scene["cnn"].shape == (6, 2, 4, 4) and np.array_equal(
        a.tile(2).scene["cnn"][3:], a.scene["cnn"]
    )
    both = Layouts.concat([a, a[[0]]])
    assert both.scene["vlm_image"].shape == (4, 5, 6) and set(both.scene) == {"cnn", "vlm_image"}
    plain = _layouts(2)
    assert (
        plain.scene is None and Layouts.concat([plain, plain]).scene is None and plain.tile(2).scene is None
    )
    with pytest.raises(ValueError, match="scene"):
        Layouts.concat([a, plain])


def test_render_draws_discs_and_targets_and_never_the_goal():
    lay = _layouts(2, {"kind": "disc", "radius": 0.08, "along": 0.5})
    img = render_layouts(lay, size=SCENE_SIZE)
    assert img.shape == (2, 3, SCENE_SIZE, SCENE_SIZE) and img.dtype == np.uint8
    x, y, r = lay.real_obstacles(0)[0]
    px, py = int(x * SCENE_SIZE), int(y * SCENE_SIZE)
    assert tuple(img[0, :, py, px]) == RENDER_PALETTE["obstacle"]  # disc center is obstacle colour
    assert tuple(img[0, :, 2, 2]) == RENDER_PALETTE["bg"]  # a corner is background
    gx, gy = (lay.goal_pos[0] * SCENE_SIZE).astype(int)
    ox, oy = (lay.targets[0, (lay.goal[0] + 1) % 4] * SCENE_SIZE).astype(int)
    assert (
        tuple(img[0, :, gy, gx]) == tuple(img[0, :, oy, ox]) == RENDER_PALETTE["target_fill"]
    )  # goal looks like any target
    no_obstacle = render_layouts(_layouts(1))
    assert not (no_obstacle == np.array(RENDER_PALETTE["obstacle"])[None, :, None, None]).all(1).any()


def test_render_keeps_the_wall_visible():
    lay = _layouts(
        1, {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14, "radius": 0.015, "extent": 0.4}
    )
    img = render_layouts(lay).astype(int)
    discs = lay.real_obstacles(0)
    for x, y, _ in discs[::3]:  # along the wall every disc center is well darker than the background
        assert img[0, :, int(y * SCENE_SIZE), int(x * SCENE_SIZE)].mean() < 128


def test_observation_and_sampler_carry_the_scene():
    lay = _layouts(3, {"kind": "disc", "radius": 0.08, "along": 0.5})
    lay.scene = {"cnn": render_layouts(lay, 64)}
    obs = observation(lay, lay.start, np.zeros((3, 4, 2)))
    assert (
        obs["scene"]["cnn"].dtype == torch.uint8
        and obs["scene"]["cnn"].shape == (3, 3, 64, 64)
        and "scene" in hide_goal(obs)
    )
    assert "scene" not in observation(_layouts(2), np.zeros((2, 2)), np.zeros((2, 4, 2)))
    policy = FlowPolicy().eval()
    ctl = SharedController(policy, lay, "policy")
    assert torch.equal(ctl.observation()["scene"]["cnn"], obs["scene"]["cnn"])
    cfg = TaskConfig(
        min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    sampler = DemoSampler(
        lay, *make_demos(np.random.default_rng(0), lay, cfg), policy.horizon, policy.hist, "cpu"
    )
    o, _, i, _ = sampler.sample(5, indexed=True)
    assert torch.equal(o["scene"]["cnn"], torch.as_tensor(lay.scene["cnn"])[i])


def _near(pixel, colour, tol=3):
    return np.abs(pixel.astype(int) - np.array(colour)).max() <= tol


def test_render_layouts_pixels_are_unchanged():
    ring = obstacle_spec({"kind": "ring", "around": "goal", "gap_angle": [15, 30]})
    lay = sample_layouts(
        np.random.default_rng(0), 4, 4, TaskConfig(min_reach=0.35, max_reach=0.5, obstacle=ring)
    )
    assert (
        hashlib.sha256(render_layouts(lay).tobytes()).hexdigest()
        == "6f2b3fdd5379ee5b294c7ef59a09f1b031f83805eb538c8e6acbcef9c2decf5b"
    )


def test_views_draw_the_cursor_and_centre_the_wrist_window_on_it():
    task = TaskConfig(
        min_reach=0.35, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    lay = sample_layouts(np.random.default_rng(0), 3, 4, task)
    disc = lay.obstacles[:, 0, :2]
    pos = disc + np.where(disc > 0.5, -1.0, 1.0) * np.array(
        [0.1, 0.05]
    )  # beside the disc, screen-centre side
    views = render_views(lay, pos, 0.5)
    assert VIEWS == ("overhead", "wrist") and views.shape == (3, 2, 3, 224, 224) and views.dtype == np.uint8
    base = render_layouts(lay)
    reach = int(np.ceil(RENDER_CURSOR_RADIUS * 224)) + 5
    for b in range(3):
        cx, cy = (pos[b] * 224).astype(int)
        assert _near(views[b, 0, :, cy, cx], RENDER_CURSOR) and _near(views[b, 1, :, 112, 112], RENDER_CURSOR)
        far = np.ones((224, 224), bool)
        far[max(cy - reach, 0) : cy + reach + 1, max(cx - reach, 0) : cx + reach + 1] = False
        assert np.array_equal(views[b, 0][:, far], base[b][:, far])  # away from the mark: render_layouts'
        col, row = ((disc[b] - pos[b]) / 0.5 + 0.5) * 224  # where the window puts the disc's centre
        assert _near(views[b, 1, :, int(row), int(col)], RENDER_PALETTE["obstacle"])


def test_a_wrist_window_as_wide_as_the_screen_at_its_centre_is_the_overhead_view():
    lay = sample_layouts(np.random.default_rng(1), 2, 4, TaskConfig(min_reach=0.35, max_reach=0.5))
    views = render_views(lay, np.full((2, 2), 0.5), 1.0)
    assert np.array_equal(views[:, 1], views[:, 0])
    corner = render_views(lay, np.full((2, 2), 0.05), 0.5)  # the window reaches 0.2 outside the screen
    assert (corner[:, 1, :, 0, 0] == 255).all()
    for bad in (
        lambda: render_views(lay, np.full((2, 2), 0.5), 0.0),
        lambda: render_views(lay, np.full((1, 2), 0.5), 0.5),
    ):
        with pytest.raises(ValueError):
            bad()
