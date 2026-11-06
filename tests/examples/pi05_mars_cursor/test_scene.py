"""The env's pictures: the overhead render, the two views, and the scene arrays the observation and samplers carry."""

import hashlib

import numpy as np
import pytest
import torch
from _view_stub import ViewStub
from cursor_sa.demos import DemoSampler, frame_steps, make_demos
from cursor_sa.policy import FlowPolicy
from cursor_sa.steering import SharedController, hide_goal, observation, view_scene_fn

from lerobot.envs.cursor import (
    RENDER_CURSOR,
    RENDER_CURSOR_RADIUS,
    RENDER_PALETTE,
    SCENE_SIZE,
    VIEWS,
    TaskConfig,
    obstacle_spec,
    render_layouts,
    render_views,
    sample_layouts,
)
from lerobot.policies.steering import LearnedTaskModule, encode_views

DISC = {"kind": "disc", "radius": 0.08, "along": 0.5}
RING = {"kind": "ring", "around": "goal", "gap_angle": [15, 30]}


def _layouts(n=3, spec=None, seed=4, reach=(0.3, 0.5)):
    cfg = TaskConfig(min_reach=reach[0], max_reach=reach[1], obstacle=obstacle_spec(spec))
    return sample_layouts(np.random.default_rng(seed), n, 4, cfg)


def _near(pixel, colour, tol=3):
    return np.abs(pixel.astype(int) - np.array(colour)).max() <= tol


def test_render_draws_discs_walls_and_targets_and_never_the_goal():
    lay = _layouts(2, DISC)
    img = render_layouts(lay, size=SCENE_SIZE)
    assert img.shape == (2, 3, SCENE_SIZE, SCENE_SIZE) and img.dtype == np.uint8
    x, y, _ = lay.real_obstacles(0)[0]
    assert tuple(img[0, :, int(y * SCENE_SIZE), int(x * SCENE_SIZE)]) == RENDER_PALETTE["obstacle"]
    assert tuple(img[0, :, 2, 2]) == RENDER_PALETTE["bg"]  # a corner is background
    gx, gy = (lay.goal_pos[0] * SCENE_SIZE).astype(int)
    ox, oy = (lay.targets[0, (lay.goal[0] + 1) % 4] * SCENE_SIZE).astype(int)
    assert (
        tuple(img[0, :, gy, gx]) == tuple(img[0, :, oy, ox]) == RENDER_PALETTE["target_fill"]
    )  # the goal is any target
    no_obstacle = render_layouts(_layouts(1))
    assert not (no_obstacle == np.array(RENDER_PALETTE["obstacle"])[None, :, None, None]).all(1).any()
    wall = _layouts(
        1, {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14, "radius": 0.015, "extent": 0.4}
    )
    img = render_layouts(wall).astype(int)
    for x, y, _ in wall.real_obstacles(0)[
        ::3
    ]:  # along the wall every disc centre is well darker than the background
        assert img[0, :, int(y * SCENE_SIZE), int(x * SCENE_SIZE)].mean() < 128
    ring = render_layouts(_layouts(4, RING, seed=0, reach=(0.35, 0.5)))  # the pixels as they were
    assert (
        hashlib.sha256(ring.tobytes()).hexdigest()
        == "6f2b3fdd5379ee5b294c7ef59a09f1b031f83805eb538c8e6acbcef9c2decf5b"
    )


def test_observation_and_sampler_carry_the_scene():
    lay = _layouts(3, DISC)
    lay.scene = {"cnn": render_layouts(lay, 64)}
    obs = observation(lay, lay.start, np.zeros((3, 4, 2)))
    assert obs["scene"]["cnn"].dtype == torch.uint8 and obs["scene"]["cnn"].shape == (3, 3, 64, 64)
    assert "scene" in hide_goal(obs) and "scene" not in observation(
        _layouts(2), np.zeros((2, 2)), np.zeros((2, 4, 2))
    )
    policy = FlowPolicy().eval()
    assert torch.equal(
        SharedController(policy, lay, "policy").observation()["scene"]["cnn"], obs["scene"]["cnn"]
    )
    cfg = TaskConfig(min_reach=0.3, max_reach=0.5, obstacle=obstacle_spec(DISC))
    sampler = DemoSampler(
        lay, *make_demos(np.random.default_rng(0), lay, cfg), policy.horizon, policy.hist, "cpu"
    )
    o, _, i, _ = sampler.sample(5, indexed=True)
    assert torch.equal(o["scene"]["cnn"], torch.as_tensor(lay.scene["cnn"])[i])


def test_views_draw_the_cursor_and_centre_the_wrist_window_on_it():
    lay = _layouts(3, DISC, seed=0, reach=(0.35, 0.5))
    disc = lay.obstacles[:, 0, :2]
    pos = disc + np.where(disc > 0.5, -1.0, 1.0) * np.array(
        [0.1, 0.05]
    )  # beside the disc, screen-centre side
    views, base = render_views(lay, pos, 0.5), render_layouts(lay)
    assert VIEWS == ("overhead", "wrist") and views.shape == (3, 2, 3, 224, 224) and views.dtype == np.uint8
    reach = int(np.ceil(RENDER_CURSOR_RADIUS * 224)) + 5
    for b in range(3):
        cx, cy = (pos[b] * 224).astype(int)
        assert _near(views[b, 0, :, cy, cx], RENDER_CURSOR) and _near(views[b, 1, :, 112, 112], RENDER_CURSOR)
        far = np.ones((224, 224), bool)
        far[max(cy - reach, 0) : cy + reach + 1, max(cx - reach, 0) : cx + reach + 1] = False
        assert np.array_equal(views[b, 0][:, far], base[b][:, far])  # away from the mark: render_layouts'
        col, row = ((disc[b] - pos[b]) / 0.5 + 0.5) * 224  # where the window puts the disc's centre
        assert _near(views[b, 1, :, int(row), int(col)], RENDER_PALETTE["obstacle"])
    wide = render_views(lay[:2], np.full((2, 2), 0.5), 1.0)  # a window as wide as the screen at its centre
    assert np.array_equal(wide[:, 1], wide[:, 0])
    assert (
        render_views(lay[:2], np.full((2, 2), 0.05), 0.5)[:, 1, :, 0, 0] == 255
    ).all()  # beyond the screen: white
    for span, n in ((0.0, 2), (0.5, 1)):
        with pytest.raises(ValueError):
            render_views(lay[:2], np.full((n, 2), 0.5), span)
    # a ring disc straddling a screen edge: the wrist window shows the part on the screen and white beyond it
    ring = _layouts(40, RING, seed=0, reach=(0.35, 0.5))
    found = [(b, c, r, axis, edge) for b, k in np.argwhere(ring.obstacle_mask) for c, r in [(ring.obstacles[b, k, :2], ring.obstacles[b, k, 2])]
             for axis in (0, 1) for edge in (0.0, 1.0) if abs(c[axis] - edge) < 0.8 * r and 0.2 < c[1 - axis] < 0.8]  # fmt: skip
    assert found, "no sampled disc straddles a screen edge"
    b, c, r, axis, edge = found[0]
    s = 1.0 if edge == 0.0 else -1.0  # into the screen
    pos, inside, outside = c.copy(), c.copy(), c.copy()
    pos[axis], inside[axis], outside[axis] = (
        edge + 0.1 * s,
        (edge + c[axis] + s * r) / 2,
        (edge + c[axis] - s * r) / 2,
    )
    view = render_views(ring[[b]], pos[None], 0.5)[0, 1]
    (ci, ri), (co, ro) = (((p - pos) / 0.5 + 0.5) * 224 for p in (inside, outside))
    assert (
        _near(view[:, int(ri), int(ci)], RENDER_PALETTE["obstacle"])
        and (view[:, int(ro), int(co)] == 255).all()
    )


def test_the_sampler_reads_each_samples_own_frame():
    demo, step, start = frame_steps(np.array([5, 0, 4, 1]), 2)
    assert (
        demo.tolist() == [0, 0, 0, 2, 2, 3]
        and step.tolist() == [0, 2, 4, 0, 2, 0]
        and start.tolist() == [0, 3, 3, 5]
    )
    assert frame_steps(np.array([3]), 1)[1].tolist() == [0, 1, 2]
    rng, task, policy = np.random.default_rng(0), TaskConfig(min_reach=0.35, max_reach=0.5), FlowPolicy()
    lay = sample_layouts(rng, 6, 4, task)
    actions, lengths = make_demos(rng, lay, task)
    demo, step, start = frame_steps(lengths, 2)
    tokens = torch.stack([torch.as_tensor(demo), torch.as_tensor(step)], -1).float()[
        :, None
    ]  # each frame names itself
    args = (lay, actions, lengths, policy.horizon, policy.hist, "cpu")
    sampler = DemoSampler(*args, frames={"vlm_views": (tokens, start, 2)})
    obs, _, i, k = sampler.sample(512, indexed=True)
    assert (k % 2 == 0).all() and (k < sampler.lengths[i]).all() and set(k.tolist()) >= {0, 2, 4}
    got = obs["scene"]["vlm_views"][:, 0]
    assert torch.equal(got[:, 0].long(), i) and torch.equal(got[:, 1].long(), k)
    torch.testing.assert_close(obs["pos"], sampler.pos[i, k])
    plain = DemoSampler(*args)
    assert "scene" not in plain.sample(8)[0]
    torch.manual_seed(5)
    a = plain.sample(64, indexed=True)
    torch.manual_seed(5)
    b = DemoSampler(*args, frames=None).sample(64, indexed=True)
    assert torch.equal(a[3], b[3]) and (a[3] % 2 == 1).any()  # no frames: every step, the same draws
    with pytest.raises(ValueError):
        DemoSampler(*args, frames={"a": (tokens, start, 2), "b": (tokens, start, 4)})


def test_a_controller_encodes_the_running_trials_views_once_per_position():
    lay, calls = sample_layouts(np.random.default_rng(0), 4, 4, TaskConfig(min_reach=0.35, max_reach=0.5)), []

    def scene_fn(layouts, pos, active):
        calls.append(None if active is None else active.copy())
        return {"vlm_views": torch.full((len(layouts), 2, 3), float(len(calls)))}

    ctl = SharedController(FlowPolicy(), lay, "policy", scene_fn=scene_fn)
    a, b = ctl.observation(), ctl.observation(goal_given=True)
    assert (
        len(calls) == 1 and torch.equal(a["scene"]["vlm_views"], b["scene"]["vlm_views"]) and calls[0] is None
    )
    ctl.pos = ctl.pos + 0.01
    assert float(ctl.observation()["scene"]["vlm_views"][0, 0, 0]) == 2.0 and len(calls) == 2
    ctl.active = np.array([True, False, True, True])
    ctl.observation()
    assert len(calls) == 3 and calls[2].tolist() == [True, False, True, True]
    assert "scene" not in SharedController(FlowPolicy(), lay, "policy").observation()
    module = LearnedTaskModule(
        horizon=16, token_source="vlm_views", scene_dim=2048, scene_tokens=512, image_size=56, wrist_span=0.5
    )
    enc, pos = ViewStub(), lay.start + 0.05
    fn = view_scene_fn(module, enc)
    rng = torch.get_rng_state()
    full = fn(lay, pos, None)["vlm_views"]
    assert torch.equal(torch.get_rng_state(), rng)  # the encoder draws on forked streams
    assert full.shape == (4, 512, 2048) and full.dtype == torch.float16 and enc.images == 8
    assert torch.equal(full, encode_views(render_views(lay, pos, 0.5, 56), ViewStub()))
    active = np.array([True, False, True, False])
    part = fn(lay, pos, active)["vlm_views"]
    assert enc.images == 12 and torch.equal(part[active], full[active]) and not part[~active].any()
    assert fn(lay, pos, np.zeros(4, bool))["vlm_views"].shape == (4, 512, 2048) and enc.images == 12
