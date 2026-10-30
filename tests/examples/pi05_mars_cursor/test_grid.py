"""The compositional adapter's grid: ring obstacles and asymmetric scenes (the side to pass is part of
the scene), the pure-pursuit expert, the curl user on the command channel, and the grid's sweeps (the
trainers: test_training.py)."""

import numpy as np
import pytest
import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.channel import InputChannel, corruption_spec, curl_commands, serve_command
from cursor_sa.demos import make_demos, plan_reach, replay
from cursor_sa.env import (TaskConfig, curl_steady_state, obstacle_count, obstacle_spec, place_obstacles, plan_route,
                           route_side, sample_layouts)
from cursor_sa.session import arm_task, channel_for, load_checkpoints, run_trials
from run_sweep import load_sweep, method_arm

STUDY = dict(min_reach=0.35, max_reach=0.5)
DISC = {"kind": "disc", "radius": 0.08, "along": 0.5, "offset": [0.012, 0.024]}
RING_GOAL = {"kind": "ring", "around": "goal", "gap_angle": [15, 30]}
RING_START = {"kind": "ring", "around": "start", "gap_angle": [15, 30]}
WALL = {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14, "radius": 0.015, "extent": 0.4}


def _lateral(points, start, goal):
    """Signed distance of points across the reach (+ left)."""
    u = (goal - start) / np.linalg.norm(goal - start)
    return (points - start) @ np.array([-u[1], u[0]])


def test_ring_specs_place_an_arc_open_away_from_the_reach():
    ring = obstacle_spec(RING_GOAL)
    assert ring == {"kind": "ring", "around": "goal", "ring_radius": 0.14, "radius": 0.02, "gap": 0.16,
                    "gap_angle": [15.0, 30.0], "side": "random"}
    assert obstacle_spec(DISC)["offset"] == [0.012, 0.024] and obstacle_spec({"kind": "disc", "radius": 0.04, "along": 0.5})["offset"] == 0.0
    for bad in ({"kind": "ring"}, {"kind": "ring", "around": "target"}, {"kind": "ring", "around": "goal", "gap": 0.3},
                {"kind": "ring", "around": "goal", "gap_angle": -5}, {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": [0.03, 0.01]},
                {"kind": "disc", "radius": 0.04, "along": 0.5, "pass": "left"}):
        with pytest.raises(ValueError, match="task.obstacle"):
            obstacle_spec(bad)
    start, goal = np.array([0.3, 0.5]), np.array([0.7, 0.5])
    straight = dict(ring, gap_angle=0.0)
    for spec, center, away in ((straight, goal, np.array([1.0, 0.0])), (dict(straight, around="start"), start, np.array([-1.0, 0.0]))):
        discs, mask, _ = place_obstacles(None, start, goal, spec)   # no gap angle: symmetric, no random number drawn
        assert mask.all() and len(discs) == obstacle_count(spec)
        assert np.allclose(np.linalg.norm(discs[:, :2] - center, axis=1), 0.14) and np.allclose(discs[:, 2], 0.02)
        assert (np.linalg.norm(np.diff(discs[:, :2], axis=0), axis=1) <= 0.04 + 1e-12).all()   # neighbors touch or overlap
        assert np.isclose(np.linalg.norm(discs[0, :2] - discs[-1, :2]) - 0.04, 0.16)             # the opening is the gap
        mouth = (discs[0, :2] + discs[-1, :2]) / 2
        assert np.allclose((mouth - center) / np.linalg.norm(mouth - center), away)              # facing away from the reach
    # a gap angle turns the gap by 15-30 degrees, to either side
    turns = []
    for seed in range(200):
        discs, _, sign = place_obstacles(np.random.default_rng(seed), start, goal, ring)
        mouth = (discs[0, :2] + discs[-1, :2]) / 2 - goal
        turns.append(np.degrees(np.arctan2(mouth[1], mouth[0])))
        assert np.sign(turns[-1]) == sign
    assert 15 - 1e-9 <= np.abs(turns).min() and np.abs(turns).max() <= 30 + 1e-9 and 0.35 < np.mean(np.array(turns) > 0) < 0.65


@pytest.mark.parametrize("spec", [DISC, RING_GOAL, RING_START])
def test_asymmetric_scenes_decide_the_side_and_both_sides_occur(spec):
    task = TaskConfig(**STUDY, obstacle=obstacle_spec(spec))
    layouts = sample_layouts(np.random.default_rng(3), 60, 4, task)
    agree, sides = [], []
    for b in range(60):
        start, goal, discs = layouts.start[b], layouts.goal_pos[b], layouts.real_obstacles(b)
        others = layouts.distractors(b, task.target_radius)
        assert (np.linalg.norm(others[:, None, :2] - discs[None, :, :2], axis=-1) >= discs[:, 2] + 0.06 - 1e-9).all()
        route = plan_route(start, goal, discs, 0.05, avoid=others)
        assert route is not None
        side = route_side(route, start, goal)
        sides.append(side)
        if spec["kind"] == "disc":   # the shorter way is away from the disc's offset
            scene = "right" if _lateral(discs[:, :2], start, goal)[0] > 0 else "left"
        else:                        # the shorter way is round to the side the gap is turned to
            scene = "left" if _lateral(((discs[0, :2] + discs[-1, :2]) / 2)[None], start, goal)[0] > 0 else "right"
        agree.append(side == scene)
    assert np.mean(agree) > 0.85 and 0.25 < np.mean(np.array(sides) == "left") < 0.75, (np.mean(agree), sides)
    with pytest.raises(ValueError, match="cannot place obstacles"):
        sample_layouts(np.random.default_rng(0), 1, 4, TaskConfig(**STUDY, obstacle=obstacle_spec(dict(RING_GOAL, ring_radius=0.45))))


def test_pursuit_expert_solves_every_grid_task_without_touching_anything():
    arm = config.load_arm("teleop.yaml", {"targets": [4, 4], "reach": [0.35, 0.5], "time_limit_s": 8.0})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    for spec in (DISC, WALL, RING_GOAL, RING_START):
        arm["obstacle"] = obstacle_spec(spec)
        layouts = sample_layouts(np.random.default_rng(4), 40, 4, arm_task(arm))
        recs = run_trials(arm, layouts, ExpertAgent(), InputChannel(None, 0.0), policies[arm["model"]], adapters, "cpu")
        assert [r["outcome"] for r in recs].count("hit") == 40, spec
        if spec is DISC:  # it passes the disc on the side its first route takes, away from the offset
            for r, b in zip(recs, range(40)):
                path = np.array(r["path"])
                start, goal, disc = layouts.start[b], layouts.goal_pos[b], layouts.obstacles[b, 0]
                route = plan_route(start, goal, disc[None], 0.05, avoid=layouts.distractors(b, 0.05))
                along = (path - start) @ ((goal - start) / np.linalg.norm(goal - start))
                c_along = (disc[:2] - start) @ ((goal - start) / np.linalg.norm(goal - start))
                lateral = _lateral(path[np.abs(along - c_along) < 0.02], start, goal) - _lateral(disc[None, :2], start, goal)
                assert (np.sign(lateral) == (1 if route_side(route, start, goal) == "left" else -1)).all()


def test_demos_follow_the_experts_route_and_plain_reaches_are_unchanged():
    for spec in (DISC, WALL, RING_GOAL, RING_START):
        task = TaskConfig(**STUDY, obstacle=obstacle_spec(spec))
        rng = np.random.default_rng(5)
        layouts = sample_layouts(rng, 20, 4, task)
        actions, lengths = make_demos(rng, layouts, task)
        env = replay(actions, layouts, task)
        assert env.success.all() and not env.collided.any()
        np.testing.assert_allclose(layouts.start + actions.sum(1), layouts.goal_pos, atol=1e-9)
        assert np.linalg.norm(actions, axis=-1).max() / task.dt < 2.5          # no faster than the policy's own demos
    # without obstacles a demo is the plain reach, drawing the same random numbers as before
    plain = sample_layouts(np.random.default_rng(6), 1, 4, TaskConfig(**STUDY))
    actions, lengths = make_demos(np.random.default_rng(9), plain, TaskConfig(**STUDY))
    reach = plan_reach(np.random.default_rng(9), plain.start[0], plain.goal_pos[0], TaskConfig(**STUDY))
    np.testing.assert_allclose(actions[0, :len(reach) - 1], np.diff(reach, axis=0))


def test_curl_channel_turns_and_slows_the_command_and_resets_per_batch():
    matrix, curl = corruption_spec({"curl": 20, "tau": 0.1})
    assert matrix is None and curl == {"curl": 20.0, "tau": 0.1}
    assert corruption_spec({"scale": [1.0, 0.25]})[1] is None
    with pytest.raises(ValueError, match="control.corruption"):
        corruption_spec({"curl": 20})
    channel = InputChannel(None, 0.0, curl=curl)
    channel.reset(2)
    raw = np.array([[1.0, 0.0], [0.0, -0.5]])
    for _ in range(200):
        served = channel(raw)
    np.testing.assert_allclose(served, raw @ curl_steady_state(0.1, 20).T, atol=1e-9)   # turned 63 degrees clockwise, slowed
    channel.reset(2)
    first = channel(raw)
    assert np.linalg.norm(first, axis=1).max() < 0.25                                    # from rest it builds up
    np.testing.assert_allclose(channel(np.zeros((2, 2))), np.zeros((2, 2)))              # idle stays idle
    # the training twin serves the same commands step by step
    pushes = np.random.default_rng(0).uniform(-1, 1, (3, 30, 2))
    live = InputChannel(None, 0.0, curl=curl)
    live.reset(3)
    stepped = np.stack([live(pushes[:, k]) for k in range(30)], 1)
    np.testing.assert_allclose(curl_commands(pushes, curl), stepped, atol=1e-12)
    with pytest.raises(ValueError, match="one command per trial"):
        live(pushes[:, :16])
    out = serve_command(torch.tensor([[0.3, 0.2], [0.3, 0.2]]), torch.tensor([[1.0, 0.0], [0.0, 0.0]]), torch.zeros(2))
    np.testing.assert_allclose(out.numpy(), [[0.3, 0.2], [0.0, 0.0]])                   # noise-free core where active
    arm = config.load_arm("teleop.yaml", {"corruption": {"curl": 20, "tau": 0.1}, "input_noise": 0.0})
    assert arm["corruption_matrix"] is None and arm["corruption_curl"] == curl
    with pytest.raises(ValueError, match="needs a corruption to copy"):
        config.load_arm("frs.yaml", {"corruption": {"curl": 20, "tau": 0.1}, "reversal_adapter": {"translation": "corruption"}})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts = sample_layouts(np.random.default_rng(1), 6, 4, TaskConfig(**STUDY))
    recs = run_trials(arm, layouts, ExpertAgent(), channel_for(arm, np.random.default_rng(0)), policies[arm["model"]],
                      adapters, "cpu")
    served = np.array(recs[0]["served_commands"])
    raw = np.array(recs[0]["raw_commands"])
    late = len(raw) // 2  # mid-reach: the served command lags the push and turns it clockwise
    assert np.cross(raw[late], served[late]) < 0 and np.linalg.norm(served[late]) < np.linalg.norm(raw[late])


def test_grid_sweeps_resolve_users_tasks_and_their_modules():
    sweep = load_sweep("grid.yaml")
    cells = sweep["grid"]["cells"]
    assert len(cells) == 20 and sweep["grid"]["users"] == ["U1", "U2", "U3", "U4"]
    assert sorted(n for n, c in cells.items() if c["train"]) == ["U1xT1", "U1xT2", "U2xT1", "U2xT3", "U3xT1", "U3xT4",
                                                                  "U4xT1", "U4xT5"]
    arm = method_arm(sweep, "mars", "U4xT4")
    assert arm["corruption_curl"] == {"curl": 20.0, "tau": 0.1} and arm["obstacle"]["around"] == "goal"
    assert arm["learned_adapter"] == {"user": "user_curl20_expert_n100_s0", "task": "task_ring_goal_n300_s0_with_rot60"}
    assert method_arm(sweep, "mars_user", "U2xT5")["learned_adapter"] == {"user": "user_gain025x_expert_n100_s0", "task": None}
    assert arm["time_limit_s"] == 8.0 and arm["input_noise"] == 1.0
    for name in ("grid_expert_check", "grid_task_check"):
        check = load_sweep(name + ".yaml")
        for level in check["grid"]["levels"]:
            for m in check["methods"]:
                assert method_arm(check, m, level)["mode"] in ("teleop", "policy")
