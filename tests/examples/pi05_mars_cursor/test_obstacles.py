"""Obstacle tasks: layouts with discs (and decoys), the collision outcome, the waypoint planner, obstacle-avoiding
demos and the obstacle-aware expert."""

import hashlib

import numpy as np
import pytest

import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.analysis import trial_metrics
from lerobot.teleoperators.readout import Readout
from cursor_sa.demos import DemoSampler, make_demos, plan_reach, replay
from lerobot.envs.cursor import (CursorEnv, Layouts, TaskConfig, center_out, obstacle_spec, plan_route, route_side,
                           sample_layouts, segment_distance)
from cursor_sa.records import OUTCOMES, arm_fields, outcome_of, summarize
from cursor_sa.session import arm_task, load_checkpoints, run_trials

DISC = {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05}
WALL = {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14}
STUDY = dict(min_reach=0.35, max_reach=0.5)
GRID_DISC = {"kind": "disc", "radius": 0.08, "along": 0.5, "offset": [0.012, 0.024]}
GRID_WALL = {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14, "radius": 0.015, "extent": 0.4}
RING_GOAL = {"kind": "ring", "around": "goal", "gap_angle": [15, 30]}
RING_START = {"kind": "ring", "around": "start", "gap_angle": [15, 30]}


def _one(start, goal, obstacles, extra_targets=()):
    """A single-trial layout with the goal first."""
    targets = np.array([goal, *extra_targets], float)[None]
    obstacles = np.array(obstacles, float).reshape(1, -1, 3)
    return Layouts(np.array([start], float), targets, np.ones(targets.shape[:2], bool), np.array([0]),
                   obstacles, np.ones(obstacles.shape[:2], bool))


def _lateral(points, start, goal):
    """Signed distance of points across the reach (+ left)."""
    u = (goal - start) / np.linalg.norm(goal - start)
    return (points - start) @ np.array([-u[1], u[0]])


def test_layouts_are_bit_identical_to_before():
    # recorded from sample_layouts before obstacles existed: the same seed must give the same layouts
    layouts = sample_layouts(np.random.default_rng(0), 3, (1, 2), TaskConfig())
    np.testing.assert_array_equal(layouts.start, [[0.7065469048855986, 0.5392624923188806],
                                                  [0.08022701777491792, 0.7066899017869497],
                                                  [0.43041849907789254, 0.07548770403091667]])
    np.testing.assert_array_equal(layouts.targets[2], [[0.2080900585423031, 0.8268610301148979],
                                                       [0.5373150982241826, 0.31974070148364625]])
    assert layouts.goal.tolist() == [0, 0, 0] and layouts.obstacles.shape == (3, 0, 3) and layouts.obstacle_mask.shape == (3, 0)
    study = sample_layouts(np.random.default_rng(5), 4, 4, TaskConfig(**STUDY))
    np.testing.assert_array_equal(study.start[0], [0.4031641979013003, 0.4937207168585683])
    assert study.goal.tolist() == [2, 3, 0, 3]
    ring = center_out(8)
    assert sorted(ring.goal) == list(range(8)) and np.allclose(ring.start, 0.5) and ring.obstacles.shape == (8, 0, 3)
    # digests recorded at 2f3e3ce7, before the decoy key existed: the key absent, False, or dropped from a
    # normalized spec (as one stored before it) all give those layouts, bit for bit
    recorded = {"disc": "935f889dae666006", "ring": "4afdd3668314bb52"}
    fields = ("start", "targets", "mask", "goal", "obstacles", "obstacle_mask")
    for spec in (DISC, RING_GOAL):
        bare = {k: v for k, v in obstacle_spec(spec).items() if k != "decoy"}
        variants = [obstacle_spec(spec), obstacle_spec(dict(spec, decoy=False)), bare]
        digests = [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()]
        for seed in (0, 1, 2):
            lays = [sample_layouts(np.random.default_rng(seed), 6, (2, 4), TaskConfig(**STUDY, obstacle=v))
                    for v in variants]
            for lay, h in zip(lays, digests):
                assert all(np.array_equal(getattr(lay, f), getattr(lays[0], f)) for f in fields)
                for f in fields:
                    h.update(np.ascontiguousarray(getattr(lay, f)).tobytes())
        assert {h.hexdigest()[:16] for h in digests} == {recorded[spec["kind"]]}


def test_obstacle_specs_place_discs_and_walls_relative_to_the_reach():
    assert obstacle_spec(None) is None
    disc = obstacle_spec(DISC)
    assert disc == {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05, "side": "random", "decoy": False}
    wall = obstacle_spec(WALL)
    assert wall["radius"] == 0.03 and wall["extent"] == 0.25 and wall["side"] == "random"
    for bad in ({"kind": "square"}, {"kind": "disc", "radius": -1, "along": 0.5}, {"kind": "wall", "along": 1.5, "gap": 0.1}):
        with pytest.raises(ValueError, match="task.obstacle"):
            obstacle_spec(bad)

    layouts = sample_layouts(np.random.default_rng(1), 300, 4, TaskConfig(**STUDY, obstacle=disc))
    assert layouts.obstacles.shape == (300, 1, 3) and layouts.obstacle_mask.all()
    sides = []
    for b in range(300):
        start, goal, (cx, cy, r) = layouts.start[b], layouts.goal_pos[b], layouts.obstacles[b, 0]
        u = (goal - start) / np.linalg.norm(goal - start)
        rel = np.array([cx, cy]) - start
        assert np.isclose(rel @ u, 0.5 * np.linalg.norm(goal - start)) and r == 0.04
        across = rel @ np.array([-u[1], u[0]])
        assert np.isclose(abs(across), 0.05)
        sides.append(np.sign(across))
        d_targets = np.linalg.norm(layouts.targets[b] - [cx, cy], axis=1)
        assert d_targets.min() >= r + 0.05 + 0.01 - 1e-9 and np.linalg.norm(start - [cx, cy]) >= r + 0.03 - 1e-9
    assert 0.35 < np.mean(np.array(sides) > 0) < 0.65

    walls = sample_layouts(np.random.default_rng(2), 100, 4, TaskConfig(**STUDY, obstacle=wall))
    assert walls.obstacles.shape[1] >= 8 and walls.obstacle_mask.any(1).all() and not walls.obstacle_mask.all()
    for b in range(100):
        start, goal = walls.start[b], walls.goal_pos[b]
        u = (goal - start) / np.linalg.norm(goal - start)
        real = walls.obstacles[b, walls.obstacle_mask[b]]
        assert np.allclose((real[:, :2] - start) @ u, 0.5 * np.linalg.norm(goal - start))  # one line across the reach
        across = (real[:, :2] - start) @ np.array([-u[1], u[0]])
        gap_side = 1.0 if (across > 1e-9).sum() < (across < -1e-9).sum() else -1.0  # the gap is on the sparser side
        assert (np.abs(across - gap_side * 0.14) >= 0.08 + 0.03 - 1e-9).all()  # no disc within gap/2 + r of the gap center
    with pytest.raises(ValueError, match="cannot place obstacles"):
        sample_layouts(np.random.default_rng(0), 1, 4, TaskConfig(**STUDY, obstacle=obstacle_spec(dict(DISC, radius=0.4))))


def test_collision_ends_the_trial_and_is_an_outcome():
    assert "collision" in OUTCOMES
    layouts = _one((0.2, 0.5), (0.8, 0.5), [[0.5, 0.5, 0.05]])
    env, path = CursorEnv(layouts, TaskConfig()), [layouts.start.copy()]
    while not env.done.all():
        env.step(np.array([[0.024, 0.0]]))
        path.append(env.pos.copy())
    assert env.collided[0] and env.selected[0] == -1 and env.steps[0] == env.t and env.t < 30
    assert env.pos[0, 0] < 0.5 and outcome_of(env, 0) == "collision"
    metrics = trial_metrics(env, np.stack(path, 1), layouts, TaskConfig())
    assert metrics["collision"][0] and not metrics["timeout"][0] and not metrics["success"][0]
    # a step that jumps across a thin disc still collides (segment test, not a point test); beside it, nothing
    for y, hit in ((0.5, True), (0.6, False)):
        env = CursorEnv(_one((0.45, 0.5), (0.8, 0.5), [[0.5, y, 0.01]]), TaskConfig())
        env.step(np.array([[0.1, 0.0]]))
        assert env.collided[0] == env.done[0] == hit
    recs = [dict(outcome="hit", duration_s=1.0), dict(outcome="collision", duration_s=0.5), dict(outcome="timeout", duration_s=5.0)]
    assert summarize(recs)["collision"] == pytest.approx(1 / 3)


def test_waypoints_route_around_discs_and_through_the_gap():
    start, goal = np.array([0.2, 0.5]), np.array([0.8, 0.5])
    np.testing.assert_array_equal(plan_route(start, goal, np.zeros((0, 3))), [goal])
    disc = np.array([[0.5, 0.5, 0.06]])
    path = plan_route(start, goal, disc, clearance=0.03)
    assert len(path) >= 2 and np.allclose(path[-1], goal)
    for p0, p1 in zip(np.vstack([start, path[:-1]]), path):
        assert segment_distance(p0, p1, disc[0, :2]) >= 0.09 - 1e-9
    # a start already inside the clearance zone still gets a finite route out
    inside = plan_route(np.array([0.43, 0.5]), goal, disc, clearance=0.03)
    assert np.isfinite(inside).all() and np.allclose(inside[-1], goal)
    # hysteresis: the hint keeps the side on a symmetric disc
    first = path[0]
    again = plan_route(start, goal, disc, clearance=0.03, hint=first * [1, -1] + [0, 1.0])  # the mirror image hint
    assert np.sign(again[0, 1] - 0.5) == -np.sign(first[1] - 0.5)
    # a wall with a gap: the route crosses the wall line inside the gap
    ys = np.arange(0.25, 0.751, 0.06)
    wall = np.array([[0.5, y, 0.03] for y in ys if not 0.56 < y < 0.72])
    pts = np.vstack([start, plan_route(start, goal, wall, clearance=0.03)])
    i = next(i for i in range(len(pts) - 1) if (pts[i, 0] - 0.5) * (pts[i + 1, 0] - 0.5) <= 0)
    y_cross = pts[i, 1] + (0.5 - pts[i, 0]) / (pts[i + 1, 0] - pts[i, 0]) * (pts[i + 1, 1] - pts[i, 1])
    assert 0.56 < y_cross < 0.72


@pytest.mark.parametrize("spec", [GRID_DISC, GRID_WALL, RING_START, dict(RING_GOAL, decoy=True)])  # one per obstacle kind
def test_obstacle_demos_avoid_the_obstacles_and_end_on_the_goal(spec):
    task = TaskConfig(**STUDY, obstacle=obstacle_spec(spec))
    rng = np.random.default_rng(3)
    layouts = sample_layouts(rng, 30, 4, task)
    actions, lengths = make_demos(rng, layouts, task)
    env = replay(actions, layouts, task)
    assert (lengths > 0).all() and env.success.all() and not env.collided.any() and (env.steps == lengths).all()
    np.testing.assert_allclose(layouts.start + actions.sum(1), layouts.goal_pos, atol=1e-9)
    assert np.linalg.norm(actions, axis=-1).max() / task.dt < 2.5  # no faster than the policy's own demos
    obs, chunk = DemoSampler(layouts, actions, lengths, 16, 4, "cpu").sample(5)  # the sampler carries the discs
    assert obs["obstacles"].shape == (5, layouts.obstacles.shape[1], 3) and obs["obstacles"].dtype == torch.float32
    assert obs["obstacle_mask"].dtype == torch.bool and chunk.shape == (5, 16, 2)
    if spec is not DISC:
        return
    # without obstacles a demo is the plain reach, drawing the same random numbers as before
    plain = sample_layouts(np.random.default_rng(6), 1, 4, TaskConfig(**STUDY))
    actions, _ = make_demos(np.random.default_rng(9), plain, TaskConfig(**STUDY))
    reach = plan_reach(np.random.default_rng(9), plain.start[0], plain.goal_pos[0], TaskConfig(**STUDY))
    np.testing.assert_allclose(actions[0, :len(reach) - 1], np.diff(reach, axis=0))
    # a layout no demonstration can reach is an error, or with allow_failed a zero-length demo the sampler never draws
    ring = [[0.5 + 0.11 * np.cos(a), 0.5 + 0.11 * np.sin(a), 0.045] for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    layouts = Layouts.concat([_one((0.2, 0.5), (0.5, 0.5), ring), _one((0.2, 0.2), (0.6, 0.2), [[0.4, 0.3, 0.03]])])
    with pytest.raises(RuntimeError, match="no successful demo"):
        make_demos(np.random.default_rng(0), layouts, TaskConfig(), max_tries=3)
    actions, lengths = make_demos(np.random.default_rng(0), layouts, TaskConfig(), max_tries=3, allow_failed=True)
    assert lengths[0] == 0 and lengths[1] > 0 and not actions[0].any()
    obs, _ = DemoSampler(layouts, actions, lengths, 16, 4, "cpu").sample(50)
    assert (obs["obstacle_mask"].sum(1) == 1).all()  # only the layout with a demonstration is ever drawn


def test_the_expert_solves_every_obstacle_task_without_touching_anything():
    arm = config.load_arm("base.yaml", {"mode": "teleop", "targets": [4, 4], "reach": [0.35, 0.5], "time_limit_s": 8.0})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    for spec in (GRID_DISC, GRID_WALL, RING_GOAL, RING_START):
        arm["obstacle"] = obstacle_spec(spec)
        layouts = sample_layouts(np.random.default_rng(4), 40, 4, arm_task(arm))
        recs = run_trials(arm, layouts, ExpertAgent(), Readout(None, 0.0), policies[arm["model"]], adapters, "cpu")
        assert [r["outcome"] for r in recs].count("hit") == 40, spec
        if spec is GRID_DISC:  # it passes the disc on the side its first route takes, away from the offset
            for b, r in enumerate(recs):
                path, start, goal, disc = np.array(r["path"]), layouts.start[b], layouts.goal_pos[b], layouts.obstacles[b, 0]
                route = plan_route(start, goal, disc[None], 0.05, avoid=layouts.distractors(b, 0.05))
                u = (goal - start) / np.linalg.norm(goal - start)
                near = np.abs((path - start) @ u - (disc[:2] - start) @ u) < 0.02
                lateral = _lateral(path[near], start, goal) - _lateral(disc[None, :2], start, goal)
                assert (np.sign(lateral) == (1 if route_side(route, start, goal) == "left" else -1)).all()
    # without obstacles the command is the closed form it always was
    plain = sample_layouts(np.random.default_rng(6), 5, (2, 4), TaskConfig())
    agent = ExpertAgent()
    agent.reset(5)
    to_goal = plain.goal_pos - plain.start
    dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
    np.testing.assert_allclose(agent.command(plain.start, plain), to_goal / dist * np.minimum(1.0, dist / 0.15))
    # no side flipping on a symmetric disc: the teleop path stays on one side of the reach line
    recs = run_trials(arm, _one((0.2, 0.5), (0.8, 0.5), [[0.5, 0.5, 0.06]]), ExpertAgent(), Readout(None, 0.0),
                      policies[arm["model"]], adapters, "cpu")
    lateral = np.array(recs[0]["path"])[:, 1] - 0.5
    signs = np.sign(lateral[np.abs(lateral) > 2e-3])
    assert recs[0]["outcome"] == "hit" and len(signs) and (signs == signs[0]).all()


def test_obstacle_arm_parameter_reaches_the_env_the_layouts_and_the_records():
    arm = config.load_arm("base.yaml", {"mode": "teleop", "obstacle": {"kind": "disc", "radius": 0.04, "along": 0.5},
                                        "targets": [4, 4], "reach": [0.35, 0.5]})
    assert arm["obstacle"] == {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.0, "side": "random",
                               "decoy": False}
    assert arm_task(arm).obstacle == arm["obstacle"] and arm_task(config.load_arm("base.yaml")).obstacle is None
    assert arm_fields(arm)["obstacle"] == arm["obstacle"]
    with pytest.raises(ValueError, match="task.obstacle"):
        config.load_arm("base.yaml", {"obstacle": {"kind": "cube"}})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts = sample_layouts(np.random.default_rng(0), 2, tuple(arm["targets"]), arm_task(arm))
    recs = run_trials(arm, layouts, ExpertAgent(), Readout(None, 0.0), policies[arm["model"]], adapters, "cpu")
    assert len(recs[0]["obstacles"]) == 1 and len(recs[0]["obstacles"][0]) == 3
    np.testing.assert_allclose(recs[0]["obstacles"], layouts.real_obstacles(0), atol=1e-4)


def test_decoy_any_places_the_obstacle_about_a_target_drawn_uniformly_the_goal_among_them():
    spec = obstacle_spec({"kind": "disc", "radius": 0.05, "along": 0.5, "decoy": "any"})
    assert spec["decoy"] == "any"
    with pytest.raises(ValueError, match="decoy must be true, false or any"):
        obstacle_spec(dict(DISC, decoy="all"))
    lay = sample_layouts(np.random.default_rng(0), 400, 4, TaskConfig(**STUDY, obstacle=spec))
    about_goal = 0
    for b in range(400):
        start, disc = lay.start[b], lay.real_obstacles(b)[0, :2]
        on = [np.allclose(disc, start + 0.5 * (t - start), atol=1e-6) for t in lay.targets[b][lay.mask[b]]]
        assert sum(on) == 1  # at `along` of the reach to exactly one target
        about_goal += on[lay.goal[b]]
        assert np.linalg.norm(lay.goal_pos[b] - start) >= 0.35 - 1e-9  # the reach rule is still about the goal
    assert 70 <= about_goal <= 130  # one target in four: 100 of 400, within three standard deviations (8.7)


def test_a_decoy_obstacle_sits_on_the_reach_to_a_non_goal_target():
    spec = obstacle_spec({"kind": "disc", "radius": 0.05, "along": 0.5, "decoy": True})
    assert spec["decoy"] is True and obstacle_spec(DISC)["decoy"] is False and obstacle_spec(WALL)["decoy"] is False
    with pytest.raises(ValueError, match="decoy"):
        obstacle_spec(dict(DISC, decoy="yes"))
    with pytest.raises(ValueError, match="decoy needs at least two targets"):
        sample_layouts(np.random.default_rng(0), 2, 1, TaskConfig(**STUDY, obstacle=obstacle_spec(dict(DISC, decoy=True))))
    lay = sample_layouts(np.random.default_rng(0), 40, 4, TaskConfig(**STUDY, obstacle=spec))
    for b in range(40):
        start, goal, disc = lay.start[b], lay.goal_pos[b], lay.real_obstacles(b)[0, :2]
        others = lay.targets[b][lay.mask[b] & ~lay.goal_mask()[b]]
        # the disc centre is at `along` of the reach to exactly one non-goal target (offset 0), never to the goal
        on = [np.allclose(disc, start + 0.5 * (t - start), atol=1e-6) for t in others]
        assert sum(on) == 1 and not np.allclose(disc, start + 0.5 * (goal - start), atol=1e-6)
        assert np.linalg.norm(goal - start) >= 0.35 - 1e-9  # the reach rule is still about the goal
    ring = sample_layouts(np.random.default_rng(1), 20, 4, TaskConfig(**STUDY, obstacle=obstacle_spec(
        dict(RING_GOAL, decoy=True))))
    for b in range(20):  # the ring is round a non-goal target
        d = np.linalg.norm(ring.targets[b][ring.mask[b]] - ring.real_obstacles(b)[:, :2].mean(0), axis=1)
        assert np.argmin(d) != ring.goal[b] and d.min() < 0.14
