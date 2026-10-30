"""Obstacle tasks: layouts with discs, the collision outcome, the waypoint planner, obstacle-avoiding
demos and the obstacle-aware expert."""

import numpy as np
import pytest

import torch

from cursor_sa import config
from cursor_sa.agents import ExpertAgent
from cursor_sa.analysis import trial_metrics
from cursor_sa.channel import InputChannel
from cursor_sa.demos import DemoSampler, make_demos, replay
from cursor_sa.env import (CursorEnv, Layouts, TaskConfig, _layout, center_out, obstacle_spec, plan_route,
                           sample_layouts, segment_distance)
from cursor_sa.records import OUTCOMES, arm_fields, outcome_of, summarize
from cursor_sa.session import Run, arm_task, load_checkpoints, run_trials
from run_sweep import cell_groups, cell_key, load_sweep, method_arm

DISC = {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05}
WALL = {"kind": "wall", "along": 0.5, "gap": 0.16, "gap_offset": 0.14}
STUDY = dict(min_reach=0.35, max_reach=0.5)


def _one(start, goal, obstacles, extra_targets=()):
    """A single-trial layout with the goal first."""
    targets = np.array([goal, *extra_targets], float)[None]
    obstacles = np.array(obstacles, float).reshape(1, -1, 3)
    return Layouts(np.array([start], float), targets, np.ones(targets.shape[:2], bool), np.array([0]),
                   obstacles, np.ones(obstacles.shape[:2], bool))


def test_layouts_without_obstacles_are_unchanged():
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
    assert center_out().obstacles.shape == (8, 0, 3)


def test_layouts_carry_obstacles_through_indexing_tile_and_concat():
    a = _one((0.2, 0.5), (0.8, 0.5), [[0.5, 0.5, 0.04]])
    b = Layouts(np.zeros((2, 2)), np.full((2, 1, 2), 0.5), np.ones((2, 1), bool), np.zeros(2, int),
                np.ones((2, 3, 3)), np.array([[True, True, False], [True, False, False]]))
    assert a[[0]].obstacles.shape == (1, 1, 3) and a.tile(3).obstacles.shape == (3, 1, 3)
    np.testing.assert_array_equal(a.tile(3).obstacle_mask, [[True]] * 3)
    both = Layouts.concat([a, b])
    assert both.obstacles.shape == (3, 3, 3) and both.obstacle_mask.tolist() == [[True, False, False], [True, True, False],
                                                                                  [True, False, False]]
    np.testing.assert_array_equal(both.obstacles[0, 0], [0.5, 0.5, 0.04])
    assert both.targets.shape == (3, 1, 2)


def test_obstacle_specs_place_discs_and_walls_relative_to_the_reach():
    assert obstacle_spec(None) is None
    disc = obstacle_spec(DISC)
    assert disc == {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.05, "side": "random"}
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
        n = np.array([-u[1], u[0]])
        real = walls.obstacles[b, walls.obstacle_mask[b]]
        along = (real[:, :2] - start) @ u
        assert np.allclose(along, 0.5 * np.linalg.norm(goal - start))  # the discs sit on one line across the reach
        across = (real[:, :2] - start) @ n
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
    # a step that jumps across a thin disc still collides (segment test, not a point test)
    thin = _one((0.45, 0.5), (0.8, 0.5), [[0.5, 0.5, 0.01]])
    env = CursorEnv(thin, TaskConfig())
    env.step(np.array([[0.1, 0.0]]))
    assert env.collided[0] and env.done[0]
    # the same step with the disc out of the way is nothing special
    env = CursorEnv(_one((0.45, 0.5), (0.8, 0.5), [[0.5, 0.6, 0.01]]), TaskConfig())
    env.step(np.array([[0.1, 0.0]]))
    assert not env.collided[0] and not env.done[0]
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
    route = plan_route(start, goal, wall, clearance=0.03)
    pts = np.vstack([start, route])
    crossing = next(i for i in range(len(pts) - 1) if (pts[i, 0] - 0.5) * (pts[i + 1, 0] - 0.5) <= 0)
    y_cross = pts[crossing, 1] + (0.5 - pts[crossing, 0]) / (pts[crossing + 1, 0] - pts[crossing, 0]) * (pts[crossing + 1, 1] - pts[crossing, 1])
    assert 0.56 < y_cross < 0.72


@pytest.mark.parametrize("spec", [DISC, dict(DISC, radius=0.08, offset=0.0), WALL, {"kind": "disc", "radius": 0.05, "along": 0.7}])
def test_obstacle_demos_avoid_the_obstacles_and_end_on_the_goal(spec):
    task = TaskConfig(**STUDY, obstacle=obstacle_spec(spec))
    rng = np.random.default_rng(3)
    layouts = sample_layouts(rng, 60, 4, task)
    actions, lengths = make_demos(rng, layouts, task)
    env = replay(actions, layouts, task)
    assert env.success.all() and not env.collided.any() and (env.steps == lengths).all()
    end = layouts.start + actions.sum(1)
    np.testing.assert_allclose(end, layouts.goal_pos, atol=1e-9)
    assert np.linalg.norm(actions, axis=-1).max() < 0.1


def test_sampler_carries_obstacles():
    task = TaskConfig(**STUDY, obstacle=obstacle_spec(WALL))
    rng = np.random.default_rng(4)
    layouts = sample_layouts(rng, 8, 4, task)
    sampler = DemoSampler(layouts, *make_demos(rng, layouts, task), 16, 4, "cpu")
    obs, chunk = sampler.sample(5)
    assert obs["obstacles"].shape == (5, layouts.obstacles.shape[1], 3) and obs["obstacle_mask"].shape == (5, layouts.obstacles.shape[1])
    assert obs["obstacles"].dtype == torch.float32 and obs["obstacle_mask"].dtype == torch.bool and chunk.shape == (5, 16, 2)


def test_expert_detours_and_is_unchanged_without_obstacles():
    arm = config.load_arm("teleop.yaml")
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    rng = np.random.default_rng(6)
    for spec in (WALL, dict(DISC, radius=0.08, offset=0.0)):
        task = TaskConfig(**STUDY, obstacle=obstacle_spec(spec))
        layouts = sample_layouts(rng, 100, 4, task)
        recs = run_trials(arm, layouts, ExpertAgent(), InputChannel(None, 0.0), policies[arm["model"]], adapters, "cpu")
        outcomes = [r["outcome"] for r in recs]
        assert outcomes.count("collision") == 0 and outcomes.count("hit") >= 95, {o: outcomes.count(o) for o in set(outcomes)}
    # without obstacles the command is the closed form it always was
    plain = sample_layouts(rng, 5, (2, 4), TaskConfig())
    agent = ExpertAgent()
    agent.reset(5)
    to_goal = plain.goal_pos - plain.start
    dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
    np.testing.assert_allclose(agent.command(plain.start, plain), to_goal / dist * np.minimum(1.0, dist / 0.15))
    # no side flipping on a symmetric disc: the teleop path stays on one side of the reach line
    layouts = _one((0.2, 0.5), (0.8, 0.5), [[0.5, 0.5, 0.06]])
    recs = run_trials(arm, layouts, ExpertAgent(), InputChannel(None, 0.0), policies[arm["model"]], adapters, "cpu")
    lateral = np.array(recs[0]["path"])[:, 1] - 0.5
    signs = np.sign(lateral[np.abs(lateral) > 2e-3])
    assert recs[0]["outcome"] == "hit" and len(signs) and (signs == signs[0]).all()


def test_obstacle_arm_parameter_reaches_the_env_the_layouts_and_the_records(tmp_path):
    from run_experiment import parse_args
    arm = config.load_arm("teleop.yaml", {"obstacle": {"kind": "disc", "radius": 0.04, "along": 0.5}, "targets": [4, 4],
                                          "reach": [0.35, 0.5]})
    assert arm["obstacle"] == {"kind": "disc", "radius": 0.04, "along": 0.5, "offset": 0.0, "side": "random"}
    assert arm_task(arm).obstacle == arm["obstacle"] and arm_task(config.load_arm("base.yaml")).obstacle is None
    assert arm_fields(arm)["obstacle"] == arm["obstacle"]
    with pytest.raises(ValueError, match="task.obstacle"):
        config.load_arm("base.yaml", {"obstacle": {"kind": "cube"}})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts = sample_layouts(np.random.default_rng(0), 2, tuple(arm["targets"]), arm_task(arm))
    recs = run_trials(arm, layouts, ExpertAgent(), InputChannel(None, 0.0), policies[arm["model"]], adapters, "cpu")
    assert len(recs[0]["obstacles"]) == 1 and len(recs[0]["obstacles"][0]) == 3
    np.testing.assert_allclose(recs[0]["obstacles"], layouts.real_obstacles(0), atol=1e-4)
    # every arm of a protocol must play the same obstacles
    base = config.CONFIG_DIR / "teleop.yaml"
    (tmp_path / "with.yaml").write_text(f"extends: {base}\ntask: {{obstacle: {{kind: disc, radius: 0.04, along: 0.5}}}}\n")
    (tmp_path / "without.yaml").write_text(f"extends: {base}\n")
    protocol = dict(name="tiny", arms={"a": str(tmp_path / "with.yaml"), "b": str(tmp_path / "without.yaml")},
                    blocks={"trials": 2, "practice": 1, "order": "listed"}, trial={"ready_hold_s": 0.0, "feedback_s": 0.0},
                    expert={"lag": 0, "slow_radius": 0.15}, display={"blind": True})
    with pytest.raises(ValueError, match="same task layouts"):
        Run(protocol, "p1", 1, ExpertAgent(), "cpu", tmp_path)
    # free play takes the spec on the command line
    args = parse_args(["--arm", "teleop.yaml", "--obstacle", "{kind: wall, along: 0.5, gap: 0.16, gap_offset: 0.14}"])
    assert args.overrides["obstacle"]["kind"] == "wall"


def test_sweep_axes_accept_named_values(tmp_path):
    (tmp_path / "named.yaml").write_text(
        "name: named\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\n"
        "grid:\n  family: none\n  noises: [0.5]\n"
        "  axes: {obstacle: {T0: null, T1: {kind: disc, radius: 0.04, along: 0.5}}, targets: [[4, 4]]}\n")
    sweep = load_sweep(tmp_path / "named.yaml")
    groups = cell_groups(sweep)
    assert groups == [(None, {"obstacle": "T0", "targets": [4, 4]}), (None, {"obstacle": "T1", "targets": [4, 4]})]
    assert cell_key(None, 0.5, groups[1][1]) == "-|0.5|obstacle=T1|targets=4-4"
    assert method_arm(sweep, "frs", None, groups[1][1])["obstacle"]["kind"] == "disc"
    assert method_arm(sweep, "teleop", None, groups[0][1])["obstacle"] is None


def test_demos_can_drop_layouts_that_have_no_route():
    ring = [[0.5 + 0.11 * np.cos(a), 0.5 + 0.11 * np.sin(a), 0.045] for a in np.linspace(0, 2 * np.pi, 10, endpoint=False)]
    boxed = _one((0.2, 0.5), (0.5, 0.5), ring)                   # the goal is walled in: no demonstration can reach it
    free = _one((0.2, 0.2), (0.6, 0.2), [[0.4, 0.3, 0.03]])
    layouts = Layouts.concat([boxed, free])
    with pytest.raises(RuntimeError, match="no successful demo"):
        make_demos(np.random.default_rng(0), layouts, TaskConfig(), max_tries=3)
    actions, lengths = make_demos(np.random.default_rng(0), layouts, TaskConfig(), max_tries=3, allow_failed=True)
    assert lengths[0] == 0 and lengths[1] > 0 and not actions[0].any()
    sampler = DemoSampler(layouts, actions, lengths, 16, 4, "cpu")
    obs, _ = sampler.sample(50)
    assert (obs["obstacle_mask"].sum(1) == 1).all()  # only the layout with a demonstration is ever drawn


def test_layouts_redraw_the_targets_when_no_start_fits_them():
    # a reach of 0.95 fits only goals near a corner: most target sets admit no start, and are redrawn
    cfg = TaskConfig(min_reach=0.95)
    layouts = sample_layouts(np.random.default_rng(0), 20, 2, cfg)
    assert (np.linalg.norm(layouts.goal_pos - layouts.start, axis=1) >= 0.95).all()
    with pytest.raises(ValueError, match="cannot place a start"):
        _layout(np.random.default_rng(0), 2, 0, cfg, max_tries=20, rounds=1)
    assert _layout(np.random.default_rng(0), 2, 0, cfg, max_tries=20, rounds=500)[0].shape == (2,)
