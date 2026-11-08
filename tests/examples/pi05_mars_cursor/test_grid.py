"""The compositional adapter's grid: ring obstacles and asymmetric scenes (the side to pass is part of the scene),
the curl user on the readout, and the users x tasks sweeps (their cells, the cells trained on, per-cell adapters).
The pursuit expert and the grid's demonstrations: test_obstacles.py; the trainers: test_training.py."""

import json

import numpy as np
import pytest
import torch

from cursor_sa import config, session
from cursor_sa.agents import ExpertAgent
from cursor_sa.readout import Readout, corruption_spec
from lerobot.envs.cursor import TaskConfig, curl_steady_state, obstacle_count, obstacle_spec, place_obstacles, plan_route, route_side, sample_layouts
from cursor_sa.session import readout_for, load_checkpoints, run_trials
from lerobot.policies.steering import LearnedTaskModule, LearnedUserModule
from run_sweep import cell_groups, cell_key, load_sweep, main as sweep_main, method_arm

STUDY = dict(min_reach=0.35, max_reach=0.5)
DISC = {"kind": "disc", "radius": 0.08, "along": 0.5, "offset": [0.012, 0.024]}
RING_GOAL = {"kind": "ring", "around": "goal", "gap_angle": [15, 30]}
RING_START = {"kind": "ring", "around": "start", "gap_angle": [15, 30]}


def _lateral(points, start, goal):
    """Signed distance of points across the reach (+ left)."""
    u = (goal - start) / np.linalg.norm(goal - start)
    return (points - start) @ np.array([-u[1], u[0]])


def test_ring_specs_place_an_arc_open_away_from_the_reach():
    ring = obstacle_spec(RING_GOAL)
    assert ring == {"kind": "ring", "around": "goal", "ring_radius": 0.14, "radius": 0.02, "gap": 0.16,
                    "gap_angle": [15.0, 30.0], "side": "random", "decoy": False}
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


def test_curl_readout_turns_and_slows_the_command_and_resets_per_batch():
    matrix, curl = corruption_spec({"curl": 20, "tau": 0.1})
    assert matrix is None and curl == {"curl": 20.0, "tau": 0.1}
    assert corruption_spec({"scale": [1.0, 0.25]})[1] is None
    with pytest.raises(ValueError, match="control.corruption"):
        corruption_spec({"curl": 20})
    s = curl_steady_state(0.1, 10.0) @ np.array([1.0, 0.0])  # a 45 degree clockwise turn at tau * b = 1
    assert np.isclose(np.degrees(np.arctan2(s[1], s[0])), -45) and np.isclose(np.linalg.norm(s), 1 / np.sqrt(2))
    readout = Readout(None, 0.0, curl=curl)
    readout.reset(2)
    raw = np.array([[1.0, 0.0], [0.0, -0.5]])
    for _ in range(200):
        served = readout(raw)
    np.testing.assert_allclose(served, raw @ curl_steady_state(0.1, 20).T, atol=1e-9)   # turned 63 degrees clockwise, slowed
    readout.reset(2)
    first = readout(raw)
    assert np.linalg.norm(first, axis=1).max() < 0.25                                    # from rest it builds up
    np.testing.assert_allclose(readout(np.zeros((2, 2))), np.zeros((2, 2)))              # idle stays idle
    # the training twin serves the same commands step by step
    pushes = np.random.default_rng(0).uniform(-1, 1, (3, 30, 2))
    live = Readout(None, 0.0, curl=curl)
    live.reset(3)
    stepped = np.stack([live(pushes[:, k]) for k in range(30)], 1)
    np.testing.assert_allclose(Readout(curl=curl).core(pushes), stepped, atol=1e-12)
    with pytest.raises(ValueError, match="one command per trial"):
        live(pushes[:, :16])
    out = Readout.serve(torch.tensor([[0.3, 0.2], [0.3, 0.2]]), torch.tensor([[1.0, 0.0], [0.0, 0.0]]), torch.zeros(2))
    np.testing.assert_allclose(out.numpy(), [[0.3, 0.2], [0.0, 0.0]])                   # noise-free core where active
    arm = config.load_arm("teleop.yaml", {"corruption": {"curl": 20, "tau": 0.1}, "input_noise": 0.0})
    assert arm["corruption_matrix"] is None and arm["corruption_curl"] == curl
    with pytest.raises(ValueError, match="needs a corruption to copy"):
        config.load_arm("frs.yaml", {"corruption": {"curl": 20, "tau": 0.1}, "reversal_adapter": {"translation": "corruption"}})
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts = sample_layouts(np.random.default_rng(1), 6, 4, TaskConfig(**STUDY))
    recs = run_trials(arm, layouts, ExpertAgent(), readout_for(arm, np.random.default_rng(0)), policies[arm["model"]],
                      adapters, "cpu")
    served = np.array(recs[0]["served_commands"])
    raw = np.array(recs[0]["raw_commands"])
    late = len(raw) // 2  # mid-reach: the served command lags the push and turns it clockwise
    assert np.cross(raw[late], served[late]) < 0 and np.linalg.norm(served[late]) < np.linalg.norm(raw[late])


GRID = """name: tiny_grid
agent: expert
arm: condition_grid.yaml
methods:
  teleop: {mode: teleop}
  mars_user: {mode: frs, reversal_adapter: {user: cell}}
  mars: {mode: frs, reversal_adapter: {user: cell, task: cell}}
  per_task: {mode: frs, model: {T0: flow_goaldrop, T1: flow_goaldrop}, reversal_adapter: {user: {T0: cell, T1: {U1: user_b, U2: user_a}}}}
grid:
  users:
    U1: {corruption: {scale: [1.0, 0.25]}, adapter: user_a}
    U2: {corruption: {curl: 20, tau: 0.1}, adapter: user_b}
  tasks:
    T0: {obstacle: null, adapter: null, time_limit_s: 5.0}
    T1: {obstacle: {kind: disc, radius: 0.04, along: 0.5, offset: 0.0}, adapter: task_a}
  noises: [0.75]
  train: [[U1, T1]]
trials: {tune: 3, test: 3}
focus: mars
against: [teleop]
seed: 7
"""


def test_grid_sweep_loads_cells_and_resolves_cell_adapters(tmp_path):
    (tmp_path / "grid.yaml").write_text(GRID)
    sweep = load_sweep(tmp_path / "grid.yaml")
    assert sweep["grid"]["levels"] == ["U1xT0", "U1xT1", "U2xT0", "U2xT1"]  # user-major
    assert sweep["grid"]["users"] == ["U1", "U2"] and sweep["grid"]["tasks"] == ["T0", "T1"]
    assert cell_groups(sweep) == [(level, {}) for level in sweep["grid"]["levels"]]
    assert cell_key("U1xT1", 0.75) == "U1xT1|0.75"
    assert [c["train"] for c in sweep["grid"]["cells"].values()] == [False, True, False, False]
    composed = method_arm(sweep, "mars", "U1xT1")
    assert composed["corruption"] == {"scale": [1.0, 0.25]} and composed["obstacle"]["radius"] == 0.04
    assert composed["learned_adapter"] == {"user": "user_a", "task": "task_a"} and composed["method"] == "mars"
    base = method_arm(sweep, "mars", "U2xT0")
    assert base["obstacle"] is None and base["learned_adapter"] == {"user": "user_b", "task": None}
    assert base["corruption_curl"] == {"curl": 20.0, "tau": 0.1}
    assert method_arm(sweep, "mars_user", "U2xT1")["learned_adapter"] == {"user": "user_b", "task": None}
    assert method_arm(sweep, "teleop", "U2xT1")["learned_adapter"] is None
    # a model or slot given per task, then per user: the cell's (`cell`: the cell's own checkpoint)
    assert method_arm(sweep, "per_task", "U1xT1")["learned_adapter"] == {"user": "user_b", "task": None}
    assert method_arm(sweep, "per_task", "U2xT1")["learned_adapter"] == {"user": "user_a", "task": None}
    assert method_arm(sweep, "per_task", "U2xT0")["learned_adapter"] == {"user": "user_b", "task": None}
    assert method_arm(sweep, "per_task", "U2xT0")["model"] == "flow_goaldrop"
    for short, kind in (("{U1: user_b}", "grid.users"), ("{T0: flow_goaldrop}", "grid.tasks")):
        (tmp_path / "short.yaml").write_text(GRID.replace("{U1: user_b, U2: user_a}" if kind == "grid.users"
                                                          else "{T0: flow_goaldrop, T1: flow_goaldrop}", short))
        with pytest.raises(ValueError, match=kind):
            load_sweep(tmp_path / "short.yaml")
    # a task's own time limit, else the base arm's
    assert method_arm(sweep, "teleop", "U2xT0")["time_limit_s"] == 5.0 and method_arm(sweep, "teleop", "U2xT1")["time_limit_s"] == 8.0
    (tmp_path / "bad.yaml").write_text(GRID.replace("train: [[U1, T1]]", "train: [[U9, T1]]"))
    with pytest.raises(ValueError, match="U9"):
        load_sweep(tmp_path / "bad.yaml")


# the run: one user, two tasks (resolution is pinned on GRID above), the shortest trials the checks allow
RUN_GRID = (GRID.replace("  per_task: {mode: frs, model: {T0: flow_goaldrop, T1: flow_goaldrop}, reversal_adapter: {user: {T0: cell, T1: {U1: user_b, U2: user_a}}}}\n", "")
            .replace("    U2: {corruption: {curl: 20, tau: 0.1}, adapter: user_b}\n", "")
            .replace("adapter: task_a}", "adapter: task_a, time_limit_s: 2.0}").replace("trials: {tune: 3, test: 3}", "trials: {tune: 1, test: 3}"))


def test_tiny_grid_sweep_end_to_end(tmp_path, monkeypatch):
    for name, module, kind in (("user_a", LearnedUserModule(16, n_dims=2), "user"), ("task_a", LearnedTaskModule(), "task")):
        (tmp_path / name).mkdir()
        torch.save(dict(kind=kind, config=module.config, state_dict=module.state_dict(), args={}, obstacle=None),
                   tmp_path / name / "adapter.pt")
    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / name / "adapter.pt")
    assert "per_task" not in RUN_GRID and "U2" not in RUN_GRID and "time_limit_s: 2.0" in RUN_GRID and "tune: 1," in RUN_GRID
    (tmp_path / "grid.yaml").write_text(RUN_GRID + "tune: {frs: {n_reverse_steps: [10]}}\n")
    argv = ["--sweep", str(tmp_path / "grid.yaml"), "--results", str(tmp_path), "--cpu", "--out", "grid"]
    sweep_main(argv)
    out = tmp_path / "sim" / "sweeps" / "grid"
    result = json.loads((out / "test.json").read_text())
    assert set(result["test"]) == {"U1xT0|0.75", "U1xT1|0.75"}
    cell = result["test"]["U1xT1|0.75"]
    assert cell["user"] == "U1" and cell["task"] == "T1" and cell["train_cell"] is True
    assert result["test"]["U1xT0|0.75"]["train_cell"] is False
    assert set(cell["metrics"]) == {"teleop", "mars_user", "mars"}
    assert set(cell["lead"]) == {"teleop", "mars_user"} and cell["best_other"] == "teleop"  # leads over all, best over `against`
    # every test trial is kept: its outcome per method and the side its route passes the obstacle on
    trials = {k: c["trials"] for k, c in result["test"].items()}
    assert len(trials["U1xT1|0.75"]["side"]) == 3
    assert set(trials["U1xT1|0.75"]["side"]) <= {"left", "right"} and set(trials["U1xT0|0.75"]["side"]) == {"straight"}
    assert trials["U1xT1|0.75"]["outcome"]["teleop"][0] in ("hit", "wrong_target", "collision", "timeout")
    by_side = cell["by_side"]
    assert sum(b["n"] for b in by_side.values()) == 3 and "by_side" not in result["test"]["U1xT0|0.75"]
    for side, b in by_side.items():
        hits = sum(o == "hit" for o, d in zip(trials["U1xT1|0.75"]["outcome"]["mars"], trials["U1xT1|0.75"]["side"]) if d == side)
        assert b["metrics"]["mars"]["hits"] == hits
    ranking = (out / "ranking.csv").read_text().splitlines()
    assert ranking[0].startswith("cell,level,noise,user,task,train_cell,") and "mars_collision" in ranking[0]
    assert (out / "figs" / "grid.png").exists() and (out / "grid_summary.md").exists()
    assert (out / "figs" / "summary.png").exists() and not (out / "figs" / "showcase.png").exists()  # every cell, not one
    assert "tiny_grid: test trials" in (out / "log.txt").read_text()  # the sweep's own log, in its folder
    summary = (out / "grid_summary.md").read_text()
    assert "held-out" in summary and "U1" in summary and "T1" in summary
    sweep_main(argv + ["--resume"])  # nothing left to run; the outputs are rewritten from the saved cells
    assert len(json.loads((out / "test.json").read_text())["test"]) == 2
    # run in two parts and merged, every cell comes out exactly as in the one run: its seeds are its own
    sweep_main(argv[:-2] + ["--out", "part_a", "--cells", "U1xT0"])
    sweep_main(argv[:-2] + ["--out", "part_b", "--cells", "U1xT1"])
    sweep_main(argv[:-2] + ["--out", "merged", "--merge", "part_a", "part_b"])
    merged = tmp_path / "sim" / "sweeps" / "merged"
    assert json.loads((merged / "test.json").read_text())["test"] == result["test"]
    assert (merged / "figs" / "grid.png").exists() and (merged / "ranking.csv").read_text() == (out / "ranking.csv").read_text()
    log = (merged / "log.txt").read_text()
    assert log.startswith("=== part_a") and "=== part_b" in log and log.index("=== part_b") < log.index("merging 2 parts")
    with pytest.raises(ValueError, match="not levels"):
        sweep_main(argv[:-2] + ["--out", "bad", "--cells", "U9xT0"])
