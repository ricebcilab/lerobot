"""Sweeps over a users x tasks grid: the cells, the cells trained on, per-cell adapters."""

import json

import pytest
import torch

from cursor_sa import session
from cursor_sa.steering import TaskModule, UserModule
from run_sweep import cell_groups, cell_key, load_sweep, main as sweep_main, method_arm

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


def test_tiny_grid_sweep_end_to_end(tmp_path, monkeypatch):
    for name, module, kind in (("user_a", UserModule(), "user"), ("user_b", UserModule(), "user"),
                               ("task_a", TaskModule(), "task")):
        (tmp_path / name).mkdir()
        torch.save(dict(kind=kind, config=module.config, state_dict=module.state_dict(), args={}, obstacle=None),
                   tmp_path / name / "adapter.pt")
    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / name / "adapter.pt")
    (tmp_path / "grid.yaml").write_text(GRID + "tune: {frs: {n_reverse_steps: [2, 10]}}\n")
    argv = ["--sweep", str(tmp_path / "grid.yaml"), "--results", str(tmp_path), "--cpu", "--out", "grid"]
    sweep_main(argv)
    out = tmp_path / "sim" / "sweeps" / "grid"
    result = json.loads((out / "test.json").read_text())
    assert set(result["test"]) == {"U1xT0|0.75", "U1xT1|0.75", "U2xT0|0.75", "U2xT1|0.75"}
    cell = result["test"]["U1xT1|0.75"]
    assert cell["user"] == "U1" and cell["task"] == "T1" and cell["train_cell"] is True
    assert result["test"]["U1xT0|0.75"]["train_cell"] is False
    assert set(cell["metrics"]) == {"teleop", "mars_user", "mars", "per_task"}
    assert set(cell["lead"]) == {"teleop", "mars_user", "per_task"} and cell["best_other"] == "teleop"  # leads over all, best over `against`
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
    assert len(json.loads((out / "test.json").read_text())["test"]) == 4
    # run in two parts and merged, every cell comes out exactly as in the one run: its seeds are its own
    sweep_main(argv[:-2] + ["--out", "part_a", "--cells", "U1xT0", "U2xT1"])
    sweep_main(argv[:-2] + ["--out", "part_b", "--cells", "U2xT0", "U1xT1"])
    sweep_main(argv[:-2] + ["--out", "merged", "--merge", "part_a", "part_b"])
    merged = tmp_path / "sim" / "sweeps" / "merged"
    assert json.loads((merged / "test.json").read_text())["test"] == result["test"]
    assert (merged / "figs" / "grid.png").exists() and (merged / "ranking.csv").read_text() == (out / "ranking.csv").read_text()
    log = (merged / "log.txt").read_text()
    assert log.startswith("=== part_a") and "=== part_b" in log and log.index("=== part_b") < log.index("merging 2 parts")
    with pytest.raises(ValueError, match="not levels"):
        sweep_main(argv[:-2] + ["--out", "bad", "--cells", "U9xT0"])
