"""The paper figures' builder (paper/figures/build_figures.py)."""

import json

import numpy as np
import pytest

from cursor_sa.paths import RESULTS
from paper.figures import build_figures as bf


def test_fig1_builds_from_its_pictograms_and_reaches(tmp_path):
    for _, name, kind in bf.FIG1_INPUTS:
        assert (bf.ASSETS / f"{name}.svg").exists() and (bf.ASSETS / f"{name}.png").exists(), name
        path = bf.served_path(kind)
        assert path.ndim == 2 and path.shape[1] == 2 and np.isfinite(path).all(), kind
        assert np.allclose(path[0], bf.START) or kind == "misjudge", kind   # misjudge's path starts at its turn
        assert ((path >= 0) & (path <= [bf.TILE_W, bf.TILE_H])).all(), kind   # inside the tile
    bf.main(["--only", "fig1", "--out", str(tmp_path), "--device", "cpu"])
    assert (tmp_path / "fig1_method.pdf").exists() and (tmp_path / "fig1_method.png").exists()
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert set(stats) == {"fig1"} and stats["fig1"]["inputs"] == [label for label, _, _ in bf.FIG1_INPUTS]


def _trial(participant, run, arm, outcome, duration):
    return dict(participant=participant, run=run, arm=arm, outcome=outcome, duration_s=duration, layout_id=duration)


def _seed_runs(per_seed, cell="U1xT2"):
    """A grid's data (grid_data's shape) for one cell from each seed's {method: trial outcomes}."""
    runs = [{"test": {f"{cell}|1.0": {"metrics": {m: {"success": float(np.mean([o == "hit" for o in v]))} for m, v in seed.items()},
                                      "trials": {"outcome": seed}}}} for seed in per_seed]
    return dict(runs=runs, users=[cell.split("x")[0]], tasks=["T1", cell.split("x")[1]], methods=list(per_seed[0]), leads={},
                test_n=len(next(iter(per_seed[0].values()))), cell=lambda u, t, k=0: runs[k]["test"][f"{u}x{t}|1.0"])


def test_the_statistics_pool_participants_and_pair_trials_within_a_seed():
    trials = [_trial("p1", "study_s1", "teleop", "hit", 1.0), _trial("p1", "study_s1", "teleop", "timeout", 5.0),
              _trial("p1", "study_s2", "teleop", "hit", 2.0), _trial("p1", "study_s2", "teleop", "hit", 3.0),
              _trial("p2", "study_s1", "teleop", "wrong_target", 2.0), _trial("p2", "study_s1", "teleop", "hit", 4.0)]
    stats = bf.participant_stats(trials, ["teleop"])
    assert list(stats) == ["p1", "p2"] and stats["p1"]["sessions"] == ["study_s1", "study_s2"]
    assert stats["p1"]["teleop"]["n"] == 4 and stats["p1"]["teleop"]["hit"] == 0.75 and stats["p1"]["teleop"]["time_s"] == 2.0
    assert stats["p2"]["teleop"]["wrong_target"] == 0.5 and stats["p2"]["teleop"]["timeout"] == 0.0
    m, lo, hi = bf.t_ci([0.75, 0.5])
    assert m == 0.625 and lo < m < hi
    D = _seed_runs([{"mars": ["hit"] * 6 + ["timeout"] * 4, "teleop": ["hit"] * 3 + ["collision"] * 7},
                    {"mars": ["hit"] * 8 + ["collision"] * 2, "teleop": ["hit"] * 4 + ["timeout"] * 6}])
    assert bf.seed_values(D, "U1", "T2", "mars") == pytest.approx([60, 80]) and bf.seed_mean(D, "U1", "T2", "teleop") == pytest.approx(35)
    mean, lo, hi = bf.lead_ci(D, "U1", "T2", "mars", "teleop", n_boot=400)
    assert mean == pytest.approx(35) and lo <= mean <= hi
    assert bf.lead_ci(D, "U1", "T2", "mars", "mars", n_boot=100) == [0, 0, 0]


def test_drawing_helpers_order_legends_bearings_and_rich_text():
    h, l = bf.row_major(list("ABCDEFG"), list("abcdefg"), 4)
    assert h == ["A", "E", "B", "F", "C", "G", "D"] and l == ["a", "e", "b", "f", "c", "g", "d"]
    assert [bf.bearing(v, [1, 0]) for v in ([0, 1], [0, -1], [-1, 0], [1, 0])] == pytest.approx([90, -90, 180, 0])
    bf.paper_style()
    fig = bf.plt.figure(figsize=(4, 2))
    texts = bf.rich_text(fig, 2.0, 1.0, [("The BCI ", {}), ("$A_U$", dict(color="red")), (" steers", dict(fontweight="bold"))], 7)
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in texts]
    assert all(abs(a.x1 - b.x0) < 0.5 for a, b in zip(boxes, boxes[1:]))   # side by side
    assert abs((boxes[0].x0 + boxes[-1].x1) / 2 - 2.0 * fig.dpi) < 0.5     # centred
    bf.plt.close(fig)


@pytest.mark.parametrize("spec", [bf.fig1_schematic, bf.fig3_schematic])
def test_schematic_draws_three_density_panels_and_its_flows(spec):
    bf.paper_style()
    fig = bf.plt.figure(figsize=(bf.FULL_W, 2.62))
    bf.schematic(fig, 0.05, 0.2, spec())
    assert len(fig.axes) == 3
    texts = [t.get_text() for t in fig.texts]
    assert "flow reversal" in texts and "flow matching" in texts and bf.MARS in texts
    assert sum("\\int_0^1" in t for t in texts) == 2
    for ax in fig.axes:   # every panel's plot sits inside the figure
        x0, y0, x1, y1 = ax.get_position().extents
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    bf.plt.close(fig)


@pytest.mark.skipif(not any((RESULTS / bf.RECORDED_STUDY / f"p{i}").exists() for i in range(1, 10)),
                    reason="no recorded study sessions")
def test_fig2_builds_from_the_recorded_sessions(tmp_path):
    bf.main(["--only", "fig2", "--out", str(tmp_path)])
    stats = json.loads((tmp_path / "stats.json").read_text())["fig2"]
    assert stats["participants"] >= 1 and set(stats["success_mean_ci"]) == set(bf.ARMS)
    assert all(len(v) == 3 for v in stats["success_mean_ci"].values())


def _grid_results(sweeps):
    """Seed 0 of a one-user grid run (configs/grid.yaml's arms: teleop, the fine-tuned baselines, MARS-T, MARS-UT and the
    fine-tuned policy with the user module), two trials a cell."""
    cells = {"U1xT1": None, "U1xT2": {"kind": "disc"}}
    methods = ["teleop", "ft_flow_control", "ft_frs", "mars_task", "mars", "ft_mars_user"]
    test = {f"{c}|1.0": dict(settings=dict.fromkeys(methods),
                             metrics={m: {"success": 0.25 if m == "ft_mars_user" else 0.5} for m in methods},
                             trials=dict(side=["left", "right"], duration_s={m: [1.0, 8.0] for m in methods},
                                         outcome={m: ["hit", "timeout"] for m in methods}))
            for c in cells}
    grid = dict(users=["U1"], tasks=["T1", "T2"], noises=[1.0], cells={c: {"obstacle": o} for c, o in cells.items()})
    (sweeps / "grid_s0").mkdir(parents=True)
    (sweeps / "grid_s0" / "test.json").write_text(json.dumps(dict(args=dict(sweep=dict(methods=dict.fromkeys(methods, {}), grid=grid),
                                                                          test_trials=2), test=test)))


def _demo_time_results(root, mars, ft, teleop=("hit", "timeout")):
    """Figure 3e's runs on two cells, one per seed (configs/grid_demo_time.yaml's arms): the vision module (mars),
    Fine-tuned + MARS-U (ft), flow control and FRS on the fine-tuned policy (ft - 0.1 and 0.1), and teleop; and the grid's
    run, whose teleop the curve's must replay."""
    cells = ["U1xT2|1.0", "U2xT3|1.0"]
    def write(name, methods, outcome=("hit", "timeout")):
        (root / name).mkdir(parents=True)
        (root / name / "test.json").write_text(json.dumps({"test": {c: {"metrics": {m: {"success": v} for m, v in methods.items()},
                                                                         "trials": {"outcome": {"teleop": list(outcome)}}}
                                                                     for c in cells}}))
    for k, (m, f) in enumerate(zip(mars, ft)):
        write(f"demo_time_s{k}", {"teleop": 0.2, **{f"mars_t{s}": m for s in bf.DEMO_SECONDS},
                                  **{f"ft_mars_user_t{s}": f for s in bf.DEMO_SECONDS},
                                  **{f"ft_flow_control_t{s}": f - 0.1 for s in bf.DEMO_SECONDS},
                                  **{f"ft_frs_t{s}": 0.1 for s in bf.DEMO_SECONDS}}, teleop)
    write("grid_s0", {"teleop": 0.2})


def test_fig3_reads_one_run_per_seed_on_the_grids_trials(tmp_path, monkeypatch):
    _grid_results(tmp_path / "cursor_sim" / "sweeps")
    D = bf.grid_data(tmp_path)
    assert D["methods"] == bf.GRID_METHODS   # the grid's arms, the fine-tuned arm among them
    assert bf.seed_values(D, "U1", "T2", "ft_mars_user") == [25.0] and "mars_user" not in D["cell"]("U1", "T2")["metrics"]
    results = tmp_path / "results"
    _demo_time_results(results / "cursor_sim" / "sweeps", mars=[0.7, 0.8], ft=[0.4, 0.5])
    D = bf.demo_time_data(results)
    assert D["seeds"] == {arm: [0, 1] for arm in bf.DEMO_ARMS} and D["curve"]["mars"][300] == pytest.approx([70, 80])
    assert D["curve"]["ft_flow_control"][300] == pytest.approx([30, 40]) and D["curve"]["ft_frs"][30] == pytest.approx([10, 10])
    _demo_time_results(tmp_path / "off" / "cursor_sim" / "sweeps", mars=[0.7], ft=[0.4], teleop=("timeout", "timeout"))
    with pytest.raises(ValueError, match="not on the grid's cells and trials"):  # a curve off the grid's trials: refused
        bf.demo_time_data(tmp_path / "off")
    monkeypatch.setattr(bf, "demo_log", lambda task, seconds, seed=0: {"args": {"n_demos": seconds}, "best_step": 5000})
    bf.paper_style()
    fig = bf.plt.figure(figsize=(bf.FULL_W, bf.FIG3_H))
    stats = bf.draw_demo_time(fig, D, (0.6, 0.4, 0.35, 0.15))
    assert [line.get_color() for line in fig.axes[0].get_lines()] == [bf.GRID_COLOR[arm] for arm in bf.DEMO_ARMS]   # MARS-UT last, on top
    assert stats["success"]["mars"]["300"][0] == pytest.approx(75) and stats["grid_demonstrations_minutes"]["T2"] == pytest.approx(5.0)
    bf.plt.close(fig)
    _demo_time_results(tmp_path / "other" / "cursor_sim" / "sweeps", mars=[0.7], ft=[0.4], teleop=("timeout", "timeout"))
    (tmp_path / "other" / "cursor_sim" / "sweeps" / "grid_s0" / "test.json").write_text(  # the grid's own teleop trials
        (results / "cursor_sim" / "sweeps" / "grid_s0" / "test.json").read_text())
    with pytest.raises(ValueError):
        bf.demo_time_data(tmp_path / "other")
