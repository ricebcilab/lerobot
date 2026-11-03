"""The paper figures' builder (paper/figures/build_figures.py)."""

import json

import numpy as np
import pytest
from PIL import Image

from cursor_sa.paths import RESULTS
from paper.figures import build_figures as bf


def test_fig1_inputs_have_their_pictograms_and_reaches():
    for _, name, kind in bf.FIG1_INPUTS:
        assert (bf.ASSETS / f"{name}.svg").exists() and (bf.ASSETS / f"{name}.png").exists(), name
        path = bf.served_path(kind)
        assert path.ndim == 2 and path.shape[1] == 2 and np.isfinite(path).all(), kind
        assert np.allclose(path[0], bf.START) or kind == "misjudge", kind   # misjudge's path starts at its turn
        assert ((path >= 0) & (path <= [bf.TILE_W, bf.TILE_H])).all(), kind   # inside the tile


def _trial(participant, run, arm, outcome, duration):
    return dict(participant=participant, run=run, arm=arm, outcome=outcome, duration_s=duration, layout_id=duration)


def test_participant_stats_pool_a_participants_sessions_once():
    trials = [_trial("p1", "study_s1", "teleop", "hit", 1.0), _trial("p1", "study_s1", "teleop", "timeout", 5.0),
              _trial("p1", "study_s2", "teleop", "hit", 2.0), _trial("p1", "study_s2", "teleop", "hit", 3.0),
              _trial("p2", "study_s1", "teleop", "wrong_target", 2.0), _trial("p2", "study_s1", "teleop", "hit", 4.0)]
    stats = bf.participant_stats(trials, ["teleop"])
    assert list(stats) == ["p1", "p2"] and stats["p1"]["sessions"] == ["study_s1", "study_s2"]
    assert stats["p1"]["teleop"]["n"] == 4 and stats["p1"]["teleop"]["hit"] == 0.75 and stats["p1"]["teleop"]["time_s"] == 2.0
    assert stats["p2"]["teleop"]["wrong_target"] == 0.5 and stats["p2"]["teleop"]["timeout"] == 0.0
    m, lo, hi = bf.t_ci([0.75, 0.5])
    assert m == 0.625 and lo < m < hi


def test_row_major_legend_order_reads_left_to_right():
    h, l = bf.row_major(list("ABCDEFG"), list("abcdefg"), 4)
    assert h == ["A", "E", "B", "F", "C", "G", "D"] and l == ["a", "e", "b", "f", "c", "g", "d"]


def test_bearing_is_signed_and_wraps():
    assert bf.bearing([0, 1], [1, 0]) == pytest.approx(90)
    assert bf.bearing([0, -1], [1, 0]) == pytest.approx(-90)
    assert bf.bearing([-1, 0], [1, 0]) == pytest.approx(180)
    assert bf.bearing([1, 0], [1, 0]) == 0


def test_rich_text_sets_its_pieces_side_by_side_centred():
    bf.paper_style()
    fig = bf.plt.figure(figsize=(4, 2))
    texts = bf.rich_text(fig, 2.0, 1.0, [("The BCI ", {}), ("$A_U$", dict(color="red")), (" steers", dict(fontweight="bold"))], 7)
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in texts]
    assert all(abs(a.x1 - b.x0) < 0.5 for a, b in zip(boxes, boxes[1:]))
    assert abs((boxes[0].x0 + boxes[-1].x1) / 2 - 2.0 * fig.dpi) < 0.5
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


def test_fig1_builds_pdf_png_and_stats(tmp_path):
    bf.main(["--only", "fig1", "--out", str(tmp_path), "--device", "cpu"])
    assert (tmp_path / "fig1_method.pdf").exists() and (tmp_path / "fig1_method.png").exists()
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert set(stats) == {"fig1"} and stats["fig1"]["inputs"] == [label for label, _, _ in bf.FIG1_INPUTS]


@pytest.mark.skipif(not any((RESULTS / f"p{i}").exists() for i in range(1, 10)), reason="no recorded study sessions")
def test_fig2_builds_from_the_recorded_sessions(tmp_path):
    bf.main(["--only", "fig2", "--out", str(tmp_path)])
    stats = json.loads((tmp_path / "stats.json").read_text())["fig2"]
    assert stats["participants"] >= 1 and set(stats["success_mean_ci"]) == set(bf.ARMS)
    assert all(len(v) == 3 for v in stats["success_mean_ci"].values())


@pytest.mark.skipif(not all((RESULTS / "sim" / "sweeps" / f"{name}_s0" / "test.json").exists()
                            for name in ("grid", "grid_ft", "demo_time_mars", "demo_time_ft")),
                    reason="no grid or demonstration-time results")
def test_fig3_builds_from_the_grid(tmp_path):
    bf.main(["--only", "fig3", "--out", str(tmp_path), "--device", "cpu"])
    stats = json.loads((tmp_path / "stats.json").read_text())["fig3"]
    cells = {f"U{u}xT{t}" for u in range(1, 5) for t in range(1, 6)}
    assert set(stats["success_per_cell"]) == cells and stats["seeds"][0] == 0
    assert all(set(v) == set(bf.GRID_METHODS) for v in stats["success_per_cell"].values())   # the fine-tuned arm, not the retrained
    assert all(len(v) == len(stats["seeds"]) for c in stats["success_per_cell_per_seed"].values() for v in c.values())
    curve = stats["demo_time"]   # panel e: both arms at every budget, on the 16 obstacle cells
    assert curve["cells"] == 16 and all(set(curve["success"][arm]) == {str(s) for s in bf.DEMO_SECONDS} for arm in bf.DEMO_ARMS)
    assert stats["composed_beats_all_baselines"]["of"] == 16 and stats["composed_beats_fine_tuned"]["of"] == 16
    assert set(stats["task_rollouts"]) == {f"T{t}" for t in range(1, 6)}
    assert Image.open(tmp_path / "fig3_grid.png").size == (round(300 * bf.FULL_W), round(300 * bf.FIG3_H))


def _seed_runs(per_seed, cell="U1xT2"):
    """A grid's data (grid_data's shape) for one cell from each seed's {method: trial outcomes}."""
    runs = [{"test": {f"{cell}|1.0": {"metrics": {m: {"success": float(np.mean([o == "hit" for o in v]))} for m, v in seed.items()},
                                      "trials": {"outcome": seed}}}} for seed in per_seed]
    return dict(runs=runs, users=[cell.split("x")[0]], tasks=["T1", cell.split("x")[1]], methods=list(per_seed[0]), leads={},
                test_n=len(next(iter(per_seed[0].values()))), cell=lambda u, t, k=0: runs[k]["test"][f"{u}x{t}|1.0"])


def _grid_results(sweeps, ft_teleop=("hit", "timeout")):
    """Seed 0 of a one-user grid (teleop, the retrained arm, MARS-UT) and its fine-tuned arm, two trials a cell."""
    cells = {"U1xT1": None, "U1xT2": {"kind": "disc"}}
    def write(name, methods, teleop):
        test = {f"{c}|1.0": dict(settings=dict.fromkeys(methods),
                                 metrics={m: {"success": 0.25 if m == "ft_mars_user" else 0.5} for m in methods},
                                 trials=dict(side=["left", "right"], duration_s={m: [1.0, 8.0] for m in methods},
                                             outcome={m: list(teleop) if m == "teleop" else ["hit", "timeout"] for m in methods}))
                for c in cells}
        grid = dict(users=["U1"], tasks=["T1", "T2"], noises=[1.0], cells={c: {"obstacle": o} for c, o in cells.items()})
        (sweeps / name).mkdir(parents=True)
        (sweeps / name / "test.json").write_text(json.dumps(dict(args=dict(sweep=dict(methods=dict.fromkeys(methods, {}), grid=grid),
                                                                           test_trials=2), test=test)))
    write("grid_s0", ["teleop", "retrained_mars_user", "mars"], ("hit", "timeout"))
    write("grid_ft_s0", ["teleop", "ft_mars_user"], ft_teleop)


def test_grid_data_joins_each_seeds_fine_tuned_arm_trial_for_trial(tmp_path):
    _grid_results(tmp_path / "sim" / "sweeps")
    D = bf.grid_data(tmp_path)
    assert D["methods"] == ["teleop", "ft_mars_user", "mars"]   # the fine-tuned arm in place of the retrained one
    assert bf.seed_values(D, "U1", "T2", "ft_mars_user") == [25.0]
    assert D["cell"]("U1", "T2")["trials"]["outcome"]["ft_mars_user"] == ["hit", "timeout"]
    other = tmp_path / "other"   # a fine-tuned arm off the grid's trials is refused
    _grid_results(other / "sim" / "sweeps", ft_teleop=("timeout", "timeout"))
    with pytest.raises(ValueError):
        bf.grid_data(other)


def test_seed_statistics_pair_trials_within_a_seed_and_average_over_the_seeds():
    D = _seed_runs([{"mars": ["hit"] * 6 + ["timeout"] * 4, "teleop": ["hit"] * 3 + ["collision"] * 7},
                    {"mars": ["hit"] * 8 + ["collision"] * 2, "teleop": ["hit"] * 4 + ["timeout"] * 6}])
    assert bf.seed_values(D, "U1", "T2", "mars") == pytest.approx([60, 80]) and bf.seed_mean(D, "U1", "T2", "teleop") == pytest.approx(35)
    mean, lo, hi = bf.lead_ci(D, "U1", "T2", "mars", "teleop", n_boot=400)
    assert mean == pytest.approx(35) and lo <= mean <= hi
    assert bf.lead_ci(D, "U1", "T2", "mars", "mars", n_boot=100) == [0, 0, 0]


def _demo_time_results(root, mars, ft, teleop=("hit", "timeout")):
    """Sweep outputs for Figure 3e on two cells: per arm one {method: success} per seed, and the grid's teleop."""
    cells = ["U1xT2|1.0", "U2xT3|1.0"]
    def run(methods):
        return {"test": {c: {"metrics": {m: {"success": v} for m, v in methods.items()},
                             "trials": {"outcome": {"teleop": list(teleop)}}} for c in cells}}
    def write(name, methods):
        (root / name).mkdir(parents=True)
        (root / name / "test.json").write_text(json.dumps(run(methods)))
    for k, (m, f) in enumerate(zip(mars, ft)):
        write(f"demo_time_mars_s{k}", {"teleop": 0.2, **{f"mars_t{s}": m for s in bf.DEMO_SECONDS}})
        write(f"demo_time_ft_s{k}", {"teleop": 0.2, **{f"ft_mars_user_t{s}": f for s in bf.DEMO_SECONDS}})
    write("grid_s0", {"teleop": 0.2})


def test_fig3e_pools_the_cells_per_budget_and_draws_one_line_per_arm(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _demo_time_results(results / "sim" / "sweeps", mars=[0.7, 0.8], ft=[0.4, 0.5])
    D = bf.demo_time_data(results)
    assert D["seeds"] == {"mars": [0, 1], "ft": [0, 1]} and D["curve"]["mars"][300] == pytest.approx([70, 80])
    monkeypatch.setattr(bf, "demo_log", lambda task, seconds, seed=0: {"args": {"n_demos": seconds}, "best_step": 5000})
    bf.paper_style()
    fig = bf.plt.figure(figsize=(bf.FULL_W, bf.FIG3_H))
    stats = bf.draw_demo_time(fig, D, (0.6, 0.4, 0.35, 0.15))
    assert [line.get_color() for line in fig.axes[0].get_lines()] == [bf.DEMO_COLOR["ft"], bf.DEMO_COLOR["mars"]]
    assert stats["success"]["mars"]["300"][0] == pytest.approx(75) and stats["grid_demonstrations_minutes"]["T2"] == pytest.approx(5.0)
    bf.plt.close(fig)
    other = tmp_path / "other"  # a sweep off the grid's trials is refused
    _demo_time_results(other / "sim" / "sweeps", mars=[0.7], ft=[0.4])
    grid = json.loads((other / "sim" / "sweeps" / "grid_s0" / "test.json").read_text())
    for c in grid["test"].values():
        c["trials"]["outcome"]["teleop"] = ["timeout", "timeout"]
    (other / "sim" / "sweeps" / "grid_s0" / "test.json").write_text(json.dumps(grid))
    with pytest.raises(ValueError):
        bf.demo_time_data(other)
