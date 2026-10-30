"""The paper figures' builder (figures/build_figures.py)."""

import json

import numpy as np
import pytest
from PIL import Image

from cursor_sa.paths import RESULTS
from figures import build_figures as bf


def test_fig1_instance_is_finite_and_its_directions_are_angles():
    inst = bf.fig1_instance("cpu", n_steps=6)
    assert np.isfinite(inst["path"]).all() and len(inst["path"]) == 7
    for k, v in inst["chunks"].items():
        assert v.shape == (16, 2) and np.isfinite(v).all(), k
        assert -1 <= inst["cosine"][k] <= 1 and -180 <= inst["direction"][k] <= 180
    assert set(inst["latents"]) == {"frs_8", "mars_8", "frs_10", "mars_10"}
    assert inst["depth_full"] == 10 and inst["depth_study"] == 8 and inst["execute"] == 2
    assert len(inst["bearings"]) == 4 and inst["bearings"][inst["goal_index"]] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(inst["served"], inst["stats"]["served"], atol=1e-3)


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
    assert set(stats) == {"fig1"} and stats["fig1"]["depth_shown"] == 10 and "chunk_cosine" in stats["fig1"]


@pytest.mark.skipif(not any((RESULTS / f"p{i}").exists() for i in range(1, 10)), reason="no recorded study sessions")
def test_fig2_builds_from_the_recorded_sessions(tmp_path):
    bf.main(["--only", "fig2", "--out", str(tmp_path)])
    stats = json.loads((tmp_path / "stats.json").read_text())["fig2"]
    assert stats["participants"] >= 1 and set(stats["success_mean_ci"]) == set(bf.ARMS)
    assert all(len(v) == 3 for v in stats["success_mean_ci"].values())


@pytest.mark.skipif(not (RESULTS / "sim" / "sweeps" / "grid" / "test.json").exists(), reason="no grid results")
def test_fig3_builds_from_the_grid(tmp_path):
    bf.main(["--only", "fig3", "--out", str(tmp_path), "--device", "cpu"])
    stats = json.loads((tmp_path / "stats.json").read_text())["fig3"]
    cells = {f"U{u}xT{t}" for u in range(1, 5) for t in range(1, 6)}
    assert set(stats["success_per_cell"]) == cells == set(stats["best_per_cell"]) and stats["seeds"][0] == 0
    assert all(len(v) == len(stats["seeds"]) for c in stats["success_per_cell_per_seed"].values() for v in c.values())
    for b in stats["best_per_cell"].values():   # the lead is the best's success over the second's, with its CI
        assert b["success"] >= b["second_success"] and b["lead"][1] <= b["lead"][0] <= b["lead"][2]
        assert b["lead"][0] == pytest.approx(b["success"] - b["second_success"], abs=0.11) and b["level"] == (b["lead"][1] <= 0)
    assert stats["composed_beats_all_baselines"]["of"] == 16 and stats["rollouts"]["trials"] == bf.ROLLOUT_TRIALS
    assert Image.open(tmp_path / "fig3_grid.png").size == (round(300 * bf.FULL_W), round(300 * bf.FIG3_H))


def _seed_runs(per_seed, cell="U1xT2"):
    """A grid's data (grid_data's shape) for one cell from each seed's {method: trial outcomes}."""
    runs = [{"test": {f"{cell}|1.0": {"metrics": {m: {"success": float(np.mean([o == "hit" for o in v]))} for m, v in seed.items()},
                                      "trials": {"outcome": seed}}}} for seed in per_seed]
    return dict(runs=runs, users=[cell.split("x")[0]], tasks=["T1", cell.split("x")[1]], methods=list(per_seed[0]), leads={},
                test_n=len(next(iter(per_seed[0].values()))), cell=lambda u, t, k=0: runs[k]["test"][f"{u}x{t}|1.0"])


def test_seed_statistics_pair_trials_within_a_seed_and_average_over_the_seeds():
    D = _seed_runs([{"mars": ["hit"] * 6 + ["timeout"] * 4, "teleop": ["hit"] * 3 + ["collision"] * 7},
                    {"mars": ["hit"] * 8 + ["collision"] * 2, "teleop": ["hit"] * 4 + ["timeout"] * 6}])
    assert bf.seed_values(D, "U1", "T2", "mars") == pytest.approx([60, 80]) and bf.seed_mean(D, "U1", "T2", "teleop") == pytest.approx(35)
    assert bf.ranked(D, "U1", "T2") == ["mars", "teleop"]
    mean, lo, hi = bf.lead_ci(D, "U1", "T2", "mars", "teleop", n_boot=400)
    assert mean == pytest.approx(35) and lo <= mean <= hi
    assert bf.lead_ci(D, "U1", "T2", "mars", "mars", n_boot=100) == [0, 0, 0]
