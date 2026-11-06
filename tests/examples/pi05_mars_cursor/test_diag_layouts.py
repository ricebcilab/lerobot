"""The layout diagnostic: the trainer's layouts rebuilt, rigid motions of them, and a tiny run end to end."""

import json

import numpy as np
import pytest
from cursor_sa.env import TaskConfig, layouts_on_screen, obstacle_spec, sample_layouts
from diag_layouts import main, moved_layouts, rebuild_layouts
from test_sweep import _image_adapter
from test_training import COMMON, _record_splits, ckpt_dir  # noqa: F401  (ckpt_dir is a fixture)
from train import train_adapter

DISC = "{kind: disc, radius: 0.08, along: 0.5}"


def test_rebuilt_layouts_are_the_trainers(ckpt_dir, monkeypatch):  # noqa: F811
    user = ["--kind", "user", "--corruption", "{rotation_deg: 60}", "--name", "u", *COMMON]
    train_adapter.train(train_adapter.parse_args(user))
    splits = _record_splits(monkeypatch)
    task = [
        "--kind", "task", "--obstacle", DISC, "--name", "t_rel", "--user-module", "u", "--corruption", "{rotation_deg: 60}",
        "--token-source", "cnn", "--render-size", "64", "--token-geometry", "relative", "--p-free", "0.25", *COMMON,
    ]  # fmt: skip
    train_adapter.train(train_adapter.parse_args(task))
    lay = rebuild_layouts(0, 60, 0.25, obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5}))
    train = splits[-1]["train"].sampler
    # 15 of the 60 without the obstacle, at the end
    assert len(lay) == 60 and lay.obstacle_mask.any(1).sum() == 45
    assert np.allclose(lay.start, train.pos[:, 0].numpy())
    assert np.allclose(lay.obstacles, train.obstacles.numpy())


def _points(lay, b):
    """Trial b's start, real targets and real disc centres, [m, 2]."""
    return np.concatenate([lay.start[b][None], lay.targets[b, lay.mask[b]], lay.real_obstacles(b)[:, :2]])


def test_moved_layouts_stay_on_screen_and_are_rigid():
    task = TaskConfig(
        min_reach=0.35, max_reach=0.5, obstacle=obstacle_spec({"kind": "disc", "radius": 0.08, "along": 0.5})
    )
    lay = sample_layouts(np.random.default_rng(0), 40, 4, task)
    moved = moved_layouts(lay, np.random.default_rng(1))
    assert len(moved) == len(lay) and moved.scene is None and layouts_on_screen(moved).all()
    assert np.array_equal(moved.goal, lay.goal) and np.array_equal(moved.mask, lay.mask)
    assert np.array_equal(moved.obstacle_mask, lay.obstacle_mask)
    assert np.array_equal(moved.obstacles[..., 2], lay.obstacles[..., 2])
    for b in range(len(lay)):  # every pairwise distance kept
        before, after = _points(lay, b), _points(moved, b)
        dist = lambda p: np.linalg.norm(p[:, None] - p[None], axis=-1)  # noqa: E731
        assert np.allclose(dist(before), dist(after))
    assert (np.linalg.norm(moved.start - lay.start, axis=1) > 1e-6).mean() > 0.9  # the rows did move
    kept = moved_layouts(lay, np.random.default_rng(1), tries=0)  # no round: every row keeps its original
    assert np.array_equal(kept.start, lay.start) and np.array_equal(kept.obstacles, lay.obstacles)


def _refusal_sweep(tmp_path):
    """A users x tasks sweep with a disc task T2 and an open-field task T0 (no checkpoints: a refusal comes first)."""
    sweep = tmp_path / "refuse.yaml"
    sweep.write_text(
        "name: tiny_refuse\nagent: expert_chunk\narm: condition_grid.yaml\nmethods:\n"
        "  oracle: {mode: teleop, readout: off}\n"
        "  oracle_task: {mode: teleop, readout: off, reversal_adapter: {task: cell}}\n"
        "  oracle_task_rel: {mode: teleop, readout: off, reversal_adapter: {task: nowhere}}\n"
        "grid:\n  users: {U0: {corruption: null, adapter: null}}\n"
        f"  tasks: {{T2: {{obstacle: {DISC}, adapter: null}}, T0: {{obstacle: null, adapter: null}}}}\n"
        "  noises: [0.0]\ntrials: {tune: 2, test: 2}\nseed: 3\n"
    )
    return sweep


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--context", "T2", "--trials", "0"], "--trials must be at least 1"),
        (["--context", "T0", "--trials", "3"], "no obstacle"),
        (["--context", "T2", "--trials", "3", "--p-free", "1.0"], "the train set is empty"),
        (["--context", "T2", "--trials", "3", "--n-demos", "0"], "the train set would be empty"),
        (["--context", "T9", "--trials", "3"], "needs exactly one cell"),
    ],
)
def test_the_diagnostic_refuses_what_it_cannot_diagnose(tmp_path, capsys, extra, message):
    sweep, out = _refusal_sweep(tmp_path), tmp_path / "d"
    with pytest.raises(SystemExit) as stop:
        main(["--sweep", str(sweep), "--method", "oracle_task_rel", "--out", str(out), *extra, "--cpu"])
    assert stop.value.code == 2 and message in capsys.readouterr().err
    assert not out.exists()


def test_the_diagnostic_runs_end_to_end_on_a_tiny_sweep(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch, geometry="relative")
    sweep = tmp_path / "tiny.yaml"
    sweep.write_text(
        "name: tiny_diag\nagent: expert_chunk\narm: condition_grid.yaml\nmethods:\n"
        "  oracle: {mode: teleop, readout: off}\n"
        "  oracle_task: {mode: teleop, readout: off, reversal_adapter: {task: cell}}\n"
        f"  oracle_task_rel: {{mode: teleop, readout: off, reversal_adapter: {{task: {name}}}}}\n"
        "grid:\n  users: {U0: {corruption: null, adapter: null}}\n"
        f"  tasks: {{T2: {{obstacle: {DISC}, adapter: null}}}}\n  noises: [0.0]\n"
        "trials: {tune: 2, test: 2}\nseed: 3\n"
    )
    main(
        [
            "--sweep", str(sweep), "--context", "T2", "--method", "oracle_task_rel", "--n-demos", "6", "--trials", "3",
            "--out", str(tmp_path / "diag"), "--scene-cache", str(tmp_path / "scene"), "--cpu",
        ]
    )  # fmt: skip
    out = json.loads((tmp_path / "diag" / "T2_oracle_task_rel.json").read_text())
    assert set(out) == {"train", "fresh", "moved"}
    assert all(set(s) == {"oracle", "oracle_task", "oracle_task_rel"} for s in out.values())
    assert all(
        v["n"] == 3 and 0 <= v["hits"] <= 1 and 0 <= v["collision"] <= 1
        for s in out.values()
        for v in s.values()
    )
    assert {p.name.split("_cnn_")[0] for p in (tmp_path / "scene").glob("*.npy")} == {
        "T2_train",
        "T2_fresh",
        "T2_moved",
    }
