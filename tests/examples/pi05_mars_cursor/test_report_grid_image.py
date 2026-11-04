# ruff: noqa: C408
import json

import pytest
from report_grid_image import gate_table, join_ft, main as report_main, pooled_over_seeds


def _cell(methods, hits, side=None, n=10):
    return dict(
        metrics={
            m: dict(success=h, collision=0.1 * i) for i, (m, h) in enumerate(zip(methods, hits, strict=True))
        },
        trials=dict(
            outcome={
                m: ["hit"] * int(h * n) + ["timeout"] * (n - int(h * n))
                for m, h in zip(methods, hits, strict=True)
            },
            side=side or [1] * n,
        ),
        train=False,
    )


def test_gate_rule_and_flag():
    cells = {
        f"U0xT{k}|0.0": dict(
            metrics={
                "oracle": dict(success=0.9, collision=0.5),
                "oracle_task": dict(success=0.95, collision=0.05),
                "oracle_task_vlm": dict(success=0.85, collision=0.10 if k != 5 else 0.3),
                "policy_task": dict(success=0.2, collision=0.1),
                "policy_task_vlm": dict(success=0.25 if k != 4 else 0.5, collision=0.1),
            }
        )
        for k in (2, 3, 4, 5)
    }
    rows = {(r["context"], r["arm"]): r for r in gate_table(dict(test=cells), ["vlm"])}
    assert rows[("T2", "vlm")]["pass"] and not rows[("T5", "vlm")]["pass"]
    assert rows[("T4", "vlm")]["flag"] and not rows[("T2", "vlm")]["flag"]


def test_join_ft_checks_the_trials_then_pools_over_seeds(tmp_path):
    ms = ["teleop", "mars", "mars_vlm"]
    grid = dict(
        test={f"U{u}xT{t}|1.0": _cell(ms, [0.2, 0.7, 0.6 + 0.01 * u]) for u in (1, 2) for t in (2, 3)}
    )
    ft = dict(test={k: _cell(["teleop", "ft_mars_user"], [0.2, 0.4]) for k in grid["test"]})
    joined = join_ft(grid, ft)
    assert joined["test"]["U1xT2|1.0"]["metrics"]["ft_mars_user"]["success"] == 0.4
    bad = dict(test={k: _cell(["teleop", "ft_mars_user"], [0.3, 0.4]) for k in grid["test"]})
    with pytest.raises(ValueError, match="trials"):
        join_ft(grid, bad)
    pooled = pooled_over_seeds([joined, joined], "mars_vlm")
    assert pooled["mean"] == pytest.approx(61.5)
    assert len(pooled["per_seed"]) == 2 and pooled["lo"] <= pooled["mean"] <= pooled["hi"]
    sweeps = tmp_path / "sim" / "sweeps"
    for name, run in (("grid_task_check_image", dict(test={})), ("grid_image_s0", grid), ("grid_ft_s0", ft)):
        (sweeps / name).mkdir(parents=True)
        (sweeps / name / "test.json").write_text(json.dumps(run))
    report_main(
        [
            "--check",
            "grid_task_check_image",
            "--grid",
            "grid_image_s0",
            "--ft",
            "grid_ft_s0",
            "--results",
            str(tmp_path),
            "--out",
            str(tmp_path / "rep"),
        ]
    )
    assert (tmp_path / "rep" / "report.md").exists()
    assert "mars_vlm" in (tmp_path / "rep" / "report.md").read_text()
    assert json.loads((tmp_path / "rep" / "report.json").read_text())["gate"] == []
