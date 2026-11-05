# ruff: noqa: C408
import json
import re

import numpy as np
import pytest
from report_grid_image import (
    crossing,
    first_count,
    gate_table,
    join_ft,
    main as report_main,
    pooled_over_seeds,
    scan_rows,
    select_recipe,
    width_check,
)


def _cell(methods, hits, side=None, n=10, train_cell=False):
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
        train_cell=train_cell,
    )


def test_gate_verdicts_pass_marginal_fail_and_flag():
    def cell(ref, arm_hits, arm_col=0.10, ft=0.5, hidden=0.25):
        m = {
            "oracle": dict(success=0.9, collision=0.5),
            "oracle_task": dict(success=ref, collision=0.05),
            "oracle_task_rel": dict(success=arm_hits, collision=arm_col),
            "policy_task": dict(success=0.2, collision=0.1),
            "policy_task_rel": dict(success=hidden, collision=0.1),
        }
        if ft is not None:
            m["oracle_ft"] = dict(success=ft, collision=0.2)
        return dict(metrics=m)

    cells = {
        "U0xT2|0.0": cell(0.95, 0.86),  # within 10 points of parametric: pass
        "U0xT3|0.0": cell(0.95, 0.80),  # above the fine-tuned floor, more than 10 under parametric: marginal
        "U0xT4|0.0": cell(0.95, 0.50, hidden=0.5),  # at the floor: FAIL, and goal-hidden flag
        "U0xT5|0.0": cell(0.742, 0.65),  # T5: parametric 74.2, so 65 passes
        "U0xT6|0.0": cell(0.95, 0.90, arm_col=0.30),  # hits fine, collisions 25 points over: FAIL
        "U0xT7|0.0": cell(0.95, 0.80, ft=None),  # no fine-tuned row: no marginal possible
        "U0xT8|0.0": cell(0.95, 0.90, ft=0.92),  # within 10 of parametric but under the floor: pass wins
        "U0xT9|0.0": cell(0.928, 0.828),  # exactly 10 points under parametric: pass
        "U0xT10|0.0": cell(0.928, 0.826),  # 10.2 points under: not a pass
        "U0xT11|0.0": cell(0.95, 0.90, arm_col=0.15),  # collisions exactly 10 points over: pass
    }
    rows = {r["context"]: r for r in gate_table(dict(test=cells), ["rel"])}
    assert [rows[f"T{k}"]["verdict"] for k in (2, 3, 4, 5, 6, 7, 8)] == [
        "pass",
        "marginal",
        "FAIL",
        "pass",
        "FAIL",
        "FAIL",
        "pass",
    ]
    assert [rows[f"T{k}"]["verdict"] for k in (9, 10, 11)] == ["pass", "marginal", "pass"]
    assert (
        rows["T2"]["pass"]
        and not rows["T3"]["pass"]
        and rows["T2"]["ref_hits"] == 0.95
        and rows["T2"]["ft_hits"] == 0.5
    )
    assert rows["T4"]["flag"] and not rows["T2"]["flag"] and rows["T7"]["ft_hits"] is None


def test_the_gate_markdown_has_the_three_verdicts_and_n_a_without_a_fine_tuned_row(tmp_path):
    def cell(ft):
        m = {
            "oracle_task": dict(success=0.95, collision=0.05),
            "oracle_task_rel": dict(success=0.80, collision=0.10),
            "policy_task": dict(success=0.2, collision=0.1),
        }
        if ft is not None:
            m["oracle_ft"] = dict(success=ft, collision=0.2)
        return dict(metrics=m)

    _write(tmp_path, check=dict(test={"U0xT2|0.0": cell(0.5), "U0xT3|0.0": cell(None)}))
    report_main(["--check", "check", "--results", str(tmp_path), "--out", str(tmp_path / "rep")])
    md = (tmp_path / "rep" / "report.md").read_text()
    assert (
        "| context | arm | hits | ref hits | ft hits | collision | ref collision | verdict | hidden hits "
        "| ref hidden | flag |"
    ) in md
    assert "| T2 | rel | 80.0 | 95.0 | 50.0 | 10.0 | 5.0 | marginal |" in md
    assert "| T3 | rel | 80.0 | 95.0 | n/a | 10.0 | 5.0 | FAIL |" in md
    report = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert {"verdict", "ref_hits", "ft_hits", "pass"} <= set(report["gate"][0])


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


def _write(tmp_path, **runs):
    sweeps = tmp_path / "sim" / "sweeps"
    for name, run in runs.items():
        (sweeps / name).mkdir(parents=True)
        (sweeps / name / "test.json").write_text(json.dumps(run))


def test_join_ft_accepts_a_fine_tuned_run_with_more_cells_and_still_checks_the_rest():
    ms = ["teleop", "mars"]
    grid = dict(test={f"U{u}xT{t}|1.0": _cell(ms, [0.2, 0.7]) for u in (1, 2) for t in (2, 3)})
    ft_cells = {k: _cell(["teleop", "ft_mars_user"], [0.2, 0.4]) for k in grid["test"]}
    ft_cells.update(
        {f"U{u}xT1|1.0": _cell(["teleop", "ft_mars_user"], [0.2, 0.9]) for u in (1, 2)}
    )  # the extra T1 cells
    joined = join_ft(grid, dict(test=ft_cells))
    assert set(joined["test"]) == set(grid["test"])  # the grid's cells only
    assert joined["test"]["U2xT3|1.0"]["metrics"]["ft_mars_user"]["success"] == 0.4
    short = dict(test={k: v for k, v in ft_cells.items() if k != "U2xT3|1.0"})
    with pytest.raises(ValueError, match="U2xT3"):
        join_ft(grid, short)
    ft_cells["U1xT2|1.0"] = _cell(["teleop", "ft_mars_user"], [0.2, 0.4], side=[-1] * 10)
    with pytest.raises(ValueError, match="side"):
        join_ft(grid, dict(test=ft_cells))
    ft_cells["U1xT2|1.0"] = _cell(["teleop", "ft_mars_user"], [0.3, 0.4])
    with pytest.raises(ValueError, match="teleop"):
        join_ft(grid, dict(test=ft_cells))


def test_the_gate_is_reported_before_the_grid_exists(tmp_path):
    check = dict(
        test={
            f"U0xT{k}|0.0": dict(
                metrics={
                    "oracle_task": dict(success=0.95, collision=0.05),
                    "oracle_task_vlm": dict(success=0.85, collision=0.10),
                    "oracle_task_cnn": dict(success=0.5, collision=0.10),
                    "policy_task": dict(success=0.2, collision=0.1),
                }
            )
            for k in (2, 3)
        }
    )
    _write(tmp_path, check=check)
    report_main(["--check", "check", "--results", str(tmp_path), "--out", str(tmp_path / "rep")])
    md = (tmp_path / "rep" / "report.md").read_text()
    out = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert {(r["context"], r["arm"], r["pass"]) for r in out["gate"]} == {
        ("T2", "vlm", True),
        ("T3", "vlm", True),
        ("T2", "cnn", False),
        ("T3", "cnn", False),
    }
    assert "Gate" in md and "Pooled" not in md and out["pooled"] == {}
    with pytest.raises(SystemExit):  # --ft pairs with --grid
        report_main(
            ["--check", "check", "--ft", "x", "--results", str(tmp_path), "--out", str(tmp_path / "rep2")]
        )


def test_the_per_cell_tables_mark_the_trained_cells_and_one_seed_has_no_interval(tmp_path):
    ms = ["teleop", "mars", "mars_vlm"]
    grid = dict(
        test={
            "U1xT2|1.0": _cell(ms, [0.2, 0.7, 0.6], train_cell=True),
            "U1xT3|1.0": _cell(ms, [0.2, 0.5, 0.4]),
        }
    )
    _write(tmp_path, check=dict(test={}), g0=grid)
    report_main(
        ["--check", "check", "--grid", "g0", "--results", str(tmp_path), "--out", str(tmp_path / "rep")]
    )
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "| cell | train |" in md
    assert "| U1xT2\\|1.0 | train |" in md and "| U1xT3\\|1.0 |  |" in md
    unescaped = lambda line: len(re.findall(r"(?<!\\)\|", line))  # noqa: E731
    lines, tables = md.splitlines(), 0
    for i, line in enumerate(lines):
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            tables += 1
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                assert unescaped(row) == unescaped(line), row
    assert tables >= 2
    assert "| mars | 60.0 | n/a | n/a |" in md
    one = json.loads((tmp_path / "rep" / "report.json").read_text())["pooled"]["mars"]
    assert one["lo"] == one["hi"] == one["mean"]  # the JSON keeps the degenerate interval


def test_pooling_over_really_different_seeds_and_the_degenerate_interval():
    ms = ["teleop", "mars"]

    def run(hit):
        return dict(test={k: _cell(ms, [0.2, hit]) for k in ("U1xT2|1.0", "U2xT2|1.0")})

    low, high = run(0.5), run(0.9)
    pooled = pooled_over_seeds([low, high], "mars")
    assert pooled["per_seed"] == pytest.approx([50.0, 90.0]) and pooled["mean"] == pytest.approx(70.0)
    # t-CI of two values: mean +- t(0.975, 1) * sd / sqrt(2) = 70 +- 12.706 * 28.28 / 1.414
    half = 12.7062 * np.std([50.0, 90.0], ddof=1) / np.sqrt(2)
    assert pooled["lo"] == pytest.approx(70.0 - half, rel=1e-3) and pooled["hi"] == pytest.approx(
        70.0 + half, rel=1e-3
    )
    same = pooled_over_seeds([low, low, low], "mars")
    assert same["lo"] == same["hi"] == same["mean"] == pytest.approx(50.0)
    single = pooled_over_seeds([low], "mars")
    assert single["lo"] == single["hi"] == single["mean"]


def _m(hits, col=0.0):
    return {"success": hits, "collision": col}


def _run(by_ctx):
    return {"test": {f"U0x{ctx}|0.0": {"metrics": m} for ctx, m in by_ctx.items()}}


def _scan_runs():
    ref = {"oracle_task": _m(0.928, 0.034), "policy_task": _m(0.764)}
    return {
        300: _run(
            {"T4": {**ref, "oracle_task_views_n300": _m(0.50, 0.45), "policy_task_views_n300": _m(0.40)}}
        ),
        1000: _run(
            {"T4": {**ref, "oracle_task_views_n1000": _m(0.83, 0.13), "policy_task_views_n1000": _m(0.87)}}
        ),
        3000: _run(
            {"T4": {**ref, "oracle_task_views_n3000": _m(0.898, 0.064), "policy_task_views_n3000": _m(0.70)}}
        ),
    }


def test_scan_rows_level_band_flag_and_first_counts():
    runs = _scan_runs()
    rows = scan_rows(runs)
    assert [(r["demos"], r["level"], r["in_band"], r["flag"]) for r in rows] == [
        (300, False, False, False),
        (1000, False, True, True),
        (3000, True, True, False),
    ]  # 3000 sits exactly 3 points off on both
    assert first_count(rows, "level") == {"T4": 3000} and first_count(rows, "in_band") == {"T4": 1000}
    assert first_count(rows[:1], "level") == {"T4": None}
    runs[1000]["test"]["U0xT4|0.0"]["metrics"]["oracle_task"] = _m(0.90, 0.034)
    with pytest.raises(ValueError, match="layouts"):
        scan_rows(runs)


def test_crossing_is_log_linear_and_marks_an_extrapolation():
    n, extra = crossing([300, 3000], [0.5, 0.9], 0.7)
    assert abs(n - 10 ** (np.log10(300) + 0.5)) < 1e-6 and not extra  # halfway in log10
    n, extra = crossing([300, 1000, 3000], [0.2, 0.4, 0.6], 0.8)
    assert extra and abs(n - 3000 * (3000 / 1000)) < 1e-6  # one more step of the last rise
    assert crossing([300, 1000, 3000], [0.6, 0.5, 0.4], 0.8) == (None, False)
    assert crossing([300, 1000], [0.9, 0.95], 0.8) == (300.0, False)


def test_select_recipe_and_width_check_apply_their_margins():
    check = _run(
        {
            "T2": {
                f"oracle_task_views_{r}": _m(h)
                for r, h in (("base", 0.50), ("span1", 0.56), ("nomask", 0.53), ("noaffine", 0.40))
            },
            "T4": {
                f"oracle_task_views_{r}": _m(h)
                for r, h in (("base", 0.30), ("span1", 0.32), ("nomask", 0.33), ("noaffine", 0.36))
            },
        }
    )
    out = select_recipe(check)
    assert (
        out["adopted"] == ["span1"] and abs(out["means"]["base"] - 0.40) < 1e-9
    )  # +4; nomask +3 exactly; noaffine -2
    width = _run({"T4": {"oracle_task_views_base": _m(0.35), "oracle_task_views_full": _m(0.40)}})
    assert width_check(width)["ok"]  # exactly 5 points under
    width["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_base"] = _m(0.348)
    assert not width_check(width)["ok"]


def _write_runs(tmp_path, runs):
    names = []
    for n, run in runs.items():
        d = tmp_path / "sim" / "sweeps" / f"scan{n}"
        d.mkdir(parents=True)
        (d / "test.json").write_text(json.dumps(run))
        names.append(f"scan{n}")
    return names


def test_main_scan_writes_report(tmp_path):
    names = _write_runs(tmp_path, _scan_runs())
    out = tmp_path / "out"
    report_main(["--scan", *names, "--results", str(tmp_path), "--out", str(out)])
    md = (out / "report.md").read_text()
    assert "## Scan" in md
    level = next(line for line in md.splitlines() if "first level" in line.lower() and "T4" in line)
    assert "3000" in level
    assert set(json.loads((out / "report.json").read_text())) >= {
        "scan",
        "first_level",
        "first_in_band",
        "crossings",
    }


def test_main_scan_says_more_than_when_no_count_is_level(tmp_path):
    runs = _scan_runs()
    runs[3000]["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_n3000"] = _m(0.60, 0.2)
    names = _write_runs(tmp_path, runs)
    report_main(["--scan", *names, "--results", str(tmp_path), "--out", str(tmp_path / "out")])
    assert "more than 3000" in (tmp_path / "out" / "report.md").read_text()


def test_main_select_and_width_write_json_and_one_mode_is_required(tmp_path, capsys):
    sel = _run({"T4": {f"oracle_task_views_{r}": _m(0.3) for r in ("base", "span1", "nomask", "noaffine")}})
    wid = _run({"T4": {"oracle_task_views_base": _m(0.4), "oracle_task_views_full": _m(0.4)}})
    for name, run in (("sel", sel), ("wid", wid)):
        d = tmp_path / "sim" / "sweeps" / name
        d.mkdir(parents=True)
        (d / "test.json").write_text(json.dumps(run))
    out = tmp_path / "out"
    report_main(["--select", "sel", "--results", str(tmp_path), "--out", str(out)])
    report_main(["--width", "wid", "--results", str(tmp_path), "--out", str(out)])
    assert json.loads((out / "select.json").read_text())["adopted"] == []
    assert json.loads((out / "width.json").read_text())["ok"] is True
    with pytest.raises(SystemExit):
        report_main(["--results", str(tmp_path), "--out", str(out)])
    with pytest.raises(SystemExit):
        report_main(["--select", "sel", "--width", "wid", "--results", str(tmp_path), "--out", str(out)])


def test_rules_refuse_a_run_that_lacks_a_row():
    runs = _scan_runs()
    del runs[1000]["test"]["U0xT4|0.0"]["metrics"]["oracle_task"]
    with pytest.raises(ValueError, match="oracle_task row for context T4"):
        scan_rows(runs)
    runs = _scan_runs()
    runs[1000]["test"]["U0xT2|0.0"] = {
        "metrics": {"oracle_task": _m(0.9), "oracle_task_views_n1000": _m(0.8)}
    }
    with pytest.raises(ValueError, match="300 demonstrations has no oracle_task row for context T2"):
        scan_rows(runs)
    runs = _scan_runs()
    del runs[3000]["test"]["U0xT4|0.0"]["metrics"]["policy_task_views_n3000"]
    assert scan_rows(runs)[-1]["hidden_hits"] is None and not scan_rows(runs)[-1]["flag"]

    check = _run(
        {
            "T2": {f"oracle_task_views_{r}": _m(0.5) for r in ("base", "span1", "nomask", "noaffine")},
            "T4": {f"oracle_task_views_{r}": _m(0.5) for r in ("base", "span1", "nomask")},
        }
    )
    with pytest.raises(ValueError, match="oracle_task_views_noaffine has no row for context T4"):
        select_recipe(check)
    del check["test"]["U0xT2|0.0"]["metrics"]["oracle_task_views_span1"]
    del check["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_span1"]
    with pytest.raises(ValueError, match="oracle_task_views_span1 has no rows"):
        select_recipe(check)

    width = _run({"T4": {"oracle_task_views_base": _m(0.4)}})
    with pytest.raises(ValueError, match="oracle_task_views_full has no row for context T4"):
        width_check(width)
    with pytest.raises(ValueError, match="oracle_task_views_base has no row for context T2"):
        width_check(width, context="T2")
