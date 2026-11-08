# ruff: noqa: C408
"""The image task module's report (reports/report_grid_image.py): the gate, the grid joined with the fine-tuned arm,
the demonstration-count scan and the recipe checks (select, width)."""

import json
import re

import numpy as np
import pytest
from report_grid_image import (
    crossing,
    first_count,
    gate_table,
    main as report_main,
    scan_rows,
    select_recipe,
    width_check,
)


def _m(hits, col=0.0, **extra):
    return {"success": hits, "collision": col, **extra}


def _run(by_ctx):
    return {"test": {f"U0x{ctx}|0.0": {"metrics": m} for ctx, m in by_ctx.items()}}


def _write(root, **runs):
    for name, run in runs.items():
        (root / "sim" / "sweeps" / name).mkdir(parents=True)
        (root / "sim" / "sweeps" / name / "test.json").write_text(json.dumps(run))
    return list(runs)


def _report(root, *argv):
    report_main([*argv, "--results", str(root), "--out", str(root / "out")])
    return (root / "out" / "report.md").read_text(), json.loads((root / "out" / "report.json").read_text())


def _cell(methods, hits, side=None, n=10, train_cell=False):
    pairs = list(zip(methods, hits, strict=True))
    outcome = {m: ["hit"] * int(h * n) + ["timeout"] * (n - int(h * n)) for m, h in pairs}
    metrics = {m: _m(h, 0.1 * i) for i, (m, h) in enumerate(pairs)}
    return dict(metrics=metrics, trials=dict(outcome=outcome, side=side or [1] * n), train_cell=train_cell)


def _scan_runs(views=None, prefix="views"):
    """Scan runs at 300 / 1000 / 3000 demonstrations on T4: parametric 92.8 / 3.4, the module's (hits, collisions)."""
    views = views or {300: (0.50, 0.45), 1000: (0.83, 0.13), 3000: (0.898, 0.064)}
    hidden = {300: 0.40, 1000: 0.87, 3000: 0.70}
    ref = {"oracle_task": _m(0.928, 0.034), "policy_task": _m(0.764)}
    arm = {
        n: {f"oracle_task_{prefix}_n{n}": _m(*v), f"policy_task_{prefix}_n{n}": _m(hidden[n])}
        for n, v in views.items()
    }
    return {n: _run({"T4": {**ref, **a}}) for n, a in arm.items()}


def test_gate_verdicts_pass_marginal_fail_and_flag():
    def cell(ref, arm_hits, arm_col=0.10, ft=0.5, hidden=0.25, ref_hidden=0.2):
        m = {"oracle": _m(0.9, 0.5), "oracle_task": _m(ref, 0.05), "oracle_task_rel": _m(arm_hits, arm_col)}
        m.update(policy_task=_m(ref_hidden, 0.1), policy_task_rel=_m(hidden, 0.1))
        return dict(metrics=m if ft is None else {**m, "oracle_ft": _m(ft, 0.2)})

    cells = {
        2: cell(0.95, 0.86),  # within 10 points of parametric: pass
        3: cell(0.95, 0.80),  # above the fine-tuned floor, more than 10 under parametric: marginal
        4: cell(0.95, 0.50, hidden=0.5),  # at the floor: FAIL, and goal-hidden flag
        5: cell(0.742, 0.65),  # T5: parametric 74.2, so 65 passes
        6: cell(0.95, 0.90, arm_col=0.30),  # hits fine, collisions 25 points over: FAIL
        7: cell(0.95, 0.80, ft=None),  # no fine-tuned row: no marginal possible
        8: cell(0.95, 0.90, ft=0.92),  # within 10 of parametric but under the floor: pass wins
        9: cell(0.928, 0.828),  # exactly 10 points under parametric: pass
        10: cell(0.928, 0.826),  # 10.2 points under: not a pass
        11: cell(0.95, 0.90, arm_col=0.15),  # collisions exactly 10 points over: pass
        # the flag needs more than 10 points over the goal-hidden reference (+10 exactly fires in raw floats)
        12: cell(0.9, 0.9, hidden=0.80, ref_hidden=0.70),
        13: cell(0.9, 0.9, hidden=0.336, ref_hidden=0.236),
        14: cell(0.9, 0.9, hidden=0.802, ref_hidden=0.70),
    }
    run = dict(test={f"U0xT{k}|0.0": c for k, c in cells.items()})
    rows = {r["context"]: r for r in gate_table(run, ["rel"])}
    verdicts = ["pass", "marginal", "FAIL", "pass", "FAIL", "FAIL", "pass", "pass", "marginal", "pass"]
    assert [rows[f"T{k}"]["verdict"] for k in range(2, 12)] == verdicts
    assert rows["T2"]["pass"] and not rows["T3"]["pass"] and rows["T7"]["ft_hits"] is None
    assert (rows["T2"]["ref_hits"], rows["T2"]["ft_hits"]) == (0.95, 0.5)
    assert [rows[f"T{k}"]["flag"] for k in (2, 4, 12, 13, 14)] == [False, True, False, False, True]


def test_the_gate_table_reports_wrong_targets(tmp_path):
    def m(hits, col, wrong=None):
        return _m(hits, col, **({} if wrong is None else {"wrong": wrong}))

    leak = {  # the goal-hidden module goes to the decoy far more often than the parametric one
        "oracle_task": m(0.9, 0.05, 0.02),
        "oracle_task_rel_rigid": m(0.88, 0.06, 0.04),
        "policy_task": m(0.5, 0.1, 0.30),
        "policy_task_rel_rigid": m(0.45, 0.1, 0.476),
    }
    old = {k: m(v["success"], v["collision"]) for k, v in leak.items()}  # without the wrong-target share
    bare = {"oracle_task": m(0.95, 0.05), "oracle_task_rel_rigid": m(0.80, 0.10), "policy_task": m(0.2, 0.1)}
    floor = {**bare, "oracle_ft": m(0.5, 0.2)}
    _write(tmp_path, check=_run({"T2": leak, "T3": bare, "T4": old, "T5": floor}))
    md, report = _report(tmp_path, "--check", "check")
    assert (
        "| context | arm | hits | ref hits | ft hits | collision | ref collision | verdict | hidden hits "
        "| ref hidden | flag | wrong target | hidden wrong | ref hidden wrong |"
    ) in md
    for row in (
        "| T2 | rel_rigid | 88.0 | 90.0 | n/a | 6.0 | 5.0 | pass | 45.0 | 50.0 | ok | 4.0 | 47.6 | 30.0 |",
        "| T3 | rel_rigid | 80.0 | 95.0 | n/a | 10.0 | 5.0 | FAIL | - | 20.0 | n/a | n/a | n/a | n/a |",
        "| T4 | rel_rigid | 88.0 | 90.0 | n/a | 6.0 | 5.0 | pass | 45.0 | 50.0 | ok | n/a | n/a | n/a |",
        "| T5 | rel_rigid | 80.0 | 95.0 | 50.0 | 10.0 | 5.0 | marginal |",
    ):
        assert row in md, row
    gate = {r["context"]: r for r in report["gate"]}
    wrong = [gate["T2"][k] for k in ("wrong_target", "hidden_wrong_target", "ref_hidden_wrong_target")]
    assert wrong == [0.04, 0.476, 0.30] and gate["T4"]["wrong_target"] is None
    assert gate["T3"]["hidden_hits"] is None and gate["T3"]["flag"] is False
    assert "Gate" in md and set(report) == {
        "gate"
    }  # the gate alone; the grid's tables are report_grid_vision.py's


def test_scan_rows_level_band_flag_and_first_counts():
    rows = scan_rows(_scan_runs())
    assert [(r["demos"], r["level"], r["in_band"], r["flag"]) for r in rows] == [
        (300, False, False, False),
        (1000, False, True, True),
        (3000, True, True, False),
    ]  # 3000 sits exactly 3 points off on both
    assert first_count(rows, "level") == {"T4": 3000} and first_count(rows, "in_band") == {"T4": 1000}
    assert first_count(rows[:1], "level") == {"T4": None}
    assert scan_rows(_scan_runs(prefix="rel"), prefix="rel") == rows  # the relative module's rows read alike
    for ref, hidden, flagged in ((0.70, 0.80, False), (0.236, 0.336, False), (0.236, 0.338, True)):
        cell = {"oracle_task": _m(0.9, 0.05), "policy_task": _m(ref), "oracle_task_views_n300": _m(0.9, 0.05)}
        run = _run({"T2": {**cell, "policy_task_views_n300": _m(hidden)}})
        assert scan_rows({300: run})[0]["flag"] is flagged, (ref, hidden)
    runs = _scan_runs()
    del runs[3000]["test"]["U0xT4|0.0"]["metrics"]["policy_task_views_n3000"]
    assert scan_rows(runs)[-1]["hidden_hits"] is None and not scan_rows(runs)[-1]["flag"]


def test_crossing_is_log_linear_and_marks_an_extrapolation():
    n, extra = crossing([300, 3000], [0.5, 0.9], 0.7)
    assert abs(n - 10 ** (np.log10(300) + 0.5)) < 1e-6 and not extra  # halfway in log10
    n, extra = crossing([300, 1000, 3000], [0.2, 0.4, 0.6], 0.8)
    assert extra and abs(n - 3000 * (3000 / 1000)) < 1e-6  # one more step of the last rise
    assert crossing([300, 1000, 3000], [0.6, 0.5, 0.4], 0.8) == (None, False)
    assert crossing([300, 1000], [0.9, 0.95], 0.8) == (300.0, False)
    n, extra = crossing([300, 3000], [0.5, 0.6], 0.7)  # exactly ten times the largest count: still given
    assert extra and abs(n - 30000) < 1e-6
    assert crossing([300, 3000], [0.5, 0.6], 0.701) == (None, False)  # past it: refused
    assert crossing([300, 1000, 3000], [0.5, 0.6, 0.602], 0.898) == (None, False)  # was 1.2e74


def test_the_scan_crossings_need_hits_and_collisions(tmp_path):
    review = _scan_runs({300: (0.60, 0.38), 1000: (0.84, 0.15), 3000: (0.90, 0.09)})
    md, report = _report(tmp_path, "--scan", *_write(tmp_path, **{f"scan{n}": r for n, r in review.items()}))
    line = next(r for r in md.splitlines() if r.startswith("- T4"))
    assert "## Scan" in md and "first level count more than 3000" in line and "hits and collisions" in line
    level = re.search(r"crossing of the level bar[^;]*? (\d+)( \(extrapolated\))?;", line)
    assert level and level.group(2) and int(level.group(1)) >= 3000, line  # hits alone: 2892, inside
    assert set(report) >= {"scan", "first_level", "first_in_band", "crossings"}
    cross = report["crossings"]["T4"]
    assert cross["level"]["extrapolated"] and cross["level"]["demos"] >= 3000
    # in band: hits reach 82.8 between 300 and 1000, collisions 13.4 between 1000 and 3000: the later one, inside
    col = 10 ** (np.log10(1000) + (0.134 - 0.15) / (0.09 - 0.15) * (np.log10(3000) - np.log10(1000)))
    assert not cross["in_band"]["extrapolated"] and abs(cross["in_band"]["demos"] - col) < 1e-6
    row = next(r for r in md.splitlines() if r.startswith("| T4 | 3000"))
    assert row.endswith("| 70.0 | 76.4 | ok |"), row
    rising = _scan_runs({300: (0.60, 0.38), 1000: (0.84, 0.15), 3000: (0.90, 0.20)})  # collisions rise
    del rising[3000]["test"]["U0xT4|0.0"]["metrics"]["policy_task_views_n3000"]  # a missing goal-hidden row
    names = _write(tmp_path / "b", **{f"scan{n}": r for n, r in rising.items()})
    md, report = _report(tmp_path / "b", "--scan", *names)
    assert report["crossings"]["T4"]["level"] == dict(demos=None, extrapolated=False)
    assert "crossing of the level bar on hits and collisions not reached on this trend" in md
    assert next(r for r in md.splitlines() if r.startswith("| T4 | 3000")).endswith("| - | 76.4 | n/a |")
    # a relative module's scan read with --prefix rel reports exactly what the views' does
    views = _report(
        tmp_path / "v", "--scan", *_write(tmp_path / "v", **{f"s{n}": r for n, r in _scan_runs().items()})
    )
    names = _write(tmp_path / "r", **{f"s{n}": r for n, r in _scan_runs(prefix="rel").items()})
    assert _report(tmp_path / "r", "--scan", *names, "--prefix", "rel") == views


def test_select_reads_a_prefix_and_a_base_arm_and_width_names_its_arms(tmp_path):
    views = {"T2": dict(base=0.50, span1=0.56, nomask=0.53, noaffine=0.40)}
    views["T4"] = dict(base=0.30, span1=0.32, nomask=0.33, noaffine=0.36)
    out = select_recipe(
        _run({c: {f"oracle_task_views_{r}": _m(h) for r, h in v.items()} for c, v in views.items()})
    )
    assert (
        out["adopted"] == ["span1"] and abs(out["means"]["base"] - 0.40) < 1e-9
    )  # +4; nomask +3; noaffine -2
    arms = {
        "T2": dict(rel=0.55, rel_jit=0.59, rel_rigid=0.57),
        "T4": dict(rel=0.35, rel_jit=0.39, rel_rigid=0.37),
    }
    rel = _run({c: {f"oracle_task_{a}": _m(h, 0.1) for a, h in v.items()} for c, v in arms.items()})
    out = select_recipe(rel, base="rel", changes=["jit", "rigid"], prefix="rel")
    assert out["adopted"] == ["jit"] and abs(out["means"]["rel"] - 0.45) < 1e-9  # jit 4 points over, rigid 2
    assert set(out["per_context"]) == {"rel", "jit", "rigid"}
    assert out["per_context"]["jit"]["T4"] == {"hits": 0.39, "collision": 0.1}
    width = _run({"T4": {"oracle_task_views_base": _m(0.35), "oracle_task_views_full": _m(0.40)}})
    assert width_check(width)["ok"]  # exactly 5 points under
    width["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_base"] = _m(0.348)
    assert not width_check(width)["ok"]
    hits = {"rel": 0.50, "rel_jit": 0.52, "rel_rigid": 0.56, "rel_both": 0.535}
    named = {"oracle_task_views_r512": _m(0.30), "oracle_task_views_raw": _m(0.40)}
    aug = _run({c: {f"oracle_task_{a}": _m(h) for a, h in hits.items()} for c in ("T2", "T4")})
    _write(tmp_path, aug=aug, wid=_run({"T4": named}))
    out, where = tmp_path / "out", ["--results", str(tmp_path), "--out", str(tmp_path / "out")]
    report_main(["--select", "aug", "--prefix", "rel", "--base", "rel", *where])
    report_main(["--width", "wid", "--reduced", "r512", "--full", "raw", *where])
    sel, wid = (json.loads((out / f).read_text()) for f in ("select.json", "width.json"))
    assert sel["adopted"] == ["rigid", "both"] and list(sel["means"]) == [
        "rel",
        "jit",
        "rigid",
        "both",
    ]  # +2, +6, +3.5
    assert set(sel) == {"means", "adopted", "per_context"}
    assert sel["per_context"]["rel"]["T2"] == {"hits": 0.5, "collision": 0.0}
    assert (wid["reduced_hits"], wid["full_hits"], wid["ok"]) == (0.30, 0.40, False)


def test_rules_refuse_a_run_that_lacks_a_row(tmp_path, capsys):
    runs = _scan_runs()
    runs[1000]["test"]["U0xT4|0.0"]["metrics"]["oracle_task"] = _m(0.90, 0.034)
    with pytest.raises(ValueError, match="layouts"):  # the parametric reference differs between counts
        scan_rows(runs)
    for n, cell, key, match in (
        (1000, "U0xT4|0.0", "oracle_task", "oracle_task row for context T4"),
        (1000, "U0xT4|0.0", "oracle_task_rel_n1000", "no oracle_task_rel_n1000 row for context T4"),
    ):
        runs = _scan_runs(prefix="rel")
        del runs[n]["test"][cell]["metrics"][key]
        with pytest.raises(ValueError, match=match):
            scan_rows(runs, prefix="rel")
    runs = _scan_runs()
    runs[1000]["test"]["U0xT2|0.0"] = {
        "metrics": {"oracle_task": _m(0.9), "oracle_task_views_n1000": _m(0.8)}
    }
    with pytest.raises(ValueError, match="300 demonstrations has no oracle_task row for context T2"):
        scan_rows(runs)
    recipes = ("base", "span1", "nomask", "noaffine")
    check = _run({c: {f"oracle_task_views_{r}": _m(0.5) for r in recipes} for c in ("T2", "T4")})
    del check["test"]["U0xT4|0.0"]["metrics"]["oracle_task_views_noaffine"]
    with pytest.raises(ValueError, match="oracle_task_views_noaffine has no row for context T4"):
        select_recipe(check)
    for c in ("T2", "T4"):
        del check["test"][f"U0x{c}|0.0"]["metrics"]["oracle_task_views_span1"]
    with pytest.raises(ValueError, match="oracle_task_views_span1 has no rows"):
        select_recipe(check)
    rel = _run(
        {c: {f"oracle_task_{a}": _m(0.5) for a in ("rel", "rel_jit", "rel_rigid")} for c in ("T2", "T4")}
    )
    with pytest.raises(ValueError, match="oracle_task_rel_both has no rows"):
        select_recipe(rel, base="rel", prefix="rel")  # the rel arms by default: jit, rigid and both
    width = _run({"T4": {"oracle_task_views_base": _m(0.4)}})
    with pytest.raises(ValueError, match="oracle_task_views_full has no row for context T4"):
        width_check(width)
    with pytest.raises(ValueError, match="oracle_task_views_base has no row for context T2"):
        width_check(width, context="T2")
    scans = _write(tmp_path, **{f"scan{n}": r for n, r in _scan_runs().items()}, again300=_scan_runs()[300])
    rels = _write(tmp_path, **{f"rel{n}": r for n, r in _scan_runs(prefix="rel").items()})
    _write(tmp_path, check=_run({}))
    for argv in (
        [],  # one mode is required
        ["--select", "scan300", "--width", "scan300"],  # and only one
        ["--scan", *scans],  # two runs with the same count
        ["--scan", *rels],  # a rel run read with the default prefix
    ):
        with pytest.raises(SystemExit):
            report_main([*argv, "--results", str(tmp_path), "--out", str(tmp_path / "x")])
    assert "300" in capsys.readouterr().err


def test_width_reads_a_prefix(tmp_path):
    """--width --prefix rel: the reduced and full rows are oracle_task_rel_<arm> (an arm named like the prefix reads the
    bare row), in --context."""
    rel = _run(
        {
            c: {"oracle_task_rel": _m(h), "oracle_task_rel_full": _m(0.60)}
            for c, h in (("T2", 0.50), ("T4", 0.58))
        }
    )
    assert width_check(rel, reduced="rel", prefix="rel") == dict(reduced_hits=0.58, full_hits=0.60, ok=True)
    assert not width_check(rel, reduced="rel", prefix="rel", context="T2")["ok"]  # 10 points under
    with pytest.raises(ValueError, match="oracle_task_views_base has no row"):
        width_check(rel)  # the default prefix is views
    _write(tmp_path, wid=rel)
    out, where = tmp_path / "out", ["--results", str(tmp_path), "--out", str(tmp_path / "out")]
    report_main(["--width", "wid", "--prefix", "rel", "--reduced", "rel", "--context", "T2", *where])
    assert json.loads((out / "width.json").read_text()) == dict(reduced_hits=0.50, full_hits=0.60, ok=False)
