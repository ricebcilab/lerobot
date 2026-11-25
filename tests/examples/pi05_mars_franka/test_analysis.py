"""The study's analysis: the three statistics, the tables, the pairing rule and the CSV.

No scipy: the Wilson bounds, McNemar's exact test and the signed-rank test are checked against values
computed by hand (or from the definitions, in closed form) on small tables. The records are written the
way the study writes them, and a foreign record is refused by name.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import analysis
import numpy as np
import pytest


def _record(
    *,
    run: str = "run1",
    arm: str = "frs",
    pair: int = 0,
    success: bool = True,
    authority: float | None = 0.5,
    scene_hash: str = "scene0",
    **extra,
) -> dict:
    """One Phase D record, with the fields the analysis reads and the provenance it ignores."""
    record = {
        "run": run,
        "arm": arm,
        "pair": pair,
        "prompt": "pick the cube",
        "success": success,
        "steps": 10,
        "duration_s": 1.5,
        "authority": authority,
        "engaged_fraction": 0.5,
        "mean_r": 0.3,
        "q0": [0.0] * 7,
        "scene_hash": scene_hash,
        "config": "/tmp/franka.toml",
        "config_sha256": "abc",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    record.update(extra)
    return record


def _write_run(directory: Path, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trials.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------- Wilson


def test_wilson_bounds_are_the_hand_computed_ones():
    """The closed form, checked against values computed outside the module (p=0.8, n=10)."""
    low, high = analysis.wilson(8, 10)
    assert low == pytest.approx(0.49016247153664183)
    assert high == pytest.approx(0.9433178485456247)

    low, high = analysis.wilson(2, 10)
    assert low == pytest.approx(0.056682151454375274)
    assert high == pytest.approx(0.5098375284633582)


def test_wilson_edges_do_not_collapse():
    """No trials is no interval; 0 and n successes keep the Wilson interval's non-zero other bound."""
    low, high = analysis.wilson(0, 0)
    assert np.isnan(low) and np.isnan(high)
    low, high = analysis.wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.27753279986288903)
    low, high = analysis.wilson(10, 10)
    assert low == pytest.approx(0.7224672001371109) and high == 1.0


# ---------------------------------------------------------------- McNemar


def test_mcnemar_matches_the_exact_binomial_on_a_discordant_table():
    """b = 5, c = 0 of 5 discordant pairs: the two-sided exact p is 2 * (1/2)^5 = 0.0625."""
    result = analysis.mcnemar([True] * 5, [False] * 5)
    assert (result["a_only"], result["b_only"], result["discordant"]) == (5, 0, 5)
    assert result["p"] == pytest.approx(0.0625)

    # b = 1, c = 9: 2 * (C(10,0) + C(10,1)) / 2^10 = 22 / 1024.
    a = [True] + [False] * 9
    b = [False] + [True] * 9
    assert analysis.mcnemar(a, b)["p"] == pytest.approx(22 / 1024)


def test_mcnemar_symmetric_and_empty_discordance():
    """Equal discordant counts are no evidence (p = 1); no discordant pair is no evidence either."""
    assert analysis.mcnemar([True, True, True, False, False, False], [False, False, False, True, True, True])[
        "p"
    ] == pytest.approx(1.0)
    result = analysis.mcnemar([True, True], [True, True])
    assert (result["discordant"], result["p"]) == (0, 1.0)
    # The both-success and both-failure cells carry no information: only discordant pairs do.
    assert analysis.mcnemar([True, False], [False, True])["p"] == pytest.approx(1.0)


# ---------------------------------------------------------------- Wilcoxon


def test_wilcoxon_on_a_known_ordering():
    """Four positive differences: the exact two-sided p is 2 * 1 / 2^4 = 0.125."""
    result = analysis.wilcoxon(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["n"] == 4 and result["p"] == pytest.approx(0.125)

    # Mixed signs: differences -3, -1, +2 give ranks 3, 1, 2, so W+ = 2 and W- = 4; the tail
    # P(W+ <= 2) is the three subsets {} {1} {2} of 2^3, so p = 2 * 3 / 8 = 0.75.
    result = analysis.wilcoxon([0.0, 0.0, 0.0], [-3.0, -1.0, 2.0])
    assert (result["w_plus"], result["w_minus"]) == (2.0, 4.0)
    assert result["p"] == pytest.approx(0.75)


def test_wilcoxon_drops_zero_differences_and_missing_pairs():
    """A zero difference is not a rank; a pair with a missing value is not a pair."""
    result = analysis.wilcoxon([1.0, 2.0], [1.0, 3.0])
    assert (result["pairs"], result["n"]) == (2, 1)
    assert result["p"] == pytest.approx(1.0)

    result = analysis.wilcoxon([0.0, 0.0, np.nan], [1.0, 2.0, 3.0])
    assert (result["pairs"], result["n"]) == (2, 2)
    assert result["p"] == pytest.approx(0.5)

    result = analysis.wilcoxon([1.0, 2.0], [1.0, 2.0])
    assert result["n"] == 0 and np.isnan(result["p"])


def test_wilcoxon_uses_the_normal_approximation_beyond_twenty_differences():
    """21 strictly positive differences: a finite, small p from the corrected normal tail."""
    result = analysis.wilcoxon(np.zeros(21), np.arange(1.0, 22.0))
    assert result["n"] == 21
    assert 0.0 < result["p"] < 0.01


# ---------------------------------------------------------------- the records and the tables


def test_analyse_groups_by_run_and_arm_and_reports_n(tmp_path):
    """The per-arm table: one row per (run, arm), every row naming its n and its Wilson interval."""
    run1 = _write_run(
        tmp_path / "run1",
        [
            _record(run="run1", arm="frs", pair=0, success=True),
            _record(run="run1", arm="frs", pair=1, success=False),
            _record(run="run1", arm="frs", pair=2, success=True),
            _record(run="run1", arm="mars-f", pair=0, success=True, authority=0.8),
        ],
    )
    result = analysis.analyse([run1])

    assert result["n_trials"] == 4
    rows = {(row["run"], row["arm"]): row for row in result["per_arm"]}
    frs = rows[("run1", "frs")]
    assert (frs["n"], frs["successes"]) == (3, 2)
    assert frs["success_rate"] == pytest.approx(2 / 3)
    assert (frs["wilson_low"], frs["wilson_high"]) == pytest.approx(analysis.wilson(2, 3))
    assert frs["n_authority"] == 3 and frs["mean_authority"] == pytest.approx(0.5)
    assert frs["authority_se"] == pytest.approx(0.0)
    mars_f = rows[("run1", "mars-f")]
    assert (mars_f["n"], mars_f["n_authority"], mars_f["mean_authority"]) == (1, 1, 0.8)
    assert mars_f["authority_se"] is None  # one trial has no standard error


def test_paired_tests_use_only_matched_pair_and_scene_trials(tmp_path):
    """Matching is on (pair, scene_hash): an unmatched trial is excluded and counted, never paired."""
    run = _write_run(
        tmp_path / "run1",
        [
            _record(arm="frs", pair=0, success=True, authority=0.1, scene_hash="h0"),
            _record(arm="frs", pair=1, success=True, authority=0.2, scene_hash="h1"),
            _record(arm="frs", pair=2, success=True, authority=0.3, scene_hash="h2"),
            _record(arm="frs", pair=3, success=False, scene_hash="h3"),  # no mars-f counterpart
            _record(arm="mars-f", pair=0, success=False, authority=0.2, scene_hash="h0"),
            _record(arm="mars-f", pair=1, success=False, authority=0.3, scene_hash="h1"),
            _record(arm="mars-f", pair=2, success=False, authority=0.4, scene_hash="h2"),
            _record(arm="mars-f", pair=3, success=True, scene_hash="other"),  # same pair, other scene
        ],
    )

    (row,) = analysis.analyse([run])["paired"]

    assert (row["a"], row["b"]) == ("frs", "mars-f")
    assert row["n_matched"] == 3
    assert row["unmatched"] == 2  # frs pair 3 and mars-f pair 3: different scenes
    # Three discordant pairs, all frs-only: p = 2 * 1 / 2^3 = 0.25.
    assert (row["a_only"], row["b_only"], row["discordant"]) == (3, 0, 3)
    assert row["mcnemar_p"] == pytest.approx(0.25)
    # Authority rises by 0.1 in every matched pair: all-positive ranks, p = 2 / 2^3 = 0.25.
    assert row["n_authority_pairs"] == 3
    assert row["authority_diff"] == pytest.approx(0.1)
    assert row["wilcoxon_p"] == pytest.approx(0.25)


def test_every_pair_of_arms_is_compared(tmp_path):
    """Three arms make three comparisons, on the trials each pair shares."""
    run = _write_run(
        tmp_path / "run1",
        [
            _record(arm=arm, pair=0, success=True, authority=value, scene_hash="h0")
            for arm, value in (("frs", 0.1), ("mars-f", 0.2), ("mars", 0.3))
        ],
    )
    rows = analysis.analyse([run])["paired"]
    assert [(row["a"], row["b"]) for row in rows] == [("frs", "mars-f"), ("frs", "mars"), ("mars-f", "mars")]
    assert all(row["n_matched"] == 1 for row in rows)


def test_a_parent_directory_expands_to_its_runs(tmp_path):
    """``--run`` may name a directory of runs: every run under it is read, once."""
    _write_run(tmp_path / "study" / "run1", [_record(run="run1")])
    _write_run(tmp_path / "study" / "run2", [_record(run="run2")])

    result = analysis.analyse([tmp_path / "study"])

    assert result["run_dirs"] == [str(tmp_path / "study" / "run1"), str(tmp_path / "study" / "run2")]
    assert result["n_trials"] == 2
    assert [row["run"] for row in result["per_arm"]] == ["run1", "run2"]


def test_a_missing_run_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no such run directory"):
        analysis.analyse([tmp_path / "absent"])


def test_a_duplicate_pairing_key_is_refused(tmp_path):
    """Two trials of one arm from the same round and scene cannot be paired unambiguously."""
    run = _write_run(
        tmp_path / "run1",
        [_record(arm="frs", pair=0, scene_hash="h0"), _record(arm="frs", pair=0, scene_hash="h0")],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        analysis.analyse([run])


# ---------------------------------------------------------------- refusing foreign records


def test_a_legacy_record_field_is_refused_by_name(tmp_path):
    """A record the writer did not produce is an error naming the field, not a silently dropped column."""
    legacy = _record()
    legacy["mean_|r|"] = 0.3  # DESIGN.md's earlier draft's name for mean_r
    run = _write_run(tmp_path / "run1", [legacy])

    with pytest.raises(ValueError, match="unknown record field 'mean_\\|r\\|'"):
        analysis.analyse([run])


def test_a_record_missing_a_field_is_refused_by_name(tmp_path):
    incomplete = _record()
    del incomplete["pair"]
    run = _write_run(tmp_path / "run1", [incomplete])

    with pytest.raises(ValueError, match="missing record field 'pair'"):
        analysis.analyse([run])


def test_a_non_boolean_success_is_refused(tmp_path):
    run = _write_run(tmp_path / "run1", [_record(success=1)])
    with pytest.raises(ValueError, match="'success' must be true or false"):
        analysis.analyse([run])


def test_a_non_json_line_is_refused_with_its_line_number(tmp_path):
    directory = tmp_path / "run1"
    directory.mkdir()
    (directory / "trials.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"trials\.jsonl:1: not a JSON record"):
        analysis.analyse([directory])


# ---------------------------------------------------------------- the command and the CSV


def _two_arm_run(tmp_path: Path) -> Path:
    return _write_run(
        tmp_path / "run1",
        [
            _record(arm="frs", pair=0, success=True, authority=0.1, scene_hash="h0"),
            _record(arm="frs", pair=1, success=False, authority=0.2, scene_hash="h1"),
            _record(arm="mars-f", pair=0, success=True, authority=0.3, scene_hash="h0"),
            _record(arm="mars-f", pair=1, success=True, authority=0.4, scene_hash="h1"),
        ],
    )


def test_analyse_prints_the_tables_and_writes_both_csvs(tmp_path, capsys):
    """The command prints the two tables and writes ``per_arm.csv`` and ``paired.csv`` under ``--out``."""
    run = _two_arm_run(tmp_path)
    output = tmp_path / "analysis"

    assert analysis.main(["--run", str(run), "--out", str(output)]) == 0

    printed = capsys.readouterr().out
    assert "per arm" in printed and "paired tests" in printed
    assert f"csv: {output / 'per_arm.csv'}, {output / 'paired.csv'}" in printed

    with (output / "per_arm.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == list(analysis.PER_ARM_FIELDS)
    assert [(row["arm"], row["n"], row["successes"]) for row in rows] == [
        ("frs", "2", "1"),
        ("mars-f", "2", "2"),
    ]

    with (output / "paired.csv").open(newline="", encoding="utf-8") as file:
        paired = list(csv.DictReader(file))
    assert list(paired[0]) == list(analysis.PAIRED_FIELDS)
    (row,) = paired
    assert (row["run"], row["a"], row["b"], row["n_matched"], row["unmatched"]) == (
        "run1",
        "frs",
        "mars-f",
        "2",
        "0",
    )
    assert float(row["mcnemar_p"]) == pytest.approx(1.0)  # one frs-only, one mars-f-only: symmetric
    assert float(row["authority_diff"]) == pytest.approx(0.2)


def test_the_default_csv_directory_is_beside_the_runs(tmp_path, capsys):
    """Without ``--out`` the CSVs land in the directory holding the runs, not in the caller's cwd."""
    _write_run(tmp_path / "study" / "run1", [_record(run="run1", arm="frs")])

    assert analysis.main(["--run", str(tmp_path / "study")]) == 0

    assert (tmp_path / "study" / "per_arm.csv").is_file()
    assert (tmp_path / "study" / "paired.csv").is_file()
    assert "csv:" in capsys.readouterr().out


def test_a_run_with_no_trials_still_writes_the_headers(tmp_path):
    """An empty block (aborted before its first outcome) analyses to no rows, not an error."""
    directory = tmp_path / "run1"
    directory.mkdir()
    (directory / "trials.jsonl").write_text("", encoding="utf-8")

    result = analysis.analyse([directory])

    assert (result["n_trials"], result["per_arm"], result["paired"]) == (0, [], [])
