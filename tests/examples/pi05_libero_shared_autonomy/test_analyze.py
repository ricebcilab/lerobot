import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("scipy")
NOTEBOOKS = Path(__file__).resolve().parents[3] / "examples/pi05/libero_shared_autonomy/notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import analyze  # noqa: E402


def write_run(root: Path, name: str, config: dict, records: list[dict], task_of=lambda i: i % 2) -> Path:
    """A run directory with a config, trial records and minimal step arrays."""
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"schedule": [0, 1, 0], **config}))
    with open(run / "trials.jsonl", "w") as f:
        for i, r in enumerate(records):
            record = {
                "trial": i,
                "task_id": task_of(i),
                "task_description": f"task {task_of(i)}",
                "steps": 100 + i,
                "duration_s": 5.0,
                "steps_file": f"trial_{i:03d}.npz",
                **r,
            }
            f.write(json.dumps(record) + "\n")
            raw = np.zeros((3, 3), dtype=np.float32)
            raw[1] = [0.5, 0.0, 0.0]
            action = np.zeros((3, 7), dtype=np.float32)
            action[:, 0] = 1.0  # executed motion follows the raw intent
            np.savez(
                run / f"trial_{i:03d}.npz",
                user_translation_raw=raw,
                user_translation=raw * 0.9,
                user_gripper=np.array([-1.0, -1.0, 1.0]),
                user_reads=np.array([1, 0, 1]),
                action=action,
            )
    return run


def config(mode="shared_flow_reversal_steering", **kw) -> dict:
    return {
        "mode": mode,
        "n_guided_steps": 8,
        "n_reversal_steps": None,
        "prompt": "do something",
        "corruption_matrix": None,
        "reversal_adapter_matrix": None,
        **kw,
    }


ROT = [[0.94, -0.34, 0.0], [0.34, 0.94, 0.0], [0.0, 0.0, 1.0]]


def outcomes(*flags):
    return [{"success": s} for s in flags]


# ---------------------------------------------------------------- loading and describing


def test_method_depth_and_label():
    c = config()
    assert (analyze.method_of(c), analyze.depth_of(c), analyze.label_for(c)) == ("FRS", 10.0, "FRS full")
    c = config(n_reversal_steps=4, corruption_matrix=ROT, reversal_adapter_matrix=np.eye(7).tolist())
    assert (analyze.method_of(c), analyze.depth_of(c)) == ("FRS+F", 4.0)
    assert analyze.label_for(c) == "FRS n_reversal_steps=4+M+F"
    c = config("shared_flow_control", n_guided_steps=6)
    assert (analyze.method_of(c), analyze.depth_of(c), analyze.label_for(c)) == (
        "FC",
        6.0,
        "FC n_guided_steps=6",
    )
    assert analyze.method_of(config("policy")) == "policy (floor)"
    assert analyze.method_of(config("policy", prompt="task")) == "policy (ceiling)"
    assert np.isnan(analyze.depth_of(config("policy")))
    assert analyze.label_for(config("policy")) == "policy"


def test_find_and_load_runs(tmp_path):
    a = write_run(tmp_path, "20260101_000000_fc", config("shared_flow_control"), outcomes(True, False, True))
    b = write_run(
        tmp_path,
        "20260101_000001_frs",
        config(corruption_matrix=ROT, operator="policy"),
        outcomes(True, True, True),
    )
    (tmp_path / "20260101_000002_aborted").mkdir()
    assert analyze.find_runs(tmp_path) == [a, b]

    trials, configs = analyze.load_runs(analyze.find_runs(tmp_path))
    assert len(trials) == 6 and set(configs) == {a.name, b.name}
    row = trials[trials["run"] == b.name].iloc[0]
    assert (row["method"], row["depth"], row["operator"], row["corrupted"], row["adapted"]) == (
        "FRS",
        10.0,
        "policy",
        True,
        False,
    )
    assert trials["mode_expressed"].isna().all()  # absent in the records -> None column
    assert "rotation about z" in analyze.describe_rotation(np.array(ROT))
    assert analyze.describe_rotation(np.eye(3)) == "identity (no-op)"


def test_differing_settings_compares_nested_specs_by_json_form():
    configs = {
        "a": {"mode": "shared_flow_control", "n_guided_steps": 8, "corruption": None, "schedule": [0, 1, 2]},
        "b": {
            "mode": "shared_flow_control",
            "n_guided_steps": 5,
            "corruption": {"rotation_z_deg": 20},
            "schedule": [2, 1, 0],
        },
    }
    result = analyze.differing_settings(configs, ["mode", "corruption"])
    assert list(result.index) == ["corruption", "schedule"]


# ---------------------------------------------------------------- rates


def test_wilson_bounds():
    low, high = analyze.wilson(2, 3)
    assert 0 <= low < 2 / 3 < high <= 1
    assert all(np.isnan(v) for v in analyze.wilson(0, 0))


def test_rate_table_and_success_table(tmp_path):
    a = write_run(tmp_path, "a", config("shared_flow_control"), outcomes(True, False, True))
    write_run(tmp_path, "b", config(corruption_matrix=ROT), outcomes(True, True, True))
    empty = tmp_path / "c_empty"
    empty.mkdir()
    (empty / "config.yaml").write_text(yaml.safe_dump(config("shared_flow_control")))
    (empty / "trials.jsonl").write_text("")
    trials, configs = analyze.load_runs(analyze.find_runs(tmp_path))

    by_run = analyze.rate_table(trials, ["run"]).set_index("run")
    assert by_run.loc[a.name, "successes"] == 2 and by_run.loc["b", "rate"] == 1.0
    assert "c_empty" not in by_run.index  # no trials, no group

    by_method = analyze.rate_table(trials, ["method", "depth"])
    assert list(by_method["method"]) == ["FC", "FRS"]  # METHODS order regardless of appearance
    assert by_method.iloc[0]["rate"] == pytest.approx(2 / 3)

    table = analyze.success_table(trials, configs)
    assert list(table.index) == [a.name, "b", "c_empty"]
    assert table.loc[a.name, "successes"] == 2 and table.loc["b", "success_rate"] == 1.0
    assert table.loc["c_empty", "trials"] == 0 and np.isnan(table.loc["c_empty", "success_rate"])
    assert table.loc[a.name, "median_steps_success"] == 101.0
    assert table.loc["b", "label"] == "FRS full+M" and bool(table.loc["b", "corrupted"])


def test_per_task_and_best_depth(tmp_path):
    write_run(tmp_path, "frs4", config(n_reversal_steps=4), outcomes(True, False, True, True))
    write_run(tmp_path, "frs10", config(), outcomes(False, False, True, False))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    best = analyze.best_depth(trials)
    assert len(best) == 1 and best.iloc[0]["method"] == "FRS" and best.iloc[0]["depth"] == 4.0
    per_task = analyze.per_task_table(trials)
    assert list(per_task.index) == ["0: task 0", "1: task 1"]
    assert per_task.loc["0: task 0", ("FRS", 4.0)] == 1.0
    assert per_task.loc["1: task 1", ("FRS", 10.0)] == 0.0


# ---------------------------------------------------------------- paired tests


def test_paired_test_by_run_and_by_method(tmp_path):
    a = write_run(
        tmp_path, "a_frs", config(n_reversal_steps=4), outcomes(True, False, False), task_of=lambda i: 0
    )
    b = write_run(
        tmp_path,
        "b_adapted",
        config(n_reversal_steps=4, reversal_adapter_matrix=np.eye(7).tolist()),
        outcomes(True, True, False),
        task_of=lambda i: 0,
    )
    # the same two methods on a second task: pairs pool over tasks when pairing on (task_id, trial)
    write_run(
        tmp_path, "c_frs_t1", config(n_reversal_steps=4), outcomes(True, False, False), task_of=lambda i: 1
    )
    write_run(
        tmp_path,
        "d_adapted_t1",
        config(n_reversal_steps=4, reversal_adapter_matrix=np.eye(7).tolist()),
        outcomes(True, True, False),
        task_of=lambda i: 1,
    )
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))

    by_run = analyze.paired_test(trials, "run", a.name, b.name)
    assert by_run["paired_trials"] == 3 and by_run["A_only"] == 0 and by_run["B_only"] == 1
    assert 0 < by_run["mcnemar_p"] <= 1
    assert len(analyze.paired_comparisons(trials[trials["run"].isin([a.name, b.name])])) == 1

    by_method = analyze.paired_test(trials, "method", "FRS", "FRS+F", ["task_id", "trial"])
    assert by_method["paired_trials"] == 6 and by_method["B_only"] == 2 and by_method["A_only"] == 0
    assert by_method["A_rate"] == pytest.approx(2 / 6) and by_method["B_rate"] == pytest.approx(4 / 6)
    assert by_method["mcnemar_p"] == pytest.approx(0.5)  # 2 discordant pairs, both in B's favour


# ---------------------------------------------------------------- expressed behaviour


def test_mode_accuracy_from_the_ceiling_anchor(tmp_path):
    def rec(ok, moved):
        return {"success": ok, "mode_expressed": moved}

    write_run(
        tmp_path,
        "a_frs4",
        config(n_reversal_steps=4),
        [rec(True, "middle"), rec(False, "top")],
        task_of=lambda i: 0,
    )
    write_run(tmp_path, "b_frs10", config(), [rec(False, "top"), rec(False, None)], task_of=lambda i: 0)
    write_run(
        tmp_path,
        "c_ceiling",
        config("policy", prompt="task"),
        [rec(True, "middle"), rec(True, "middle")],
        task_of=lambda i: 0,
    )
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    assert analyze.task_targets(trials) == {0: "middle"}
    tagged = analyze.with_mode_accuracy(trials)
    assert tagged.groupby("run")["on_target"].mean().to_dict() == {
        "a_frs4": 0.5,
        "b_frs10": 0.0,
        "c_ceiling": 1.0,
    }
    assert analyze.rate_table(tagged, ["method", "depth"], "on_target").set_index("method").loc[
        "FRS", "rate"
    ].tolist() == [0.5, 0.0]


# ---------------------------------------------------------------- step arrays and plots


def test_step_metrics_reads_engagement_and_intent(tmp_path):
    write_run(tmp_path, "r", config(corruption_matrix=ROT), outcomes(True))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    out = analyze.step_metrics(trials).iloc[0]
    assert out["n_steps"] == 3 and out["commanding_frac"] == pytest.approx(1 / 3)
    assert out["reads_this_trial"] == 2 and out["gripper_closed_frac"] == pytest.approx(1 / 3)
    assert out["mean_speed_when_active"] == pytest.approx(0.5)
    assert out["mean_corruption_shift"] == pytest.approx(0.05)
    assert out["cos_raw"] == pytest.approx(1.0) and out["cos_served"] == pytest.approx(1.0)


def test_plot_performance_vs_depth(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_run(tmp_path, "frs4", config(n_reversal_steps=4), outcomes(True, False))
    write_run(tmp_path, "ceiling", config("policy", prompt="task"), outcomes(True, True))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    fig, ax = plt.subplots()
    curve = analyze.plot_performance_vs_depth(trials, ax)
    assert len(ax.lines) >= 2  # the FRS line plus the ceiling band
    assert list(curve["method"]) == ["FRS", "policy (ceiling)"]
    plt.close(fig)
