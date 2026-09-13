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
    """A run directory with a config, trial records and minimal step arrays.

    A record's `arrays` entry (not written to the JSON) adds or overrides step arrays,
    e.g. `eef_pos` or `object_pos`, for the trajectory metrics.
    """
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"schedule": [0, 1, 0], **config}))
    with open(run / "trials.jsonl", "w") as f:
        for i, r in enumerate(records):
            r = dict(r)
            arrays = r.pop("arrays", {})
            record = {
                "trial": i,
                "task_id": task_of(i),
                "task_description": f"task {task_of(i)}",
                "pair_id": f"task-{task_of(i)}-state-{i}",
                "initial_state_hash": f"state-{task_of(i)}-{i}",
                "steps": 100 + i,
                "duration_s": 5.0,
                "steps_file": f"trial_{i:03d}.npz",
                "mode_expressed": None,
                **r,
            }
            f.write(json.dumps(record) + "\n")
            raw = np.zeros((3, 3), dtype=np.float32)
            raw[1] = [0.5, 0.0, 0.0]
            action = np.zeros((3, 7), dtype=np.float32)
            action[:, 0] = 1.0  # executed motion follows the raw intent
            defaults = {
                "user_translation_raw": raw,
                "user_translation": raw * 0.9,
                "user_gripper": np.array([-1.0, -1.0, 1.0]),
                "user_reads": np.array([1, 0, 1]),
                "action": action,
                "eef_pos": np.zeros((3, 3), dtype=np.float32),
            }
            np.savez(run / f"trial_{i:03d}.npz", **{**defaults, **arrays})
    return run


def path(*points) -> dict:
    """Step arrays for an end-effector that visits `points` while a bowl tracks it and a knob turns."""
    eef = np.asarray(points, dtype=np.float32)
    return {
        "eef_pos": eef,
        "object_names": np.array(["bowl", "stove"]),
        "object_pos": np.stack([eef + [0.0, 0.0, -0.1], np.zeros_like(eef)], axis=1),
        "articulation_names": np.array(["stove_knob"]),
        "articulation_qpos": np.linspace(0.0, 1.0, len(eef), dtype=np.float32)[:, None],
    }


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


def test_operator_profiles_cannot_be_silently_pooled(tmp_path):
    a = write_run(tmp_path, "immediate", config(operator="policy"), outcomes(True))
    b = write_run(
        tmp_path, "delayed", config(operator="policy", policy_operator={"delay_steps": 4}), outcomes(False)
    )
    with pytest.raises(ValueError, match="operator timing profiles"):
        analyze.load_runs([a, b])
    c = write_run(
        tmp_path,
        "explicit_default",
        config(operator="policy", policy_operator={"update_every_steps": 10, "delay_steps": 0}),
        outcomes(True),
    )
    df, _ = analyze.load_runs([a, c])
    assert df.operator_profile.nunique() == 1


def test_method_and_depth():
    c = config()
    assert (analyze.method_of(c), analyze.depth_of(c)) == ("FRS", 10.0)  # null = full reversal
    c = config(n_reversal_steps=4, corruption_matrix=ROT, reversal_adapter_matrix=np.eye(7).tolist())
    assert (analyze.method_of(c), analyze.depth_of(c)) == ("FRS-RA", 4.0)
    c = config("shared_flow_control", n_guided_steps=6)
    assert (analyze.method_of(c), analyze.depth_of(c)) == ("FC", 6.0)
    assert analyze.method_of(config("policy")) == "policy (floor)"
    assert analyze.method_of(config("policy", prompt="task")) == "policy (ceiling)"
    assert np.isnan(analyze.depth_of(config("policy")))


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
    assert (row["method"], row["depth"], row["operator"]) == ("FRS", 10.0, "policy")
    assert trials["mode_expressed"].isna().all()


def test_wilson_bounds():
    low, high = analyze.wilson(2, 3)
    assert 0 <= low < 2 / 3 < high <= 1
    assert all(np.isnan(v) for v in analyze.wilson(0, 0))


def test_summary_table(tmp_path):
    a = write_run(tmp_path, "a", config("shared_flow_control"), outcomes(True, False, True))
    write_run(tmp_path, "b", config(corruption_matrix=ROT), outcomes(True, True, True))
    empty = tmp_path / "c_empty"
    empty.mkdir()
    (empty / "config.yaml").write_text(yaml.safe_dump(config("shared_flow_control")))
    (empty / "trials.jsonl").write_text("")
    trials, configs = analyze.load_runs(analyze.find_runs(tmp_path))
    assert set(configs) == {a.name, "b", "c_empty"}

    by_run = analyze.summary_table(trials, ["run"]).set_index("run")
    assert by_run.loc[a.name, "value"] == pytest.approx(2 / 3) and by_run.loc["b", "value"] == 1.0
    assert "c_empty" not in by_run.index  # no trials, no group

    by_method = analyze.summary_table(trials, ["method", "depth"])
    assert list(by_method["method"]) == ["FC", "FRS"]  # METHODS order regardless of appearance
    assert by_method.iloc[0]["value"] == pytest.approx(2 / 3)
    assert 0 < by_method.iloc[0]["ci95_low"] < 2 / 3 < by_method.iloc[0]["ci95_high"] < 1


def test_best_depth(tmp_path):
    write_run(tmp_path, "frs4", config(n_reversal_steps=4), outcomes(True, False, True, True))
    write_run(tmp_path, "frs10", config(), outcomes(False, False, True, False))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    best = analyze.best_depth(trials)
    assert len(best) == 1 and best.iloc[0]["method"] == "FRS" and best.iloc[0]["depth"] == 4.0


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

    by_method = analyze.paired_test(trials, "method", "FRS", "FRS-RA", ["task_id", "trial"])
    assert by_method["paired_trials"] == 6 and by_method["B_only"] == 2 and by_method["A_only"] == 0
    assert by_method["A_rate"] == pytest.approx(2 / 6) and by_method["B_rate"] == pytest.approx(4 / 6)
    assert by_method["mcnemar_p"] == pytest.approx(0.5)  # 2 discordant pairs, both in B's favour


# ---------------------------------------------------------------- expressed behaviour


def test_paired_test_rejects_missing_mismatched_and_duplicate_pairs(tmp_path):
    write_run(tmp_path, "a", config(), outcomes(True))
    write_run(tmp_path, "b", config(reversal_adapter_matrix=np.eye(7).tolist()), outcomes(False))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    with pytest.raises(ValueError, match="Unverified pairing"):
        analyze.paired_test(trials.drop(columns="pair_id"), "method", "FRS", "FRS-RA")
    bad = trials.copy()
    bad.loc[bad["method"] == "FRS-RA", "initial_state_hash"] = "different"
    with pytest.raises(ValueError, match="Unmatched initial_state_hash"):
        analyze.paired_test(bad, "method", "FRS", "FRS-RA")
    with pytest.raises(ValueError, match="Duplicate pairing"):
        analyze.paired_test(analyze.pd.concat([trials, trials]), "method", "FRS", "FRS-RA")


def test_task_targets_prefer_the_ceiling_anchor(tmp_path):
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
    # Without an anchor, the target is what the successful trials moved most.
    steered = trials[trials["method"] != "policy (ceiling)"]
    assert analyze.task_targets(steered) == {0: "middle"}
    assert analyze.task_targets(steered[~steered["success"]]) == {0: None}


# ---------------------------------------------------------------- step arrays and plots


def test_trial_metrics_read_engagement_and_intent(tmp_path):
    write_run(tmp_path, "r", config(corruption_matrix=ROT), outcomes(True))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    out = analyze.trial_metrics(trials).iloc[0]
    assert out["n_steps"] == 3 and out["commanding_frac"] == pytest.approx(1 / 3)
    assert out["reads_this_trial"] == 2 and out["gripper_closed_frac"] == pytest.approx(1 / 3)
    assert out["mean_speed_when_active"] == pytest.approx(0.5)
    assert out["mean_corruption_shift"] == pytest.approx(0.05)
    assert out["cos_raw"] == pytest.approx(1.0) and out["cos_served"] == pytest.approx(1.0)


def test_trial_metrics_path_time_and_progress(tmp_path):
    # Ceiling: two successes whose bowl ends near (1, 0, -0.1) define the goal set.
    write_run(
        tmp_path,
        "c_ceiling",
        config("policy", prompt="task"),
        [
            {"success": True, "mode_expressed": "bowl", "arrays": path([0, 0, 0], [0.5, 0, 0], [1, 0, 0])},
            {
                "success": True,
                "mode_expressed": "bowl",
                "arrays": path([0, 0, 0], [0.5, 0.5, 0], [1, 0.1, 0]),
            },
        ],
        task_of=lambda i: 0,
    )
    # Steered: a detour that succeeds, and a failure that gets halfway then backs off.
    write_run(
        tmp_path,
        "a_frs4",
        config(n_reversal_steps=4),
        [
            {"success": True, "mode_expressed": "bowl", "arrays": path([0, 0, 0], [0.5, 1, 0], [1, 0, 0])},
            {
                "success": False,
                "mode_expressed": "bowl",
                "arrays": path([0, 0, 0], [0.5, 0, 0], [0.25, 0, 0]),
            },
        ],
        task_of=lambda i: 0,
    )
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))

    goals = analyze.task_goals(trials)
    assert goals[0]["element"] == "bowl" and goals[0]["kind"] == "object"
    assert goals[0]["positions"].shape == (2, 3)

    out = analyze.trial_metrics(trials).set_index(["run", "trial"])
    detour, backoff = out.loc[("a_frs4", 0)], out.loc[("a_frs4", 1)]
    assert detour["path_length"] == pytest.approx(2 * np.hypot(0.5, 1.0))
    assert detour["displacement"] == pytest.approx(1.0)
    assert detour["path_efficiency"] == pytest.approx(1.0 / (2 * np.hypot(0.5, 1.0)))
    assert detour["time_to_success"] == 100 and np.isnan(backoff["time_to_success"])
    assert detour["progress_final"] == pytest.approx(1.0) and detour["progress_max"] == pytest.approx(1.0)
    assert backoff["progress_max"] == pytest.approx(0.5) and backoff["progress_final"] == pytest.approx(0.25)
    assert backoff["goal_distance_min"] == pytest.approx(0.5)
    # Ceiling trials are scored too, and all reach their own goal set.
    assert out.loc["c_ceiling", "progress_final"].tolist() == pytest.approx([1.0, 1.0])


def test_trial_metrics_articulation_goal_and_missing_scene(tmp_path):
    write_run(
        tmp_path,
        "c_ceiling",
        config("policy", prompt="task"),
        [{"success": True, "mode_expressed": "stove_knob", "arrays": path([0, 0, 0], [0, 0, 0])}],
        task_of=lambda i: 0,
    )
    write_run(
        tmp_path,
        "a_frs4",
        config(n_reversal_steps=4),
        [
            {
                "success": False,
                "mode_expressed": "stove_knob",
                "arrays": path([0, 0, 0], [0, 0, 0], [0, 0, 0]),
            },
            {"success": False, "mode_expressed": None},  # no scene arrays, degenerate path
        ],
        task_of=lambda i: 0,
    )
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    assert analyze.task_goals(trials)[0]["kind"] == "articulation"
    out = analyze.trial_metrics(trials).set_index(["run", "trial"])
    # The knob goes 0 -> 1 in the ceiling run; the steered trial's knob also reaches 1 (linspace).
    assert out.loc[("a_frs4", 0), "progress_final"] == pytest.approx(1.0)
    bare = out.loc[("a_frs4", 1)]
    assert bare["path_length"] == 0 and np.isnan(bare["path_efficiency"]) and np.isnan(bare["progress_max"])
    assert np.isnan(bare["time_to_success"])


def test_success_over_time_and_continuous_summary(tmp_path):
    write_run(tmp_path, "frs4", config(n_reversal_steps=4), outcomes(True, False, True), task_of=lambda i: 0)
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    # steps are 100, 101, 102 by the fixture; trials 0 and 2 succeed.
    curve = analyze.success_over_time(trials, ["method"], horizon=103)
    frs = curve[curve["method"] == "FRS"].set_index("step")["fraction"]
    assert frs.loc[99] == 0 and frs.loc[100] == pytest.approx(1 / 3) and frs.loc[102] == pytest.approx(2 / 3)
    assert frs.loc[103] == pytest.approx(2 / 3)

    table = analyze.summary_table(trials.assign(x=[1.0, 3.0, np.nan]), ["method"], "x")
    row = table.iloc[0]
    assert row["n"] == 2 and row["value"] == pytest.approx(2.0)
    assert row["ci95_low"] < 2.0 < row["ci95_high"]


def test_best_vs_best_selects_on_the_data_and_pairs_by_reset(tmp_path):
    for name, cfg, flags in [
        ("fc4", config("shared_flow_control", n_guided_steps=4), (False, False, True)),
        ("fc8", config("shared_flow_control", n_guided_steps=8), (True, False, True)),
        ("frs1", config(n_reversal_steps=1), (True, True, True)),
        ("frs4", config(n_reversal_steps=4), (False, True, False)),
    ]:
        write_run(tmp_path, name, cfg, outcomes(*flags), task_of=lambda i: 0)
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    best, comparisons, selected = analyze.best_vs_best(trials)
    assert best.set_index("method")["depth"].to_dict() == {"FC": 8.0, "FRS": 1.0}
    assert set(selected["run"]) == {"fc8", "frs1"}
    row = comparisons.iloc[0]
    assert (row["A"], row["B"], row["paired_trials"]) == ("FC", "FRS", 3)
    assert row["A_rate"] == pytest.approx(2 / 3) and row["B_rate"] == pytest.approx(1.0)
    assert row["B_only"] == 1 and row["A_only"] == 0


def test_plot_metric_handles_binary_and_continuous_metrics_on_either_axis(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_run(tmp_path, "fc8", config("shared_flow_control", n_guided_steps=8), outcomes(True, False))
    write_run(tmp_path, "frs2", config(n_reversal_steps=2), outcomes(True, True))
    write_run(tmp_path, "ceiling", config("policy", prompt="task"), outcomes(True, True))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))

    fig, ax = plt.subplots()
    measured = trials.assign(user_authority=[np.nan, np.nan, 0.5, 0.7, 0.2, 0.4])  # runs load alphabetically
    table = analyze.plot_metric(measured, ax)  # success vs measured user authority by default
    assert list(table.columns[:4]) == ["method", "depth", "n", "value"]
    assert table.set_index("method")["user_authority"].to_dict() == pytest.approx(
        {"FC": 0.6, "FRS": 0.3, "policy (ceiling)": np.nan}, nan_ok=True
    )
    assert ax.get_xlabel() == "User authority" and ax.get_ylabel() == "Success rate"
    assert len(ax.lines) >= 3  # two method lines plus the ceiling band
    assert any(c.has_xerr for c in ax.containers if hasattr(c, "has_xerr"))  # horizontal error bars
    plt.close(fig)

    fig, ax = plt.subplots()
    table = analyze.plot_metric(trials.assign(x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ax, "x", x="depth")
    assert list(table["depth"].dropna()) == [8.0, 2.0] and list(table["value"]) == [3.5, 5.5, 1.5]
    assert ax.get_ylabel() == "X" and ax.get_ylim() != pytest.approx((-0.02, 1.02))
    plt.close(fig)


def test_plot_best_vs_best_draws_a_bar_per_method_and_a_bracket_per_pair(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, cfg, flags in [
        ("fc8", config("shared_flow_control", n_guided_steps=8), (True, False, True)),
        ("frs1", config(n_reversal_steps=1), (True, True, True)),
        ("ra2", config(n_reversal_steps=2, reversal_adapter_matrix=np.eye(7).tolist()), (False, True, True)),
    ]:
        write_run(tmp_path, name, cfg, outcomes(*flags), task_of=lambda i: 0)
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    best, comparisons, selected = analyze.best_vs_best(trials)
    fig, ax = plt.subplots()
    table, drawn = analyze.plot_best_vs_best(selected, ax)
    assert list(table["method"]) == ["FC", "FRS", "FRS-RA"]
    assert len(ax.patches) == 3 and len(comparisons) == 3 and drawn["p"].tolist() == comparisons["p"].tolist()
    assert [p.get_facecolor()[:3] for p in ax.patches] == [
        matplotlib.colors.to_rgb(analyze.METHOD_COLORS[m]) for m in ("FC", "FRS", "FRS-RA")
    ]
    assert sum(t.get_text() in ("n.s.", "*", "**", "***") for t in ax.texts) == 3  # one mark per pair
    assert analyze.significance_stars(0.03) == "*" and analyze.significance_stars(np.nan) == "n.s."
    assert [t.get_text() for t in ax.get_xticklabels()][0] == "FC\ndepth 8"
    plt.close(fig)


# ---------------------------------------------------------------- authority per step


def test_step_authority_is_a_clipped_barycentric_coordinate_and_symmetric():
    p = np.array([[0.0, 0.0, 0.0]] * 5)
    r = np.array([[1.0, 0.0, 0.0]] * 5)
    a = np.array(
        [
            [0.0, 0.0, 0.0],  # the policy's plan: user 0
            [1.0, 0.0, 0.0],  # the command: user 1
            [0.3, 0.5, 0.0],  # 30% of the way, with motion that follows neither: 0.3
            [1.4, 0.0, 0.0],  # overshoot clips to 1
            [-0.5, 0.0, 0.0],  # moved away clips to 0
        ]
    )
    user = analyze.step_authority(a, p, r)
    assert user.tolist() == pytest.approx([0.0, 1.0, 0.3, 1.0, 0.0])
    policy = analyze.step_authority(a, r, p)  # swap the roles: the complement, exactly
    assert (user + policy).tolist() == pytest.approx([1.0] * 5)
    # Nothing to decide when plan and command coincide.
    assert np.isnan(analyze.step_authority(a[:1], p[:1], p[:1])).all()


def test_chunk_authority_pools_a_chunk_and_skips_idle_ones():
    a = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.9, 0.0, 0.0]])
    p = np.zeros((4, 3))
    r = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pushing = np.array([True, True, False, True])
    # chunks of two steps: the first pools its two pushing steps (0.5), the second has one (0.9)
    assert analyze.chunk_authority(a, p, r, pushing, 2).tolist() == pytest.approx([0.5, 0.9])
    assert np.isnan(analyze.chunk_authority(a, p, r, np.zeros(4, bool), 2)).all()
    swapped = analyze.chunk_authority(a, r, p, pushing, 2)
    assert (analyze.chunk_authority(a, p, r, pushing, 2) + swapped).tolist() == pytest.approx([1.0, 1.0])


def test_trial_metrics_measure_authority_from_the_policy_plan(tmp_path):
    raw = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    action = np.zeros((3, 7), dtype=np.float32)
    action[0, 0] = 0.45  # served command is 0.9 * raw = 0.45: executed it fully
    action[1, 0] = 0.2625  # plan 0.2 -> command 0.45: a quarter of the way
    action[2, 0] = 0.3  # idle operator: executed the plan
    plan = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
    arrays = {
        "user_translation_raw": raw,
        "user_translation": raw * 0.9,
        "action": action,
        "policy_translation": plan,
    }
    write_run(tmp_path, "r", config(), [{"success": True, "arrays": arrays}], task_of=lambda i: 0)
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    out = analyze.trial_metrics(trials).iloc[0]
    # One chunk (n_action_steps = 10 > 3 steps) pooling the two pushing steps:
    # <(0.45, 0.0625), (0.45, 0.25)> / |(0.45, 0.25)|^2
    assert out["user_authority"] == pytest.approx((0.45 * 0.45 + 0.0625 * 0.25) / (0.45**2 + 0.25**2))
    assert out["user_authority_all_steps"] == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    assert out["policy_authority"] == pytest.approx(1 - out["user_authority"])

    write_run(tmp_path, "legacy", config(), outcomes(True), task_of=lambda i: 0)  # no policy_translation
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    legacy = analyze.trial_metrics(trials).set_index("run").loc["legacy"]
    assert np.isnan(legacy["user_authority"])


def test_plot_metric_places_cells_at_the_mean_of_a_measured_x(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_run(tmp_path, "fc4", config("shared_flow_control", n_guided_steps=4), outcomes(True, False))
    write_run(tmp_path, "fc8", config("shared_flow_control", n_guided_steps=8), outcomes(True, True))
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    fig, ax = plt.subplots()
    table = analyze.plot_metric(
        trials.assign(user_authority=[0.1, 0.3, 0.8, 1.0]), ax, "success", x="user_authority"
    )
    assert table.set_index("depth")["user_authority"].to_dict() == pytest.approx({4.0: 0.2, 8.0: 0.9})
    assert ax.get_xlabel() == "User authority"
    plt.close(fig)


def test_paired_test_uses_wilcoxon_for_a_continuous_metric_and_head_to_head_draws_it(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_run(
        tmp_path,
        "fc8",
        config("shared_flow_control", n_guided_steps=8),
        outcomes(*[True] * 6),
        task_of=lambda i: 0,
    )
    write_run(tmp_path, "frs1", config(n_reversal_steps=1), outcomes(*[True] * 6), task_of=lambda i: 0)
    trials, _ = analyze.load_runs(analyze.find_runs(tmp_path))
    trials["path_length"] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.5, 0.6, 0.7, 0.8, 0.9, np.nan]
    result = analyze.paired_test(trials, "method", "FC", "FRS", metric="path_length")
    assert result["paired_trials"] == 5 and result["mean_difference"] == pytest.approx(-0.5)
    assert 0 < result["wilcoxon_p"] == result["p"] < 0.1
    fig, ax = plt.subplots()
    table, comparisons = analyze.plot_best_vs_best(trials, ax, "path_length")
    assert ax.get_ylabel() == "End-effector path length (m)" and len(ax.patches) == 2
    assert comparisons.iloc[0]["p"] == pytest.approx(result["p"])
    plt.close(fig)
