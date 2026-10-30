"""Chunk references, the expert_chunk agent, and a tiny sweep end to end."""

import json

import numpy as np
import pytest
import torch

from cursor_sa import config
from cursor_sa.agents import ExpertChunkAgent
from cursor_sa.channel import InputChannel
from cursor_sa.env import TaskConfig, center_out, curl_steady_state, sample_layouts
from cursor_sa.policy import FlowPolicy
from cursor_sa.session import load_checkpoints, run_trials
from cursor_sa.steering import MAX_SPEED, steered_chunk
from run_sweep import cell_groups, cell_key, depths_for, load_sweep, main as sweep_main, method_arm, with_depth


def _obs(B=3, N=5, hist=4):
    g = torch.Generator().manual_seed(0)
    goal = torch.zeros(B, N, dtype=torch.bool)
    goal[torch.arange(B), torch.randint(N, (B,), generator=g)] = True
    return dict(pos=torch.rand(B, 2, generator=g), hist=0.01 * torch.randn(B, hist, 2, generator=g),
                targets=torch.rand(B, N, 2, generator=g), mask=torch.ones(B, N, dtype=torch.bool), goal=goal)


@pytest.mark.parametrize("mode", ["flow_control", "frs"])
def test_constant_reference_equals_the_command(mode):
    torch.manual_seed(0)
    model, obs = FlowPolicy().eval(), _obs()
    command = torch.tensor([[0.8, 0.0], [0.0, -0.5], [0.3, 0.3]])
    torch.manual_seed(1)
    plain = steered_chunk(model, obs, command, mode)
    torch.manual_seed(1)
    ref = command[:, None].expand(-1, model.horizon, -1)
    torch.testing.assert_close(steered_chunk(model, obs, command, mode, reference=ref), plain)
    torch.manual_seed(1)
    other = ref.clone()
    other[:, 2:] *= -1  # a reference that is not constant steers differently (from step 2: inside flow_control's tau)
    assert not torch.allclose(steered_chunk(model, obs, command, mode, reference=other), plain)


def test_expert_chunk_agent_plans_the_goal_given_chunk_and_oracle_hits():
    arm = config.load_arm("teleop.yaml")
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    policy = policies[arm["model"]]
    layouts = sample_layouts(np.random.default_rng(3), 6, (2, 4), TaskConfig())
    agent = ExpertChunkAgent()
    recs = run_trials(arm, layouts, agent, InputChannel(None, 0.0), policy, adapters, "cpu")  # the oracle
    assert all(r["outcome"] == "hit" for r in recs)
    assert all(np.isfinite(r["raw_commands"]).all() and abs(np.array(r["raw_commands"])).max() < 4 for r in recs)
    frs = config.load_arm("frs.yaml")
    recs = run_trials(frs, layouts, agent, InputChannel(None, 0.0), policy, adapters, "cpu")
    assert all(r["outcome"] in ("hit", "wrong_target", "timeout") for r in recs)
    assert all(r["steered_chunks"] > 0 for r in recs)


def test_center_out_and_curl_steady_state():
    lay = center_out(8)
    assert len(lay) == 8 and sorted(lay.goal) == list(range(8)) and np.allclose(lay.start, 0.5)
    s = curl_steady_state(0.1, 10.0) @ np.array([1.0, 0.0])  # 45 degree clockwise turn at tau*b = 1
    assert np.isclose(np.degrees(np.arctan2(s[1], s[0])), -45) and np.isclose(np.linalg.norm(s), 1 / np.sqrt(2))


def test_sweep_configs_load_and_methods_resolve():
    for name in ("gain", "rotation", "curl", "study_condition", "learned_adapter_rot90", "learned_adapter_gain025y",
                 "task_space", "task_space_refine", "grid", "grid_expert_check", "grid_task_check", "flow_control_gain"):
        sweep = load_sweep(name + ".yaml")
        assert sweep["focus"] in sweep["methods"] and all(m in sweep["methods"] for m in sweep["against"])
        level = sweep["grid"]["levels"][0]
        for m in sweep["methods"]:
            arm = method_arm(sweep, m, level)
            assert arm["mode"] in ("teleop", "policy", "flow_control", "frs")
            if arm["mode"] in ("flow_control", "frs"):
                depths_for(sweep, arm["mode"])  # every mode a sweep plays has its depths
    curl = load_sweep("curl.yaml")
    arm = method_arm(curl, "mars_fixed", 20)
    np.testing.assert_allclose(arm["adapter_matrix"] @ curl_steady_state(0.1, 20.0), np.eye(2), atol=1e-9)
    assert arm["curl"] == 20 and arm["velocity_tau"] == 0.1
    assert with_depth(method_arm(curl, "frs", 0), 10)["n_reverse_steps"] is None
    assert with_depth(method_arm(curl, "flow_control", 0), 3)["tau"] == 3
    assert depths_for(curl, "frs") == list(range(1, 11)) and depths_for(curl, "flow_control") == list(range(2, 17, 2))


def test_depths_can_be_given_per_mode(tmp_path):
    (tmp_path / "pm.yaml").write_text(
        "name: pm\nmethods: {teleop: {mode: teleop}, flow_control: {mode: flow_control}, frs: {mode: frs}}\n"
        "depths: {flow_control: [4, 8, 16], frs: [2, 10]}\ngrid: {family: gain, levels: [0.25], noises: [1]}\n")
    sweep = load_sweep(tmp_path / "pm.yaml")
    assert depths_for(sweep, "flow_control") == [4, 8, 16] and depths_for(sweep, "frs") == [2, 10]
    (tmp_path / "fc.yaml").write_text(
        "name: fc\nmethods: {flow_control: {mode: flow_control}}\ndepths: {flow_control: [4, 8]}\n"
        "grid: {family: gain, levels: [0.25], noises: [1]}\n")
    with pytest.raises(ValueError, match="no list"):
        depths_for(load_sweep(tmp_path / "fc.yaml"), "frs")
    for bad in ("depths: {teleop: [1]}", "depths: {frs: 3}", "depths: {frs: []}"):
        (tmp_path / "bad.yaml").write_text(f"name: bad\nmethods: {{frs: {{mode: frs}}}}\n{bad}\n"
                                           "grid: {family: gain, levels: [0.25], noises: [1]}\n")
        with pytest.raises(ValueError, match="depths"):
            load_sweep(tmp_path / "bad.yaml")


def test_tiny_sweep_end_to_end(tmp_path):
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\n"
        "  mars: {mode: frs, reversal_adapter: {learned: gain025y_free}}\n"
        "grid: {family: gain, levels: [0.25], noises: [0.5, 1.25]}\nfocus: mars\n")
    sweep_main(["--sweep", str(tmp_path / "tiny.yaml"), "--tune-trials", "4", "--test-trials", "4", "--depths", "2", "10",
                "--results", str(tmp_path), "--out", "tiny"])
    out = tmp_path / "sim" / "sweeps" / "tiny"
    test = json.loads((out / "test.json").read_text())
    tuning = json.loads((out / "tuning.json").read_text())
    assert set(test["test"]) == {"0.25|0.5", "0.25|1.25"}
    cell = test["test"]["0.25|1.25"]
    assert set(cell["metrics"]) == {"teleop", "frs", "mars"} and set(cell["lead"]) == {"teleop", "frs"}
    assert cell["depths"]["teleop"] is None and cell["depths"]["frs"] in (2, 10)
    assert set(tuning["tuning"]["0.25|1.25"]["frs"]) == {"2", "10"} and "None" in tuning["tuning"]["0.25|1.25"]["teleop"]
    assert (out / "depths.yaml").exists() and (out / "figs" / "test.png").exists() and (out / "figs" / "showcase.png").exists()


def test_fixed_depths_skips_tuning(tmp_path):
    """--fixed-depths tests every method at the depths in the file (depths.yaml's format) without tuning: an
    empty tuning.json, depths.yaml a copy of the input, and the same test trials a tuned run at those depths gives."""
    (tmp_path / "fx.yaml").write_text(
        "name: fx\nmethods:\n  frs: {mode: frs}\n  mars: {mode: frs, reversal_adapter: {learned: gain025y_free}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1.0]}\nfocus: mars\n")
    (tmp_path / "depths.yaml").write_text("0.25|1.0:\n  frs: 3\n  mars: 7\n")
    common = ["--sweep", str(tmp_path / "fx.yaml"), "--tune-trials", "4", "--test-trials", "4", "--cpu",
              "--results", str(tmp_path)]
    sweep_main(common + ["--fixed-depths", str(tmp_path / "depths.yaml"), "--out", "fixed"])
    out = tmp_path / "sim" / "sweeps" / "fixed"
    test = json.loads((out / "test.json").read_text())
    assert json.loads((out / "tuning.json").read_text())["tuning"] == {}
    assert (out / "depths.yaml").read_text() == (tmp_path / "depths.yaml").read_text()
    cell = test["test"]["0.25|1.0"]
    assert cell["depths"] == {"frs": 3, "mars": 7} and set(cell["metrics"]) == {"frs", "mars"}

    # the methods really ran at those depths: a tuned run offered only that depth tests the same trials
    for m, d in cell["depths"].items():
        sweep_main(common + ["--depths", str(d), "--out", f"tuned{d}"])
        tuned = json.loads((tmp_path / "sim" / "sweeps" / f"tuned{d}" / "test.json").read_text())["test"]["0.25|1.0"]
        assert tuned["depths"][m] == d and tuned["trials"]["outcome"][m] == cell["trials"]["outcome"][m]
        assert tuned["trials"]["duration_s"][m] == cell["trials"]["duration_s"][m]

    # a cell the file does not give, or a method it leaves out, is an error rather than a guess
    (tmp_path / "short.yaml").write_text("0.25|1.0:\n  frs: 3\n")
    with pytest.raises(ValueError, match="fixed-depths"):
        sweep_main(common + ["--fixed-depths", str(tmp_path / "short.yaml"), "--out", "short"])


def test_sweep_task_axes_make_cells_and_arms(tmp_path):
    (tmp_path / "ax.yaml").write_text(
        "name: ax\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [0.5, 1]\n"
        "  axes: {targets: [[2, 2], [4, 4]], dwell_s: [0.3, 0.5]}\n")
    sweep = load_sweep(tmp_path / "ax.yaml")
    assert sweep["grid"]["noises"] == [0.5, 1.0]
    groups = cell_groups(sweep)  # every (level, task parameters) combination, in grid order
    assert len(groups) == 4 and groups[0] == (0.25, {"targets": [2, 2], "dwell_s": 0.3})
    assert groups[1] == (0.25, {"targets": [2, 2], "dwell_s": 0.5})
    assert cell_key(0.25, 1.0, groups[1][1]) == "0.25|1.0|targets=2-2|dwell_s=0.5"
    arm = method_arm(sweep, "frs", 0.25, groups[3][1])
    assert arm["targets"] == [4, 4] and arm["dwell_s"] == 0.5 and arm["corruption"] == {"scale": [1.0, 0.25]}
    assert load_sweep("study_condition.yaml")["grid"]["axes"] == {}
    assert cell_groups(load_sweep("curl.yaml"))[1] == (5, {}) and cell_key(5, 0.0) == "5|0.0"
    (tmp_path / "bad.yaml").write_text("methods: {teleop: {mode: teleop}}\ngrid: {noises: [1], axes: {tau: [1, 2]}}\n")
    with pytest.raises(ValueError, match="grid.axes"):
        load_sweep(tmp_path / "bad.yaml")


def test_tiny_sweep_with_task_axes_skips_impossible_cells_ranks_and_resumes(tmp_path):
    (tmp_path / "ax.yaml").write_text(
        "name: ax\nmethods:\n  teleop: {mode: teleop}\n"
        "  mars: {mode: frs, reversal_adapter: {learned: gain025y_free}}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [1.25]\n"
        "  axes: {targets: [[2, 2], [3, 3]], min_target_sep: [0.12, 1.5]}\n")  # 1.5 apart does not fit the screen
    argv = ["--sweep", str(tmp_path / "ax.yaml"), "--tune-trials", "3", "--test-trials", "3", "--depths", "2", "10",
            "--results", str(tmp_path), "--out", "ax"]
    sweep_main(argv)
    out = tmp_path / "sim" / "sweeps" / "ax"
    result = json.loads((out / "test.json").read_text())
    keys = {"0.25|1.25|targets=2-2|min_target_sep=0.12", "0.25|1.25|targets=3-3|min_target_sep=0.12"}
    assert set(result["test"]) == keys and result["showcase"] in keys
    assert set(result["skipped"]) == {"0.25|1.25|targets=2-2|min_target_sep=1.5", "0.25|1.25|targets=3-3|min_target_sep=1.5"}
    assert all("cannot place" in why for why in result["skipped"].values())
    cell = result["test"]["0.25|1.25|targets=3-3|min_target_sep=0.12"]
    assert cell["axes"] == {"targets": [3, 3], "min_target_sep": 0.12} and cell["depths"]["mars"] in (2, 10)
    ranking = (out / "ranking.csv").read_text().splitlines()
    assert len(ranking) == 3 and ranking[0].startswith("cell,level,noise,targets,min_target_sep,") and "lead_lo" in ranking[0]
    for fig in ("axes.png", "showcase.png", "depth_curves.png"):
        assert (out / "figs" / fig).exists()

    # --resume keeps the cells already tested and runs only the rest
    done = next(iter(keys))
    result["test"][done]["metrics"]["teleop"]["success"] = 0.123
    result["test"] = {done: result["test"][done]}
    (out / "test.json").write_text(json.dumps(result))
    sweep_main(argv + ["--resume"])
    again = json.loads((out / "test.json").read_text())
    assert set(again["test"]) == keys and again["test"][done]["metrics"]["teleop"]["success"] == 0.123


def test_channel_broadcasts_per_trial_parameters_over_chunks():
    from cursor_sa.channel import matrix_spec
    ms = np.stack([matrix_spec({"scale": [1.0, 0.5]}), matrix_spec({"rotation_deg": 90})])
    chunk = np.tile([[0.4, 0.8]], (2, 16, 1))
    out = InputChannel(ms, np.array([0.0, 0.0]))(chunk)
    np.testing.assert_allclose(out[0], np.tile([[0.4, 0.4]], (16, 1)))
    np.testing.assert_allclose(out[1], np.tile([[-0.8, 0.4]], (16, 1)), atol=1e-12)
    noisy = InputChannel(None, np.array([0.0, 0.2]), np.random.default_rng(0))(np.tile([[0.0, 0.5]], (2, 16, 1)))
    assert np.ptp(noisy[0], axis=0).max() == 0 and 0.1 < noisy[1, :, 0].std() < 0.35


def test_expert_chunk_under_gain_by_time_holds_each_planned_step():
    """Mode "time" at gain 0.5: the chunk agent pushes each planned step for two control steps, gives the
    reference only when a chunk starts, and the controller's history receives the planned steps whole."""
    from cursor_sa.steering import SharedController
    arm = config.load_arm("teleop.yaml")
    policies, _ = load_checkpoints({arm["name"]: arm}, "cpu")
    layouts = sample_layouts(np.random.default_rng(3), 2, (2, 4), TaskConfig())
    ctl = SharedController(policies[arm["model"]], layouts, "teleop", execute=2, speed_gain=0.5, gain_mode="time")
    agent = ExpertChunkAgent()
    agent.reset(2, InputChannel(None, 0.0))
    out = []
    for _ in range(5):
        raw, served, ref = agent.step(ctl.pos, layouts, ctl)
        move = ctl.act(served, ref)
        ctl.record(move)
        out.append((raw, ref, move))
    raws, refs, moves = zip(*out)
    np.testing.assert_allclose(raws[0], raws[1])
    np.testing.assert_allclose(raws[2], raws[3])
    assert not np.allclose(raws[1], raws[2]) and refs[0] is not None and refs[1] is None and refs[4] is not None
    np.testing.assert_allclose(moves[0], 0.5 * raws[0] * ctl.max_speed * ctl.dt)
    np.testing.assert_allclose(ctl.hist[:, -2], raws[0] * ctl.max_speed * ctl.dt, rtol=1e-6)
    np.testing.assert_allclose(ctl.hist[:, -1], raws[2] * ctl.max_speed * ctl.dt, rtol=1e-6)


def test_port_check_wilson_intervals_and_key_mismatches(tmp_path):
    from research.port_check import compare, main as port_check_main, wilson
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-4 and abs(hi - 0.5962) < 1e-4  # the textbook 95% Wilson interval at 50/100
    assert wilson(0, 500)[0] < 1e-12 and wilson(500, 500)[1] > 1 - 1e-12

    def cell(**hits):
        return {"metrics": {m: {"n": 100, "hits": h, "success": h / 100} for m, h in hits.items()},
                "trials": {"outcome": {m: ["hit"] * h + ["timeout"] * (100 - h) for m, h in hits.items()}}}
    old = {"A|1.0": cell(teleop=50, mars=80)}
    rows, summary = compare(old, {"A|1.0": cell(teleop=50, mars=85)})
    assert summary["inside"] == 2 and summary["equal"] == 1 and summary["trials_identical"] == 1 and summary["failed"] == 0
    assert summary["largest_at"] == "A|1.0 mars" and abs(summary["largest_abs_diff"] - 0.05) < 1e-12
    _, summary = compare(old, {"A|1.0": cell(teleop=50, mars=95)})  # 0.95 is outside [0.71, 0.87]
    assert summary["failed"] == 1
    _, summary = compare(old, {"A|1.0": cell(teleop=50), "B|1.0": cell(teleop=50)})  # a missing method, an extra cell
    assert summary["failed"] == 2 and summary["values"] == 3

    for name, data in (("old", old), ("same", old), ("off", {"A|1.0": cell(teleop=50, mars=95)})):
        (tmp_path / f"{name}.json").write_text(json.dumps({"test": data}))
    common = ["--old", str(tmp_path / "old.json"), "--out", str(tmp_path / "pc")]
    assert port_check_main(common + ["--new", str(tmp_path / "same.json")]) == 0
    assert json.loads((tmp_path / "pc" / "port_check.json").read_text())["summary"]["equal"] == 2
    assert port_check_main(common + ["--new", str(tmp_path / "off.json")]) == 1


def test_port_check_fails_when_nothing_is_compared(tmp_path):
    from research.port_check import main as port_check_main
    (tmp_path / "empty.json").write_text(json.dumps({"test": {}}))
    empty = str(tmp_path / "empty.json")
    assert port_check_main(["--old", empty, "--new", empty, "--out", str(tmp_path / "pc")]) == 1
