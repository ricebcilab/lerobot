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
from run_sweep import cell_groups, cell_key, load_sweep, main as sweep_main, method_arm, settings_for, with_setting


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
    for name in ("study_tuning", "grid", "grid_expert_check", "grid_task_check"):
        sweep = load_sweep(name + ".yaml")
        assert sweep["focus"] in sweep["methods"] and all(m in sweep["methods"] for m in sweep["against"])
        level = sweep["grid"]["levels"][0]
        for m in sweep["methods"]:
            arm = method_arm(sweep, m, level)
            assert arm["mode"] in ("teleop", "policy", "flow_control", "frs")
            if arm["mode"] in ("flow_control", "frs"):
                assert settings_for(sweep, arm["mode"])  # every mode a sweep plays has its settings
    grid = load_sweep("grid.yaml")  # the baselines with their papers' fixed knob (lambda 1, the full reversal); MARS both
    assert settings_for(grid, "flow_control", "flow_control") == [{"tau": t} for t in range(2, 17, 2)]
    assert settings_for(grid, "frs", "frs") == [{"pinned_steps": p} for p in (2, 4, 8, 16)]
    assert method_arm(grid, "frs", "U1xT1")["n_reverse_steps"] is None and method_arm(grid, "flow_control", "U1xT1")["lambda"] == 1
    for m in ("mars_user", "mars_task", "mars", "retrained_mars_user"):
        assert len(settings_for(grid, "frs", m)) == 10 * 4 and "tune" not in method_arm(grid, m, "U1xT1")
    arm = method_arm(grid, "frs", "U1xT1")
    assert with_setting(arm, {"n_reverse_steps": 10, "pinned_steps": 4})["n_reverse_steps"] is None
    assert with_setting(arm, {"n_reverse_steps": 6, "pinned_steps": 4})["pinned_steps"] == 4
    assert with_setting(method_arm(grid, "flow_control", "U1xT1"), {"tau": 12, "lambda": 2})["lambda"] == 2
    assert with_setting(arm, 10)["n_reverse_steps"] is None  # a bare number: an older run's depth


def test_tune_lists_each_modes_knobs(tmp_path):
    (tmp_path / "pm.yaml").write_text(
        "name: pm\nmethods: {teleop: {mode: teleop}, flow_control: {mode: flow_control}, frs: {mode: frs}}\n"
        "tune: {flow_control: {tau: [4, 8, 16]}, frs: {n_reverse_steps: [2, 10], pinned_steps: [4, 16]}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1]}\n")
    sweep = load_sweep(tmp_path / "pm.yaml")
    assert settings_for(sweep, "flow_control") == [{"tau": 4}, {"tau": 8}, {"tau": 16}]
    assert len(settings_for(sweep, "frs")) == 4
    (tmp_path / "fc.yaml").write_text(
        "name: fc\nmethods: {flow_control: {mode: flow_control}}\ntune: {flow_control: {tau: [4, 8]}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1]}\n")
    with pytest.raises(ValueError, match="no knobs"):
        settings_for(load_sweep(tmp_path / "fc.yaml"), "frs")
    (tmp_path / "own.yaml").write_text(  # a method's own tune replaces its mode's
        "name: own\nmethods: {frs: {mode: frs}, mars: {mode: frs, tune: {n_reverse_steps: [5, 10], pinned_steps: [8]}}}\n"
        "tune: {frs: {pinned_steps: [4, 16]}}\ngrid: {family: gain, levels: [0.25], noises: [1]}\n")
    own = load_sweep(tmp_path / "own.yaml")
    assert settings_for(own, "frs", "frs") == [{"pinned_steps": 4}, {"pinned_steps": 16}]
    assert settings_for(own, "frs", "mars") == [{"n_reverse_steps": 5, "pinned_steps": 8}, {"n_reverse_steps": 10, "pinned_steps": 8}]
    (tmp_path / "bad.yaml").write_text("name: bad\nmethods: {mars: {mode: frs, tune: {tau: [4]}}}\n"
                                       "grid: {family: gain, levels: [0.25], noises: [1]}\n")
    with pytest.raises(ValueError, match="methods.mars.tune"):
        load_sweep(tmp_path / "bad.yaml")
    for bad in ("tune: {teleop: {tau: [1]}}", "tune: {frs: {n_reverse_steps: 3}}", "tune: {frs: {n_reverse_steps: []}}",
                "tune: {frs: {tau: [4]}}", "tune: {frs: {}}"):
        (tmp_path / "bad.yaml").write_text(f"name: bad\nmethods: {{frs: {{mode: frs}}}}\n{bad}\n"
                                           "grid: {family: gain, levels: [0.25], noises: [1]}\n")
        with pytest.raises(ValueError, match="tune"):
            load_sweep(tmp_path / "bad.yaml")


def test_tiny_sweep_end_to_end(tmp_path):
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\n"
        "  mars: {mode: frs, reversal_adapter: {learned: learned_adapter/gain025y_free}}\n"
        "tune: {frs: {n_reverse_steps: [2, 10]}}\ngrid: {family: gain, levels: [0.25], noises: [0.5, 1.25]}\nfocus: mars\n")
    sweep_main(["--sweep", str(tmp_path / "tiny.yaml"), "--tune-trials", "4", "--test-trials", "4",
                "--results", str(tmp_path), "--out", "tiny"])
    out = tmp_path / "sim" / "sweeps" / "tiny"
    test = json.loads((out / "test.json").read_text())
    tuning = json.loads((out / "tuning.json").read_text())
    assert set(test["test"]) == {"0.25|0.5", "0.25|1.25"}
    cell = test["test"]["0.25|1.25"]
    assert set(cell["metrics"]) == {"teleop", "frs", "mars"} and set(cell["lead"]) == {"teleop", "frs"}
    assert cell["settings"]["teleop"] is None and cell["settings"]["frs"]["n_reverse_steps"] in (2, 10)
    assert set(tuning["tuning"]["0.25|1.25"]["frs"]) == {"n_reverse_steps=2", "n_reverse_steps=10"}
    assert "None" in tuning["tuning"]["0.25|1.25"]["teleop"]
    assert (out / "settings.yaml").exists() and (out / "figs" / "test.png").exists() and (out / "figs" / "showcase.png").exists()


def test_settings_are_tuned_jointly_and_replayed_as_fixed_settings(tmp_path):
    """`tune` tunes every combination of a mode's knobs (flow control's tau x lambda, frs's n_reverse_steps x
    pinned_steps), records the best as {knob: value}, and --fixed-settings replays it on the same test trials."""
    (tmp_path / "jt.yaml").write_text(
        "name: jt\nmethods:\n  flow_control: {mode: flow_control}\n  frs: {mode: frs}\n"
        "tune: {flow_control: {tau: [8, 16], lambda: [1, 2]}, frs: {n_reverse_steps: [4, 10], pinned_steps: [2, 16]}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1.0]}\nfocus: frs\n")
    common = ["--sweep", str(tmp_path / "jt.yaml"), "--tune-trials", "4", "--test-trials", "4", "--cpu", "--results", str(tmp_path)]
    sweep_main(common + ["--out", "joint"])
    out = tmp_path / "sim" / "sweeps" / "joint"
    tuning = json.loads((out / "tuning.json").read_text())["tuning"]["0.25|1.0"]
    assert set(tuning["flow_control"]) == {"tau=8,lambda=1", "tau=8,lambda=2", "tau=16,lambda=1", "tau=16,lambda=2"}
    assert set(tuning["frs"]) == {f"n_reverse_steps={n},pinned_steps={p}" for n in (4, 10) for p in (2, 16)}
    cell = json.loads((out / "test.json").read_text())["test"]["0.25|1.0"]
    assert set(cell["settings"]["flow_control"]) == {"tau", "lambda"}
    assert set(cell["settings"]["frs"]) == {"n_reverse_steps", "pinned_steps"}
    sweep_main(common + ["--fixed-settings", str(out / "settings.yaml"), "--out", "replay"])
    replay = json.loads((tmp_path / "sim" / "sweeps" / "replay" / "test.json").read_text())["test"]["0.25|1.0"]
    assert replay["settings"] == cell["settings"] and replay["trials"]["outcome"] == cell["trials"]["outcome"]


def test_fixed_settings_skips_tuning(tmp_path):
    """--fixed-settings tests every method at the settings in the file (settings.yaml's format) without tuning: an
    empty tuning.json, settings.yaml a copy of the input, and the same test trials a tuned run at those settings gives."""
    (tmp_path / "fx.yaml").write_text(
        "name: fx\nmethods:\n  frs: {mode: frs}\n  mars: {mode: frs, reversal_adapter: {learned: learned_adapter/gain025y_free}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1.0]}\nfocus: mars\n")
    (tmp_path / "settings.yaml").write_text("0.25|1.0:\n  frs: {n_reverse_steps: 3}\n  mars: {n_reverse_steps: 7}\n")
    common = ["--tune-trials", "4", "--test-trials", "4", "--cpu", "--results", str(tmp_path)]
    sweep_main(["--sweep", str(tmp_path / "fx.yaml")] + common + ["--fixed-settings", str(tmp_path / "settings.yaml"), "--out", "fixed"])
    out = tmp_path / "sim" / "sweeps" / "fixed"
    test = json.loads((out / "test.json").read_text())
    assert json.loads((out / "tuning.json").read_text())["tuning"] == {}
    assert (out / "settings.yaml").read_text() == (tmp_path / "settings.yaml").read_text()
    cell = test["test"]["0.25|1.0"]
    assert cell["settings"] == {"frs": {"n_reverse_steps": 3}, "mars": {"n_reverse_steps": 7}}

    # the methods really ran at those settings: a tuned run offered only that setting tests the same trials
    for m, d in cell["settings"].items():
        n = d["n_reverse_steps"]
        (tmp_path / f"t{n}.yaml").write_text((tmp_path / "fx.yaml").read_text() + f"tune: {{frs: {{n_reverse_steps: [{n}]}}}}\n")
        sweep_main(["--sweep", str(tmp_path / f"t{n}.yaml")] + common + ["--out", f"tuned{n}"])
        tuned = json.loads((tmp_path / "sim" / "sweeps" / f"tuned{n}" / "test.json").read_text())["test"]["0.25|1.0"]
        assert tuned["settings"][m] == d and tuned["trials"]["outcome"][m] == cell["trials"]["outcome"][m]
        assert tuned["trials"]["duration_s"][m] == cell["trials"]["duration_s"][m]

    # a cell the file does not give, or a method it leaves out, is an error rather than a guess
    (tmp_path / "short.yaml").write_text("0.25|1.0:\n  frs: {n_reverse_steps: 3}\n")
    with pytest.raises(ValueError, match="fixed-settings"):
        sweep_main(["--sweep", str(tmp_path / "fx.yaml")] + common + ["--fixed-settings", str(tmp_path / "short.yaml"), "--out", "short"])


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
    assert load_sweep("study_tuning.yaml")["grid"]["axes"] == {} and cell_key(5, 0.0) == "5|0.0"
    (tmp_path / "bad.yaml").write_text("methods: {teleop: {mode: teleop}}\ngrid: {noises: [1], axes: {tau: [1, 2]}}\n")
    with pytest.raises(ValueError, match="grid.axes"):
        load_sweep(tmp_path / "bad.yaml")


def test_tiny_sweep_with_task_axes_skips_impossible_cells_ranks_and_resumes(tmp_path):
    (tmp_path / "ax.yaml").write_text(
        "name: ax\nmethods:\n  teleop: {mode: teleop}\n"
        "  mars: {mode: frs, reversal_adapter: {learned: learned_adapter/gain025y_free}}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [1.25]\n"
        "  axes: {targets: [[2, 2], [3, 3]], min_target_sep: [0.12, 1.5]}\n"  # 1.5 apart does not fit the screen
        "tune: {frs: {n_reverse_steps: [2, 10]}}\n")
    argv = ["--sweep", str(tmp_path / "ax.yaml"), "--tune-trials", "3", "--test-trials", "3",
            "--results", str(tmp_path), "--out", "ax"]
    sweep_main(argv)
    out = tmp_path / "sim" / "sweeps" / "ax"
    result = json.loads((out / "test.json").read_text())
    keys = {"0.25|1.25|targets=2-2|min_target_sep=0.12", "0.25|1.25|targets=3-3|min_target_sep=0.12"}
    assert set(result["test"]) == keys and result["showcase"] in keys
    assert set(result["skipped"]) == {"0.25|1.25|targets=2-2|min_target_sep=1.5", "0.25|1.25|targets=3-3|min_target_sep=1.5"}
    assert all("cannot place" in why for why in result["skipped"].values())
    cell = result["test"]["0.25|1.25|targets=3-3|min_target_sep=0.12"]
    assert cell["axes"] == {"targets": [3, 3], "min_target_sep": 0.12} and cell["settings"]["mars"]["n_reverse_steps"] in (2, 10)
    ranking = (out / "ranking.csv").read_text().splitlines()
    assert len(ranking) == 3 and ranking[0].startswith("cell,level,noise,targets,min_target_sep,") and "lead_lo" in ranking[0]
    for fig in ("axes.png", "showcase.png", "tuning_curves.png"):
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
