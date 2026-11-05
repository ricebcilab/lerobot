"""Chunk references, the expert_chunk agent, and a tiny sweep end to end."""

import json
import re

import numpy as np
import pytest
import torch

from _view_stub import ViewStub
from cursor_sa import config
from cursor_sa.agents import ExpertChunkAgent
from cursor_sa.readout import Readout
from cursor_sa.env import TaskConfig, center_out, curl_steady_state, sample_layouts
from cursor_sa.policy import FlowPolicy
from cursor_sa.session import load_checkpoints, run_trials
from cursor_sa.steering import MAX_SPEED, steered_chunk
from lerobot.policies.steering import LearnedTaskModule, save_adapter_module
import run_sweep
from run_sweep import (attach_scene, scene_sources, cell_groups, cell_key, load_sweep, main as sweep_main, method_arm, settings_for, with_module_seed,
                       with_setting)


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
    recs = run_trials(arm, layouts, agent, Readout(None, 0.0), policy, adapters, "cpu")  # the oracle
    assert all(r["outcome"] == "hit" for r in recs)
    assert all(np.isfinite(r["raw_commands"]).all() and abs(np.array(r["raw_commands"])).max() < 4 for r in recs)
    frs = config.load_arm("frs.yaml")
    recs = run_trials(frs, layouts, agent, Readout(None, 0.0), policy, adapters, "cpu")
    assert all(r["outcome"] in ("hit", "wrong_target", "timeout") for r in recs)
    assert all(r["steered_chunks"] > 0 for r in recs)


def test_center_out_and_curl_steady_state():
    lay = center_out(8)
    assert len(lay) == 8 and sorted(lay.goal) == list(range(8)) and np.allclose(lay.start, 0.5)
    s = curl_steady_state(0.1, 10.0) @ np.array([1.0, 0.0])  # 45 degree clockwise turn at tau*b = 1
    assert np.isclose(np.degrees(np.arctan2(s[1], s[0])), -45) and np.isclose(np.linalg.norm(s), 1 / np.sqrt(2))


def test_sweep_configs_load_and_methods_resolve():
    for name in ("study_tuning", "grid", "grid_expert_check", "grid_task_check", "demo_time_check", "demo_time_mars",
                 "demo_time_ft", "grid_ft"):
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

    def ft(cell, seed=None):  # the fine-tuned arm: on T1 the frozen policy, on the obstacle tasks fine-tuned, per seed
        arm = method_arm(load_sweep("grid_ft.yaml", seed), "ft_mars_user", cell)
        return arm["model"], arm["reversal_adapter"]["user"]

    assert ft("U1xT1") == ("flow_goaldrop", "user_gain025y_expert_n100_s0")
    assert ft("U2xT3", 1) == ("flow_goaldrop_wall_gap_ft_n300_s1", "user_gain025x_expert_n100_s1_on_wall_gap_ft_n300")
    for m in ("mars_user", "mars_task", "mars", "retrained_mars_user"):
        assert len(settings_for(grid, "frs", m)) == 10 * 4 and "tune" not in method_arm(grid, m, "U1xT1")
    arm = method_arm(grid, "frs", "U1xT1")
    assert with_setting(arm, {"n_reverse_steps": 10, "pinned_steps": 4})["n_reverse_steps"] is None
    assert with_setting(arm, {"n_reverse_steps": 6, "pinned_steps": 4})["pinned_steps"] == 4
    assert with_setting(method_arm(grid, "flow_control", "U1xT1"), {"tau": 12, "lambda": 2})["lambda"] == 2
    assert with_setting(arm, 10)["n_reverse_steps"] is None  # a bare number: an older run's depth


def test_module_seed_swaps_the_seed_0_tag_of_a_count_or_a_time_budget():
    """--module-seed k: `_n<demos>_s0` and a time budget's `_t<seconds>_s0` become `_s<k>` in every checkpoint name;
    a name without either tag (one policy for every seed) stays."""
    names = {"model": {"T2": "demo_time/flow_goaldrop_disc_large_ft_t300_s0", "T3": "flow_goaldrop_wall_gap"},
             "user": "demo_time/user_gain025y_expert_n100_s0_on_disc_large_ft_t300",
             "task": ["task_disc_large_n300_s0_with_gain025y", "demo_time/task_disc_large_t10800_s0_with_gain025y"]}
    assert with_module_seed(names, 2) == {
        "model": {"T2": "demo_time/flow_goaldrop_disc_large_ft_t300_s2", "T3": "flow_goaldrop_wall_gap"},
        "user": "demo_time/user_gain025y_expert_n100_s2_on_disc_large_ft_t300",
        "task": ["task_disc_large_n300_s2_with_gain025y", "demo_time/task_disc_large_t10800_s2_with_gain025y"]}


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


def test_readout_broadcasts_per_trial_parameters_over_chunks():
    from cursor_sa.readout import matrix_spec
    ms = np.stack([matrix_spec({"scale": [1.0, 0.5]}), matrix_spec({"rotation_deg": 90})])
    chunk = np.tile([[0.4, 0.8]], (2, 16, 1))
    out = Readout(ms, np.array([0.0, 0.0]))(chunk)
    np.testing.assert_allclose(out[0], np.tile([[0.4, 0.4]], (16, 1)))
    np.testing.assert_allclose(out[1], np.tile([[-0.8, 0.4]], (16, 1)), atol=1e-12)
    noisy = Readout(None, np.array([0.0, 0.2]), np.random.default_rng(0))(np.tile([[0.0, 0.5]], (2, 16, 1)))
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
    agent.reset(2, Readout(None, 0.0))
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


def test_the_torch_twins_follow_the_shared_readout():
    """The cursor's torch versions of the readout (training) give the shared numpy rule's numbers for the same draws."""
    import torch
    from cursor_sa.readout import corrupt_chunk, serve_command

    from lerobot.teleoperators import readout

    g = np.random.default_rng(5)
    raw = g.uniform(-1, 1, (6, 2))
    raw[1] = 0.02  # idle
    m = np.array([[1.0, 0.0], [0.0, 0.25]])
    core, sigma, noise = readout.corrupt(raw, m), g.uniform(0, 2, 6), g.normal(size=(6, 2))
    served = serve_command(torch.tensor(core), torch.tensor(raw), torch.tensor(sigma), torch.tensor(noise))
    np.testing.assert_allclose(served.numpy(), readout.add_noise(core, raw, sigma, noise), atol=1e-12)
    chunk = torch.tensor(g.uniform(-1, 1, (6, 4, 2)))
    np.testing.assert_allclose(corrupt_chunk(chunk, torch.tensor(m), torch.zeros(6)).numpy(), readout.corrupt(chunk.numpy(), m),
                               atol=1e-12)




def _image_adapter(tmp_path, monkeypatch, name="image/t_cnn", geometry="absolute"):
    # the policies stay where they are; only the adapters are read from the temp folder
    import cursor_sa.session as session_mod

    monkeypatch.setattr(session_mod, "adapter_path", lambda n: tmp_path / "ckpt" / "adapters" / n / "adapter.pt")
    save_adapter_module(
        LearnedTaskModule(horizon=16, token_source="cnn", image_size=64, token_geometry=geometry),
        tmp_path / "ckpt" / "adapters" / name / "adapter.pt",
        "task",
    )
    return name


def _lay(seed=0, n=3):
    return sample_layouts(np.random.default_rng(seed), n, 4, TaskConfig(min_reach=0.3, max_reach=0.5))


def _files(folder):
    return sorted(p.name for p in folder.iterdir())


def test_scene_cache_is_keyed_by_cell_split_source_size_and_the_digest(tmp_path):
    lay, calls = _lay(), []
    attach_scene(lay, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=lambda: calls.append(1))
    assert lay.scene["cnn"].shape == (3, 3, 64, 64) and calls == []
    (name,) = _files(tmp_path)
    assert re.fullmatch(r"U1xT2\|1\.0_test_cnn_64_[0-9a-f]{12}\.npy", name)  # no temporary file is left behind
    again = _lay()
    attach_scene(again, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)
    assert np.array_equal(again.scene["cnn"], lay.scene["cnn"]) and _files(tmp_path) == [name]
    attach_scene(again, {"cnn": 32}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)
    assert len(list(tmp_path.glob("U1xT2|1.0_test_cnn_32_*.npy"))) == 1


def test_scene_cache_hit_recomputes_nothing(tmp_path, monkeypatch):
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", encoder_factory=None)
    monkeypatch.setattr(run_sweep, "scene_for", lambda *a, **k: pytest.fail("recomputed"))
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", encoder_factory=None)


def test_scene_cache_misses_when_the_render_or_the_draw_or_the_size_changes(tmp_path, monkeypatch):
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", encoder_factory=None)
    assert len(_files(tmp_path)) == 1
    attach_scene(_lay(seed=1), {"cnn": 64}, tmp_path, "k", encoder_factory=None)  # another layout draw
    attach_scene(_lay(), {"cnn": 32}, tmp_path, "k", encoder_factory=None)  # another size
    assert len(_files(tmp_path)) == 3
    obstructed = _lay()
    obstructed.obstacles = np.tile([0.5, 0.5, 0.08], (3, 1, 1))  # the same targets, an obstacle
    obstructed.obstacle_mask = np.ones((3, 1), dtype=bool)
    attach_scene(obstructed, {"cnn": 64}, tmp_path, "k", encoder_factory=None)
    assert len(_files(tmp_path)) == 4
    monkeypatch.setitem(run_sweep.RENDER_PALETTE, "obstacle", (1, 2, 3))  # the dict render_layouts draws with
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", encoder_factory=None)
    assert len(_files(tmp_path)) == 5


def test_the_goal_is_not_in_the_digest_and_the_module_seed_is_not_either():
    lay, other = _lay(), _lay()
    other.goal = (other.goal + 1) % 3
    assert run_sweep.scene_digest(lay, "cnn", 64) == run_sweep.scene_digest(other, "cnn", 64)
    # the digest reads the sweep's layouts and render spec only: one --module-seed's modules read the same files
    import inspect
    assert "seed" not in inspect.signature(run_sweep.scene_digest).parameters


def test_a_vlm_source_digest_depends_on_the_encoder_path(tmp_path):
    class Stub:
        def image_tokens(self, images):
            return np.zeros((len(images), 256, 2048), np.float16)

    for vlm in ("a/one", "a/two", "a/one"):
        attach_scene(_lay(), {"vlm_image": 56}, tmp_path, "k", lambda: Stub(), vlm=vlm)
    assert len(_files(tmp_path)) == 2
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", None, vlm="a/one")
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "k", None, vlm="a/two")  # a cnn source does not read the encoder
    assert len(_files(tmp_path)) == 3


def test_a_vlm_source_makes_the_encoder_once_across_keys_and_matches_the_trainer(tmp_path, monkeypatch):
    from cursor_sa.readout import corruption_spec
    from train import train_adapter

    made, calls = [], []

    class Stub:
        def image_tokens(self, images):
            calls.append(len(images))
            flat = np.asarray(images).reshape(len(images), -1).astype(np.float32)
            return (flat[:, :1, None] + flat[:, 1:2, None] * np.arange(256)[None, :, None] * np.ones((1, 1, 2048))).astype(np.float16)

    encoders = []

    def factory():  # as main's: one encoder for the whole run
        if not encoders:
            made.append(1)
            encoders.append(Stub())
        return encoders[0]

    scenes = {}
    for key, seed in (("a_tune", 0), ("a_test", 1)):
        lay = _lay(seed)
        attach_scene(lay, {"vlm_image": 56}, tmp_path, key, factory)
        scenes[key] = lay.scene["vlm_image"]
    assert made == [1] and calls == [3, 3]
    assert len(list(tmp_path.glob("a_tune_vlm_image_56_*.npy"))) == 1
    assert scenes["a_tune"].dtype == np.float16 and scenes["a_tune"].shape == (3, 256, 2048)
    attach_scene(_lay(0), {"vlm_image": 56}, tmp_path, "a_tune", lambda: pytest.fail("encoder made on a hit"))

    # the trainer's attach (expert_splits -> scene_for) and attach_scene give one array for the same layouts and stub
    seen = []
    real = train_adapter.scene_for
    monkeypatch.setattr(train_adapter, "scene_for", lambda lay, *a: (seen.append((lay, real(lay, *a))), seen[-1][1])[1])
    args = train_adapter.parse_args(["--name", "x", "--corruption", "{rotation_deg: 60}", "--n-demos", "6", "--val-demos", "4",
                                     "--token-source", "vlm_image", "--render-size", "56"])
    train_adapter.expert_splits(args, None, FlowPolicy().eval(), "cpu", np.random.default_rng(0), *corruption_spec(args.corruption),
                                encoder=Stub())
    assert len(seen) == 2
    for lay, trained in seen:
        attach_scene(lay, {"vlm_image": 56}, tmp_path / "trainer", "t", lambda: Stub())
        assert np.array_equal(lay.scene["vlm_image"], trained)


@pytest.mark.parametrize("geometry", ["absolute", "relative"])
def test_tiny_sweep_attaches_the_scene_for_an_image_task_module(tmp_path, monkeypatch, geometry):
    name = _image_adapter(tmp_path, monkeypatch, geometry=geometry)
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_image\nagent: expert\narm: condition_grid.yaml\nmethods:\n  teleop: {mode: teleop}\n"
        f"  mars_cnn: {{mode: frs, reversal_adapter: {{task: {name}}}, tune: {{n_reverse_steps: [2]}}}}\n"
        "grid:\n  users: {U1: {corruption: {scale: [1.0, 0.25]}, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [1.0]\nfocus: mars_cnn\nseed: 3\n"
    )
    sweep_main(
        [
            "--sweep", str(tmp_path / "tiny.yaml"), "--tune-trials", "2", "--test-trials", "2",
            "--results", str(tmp_path), "--out", "tiny_image", "--scene-cache", str(tmp_path / "scene"),
        ]
    )  # fmt: skip
    assert {re.sub(r"_[0-9a-f]{12}\.npy$", "", p.name) for p in (tmp_path / "scene").glob("*.npy")} == {
        "U1xT2|1.0_tune_cnn_64",
        "U1xT2|1.0_test_cnn_64",
    }
    test_json = tmp_path / "sim" / "sweeps" / "tiny_image" / "test.json"
    assert set(json.loads(test_json.read_text())["test"]["U1xT2|1.0"]["metrics"]) == {"teleop", "mars_cnn"}
    sweep_main(
        [
            "--sweep", str(tmp_path / "tiny.yaml"), "--results", str(tmp_path), "--out", "pre",
            "--scene-cache", str(tmp_path / "scene2"), "--precompute-scene",
        ]
    )  # fmt: skip
    assert len(list((tmp_path / "scene2").glob("*.npy"))) == 2
    assert not (tmp_path / "sim" / "sweeps" / "pre" / "test.json").exists()


def test_module_seed_runs_use_the_same_default_scene_folder(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch)
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_seed\nagent: expert\narm: condition_grid.yaml\nmethods:\n  teleop: {mode: teleop}\n"
        f"  mars_cnn: {{mode: frs, reversal_adapter: {{task: {name}}}, tune: {{n_reverse_steps: [2]}}}}\n"
        "grid:\n  users: {U1: {corruption: {scale: [1.0, 0.25]}, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [1.0]\nfocus: mars_cnn\nseed: 3\n"
    )  # the module name has no _n<demos>_s0 tag, so --module-seed renames the sweep and nothing else
    real = run_sweep.cursor_paths.CHECKPOINTS
    (tmp_path / "ck").mkdir()
    for p in real.iterdir():  # the policies are read from the real folder; the scene cache goes to the temp one
        if p.name != "scene":
            (tmp_path / "ck" / p.name).symlink_to(p)
    monkeypatch.setattr(run_sweep.cursor_paths, "CHECKPOINTS", tmp_path / "ck")
    argv = ["--sweep", str(tmp_path / "tiny.yaml"), "--results", str(tmp_path), "--precompute-scene",
            "--tune-trials", "2", "--test-trials", "2"]
    sweep_main(argv + ["--out", "plain"])
    folders = sorted(p for p in (tmp_path / "ck" / "scene").iterdir())
    assert [p.name for p in folders] == ["tiny_seed"]
    files = _files(folders[0])
    assert len(files) == 2
    sweep_main(argv + ["--out", "seeded", "--module-seed", "1"])
    assert sorted(p.name for p in (tmp_path / "ck" / "scene").iterdir()) == ["tiny_seed"]
    assert _files(folders[0]) == files  # seed 1 hit seed 0's files


def test_precompute_scene_writes_nothing_under_out_for_skipped_cells_and_visits_done_ones(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch)
    (tmp_path / "ax.yaml").write_text(
        "name: ax\nagent: expert\narm: condition_grid.yaml\nmethods:\n  teleop: {mode: teleop}\n"
        f"  mars_cnn: {{mode: frs, reversal_adapter: {{task: {name}}}, tune: {{n_reverse_steps: [2]}}}}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [1.25]\n"
        "  axes: {targets: [[2, 2]], min_target_sep: [0.12, 1.5]}\n"  # 1.5 apart does not fit the screen
    )
    argv = ["--sweep", str(tmp_path / "ax.yaml"), "--results", str(tmp_path), "--out", "pre", "--scene-cache",
            str(tmp_path / "scene"), "--precompute-scene", "--tune-trials", "2", "--test-trials", "2", "--resume"]
    out = tmp_path / "sim" / "sweeps" / "pre"
    out.mkdir(parents=True)
    (out / "test.json").write_text(json.dumps({"test": {"0.25|1.25|targets=2-2|min_target_sep=0.12": {}}, "skipped": {}}))
    before = (out / "test.json").read_text()
    sweep_main(argv)  # the cell test.json calls done is visited all the same
    assert {re.sub(r"_[0-9a-f]{12}\.npy$", "", p.name) for p in (tmp_path / "scene").glob("*.npy")} == {
        "0.25|1.25|targets=2-2|min_target_sep=0.12_tune_cnn_64",
        "0.25|1.25|targets=2-2|min_target_sep=0.12_test_cnn_64",
    }
    assert (out / "test.json").read_text() == before
    assert sorted(p.name for p in out.iterdir()) == ["test.json"]


def test_image_sweep_configs_resolve_every_arm_per_cell():
    check = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image.yaml")
    assert check["focus"] == "oracle_task_vlm" and method_arm(check, "oracle_task_vlm", "U0xT3")["learned_adapter"]["task"] == "image/task_wall_gap_n300_s0_with_gain025x_vlm_image"
    assert method_arm(check, "oracle_task_cnn_all", "U0xT5")["learned_adapter"]["task"] == "image/task_all_n1200_s0_cnn"
    grid = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_image.yaml")
    assert set(grid["methods"]) == {"teleop", "mars", "mars_vlm", "mars_vlm_all", "mars_cnn", "mars_cnn_all"} and set(grid["grid"]["tasks"]) == {"T2", "T3", "T4", "T5"}
    arm = method_arm(grid, "mars_vlm", "U2xT4")
    assert arm["learned_adapter"] == {"user": "user_gain025x_expert_n100_s0", "task": "image/task_ring_goal_n300_s0_with_rot60_vlm_image"}
    assert method_arm(grid, "mars", "U2xT4")["learned_adapter"]["task"] == "task_ring_goal_n300_s0_with_rot60"
    seeded = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_image.yaml", module_seed=2)
    assert method_arm(seeded, "mars_vlm_all", "U1xT2")["learned_adapter"]["task"] == "image/task_all_n1200_s2_vlm_image"
    base = load_sweep(config.CONFIG_DIR / "sweeps" / "grid.yaml")
    assert grid["seed"] == base["seed"] and grid["trials"] == base["trials"] and {k for k, c in grid["grid"]["cells"].items() if c["train"]} == {k for k, c in base["grid"]["cells"].items() if c["train"] and c["task"] != "T1"}
    quick = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2_quick.yaml")
    assert set(quick["grid"]["tasks"]) == {"T2", "T4"} and quick["focus"] == "oracle_task_rel" and quick["against"] == ["oracle_task", "oracle_ft"]
    assert set(quick["methods"]) == {"oracle", "oracle_task", "oracle_ft", "oracle_task_rel", "oracle_task_teach_rel", "policy", "policy_task", "policy_task_rel", "policy_task_teach_rel"}
    assert method_arm(quick, "oracle_ft", "U0xT4")["model"] == "flow_goaldrop_ring_goal_ft_n300_s0" and method_arm(quick, "oracle_ft", "U0xT4")["learned_adapter"] is None
    assert method_arm(quick, "oracle_task_rel", "U0xT2")["learned_adapter"]["task"] == "image/task_disc_large_n300_s0_with_gain025y_vlm_image_rel"
    assert method_arm(quick, "oracle_task_teach_rel", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n10000_s0_with_rot60_vlm_image_teacher_rel"
    assert quick["seed"] == check["seed"] and quick["trials"] == check["trials"] and quick["arm"] == check["arm"] and quick["agent"] == check["agent"]
    full = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2.yaml")
    assert set(full["grid"]["tasks"]) == {"T2", "T3", "T4", "T5"} and set(full["methods"]) == {"oracle", "oracle_task", "oracle_ft", "oracle_task_rel", "oracle_task_rel_all", "policy", "policy_task", "policy_task_rel", "policy_task_rel_all"}
    assert method_arm(full, "oracle_ft", "U0xT3")["model"] == "flow_goaldrop_wall_gap_ft_n300_s0"
    assert method_arm(full, "oracle_task_rel", "U0xT5")["learned_adapter"]["task"] == "image/task_ring_start_n300_s0_with_curl20_vlm_image_rel"
    assert method_arm(full, "oracle_task_rel_all", "U0xT3")["learned_adapter"]["task"] == "image/task_all_n1200_s0_vlm_image_rel"


def test_precompute_scene_with_merge_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        sweep_main(["--sweep", "x.yaml", "--results", str(tmp_path), "--precompute-scene", "--merge", "a", "b"])


def test_view_sweep_configs_resolve_every_arm():
    sel = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_views_select.yaml")
    assert sel["seed"] == 821 and set(sel["grid"]["tasks"]) == {"T2", "T4"} and sel["focus"] == "oracle_task_views_base"
    assert set(sel["methods"]) == {"oracle", "oracle_task", "oracle_ft", *(f"oracle_task_views_{r}" for r in ("base", "span1", "nomask", "noaffine"))}
    assert method_arm(sel, "oracle_task_views_span1", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n300_s0_with_rot60_views_span1"
    width = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_views_width.yaml")
    assert width["seed"] == 821 and set(width["grid"]["tasks"]) == {"T4"}
    assert method_arm(width, "oracle_task_views_full", "U0xT4")["learned_adapter"]["task"] == "image/task_ring_goal_n300_s0_with_rot60_views_full"
    quick = load_sweep(config.CONFIG_DIR / "sweeps" / "grid_task_check_image2_quick.yaml")
    for n in (300, 1000, 3000):
        scan = load_sweep(config.CONFIG_DIR / "sweeps" / f"grid_task_check_views_n{n}.yaml")
        assert scan["seed"] == 820 and scan["trials"] == quick["trials"] and scan["arm"] == quick["arm"] and scan["agent"] == quick["agent"]
        assert {"oracle", "oracle_task", "oracle_ft", "policy", "policy_task", f"oracle_task_views_n{n}", f"policy_task_views_n{n}"} <= set(scan["methods"])
        assert method_arm(scan, f"oracle_task_views_n{n}", "U0xT2")["learned_adapter"]["task"] == f"image/task_disc_large_n{n}_s0_with_gain025y_views"
        assert method_arm(scan, f"policy_task_views_n{n}", "U0xT4")["learned_adapter"]["task"] == f"image/task_ring_goal_n{n}_s0_with_rot60_views"
        assert ("oracle_task_teach_rel" in scan["methods"]) == (n == 300)


def _view_sweep(tmp_path, monkeypatch):
    """Two vlm_views task modules at spans 0.5 and 1.0 in the temp adapter folder, and a tiny sweep playing them."""
    import cursor_sa.session as session_mod

    monkeypatch.setattr(session_mod, "adapter_path", lambda n: tmp_path / "ckpt" / "adapters" / n / "adapter.pt")
    for name, span in (("image/v05", 0.5), ("image/v10", 1.0)):
        save_adapter_module(LearnedTaskModule(horizon=16, token_source="vlm_views", scene_dim=2048, scene_tokens=512, image_size=56,
                                              wrist_span=span, scene_rank=2), tmp_path / "ckpt" / "adapters" / name / "adapter.pt", "task")
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_views\nagent: expert_chunk\narm: condition_grid.yaml\nmethods:\n"
        "  oracle_a: {mode: teleop, readout: off, reversal_adapter: {task: image/v05}}\n"
        "  policy_b: {mode: policy, reversal_adapter: {task: image/v10}}\n"
        "tune: {frs: {n_reverse_steps: [10]}}\ngrid:\n  users: {U0: {corruption: null, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [0.0]\nfocus: oracle_a\nseed: 3\n"
    )
    return tmp_path / "tiny.yaml"


def test_tiny_sweep_runs_two_view_modules_at_their_own_spans(tmp_path, monkeypatch):
    import cursor_sa.steering as steering_mod

    sweep = _view_sweep(tmp_path, monkeypatch)
    rendered, real, ViewStub.made = [], steering_mod.render_views, 0

    def spy(layouts, pos, span, size, **k):
        rendered.append((span, size))
        return real(layouts, pos, span, size, **k)

    monkeypatch.setattr(steering_mod, "render_views", spy)
    monkeypatch.setattr(run_sweep, "SceneEncoder", ViewStub)
    args = ["--sweep", str(sweep), "--results", str(tmp_path), "--scene-cache", str(tmp_path / "scene")]
    sweep_main([*args, "--out", "pre", "--precompute-scene"])
    assert ViewStub.made == 0 and not rendered  # nothing to precompute for per-step views
    sweep_main([*args, "--out", "tiny_views", "--tune-trials", "2", "--test-trials", "3"])
    assert {span for span, _ in rendered} == {0.5, 1.0} and {size for _, size in rendered} == {56} and ViewStub.made == 1
    assert not list((tmp_path / "scene").glob("*.npy"))
    test_json = tmp_path / "sim" / "sweeps" / "tiny_views" / "test.json"
    assert set(json.loads(test_json.read_text())["test"]["U0xT2|0.0"]["metrics"]) == {"oracle_a", "policy_b"}


def test_run_trials_refuses_a_view_module_without_an_encoder(tmp_path, monkeypatch):
    arm = method_arm(load_sweep(_view_sweep(tmp_path, monkeypatch)), "policy_b", "U0xT2")
    policies, adapters = load_checkpoints({"policy_b": arm}, "cpu")
    with pytest.raises(ValueError, match="scene_encoder"):
        run_trials(arm, _lay(), ExpertChunkAgent(), Readout(None, 0.0), policies[arm["model"]], adapters, "cpu")
