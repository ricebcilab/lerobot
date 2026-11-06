"""The sweeps (run_sweep.py): chunk references, the expert_chunk agent, the shipped sweep files, tuning and fixed
settings, task axes, the scene cache, and tiny sweeps end to end."""

import json
import re

import numpy as np
import pytest
import torch

from _view_stub import ViewStub
from cursor_sa import config
from cursor_sa.agents import ExpertChunkAgent
from cursor_sa.readout import Readout
from cursor_sa.env import TaskConfig, sample_layouts
from cursor_sa.policy import FlowPolicy
from cursor_sa.session import load_checkpoints, run_trials
from cursor_sa.steering import SharedController, steered_chunk
from lerobot.policies.steering import LearnedTaskModule, save_adapter_module
import run_sweep
from run_sweep import (attach_scene, cell_groups, cell_key, load_sweep, main as sweep_main, method_arm, settings_for, with_module_seed,
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


def test_the_expert_chunk_agent_plans_goal_given_chunks_and_holds_each_planned_step():
    arm = config.load_arm("teleop.yaml")
    policies, adapters = load_checkpoints({arm["name"]: arm}, "cpu")
    policy = policies[arm["model"]]
    layouts = sample_layouts(np.random.default_rng(3), 6, (2, 4), TaskConfig())
    recs = run_trials(arm, layouts, ExpertChunkAgent(), Readout(None, 0.0), policy, adapters, "cpu")  # the oracle
    assert all(r["outcome"] == "hit" and np.isfinite(r["raw_commands"]).all() and abs(np.array(r["raw_commands"])).max() < 4 for r in recs)
    recs = run_trials(config.load_arm("frs.yaml"), layouts, ExpertChunkAgent(), Readout(None, 0.0), policy, adapters, "cpu")
    assert all(r["outcome"] in ("hit", "wrong_target", "timeout") and r["steered_chunks"] > 0 for r in recs)
    # gain 0.5 by time: each planned step pushed for two control steps, the reference given only when a chunk starts,
    # the controller's history receiving the planned steps whole
    ctl = SharedController(policy, layouts[:2], "teleop", execute=2, speed_gain=0.5, gain_mode="time")
    agent = ExpertChunkAgent()
    agent.reset(2, Readout(None, 0.0))
    raws, refs, moves = [], [], []
    for _ in range(5):
        raw, served, ref = agent.step(ctl.pos, layouts[:2], ctl)
        moves.append(ctl.act(served, ref))
        ctl.record(moves[-1])
        raws.append(raw)
        refs.append(ref)
    np.testing.assert_allclose(raws[0], raws[1])
    np.testing.assert_allclose(raws[2], raws[3])
    assert not np.allclose(raws[1], raws[2]) and refs[0] is not None and refs[1] is None and refs[4] is not None
    np.testing.assert_allclose(moves[0], 0.5 * raws[0] * ctl.max_speed * ctl.dt)
    np.testing.assert_allclose(ctl.hist[:, -2], raws[0] * ctl.max_speed * ctl.dt, rtol=1e-6)
    np.testing.assert_allclose(ctl.hist[:, -1], raws[2] * ctl.max_speed * ctl.dt, rtol=1e-6)


def test_every_shipped_sweep_resolves_every_method():
    for path in sorted((config.CONFIG_DIR / "sweeps").glob("*.yaml")):
        sweep = load_sweep(path)
        assert sweep.get("focus") is None or sweep["focus"] in sweep["methods"], path.name
        assert all(m in sweep["methods"] for m in sweep.get("against") or []), path.name
        for level, axes in cell_groups(sweep):
            for m in sweep["methods"]:
                arm = method_arm(sweep, m, level, axes)
                assert arm["mode"] in ("teleop", "policy", "flow_control", "frs"), (path.name, m)
                if arm["mode"] in ("flow_control", "frs"):
                    assert settings_for(sweep, arm["mode"], m), (path.name, m)  # every mode a sweep plays has its knobs
    grid = load_sweep("grid.yaml")  # the baselines with their papers' fixed knob (lambda 1, the full reversal); MARS both
    assert settings_for(grid, "flow_control", "flow_control") == [{"tau": t} for t in range(2, 17, 2)]
    assert settings_for(grid, "frs", "frs") == [{"pinned_steps": p} for p in (2, 4, 8, 16)]
    assert method_arm(grid, "frs", "U1xT1")["n_reverse_steps"] is None and method_arm(grid, "flow_control", "U1xT1")["lambda"] == 1
    for m in ("mars_user", "mars_task", "mars", "retrained_mars_user"):
        assert len(settings_for(grid, "frs", m)) == 10 * 4 and "tune" not in method_arm(grid, m, "U1xT1")
    cells = grid["grid"]["cells"]
    assert len(cells) == 20 and sorted(n for n, c in cells.items() if c["train"]) == [
        "U1xT1", "U1xT2", "U2xT1", "U2xT3", "U3xT1", "U3xT4", "U4xT1", "U4xT5"]
    arm = method_arm(grid, "mars", "U4xT4")
    assert arm["corruption_curl"] == {"curl": 20.0, "tau": 0.1} and arm["obstacle"]["around"] == "goal"
    assert arm["learned_adapter"] == {"user": "user_curl20_expert_n100_s0", "task": "task_ring_goal_n300_s0_with_rot60"}
    assert arm["time_limit_s"] == 8.0 and arm["input_noise"] == 1.0
    ft = method_arm(load_sweep("grid_ft.yaml", 1), "ft_mars_user", "U2xT3")  # on the obstacle tasks fine-tuned, per seed
    assert (ft["model"], ft["reversal_adapter"]["user"]) == ("flow_goaldrop_wall_gap_ft_n300_s1", "user_gain025x_expert_n100_s1_on_wall_gap_ft_n300")
    arm = method_arm(grid, "frs", "U1xT1")
    assert with_setting(arm, {"n_reverse_steps": 10, "pinned_steps": 4})["n_reverse_steps"] is None
    assert with_setting(arm, {"n_reverse_steps": 6, "pinned_steps": 4})["pinned_steps"] == 4
    assert with_setting(arm, 10)["n_reverse_steps"] is None  # a bare number: an older run's depth
    # the leak check differs from the rigid recipe's check only by the decoy and the fine-tuned arm
    rigid, leak = load_sweep("grid_task_check_rel_rigid.yaml"), load_sweep("grid_task_check_rel_leak.yaml")
    assert set(leak["methods"]) == set(rigid["methods"]) - {"oracle_ft"} and leak["focus"] == "oracle_task_rel_rigid"
    assert leak["against"] == ["oracle_task"] and (leak["seed"], leak["trials"]) == (820, {"tune": 20, "test": 500})
    for cell in ("U0xT2", "U0xT4"):
        for m in leak["methods"]:
            a, b = method_arm(leak, m, cell), method_arm(rigid, m, cell)
            assert a["learned_adapter"] == b["learned_adapter"] and a["obstacle"] == dict(b["obstacle"], decoy=True)


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


def test_tiny_sweep_tunes_jointly_and_replays_fixed_settings(tmp_path):
    """`tune` tunes every combination of a mode's knobs, records the best as {knob: value}, and --fixed-settings replays
    a settings file (settings.yaml's format) on the same test trials without tuning; a cell or method it leaves out is
    an error rather than a guess."""
    (tmp_path / "jt.yaml").write_text(
        "name: jt\nmethods:\n  teleop: {mode: teleop}\n  flow_control: {mode: flow_control}\n  frs: {mode: frs}\n"
        "  mars: {mode: frs, reversal_adapter: {learned: learned_adapter/gain025y_free}}\n"
        "tune: {flow_control: {tau: [8, 16], lambda: [1, 2]}, frs: {n_reverse_steps: [10], pinned_steps: [2, 16]}}\n"
        "grid: {family: gain, levels: [0.25], noises: [1.25]}\nfocus: mars\n")
    common = ["--sweep", str(tmp_path / "jt.yaml"), "--tune-trials", "2", "--test-trials", "4", "--cpu", "--results", str(tmp_path)]
    sweep_main(common + ["--out", "joint"])
    out = tmp_path / "sim" / "sweeps" / "joint"
    test, tuning = (json.loads((out / f"{n}.json").read_text()) for n in ("test", "tuning"))
    assert set(test["test"]) == {"0.25|1.25"}
    cell = test["test"]["0.25|1.25"]
    assert set(cell["metrics"]) == {"teleop", "flow_control", "frs", "mars"} and set(cell["lead"]) == {"teleop", "flow_control", "frs"}
    tuned = tuning["tuning"]["0.25|1.25"]
    assert set(tuned["flow_control"]) == {"tau=8,lambda=1", "tau=8,lambda=2", "tau=16,lambda=1", "tau=16,lambda=2"}
    assert set(tuned["frs"]) == {f"n_reverse_steps=10,pinned_steps={p}" for p in (2, 16)} and "None" in tuned["teleop"]
    assert cell["settings"]["teleop"] is None and set(cell["settings"]["flow_control"]) == {"tau", "lambda"}
    assert set(cell["settings"]["frs"]) == {"n_reverse_steps", "pinned_steps"}
    for fig in ("test.png", "showcase.png"):
        assert (out / "figs" / fig).exists()
    sweep_main(common + ["--fixed-settings", str(out / "settings.yaml"), "--out", "replay"])
    replay = tmp_path / "sim" / "sweeps" / "replay"
    assert json.loads((replay / "tuning.json").read_text())["tuning"] == {}
    assert (replay / "settings.yaml").read_text() == (out / "settings.yaml").read_text()
    for key, cell in test["test"].items():
        again = json.loads((replay / "test.json").read_text())["test"][key]
        assert again["settings"] == cell["settings"] and again["trials"]["outcome"] == cell["trials"]["outcome"]
        assert again["trials"]["duration_s"] == cell["trials"]["duration_s"]
    (tmp_path / "short.yaml").write_text("0.25|1.25:\n  frs: {n_reverse_steps: 3}\n")
    with pytest.raises(ValueError, match="fixed-settings"):
        sweep_main(common + ["--fixed-settings", str(tmp_path / "short.yaml"), "--out", "short"])


def test_task_axes_make_cells_skip_impossible_ones_rank_and_resume(tmp_path):
    (tmp_path / "ax.yaml").write_text(
        "name: ax\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [0.5, 1]\n  axes: {targets: [[2, 2], [4, 4]], dwell_s: [0.3, 0.5]}\n")
    sweep = load_sweep(tmp_path / "ax.yaml")
    groups = cell_groups(sweep)  # every (level, task parameters) combination, in grid order
    assert sweep["grid"]["noises"] == [0.5, 1.0] and len(groups) == 4 and groups[1] == (0.25, {"targets": [2, 2], "dwell_s": 0.5})
    assert cell_key(0.25, 1.0, groups[1][1]) == "0.25|1.0|targets=2-2|dwell_s=0.5" and cell_key(5, 0.0) == "5|0.0"
    arm = method_arm(sweep, "frs", 0.25, groups[3][1])
    assert arm["targets"] == [4, 4] and arm["dwell_s"] == 0.5 and arm["corruption"] == {"scale": [1.0, 0.25]}
    assert load_sweep("study_tuning.yaml")["grid"]["axes"] == {}
    (tmp_path / "named.yaml").write_text(  # named values: an axis of obstacles
        "name: named\nmethods:\n  teleop: {mode: teleop}\n  frs: {mode: frs}\ngrid:\n  family: none\n  noises: [0.5]\n"
        "  axes: {obstacle: {T0: null, T1: {kind: disc, radius: 0.04, along: 0.5}}, targets: [[4, 4]]}\n")
    named = load_sweep(tmp_path / "named.yaml")
    assert cell_groups(named) == [(None, {"obstacle": "T0", "targets": [4, 4]}), (None, {"obstacle": "T1", "targets": [4, 4]})]
    assert cell_key(None, 0.5, {"obstacle": "T1", "targets": [4, 4]}) == "-|0.5|obstacle=T1|targets=4-4"
    assert method_arm(named, "frs", None, {"obstacle": "T1"})["obstacle"]["kind"] == "disc"
    assert method_arm(named, "teleop", None, {"obstacle": "T0"})["obstacle"] is None
    (tmp_path / "bad.yaml").write_text("methods: {teleop: {mode: teleop}}\ngrid: {noises: [1], axes: {tau: [1, 2]}}\n")
    with pytest.raises(ValueError, match="grid.axes"):
        load_sweep(tmp_path / "bad.yaml")
    (tmp_path / "run.yaml").write_text(
        "name: ax\nmethods:\n  teleop: {mode: teleop}\n  mars: {mode: frs, reversal_adapter: {learned: learned_adapter/gain025y_free}}\n"
        "grid:\n  family: gain\n  levels: [0.25]\n  noises: [1.25]\n"
        "  axes: {targets: [[2, 2], [3, 3]], min_target_sep: [0.12, 1.5]}\n"  # 1.5 apart does not fit the screen
        "tune: {frs: {n_reverse_steps: [2, 10]}}\n")
    argv = ["--sweep", str(tmp_path / "run.yaml"), "--tune-trials", "3", "--test-trials", "3", "--results", str(tmp_path), "--out", "ax"]
    sweep_main(argv)
    out = tmp_path / "sim" / "sweeps" / "ax"
    result = json.loads((out / "test.json").read_text())
    keys = {"0.25|1.25|targets=2-2|min_target_sep=0.12", "0.25|1.25|targets=3-3|min_target_sep=0.12"}
    assert set(result["test"]) == keys and result["showcase"] in keys
    assert set(result["skipped"]) == {k.replace("0.12", "1.5") for k in keys} and all("cannot place" in w for w in result["skipped"].values())
    cell = result["test"]["0.25|1.25|targets=3-3|min_target_sep=0.12"]
    assert cell["axes"] == {"targets": [3, 3], "min_target_sep": 0.12} and cell["settings"]["mars"]["n_reverse_steps"] in (2, 10)
    ranking = (out / "ranking.csv").read_text().splitlines()
    assert len(ranking) == 3 and ranking[0].startswith("cell,level,noise,targets,min_target_sep,") and "lead_lo" in ranking[0]
    for fig in ("axes.png", "showcase.png", "tuning_curves.png"):
        assert (out / "figs" / fig).exists()
    done = next(iter(keys))  # --resume keeps the cells already tested and runs only the rest
    result["test"][done]["metrics"]["teleop"]["success"] = 0.123
    result["test"] = {done: result["test"][done]}
    (out / "test.json").write_text(json.dumps(result))
    sweep_main(argv + ["--resume"])
    again = json.loads((out / "test.json").read_text())
    assert set(again["test"]) == keys and again["test"][done]["metrics"]["teleop"]["success"] == 0.123


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


def test_scene_cache_is_keyed_by_cell_split_source_size_and_the_digest(tmp_path, monkeypatch):
    lay, calls = _lay(), []
    attach_scene(lay, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=lambda: calls.append(1))
    assert lay.scene["cnn"].shape == (3, 3, 64, 64) and calls == []
    (name,) = _files(tmp_path)
    assert re.fullmatch(r"U1xT2\|1\.0_test_cnn_64_[0-9a-f]{12}\.npy", name)  # no temporary file is left behind
    real = run_sweep.scene_for
    monkeypatch.setattr(run_sweep, "scene_for", lambda *a, **k: pytest.fail("recomputed"))
    again = _lay()
    attach_scene(again, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)  # a hit recomputes nothing
    assert np.array_equal(again.scene["cnn"], lay.scene["cnn"]) and _files(tmp_path) == [name]
    goal = _lay()
    goal.goal = (goal.goal + 1) % 3  # the goal is not in the digest (nor is the module seed: it is not an input)
    attach_scene(goal, {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)
    monkeypatch.setattr(run_sweep, "scene_for", real)
    obstructed = _lay()
    obstructed.obstacles = np.tile([0.5, 0.5, 0.08], (3, 1, 1))  # the same targets, an obstacle
    obstructed.obstacle_mask = np.ones((3, 1), dtype=bool)
    for changed, size in ((_lay(seed=1), 64), (_lay(), 32), (obstructed, 64)):  # another draw, size, render
        attach_scene(changed, {"cnn": size}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)
    assert len(_files(tmp_path)) == 4
    monkeypatch.setitem(run_sweep.RENDER_PALETTE, "obstacle", (1, 2, 3))  # the dict render_layouts draws with
    attach_scene(_lay(), {"cnn": 64}, tmp_path, "U1xT2|1.0_test", encoder_factory=None)
    assert len(_files(tmp_path)) == 5


def test_a_vlm_source_makes_the_encoder_once_and_matches_the_trainer(tmp_path, monkeypatch):
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
    for vlm in ("a/one", "a/two", "a/one"):  # the digest of a vlm source depends on the encoder's path, a cnn's does not
        attach_scene(_lay(), {"vlm_image": 56}, tmp_path / "paths", "k", factory, vlm=vlm)
        attach_scene(_lay(), {"cnn": 64}, tmp_path / "paths", "k", None, vlm=vlm)
    assert len(_files(tmp_path / "paths")) == 3

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


def test_a_tiny_sweep_attaches_the_scene_for_an_image_task_module(tmp_path, monkeypatch):
    name = _image_adapter(tmp_path, monkeypatch, geometry="relative")
    (tmp_path / "tiny.yaml").write_text(
        "name: tiny_image\nagent: expert\narm: condition_grid.yaml\nmethods:\n  teleop: {mode: teleop}\n"
        f"  mars_cnn: {{mode: frs, reversal_adapter: {{task: {name}}}, tune: {{n_reverse_steps: [2]}}}}\n"
        "grid:\n  users: {U1: {corruption: {scale: [1.0, 0.25]}, adapter: null}}\n"
        "  tasks: {T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}}\n  noises: [1.0]\nfocus: mars_cnn\nseed: 3\n"
    )  # the module name has no _n<demos>_s0 tag, so --module-seed renames the sweep and nothing else
    argv = ["--sweep", str(tmp_path / "tiny.yaml"), "--tune-trials", "2", "--test-trials", "2", "--results", str(tmp_path)]
    sweep_main(argv + ["--out", "tiny_image", "--scene-cache", str(tmp_path / "scene")])
    assert {re.sub(r"_[0-9a-f]{12}\.npy$", "", p.name) for p in (tmp_path / "scene").glob("*.npy")} == {"U1xT2|1.0_tune_cnn_64", "U1xT2|1.0_test_cnn_64"}
    assert set(json.loads((tmp_path / "sim" / "sweeps" / "tiny_image" / "test.json").read_text())["test"]["U1xT2|1.0"]["metrics"]) == {"teleop", "mars_cnn"}
    # --precompute-scene fills the default folder, one per sweep that every --module-seed run shares, and tests nothing
    real = run_sweep.cursor_paths.CHECKPOINTS
    (tmp_path / "ck").mkdir()
    for p in real.iterdir():  # the policies are read from the real folder; the scene cache goes to the temp one
        if p.name != "scene":
            (tmp_path / "ck" / p.name).symlink_to(p)
    monkeypatch.setattr(run_sweep.cursor_paths, "CHECKPOINTS", tmp_path / "ck")
    sweep_main(argv + ["--precompute-scene", "--out", "plain"])
    (folder,) = (tmp_path / "ck" / "scene").iterdir()
    files = _files(folder)
    assert folder.name == "tiny_image" and len(files) == 2 and not (tmp_path / "sim" / "sweeps" / "plain" / "test.json").exists()
    sweep_main(argv + ["--precompute-scene", "--out", "seeded", "--module-seed", "1"])
    assert [p.name for p in (tmp_path / "ck" / "scene").iterdir()] == ["tiny_image"] and _files(folder) == files  # seed 1 hit seed 0's files


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
    with pytest.raises(SystemExit):  # nothing to merge in a precompute
        sweep_main(["--sweep", "x.yaml", "--results", str(tmp_path), "--precompute-scene", "--merge", "a", "b"])


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
    arm = method_arm(load_sweep(sweep), "policy_b", "U0xT2")
    policies, adapters = load_checkpoints({"policy_b": arm}, "cpu")
    with pytest.raises(ValueError, match="scene_encoder"):  # run_trials needs an encoder for a view module
        run_trials(arm, _lay(), ExpertChunkAgent(), Readout(None, 0.0), policies[arm["model"]], adapters, "cpu")
