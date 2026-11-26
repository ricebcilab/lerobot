"""The user study built from a protocol's cells: the cell's tuned arms (run_experiment.protocol_arms), the scene an
image task module reads in a study run, the shipped protocols, a visit calibrated on the participant's own commands
(--calibrate), and what a window without cues draws."""

import functools
import os
import time

import numpy as np
import pytest
import run_experiment
import torch
import yaml
from cursor_sa import config, session
from cursor_sa.agents import ExpertAgent
from cursor_sa.policy import FlowPolicy
from cursor_sa.records import load_run
from cursor_sa.steering import SharedController, to_action

from lerobot.envs.cursor import CursorEnv, TaskConfig, sample_layouts
from lerobot.policies.steering import AdapterModules, LearnedTaskModule

DISC = {"kind": "disc", "radius": 0.08, "along": 0.5}
CELLS = """
arm: condition_grid.yaml
methods:
  teleop: {mode: teleop}
  flow_control: {mode: flow_control, model: flow_other}
  mars:
    mode: frs
    tune: {n_reverse_steps: [2, 10], pinned_steps: [4, 16]}
    reversal_adapter: {user: cell, task: task_x}
tune:
  flow_control: {tau: [4, 8]}
grid:
  users:
    U3: {corruption: {rotation_deg: 60}, adapter: user_c}
  tasks:
    T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}
  noises: [0.5]
seed: 7
"""
TUNED = {"teleop": None, "flow_control": {"tau": 8}, "mars": {"n_reverse_steps": 2, "pinned_steps": 4}}
PROTOCOL = {
    "name": "tiny",
    "blocks": {"trials": 2, "practice": 1, "order": "latin_square"},
    "trial": {"ready_s": 0.0, "feedback_s": 0.0},
    "expert": {"lag": 0, "slow_radius": 0.15},
}
USERS = ("U1", "U2", "U3", "U4")
TASKS = ("T1", "T2", "T3", "T4", "T5")
CALIBRATED = """
name: tiny
cells:
  arm: condition_study.yaml
  methods:
    teleop: {mode: teleop}
    mars: {mode: frs, reversal_adapter: {user: cell}, tune: {n_reverse_steps: [2], pinned_steps: [4]}}
  grid:
    users:
      U3: {corruption: {rotation_deg: 60}, adapter: shared_module}
    tasks:
      T1: {obstacle: null, adapter: null, time_limit_s: 5.0}
    noises: [0.5]
  settings:
    U3xT1|0.5:
      mars: {n_reverse_steps: 2, pinned_steps: 4}
blocks: {trials: 2, practice: 1}
trial: {ready_s: 0.0, feedback_s: 0.0}
calibration: {reaches: 6, practice: 1, steps: 3}
"""


def test_protocol_cells_give_the_cells_tuned_arms(tmp_path):
    """A protocol's cells are a sweep of their own (one user on one task): every method's arm built as that sweep
    builds it for the cell, at the cell's tuned setting and the sweep's noise (the base arm's own noise is another
    level); run_sweep.py tunes the same file. A second user or task is another protocol."""
    import run_sweep

    settings = {
        "U3xT2|0.5": {
            "teleop": None,
            "flow_control": {"tau": 4},
            "mars": {"n_reverse_steps": 10, "pinned_steps": 16},
        }
    }

    def load(cells, settings, name="p.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump({"name": "tiny", "cells": {**cells, "settings": settings}}, sort_keys=False)
        )
        return config.load_protocol(tmp_path / name)

    cells = yaml.safe_load(CELLS)
    arms, user, task = run_experiment.protocol_arms(load(cells, settings))
    assert (user, task, list(arms)) == ("U3", "T2", ["teleop", "flow_control", "mars"])
    assert arms["flow_control"]["tau"] == 4 and arms["flow_control"]["model"] == "flow_other"
    assert (arms["mars"]["n_reverse_steps"], arms["mars"]["pinned_steps"]) == (
        None,
        16,
    )  # 10 of 10 steps: the full reversal
    assert {arm["input_noise"] for arm in arms.values()} == {0.5}
    assert arms["teleop"]["corruption"] == {"rotation_deg": 60} and arms["mars"]["learned_adapter"] == {
        "user": "user_c",
        "task": "task_x",
    }
    assert arms["teleop"]["obstacle"]["kind"] == "disc" and arms["teleop"]["time_limit_s"] == 8.0
    sweep = run_sweep.load_sweep(tmp_path / "p.yaml")  # the protocol is the sweep that tunes it
    assert (sweep["name"], sweep["grid"]["levels"], sweep["seed"]) == ("tiny", ["U3xT2"], 7)
    tuned = run_sweep.with_setting(
        run_sweep.method_arm(sweep, "mars", "U3xT2"), settings["U3xT2|0.5"]["mars"]
    )
    assert all(
        tuned[k] == arms["mars"][k]
        for k in ("learned_adapter", "model", "n_reverse_steps", "pinned_steps", "obstacle")
    )
    grid = cells["grid"]
    for extra in (
        {"users": {**grid["users"], "U1": {"corruption": None, "adapter": "user_a"}}},
        {"tasks": {**grid["tasks"], "T1": {"obstacle": None, "adapter": None, "time_limit_s": 5.0}}},
    ):
        with pytest.raises(ValueError, match="one user on one task"):
            run_experiment.protocol_arms(load({**cells, "grid": {**grid, **extra}}, settings, "two.yaml"))
    untuned = load(cells, {"U3xT2|0.5": {**settings["U3xT2|0.5"], "mars": None}}, "untuned.yaml")
    with pytest.raises(ValueError, match=r"U3xT2\|0.5, mars"):  # a tuned method without its setting
        run_experiment.protocol_arms(untuned)


def test_shipped_study_protocols_resolve():
    """study_T<t>_U<k>.yaml, each grid user on each task: the arms at their tuned settings under the grid's condition,
    noise 1.0, the condition hidden and no cues; on the obstacle tasks flow control and FRS play the task's fine-tuned
    policy and MARS adds its task module. A task's four protocols differ only in their user and its settings, and a
    user is the same readout and module on every task."""
    import run_sweep

    ic_gain = None
    files = {}
    for t in TASKS:
        for u in USERS:
            name = f"study_{t}_{u}"
            protocol = config.load_protocol(f"{name}.yaml")
            assert (protocol["name"], protocol["study"]) == (name, "study")  # results/cursor_study/
            assert protocol["display"] == {"blind": True, "cues": False}
            sweep = run_sweep.load_sweep(name)  # run_sweep.py --sweep <protocol>
            assert (sweep["grid"]["users"], sweep["grid"]["tasks"]) == ([u], [t])
            arms, user, task = run_experiment.protocol_arms(protocol)
            assert (user, task, list(arms)) == (u, t, ["teleop", "flow_control", "frs", "mars"])
            assert {arm["input_noise"] for arm in arms.values()} == {1.0} and {
                arm["gain_mode"] for arm in arms.values()
            } == {"time"}
            assert (
                arms["flow_control"]["lambda"] == 1 and arms["frs"]["n_reverse_steps"] is None
            )  # as their papers define them
            assert (arms["mars"]["learned_adapter"]["task"] is None) == (t == "T1")
            assert (arms["frs"]["model"] == "flow_goaldrop") == (t == "T1") and arms["mars"][
                "model"
            ] == "flow_goaldrop"
            if ic_gain is None:  # a full push, in action units
                policy = session.load_checkpoints({"a": arms["teleop"]}, "cpu")[0]["flow_goaldrop"]
                ic_gain = float(to_action(1.0, policy))
            assert arms["flow_control"]["ic_gain"] == pytest.approx(ic_gain, abs=1e-4)
            files[t, u] = yaml.safe_load((config.CONFIG_DIR / f"{name}.yaml").read_text())

    def shared(p):  # all of a protocol but its name, its user and that user's settings
        cells = {k: v for k, v in p["cells"].items() if k != "settings"}
        return {**p, "name": None, "cells": {**cells, "grid": {**cells["grid"], "users": None}}}

    for t in TASKS:
        assert all(shared(files[t, u]) == shared(files[t, USERS[0]]) for u in USERS), t
    for u in USERS:
        assert all(
            files[t, u]["cells"]["grid"]["users"] == files[TASKS[0], u]["cells"]["grid"]["users"]
            for t in TASKS
        ), u


def test_a_task_and_user_on_the_command_line_pick_the_studys_protocol():
    """--task and --user pick the protocol played: the study's file for them (study, T3 and U2 -> study_T3_U2.yaml).
    A protocol file named whole needs neither, and a study named without them, or with a pair it has no file for, is
    an error that lists the pairs it has."""
    args = run_experiment.parse_args(
        ["--protocol", "study", "--task", "T3", "--participant", "p2", "--user", "U2"]
    )
    assert (args.protocol, args.participant) == ("study_T3_U2.yaml", "p2")
    assert run_experiment.protocol_arms(config.load_protocol(args.protocol))[1:] == ("U2", "T3")
    assert run_experiment.protocol_file("study_T4_U4.yaml") == "study_T4_U4.yaml"
    for task, user in ((None, None), ("T1", None), ("T9", "U1")):
        with pytest.raises(ValueError, match="T1_U1, .*--task T<t> --user U<k>"):
            run_experiment.protocol_file("study", user, task)


def test_a_calibrated_visit_plays_mars_with_the_participants_own_module(tmp_path, monkeypatch):
    """--calibrate: before the first block the cursor plays demonstration reaches by itself while the participant
    pushes through the user's readout; a user module trained on what the readout served takes the shared module's
    place in MARS, by name in every manifest and record, and the shared module is never loaded. A resumed visit keeps
    the module it calibrated."""
    from train import train_adapter

    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / "adapters" / name / "adapter.pt")
    small = functools.partial(
        train_adapter.train_calibrated_user, batch=16, check=64
    )  # a tiny run of the trainer
    monkeypatch.setattr(train_adapter, "train_calibrated_user", small)
    (tmp_path / "tiny.yaml").write_text(CALIBRATED)
    assert config.load_protocol("study_T1_U1.yaml")["calibration"] == {
        "reaches": 50,
        "practice": 5,
        "steps": 500,
        "slow": 2,
    }
    argv = ["--protocol", str(tmp_path / "tiny.yaml"), "--participant", "sim", "--cpu", "--calibrate"]
    argv += ["--results", str(tmp_path / "results")]
    run_experiment.main(argv)
    runs = tmp_path / "results" / "cursor_tiny" / "sim"
    manifest, reaches = load_run(runs / "tiny_s1_calibration")
    assert (
        manifest["block_order"] == ["calibration"] and manifest["arms"]["calibration"]["mode"] == "playback"
    )
    assert [r["practice"] for r in reaches] == [True] + [False] * 6
    for r in reaches:  # the demonstration lands on the goal; the readout is the user's, read at every step
        assert (r["outcome"], r["user"], r["task"]) == ("hit", "U3", "calibration")
        assert (r["corruption"], r["input_noise"], r["obstacle"]) == ({"rotation_deg": 60}, 0.5, None)
        assert len(r["served_commands"]) == len(r["path"]) - 1 == round(r["duration_s"] / 0.02)
        path, start, goal = np.array(r["path"]), np.array(r["start"]), np.array(r["targets"][r["goal"]])
        along = (goal - start) / np.linalg.norm(goal - start)
        offset = path - start
        assert (
            np.abs(offset[:, 0] * along[1] - offset[:, 1] * along[0]).max() < 2e-3
        )  # a straight line to the goal
        assert (np.diff(offset @ along) > -1e-3).all()  # that never turns back: no overshoot and correction
        steps = np.diff(path, axis=0)[: 2 * ((len(path) - 1) // 2)].reshape(-1, 2, 2)
        np.testing.assert_allclose(
            steps[:, 0], steps[:, 1], atol=2e-3
        )  # at half speed: every step over two frames
    name = "cursor_participants/tiny/sim/tiny_s1"
    path = tmp_path / "adapters" / name / "adapter.pt"
    assert AdapterModules.load(path).kind == "user"
    meta = torch.load(path)["meta"]["calibration"]
    assert (meta["participant"], meta["user"], meta["reaches"], meta["steps"]) == ("sim", "U3", 6, 3)
    assert {"heading_cos_none", "heading_cos_user"} <= set(meta["log"][-1])
    played, records = load_run(runs / "tiny_s1")
    assert played["arms"]["mars"]["reversal_adapter"] == {"user": name, "task": None}
    assert (
        f"adapters/{name}" in played["checkpoints"] and "adapters/shared_module" not in played["checkpoints"]
    )
    assert {r["arm"] for r in records} == {"teleop", "mars"} and len(records) == 6
    assert {r["reversal_adapter"]["user"] for r in records if r["arm"] == "mars"} == {name}
    trained, real = [], run_experiment.train_user_module
    monkeypatch.setattr(
        run_experiment, "train_user_module", lambda *a, **k: trained.append(1) or real(*a, **k)
    )
    run_experiment.main([*argv, "--resume"])  # nothing left to play, nothing trained again
    path.unlink()  # a visit stopped while its module was trained: the reaches stand, the training runs again
    run_experiment.main([*argv, "--resume"])
    assert trained == [1] and AdapterModules.load(path).kind == "user"
    assert len(load_run(runs / "tiny_s1_calibration")[1]) == 7
    # another study of the same user, without --calibrate: MARS plays the participant's own module by default
    (tmp_path / "other.yaml").write_text(CALIBRATED.replace("name: tiny", "name: other"))
    run_experiment.main(
        [
            "--protocol",
            str(tmp_path / "other.yaml"),
            "--participant",
            "sim",
            "--cpu",
            "--results",
            str(tmp_path / "results"),
        ]
    )
    other, records = load_run(tmp_path / "results" / "cursor_other" / "sim" / "other_s1")
    assert other["arms"]["mars"]["reversal_adapter"]["user"] == name and trained == [
        1
    ]  # reused, not trained again
    assert {r["reversal_adapter"]["user"] for r in records if r["arm"] == "mars"} == {name}
    assert session.participant_module("p9", "U3") is None  # a participant without one plays the shared module


def test_a_study_run_attaches_and_caches_an_image_task_modules_scene(tmp_path, monkeypatch):
    """A task module that reads a picture of the scene: the study attaches the run's scene to its layouts from a cache
    keyed by participant, visit and task, a second pass renders nothing, and a run without it stops before its first
    step."""
    monkeypatch.setattr(session, "adapter_path", lambda name: tmp_path / "adapters" / name / "adapter.pt")
    AdapterModules.save_module(
        LearnedTaskModule(horizon=16, token_source="cnn", image_size=32),
        tmp_path / "adapters" / "t_cnn" / "adapter.pt",
        "task",
        meta={},
    )
    arms = {
        "mars": config.load_arm(
            "base.yaml",
            {"mode": "frs", "obstacle": DISC, "reversal_adapter": {"user": None, "task": "t_cnn"}},
        )
    }

    def make(who):
        return session.Run(
            {**PROTOCOL, "display": {"blind": True, "cues": False}},
            who,
            1,
            ExpertAgent(),
            "cpu",
            tmp_path / "results",
            arms=arms,
            task="T2",
        )

    run = make("sim")
    run.start()
    with pytest.raises(ValueError, match="scene"):
        run.run()
    run_experiment.attach_scenes(run, tmp_path / "scene")
    files = sorted(p.name for p in (tmp_path / "scene").iterdir())
    assert len(files) == 2 and all(
        f.startswith("sim_s1_T2_") and "_cnn_32_" in f for f in files
    )  # trials and practice
    assert run.layouts.scene["cnn"].shape[0] == 2 and run.practice.scene["cnn"].shape[0] == 1
    assert make("sim").layouts.scene is None and run.run() == 3  # the trials play with their scene
    again = make("sim")
    monkeypatch.setattr(
        "run_sweep.render_scene_array", lambda *a: pytest.fail("the cache holds these scenes")
    )
    again.policies, again.adapters = session.load_checkpoints(again.arms, "cpu")
    run_experiment.attach_scenes(again, tmp_path / "scene")
    np.testing.assert_array_equal(again.layouts.scene["cnn"], run.layouts.scene["cnn"])


def test_the_renderer_hides_the_assist_cues_and_the_arm(monkeypatch):
    """`cues=False` (a blinded study): no command line, no planned path and no STEERING label, so a steered block
    looks like a teleop one; the default window draws them. The block and trial count is always shown, and the
    arm's name beside it only when the window is not blind (--no-blind)."""
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame
    from cursor_sa import renderer as window

    task = TaskConfig()
    layouts = sample_layouts(np.random.default_rng(0), 1, 4, task)
    env = CursorEnv(layouts, task)
    ctl = SharedController(FlowPolicy().eval(), layouts, "frs", execute=8)
    served = np.array([[0.8, 0.3]])
    env.step(ctl.act(served))  # a steered chunk: seven planned steps reach out from under the cursor's mark

    def colours(cues):
        screen = window.Renderer(cues=cues)
        screen.draw({"env": env, "ctl": ctl, "layouts": layouts, "task": task, "served": served, "hud": {}})
        pixels = {tuple(c) for c in pygame.surfarray.array3d(screen.screen).reshape(-1, 3)}
        screen.close()
        return pixels

    shown, hidden = colours(True), colours(False)
    assert window.COMMAND in shown and window.PLAN in shown
    assert window.COMMAND not in hidden and window.PLAN not in hidden and window.CURSOR in hidden
    assert window.GUIDE not in shown
    ctl = session.Playback(
        np.full((1, 30, 2), 0.01), slow=2, start=env.pos
    )  # a calibration reach: its whole path
    assert window.GUIDE in colours(False)

    def hud_text(blind):
        screen, said = window.Renderer(blind=blind), []
        monkeypatch.setattr(screen, "text", lambda s, *a, **k: said.append(s))
        screen.draw(
            {
                "env": env,
                "ctl": ctl,
                "layouts": layouts,
                "task": task,
                "served": served,
                "hud": {"block": 2, "n_blocks": 4, "trial": 3, "n_trials": 40, "arm": "mars"},
            }
        )
        screen.close()
        return " ".join(said)

    assert "Block 2 of 4   trial 3 of 40" in hud_text(True) and "mars" not in hud_text(True)
    assert "Block 2 of 4   trial 3 of 40" in hud_text(False) and "mars" in hud_text(False)


def test_trial_frames_keep_their_deadlines(monkeypatch):
    """Frame k of a trial ends k frame periods after the trial began, so a frame that runs late (a chunk computed in
    it, 33 ms of a 20 ms period) is followed by a shorter one and the trial keeps the task's rate; after a stall the
    deadlines start again from there instead of a burst of short frames."""
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    from cursor_sa import renderer as window

    class Clock:  # the renderer's time, in ms
        t = 0.0

        def monotonic(self):
            return self.t / 1000

        def sleep(self, s):
            self.t += s * 1000

    clock = Clock()
    monkeypatch.setattr(window, "time", clock)
    task = TaskConfig()
    layouts = sample_layouts(np.random.default_rng(0), 1, 4, task)
    ctx = {
        "env": CursorEnv(layouts, task),
        "ctl": SharedController(FlowPolicy().eval(), layouts, "teleop"),
        "layouts": layouts,
        "task": task,
        "served": np.zeros((1, 2)),
        "hud": {},
        "agent": ExpertAgent(),
    }
    screen = window.Renderer(fps=50)
    screen.begin_trial()

    def frame(work_ms):  # the step's own work, then the frame
        clock.t += work_ms
        screen.frame(ctx)
        return round(clock.t, 6)

    ends = [frame(33 if k % 4 == 0 else 1) for k in range(8)]  # a chunk every fourth frame
    assert ends == [33, 40, 60, 80, 113, 120, 140, 160]
    assert [frame(200), frame(1)] == [360, 380]  # a stall: no catching up
    screen.close()


def test_a_trial_starts_after_the_ready_pause_whatever_the_stick_does(monkeypatch):
    """Before each trial the window shows the layout under "Get ready" for ready_s and then starts: a person still
    pushing the stick is not asked to release it first."""
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    from cursor_sa import renderer as window

    class Pushing:  # a person who never lets go of the stick
        idle = False

    screen, banners = window.Renderer(ready_s=0.05), []
    monkeypatch.setattr(screen, "draw", lambda ctx, banner=None, **k: banners.append(banner))
    tic = time.monotonic()
    screen.ready_phase({"agent": Pushing()})
    assert (
        time.monotonic() - tic >= 0.05 and screen.deadline is not None
    )  # the trial's frame grid starts there
    assert banners and set(banners) == {"Get ready"}
    screen.close()
