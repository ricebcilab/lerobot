"""The user study built from a protocol's cells: each task section's tuned arms (run_experiment.protocol_sections), the
scene an image task module reads in a study run, the shipped protocols, a visit calibrated on the participant's own
commands (--calibrate), and what a window without cues draws."""

import functools
import os

import numpy as np
import pytest
import run_experiment
import torch
import yaml
from cursor_sa import config, session
from cursor_sa.agents import ExpertAgent
from cursor_sa.policy import FlowPolicy
from cursor_sa.records import load_run
from cursor_sa.session import Visit
from cursor_sa.steering import SharedController, to_action

from lerobot.envs.cursor import CursorEnv, TaskConfig, sample_layouts
from lerobot.policies.steering import AdapterModules, LearnedTaskModule

DISC = {"kind": "disc", "radius": 0.08, "along": 0.5}
CELLS = """
arm: condition_grid.yaml
methods:
  teleop: {mode: teleop}
  flow_control: {mode: flow_control, model: {T1: flow_goaldrop, T2: flow_other}}
  mars:
    mode: frs
    tune: {n_reverse_steps: [2, 10], pinned_steps: [4, 16]}
    reversal_adapter: {user: cell, task: {T1: null, T2: task_x}}
tune:
  flow_control: {tau: [4, 8]}
grid:
  users:
    U3: {corruption: {rotation_deg: 60}, adapter: user_c}
  tasks:
    T1: {obstacle: null, adapter: null, time_limit_s: 5.0}
    T2: {obstacle: {kind: disc, radius: 0.08, along: 0.5}, adapter: null}
  noises: [0.5]
seed: 7
"""
TUNED = {"teleop": None, "flow_control": {"tau": 8}, "mars": {"n_reverse_steps": 2, "pinned_steps": 4}}
PROTOCOL = {
    "name": "tiny",
    "blocks": {"trials": 2, "practice": 1, "order": "latin_square"},
    "trial": {"ready_hold_s": 0.0, "feedback_s": 0.0},
    "expert": {"lag": 0, "slow_radius": 0.15},
}
USERS = ("U1", "U2", "U3", "U4")
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
trial: {ready_hold_s: 0.0, feedback_s: 0.0}
calibration: {reaches: 6, practice: 1, steps: 3}
"""


def test_protocol_cells_give_each_task_its_tuned_arms(tmp_path):
    """A protocol's cells are a sweep of their own (one user x the tasks played): one section per task, every
    method's arm built as that sweep builds it for the cell, at the cell's tuned setting and the sweep's noise (the
    base arm's own noise is another level); run_sweep.py tunes the same file."""
    import run_sweep

    settings = {
        "U3xT1|0.5": TUNED,
        "U3xT2|0.5": {
            "teleop": None,
            "flow_control": {"tau": 4},
            "mars": {"n_reverse_steps": 10, "pinned_steps": 16},
        },
    }

    def load(cells, settings, name="p.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump({"name": "tiny", "cells": {**cells, "settings": settings}}, sort_keys=False)
        )
        return config.load_protocol(tmp_path / name)

    cells = yaml.safe_load(CELLS)
    protocol = load(cells, settings)
    sections, user = run_experiment.protocol_sections(protocol)
    assert user == "U3" and list(sections) == ["T1", "T2"]
    assert all(list(arms) == ["teleop", "flow_control", "mars"] for arms in sections.values())
    t1, t2 = sections["T1"], sections["T2"]
    assert (t1["flow_control"]["tau"], t2["flow_control"]["tau"]) == (8, 4)
    assert (t1["mars"]["n_reverse_steps"], t1["mars"]["pinned_steps"]) == (2, 4)
    assert (t2["mars"]["n_reverse_steps"], t2["mars"]["pinned_steps"]) == (
        None,
        16,
    )  # 10 of 10 steps is the full reversal
    assert {arm["input_noise"] for arms in sections.values() for arm in arms.values()} == {0.5}
    assert t1["teleop"]["corruption"] == {"rotation_deg": 60}
    assert t1["mars"]["learned_adapter"] == {"user": "user_c", "task": None}
    assert (
        t2["mars"]["learned_adapter"] == {"user": "user_c", "task": "task_x"}
        and t2["flow_control"]["model"] == "flow_other"
    )
    assert t2["teleop"]["obstacle"]["kind"] == "disc" and (
        t1["teleop"]["time_limit_s"],
        t2["teleop"]["time_limit_s"],
    ) == (5.0, 8.0)
    sweep = run_sweep.load_sweep(tmp_path / "p.yaml")  # the protocol is the sweep that tunes it
    assert (sweep["name"], sweep["grid"]["levels"], sweep["seed"]) == ("tiny", ["U3xT1", "U3xT2"], 7)
    tuned = run_sweep.with_setting(
        run_sweep.method_arm(sweep, "mars", "U3xT2"), settings["U3xT2|0.5"]["mars"]
    )
    assert all(
        tuned[k] == t2["mars"][k]
        for k in ("learned_adapter", "model", "n_reverse_steps", "pinned_steps", "obstacle")
    )
    two = {
        **cells,
        "grid": {
            **cells["grid"],
            "users": {**cells["grid"]["users"], "U1": {"corruption": None, "adapter": "user_a"}},
        },
    }
    with pytest.raises(
        ValueError, match="one user"
    ):  # a protocol plays one user; another user is another protocol
        run_experiment.protocol_sections(load(two, settings, "two.yaml"))
    untuned = load(cells, {**settings, "U3xT2|0.5": {**settings["U3xT2|0.5"], "mars": None}}, "untuned.yaml")
    with pytest.raises(ValueError, match=r"U3xT2\|0.5, mars"):  # a tuned method without its setting
        run_experiment.protocol_sections(untuned)


def test_shipped_study_protocols_resolve():
    """study_short_<user>.yaml (each grid user on T1) and study_long_<user>.yaml (each on T1-T5): the arms of every
    task section at their tuned settings, noise 0.5, the condition hidden and no cues; the long ones' obstacle tasks
    play the fine-tuned policy and MARS's task module, and a study's four differ only in their user and its settings."""
    import run_sweep

    studies = {"study_short": ["T1"], "study_long": ["T1", "T2", "T3", "T4", "T5"]}
    for study, tasks in studies.items():
        for expected in USERS:
            protocol = config.load_protocol(f"{study}_{expected}.yaml")
            assert (
                protocol["name"] == f"{study}_{expected}" and protocol["study"] == study
            )  # results/<study>/
            assert protocol["display"] == {"blind": True, "cues": False}
            assert run_sweep.load_sweep(protocol["name"])["grid"]["users"] == [
                expected
            ]  # run_sweep.py --sweep <protocol>
            sections, user = run_experiment.protocol_sections(protocol)
            assert user == expected and list(sections) == tasks, protocol["name"]
            for task, arms in sections.items():
                assert list(arms) == ["teleop", "flow_control", "frs", "mars"]
                assert {arm["input_noise"] for arm in arms.values()} == {0.5}
                assert (
                    arms["flow_control"]["lambda"] == 1 and arms["frs"]["n_reverse_steps"] is None
                )  # as their papers define them
                assert (arms["mars"]["learned_adapter"]["task"] is None) == (task == "T1")
                assert (arms["frs"]["model"] == "flow_goaldrop") == (task == "T1") and arms["mars"][
                    "model"
                ] == "flow_goaldrop"
                policy = session.load_checkpoints({"a": arms["teleop"]}, "cpu")[0][
                    "flow_goaldrop"
                ]  # a full push, in action units
                assert arms["flow_control"]["ic_gain"] == pytest.approx(
                    float(to_action(1.0, policy)), abs=1e-4
                )
    assert sections["T2"]["teleop"]["gain_mode"] == "time"

    def shared(p):  # all of a protocol but its name, its user and that user's settings
        cells = {k: v for k, v in p["cells"].items() if k != "settings"}
        return {**p, "name": None, "cells": {**cells, "grid": {**cells["grid"], "users": None}}}

    for study in studies:
        files = [yaml.safe_load((config.CONFIG_DIR / f"{study}_{u}.yaml").read_text()) for u in USERS]
        assert all(shared(p) == shared(files[0]) for p in files), study
        assert [list(p["cells"]["grid"]["users"]) for p in files] == [[u] for u in USERS]
    users = [
        yaml.safe_load((config.CONFIG_DIR / f"{study}_{u}.yaml").read_text())["cells"]["grid"]["users"]
        for study in studies
        for u in USERS
    ]
    assert users[:4] == users[4:]  # a user is the same readout and module in both studies


def test_a_user_on_the_command_line_picks_the_studys_protocol():
    """--user names the user profile on the command line: the protocol played is the study's file for that user
    (study_short and U3 -> study_short_U3.yaml). A protocol file named whole needs no --user, and a study named
    without its user, or with one it has no file for, is an error that lists the users it has."""
    args = run_experiment.parse_args(["--protocol", "study_short", "--participant", "p2", "--user", "U3"])
    assert (args.protocol, args.participant) == ("study_short_U3.yaml", "p2")
    assert run_experiment.protocol_sections(config.load_protocol(args.protocol))[1] == "U3"
    assert run_experiment.protocol_file("study_long.yaml", "U2") == "study_long_U2.yaml"
    assert run_experiment.protocol_file("study_long_U4.yaml") == "study_long_U4.yaml"
    for user in (None, "U9"):
        with pytest.raises(ValueError, match="--user U1, U2, U3 or U4"):
            run_experiment.protocol_file("study_short", user)


def test_a_calibrated_visit_plays_mars_with_the_participants_own_module(tmp_path, monkeypatch):
    """--calibrate: before the first section the cursor plays demonstration reaches by itself while the participant
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
    assert config.load_protocol("study_short_U1.yaml")["calibration"] == {
        "reaches": 50,
        "practice": 5,
        "steps": 500,
        "slow": 2,
    }
    argv = ["--protocol", str(tmp_path / "tiny.yaml"), "--participant", "sim", "--cpu", "--calibrate"]
    argv += ["--results", str(tmp_path / "results")]
    run_experiment.main(argv)
    runs = tmp_path / "results" / "tiny" / "sim"
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
    name = "participants/tiny/sim/tiny_s1"
    path = tmp_path / "adapters" / name / "adapter.pt"
    assert AdapterModules.load(path).kind == "user"
    meta = torch.load(path)["meta"]["calibration"]
    assert (meta["participant"], meta["user"], meta["reaches"], meta["steps"]) == ("sim", "U3", 6, 3)
    assert {"heading_cos_none", "heading_cos_user"} <= set(meta["log"][-1])
    played, records = load_run(runs / "tiny_s1_T1")
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


def test_a_study_run_attaches_and_caches_an_image_task_modules_scene(tmp_path, monkeypatch):
    """A task module that reads a picture of the scene: the study attaches each section's scene to its layouts from
    a cache keyed by the visit, a second pass renders nothing, and a run without it stops before its first step."""
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
        return Visit(
            {**PROTOCOL, "display": {"blind": True, "cues": False}},
            {"T2": arms},
            who,
            1,
            ExpertAgent(),
            "cpu",
            tmp_path / "results",
        )

    visit = make("sim")
    visit.start()
    with pytest.raises(ValueError, match="scene"):
        visit.run()
    run_experiment.attach_scenes(visit.runs, tmp_path / "scene")
    files = sorted(p.name for p in (tmp_path / "scene").iterdir())
    assert len(files) == 2 and all(
        f.startswith("sim_s1_T2_") and "_cnn_32_" in f for f in files
    )  # trials and practice
    run = visit.runs[0]
    assert run.layouts.scene["cnn"].shape[0] == 2 and run.practice.scene["cnn"].shape[0] == 1
    assert make("sim").runs[0].layouts.scene is None and visit.run() == 3  # the trials play with their scene
    again = make("sim")
    monkeypatch.setattr(
        "run_sweep.render_scene_array", lambda *a: pytest.fail("the cache holds these scenes")
    )
    for r in again.runs:
        r.policies, r.adapters = session.load_checkpoints(r.arms, "cpu")
    run_experiment.attach_scenes(again.runs, tmp_path / "scene")
    np.testing.assert_array_equal(again.runs[0].layouts.scene["cnn"], run.layouts.scene["cnn"])


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
