"""The study runner: the schedule, the per-trial records, authority, the scene digest, and aborts.

No hardware and no checkpoint: the station, the bundle and the steering policy are fakes (the CLI's
factory names are monkeypatched, so the runner's own composition path runs), the loop is the real one,
and the operator's outcome entry is a monkeypatched ``input``. The records are read back the way the
analysis reads them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import analysis
import cli
import numpy as np
import pytest
import study
from config import load_config
from source import FakeSource

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
"""A DROID-like reset pose: every arm resets to it, so every trial of a config shares its q0."""

DEFLECTION = np.array([0.5, -0.5, 0.4, 0.0, 0.0, 0.0])
"""A deflection well above ``[command] deadband``, so a steered trial has engaged steps."""

ACTION = np.array([0.3, -0.2, 0.1, 0.0, 0.4, -0.1, 0.2, 1.0])
"""The fake bundle's action: dims 0-6 differ from the plan, so the authority is measurable."""

PLAN = np.array([0.1, 0.0, -0.2, 0.3, 0.0, 0.2, -0.1, 0.0])
"""The fake bundle's own chunk step: the plan the authority is measured against."""


# ---------------------------------------------------------------- the fakes


class _Station:
    """A deterministic station: one observation, the same every reset and step, recording its actions."""

    def __init__(self, frame_value: int = 0) -> None:
        self.arm = "follower"
        self.frame_value = frame_value
        self.resets = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    @property
    def manifest(self):
        return SimpleNamespace(arms=("follower",), cameras=("scene_zed", "wrist_zed"))

    def reset(self) -> dict[str, np.ndarray]:
        self.resets += 1
        return _observation(self.frame_value)

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        self.actions.append(np.array(action, copy=True))
        return _observation(self.frame_value)

    def close(self) -> None:
        self.closed = True

    def cancel_episode(self) -> None:
        pass


class _ChangingStation(_Station):
    """A station whose every reset is a different scene, so a trial's digest can be tied to its reset.

    ``reset()`` returns the pose ``JOINT + 0.01 * reset count`` on frame ``reset count``; ``step()``
    returns the steady observation (``JOINT``, frame 0), so the first control step of a trial is the
    only one computed from the reset observation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.observed: list[dict[str, np.ndarray]] = []

    def reset(self) -> dict[str, np.ndarray]:
        self.resets += 1
        observation = _observation(self.resets)
        observation["follower/joint_position"] = JOINT + 0.01 * self.resets
        self.observed.append(observation)
        return observation

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        self.actions.append(np.array(action, copy=True))
        return _observation()


def _observation(frame_value: int = 0) -> dict[str, np.ndarray]:
    return {
        "follower/joint_position": JOINT.copy(),
        "follower/gripper_position": np.zeros(1),
        "scene_zed": np.full((8, 8, 3), frame_value, dtype=np.uint8),
        "wrist_zed": np.zeros((8, 8, 3), dtype=np.uint8),
    }


class _Bundle:
    """The bundle as the loop reads it: the fixed action, the plan before it, and the two counters."""

    scene = None

    def __init__(self, action: np.ndarray = ACTION, plan: np.ndarray | None = PLAN) -> None:
        self.action = np.asarray(action, dtype=np.float64)
        self._plan = None if plan is None else np.asarray(plan, dtype=np.float64)
        self.plan8: np.ndarray | None = None
        self.prepares = 0
        self.acts = 0
        self.resets = 0

    def prepare(self, raw_batch: dict) -> dict:
        self.prepares += 1
        return {"prepared": self.prepares}

    def act(self, batch: dict, command7: np.ndarray | None = None) -> np.ndarray:
        self.acts += 1
        self.plan8 = (
            self.action.copy() if command7 is None else None if self._plan is None else self._plan.copy()
        )
        return self.action.copy()

    def reset(self) -> None:
        self.resets += 1
        self.plan8 = None


class _Adapter:
    """The steering policy's adapter holder: records every matrix the loop assigns."""

    def __init__(self) -> None:
        self.matrices: list[np.ndarray] = []

    @property
    def matrix(self) -> np.ndarray | None:
        return self.matrices[-1] if self.matrices else None

    @matrix.setter
    def matrix(self, value) -> None:
        self.matrices.append(np.array(value, copy=True))


class _Steering:
    """A steering policy as the loop uses it: only the adapter holder, written by the ``mars-f`` sync."""

    def __init__(self) -> None:
        self.adapter = _Adapter()


@pytest.fixture
def rig(monkeypatch):
    """A fake station/source/bundle/steering through the CLI's factories; returns what was built."""
    station = _Station()
    bundles: list[_Bundle] = []
    steerings: list[_Steering] = []
    made: list[tuple[_Bundle, object]] = []
    loads: list[object] = []
    tasks: list[object] = []

    def load(config, task=None):
        loads.append(config)
        tasks.append(task)
        bundle = _Bundle()
        bundles.append(bundle)
        return bundle

    def make(bundle, steering_cfg):
        made.append((bundle, steering_cfg))
        steering = _Steering()
        steerings.append(steering)
        return steering

    monkeypatch.setattr(cli, "C3poStation", lambda *args: station)
    monkeypatch.setattr(cli, "SpaceMouseSource", lambda **kwargs: FakeSource(deflection=DEFLECTION))
    monkeypatch.setattr(cli, "load_bundle", load)
    monkeypatch.setattr(cli, "make_steering", make)
    return SimpleNamespace(
        station=station, bundles=bundles, steerings=steerings, made=made, loads=loads, tasks=tasks
    )


def _operator(monkeypatch, *answers: str | BaseException) -> None:
    """The terminal's outcome entry: the answers in order; an exception instance is raised instead."""
    pending = list(answers)

    def answer(prompt: str = "") -> str:
        value = pending.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", answer)


def _config(
    tmp_path: Path,
    arms: list[str],
    *,
    order: str = "shuffled",
    trials: int = 2,
    max_steps: int = 2,
    seed: int = 0,
) -> Path:
    """A config whose ``[study]`` names the block, with the two steered modes the tests use."""
    text = f"""
[modes.frs]
mode = "frs"

[modes.mars-f]
mode = "frs"
adapter_matrix = "corruption"

[study]
arms = {json.dumps(arms)}
trials = {trials}
order = "{order}"
max_steps = {max_steps}
out = "{tmp_path}/study"
seed = {seed}
"""
    path = tmp_path / "franka.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _npz(run_dir: Path, arm: str, pair: int) -> dict[str, np.ndarray]:
    with np.load(run_dir / "trials" / f"{arm}-{pair}.npz") as data:
        return {name: np.array(data[name]) for name in data.files}


# ---------------------------------------------------------------- the schedule and the records


def test_shuffled_order_is_each_rounds_seeded_permutation(tmp_path, rig, monkeypatch):
    """``shuffled``: every round runs every arm once in a seeded order, and ``pair`` is the round.

    The expected order is the one ``[study] seed`` fixes (``default_rng(0)`` over the three arms); the
    records' ``(pair, arm)`` sequence is that order, so the pairing key does not depend on the order.
    """
    path = _config(tmp_path, ["autonomous", "frs", "mars-f"], trials=2, max_steps=2)
    _operator(monkeypatch, "y", "n", "y", "n", "y", "n")

    run_dir = study.run(load_config(path), prompt="pick the cube", config_path=path)

    records = _records(run_dir)
    assert [(record["pair"], record["arm"]) for record in records] == [
        (0, "mars-f"),
        (0, "autonomous"),
        (0, "frs"),
        (1, "mars-f"),
        (1, "frs"),
        (1, "autonomous"),
    ]
    assert [record["success"] for record in records] == [True, False, True, False, True, False]
    assert len(list((run_dir / "trials").glob("*.npz"))) == 6


def test_blocked_order_runs_one_arms_trials_before_the_next(tmp_path, rig, monkeypatch):
    """``blocked``: all of one arm's rounds first; ``pair`` still counts the rounds."""
    path = _config(tmp_path, ["frs", "mars-f"], order="blocked", trials=2)
    _operator(monkeypatch, "y", "y", "n", "n")

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    assert [(record["pair"], record["arm"]) for record in _records(run_dir)] == [
        (0, "frs"),
        (1, "frs"),
        (0, "mars-f"),
        (1, "mars-f"),
    ]


def test_a_trial_records_one_jsonl_line_one_npz_and_the_run_provenance(tmp_path, rig, monkeypatch):
    """The record format: the jsonl fields, the four arrays, the config copy and ``run.json``."""
    path = _config(tmp_path, ["autonomous", "frs"], trials=2)
    _operator(monkeypatch, "y", "n", "y", "n")

    run_dir = study.run(load_config(path), prompt="pick the cube", config_path=path)

    records = _records(run_dir)
    assert len(records) == 4
    for record in records:
        assert set(record) == set(analysis.RECORD_FIELDS)
        assert record["run"] == run_dir.name
        assert record["prompt"] == "pick the cube"
        assert record["steps"] == 2
        assert record["config"] == str(path)
        assert record["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["q0"] == JOINT.tolist()
        assert record["scene_hash"] == study.scene_digest(_observation(), "scene_zed")
        assert record["timestamp"]
        assert record["duration_s"] > 0
    assert [record["success"] for record in records] == [True, False, True, False]

    for record in records:
        arrays = _npz(run_dir, record["arm"], record["pair"])
        assert set(arrays) == {"executed", "plan", "served", "q"}
        assert arrays["executed"].shape == (2, 8)
        assert arrays["plan"].shape == (2, 8)
        assert arrays["served"].shape == (2, 7)
        assert arrays["q"].shape == (2, 7)
        np.testing.assert_array_equal(arrays["q"], np.tile(JOINT, (2, 1)))
        # The record's numbers are the ones the analysis computes from the arrays; a mode with
        # nothing to measure records null, not a fabricated zero.
        measured = study.metrics(arrays["executed"], arrays["plan"], arrays["served"], 0.02)
        assert record["engaged_fraction"] == pytest.approx(measured.engaged_fraction)
        for field in ("authority", "mean_r"):
            if getattr(measured, field) is None or np.isnan(getattr(measured, field)):
                assert record[field] is None
            else:
                assert record[field] == pytest.approx(getattr(measured, field))

    provenance = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert provenance["run"] == run_dir.name
    assert provenance["config_path"] == str(path)
    assert provenance["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["arms"] == ["autonomous", "frs"]
    assert (provenance["trials"], provenance["order"], provenance["seed"]) == (2, "shuffled", 0)
    assert provenance["max_steps"] == 2 and provenance["prompt"] == "pick the cube"
    assert provenance["timestamp"]
    assert isinstance(provenance["git_head"], str) and provenance["git_head"]
    assert isinstance(provenance["git_dirty"], bool)
    assert (provenance["completed"], provenance["aborted"]) == (4, False)
    assert (run_dir / "config.toml").read_bytes() == path.read_bytes()
    assert rig.station.closed


def test_the_arm_bundle_and_steering_are_built_once_and_a_new_loop_per_trial(tmp_path, rig, monkeypatch):
    """Two rounds reuse one bundle and one steering policy per arm; the trial's loops are separate.

    The station is reset once per trial: the runner digests that observation and the loop starts the
    episode from it (``initial_obs``), so no trial resets the scene twice.
    """
    path = _config(tmp_path, ["autonomous", "frs"], trials=3)
    _operator(monkeypatch, *(["y"] * 6))

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    assert len(rig.loads) == 2  # autonomous's bundle and frs's, once each
    assert len(rig.bundles) == 2 and len(rig.made) == 1  # only frs has a steering policy
    assert len(_records(run_dir)) == 6
    assert rig.station.resets == 6
    assert len(rig.station.actions) == 2 * 6


def test_a_teleop_trial_records_no_plan_and_an_autonomous_trial_records_no_command(
    tmp_path, rig, monkeypatch
):
    """The two session anchors: teleop has a command and no plan, autonomous a plan and no command.

    The arrays stay (T, 8)/(T, 7) with NaN in the mode that has no such value, and the record's
    authority is null because nothing was steered.
    """
    path = _config(tmp_path, ["teleop", "autonomous"], order="blocked", trials=1)
    _operator(monkeypatch, "y", "y")

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    teleop, autonomous = _records(run_dir)
    assert (teleop["arm"], autonomous["arm"]) == ("teleop", "autonomous")
    teleop_arrays = _npz(run_dir, "teleop", 0)
    assert np.all(np.isnan(teleop_arrays["plan"]))
    assert np.all(np.isfinite(teleop_arrays["served"]))
    assert teleop["authority"] is None and teleop["mean_r"] is not None
    autonomous_arrays = _npz(run_dir, "autonomous", 0)
    assert np.all(np.isfinite(autonomous_arrays["plan"]))
    assert np.all(np.isnan(autonomous_arrays["served"]))
    assert autonomous["authority"] is None
    assert autonomous["engaged_fraction"] == 0.0 and autonomous["mean_r"] is None


def test_a_mode_arm_is_composed_exactly_as_the_session_is(tmp_path, rig, monkeypatch):
    """``mars-f`` reaches the loop as the session's composition: the table's string is cleared for
    ``make_steering`` and handed to the loop as ``adapter_from_corruption``, and the ``mars-f`` sync
    writes the resolved matrix into the steering policy's adapter holder."""
    path = _config(tmp_path, ["frs", "mars-f"], order="blocked", trials=1)
    _operator(monkeypatch, "y", "y")

    study.run(load_config(path), prompt="pick", config_path=path)

    configured = load_config(path)
    (frs_bundle, frs_steering), (mars_f_bundle, mars_f_steering) = rig.made
    assert frs_steering == configured.modes["frs"]
    assert mars_f_steering == replace(configured.modes["mars-f"], adapter_matrix=None)
    # The steering policy's holder gets F (the identity for a clean readout) every episode.
    assert len(rig.steerings[0].adapter.matrices) == 0  # plain frs has no adapter sync
    assert len(rig.steerings[1].adapter.matrices) == 1  # one episode: one matrix


def test_a_trial_resets_once_and_its_digest_is_the_scene_the_first_action_saw(tmp_path, rig, monkeypatch):
    """One reset per trial, and the record's ``q0``/``scene_hash`` are that observation: the digest is
    the scene the trial's first action was computed on, which a second ``reset()`` could change."""
    station = _ChangingStation()
    monkeypatch.setattr(cli, "C3poStation", lambda *args: station)
    path = _config(tmp_path, ["autonomous", "frs"], trials=1, max_steps=2)
    _operator(monkeypatch, "y", "y")

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    records = _records(run_dir)
    assert station.resets == len(records) == 2  # exactly one reset per trial
    for record, observation in zip(records, station.observed, strict=True):
        assert record["q0"] == observation["follower/joint_position"].tolist()
        assert record["scene_hash"] == study.scene_digest(observation, "scene_zed")
        arrays = _npz(run_dir, record["arm"], record["pair"])
        # Tick 0's action was computed from the reset observation; tick 1's from the station's step.
        np.testing.assert_array_equal(arrays["q"][0], observation["follower/joint_position"])
        np.testing.assert_array_equal(arrays["q"][1], JOINT)


def test_a_task_arm_hands_its_checkpoint_to_the_bundle_and_the_steering_policy(tmp_path, rig, monkeypatch):
    """A ``mars-t`` arm composes like the session: ``load_bundle(config, task=...)`` carries the
    checkpoint's scene into the bundle, and ``make_steering`` gets the table that names it."""
    checkpoint = tmp_path / "task.pt"
    checkpoint.write_bytes(b"a task checkpoint the fake loader only records")
    path = tmp_path / "franka.toml"
    path.write_text(
        f'[modes.mars-t]\nmode = "frs"\nadapter_task = "{checkpoint}"\n\n'
        f'[study]\narms = ["mars-t"]\ntrials = 1\nmax_steps = 2\nout = "{tmp_path}/study"\n',
        encoding="utf-8",
    )
    _operator(monkeypatch, "y")

    study.run(load_config(path), prompt="pick", config_path=path)

    assert rig.tasks == [checkpoint]
    assert rig.made[0][1].adapter_task == checkpoint


def test_the_scene_digest_is_stable_for_equal_observations_and_differs_otherwise():
    """sha256 of the rounded pose and a 32x32 frame: equal resets pair, a moved joint or a changed
    frame does not."""
    digest = study.scene_digest(_observation(), "scene_zed")
    assert len(digest) == 64
    assert digest == study.scene_digest(_observation(), "scene_zed")
    assert digest != study.scene_digest(_observation(frame_value=7), "scene_zed")
    moved = _observation()
    moved["follower/joint_position"] = JOINT + np.array([1e-2, 0, 0, 0, 0, 0, 0])
    assert digest != study.scene_digest(moved, "scene_zed")
    # The pose is rounded: a change below 1e-4 rad is not a different scene.
    unchanged = _observation()
    unchanged["follower/joint_position"] = JOINT + np.array([1e-6, 0, 0, 0, 0, 0, 0])
    assert digest == study.scene_digest(unchanged, "scene_zed")


def test_the_authority_of_a_hand_built_row_sequence():
    """The formula on known rows: half-way 0.5, clipped to 0 and 1, idle and undefined rows excluded."""
    executed = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # projects to 0.5 of the way to the command
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # idle: not engaged
            [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # behind the plan: clipped to 0
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # past the command: clipped to 1
        ]
    )
    plan = np.zeros((4, 8))
    served = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    measured = study.metrics(executed, plan, served, 0.02)

    assert measured.authority == pytest.approx((0.5 + 0.0 + 1.0) / 3.0)
    assert measured.engaged_fraction == pytest.approx(3.0 / 4.0)
    assert measured.mean_r == pytest.approx(1.0)


def test_the_authority_skips_a_step_where_plan_and_command_coincide():
    """``p = r`` has nothing to decide: the step is not measured, not called authority zero."""
    executed = np.array([[0.5] * 7 + [1.0], [3.0, 0, 0, 0, 0, 0, 0, 0.0]])
    plan = np.array([[1.0] * 7 + [0.0], [0.0] * 8])
    served = np.array([[1.0] * 7, [1.0, 0, 0, 0, 0, 0, 0]])

    measured = study.metrics(executed, plan, served, 0.02)

    assert measured.authority == pytest.approx(1.0)  # only the second row is measurable
    assert measured.engaged_fraction == pytest.approx(1.0)


def test_an_aborted_block_still_writes_the_trials_it_finished(tmp_path, rig, monkeypatch):
    """``q`` at the outcome stops the block: the finished trial's line and npz are on disk, and the
    trial whose outcome was never entered is not recorded."""
    path = _config(tmp_path, ["autonomous", "frs"], order="blocked", trials=2)
    _operator(monkeypatch, "y", "q")

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    records = _records(run_dir)
    assert [(record["pair"], record["arm"]) for record in records] == [(0, "autonomous")]
    assert len(list((run_dir / "trials").glob("*.npz"))) == 1
    provenance = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (provenance["completed"], provenance["aborted"]) == (1, True)
    assert rig.station.closed
    assert rig.station.resets == 2  # the second trial ran before its outcome was asked


def test_eof_at_the_outcome_stops_the_block_before_the_first_record(tmp_path, rig, monkeypatch):
    """EOF is an abort too: the run directory and ``run.json`` exist, and the record is empty."""
    path = _config(tmp_path, ["autonomous"], trials=1)
    _operator(monkeypatch, EOFError())

    run_dir = study.run(load_config(path), prompt="pick", config_path=path)

    assert _records(run_dir) == []
    assert (run_dir / "trials.jsonl").is_file()
    provenance = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (provenance["completed"], provenance["aborted"]) == (0, True)


def test_the_runner_without_a_config_path_marks_the_provenance_absent(tmp_path, rig, monkeypatch):
    """``run``'s required signature has no config path: the copy and the digest are then absent, and
    the records say null rather than guessing a file."""
    path = _config(tmp_path, ["autonomous"], trials=1)
    _operator(monkeypatch, "y")

    run_dir = study.run(load_config(path), prompt="pick", out=tmp_path / "runs")

    provenance = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert provenance["config_path"] is None and provenance["config_sha256"] is None
    assert not (run_dir / "config.toml").exists()
    assert _records(run_dir)[0]["config"] is None
    assert _records(run_dir)[0]["config_sha256"] is None


def test_a_block_without_arms_or_with_a_duplicate_arm_is_refused(tmp_path, rig):
    """A study is a block of distinct arms: an absent one cannot run, and a duplicate would double the
    pairing key the analysis relies on."""
    empty = tmp_path / "empty.toml"
    empty.write_text('[modes.frs]\nmode = "frs"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no study block"):
        study.run(load_config(empty), prompt="pick")

    duplicate = _config(tmp_path, ["frs", "frs"], trials=1)
    with pytest.raises(ValueError, match="named more than once"):
        study.run(load_config(duplicate), prompt="pick")


# ---------------------------------------------------------------- the command


def test_main_runs_the_block_prints_the_directory_and_returns_zero(tmp_path, rig, monkeypatch, capsys):
    path = _config(tmp_path, ["autonomous"], trials=1)
    _operator(monkeypatch, "y")

    assert study.main(["--config", str(path), "--prompt", "pick"]) == 0

    assert len(list((tmp_path / "study").glob("*/trials.jsonl"))) == 1
    assert "study:" in capsys.readouterr().out


def test_main_returns_130_when_the_operator_stops_the_block(tmp_path, rig, monkeypatch, capsys):
    path = _config(tmp_path, ["autonomous", "frs"], order="blocked", trials=2)
    _operator(monkeypatch, "y", "q")

    assert study.main(["--config", str(path), "--prompt", "pick"]) == 130

    assert "stopped early" in capsys.readouterr().err


def test_main_needs_its_prompt(tmp_path):
    with pytest.raises(SystemExit) as exit:
        study.main(["--config", str(_config(tmp_path, ["autonomous"], trials=1))])
    assert exit.value.code == 2
