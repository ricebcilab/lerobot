"""The one control loop: one station step per action, per-chunk batches, readout, recording pattern.

No c3po, no torch, no checkpoint: the station is ``station.FakeStation``, the bundle and the steering
policy are counting fakes, and the command source is ``source.py``'s ``FakeSource``. The tests check the
contract the composition root relies on — one ``station.step`` per action and none elsewhere, one
``bundle.prepare`` per chunk and one ``bundle.act`` per tick with the chunk's held command, the bridge's
readout for a known deflection, the seeded corruption draws, the ``mars-f`` adapter sync, and c3po's
documented recording episode pattern.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import bridge
import numpy as np
import pytest
from bridge import ARM_DIMS, TWIST_DIMS
from config import CommandConfig, Config, ReadoutConfig
from loop import Loop
from policy import STATION_DIMS, fake_bundle, teleop_action
from scene import scene_images
from source import FakeSource
from statefiles import Recorder, load
from station import SCENE_IMAGES, FakeStation, observation_batch

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"

TRANSLATION_GAIN = CommandConfig().translation_gain
ROTATION_GAIN = CommandConfig().rotation_gain
TWIST_LENGTH = CommandConfig().twist_length

JOINT = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.75])
"""A DROID-like reset pose: the IK's ``q``, and the default corruption lift's ``q0``."""

Q0 = tuple(float(value) for value in JOINT)

DEFLECTION = np.array([0.2, -0.1, 0.3, 0.05, -0.05, 0.1])
"""A known stick deflection, in the source's device axes ([x, y, z, rx, ry, rz])."""


# ---------------------------------------------------------------- the fakes


class _Station(FakeStation):
    """``FakeStation`` plus the per-step hook a test scripts."""

    def __init__(self, observations, rate: float = 15.0) -> None:
        super().__init__(observations, rate)
        self.on_step = None
        self.resets = 0
        self.steps = 0

    def reset(self) -> dict[str, np.ndarray]:
        self.resets += 1
        return super().reset()

    def step(self, action):
        observation = super().step(action)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self)
        return observation


class _Scene:
    """A bundle's scene as the loop reads it: only the render size matters (the bundle encodes it)."""

    def __init__(self, render_size: int) -> None:
        self.render_size = render_size


class _CountingBundle:
    """A numpy-only bundle that records the tick's halves; ``act`` returns a fixed action.

    ``scene`` is ``None`` (no scene step: the raw batch is the station's) or a task checkpoint's scene
    (the loop adds the scene camera's pictures to the raw batch; the bundle's ``prepare`` would encode
    them, which this fake records instead).
    """

    def __init__(self, std: np.ndarray | None = None, scene: _Scene | None = None) -> None:
        self.std = np.linspace(0.2, 0.9, STATION_DIMS) if std is None else np.asarray(std)
        self.scene = scene
        self.batches: list[dict] = []
        self.commands: list[np.ndarray | None] = []
        self.resets = 0

    def prepare(self, raw_batch: dict) -> dict:
        self.batches.append(raw_batch)
        return {"prepared": len(self.batches)}

    def act(self, batch: dict, command7: np.ndarray | None = None) -> np.ndarray:
        self.commands.append(None if command7 is None else np.array(command7, copy=True))
        return np.linspace(-0.3, 0.4, STATION_DIMS)

    def reset(self) -> None:
        self.resets += 1


class _Adapter:
    """A fake steering policy's adapter holder: records every matrix the loop assigns."""

    def __init__(self) -> None:
        self.matrices: list[np.ndarray] = []

    @property
    def matrix(self) -> np.ndarray | None:
        return self.matrices[-1] if self.matrices else None

    @matrix.setter
    def matrix(self, value) -> None:
        self.matrices.append(np.array(value, copy=True))


def _steering() -> SimpleNamespace:
    return SimpleNamespace(adapter=_Adapter())


class _ScriptedRecording:
    """c3po's ``Recording`` flags, driven by ``(episode, tick, flag)`` entries the test writes.

    ``advance`` is called after each station step and sets the named flag when the schedule matches;
    ``clear`` is c3po's: it resets ``done``/``rerecord`` (never ``stop``) and starts the next episode.
    """

    def __init__(self, schedule) -> None:
        self.schedule = list(schedule)
        self.episode = 0
        self.tick = 0
        self.stop = self.done = self.rerecord = False
        self.clears = 0

    def advance(self) -> None:
        self.tick += 1
        for episode, tick, flag in self.schedule:
            if episode == self.episode and tick == self.tick:
                setattr(self, flag, True)

    def clear(self) -> None:
        self.clears += 1
        self.done = self.rerecord = False
        self.episode += 1
        self.tick = 0

    def __enter__(self) -> _ScriptedRecording:
        return self

    def __exit__(self, *exception) -> None:
        pass


class _RecordingStation(_Station):
    """``_Station`` plus c3po's recording surface: ``record``, ``download`` and ``cancel_episode``."""

    def __init__(self, observations, schedule) -> None:
        super().__init__(observations)
        self.schedule = list(schedule)
        self.recording: _ScriptedRecording | None = None
        self.record_names: list[str] = []
        self.downloads: list[tuple[str, Path]] = []
        self.cancels = 0

    def record(self, name: str) -> _ScriptedRecording:
        self.record_names.append(name)
        self.recording = _ScriptedRecording(self.schedule)
        return self.recording

    def download(self, name: str, dest: Path) -> Path:
        self.downloads.append((name, Path(dest)))
        path = Path(dest) / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cancel_episode(self) -> None:
        self.cancels += 1

    def step(self, action):
        observation = super().step(action)
        if self.recording is not None:
            self.recording.advance()
        return observation


TELEOP_STD = np.linspace(0.5, 4.0, STATION_DIMS)
"""The synthetic statistics file's action q99 (and so its per-dim standard deviation, halved)."""


def _stats_file(path: Path) -> Path:
    """A hermetic ``norm_stats.json``: 32 wide, meaningful in the 8 station dims (``stats.action_std``'s
    layout check), with an action q99 a known multiple away from zero."""
    zeros = np.zeros(32 - STATION_DIMS)
    q01 = np.concatenate([np.zeros(STATION_DIMS), zeros])
    q99 = np.concatenate([TELEOP_STD, zeros])

    def group() -> dict:
        return {
            "mean": q01.tolist(),
            "std": np.ones(32).tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }

    payload = {"norm_stats": {"state": group(), "actions": group()}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _observations(
    count: int = 3, joint: np.ndarray | None = None, joints: list[np.ndarray] | None = None
) -> list[dict]:
    """A fake station's canned observations: two camera feeds and the arm state.

    ``joints`` gives each canned observation its own pose (the chunk test changes ``q`` mid-chunk);
    otherwise every observation is at ``joint`` (the default: the DROID-like ``JOINT``).
    """
    poses = [JOINT if joint is None else np.asarray(joint)] * count if joints is None else joints
    return [
        {
            "follower/joint_position": np.asarray(pose, dtype=np.float64).copy(),
            "follower/gripper_position": np.zeros(1),
            "scene_zed": np.full((8, 8, 3), 10 * index, dtype=np.uint8),
            "wrist_zed": np.full((8, 8, 3), 20 * index, dtype=np.uint8),
        }
        for index, pose in enumerate(poses)
    ]


def _config(*, n_action_steps: int = 4, readout: ReadoutConfig | None = None, command=None) -> Config:
    config = Config()
    return replace(
        config,
        policy=replace(config.policy, n_action_steps=n_action_steps),
        readout=config.readout if readout is None else readout,
        command=config.command if command is None else command,
    )


def _expected_served(deflection, q, *, spec=None, q0=None, gain=None, sigma=None, rng=None) -> np.ndarray:
    """The bridge's chain for ``deflection`` at ``q``, with the corruption lifted as the loop lifts it."""
    twist = np.asarray(deflection, dtype=np.float64).copy()
    twist[:3] *= TRANSLATION_GAIN
    twist[3:] *= ROTATION_GAIN
    intent = bridge.operator_command(twist, q, TWIST_LENGTH)
    if spec is None:
        return bridge.readout(intent, None, sigma, 1.0 if gain is None else gain, rng)
    f7 = bridge.corruption_lift(bridge.matrix6_from_spec(spec), q0, TWIST_LENGTH)
    auto = 1.0 / float(np.linalg.norm(f7, 2))
    return bridge.readout(intent, f7, sigma, auto if gain is None else gain, rng)


# ---------------------------------------------------------------- teleop


def test_teleop_sends_one_station_step_per_tick_and_records_the_teleop_action():
    """c3po owns the clock: exactly one ``step(action8)`` per action, and teleop's action is the
    served command plus the binarized gripper, with no policy in the chain."""
    station = _Station(_observations())
    source = FakeSource(deflection=DEFLECTION)
    loop = Loop(config=_config(n_action_steps=4), station=station, mode="teleop", source=source)

    loop.run(prompt="pick the cube", max_steps=6)

    assert station.steps == 6
    assert len(station.actions) == 6
    expected = teleop_action(_expected_served(DEFLECTION, JOINT), False)
    for action in station.actions:
        np.testing.assert_allclose(action, expected)
    assert source.reads == 6  # teleop reads the stick every tick


def test_teleop_carries_the_operator_gripper_latch_into_the_action():
    station = _Station(_observations())
    source = FakeSource(deflection=DEFLECTION)
    station.on_step = lambda fake: setattr(source, "gripper_closed", fake.steps >= 1)
    loop = Loop(config=_config(), station=station, mode="teleop", source=source)

    loop.run(prompt="pick", max_steps=3)

    assert [int(action[ARM_DIMS]) for action in station.actions] == [0, 1, 1]
    expected = _expected_served(DEFLECTION, JOINT)
    for action, closed in zip(station.actions, (False, True, True), strict=True):
        np.testing.assert_allclose(action, teleop_action(expected, closed))


# ---------------------------------------------------------------- the chunk cadence


def test_steered_prepares_once_per_chunk_and_holds_the_boundary_command():
    """One ``prepare`` per chunk and one ``act`` per tick; the command read at the chunk's first tick
    is held for the chunk, so a stick moved mid-chunk does not move the executed command until the
    next boundary."""
    station = _Station(_observations())
    source = FakeSource(deflection=DEFLECTION)
    bundle = _CountingBundle()
    changed = np.array([-0.3, 0.2, 0.0, 0.0, 0.0, 0.0])

    def change_mid_chunk(fake: _Station) -> None:
        if fake.steps == 1:  # after the first tick, two ticks before the next boundary
            source.deflection_vector = changed

    station.on_step = change_mid_chunk
    loop = Loop(
        config=_config(n_action_steps=4),
        station=station,
        mode="steered",
        source=source,
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick the cube", max_steps=8)

    assert len(bundle.batches) == 2
    assert len(bundle.commands) == 8
    assert source.reads == 2  # the stick is read at the two chunk boundaries only
    assert bundle.resets == 1
    for command in bundle.commands[:4]:
        np.testing.assert_allclose(command, _expected_served(DEFLECTION, JOINT))
    for command in bundle.commands[4:]:
        np.testing.assert_allclose(command, _expected_served(changed, JOINT))


def test_steered_holds_one_served_command_for_the_chunk_when_q_and_the_stick_move():
    """The whole served command is the chunk boundary's, not each tick's: moving the stick *and* the
    arm pose mid-chunk changes neither what ``act`` receives nor the trace's ``cmd[...]``."""
    station = _Station(_observations(joints=[JOINT, JOINT + 0.1, JOINT + 0.2]))
    source = FakeSource(deflection=DEFLECTION)
    bundle = _CountingBundle()
    changed = np.array([-0.3, 0.2, 0.0, 0.0, 0.0, 0.0])

    def change_mid_chunk(fake: _Station) -> None:
        if fake.steps == 1:  # after the first tick, two ticks before the next boundary
            source.deflection_vector = changed

    station.on_step = change_mid_chunk
    loop = Loop(
        config=_config(n_action_steps=4),
        station=station,
        mode="steered",
        source=source,
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick the cube", max_steps=8, trace=True)

    assert len(bundle.commands) == 8
    first, second = bundle.commands[0], bundle.commands[4]
    for command in bundle.commands[:4]:
        np.testing.assert_array_equal(command, first)  # identical, not merely close
    for command in bundle.commands[4:]:
        np.testing.assert_array_equal(command, second)
    np.testing.assert_allclose(first, _expected_served(DEFLECTION, JOINT))  # tick 0's observation
    np.testing.assert_allclose(second, _expected_served(changed, JOINT + 0.1))  # tick 4's observation
    assert not np.allclose(first, second)
    assert source.reads == 2  # the stick is read at the two chunk boundaries only


def test_the_trace_prints_the_held_command_and_no_operator_gripper(capsys):
    """The trace's ``cmd[...]`` is the command in force, i.e. the held per-chunk one, and the gripper
    column is ``-`` in the shared modes: that button is not what moves the gripper there."""
    station = _Station(_observations())
    source = FakeSource(deflection=DEFLECTION, gripper=True)
    loop = Loop(
        config=_config(n_action_steps=2),
        station=station,
        mode="steered",
        source=source,
        bundle=_CountingBundle(),
        steering=_steering(),
    )

    loop.run(prompt="pick", max_steps=4, trace=True)

    lines = [line for line in capsys.readouterr().out.splitlines() if " cmd[" in line]
    assert len(lines) == 4
    held = " ".join(f"{value:+.3f}" for value in _expected_served(DEFLECTION, JOINT))
    for line in lines:
        assert f"cmd[{held}]" in line
        assert line.endswith("grip -")


def test_the_trace_prints_the_operators_gripper_in_teleop(capsys):
    station = _Station(_observations())
    source = FakeSource(deflection=DEFLECTION)

    def close_after_first_step(fake: _Station) -> None:
        if fake.steps >= 1:
            source.gripper_closed = True

    station.on_step = close_after_first_step
    loop = Loop(config=_config(), station=station, mode="teleop", source=source)

    loop.run(prompt="pick", max_steps=2, trace=True)

    lines = [line for line in capsys.readouterr().out.splitlines() if " cmd[" in line]
    assert [line.rsplit("grip ", 1)[1] for line in lines] == ["0", "1"]


def test_the_noise_is_drawn_once_per_chunk_not_once_per_tick(monkeypatch):
    """The chunk boundary draws the noise: the same seed's commands are the same for 2 chunks whether
    the chunk is 2 or 4 ticks (and the readout is drawn twice in both), so a seeded run's noise does
    not depend on ``n_action_steps``."""
    draws = []
    real = bridge.readout

    def spy(intent7, matrix7, noise_std7, gain=1.0, rng=None):
        draws.append(1)
        return real(intent7, matrix7, noise_std7, gain, rng)

    monkeypatch.setattr(bridge, "readout", spy)

    def chunk_commands(n_action_steps: int) -> tuple[list[np.ndarray], int]:
        draws.clear()
        bundle = _CountingBundle()
        loop = Loop(
            config=_config(n_action_steps=n_action_steps, readout=ReadoutConfig(noise=0.3, seed=7)),
            station=_Station(_observations()),
            mode="steered",
            source=FakeSource(deflection=DEFLECTION),
            bundle=bundle,
            steering=_steering(),
        )
        loop.run(prompt="pick", max_steps=2 * n_action_steps)  # two chunks either way
        return [bundle.commands[0], bundle.commands[n_action_steps]], len(draws)

    short, short_draws = chunk_commands(2)
    long, long_draws = chunk_commands(4)

    assert short_draws == 2 and long_draws == 2  # one readout per chunk
    for first, second in zip(short, long, strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.allclose(short[0], short[1])  # the noise is still there


def test_the_raw_batch_is_the_stations_observation_under_the_configs_cameras():
    """The batch the pre-processor sees is ``station.observation_batch``'s dict: the two configured
    feeds, the 32-wide state and the prompt; the third image key stays omitted."""
    bundle = _CountingBundle()
    station = _Station(_observations())
    loop = Loop(
        config=_config(n_action_steps=4),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick the cube", max_steps=1)

    batch = bundle.batches[0]
    assert batch["task"] == "pick the cube"
    assert set(batch) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.state",
        "task",
    }
    assert batch["observation.state"].shape == (32,)
    assert SCENE_IMAGES not in batch  # a bundle without a scene adds nothing


def test_the_raw_batch_carries_the_scene_camera_when_the_bundle_has_a_scene():
    """A task checkpoint's bundle gives the loop a scene: the configured exterior feed (the ZED2) goes
    into the raw batch letterboxed to the module's render size, uint8 channel-first, so the bundle's
    scene step can encode it; every other key is the station's."""
    observations = _observations()
    bundle = _CountingBundle(scene=_Scene(8))
    loop = Loop(
        config=_config(n_action_steps=4), station=_Station(observations), mode="autonomous", bundle=bundle
    )

    loop.run(prompt="pick", max_steps=1)

    batch = bundle.batches[0]
    assert set(batch) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.state",
        "task",
        SCENE_IMAGES,
    }
    pictures = batch[SCENE_IMAGES]
    assert pictures.shape == (3, 8, 8) and pictures.dtype == np.uint8
    # the configured exterior feed, letterboxed to the scene's render size
    np.testing.assert_array_equal(pictures, scene_images(observations[0], "scene_zed", 8))


def test_the_scene_pictures_do_not_change_the_actions():
    """The scene camera's pictures are one more input key: same station, same bundle behaviour, same
    actions sent."""
    plain, with_scene = _Station(_observations()), _Station(_observations())
    Loop(config=_config(), station=plain, mode="autonomous", bundle=_CountingBundle()).run(
        prompt="pick", max_steps=3
    )
    Loop(
        config=_config(), station=with_scene, mode="autonomous", bundle=_CountingBundle(scene=_Scene(8))
    ).run(prompt="pick", max_steps=3)

    np.testing.assert_array_equal(np.stack(plain.actions), np.stack(with_scene.actions))


def test_autonomous_prepares_once_per_chunk_and_acts_without_a_command():
    station = _Station(_observations())
    bundle = _CountingBundle()
    loop = Loop(config=_config(n_action_steps=4), station=station, mode="autonomous", bundle=bundle)

    loop.run(prompt="pick", max_steps=8)

    assert len(bundle.batches) == 2
    assert len(bundle.commands) == 8
    assert all(command is None for command in bundle.commands)


# ---------------------------------------------------------------- the readout


SPEC = {"scale": [2.0, 1.0, 1.0]}
"""(6, 6)'s translation block doubled on x: the lift through ``J(q0)`` is the loop's fixed F."""


def test_a_corruption_is_lifted_through_the_configured_q0():
    """DESIGN's chain: ``M6`` is lifted to a fixed 7x7 ``F = I + J+(q0)(M6 - I)J(q0)`` and read out
    with ``gain = auto = 1 / ||F||_2``, so a known deflection's served command is bridge's own."""
    config = _config(
        n_action_steps=1,
        readout=ReadoutConfig(corruption=SPEC, noise=0.0, seed=0),
        command=CommandConfig(q0=Q0),
    )
    station = _Station(_observations())
    bundle = _CountingBundle()
    loop = Loop(
        config=config,
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick", max_steps=1)

    expected = _expected_served(DEFLECTION, JOINT, spec=SPEC, q0=JOINT)
    np.testing.assert_allclose(bundle.commands[0], expected)
    assert not np.allclose(bundle.commands[0], _expected_served(DEFLECTION, JOINT))  # not the clean readout


def test_noise_off_hands_the_readout_no_noise_std(monkeypatch):
    """``readout.noise = 0`` means no draw at all: ``bridge.readout`` gets no sigma (not zeros)."""
    seen = []
    real = bridge.readout

    def spy(intent7, matrix7, noise_std7, gain=1.0, rng=None):
        seen.append(noise_std7)
        return real(intent7, matrix7, noise_std7, gain, rng)

    monkeypatch.setattr(bridge, "readout", spy)
    station = _Station(_observations())
    loop = Loop(
        config=_config(n_action_steps=1, readout=ReadoutConfig(noise=0.0)),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=_CountingBundle(),
        steering=_steering(),
    )

    loop.run(prompt="pick", max_steps=2)

    assert seen and all(sigma is None for sigma in seen)


def test_the_noise_std_is_the_configured_noise_in_action_std_units(monkeypatch):
    seen = []
    real = bridge.readout

    def spy(intent7, matrix7, noise_std7, gain=1.0, rng=None):
        seen.append(noise_std7)
        return real(intent7, matrix7, noise_std7, gain, rng)

    monkeypatch.setattr(bridge, "readout", spy)
    station = _Station(_observations())
    bundle = _CountingBundle()
    loop = Loop(
        config=_config(n_action_steps=1, readout=ReadoutConfig(noise=0.3)),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick", max_steps=1)

    np.testing.assert_allclose(seen[0], 0.3 * bundle.std[:ARM_DIMS])


def test_a_seeded_run_reproduces_the_noise():
    """The readout's generator is ``readout.seed``'s: the same seed gives the same commands, the same
    actions as ``bridge.readout`` draws them, and another seed does not."""
    noise, seed = 0.3, 7

    def commands(run_seed: int) -> list[np.ndarray]:
        station = _Station(_observations())
        bundle = _CountingBundle()
        loop = Loop(
            config=_config(n_action_steps=1, readout=ReadoutConfig(noise=noise, seed=run_seed)),
            station=station,
            mode="steered",
            source=FakeSource(deflection=DEFLECTION),
            bundle=bundle,
            steering=_steering(),
        )
        loop.run(prompt="pick", max_steps=6)
        return bundle.commands

    one, two, other = commands(seed), commands(seed), commands(seed + 1)
    for first, second in zip(one, two, strict=True):
        np.testing.assert_array_equal(first, second)
    assert not np.allclose(np.stack(one), np.stack(other))
    expected = _expected_served(
        DEFLECTION, JOINT, sigma=noise * _CountingBundle().std[:ARM_DIMS], rng=np.random.default_rng(seed)
    )
    np.testing.assert_allclose(one[0], expected)


def test_teleop_noise_reads_the_action_std_from_the_statistics_file(tmp_path, monkeypatch):
    """Teleop must not load the checkpoint to know what its noise means: it reads the file's action
    quantiles directly (``stats.action_std``)."""
    stats = _stats_file(tmp_path / "stats.json")
    config = _config(n_action_steps=1, readout=ReadoutConfig(noise=0.3))
    config = replace(config, policy=replace(config.policy, stats=stats))
    seen = []
    real = bridge.readout

    def spy(intent7, matrix7, noise_std7, gain=1.0, rng=None):
        seen.append(noise_std7)
        return real(intent7, matrix7, noise_std7, gain, rng)

    monkeypatch.setattr(bridge, "readout", spy)
    loop = Loop(
        config=config,
        station=_Station(_observations()),
        mode="teleop",
        source=FakeSource(deflection=DEFLECTION),
    )

    loop.run(prompt="pick", max_steps=1)

    expected = 0.3 * np.maximum(TELEOP_STD / 2.0, 1e-6)[:ARM_DIMS]
    np.testing.assert_allclose(seen[0], expected)


def test_a_bundle_without_statistics_refuses_the_noise(monkeypatch):
    """Noise is in action-standard-deviation units: a bundle with ``std=None`` has no scale for it, so
    the run fails clearly instead of silently drawing none."""
    bundle = _CountingBundle()
    bundle.std = None
    loop = Loop(
        config=_config(n_action_steps=1, readout=ReadoutConfig(noise=0.3)),
        station=_Station(_observations()),
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=bundle,
        steering=_steering(),
    )

    with pytest.raises(ValueError, match="action statistics"):
        loop.run(prompt="pick", max_steps=1)


def test_the_dry_run_bundle_carries_a_stand_in_std_for_the_noise(monkeypatch):
    """``fake_bundle`` has no checkpoint statistics, but noise must still be expressible in a dry run:
    its stand-in std (0.3 in every dim) is what ``readout.noise`` multiplies."""
    seen = []
    real = bridge.readout

    def spy(intent7, matrix7, noise_std7, gain=1.0, rng=None):
        seen.append(noise_std7)
        return real(intent7, matrix7, noise_std7, gain, rng)

    monkeypatch.setattr(bridge, "readout", spy)
    loop = Loop(
        config=_config(n_action_steps=1, readout=ReadoutConfig(noise=0.35)),
        station=_Station(_observations()),
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=fake_bundle(),
        steering=_steering(),
    )

    loop.run(prompt="pick", max_steps=1)

    np.testing.assert_allclose(seen[0], 0.35 * np.full(ARM_DIMS, 0.3))


def test_autonomous_draws_no_readout():
    """Autonomous has no operator command, so there is nothing to corrupt: a corruption/noise config
    must not fail a run that reads neither (and a fake bundle has no statistics)."""
    bundle = _CountingBundle()
    bundle.std = None
    loop = Loop(
        config=_config(
            n_action_steps=1,
            readout=ReadoutConfig(corruption={"scale": [2.0, 1.0, 1.0]}, noise=0.3),
        ),
        station=_Station(_observations()),
        mode="autonomous",
        bundle=bundle,
    )

    loop.run(prompt="pick", max_steps=2)

    assert bundle.commands == [None, None]


# ---------------------------------------------------------------- mars-f's adapter sync


def test_mars_f_follows_the_fixed_corruption_at_every_episode(tmp_path):
    """The 7x7 ``F`` of the episode goes into the steering policy's holder at each episode, so a
    re-record (or a per-trial redraw) cannot leave a stale matrix behind."""
    schedule = [(0, 1, "done"), (1, 1, "stop")]
    config = _config(
        n_action_steps=1,
        readout=ReadoutConfig(corruption=SPEC, noise=0.0, seed=0),
        command=CommandConfig(q0=Q0),
    )
    station = _RecordingStation(_observations(), schedule)
    steering = _steering()
    loop = Loop(
        config=config,
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=_CountingBundle(),
        steering=steering,
        adapter_from_corruption="corruption",
    )

    loop.run_recording(name="mars_f", dest=tmp_path / "mars_f", prompt="pick")

    f7 = bridge.corruption_lift(bridge.matrix6_from_spec(SPEC), JOINT, TWIST_LENGTH)
    assert len(steering.adapter.matrices) == 2
    for matrix in steering.adapter.matrices:
        np.testing.assert_allclose(matrix, f7)


def test_mars_f_can_hold_the_corruptions_inverse():
    config = _config(
        n_action_steps=1,
        readout=ReadoutConfig(corruption=SPEC, noise=0.0, seed=0),
        command=CommandConfig(q0=Q0),
    )
    station = _Station(_observations())
    steering = _steering()
    loop = Loop(
        config=config,
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=_CountingBundle(),
        steering=steering,
        adapter_from_corruption="corruption_inverse",
    )

    loop.run(prompt="pick", max_steps=1)

    inverse = bridge.lift_inverse(bridge.matrix6_from_spec(SPEC), JOINT, TWIST_LENGTH)
    np.testing.assert_allclose(steering.adapter.matrices[0], inverse)
    np.testing.assert_allclose(
        inverse @ bridge.corruption_lift(bridge.matrix6_from_spec(SPEC), JOINT, TWIST_LENGTH),
        np.eye(ARM_DIMS),
        atol=1e-12,
    )


def _expected_family_f7(family: dict, *, episode: int, seed: int) -> np.ndarray:
    matrix6 = bridge.matrix6_from_spec(family, np.random.default_rng([seed, episode]))
    return bridge.corruption_lift(matrix6, JOINT, TWIST_LENGTH)


def test_a_random_family_redraws_the_adapter_per_episode_for_the_seed(tmp_path):
    """A ``{random: ...}`` family draws its matrix at every episode from
    ``default_rng([readout.seed, episode])``: the same seed replays the sequence, and F changes."""
    family = {"random": {"scale": [0.5, 2.0], "rotation_deg": [0.0, 90.0]}}
    schedule = [(0, 1, "done"), (1, 1, "done"), (2, 1, "stop")]

    def matrices(seed: int, dest: Path) -> list[np.ndarray]:
        station = _RecordingStation(_observations(), schedule)
        steering = _steering()
        loop = Loop(
            config=_config(n_action_steps=1, readout=ReadoutConfig(corruption=family, noise=0.0, seed=seed)),
            station=station,
            mode="steered",
            source=FakeSource(deflection=DEFLECTION),
            bundle=_CountingBundle(),
            steering=steering,
            adapter_from_corruption="corruption",
        )
        loop.run_recording(name="random", dest=dest, prompt="pick")
        return steering.adapter.matrices

    first = matrices(5, tmp_path / "random_a")
    replay = matrices(5, tmp_path / "random_b")
    assert len(first) == 3
    for one, two in zip(first, replay, strict=True):
        np.testing.assert_array_equal(one, two)
    assert not np.allclose(first[0], first[1])
    assert not np.allclose(first[1], first[2])
    np.testing.assert_allclose(first[0], _expected_family_f7(family, episode=0, seed=5))


def test_mars_f_with_the_corruption_off_holds_the_identity():
    station = _Station(_observations())
    steering = _steering()
    loop = Loop(
        config=_config(n_action_steps=1),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=_CountingBundle(),
        steering=steering,
        adapter_from_corruption="corruption",
    )

    loop.run(prompt="pick", max_steps=1)

    np.testing.assert_allclose(steering.adapter.matrices[0], np.eye(ARM_DIMS))


# ---------------------------------------------------------------- recording


SCHEDULE = [(0, 2, "done"), (1, 1, "done"), (1, 1, "rerecord"), (2, 1, "stop")]
"""Three episodes: the first ends done at tick 2, the second is re-recorded after tick 1, the third
stops at tick 1 (``r`` alone does not end an episode: c3po's inner loop checks only done/stop)."""


def test_run_recording_follows_c3pos_episode_pattern(tmp_path):
    destination = tmp_path / "outputs" / "demos"
    station = _RecordingStation(_observations(), SCHEDULE)
    source = FakeSource(deflection=DEFLECTION)
    loop = Loop(config=_config(), station=station, mode="teleop", source=source)

    path = loop.run_recording(name="pilot", dest=destination, prompt="pick")

    assert station.record_names == ["pilot"]
    assert station.resets == 3  # reset once per episode, inside the recording context
    assert station.steps == 4  # 2 + 1 + 1 ticks, one station step each
    assert len(station.actions) == 4
    assert source.reads == 4  # teleop reads the stick every tick
    assert station.recording.clears == 3
    assert station.cancels == 1  # the re-recorded episode is discarded
    assert station.downloads == [("pilot", destination)]
    assert path == destination / "pilot"
    assert path.is_dir()
    assert destination.is_dir()


def test_run_recording_refuses_a_station_that_cannot_record(tmp_path):
    """The recording surface is a declared port: a plain FakeStation has no ``record``/``download``,
    so the loop says so instead of failing on a missing attribute mid-episode."""
    station = _Station(_observations())
    loop = Loop(config=_config(), station=station, mode="teleop", source=FakeSource(deflection=DEFLECTION))

    with pytest.raises(ValueError, match="recording station"):
        loop.run_recording(name="pilot", dest=tmp_path, prompt="pick")

    assert station.steps == 0


# ---------------------------------------------------------------- states recording


class _Recorder:
    """A fake ``statefiles.Recorder``: keeps every ``add`` the loop makes, in order."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, batch, **values) -> None:
        self.calls.append({"batch": batch, **values})

    def __len__(self) -> int:
        return len(self.calls)


def _recording_loop(mode: str, states, *, config=None, station=None):
    """One mode's composition for the recording tests: ``(loop, station, source, bundle)``."""
    station = _Station(_observations()) if station is None else station
    source = FakeSource(deflection=DEFLECTION)
    bundle = _CountingBundle()
    config = _config(n_action_steps=4) if config is None else config
    if mode == "teleop":
        return Loop(config=config, station=station, mode="teleop", source=source), station, source, bundle
    if mode == "autonomous":
        return Loop(config=config, station=station, mode="autonomous", bundle=bundle), station, source, bundle
    loop = Loop(
        config=config,
        station=station,
        mode="steered",
        source=source,
        bundle=bundle,
        steering=_steering(),
    )
    return loop, station, source, bundle


@pytest.mark.parametrize("mode", ["teleop", "autonomous", "steered"])
def test_a_recorder_gets_one_row_per_chunk_boundary_in_every_mode(mode):
    """Recording is a chunk-boundary event in all three modes, teleop included (which has no batch)."""
    recorder = _Recorder()
    loop, station, _, _ = _recording_loop(mode, recorder)

    loop.run(prompt="pick the cube", max_steps=9, states=recorder)

    assert [(call["tick"], call["episode"], call["mode"]) for call in recorder.calls] == [
        (0, 0, mode),
        (4, 0, mode),
        (8, 0, mode),
    ]
    assert all(call["prompt"] == "pick the cube" for call in recorder.calls)
    for call in recorder.calls:
        np.testing.assert_array_equal(call["joint_position"], JOINT)
        np.testing.assert_array_equal(call["gripper"], np.zeros(1))
        np.testing.assert_array_equal(call["q0"], JOINT)  # command.q0 empty: the reset pose
    assert len(station.actions) == 9  # the recording adds no station step


def test_the_steered_row_is_the_boundarys_raw_batch_and_held_command():
    """The recorded batch is the pre-processor's input, and the served command is the chunk's held one."""
    observations = _observations(joints=[JOINT, JOINT + 0.1, JOINT + 0.2])
    station = _Station(observations)
    bundle = _CountingBundle()
    recorder = _Recorder()
    loop = Loop(
        config=_config(n_action_steps=4, readout=ReadoutConfig(noise=0.0)),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=bundle,
        steering=_steering(),
    )

    loop.run(prompt="pick the cube", max_steps=8, states=recorder)

    first, second = recorder.calls
    assert first["batch"] is bundle.batches[0] and "prepared" not in first["batch"]
    assert second["batch"] is bundle.batches[1]
    expected = observation_batch(
        observations[0],
        arm="follower",
        cameras={"exterior": "scene_zed", "wrist": "wrist_zed", "third": None},
        task="pick the cube",
    )
    np.testing.assert_array_equal(
        first["batch"]["observation.images.base_0_rgb"],
        expected["observation.images.base_0_rgb"],
    )
    np.testing.assert_array_equal(first["served"], bundle.commands[0])
    np.testing.assert_array_equal(second["served"], bundle.commands[4])
    np.testing.assert_array_equal(first["deflection"], DEFLECTION)
    # the state before the action: tick 4's boundary observation is the second canned pose
    np.testing.assert_array_equal(first["joint_position"], JOINT)
    np.testing.assert_array_equal(second["joint_position"], JOINT + 0.1)


def test_a_teleop_recording_stores_the_action_it_sends():
    """Teleop has no bundle: the loop builds the raw batch for the recording alone, and the served
    command it stores is the one the tick's action carries."""
    station = _Station(_observations())
    recorder = _Recorder()
    loop = Loop(
        config=_config(n_action_steps=4),
        station=station,
        mode="teleop",
        source=FakeSource(deflection=DEFLECTION),
    )

    loop.run(prompt="pick", max_steps=8, states=recorder)

    assert len(recorder.calls) == 2
    for call, action in zip(recorder.calls, (station.actions[0], station.actions[4]), strict=True):
        np.testing.assert_array_equal(call["served"], action[:ARM_DIMS])
        np.testing.assert_array_equal(call["deflection"], DEFLECTION)
        assert call["batch"]["task"] == "pick"
        assert call["batch"]["observation.images.base_0_rgb"].shape == (3, 224, 224)


def test_an_autonomous_recording_stores_zeros_for_the_operator_columns():
    """Autonomous has no operator and no command: its rows say zeros rather than inventing one."""
    station = _Station(_observations())
    recorder = _Recorder()
    loop = Loop(
        config=_config(n_action_steps=4), station=station, mode="autonomous", bundle=_CountingBundle()
    )

    loop.run(prompt="pick", max_steps=8, states=recorder)

    assert len(recorder.calls) == 2
    for call in recorder.calls:
        np.testing.assert_array_equal(call["served"], np.zeros(ARM_DIMS))
        np.testing.assert_array_equal(call["deflection"], np.zeros(TWIST_DIMS))
        np.testing.assert_array_equal(call["q0"], JOINT)


@pytest.mark.parametrize("mode", ["teleop", "autonomous", "steered"])
def test_the_actions_are_the_same_with_and_without_a_recorder(mode, tmp_path):
    """Recording is observation only: same station, same seed, same actions, with and without it."""
    stats = _stats_file(tmp_path / "stats.json")

    def actions(states) -> list[np.ndarray]:
        config = _config(n_action_steps=3, readout=ReadoutConfig(corruption=SPEC, noise=0.3, seed=5))
        config = replace(config, policy=replace(config.policy, stats=stats))
        loop, station, _, _ = _recording_loop(mode, states, config=config)
        loop.run(prompt="pick", max_steps=7, states=states)
        return station.actions

    without = actions(None)
    recorder = _Recorder()
    with_states = actions(recorder)

    assert len(recorder.calls) == 3  # boundaries at ticks 0, 3, 6
    assert len(without) == len(with_states) == 7
    for one, two in zip(without, with_states, strict=True):
        np.testing.assert_array_equal(one, two)


def test_run_recording_records_every_episode_and_numbers_them_from_zero(tmp_path):
    """The recording loop's episodes are 0-based and each restarts its ticks; the recorder sees the
    boundaries of the episode the loop is in."""
    schedule = [(0, 2, "done"), (1, 1, "stop")]
    recorder = _Recorder()
    station = _RecordingStation(_observations(), schedule)
    loop = Loop(
        config=_config(n_action_steps=1),
        station=station,
        mode="teleop",
        source=FakeSource(deflection=DEFLECTION),
    )

    loop.run_recording(name="pilot", dest=tmp_path, prompt="pick", states=recorder)

    assert [(call["episode"], call["tick"]) for call in recorder.calls] == [(0, 0), (0, 1), (1, 0)]
    assert {call["mode"] for call in recorder.calls} == {"teleop"}
    assert station.steps == 3


RERECORD_SCHEDULE = [(0, 1, "done"), (0, 1, "rerecord"), (1, 1, "stop")]
"""Two takes: the first ends re-recorded, the second stops. With ``n_action_steps = 1`` each take
contributes one row at tick 0."""


class _DropLog(Recorder):
    """The real recorder, plus a log of every drop the loop asked it for."""

    def __init__(self, meta: dict) -> None:
        super().__init__(meta)
        self.drops: list[tuple[int, int]] = []

    def drop_episode(self, episode: int) -> int:
        removed = super().drop_episode(episode)
        self.drops.append((episode, removed))
        return removed


def test_a_re_recorded_take_leaves_no_states_behind(tmp_path):
    """c3po's ``r`` discards the take's dataset; the recorder drops the same take's rows, so the saved
    file holds only the takes the operator kept."""
    recorder = _DropLog({"session": "test"})
    station = _RecordingStation(_observations(), RERECORD_SCHEDULE)
    loop = Loop(
        config=_config(n_action_steps=1),
        station=station,
        mode="teleop",
        source=FakeSource(deflection=DEFLECTION),
    )

    loop.run_recording(name="pilot", dest=tmp_path / "dropped", prompt="pick", states=recorder)

    assert recorder.drops == [(0, 1)]  # the rejected first take's one boundary row
    assert station.cancels == 1
    arrays, _ = load(recorder.save(tmp_path / "states.npz"))
    assert len(arrays["episode"]) == 1
    assert arrays["episode"].tolist() == [1]  # only the kept take
    assert arrays["tick"].tolist() == [0]


def test_a_re_recorded_take_changes_no_actions(tmp_path):
    """Dropping the rejected take's states is observation only: the actions are the same either way."""

    def actions(states, dest) -> list[np.ndarray]:
        station = _RecordingStation(_observations(), RERECORD_SCHEDULE)
        loop = Loop(
            config=_config(n_action_steps=1),
            station=station,
            mode="teleop",
            source=FakeSource(deflection=DEFLECTION),
        )
        loop.run_recording(name="pilot", dest=dest, prompt="pick", states=states)
        return station.actions

    without = actions(None, tmp_path / "without")
    with_states = actions(_DropLog({"session": "test"}), tmp_path / "with")

    assert len(without) == len(with_states) == 2  # one action per take
    for one, two in zip(without, with_states, strict=True):
        np.testing.assert_array_equal(one, two)


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize(
    ("mode", "parts"),
    [
        ("teleop", {}),
        ("teleop", {"source": FakeSource(deflection=DEFLECTION), "bundle": _CountingBundle()}),
        ("teleop", {"source": FakeSource(deflection=DEFLECTION), "steering": _steering()}),
        ("autonomous", {}),
        ("autonomous", {"bundle": _CountingBundle(), "source": FakeSource(deflection=DEFLECTION)}),
        ("autonomous", {"bundle": _CountingBundle(), "steering": _steering()}),
        ("steered", {"source": FakeSource(deflection=DEFLECTION)}),
        ("steered", {"source": FakeSource(deflection=DEFLECTION), "bundle": _CountingBundle()}),
        ("steered", {"bundle": _CountingBundle(), "steering": _steering()}),
        (
            "steering",
            {
                "source": FakeSource(deflection=DEFLECTION),
                "bundle": _CountingBundle(),
                "steering": _steering(),
            },
        ),
    ],
)
def test_construction_refuses_a_mode_whose_parts_do_not_fit(mode, parts):
    with pytest.raises(ValueError, match=r"teleop|autonomous|steered|mode"):
        Loop(config=_config(), station=_Station(_observations()), mode=mode, **parts)


def test_construction_refuses_an_adapter_sync_the_mode_cannot_use():
    with pytest.raises(ValueError, match="adapter"):
        Loop(
            config=_config(),
            station=_Station(_observations()),
            mode="teleop",
            source=FakeSource(deflection=DEFLECTION),
            adapter_from_corruption="corruption",
        )
    with pytest.raises(ValueError, match="adapter"):
        Loop(
            config=_config(),
            station=_Station(_observations()),
            mode="steered",
            source=FakeSource(deflection=DEFLECTION),
            bundle=_CountingBundle(),
            steering=_steering(),
            adapter_from_corruption="stale",
        )


def test_each_mode_constructs_with_its_own_parts():
    station = _Station(_observations())
    assert Loop(config=_config(), station=station, mode="teleop", source=FakeSource(deflection=DEFLECTION))
    assert Loop(config=_config(), station=station, mode="autonomous", bundle=_CountingBundle())
    assert Loop(
        config=_config(),
        station=station,
        mode="steered",
        source=FakeSource(deflection=DEFLECTION),
        bundle=_CountingBundle(),
        steering=_steering(),
    )


# ---------------------------------------------------------------- the import boundary


def test_importing_the_loop_needs_neither_torch_nor_c3po():
    """The loop is on ``--dry-run``'s path: c3po is the station's, torch the policy's, and the
    SpaceMouse import lives inside ``SpaceMouseSource.__init__``."""
    code = (
        "import sys, loop; "
        "assert 'torch' not in sys.modules, 'loop imported torch'; "
        "assert 'c3po' not in sys.modules, 'loop imported c3po'; "
        "assert hasattr(loop, 'Loop')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )
