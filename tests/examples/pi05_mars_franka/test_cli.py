"""The command surface: argument rules, composition, the dry run, and the console-script shim.

No c3po, no checkpoint, no SpaceMouse: the real-run tests replace ``cli.Loop``, ``C3poStation`` and
``SpaceMouseSource`` (and ``load_bundle``/``make_steering`` for the policy commands) with recording
fakes, and the ``--dry-run`` tests drive the real composition over ``FakeStation``/``FakeSource``/
``fake_bundle``. The shim is tested through ``lerobot.scripts.mars_franka.main``, the closest thing to
``uv run mars-franka`` that does not need a reinstall.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cli
import numpy as np
import pytest
import torch
from config import load_config
from policy import Bundle
from scene import TASK_SOURCE, Scene, from_checkpoint
from source import FakeSource
from statefiles import Recorder, load
from station import FakeStation as RealFakeStation
from stats import CHECKPOINT_WIDTH

from lerobot.policies.steering import AdapterModules, LearnedTaskModule, LearnedUserModule

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "pi05" / "mars" / "franka"
SHIPPED = EXAMPLE_DIR / "configs" / "franka.toml"

MODE_COMMANDS = ("autonomous", "fc", "frs", "mars-f", "mars-u", "mars-t", "mars")
SHIPPED_COMMANDS = ("autonomous", "teleop", "fc", "frs", "mars-f", "mars-u", "mars-t", "mars")

TASK_SIZE = 128
"""The fake task checkpoints' scene render size: small, so saving one is instant (the VLM resizes to 224)."""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "franka.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _observations() -> list[dict]:
    """One canned station observation for the real loop: the arm and the two shipped camera feeds."""
    return [
        {
            "follower/joint_position": np.zeros(7),
            "follower/gripper_position": np.zeros(1),
            "scene_zed": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_zed": np.zeros((8, 8, 3), dtype=np.uint8),
        }
    ]


# ---------------------------------------------------------------- the fakes


class _CapturingFakeStation(RealFakeStation):
    """``FakeStation`` that remembers every instance and the observations it was built from."""

    instances: list[_CapturingFakeStation] = []
    observations: list[list[dict]] = []

    def __init__(self, observations, rate: float) -> None:
        type(self).observations.append(observations)
        type(self).instances.append(self)
        super().__init__(observations, rate)


@pytest.fixture
def dry_station(monkeypatch):
    _CapturingFakeStation.instances = []
    _CapturingFakeStation.observations = []
    monkeypatch.setattr(cli, "FakeStation", _CapturingFakeStation)
    return _CapturingFakeStation


class _RecordingLoop:
    """``cli.Loop``'s stand-in: records what the composition handed it and what it was asked to run."""

    instances: list[_RecordingLoop] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.runs: list[dict] = []
        self.states: list = []
        type(self).instances.append(self)

    def run(self, *, prompt: str, max_steps: int | None = None, trace: bool = False, states=None) -> None:
        self.states.append(states)
        self.runs.append({"prompt": prompt, "max_steps": max_steps, "trace": trace})

    def run_recording(self, *, name: str, dest: Path, prompt: str, trace: bool = False, states=None) -> Path:
        self.states.append(states)
        self.runs.append({"record": name, "dest": Path(dest), "prompt": prompt, "trace": trace})
        return Path(dest) / name


class _FakeC3poStation:
    """The real station's constructor and ``close``, without c3po."""

    arm = "follower"

    def __init__(self, host: str, port: int, rate: float) -> None:
        self.init = (host, port, rate)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeSpaceMouse:
    """The source's constructor and ``close``, without a device or torch."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def real_components(monkeypatch):
    """Replace the hardware and the loop; return the recording loop class."""
    _RecordingLoop.instances = []
    monkeypatch.setattr(cli, "Loop", _RecordingLoop)
    monkeypatch.setattr(cli, "C3poStation", _FakeC3poStation)
    monkeypatch.setattr(cli, "SpaceMouseSource", _FakeSpaceMouse)
    return _RecordingLoop


def _steering_bundle() -> Bundle:
    """The little ``make_steering`` reads before it refuses a missing adapter file: action statistics."""
    return Bundle(
        policy=object(),
        preprocessor=None,
        postprocessor=None,
        mean=np.zeros(8),
        std=np.full(8, 0.3),
    )


class _SamplerConfig:
    """pi0.5's sampler interface, as ``make_steering`` builds its core from it."""

    chunk_size = 8
    max_action_dim = CHECKPOINT_WIDTH
    num_inference_steps = 10
    n_action_steps = 8
    noise_time = 1.0
    type = "pi05"
    device = "cpu"


class _SamplerPolicy:
    """A base policy with the sampler contract and nothing loaded."""

    def __init__(self) -> None:
        self.config = _SamplerConfig()

    def reset(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        return torch.zeros(1, self.config.chunk_size, self.config.max_action_dim)

    def velocity_field(self, batch: dict):
        return lambda x, time: torch.zeros_like(x)


class _StatsStep:
    """An unnormalizer-like step for the core's ``action_mean_std``: action mean/std in raw units."""

    def __init__(self) -> None:
        mean = np.zeros(CHECKPOINT_WIDTH)
        std = np.full(CHECKPOINT_WIDTH, 0.25)
        self.stats = {"action": {"mean": mean, "std": std}}
        self.norm_map = {}


class _Postprocessor:
    def __init__(self) -> None:
        self.steps = [_StatsStep()]

    def __call__(self, action):
        return torch.as_tensor(action)


def _trained_bundle() -> Bundle:
    """A fake bundle the real ``make_steering`` can build its core over: sampler fields and statistics."""
    return Bundle(
        policy=_SamplerPolicy(),
        preprocessor=lambda raw: raw,
        postprocessor=_Postprocessor(),
        mean=np.zeros(8),
        std=np.full(8, 0.25),
    )


def _task_checkpoint(path: Path) -> Path:
    """A MARS task module checkpoint built with no training: its config is the scene ``load_bundle`` reads.

    ``n_dims`` is the station's 8 (the task module learns the gripper too) and its horizon the checkpoint's
    chunk, as ``train-task`` builds it; the scene is this phase's ``vlm_image``. No weights are trained.
    """
    module = LearnedTaskModule(
        horizon=_SamplerConfig.chunk_size,
        n_dims=8,
        d=16,
        tdim=8,
        heads=2,
        layers=1,
        **LearnedTaskModule.spec(TASK_SOURCE, TASK_SIZE),
    )
    return AdapterModules.save_module(module, path, kind="task", meta={"prompt": "pick"})


def _user_checkpoint(path: Path) -> Path:
    """A MARS user module checkpoint, for the composed ``mars`` arm that names both."""
    module = LearnedUserModule(_SamplerConfig.chunk_size, 7, d=16, tdim=8)
    return AdapterModules.save_module(module, path, kind="user", meta={"prompt": "pick"})


def _task_arm_config(config_dir: Path, command: str, *, task: Path, user: Path) -> Path:
    """A config whose ``[modes.<command>]`` names the saved (fake) task checkpoint, and the user too for ``mars``."""
    adapter_user = f'adapter_user = "{user}"\n' if command == "mars" else ""
    return _write(config_dir, f'[modes.{command}]\nmode = "frs"\n{adapter_user}adapter_task = "{task}"\n')


@pytest.fixture
def fake_hardware(monkeypatch):
    """The real loop over a fake station and a fake source: no c3po, no mouse, no faked loop."""
    monkeypatch.setattr(cli, "C3poStation", lambda *args: RealFakeStation(_observations(), 15.0))
    monkeypatch.setattr(cli, "SpaceMouseSource", lambda **kwargs: FakeSource())


def _patch_policy(monkeypatch) -> tuple[dict, object, object]:
    """Replace the bundle load and ``make_steering``; return their call log and the two sentinels."""
    calls: dict = {}
    bundle, steering = object(), object()

    def fake_load(config):
        calls["config"] = config
        return bundle

    def fake_make(got_bundle, steering_cfg):
        calls["bundle"] = got_bundle
        calls["steering"] = steering_cfg
        return steering

    monkeypatch.setattr(cli, "load_bundle", fake_load)
    monkeypatch.setattr(cli, "make_steering", fake_make)
    return calls, bundle, steering


# ---------------------------------------------------------------- the argument rules


@pytest.mark.parametrize("command", MODE_COMMANDS)
def test_a_policy_command_requires_a_prompt(command, capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main([command, "--config", str(SHIPPED), "--dry-run"])
    assert exit.value.code == 2
    assert "--prompt is required" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["autonomous", "fc", "probe"])
def test_corrupted_is_only_for_teleop(command, capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main([command, "--prompt", "pick", "--corrupted"])
    assert exit.value.code == 2
    assert "--corrupted is teleop only" in capsys.readouterr().err


def test_an_unknown_command_is_refused_naming_the_commands_and_the_mode_tables(capsys):
    """A first positional that is neither a fixed command nor a ``[modes.<name>]`` table is a usage
    error; it names both what the CLI accepts and what the config it read actually defines."""
    with pytest.raises(SystemExit) as exit:
        cli.main(["fly"])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "unknown command 'fly'" in error
    assert "autonomous" in error and "train-task" in error  # the fixed commands are named
    assert "[modes.fc]" in error and "[modes.mars]" in error  # and the shipped config's mode tables


def test_an_unknown_command_names_the_configs_own_mode_tables(tmp_path, capsys):
    """The tables named are the file's, not the shipped config's: a variant config's are its own."""
    path = _write(tmp_path, '[modes.task-only]\nmode = "none"\nadapter_task = "assets/task/t.pt"\n')
    with pytest.raises(SystemExit) as exit:
        cli.main(["fly", "--config", str(path), "--prompt", "pick"])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "'fly'" in error and "[modes.task-only]" in error
    assert "autonomous" in error and "[modes.fc]" not in error


GENERIC_MODE = """
[modes.task-only]
mode = "none"
adapter_task = "assets/task/pilot/task.pt"
"""


def test_a_mode_table_runs_as_a_steered_command_in_a_dry_run(dry_station, tmp_path, capsys):
    """Phase D's generic dispatch: a first positional naming a ``[modes.<name>]`` table runs that table
    exactly as ``fc``/``frs`` do (bundle + ``--prompt`` + ``[readout]``), which is how the S2
    "policy + task module, no steering" anchor (``mode = "none"``, ``adapter_task``) runs."""
    path = _write(tmp_path, GENERIC_MODE)
    assert (
        cli.main(["task-only", "--config", str(path), "--prompt", "pick", "--dry-run", "--steps", "4"]) == 0
    )
    station = dry_station.instances[-1]
    assert len(station.actions) == 4
    assert capsys.readouterr().out.count("act[") == 4


def test_a_generic_mode_command_requires_a_prompt(tmp_path, capsys):
    """A mode table is a policy command like ``fc``/``frs``: it runs a task, so its prompt is required."""
    path = _write(tmp_path, GENERIC_MODE)
    with pytest.raises(SystemExit) as exit:
        cli.main(["task-only", "--config", str(path), "--dry-run"])
    assert exit.value.code == 2
    assert "--prompt is required for 'task-only'" in capsys.readouterr().err


def test_a_generic_mode_dispatches_exactly_like_a_steered_command(real_components, tmp_path, monkeypatch):
    """The composition is the steered one: the mode table reaches ``make_steering`` and the loop gets
    the bundle, the steering policy and the ``steered`` loop mode."""
    calls, bundle, steering = _patch_policy(monkeypatch)
    path = _write(tmp_path, '[modes.task-only]\nmode = "none"\n')
    assert cli.main(["task-only", "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["mode"] == "steered"
    assert loop.kwargs["bundle"] is bundle and loop.kwargs["steering"] is steering
    assert calls["steering"] == load_config(path).modes["task-only"]
    assert calls["steering"].mode == "none"
    assert loop.runs == [{"prompt": "pick", "max_steps": 1, "trace": False}]


def test_probe_is_not_asked_to_record(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["probe", "--record", "pilot"])
    assert exit.value.code == 2
    assert "probe records nothing" in capsys.readouterr().err


# ---------------------------------------------------------------- the dry run


def test_the_shipped_config_is_the_study_config():
    config = load_config(SHIPPED)
    assert (config.station.host, config.station.port) == ("10.42.0.1", 9090)
    assert (config.station.arm, config.station.rate) == ("follower", 15.0)
    assert (config.cameras.exterior, config.cameras.wrist) == ("scene_zed", "wrist_zed")
    assert (config.policy.path, config.policy.device, config.policy.n_action_steps) == (
        "lerobot/pi05_droid",
        "cuda",
        8,
    )
    assert config.command.translation_gain == 0.11 and config.command.gain == "auto"
    assert config.readout.corruption == {"rotation_z_deg": 20} and config.readout.noise == 0.35
    # The six method arms the config ships: Phase B's user module and Phase C's task module name
    # gitignored checkpoints, so the shipped tables are the ones their trainers write to.
    assert set(config.modes) == {"fc", "frs", "mars-f", "mars-u", "mars-t", "mars"}
    assert config.modes["fc"].ic_gain == 15.0 and config.modes["mars-f"].adapter_matrix == "corruption"
    assert config.modes["mars-u"].mode == "frs"
    assert config.modes["mars-u"].adapter_user == config.user.checkpoint
    assert config.modes["mars-t"].mode == "frs"
    assert config.modes["mars-t"].adapter_task == config.task.checkpoint
    assert config.modes["mars"].mode == "frs"
    assert config.modes["mars"].adapter_user == config.user.checkpoint
    assert config.modes["mars"].adapter_task == config.task.checkpoint
    assert (config.steering.n_reversal_steps, config.steering.pinned_steps) == (10, 8)
    assert config.spacemouse.source == (0, 1, 2, 3, 4, 5)
    assert config.recording.dir == EXAMPLE_DIR / "outputs" / "mars_franka"
    # Phase B's tables: the study's calibration session and the trainer's loop
    assert config.user.states == EXAMPLE_DIR / "assets" / "states" / "pilot"
    assert config.user.out == EXAMPLE_DIR / "assets" / "user" / "rotz20"
    assert config.user.checkpoint == EXAMPLE_DIR / "assets" / "user" / "rotz20" / "user.pt"
    assert (config.user.cues, config.user.sigma) == (3, (0.2, 0.5))
    assert (config.user.step_rank, config.user.cross_step_rank, config.user.additive) == (None, 0, False)
    assert (config.training.steps, config.training.batch, config.training.lr) == (3000, 8, 1e-3)
    assert (config.training.val_frac, config.training.seed, config.training.log_every) == (0.1, 0, 100)
    assert (config.training.val_depths, config.training.device) == ((10, 4), "cuda")
    # Phase C's table: the pilot's recorded demonstrations and the task module train-task writes
    assert config.task.demos == EXAMPLE_DIR / "assets" / "demos" / "pilot"
    assert config.task.out == EXAMPLE_DIR / "assets" / "task" / "pilot"
    assert config.task.checkpoint == EXAMPLE_DIR / "assets" / "task" / "pilot" / "task.pt"
    assert (config.task.token_source, config.task.render_size) == ("vlm_image", 224)
    assert (config.task.scene_rank, config.task.d, config.task.tdim) == (None, 128, 64)
    assert config.task.lambda_id == 1e-3
    # Phase D's block: the shipped study is S2's first row, FRS vs FRS with the task module
    assert config.study.arms == ("frs", "mars-t")
    assert (config.study.trials, config.study.order, config.study.max_steps) == (10, "shuffled", 600)
    assert config.study.out == EXAMPLE_DIR / "outputs" / "mars_franka" / "study"
    assert config.study.seed == 0


@pytest.mark.parametrize("command", SHIPPED_COMMANDS)
def test_a_dry_run_composes_and_runs_every_shipped_command(command, dry_station, capsys):
    """Every command the shipped config supports runs offline for the requested steps, through the
    real loop and fake station, and the dry run prints one trace line per action."""
    argv = [command, "--config", str(SHIPPED), "--dry-run", "--steps", "4"]
    if command != "teleop":
        argv += ["--prompt", "pick the cube"]
    assert cli.main(argv) == 0
    station = dry_station.instances[-1]
    assert len(station.actions) == 4
    assert all(action.shape == (8,) for action in station.actions)
    assert capsys.readouterr().out.count("act[") == 4


def test_a_dry_run_builds_no_station_no_mouse_and_no_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a dry run built a real object")

    monkeypatch.setattr(cli, "C3poStation", forbidden)
    monkeypatch.setattr(cli, "SpaceMouseSource", forbidden)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    assert cli.main(["fc", "--prompt", "pick", "--dry-run", "--steps", "3"]) == 0


def test_a_dry_run_synthesizes_the_configured_arm_and_camera_feeds(dry_station, tmp_path):
    path = _write(
        tmp_path,
        '[station]\narm = "test_arm"\n[cameras]\nexterior = "cam_a"\nwrist = "cam_b"\nthird = "cam_c"\n',
    )
    assert cli.main(["teleop", "--config", str(path), "--dry-run", "--steps", "1"]) == 0
    observation = dry_station.observations[0][0]
    assert set(observation) == {
        "test_arm/joint_position",
        "test_arm/gripper_position",
        "cam_a",
        "cam_b",
        "cam_c",
    }
    assert observation["cam_a"].dtype == np.uint8 and observation["cam_a"].ndim == 3
    assert dry_station.instances[-1].arm == "test_arm"


def test_a_dry_run_without_the_statistics_file_uses_a_stand_in_std(dry_station, tmp_path, capsys):
    """A fresh clone has no openpi asset; teleop --corrupted must still dry-run, with a stated stand-in."""
    missing = tmp_path / "norm_stats.json"  # never written: the gitignored asset is absent
    path = _write(tmp_path, f'[policy]\nstats = "{missing}"\n[readout]\nnoise = 0.35\n')
    assert cli.main(["teleop", "--config", str(path), "--corrupted", "--dry-run", "--steps", "2"]) == 0
    out = capsys.readouterr().out
    assert f"dry run: no statistics file at {missing}; noise uses a stand-in std 0.3" in out
    assert out.count("act[") == 2
    assert len(dry_station.instances[-1].actions) == 2


def test_a_real_run_without_the_statistics_file_still_fails_naming_it(fake_hardware, tmp_path, capsys):
    """The stand-in is the dry run's alone: the real teleop path keeps failing on the missing asset."""
    missing = tmp_path / "norm_stats.json"
    path = _write(tmp_path, f'[policy]\nstats = "{missing}"\n[readout]\nnoise = 0.35\n')
    assert cli.main(["teleop", "--config", str(path), "--corrupted", "--steps", "2"]) == 2
    assert str(missing) in capsys.readouterr().err


def test_the_command_surface_imports_without_torch_or_c3po():
    """A dry run must not need either: cli's imports are the loop's, which are pure at module level."""
    code = (
        "import sys, cli; "
        "assert 'torch' not in sys.modules, 'cli imported torch'; "
        "assert 'c3po' not in sys.modules, 'cli imported c3po'; "
        "assert hasattr(cli, 'main')"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(EXAMPLE_DIR)},
    )


# ---------------------------------------------------------------- the readout choice


def test_plain_teleop_runs_the_clean_readout(real_components):
    assert cli.main(["teleop", "--config", str(SHIPPED), "--steps", "3"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["mode"] == "teleop"
    assert (loop.kwargs["config"].readout.corruption, loop.kwargs["config"].readout.noise) == (None, 0.0)
    assert loop.kwargs["bundle"] is None and loop.kwargs["steering"] is None
    assert loop.kwargs["adapter_from_corruption"] is None
    assert loop.kwargs["station"].closed and loop.kwargs["source"].closed
    assert loop.runs == [{"prompt": "", "max_steps": 3, "trace": False}]


def test_teleop_corrupted_uses_the_configs_readout(real_components):
    configured = load_config(SHIPPED)
    assert cli.main(["teleop", "--config", str(SHIPPED), "--steps", "1", "--corrupted"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["config"].readout == configured.readout
    assert loop.kwargs["config"].readout.corruption == {"rotation_z_deg": 20}


def test_a_bug_in_the_run_keeps_its_traceback_and_still_closes_the_devices(real_components, monkeypatch):
    """Expected failures are one-line errors; an unexpected exception is a bug and propagates, after
    the run's context has closed the station and the source."""

    def broken_run(self, **kwargs):
        raise RuntimeError("a bug in the loop")

    monkeypatch.setattr(real_components, "run", broken_run)
    with pytest.raises(RuntimeError, match="a bug in the loop"):
        cli.main(["teleop", "--config", str(SHIPPED), "--steps", "1"])
    loop = real_components.instances[-1]
    assert loop.kwargs["station"].closed and loop.kwargs["source"].closed


# ---------------------------------------------------------------- the policy commands


@pytest.mark.parametrize(
    ("command", "mode"),
    [("fc", "flow_control"), ("frs", "frs"), ("mars-f", "frs"), ("mars-u", "frs")],
)
def test_a_steered_command_builds_its_mode_table(command, mode, real_components, monkeypatch):
    calls, bundle, steering = _patch_policy(monkeypatch)
    assert cli.main([command, "--prompt", "pick", "--config", str(SHIPPED), "--steps", "1"]) == 0
    loop = real_components.instances[-1]
    configured = load_config(SHIPPED)
    expected = configured.modes[command]
    if command == "mars-f":  # the string is the loop's to resolve, not make_steering's
        expected = replace(expected, adapter_matrix=None)
        assert loop.kwargs["adapter_from_corruption"] == "corruption"
        assert configured.modes[command].adapter_matrix == "corruption"
    else:
        assert loop.kwargs["adapter_from_corruption"] is None
    assert calls["steering"] == expected
    assert calls["steering"].mode == mode
    assert calls["bundle"] is bundle and calls["config"].policy.path == configured.policy.path
    assert loop.kwargs["mode"] == "steered"
    assert loop.kwargs["bundle"] is bundle and loop.kwargs["steering"] is steering
    assert loop.kwargs["station"].closed and loop.kwargs["source"].closed


def test_autonomous_loads_the_bundle_and_has_no_operator(real_components, monkeypatch):
    calls, bundle, _ = _patch_policy(monkeypatch)
    assert cli.main(["autonomous", "--prompt", "pick", "--config", str(SHIPPED), "--steps", "2"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["mode"] == "autonomous"
    assert loop.kwargs["bundle"] is bundle
    assert loop.kwargs["source"] is None and loop.kwargs["steering"] is None
    assert "steering" not in calls  # make_steering is never reached


def test_mars_u_composes_the_real_steering_policy_from_a_saved_user_module(
    tmp_path, real_components, monkeypatch
):
    """Once ``[modes.mars-u]`` points at a trained checkpoint, the command builds the real MARS-U arm:
    the module is loaded into the steering policy's adapter and the loop gets that policy."""
    checkpoint = AdapterModules.save_module(
        LearnedUserModule(_SamplerConfig.chunk_size, 7, d=16, tdim=8),
        tmp_path / "user" / "user.pt",
        kind="user",
        meta={"prompt": "pick"},
    )
    monkeypatch.setattr(cli, "load_bundle", lambda config: _trained_bundle())
    path = _write(
        tmp_path,
        f'[modes.mars-u]\nmode = "frs"\nadapter_user = "{checkpoint}"\n',
    )
    assert cli.main(["mars-u", "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["mode"] == "steered"
    assert loop.kwargs["adapter_from_corruption"] is None
    steering = loop.kwargs["steering"]
    assert steering.config.adapter_user == str(checkpoint)
    assert isinstance(steering.adapter.user, LearnedUserModule)
    assert steering.adapter.user.horizon == _SamplerConfig.chunk_size
    assert steering.adapter.user.n_dims == 7
    assert steering.adapter.matrix is None  # mars-u is the module alone, no fixed lift


@pytest.mark.parametrize("command", ["mars-t", "mars"])
def test_a_missing_mode_table_names_it_and_how_to_add_one(command, tmp_path, capsys):
    """A variant config without the arm's table is told how to add it (the shipped config ships both)."""
    path = _write(tmp_path, '[modes.frs]\nmode = "frs"\n')
    assert cli.main([command, "--prompt", "pick", "--config", str(path), "--dry-run", "--steps", "1"]) == 2
    error = capsys.readouterr().err
    assert f"[modes.{command}]" in error
    assert "add one" in error


@pytest.mark.parametrize("command", ["autonomous", "fc", "frs", "mars-f"])
def test_a_mode_without_a_task_adapter_calls_load_bundle_exactly_as_before(
    command, real_components, monkeypatch
):
    """No ``adapter_task`` anywhere in the mode's table: ``load_bundle(config)``, with no ``task=`` keyword
    at all -- the fake loader does not accept one, so an extra argument would fail the run."""
    seen = {}

    def fake_load(config):
        seen["config"] = config
        return _trained_bundle()

    monkeypatch.setattr(cli, "load_bundle", fake_load)
    assert cli.main([command, "--prompt", "pick", "--config", str(SHIPPED), "--steps", "1"]) == 0
    assert seen["config"].policy.path == load_config(SHIPPED).policy.path
    assert real_components.instances[-1].kwargs["bundle"].scene is None


@pytest.mark.parametrize("command", ["mars-t", "mars"])
def test_a_dry_run_of_the_task_arms_loads_no_checkpoint(command, dry_station, monkeypatch, capsys):
    """The shipped task arms name the checkpoints ``train-task`` writes (gitignored, absent on a fresh
    checkout); a dry run composes them from the fakes and reads no checkpoint, as ``mars-u`` does."""

    def forbidden(*args, **kwargs):
        pytest.fail("a dry run loaded a checkpoint")

    monkeypatch.setattr(cli, "load_bundle", forbidden)
    assert cli.main([command, "--config", str(SHIPPED), "--dry-run", "--steps", "2", "--prompt", "pick"]) == 0
    assert len(dry_station.instances[-1].actions) == 2
    assert capsys.readouterr().out.count("act[") == 2


@pytest.mark.parametrize("command", ["mars-t", "mars"])
def test_the_task_arms_load_their_checkpoint_into_the_bundles_scene(
    command, tmp_path, real_components, monkeypatch
):
    """Once a mode table names ``adapter_task``, that checkpoint reaches ``load_bundle(task=...)``: the
    bundle the loop gets carries the checkpoint's recorded scene, so ``prepare`` encodes the scene camera
    and the task module reads its tokens; the same file is the steering policy's adapter."""
    task = _task_checkpoint(tmp_path / "task" / "task.pt")
    user = _user_checkpoint(tmp_path / "user" / "user.pt")
    path = _task_arm_config(tmp_path, command, task=task, user=user)
    calls = {}

    def fake_load(config, task=None):
        calls["config"] = config
        calls["task"] = task
        return replace(_trained_bundle(), scene=None if task is None else from_checkpoint(task))

    monkeypatch.setattr(cli, "load_bundle", fake_load)
    assert cli.main([command, "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 0

    configured = load_config(path)
    checkpoint = configured.modes[command].adapter_task
    loop = real_components.instances[-1]
    assert calls["config"] == configured
    assert calls["task"] == checkpoint
    bundle = loop.kwargs["bundle"]
    assert bundle.scene == Scene(TASK_SOURCE, TASK_SIZE)
    assert loop.kwargs["mode"] == "steered"
    assert loop.kwargs["steering"].config.adapter_task == str(checkpoint)
    if command == "mars":
        assert loop.kwargs["steering"].config.adapter_user == str(configured.modes[command].adapter_user)
    else:
        assert loop.kwargs["steering"].config.adapter_user is None
    assert loop.kwargs["station"].closed and loop.kwargs["source"].closed


@pytest.mark.parametrize("command", ["mars-t", "mars"])
def test_a_task_arm_names_its_missing_checkpoint_in_one_line(
    command, tmp_path, real_components, monkeypatch, capsys
):
    """``train-task`` has not run: the run names the file and the command that writes it, no traceback."""
    missing = tmp_path / "task" / "task.pt"  # never written
    path = _task_arm_config(tmp_path, command, task=missing, user=tmp_path / "user" / "user.pt")
    monkeypatch.setattr(cli, "load_bundle", lambda config: pytest.fail("loaded a checkpoint first"))
    assert cli.main([command, "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 2
    error = capsys.readouterr().err
    assert str(missing) in error and "train-task" in error
    assert "Traceback" not in error
    assert len(error.strip().splitlines()) == 1  # one line, the CLI's own


def test_a_relative_task_checkpoint_resolves_beside_the_example(tmp_path, real_components, capsys):
    """A config's relative path is the config layer's: beside the example directory, not the caller's cwd."""
    path = _write(tmp_path, '[modes.mars-t]\nmode = "frs"\nadapter_task = "assets/task/absent/task.pt"\n')
    assert cli.main(["mars-t", "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 2
    expected = EXAMPLE_DIR / "assets" / "task" / "absent" / "task.pt"
    error = capsys.readouterr().err
    assert str(expected) in error and "train-task" in error


def test_mars_u_names_a_missing_user_checkpoint(tmp_path, real_components, monkeypatch, capsys):
    """The shipped table's checkpoint is gitignored until `train-user` has run: the run names the file."""
    missing = tmp_path / "user" / "user.pt"  # never written
    monkeypatch.setattr(cli, "load_bundle", lambda config: _steering_bundle())
    path = _write(tmp_path, f'[modes.mars-u]\nmode = "frs"\nadapter_user = "{missing}"\n')
    assert cli.main(["mars-u", "--config", str(path), "--prompt", "pick", "--steps", "1"]) == 2
    error = capsys.readouterr().err
    assert str(missing) in error and "adapter_user" in error and "train-user" in error


# ---------------------------------------------------------------- the train-user command


def _user_config(tmp_path: Path) -> Path:
    """A config whose ``[user]`` names a states directory and an output under tmp_path."""
    return _write(
        tmp_path,
        f'[user]\nstates = "{tmp_path}/states/pilot"\nout = "{tmp_path}/user"\n\n[modes.frs]\nmode = "frs"\n',
    )


def test_train_user_requires_a_prompt(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["train-user", "--config", str(SHIPPED)])
    assert exit.value.code == 2
    assert "--prompt is required" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["train-user", "validate-user"])
@pytest.mark.parametrize(
    "flags",
    [["--steps", "2"], ["--record", "pilot"], ["--record-states", "pilot"], ["--dry-run"], ["--trace"]],
)
def test_the_user_commands_take_no_session_flags(command, flags, capsys):
    """``[user]`` names the trainer's and validator's input and output; a session flag has no meaning."""
    with pytest.raises(SystemExit) as exit:
        cli.main([command, "--config", str(SHIPPED), "--prompt", "pick", *flags])
    assert exit.value.code == 2
    assert "takes --config and --prompt only" in capsys.readouterr().err


def test_train_user_reaches_the_trainer_with_the_configs_values(tmp_path, monkeypatch, capsys):
    """The command is the trainer's composition root: it hands the trainer the file's ``[user] states``
    and ``out`` and the session's prompt, and prints the checkpoint the trainer returns."""
    import train_user

    config_path = _user_config(tmp_path)
    calls = []

    def fake_train(config, *, prompt, states_path, out, device=None):
        calls.append({"config": config, "prompt": prompt, "states_path": Path(states_path), "out": Path(out)})
        return Path(out) / "user.pt"

    monkeypatch.setattr(train_user, "train", fake_train)
    assert cli.main(["train-user", "--config", str(config_path), "--prompt", "pick the cube"]) == 0

    configured = load_config(config_path)
    assert configured.user.states == tmp_path / "states" / "pilot"
    assert calls == [
        {
            "config": configured,
            "prompt": "pick the cube",
            "states_path": configured.user.states,
            "out": configured.user.out,
        }
    ]
    assert f"user module: {configured.user.out / 'user.pt'}" in capsys.readouterr().out


def test_train_user_reports_a_user_error_as_one_line(tmp_path, monkeypatch, capsys):
    import train_user

    def refuses(*args, **kwargs):
        raise ValueError("[user] states: no such states file")

    monkeypatch.setattr(train_user, "train", refuses)
    assert cli.main(["train-user", "--config", str(_user_config(tmp_path)), "--prompt", "pick"]) == 2
    assert "[user] states" in capsys.readouterr().err


def test_validate_user_requires_a_prompt(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["validate-user", "--config", str(SHIPPED)])
    assert exit.value.code == 2
    assert "--prompt is required" in capsys.readouterr().err


def test_validate_user_reaches_the_validator_with_the_configs_values(tmp_path, monkeypatch, capsys):
    """The command is the validator's composition root: it hands it the file's ``[user]`` states and
    checkpoint and the session's prompt, and prints the validation JSON's path."""
    import validate

    config_path = _write(
        tmp_path,
        f'[user]\nstates = "{tmp_path}/states/pilot"\ncheckpoint = "{tmp_path}/user/user.pt"\n'
        f'out = "{tmp_path}/validation"\n',
    )
    calls = []

    def fake_validate(config, *, prompt, states_path, checkpoint, out=None):
        calls.append(
            {
                "config": config,
                "prompt": prompt,
                "states_path": Path(states_path),
                "checkpoint": Path(checkpoint),
                "out": out,
            }
        )
        return {"out": str(Path(config.user.out) / "validation.json")}

    monkeypatch.setattr(validate, "validate", fake_validate)
    assert cli.main(["validate-user", "--config", str(config_path), "--prompt", "pick the cube"]) == 0

    configured = load_config(config_path)
    assert configured.user.states == tmp_path / "states" / "pilot"
    assert calls == [
        {
            "config": configured,
            "prompt": "pick the cube",
            "states_path": configured.user.states,
            "checkpoint": configured.user.checkpoint,
            "out": None,
        }
    ]
    assert f"validation: {configured.user.out / 'validation.json'}" in capsys.readouterr().out


def test_validate_user_reports_a_user_error_as_one_line(tmp_path, monkeypatch, capsys):
    import validate

    def refuses(*args, **kwargs):
        raise ValueError("[user] checkpoint: no such user module")

    monkeypatch.setattr(validate, "validate", refuses)
    assert cli.main(["validate-user", "--config", str(_user_config(tmp_path)), "--prompt", "pick"]) == 2
    assert "[user] checkpoint" in capsys.readouterr().err


# ---------------------------------------------------------------- the train-task command


def _task_config(tmp_path: Path) -> Path:
    """A config whose ``[task]`` names a demonstrations directory and an output under tmp_path."""
    return _write(
        tmp_path,
        f'[task]\ndemos = "{tmp_path}/demos/pilot"\nout = "{tmp_path}/task"\n',
    )


def test_train_task_requires_a_prompt(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["train-task", "--config", str(SHIPPED)])
    assert exit.value.code == 2
    assert "--prompt is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [["--steps", "2"], ["--record", "pilot"], ["--record-states", "pilot"], ["--dry-run"], ["--trace"]],
)
def test_train_task_takes_no_session_flags(flags, capsys):
    """``[task]`` names the trainer's input and output; a session flag has no meaning."""
    with pytest.raises(SystemExit) as exit:
        cli.main(["train-task", "--config", str(SHIPPED), "--prompt", "pick", *flags])
    assert exit.value.code == 2
    error = capsys.readouterr().err
    assert "takes --config and --prompt only" in error and "[task]" in error


def test_train_task_reaches_the_trainer_with_the_configs_values(tmp_path, monkeypatch, capsys):
    """The command is the trainer's composition root: it hands the trainer the file's ``[task] demos`` and
    ``out`` and the session's prompt, and prints the checkpoint the trainer returns."""
    import train_task

    config_path = _task_config(tmp_path)
    calls = []

    def fake_train(config, *, prompt, demos_path, out, device=None):
        calls.append({"config": config, "prompt": prompt, "demos_path": Path(demos_path), "out": Path(out)})
        return Path(out) / "task.pt"

    monkeypatch.setattr(train_task, "train", fake_train)
    assert cli.main(["train-task", "--config", str(config_path), "--prompt", "pick the cube"]) == 0

    configured = load_config(config_path)
    assert configured.task.demos == tmp_path / "demos" / "pilot"
    assert calls == [
        {
            "config": configured,
            "prompt": "pick the cube",
            "demos_path": configured.task.demos,
            "out": configured.task.out,
        }
    ]
    assert f"task module: {configured.task.out / 'task.pt'}" in capsys.readouterr().out


def test_train_task_reports_a_user_error_as_one_line(tmp_path, monkeypatch, capsys):
    import train_task

    def refuses(*args, **kwargs):
        raise ValueError("[task] demos: no such dataset")

    monkeypatch.setattr(train_task, "train", refuses)
    assert cli.main(["train-task", "--config", str(_task_config(tmp_path)), "--prompt", "pick"]) == 2
    assert "[task] demos" in capsys.readouterr().err


def test_train_task_names_the_missing_demonstrations_dataset(tmp_path, capsys):
    """The shipped command on a fresh checkout has no demonstrations: one line names the dataset, no
    traceback (the demo set is what ``teleop --record`` downloads and ``[task] demos`` points at)."""
    missing = tmp_path / "demos" / "pilot"
    config_path = _task_config(tmp_path)
    assert cli.main(["train-task", "--config", str(config_path), "--prompt", "pick the cube"]) == 2
    error = capsys.readouterr().err
    assert str(missing) in error and "teleop --record" in error
    assert error.count("Traceback") == 0


# ---------------------------------------------------------------- the config flag and the SpaceMouse


def test_the_config_flag_chooses_another_file(real_components, tmp_path, capsys):
    path = _write(tmp_path, '[station]\nhost = "127.0.0.1"\nport = 4242\nrate = 10.0\n')
    assert cli.main(["teleop", "--config", str(path), "--steps", "1"]) == 0
    loop = real_components.instances[-1]
    assert loop.kwargs["station"].init == ("127.0.0.1", 4242, 10.0)
    assert str(path) in capsys.readouterr().out  # the header names the file the session ran


def test_the_spacemouse_table_reaches_the_source(real_components, tmp_path):
    path = _write(
        tmp_path,
        "[spacemouse]\n"
        'path = "/dev/hidraw9"\n'
        'hid_id = "0000046D:0000C62B"\n'
        "source = [3, 4, 5, 0, 1, 2]\n"
        "sign = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]\n",
    )
    assert cli.main(["teleop", "--config", str(path), "--steps", "1"]) == 0
    mouse = real_components.instances[-1].kwargs["source"]
    assert mouse.kwargs == {
        "path": "/dev/hidraw9",
        "hid_id": "0000046D:0000C62B",
        "full_scale": 350.0,
        "rotation_full_scale": 350.0,
        "source": (3, 4, 5, 0, 1, 2),
        "sign": (1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
    }


def test_an_empty_spacemouse_path_means_search(real_components):
    assert cli.main(["teleop", "--config", str(SHIPPED), "--steps", "1"]) == 0
    mouse = real_components.instances[-1].kwargs["source"]
    assert mouse.kwargs["path"] is None and mouse.kwargs["hid_id"] is None


@pytest.mark.parametrize(
    "text",
    [
        "[spacemouse]\nsource = [0, 1, 2, 3, 4]\n",
        "[spacemouse]\nsource = [0, 1, 2, 3, 4, 6]\n",
        "[spacemouse]\nsign = [1.0, 1.0, 1.0]\n",
        "[spacemouse]\nfull_scale = 0.0\n",
    ],
)
def test_a_bad_spacemouse_table_is_refused(text, tmp_path, capsys):
    assert cli.main(["teleop", "--config", str(_write(tmp_path, text)), "--steps", "1"]) == 2
    assert "[spacemouse]" in capsys.readouterr().err


# ---------------------------------------------------------------- recording


def test_record_switches_to_run_recording_and_prints_the_dataset(real_components, capsys):
    configured = load_config(SHIPPED)
    assert cli.main(["teleop", "--config", str(SHIPPED), "--record", "pilot"]) == 0
    run = real_components.instances[-1].runs[0]
    assert run["record"] == "pilot" and run["dest"] == configured.recording.dir
    assert str(configured.recording.dir / "pilot") in capsys.readouterr().out


def test_record_needs_a_recording_station_so_a_dry_run_says_so(dry_station, capsys):
    assert cli.main(["teleop", "--config", str(SHIPPED), "--dry-run", "--record", "pilot"]) == 2
    assert "recording station" in capsys.readouterr().err


# ---------------------------------------------------------------- the states flag


def _states_config(tmp_path: Path, *, n_action_steps: int | None = None) -> Path:
    """The shipped config with its states directory (and optionally its chunk) moved into tmp_path."""
    text = SHIPPED.read_text(encoding="utf-8")
    text = text.replace('dir = "assets/states"', f'dir = "{tmp_path}/states"')
    if n_action_steps is not None:
        text = text.replace("n_action_steps = 8", f"n_action_steps = {n_action_steps}")
    return _write(tmp_path, text)


@pytest.mark.parametrize("command", SHIPPED_COMMANDS)
def test_record_states_writes_and_prints_its_file_on_every_shipped_command(command, tmp_path, capsys):
    """Every session command takes the flag; a dry run collects one row per chunk boundary."""
    path = _states_config(tmp_path)
    argv = [command, "--config", str(path), "--dry-run", "--steps", "8", "--record-states", "pilot"]
    if command != "teleop":
        argv += ["--prompt", "pick the cube"]

    assert cli.main(argv) == 0

    states_path = tmp_path / "states" / "pilot" / "states.npz"
    out = capsys.readouterr().out
    assert f"states: {states_path} (1 state)" in out
    arrays, meta = load(states_path)
    assert len(arrays["episode"]) == 1  # steps 8 / n_action_steps 8: one chunk boundary
    expected_mode = (
        "teleop" if command == "teleop" else "autonomous" if command == "autonomous" else "steered"
    )
    assert arrays["mode"].tolist() == [expected_mode]
    assert meta["config"] == str(path)
    assert meta["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["policy"] == "lerobot/pi05_droid"
    assert meta["station"] == {"host": "10.42.0.1", "port": 9090, "arm": "follower", "rate": 15.0}
    assert meta["timestamp"]
    if command == "teleop":  # the clean readout is what the loop ran
        assert (meta["corruption"], meta["noise"]) == (None, 0.0)
    else:
        assert meta["corruption"] == {"rotation_z_deg": 20} and meta["noise"] == 0.35


def test_record_states_is_refused_on_probe(capsys):
    with pytest.raises(SystemExit) as exit:
        cli.main(["probe", "--record-states", "pilot"])
    assert exit.value.code == 2
    assert "probe records nothing" in capsys.readouterr().err


def test_steps_still_caps_the_run_with_record_states(tmp_path, capsys, dry_station):
    """``--steps`` still bounds the run: one state per tick with a one-tick chunk, and it stops."""
    path = _states_config(tmp_path, n_action_steps=1)
    assert (
        cli.main(["teleop", "--config", str(path), "--dry-run", "--steps", "3", "--record-states", "pilot"])
        == 0
    )
    arrays, _ = load(tmp_path / "states" / "pilot" / "states.npz")
    assert arrays["tick"].tolist() == [0, 1, 2] and arrays["episode"].tolist() == [0, 0, 0]
    assert len(dry_station.instances[-1].actions) == 3
    assert "(3 states)" in capsys.readouterr().out


def test_record_states_hands_the_loop_a_recorder_and_saves_it_after_the_run(
    real_components, tmp_path, capsys
):
    path = _states_config(tmp_path)
    assert cli.main(["teleop", "--config", str(path), "--steps", "2", "--record-states", "pilot"]) == 0
    loop = real_components.instances[-1]
    assert isinstance(loop.states[0], Recorder)  # the real recorder, not a stand-in
    states_path = tmp_path / "states" / "pilot" / "states.npz"
    assert states_path.is_file()  # saved at exit, even with no rows from the fake loop
    assert f"states: {states_path} (0 states)" in capsys.readouterr().out


def test_record_states_composes_with_record(real_components, tmp_path):
    path = _states_config(tmp_path)
    assert cli.main(["teleop", "--config", str(path), "--record", "pilot", "--record-states", "pilot"]) == 0
    loop = real_components.instances[-1]
    assert loop.runs[0]["record"] == "pilot"
    assert isinstance(loop.states[0], Recorder)


# ---------------------------------------------------------------- the console-script shim


def test_the_shim_help_exits_zero():
    from lerobot.scripts.mars_franka import main

    with pytest.raises(SystemExit) as exit:
        main(["--help"])
    assert exit.value.code == 0


def test_the_shim_runs_a_dry_run_teleoperation_end_to_end(capsys):
    from lerobot.scripts.mars_franka import main

    assert main(["teleop", "--dry-run", "--steps", "2"]) == 0
    assert capsys.readouterr().out.count("act[") == 2
