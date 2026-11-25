"""``mars-franka <command>``: the one command surface and the one composition root.

This is the only place real objects are built. ``--dry-run`` builds the fakes instead (a
``station.FakeStation`` of synthesized observations for the configured arm and camera feeds, a
``source.FakeSource`` and ``policy.fake_bundle()``), never imports c3po or torch and never loads a
checkpoint; it always traces, so every action the loop would send is printed. It also needs no
statistics asset: if ``[policy] stats`` is absent, teleop's noise uses a generated stand-in std and
the run says so (a real run still fails on the missing file).

``--record-states NAME`` keeps the chunk-boundary states a session visits, in
``<config.states.dir>/NAME/states.npz`` (``statefiles.Recorder``): the recorder is handed to the loop,
which appends one pre-action row per chunk in every mode, and this module saves it after the run and
prints its path and count. It composes with ``--record`` (the dataset and the states describe the same
session) and every session command takes it; ``probe`` records nothing. The states' ``meta`` names the
config path and its SHA-256, the policy and statistics paths, the station, the effective readout and a
timestamp, so a collected file can be replayed against the config that produced it.

``teleop`` is the operator alone (no bundle, no steering), ``autonomous`` the policy alone, and
``fc``/``frs``/``mars-f``/``mars-u``/``mars-t``/``mars`` read their ``[modes.<command>]`` table
and steer the policy with it; a missing table is named with how to add it, and a mode whose adapter
checkpoint is absent names that file (the shipped config ships ``mars-u``, ``mars-t`` and the composed
``mars``, whose checkpoints Phase B's ``train-user`` and Phase C's ``train-task`` write).

Any other first positional that names a ``[modes.<name>]`` table runs that table exactly as a shipped
steered command does (bundle + ``--prompt`` + the ``[readout]``): this is how Phase D's "policy + task
module, no steering" anchor (``mode = "none"``, ``adapter_task = ...``) is run. A name that is neither
a fixed command nor a mode table is refused, naming the fixed commands and the mode tables the config
has.

A mode that names ``adapter_task`` hands that checkpoint to ``policy.load_bundle(config, task=...)``,
so the bundle carries the task module's recorded scene step and the loop adds the scene camera's
frame to every chunk's batch; a mode without one loads the bundle exactly as before. ``--prompt`` is
required by every policy command, optional for teleop and probe. ``teleop`` without ``--corrupted`` uses the
**clean** readout (no corruption, no noise), the demonstration-collection path; ``teleop --corrupted``
and every shared command use the config's ``[readout]``. The loop owns the readout and prints what it
resolves; this module only chooses the config. ``--record NAME`` switches to
``Loop.run_recording`` and prints the dataset path it returns.

``train-user`` and ``validate-user`` (DESIGN.md's Phase B) and ``train-task`` (Phase C) are the
trainers and the offline validator, not sessions: they take ``--config`` and ``--prompt`` only and read
their input, output and checkpoint paths from their own table (``[user]`` or ``[task]``);
``train-user`` calls ``train_user.train``, ``validate-user`` calls ``validate.validate``, and
``train-task`` calls ``train_task.train``, printing the file each writes. All three load the checkpoint,
so their imports are lazy (another command's path never pulls torch); they print their own progress,
and ``validate-user`` its table.

Phase D adds the two commands that read a block instead of running one session. ``study`` runs what
``[study]`` describes: rounds of trials over its arms, a scene reset and a y/n outcome question per
trial, and the run's records under ``<[study] out>/<timestamp>_<config stem>/`` (``trials.jsonl``, one
``trials/<arm>-<pair>.npz`` per trial, the config copy and ``run.json``). ``study.main`` builds every
arm here (:func:`compose_arm`), so an arm is a fixed session command or a ``[modes.<name>]`` table and
nothing new; a new ``Loop`` per trial reuses the arm's bundle. ``analyse`` reads one or more run
directories (or a directory holding them) and prints and writes the per-arm table and the paired
tests; it needs no config, so its only flag is its own repeatable ``--run``. Neither command is a
session: ``--steps``, ``--record``, ``--record-states``, ``--trace``, ``--dry-run`` and ``--corrupted``
are refused (the block comes from ``[study]``, the runs from ``--run``), and their imports are lazy.

``probe`` is a later slice: its import is lazy, and it records nothing, so ``--record`` is
refused. Ctrl-C is caught here, and only here, after the station and the source are closed; 130 is
returned. A user-facing failure (config, composition, a missing device/asset, missing c3po) prints one
line to stderr and exits 2; any other exception keeps its traceback, because that one is a bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from bridge import ARM_DIMS
from config import STEERED_COMMANDS, Config, ReadoutConfig, SteeringConfig, load_config, mode_tables
from loop import Loop
from numpy.typing import NDArray
from policy import fake_bundle, load_bundle, make_steering
from source import FakeSource, SpaceMouseSource
from statefiles import Recorder
from station import C3poStation, FakeStation
from stats import CHECKPOINT_WIDTH, STATION_DIMS

EXAMPLE_DIR = Path(__file__).parent  # the shipped config and every relative asset path are beside cli.py

DEFAULT_CONFIG = EXAMPLE_DIR / "configs" / "franka.toml"

NON_SESSION_COMMANDS = ("train-user", "validate-user", "train-task")
"""The commands that are not a control session: no station, no source, no loop (the Phase B/C trainers and validator)."""
COMMANDS = (
    "autonomous",
    "teleop",
    *STEERED_COMMANDS,
    *NON_SESSION_COMMANDS,
    "study",
    "analyse",
    "probe",
)
"""Every fixed command: the sessions, the Phase B/C tools, the Phase D block and analysis, and ``probe``."""

_PROMPT_OPTIONAL = ("teleop", "probe", "analyse")
"""The commands that run without a task text; every other command requires ``--prompt``."""

_SESSION_FLAGS = ("record", "record_states", "steps", "trace", "corrupted", "dry_run")
"""The flags a session's run reads; every command that is not a session refuses them."""

_NON_SESSION_SECTION = {"train-task": "[task]"}
"""The config table a non-session command's input and output come from (``[user]`` when not named)."""

_DRY_RUN_ACTION_STD = 0.3
"""One stand-in action standard deviation per dim, when a dry run has no statistics file."""


@dataclass(frozen=True)
class _Method:
    """One command resolved to the loop's mode and, for the shared commands, its mode table."""

    loop_mode: str
    steering: SteeringConfig | None = None
    adapter: str | None = None
    """``"corruption"``/``"corruption_inverse"`` when the table names a 7x7 the loop must resolve."""


# ---------------------------------------------------------------- the command line


def main(argv: list[str] | None = None) -> int:
    """Run one command; 0 on success, 2 for a user error, 130 after Ctrl-C."""
    parser, args = _parse_args(argv)
    try:
        if args.command == "analyse":
            return _analyse(args)
        if args.command == "probe":
            return _probe(args)
        if args.command == "study":
            return _study(args)
        config = load_config(args.config)
        _check_command(args, config, parser)
        _print_session(args, config)
        if args.command == "train-user":
            return _train_user(args, config)
        if args.command == "validate-user":
            return _validate_user(args, config)
        if args.command == "train-task":
            return _train_task(args, config)
        return _run(args, config)
    except KeyboardInterrupt:  # only here: the run has closed the station and the source on the way out
        print("\nmars-franka: interrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError) as error:  # config, composition, missing device/asset/c3po
        print(f"mars-franka: {error}", file=sys.stderr)
        return 2


def _parse_args(argv: list[str] | None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        prog="mars-franka",
        description="Drive the Franka station with pi0.5: one MARS method, teleop or autonomous.",
    )
    parser.add_argument(
        "command",
        metavar="COMMAND",
        help="what to run: a fixed command, or a [modes.<name>] table run as a steered command",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"the config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--prompt", metavar="TEXT", help="the task text; required for the policy commands")
    parser.add_argument("--record", metavar="NAME", help="record episodes as NAME and print the dataset path")
    parser.add_argument(
        "--record-states",
        metavar="NAME",
        dest="record_states",
        help="record chunk-boundary states as NAME (in [states] dir) and print the states path",
    )
    parser.add_argument("--steps", type=_positive_int, metavar="N", help="stop run() after N control steps")
    parser.add_argument("--trace", action="store_true", help="print the loop's per-tick line")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="fake station, source and bundle: no c3po, no torch, no checkpoint; every action traced",
    )
    parser.add_argument(
        "--corrupted",
        action="store_true",
        help="teleop only: apply [readout]; plain teleop commands a clean readout",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        metavar="DIR",
        help="analyse only: a run directory or a directory of them (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="analyse only: the directory the CSVs are written to (default: beside the run directories)",
    )
    args = parser.parse_args(argv)
    # The default config is filled in here, but whether the flag was given is kept: analyse takes none.
    args.config_explicit = args.config is not None
    if args.config is None:
        args.config = DEFAULT_CONFIG
    _check_args(args, parser)
    return parser, args


def _check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """The command surface's rules argparse cannot state, each named for the flag to change.

    The prompt rule covers the fixed commands here; a generic ``[modes.<name>]`` name is checked once
    the config is read (``_check_command``), because only the file says whether it is a mode at all.
    """
    if args.run is not None and args.command != "analyse":
        parser.error("--run is analyse only: it names the run directories the analysis reads")
    if args.out is not None and args.command != "analyse":
        parser.error("--out is analyse only: it names the directory the analysis writes its CSVs to")
    if args.command in COMMANDS and args.command not in _PROMPT_OPTIONAL and args.prompt is None:
        parser.error(f"--prompt is required for {args.command!r}")
    if args.command in NON_SESSION_COMMANDS:
        given = [f"--{name.replace('_', '-')}" for name in _SESSION_FLAGS if getattr(args, name)]
        if given:
            section = _NON_SESSION_SECTION.get(args.command, "[user]")
            parser.error(
                f"{args.command} takes --config and --prompt only ({given[0]} is not one): "
                f"{section} names its input and its output"
            )
    if args.command == "study":
        given = [f"--{name.replace('_', '-')}" for name in _SESSION_FLAGS if getattr(args, name)]
        if given:
            parser.error(
                f"study takes --config and --prompt only ({given[0]} is not one): [study] names the "
                f"block, its rounds and its cap"
            )
    if args.command == "analyse":
        given = [f"--{name.replace('_', '-')}" for name in _SESSION_FLAGS if getattr(args, name)]
        if args.config_explicit:
            given.insert(0, "--config")
        if args.prompt is not None:
            given.append("--prompt")
        if given:
            parser.error(
                f"analyse takes --run DIR only ({given[0]} is not one): it reads the run records, "
                f"not a config or a session"
            )
        if not args.run:
            parser.error("analyse needs --run DIR (repeatable): the run directories to read")
    if args.corrupted and args.command != "teleop":
        parser.error("--corrupted is teleop only: the shared modes always apply [readout]")
    if args.command == "probe" and (args.record is not None or args.record_states is not None):
        parser.error("probe records nothing: it prints the station and checkpoint facts and moves nothing")


def _check_command(args: argparse.Namespace, config: Config, parser: argparse.ArgumentParser) -> None:
    """The first positional: a fixed command, or a ``[modes.<name>]`` table the config has.

    Generic dispatch is what runs a study's own composition (Phase D's "policy + task module, no
    steering" anchor: ``mode = "none"`` with ``adapter_task``); it is a steered command in every other
    way, so it requires ``--prompt`` like ``fc``/``frs``. A name that is neither is a usage error
    naming the fixed commands and the mode tables this config actually has.
    """
    if args.command in COMMANDS:
        return
    if args.command in config.modes:
        if args.prompt is None:
            parser.error(f"--prompt is required for {args.command!r}")
        return
    parser.error(
        f"unknown command {args.command!r}: expected one of {', '.join(COMMANDS)}, or a mode table "
        f"({mode_tables(config.modes)})"
    )


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _probe(args: argparse.Namespace) -> int:
    """``probe.py`` is a later slice: imported lazily, and assumed to take the same ``--config`` flag.

    While it does not exist the import simply raises; the probe is the hardware bring-up command, so
    it is the one command a machine without the station cannot run.
    """
    from probe import main as probe_main

    return int(probe_main(["--config", str(args.config)]) or 0)


def _train_user(args: argparse.Namespace, config: Config) -> int:
    """``train-user`` over the config's ``[user]`` states and output; the trainer prints its own progress.

    The trainer loads the checkpoint (torch, its processor pipelines), so it is imported here, lazily,
    like the probe: no other command's path pulls either.
    """
    import train_user

    path = train_user.train(config, prompt=args.prompt, states_path=config.user.states, out=config.user.out)
    print(f"user module: {path}", flush=True)
    return 0


def _validate_user(args: argparse.Namespace, config: Config) -> int:
    """``validate-user`` over the config's ``[user]`` states and checkpoint; the validator prints its own table.

    Lazy import for the same reason as the trainer's: the validator loads the checkpoint (torch, its
    processor pipelines), and only this command's path needs it. The output path is ``[user] out``, so
    ``validation.json`` lands beside the checkpoint its run compares.
    """
    import validate

    result = validate.validate(
        config, prompt=args.prompt, states_path=config.user.states, checkpoint=config.user.checkpoint
    )
    print(f"validation: {result['out']}", flush=True)
    return 0


def _train_task(args: argparse.Namespace, config: Config) -> int:
    """``train-task`` over the config's ``[task]`` demonstrations and output; the trainer prints its progress.

    Lazy import for the same reason as the other non-session commands: the trainer loads the checkpoint
    (torch, its processor pipelines) and the scene encoder, so no other command's path should pull them.
    """
    import train_task

    path = train_task.train(config, prompt=args.prompt, demos_path=config.task.demos, out=config.task.out)
    print(f"task module: {path}", flush=True)
    return 0


def _study(args: argparse.Namespace) -> int:
    """``study``: run the block ``[study]`` describes and print the run directory.

    Phase D's runner is its own module (``study.py``): the block, the outcome entry and the records are
    not a session's concerns, and its import is lazy so only this command's path pulls it. The command
    hands it the same config file and prompt the session commands take; everything else -- the arms,
    the round count, the cap, the output parent -- is the ``[study]`` table's. The runner returns 130
    when the operator stopped the block early (its records are complete for the trials it finished).
    """
    import study

    return study.main(["--config", str(args.config), "--prompt", args.prompt])


def _analyse(args: argparse.Namespace) -> int:
    """``analyse``: read the runs' records and print (and write) the tables.

    No config and no station: ``analysis.py`` reads only the ``trials.jsonl`` files under the ``--run``
    directories (or under a directory holding them). Lazy import, so no session's path pulls it.
    """
    import analysis

    argv = [argument for run in args.run for argument in ("--run", str(run))]
    if args.out is not None:
        argv += ["--out", str(args.out)]
    return analysis.main(argv)


# ---------------------------------------------------------------- composition


@dataclass(frozen=True)
class Arm:
    """One command composed as the loop runs it: the loop's mode, its config, and its policy pieces.

    ``loop_config`` is the config the loop runs under (plain teleop's clean readout, everything else
    the file's), ``adapter_from_corruption`` the name a ``mars-f`` table's 7x7 is resolved by, and
    ``bundle``/``steering`` the policy objects (None for teleop, None steering for autonomous). The CLI
    builds one arm per run; the study builds each arm once and runs a new ``Loop`` over it per trial.
    """

    command: str
    loop_mode: str
    loop_config: Config
    bundle: Any = None
    steering: Any = None
    adapter_from_corruption: str | None = None


def compose_arm(
    command: str,
    config: Config,
    config_path: Path = DEFAULT_CONFIG,
    *,
    dry_run: bool = False,
    corrupted: bool = False,
) -> Arm:
    """The command's whole composition, in one place: what a session and a study arm both run.

    A study's arm is a command name, and DESIGN.md freezes that it is "composed exactly as the CLI
    composes it"; this function is that sentence. ``dry_run`` builds the fake bundle and steering and
    never touches a checkpoint (the CLI's dry runs); a study never asks for it. ``corrupted`` is
    teleop's clean-readout choice: a session's flag, and the study's arms are always the file's readout.
    """
    method = _method(command, config, config_path)
    bundle, steering = _policy(method, config, dry_run=dry_run, command=command)
    return Arm(
        command=command,
        loop_mode=method.loop_mode,
        loop_config=_loop_config(command, corrupted=corrupted, config=config),
        bundle=bundle,
        steering=steering,
        adapter_from_corruption=method.adapter,
    )


def _run(args: argparse.Namespace, config: Config) -> int:
    """Build the command's objects, close the devices whichever way the run ends, and print the dataset."""
    arm = compose_arm(args.command, config, args.config, dry_run=args.dry_run, corrupted=args.corrupted)
    prompt = "" if args.prompt is None else args.prompt
    with ExitStack() as stack:
        source = None
        if arm.loop_mode != "autonomous":  # autonomous is the policy alone: no operator device
            if not args.dry_run and config.spacemouse.tare:
                print(
                    "spacemouse: leave the stick at rest while the source opens (it captures the tare)",
                    flush=True,
                )
            source = stack.enter_context(closing(open_source(config, args.dry_run)))
            print_source(source)
        station = stack.enter_context(closing(open_station(config, args.dry_run)))
        loop_config = arm.loop_config
        if args.dry_run:
            loop_config = _dry_run_statistics(stack, loop_config, arm.loop_mode)
        states = _recorder(args, config, loop_config)
        loop = Loop(
            config=loop_config,
            station=station,
            mode=arm.loop_mode,
            source=source,
            bundle=arm.bundle,
            steering=arm.steering,
            adapter_from_corruption=arm.adapter_from_corruption,
        )
        if args.record is not None:
            path = loop.run_recording(
                name=args.record,
                dest=config.recording.dir,
                prompt=prompt,
                trace=args.trace,
                states=states,
            )
            print(f"dataset: {path}", flush=True)
        else:
            loop.run(
                prompt=prompt,
                max_steps=args.steps,
                trace=args.trace or args.dry_run,
                states=states,
            )
        if states is not None:
            _save_states(states, config, args.record_states)
    return 0


def _method(command: str, config: Config, config_path: Path) -> _Method:
    """The command as the loop's mode: teleop, autonomous, or a steered mode's table."""
    if command == "teleop":
        return _Method(loop_mode="teleop")
    if command == "autonomous":
        return _Method(loop_mode="autonomous")
    steering = config.modes.get(command)
    if steering is None:
        raise ValueError(
            f"{config_path}: command {command!r} has no [modes.{command}] table; add one, e.g.\n"
            f'[modes.{command}]\nmode = "frs"'
        )
    adapter = steering.adapter_matrix if isinstance(steering.adapter_matrix, str) else None
    return _Method(loop_mode="steered", steering=steering, adapter=adapter)


def _policy(method: _Method, config: Config, *, dry_run: bool, command: str) -> tuple[Any, Any]:
    """The bundle and steering policy the mode needs: fakes for a dry run, the real ones otherwise.

    A real run of a mode whose table names ``adapter_task`` loads that task module's checkpoint into the
    bundle (``load_bundle(config, task=<path>)``): the bundle then carries the checkpoint's recorded
    scene (``Bundle.scene``), so the loop includes the scene camera's frame in every chunk's batch and
    ``prepare`` encodes it into the tokens the task module reads. The module itself is still the steering
    policy's to load from the same path, so this adds the scene step and nothing else.

    A mode without a task adapter calls ``load_bundle(config)`` exactly as before. A dry run builds the
    fake bundle and never touches a checkpoint (no torch), so a shipped-but-not-yet-trained checkpoint
    does not stop it -- the same contract as ``mars-u``'s user module.
    """
    if method.loop_mode == "teleop":
        return None, None
    bundle = fake_bundle() if dry_run else _loaded_bundle(method, config, command)
    if method.loop_mode == "autonomous":
        return bundle, None
    if dry_run:
        return bundle, _DrySteering()
    return bundle, make_steering(bundle, _effective_steering(method))


def _loaded_bundle(method: _Method, config: Config, command: str) -> Any:
    """``load_bundle`` with the mode's task checkpoint when it names one, exactly as before when it does not."""
    task = _task_checkpoint(method.steering, command)
    return load_bundle(config) if task is None else load_bundle(config, task=task)


def _task_checkpoint(steering: SteeringConfig | None, command: str) -> Path | None:
    """The mode's task-module checkpoint, or None when the table names no ``adapter_task``.

    The path is the config layer's: a relative one resolves beside the example directory, as every other
    config path does. The file is checked before the base checkpoint is loaded, so an absent Phase C asset
    is one line naming the checkpoint and the command that writes it (``train-task``), never a traceback
    from reading a checkpoint that is not there.
    """
    if steering is None or steering.adapter_task is None:
        return None
    path = Path(steering.adapter_task)
    path = path if path.is_absolute() else EXAMPLE_DIR / path
    if not path.is_file():
        raise ValueError(
            f"[modes.{command}] adapter_task: {path} is not a file. Phase C's `mars-franka train-task "
            "--prompt <task>` writes it from [task] demos ([task] out names where)"
        )
    return path


def _effective_steering(method: _Method) -> SteeringConfig:
    """The table ``make_steering`` gets.

    A named ``adapter_matrix`` needs ``q0``, which the loop has, so the table reaches it with that
    field cleared and the loop gets the name; literal 7x7 rows go through unchanged.
    """
    if method.adapter is not None:
        return replace(method.steering, adapter_matrix=None)
    return method.steering


class _DrySteering:
    """The fake bundle returns its fixed action before any steering call, so this only satisfies the
    loop's construction contract and carries the adapter slot the ``mars-f`` sync writes."""

    def __init__(self) -> None:
        self.adapter = SimpleNamespace(matrix=None)


def open_source(config: Config, dry_run: bool = False) -> Any:
    """The operator's device: a fake for a dry run, the configured SpaceMouse otherwise ("" = search).

    Public because the study builds one source for its whole block (the loop resets the latch per
    episode); a session builds one for its single run.
    """
    if dry_run:
        return FakeSource()
    mouse = config.spacemouse
    return SpaceMouseSource(
        path=mouse.path or None,
        hid_id=mouse.hid_id or None,
        full_scale=mouse.full_scale,
        rotation_full_scale=mouse.rotation_full_scale,
        source=mouse.source,
        sign=mouse.sign,
        tare=mouse.tare,
    )


def print_source(source: Any) -> None:
    """The run header's device line: the node the source took and whether it tared (a fake names none).

    The chosen node is only known after the search has opened the device, so this prints once the
    source exists; the rest of the header is printed before the devices are opened.
    """
    describe = getattr(source, "describe", None)
    if describe is not None:
        print(f"source: {describe()}", flush=True)


def open_station(config: Config, dry_run: bool = False) -> Any:
    """The station: the fake for a dry run, c3po otherwise (the single optional hardware dependency).

    Public for the same reason as :func:`open_source`: a study resets one station for every trial of the
    block, so it opens one.
    """
    if dry_run:
        return FakeStation(_dry_observations(config), config.station.rate)
    return C3poStation(config.station.host, config.station.port, config.station.rate)


def _dry_observations(config: Config) -> list[dict[str, NDArray]]:
    """One synthesized observation for a dry run: the configured arm's state and camera feeds.

    The state is zeros at the configured arm's keys and each feed is a zero uint8 frame at a plausible
    480x640 size, so the loop's batch builder runs the real 32-wide state and letterbox path.
    """
    arm = config.station.arm
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    observation: dict[str, NDArray] = {
        f"{arm}/joint_position": np.zeros(ARM_DIMS),
        f"{arm}/gripper_position": np.zeros(1),
        config.cameras.exterior: frame.copy(),
        config.cameras.wrist: frame.copy(),
    }
    if config.cameras.third is not None:
        observation[config.cameras.third] = frame.copy()
    return [observation]


def _loop_config(command: str, corrupted: bool, config: Config) -> Config:
    """The config the loop runs under: plain teleop's clean readout, everything else the file's.

    The loop always applies ``config.readout``; plain teleop is the demonstration-collection path, so
    the CLI hands it the explicit off state rather than relying on the file's opinion.
    """
    if command == "teleop" and not corrupted:
        return replace(config, readout=ReadoutConfig(corruption=None, noise=0.0))
    return config


def _recorder(args: argparse.Namespace, config: Config, loop_config: Config) -> Recorder | None:
    """The session's state recorder, when ``--record-states`` asked for one; the CLI saves it at exit."""
    if args.record_states is None:
        return None
    return Recorder(_states_meta(args.config, config, loop_config))


def _states_meta(config_path: Path, config: Config, loop_config: Config) -> dict[str, object]:
    """The provenance a states file carries: what replaying its rows has to be reproduced against.

    The readout is ``loop_config``'s -- the effective one the loop ran -- because plain teleop runs the
    clean readout even though the file's ``[readout]`` may say otherwise. The digest is of the config
    file's bytes, so a file can be matched to the exact config that collected it.
    """
    return {
        "config": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "policy": config.policy.path,
        "stats": str(config.policy.stats),
        "station": {
            "host": config.station.host,
            "port": config.station.port,
            "arm": config.station.arm,
            "rate": config.station.rate,
        },
        "corruption": loop_config.readout.corruption,
        "noise": loop_config.readout.noise,
        "seed": loop_config.readout.seed,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _save_states(recorder: Recorder, config: Config, name: str) -> None:
    """Write ``<config.states.dir>/NAME/states.npz`` and print its path and state count."""
    path = recorder.save(Path(config.states.dir) / name / "states.npz")
    count = len(recorder)
    print(f"states: {path} ({count} {'state' if count == 1 else 'states'})", flush=True)


def _dry_run_statistics(stack: ExitStack, config: Config, loop_mode: str) -> Config:
    """A dry run's stand-in for an absent openpi statistics file, when teleop's noise would read it.

    ``readout.noise`` is in action-standard-deviation units and teleop's loop reads the scale from
    ``[policy] stats`` directly (teleop must not load the checkpoint). The file is gitignored, and a
    dry run on a fresh clone must still compose every command, so an absent file is replaced with a
    generated one whose action std is ``_DRY_RUN_ACTION_STD`` per station dim; a real run keeps
    failing loudly on the missing asset because this only ever runs under ``--dry-run``.
    """
    stats = Path(config.policy.stats)
    if loop_mode != "teleop" or not config.readout.noise or stats.is_file():
        return config
    directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="mars-franka-dry-run-"))
    path = Path(directory) / "norm_stats.json"
    path.write_text(json.dumps(_stand_in_stats()), encoding="utf-8")
    print(
        f"dry run: no statistics file at {stats}; noise uses a stand-in std {_DRY_RUN_ACTION_STD:g}",
        flush=True,
    )
    return replace(config, policy=replace(config.policy, stats=path))


def _stand_in_stats() -> dict:
    """The 32-wide openpi shape, meaningful in the 8 station dims: ``q99 - q01 = 2 * std``."""
    q99 = [2.0 * _DRY_RUN_ACTION_STD] * STATION_DIMS + [0.0] * (CHECKPOINT_WIDTH - STATION_DIMS)
    group = {
        "mean": [0.0] * CHECKPOINT_WIDTH,
        "std": [1.0] * CHECKPOINT_WIDTH,
        "q01": [0.0] * CHECKPOINT_WIDTH,
        "q99": q99,
    }
    return {"norm_stats": {"actions": group, "state": group}}


def _print_session(args: argparse.Namespace, config: Config) -> None:
    """The session header: what this run is, before anything moves."""
    prompt = "(none)" if args.prompt is None else repr(args.prompt)
    policy = "fake_bundle (no checkpoint)" if args.dry_run else str(config.policy.path)
    print(
        f"mars-franka {args.command}: config {args.config} prompt {prompt} "
        f"station {config.station.host}:{config.station.port} at {config.station.rate:g} Hz, "
        f"policy {policy} on {config.policy.device}, n_action_steps {config.policy.n_action_steps}",
        flush=True,
    )
    if args.command == "teleop":
        readout = (
            "the config's [readout] (--corrupted)" if args.corrupted else "clean (no corruption, no noise)"
        )
        print(f"readout: {readout}", flush=True)
    if args.dry_run:
        print(
            "dry run: FakeStation + FakeSource + fake_bundle; trace is on, every action is printed",
            flush=True,
        )
