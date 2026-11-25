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
and steer the policy with it; a missing table is named with how to add it (the shipped config
ships only the methods whose checkpoints exist). ``--prompt`` is required by every policy
command, optional for teleop and probe. ``teleop`` without ``--corrupted`` uses the **clean**
readout (no corruption, no noise), the demonstration-collection path; ``teleop --corrupted`` and
every shared command use the config's ``[readout]``. The loop owns the readout and prints what it
resolves; this module only chooses the config. ``--record NAME`` switches to
``Loop.run_recording`` and prints the dataset path it returns.

``train-user`` (DESIGN.md's Phase B) is the user-module trainer, not a session: it takes
``--config`` and ``--prompt`` only, reads the states and the output path from ``[user]``, and calls
``train_user.train``. It loads the checkpoint, so its import is lazy (a dry run never pulls torch);
the trainer itself prints its split, cue and per-log lines, and this module prints the checkpoint's
path at the end.

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
from config import Config, ReadoutConfig, SteeringConfig, load_config
from loop import Loop
from numpy.typing import NDArray
from policy import fake_bundle, load_bundle, make_steering
from source import FakeSource, SpaceMouseSource
from statefiles import Recorder
from station import C3poStation, FakeStation
from stats import CHECKPOINT_WIDTH, STATION_DIMS

EXAMPLE_DIR = Path(__file__).parent  # the shipped config and every relative asset path are beside cli.py

DEFAULT_CONFIG = EXAMPLE_DIR / "configs" / "franka.toml"

STEERED_COMMANDS = ("fc", "frs", "mars-f", "mars-u", "mars-t", "mars")
NON_SESSION_COMMANDS = ("train-user",)
"""The commands that are not a control session: no station, no source, no loop (Phase B's trainer)."""
COMMANDS = ("autonomous", "teleop", *STEERED_COMMANDS, *NON_SESSION_COMMANDS, "probe")

_PROMPT_OPTIONAL = ("teleop", "probe")
"""The commands that run without a task text; every other command requires ``--prompt``."""

_TRAIN_ONLY_FLAGS = ("record", "record_states", "steps", "trace", "corrupted", "dry_run")
"""The session flags ``train-user`` refuses: ``[user]`` names its input and output, and it loads the real checkpoint."""

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
    args = _parse_args(argv)
    if args.command == "probe":
        return _probe(args)
    try:
        config = load_config(args.config)
        _print_session(args, config)
        if args.command in NON_SESSION_COMMANDS:
            return _train_user(args, config)
        return _run(args, config)
    except KeyboardInterrupt:  # only here: the run has closed the station and the source on the way out
        print("\nmars-franka: interrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError) as error:  # config, composition, missing device/asset/c3po
        print(f"mars-franka: {error}", file=sys.stderr)
        return 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mars-franka",
        description="Drive the Franka station with pi0.5: one MARS method, teleop or autonomous.",
    )
    parser.add_argument("command", choices=COMMANDS, help="what to run")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
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
    args = parser.parse_args(argv)
    _check_args(args, parser)
    return args


def _check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """The command surface's rules argparse cannot state, each named for the flag to change."""
    if args.command not in _PROMPT_OPTIONAL and args.prompt is None:
        parser.error(f"--prompt is required for {args.command!r}")
    if args.corrupted and args.command != "teleop":
        parser.error("--corrupted is teleop only: the shared modes always apply [readout]")
    if args.command == "probe" and (args.record is not None or args.record_states is not None):
        parser.error("probe records nothing: it prints the station and checkpoint facts and moves nothing")
    if args.command in NON_SESSION_COMMANDS:
        given = [f"--{name.replace('_', '-')}" for name in _TRAIN_ONLY_FLAGS if getattr(args, name)]
        if given:
            parser.error(
                f"{args.command} takes --config and --prompt only ({given[0]} is not one): "
                "[user] names its input and its output"
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


# ---------------------------------------------------------------- composition


def _run(args: argparse.Namespace, config: Config) -> int:
    """Build the command's objects, close the devices whichever way the run ends, and print the dataset."""
    method = _method(args.command, config, args.config)
    bundle, steering = _policy(method, config, dry_run=args.dry_run)
    prompt = "" if args.prompt is None else args.prompt
    with ExitStack() as stack:
        source = None
        if method.loop_mode != "autonomous":  # autonomous is the policy alone: no operator device
            source = stack.enter_context(closing(_source(config, args.dry_run)))
        station = stack.enter_context(closing(_station(config, args.dry_run)))
        loop_config = _loop_config(args.command, args.corrupted, config)
        if args.dry_run:
            loop_config = _dry_run_statistics(stack, loop_config, method.loop_mode)
        states = _recorder(args, config, loop_config)
        loop = Loop(
            config=loop_config,
            station=station,
            mode=method.loop_mode,
            source=source,
            bundle=bundle,
            steering=steering,
            adapter_from_corruption=method.adapter,
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


def _policy(method: _Method, config: Config, *, dry_run: bool) -> tuple[Any, Any]:
    """The bundle and steering policy the mode needs: fakes for a dry run, the real ones otherwise."""
    if method.loop_mode == "teleop":
        return None, None
    bundle = fake_bundle() if dry_run else load_bundle(config)
    if method.loop_mode == "autonomous":
        return bundle, None
    if dry_run:
        return bundle, _DrySteering()
    return bundle, make_steering(bundle, _effective_steering(method))


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


def _source(config: Config, dry_run: bool) -> Any:
    """The operator's device: a fake for a dry run, the configured SpaceMouse otherwise ("" = search)."""
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
    )


def _station(config: Config, dry_run: bool) -> Any:
    """The station: the fake for a dry run, c3po otherwise (the single optional hardware dependency)."""
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
