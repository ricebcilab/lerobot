"""``mars-franka <command>``: the one command surface and the one composition root.

This is the only place real objects are built. ``--dry-run`` builds the fakes instead (a
``station.FakeStation`` of synthesized observations for the configured arm and camera feeds, a
``source.FakeSource`` and ``policy.fake_bundle()``), never imports c3po and never loads a
checkpoint; it always traces, so every action the loop would send is printed.

``teleop`` is the operator alone (no bundle, no steering), ``autonomous`` the policy alone, and
``fc``/``frs``/``mars-f``/``mars-u``/``mars-t``/``mars`` read their ``[modes.<command>]`` table
and steer the policy with it; a missing table is named with how to add it (the shipped config
ships only the methods whose checkpoints exist). ``--prompt`` is required by every policy
command, optional for teleop and probe. ``teleop`` without ``--corrupted`` uses the **clean**
readout (no corruption, no noise), the demonstration-collection path; ``teleop --corrupted`` and
every shared command use the config's ``[readout]``. The loop owns the readout and prints what it
resolves; this module only chooses the config. ``--record NAME`` switches to
``Loop.run_recording`` and prints the dataset path it returns.

``probe`` is a later slice: its import is lazy, and it records nothing, so ``--record`` is
refused. Ctrl-C is caught here, and only here, after the station and the source are closed; 130 is
returned. A user-facing failure (config, composition, a missing device/asset, missing c3po) prints one
line to stderr and exits 2; any other exception keeps its traceback, because that one is a bug.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
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
from station import C3poStation, FakeStation

EXAMPLE_DIR = Path(__file__).parent  # the shipped config and every relative asset path are beside cli.py

DEFAULT_CONFIG = EXAMPLE_DIR / "configs" / "franka.toml"

STEERED_COMMANDS = ("fc", "frs", "mars-f", "mars-u", "mars-t", "mars")
COMMANDS = ("autonomous", "teleop", *STEERED_COMMANDS, "probe")

_PROMPT_OPTIONAL = ("teleop", "probe")
"""The commands that run without a task text; every other command requires ``--prompt``."""


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
    parser.add_argument("--steps", type=_positive_int, metavar="N", help="stop run() after N control steps")
    parser.add_argument("--trace", action="store_true", help="print the loop's per-tick line")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="fake station, source and bundle: no c3po, no checkpoint, every action traced",
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
    if args.command == "probe" and args.record is not None:
        parser.error("probe records nothing: it prints the station and checkpoint facts and moves nothing")


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
        loop = Loop(
            config=_loop_config(args.command, args.corrupted, config),
            station=station,
            mode=method.loop_mode,
            source=source,
            bundle=bundle,
            steering=steering,
            adapter_from_corruption=method.adapter,
        )
        if args.record is not None:
            path = loop.run_recording(
                name=args.record, dest=config.recording.dir, prompt=prompt, trace=args.trace
            )
            print(f"dataset: {path}", flush=True)
        else:
            loop.run(prompt=prompt, max_steps=args.steps, trace=args.trace or args.dry_run)
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
        return bundle, _dry_steering()
    return bundle, make_steering(bundle, _effective_steering(method))


def _effective_steering(method: _Method) -> SteeringConfig:
    """The table ``make_steering`` gets.

    A named ``adapter_matrix`` needs ``q0``, which the loop has, so the table reaches it with that
    field cleared and the loop gets the name; literal 7x7 rows go through unchanged.
    """
    if method.adapter is not None:
        return replace(method.steering, adapter_matrix=None)
    return method.steering


def _dry_steering() -> Any:
    """The adapter holder a dry run needs instead of a ``SteeringPolicy``.

    ``make_steering`` builds ``lerobot.policies.steering`` objects (torch, the checkpoint's action
    statistics), so a dry run cannot call it; ``fake_bundle`` returns its fixed action whatever the
    command, and the adapter holder is the only part of the steering policy the loop touches.
    """
    return SimpleNamespace(adapter=SimpleNamespace(matrix=None))


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
