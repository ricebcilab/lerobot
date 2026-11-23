#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Slice 0's probe: read the checkpoint's config and the station's manifest, print DECISION.md's inputs.

    python examples/pi05/mars/franka/probe.py --policy <ckpt> --station <host> --port 9090

The probe answers the questions the first decision record must answer, without loading weights and without
moving the arm: what action the checkpoint was trained on (its width, chunk, inference steps and image
keys, and whether it is relative), what the station accepts (action keys, cameras, control rate), whether a
command-free ``step(None)`` holds, and which SpaceMouse a person would hold. Nothing here is on the
runner's path; this is the one script that may be run alone.

Each half reports ``unavailable: <reason>`` instead of raising: a probe that dies on the first missing
piece cannot tell you what the other half said.

``--worktree`` names the LeRobot checkout to read; its ``src/`` goes first on ``sys.path`` so the probe
measures that worktree's config classes, not an installed copy. It defaults to the checkout this file
lives in.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from interfaces import StationManifest

POSTPROCESSOR_FILES = ("postprocessor.json", "policy_postprocessor.json")
UNNORMALIZER_STATS = "*unnormalizer_processor.safetensors"


def _display(value) -> str:
    """A field as a DECISION.md line shows it: lowercase booleans, and a field a config lacks is named."""
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _image_keys(config) -> list[str]:
    """The visual input feature keys: the cameras the checkpoint expects, in config order."""
    features = getattr(config, "input_features", None) or {}
    return [
        key
        for key, feature in features.items()
        if "image" in key or "VISUAL" in str(getattr(feature, "type", "")).upper()
    ]


def _action_shape(config) -> tuple | None:
    """The action feature's shape, which is padded to ``max_action_dim`` (only its width is meaningful)."""
    action = (getattr(config, "output_features", None) or {}).get("action")
    shape = getattr(action, "shape", None)
    return None if shape is None else tuple(shape)


def _find_action_stats(document):
    """The ``{"mean": ..., "std": ...}`` mapping under an ``"action"`` key, wherever the JSON nests it."""
    if isinstance(document, dict):
        action = document.get("action")
        if isinstance(action, dict) and "mean" in action and "std" in action:
            return action
        children = document.values()
    elif isinstance(document, list):
        children = document
    else:
        return None
    for child in children:
        found = _find_action_stats(child)
        if found is not None:
            return found
    return None


def _width(values) -> int | str:
    """How many entries a mean/std vector has; a bare scalar is possible in a hand-written file."""
    try:
        return len(values)
    except TypeError:
        return "scalar"


def _safetensors_stats(folder: Path):
    """Action mean/std widths from the unnormalizer's safetensors, which is where lerobot keeps them.

    ``*_unnormalizer_processor.safetensors`` is the serialized postprocessor's state; the inline case in
    ``postprocessor.json`` is a hand-written or older format, so both are read.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    for path in sorted(folder.glob(UNNORMALIZER_STATS)):
        try:
            with safe_open(path, framework="numpy") as tensors:
                keys = set(tensors.keys())
                if {"action.mean", "action.std"} <= keys:
                    mean = tensors.get_slice("action.mean").get_shape()
                    std = tensors.get_slice("action.std").get_shape()
                    return (mean[0] if mean else 1), (std[0] if std else 1)
        except (OSError, ValueError):
            continue
    return None


def _action_stats(config):
    """(mean width, std width) of the action's statistics, or None when the folder has none."""
    folder = Path(getattr(config, "pretrained_path", "") or "")
    if not folder.is_dir():
        return None
    for name in POSTPROCESSOR_FILES:
        path = folder / name
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        stats = _find_action_stats(document)
        if stats is not None:
            return _width(stats["mean"]), _width(stats["std"])
    return _safetensors_stats(folder)


def format_checkpoint(config, stats_path: Path | None = None) -> str:
    """A DECISION.md block for a loaded policy config: the action and the observation keys, no weights.

    The widths come from the checkpoint's own postprocessor, which for the published DROID port is absent: the
    file the runner installs instead is `stats_path`, so its presence and the action width it describes are
    reported on their own line rather than as a bare "unavailable".
    """
    stats = _action_stats(config)
    shape = _action_shape(config)
    widths = "unavailable: the checkpoint ships none" if stats is None else f"{stats[0]} / {stats[1]}"
    lines = [
        "checkpoint:",
        f"  path: {_display(getattr(config, 'pretrained_path', None))}",
        f"  type: {_display(getattr(config, 'type', None))}",
        f"  chunk_size: {_display(getattr(config, 'chunk_size', None))}",
        f"  n_action_steps: {_display(getattr(config, 'n_action_steps', None))}",
        f"  num_inference_steps: {_display(getattr(config, 'num_inference_steps', None))}",
        f"  max_action_dim: {_display(getattr(config, 'max_action_dim', None))}",
        f"  use_relative_actions: {_display(getattr(config, 'use_relative_actions', None))}",
        f"  image keys: {', '.join(_image_keys(config)) or 'none'}",
        f"  action shape: {shape or 'unavailable'}",
        f"  action mean/std widths: {widths}",
    ]
    if stats_path is not None:
        lines.append(f"  statistics: {_format_stats_file(stats_path)}")
    return "\n".join(lines)


def _format_stats_file(path: Path) -> str:
    """What `--stats` names: present with the width it describes, or why it is missing."""
    if not path.exists():
        return (
            f"[--stats {path} absent: download it from "
            "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json]"
        )
    try:
        import stats as checkpoint_stats

        found = checkpoint_stats.read_openpi_stats(path)
    except Exception as error:  # the probe reports, it does not die on the other half's input
        return f"[--stats {path} unreadable: {error}]"
    return f"[--stats {path}: action width {checkpoint_stats.real_width(found)}]"


def format_manifest(manifest: StationManifest) -> str:
    """A DECISION.md block for what the station says it accepts and observes."""
    lines = [
        "manifest:",
        f"  arms: {', '.join(manifest.arms) or 'none'}",
        f"  cameras: {', '.join(manifest.cameras) or 'none'}",
        f"  action keys: {', '.join(manifest.action_keys) or 'none'}",
        f"  control rate: {float(manifest.control_rate):g} Hz",
    ]
    return "\n".join(lines)


def format_spacemouse(path, full_scale) -> str:
    """One line for the operator's device; ``path`` is None when no node was found."""
    if not path:
        return "spacemouse: unavailable: no device found"
    scale = "unknown" if full_scale is None else f"{float(full_scale):g} counts"
    return f"spacemouse: {path} (full scale {scale})"


def _use_worktree(worktree) -> Path:
    """Read the chosen checkout: its ``src/`` first, so the probe imports that code rather than an install."""
    root = Path(worktree).expanduser().resolve()
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _load_config(policy: Path):
    """The checkpoint's config alone: ``from_pretrained`` reads ``config.json`` and never the weights.

    Importing the pi05 module registers its config subclass with draccus; without it the registry cannot
    decode the folder's ``"type": "pi05"``.
    """
    import lerobot.policies.pi05  # noqa: F401  registers PI05Config
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(str(policy))
    config.pretrained_path = str(policy)  # the loader drops it; format_checkpoint reads the folder from it
    return config


def _checkpoint_block(policy: Path, stats_path: Path) -> str:
    """The checkpoint half, reporting a failure as its own line rather than ending the probe."""
    try:
        return format_checkpoint(_load_config(policy), stats_path)
    except Exception as e:
        return f"checkpoint: unavailable: {e}"


def _manifest(robot) -> StationManifest:
    """c3po's manifest properties in the example's frozen type."""
    return StationManifest(
        arms=tuple(robot.arms),
        cameras=tuple(robot.cameras),
        action_keys=tuple(robot.action_keys),
        control_rate=float(robot.control_rate),
    )


def _read_only_step(robot) -> str:
    """One ``step(None)``: the station's answer to a command-free read, the bring-up check."""
    try:
        observation = robot.step(None)
    except Exception as e:
        return f"  step(None): rejected: {e}"
    try:
        keys = ", ".join(sorted(observation)) or "none"
    except TypeError:
        keys = str(observation)
    return f"  step(None): holds; observation keys: {keys}"


def _station_block(host: str, port: int) -> str:
    """The station half: connect, print the manifest, then try one read-only step and close.

    c3po is imported here and inside the try, so the module imports (and the checkpoint half runs) on a
    machine without it.
    """
    try:
        from c3po import Robot
    except Exception as e:
        return f"station: unavailable: c3po is not importable ({e})"
    robot = None
    try:
        robot = Robot(host=host, port=port)
        manifest = _manifest(robot)
        lines = [
            f"station: ws://{host}:{port}",
            format_manifest(manifest),
            f"suggested cameras: {_suggested_cameras(manifest)}",
            _read_only_step(robot),
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"station: unavailable: {e}"
    finally:
        if robot is not None:
            with contextlib.suppress(Exception):
                robot.close()


def _suggested_cameras(manifest: StationManifest) -> str:
    """The `cameras:` line a condition file should carry, from the feeds' names, or why it cannot be guessed."""
    try:
        import batches

        roles = batches.suggested_cameras(manifest)
    except Exception as error:
        return f"unavailable: {error}"
    return f"{{exterior: {roles['exterior']}, wrist: {roles['wrist']}}}"


def _spacemouse_block() -> str:
    """The operator's device, found the same way the chain finds it for a run."""
    try:
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import env_full_scale, find_spacemouse

        return format_spacemouse(find_spacemouse(), env_full_scale())
    except Exception as e:
        return f"spacemouse: unavailable: {e}"


def main(argv=None) -> int:
    """Print the one block; nothing else, so the output can be pasted into DECISION.md."""
    args = _parse_args(argv)
    worktree = _use_worktree(args.worktree)
    stats_path = Path(args.stats) if args.stats is not None else DEFAULT_STATS
    blocks = [
        f"worktree: {worktree}",
        _checkpoint_block(args.policy, stats_path),
        _station_block(args.station, args.port),
        _spacemouse_block(),
    ]
    print("\n".join(blocks))
    return 0


DEFAULT_STATS = Path(__file__).resolve().parent / "assets" / "droid" / "norm_stats.json"
"""Where the runner looks for the statistics the checkpoint does not ship (see `assets/README.md`)."""


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", required=True, type=Path, help="checkpoint folder (weights are not read)")
    parser.add_argument("--station", default="10.42.0.1", help="c3po station host")
    parser.add_argument("--port", default=9090, type=int, help="c3po WebSocket port")
    parser.add_argument(
        "--stats", default=None, help=f"openpi's norm_stats.json (default: {DEFAULT_STATS})"
    )
    default_worktree = Path(__file__).resolve().parents[4]
    parser.add_argument(
        "--worktree",
        nargs="?",
        const=default_worktree,
        default=default_worktree,
        type=Path,
        help="LeRobot checkout to read; its src/ goes first on sys.path (default: this one)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
