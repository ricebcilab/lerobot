#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Interactive LIBERO runner: type a text prompt, watch the policy act on it.

Loads the pi0.5 policy once, then loops: read an instruction from the terminal,
run one rollout in the chosen LIBERO scene feeding *your* text to the policy
(instead of the task's built-in instruction), and stream frames live to a
browser tab. An MP4 of each rollout is also saved.

Launch via ./interactive.sh (sets MUJOCO_GL/caches and runs under uv), then
open the printed URL. REPL commands:

    <any text>          run a rollout with that instruction
    <empty line>        run a rollout with the scene's built-in instruction
    tasks               list task ids + instructions for the current suite
    task <suite> <id>   switch scene (e.g. `task libero_spatial 3`)
    mode <name> [n]     policy: pi0.5 only
                        shared_override: your x/y/z replaces pi0.5's translation
                        shared_flow_control [tau]: your x/y/z steers dims 0-2 of x_t
                        for the first tau denoising steps (default 5)
                        shared_reverse_flow_steering [n_reverse_steps]: your x/y/z
                        defines a reference chunk that is integrated backward
                        through the flow to its latent noise; the forward flow
                        starts from there (Flow Reversal Steering, arXiv:2606.13675)
                        teleop: you drive x/y/z and the gripper, no model
    noise [std]         show / set the std of the Gaussian noise added to your
                        x/y/z command at every step (0 = off)
    corruption [file|off]
                        show / load / clear the command corruption M (3x3):
                        a YAML file with {rotation_z_deg: d}, {scale: [...]} or M: rows
    adapter [file|off]  show / load / clear the reversal adapter F (7x7): a YAML
                        file with translation/orientation/gripper blocks or F: rows
    quit / Ctrl-D       exit

Teleop input comes from a 3Dconnexion SpaceMouse if one is plugged in (any
button toggles the gripper) and/or from the keyboard in the browser live view:
arrows move x/y, PageUp/PageDown or W/S move z, Space toggles the gripper, hold
Shift for full speed. The SpaceMouse wins while it is deflected.
"""

import argparse
import datetime as dt
import re
import time
from pathlib import Path

from config import DEFAULT_INTERACTIVE_CONFIG, MODES, load_interactive_settings
from session import VIDEO_FPS, Session
from teleop import build_corruption, read_matrix_spec

from lerobot.policies.pi05.steering import build_reversal_adapter
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import init_logging


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "prompt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        nargs="?",
        const=str(DEFAULT_INTERACTIVE_CONFIG),
        default=None,
        metavar="PATH",
        help="read settings from a YAML file (flags passed here still win); PATH defaults to "
        f"{DEFAULT_INTERACTIVE_CONFIG.relative_to(Path(__file__).resolve().parent)}",
    )
    parser.add_argument("--policy-path", dest="policy_path", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task-id", dest="task_id", type=int, default=None)
    parser.add_argument("--n-action-steps", dest="n_action_steps", type=int, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=MODES, help="control mode to start in")
    parser.add_argument("--tau", type=int, default=None, help="shared_flow_control: steps your input steers")
    parser.add_argument(
        "--n-reverse-steps",
        dest="n_reverse_steps",
        type=int,
        default=None,
        metavar="N",
        help="shared_reverse_flow_steering: reverse N of the denoising steps (default: all)",
    )
    parser.add_argument(
        "--corruption",
        default=None,
        metavar="FILE",
        help="YAML matrix spec applied to your x/y/z command at every step (x -> M @ x)",
    )
    parser.add_argument(
        "--reversal-adapter",
        dest="reversal_adapter",
        default=None,
        metavar="FILE",
        help="YAML adapter spec for shared_reverse_flow_steering's reverse integration (x_t += h * F @ v)",
    )
    parser.add_argument(
        "--input-noise",
        dest="input_noise",
        type=float,
        default=None,
        metavar="STD",
        help="std of Gaussian noise added to x, y, z of your command while you are commanding",
    )
    parser.add_argument(
        "--compile", action="store_const", const=True, default=None, help="torch.compile the model"
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None, help="where rollout MP4s are written"
    )
    args = parser.parse_args()
    overrides = vars(args).copy()
    overrides.pop("config")
    for key in ("corruption", "reversal_adapter"):
        if overrides[key] is not None:
            try:
                overrides[key] = read_matrix_spec(overrides[key])
            except (OSError, ValueError) as e:
                parser.error(f"--{key.replace('_', '-')}: {e}")
    try:
        settings = load_interactive_settings(args.config, overrides)
    except (OSError, ValueError) as e:
        parser.error(str(e))
    return settings


# ---------------------------------------------------------------- REPL commands


def cmd_mode(session: Session, tokens: list[str]) -> None:
    usage = (
        "usage: mode policy|shared_override|shared_flow_control [tau]|shared_reverse_flow_steering [n]|teleop"
    )
    if len(tokens) < 2 or tokens[1] not in MODES:
        print(usage)
        return
    mode, arg = tokens[1], (tokens[2] if len(tokens) > 2 else None)
    kwargs = {}
    if arg is not None:
        if mode not in ("shared_flow_control", "shared_reverse_flow_steering") or not arg.isdigit():
            print(usage)
            return
        kwargs["tau" if mode == "shared_flow_control" else "n_reverse_steps"] = int(arg)
    try:
        session.set_mode(mode, **kwargs)
    except ValueError as e:
        print(e)
        return
    text = session.announce_mode()
    if text:
        print(text)


def cmd_noise(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2:
        try:
            session.chain.noisy.std = float(tokens[1])
        except ValueError:
            print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
            return
    elif len(tokens) > 2:
        print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
        return
    std = session.chain.noisy.std
    print(f"Input noise: std {std:.3f} on x/y/z while you command." if std > 0 else "Input noise: off.")


def cmd_corruption(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2 and tokens[1] == "off":
        session.set_corruption(None, None)
    elif len(tokens) == 2:
        try:
            session.set_corruption(build_corruption(read_matrix_spec(tokens[1])), Path(tokens[1]).name)
        except (OSError, ValueError) as e:
            print(f"could not load corruption: {e}")
            return
    elif len(tokens) > 2:
        print("usage: corruption [<file.yaml>|off]")
        return
    hint = "" if session.chain.corruption.matrix is not None else " Load one with `corruption <file.yaml>`."
    print(session.chain.corruption.describe() + hint)


def cmd_adapter(session: Session, tokens: list[str]) -> None:
    if len(tokens) == 2 and tokens[1] == "off":
        session.set_reversal_adapter(None, None)
    elif len(tokens) == 2:
        try:
            spec = read_matrix_spec(tokens[1])
            matrix = build_reversal_adapter(spec, session.chain.corruption.matrix)
            session.set_reversal_adapter(matrix, Path(tokens[1]).name)
        except (OSError, ValueError) as e:
            print(f"could not load reversal adapter: {e}")
            return
    elif len(tokens) > 2:
        print("usage: adapter [<file.yaml>|off]")
        return
    hint = "" if session.adapter.matrix is not None else " Load one with `adapter <file.yaml>`."
    print(session.adapter.describe() + hint)


def cmd_tasks(session: Session, tokens: list[str]) -> None:
    for i, lang in enumerate(session.list_tasks(session.suite)):
        print(f"  {i}: {lang}")


def cmd_task(session: Session, tokens: list[str]) -> None:
    if len(tokens) != 3 or not tokens[2].lstrip("-").isdigit():
        print("usage: task <suite> <id>   e.g. task libero_spatial 3")
        return
    try:
        session.set_scene(tokens[1], int(tokens[2]))
    except (ValueError, FileNotFoundError) as e:
        print(f"could not build env: {e}")
        return
    session.show_scene()
    print(f"Scene: {session.suite} task {session.task_id}")
    print(f'Built-in instruction: "{session.task_description}"')


COMMANDS = {
    "mode": cmd_mode,
    "noise": cmd_noise,
    "corruption": cmd_corruption,
    "adapter": cmd_adapter,
    "tasks": cmd_tasks,
    "task": cmd_task,
}


def run_prompt(session: Session, line: str, output_dir: Path) -> None:
    prompt = "manual teleop" if session.mode == "teleop" else (line or session.task_description)
    print(f'Running [{session.mode_label()}]: "{prompt}"')
    start = time.time()
    result = session.rollout(prompt)
    video_path = output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{slugify(prompt)}.mp4"
    write_video(video_path, result.frames, fps=VIDEO_FPS)
    outcome = "SUCCESS" if result.success else "no success"
    print(f"{outcome} after {result.steps} steps ({time.time() - start:.0f}s), video: {video_path}")
    stats = session.stats_line()
    if stats:
        print(stats)
    print()


def main():
    settings = parse_args()
    init_logging()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    session = Session(settings)
    session.show_scene()
    session.view.stream.set_status(state="idle: type a prompt in the terminal")
    print(f"Scene: {session.suite} task {session.task_id}")
    print(f'Built-in instruction: "{session.task_description}"')
    print(
        "Type an instruction (empty = built-in), `tasks`, `task <suite> <id>`, `mode <name> [n]`, "
        "`noise [std]`, `corruption [file|off]`, `adapter [file|off]`, or `quit`.\n"
    )
    if session.chain.corruption.matrix is not None:
        print(session.chain.corruption.describe() + "\n")
    if session.adapter.matrix is not None:
        print(session.adapter.describe() + "\n")
    if session.mode != "policy":
        print(session.announce_mode() + "\n")
    if session.chain.noisy.std > 0:
        print(f"Input noise: std {session.chain.noisy.std:.3f} on x/y/z while you command.\n")

    try:
        while True:
            try:
                line = input(f"{session.mode}> ").strip()
            except EOFError:
                break
            if line in ("quit", "exit"):
                break
            tokens = line.split()
            if tokens and tokens[0] in COMMANDS:
                COMMANDS[tokens[0]](session, tokens)
                continue
            run_prompt(session, line, settings.output_dir)
    finally:
        session.close()


if __name__ == "__main__":
    main()
