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
    mode <name>         policy: pi0.5 only
                        shared_override: SpaceMouse x/y/z replaces pi0.5's
                        translation in the executed action
                        shared_flow_control [tau]: SpaceMouse steers dims 0-2 of
                        x_t for the first tau denoising steps (default 5), the
                        rest denoise freely; executed action is the model's own
                        teleop: SpaceMouse only, any button toggles the gripper
    quit / Ctrl-D       exit
"""

import argparse
import datetime as dt
import io
import json
import logging
import re
import threading
import time
from collections import deque
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spacemouse import SpaceMouseReader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import init_logging

VIDEO_FPS = 30
RATE_HZ = 20  # LIBERO control rate; caps the loop so teleop feels real-time
# Suite defaults (~14 s) are too short for manual driving. Must stay below
# robosuite's horizon of 1000 (which includes the 10 settle steps at reset):
# LIBERO's step() replaces the horizon `done` with task success, so exceeding
# the horizon raises "executing action in terminated episode" instead of ending.
TELEOP_MAX_STEPS = 900

# LIBERO's robosuite OSC_POSE controller: end-effector position deltas,
# orientation (axis-angle) deltas, then gripper (-1 = open, +1 = close).
ACTION_LABELS = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

PAGE = """<!doctype html>
<html>
<head>
<title>LIBERO interactive</title>
<style>
  body { background: #111; color: #ddd; font-family: monospace; text-align: center; }
  img { width: 540px; max-width: 95vw; image-rendering: auto; margin-top: 1em; }
  #status { margin: 1em auto; max-width: 640px; }
  .prompt { color: #8fd; }
  .ok { color: #6f6; }
  .fail { color: #f66; }
  #actions { width: 360px; max-width: 95vw; margin: 0.5em auto; }
  .arow { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  .alabel { width: 64px; text-align: right; color: #aaa; }
  .aval { width: 52px; text-align: right; }
  .abar { position: relative; flex: 1; height: 10px; background: #222; border-radius: 3px; }
  .abar::after { content: ""; position: absolute; left: 50%; top: 0; bottom: 0;
                 width: 1px; background: #555; }
  .afill { position: absolute; top: 0; bottom: 0; border-radius: 2px; }
</style>
</head>
<body>
<h3>LIBERO interactive</h3>
<img src="/stream">
<div id="status">connecting...</div>
<div id="actions"></div>
<script>
function actionRow(label, v) {
  // Signed bar centered at zero; values live in [-1, 1] (clamped for display).
  const pct = Math.min(Math.abs(v), 1) * 50;
  const left = v >= 0 ? 50 : 50 - pct;
  const color = v >= 0 ? "#6cf" : "#f96";
  return '<div class="arow"><span class="alabel">' + label + '</span>' +
    '<span class="abar"><span class="afill" style="left:' + left + '%;width:' +
    pct + '%;background:' + color + '"></span></span>' +
    '<span class="aval">' + v.toFixed(2) + '</span></div>';
}
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    let state = s.state;
    if (state === "success") state = '<span class="ok">SUCCESS</span>';
    if (state === "failed") state = '<span class="fail">no success</span>';
    document.getElementById("status").innerHTML =
      '<div class="prompt">&quot;' + s.prompt + '&quot;</div>' +
      '<div>step ' + s.step + ' / ' + s.max_steps + ' &mdash; ' + state +
      ' <span style="color:#888">(' + s.mode + ')</span></div>';
    if (s.action) {
      document.getElementById("actions").innerHTML =
        s.action_labels.map((l, i) => actionRow(l, s.action[i])).join("");
    }
  } catch (e) {}
  setTimeout(poll, 200);
}
poll();
</script>
</body>
</html>
"""


class FrameStream:
    """Holds the latest JPEG frame + rollout status for the HTTP server."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._status = {
            "prompt": "",
            "step": 0,
            "max_steps": 0,
            "state": "loading model...",
            "mode": "policy",
            "action": None,
            "action_labels": ACTION_LABELS,
        }

    def publish(self, rgb: np.ndarray) -> None:
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=85)
        with self._cond:
            self._jpeg = buf.getvalue()
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def set_status(self, **kwargs) -> None:
        with self._cond:
            self._status.update(kwargs)

    def get_status(self) -> dict:
        with self._cond:
            return dict(self._status)


def make_handler(stream: FrameStream):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal clean for the REPL
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                body = json.dumps(stream.get_status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        jpeg, seq = stream.wait_frame(seq)
                        if jpeg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


class ZeroPolicy:
    """Stand-in policy for teleop mode: outputs zeros, the SpaceMouse hook fills them in."""

    def reset(self) -> None:
        pass

    def select_action(self, obs) -> torch.Tensor:
        return torch.zeros(1, 7)


def identity(x):
    """Stand-in for the processor pipelines in teleop mode (they are only called)."""
    return x


def build_env(suite: str, task_id: int):
    """Build a 1-env SyncVectorEnv for one LIBERO scene, mirroring eval.sh settings."""
    env_cfg = LiberoEnvConfig(task=suite, task_ids=[task_id])
    envs_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs_dict[suite][task_id]
    check_env_attributes_and_types(vec_env)
    return env_cfg, envs_dict, vec_env


def list_tasks(suite: str) -> list[str]:
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    if suite not in bench:
        raise ValueError(f"Unknown suite '{suite}'. Available: {', '.join(sorted(bench))}")
    return [t.language for t in bench[suite]().tasks]


def run_rollout(
    vec_env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    prompt: str,
    stream: FrameStream,
    autocast_ctx,
    action_hook=None,
    max_steps_override: int | None = None,
) -> tuple[bool, int, list[np.ndarray]]:
    """One rollout of the policy in vec_env (1 sub-env), driven by `prompt`.

    Mirrors the step loop of lerobot_eval.rollout(), except observation["task"]
    is the user's text instead of the env's task_description. If given,
    `action_hook` edits the env-space action right before it is executed
    (used to paste in SpaceMouse commands).
    """
    policy.reset()
    observation, _ = vec_env.reset()
    max_steps = max_steps_override or int(vec_env.call("_max_episode_steps")[0])
    frames = [vec_env.envs[0].render()]
    stream.publish(frames[-1])
    stream.set_status(prompt=prompt, step=0, max_steps=max_steps, state="running", action=None)

    success = False
    step = 0
    while step < max_steps:
        step_start = time.time()
        obs = preprocess_observation(observation)
        obs["task"] = [prompt]
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        with torch.inference_mode(), autocast_ctx:
            action = policy.select_action(obs)
        action = postprocessor(action)
        transition = env_postprocessor({ACTION: action})
        action_numpy = transition[ACTION].to("cpu").numpy()
        if action_hook is not None:
            action_numpy = action_hook(action_numpy)

        observation, _reward, terminated, truncated, info = vec_env.step(action_numpy)
        step += 1
        stream.set_status(step=step, action=action_numpy[0].tolist())

        if "final_info" in info:
            success = bool(info["final_info"]["is_success"][0])
        elif "is_success" in info:
            is_success = info["is_success"]
            success = bool(is_success[0] if hasattr(is_success, "__len__") else is_success)

        if terminated[0] or truncated[0]:
            # The sub-env auto-resets on termination, so render() would show a
            # fresh scene — use the final observation's agentview pixels instead
            # (flipped to match render() orientation).
            final = observation["pixels"]["image"][0][::-1, ::-1]
            frames.append(final)
            stream.publish(final)
            break

        frames.append(vec_env.envs[0].render())
        stream.publish(frames[-1])
        print(f"\r  step {step}/{max_steps}", end="", flush=True)

        leftover = 1.0 / RATE_HZ - (time.time() - step_start)
        if leftover > 0:
            time.sleep(leftover)

    print()
    stream.set_status(state="success" if success else "failed")
    return success, step, frames


def make_spacemouse_hook(reader: SpaceMouseReader, paste_gripper: bool):
    """Paste SpaceMouse translation (and, for teleop, gripper) into the env action."""

    def hook(action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[0, :3] = reader.translation
        if paste_gripper:
            action[0, 6] = reader.gripper
        return action

    return hook


def get_action_mean_std(postprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Translation mean/std from the checkpoint's action unnormalizer stats."""
    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats and "action" in stats:
            mean = np.asarray(stats["action"]["mean"], dtype=np.float64)[:3]
            std = np.maximum(np.asarray(stats["action"]["std"], dtype=np.float64)[:3], 1e-6)
            return mean, std
    raise RuntimeError("No action stats in the postprocessor; shared_flow_control needs them.")


class FlowControlPolicy:
    """pi0.5 wrapper implementing shared_flow_control.

    For the first `tau` of the chunk's denoising steps, the SpaceMouse
    translation (normalized into the model's action space) is written into
    dims 0-2 of x_t across the whole chunk; the remaining steps denoise
    freely. The executed action is entirely the model's output — the
    SpaceMouse steers only the early flow. Guidance is skipped while the
    stick is inside DEADBAND, so an idle SpaceMouse means pure policy.
    """

    DEADBAND = 0.05

    def __init__(self, policy, reader: SpaceMouseReader, tau: int, postprocessor):
        self._policy = policy
        self._reader = reader
        self.tau = tau
        self._mean, self._std = get_action_mean_std(postprocessor)
        self._queue: deque = deque()
        self.hook_calls = 0  # per-rollout count of guided denoising steps

    def reset(self) -> None:
        self._policy.reset()
        self._queue.clear()
        self.hook_calls = 0

    @torch.compiler.disable
    def _x_t_hook(self, step: int, time_: float, x_t: torch.Tensor) -> torch.Tensor:
        # compiler.disable: with --compile, sample_actions is dynamo-traced;
        # the hook reads live SpaceMouse state and must stay eager.
        if step >= self.tau:
            return x_t
        cmd = self._reader.translation
        if np.max(np.abs(cmd)) < self.DEADBAND:
            return x_t
        target = (cmd - self._mean) / self._std
        x_t[..., :3] = torch.as_tensor(target, dtype=x_t.dtype, device=x_t.device)
        self.hook_calls += 1
        return x_t

    def select_action(self, batch) -> torch.Tensor:
        # Same queue logic as PI05Policy.select_action, plus the guidance hook.
        if len(self._queue) == 0:
            chunk = self._policy.predict_action_chunk(batch, x_t_hook=self._x_t_hook)
            actions = chunk[:, : self._policy.config.n_action_steps]
            self._queue.extend(actions.transpose(0, 1))
        return self._queue.popleft()


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "prompt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model as in eval.sh (slower first rollout, faster after)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pi05_libero_interactive"),
        help="Where rollout MP4s are written",
    )
    args = parser.parse_args()

    init_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream = FrameStream()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(stream))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\nLive view: http://localhost:{args.port}  (VSCode should auto-forward the port)\n")

    logging.info(f"Loading policy {args.policy_path} ...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.n_action_steps = args.n_action_steps
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = args.compile

    env_cfg, envs_dict, vec_env = build_env(args.suite, args.task_id)

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    autocast_ctx = (
        torch.autocast(device_type=torch.device(policy.config.device).type)
        if policy_cfg.use_amp
        else nullcontext()
    )

    # Show the scene while the user types the first prompt.
    vec_env.reset()
    stream.publish(vec_env.envs[0].render())
    stream.set_status(state="idle — type a prompt in the terminal")

    suite, task_id = args.suite, args.task_id
    mode = "policy"
    tau = 5  # flow-control default: guide the first half of the 10 denoising steps
    sm_reader: SpaceMouseReader | None = None
    flow_policy: FlowControlPolicy | None = None
    zero_policy = ZeroPolicy()
    print(f"\nScene: {suite} task {task_id}")
    print(f'Built-in instruction: "{vec_env.envs[0].task_description}"')
    print(
        "Type an instruction (empty = built-in), `tasks`, `task <suite> <id>`, "
        "`mode policy|shared_override|shared_flow_control [tau]|teleop`, or `quit`.\n"
    )

    try:
        while True:
            try:
                line = input(f"{mode}> ").strip()
            except EOFError:
                break
            if line in ("quit", "exit"):
                break

            if line == "mode" or line.startswith("mode "):
                tokens = line.split()
                new_mode = tokens[1] if len(tokens) > 1 else ""
                if new_mode not in ("policy", "shared_override", "shared_flow_control", "teleop"):
                    print("usage: mode policy|shared_override|shared_flow_control [tau]|teleop")
                    continue
                if new_mode == "shared_flow_control" and len(tokens) > 2:
                    n_flow_steps = policy.config.num_inference_steps
                    if not tokens[2].isdigit() or not 0 <= int(tokens[2]) <= n_flow_steps:
                        print(f"tau must be an integer in [0, {n_flow_steps}] (denoising steps)")
                        continue
                    tau = int(tokens[2])
                if new_mode != "policy" and sm_reader is None:
                    try:
                        sm_reader = SpaceMouseReader()
                        print(f"SpaceMouse connected ({sm_reader.device_path})")
                    except (FileNotFoundError, PermissionError, OSError) as e:
                        print(f"SpaceMouse unavailable: {e}")
                        continue
                mode = new_mode
                if mode == "teleop":
                    print("Teleop: SpaceMouse drives x/y/z, any button toggles the gripper.")
                    print("Press Enter to start a rollout.")
                elif mode == "shared_override":
                    print("Shared override: pi0.5 drives, your SpaceMouse x/y/z replaces its translation.")
                elif mode == "shared_flow_control":
                    print(
                        f"Shared flow control: SpaceMouse steers the first tau={tau} of "
                        f"{policy.config.num_inference_steps} denoising steps (idle stick = pure policy)."
                    )
                continue

            if line == "tasks":
                for i, lang in enumerate(list_tasks(suite)):
                    print(f"  {i}: {lang}")
                continue

            if line.startswith("task "):
                parts = line.split()
                if len(parts) != 3 or not parts[2].lstrip("-").isdigit():
                    print("usage: task <suite> <id>   e.g. task libero_spatial 3")
                    continue
                new_suite, new_task_id = parts[1], int(parts[2])
                try:
                    new_env_cfg, new_envs_dict, new_vec_env = build_env(new_suite, new_task_id)
                except (ValueError, FileNotFoundError) as e:
                    print(f"could not build env: {e}")
                    continue
                close_envs(envs_dict)
                suite, task_id = new_suite, new_task_id
                env_cfg, envs_dict, vec_env = new_env_cfg, new_envs_dict, new_vec_env
                env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                    env_cfg=env_cfg, policy_cfg=policy_cfg
                )
                vec_env.reset()
                stream.publish(vec_env.envs[0].render())
                print(f"Scene: {suite} task {task_id}")
                print(f'Built-in instruction: "{vec_env.envs[0].task_description}"')
                continue

            if mode == "teleop":
                prompt = "SpaceMouse teleop"
                rollout_kwargs = {
                    "policy": zero_policy,
                    "env_preprocessor": identity,
                    "env_postprocessor": identity,
                    "preprocessor": identity,
                    "postprocessor": identity,
                    "action_hook": make_spacemouse_hook(sm_reader, paste_gripper=True),
                    "max_steps_override": TELEOP_MAX_STEPS,
                }
            else:
                prompt = line if line else vec_env.envs[0].task_description
                rollout_kwargs = {
                    "policy": policy,
                    "env_preprocessor": env_preprocessor,
                    "env_postprocessor": env_postprocessor,
                    "preprocessor": preprocessor,
                    "postprocessor": postprocessor,
                }
                if mode == "shared_override":
                    rollout_kwargs["action_hook"] = make_spacemouse_hook(sm_reader, paste_gripper=False)
                elif mode == "shared_flow_control":
                    if flow_policy is None:
                        flow_policy = FlowControlPolicy(policy, sm_reader, tau, postprocessor)
                    flow_policy.tau = tau
                    rollout_kwargs["policy"] = flow_policy

            mode_label = f"{mode} tau={tau}" if mode == "shared_flow_control" else mode
            print(f'Running [{mode_label}]: "{prompt}"')
            start = time.time()
            stream.set_status(mode=mode_label)
            success, steps, frames = run_rollout(
                vec_env,
                prompt=prompt,
                stream=stream,
                autocast_ctx=autocast_ctx,
                **rollout_kwargs,
            )
            video_path = (
                args.output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{slugify(prompt)}.mp4"
            )
            write_video(video_path, frames, fps=VIDEO_FPS)
            outcome = "SUCCESS" if success else "no success"
            print(f"{outcome} after {steps} steps ({time.time() - start:.0f}s) — video: {video_path}")
            if mode == "shared_flow_control":
                print(f"flow guidance applied on {flow_policy.hook_calls} denoising steps")
            print()
    finally:
        close_envs(envs_dict)
        server.shutdown()


if __name__ == "__main__":
    main()
