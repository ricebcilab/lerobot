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
                        shared_override: your x/y/z replaces pi0.5's
                        translation in the executed action
                        shared_flow_control [tau]: your x/y/z steers dims 0-2 of
                        x_t for the first tau denoising steps (default 5), the
                        rest denoise freely; executed action is the model's own
                        shared_reverse_flow_steering: your x/y/z defines a
                        uniform-velocity reference chunk that is integrated
                        backward through the flow to its latent noise; the
                        forward flow then starts from that noise (Flow
                        Reversal Steering, arXiv:2606.13675)
                        teleop: you drive x/y/z and the gripper, no model
    noise [std]         show / set the std of the isotropic Gaussian noise
                        added to your x/y/z command at every step (0 = off)
    corruption [path|off]
                        show / load / clear the deterministic corruption: a
                        3x3 matrix M from a YAML file, applied to your x/y/z
                        command at every step (x -> M @ x)
    adapter [path|off]  show / load / clear the flow-reversal adapter: a 7x7
                        matrix F from a YAML file that adapts the velocity
                        field used by shared_reverse_flow_steering's reverse
                        integration (x_t += h * F @ v)
    adapter [path|off]  show / load / clear the flow-reversal adapter: a 7x7
                        matrix F from a YAML file that adapts the velocity
                        field used by shared_reverse_flow_steering's reverse
                        integration (x_t += h * F @ v)
    quit / Ctrl-D       exit

Teleop input ("your x/y/z") comes from a 3Dconnexion SpaceMouse if one is
plugged in (any button toggles the gripper) and/or from the keyboard in the
browser live view: arrows move x/y, PageUp/PageDown or W/S move z, Space
toggles the gripper, hold Shift for full speed. The SpaceMouse wins while it
is deflected; the keyboard works whenever the page has focus. Two optional
corruptions of that command emulate an imperfect operator: `--deterministic-corruption`
(or `corruption` in the REPL) multiplies x/y/z by a 3x3 matrix read from a YAML
file, and `--input-noise` (or `noise`) then adds independent N(0, std^2) noise
to x, y and z at every control step while you are commanding. Separately,
`--flow-reversal-adapter` (or `adapter`) loads a 7x7 matrix F that adapts the
velocity field used by shared_reverse_flow_steering's reverse integration.
"""

import argparse
import datetime as dt
import io
import json
import logging
import re
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from teleop import (
    CommandCorruption,
    KeyboardReader,
    NoisyReader,
    TeleopChain,
    build_corruption,
    read_matrix_spec,
)

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
from lerobot.policies.pi05.steering import (
    FlowControlPolicy,
    ReversalAdapter as FlowAdapter,
    ReverseFlowSteeringPolicy,
    build_reversal_adapter,
)
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
  .task { color: #fff; font-weight: 600; }
  .vla { color: #9a9; font-size: 90%; }
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
  #keys { margin: 0.5em auto; max-width: 640px; color: #888; font-size: 0.9em; }
  #keys .held { color: #6cf; }
  #keys .focus { color: #fc6; }
  kbd { background: #222; border: 1px solid #444; border-radius: 3px; padding: 0 4px; color: #ccc; }
</style>
</head>
<body>
<h3>LIBERO interactive</h3>
<img src="/stream">
<div id="status">connecting...</div>
<div id="actions"></div>
<div id="keys"></div>
<script>
// Keyboard teleop: the set of held keys (and a running count of gripper
// toggles) is POSTed to /keys on every change and as a heartbeat while any key
// is held, so a lost keyup or a closed tab cannot leave a key stuck.
const KEYMAP = {ArrowUp: "ArrowUp", ArrowDown: "ArrowDown", ArrowLeft: "ArrowLeft",
                ArrowRight: "ArrowRight", PageUp: "PageUp", PageDown: "PageDown",
                KeyW: "w", KeyS: "s", ShiftLeft: "Shift", ShiftRight: "Shift", Space: "Space"};
const held = new Set();
let toggles = 0;
function sendKeys() {
  fetch("/keys", {method: "POST", headers: {"Content-Type": "application/json"},
                  body: JSON.stringify({held: [...held], toggles: toggles})}).catch(() => {});
}
window.addEventListener("keydown", e => {
  const name = KEYMAP[e.code];
  if (!name) return;
  e.preventDefault();
  if (e.repeat) return;
  if (name === "Space") toggles += 1; else held.add(name);
  sendKeys();
});
window.addEventListener("keyup", e => {
  const name = KEYMAP[e.code];
  if (!name) return;
  e.preventDefault();
  held.delete(name);
  sendKeys();
});
window.addEventListener("blur", () => { held.clear(); sendKeys(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) { held.clear(); sendKeys(); } });
setInterval(() => { if (held.size) sendKeys(); }, 300);
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
      (s.task
        ? '<div class="task">' + s.task + '</div>' +
          '<div class="vla">VLA prompt: &quot;' + s.prompt + '&quot;</div>'
        : '<div class="prompt">&quot;' + s.prompt + '&quot;</div>') +
      '<div>step ' + s.step + ' / ' + s.max_steps + ' &mdash; ' + state +
      ' <span style="color:#888">(' + s.mode + ')</span></div>';
    if (s.action) {
      document.getElementById("actions").innerHTML =
        s.action_labels.map((l, i) => actionRow(l, s.action[i])).join("");
    }
    const focus = document.hasFocus() ? "" : ' <span class="focus">(click the page to give it keyboard focus)</span>';
    const heldKeys = s.keys.length ? ' <span class="held">held: ' + s.keys.join(" ") + '</span>' : "";
    const noise = s.input_noise > 0 ? ' &middot; <span class="held">input noise &sigma;=' + s.input_noise.toFixed(2) + '</span>' : "";
    const corruption = s.corruption ? ' &middot; <span class="held">corruption: ' + s.corruption + '</span>' : "";
    const adapter = s.flow_adapter ? ' &middot; <span class="held">flow adapter: ' + s.flow_adapter + '</span>' : "";
    document.getElementById("keys").innerHTML =
      'keyboard: <kbd>&uarr;</kbd><kbd>&darr;</kbd><kbd>&larr;</kbd><kbd>&rarr;</kbd> move &middot; ' +
      '<kbd>PgUp</kbd>/<kbd>PgDn</kbd> or <kbd>W</kbd>/<kbd>S</kbd> up/down &middot; ' +
      '<kbd>Space</kbd> gripper (' + (s.keyboard_gripper > 0 ? "closed" : "open") + ') &middot; ' +
      '<kbd>Shift</kbd> fast' + corruption + noise + adapter + heldKeys + focus;
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
            "task": None,  # human-facing task (experiment.py); None = show the prompt alone
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


DEFAULT_FLOW_ADAPTER_FILE = Path(__file__).resolve().parent / "flow_reversal_adapter.yaml"


def make_handler(
    stream: FrameStream,
    keyboard: KeyboardReader,
    noisy: NoisyReader,
    corruption: CommandCorruption,
    adapter: FlowAdapter,
):
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
                status = stream.get_status()
                status["keys"] = sorted(keyboard.held)
                status["keyboard_gripper"] = keyboard.gripper
                status["input_noise"] = noisy.std
                status["corruption"] = corruption.label if corruption.matrix is not None else None
                status["flow_adapter"] = adapter.label if adapter.matrix is not None else None
                body = json.dumps(status).encode()
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

        def do_POST(self):
            if self.path != "/keys":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                held, toggles = payload["held"], payload["toggles"]
                if not (
                    isinstance(held, list)
                    and all(isinstance(k, str) for k in held)
                    and isinstance(toggles, int)
                ):
                    raise TypeError("held must be a list of str, toggles an int")
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
                return
            keyboard.update(held, toggles)
            self.send_response(204)
            self.end_headers()

    return Handler


MODES = ("policy", "shared_override", "shared_flow_control", "shared_reverse_flow_steering", "teleop")


def announce_mode(
    mode: str, tau: int, policy, flow_adapter: FlowAdapter, n_reverse_steps: int | None = None
) -> None:
    """Print what the newly selected mode does (shared by the REPL and startup)."""
    if mode != "policy":
        print(
            "Keyboard (click the live view first): arrows = x/y, PgUp/PgDn or W/S = z, "
            "Space = gripper, hold Shift = full speed."
        )
    if mode == "teleop":
        print("Teleop: you drive x/y/z and the gripper; the model is not involved.")
        print("Press Enter to start a rollout.")
    elif mode == "shared_override":
        print("Shared override: pi0.5 drives, your x/y/z replaces its translation.")
    elif mode == "shared_flow_control":
        print(
            f"Shared flow control: your x/y/z steers the first tau={tau} of "
            f"{policy.config.num_inference_steps} denoising steps (idle input = pure policy)."
        )
    elif mode == "shared_reverse_flow_steering":
        total = policy.config.num_inference_steps
        n = total if n_reverse_steps is None else n_reverse_steps
        if n >= total:
            print(
                "Shared reverse flow steering: your x/y/z defines a reference chunk that is "
                f"inverted through the flow ({total} reverse steps) to its noise; pi0.5 then "
                "denoises from that noise (idle input = pure policy)."
            )
        else:
            print(
                "Shared reverse flow steering: your x/y/z defines a reference chunk that is "
                f"inverted {n} of {total} steps (partway to noise, t={n / total:.1f}); pi0.5 then "
                f"denoises from there in {total - n} steps (idle input = pure policy)."
            )
        if flow_adapter.matrix is not None:
            print(flow_adapter.describe())


class ZeroPolicy:
    """Stand-in policy for teleop mode: outputs zeros, the teleop hook fills them in."""

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
    recorder=None,
) -> tuple[bool, int, list[np.ndarray]]:
    """One rollout of the policy in vec_env (1 sub-env), driven by `prompt`.

    Mirrors the step loop of lerobot_eval.rollout(), except observation["task"]
    is the user's text instead of the env's task_description. If given,
    `action_hook` edits the env-space action right before it is executed
    (used to paste in SpaceMouse / keyboard commands). If given, `recorder` is
    called once per step with the observation the policy saw, the executed
    action and the env's response (used by experiment.py to log trials).
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

        previous_observation = observation
        observation, reward, terminated, truncated, info = vec_env.step(action_numpy)
        step += 1
        if recorder is not None:
            recorder(
                step=step,
                observation=previous_observation,
                action=action_numpy,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
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


DEFAULT_CORRUPTION_FILE = Path(__file__).resolve().parent / "deterministic_corruption.yaml"


DEFAULT_INTERACTIVE_CONFIG_FILE = Path(__file__).resolve().parent / "config_interactive.yaml"

# yaml key -> argparse dest, for --config. Nested sections keep the file readable.
INTERACTIVE_CONFIG_KEYS = {
    "policy": {"path": "policy_path", "n_action_steps": "n_action_steps", "compile": "compile"},
    "scene": {"suite": "suite", "task_id": "task_id"},
    "control": {
        "mode": "mode",
        "tau": "tau",
        "n_reverse_steps": "n_reverse_steps",
        "input_noise": "input_noise",
        "deterministic_corruption": "deterministic_corruption",
        "flow_reversal_adapter": "flow_reversal_adapter",
    },
    "server": {"port": "port"},
    "output_dir": "output_dir",
}


def read_yaml_config(path: str | Path) -> dict:
    """Read a YAML config file into a dict (empty file -> {})."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def flatten_config(data: dict, schema: dict, where: str) -> dict:
    """Map a nested config dict onto flat argparse dests, rejecting unknown keys."""
    flat = {}
    for key, value in data.items():
        if key not in schema:
            raise ValueError(f"{where}: unknown key '{key}' (expected one of {', '.join(schema)})")
        spec = schema[key]
        if isinstance(spec, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{where}: '{key}' must be a mapping of {', '.join(spec)}")
            for sub, sub_value in value.items():
                if sub not in spec:
                    raise ValueError(
                        f"{where}: unknown key '{key}.{sub}' (expected one of {', '.join(spec)})"
                    )
                if sub_value is not None:
                    flat[spec[sub]] = sub_value
        elif value is not None:
            flat[spec] = value
    return flat


def make_teleop_hook(reader: NoisyReader, paste_gripper: bool):
    """Paste the teleop translation (and, for teleop mode, gripper) into the env action."""

    def hook(action: np.ndarray) -> np.ndarray:
        action = action.copy()
        action[0, :3] = reader.translation
        if paste_gripper:
            action[0, 6] = reader.gripper
        return action

    return hook


class ModeRunner:
    """Turns a control mode into `run_rollout` kwargs, reusing the mode wrappers.

    The shared-mode wrappers (FlowControlPolicy, ReverseFlowSteeringPolicy) are
    built lazily and kept, so their per-rollout counters (reset by
    `run_rollout`) can be read afterwards through `metrics()`. The processors
    are attributes because switching scene rebuilds the env processors.
    """

    def __init__(
        self,
        policy,
        reader,
        preprocessor,
        postprocessor,
        env_preprocessor,
        env_postprocessor,
        flow_adapter: FlowAdapter | None = None,
        n_reverse_steps: int | None = None,
    ):
        self.policy = policy
        self.n_reverse_steps = n_reverse_steps
        self.reader = reader
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.flow_adapter = flow_adapter
        self.zero_policy = ZeroPolicy()
        self.flow_policy: FlowControlPolicy | None = None
        self.frs_policy: ReverseFlowSteeringPolicy | None = None

    def kwargs(self, mode: str, tau: int = 5, n_reverse_steps: int | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"unknown mode '{mode}' (expected one of {', '.join(MODES)})")
        if mode == "teleop":
            return {
                "policy": self.zero_policy,
                "env_preprocessor": identity,
                "env_postprocessor": identity,
                "preprocessor": identity,
                "postprocessor": identity,
                "action_hook": make_teleop_hook(self.reader, paste_gripper=True),
                "max_steps_override": TELEOP_MAX_STEPS,
            }
        kwargs = {
            "policy": self.policy,
            "env_preprocessor": self.env_preprocessor,
            "env_postprocessor": self.env_postprocessor,
            "preprocessor": self.preprocessor,
            "postprocessor": self.postprocessor,
        }
        if mode == "shared_override":
            kwargs["action_hook"] = make_teleop_hook(self.reader, paste_gripper=False)
        elif mode == "shared_flow_control":
            if self.flow_policy is None:
                self.flow_policy = FlowControlPolicy(self.policy, self.reader, tau, self.postprocessor)
            self.flow_policy.tau = tau
            kwargs["policy"] = self.flow_policy
        elif mode == "shared_reverse_flow_steering":
            if self.frs_policy is None:
                self.frs_policy = ReverseFlowSteeringPolicy(
                    self.policy, self.reader, self.postprocessor, adapter=self.flow_adapter
                )
            self.frs_policy.n_reverse_steps = (
                self.n_reverse_steps if n_reverse_steps is None else n_reverse_steps
            )
            kwargs["policy"] = self.frs_policy
        return kwargs

    def metrics(self, mode: str) -> dict:
        """Per-rollout steering statistics of the last rollout in `mode`."""
        if mode == "shared_flow_control" and self.flow_policy is not None:
            return {"guided_denoising_steps": int(self.flow_policy.hook_calls)}
        if mode == "shared_reverse_flow_steering" and self.frs_policy is not None:
            errors = self.frs_policy.reconstruction_errors
            total = self.policy.config.num_inference_steps
            return {
                "n_reverse_steps": self.frs_policy.n_reverse_steps or total,
                "steered_chunks": int(self.frs_policy.steered_chunks),
                "reconstruction_error_mean": float(np.mean(errors)) if errors else None,
            }
        return {}

    def stats_line(self, mode: str) -> str:
        metrics = self.metrics(mode)
        if mode == "shared_flow_control" and metrics:
            return f"flow guidance applied on {metrics['guided_denoising_steps']} denoising steps"
        if mode == "shared_reverse_flow_steering" and metrics:
            error = metrics["reconstruction_error_mean"]
            detail = f" (mean |executed - reference| translation: {error:.2f} std)" if error else ""
            return f"reverse flow steering applied on {metrics['steered_chunks']} chunks{detail}"
        return ""


def mode_display(mode: str, tau: int, n_reverse_steps: int | None, num_inference_steps: int) -> str:
    """Short label for the terminal and the live view."""
    if mode == "shared_flow_control":
        return f"{mode} tau={tau}"
    if mode == "shared_reverse_flow_steering":
        return f"{mode} n={n_reverse_steps or num_inference_steps}"
    return mode


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "_".join(words) or "prompt"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        nargs="?",
        const=str(DEFAULT_INTERACTIVE_CONFIG_FILE),
        default=None,
        metavar="PATH",
        help="read the settings below from a YAML file (any flag you also pass wins); PATH defaults to "
        f"{DEFAULT_INTERACTIVE_CONFIG_FILE.name} next to this script",
    )
    parser.add_argument("--policy-path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--mode",
        default="policy",
        choices=MODES,
        help="control mode to start in (changeable live with `mode`)",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=5,
        help="shared_flow_control: number of leading denoising steps your input steers",
    )
    parser.add_argument(
        "--n-reverse-steps",
        type=int,
        default=None,
        metavar="N",
        help="shared_reverse_flow_steering: how many of the policy's denoising steps the reference "
        "is reversed through (default: all of them, i.e. all the way to noise). A smaller N stops "
        "part-way, keeping more of your reference; the forward flow then runs the remaining steps",
    )
    parser.add_argument(
        "--deterministic-corruption",
        nargs="?",
        const=str(DEFAULT_CORRUPTION_FILE),
        default=None,
        metavar="PATH",
        help="apply a fixed 3x3 matrix M (read from PATH, a YAML file with an `M:` key) to x, y and z of the "
        "SpaceMouse/keyboard command at every control step (x -> M @ x); PATH defaults to "
        f"{DEFAULT_CORRUPTION_FILE.name} next to this script; changeable live with `corruption`",
    )
    parser.add_argument(
        "--flow-reversal-adapter",
        nargs="?",
        const=str(DEFAULT_FLOW_ADAPTER_FILE),
        default=None,
        metavar="PATH",
        help="adapt the velocity field used by shared_reverse_flow_steering's reverse integration with a "
        "7x7 matrix F (read from PATH, a YAML file with an `F:` key): x_t += h * F @ v instead of "
        f"x_t += h * v. PATH defaults to {DEFAULT_FLOW_ADAPTER_FILE.name} next to this script "
        "(the identity, a no-op); changeable live with `adapter`",
    )
    parser.add_argument(
        "--input-noise",
        type=float,
        default=0.0,
        metavar="STD",
        help="std of the isotropic Gaussian noise added independently to x, y and z of the "
        "SpaceMouse/keyboard command at every control step while you are commanding "
        "(translation action units, full deflection = 1); 0 = off, changeable live with `noise`",
    )
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
    if args.config is not None:
        try:
            defaults = flatten_config(
                read_yaml_config(args.config), INTERACTIVE_CONFIG_KEYS, str(args.config)
            )
        except (OSError, ValueError) as e:
            parser.error(f"--config: {e}")
        # config sets the defaults; anything passed on the command line still wins
        parser.set_defaults(**defaults)
        args = parser.parse_args()
    if args.input_noise < 0:
        parser.error("--input-noise must be >= 0")
    if not 0 <= args.tau <= 100:
        parser.error("--tau must be >= 0")
    if args.n_reverse_steps is not None and args.n_reverse_steps < 1:
        parser.error("--n-reverse-steps must be >= 1")

    init_logging()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream = FrameStream()
    keyboard = KeyboardReader()
    chain = TeleopChain(keyboard, input_noise=args.input_noise)
    corruption, noisy, reader = chain.corruption, chain.noisy, chain.reader
    if args.deterministic_corruption is not None:
        try:
            path = Path(args.deterministic_corruption)
            corruption.matrix = build_corruption(read_matrix_spec(path))
            corruption.label = path.name
        except (OSError, ValueError) as e:
            parser.error(f"--deterministic-corruption: {e}")
    flow_adapter = FlowAdapter()  # 7x7 F for the reverse integration (off until a file is loaded)
    if args.flow_reversal_adapter is not None:
        try:
            path = Path(args.flow_reversal_adapter)
            flow_adapter.matrix = build_reversal_adapter(read_matrix_spec(path))
            flow_adapter.label = path.name
        except (OSError, ValueError) as e:
            parser.error(f"--flow-reversal-adapter: {e}")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(stream, keyboard, noisy, corruption, flow_adapter)
    )
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
    mode = args.mode
    tau = args.tau  # flow-control: how many leading denoising steps your input steers
    n_reverse_steps = args.n_reverse_steps  # reverse-flow steering: None = all the way to noise
    runner = ModeRunner(
        policy,
        reader,
        preprocessor,
        postprocessor,
        env_preprocessor,
        env_postprocessor,
        flow_adapter=flow_adapter,
    )
    print(f"\nScene: {suite} task {task_id}")
    print(f'Built-in instruction: "{vec_env.envs[0].task_description}"')
    print(
        "Type an instruction (empty = built-in), `tasks`, `task <suite> <id>`, "
        "`mode policy|shared_override|shared_flow_control [tau]"
        "|shared_reverse_flow_steering [n_reverse_steps]|teleop`, "
        "`noise [std]`, `corruption [path|off]`, `adapter [path|off]`, or `quit`.\n"
    )
    if corruption.matrix is not None:
        print(corruption.describe() + "\n")
    if flow_adapter.matrix is not None:
        print(flow_adapter.describe() + "\n")
    if mode != "policy":
        chain.attach_spacemouse()
        announce_mode(mode, tau, policy, flow_adapter, n_reverse_steps)
        print()
    if noisy.std > 0:
        print(f"Input noise: std {noisy.std:.3f} on x/y/z at every step while you command.\n")

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
                if new_mode not in MODES:
                    print(
                        "usage: mode policy|shared_override|shared_flow_control [tau]"
                        "|shared_reverse_flow_steering [n_reverse_steps]|teleop"
                    )
                    continue
                if new_mode == "shared_reverse_flow_steering" and len(tokens) > 2:
                    n_flow_steps = policy.config.num_inference_steps
                    if not tokens[2].isdigit() or not 1 <= int(tokens[2]) <= n_flow_steps:
                        print(f"n_reverse_steps must be an integer in [1, {n_flow_steps}]")
                        continue
                    n_reverse_steps = int(tokens[2])
                if new_mode == "shared_flow_control" and len(tokens) > 2:
                    n_flow_steps = policy.config.num_inference_steps
                    if not tokens[2].isdigit() or not 0 <= int(tokens[2]) <= n_flow_steps:
                        print(f"tau must be an integer in [0, {n_flow_steps}] (denoising steps)")
                        continue
                    tau = int(tokens[2])
                if new_mode != "policy":
                    chain.attach_spacemouse()
                mode = new_mode
                announce_mode(mode, tau, policy, flow_adapter, n_reverse_steps)
                continue

            if line == "noise" or line.startswith("noise "):
                tokens = line.split()
                if len(tokens) == 2:
                    try:
                        noisy.std = float(tokens[1])
                    except ValueError:
                        print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
                        continue
                elif len(tokens) > 2:
                    print("usage: noise <std>   e.g. noise 0.1   (std >= 0, 0 = off)")
                    continue
                if noisy.std > 0:
                    print(
                        f"Input noise: std {noisy.std:.3f} — independent N(0, std²) added to x, y and z "
                        "at every control step while you command (units: full deflection = 1)."
                    )
                else:
                    print("Input noise: off (set with `noise <std>`).")
                continue

            if line == "adapter" or line.startswith("adapter "):
                tokens = line.split()
                if len(tokens) == 2 and tokens[1] == "off":
                    flow_adapter.matrix = None
                    flow_adapter.label = None
                elif len(tokens) == 2:
                    try:
                        path = Path(tokens[1])
                        flow_adapter.matrix = build_reversal_adapter(read_matrix_spec(path))
                        flow_adapter.label = path.name
                    except (OSError, ValueError) as e:
                        print(f"could not load flow-reversal adapter: {e}")
                        continue
                elif len(tokens) > 2:
                    print("usage: adapter [<file.yaml>|off]")
                    continue
                print(flow_adapter.describe())
                continue

            if line == "corruption" or line.startswith("corruption "):
                tokens = line.split()
                if len(tokens) == 2 and tokens[1] == "off":
                    corruption.matrix = None
                    corruption.label = None
                elif len(tokens) == 2:
                    try:
                        path = Path(tokens[1])
                        corruption.matrix = build_corruption(read_matrix_spec(path))
                        corruption.label = path.name
                    except (OSError, ValueError) as e:
                        print(f"could not load corruption matrix: {e}")
                        continue
                elif len(tokens) > 2:
                    print("usage: corruption [<file.yaml>|off]")
                    continue
                print(corruption.describe())
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
                runner.env_preprocessor, runner.env_postprocessor = env_preprocessor, env_postprocessor
                vec_env.reset()
                stream.publish(vec_env.envs[0].render())
                print(f"Scene: {suite} task {task_id}")
                print(f'Built-in instruction: "{vec_env.envs[0].task_description}"')
                continue

            prompt = "manual teleop" if mode == "teleop" else (line or vec_env.envs[0].task_description)
            rollout_kwargs = runner.kwargs(mode, tau, n_reverse_steps)

            mode_label = mode_display(mode, tau, n_reverse_steps, policy.config.num_inference_steps)
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
            video_path = args.output_dir / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{slugify(prompt)}.mp4"
            write_video(video_path, frames, fps=VIDEO_FPS)
            outcome = "SUCCESS" if success else "no success"
            print(f"{outcome} after {steps} steps ({time.time() - start:.0f}s) — video: {video_path}")
            stats = runner.stats_line(mode)
            if stats:
                print(stats)
            print()
    finally:
        close_envs(envs_dict)
        server.shutdown()


if __name__ == "__main__":
    main()
