#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The operator's browser window: live frames, status, and keyboard capture."""

import io
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from teleop import KeyboardReader

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


class LiveView:
    """Serves the page, the MJPEG stream, a status JSON, and takes keyboard events.

    `status_extra()` is called on every /status request and merged into the
    stream's status dict, so the view needs no knowledge of the teleop chain
    beyond the keyboard reader it feeds.
    """

    def __init__(self, port: int, keyboard: KeyboardReader, status_extra: Callable[[], dict]):
        self.port = port
        self.stream = FrameStream()
        self._keyboard = keyboard
        self._status_extra = status_extra
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def start(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def _handler(self):
        stream, keyboard, status_extra = self.stream, self._keyboard, self._status_extra

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
                    status.update(status_extra())
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
