"""The bring-up diagnostic: print what the station, the SpaceMouse and the checkpoint are.

``mars-franka probe --config PATH`` runs four sections and prints each one's status. It sends no
action: the one ``reset()`` is the station's own episode signal and the only other station call is a
read-only ``step(None)``. A section that cannot run says why and the next one still runs; the exit
status is 0 only when every section passed.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import shutil
import struct
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import bridge
import numpy as np
import stats
from bridge import ARM_DIMS
from config import Config, load_config
from station import observation_batch

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "franka.toml"
Report = tuple[list[str], int, str | None]  # (lines, failed checks, unavailable reason)

HOLD_SECONDS = 1.0  # the read-only step(None) window: about 15 control steps at 15 Hz
HOLD_TOLERANCE_RAD = 0.1  # measured uncommanded drift is 0.0001-0.005 rad per step (plant.md)
SPACEMOUSE_SECONDS = 5.0  # two full pushes per axis
REST_WINDOW = 0.5  # after the pushes the operator releases; this catches the zeroing reports


def _error(error: BaseException | None) -> str:
    message = str(error).strip() if error is not None else ""
    name = type(error).__name__ if error is not None else "no error"
    return f"{name}: {message}" if message else name


def _vec(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{float(value):+.3f}" for value in values) + "]"


def _check(lines: list[str], passed: bool, label: str, detail: str = "") -> int:
    lines.append(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return 0 if passed else 1


def _unavailable(reason: str) -> Report:
    return [f"[UNAVAILABLE] {reason}"], 0, reason


def configuration_section(config: Config, path, observation, reset_error) -> Report:
    """The config summary, then q0 / F7 / gain / noise sigma as the loop would resolve them."""
    lines: list[str] = []
    hi = lines.append
    st, cmd, pol, cam, ro = config.station, config.command, config.policy, config.cameras, config.readout
    hi(f"config: {path} (unknown keys refused); command: probe")
    hi(f"station: {st.arm} at {st.host}:{st.port}, rate {st.rate:g} Hz; policy: {pol.path} on {pol.device}")
    hi(
        f"policy: n_action_steps {pol.n_action_steps}, dtype {pol.dtype}; cameras: exterior "
        f"{cam.exterior!r}, wrist {cam.wrist!r}, third {cam.third or 'omitted (zero mask)'}"
    )
    hi(
        f"command: translation_gain {cmd.translation_gain:g}, rotation_gain {cmd.rotation_gain:g}, "
        f"twist_length {cmd.twist_length:g}, deadband {cmd.deadband:g}, gain {cmd.gain}"
    )
    hi(
        f"readout: corruption {ro.corruption!r}, noise {ro.noise:g} sigma, seed {ro.seed}; steering: "
        f"n_reversal_steps {config.steering.n_reversal_steps}, pinned_steps {config.steering.pinned_steps}"
    )
    modes = []
    for name, mode in sorted(config.modes.items()):
        if mode.mode == "flow_control":
            knobs = f"tau {mode.tau}, lam {mode.lam}, ic_gain {mode.ic_gain:g}"
        else:
            knobs = f"n_reversal_steps {mode.n_reversal_steps}, pinned_steps {mode.pinned_steps}"
        adapter = f", adapter_matrix {mode.adapter_matrix}" if mode.adapter_matrix is not None else ""
        modes.append(f"{name}={mode.mode}({knobs}{adapter})")
    hi("modes: " + "; ".join(modes) + "; plain teleop is clean, --corrupted/shared use [readout]")
    q0, source = None, None
    if cmd.q0:
        q0, source = np.asarray(cmd.q0, dtype=np.float64), "from [command] q0"
    elif observation is not None:
        try:
            q0, source = bridge.nominal_pose(observation, st.arm), "the station reset pose"
        except ValueError as error:
            hi(f"[INFO] q0: {error}")
    if q0 is None:
        why = _error(reset_error) if reset_error is not None else "no reset observation"
        hi(f"[INFO] q0: the reset pose is unavailable ({why}); set [command] q0 to pin it")
    else:
        hi(f"q0: {_vec(q0)} ({source})")
    spec, matrix7 = ro.corruption, None
    if spec is None:
        hi("corruption: off (F7 is the identity)")
    elif q0 is None:
        hi("[INFO] F7, gain: unresolved without q0; the identity is not this session's corruption")
    else:
        random = isinstance(spec, dict) and set(spec) == {"random"}
        rng = np.random.default_rng([int(ro.seed), 0]) if random else None
        matrix7 = bridge.corruption_lift(bridge.matrix6_from_spec(spec, rng), q0, cmd.twist_length)
        hi(f"F7 (7x7, ||F7||2 = {float(np.linalg.norm(matrix7, 2)):.4f}; rows omitted)")
        if rng is not None:
            hi("  one draw for episode 0: a {random} family is redrawn every episode")
    if cmd.gain != "auto":
        hi(f"gain: {float(cmd.gain):.4f} (set in [command])")
    elif spec is None:
        hi("gain: 1.0 (auto with the corruption off)")
    elif matrix7 is not None:
        hi(f"gain: {1.0 / float(np.linalg.norm(matrix7, 2)):.4f} (auto = 1/||F7||2)")
    if not ro.noise:
        hi("noise sigma: off (readout.noise 0.0)")
    else:
        try:
            sigma = ro.noise * stats.action_std(pol.stats)
        except (OSError, ValueError) as error:
            hi(f"[INFO] noise sigma: cannot be scaled: {_error(error)}; the loop refuses noise")
        else:
            hi(f"noise sigma (7, action units): {_vec(sigma)} ({ro.noise:g} x action std)")
    return lines, 0, None


def station_section(config: Config, station, observation, reset_error) -> Report:
    """The station: manifest, rates, one reset, the camera feeds, a held second, the action contract."""
    if station is None:
        reason = _error(reset_error) if reset_error is not None else "the station is not available"
        return _unavailable(reason)
    lines, failures = [], 0
    hi = lines.append
    m, cfg = station.manifest, config.station
    robot = getattr(station, "_robot", None)
    raw = getattr(robot, "_manifest", None) or {}
    hi(
        f"manifest: arms {list(m.arms) or 'none'}, cameras {list(m.cameras) or 'none'}, "
        f"action keys {list(m.action_keys) or 'none'}, native rate {m.control_rate:g} Hz"
    )
    arms = m.arms
    failures += _check(lines, len(arms) == 1, "single arm", f"{list(arms)} (the loop drives exactly one)")
    rate = f"native {m.control_rate:g} vs requested {cfg.rate:g} Hz"
    failures += _check(lines, m.control_rate >= cfg.rate, "rate", rate)
    arm = arms[0] if arms else cfg.arm
    if observation is None:
        reason = _error(reset_error) if reset_error is not None else "no observation"
        failures += _check(lines, False, "reset()", reason)
    else:
        for key in sorted(observation):
            value = np.asarray(observation[key])
            hi(f"  {key}: {value.shape} {value.dtype}")
        try:
            hi(f"reset pose q: {_vec(bridge.nominal_pose(observation, arm))}")
        except ValueError as error:
            failures += _check(lines, False, "reset pose", str(error))
    for role in ("exterior", "wrist", "third"):
        feed = getattr(config.cameras, role)
        if feed is None:
            hi("camera third: omitted (the model fills it with a -1 image and a zero mask)")
            continue
        if observation is None:
            continue
        if feed not in observation:
            failures += _check(lines, False, f"camera {role}", f"{feed!r} is not in the observation")
            continue
        frame = np.asarray(observation[feed])
        valid = frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        detail = f"{feed}: {frame.shape} {frame.dtype}, mean {frame.mean():.1f}/255"
        failures += _check(lines, valid, f"camera {role}", detail)
    if observation is not None:
        try:
            start = bridge.nominal_pose(observation, arm)
            started = time.monotonic()
            pose, steps = start, 0
            while (elapsed := time.monotonic() - started) < HOLD_SECONDS:
                pose = bridge.nominal_pose(robot.step(None), arm)
                steps += 1
        except Exception as error:
            failures += _check(lines, False, "step(None)", _error(error))
        else:
            delta = np.abs(pose - start)
            detail = f"max |dq| {delta.max():.4f} rad over {elapsed:.2f} s, {steps} steps"
            failures += _check(lines, delta.max() <= HOLD_TOLERANCE_RAD, "holds without a command", detail)
    entry = next((item for item in raw.get("arms", []) if item.get("name") == arm), {})
    keys = list(m.action_keys)
    contract = ((f"{arm}/joint_velocity", "joint velocities"), (f"{arm}/gripper_position", "gripper bit"))
    for key, what in contract:
        passed = key in keys
        detail = what if passed else f"not in action_keys {keys}"
        hi(f"[{'PASS' if passed else 'FAIL'}] {key}: {detail}")
        failures += not passed
    mode = entry.get("command_mode")
    if mode is None:
        hi("[INFO] command mode: not advertised in the arm table; the action keys are the contract")
    else:
        failures += _check(lines, mode == "joint_velocity", "command mode", f"advertised {mode!r}")
    return lines, failures, None


def hidraw_nodes() -> list[str]:
    """Every /dev/hidraw* node, sorted; empty on a machine without Linux hidraw devices."""
    return sorted(str(path) for path in Path("/dev").glob("hidraw*"))


def _tally(node: str, data: bytes, ids, seen: dict, peaks: dict, last: dict) -> None:
    report = data[0]
    counts = seen.setdefault(node, {})
    counts[report] = counts.get(report, 0) + 1
    if report in ids[:2] and len(data) >= 7:
        values = struct.unpack_from("<hhh", data, 1)
        last[report] = values
        peaks[report] = tuple(
            max(abs(value), peak) for value, peak in zip(values, peaks.get(report, (0, 0, 0)), strict=True)
        )


def _read_reports(opened: dict, seconds: float, ids, seen: dict, peaks: dict, last: dict) -> None:
    fds, deadline = list(opened), time.monotonic() + seconds
    while fds and time.monotonic() < deadline:
        ready, _, _ = select.select(fds, [], [], 0.25)
        for fd in ready:
            try:
                data = os.read(fd, 64)
            except OSError:
                fds.remove(fd)
                continue
            if data:
                _tally(opened[fd], data, ids, seen, peaks, last)
            else:
                fds.remove(fd)  # b"" means the device went away, not a report


def _mouse_report(
    lines: list[str], failures: int, config: Config, seen: dict, peaks: dict, last: dict, ids
) -> tuple[list[str], int]:
    hi = lines.append
    for node in sorted(seen):
        hi(f"{node}: " + ", ".join(f"id {report} x {count}" for report, count in sorted(seen[node].items())))
    for report, name in ((ids[0], "translation"), (ids[1], "rotation")):
        peak = peaks.get(report)
        detail = "none arrived" if peak is None else f"max |counts| x {peak[0]} y {peak[1]} z {peak[2]}"
        failures += _check(lines, peak is not None, f"id {report} ({name}) reports", detail)
    if ids[0] in peaks:
        hi(f"copy into [spacemouse] full_scale = {max(peaks[ids[0]]):g}; the shipped 350.0 is unverified")
    if ids[1] in peaks:
        hi(
            f"copy into [spacemouse] rotation_full_scale = {max(peaks[ids[1]]):g}; the shipped 350.0 is unverified"
        )
    source, sign = list(config.spacemouse.source), [float(value) for value in config.spacemouse.sign]
    hi(f"configured source {source}, sign {sign}; channels 0 x, 1 y, 2 z, 3 rx, 4 ry, 5 rz")
    raw = np.asarray([*last.get(ids[0], (0, 0, 0)), *last.get(ids[1], (0, 0, 0))], dtype=np.float64)
    scale = np.asarray([config.spacemouse.full_scale] * 3 + [config.spacemouse.rotation_full_scale] * 3)
    hi(f"rest-state six-vector [x y z rx ry rz]: {_vec(raw[source] * np.asarray(sign) / scale)}")
    return lines, failures


def spacemouse_section(config: Config) -> Report:
    """The 3Dconnexion nodes, their reports, the full scales and the rest state."""
    nodes = hidraw_nodes()
    if not nodes:
        return _unavailable("no /dev/hidraw* node: the device is read straight from Linux hidraw")
    try:
        from lerobot.teleoperators.spacemouse import teleop_spacemouse as sm

        _is_3dconnexion, find_spacemouse = sm._is_3dconnexion, sm.find_spacemouse
    except Exception as error:
        return _unavailable(f"lerobot's spacemouse module is not importable ({_error(error)})")
    lines, failures = [], 0
    hi = lines.append
    spacemice = []
    for node in nodes:
        try:
            uevent = (Path("/sys/class/hidraw") / Path(node).name / "device" / "uevent").read_text()
        except OSError:
            uevent = ""
        hid = next((line[7:].strip() for line in uevent.splitlines() if line.startswith("HID_ID=")), "?")
        is_mouse = _is_3dconnexion(uevent)
        hi(f"hidraw {node}: HID_ID {hid} ({'3Dconnexion' if is_mouse else 'other'})")
        if is_mouse:
            spacemice.append(node)
    if not spacemice:
        return _unavailable(f"no 3Dconnexion node among the {len(nodes)} hidraw nodes")
    try:
        configured = config.spacemouse.path or find_spacemouse(config.spacemouse.hid_id or None)
        lines.append(f"configured node: {configured!r}")
    except OSError as error:
        lines.append(f"[INFO] configured node: {_error(error)}")
    opened = {}
    for node in spacemice:
        try:
            opened[os.open(node, os.O_RDONLY | os.O_NONBLOCK)] = node
        except OSError as error:
            failures += _check(lines, False, f"readable {node}", _error(error))
    if not opened:
        return _unavailable("no 3Dconnexion node could be opened for reading")
    ids, seen, peaks, last = (sm.REPORT_TRANSLATION, sm.REPORT_ROTATION, sm.REPORT_BUTTONS), {}, {}, {}
    try:
        print("  push each translation axis to its stop (+x right, +y toward you, +z down), then", flush=True)
        print("  twist each rotation axis to its stop (rx, ry, rz)", flush=True)
        print(f"  reading {len(opened)} node(s) for {SPACEMOUSE_SECONDS:g} s; move the stick now", flush=True)
        _read_reports(opened, SPACEMOUSE_SECONDS, ids, seen, peaks, last)
        print("  release the stick and hold it still; reading the rest state", flush=True)
        _read_reports(opened, REST_WINDOW, ids, seen, peaks, last)
    finally:
        for fd in opened:
            with contextlib.suppress(OSError):
                os.close(fd)
    return _mouse_report(lines, failures, config, seen, peaks, last, ids) + (None,)


def policy_section(config: Config, observation) -> Report:
    """The checkpoint, the statistics file, and one timed prepare + act."""
    if shutil.which("nvidia-smi") is None:
        return _unavailable("needs CUDA: no nvidia-smi on PATH; the checkpoint is loaded on the GPU")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        import lerobot.policies  # noqa: F401  registers the pi05 config class
        from lerobot.configs.policies import PreTrainedConfig

        ck = PreTrainedConfig.from_pretrained(str(config.policy.path))
    except Exception as error:
        return _unavailable(f"needs CUDA and the checkpoint's config ({_error(error)})")
    lines, failures = [], 0
    hi = lines.append
    images = [key for key in (getattr(ck, "input_features", None) or {}) if "image" in key]
    steps = f"n_action_steps {ck.n_action_steps} (config {config.policy.n_action_steps})"
    hi(f"checkpoint: {config.policy.path}; type {ck.type}; chunk_size {ck.chunk_size}")
    hi(f"{steps}; num_inference_steps {ck.num_inference_steps}")
    hi(
        f"max_action_dim {ck.max_action_dim}, max_state_dim {ck.max_state_dim}, "
        f"use_relative_actions {ck.use_relative_actions}, dtype {ck.dtype} (config {config.policy.dtype})"
    )
    hi(f"image keys: {', '.join(images) or 'none'}")
    path = Path(config.policy.stats)
    if not path.is_file():
        return _unavailable(f"statistics file {path} is absent: download openpi's droid/norm_stats.json")
    try:
        groups = stats._groups(path)
        panels = stats._quantile_pair(groups, "actions", path, stats.STATION_DIMS)
        stats._quantile_pair(groups, "state", path, stats.STATION_DIMS)
        dims = f"{stats.CHECKPOINT_WIDTH} wide, {stats.STATION_DIMS} meaningful dims"
        hi(f"statistics file: {path}, {dims}")
    except (OSError, ValueError) as error:
        return _unavailable(f"statistics {path}: {_error(error)}")
    hi(f"action mean (8): {_vec((panels['q01'] + panels['q99']) / 2.0)}")
    hi(f"action std (8): {_vec(np.maximum((panels['q99'] - panels['q01']) / 2.0, 1e-6))}")
    try:
        from policy import load_bundle

        started = time.monotonic()
        bundle = load_bundle(config)
        load_ms = 1000.0 * (time.monotonic() - started)
    except Exception as error:
        return _unavailable(f"load_bundle: {_error(error)}")
    hi(f"load_bundle: {load_ms:.0f} ms")
    if observation is None:
        frame, arm = np.zeros((480, 640, 3), dtype=np.uint8), config.station.arm
        observation = {f"{arm}/joint_position": np.zeros(ARM_DIMS), f"{arm}/gripper_position": np.zeros(1)}
        for feed in (config.cameras.exterior, config.cameras.wrist, config.cameras.third):
            if feed is not None:
                observation[feed] = frame.copy()
    cameras = {
        "exterior": config.cameras.exterior,
        "wrist": config.cameras.wrist,
        "third": config.cameras.third,
    }
    batch = observation_batch(observation, arm=config.station.arm, cameras=cameras, task="probe")
    try:
        started = time.monotonic()
        prepared = bundle.prepare(batch)
        prepare_ms = 1000.0 * (time.monotonic() - started)
        started = time.monotonic()
        action = bundle.act(prepared, None)
        act_ms = 1000.0 * (time.monotonic() - started)
    except Exception as error:
        return _unavailable(f"prepare + act: {_error(error)}")
    tick = 1000.0 / config.station.rate
    hi(
        f"chunk time {prepare_ms + act_ms:.0f} ms (prepare {prepare_ms:.0f} + act {act_ms:.0f}); "
        f"one {config.station.rate:g} Hz tick is {tick:.1f} ms"
    )
    hi(f"action (8, action units): {_vec(action)}")
    keys = sorted(key for key in prepared if key.startswith("observation.images."))
    third = "present" if "observation.images.right_wrist_0_rgb" in prepared else "omitted (zero mask)"
    hi(f"image keys present: {', '.join(keys) or 'none'}; third camera: {third}")
    return lines, failures, None


def main(argv: list[str] | None = None) -> int:
    """Print the four sections; 0 only when all passed, 1 otherwise, 2 for an unreadable config."""
    parser = argparse.ArgumentParser(
        prog="mars-franka probe",
        description="Print the station, the SpaceMouse and the checkpoint; move nothing.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, metavar="PATH")
    args = parser.parse_args(argv)
    print(
        f"mars-franka probe: config {args.config}; sends no action (one reset, read-only steps)", flush=True
    )
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"mars-franka probe: {args.config}: {error}", file=sys.stderr)
        return 2
    print("reading the station: one reset(), no action", flush=True)
    station = observation = reset_error = None
    try:
        from station import C3poStation

        station = C3poStation(config.station.host, config.station.port, config.station.rate)
        observation = dict(station.reset())
    except Exception as error:
        reset_error = error
    builders = (
        ("configuration", lambda: configuration_section(config, args.config, observation, reset_error)),
        ("station", lambda: station_section(config, station, observation, reset_error)),
        ("spacemouse", lambda: spacemouse_section(config)),
        ("policy", lambda: policy_section(config, observation)),
    )
    passed = 0
    try:
        for index, (title, build) in enumerate(builders, 1):
            print(f"=== {index}/{len(builders)} {title} ===", flush=True)
            try:
                lines, failures, unavailable = build()
            except Exception as error:  # one section's bug must not hide the other three
                lines, failures, unavailable = [f"[FAIL] the section raised {_error(error)}"], 1, None
            for line in lines:
                print(f"  {line}", flush=True)
            status = f"FAIL ({failures} checks failed)" if failures else "PASS"
            if unavailable is not None:
                status = f"UNAVAILABLE: {unavailable}"
            print(f"status: {status}", flush=True)
            passed += failures == 0 and unavailable is None
    finally:
        if station is not None:
            with contextlib.suppress(Exception):
                station.close()
    print(f"probe: {passed}/{len(builders)} sections passed", flush=True)
    return 0 if passed == len(builders) else 1


if __name__ == "__main__":
    raise SystemExit(main())
