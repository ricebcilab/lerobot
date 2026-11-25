"""The bring-up diagnostic: print what the station, the SpaceMouse and the checkpoint are.

``mars-franka probe --config PATH`` runs four sections and prints each one's status. It sends no
action: the one ``reset()`` is the station's own episode signal and the only other station call is a
read-only ``step(None)``. A section that cannot run says why and the next one still runs; the exit
status is 0 only when every section passed.

Two rules make the report usable when something goes wrong. Every line a section collected is printed
even when a later step of that section fails (the session lost the checkpoint facts because the policy
step raised after collecting them; the section now returns its lines with the failure). And an
unexpected exception prints its traceback: a diagnostic must not swallow the stack.

The SpaceMouse section answers the question the session exposed: a 3Dconnexion device can expose five
hidraw nodes for one mouse and only one carries the reports. It prints every 3Dconnexion node's
``HID_NAME`` and report-descriptor size, tallies every ``(report id, byte length)`` pair over the read
window, and for each pair prints the max ``|int16|`` of the triple at offset 1 and — for a payload of 12
bytes or more — of the second triple at offset 7, so a six-axis single report (or a rotation report
under another id) is visible instead of lost. Then it builds the real ``SpaceMouseSource`` and prints
the node it picks, the rest six-vector and whether it tared.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import select
import shutil
import struct
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import bridge
import numpy as np
import stats
from bridge import ARM_DIMS
from config import Config, load_config
from numpy.typing import NDArray
from station import observation_batch

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "franka.toml"
Report = tuple[list[str], int, str | None]  # (lines, failed checks, unavailable reason)

HOLD_SECONDS = 1.0  # the read-only step(None) window: about 15 control steps at 15 Hz
HOLD_TOLERANCE_RAD = 0.1  # measured uncommanded drift is 0.0001-0.005 rad per step (plant.md)
SPACEMOUSE_SECONDS = 5.0  # two full pushes per axis
REST_WINDOW = 0.5  # after the pushes the operator releases; this catches the zeroing reports
HIDRAW_SYSFS = Path("/sys/class/hidraw")
"""Where a hidraw node's uevent and report descriptor live."""

# lerobot's teleop_spacemouse.REPORT_TRANSLATION/REPORT_ROTATION, kept local so the formatting below
# is a pure function of synthetic report bytes (the section imports lerobot only to find the device).
REPORT_TRANSLATION = 1
REPORT_ROTATION = 2


def _error(error: BaseException | None) -> str:
    message = str(error).strip() if error is not None else ""
    name = type(error).__name__ if error is not None else "no error"
    return f"{name}: {message}" if message else name


def _vec(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{float(value):+.3f}" for value in values) + "]"


def _counts(values: Sequence[int]) -> str:
    """The tally's maxima as raw counts, signed: ``[+350 -40 +12]``."""
    return "[" + " ".join(f"{int(value):+d}" for value in values) + "]"


def _check(lines: list[str], passed: bool, label: str, detail: str = "") -> int:
    lines.append(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return 0 if passed else 1


def _unavailable(lines: list[str], reason: str) -> Report:
    lines.append(f"[UNAVAILABLE] {reason}")
    return lines, 0, reason


def configuration_section(
    config: Config, path, observation, reset_error, lines: list[str] | None = None
) -> Report:
    """The config summary, then q0 / F7 / gain / noise sigma as the loop would resolve them."""
    lines = [] if lines is None else lines
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
        adapters = []
        if mode.adapter_matrix is not None:
            adapters.append(f"matrix {mode.adapter_matrix}")
        if mode.adapter_user is not None:
            adapters.append(f"user {mode.adapter_user}")
        if mode.adapter_task is not None:
            adapters.append(f"task {mode.adapter_task}")
        adapter = f", adapter {', '.join(adapters)}" if adapters else ""
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
        hi("F7 (7x7, the corruption lift the loop applies):")
        for row in matrix7:
            hi("  " + _vec(row))
        hi(f"||F7||2 = {float(np.linalg.norm(matrix7, 2)):.4f}")
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
            hi(
                f"noise sigma (8 dims: 7 command dims + gripper; the loop's noise is the first "
                f"{ARM_DIMS}): {_vec(sigma)} ({ro.noise:g} x action std)"
            )
    return lines, 0, None


def station_section(
    config: Config, station, observation, reset_error, lines: list[str] | None = None
) -> Report:
    """The station: manifest, rates, one reset, the camera feeds, a held second, the action contract."""
    if station is None:
        reason = _error(reset_error) if reset_error is not None else "the station is not available"
        return _unavailable([] if lines is None else lines, reason)
    lines = [] if lines is None else lines
    failures = 0
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


# ---------------------------------------------------------------- the SpaceMouse


@dataclass
class Tally:
    """The SpaceMouse read window: every ``(report id, byte length)`` pair the nodes delivered.

    ``seen`` counts each pair per node; ``peaks`` holds the maxima of the int16 triples at offset 1
    (and at offset 7 when the report is long enough), keyed by ``(node, report id, length)``;
    ``last`` holds those triples as of the newest report and ``order`` records arrival sequence, so
    the rest six-vector can use the newest translation/rotation report.
    """

    seen: dict[str, dict[tuple[int, int], int]] = field(default_factory=dict)
    peaks: dict[tuple[str, int, int], tuple[tuple[int, int, int], tuple[int, int, int] | None]] = field(
        default_factory=dict
    )
    last: dict[tuple[str, int, int], tuple[tuple[int, int, int], tuple[int, int, int] | None]] = field(
        default_factory=dict
    )
    order: dict[tuple[str, int, int], int] = field(default_factory=dict)
    serial: int = 0


@dataclass(frozen=True)
class SourceFacts:
    """What the real ``SpaceMouseSource`` reported when the probe built one after the read window."""

    path: str
    tare: bool
    rest: NDArray | None
    at_rest: NDArray


def hidraw_nodes() -> list[str]:
    """Every /dev/hidraw* node, sorted; empty on a machine without Linux hidraw devices."""
    return sorted(str(path) for path in Path("/dev").glob("hidraw*"))


def _uevent(node: str, sysfs: Path | str | None = None) -> str:
    """A node's uevent (``HID_ID``, ``HID_NAME``, ...); "" when the sysfs entry is unreadable."""
    root = HIDRAW_SYSFS if sysfs is None else Path(sysfs)
    try:
        return (root / Path(node).name / "device" / "uevent").read_text()
    except OSError:
        return ""


def _uevent_field(uevent: str, name: str) -> str | None:
    prefix = f"{name}="
    for line in uevent.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _report_descriptor_size(node: str, sysfs: Path | str | None = None) -> int | None:
    """The HID report descriptor's byte length from sysfs, or None when it is not readable."""
    root = HIDRAW_SYSFS if sysfs is None else Path(sysfs)
    try:
        return (root / Path(node).name / "device" / "report_descriptor").stat().st_size
    except OSError:
        return None


def _node_line(node: str, is_spacemouse: bool, sysfs: Path | str | None = None) -> str:
    """One hidraw node; a 3Dconnexion node also says its ``HID_NAME`` and descriptor size."""
    uevent = _uevent(node, sysfs)
    hid = _uevent_field(uevent, "HID_ID") or "?"
    kind = "3Dconnexion" if is_spacemouse else "other"
    line = f"hidraw {node}: HID_ID {hid} ({kind})"
    if not is_spacemouse:
        return line
    name = _uevent_field(uevent, "HID_NAME")
    line += f", HID_NAME {name!r}" if name else ", HID_NAME ?"
    size = _report_descriptor_size(node, sysfs)
    line += f", report descriptor {size} bytes" if size is not None else ", report descriptor unavailable"
    return line


def _tally(tally: Tally, node: str, data: bytes) -> None:
    """Count one raw hidraw report and keep the maxima of its int16 triples.

    The first byte is the report id, so a translation report (three int16) is 7 bytes and the second
    triple at offset 7 needs 12 bytes after the id (a 13-byte report: the six-axis single report's
    shape). Reports shorter than a triple are counted but have no maxima; that which ids arrive is
    the finding, so a short report still counts toward its pair.
    """
    pair = (data[0], len(data))
    counts = tally.seen.setdefault(node, {})
    counts[pair] = counts.get(pair, 0) + 1
    if len(data) < 7:
        return
    first = struct.unpack_from("<hhh", data, 1)
    second = struct.unpack_from("<hhh", data, 7) if len(data) >= 13 else None
    key = (node, *pair)
    tally.serial += 1
    tally.order[key] = tally.serial
    previous = tally.peaks.get(key)
    if previous is None:
        tally.peaks[key] = (
            tuple(abs(value) for value in first),
            None if second is None else tuple(abs(value) for value in second),
        )
    else:
        peak_first, peak_second = previous
        tally.peaks[key] = (
            tuple(max(abs(value), peak) for value, peak in zip(first, peak_first, strict=True)),
            None
            if peak_second is None or second is None
            else tuple(max(abs(value), peak) for value, peak in zip(second, peak_second, strict=True)),
        )
    tally.last[key] = (first, second)


def _read_reports(opened: dict, seconds: float, tally: Tally) -> None:
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
                _tally(tally, opened[fd], data)
            else:
                fds.remove(fd)  # b"" means the device went away, not a report


def _max_counts(tally: Tally, report_id: int) -> tuple[int, int, int] | None:
    """The per-axis largest |int16| of a report id's triples, across nodes and lengths; None if absent."""
    triples = [maxima[0] for key, maxima in tally.peaks.items() if key[1] == report_id]
    if not triples:
        return None
    return tuple(max(triple[axis] for triple in triples) for axis in range(3))  # type: ignore[return-value]


def _newest(tally: Tally, report_id: int) -> tuple[int, int, int] | None:
    """The first triple of the newest report of ``report_id`` across the nodes; None if absent."""
    keys = [key for key in tally.last if key[1] == report_id]
    if not keys:
        return None
    return tally.last[max(keys, key=tally.order.__getitem__)][0]


def _tally_lines(tally: Tally) -> list[str]:
    """Every pair's count per node, then every pair's triple maxima — the rotation-shape evidence."""
    lines: list[str] = []
    for node in sorted(tally.seen):
        pairs = ", ".join(
            f"id {report} ({length} bytes) x {count}"
            for (report, length), count in sorted(tally.seen[node].items())
        )
        lines.append(f"  {node} reports: {pairs}")
    for key in sorted(tally.peaks, key=lambda item: (item[0], item[1], item[2])):
        node, report, length = key
        first, second = tally.peaks[key]
        line = f"  {node} id {report} ({length} bytes): max |int16| offset 1 {_counts(first)}"
        if second is not None:
            line += f", offset 7 {_counts(second)}"
        lines.append(line)
    return lines


def _rest_from_tally(tally: Tally, config: Config) -> NDArray:
    """The rest six-vector from the newest raw reports, scaled and mapped like the source would."""
    translation = _newest(tally, REPORT_TRANSLATION) or (0, 0, 0)
    rotation = _newest(tally, REPORT_ROTATION) or (0, 0, 0)
    raw = np.asarray([*translation, *rotation], dtype=np.float64)
    scale = np.asarray([config.spacemouse.full_scale] * 3 + [config.spacemouse.rotation_full_scale] * 3)
    source = np.asarray(config.spacemouse.source, dtype=np.intp)
    sign = np.asarray(config.spacemouse.sign, dtype=np.float64)
    return raw[source] * sign / scale


def _source_facts(config: Config) -> tuple[SourceFacts | None, Exception | None]:
    """Build the real ``SpaceMouseSource`` the session uses, and record what it chose and tared."""
    try:
        from source import SpaceMouseSource

        source = SpaceMouseSource(
            path=config.spacemouse.path or None,
            hid_id=config.spacemouse.hid_id or None,
            full_scale=config.spacemouse.full_scale,
            rotation_full_scale=config.spacemouse.rotation_full_scale,
            source=config.spacemouse.source,
            sign=config.spacemouse.sign,
            tare=config.spacemouse.tare,
        )
    except (OSError, ImportError) as error:  # a quiet or unreadable device is a report line
        return None, error
    try:
        facts = SourceFacts(
            path=source.path,
            tare=source.tare,
            rest=source.rest,
            at_rest=np.asarray(source.deflection(), dtype=np.float64).copy(),
        )
    finally:
        source.close()
    return facts, None


def _mouse_report(
    lines: list[str],
    failures: int,
    config: Config,
    tally: Tally,
    source: SourceFacts | None,
    source_error: Exception | None,
) -> tuple[list[str], int]:
    """The tallied report, the checks, the scales to copy, and the source's pick, rest and tare."""
    hi = lines.append
    lines.extend(_tally_lines(tally))
    translation = _max_counts(tally, REPORT_TRANSLATION)
    detail = (
        "none arrived"
        if translation is None
        else f"max |counts| x {translation[0]} y {translation[1]} z {translation[2]}"
    )
    failures += _check(
        lines, translation is not None, f"id {REPORT_TRANSLATION} (translation) reports", detail
    )
    rotation = _max_counts(tally, REPORT_ROTATION)
    if rotation is None:
        failures += _check(
            lines, False, f"id {REPORT_ROTATION} (rotation) reports", "none arrived on any node"
        )
        hi(
            "[DIAG] no rotation report arrived on any node: twist each rotation axis to its stop again "
            "during the 5 s window. The device may report rotation under another id or as the second "
            "triple of a wider single report; the (id, bytes) tally above shows both, so a pair that is "
            "not id 1 with 7 bytes — a six-axis report is 13 bytes — is the place to look, and its "
            "second triple would be the rotation."
        )
    else:
        detail = f"max |counts| x {rotation[0]} y {rotation[1]} z {rotation[2]}"
        failures += _check(lines, True, f"id {REPORT_ROTATION} (rotation) reports", detail)
    if translation is not None:
        hi(f"copy into [spacemouse] full_scale = {max(translation):g}; the shipped 350.0 is unverified")
    if rotation is not None:
        hi(f"copy into [spacemouse] rotation_full_scale = {max(rotation):g}; the shipped 350.0 is unverified")
    source_map, sign = list(config.spacemouse.source), [float(value) for value in config.spacemouse.sign]
    hi(f"configured source {source_map}, sign {sign}; channels 0 x, 1 y, 2 z, 3 rx, 4 ry, 5 rz")
    fallback = _rest_from_tally(tally, config)
    if source is not None:
        hi(f"source would pick: {source.path}")
        hi(f"source tare: {'on' if source.tare else 'off'}")
        if source.rest is not None:
            hi(
                f"rest six-vector [x y z rx ry rz]: {_vec(source.rest)} "
                "(captured at start, after the node reported and the stick settled)"
            )
            hi(f"at rest after tare: {_vec(source.at_rest)}")
        else:
            hi(f"rest six-vector [x y z rx ry rz]: {_vec(fallback)} (no tare captured; last raw reports)")
    else:
        failures += _check(lines, False, "source", _error(source_error))
        hi(
            f"rest six-vector [x y z rx ry rz]: {_vec(fallback)} (from the last raw reports; the source is absent)"
        )
    return lines, failures


def spacemouse_section(config: Config, lines: list[str] | None = None) -> Report:
    """The 3Dconnexion nodes and their reports, the tallies, the scales, the source's pick and tare."""
    nodes = hidraw_nodes()
    if not nodes:
        return _unavailable(
            [] if lines is None else lines, "no /dev/hidraw* node: the device is read from Linux hidraw"
        )
    try:
        sm = importlib.import_module("lerobot.teleoperators.spacemouse.teleop_spacemouse")

        _is_3dconnexion = sm._is_3dconnexion
    except (ImportError, OSError) as error:
        return _unavailable(
            [] if lines is None else lines,
            f"lerobot's spacemouse module is not importable ({_error(error)})",
        )
    lines = [] if lines is None else lines
    failures = 0
    hi = lines.append
    spacemice = []
    for node in nodes:
        uevent = _uevent(node)
        is_mouse = _is_3dconnexion(uevent)
        hi(_node_line(node, is_mouse))
        if is_mouse:
            spacemice.append(node)
    if not spacemice:
        return _unavailable(lines, f"no 3Dconnexion node among the {len(nodes)} hidraw nodes")
    mouse = config.spacemouse
    hi(f"configured [spacemouse]: path {mouse.path!r}, hid_id {mouse.hid_id!r}, tare {mouse.tare}")
    opened = {}
    for node in spacemice:
        try:
            opened[os.open(node, os.O_RDONLY | os.O_NONBLOCK)] = node
        except OSError as error:
            failures += _check(lines, False, f"readable {node}", _error(error))
    if not opened:
        return _unavailable(lines, "no 3Dconnexion node could be opened for reading")
    tally = Tally()
    try:
        print("  push each translation axis to its stop (+x right, +y toward you, +z down), then", flush=True)
        print("  twist each rotation axis to its stop (rx, ry, rz)", flush=True)
        print(f"  reading {len(opened)} node(s) for {SPACEMOUSE_SECONDS:g} s; move the stick now", flush=True)
        _read_reports(opened, SPACEMOUSE_SECONDS, tally)
        print("  release the stick and hold it still; reading the rest state", flush=True)
        _read_reports(opened, REST_WINDOW, tally)
    finally:
        for fd in opened:
            with contextlib.suppress(OSError):
                os.close(fd)
    source, source_error = _source_facts(config)
    lines, failures = _mouse_report(lines, failures, config, tally, source, source_error)
    return lines, failures, None


# ---------------------------------------------------------------- the checkpoint


def policy_section(config: Config, observation, lines: list[str] | None = None) -> Report:
    """The checkpoint, the statistics file, and one timed prepare + act."""
    if shutil.which("nvidia-smi") is None:
        return _unavailable(
            [] if lines is None else lines,
            "needs CUDA: no nvidia-smi on PATH; the checkpoint is loaded on the GPU",
        )
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        import lerobot.policies  # noqa: F401  registers the pi05 config class
        from lerobot.configs.policies import PreTrainedConfig

        ck = PreTrainedConfig.from_pretrained(str(config.policy.path))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return _unavailable(
            [] if lines is None else lines,
            f"needs CUDA and the checkpoint's config ({_error(error)})",
        )
    lines = [] if lines is None else lines
    failures = 0
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
        return _unavailable(
            lines, f"statistics file {path} is absent: download openpi's droid/norm_stats.json"
        )
    try:
        groups = stats._groups(path)
        panels = stats._quantile_pair(groups, "actions", path, stats.STATION_DIMS)
        stats._quantile_pair(groups, "state", path, stats.STATION_DIMS)
        dims = f"{stats.CHECKPOINT_WIDTH} wide, {stats.STATION_DIMS} meaningful dims"
        hi(f"statistics file: {path}, {dims}")
    except (OSError, ValueError) as error:
        return _unavailable(lines, f"statistics {path}: {_error(error)}")
    hi(f"action mean (8): {_vec((panels['q01'] + panels['q99']) / 2.0)}")
    hi(f"action std (8): {_vec(np.maximum((panels['q99'] - panels['q01']) / 2.0, 1e-6))}")
    try:
        from policy import load_bundle

        started = time.monotonic()
        bundle = load_bundle(config)
        load_ms = 1000.0 * (time.monotonic() - started)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return _unavailable(lines, f"load_bundle: {_error(error)}")
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
    started = time.monotonic()
    prepared = bundle.prepare(batch)  # an unexpected failure here is main's traceback, not a one-liner
    prepare_ms = 1000.0 * (time.monotonic() - started)
    started = time.monotonic()
    action = bundle.act(prepared, None)
    act_ms = 1000.0 * (time.monotonic() - started)
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
        (
            "configuration",
            lambda lines: configuration_section(config, args.config, observation, reset_error, lines),
        ),
        ("station", lambda lines: station_section(config, station, observation, reset_error, lines)),
        ("spacemouse", lambda lines: spacemouse_section(config, lines)),
        ("policy", lambda lines: policy_section(config, observation, lines)),
    )
    passed = 0
    try:
        for index, (title, build) in enumerate(builders, 1):
            print(f"=== {index}/{len(builders)} {title} ===", flush=True)
            lines: list[str] = []
            raised: BaseException | None = None
            try:
                _, failures, unavailable = build(lines)
            except Exception as error:  # one section's bug must not hide the other three
                raised, failures, unavailable = error, 1, None
                lines.append(f"[FAIL] the section raised {_error(error)} (traceback below)")
            for line in lines:
                print(f"  {line}", flush=True)
            if raised is not None:  # a diagnostic must not swallow the stack
                traceback.print_exception(type(raised), raised, raised.__traceback__)
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
