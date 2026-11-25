"""The bring-up diagnostic: print what the station, the operator's device and the checkpoint are.

``mars-franka probe --config PATH`` runs four independent sections and prints each one's status:

1. configuration — the parsed config, the readout and the resolved corruption (``q0``, ``F7``, the
   ``auto`` gain, the noise sigma), using ``config.py``, ``bridge.py`` and ``stats.py`` rather than
   re-deriving their numbers;
2. station — c3po's manifest, its native rate against the loop's requested rate, one ``reset()`` and
   the observation it returns, the camera feeds' frames, a one-second read-only ``step(None)``, and the
   action contract (``{arm}/joint_velocity`` and ``{arm}/gripper_position`` in ``action_keys``);
3. SpaceMouse — every 3Dconnexion hidraw node, the report ids each one sends, the peak raw counts per
   translation and rotation axis (the full scales to write into ``[spacemouse]``), the configured
   ``source``/``sign`` mapping and the rest-state six-vector;
4. policy — the checkpoint's facts from ``PreTrainedConfig.from_pretrained``, the statistics file,
   a timed ``load_bundle`` and a timed ``prepare`` + ``act`` on one observation.

**It sends no action.** The one ``reset()`` is the station's own episode signal and the only other
station call is a read-only ``step(None)``; no action dictionary is ever built, so nothing here can
drive the arm. The one thing the probe cannot promise is the station's own behaviour on ``reset()``:
if r2d2 moves the arm there, that is r2d2's reset, not this script's command.

This is a report tool, so a section that cannot run says why and the next section runs anyway. The
exit status is 0 only when every section passed; an unavailable section (no c3po, no device, no
CUDA) is not a pass. A missing station is the normal case on a machine that is not chopper, and the
report is still useful there: it prints the config and the checkpoint's numbers without hardware.

Each section prints when it has something to say; the SpaceMouse section prints its instruction
before opening the read window, which is why the lines are streamed rather than collected.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import bridge
import numpy as np
import stats
from bridge import ARM_DIMS
from config import Config, load_config
from numpy.typing import NDArray
from station import StationManifest, observation_batch

EXAMPLE_DIR = Path(__file__).parent
"""The shipped config and the relative asset paths beside this file."""

DEFAULT_CONFIG = EXAMPLE_DIR / "configs" / "franka.toml"

SECTIONS = ("configuration", "station", "spacemouse", "policy")
"""The report's four sections, in the order they print."""

HOLD_SECONDS = 1.0
"""How long the read-only ``step(None)`` window runs: about 15 control steps at 15 Hz."""

HOLD_TOLERANCE_RAD = 0.1
"""A station that holds the arm through a command-free second moves less than this. The value is a
session fact, not a checkout constant: the measured plant's uncommanded drift is 0.0001-0.005 rad per
step (flow-steering/docs/plant.md), so a held arm can accumulate a few hundredths of a radian in a
second and still be holding."""

SPACEMOUSE_SECONDS = 5.0
"""How long the SpaceMouse section reads raw reports. Five seconds is two pushes per axis."""

REST_SETTLE_SECONDS = 0.5
"""After the push window the operator releases the stick; this window captures the zeroing reports."""

REST_TOLERANCE = 0.1
"""The rest-state six-vector is in deflection units ([-1, 1]): an axis above this at rest is not
re-zeroing, or the stick was not released."""

TRANSLATION_REPORT = 1
ROTATION_REPORT = 2
BUTTON_REPORT = 3
"""``spacemouse.teleop_spacemouse``'s report ids. The live section passes the module's own constants
to the tally; these defaults are what the pure tests feed it."""

HID_EVENT = Path("/sys/class/hidraw")
"""``/sys/class/hidraw/hidrawN/device/uevent`` is where a node's HID id lives."""


# ---------------------------------------------------------------- the report


@dataclass
class Section:
    """One printed section: the lines it said and whether its checks passed.

    A section that cannot run sets ``unavailable_reason`` (an absent device, a missing package) rather
    than pretending to pass; the exit status treats that as a failure, because the probe exists to
    report on a session and a missing piece of the session is the report's most useful line.
    """

    title: str
    lines: list[str] = field(default_factory=list)
    failures: int = 0
    unavailable_reason: str | None = None

    def say(self, text: str) -> None:
        """A plain line: a printed fact, not a judgement."""
        self.lines.append(text)
        print(f"  {text}", flush=True)

    def info(self, label: str, detail: str = "") -> None:
        """A line that is neither pass nor fail: a fact the operator needs, without a verdict."""
        self.say(f"[INFO] {label}: {detail}" if detail else f"[INFO] {label}")

    def check(self, passed: bool, label: str, detail: str = "") -> bool:
        """Print one PASS/FAIL check; return what it found so a caller can branch on the same fact."""
        status = "PASS" if passed else "FAIL"
        if not passed:
            self.failures += 1
        self.say(f"[{status}] {label}: {detail}" if detail else f"[{status}] {label}")
        return passed

    def unavailable(self, reason: str) -> None:
        """This section cannot run here; say why once and let the next section run."""
        self.unavailable_reason = reason
        self.say(f"[UNAVAILABLE] {reason}")

    @property
    def ok(self) -> bool:
        return self.unavailable_reason is None and self.failures == 0

    @property
    def status(self) -> str:
        if self.unavailable_reason is not None:
            return f"UNAVAILABLE: {self.unavailable_reason}"
        if self.failures:
            return f"FAIL ({self.failures} checks failed)"
        return "PASS"


# ---------------------------------------------------------------- the raw-report tally


@dataclass
class ReportTally:
    """What the SpaceMouse's raw hidraw reports said: ids per node and the peak |counts| per axis.

    A 3Dconnexion device exposes several hidraw interfaces and only one carries the reports, so the
    tally keys everything by node: "which report ids arrive on which node" is how a session learns
    whether id 2 (rotation) arrives at all and where. The peak counts are the operator's full
    deflection in raw counts, which is what ``[spacemouse] full_scale`` and
    ``rotation_full_scale`` are expressed in.

    The ids default to the module's and the live section passes the module's own constants, so a
    firmware that renumbered its reports would be read by the numbers the reader uses, not by this
    class's memory of them.
    """

    translation_id: int = TRANSLATION_REPORT
    rotation_id: int = ROTATION_REPORT
    report_ids: dict[str, dict[int, int]] = field(default_factory=dict)
    """node -> report id -> the number of reports of that id seen on that node."""
    peaks: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    """report id -> the largest |count| seen on each axis, so a sign never cancels a peak."""
    last_counts: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    """report id -> the last counts seen, which is the rest state once the stick is released."""
    reports: int = 0
    """Every report seen, buttons included; 0 means no node sent anything."""

    def add(self, node: str, data: bytes) -> int | None:
        """Record one raw report ``data`` read from hidraw ``node``; return its id, or None for no data.

        The first byte is the report id. The motion reports (the translation and rotation ids) carry
        three little-endian int16 counts after it; a short report still counts toward its id, because
        "which ids arrive" is itself the finding, but it cannot move a peak.
        """
        if not data:
            return None
        report = int(data[0])
        counts = self.report_ids.setdefault(node, {})
        counts[report] = counts.get(report, 0) + 1
        self.reports += 1
        if report in (self.translation_id, self.rotation_id) and len(data) >= 7:
            values = struct.unpack_from("<hhh", data, 1)
            self.last_counts[report] = values
            previous = self.peaks.get(report, (0, 0, 0))
            self.peaks[report] = tuple(max(abs(now), was) for now, was in zip(values, previous, strict=True))
        return report

    def ids(self, node: str) -> dict[int, int]:
        """The report ids seen on ``node`` and their counts (empty when the node was silent)."""
        return dict(self.report_ids.get(node, {}))

    def ids_by_report(self, report: int) -> dict[str, int]:
        """The nodes that sent ``report`` and how many times each (empty when none did)."""
        return {node: counts[report] for node, counts in self.report_ids.items() if report in counts}

    def suggestion(self) -> tuple[float | None, float | None]:
        """The full scales the peaks imply: one count for translation, one for rotation.

        A single config scale covers all three axes of a report, so take the largest axis: the axis
        that travelled furthest must read at most 1.0. ``None`` for a report id that never arrived;
        that axis cannot be scaled, and is the finding that matters.
        """
        translation = self.peaks.get(self.translation_id)
        rotation = self.peaks.get(self.rotation_id)
        return (
            None if translation is None else float(max(translation)),
            None if rotation is None else float(max(rotation)),
        )

    def rest_vector(
        self,
        source: Sequence[int],
        sign: Sequence[float],
        full_scale: float,
        rotation_full_scale: float,
    ) -> NDArray:
        """The six mapped channels at rest: device counts divided by the scales, then mapped.

        This is the value ``SpaceMouseTwist.twist`` reports when nothing is touched: the last
        translation and rotation counts, mapped through ``source``/``sign`` and divided by the
        configured scales. A released device sends a zeroing report, so it should be the zero vector;
        a non-zero axis is the stick held, a stale report, or a scale so wrong it cannot return to 0.
        """
        translation = self.last_counts.get(self.translation_id, (0, 0, 0))
        rotation = self.last_counts.get(self.rotation_id, (0, 0, 0))
        device = np.asarray([*translation, *rotation], dtype=np.float64)[list(source)] * np.asarray(
            list(sign)
        )
        scale = np.asarray([full_scale] * 3 + [rotation_full_scale] * 3, dtype=np.float64)
        return device / scale

    def summary(self) -> list[str]:
        """The report's lines: which ids arrived where, then the peak counts per axis."""
        lines = []
        if self.report_ids:
            for node in sorted(self.report_ids):
                seen = ", ".join(f"id {report} x {count}" for report, count in sorted(self.ids(node).items()))
                lines.append(f"{node}: {seen}")
        else:
            lines.append("no node sent a report")
        for report, name in ((self.translation_id, "translation"), (self.rotation_id, "rotation")):
            peak = self.peaks.get(report)
            if peak is None:
                lines.append(f"peak |counts| id {report} ({name}): no reports")
            else:
                axes = ", ".join(f"{axis} {value}" for axis, value in zip(("x", "y", "z"), peak, strict=True))
                lines.append(f"peak |counts| id {report} ({name}): {axes}")
        return lines


def format_manifest(
    manifest: StationManifest, joint_names: Mapping[str, Sequence[str]] | None = None
) -> list[str]:
    """The station's advertised facts as report lines; ``joint_names`` is c3po's optional extra."""
    lines = [
        f"arms: {', '.join(manifest.arms) or 'none'}",
        f"cameras: {', '.join(manifest.cameras) or 'none'}",
        f"action keys: {', '.join(manifest.action_keys) or 'none'}",
        f"control rate: {float(manifest.control_rate):g} Hz (the station's native rate)",
    ]
    for arm, names in (joint_names or {}).items():
        lines.append(f"joint names for {arm}: {', '.join(names) or 'none advertised'}")
    return lines


@dataclass(frozen=True)
class Check:
    """One action-contract check: what the loop sends against what the station advertises."""

    label: str
    status: str
    detail: str

    def line(self) -> str:
        return (
            f"[{self.status}] {self.label}: {self.detail}" if self.detail else f"[{self.status}] {self.label}"
        )


def action_contract(arm: str, action_keys: Sequence[str], command_mode: str | None = None) -> list[Check]:
    """The two action keys the loop writes, against the manifest's ``action_keys``.

    ``station.action_dict`` sends exactly ``{arm}/joint_velocity`` and ``{arm}/gripper_position``; a
    station that advertises anything else refuses every step (c3po raises ``ValueError`` naming the
    missing keys). The arm's own ``command_mode`` from c3po's raw manifest is reported too: it is
    what decides the action key, so a mode that is not ``joint_velocity`` would send velocities to a
    position controller.
    """
    velocity = f"{arm}/joint_velocity"
    gripper = f"{arm}/gripper_position"
    keys = list(action_keys)
    checks = [
        Check(
            velocity,
            "PASS" if velocity in keys else "FAIL",
            "the loop's seven joint velocities"
            if velocity in keys
            else f"not in the manifest's action_keys {keys}; every step() would raise",
        ),
        Check(
            gripper,
            "PASS" if gripper in keys else "FAIL",
            "the loop's binarized gripper bit"
            if gripper in keys
            else f"not in the manifest's action_keys {keys}; every step() would raise",
        ),
    ]
    if command_mode == "joint_velocity":
        checks.append(Check("command mode", "PASS", "the arm advertises joint_velocity"))
    elif command_mode is None:
        checks.append(
            Check(
                "command mode",
                "INFO",
                "not advertised in the arm's manifest table; the action key above is the contract",
            )
        )
    else:
        checks.append(
            Check(
                "command mode",
                "FAIL",
                f"the arm advertises {command_mode!r}, not 'joint_velocity': the loop's velocities "
                "would be interpreted as that command",
            )
        )
    return checks


# ---------------------------------------------------------------- the configuration section


def configuration_section(
    config: Config,
    *,
    path: str | Path = DEFAULT_CONFIG,
    command: str = "probe",
    reset_observation: Mapping[str, NDArray] | None = None,
    session: _StationSession | None = None,
) -> Section:
    """The config, the knobs, and the corruption the loop would actually apply.

    ``q0`` is ``[command] q0`` when the file sets it, else the station's reset pose; without either
    the lift cannot be evaluated, and the ``q0``/``F7``/gain lines say so rather than printing an
    identity that is not the session's corruption. The numbers are ``config.py``'s,
    ``bridge.corruption_lift``'s and ``stats.action_std``'s; nothing here re-derives them.
    """
    section = Section("configuration")
    section.say(f"config: {path} (parsed; an unknown key is refused by config.py)")
    section.say(f"command: {command}")
    section.say(
        f"station: {config.station.arm} at {config.station.host}:{config.station.port}, "
        f"rate {config.station.rate:g} Hz"
    )
    section.say(
        f"policy: {config.policy.path} on {config.policy.device} "
        f"(n_action_steps {config.policy.n_action_steps}, dtype {config.policy.dtype})"
    )
    section.say(f"cameras: {_cameras_text(config)}")
    section.say(
        f"command knobs: translation_gain {config.command.translation_gain:g} m/s, "
        f"rotation_gain {config.command.rotation_gain:g} rad/s, "
        f"twist_length {config.command.twist_length:g} m, "
        f"deadband {config.command.deadband:g} action units, "
        f"gain {config.command.gain if isinstance(config.command.gain, str) else f'{config.command.gain:g}'}"
    )
    section.say(
        f"readout: corruption {_corruption_text(config.readout.corruption)}, "
        f"noise {config.readout.noise:g} sigma, seed {config.readout.seed}"
    )
    section.say(
        f"steering: n_reversal_steps {config.steering.n_reversal_steps}, "
        f"pinned_steps {config.steering.pinned_steps}"
    )
    section.say("modes:")
    for name in sorted(config.modes):
        section.say(f"  {name}: {_mode_text(config.modes[name])}")
    absent = [name for name in ("mars-u", "mars-t", "mars") if name not in config.modes]
    if absent:
        section.say(
            f"  absent: {', '.join(absent)} (the Phase B/C adapters do not exist; the CLI names the "
            "missing [modes.*] table instead of loading nothing)"
        )
    section.say(
        "readout in force: plain teleop is clean (corruption off, no noise); teleop --corrupted and "
        "every shared mode use [readout]"
    )

    q0 = _resolve_q0(config, reset_observation=reset_observation, session=session, section=section)
    _corruption_numbers(section, config, q0)
    return section


def _resolve_q0(
    config: Config,
    *,
    reset_observation: Mapping[str, NDArray] | None,
    session: _StationSession | None,
    section: Section,
) -> NDArray | None:
    """The lift's nominal pose: ``[command] q0``, else the station's reset pose, else nothing."""
    configured = np.asarray(config.command.q0, dtype=np.float64)
    if configured.shape == (ARM_DIMS,):
        section.say(f"q0: {_vector_text(configured)} (from [command] q0)")
        return configured
    if reset_observation is None and session is not None:
        try:
            reset_observation = session.reset_observation()
        except Exception as error:  # a report tool: the missing station is the section's next line
            section.info("q0", f"the station's reset pose is unavailable ({_error_text(error)})")
            section.info(
                "q0 needed", "set [command] q0 to the seven joint angles to resolve F7 without the station"
            )
            return None
    if reset_observation is None:
        section.info("q0", "neither [command] q0 nor a station reset pose is available")
        section.info(
            "q0 needed", "set [command] q0 to the seven joint angles to resolve F7 without the station"
        )
        return None
    try:
        q0 = bridge.nominal_pose(reset_observation, config.station.arm)
    except ValueError as error:
        section.info("q0", f"the reset observation has no {config.station.arm}/joint_position ({error})")
        return None
    section.say(f"q0: {_vector_text(q0)} (the station's reset pose)")
    return q0


def _corruption_numbers(section: Section, config: Config, q0: NDArray | None) -> None:
    """The lift, the gain and the noise sigma: the readout the loop would apply in a shared mode."""
    spec = config.readout.corruption
    matrix7 = None
    if spec is None:
        section.say("corruption: off (F7 is the identity and the auto gain is 1.0)")
    else:
        rng = np.random.default_rng([int(config.readout.seed), 0]) if _is_random(spec) else None
        matrix6 = bridge.matrix6_from_spec(spec, rng)
        if q0 is None:
            section.info(
                "F7",
                "not resolved: the lift needs q0, and the identity is not this session's corruption",
            )
        else:
            matrix7 = bridge.corruption_lift(matrix6, q0, float(config.command.twist_length))
            norm = float(np.linalg.norm(matrix7, 2))
            section.say(f"F7 (7x7, ||F7||2 = {norm:.4f}):")
            for row in matrix7:
                section.say("  " + _vector_text(row))
            if _is_random(spec):
                section.say("  one draw for episode 0: a {random} family is redrawn at every episode")
        gain = _resolved_gain(config.command.gain, spec, matrix7)
        if gain is None:
            section.info("gain", "auto = 1/||F7||2 prints once F7 is resolved")
        elif isinstance(config.command.gain, str):
            section.say(f"gain: {gain:.4f} (auto = 1/||F7||2, corruption on)")
        else:
            section.say(f"gain: {gain:.4f} (set explicitly in [command])")
    _noise_sigma(section, config)


def _resolved_gain(gain: str | float, spec: dict | None, matrix7: NDArray | None) -> float | None:
    """The loop's gain rule: the configured number, or ``auto`` = 1/||F7||2 when corruption is on."""
    if gain != "auto":
        return float(gain)
    if spec is None:
        return 1.0
    if matrix7 is None:
        return None
    return 1.0 / float(np.linalg.norm(matrix7, 2))


def _noise_sigma(section: Section, config: Config) -> None:
    """``readout.noise`` in action-standard-deviation units, from the statistics file."""
    noise = float(config.readout.noise)
    if not noise:
        section.say("noise sigma: off (readout.noise 0.0)")
        return
    try:
        std = stats.action_std(config.policy.stats)
    except (OSError, ValueError) as error:
        section.info(
            "noise sigma",
            f"{noise:g} sigma cannot be scaled: {_error_text(error)}; the loop refuses noise without the "
            "statistics file (loop._device_noise)",
        )
        return
    section.say(f"noise sigma (7, action units): {_vector_text(noise * std)} ({noise:g} x the action std)")


# ---------------------------------------------------------------- the station section


def station_section(
    config: Config,
    session: _StationSession,
    *,
    reset_observation: Mapping[str, NDArray] | None = None,
    reset_error: Exception | None = None,
) -> Section:
    """The station: manifest, rates, one reset, the feeds, a held second, the action contract.

    ``station.C3poStation`` is the loop's port, so the manifest is the same object the loop sees. Its
    wrapped ``c3po.Robot`` supplies the three facts the port does not expose — ``joint_names``, the
    arm table's ``command_mode``/``gripper`` entry, and c3po's read-only ``step(None)``.
    """
    section = Section("station")
    try:
        station = session.station()
    except Exception as error:  # no c3po, no station, a refused handshake: say which and move on
        section.unavailable(_error_text(error))
        return section

    raw = _raw_manifest(station)
    manifest = station.manifest
    for line in format_manifest(manifest, _joint_names(station)):
        section.say(line)
    if raw:
        section.say(
            f"station model: {raw.get('station_model', 'unknown')} (c3po API {raw.get('api_version', '?')})"
        )

    arm = _single_arm(section, config, manifest)
    _rate_check(section, config, manifest)
    observation, reset_error = _observation_for(session, reset_observation, reset_error)
    if observation is None:
        section.check(False, "reset()", _error_text(reset_error))
    else:
        _observation_report(section, observation, arm)
    _camera_report(section, config, observation)
    _hold_check(section, station, observation, arm)
    _contract_report(section, raw, arm, manifest)
    return section


def _single_arm(section: Section, config: Config, manifest: StationManifest) -> str:
    """The one arm the loop would drive; a station with none or two is not what this example does."""
    arms = manifest.arms
    if len(arms) == 1:
        section.check(True, "single arm", f"{arms[0]!r} (the loop drives exactly one)")
        if config.station.arm != arms[0]:
            section.check(
                False,
                "configured arm",
                f"[station] arm is {config.station.arm!r} but the station advertises {arms[0]!r}",
            )
        return arms[0]
    section.check(
        False, "single arm", f"the station advertises {len(arms)} arms {list(arms)}; the loop needs one"
    )
    return arms[0] if arms else config.station.arm


def _rate_check(section: Section, config: Config, manifest: StationManifest) -> None:
    """The loop asks for ``[station] rate``; r2d2 may cap it at the native ``control_rate``."""
    requested = float(config.station.rate)
    native = float(manifest.control_rate)
    section.say(
        f"control rate: the loop asks for {requested:g} Hz; the station's native rate is {native:g} Hz"
    )
    if native >= requested:
        section.check(
            True,
            "rate",
            f"native {native:g} Hz >= requested {requested:g} Hz: the loop runs at the checkpoint's rate",
        )
    else:
        section.check(
            False,
            "rate",
            f"native {native:g} Hz < requested {requested:g} Hz: r2d2 caps the loop below the trained "
            "action rate, so one action covers more wall time than the checkpoint assumes",
        )


def _observation_for(
    session: _StationSession,
    observation: Mapping[str, NDArray] | None,
    error: Exception | None,
) -> tuple[Mapping[str, NDArray] | None, Exception | None]:
    """The one reset observation, from the caller or from the session's cached reset."""
    if observation is not None or error is not None:
        return observation, error
    try:
        return session.reset_observation(), None
    except Exception as caught:  # the reset is the station's; a report tool prints why it failed
        return None, caught


def _observation_report(section: Section, observation: Mapping[str, NDArray], arm: str) -> None:
    """One reset's keys, shapes and dtypes, and the reset pose the lift would use."""
    section.check(True, "reset()", f"{len(observation)} keys")
    for key in sorted(observation):
        value = np.asarray(observation[key])
        section.say(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    try:
        pose = bridge.nominal_pose(observation, arm)
    except ValueError as error:
        section.check(False, "reset pose", str(error))
        return
    section.say(f"reset pose q ({arm}/joint_position): {_vector_text(pose)}")


def _camera_report(
    section: Section,
    config: Config,
    observation: Mapping[str, NDArray] | None,
) -> None:
    """Each configured feed's frame: shape, dtype and mean brightness (0-255)."""
    for role in ("exterior", "wrist", "third"):
        feed = getattr(config.cameras, role)
        if feed is None:
            section.say(
                "camera third: omitted (the model fills the third image with a -1 image and a zero mask)"
            )
            continue
        if observation is None:
            continue
        if feed not in observation:
            section.check(False, f"camera {role}", f"{feed!r} is not in the observation")
            continue
        frame = np.asarray(observation[feed])
        valid = frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        brightness = float(frame.mean()) if frame.size else 0.0
        detail = f"{feed}: shape {frame.shape}, dtype {frame.dtype}, mean brightness {brightness:.1f}/255"
        if not valid:
            detail += " (the batch builder needs an (H, W, 3) uint8 frame)"
        elif brightness < 5.0:
            detail += " (dark: check the lens, the light and the exposure)"
        elif brightness > 250.0:
            detail += " (blown out: check the exposure)"
        section.check(valid, f"camera {role}", detail)


def _hold_check(
    section: Section,
    station: object,
    observation: Mapping[str, NDArray] | None,
    arm: str,
) -> None:
    """One read-only ``step(None)`` window: does the station hold the arm when nothing is commanded?"""
    robot = getattr(station, "_robot", None)
    if robot is None or not callable(getattr(robot, "step", None)):
        section.check(False, "read-only step(None)", "the station object does not wrap c3po's Robot")
        return
    if observation is None:
        section.info("read-only step(None)", "skipped: there is no reset observation to compare against")
        return
    try:
        start_pose = bridge.nominal_pose(observation, arm)
        start = time.monotonic()
        steps = 0
        pose = start_pose
        elapsed = 0.0
        while elapsed < HOLD_SECONDS:
            pose = bridge.nominal_pose(robot.step(None), arm)
            steps += 1
            elapsed = time.monotonic() - start
    except Exception as error:  # c3po's timeout or a malformed observation: report, do not crash
        section.check(False, "read-only step(None)", _error_text(error))
        return
    delta = np.abs(pose - start_pose)
    section.say(f"read-only step(None): {steps} steps in {elapsed:.2f} s, no action sent")
    per_joint = " ".join(f"{value:.4f}" for value in delta)
    section.check(
        float(delta.max()) <= HOLD_TOLERANCE_RAD,
        "holds without a command",
        f"max |dq| {float(delta.max()):.4f} rad over {elapsed:.2f} s "
        f"(tolerance {HOLD_TOLERANCE_RAD:g} rad) per joint [{per_joint}]",
    )


def _contract_report(
    section: Section, raw: Mapping[str, object], arm: str, manifest: StationManifest
) -> None:
    """The action contract: the two keys the loop sends, and the arm table's command mode."""
    entry = _arm_entry(raw, arm)
    command_mode = entry.get("command_mode") if isinstance(entry, dict) else None
    gripper = entry.get("gripper") if isinstance(entry, dict) else None
    gripper_key = gripper.get("key") if isinstance(gripper, dict) else None
    section.say(f"action contract ({arm}):")
    for check in action_contract(arm, manifest.action_keys, command_mode):
        section.say("  " + check.line())
    if gripper_key is not None:
        section.say(f"  the manifest's gripper key: {gripper_key!r}")


# ---------------------------------------------------------------- the SpaceMouse section


def spacemouse_section(config: Config, *, seconds: float = SPACEMOUSE_SECONDS) -> Section:
    """Every 3Dconnexion hidraw node, the report ids and counts it sends, and the mapped rest state.

    The vendor test and the device lookup are ``spacemouse.teleop_spacemouse``'s: a device's
    ``HID_ID`` is matched there (vendor ``256F``, or ``046D`` with a ``C6xx`` product), so this section
    imports the module and does not re-spell the rule. The import is inside the section and after the
    node check, so a machine with no ``/dev/hidraw`` never loads lerobot's torch path.
    """
    section = Section("spacemouse")
    nodes = hidraw_nodes()
    if not nodes:
        section.unavailable(
            "no /dev/hidraw* node: the device is read straight from Linux hidraw, so there is nothing "
            "to enumerate here (on chopper, check that the SpaceMouse is plugged in)"
        )
        return section
    try:
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import (
            REPORT_BUTTONS,
            REPORT_ROTATION,
            REPORT_TRANSLATION,
            _is_3dconnexion,
            find_spacemouse,
        )
    except Exception as error:  # lerobot's module is torch-heavy; a missing import is a fact
        section.unavailable(f"lerobot's spacemouse module is not importable ({_error_text(error)})")
        return section

    spacemice = []
    for node in nodes:
        uevent = _uevent(node)
        kind = "3Dconnexion" if _is_3dconnexion(uevent) else "other"
        section.say(f"hidraw {node}: HID_ID {_hid_id(uevent)} ({kind})")
        if kind == "3Dconnexion":
            spacemice.append(node)
    if not spacemice:
        section.unavailable(
            f"no 3Dconnexion node among the {len(nodes)} hidraw nodes (vendor 256F, or 046D with a "
            "C6xx product); the udev rule only affects reading a node, not finding it"
        )
        return section

    configured = config.spacemouse.path or None
    if configured is None:
        try:
            configured = find_spacemouse(config.spacemouse.hid_id or None)
        except (FileNotFoundError, OSError) as error:
            section.check(False, "configured node", _error_text(error))
    if configured is not None:
        section.check(
            configured in spacemice,
            "configured node",
            f"{configured!r} is {'' if configured in spacemice else 'not '}one of the readable "
            f"3Dconnexion nodes {spacemice}",
        )

    opened, unreadable = _open_nodes(spacemice)
    for node, error in unreadable:
        section.check(
            False,
            f"readable {node}",
            f"{_error_text(error)}; a /dev/hidraw node needs the udev rule from the README",
        )
    if not opened:
        section.unavailable("no 3Dconnexion node could be opened for reading")
        return section
    section.check(True, "readable nodes", f"{len(opened)} of {len(spacemice)} 3Dconnexion nodes")
    ids = (REPORT_TRANSLATION, REPORT_ROTATION, REPORT_BUTTONS)
    _read_mouse(section, opened, config, seconds, ids)
    return section


def _read_mouse(
    section: Section,
    opened: Mapping[int, str],
    config: Config,
    seconds: float,
    ids: tuple[int, int, int],
) -> None:
    """The live read window: instruct the operator, tally every node, report the measured facts."""
    section.say(
        "push each translation axis to its stop in turn: +x (right), +y (toward you), +z (down); "
        "then twist each rotation axis to its stop in turn: rx, ry, rz"
    )
    section.say(f"reading {len(opened)} node(s) for {seconds:g} s; the operator moves the stick now")
    tally = ReportTally(translation_id=ids[0], rotation_id=ids[1])
    try:
        _tally_reports(opened, tally, seconds)
        section.say("release the stick and hold it still; reading the rest state")
        _tally_reports(opened, tally, REST_SETTLE_SECONDS)
    finally:
        for fd in opened:
            with contextlib.suppress(OSError):
                os.close(fd)
    for line in tally.summary():
        section.say(line)
    _mouse_checks(section, tally, config, ids)


def _tally_reports(opened: Mapping[int, str], tally: ReportTally, seconds: float) -> None:
    """Read every open node with ``select`` until the window closes, feeding the tally."""
    fds = list(opened)
    deadline = time.monotonic() + seconds
    while fds and time.monotonic() < deadline:
        ready, _, _ = select.select(fds, [], [], 0.25)
        for fd in ready:
            try:
                data = os.read(fd, 64)
            except (BlockingIOError, OSError):  # unplugged or closed: drop the node, keep the rest
                fds.remove(fd)
                continue
            if not data:  # a hidraw read of b"" means the device went away, not a report
                fds.remove(fd)
                continue
            tally.add(opened[fd], data)


def _mouse_checks(section: Section, tally: ReportTally, config: Config, ids: tuple[int, int, int]) -> None:
    """What the tally means: the scales to copy, the id-2 finding, the mapping, the rest state."""
    translation = tally.ids_by_report(ids[0])
    rotation = tally.ids_by_report(ids[1])
    section.check(
        bool(translation),
        "id 1 (translation) reports",
        f"{sum(translation.values())} reports on {', '.join(sorted(translation))}"
        if translation
        else "none arrived: the device is not sending translations on a readable node",
    )
    section.check(
        bool(rotation),
        "id 2 (rotation) reports",
        f"{sum(rotation.values())} reports on {', '.join(sorted(rotation))}"
        if rotation
        else "none arrived: either the operator did not twist, or this device/firmware does not send "
        "rotation on hidraw; the rotation axes cannot be used until it does",
    )
    buttons = tally.ids_by_report(ids[2])
    section.info("id 3 (button) reports", f"{sum(buttons.values())}" if buttons else "none arrived")

    full_scale, rotation_full_scale = tally.suggestion()
    if full_scale is not None:
        section.say(
            f"copy the measured translation peak into [spacemouse] full_scale = {full_scale:g} "
            "(raw counts at full deflection; a peak short of the stop is a lower bound)"
        )
    if rotation_full_scale is not None:
        section.say(
            f"copy the measured rotation peak into [spacemouse] rotation_full_scale = {rotation_full_scale:g}"
        )
    section.say(
        "the shipped [spacemouse] full_scale = 350.0 and rotation_full_scale = 350.0 are unverified "
        "defaults: copy the numbers above into configs/franka.toml after a full push on each axis"
    )
    source = list(config.spacemouse.source)
    sign = list(config.spacemouse.sign)
    section.say(f"configured source {source} and sign {[float(value) for value in sign]}")
    section.say("  device channels: 0 x, 1 y, 2 z, 3 rx, 4 ry, 5 rz (a positive sign keeps the channel)")
    section.say("  the probe cannot see a direction: confirm the source/sign map by jogging at a low gain")
    rest = tally.rest_vector(
        config.spacemouse.source,
        config.spacemouse.sign,
        float(config.spacemouse.full_scale),
        float(config.spacemouse.rotation_full_scale),
    )
    section.say(f"rest-state six-vector [x y z rx ry rz]: {_vector_text(rest)}")
    section.check(
        bool(np.max(np.abs(rest)) <= REST_TOLERANCE),
        "rest state",
        f"max |axis| {float(np.max(np.abs(rest))):.3f} in deflection units (tolerance "
        f"{REST_TOLERANCE:g}); a non-zero axis is a held stick or a device that is not re-zeroing",
    )


# ---------------------------------------------------------------- the policy section


def policy_section(config: Config, session: _StationSession | None = None) -> Section:
    """The checkpoint, the statistics file, and one timed ``prepare`` + ``act``.

    The chunk time is the number that decides whether a chunk pause survives the 15 Hz loop: Phase A
    has no prefetch and no compile, so the pre-processor and the denoising pass cost what they cost,
    and the report prints them rather than hiding them.
    """
    section = Section("policy")
    if not _cuda_available():
        section.unavailable(
            "needs CUDA: no nvidia-smi on PATH or torch.cuda.is_available() is False; the checkpoint "
            "is loaded on the GPU"
        )
        return section
    try:
        facts = _checkpoint_facts(config)
    except Exception as error:
        section.check(False, "checkpoint", _error_text(error))
        return section
    for line in facts:
        section.say(line)
    if not _statistics_block(section, config.policy.stats):
        section.info("load skipped", "the bundle cannot produce an action without the action statistics")
        return section

    try:
        from policy import load_bundle

        start = time.monotonic()
        bundle = load_bundle(config)
        load_seconds = time.monotonic() - start
    except Exception as error:
        section.check(False, "load_bundle", _error_text(error))
        return section
    section.say(f"load_bundle: {load_seconds:.1f} s (checkpoint and processors on {config.policy.device})")

    observation, synthetic = _policy_observation(config, session)
    if observation is None:
        section.check(False, "observation", "neither a station observation nor a synthetic one")
        return section
    batch = observation_batch(
        observation,
        arm=config.station.arm,
        cameras={
            "exterior": config.cameras.exterior,
            "wrist": config.cameras.wrist,
            "third": config.cameras.third,
        },
        task="probe",
    )
    try:
        start = time.monotonic()
        prepared = bundle.prepare(batch)
        prepare_ms = 1000.0 * (time.monotonic() - start)
        start = time.monotonic()
        action = bundle.act(prepared, None)
        act_ms = 1000.0 * (time.monotonic() - start)
    except Exception as error:
        section.check(False, "prepare + act", _error_text(error))
        return section
    source = "synthetic (the station section did not deliver one)" if synthetic else "the station's reset()"
    section.say(f"observation for the test chunk: {source}")
    section.say(f"prepare: {prepare_ms:.1f} ms; act: {act_ms:.1f} ms")
    _chunk_time(section, config, prepare_ms + act_ms)
    section.say(f"action (8, action units): {_vector_text(action)}")
    _image_keys_block(section, prepared)
    return section


def _chunk_time(section: Section, config: Config, total_ms: float) -> None:
    """The chunk's wall time against one control period: a pause that fits costs nothing."""
    period_ms = 1000.0 / float(config.station.rate)
    section.say(f"chunk time: {total_ms:.1f} ms for the first action of a chunk (prepare + act)")
    if total_ms <= period_ms:
        section.check(
            True,
            "chunk within one control period",
            f"{total_ms:.1f} ms <= {period_ms:.1f} ms at {config.station.rate:g} Hz: no pause",
        )
    else:
        section.info(
            "chunk pause",
            f"{total_ms:.1f} ms > {period_ms:.1f} ms at {config.station.rate:g} Hz: a "
            f"{total_ms - period_ms:.1f} ms pause every {config.policy.n_action_steps} actions "
            "(Phase A leaves it unhidden; this is the measurement)",
        )


def _image_keys_block(section: Section, prepared: Mapping[str, object]) -> None:
    """Which image keys reached the policy, and the third-camera convention the session ran."""
    keys = sorted(key for key in prepared if key.startswith("observation.images."))
    section.say(f"image keys in the prepared batch: {', '.join(keys) or 'none'}")
    section.check(
        "observation.images.base_0_rgb" in prepared and "observation.images.left_wrist_0_rgb" in prepared,
        "base and wrist images",
        "the two feeds the batch builder always writes",
    )
    if "observation.images.right_wrist_0_rgb" in prepared:
        section.say("third camera: present (cameras.third names a feed)")
    else:
        section.say(
            "third camera: omitted (the model fills it with a -1 image and a zero mask, not a zeros frame)"
        )


def _image_keys(checkpoint: object) -> list[str]:
    """The checkpoint's visual input keys: what the batch builder must provide."""
    features = getattr(checkpoint, "input_features", None) or {}
    return [
        key
        for key, feature in features.items()
        if "image" in key or "VISUAL" in str(getattr(feature, "type", "")).upper()
    ]


def _checkpoint_facts(config: Config) -> list[str]:
    """The checkpoint's own config, without loading its weights."""
    import lerobot.policies  # noqa: F401  registers the pi05 config class for the registry
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = PreTrainedConfig.from_pretrained(str(config.policy.path))
    return [
        f"checkpoint: {config.policy.path}",
        f"type: {getattr(checkpoint, 'type', '?')}",
        f"chunk_size: {getattr(checkpoint, 'chunk_size', '?')}, "
        f"n_action_steps: {getattr(checkpoint, 'n_action_steps', '?')} (the config overrides it to "
        f"{config.policy.n_action_steps}), "
        f"num_inference_steps: {getattr(checkpoint, 'num_inference_steps', '?')}",
        f"max_action_dim: {getattr(checkpoint, 'max_action_dim', '?')}, "
        f"max_state_dim: {getattr(checkpoint, 'max_state_dim', '?')}",
        f"dtype: {getattr(checkpoint, 'dtype', '?')} (the config overrides it to {config.policy.dtype})",
        f"use_relative_actions: {getattr(checkpoint, 'use_relative_actions', '?')}",
        f"image keys: {', '.join(_image_keys(checkpoint)) or 'none'}",
    ]


def _statistics_block(section: Section, stats_path: Path | str) -> bool:
    """The statistics file: present, 32 wide, meaningful in the 8 station dims, with its mean/std."""
    path = Path(stats_path)
    if not path.is_file():
        section.check(
            False,
            "statistics file",
            f"{path} is absent: download openpi's droid/norm_stats.json (assets/README.md); the shared "
            "modes cannot act without it and readout noise cannot be drawn",
        )
        return False
    try:
        groups = stats._groups(path)
        actions = stats._quantile_pair(groups, "actions", path, stats.STATION_DIMS)
        stats._quantile_pair(groups, "state", path, stats.STATION_DIMS)
    except (OSError, ValueError) as error:
        section.check(False, "statistics file", f"{path}: {_error_text(error)}")
        return False
    section.check(
        True,
        "statistics file",
        f"{path}: {stats.CHECKPOINT_WIDTH} wide, {stats.STATION_DIMS} meaningful dims in state and actions",
    )
    # The affine form of the QUANTILES map, the conversion FlowControlSteering.action_mean_std makes.
    mean = (actions["q01"] + actions["q99"]) / 2.0
    std = np.maximum((actions["q99"] - actions["q01"]) / 2.0, 1e-6)
    section.say(f"action mean (8): {_vector_text(mean)}")
    section.say(f"action std (8): {_vector_text(std)}")
    return True


def _policy_observation(
    config: Config, session: _StationSession | None
) -> tuple[dict[str, NDArray] | None, bool]:
    """The station's reset observation, or the same shapes synthesized when the station failed."""
    if session is not None:
        try:
            return dict(session.reset_observation()), False
        except Exception:  # the station section already said why; the policy can still be measured
            pass
    return _synthetic_observation(config), True


def _synthetic_observation(config: Config) -> dict[str, NDArray]:
    """The configured arm and feeds at zero: the same shapes the station would hand over.

    The policy's batch path is what the chunk timing measures, and it needs images at the batch
    builder's dtype and size; zeros are honest about there being no station.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    arm = config.station.arm
    observation = {
        f"{arm}/joint_position": np.zeros(ARM_DIMS),
        f"{arm}/gripper_position": np.zeros(1),
        config.cameras.exterior: frame.copy(),
        config.cameras.wrist: frame.copy(),
    }
    if config.cameras.third is not None:
        observation[config.cameras.third] = frame.copy()
    return observation


def _cuda_available() -> bool:
    """Whether a CUDA device is there, without importing torch when there cannot be one.

    ``nvidia-smi`` is a cheap first question: a machine without it cannot have a CUDA GPU, so the
    probe stays torch-free on a laptop (and in the offline tests).
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------- the station session


class _StationSession:
    """One lazy c3po station and one cached reset: the probe's only connection, closed by main.

    Every section that needs the station calls through this, so a failed connect is attempted once and
    the later section reports the same error instead of waiting on the handshake again. The cached
    reset observation is the single ``reset()``: the configuration section reads the pose from it, the
    station section reports it, and the policy section measures a batch on it.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._station: object | None = None
        self._observation: dict[str, NDArray] | None = None
        self._error: Exception | None = None
        self._closed = False

    def station(self) -> object:
        if self._station is not None:
            return self._station
        if self._error is not None:
            raise self._error
        try:
            self._station = _open_station(self._config)
        except Exception as error:  # remembered so the next section reports it, not reconnects
            self._error = error
            raise
        return self._station

    def reset_observation(self) -> dict[str, NDArray]:
        if self._observation is not None:
            return self._observation
        try:
            self._observation = dict(self.station().reset())
        except Exception as error:
            self._error = error
            raise
        return self._observation

    def close(self) -> None:
        if self._station is None or self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):  # closing a station that already failed is not a line
            self._station.close()


def _open_station(config: Config) -> object:
    """The station the loop would build; ``station.C3poStation`` imports c3po in its constructor."""
    from station import C3poStation

    return C3poStation(config.station.host, config.station.port, config.station.rate)


# ---------------------------------------------------------------- the pieces


def hidraw_nodes() -> list[str]:
    """Every ``/dev/hidraw*`` node, sorted; empty on a machine without Linux's hidraw devices."""
    return sorted(str(path) for path in Path("/dev").glob("hidraw*"))


def _uevent(node: str) -> str:
    """A hidraw node's ``/sys`` uevent text (``HID_ID=bus:vendor:product``); empty when unreadable."""
    try:
        return (HID_EVENT / Path(node).name / "device" / "uevent").read_text()
    except OSError:
        return ""


def _hid_id(uevent: str) -> str:
    """The ``HID_ID`` line of a uevent, or ``?`` when the node did not publish one."""
    for line in uevent.splitlines():
        if line.startswith("HID_ID="):
            return line[len("HID_ID=") :].strip()
    return "?"


def _open_nodes(nodes: Sequence[str]) -> tuple[dict[int, str], list[tuple[str, OSError]]]:
    """Open every node non-blocking: the readable ones, and the per-node reasons the rest were not."""
    opened: dict[int, str] = {}
    unreadable: list[tuple[str, OSError]] = []
    for node in nodes:
        try:
            opened[os.open(node, os.O_RDONLY | os.O_NONBLOCK)] = node
        except OSError as error:
            unreadable.append((node, error))
    return opened, unreadable


def _raw_manifest(station: object) -> dict:
    """c3po's parsed manifest table from the wrapped ``Robot`` (empty when the port is not c3po's)."""
    robot = getattr(station, "_robot", None)
    manifest = getattr(robot, "_manifest", None)
    return manifest if isinstance(manifest, dict) else {}


def _joint_names(station: object) -> dict:
    """c3po's ``Robot.joint_names`` (arm -> ordered joint names), when the station has one."""
    names = getattr(getattr(station, "_robot", None), "joint_names", None)
    return names if isinstance(names, dict) else {}


def _arm_entry(raw: Mapping[str, object], arm: str) -> dict:
    """The arm's entry in c3po's raw manifest: its ``command_mode`` and gripper declaration."""
    for entry in raw.get("arms", []) if isinstance(raw.get("arms"), list) else []:
        if isinstance(entry, dict) and entry.get("name") == arm:
            return entry
    return {}


def _is_random(spec: object) -> bool:
    """Whether the corruption is the per-episode ``{random: ...}`` family rather than one matrix."""
    return isinstance(spec, dict) and set(spec) == {"random"}


def _cameras_text(config: Config) -> str:
    """The three camera roles as the batch builder reads them."""
    third = (
        f"third {config.cameras.third!r}"
        if config.cameras.third is not None
        else "third omitted (the model fills it with a zero-mask image)"
    )
    return f"exterior {config.cameras.exterior!r}, wrist {config.cameras.wrist!r}, {third}"


def _corruption_text(spec: dict | None) -> str:
    """The readout's corruption as one phrase: the spec, the per-episode family, or the explicit off."""
    if spec is None:
        return "off (the clean readout)"
    if _is_random(spec):
        return f"a {spec!r} family, redrawn at every episode"
    return repr(spec)


def _mode_text(mode: object) -> str:
    """One ``[modes.*]`` entry as its mode and the knobs that mode reads."""
    parts = [str(getattr(mode, "mode", "none"))]
    if parts[0] == "flow_control":
        parts += [
            f"tau {getattr(mode, 'tau', '?')}",
            f"lam {getattr(mode, 'lam', '?')}",
            f"ic_gain {float(getattr(mode, 'ic_gain', 0.0)):g}",
        ]
    elif parts[0] == "frs":
        parts += [
            f"n_reversal_steps {getattr(mode, 'n_reversal_steps', '?')}",
            f"pinned_steps {getattr(mode, 'pinned_steps', '?')}",
        ]
    for key in ("adapter_matrix", "adapter_user", "adapter_task"):
        value = getattr(mode, key, None)
        if value is not None:
            parts.append(f"{key} {value}")
    return ", ".join(parts)


def _vector_text(values: Sequence[float]) -> str:
    """A vector as a short bracketed line with a sign on every entry."""
    return "[" + " ".join(f"{float(value):+.3f}" for value in values) + "]"


def _error_text(error: BaseException | None) -> str:
    """An exception as one report line: the type, then its message (never an empty string)."""
    if error is None:
        return "no error recorded"
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


# ---------------------------------------------------------------- the command


def main(argv: list[str] | None = None) -> int:
    """Print the four sections; 0 only when all four passed, 1 otherwise, 2 for an unreadable config.

    A section that raises is reported as a failed section and the rest still run: this is the tool a
    person runs to find out what is wrong, so it must not die on the first wrong thing it meets.
    """
    parser = argparse.ArgumentParser(
        prog="mars-franka probe",
        description="Print what the station, the SpaceMouse and the checkpoint are; move nothing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        metavar="PATH",
        help=f"the config file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    print(f"mars-franka probe: config {args.config}", flush=True)
    print(
        "this probe sends no action: one reset(), read-only steps, then close. Nothing here moves the arm.",
        flush=True,
    )
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        print(f"mars-franka probe: {args.config}: {error}", file=sys.stderr)
        return 2

    session = _StationSession(config)
    builders = (
        (
            SECTIONS[0],
            lambda: configuration_section(config, path=args.config, session=session),
        ),
        (SECTIONS[1], lambda: station_section(config, session)),
        (SECTIONS[2], lambda: spacemouse_section(config)),
        (SECTIONS[3], lambda: policy_section(config, session)),
    )
    passed = 0
    try:
        for index, (title, build) in enumerate(builders, 1):
            print(f"=== {index}/{len(builders)} {title} ===", flush=True)
            try:
                section = build()
            except Exception as error:  # one section's bug must not hide the other three
                section = Section(title)
                section.check(False, title, f"the section raised {_error_text(error)}")
            print(f"status: {section.status}", flush=True)
            if section.ok:
                passed += 1
    finally:
        session.close()
    print(f"probe: {passed}/{len(builders)} sections passed", flush=True)
    return 0 if passed == len(builders) else 1


if __name__ == "__main__":
    raise SystemExit(main())
