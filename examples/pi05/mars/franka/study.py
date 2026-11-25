"""The study block: rounds of trials over the config's arms, the outcome entry, and the run's records.

Phase D's protocol (DESIGN.md, "Phase D: running a study"). A block is ``[study] arms`` x ``[study]
trials`` rounds: round ``r`` runs every arm once, so ``pair = r`` is the pairing key. ``order =
"shuffled"`` draws each round's arm order from ``[study] seed``; ``order = "blocked"`` runs every trial
of one arm before the next arm. Between trials the operator resets the scene; after each trial the
operator answers the one question the study needs -- did the task succeed? -- on the terminal. A blank
answer is a failure, ``q`` stops the block, and EOF/Ctrl-C stop it too, cleanly, leaving the trials
already recorded as the record.

Every trial is one arm, composed by the CLI's own composition root (:func:`cli.compose_arm`), so an arm
is a fixed session command (``autonomous``, ``teleop``, ``fc``, ``frs``, ``mars-f``, ``mars-u``,
``mars-t``, ``mars``) or a ``[modes.<name>]`` table -- including the S2 "policy + task module, no
steering" anchor -- and a study needs no new composition. Each arm's bundle and steering policy are
built **once per block** and reused; a new :class:`loop.Loop` is built per trial, which is cheap and
gives the trial a fresh episode (the base policy's queues, the operator's latch, the readout's draw).

A trial is: ``station.reset()``, a digest of that reset observation (the pairing key with ``pair``),
``Loop.run(prompt=..., max_steps=[study] max_steps, on_tick=rows, initial_obs=observation)``, the
outcome question, and the records. The reset happens **once**: the loop starts the episode from that
same observation instead of resetting again, so the digest and the record's ``q0`` describe exactly
the scene the trial's first action was computed on -- which matters because a station's ``reset()``
is not known to be idempotent. The loop's hook is the per-tick record, called before
``station.step``: the executed action, the base policy's own chunk step (``Bundle.plan8``), the
served command and the joint positions the action came from.

**Authority** (DESIGN.md's formula, the number the depth knobs are supposed to buy): with ``a`` the
executed action, ``p`` the policy's plan and ``r`` the served command, all in action units,
``authority = clip(<a - p, r - p> / ||r - p||^2, 0, 1)`` on dims 0-6, over the steps where ``r`` is
above ``[command] deadband`` (``lerobot.teleoperators.readout.is_active``). A trial reports the mean
over its engaged steps, the fraction of steps engaged, and the mean command magnitude while engaged.
Steps with nothing to decide (``p = r``, no plan in teleop, no command in autonomous) are not measured;
a trial with no measurable step reports ``null``, not zero.

**Records.** ``<out>/<timestamp>_<config stem>/`` holds:

- ``trials.jsonl`` -- one line per completed trial: ``run, arm, pair, prompt, success, steps,
  duration_s, authority, engaged_fraction, mean_r, q0, scene_hash, config, config_sha256,
  timestamp``. Non-finite numbers are JSON ``null`` (a teleop trial's authority is undefined).
- ``trials/<arm>-<pair>.npz`` -- that trial's rows: ``executed`` (T, 8), ``plan`` (T, 8), ``served``
  (T, 7), ``q`` (T, 7). A mode without one of the three (teleop has no plan, autonomous no command)
  stores NaN in that array, so "there was nothing here" stays visible.
- ``config.toml`` -- the block's config file, byte for byte; ``config_sha256`` is its digest.
- ``run.json`` -- the block's provenance: the config path and digest, the git HEAD and whether the
  working tree was dirty, the timestamp, the arms, the round count, the order and the seed, and at the
  end how many trials were recorded and whether the block stopped early.

``out`` is the run-directory parent (``[study] out`` when None), so a block never overwrites another.
``config_path`` is the file the config was loaded from: the command hands it over so the copy and the
digest name the exact file the block ran. A caller that already holds a parsed ``Config`` and no path
(some tests) can omit it; the copy and the digest are then absent and the record says so.

The scene digest is honest, not a guarantee: it hashes the reset pose (rounded) and a 32x32 downsample
of the exterior camera's frame, so two trials pair only if their resets looked the same. A station whose
``reset()`` does not restore the scene produces different digests, and the analysis reports those trials
as unpaired rather than pairing on ``pair`` alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from contextlib import ExitStack, closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cli
import numpy as np
from bridge import ARM_DIMS
from config import Config, StudyConfig, load_config
from loop import Loop
from numpy.typing import NDArray
from policy import STATION_DIMS
from station import resize_with_pad

EXAMPLE_DIR = Path(__file__).parent
"""The example's directory: the default config and the ``git`` commands' working directory."""

DEFAULT_CONFIG = EXAMPLE_DIR / "configs" / "franka.toml"

DIGEST_SIZE = 32
"""The scene digest's frame side: a 32x32 uint8 downsample of the exterior camera."""

Q_DECIMALS = 4
"""The reset pose's rounding before hashing: 1e-4 rad is below the station's repeatability."""

SUCCESS_ANSWERS = ("y", "yes")
"""The answers that record a success; every other answer (blank included) is a failure."""

ABORT_ANSWERS = ("q", "quit")
"""The answers that stop the block cleanly: no more trials, the finished ones stay recorded."""


class _AbortedError(Exception):
    """The operator (or EOF) stopped the block at the outcome question: not an error."""


@dataclass(frozen=True)
class Metrics:
    """One trial's authority numbers, from its per-step rows."""

    authority: float
    engaged_fraction: float
    mean_r: float


class _Rows:
    """The study's per-tick record: the loop's ``on_tick`` arguments, appended in control-step order."""

    def __init__(self) -> None:
        self.executed: list[NDArray] = []
        self.plan: list[NDArray | None] = []
        self.served: list[NDArray | None] = []
        self.q: list[NDArray] = []

    def __call__(
        self, tick: int, action8: NDArray, plan8: NDArray | None, served7: NDArray | None, q7: NDArray
    ) -> None:
        self.executed.append(np.array(action8, dtype=np.float64, copy=True))
        self.plan.append(None if plan8 is None else np.array(plan8, dtype=np.float64, copy=True))
        self.served.append(None if served7 is None else np.array(served7, dtype=np.float64, copy=True))
        self.q.append(np.array(q7, dtype=np.float64, copy=True))

    def arrays(self) -> dict[str, NDArray]:
        """The npz arrays: ``executed`` (T, 8), ``plan`` (T, 8), ``served`` (T, 7), ``q`` (T, 7).

        A missing value -- teleop has no plan, autonomous no command -- is NaN in its row, so a
        re-analysis can tell "not measured" from a measured zero.
        """
        return {
            "executed": _stacked(self.executed, STATION_DIMS),
            "plan": _stacked(self.plan, STATION_DIMS),
            "served": _stacked(self.served, ARM_DIMS),
            "q": _stacked(self.q, ARM_DIMS),
        }


# ---------------------------------------------------------------- the numbers


def scene_digest(observation: dict[str, NDArray], camera: str) -> str:
    """The pairing key's other half: sha256 of the reset pose (rounded) and a 32x32 exterior frame.

    ``observation`` is the station's reset observation; ``camera`` is the exterior feed's name
    (``[cameras] exterior``). The pose is rounded to ``Q_DECIMALS`` so a repeated reset hashes the same,
    and the frame is letterboxed to ``DIGEST_SIZE`` through the same ``station.resize_with_pad`` the
    policy's images take. A scene the station did not restore produces a different digest, which is how
    the analysis finds trials that cannot be paired.
    """
    pose = np.round(_joint_positions(observation), Q_DECIMALS).astype(np.float64)
    if camera not in observation:
        raise ValueError(
            f"the observation has no exterior camera feed {camera!r} (keys: {sorted(observation)})"
        )
    picture = resize_with_pad(np.asarray(observation[camera]), DIGEST_SIZE)
    return hashlib.sha256(b"q" + pose.tobytes() + b"frame" + picture.tobytes()).hexdigest()


def metrics(executed: NDArray, plan: NDArray, served: NDArray, deadband: float) -> Metrics:
    """The trial's authority, engaged fraction and mean command magnitude, from its rows.

    ``executed`` is (T, 8), ``plan`` (T, 8) and ``served`` (T, 7), all in action units (NaN where a
    mode has no such value). The authority of a step is DESIGN.md's barycentric coordinate of the
    executed action between the policy's plan and the operator's command, on dims 0-6,
    ``clip(<a - p, r - p> / ||r - p||^2, 0, 1)``; it is measured only where the command is above
    ``deadband`` and the step's three values exist. A step where ``p = r`` has nothing to decide and is
    not measured. ``mean_r`` is the mean L2 norm of the served command over the engaged steps.
    """
    from lerobot.teleoperators.readout import is_active  # torch's module: lazy, so study imports stay light

    a = np.asarray(executed, dtype=np.float64)[:, :ARM_DIMS]
    p = np.asarray(plan, dtype=np.float64)[:, :ARM_DIMS]
    r = np.asarray(served, dtype=np.float64)[:, :ARM_DIMS]
    active = is_active(r, deadband)  # NaN is never active, so autonomous has no engaged step
    direction = r - p
    denominator = (direction * direction).sum(axis=1)
    moved = a - p
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.clip((moved * direction).sum(axis=1) / denominator, 0.0, 1.0)
    measured = active & np.isfinite(step)
    return Metrics(
        authority=float(step[measured].mean()) if measured.any() else math.nan,
        engaged_fraction=float(active.mean()) if active.size else math.nan,
        mean_r=float(np.linalg.norm(r, axis=1)[active].mean()) if active.any() else math.nan,
    )


def schedule(study: StudyConfig) -> list[tuple[int, str]]:
    """The block's trials in run order: ``(pair, arm)``.

    ``shuffled`` is each round's seeded permutation of the arms, from ``[study] seed``; ``blocked`` is
    every trial of one arm before the next. ``pair`` is the round either way, so the pairing key does
    not depend on the order the block ran in.
    """
    arms = list(study.arms)
    if study.order == "blocked":
        return [(pair, arm) for arm in arms for pair in range(study.trials)]
    rng = np.random.default_rng(int(study.seed))
    return [(pair, str(arm)) for pair in range(study.trials) for arm in rng.permutation(arms)]


# ---------------------------------------------------------------- the block


def run(
    config: Config,
    *,
    prompt: str,
    out: Path | None = None,
    config_path: Path | None = None,
) -> Path:
    """Run one study block; return the run directory the records landed in.

    The arms come from ``[study]``; each is composed once (the CLI's composition, so the readout and
    the mode's adapter strings are the session's) and the station and the source are opened once for
    the block. The operator resets the scene between trials and enters the outcome on the terminal;
    Ctrl-C, EOF or ``q`` stop the block, and the trials whose outcome was entered stay recorded.

    ``out`` is the run-directory parent (``[study] out`` when None). ``config_path`` is the file the
    config was loaded from: it is copied into the run directory and hashed into the records. The
    command always passes it; a caller that has only the parsed config omits it, and the records mark
    the copy and the digest as absent.
    """
    study = config.study
    arms = list(study.arms)
    if not arms:
        raise ValueError(
            "no study block: [study] arms names the arms to run (a fixed session command or a "
            "[modes.<name>] table)"
        )
    if len(set(arms)) != len(arms):
        duplicate = next(arm for arm in arms if arms.count(arm) > 1)
        raise ValueError(f"[study] arms: {duplicate!r} is named more than once")
    parent = Path(study.out) if out is None else Path(out)
    with ExitStack() as stack:
        station = stack.enter_context(closing(cli.open_station(config)))
        source = None
        if any(arm != "autonomous" for arm in arms):
            source = stack.enter_context(closing(cli.open_source(config)))
        composed = _compose(config, config_path, arms)
        run_dir = _create_run_dir(parent, config_path)
        digest = _copy_config(run_dir, config_path)
        provenance = _provenance(config, prompt, run_dir, config_path, digest)
        _write_json(run_dir / "run.json", provenance)
        print(
            f"study: {len(arms)} arm(s) x {study.trials} round(s), {study.order}; prompt {prompt!r}",
            flush=True,
        )
        print(f"study: run directory {run_dir}", flush=True)
        trial = _Trial(
            config=config,
            prompt=prompt,
            station=station,
            source=source,
            arms=composed,
            run_dir=run_dir,
            config_path=config_path,
            config_sha256=digest,
        )
        aborted = False
        try:
            _run_block(trial, study, run_dir)
        except (_AbortedError, KeyboardInterrupt):
            aborted = True
        completed = _count_trials(run_dir / "trials.jsonl")
        provenance.update(completed=completed, aborted=aborted)
        _write_json(run_dir / "run.json", provenance)
    print(f"study: {completed} trial(s) recorded in {run_dir}", flush=True)
    if aborted:
        print("study: the block stopped early; the completed trials are the record", file=sys.stderr)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    """``mars-franka study --config ... --prompt ...``: run the block, print the run directory.

    130 when the block stopped early (the records are complete for the trials it finished), else 0.
    """
    parser = argparse.ArgumentParser(
        prog="mars-franka study",
        description="Run the block [study] describes: rounds of trials, an outcome per trial, the records.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, metavar="PATH", help="the config file")
    parser.add_argument("--prompt", required=True, metavar="TEXT", help="the task text every trial runs")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_dir = run(config, prompt=args.prompt, config_path=Path(args.config))
    print(f"study: {run_dir}", flush=True)
    return _status(run_dir)


@dataclass
class _Trial:
    """One trial's fixed objects and identity: the block's station, source and arms, and its records."""

    config: Config
    prompt: str
    station: Any
    source: Any
    arms: dict[str, cli.Arm]
    run_dir: Path
    config_path: Path | None
    config_sha256: str | None

    def run(self, pair: int, arm: str) -> dict:
        """Run one trial and return its ``trials.jsonl`` record; raises ``_AbortedError`` at the outcome.

        The station resets here, once: its observation is the trial's start, digested for the pairing
        key, recorded as ``q0`` and handed to ``Loop.run`` as ``initial_obs``, so the loop does not
        reset again and the digest describes the scene the first action was computed from.
        """
        pieces = self.arms[arm]
        started = time.monotonic()
        observation = self.station.reset()
        digest = scene_digest(observation, self.config.cameras.exterior)
        pose = _joint_positions(observation)
        rows = _Rows()
        loop = Loop(
            config=pieces.loop_config,
            station=self.station,
            mode=pieces.loop_mode,
            source=None if pieces.loop_mode == "autonomous" else self.source,
            bundle=pieces.bundle,
            steering=pieces.steering,
            adapter_from_corruption=pieces.adapter_from_corruption,
        )
        loop.run(
            prompt=self.prompt,
            max_steps=self.config.study.max_steps,
            on_tick=rows,
            initial_obs=observation,
        )
        duration = time.monotonic() - started
        success = _ask_outcome(arm, pair)
        arrays = rows.arrays()
        measured = metrics(arrays["executed"], arrays["plan"], arrays["served"], self.config.command.deadband)
        np.savez(self.run_dir / "trials" / f"{arm}-{pair}.npz", **arrays)
        return {
            "run": self.run_dir.name,
            "arm": arm,
            "pair": pair,
            "prompt": self.prompt,
            "success": success,
            "steps": len(rows.executed),
            "duration_s": duration,
            "authority": _finite(measured.authority),
            "engaged_fraction": _finite(measured.engaged_fraction),
            "mean_r": _finite(measured.mean_r),
            "q0": [float(value) for value in pose],
            "scene_hash": digest,
            "config": None if self.config_path is None else str(self.config_path),
            "config_sha256": self.config_sha256,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }


# ---------------------------------------------------------------- the block's pieces


def _compose(config: Config, config_path: Path | None, arms: list[str]) -> dict[str, cli.Arm]:
    """Build each arm once, through the CLI's own composition root.

    A mode arm's table is resolved exactly as a session resolves it: the bundle (with the task
    checkpoint when the table names ``adapter_task``), the steering policy, and the
    ``adapter_from_corruption`` name the loop resolves for a ``mars-f`` table.
    """
    return {arm: cli.compose_arm(arm, config, config_path or DEFAULT_CONFIG) for arm in arms}


def _run_block(trial: _Trial, study: StudyConfig, run_dir: Path) -> None:
    """The block's trials in order, each written to ``trials.jsonl`` once its outcome is in."""
    order = schedule(study)
    total = len(order)
    with (run_dir / "trials.jsonl").open("w", encoding="utf-8") as lines:
        for index, (pair, arm) in enumerate(order, start=1):
            print(f"study: trial {index}/{total}: arm {arm}, pair {pair}", flush=True)
            lines.write(json.dumps(trial.run(pair, arm)) + "\n")
            lines.flush()


def _ask_outcome(arm: str, pair: int) -> bool:
    """The operator's one label: ``y``/``yes`` is a success, anything else (blank included) a failure.

    ``q``/``quit`` stops the block, and EOF (Ctrl-D) stops it as ``_AbortedError``; Ctrl-C propagates as
    ``KeyboardInterrupt`` to :func:`run`, which stops the block the same way.
    """
    question = f"study: {arm} pair {pair}: success? [y = yes, n/blank = no, q = stop] "
    try:
        answer = input(question).strip().lower()
    except EOFError as error:
        raise _AbortedError from error
    if answer in ABORT_ANSWERS:
        raise _AbortedError
    return answer in SUCCESS_ANSWERS


def _create_run_dir(parent: Path, config_path: Path | None) -> Path:
    """``<parent>/<timestamp>_<config stem>/``, with a numeric suffix if that name already exists.

    ``trials/`` is the per-tick record's directory, so it is made here: a trial writes its npz before
    its jsonl line, and the block's first line never lands without a place for the arrays.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = "config" if config_path is None else Path(config_path).stem
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / f"{stamp}_{stem}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{stamp}_{stem}-{suffix}"
        suffix += 1
    (candidate / "trials").mkdir(parents=True)
    return candidate


def _copy_config(run_dir: Path, config_path: Path | None) -> str | None:
    """Copy the block's config file into the run directory; return its SHA-256, or None without a path."""
    if config_path is None:
        return None
    data = Path(config_path).read_bytes()
    (run_dir / "config.toml").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _provenance(
    config: Config, prompt: str, run_dir: Path, config_path: Path | None, digest: str | None
) -> dict[str, Any]:
    """``run.json``: what the block is, beyond its per-trial records, and where it ran."""
    head, dirty = _git_state()
    return {
        "run": run_dir.name,
        "config_path": None if config_path is None else str(config_path),
        "config_sha256": digest,
        "git_head": head,
        "git_dirty": dirty,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "prompt": prompt,
        "arms": list(config.study.arms),
        "trials": int(config.study.trials),
        "order": config.study.order,
        "seed": int(config.study.seed),
        "max_steps": int(config.study.max_steps),
    }


def _git_state() -> tuple[str | None, bool | None]:
    """The checkout's HEAD and whether the working tree is dirty; ``(None, None)`` outside a repository."""

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=EXAMPLE_DIR,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return head, None if status is None else bool(status)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """One JSON object, indented; the run's ``run.json`` is rewritten as the block progresses."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _count_trials(path: Path) -> int:
    """How many trial lines the block has written: the records are flushed per trial, so this is the truth."""
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _status(run_dir: Path) -> int:
    """The command's exit status: 130 when the block stopped early, else 0."""
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return 130 if payload.get("aborted") else 0


# ---------------------------------------------------------------- arrays and poses


def _joint_positions(observation: dict[str, NDArray]) -> NDArray:
    """The single arm's reset pose, (7,), found by its ``/joint_position`` key."""
    keys = [key for key in observation if key.endswith("/joint_position")]
    if len(keys) != 1:
        raise ValueError(
            f"a station observation needs exactly one '<arm>/joint_position' key, got {sorted(keys)}"
        )
    pose = np.asarray(observation[keys[0]], dtype=np.float64).reshape(-1)
    if pose.shape != (ARM_DIMS,):
        raise ValueError(f"{keys[0]!r}: expected shape ({ARM_DIMS},), got {pose.shape}")
    return pose


def _stacked(rows: list[NDArray | None], width: int) -> NDArray:
    """One (T, width) float64 array; a missing row (None) is NaN, so it reads as "not measured"."""
    stacked = np.full((len(rows), width), math.nan)
    for index, row in enumerate(rows):
        if row is not None:
            stacked[index] = row
    return stacked


def _finite(value: float) -> float | None:
    """A JSON-safe number: None for NaN/inf, so a record is valid JSON rather than a Python NaN literal."""
    return float(value) if math.isfinite(value) else None
