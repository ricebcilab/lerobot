"""The one control loop: the station's clock, the operator's readout, the policy's action.

c3po owns the clock. One iteration sends one action with ``station.step(action8)``, which blocks
until the station's next observation; nothing here measures time (no ``time.sleep``, no timers, no
tick counters beyond the action index, no resent actions). The station is asked for the checkpoint's
own 15 Hz, so one shipped action per control step is one trained control step.

The chain is ``bridge.py``'s, in DESIGN.md's order: the operator's 6-axis deflection in [-1, 1]
scaled to a twist (``command.translation_gain`` m/s and ``command.rotation_gain`` rad/s at full
deflection) -> ``bridge.operator_command`` (IK of the twist at the current pose, in action units) ->
``bridge.readout`` through the corruption lift ``F`` and the noise -> the served 7-wide command ->
the method -> the 8-wide station action. In teleop the whole chain runs every tick; in the shared
modes it runs once per chunk (at ``tick % n_action_steps == 0``) and the *served* command is held for
the chunk: the steering core reads a batch's command only when its action queue is empty, so a
mid-chunk re-read would change nothing the policy sees and would only re-draw the readout's noise.

The station's raw batch is built once per chunk (``bundle.prepare``) and reused for the
``n_action_steps - 1`` mid-chunk ticks (``bundle.act``): the pre-processor (two images, the
tokenizer, the device transfer) is the expensive half of a tick, and the policies read a batch only
when their queue is empty. Teleop has no bundle and never builds a batch. The operator's device port
lives in ``source.py`` (``DeflectionSource`` and its two adapters); this module only consumes it.

The chunk boundary is also where a ``statefiles.Recorder`` gets its rows when one is passed to
``run``/``run_recording``: once per chunk in every mode, teleop included, the raw batch (the
pre-processor's input), the boundary's joint positions and gripper, the deflection in force and the
served command in force are handed over *before* the action is computed. Recording is observation
only -- it draws no noise and reads no device a second time -- so a run sends the same actions with
and without it. The loop never writes a file: saving the recorder is the caller's job.

Corruption is resolved lazily, at the first episode, because it needs ``q0``: the configured
``command.q0`` or else the reset pose. A fixed corruption is drawn once and printed in the run
header; a ``{random: ...}`` family is redrawn at every episode from
``numpy.random.default_rng([readout.seed, episode])``. ``adapter_from_corruption`` keeps a ``mars-f``
steering policy's ``adapter.matrix`` in sync with the episode's ``F`` (or its exact inverse), which a
per-trial redraw would otherwise leave stale.

``trace`` prints one flushed line per tick: tick, mode, the served command, the action, the joint
positions and the gripper bit (``-`` in the shared modes: the gripper there is the policy's). The
joint positions are the observation the action was computed from, i.e. the state *before* that row's
action. ``run`` catches nothing: Ctrl-C propagates and the caller closes the station. ``run_recording`` follows c3po's documented ``recording()`` pattern (``n``/``r``/
``q`` flags, ``cancel_episode``), downloads the dataset before the caller closes the station, and
returns its local path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import bridge
import numpy as np
from bridge import ARM_DIMS, TWIST_DIMS
from numpy.typing import NDArray
from policy import teleop_action
from readout import ADAPTER_SOURCES, Readout, resolve
from source import DeflectionSource
from station import RecordingStation, observation_batch
from stats import action_std

if TYPE_CHECKING:
    from config import Config
    from policy import Bundle
    from statefiles import Recorder
    from station import Station

MODES = ("teleop", "autonomous", "steered")
"""The loop's three modes: the operator alone, the policy alone, the operator through a method."""


class Loop:
    """One run: one station, one mode, one command source and (in the shared modes) one method.

    ``mode`` fixes what must be present: ``teleop`` a source and nothing else, ``autonomous`` a
    bundle and nothing else, ``steered`` a source, a bundle and the steering policy. The mode is
    checked here, before the first observation, so a mis-composed run fails on the ground rather than
    after a reset. ``adapter_from_corruption`` is the shared modes' adapter sync (``mars-f``); it
    needs the steering policy, so it is refused by the other modes.
    """

    def __init__(
        self,
        *,
        config: Config,
        station: Station,
        mode: str,
        source: DeflectionSource | None = None,
        bundle: Bundle | None = None,
        steering: Any = None,
        adapter_from_corruption: str | None = None,
    ) -> None:
        _validate(mode, source, bundle, steering, adapter_from_corruption)
        self._config = config
        self._station = station
        self._mode = mode
        self._source = source
        self._bundle = bundle
        self._steering = steering
        self._adapter_from_corruption = adapter_from_corruption

        self._arm = station.arm
        self._n_action_steps = int(config.policy.n_action_steps)
        self._length = float(config.command.twist_length)
        self._cameras = _cameras(config.cameras)
        self._noise_rng = np.random.default_rng(int(config.readout.seed))
        self._episode = 0  # the episode index: the random family's seed
        self._q0: NDArray | None = None  # the corruption's nominal pose, captured at the first episode
        self._fixed_readout: Readout | None = None  # a fixed corruption, resolved once
        self._readout: Readout | None = None  # the episode's readout in force
        self._held_served: NDArray | None = None  # the shared modes' per-chunk command, in force
        self._held_deflection: NDArray | None = None  # the deflection that command came from
        self._batch: dict | None = None  # the chunk's prepared batch, reused mid-chunk
        self._states: Recorder | None = None  # the caller's recorder, set by run/run_recording

    # ------------------------------------------------------------ the run

    def run(
        self,
        *,
        prompt: str,
        max_steps: int | None = None,
        trace: bool = False,
        states: Recorder | None = None,
    ) -> None:
        """Drive one episode: ``max_steps`` ticks, or until interrupted (Ctrl-C propagates).

        ``prompt`` is the task text in the batch. ``trace`` prints one line per tick; its joint
        positions are the observation that tick's action was computed from, i.e. the state *before*
        the action. ``states`` is an optional ``statefiles.Recorder``: one pre-action state row is
        appended at every chunk boundary (``_record``), in every mode. The loop never saves it -- the
        CLI does, like the dataset download -- so an interrupted run writes no partial file. The
        caller closes the station.
        """
        self._states = states
        obs = self._station.reset()
        self._begin_episode(obs)
        tick = 0
        while max_steps is None or tick < max_steps:
            obs = self._station.step(self._tick(obs, prompt=prompt, tick=tick, trace=trace))
            tick += 1

    def run_recording(
        self,
        *,
        name: str,
        dest: Path,
        prompt: str,
        trace: bool = False,
        states: Recorder | None = None,
    ) -> Path:
        """Record episodes with c3po's keyboard flags, download the dataset, return its local path.

        The station must be a ``station.RecordingStation`` (``C3poStation``): this raises rather than
        hoping for ``record``/``download`` on whatever it was given. c3po's documented ``recording()``
        pattern: ``q`` stops, ``n`` ends an episode, ``r`` discards the episode that just ended
        (``station.cancel_episode()``) before it is re-recorded. ``dest`` is created if absent; the
        returned path is the station's (``dest / name`` for c3po). ``states`` is the same optional
        recorder ``run`` takes; it collects rows from every episode, and a ``r`` re-record retracts the
        discarded take's rows (``Recorder.drop_episode``) together with c3po's dataset cancel, so only
        takes the operator kept reach the file.
        """
        station = self._station
        if not isinstance(station, RecordingStation):
            raise ValueError(
                "run_recording needs a recording station (station.RecordingStation: record/download, "
                f"C3poStation); got {type(station).__name__}, which records nothing"
            )
        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        self._states = states
        with station.record(name) as rec:
            while not rec.stop:
                obs = station.reset()
                self._begin_episode(obs)
                tick = 0
                while not rec.stop and not rec.done:
                    obs = station.step(self._tick(obs, prompt=prompt, tick=tick, trace=trace))
                    tick += 1
                if rec.rerecord:
                    station.cancel_episode()  # c3po discards the take's dataset ...
                    if self._states is not None:
                        self._states.drop_episode(self._episode - 1)  # ... and its states
                rec.clear()
        return station.download(name, destination)

    # ------------------------------------------------------------ the episode

    def _begin_episode(self, obs: dict[str, NDArray]) -> None:
        """A fresh episode: the policy's queues, the operator's latch, and the corruption's draw."""
        if self._bundle is not None:
            self._bundle.reset()
        if self._source is not None:  # autonomous has no operator device
            self._source.reset()
        self._held_served = None
        self._held_deflection = None
        self._batch = None
        # autonomous has no operator command, so there is nothing to corrupt: no readout is drawn
        self._readout = None if self._mode == "autonomous" else self._resolve_readout(obs)
        if self._readout is not None and self._readout.adapter is not None:
            self._steering.adapter.matrix = self._readout.adapter.copy()  # consumers read the holder
        self._episode += 1

    def _tick(self, obs: dict[str, NDArray], *, prompt: str, tick: int, trace: bool) -> NDArray:
        """One control step: the command, the action; the station's next observation is the caller's.

        In teleop the operator's whole chain runs every tick. In the shared modes it runs once per
        chunk, at the boundary observation, and the served command is held for the chunk's ticks
        (a mid-chunk read is not what the steering core consumes, and it would redraw the noise).

        A recorder, when the caller passed one, is called once at each chunk boundary with that
        boundary's *pre-action* state: the raw batch before the pre-processor, the observation's joint
        positions and gripper, the deflection the command was read from and the served command in
        force. The call happens before the action is computed, and it neither draws a random number
        nor reads the operator's device again, so the actions are the same with or without it.
        """
        chunk_start = tick % self._n_action_steps == 0
        if self._mode == "teleop":
            self._held_served = self._command(obs)  # every tick: the operator is the loop
        elif self._mode == "steered" and chunk_start:
            self._held_served = self._command(obs)  # once per chunk: the core reads it at the boundary
        if self._mode != "teleop" and chunk_start:
            raw_batch = self._raw_batch(obs, prompt)
            self._record(raw_batch, obs, tick=tick, prompt=prompt)  # before the pre-processor sees it
            self._batch = self._bundle.prepare(raw_batch)
        elif chunk_start and self._states is not None:
            # teleop has no bundle: a recording builds a batch for the recording alone
            self._record(self._raw_batch(obs, prompt), obs, tick=tick, prompt=prompt)
        if self._mode == "teleop":
            served = self._held_served
            action = teleop_action(served, self._source.gripper())
        else:
            served = None if self._mode == "autonomous" else self._held_served
            action = self._bundle.act(self._batch, served)
        if trace:
            self._trace(tick, served, action, obs)
        return action

    def _command(self, obs: dict[str, NDArray]) -> NDArray:
        """The operator's served command: deflection -> twist -> IK -> F, gain, one noise draw.

        The deflection read here is kept as the one in force, so a recorder stores the exact stick
        position the command came from without reading the device a second time.
        """
        if self._readout is None:
            raise RuntimeError("the episode's readout is not resolved: call run() to begin an episode")
        deflection = self._source.deflection()
        self._held_deflection = np.array(deflection, dtype=np.float64, copy=True)
        twist = _twist(deflection, self._config.command)
        intent = bridge.operator_command(twist, bridge.nominal_pose(obs, self._arm), self._length)
        return self._readout.serve(intent, self._noise_rng)

    def _raw_batch(self, obs: dict[str, NDArray], prompt: str) -> dict:
        """The station's observation as the policy's raw batch (before the pre-processor)."""
        return observation_batch(obs, arm=self._arm, cameras=self._cameras, task=prompt)

    def _record(self, raw_batch: dict, obs: dict[str, NDArray], *, tick: int, prompt: str) -> None:
        """Append one chunk-boundary row to the caller's recorder, if there is one.

        Called from ``_tick`` at the boundary, after the boundary's command and raw batch are in hand
        and before the action is computed: the row is the state the action is about to be computed
        from, i.e. the state *before* the action -- the field where steering actually was. ``raw_batch``
        is the un-preprocessed batch the policy's chunk starts from (teleop builds one for this call
        alone); the joint positions and gripper are the observation's, the deflection and the served
        command are the ones in force at the boundary, and ``autonomous`` -- which has no operator and
        no command -- stores zeros in both. ``q0`` is the session's nominal pose, which only the loop
        resolves (the CLI cannot know a station's reset pose before the run); ``episode`` is 0-based
        and is the same index the episode's readout drew a ``{random: ...}`` family from. It draws no
        random number and reads no device, so the actions sent do not depend on whether it ran.
        """
        if self._states is None:
            return
        if self._mode == "autonomous":
            deflection, served = np.zeros(TWIST_DIMS), np.zeros(ARM_DIMS)
        else:
            deflection, served = self._held_deflection, self._held_served
        self._states.add(
            raw_batch,
            joint_position=bridge.nominal_pose(obs, self._arm),
            gripper=_gripper(obs, self._arm),
            deflection=deflection,
            served=served,
            q0=self._nominal_pose(obs),
            episode=self._episode - 1,
            tick=tick,
            prompt=prompt,
            mode=self._mode,
        )

    # ------------------------------------------------------------ corruption

    def _resolve_readout(self, obs: dict[str, NDArray]) -> Readout:
        """This episode's readout: once for a fixed spec, redrawn per episode for a random family."""
        spec = self._config.readout.corruption
        random_family = isinstance(spec, dict) and set(spec) == {"random"}
        if self._fixed_readout is not None and not random_family:
            return self._fixed_readout
        episode_readout = resolve(
            self._config,
            q0=self._nominal_pose(obs),
            action_std7=self._action_std(),
            episode=self._episode,
            adapter_from_corruption=self._adapter_from_corruption,
        )
        if not random_family:
            self._fixed_readout = episode_readout
            if episode_readout.label:  # a fixed corruption names itself, once, in the run header
                print(f"readout: {episode_readout.label}", flush=True)
        return episode_readout

    def _nominal_pose(self, obs: dict[str, NDArray]) -> NDArray:
        """``command.q0`` when configured, else the first reset pose: the lift's fixed pose, once."""
        if self._q0 is None:
            configured = tuple(self._config.command.q0)
            self._q0 = (
                np.asarray(configured, dtype=np.float64)
                if len(configured) == ARM_DIMS
                else bridge.nominal_pose(obs, self._arm)
            )
        return self._q0

    def _action_std(self) -> NDArray | None:
        """The per-dim action standard deviation the noise is expressed in, or None when off.

        The bundle carries the statistics in the shared modes; teleop never loads the checkpoint (or
        torch), so it reads the statistics file directly. ``readout.noise = 0`` needs no scale at all,
        so a clean teleop run neither reads nor requires the file. A bundle without statistics has no
        unit for the noise, and ``readout.resolve`` refuses that rather than silently drawing none:
        ``--dry-run``'s ``fake_bundle`` carries a stand-in std for this reason, and a real steered
        bundle always has them (``make_steering`` refuses otherwise).
        """
        if not float(self._config.readout.noise):
            return None
        if self._bundle is None:
            return action_std(self._config.policy.stats)
        return self._bundle.std

    # ------------------------------------------------------------ the trace

    def _trace(self, tick: int, served: NDArray | None, action: NDArray, obs: dict[str, NDArray]) -> None:
        """One flushed line: tick, mode, the served command, the action, the state before it, gripper.

        The gripper column is the operator's button only in teleop; in the shared modes the gripper is
        the policy's, so the column is ``-`` rather than a button state nobody acts on.
        """
        command = "policy" if served is None else " ".join(f"{value:+.3f}" for value in served)
        executed = " ".join(f"{value:+.3f}" for value in action)
        joints = " ".join(f"{value:+.3f}" for value in bridge.nominal_pose(obs, self._arm))
        gripper = ("1" if action[ARM_DIMS] > 0.5 else "0") if self._mode == "teleop" else "-"
        print(
            f"{tick:6d} {self._mode:10s} cmd[{command}] act[{executed}] q[{joints}] grip {gripper}",
            flush=True,
        )


# ---------------------------------------------------------------- the pieces the loop reads


def _validate(
    mode: str,
    source: DeflectionSource | None,
    bundle: Bundle | None,
    steering: Any,
    adapter_from_corruption: str | None,
) -> None:
    """The mode's composition, checked before anything is loaded or moved."""
    if mode not in MODES:
        raise ValueError(f"mode: expected one of {MODES}, got {mode!r}")
    if mode == "teleop":
        if source is None:
            raise ValueError("teleop needs a source and no bundle or steering: source is None")
        if bundle is not None or steering is not None:
            raise ValueError("teleop has no policy: bundle and steering must be None")
    elif mode == "autonomous":
        if bundle is None:
            raise ValueError("autonomous needs a bundle and no source or steering: bundle is None")
        if source is not None or steering is not None:
            raise ValueError("autonomous has no operator: source and steering must be None")
    else:
        missing = [
            name
            for name, value in (("source", source), ("bundle", bundle), ("steering", steering))
            if value is None
        ]
        if missing:
            raise ValueError(
                f"steered needs a source, a bundle and a steering policy; missing {', '.join(missing)}"
            )
    if adapter_from_corruption is not None:
        if mode != "steered":
            raise ValueError(
                f"adapter_from_corruption is the shared modes' adapter sync; mode {mode!r} has no steering"
            )
        if adapter_from_corruption not in ADAPTER_SOURCES:
            raise ValueError(
                f"adapter_from_corruption: expected None, 'corruption' or 'corruption_inverse', "
                f"got {adapter_from_corruption!r}"
            )


def _cameras(cameras: Any) -> dict[str, str | None]:
    """The station's camera roles from the config's ``[cameras]`` section (third may be absent)."""
    return {"exterior": cameras.exterior, "wrist": cameras.wrist, "third": cameras.third}


def _twist(deflection: NDArray, command: Any) -> NDArray:
    """The 6 deflection axes as a twist: translation m/s, then rotation rad/s."""
    twist = np.array(deflection, dtype=np.float64)
    if twist.shape != (TWIST_DIMS,):
        raise ValueError(f"a deflection must be ({TWIST_DIMS},), got shape {twist.shape}")
    twist[:3] *= float(command.translation_gain)
    twist[3:] *= float(command.rotation_gain)
    return twist


def _gripper(obs: dict[str, NDArray], arm: str) -> NDArray:
    """(1,) the station's gripper position from an observation: the state's dim 8 in the batch."""
    key = f"{arm}/gripper_position"
    if key not in obs:
        raise ValueError(f"the observation has no {key!r} (keys: {sorted(obs)})")
    value = np.asarray(obs[key], dtype=np.float64)
    if value.shape != (1,):
        raise ValueError(f"{key!r}: expected shape (1,), got {value.shape}")
    return value
