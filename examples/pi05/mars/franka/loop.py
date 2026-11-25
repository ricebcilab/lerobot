"""The one control loop: the station's clock, the operator's readout, the policy's action.

c3po owns the clock. One iteration sends one action with ``station.step(action8)``, which blocks
until the station's next observation; nothing here measures time (no ``time.sleep``, no timers, no
tick counters beyond the action index, no resent actions). The station is asked for the checkpoint's
own 15 Hz, so one shipped action per control step is one trained control step.

The chain is ``bridge.py``'s, in DESIGN.md's order: the operator's 6-axis deflection in [-1, 1]
scaled to a twist (``command.translation_gain`` m/s and ``command.rotation_gain`` rad/s at full
deflection) -> ``bridge.operator_command`` (IK of the twist at the current pose, in action units) ->
``bridge.readout`` through the corruption lift ``F`` and the noise -> the served 7-wide command ->
the method -> the 8-wide station action. The command is read every tick in teleop and once per chunk
(``tick % n_action_steps == 0``) in the shared modes, where it is held for the chunk: the steering
core reads a batch's command only when its action queue is empty, so a per-tick read would change
nothing.

The station's raw batch is built once per chunk (``bundle.prepare``) and reused for the
``n_action_steps - 1`` mid-chunk ticks (``bundle.act``): the pre-processor (two images, the
tokenizer, the device transfer) is the expensive half of a tick, and the policies read a batch only
when their queue is empty. Teleop has no bundle and never builds a batch.

Corruption is resolved lazily, at the first episode, because it needs ``q0``: the configured
``command.q0`` or else the reset pose. A fixed corruption is drawn once and printed in the run
header; a ``{random: ...}`` family is redrawn at every episode from
``numpy.random.default_rng([readout.seed, episode])``. ``adapter_from_corruption`` keeps a ``mars-f``
steering policy's ``adapter.matrix`` in sync with the episode's ``F`` (or its exact inverse), which a
per-trial redraw would otherwise leave stale.

``trace`` prints one flushed line per tick: tick, mode, the served command, the action, the joint
positions and the gripper bit. The joint positions are the observation the action was computed from,
i.e. the state *before* that row's action. ``run`` catches nothing: Ctrl-C propagates and the caller
closes the station. ``run_recording`` follows c3po's documented ``recording()`` pattern (``n``/``r``/
``q`` flags, ``cancel_episode``), downloads the dataset before the caller closes the station, and
returns its local path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import bridge
import numpy as np
from bridge import ARM_DIMS, TWIST_DIMS
from numpy.typing import NDArray
from policy import teleop_action
from station import observation_batch
from stats import action_std

if TYPE_CHECKING:
    from collections.abc import Sequence

    from config import Config
    from policy import Bundle
    from station import Station

MODES = ("teleop", "autonomous", "steered")
"""The loop's three modes: the operator alone, the policy alone, the operator through a method."""

ADAPTER_SOURCES = ("corruption", "corruption_inverse")
"""How ``mars-f``'s fixed adapter follows the episode's corruption; None leaves the holder alone."""


class DeflectionSource(Protocol):
    """The operator's device: six deflection axes in [-1, 1] and a latched gripper button.

    The axes are the device's, ``[x, y, z, rx, ry, rz]``; the loop scales the first three to m/s and
    the last three to rad/s. ``gripper()`` is a latch, not a momentary press: it returns True while
    the operator has the gripper closed. The protocol has no episode reset; the concrete sources each
    provide ``reset()`` for the latch (the loop calls it when it has one).
    """

    def deflection(self) -> NDArray: ...

    def gripper(self) -> bool: ...

    def close(self) -> None: ...


class SpaceMouseSource:
    """A 3Dconnexion SpaceMouse as a :class:`DeflectionSource`: lerobot's six-channel twist, latched.

    ``path``/``hid_id``/``full_scale``/``rotation_full_scale`` are ``SpaceMouseTwist``'s (environment
    variables' defaults included); ``source``/``sign`` map the device's six channels onto the twist's
    axes, identity by default. A press of any button toggles the gripper latch; the device's button
    report is a bitmask, so ``reader.take_press()`` is the once-per-press edge. lerobot is imported
    inside ``__init__``: its teleoperator package imports torch, and ``import loop`` must not.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        hid_id: str | None = None,
        full_scale: float | None = None,
        rotation_full_scale: float | None = None,
        source: Sequence[int] | None = None,
        sign: Sequence[float] | None = None,
    ) -> None:
        from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTwist

        self._twist = SpaceMouseTwist(
            source=(0, 1, 2, 3, 4, 5) if source is None else source,
            sign=(1.0,) * TWIST_DIMS if sign is None else sign,
            path=path,
            hid_id=hid_id,
            full_scale=full_scale,
            rotation_full_scale=rotation_full_scale,
        )
        self._closed = False

    def deflection(self) -> NDArray:
        return np.asarray(self._twist.twist, dtype=np.float64)

    def gripper(self) -> bool:
        """True while the gripper is latched closed; each button press toggles the latch."""
        if self._twist.reader.take_press():
            self._closed = not self._closed
        return self._closed

    def reset(self) -> None:
        """A new episode starts with the gripper open (the latch, not the device)."""
        self._closed = False

    def close(self) -> None:
        self._twist.close()


class FakeSource:
    """A deterministic :class:`DeflectionSource` for tests and ``--dry-run``: fields, not a device.

    ``deflection_vector`` and ``gripper_closed`` are settable between ticks (a test changes the stick
    mid-chunk); ``reset`` opens the gripper latch at each episode. ``reads`` counts the deflection
    reads, which is how a test checks the command cadence.
    """

    def __init__(self, *, deflection: NDArray | None = None, gripper: bool = False) -> None:
        self.deflection_vector = np.zeros(TWIST_DIMS) if deflection is None else np.asarray(deflection)
        if self.deflection_vector.shape != (TWIST_DIMS,):
            raise ValueError(
                f"deflection must be ({TWIST_DIMS},) in [-1, 1], got shape {self.deflection_vector.shape}"
            )
        self.gripper_closed = bool(gripper)
        self.reads = 0

    def deflection(self) -> NDArray:
        self.reads += 1
        return np.asarray(self.deflection_vector, dtype=np.float64).copy()

    def gripper(self) -> bool:
        return self.gripper_closed

    def reset(self) -> None:
        self.gripper_closed = False

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _Readout:
    """One episode's readout in force: the lifted corruption (None off), the gain and the noise std."""

    matrix7: NDArray | None = field(repr=False)  # F, the lifted corruption; None = corruption off
    adapter: NDArray | None = field(repr=False)  # the matrix mars-f's holder gets; None = leave it
    gain: float
    sigma7: NDArray | None = field(repr=False)


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

        self._arm = _arm(station)
        self._n_action_steps = int(config.policy.n_action_steps)
        self._length = float(config.command.twist_length)
        self._cameras = _cameras(config.cameras)
        self._noise_rng = np.random.default_rng(int(config.readout.seed))
        self._episode = 0  # the episode index: the random family's seed
        self._q0: NDArray | None = None  # the corruption's nominal pose, captured at the first episode
        self._fixed_readout: _Readout | None = None  # a fixed corruption, resolved once
        self._readout: _Readout | None = None  # the episode's readout in force
        self._held_deflection: NDArray | None = None  # the shared modes' per-chunk command
        self._batch: dict | None = None  # the chunk's prepared batch, reused mid-chunk

    # ------------------------------------------------------------ the run

    def run(self, *, prompt: str, max_steps: int | None = None, trace: bool = False) -> None:
        """Drive one episode: ``max_steps`` ticks, or until interrupted (Ctrl-C propagates).

        ``prompt`` is the task text in the batch. ``trace`` prints one line per tick; its joint
        positions are the observation that tick's action was computed from, i.e. the state *before*
        the action. The caller closes the station.
        """
        obs = self._station.reset()
        self._begin_episode(obs)
        tick = 0
        while max_steps is None or tick < max_steps:
            obs = self._station.step(self._tick(obs, prompt=prompt, tick=tick, trace=trace))
            tick += 1

    def run_recording(self, *, name: str, dest: Path, prompt: str, trace: bool = False) -> Path:
        """Record episodes with c3po's keyboard flags, download the dataset, return its local path.

        c3po's documented ``recording()`` pattern: ``q`` stops, ``n`` ends an episode, ``r`` discards
        the episode that just ended (``station.cancel_episode()``) before it is re-recorded. ``dest``
        is created if absent; the returned path is the station's (``dest / name`` for c3po).
        """
        destination = Path(dest)
        destination.mkdir(parents=True, exist_ok=True)
        with self._station.record(name) as rec:
            while not rec.stop:
                obs = self._station.reset()
                self._begin_episode(obs)
                tick = 0
                while not rec.stop and not rec.done:
                    obs = self._station.step(self._tick(obs, prompt=prompt, tick=tick, trace=trace))
                    tick += 1
                if rec.rerecord:
                    self._station.cancel_episode()
                rec.clear()
        return self._station.download(name, destination)

    # ------------------------------------------------------------ the episode

    def _begin_episode(self, obs: dict[str, NDArray]) -> None:
        """A fresh episode: the policy's queues, the operator's latch, and the corruption's draw."""
        if self._bundle is not None:
            self._bundle.reset()
        self._reset_source()
        self._held_deflection = None
        self._batch = None
        # autonomous has no operator command, so there is nothing to corrupt: no readout is drawn
        self._readout = None if self._mode == "autonomous" else self._resolve_readout(obs)
        if self._readout is not None and self._readout.adapter is not None:
            self._steering.adapter.matrix = self._readout.adapter.copy()  # consumers read the holder
        self._episode += 1

    def _tick(self, obs: dict[str, NDArray], *, prompt: str, tick: int, trace: bool) -> NDArray:
        """One control step: the command, the action; the station's next observation is the caller's."""
        if self._mode == "teleop":
            self._held_deflection = self._source.deflection()  # every tick: the operator is the loop
        elif self._mode == "steered" and tick % self._n_action_steps == 0:
            self._held_deflection = self._source.deflection()  # once per chunk, held for the chunk
        if self._mode != "teleop" and tick % self._n_action_steps == 0:
            self._batch = self._bundle.prepare(self._raw_batch(obs, prompt))
        served = None if self._mode == "autonomous" else self._served(obs)
        if self._mode == "teleop":
            action = teleop_action(served, self._source.gripper())
        else:
            action = self._bundle.act(self._batch, served)
        if trace:
            self._trace(tick, served, action, obs)
        return action

    def _served(self, obs: dict[str, NDArray]) -> NDArray:
        """The operator's command through the bridge: deflection -> twist -> IK -> F, gain, noise."""
        if self._held_deflection is None:
            raise RuntimeError("the loop has no deflection: read the source before serving a command")
        twist = _twist(self._held_deflection, self._config.command)
        intent = bridge.operator_command(twist, bridge.nominal_pose(obs, self._arm), self._length)
        if self._readout is None:
            raise RuntimeError("the episode's readout is not resolved: call run() to begin an episode")
        readout = self._readout
        return bridge.readout(intent, readout.matrix7, readout.sigma7, readout.gain, self._noise_rng)

    def _raw_batch(self, obs: dict[str, NDArray], prompt: str) -> dict:
        """The station's observation as the policy's raw batch (before the pre-processor)."""
        return observation_batch(obs, arm=self._arm, cameras=self._cameras, task=prompt)

    def _reset_source(self) -> None:
        """Reset the operator's gripper latch for the new episode (the protocol's concrete sources do)."""
        reset = getattr(self._source, "reset", None)
        if callable(reset):
            reset()

    # ------------------------------------------------------------ corruption

    def _resolve_readout(self, obs: dict[str, NDArray]) -> _Readout:
        """This episode's corruption, gain and noise: once for a fixed spec, per episode for random."""
        spec = self._config.readout.corruption
        random_family = isinstance(spec, dict) and set(spec) == {"random"}
        if self._fixed_readout is not None and not random_family:
            return self._fixed_readout
        rng = (
            np.random.default_rng([int(self._config.readout.seed), self._episode]) if random_family else None
        )
        matrix6 = bridge.matrix6_from_spec(spec, rng) if spec is not None else np.eye(TWIST_DIMS)
        lifted = spec is not None or self._adapter_from_corruption is not None
        q0 = self._nominal_pose(obs) if lifted else None
        matrix7 = bridge.corruption_lift(matrix6, q0, self._length) if lifted else None
        adapter = None
        if self._adapter_from_corruption == "corruption":
            adapter = matrix7
        elif self._adapter_from_corruption == "corruption_inverse":
            adapter = bridge.lift_inverse(matrix6, q0, self._length)
        readout = _Readout(
            matrix7=matrix7 if spec is not None else None,
            adapter=adapter,
            gain=self._gain(spec, matrix7),
            sigma7=self._device_noise(),
        )
        if not random_family:
            self._fixed_readout = readout
            if spec is not None:
                print(
                    f"readout: fixed corruption {spec!r} "
                    f"(gain {readout.gain:.4f}, noise {self._config.readout.noise:g} sigma)",
                    flush=True,
                )
        return readout

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

    def _gain(self, spec: Any, matrix7: NDArray | None) -> float:
        """``command.gain``, with ``auto`` = 1 / ||F||_2 when the corruption is on, else 1.0."""
        gain = self._config.command.gain
        if gain != "auto":
            return float(gain)
        return 1.0 if spec is None or matrix7 is None else 1.0 / float(np.linalg.norm(matrix7, 2))

    def _device_noise(self) -> NDArray | None:
        """``readout.noise`` in action-std units, or None (no draws) when the noise is off.

        The per-dim standard deviations come from the bundle's statistics in the shared modes and from
        the statistics file directly in teleop, which never loads the checkpoint (or torch). A bundle
        without statistics (``--dry-run``'s fixed-action stand-in) has no scale to express the noise
        in, so it draws none; a real steered bundle always has them (``make_steering`` refuses else).
        """
        noise = float(self._config.readout.noise)
        if not noise:
            return None
        if self._bundle is None:
            std = action_std(self._config.policy.stats)
        else:
            std = self._bundle.std
            if std is None:  # --dry-run's fixed-action bundle has no statistics to scale
                return None
        return noise * np.asarray(std, dtype=np.float64)[:ARM_DIMS]

    # ------------------------------------------------------------ the trace

    def _trace(self, tick: int, served: NDArray | None, action: NDArray, obs: dict[str, NDArray]) -> None:
        """One flushed line: tick, mode, the served command, the action, the state before it, gripper."""
        command = "policy" if served is None else " ".join(f"{value:+.3f}" for value in served)
        executed = " ".join(f"{value:+.3f}" for value in action)
        joints = " ".join(f"{value:+.3f}" for value in bridge.nominal_pose(obs, self._arm))
        gripper = 1 if action[ARM_DIMS] > 0.5 else 0
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


def _arm(station: Station) -> str:
    """The arm the loop drives: the station's own ``arm``, else the manifest's single arm.

    ``C3poStation`` has an ``arm`` property; ``FakeStation`` has only the manifest, and both the
    tests and ``--dry-run`` run the loop on the fake, so the manifest is the fallback.
    """
    arm = getattr(station, "arm", None)
    if arm is not None:
        return arm
    arms = station.manifest.arms
    if len(arms) != 1:
        raise ValueError(f"the station has {len(arms)} arms {arms}; this example drives exactly one")
    return arms[0]


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
