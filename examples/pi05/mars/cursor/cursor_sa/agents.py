"""Agents that produce the command: a simulated expert, or a person on the SpaceMouse.

An agent is `reset` for a batch of trials with the input channel, then
`step(pos, layouts, ctl)` is called once per time step and returns

    raw        [B, 2]      the deflection in [-1, 1] the agent means this step
    served     [B, 2]      the same after the channel (corruption, noise): what the controller gets
    reference  None, or at a replan the served chunk [B, H, 2] an expert_chunk agent means

The channel (`channel.InputChannel`) is the same object for a person and for the
expert, so the same corruption and noise apply to both.
"""

import time
from collections import deque

import numpy as np
import torch

from .channel import InputChannel, is_active
from .env import AVOID_CLEARANCE, densify, plan_route
from .devices import Keyboard, SpaceMouse, find_spacemouse, spacemouse_full_scale, spacemouse_hid_id
from .steering import to_deflection


class StepAgent:
    """An agent that means one deflection per step; subclasses define `command(pos, layouts) -> [B, 2]`."""

    def reset(self, n_trials=1, channel=None):
        self.channel = channel if channel is not None else InputChannel()

    def step(self, pos, layouts, ctl=None):
        raw = self.command(pos, layouts)
        return raw, self.channel(raw), None


def pursuit_index(pos, pts, valid, discs, dmask):
    """For each row, the index of the farthest valid point of its route pts [n, P, 2] (valid where
    `valid` [n, P]) whose straight line from pos [n, 2] clears every disc of discs [n, K, 3] (valid
    where dmask [n, K]); -1 where none does."""
    d = pts - pos[:, None]                                          # [n, P, 2]
    rel = discs[:, None, :, :2] - pos[:, None, None]                # [n, 1, K, 2]
    dd = np.maximum((d * d).sum(-1), 1e-12)[..., None]              # [n, P, 1]
    tau = np.clip((rel * d[:, :, None]).sum(-1) / dd, 0.0, 1.0)     # [n, P, K]
    gap2 = ((rel - tau[..., None] * d[:, :, None]) ** 2).sum(-1)    # [n, P, K]
    blocked = ((gap2 < discs[:, None, :, 2] ** 2) & dmask[:, None]).any(-1)
    return np.where(valid & ~blocked, np.arange(pts.shape[1]), -1).max(-1)


class ExpertAgent(StepAgent):
    """Pushes at the goal, around the task's obstacles if there are any: the intent a person would have.

    Full deflection toward the goal, tapering linearly inside `slow_radius`, seen `lag` steps late.
    With obstacles it plans a route once (`env.plan_route`: discs inflated by `clearance`, the other
    targets avoided, the shorter way round) and pushes toward the farthest point of the route it
    can see with every disc `margin` clear of the line of sight (pure pursuit, so no corner is cut).
    When it sees no point of its route (pushed off it) it heads for the nearest one and, at most every
    10 steps, re-plans from where it is, keeping its side. Batched over trials.
    """

    name = "expert"

    def __init__(self, lag=0, slow_radius=0.15, clearance=0.05, margin=0.03, target_radius=0.05, spacing=0.02):
        self.lag, self.slow_radius, self.clearance, self.margin = lag, slow_radius, clearance, margin
        self.target_radius, self.spacing = target_radius, spacing  # distractors are discs this big; route point spacing
        self.reset()

    def reset(self, n_trials=1, channel=None):
        super().reset(n_trials, channel)
        self._history = deque(maxlen=self.lag + 1)
        self._routes = [None] * n_trials            # per trial: (waypoints [W, 2], points every `spacing` along them)
        self._since = np.zeros(n_trials, dtype=int)  # steps since the last plan
        self._sight = None                           # per trial: the discs a line of sight must clear

    def command(self, pos, layouts):
        goal_pos = layouts.goal_pos
        target, at_goal = goal_pos.copy(), np.ones(len(pos), dtype=bool)
        rows = np.flatnonzero(layouts.obstacle_mask.any(1))
        if len(rows):
            target[rows], at_goal[rows] = self._pursue(rows, pos, layouts)
        to_target = target - pos
        dist = np.linalg.norm(to_target, axis=-1, keepdims=True)
        taper = np.where(at_goal[:, None], np.minimum(1.0, dist / self.slow_radius), 1.0)
        command = to_target / np.maximum(dist, 1e-9) * taper
        self._history.append(np.clip(command, -1.0, 1.0))
        return self._history[0].copy()

    def _plan(self, b, pos, layouts, hint=None):
        """Trial b's route from pos: the planner at the expert's clearance, then at less if that finds no
        way through (pushed close to a disc), the straight line as a last resort."""
        obstacles, goal = layouts.real_obstacles(b), layouts.goal_pos[b]
        avoid = layouts.distractors(b, self.target_radius)
        route = None
        for clearance in (self.clearance, 0.6 * self.clearance, 0.3 * self.clearance):
            route = plan_route(pos, goal, obstacles, clearance, hint=hint, avoid=avoid)
            if route is not None:
                break
        route = np.array([goal]) if route is None else route
        self._routes[b] = (route, densify(np.vstack([pos, route]), self.spacing))
        self._since[b] = 0

    def _sight_discs(self, layouts):
        """[B, K, 4] (x, y, radius with the margin, radius) of every obstacle and distractor target, and the mask."""
        B = len(layouts)
        parts = [np.concatenate([layouts.real_obstacles(b)[:, :2], layouts.real_obstacles(b)[:, 2:] + self.margin,
                                 layouts.real_obstacles(b)[:, 2:]], 1) for b in range(B)]
        parts = [np.concatenate([p, np.concatenate([d[:, :2], d[:, 2:] + AVOID_CLEARANCE, d[:, 2:]], 1)])
                 for p, d in zip(parts, (layouts.distractors(b, self.target_radius) for b in range(B)))]
        K = max(len(p) for p in parts)
        discs, mask = np.zeros((B, K, 4)), np.zeros((B, K), dtype=bool)
        for b, p in enumerate(parts):
            discs[b, :len(p)], mask[b, :len(p)] = p, True
        return discs, mask

    def _pursue(self, rows, pos, layouts):
        """The point each trial in `rows` pushes toward, and whether it is the goal."""
        if self._sight is None:
            self._sight = self._sight_discs(layouts)
        for b in rows:
            if self._routes[b] is None:
                self._plan(b, pos[b], layouts)
        self._since[rows] += 1
        P = max(len(self._routes[b][1]) for b in rows)
        pts, valid = np.zeros((len(rows), P, 2)), np.zeros((len(rows), P), dtype=bool)
        for i, b in enumerate(rows):
            route = self._routes[b][1]
            pts[i, :len(route)], valid[i, :len(route)] = route, True
            pts[i, len(route):] = route[-1]
        discs, dmask = self._sight[0][rows], self._sight[1][rows]
        best = pursuit_index(pos[rows], pts, valid, discs[..., [0, 1, 2]], dmask)
        lost = best < 0
        if lost.any():  # inside a margin already: sight lines need only clear the discs themselves
            best[lost] = pursuit_index(pos[rows[lost]], pts[lost], valid[lost], discs[lost][..., [0, 1, 3]], dmask[lost])
        target, at_goal = np.empty((len(rows), 2)), np.zeros(len(rows), dtype=bool)
        for i, b in enumerate(rows):
            if best[i] < 0:
                if self._since[b] >= 10:
                    self._plan(b, pos[b], layouts, hint=self._routes[b][0][0])
                route = self._routes[b][1]
                target[i] = route[np.linalg.norm(route - pos[b], axis=1).argmin()]
                at_goal[i] = len(self._routes[b][0]) == 1 and np.allclose(target[i], route[-1])
            else:
                target[i], at_goal[i] = pts[i, best[i]], best[i] == len(self._routes[b][1]) - 1
        return target, at_goal

    @property
    def idle(self):
        return True

    def params(self):
        return dict(lag=self.lag, slow_radius=self.slow_radius, clearance=self.clearance, margin=self.margin)


class ExpertChunkAgent:
    """Means the controller's own policy chunk, goal given: at every replan it plans
    with the goal flag set and pushes that chunk step by step, at the controller's
    pace (its gain mode). This is the BCI model of the learned-adapter study: what
    the operator intends is a policy chunk, the channel corrupts each of its steps,
    and the served chunk is the controller's reference.
    """

    name = "expert_chunk"

    def reset(self, n_trials=1, channel=None):
        self.channel = channel if channel is not None else InputChannel()
        self.raw = self.served = None

    @torch.no_grad()
    def plan(self, ctl, layouts):
        """The intended chunk [B, H, 2] as a deflection per step (full deflection = max speed)."""
        obs = ctl.observation(goal_given=True)
        noise = torch.randn(len(layouts), ctl.policy.horizon, 2, device=obs["pos"].device)
        return to_deflection(ctl.sample_chunk(obs, noise), ctl.policy, ctl.dt, ctl.max_speed).cpu().numpy()

    def step(self, pos, layouts, ctl):
        new = ctl.t >= ctl.cycle  # the controller starts a chunk at this control step
        if new:  # a replan: plan the chunk, and let the channel corrupt every step of it at once
            self.raw = self.plan(ctl, layouts)
            self.served = self.channel(self.raw)
        parts = ctl.steps_of(0 if new else ctl.t)  # the step(s) the controller plays now, at the gain's pace
        play = lambda chunk: sum((share / ctl.gain) * chunk[:, i] for i, share in parts)
        return play(self.raw), play(self.served), self.served if new else None

    @property
    def idle(self):
        return True

    def params(self):
        return {}


class HumanAgent(StepAgent):
    """A person on the SpaceMouse; the arrow keys only when `allow_keyboard`.

    The SpaceMouse is found on the first `poll()` and again if it is unplugged;
    the SPACEMOUSE_HID_ID and SPACEMOUSE_FULL_SCALE environment variables pin the
    device and set its full-scale count (see devices.py).
    The renderer feeds the keyboard state through `keyboard.update(...)`.
    """

    name = "human"

    def __init__(self, allow_keyboard=False, rescan_s=2.0):
        self.allow_keyboard, self.rescan_s = allow_keyboard, rescan_s
        self.keyboard = Keyboard()
        self.spacemouse, self.status, self._next_scan = None, "not found", 0.0
        self.reset()

    def poll(self):
        """Connect the SpaceMouse when it appears; drop it when it is unplugged."""
        now = time.monotonic()
        if (self.spacemouse is None or not self.spacemouse.connected) and now >= self._next_scan:
            self._next_scan, self.spacemouse = now + self.rescan_s, None
            path = find_spacemouse(spacemouse_hid_id())
            if path is None:
                self.status = "not found"
            else:
                try:
                    self.spacemouse, self.status = SpaceMouse(path, spacemouse_full_scale()), f"connected ({path})"
                except OSError as e:
                    self.status = f"found at {path} but cannot open it ({e.strerror})"
        return self.spacemouse is not None

    def reset(self, n_trials=1, channel=None):
        super().reset(n_trials, channel)
        self._last = np.zeros(2)

    def command(self, pos=None, layouts=None):
        self.poll()
        if self.spacemouse is not None and is_active(self.spacemouse.translation):
            self._last = self.spacemouse.translation
        elif self.allow_keyboard:
            self._last = self.keyboard.translation
        else:
            self._last = np.zeros(2)
        return self._last[None].copy()

    @property
    def connected(self):
        return self.spacemouse is not None and self.spacemouse.connected

    @property
    def idle(self):
        return not is_active(self.command()[0])

    def take_button(self):
        """True once per SpaceMouse button press."""
        return self.spacemouse is not None and self.spacemouse.take_button()

    def params(self):
        return dict(allow_keyboard=self.allow_keyboard, device=self.spacemouse.path if self.spacemouse else None)

    def close(self):
        if self.spacemouse is not None:
            self.spacemouse.close()


AGENTS = {"human": HumanAgent, "expert": ExpertAgent, "expert_chunk": ExpertChunkAgent}


def make_agent(name, **params):
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; expected one of {tuple(AGENTS)}")
    return AGENTS[name](**params)
