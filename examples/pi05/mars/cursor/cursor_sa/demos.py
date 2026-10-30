"""Synthetic human-like reaching demonstrations.

Each reach is a minimum-jerk movement (Flash & Hogan 1985) whose duration
follows Fitts' law, bowed sideways by a random amount. On some reaches the
primary movement misses, and an overlapping corrective submovement brings the
cursor onto the goal (Meyer et al. 1988). The cursor then holds still until the
dwell selects the goal. Reaches between the same start and goal therefore vary,
which gives the flow head a real distribution to learn.

Around obstacles a demonstration follows the expert's own path (the pure-pursuit
`agents.ExpertAgent` along its planned route), timed as one minimum-jerk
movement whose Fitts duration grows with every radian the path turns.

`DemoSampler` draws (observation, action chunk) training pairs from the demos.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch

from .env import CursorEnv, TaskConfig

CURVE_SLOWDOWN = 0.15  # a demonstration around obstacles takes this fraction longer per radian its path turns


@dataclass
class DemoConfig:
    fitts_a: float = 0.3            # s
    fitts_b: float = 0.2            # s per bit
    speed_sd: float = 0.2           # log-normal jitter on movement duration
    max_bow: float = 0.25           # peak sideways deviation, as a fraction of reach distance
    p_correct: float = 0.4          # chance the primary movement misses and is corrected
    err_along: float = 0.08         # sd of the miss along the reach, fraction of distance
    err_across: float = 0.05        # sd of the miss across the reach, fraction of distance
    corr_onset: tuple = (0.6, 0.9)  # correction start, fraction of primary duration
    corr_a: float = 0.15            # Fitts' constants for the correction (s, s per bit)
    corr_b: float = 0.1


def min_jerk(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15 * x + 6 * x**2)


def plan_reach(rng, start, goal, task=TaskConfig(), demo=DemoConfig()):
    """Cursor positions [n + 1, 2] for one reach, ending at rest on the goal."""
    diff = goal - start
    dist = np.linalg.norm(diff)
    along = diff / dist
    across = np.array([-along[1], along[0]])
    width = 2 * task.target_radius

    T1 = (demo.fitts_a + demo.fitts_b * np.log2(dist / width + 1)) * rng.lognormal(0, demo.speed_sd)
    T, err = T1, np.zeros(2)
    if rng.random() < demo.p_correct:
        err = dist * (rng.normal(0, demo.err_along) * along + rng.normal(0, demo.err_across) * across)
        t_corr = rng.uniform(*demo.corr_onset) * T1
        T2 = demo.corr_a + demo.corr_b * np.log2(np.linalg.norm(err) / width + 1)
        T = max(T1, t_corr + T2)

    t = np.arange(int(np.ceil(T / task.dt)) + 1) * task.dt
    s = min_jerk(t / T1)
    bow = rng.uniform(-demo.max_bow, demo.max_bow) * dist * np.sin(np.pi * s)
    pos = start + s[:, None] * (diff + err) + bow[:, None] * across
    if err.any():
        pos -= min_jerk((t - t_corr) / T2)[:, None] * err
    return pos


def trace_routes(layouts, step=0.005, max_steps=4000):
    """The expert's path from each layout's start to its goal (`agents.ExpertAgent`: pure pursuit along
    its planned route), traced at a constant `step`: a list of [n, 2] arrays, each ending exactly on its goal."""
    from .agents import ExpertAgent  # the agents use the planner; the demos follow the agent's own route
    agent = ExpertAgent()
    agent.reset(len(layouts))
    goal, pos = layouts.goal_pos, layouts.start.copy()
    paths, done = [[p.copy()] for p in pos], np.zeros(len(layouts), dtype=bool)
    for _ in range(max_steps):
        push = agent.command(pos, layouts)
        to_goal = np.linalg.norm(goal - pos, axis=1)
        arrive = ~done & (to_goal <= step)
        move = push / np.maximum(np.linalg.norm(push, axis=1, keepdims=True), 1e-12) * step
        pos = np.where((done | arrive)[:, None], np.where(arrive[:, None], goal, pos), pos + move)
        for b in np.flatnonzero(~done):
            paths[b].append(pos[b].copy())
        done |= arrive
        if done.all():
            break
    return [np.array(p) for p in paths]


def plan_along(rng, path, task=TaskConfig(), demo=DemoConfig(), slowdown=CURVE_SLOWDOWN):
    """Cursor positions [n + 1, 2] along `path` [m, 2] (start to goal, densely sampled), ending at rest on
    the goal: one minimum-jerk movement over the path's length, with the Fitts duration of that length
    (and its log-normal jitter) stretched by `slowdown` per radian the path turns."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    coarse = np.stack([np.interp(np.arange(0, arc[-1], 0.02), arc, path[:, i]) for i in range(2)], 1)
    heading = np.arctan2(*np.diff(np.vstack([coarse, path[-1:]]), axis=0).T[::-1])
    turn = np.abs(np.angle(np.exp(1j * np.diff(heading)))).sum() if len(heading) > 1 else 0.0
    T = ((demo.fitts_a + demo.fitts_b * np.log2(arc[-1] / (2 * task.target_radius) + 1))
         * rng.lognormal(0, demo.speed_sd) * (1 + slowdown * turn))
    s = arc[-1] * min_jerk(np.arange(int(np.ceil(T / task.dt)) + 1) * task.dt / T)
    return np.stack([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])], 1)


def make_demos(rng, layouts, task=TaskConfig(), demo=DemoConfig(), max_tries=50, allow_failed=False):
    """One successful reach per layout, around the layout's obstacles if any (along the expert's path,
    `trace_routes` and `plan_along`).

    Returns actions [B, L, 2] (per-step cursor displacement, zero once the
    cursor is holding on the goal) and lengths [B], the step at which the dwell
    selected the goal. Plans that leave the screen, touch an obstacle, dwell on
    a distractor, or time out are redrawn. A layout with no successful plan
    after max_tries raises, or with `allow_failed` gets zero actions and length
    0 (the sampler never draws it): for training data, where a rare boxed-in
    layout is not worth a crash.
    """
    task = replace(task, cursor_noise=0.0)
    rows = np.flatnonzero(layouts.obstacle_mask.any(1))
    paths = {}
    for chunk in np.array_split(rows, max(1, int(np.ceil(len(rows) / 500)))):
        if len(chunk):
            paths.update(zip(chunk.tolist(), trace_routes(layouts[chunk])))
    B = len(layouts)
    plans = [None] * B
    todo = np.arange(B)
    for _ in range(max_tries):
        for b in todo:
            plans[b] = (plan_along(rng, paths[b], task, demo) if b in paths
                        else plan_reach(rng, layouts.start[b], layouts.goal_pos[b], task, demo))
        on_screen = np.array([((plans[b] >= 0) & (plans[b] <= 1)).all() for b in todo])
        env = replay(_pad([np.diff(plans[b], axis=0) for b in todo]), layouts[todo], task)
        todo = todo[~(env.success & on_screen)]
        if not len(todo):
            break
    else:
        if not allow_failed:
            raise RuntimeError(f"{len(todo)} layouts had no successful demo after {max_tries} tries")
        print(f"make_demos: dropping {len(todo)} of {B} layouts with no successful demo after {max_tries} tries", flush=True)
        for b in todo:
            plans[b] = layouts.start[b][None]

    actions = _pad([np.diff(p, axis=0) for p in plans])
    lengths = replay(actions, layouts, task).steps
    lengths[todo] = 0
    # The dwell can finish after the last planned step; cover the whole trial.
    extra = max(0, lengths.max() - actions.shape[1])
    return np.concatenate([actions, np.zeros((B, extra, 2))], 1), lengths


def replay(actions, layouts, task=TaskConfig(), rng=None):
    """Step action sequences [B, L, 2] open loop; the cursor holds after they end."""
    env = CursorEnv(layouts, task, rng)
    hold = np.zeros_like(actions[:, 0])
    for t in range(task.max_steps):
        if env.step(actions[:, t] if t < actions.shape[1] else hold).all():
            break
    return env


def _pad(seqs):
    out = np.zeros((len(seqs), max(len(s) for s in seqs), 2))
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


class DemoSampler:
    """Draws (observation, action chunk) pairs uniformly over all demo time steps.
    Steps before a demo starts count as rest; steps after it ends are holds."""

    def __init__(self, layouts, actions, lengths, horizon, hist, device):
        B = len(actions)
        zeros = lambda n: np.zeros((B, n, 2))
        pos = layouts.start[:, None] + np.concatenate([zeros(1), np.cumsum(actions, 1)], 1)
        padded = np.concatenate([zeros(hist), actions, zeros(horizon)], 1)
        f32 = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.pos, self.actions, self.targets = f32(pos), f32(padded), f32(layouts.targets)
        self.mask = torch.as_tensor(layouts.mask, device=device)
        self.goal = torch.as_tensor(layouts.goal_mask(), device=device)
        self.obstacles = f32(layouts.obstacles)
        self.obstacle_mask = torch.as_tensor(layouts.obstacle_mask, device=device)
        self.lengths = torch.as_tensor(lengths, device=device)
        self.valid = torch.nonzero(self.lengths > 0).squeeze(1)   # demos that exist (length 0: none was found)
        self.window = torch.arange(hist + horizon, device=device)
        self.hist = hist

    def sample(self, n, goal_dropout=0.0, indexed=False):
        """goal_dropout: fraction of samples whose goal flag is hidden (all targets 0). With `indexed`
        also the demo i and its step k of every sample (to look up what the expert pushed there)."""
        i = self.valid[torch.randint(len(self.valid), (n,), device=self.lengths.device)]
        k = (torch.rand(n, device=i.device) * self.lengths[i]).long()
        seq = self.actions[i[:, None], k[:, None] + self.window]
        goal = self.goal[i] & (torch.rand(n, 1, device=i.device) >= goal_dropout)
        obs = dict(pos=self.pos[i, k], hist=seq[:, :self.hist], targets=self.targets[i],
                   mask=self.mask[i], goal=goal, obstacles=self.obstacles[i], obstacle_mask=self.obstacle_mask[i])
        return (obs, seq[:, self.hist:], i, k) if indexed else (obs, seq[:, self.hist:])
