#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Headless 2D cursor-reaching task with a variable number of targets.

Same rules as CursorControlTask in ../shared_autonomy_benchmark/task_env.py
(cursor += action + noise, clipped to the screen), but batched, in normalized
screen coordinates ([0, 1]^2), timed in steps rather than wall-clock seconds,
and with dwell selection: holding the cursor on any target for `dwell_steps`
selects that target and ends the trial.

Optionally the cursor has inertia and feels a curl force field (Shadmehr &
Mussa-Ivaldi 1994). The action then sets a commanded velocity u = action / dt,
and the cursor's velocity follows dv/dt = (u - v) / velocity_tau + curl * R v,
with R a 90-degree clockwise rotation. At steady state this turns the command
clockwise by atan(velocity_tau * curl) and scales the speed by
1 / sqrt(1 + (velocity_tau * curl)^2); the curl acts most while speeding up.

Optionally a task has obstacles: static discs placed relative to the line from
the cursor's start to the goal (`obstacle_spec`): one disc, a wall of discs with
a gap, or a ring of discs around the goal or the start with a gap facing away
from the reach. Touching one ends the trial (outcome `collision`).
`plan_route` routes around them; the simulated expert follows its route, and
the demonstrations follow the expert.
"""

import heapq
from dataclasses import dataclass

import numpy as np

OBSTACLE_KINDS = ("disc", "wall", "ring")
TARGET_CLEARANCE, START_CLEARANCE = 0.01, 0.03  # an obstacle keeps this far from a target's edge / the start
LANE = 0.1  # a disc leaves a lane this wide between itself and the screen edge on at least one side
SCREEN_EDGE = 0.01  # routes keep this far from the screen edge
ROUTE_CLEARANCE = 0.05  # a ring needs a route this clear of it (the expert's clearance)
AVOID_CLEARANCE = (
    0.01  # routes keep this far from the edge of a distractor target (crossing one briefly selects nothing)
)
RING_PATIENCE = 300  # starts a set of targets may refuse a ring before the targets are drawn again


@dataclass
class TaskConfig:
    dt: float = 0.02  # s per step (50 Hz, a typical BCI bin)
    target_radius: float = 0.05
    min_target_sep: float = 0.12  # center-to-center distance between targets
    min_reach: float = 0.2  # cursor start to goal, at least
    max_reach: float | None = None  # and at most (None: no limit)
    margin: float = 0.05  # targets and start stay this far from the screen edge
    dwell_steps: int = 15  # 0.3 s hold selects a target
    max_steps: int = 250  # 5 s timeout
    cursor_noise: float = 0.0  # std of Gaussian noise added to each cursor step
    velocity_tau: float = 0.0  # s; > 0 gives the cursor inertia (velocity follows the command)
    curl: float = 0.0  # 1/s; curl force field strength, + = clockwise (needs velocity_tau > 0)
    obstacle: dict | None = None  # a normalized obstacle spec (see obstacle_spec), or None


@dataclass
class Layouts:
    """A batch of trial layouts. Target rows where mask is False are padding;
    obstacle rows where obstacle_mask is False too."""

    start: np.ndarray  # [B, 2] cursor start
    targets: np.ndarray  # [B, N, 2] target centers
    mask: np.ndarray  # [B, N] True for real targets
    goal: np.ndarray  # [B] index of the goal target
    obstacles: np.ndarray | None = None  # [B, K, 3] disc center x, y and radius
    obstacle_mask: np.ndarray | None = None  # [B, K] True for real discs

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = np.zeros((len(self.start), 0, 3))
        if self.obstacle_mask is None:
            self.obstacle_mask = np.zeros(self.obstacles.shape[:2], dtype=bool)

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx):
        return Layouts(
            self.start[idx],
            self.targets[idx],
            self.mask[idx],
            self.goal[idx],
            self.obstacles[idx],
            self.obstacle_mask[idx],
        )

    @property
    def goal_pos(self):
        return self.targets[np.arange(len(self)), self.goal]

    def goal_mask(self):
        """[B, N] True at the goal target (the policy's goal flag)."""
        return np.arange(self.mask.shape[1])[None] == self.goal[:, None]

    def real_obstacles(self, b):
        """The discs of trial b, [k, 3]."""
        return self.obstacles[b, self.obstacle_mask[b]]

    def distractors(self, b, radius):
        """Trial b's non-goal targets as discs [m, 3]: a route around obstacles keeps clear of them,
        since lingering on one would select it."""
        pts = self.targets[b, self.mask[b] & ~self.goal_mask()[b]]
        return np.concatenate([pts, np.full((len(pts), 1), radius)], 1)

    def tile(self, k):
        """The whole batch k times over: trial i of copy j is row j * B + i."""
        return Layouts(
            np.tile(self.start, (k, 1)),
            np.tile(self.targets, (k, 1, 1)),
            np.tile(self.mask, (k, 1)),
            np.tile(self.goal, k),
            np.tile(self.obstacles, (k, 1, 1)),
            np.tile(self.obstacle_mask, (k, 1)),
        )

    @staticmethod
    def concat(parts):
        """One batch from several, padding the target and obstacle rows to the widest."""

        def pad(a, n):
            return np.concatenate([a, np.zeros((a.shape[0], n - a.shape[1]) + a.shape[2:], a.dtype)], 1)

        n = max(p.mask.shape[1] for p in parts)
        k = max(p.obstacle_mask.shape[1] for p in parts)
        return Layouts(
            np.concatenate([p.start for p in parts]),
            np.concatenate([pad(p.targets, n) for p in parts]),
            np.concatenate([pad(p.mask, n) for p in parts]),
            np.concatenate([p.goal for p in parts]),
            np.concatenate([pad(p.obstacles, k) for p in parts]),
            np.concatenate([pad(p.obstacle_mask, k) for p in parts]),
        )


# ---------------------------------------------------------------- obstacles


def obstacle_spec(spec, where="task.obstacle"):
    """A normalized obstacle spec, or None. Obstacles are placed relative to the reach (the line from the
    start to the goal): `along` is the fraction of the reach, `offset` the distance across it, `side`
    whether the side is drawn per trial (random) or always the left of the reach (fixed).

        {kind: disc, radius: r, along: a, offset: o, side: random}
        {kind: wall, along: a, gap: g, gap_offset: go, radius: 0.03, extent: 0.25, side: random}
        {kind: ring, around: goal, ring_radius: 0.14, radius: 0.02, gap: 0.16, gap_angle: 0, side: random}

    A wall is a row of touching discs across the reach, `extent` to each side, with a gap of width
    `gap` centered `gap_offset` across the line (on the drawn side). A ring is an arc of touching discs
    of radius `radius` on a circle of radius `ring_radius` around the goal or the start, open by `gap`
    (between the discs' edges) on the side facing away from the reach, behind the goal or behind the
    start, turned by `gap_angle` degrees toward the drawn side. A disc's `offset` and a ring's
    `gap_angle` may be a range [lo, hi], drawn per trial: an off-center disc or a turned gap makes one
    way round shorter, so the side to pass is part of the scene."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec.get("kind") not in OBSTACLE_KINDS:
        raise ValueError(f"{where}: kind must be one of {OBSTACLE_KINDS}, got {spec!r}")
    defaults = {
        "disc": {"radius": None, "along": None, "offset": 0.0, "side": "random"},
        "wall": {
            "along": None,
            "gap": None,
            "gap_offset": 0.0,
            "radius": 0.03,
            "extent": 0.25,
            "side": "random",
        },
        "ring": {
            "around": None,
            "ring_radius": 0.14,
            "radius": 0.02,
            "gap": 0.16,
            "gap_angle": 0.0,
            "side": "random",
        },
    }[spec["kind"]]
    unknown = set(spec) - set(defaults) - {"kind"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} for kind {spec['kind']}")
    out = {"kind": spec["kind"], **{k: spec.get(k, v) for k, v in defaults.items()}}
    for key, value in out.items():
        if value is None:
            raise ValueError(f"{where}: {key} is required for kind {spec['kind']}")
    if out["side"] not in ("random", "fixed"):
        raise ValueError(f"{where}: side must be random or fixed, got {out['side']!r}")
    if "around" in out and out["around"] not in ("goal", "start"):
        raise ValueError(f"{where}: around must be goal or start, got {out['around']!r}")
    if "along" in out and not 0 < out["along"] < 1:
        raise ValueError(f"{where}: along must be in (0, 1), got {out['along']!r}")
    for key in ("radius", "gap", "extent", "ring_radius"):
        if key in out and not out[key] > 0:
            raise ValueError(f"{where}: {key} must be positive, got {out[key]!r}")
    for key in ("offset", "gap_angle"):  # a number, or a range [lo, hi] drawn per trial
        if key in out:
            value = out[key]
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0) or (
                isinstance(value, (list, tuple)) and len(value) == 2 and 0 <= value[0] <= value[1]
            )
            if not ok:
                raise ValueError(f"{where}: {key} must be a number >= 0 or a range [lo, hi], got {value!r}")
            out[key] = [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)
    if out["kind"] == "ring" and not out["gap"] / 2 + out["radius"] < out["ring_radius"]:
        raise ValueError(f"{where}: a ring's gap/2 + radius must be below its ring_radius, got {out!r}")
    return {
        k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in out.items()
    }


def _draw(rng, value):
    """A spec value: the number itself, or a uniform draw from a range [lo, hi]."""
    return rng.uniform(*value) if isinstance(value, list) else value


def _ring_angles(spec):
    """Angles of a ring's discs from the axis of its gap: an arc from the gap's edge round to its other
    edge, discs at most 2 * radius apart (so they touch or overlap), the opening exactly `gap` wide."""
    R, r = spec["ring_radius"], spec["radius"]
    edge = np.arcsin(
        (spec["gap"] / 2 + r) / R
    )  # the discs flanking the gap are gap + 2r apart, center to center
    n = int(np.ceil(R * (2 * np.pi - 2 * edge) / (2 * r) - 1e-9)) + 1
    return np.linspace(edge, 2 * np.pi - edge, n)


def obstacle_count(spec):
    """Rows per layout the spec needs (its discs, dropped ones included)."""
    if spec is None:
        return 0
    if spec["kind"] == "disc":
        return 1
    if spec["kind"] == "ring":
        return len(_ring_angles(spec))
    return 2 * int(np.floor(spec["extent"] / (2 * spec["radius"]) + 1e-9)) + 1


def place_obstacles(rng, start, goal, spec):
    """The spec's discs for one reach: ([K, 3], mask [K], side). Dropped discs (a wall's gap) are masked out.
    A ring with no `gap_angle` draws no random number (it is symmetric about the reach)."""
    u = (goal - start) / np.linalg.norm(goal - start)
    n = np.array([-u[1], u[0]])
    r = spec["radius"]
    if spec["kind"] == "ring":
        center, away = (goal, u) if spec["around"] == "goal" else (start, -u)
        sign, turn = 1.0, 0.0
        if spec["gap_angle"]:
            sign = 1.0 if spec["side"] == "fixed" else (1.0 if rng.random() < 0.5 else -1.0)
            turn = sign * np.radians(_draw(rng, spec["gap_angle"]))
        ang = np.arctan2(away[1], away[0]) + turn + _ring_angles(spec)
        discs = np.concatenate(
            [
                center + spec["ring_radius"] * np.stack([np.cos(ang), np.sin(ang)], 1),
                np.full((len(ang), 1), r),
            ],
            1,
        )
        return discs, np.ones(len(ang), dtype=bool), sign
    sign = 1.0 if spec["side"] == "fixed" else (1.0 if rng.random() < 0.5 else -1.0)
    c0 = start + spec["along"] * np.linalg.norm(goal - start) * u
    if spec["kind"] == "disc":
        return np.array([[*(c0 + sign * _draw(rng, spec["offset"]) * n), r]]), np.ones(1, dtype=bool), sign
    k2 = (obstacle_count(spec) - 1) // 2
    across = np.arange(-k2, k2 + 1) * 2 * r
    keep = np.abs(across - sign * spec["gap_offset"]) >= spec["gap"] / 2 + r
    discs = np.concatenate([c0 + across[:, None] * n, np.full((len(across), 1), r)], 1)
    discs[~keep] = 0.0
    return discs, keep, sign


def _passable(obstacles, mask, start, pts, goal, sign, cfg):
    """Clear of the targets and the start, and passable: a disc leaves a lane on screen on at least
    one side of the reach; a wall's gap is on screen; a ring has a lane round it on screen and clear of
    the other targets on at least one side, and a route (`plan_route`, ROUTE_CLEARANCE)."""
    real, goal_pos = obstacles[mask], pts[goal]
    if not len(real):
        return True
    d_targets = np.linalg.norm(pts[None] - real[:, None, :2], axis=-1)
    d_start = np.linalg.norm(start - real[:, :2], axis=-1)
    clear = real[:, 2] + cfg.target_radius + TARGET_CLEARANCE
    if not ((d_targets >= clear[:, None]).all() and (d_start >= real[:, 2] + START_CLEARANCE).all()):
        return False
    spec = cfg.obstacle
    u = (goal_pos - start) / np.linalg.norm(goal_pos - start)
    n = np.array([-u[1], u[0]])
    on_screen = lambda p: bool(((p >= SCREEN_EDGE) & (p <= 1 - SCREEN_EDGE)).all())
    if spec["kind"] == "disc":
        c, r = real[0, :2], real[0, 2]
        return on_screen(c + (r + LANE) * n) or on_screen(c - (r + LANE) * n)
    if spec["kind"] == "wall":
        return on_screen(
            start + spec["along"] * np.linalg.norm(goal_pos - start) * u + sign * spec["gap_offset"] * n
        )
    # a ring, cheap first: a lane round it, from its gap to the far side, on screen and free on one side at least
    center = goal_pos if spec["around"] == "goal" else start
    away = (real[0, :2] + real[-1, :2]) / 2 - center  # toward the gap's mouth, between the arc's two ends
    away /= np.linalg.norm(away)
    across = np.array([-away[1], away[0]])
    rho = spec["ring_radius"] + spec["radius"] + ROUTE_CLEARANCE + SCREEN_EDGE
    phi = np.linspace(0.0, np.pi, 13)[:, None]
    others = np.delete(pts, goal, 0)
    free = lambda lane: (
        on_screen(lane)
        and bool(
            (
                np.linalg.norm(lane[:, None] - others[None], axis=-1) >= cfg.target_radius + AVOID_CLEARANCE
            ).all()
        )
    )
    if not any(
        free(center + rho * (np.cos(phi) * away + np.sin(phi) * side * across)) for side in (1.0, -1.0)
    ):
        return False
    avoid = np.concatenate([others, np.full((len(others), 1), cfg.target_radius)], 1)
    return plan_route(start, goal_pos, real, ROUTE_CLEARANCE, avoid=avoid) is not None


def segment_distance(p0, p1, c):
    """Distance from points c [..., 2] to the segments p0-p1 [..., 2] (broadcasting)."""
    p0, p1, c = np.asarray(p0, float), np.asarray(p1, float), np.asarray(c, float)
    d = p1 - p0
    dd = (d * d).sum(-1)
    tau = np.clip(((c - p0) * d).sum(-1) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    return np.linalg.norm(c - (p0 + tau[..., None] * d), axis=-1)


def plan_route(start, goal, obstacles, clearance=0.03, n_angles=12, hint=None, avoid=None):
    """Waypoints [W, 2] from the start to the goal around the discs `obstacles` [K, 3], the goal last, or
    None when there is no way through; the straight line when there are no discs. A visibility graph
    over the discs inflated by `clearance` (nodes on each inflated circle, laid out from the reach's
    direction, so a disc centered on the reach gives two mirror-image routes), shortest path by
    Dijkstra. `hint` (a previous first waypoint) penalizes first legs on the other side of the reach
    line, so a route does not flip sides between re-plans. `avoid` [M, 3] are further discs (the distractor targets) a route keeps AVOID_CLEARANCE clear
    of once it has to detour at all. A start (or goal) already inside an inflated disc ignores that disc."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    obstacles = np.asarray(obstacles, float).reshape(-1, 3)
    if not len(obstacles):
        return np.array([goal])
    R = obstacles[:, 2] + clearance
    if avoid is not None and len(avoid):
        avoid = np.asarray(avoid, float).reshape(-1, 3)
        obstacles = np.concatenate([obstacles, avoid])
        R = np.concatenate([R, avoid[:, 2] + min(clearance, AVOID_CLEARANCE)])
    c = obstacles[:, :2]
    u = goal - start
    angles = np.arctan2(u[1], u[0]) + np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    ring = c[:, None] + (1.06 * R)[:, None, None] * np.stack([np.cos(angles), np.sin(angles)], -1)[None]
    nodes = np.concatenate([start[None], goal[None], ring.reshape(-1, 2)])
    inside = np.linalg.norm(nodes[:, None] - c[None], axis=-1) < R[None]  # [N, K] node inside inflated disc k
    keep = np.ones(len(nodes), dtype=bool)
    keep[2:] = ~inside[2:].any(1) & ((nodes[2:] >= SCREEN_EDGE) & (nodes[2:] <= 1 - SCREEN_EDGE)).all(
        1
    )  # on screen, outside other discs
    nodes, inside = nodes[keep], inside[keep]  # the start and the goal stay nodes 0 and 1
    # an edge is valid when it clears every inflated disc, except discs that already contain its endpoint
    p0, d = nodes[:, None], nodes[None] - nodes[:, None]  # [N, 1, 2], [N, N, 2]
    rel = c[None, None] - p0[..., None, :]  # [N, 1, K, 2]
    dd = np.maximum((d * d).sum(-1), 1e-12)[..., None]  # [N, N, 1]
    tau = np.clip((rel * d[..., None, :]).sum(-1) / dd, 0.0, 1.0)  # [N, N, K]
    dist2 = ((rel - tau[..., None] * d[..., None, :]) ** 2).sum(-1)  # [N, N, K]
    excused = inside[:, None] | inside[None]  # [N, N, K]
    valid = ((dist2 >= R**2) | excused).all(-1)
    np.fill_diagonal(valid, False)
    weight = np.linalg.norm(d, axis=-1)
    side = u[0] * (nodes[:, 1] - start[1]) - u[1] * (nodes[:, 0] - start[0])  # > 0: left of the reach
    side = np.where(np.abs(side) > 1e-9, np.sign(side), 0.0)
    if hint is not None:
        hint_side = np.sign(u[0] * (hint[1] - start[1]) - u[1] * (hint[0] - start[0]))
        penalty = (side != 0) & (hint_side != 0) & (side != hint_side)
        weight[0, penalty] += 0.25
    best, prev, done = np.full(len(nodes), np.inf), np.full(len(nodes), -1), np.zeros(len(nodes), bool)
    best[0], heap = 0.0, [(0.0, 0)]
    while heap:
        d0, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = True
        if i == 1:
            break
        for j in np.flatnonzero(valid[i]):
            if d0 + weight[i, j] < best[j]:
                best[j], prev[j] = d0 + weight[i, j], i
                heapq.heappush(heap, (best[j], j))
    if not np.isfinite(best[1]):
        return None
    path, i = [], 1
    while i != 0:
        path.append(nodes[i])
        i = prev[i]
    return np.array(path[::-1])


def route_side(route, start, goal, tol=1e-6):
    """Which side of the reach a route passes on: "left", "right", or "straight" when its waypoints keep to
    the line (no detour); by the sign of the waypoint farthest from the line."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    u = (goal - start) / np.linalg.norm(goal - start)
    lateral = (np.asarray(route, float) - start) @ np.array([-u[1], u[0]])
    far = lateral[np.abs(lateral).argmax()]
    return "straight" if abs(far) <= tol else ("left" if far > 0 else "right")


def densify(points, spacing):
    """Points every `spacing` (at most) along the polyline `points` [W, 2], its first point left out and
    every later vertex kept."""
    out = []
    for p0, p1 in zip(points[:-1], points[1:]):
        k = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / spacing)))
        out.append(p0 + (np.arange(1, k + 1) / k)[:, None] * (p1 - p0))
    return np.concatenate(out) if out else points[1:]


# ---------------------------------------------------------------- layouts


def _ring_crowded(pts, goal, cfg):
    """True when a ring around the goal could not clear the other targets, whatever the start."""
    spec = cfg.obstacle
    if spec is None or spec["kind"] != "ring" or spec["around"] != "goal":
        return False
    reach = spec["ring_radius"] + spec["radius"] + cfg.target_radius + TARGET_CLEARANCE
    return bool((np.linalg.norm(np.delete(pts, goal, 0) - pts[goal], axis=1) < reach).any())


def _layout(rng, n, goal, cfg, max_tries=20_000, rounds=20):
    """n targets at least min_target_sep apart, a start clear of all of them with the goal within
    reach, and the task's obstacles clear of the targets and the start. Rejection sampling: up to
    max_tries starts per set of targets, and a set that admits none is redrawn, up to `rounds`
    times. Raises when the constraints cannot be met (too many targets for the spacing, an
    impossible reach, an obstacle that cannot fit). Without obstacles no extra random numbers are
    drawn, so layouts reproduce those of a task without them."""
    k, reach_ok = obstacle_count(cfg.obstacle), 0
    for _ in range(rounds):
        pts = np.empty((0, 2))
        for _ in range(max_tries):
            p = rng.uniform(cfg.margin, 1 - cfg.margin, 2)
            if np.all(np.linalg.norm(pts - p, axis=1) >= cfg.min_target_sep):
                pts = np.vstack([pts, p])
                if len(pts) == n:
                    break
        else:
            raise ValueError(f"cannot place {n} targets at least {cfg.min_target_sep} apart")
        if _ring_crowded(pts, goal, cfg):
            reach_ok += 1
            continue
        obstacles, mask = np.zeros((k, 3)), np.zeros(k, dtype=bool)
        refused = 0
        for _ in range(max_tries):
            s = rng.uniform(cfg.margin, 1 - cfg.margin, 2)
            d = np.linalg.norm(pts - s, axis=1)
            if not (
                cfg.min_reach <= d[goal] <= (np.inf if cfg.max_reach is None else cfg.max_reach)
                and d.min() >= 2 * cfg.target_radius
            ):
                continue
            reach_ok += 1
            if cfg.obstacle is not None:
                obstacles, mask, sign = place_obstacles(rng, s, pts[goal], cfg.obstacle)
                if not _passable(obstacles, mask, s, pts, goal, sign, cfg):
                    refused += 1
                    if refused >= RING_PATIENCE and cfg.obstacle["kind"] == "ring":
                        break  # these targets hardly ever leave a way round a ring: draw new ones
                    continue
            return s, pts, obstacles, mask
    if cfg.obstacle is not None and reach_ok:
        raise ValueError(
            f"cannot place obstacles {cfg.obstacle} clear of the targets and the start among {n} targets"
        )
    raise ValueError(f"cannot place a start {cfg.min_reach}-{cfg.max_reach} from the goal among {n} targets")


def center_out(n=8, radius=0.32):
    """n trials: the cursor at the center, n targets on a ring, each one the goal once."""
    angles = np.radians(90 - 360 / n * np.arange(n))
    targets = 0.5 + radius * np.stack([np.cos(angles), np.sin(angles)], 1)
    return Layouts(np.full((n, 2), 0.5), np.repeat(targets[None], n, 0), np.ones((n, n), bool), np.arange(n))


def curl_discretization(tau, curl, dt):
    """(decay, drive): the exact one-step update v <- decay v + drive u of dv/dt = (u - v) / tau + curl * R v,
    R a 90-degree clockwise rotation, for a command u held over the step."""
    eye, rot = np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]])
    decay = np.exp(-dt / tau) * (np.cos(curl * dt) * eye + np.sin(curl * dt) * rot)
    m_inv = (-eye / tau - curl * rot) / (1 / tau**2 + curl**2)
    return decay, m_inv @ (decay - eye) / tau


def curl_steady_state(tau, curl):
    """The linear map from commanded to steady-state cursor velocity under a curl field:
    a clockwise turn by atan(tau * curl) and a slowdown by 1 / sqrt(1 + (tau * curl)^2)."""
    rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return (np.eye(2) + tau * curl * rot) / (1 + (tau * curl) ** 2)


def sample_layouts(rng, n_trials, n_targets, cfg=TaskConfig()):
    """Random start, targets, goal (and the task's obstacles) for each trial.

    n_targets is an int, or an inclusive (lo, hi) range drawn per trial.
    """
    lo, hi = (n_targets, n_targets) if np.isscalar(n_targets) else n_targets
    counts = rng.integers(lo, hi + 1, n_trials)
    goal = rng.integers(0, counts)
    k = obstacle_count(cfg.obstacle)
    start = np.zeros((n_trials, 2))
    targets = np.zeros((n_trials, hi, 2))
    mask = np.zeros((n_trials, hi), dtype=bool)
    obstacles = np.zeros((n_trials, k, 3))
    obstacle_mask = np.zeros((n_trials, k), dtype=bool)
    for b in range(n_trials):
        start[b], targets[b, : counts[b]], obstacles[b], obstacle_mask[b] = _layout(
            rng, counts[b], goal[b], cfg
        )
        mask[b, : counts[b]] = True
    return Layouts(start, targets, mask, goal, obstacles, obstacle_mask)


# ---------------------------------------------------------------- environment


class CursorEnv:
    """Runs a batch of trials in lockstep; finished trials ignore further actions."""

    def __init__(self, layouts, cfg=TaskConfig(), rng=None):
        self.layouts, self.cfg = layouts, cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        B = len(layouts)
        self.pos = layouts.start.copy()
        self.t = 0
        self.on = np.full(B, -1)  # target under the cursor (-1: none)
        self.dwell = np.zeros(B, dtype=int)  # consecutive steps on that target
        self.done = np.zeros(B, dtype=bool)
        self.selected = np.full(B, -1)  # target selected by dwell (-1: none)
        self.collided = np.zeros(B, dtype=bool)  # the trial ended on an obstacle
        self.steps = np.full(B, cfg.max_steps)  # step at which each trial ended
        self.vel = np.zeros((B, 2))  # cursor velocity (screens/s), with inertia only
        if cfg.curl and not cfg.velocity_tau:
            raise ValueError("a curl force field needs velocity_tau > 0")
        if cfg.velocity_tau:
            self._decay, self._drive = curl_discretization(cfg.velocity_tau, cfg.curl, cfg.dt)

    @property
    def success(self):
        return self.selected == self.layouts.goal

    def step(self, action):
        cfg, live = self.cfg, ~self.done
        if cfg.cursor_noise:
            action = action + self.rng.normal(0, cfg.cursor_noise, action.shape)
        if cfg.velocity_tau:
            vel = self.vel @ self._decay.T + (action / cfg.dt) @ self._drive.T
            action = 0.5 * (self.vel + vel) * cfg.dt
            self.vel[live] = vel[live]
        prev = self.pos.copy()
        self.pos[live] = np.clip(self.pos[live] + action[live], 0, 1)
        self.t += 1

        hit = np.zeros(len(live), dtype=bool)
        if self.layouts.obstacles.shape[1]:  # the step's segment touches a disc
            dist = segment_distance(prev[:, None], self.pos[:, None], self.layouts.obstacles[..., :2])
            hit = live & ((dist < self.layouts.obstacles[..., 2]) & self.layouts.obstacle_mask).any(1)
            self.collided[hit] = True
            self.steps[hit] = self.t

        d = np.linalg.norm(self.layouts.targets - self.pos[:, None], axis=-1)
        d[~self.layouts.mask] = np.inf
        near = d.argmin(1)
        on = np.where(d[np.arange(len(d)), near] < cfg.target_radius, near, -1)
        self.dwell = np.where(on < 0, 0, np.where(on == self.on, self.dwell + 1, 1))
        self.on = on

        sel = live & ~hit & (self.dwell >= cfg.dwell_steps)
        self.selected[sel] = on[sel]
        self.steps[sel] = self.t
        self.done |= sel | hit | (self.t >= cfg.max_steps)
        return self.done
