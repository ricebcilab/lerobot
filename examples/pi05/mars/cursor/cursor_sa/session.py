"""The experiment mechanism: a protocol run by a participant, human or simulated.

    protocol  arms (control configs), trials and practice trials per block,
              block order rule, trial timing
    run       one participant under one protocol: blocks in a counterbalanced
              order, the same target layouts in every arm, one jsonl per block
              and a manifest
    trial     agent -> channel -> controller -> env, stepped at 50 Hz, logged

`run_trials` steps a batch of trials in lockstep. A person runs one trial at a
time in real time with a renderer; the expert runs a whole block at once,
headless. Everything else is the same code and the same records.
"""

import time
import zlib
from pathlib import Path

import numpy as np

from . import config
from .channel import InputChannel
from .env import CursorEnv, Layouts, TaskConfig, sample_layouts
from .paths import RESULTS, ROOT, adapter_path, policy_path, policy_weights
from .policy import FlowPolicy
from .records import (append_record, arm_fields, block_filename, environment, file_sha256, git_commit, now_iso,
                      outcome_of, read_manifest, read_records, rounded, trial_record, write_manifest)
from .steering import FixedAdapter, ReversalAdapter, SharedController


class Aborted(Exception):
    """The person ended the run (Esc or closing the window)."""


# ---------------------------------------------------------------- design


def participant_index(participant):
    """p7 -> 7, sim -> 0; used to pick the row of the Latin square."""
    digits = "".join(ch for ch in participant if ch.isdigit())
    return int(digits) if digits else 0


def balanced_latin_square(n):
    """Williams design: every arm in every position, each arm following each other
    arm equally often (for even n); rows for n participants."""
    first = [0]
    lo, hi = 1, n - 1
    while len(first) < n:
        first.append(hi)
        hi -= 1
        if len(first) < n:
            first.append(lo)
            lo += 1
    rows = [[(a + r) % n for a in first] for r in range(n)]
    if n % 2:  # odd n: add the reversed rows to balance the carry-over
        rows += [row[::-1] for row in rows]
    return rows


def block_order(arm_names, participant, order="latin_square", session=1):
    """The arms in the order a visit plays them: a row of the balanced Latin square chosen by the
    participant number and advanced by one row per visit, so the same person's visits are
    counterbalanced too (p1 -> row 1, then 2, ...; sim -> row 0)."""
    if order == "listed":
        return list(arm_names)
    if order == "latin_square":
        rows = balanced_latin_square(len(arm_names))
        row = rows[(participant_index(participant) + session - 1) % len(rows)]
        return [arm_names[i] for i in row]
    raise ValueError(f"unknown block order {order!r}; expected latin_square or listed")


def layout_seed(participant, session):
    """One seed per visit: the same layouts in every arm of the visit, new ones next visit."""
    return zlib.crc32(f"{participant}:{session}".encode())


def make_layout_sets(seed, n_trials, n_practice, targets, task):
    rng = np.random.default_rng(seed)
    return sample_layouts(rng, n_trials, targets, task), sample_layouts(rng, n_practice, targets, task)


def block_permutation(seed, block, n, practice=False):
    """The order in which block `block` (1-based) visits the layout set."""
    return np.random.default_rng([seed, block, int(practice)]).permutation(n)


# ---------------------------------------------------------------- trials


def arm_task(arm):
    """The arm's task: its geometry and timing (seconds in the arm, steps in the env)."""
    task = TaskConfig(target_radius=arm["target_radius"], min_target_sep=arm["min_target_sep"],
                      min_reach=arm["reach"][0], max_reach=arm["reach"][1],
                      velocity_tau=arm["velocity_tau"], curl=arm["curl"], obstacle=arm["obstacle"])
    task.dwell_steps, task.max_steps = round(arm["dwell_s"] / task.dt), round(arm["time_limit_s"] / task.dt)
    return task


def load_checkpoints(arms, device):
    """One policy per model name and one adapter checkpoint per name the arms reference."""
    policies = {name: FlowPolicy.load(policy_path(name), device) for name in {a["model"] for a in arms.values()}}
    adapters = {name: ReversalAdapter.load(adapter_path(name), device)
                for name in {n for a in arms.values() for n in config.adapter_names(a)}}
    return policies, adapters


def adapter_checkpoints(arms):
    """'adapters/<name>' -> file hash for every adapter checkpoint the arms reference."""
    return {f"adapters/{n}": file_sha256(adapter_path(n)) for a in arms.values() for n in config.adapter_names(a)}


def arm_adapter(arm, adapters):
    """The arm's reversal adapter: one loaded checkpoint by name, a user module and a task module from
    their checkpoints composed, or MARS's fixed matrix F. Served in every mode: a task module shapes
    the chunks of every mode (and the expert_chunk agent's intent), a user module acts only in the
    reversals of frs."""
    names = arm["learned_adapter"]
    if isinstance(names, str):
        return adapters[names]
    if isinstance(names, dict):
        return ReversalAdapter(user=adapters[names["user"]].user if names["user"] else None,
                               task=adapters[names["task"]].task if names["task"] else None)
    return None if arm["adapter_matrix"] is None else ReversalAdapter(user=FixedAdapter(arm["adapter_matrix"]))


def channel_for(arm, rng=None, input_noise=None, off=False):
    """The arm's input channel: its corruption (a matrix or a curl) and noise (`input_noise` overrides
    the arm's, e.g. one std per trial); with `off` a clean channel, nothing corrupted, no noise."""
    if off:
        return InputChannel(None, 0.0, rng)
    noise = arm["input_noise"] if input_noise is None else input_noise
    return InputChannel(arm["corruption_matrix"], noise, rng, curl=arm["corruption_curl"])


def run_trials(arm, layouts, agent, channel, policy, adapters, device, renderer=None, hud=None):
    """Step len(layouts) trials in lockstep until all are done; return one record per trial.

    With a renderer the trials run in real time (one at a time: len(layouts) must
    be 1), with a ready gate before and feedback after. Raises Aborted if the
    person quits.
    """
    task = arm_task(arm)
    B = len(layouts)
    if renderer is not None and B != 1:
        raise ValueError("real-time runs step one trial at a time")
    env = CursorEnv(layouts, task)
    ctl = SharedController(policy, layouts, arm["mode"], arm["tau"], arm["n_reverse_steps"], arm["flow_steps"],
                           arm["execute"], task.dt, speed_gain=arm["speed_gain"], adapter=arm_adapter(arm, adapters),
                           device=device, gain_mode=arm["gain_mode"], ic_gain=arm["ic_gain"], lam=arm["lambda"],
                           pinned_steps=arm["pinned_steps"])
    channel.reset(B)
    agent.reset(B, channel)
    ctx = dict(env=env, ctl=ctl, layouts=layouts, task=task, agent=agent, served=np.zeros((B, 2)), hud=hud or {})
    if renderer is not None:
        renderer.ready_phase(ctx)
    t_start = now_iso()
    tic = time.perf_counter()
    path, raw_log, served_log, frame_dt = [env.pos.copy()], [], [], []
    aborted = False
    try:
        while not env.done.all():
            raw, served, reference = agent.step(env.pos, layouts, ctl)
            prev = env.pos.copy()
            env.step(ctl.act(served, reference))
            ctl.record(env.pos - prev)
            path.append(env.pos.copy())
            raw_log.append(raw.copy())
            served_log.append(served.copy())
            ctx["served"] = served
            if renderer is not None:
                renderer.frame(ctx)
                now = time.perf_counter()
                frame_dt.append(now - tic)
                tic = now
    except Aborted:
        aborted = True
    t_end = now_iso()
    if renderer is not None and not aborted:
        renderer.feedback_phase(ctx)

    path, raw_log, served_log = np.stack(path, 1), np.stack(raw_log, 1), np.stack(served_log, 1)
    records = []
    for b in range(B):
        n = int(env.steps[b]) if env.done[b] else int(env.t)
        records.append(dict(
            t_start=t_start, t_end=t_end, duration_s=round(n * task.dt, 3),
            outcome="aborted" if aborted and not env.done[b] else outcome_of(env, b),
            selected=int(env.selected[b]), goal=int(layouts.goal[b]), n_targets=int(layouts.mask[b].sum()),
            targets=rounded(layouts.targets[b, layouts.mask[b]]), start=rounded(layouts.start[b]),
            obstacles=rounded(layouts.real_obstacles(b)),
            path=rounded(path[b, :n + 1]), raw_commands=rounded(raw_log[b, :n], 3),
            served_commands=rounded(served_log[b, :n], 3),
            frame_dt_ms=rounded(np.array(frame_dt) * 1000, 1) if renderer is not None else None,
            chunks=int(ctl.chunks), steered_chunks=int(ctl.steered_chunks[b])))
    if aborted:
        raise Aborted(records)
    return records


# ---------------------------------------------------------------- runs


class Run:
    """One participant's visit under a protocol. `start()` writes the manifest and
    finds where to resume; `run()` plays the blocks."""

    def __init__(self, protocol, participant, session, agent, device="cpu", results_dir=RESULTS, resume=False,
                 run_name=None):
        self.protocol, self.participant, self.session, self.agent = protocol, participant, session, agent
        self.device, self.resume = device, resume
        self.arms = {name: config.load_arm(file) for name, file in protocol["arms"].items()}
        self.order = block_order(list(self.arms), participant, protocol["blocks"]["order"], session)
        self.run_name = run_name or f"{protocol['name']}_s{session}"
        self.dir = Path(results_dir) / participant / self.run_name
        self.seed = layout_seed(participant, session)
        blocks = protocol["blocks"]
        first = next(iter(self.arms.values()))
        for name, arm in self.arms.items():  # every arm plays the same layouts, so they must agree on them
            if any(arm[k] != first[k] for k in config.LAYOUT_KEYS):
                raise ValueError(f"arms {next(iter(self.arms))} and {name} must have the same task layouts "
                                 f"({', '.join(config.LAYOUT_KEYS)})")
        self.layouts, self.practice = make_layout_sets(self.seed, blocks["trials"], blocks["practice"],
                                                       tuple(first["targets"]), arm_task(first))
        self.policies = self.adapters = None

    def manifest(self):
        checkpoints = {f"policies/{a['model']}": file_sha256(policy_weights(a["model"])) for a in self.arms.values()}
        checkpoints.update(adapter_checkpoints(self.arms))
        return dict(participant=self.participant, session=self.session, run=self.run_name,
                    protocol=self.protocol, arms={k: arm_fields(v) for k, v in self.arms.items()},
                    block_order=self.order, layout_seed=self.seed,
                    agent=dict(name=self.agent.name, **self.agent.params()), checkpoints=checkpoints,
                    git_commit=git_commit(ROOT), environment=environment(), started=now_iso(), finished=None)

    def start(self):
        if self.dir.exists() and any(self.dir.glob("block*.jsonl")) and not self.resume:
            raise FileExistsError(f"{self.dir} already has trials; pass --resume to continue it, "
                                  f"or --session {self.session + 1} for a new visit")
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.resume and (self.dir / "manifest.json").exists():
            old = read_manifest(self.dir)
            if old["block_order"] != self.order or old["layout_seed"] != self.seed:
                raise ValueError(f"{self.dir}: the existing run has a different order or layout seed")
        write_manifest(self.dir, self.manifest())
        self.policies, self.adapters = load_checkpoints(self.arms, self.device)

    def done_trials(self, block, arm):
        path = self.dir / block_filename(block, arm)
        return [r for r in read_records(path)] if path.exists() else []

    def block_plan(self, block):
        """(practice flag, trial index, layout, layout id) for every trial of a block, in order."""
        n_p, n_t = len(self.practice), len(self.layouts)
        plan = [(True, i + 1, self.practice[[j]], int(j))
                for i, j in enumerate(block_permutation(self.seed, block, n_p, practice=True))]
        plan += [(False, i + 1, self.layouts[[j]], int(j)) for i, j in enumerate(block_permutation(self.seed, block, n_t))]
        return plan

    def run(self, renderer=None):
        """Play every block; returns the number of trials played.

        A person plays one trial at a time in real time; headless, the practice
        trials and then the trials of a block run as one batch each.
        """
        played = 0
        try:
            for block, arm_name in enumerate(self.order, 1):
                arm = self.arms[arm_name]
                done = self.done_trials(block, arm_name)
                plan = self.block_plan(block)[len(done):]
                if not plan:
                    continue
                log = self.dir / block_filename(block, arm_name)
                channel = channel_for(arm, np.random.default_rng([self.seed, block]))
                hud = dict(block=block, n_blocks=len(self.order), arm=arm_name)
                if renderer is not None:
                    renderer.block_screen(dict(**hud, n_practice=len(self.practice), n_trials=len(self.layouts),
                                               resumed=len(done)), self.agent)
                groups = ([[p] for p in plan] if renderer is not None
                          else [[p for p in plan if p[0] == practice] for practice in (True, False)])
                for rows in filter(None, groups):
                    practice = rows[0][0]
                    trial_hud = dict(**hud, trial=rows[0][1], n_trials=len(self.practice if practice else self.layouts),
                                     practice=practice)
                    try:
                        recs = run_trials(arm, Layouts.concat([r[2] for r in rows]), self.agent, channel,
                                          self.policies[arm["model"]], self.adapters, self.device, renderer, trial_hud)
                    except Aborted as e:
                        self.append(log, block, arm_name, rows, e.args[0])
                        raise
                    self.append(log, block, arm_name, rows, recs)
                    played += len(rows)
                if renderer is not None:
                    renderer.break_screen(dict(**hud, last=block == len(self.order)), self.agent)
        finally:
            m = read_manifest(self.dir)
            m["finished"] = now_iso()
            write_manifest(self.dir, m)
        return played

    def append(self, log, block, arm_name, rows, recs):
        for (practice, trial, _, layout_id), rec in zip(rows, recs):
            append_record(log, trial_record(rec, self.arms[arm_name], self.participant, self.run_name,
                                            self.protocol["name"], block, arm_name, trial, practice, layout_id,
                                            self.agent.name))
