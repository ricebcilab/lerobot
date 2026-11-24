"""The experiment mechanism: a protocol run by a participant, human or simulated.

    protocol  arms (control configs), trials and practice trials per block,
              block order rule, trial timing
    run       one participant under one protocol: blocks in a counterbalanced
              order, the same target layouts in every arm, one jsonl per block
              and a manifest
    visit     a participant's visit over several tasks: one run per task
              section, in a counterbalanced task order
    calibration  the reaches a participant's own user module is trained on,
              before a visit: the cursor plays demonstrations by itself
              while the person pushes
    trial     agent -> readout -> controller -> env, stepped at 50 Hz, logged

`run_trials` steps a batch of trials in lockstep. A person runs one trial at a
time in real time with a renderer; the expert runs a whole block at once,
headless. Everything else is the same code and the same records.
"""

import json
import time
import zlib
from pathlib import Path

import numpy as np

from . import config
from lerobot.teleoperators.readout import Readout
from lerobot.envs.cursor import CursorEnv, Layouts, TaskConfig, sample_layouts
from .demos import make_demos
from .paths import RESULTS, ROOT, adapter_path, policy_path, policy_weights
from .policy import load_policy
from .records import (append_record, arm_fields, block_filename, environment, file_sha256, git_commit, now_iso,
                      outcome_of, read_manifest, read_records, rounded, trial_record, write_manifest)
from .steering import SharedController, view_scene_fn
from lerobot.policies.steering import AdapterModules


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


def block_order(arm_names, participant, order="latin_square", session=1, shift=0):
    """The arms in the order a visit plays them: a row of the balanced Latin square chosen by the
    participant number and advanced by one row per visit, so the same person's visits are
    counterbalanced too (p1 -> row 1, then 2, ...; sim -> row 0). `shift` advances it further: each
    task section of a visit takes the next row."""
    if order == "listed":
        return list(arm_names)
    if order == "latin_square":
        rows = balanced_latin_square(len(arm_names))
        row = rows[(participant_index(participant) + session - 1 + shift) % len(rows)]
        return [arm_names[i] for i in row]
    raise ValueError(f"unknown block order {order!r}; expected latin_square or listed")


def task_order(tasks, participant, order="latin_square", session=1):
    """The tasks in the order a visit plays them: the first as listed (the open field, where every visit
    starts), the others in a row of their own balanced Latin square, chosen as the block order's row is."""
    tasks = list(tasks)
    return tasks[:1] + block_order(tasks[1:], participant, order, session) if len(tasks) > 2 else tasks


def layout_seed(participant, session, section=None):
    """One seed per visit: the same layouts in every arm of the visit, new ones next visit. A task section of
    a visit (`section`, its task's name) takes its own."""
    return zlib.crc32((f"{participant}:{session}" + ("" if section is None else f":{section}")).encode())


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
    """One policy per model name (a `CursorFlowPolicy`, `load_policy`) and one adapter checkpoint per name the arms
    reference."""
    policies = {name: load_policy(policy_path(name), device) for name in {a["model"] for a in arms.values()}}
    adapters = {name: AdapterModules.load(adapter_path(name), device)
                for name in {n for a in arms.values() for n in config.adapter_names(a)}}
    return policies, adapters


def adapter_checkpoints(arms):
    """'adapters/<name>' -> file hash for every adapter checkpoint the arms reference."""
    names = dict.fromkeys(n for a in arms.values() for n in config.adapter_names(a))  # each checkpoint hashed once
    return {f"adapters/{n}": file_sha256(adapter_path(n)) for n in names}


def arm_adapter(arm, adapters):
    """The arm's reversal adapter: one loaded checkpoint by name, a user module and a task module from
    their checkpoints composed, or MARS's fixed matrix F. Served in every mode: a task module shapes
    the chunks of every mode (and the expert_chunk agent's intent), a user module acts only in the
    reversals of frs."""
    names = arm["learned_adapter"]
    if isinstance(names, str):
        return adapters[names]
    if isinstance(names, dict):
        return AdapterModules(user=adapters[names["user"]].user if names["user"] else None,
                              task=adapters[names["task"]].task if names["task"] else None)
    return None if arm["adapter_matrix"] is None else AdapterModules(matrix=arm["adapter_matrix"])


def readout_for(arm, rng=None, input_noise=None, off=False):
    """The arm's readout: its corruption (a matrix or a curl) and noise (`input_noise` overrides
    the arm's, e.g. one std per trial); with `off` a clean readout, nothing corrupted, no noise."""
    if off:
        return Readout(None, 0.0, rng)
    noise = arm["input_noise"] if input_noise is None else input_noise
    return Readout(arm["corruption_matrix"], noise, rng, curl=arm["corruption_curl"])


class Playback:
    """The controller of a calibration reach: the cursor moves by the recorded steps `actions` [B, L, 2] on its own
    and holds after they end, whatever the command. What `run_trials` and the window ask of a controller."""

    mode, chunks = "playback", 0

    def __init__(self, actions):
        self.actions, self.t = np.asarray(actions, dtype=float), 0
        self.steered_chunks = np.zeros(len(self.actions), dtype=int)

    def act(self, command, reference=None):
        step = self.actions[:, self.t] if self.t < self.actions.shape[1] else np.zeros_like(self.actions[:, 0])
        self.t += 1
        return step

    def record(self, displacement):
        pass

    def planned(self):
        return [()] * len(self.actions)


def run_trials(arm, layouts, agent, readout, policy, adapters, device, renderer=None, hud=None, scene_encoder=None,
               playback=None):
    """Step len(layouts) trials in lockstep until all are done; return one record per trial.

    `policy` (a `CursorFlowPolicy`, `load_checkpoints`', or a bare `FlowPolicy`) goes to the controller
    (`SharedController`), which takes every chunk from lerobot's `SteeringPolicy` over it. With a renderer
    the trials run in real time (one at a time: len(layouts) must be 1), with a ready gate before and
    feedback after. Raises Aborted if the
    person quits. When the arm's task module reads vlm_views, `scene_encoder` (an
    encoder, or a zero-argument factory of one, called once) is required: the
    running trials' two views are encoded at every observation (`view_scene_fn`).
    With `playback` (steps [B, L, 2], a calibration's demonstrations) the cursor plays those
    steps by itself instead of a controller's (`Playback`): the commands are read and
    logged as in any trial, nothing steers, and no feedback follows.
    """
    task = arm_task(arm)
    B = len(layouts)
    if renderer is not None and B != 1:
        raise ValueError("real-time runs step one trial at a time")
    adapter = arm_adapter(arm, adapters)
    module = None if adapter is None else adapter.task
    scene_fn = None
    if getattr(module, "token_source", None) == "vlm_views":
        if scene_encoder is None:
            raise ValueError("the arm's task module reads vlm_views, made at every observation: run_trials needs a "
                             "scene_encoder (an encoder, or a factory of one)")
        scene_fn = view_scene_fn(module, scene_encoder if hasattr(scene_encoder, "encode") else scene_encoder())
    env = CursorEnv(layouts, task)
    ctl = Playback(playback) if playback is not None else SharedController(
        policy, layouts, arm["mode"], arm["tau"], arm["n_reverse_steps"], arm["flow_steps"], arm["execute"], task.dt,
        speed_gain=arm["speed_gain"], adapter=adapter, device=device, gain_mode=arm["gain_mode"], ic_gain=arm["ic_gain"],
        lam=arm["lambda"], pinned_steps=arm["pinned_steps"], scene_fn=scene_fn)
    readout.reset(B)
    agent.reset(B, readout)
    ctx = dict(env=env, ctl=ctl, layouts=layouts, task=task, agent=agent, served=np.zeros((B, 2)), hud=hud or {})
    if renderer is not None:
        renderer.ready_phase(ctx)
    t_start = now_iso()
    tic = time.perf_counter()
    path, raw_log, served_log, frame_dt = [env.pos.copy()], [], [], []
    aborted = False
    try:
        while not env.done.all():
            ctl.active = ~env.done  # the trials still running, the ones a scene provider renders
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
    if renderer is not None and not aborted and playback is None:
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
    """One participant's run of one task under a protocol: `arms` ({arm name: arm}) on the same layouts, a
    block each. `start()` writes the manifest and finds where to resume; `run()` plays the blocks.

    `section` makes the run one task section of a `Visit`: {task, index, first_block, n_blocks, last, ...}
    names its task (in the run's folder, its layout seed and every record), takes the block order's next
    row per section (`index`), and counts the blocks over the visit; `labels` go into its records and
    manifest too."""

    calibration = False  # a `Calibration`'s trials are reaches the cursor plays by itself

    def __init__(self, protocol, participant, session, agent, device="cpu", results_dir=RESULTS, resume=False,
                 run_name=None, *, arms, section=None, labels=None):
        self.protocol, self.participant, self.session, self.agent = protocol, participant, session, agent
        self.device, self.resume, self.section = device, resume, section
        task = section["task"] if section else None
        self.labels = {"task": task, **(labels or {})} if section else {}
        self.arms = arms
        self.order = block_order(list(self.arms), participant, protocol["blocks"]["order"], session,
                                 section["index"] if section else 0)
        self.run_name = run_name or f"{protocol['name']}_s{session}" + (f"_{task}" if section else "")
        self.dir = Path(results_dir) / participant / self.run_name
        self.seed = layout_seed(participant, session, task)
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
        models = dict.fromkeys(a["model"] for a in self.arms.values())  # each policy hashed once
        checkpoints = {f"policies/{m}": file_sha256(policy_weights(m)) for m in models}
        checkpoints.update(adapter_checkpoints(self.arms))
        section = {} if self.section is None else {"section": {**self.section, **self.labels}}
        return dict(participant=self.participant, session=self.session, run=self.run_name, **section,
                    protocol=self.protocol, arms={k: arm_fields(v) for k, v in self.arms.items()},
                    block_order=self.order, layout_seed=self.seed,
                    agent=dict(name=self.agent.name, **self.agent.params()), checkpoints=checkpoints,
                    git_commit=git_commit(ROOT), environment=environment(), started=now_iso(), finished=None)

    def check(self):
        """A run that has trials is continued only on --resume."""
        if self.dir.exists() and any(self.dir.glob("block*.jsonl")) and not self.resume:
            raise FileExistsError(f"{self.dir} already has trials; pass --resume to continue it, "
                                  f"or --session {self.session + 1} for a new visit")

    def start(self):
        self.check()
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.resume and (self.dir / "manifest.json").exists():
            old = read_manifest(self.dir)
            if old["block_order"] != self.order or old["layout_seed"] != self.seed:
                raise ValueError(f"{self.dir}: the existing run has a different order or layout seed")
            if old["arms"] != json.loads(json.dumps(self.manifest()["arms"])):
                raise ValueError(f"{self.dir}: the existing run has different arms (their settings changed since it began)")
            if old.get("finished") and not self.remaining():
                return  # a complete run keeps its manifest; nothing is left to play
        write_manifest(self.dir, self.manifest())
        self.policies, self.adapters = load_checkpoints(self.arms, self.device)

    def done_trials(self, block, arm):
        path = self.dir / block_filename(block, arm)
        return [r for r in read_records(path)] if path.exists() else []

    def remaining(self):
        """The trials still to play, over every block."""
        return sum(len(self.block_plan(block)) - len(self.done_trials(block, arm)) for block, arm in enumerate(self.order, 1))

    def block_plan(self, block):
        """(practice flag, trial index, layout, layout id) for every trial of a block, in order."""
        n_p, n_t = len(self.practice), len(self.layouts)
        plan = [(True, i + 1, self.practice[[j]], int(j))
                for i, j in enumerate(block_permutation(self.seed, block, n_p, practice=True))]
        plan += [(False, i + 1, self.layouts[[j]], int(j)) for i, j in enumerate(block_permutation(self.seed, block, n_t))]
        return plan

    def playback(self, rows):
        """The steps the cursor plays by itself in these trials of a block's plan, or None: a run's trials are steered."""
        return None

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
                readout = readout_for(arm, np.random.default_rng([self.seed, block]))
                before, n_blocks = (self.section["first_block"], self.section["n_blocks"]) if self.section else (0, len(self.order))
                hud = dict(block=before + block, n_blocks=n_blocks, arm=arm_name,  # counted over the visit
                           calibration=self.calibration)
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
                        recs = run_trials(arm, Layouts.concat([r[2] for r in rows]), self.agent, readout,
                                          self.policies[arm["model"]], self.adapters, self.device, renderer, trial_hud,
                                          playback=self.playback(rows))
                    except Aborted as e:
                        self.append(log, block, arm_name, rows, e.args[0])
                        raise
                    self.append(log, block, arm_name, rows, recs)
                    played += len(rows)
                if renderer is not None and not self.calibration:  # a calibration ends in its module's training
                    end = block == len(self.order)  # of this run; the visit ends with its last section
                    last = end and (self.section is None or self.section["last"])
                    renderer.break_screen(dict(**hud, last=last, section_end=end and not last), self.agent)
        finally:
            m = read_manifest(self.dir)
            m["finished"] = now_iso()
            write_manifest(self.dir, m)
        return played

    def append(self, log, block, arm_name, rows, recs):
        for (practice, trial, _, layout_id), rec in zip(rows, recs):
            append_record(log, trial_record(rec, self.arms[arm_name], self.participant, self.run_name,
                                            self.protocol["name"], block, arm_name, trial, practice, layout_id,
                                            self.agent.name, self.labels))


class Calibration(Run):
    """The reaches a participant's own user module is trained on (--calibrate), before a visit's first section: on
    the open field, under the user's readout, the cursor plays one demonstration per layout by itself (`make_demos`)
    while the participant pushes as if steering it, and every step keeps what the readout served. A run of one
    block in <protocol>_s<session>_calibration/ (`protocol.calibration`: its reaches and practice reaches), resumed
    as any run is.

    `sections` are the visit's ({task: {arm name: arm}}): the readout and the layouts' task are their first arm's,
    the policy the one their user module steers. `module` names the module trained on the calibration
    (checkpoints/adapters/participants/<study>/<participant>/<protocol>_s<session>), `with_module` puts it in the
    shared module's place and `data` is what its trainer reads."""

    calibration = True

    def __init__(self, protocol, sections, participant, session, agent, device="cpu", results_dir=RESULTS, resume=False,
                 labels=None):
        arms = [arm for section in sections.values() for arm in section.values()]
        models = {arm["model"] for arm in arms if self.user_module(arm)}
        if len(models) != 1:
            raise ValueError(f"protocol {protocol['name']}: a calibration trains the user module of one policy; its arms "
                             f"play a user module on {sorted(models) or 'none'}")
        arm = {**arms[0], "mode": "playback", "method": "calibration", "model": models.pop(), "obstacle": None,
               "reversal_adapter": None, "learned_adapter": None, "adapter_matrix": None}
        plan = protocol["calibration"]
        blocks = {"trials": plan["reaches"], "practice": plan["practice"], "order": "listed"}
        section = dict(task="calibration", index=0, first_block=0, n_blocks=1, last=False)
        super().__init__({**protocol, "blocks": blocks}, participant, session, agent, device, results_dir, resume,
                         arms={"calibration": arm}, section=section, labels=labels)
        self.module = f"participants/{protocol['study']}/{participant}/{protocol['name']}_s{session}"
        rng, task = np.random.default_rng([self.seed, 2]), arm_task(arm)
        self.demos = {practice: make_demos(rng, layouts, task)  # (steps [B, L, 2], lengths [B]) per layout set
                      for practice, layouts in ((False, self.layouts), (True, self.practice))}

    @staticmethod
    def user_module(arm):
        """The user module an arm plays by name, or None."""
        names = arm["learned_adapter"]
        return names.get("user") if isinstance(names, dict) else None

    @property
    def module_path(self):
        return adapter_path(self.module)

    @property
    def ready(self):
        """Whether the visit has its module: every reach played and the module trained on them saved."""
        return self.module_path.exists() and (self.dir / "manifest.json").exists() and not self.remaining()

    def playback(self, rows):
        return self.demos[rows[0][0]][0][[row[3] for row in rows]]

    def with_module(self, sections):
        """`sections` with the participant's module in the shared user module's place, in every arm that plays one."""
        def swap(arm):
            if not self.user_module(arm):
                return arm
            return {**arm, "reversal_adapter": {**arm["reversal_adapter"], "user": self.module},
                    "learned_adapter": {**arm["learned_adapter"], "user": self.module}}

        return {task: {name: swap(arm) for name, arm in arms.items()} for task, arms in sections.items()}

    def data(self):
        """The played calibration as its trainer reads it: (layouts, steps [B, L, 2], lengths [B], served [B, L, 2]),
        one row per finished reach: its layout, the demonstration the cursor played on it, and what the readout
        served at every one of its steps."""
        records = [r for r in self.done_trials(1, "calibration") if not r["practice"] and r["outcome"] == "hit"]
        ids = [r["layout_id"] for r in records]
        actions, lengths = self.demos[False]
        served = np.zeros((len(ids), actions.shape[1], 2))
        for row, (record, i) in enumerate(zip(records, ids)):
            if len(record["served_commands"]) != lengths[i]:
                raise ValueError(f"{self.dir}: reach {record['trial']} has {len(record['served_commands'])} commands for a "
                                 f"demonstration of {lengths[i]} steps (was the code changed since it was played?)")
            served[row, :lengths[i]] = record["served_commands"]
        return self.layouts[ids], actions[ids], lengths[ids], served


class Visit:
    """One participant's visit over several task sections: a `Run` per section, in the visit's task order
    (`task_order`), each with its own layouts, run folder and block order (the next row per section).

    `sections` maps a task's name to its arms {arm name: arm}, in the protocol's order; `order` is the task
    order rule (latin_square | listed); `labels` (e.g. the user) go into every record and manifest."""

    def __init__(self, protocol, sections, participant, session, agent, device="cpu", results_dir=RESULTS, resume=False,
                 order="latin_square", labels=None):
        names = list(sections)
        self.tasks = task_order(names, participant, order, session)
        n_blocks, first, self.runs = sum(len(arms) for arms in sections.values()), 0, []
        for k, task in enumerate(self.tasks):
            section = dict(task=task, index=names.index(task), position=k + 1, of=len(names), first_block=first,
                           n_blocks=n_blocks, last=k == len(names) - 1, task_order=self.tasks,
                           obstacles=any(arm["obstacle"] is not None for arm in sections[task].values()))
            self.runs.append(Run(protocol, participant, session, agent, device, results_dir, resume,
                                 arms=sections[task], section=section, labels=labels))
            first += len(sections[task])

    def check(self):
        """Every section's folder before anything is played (a calibration, which comes before `start`)."""
        for run in self.runs:
            run.check()

    def start(self):
        """Every section's manifest and checkpoints before the first screen, so an existing folder or a
        missing checkpoint stops the visit before it begins."""
        for run in self.runs:
            run.start()

    def run(self, renderer=None):
        """Play every section with trials left; returns the number of trials played."""
        played = 0
        for run in self.runs:
            if not run.remaining():
                continue  # a section a resumed visit had finished
            if renderer is not None and len(self.runs) > 1:  # a one-task visit needs no task screen
                renderer.section_screen({**run.section, **run.labels}, run.agent)
            played += run.run(renderer)
        return played
