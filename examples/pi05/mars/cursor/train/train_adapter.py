"""Train one module of the reversal adapter: a user module for a user's command corruption, or a task
module for an obstacle task with a user's frozen user module in the loop.

    python train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' --name grid_20000demos/user_gain025y_expert
    python train/train_adapter.py --kind task --obstacle '{kind: ring, around: goal, pass: left}' \
        --user-module grid_20000demos/user_rot60_expert --corruption '{rotation_deg: 60}' --name grid_20000demos/task_ring_goal_with_rot60

USER MODULE: open-loop calibration with an expert that knows the goal. On demonstrations of the open-field
task, the label is the demonstration's next chunk L, and the command is what the expert
(`agents.ExpertAgent`) pushes at that moment toward the goal, through the user's readout (corruption M or
curl, then noise of std sigma), held over the chunk as the controller holds it: a_bci. The user module
U(x_t, t) acts on the backward pass; the field objective pulls it toward the label's latent z_L (L reversed
through the policy's flow), along the straight path from z_L to a_bci:

    U(x_t, t) P(x_t, t) = a_bci - z_L,   x_t = (1 - t) z_L + t a_bci

so a full reversal of a_bci lands on z_L, which regenerates L. Two knobs set the module's flexibility (UserModule):
`--step-rank r` builds every step's matrix from r matrices shared by the chunk, and `--cross-step-rank q` adds a
rank-q coupling between the chunk's steps (`--cross-step-dims 0` or `0,1` picks the dims it couples; default both);
the defaults (every step's matrix free, no coupling) are the grid's. They are saved in the module's config.

CLOSED-LOOP RECALIBRATION (ReFIT-style, `--closed-loop <initial module>`): expert demonstrations always head
for the goal, so the calibration above never shows the off-course states closed loop produces (and the module
trusts the policy's course there). The recalibration runs the initial module in closed loop on the open-field
task (the expert pushing through the user's readout at `--closed-loop-noise`, the grid's controller), keeps
every visited state with the expert's push there (what the user meant: toward the goal), relabels the state
with the reach the demonstrator would make from it (a straight min-jerk reach to the goal, its first chunk),
and trains a fresh module on batches mixing the open-loop data with these (`--closed-loop-mix`). Several
initial modules pool their trials (DAgger-style rounds: the first module's states and the recalibrated one's):

    python train/train_adapter.py --kind user --corruption '{scale: [1.0, 0.25]}' --name refit/user_gain025y_refit \
        --closed-loop grid_20000demos/user_gain025y_expert --closed-loop-depth 6

TASK MODULE: the policy is frozen and never saw obstacles. On demonstrations that go around the task's
obstacles (along the expert's path), the task module T = (A, b) is trained by flow matching through the
adapted field, on the policy's own interpolation x_t = (1 - t) z + t a,

    L_fm = || A(x_t, t, v, obstacles) v + b(...) - (a - z) ||^2 + lambda_id (||A - I||^2 + ||b||^2),   v = P(x_t, t | o)

with the policy's goal dropout, so the adapted field reproduces the detours with the goal given and
hidden alike; a share `p_free` of the layouts has no obstacle, where the module should be the identity.
With `--token-source` vlm_image / vlm_prefix / cnn the module reads the obstacles from the rendered scene
(`--render-size` pixels) instead of the disc parameters: pi0.5's vision tokens (`--vlm`) or a CNN trained with it;
`--token-geometry relative` (vlm_image, cnn) gives each scene token its offset from the cursor instead of a learned position.
TOKEN STORE: by default the layouts carry their tokens and the sampler holds them on the device (float16, 1 MB a
scene). `--scene-store DIR` (vlm_image, vlm_prefix) keeps them instead in a disk store under DIR
(`cursor_sa.dataset.SceneDataset`, one row per distinct layout, keyed by the layout, the render size and the encoder
`--vlm`, so the scenes a store holds are encoded once for every run that draws them), and the sampler gathers each
batch's rows from it to the device; the run is the one the resident tokens give. `--scene-rank r` (vlm_image,
vlm_prefix; default full width) fits a PCA token reduction to width r on REDUCTION_SAMPLE of the train split's scenes
(the moved copies among them), drawn with default_rng([seed, 13]) (centred unless `--scene-center no`), and the module
carries it (its `scene_mean`, `scene_basis`): the stored sampler feeds the reduced tokens, the resident one the full
ones, which the module reduces. The val split reuses the train split's reduction; a pooled run fits it on its first
entry's train split. The meta records `scene_store` (the store's folder name), `scene_rank` and `scene_center`.
`--jitter s` moves every training sample's position off the path by N(0, s^2 I) and has the first `--jitter-steps` steps of
its chunk take the move back out (a recovery label). `--rigid-motions k` adds up to k copies of every training demonstration
under random rigid motions of the screen (`augment_demos`), the expert's pushes moved with them; neither touches the val split.
The user's frozen module is in the loop through lambda_field x the field regression above through both
modules, on that user's commands around the obstacles (the layouts with the obstacle only):

    U(x_t, t) (A v + b) = a_bci - z_L,   z_L = L reversed through the task-adapted flow.

POOLED: `--pooled FILE` (YAML `entries: [{obstacle: <spec>, user_module: <name>, corruption: <spec>}, ...]`) trains
ONE task module over several contexts, each with its own demonstrations (`--n-demos` per entry) and its own frozen
user module in the lambda_field term; batches alternate between the entries. The log reports every number per entry
(suffixed _0, _1, ...) and, under the unsuffixed names, its mean over the entries. A single-context run is the
one-entry case.

TEACHER: `--teacher NAME` (task) swaps the scripted expert for a parametric task module's demonstrations: that
module's `oracle_task` rollouts (the gate's arm with a clean readout and the expert-chunk agent) on random layouts of
the task, their hits only. The rollout runs at half speed, two control steps per planned step, so its path is resampled
to the planned steps (`planned_steps`), the policy's own step rate, which is the rate a module trained on them acts at.
The student learns what the teacher's composed field does, with the teacher's own mistakes (non-hits) left out.
The kept set's obstacle-free share drifts a point or two above `--p-free` (free layouts hit more often, and the last
chunk is truncated from its tail, where `layouts_for` puts them).

VIEWS: `--token-source vlm_views` (task, single context, scripted expert) reads two camera views of the scene, each
through pi0.5's frozen vision model: the overhead view (the whole screen with the cursor's mark) and the wrist view (a
window of `--wrist-span` screens centred on the cursor), 256 tokens each. The demonstrations are drawn as for any
source; then every split gets a frame table (`view_frames`): a frame every `--view-stride` demo steps, its views drawn
at the demonstration's position at that step, and the sampler floors every sampled step to its frame (the step's
action chunk and the frame's tokens). `--view-affine p` moves a share p of the train split's stored pictures (each view
on its own) by a small rotation and shift, drawn once and frozen in the table (`move_pictures`); the val split's are
never moved. `--scene-rank r` (default 256; 0: full width) fits a PCA token reduction on `REDUCTION_SAMPLE` train
frames, stores the reduced tokens and carries the reduction in the module (its `scene_mean`, `scene_basis`);
`--scene-center no` fits it uncentred (the tokens' constant part kept: a zero `scene_mean` and the top directions of
the second moment). `--token-drop p` hides a share p of the scene tokens, drawn anew every batch, from the flow-matching loss only; the
field term and the validation see every token. The views pipeline draws from its own `default_rng([seed, 7])` and
leaves torch's generators as it found them (the encoder's load included), so the demonstrations, the frozen noise and
the module's initialisation are those of any other source at the seed. The checkpoint's meta records the flags and
every split's frames, moved pictures, variance kept and seconds (`views`). `--save-every N` (any task module, a
multiple of `--log-every`) also saves the module as it is at every N-th step as the module `<name>_step<N>`
(`meta["step"]`); the log's `train_fm` and `train_field` are the window's means of the flow-matching part and of the
field part (before its `--lambda-field` weight; None without the field term) of the training loss.

DATA: `--n-demos` demonstrations (one reach each) train a module; the default 20000 is as many as the base
policy's own. Every batch draws the readout's noise anew, so even a few demonstrations give endless noisy
commands; `--frozen-noise` draws it once per demonstration step instead, as a recorded calibration has one
command per step (the validation split keeps fresh draws).

Saves checkpoints/adapters/<name>/{adapter.pt, train_log.json}; adapter.pt holds kind, config,
state_dict, args and the obstacle spec (None for a user module and a pooled run, which has `pooled`: its entries). The trainer up to git commit 5c2167cd
also trained user modules from the policy's own chunks (grid 1 and the study's learned_adapter/gain025y_free) and task
modules without a user.
"""

import argparse
import json
import multiprocessing
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cursor_sa import config  # noqa: E402
from cursor_sa.agents import ExpertAgent, ExpertChunkAgent  # noqa: E402
from cursor_sa.readout import DEADBAND, Readout, corruption_spec  # noqa: E402
from cursor_sa.demos import (  # noqa: E402
    DemoConfig, DemoSampler, augment_demos, demo_positions, frame_steps, make_demos, plan_reach,
)
from lerobot.envs.cursor import (RENDER_PALETTE, SCENE_SIZE, Layouts, TaskConfig, obstacle_spec, render_views,  # noqa: E402
                           sample_layouts)
from cursor_sa.paths import adapter_dir, adapter_path, policy_path  # noqa: E402
from cursor_sa.policy import FlowPolicy  # noqa: E402
from cursor_sa.dataset import SceneDataset, encoder_digest, store_name  # noqa: E402
from cursor_sa.session import arm_task, readout_for, run_trials  # noqa: E402
from cursor_sa.steering import (ReversalAdapter, TaskModule, UserModule, forward_flow, hide_goal,  # noqa: E402
                                reverse_flow, to_action)

from lerobot.policies.steering import (SCENE_SOURCES, TOKEN_GEOMETRIES, SceneEncoder,  # noqa: E402
                                       heading_cosine, load_adapter_module, reduce_tokens, regenerate,
                                       save_adapter_module, scene_spec, task_field_loss, train_module,
                                       user_field_loss)

# ---------------------------------------------------------------- data


def layouts_for(args, obstacle, rng, n, p_free=0.0):
    """n layouts of the task (a share p_free of them without its obstacle)."""
    task = TaskConfig(min_reach=args.reach[0], max_reach=args.reach[1], obstacle=obstacle)
    n_free = int(round(p_free * n)) if obstacle is not None else 0
    parts = [sample_layouts(rng, n - n_free, tuple(args.targets), task)] if n - n_free else []
    if n_free:
        parts.append(sample_layouts(rng, n_free, tuple(args.targets), replace(task, obstacle=None)))
    return (Layouts.concat(parts) if len(parts) > 1 else parts[0]), task


def planned_steps(path):
    """The planned steps a `gain_mode: time, speed_gain: 0.5` rollout executed from its path (a position per control
    step): the gate's arm plays each planned step over two control steps, so every second position is a planned
    one. An odd final control step is half a planned step inside the hold and is dropped."""
    return np.diff(np.asarray(path)[::2], axis=0)


def teacher_demos(args, obstacle, policy, device, rng, n, p_free=0.0):
    """n hits of the teacher's `oracle_task` rollouts (`--teacher`) as demonstrations: random layouts of the task
    run `--teacher-chunk` at a time under the gate's arm (teleop, a clean readout, the expert-chunk agent) with the
    teacher in the reversal, the hits kept until n are. Returns the layouts, the task, the actions [n, L, 2] (the
    planned steps, zero after each demonstration's length), the lengths [n] and info: drawn, kept, outcomes (counts
    over everything drawn) and records (the kept trials')."""
    adapter = ReversalAdapter.load(adapter_path(args.teacher), device)
    taught = obstacle_spec(load_adapter_module(adapter_path(args.teacher))[2]["obstacle"])
    if taught != obstacle_spec(obstacle):
        raise ValueError(f"--teacher {args.teacher} was trained on the obstacle {taught}, not this run's "
                         f"{obstacle_spec(obstacle)}: a teacher's rollouts are only demonstrations of its own obstacle")
    if adapter.task is None or adapter.task.token_source != "discs":
        raise ValueError(f"--teacher must be a parametric (discs) task module: {args.teacher} has "
                         f"{'no task module' if adapter.task is None else 'token source ' + adapter.task.token_source}")
    arm = config.load_arm("condition_grid.yaml", dict(
        model=args.model or config.load_arm("base.yaml")["model"], mode="teleop", reversal_adapter={"user": None, "task": args.teacher},
        obstacle=obstacle, corruption=None, targets=list(args.targets), reach=list(args.reach)))
    assert (arm["mode"], arm["gain_mode"], arm["speed_gain"], arm["execute"]) == ("teleop", "time", 0.5, 2), (
        "the teacher's rollout must play each planned step over two control steps (teleop, gain_mode time, speed_gain 0.5, "
        f"execute 2): `planned_steps` resamples by that pacing; got {arm['mode']}, {arm['gain_mode']}, {arm['speed_gain']}, {arm['execute']}")
    assert arm_task(arm).obstacle == TaskConfig(obstacle=obstacle).obstacle, "the rollouts must run on the trainer's task"
    tic, parts, records, drawn, outcomes = time.time(), [], [], 0, {}
    while len(records) < n:
        if drawn >= 20 * n + args.teacher_chunk:
            raise RuntimeError(f"teacher {args.teacher}: {len(records)} hits in {drawn} rolled-out layouts (keep rate "
                               f"{len(records) / drawn:.1%}), fewer than the {n} wanted; the teacher does not solve this task")
        layouts, task = layouts_for(args, obstacle, rng, args.teacher_chunk, p_free)
        trials = run_trials(arm, layouts, ExpertChunkAgent(), readout_for(arm, rng, off=True), policy, {args.teacher: adapter}, device)
        drawn += len(layouts)
        for r in trials:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        hits = [i for i, r in enumerate(trials) if r["outcome"] == "hit"]
        if hits:
            parts.append(layouts[np.array(hits)])
            records += [trials[i] for i in hits]
    records = records[:n]
    layouts = (Layouts.concat(parts) if len(parts) > 1 else parts[0])[np.arange(n)]
    steps = [planned_steps(r["path"]) for r in records]
    lengths = np.array([len(s) for s in steps])
    actions = np.zeros((n, lengths.max(), 2))
    for i, s in enumerate(steps):
        actions[i, :len(s)] = s
    print(f"teacher {args.teacher}: kept {n} of {drawn} rollouts {outcomes} ({time.time() - tic:.0f}s)", flush=True)
    return layouts, task, actions, lengths, dict(drawn=drawn, kept=n, outcomes=outcomes, records=records)


def budget_demos(args, obstacle, rng, seconds, p_free=0.0, chunk=500):
    """The task's demonstrations that fit in `seconds` of demonstrating, each taking its duration up to the goal's
    selection (the reach and the hold): layouts drawn `chunk` at a time as `layouts_for` draws them (a share p_free
    without the obstacle), shuffled, and kept while their running total fits. Both trainers draw a time budget this
    way (`--demo-seconds`), as the first draw of their seed, so a task module and a fine-tuned policy given one budget
    and seed train on the same demonstrations. Returns the layouts, the task, the actions and the lengths."""
    parts, total = [], 0.0
    while total <= seconds:  # past the budget, so the cut falls inside what was drawn
        layouts, task = layouts_for(args, obstacle, rng, chunk, p_free)
        layouts = layouts[rng.permutation(chunk)]
        actions, lengths = make_demos(rng, layouts, task, allow_failed=True)
        parts.append((layouts, actions, lengths))
        total += lengths.sum() * task.dt
    width = max(a.shape[1] for _, a, _ in parts)
    actions = np.concatenate([np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0))) for _, a, _ in parts])
    lengths = np.concatenate([n for _, _, n in parts])
    layouts = Layouts.concat([l for l, _, _ in parts]) if len(parts) > 1 else parts[0][0]
    keep = int(np.searchsorted(np.cumsum(lengths) * task.dt, seconds, side="right"))
    return layouts[:keep], task, actions[:keep], lengths[:keep]


def expert_pushes(layouts, pos, rows_per_call=500):
    """What the expert (`agents.ExpertAgent`) pushes at every demo step [B, L, 2], watching the
    demonstration's positions pos [B, L + 1, 2]: toward the goal, around the task's obstacles."""
    B, L = pos.shape[0], pos.shape[1] - 1
    out = np.zeros((B, L, 2))
    for rows in np.array_split(np.arange(B), max(1, int(np.ceil(B / rows_per_call)))):
        sub, agent = layouts[rows], ExpertAgent()
        agent.reset(len(rows))
        for k in range(L):
            out[rows, k] = agent.command(pos[rows, k], sub)
    return out


VIEW_AFFINE_DEGREES = 5.0     # --view-affine: a moved picture turns by up to this many degrees about its centre
VIEW_AFFINE_TRANSLATE = 0.05  # and shifts by up to this share of its side per axis
REDUCTION_SAMPLE = 2000       # frames (vlm_views) or scenes (vlm_image, vlm_prefix) the token reduction is fitted on
REDUCTION_CHUNK = 64          # scenes per batch of the reduction's fit (its float64 sums take 8 bytes a value)
STORED_SOURCES = ("vlm_image", "vlm_prefix")   # one token array per layout: --scene-store, --scene-rank
VIEW_CHUNK = 256              # frames rendered and encoded at a time
VIEW_GPU_BYTES = 12 * 2**30   # a split's frame table above this stays in RAM


def move_pictures(images, rng, p, block=64):
    """Each uint8 picture [N, 3, S, S] independently, with probability p, turned about its centre by an angle in
    +-VIEW_AFFINE_DEGREES and shifted by up to VIEW_AFFINE_TRANSLATE of S per axis (one uniform draw each), resampled
    bilinearly with the uncovered border in the background's colour (not black: an obstacle is a dark disc). Every
    draw is made before any picture moves (`block` pictures are resampled at a time). Returns the pictures and the
    bool mask [N] of the moved ones."""
    n, _, s, _ = images.shape
    moved = rng.random(n) < p
    out = np.array(images, dtype=np.uint8)
    m = int(moved.sum())
    if not m:
        return out, moved
    angle = np.deg2rad(rng.uniform(-VIEW_AFFINE_DEGREES, VIEW_AFFINE_DEGREES, m)).astype(np.float32)
    shift = (rng.uniform(-VIEW_AFFINE_TRANSLATE, VIEW_AFFINE_TRANSLATE, (m, 2)) * s).astype(np.float32)   # (x, y) pixels
    c, rows = (s - 1) / 2, np.flatnonzero(moved)
    ys, xs = np.mgrid[:s, :s].astype(np.float32)
    bg = np.asarray(RENDER_PALETTE["bg"], np.float32)[None, :, None, None]
    for lo in range(0, m, block):
        sl = slice(lo, lo + block)
        cos, sin = np.cos(angle[sl])[:, None, None], np.sin(angle[sl])[:, None, None]
        dx, dy = xs - c - shift[sl, 0, None, None], ys - c - shift[sl, 1, None, None]
        # each output pixel's source: the inverse of q = R (p - c) + c + shift; beyond the picture lies the background
        sx, sy = np.clip(cos * dx + sin * dy + c, -1, s), np.clip(-sin * dx + cos * dy + c, -1, s)
        x0, y0 = np.clip(np.floor(sx), -1, s - 1), np.clip(np.floor(sy), -1, s - 1)
        fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
        padded = np.empty((len(sx), 3, s + 2, s + 2), np.float32)
        padded[:] = bg
        padded[:, :, 1:-1, 1:-1] = images[rows[sl]]
        b, x0, y0 = np.arange(len(sx))[:, None, None], x0.astype(np.int64) + 1, y0.astype(np.int64) + 1
        corner = lambda y, x: padded[b, :, y, x]                                               # [m, S, S, 3]
        value = ((1 - fy) * ((1 - fx) * corner(y0, x0) + fx * corner(y0, x0 + 1))
                 + fy * ((1 - fx) * corner(y0 + 1, x0) + fx * corner(y0 + 1, x0 + 1)))
        out[rows[sl]] = np.clip(np.rint(value), 0, 255).astype(np.uint8).transpose(0, 3, 1, 2)
    return out, moved


def view_chunks(pool, workers, layouts, pos, demo, step, chunks, wrist_span, size):
    """`render_views` of each chunk of frames (index arrays into demo, step), in order: in-process without a `pool`,
    else in the pool's `workers` spawned processes (which run the env's rendering only and never touch CUDA), at most
    two chunks per process in flight."""
    job = lambda rows: (layouts[demo[rows]], pos[demo[rows], step[rows]], wrist_span, size)
    if pool is None:
        for rows in chunks:
            yield render_views(*job(rows))
        return
    pending = deque()
    for rows in chunks:
        pending.append(pool.submit(render_views, *job(rows)))
        if len(pending) > 2 * workers:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def view_frames(args, layouts, actions, lengths, encoder, device, reduction=None, affine_rng=None):
    """The frame table of one split for --token-source vlm_views: a frame every --view-stride steps of every
    demonstration (`frame_steps`), its two views (`render_views`, overhead and wrist at --wrist-span) drawn at the
    demonstration's position at that step, rendered VIEW_CHUNK frames at a time (--render-workers processes, 0:
    in-process), with `affine_rng` every picture moved (`move_pictures` at --view-affine, the two views independently),
    encoded (`SceneEncoder.view_tokens`). With --scene-rank the tokens are reduced by `reduction` (mean, basis) or, when it is None,
    by a PCA (`SceneEncoder.fit_reduction`, streamed) of up to REDUCTION_SAMPLE frames chosen with `affine_rng` (or a
    default_rng([seed, 7])), whose pictures are rendered, moved and encoded for the fit first. Draws from no other
    generator, and leaves torch's CPU and CUDA generator states as it found them (the encoder runs inside `fork_rng`).
    Returns float16 tokens [F, 512, rank or 2048] (on `device` when at most VIEW_GPU_BYTES, else on the CPU), start
    (demo i's frames from row start[i]) and info: frames, moved (pictures in the table), kept (the
    variance share when fitted here, else None), reduction (the one used, or None) and seconds."""
    tic = time.time()
    demo, step, start = frame_steps(lengths, args.view_stride)
    pos = layouts.start[:, None] + np.concatenate([np.zeros((len(actions), 1, 2)), np.cumsum(actions, 1)], 1)
    spec = scene_spec("vlm_views", args.render_size, wrist_span=args.wrist_span)
    rng = affine_rng if affine_rng is not None else np.random.default_rng([args.seed, 7])
    workers = args.render_workers

    def encoded(frames, pool):
        """(rows, tokens on the encoder's device, pictures moved) for every chunk of `frames`, in order."""
        chunks = [frames[i:i + VIEW_CHUNK] for i in range(0, len(frames), VIEW_CHUNK)]
        for rows, views in zip(chunks, view_chunks(pool, workers, layouts, pos, demo, step, chunks, args.wrist_span,
                                                    args.render_size)):
            hits = 0
            if affine_rng is not None:
                flat, hit = move_pictures(views.reshape(-1, *views.shape[2:]), affine_rng, args.view_affine)
                views, hits = flat.reshape(views.shape), int(hit.sum())
            yield rows, SceneEncoder.view_tokens(views, encoder), hits

    # the encoder (its lazy first load builds a model, initialised from torch's generators) runs on forked streams
    with (torch.random.fork_rng(devices=range(torch.cuda.device_count())),
          ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) if workers else nullcontext() as pool):
        kept = None
        if args.scene_rank and reduction is None:
            sample = np.sort(rng.choice(len(demo), min(len(demo), REDUCTION_SAMPLE), replace=False))
            mean, basis, kept = SceneEncoder.fit_reduction((t for _, t, _ in encoded(sample, pool)), args.scene_rank,
                                                    center=args.scene_center == "yes")
            reduction = (mean, basis)
        shape = (len(demo), spec["scene_tokens"], args.scene_rank or spec["scene_dim"])
        on_device = int(np.prod(shape)) * 2 <= VIEW_GPU_BYTES
        tokens = torch.empty(shape, dtype=torch.float16, device=device if on_device else "cpu")
        moved = 0
        for rows, t, hits in encoded(np.arange(len(demo)), pool):
            if args.scene_rank:
                t = reduce_tokens(t, reduction[0].to(t.device), reduction[1].to(t.device))
            tokens[rows[0]:rows[-1] + 1] = t.to(tokens.device)
            moved += hits
    return tokens, start, dict(frames=len(demo), moved=moved, kept=kept, reduction=reduction if args.scene_rank else None,
                               seconds=time.time() - tic)


def scene_reduction(args, tokens_of, n, device):
    """--scene-rank's PCA token reduction (vlm_image, vlm_prefix) of a split of n scenes: fitted
    (`SceneEncoder.fit_reduction`, centred unless --scene-center no) on REDUCTION_SAMPLE of them (`SceneDataset.sample_rows`,
    default_rng([seed, 13]); all of them when fewer), `tokens_of(scenes)` giving their full-width tokens,
    REDUCTION_CHUNK at a time on `device`. Returns ((mean, basis), the share of the variance kept)."""
    pick = SceneDataset.sample_rows(np.arange(n), REDUCTION_SAMPLE, np.random.default_rng([args.seed, 13]))
    batches = (torch.as_tensor(tokens_of(pick[at:at + REDUCTION_CHUNK])).to(device)
               for at in range(0, len(pick), REDUCTION_CHUNK))
    mean, basis, kept = SceneEncoder.fit_reduction(batches, args.scene_rank, center=args.scene_center == "yes")
    return (mean, basis), kept


def token_store(args):
    """The --scene-store SceneDataset of this run's source, render size and encoder."""
    return SceneDataset(args.scene_store, args.token_source, args.render_size, encoder_digest(args.vlm), args.vlm)


class ExpertData:
    """One split of expert calibration data: the demonstrations (a DemoSampler; `frames` its frame table, `jitter` and
    `jitter_steps` its recovery labels, `store` and `reduction` its stored tokens, see `DemoSampler`) and, at every demo step, the expert's push (`push` [B, L, 2]
    when given, else `expert_pushes`) and what the user's readout makes of it before noise."""

    def __init__(self, layouts, actions, lengths, policy, device, corruption, curl, frozen_noise=False, frames=None, push=None,
                 jitter=0.0, jitter_steps=4, store=None, reduction=None):
        self.sampler = DemoSampler(layouts, actions, lengths, policy.horizon, policy.hist, device, frames, jitter=jitter,
                                   jitter_steps=jitter_steps, store=store, reduction=reduction)
        if push is None:
            push = expert_pushes(layouts, demo_positions(layouts, actions))
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.push, self.core = f32(push), f32(Readout(corruption, curl=curl).core(push))
        self.noise = torch.randn_like(self.core) if frozen_noise else None  # one draw per demo step, or fresh ones

    def noise_at(self, i, k):
        """The readout's noise draws at demo steps (i, k) [B, 2] when frozen, else None (fresh draws)."""
        return None if self.noise is None else self.noise[i, k]

    def sample(self, n, goal_dropout=0.0):
        """(obs, chunk, push, core, i, k): demo samples and the expert's push and its readout core there."""
        obs, chunk, i, k = self.sampler.sample(n, goal_dropout, indexed=True)
        return obs, chunk, self.push[i, k], self.core[i, k], i, k


def expert_splits(args, obstacle, policy, device, rng, corruption, curl, p_free=0.0, encoder=None):
    """ExpertData for the train and val splits of a task: the train split --n-demos demonstrations, or with
    --demo-seconds as many as fit in that much demonstrating (`budget_demos`; args.n_demos becomes their count).
    With --teacher both splits are the teacher's hits instead (`teacher_demos`; their counts go to
    args.teacher_demos). With an image --token-source each split's layouts carry that source's scene arrays (`encoder` for vlm_*; the
    caller closes it). With vlm_views each split gets a frame table instead (`view_frames`, after its demonstrations are
    drawn as for any source): the train split's fits the reduction and moves pictures with default_rng([seed, 7]), the val
    split's reuses that reduction and moves none; args.view_info gets each split's info (without the reduction) and
    args.view_reduction the reduction. With --scene-store (vlm_image, vlm_prefix) each split's tokens stay in the store
    instead (`SceneDataset.rows_for` encodes the scenes it lacks with `encoder`) and its sampler gathers them per batch.
    With --scene-rank (those sources) the train split fits the token reduction (`scene_reduction`) unless
    args.scene_reduction already holds one (a pooled run's first entry's), stores it there with its kept share in
    args.scene_kept, and a stored split's sampler reduces with it. With --rigid-motions the train split gains its
    demonstrations' moved copies (`augment_demos` on the expert's pushes, drawing from default_rng([seed, 11])) before
    anything is rendered; the count is added to args.rigid_kept. --jitter goes to the train split's sampler."""
    out = {}
    if args.teacher:
        args.teacher_demos = {}
    if args.token_source == "vlm_views":
        args.view_info, args.view_reduction = {}, None
    for split, n in [("train", args.n_demos), ("val", args.val_demos)]:
        if args.teacher:
            tic = time.time()
            layouts, task, actions, lengths, info = teacher_demos(args, obstacle, policy, device, rng, n, p_free)
            args.teacher_demos[split] = {k: v for k, v in info.items() if k != "records"}
        elif split == "train" and args.demo_seconds is not None:
            tic = time.time()
            layouts, task, actions, lengths = budget_demos(args, obstacle, rng, args.demo_seconds, p_free)
            args.n_demos = n = len(layouts)
        else:
            layouts, task = layouts_for(args, obstacle, rng, n, p_free)
            tic = time.time()
            actions, lengths = make_demos(rng, layouts, task, allow_failed=True)
        frames, push, stored, reduction = None, None, None, None
        if split == "train" and args.rigid_motions:  # before any scene: `Layouts.concat` refuses layouts that carry one
            push = expert_pushes(layouts, demo_positions(layouts, actions))
            layouts, actions, lengths, kept, push = augment_demos(np.random.default_rng([args.seed, 11]), layouts, actions, lengths,
                                                                  args.rigid_motions, task, push=push)
            if getattr(args, "rigid_kept", None) is None:  # `train_task` sets it; a direct call starts the count here
                args.rigid_kept = 0
            args.rigid_kept += int(kept.sum())
            print(f"train: {len(kept)} demonstrations and {kept.sum()} moved copies ({kept.min()}-{kept.max()} per demonstration)",
                  flush=True)
        if args.token_source == "vlm_views":
            train = split == "train"
            tokens, start, info = view_frames(args, layouts, actions, lengths, encoder, device, args.view_reduction,
                                              np.random.default_rng([args.seed, 7]) if train else None)
            if train:
                args.view_reduction = info["reduction"]
            args.view_info[split] = {k: v for k, v in info.items() if k != "reduction"}
            frames = {"vlm_views": (tokens, start, args.view_stride)}
            what = "variance" if args.scene_center == "yes" else "second moment"
            kept = "" if info["kept"] is None else f", {100 * info['kept']:.1f}% of the token {what} kept"
            print(f"{split}: {info['frames']} frames of two views, {info['moved']} pictures moved{kept} "
                  f"({info['seconds']:.0f}s)", flush=True)
        elif args.scene_store is not None:
            store, started = token_store(args), time.time()
            rows = store.rows_for(layouts, encoder)
            print(f"{split}: {len(layouts)} scenes, {store.encoded} newly encoded ({time.time() - started:.0f} s), "
                  f"store {store.path}", flush=True)
            stored = (store, rows, args.token_source)
        elif args.token_source != "discs":
            layouts.scene = {args.token_source: SceneEncoder.scene_array(layouts, args.token_source, args.render_size, encoder)}
        if args.token_source in STORED_SOURCES and args.scene_rank:
            if split == "train" and getattr(args, "scene_reduction", None) is None:
                tokens_of = ((lambda idx: stored[0].gather(stored[1][idx])) if stored is not None
                             else (lambda idx: layouts.scene[args.token_source][idx]))
                args.scene_reduction, args.scene_kept = scene_reduction(args, tokens_of, len(layouts), device)
                what = "variance" if args.scene_center == "yes" else "second moment"
                print(f"train: a token reduction to width {args.scene_rank}, {100 * args.scene_kept:.1f}% of the token {what} "
                      f"kept", flush=True)
            reduction = args.scene_reduction if stored is not None else None   # the resident tokens: the module reduces
        out[split] = ExpertData(layouts, actions, lengths, policy, device, corruption, curl,
                                frozen_noise=split == "train" and args.frozen_noise, frames=frames, push=push,
                                jitter=args.jitter if split == "train" else 0.0, jitter_steps=args.jitter_steps,
                                store=stored, reduction=reduction)
        # the originals come first: their seconds, not the moved copies'
        print(f"{split}: {n} demonstrations ({lengths[:n].sum() * task.dt:.0f} s of demonstrating) and the expert's pushes "
              f"({time.time() - tic:.0f}s)", flush=True)
    return out


STRAIGHT = DemoConfig(max_bow=0.0, p_correct=0.0)   # the relabel: a straight min-jerk reach to the goal


class ClosedLoopData:
    """One split of closed-loop recalibration data: the states a controller visited with the user's readout in
    the loop (trajectories as a DemoSampler, so an observation carries the real history), at every step the
    expert's push there and what the readout makes of it before noise, and as the label the first chunk of a
    straight min-jerk reach from that state to the goal (what the demonstrator would do from there)."""

    def __init__(self, layouts, actions, lengths, raw, policy, device, corruption, curl, rng, task):
        self.sampler = DemoSampler(layouts, actions, lengths, policy.horizon, policy.hist, device)
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.push, self.core = f32(raw), f32(Readout(corruption, curl=curl).core(raw))
        H = policy.horizon
        pos = demo_positions(layouts, actions)
        labels = np.zeros((len(actions), actions.shape[1], H, 2))
        for i in range(len(actions)):
            goal = layouts.goal_pos[i]
            for k in range(int(lengths[i])):
                if np.linalg.norm(goal - pos[i, k]) < 1e-6:
                    continue                                         # on the goal's centre: hold
                steps = np.diff(plan_reach(rng, pos[i, k], goal, task, STRAIGHT), axis=0)[:H]
                labels[i, k, :len(steps)] = steps
        self.labels = f32(labels)

    def sample(self, n, goal_dropout=0.0):
        """(obs, label chunk, push, core, i, k), as ExpertData.sample."""
        obs, _, i, k = self.sampler.sample(n, goal_dropout, indexed=True)
        return obs, self.labels[i, k], self.push[i, k], self.core[i, k], i, k


def closed_loop_splits(args, policy, model, device, rng):
    """ClosedLoopData for the train and val splits (a tenth of every module's trials): the open-field task under
    `--closed-loop-arm` with each initial module of `--closed-loop` in the reversal at `--closed-loop-depth`, the
    user's corruption and `--closed-loop-noise`, the expert pushing at the goal. Prints how those trials went."""
    corruption, curl = corruption_spec(args.corruption)
    parts, records, val = [], [], []
    for name in args.closed_loop:
        arm = config.load_arm(args.closed_loop_arm, dict(
            model=model, mode="frs", reversal_adapter={"user": name, "task": None}, n_reverse_steps=args.closed_loop_depth,
            corruption=args.corruption, input_noise=args.closed_loop_noise, obstacle=None, time_limit_s=args.closed_loop_time_limit,
            targets=list(args.targets), reach=list(args.reach)))
        adapters, task = {name: ReversalAdapter.load(adapter_path(name), device)}, arm_task(arm)
        tic, first = time.time(), len(records)
        for start in range(0, args.closed_loop_trials, 500):
            layouts = sample_layouts(rng, min(500, args.closed_loop_trials - start), tuple(args.targets), task)
            parts.append(layouts)
            records += run_trials(arm, layouts, ExpertAgent(), readout_for(arm, rng), policy, adapters, device)
        mine = records[first:]
        rate = lambda o: np.mean([r["outcome"] == o for r in mine])
        print(f"closed loop with {name} at depth {args.closed_loop_depth}, noise {args.closed_loop_noise}: {len(mine)} trials, "
              f"success {100 * rate('hit'):.1f}%, wrong target {100 * rate('wrong_target'):.1f}%, timeout {100 * rate('timeout'):.1f}%, "
              f"{sum(len(r['raw_commands']) for r in mine)} states ({time.time() - tic:.0f}s)", flush=True)
        val += list(range(len(records) - max(1, len(mine) // 10), len(records)))   # the last tenth of each module's trials
    layouts = Layouts.concat(parts) if len(parts) > 1 else parts[0]
    lengths = np.array([len(r["raw_commands"]) for r in records])
    actions, raw = np.zeros((len(records), lengths.max(), 2)), np.zeros((len(records), lengths.max(), 2))
    for b, r in enumerate(records):
        actions[b, :lengths[b]], raw[b, :lengths[b]] = np.diff(np.asarray(r["path"]), axis=0), np.asarray(r["raw_commands"])
    train = np.setdiff1d(np.arange(len(records)), val)
    split = lambda idx: ClosedLoopData(layouts[idx], actions[idx], lengths[idx], raw[idx], policy, device, corruption, curl, rng, task)
    return {"train": split(train), "val": split(np.array(val))}


def mixed_batch(open_data, closed_data, batch, mix):
    """A batch with a share `mix` of closed-loop samples: (obs, chunk, push, core)."""
    n_c = int(round(mix * batch))
    a, b = open_data.sample(batch - n_c), closed_data.sample(n_c)
    return ({k: torch.cat([a[0][k], b[0][k]]) for k in a[0]},) + tuple(torch.cat([x, y]) for x, y in zip(a[1:4], b[1:4]))


def bci_command(policy, push, core, sigma, noise=None):
    """The controller's reference for one served command per sample: noise on the readout core (where the
    push is active; `noise` the standard normal draws, default fresh), held over the chunk, in the policy's
    units [B, H, 2]."""
    served = Readout.serve(core, push, sigma, noise)
    return to_action(served[:, None].expand(-1, policy.horizon, -1), policy)


def draw_sigma(args, n, device):
    return args.sigma_min + (args.sigma_max - args.sigma_min) * torch.rand(n, device=device)


def active_of(push):
    return push.abs().amax(-1) >= DEADBAND


# ---------------------------------------------------------------- losses and checks


def task_loss(policy, task, obs, chunk, z, t, lambda_id):
    """Flow matching through the adapted field on the policy's interpolation x_t = (1 - t) z + t a (the cursor's
    time convention, t = 0 noise; a the chunk in normalized units), plus the pull of A, b toward the identity.
    Returns the loss and its two parts. The core's `task_field_loss` on this policy and observation."""
    return task_field_loss(task.bind(obs), lambda x, t_: policy(obs, x, t_), chunk / policy.action_scale, z, t,
                           lambda_id, n_dims=2)


def field_loss(policy, user, task, free, latent, bci, weight=None):
    """The regression along the straight path from `latent` to a_bci, x_t = (1 - t) latent + t a_bci (the cursor's
    time convention, t = 0 noise; t drawn uniformly here): U(x_t, t) T(v) = a_bci - latent, v the policy's field
    and T the task module (A v + b) if any. Gradients reach whichever module trains. `weight` [B] (e.g. the active
    commands) averages the per-sample errors. The core's `user_field_loss` on this policy and observation."""
    t = torch.rand(len(latent), device=latent.device)
    return user_field_loss(user, lambda x, t_: policy(free, x, t_), latent, bci, t,
                           None if task is None else task.bind(free), weight)


def collides(obs, chunk):
    """Per sample: whether the chunk (screen units [B, H, 2]) from the cursor runs into one of its discs."""
    pts = torch.cat([obs["pos"][:, None], obs["pos"][:, None] + torch.cumsum(chunk, 1)], 1)   # [B, H + 1, 2]
    p0, p1 = pts[:, :-1, None], pts[:, 1:, None]                                            # [B, H, 1, 2]
    c, r = obs["obstacles"][:, None, :, :2], obs["obstacles"][:, None, :, 2]                 # [B, 1, K, 2], [B, 1, K]
    d = p1 - p0
    tau = (((c - p0) * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    dist = (c - (p0 + tau[..., None] * d)).norm(dim=-1)                                      # [B, H, K]
    return ((dist < r) & obs["obstacle_mask"][:, None]).any(-1).any(-1)


def rows_of(obs, rows):
    """The observation's `rows`; its scene (a dict source -> tensor) too."""
    return {k: rows_of(v, rows) if isinstance(v, dict) else v[rows] for k, v in obs.items()}


@torch.no_grad()
def chunk_collision(policy, adapter, obs, steps):
    """Share of the obstacle trials among `obs` whose chunk, sampled from the (adapted) forward flow,
    runs into a disc (every step's segment tested). NaN when no trial has an obstacle."""
    rows = obs["obstacle_mask"].any(1)
    if not rows.any():
        return float("nan")
    obs = rows_of(obs, rows)
    z = torch.randn(len(obs["pos"]), policy.horizon, 2, device=obs["pos"].device)
    return collides(obs, forward_flow(policy, obs, z, 0.0, 1.0, steps, adapter) * policy.action_scale).float().mean().item()


@torch.no_grad()
def expert_check(policy, data, args, user=None, task=None, n=2048):
    """Open loop on held-out demo states, at full reversal: how well the regenerated chunk heads where the
    label does (cosine of their displacements, over active commands), with no module, the user module, and
    the task module with it (and how often the regenerated chunks collide)."""
    obs, chunk, push, core, _, _ = data.sample(n)
    free, active = hide_goal(obs), active_of(push)
    bci = bci_command(policy, push, core, draw_sigma(args, n, push.device))
    out = {}
    variants = [("none", None, None), ("user", user, None)] + ([("composed", user, task)] if task is not None else [])
    for name, u, tm in variants:
        adapter = ReversalAdapter(user=u, task=tm) if u is not None or tm is not None else None
        regen = regenerate(lambda x, t_: policy(free, x, t_), bci, args.flow_steps, None if adapter is None else adapter.field(free),
                           noise_time=0.0, batch_time=True) * policy.action_scale
        out[f"heading_cos_{name}"] = float(heading_cosine(regen, chunk, n_dims=2)[active].mean())
        if obs["obstacle_mask"].any():
            rows = active & obs["obstacle_mask"].any(1)
            out[f"collide_{name}"] = float(collides(rows_of(obs, rows), regen[rows]).float().mean())
    return out


# ---------------------------------------------------------------- the task module


def entries_of(args):
    """The task module's contexts: the `--pooled` file's entries, or the one the command line names."""
    if args.pooled:
        return args.pooled
    return [dict(obstacle=args.obstacle, user_module=args.user_module, corruption=args.corruption)]


def train_task(args, policy, device, rng):
    """One task module over the contexts of `entries_of(args)` (one for a single-context run): each entry has its
    own demonstrations and frozen user module; batches alternate between entries. The first entry draws from `rng`
    (seeded by `train`), each later one from a fresh stream of `args.seed` (numpy and torch), so every entry's
    demonstrations are those of the single-context module for it: the pooled arm trains on the per-task modules'
    demonstrations together."""
    encoder = SceneEncoder(args.vlm, device) if args.token_source.startswith("vlm_") else None
    entries = entries_of(args)
    args.rigid_kept = 0 if args.rigid_motions else None
    args.scene_reduction, args.scene_kept = None, None   # --scene-rank (vlm_image, vlm_prefix): fitted on the first train split
    try:
        pairs = []
        for j, e in enumerate(entries):
            if j:  # every entry after the first draws as a single-context run of it does: the seed's numpy and torch streams
                rng = np.random.default_rng(args.seed)
                torch.manual_seed(args.seed)
            data = expert_splits(args, obstacle_spec(e["obstacle"]), policy, device, rng, *corruption_spec(e["corruption"]),
                                 args.p_free, encoder)
            pairs.append((data, ReversalAdapter.load(adapter_path(e["user_module"]), device).user.requires_grad_(False)))
    finally:
        if encoder is not None:
            encoder.close()
    views = dict(wrist_span=args.wrist_span, scene_rank=args.scene_rank or None) if args.token_source == "vlm_views" else {}
    if args.token_source in STORED_SOURCES and args.scene_rank:
        views = dict(scene_rank=args.scene_rank)
    task = TaskModule(policy.horizon, d=args.d, layers=args.layers, token_geometry=args.token_geometry,
                      **scene_spec(args.token_source, args.render_size, **views)).to(device)
    if views.get("scene_rank"):
        task.set_reduction(*(args.view_reduction if args.token_source == "vlm_views" else args.scene_reduction))
    adapted = ReversalAdapter(task=task)
    parts = dict(step=0, fm=[], field=[])  # the window's loss parts, reported and cleared by validate

    def batch_loss(step):
        parts["step"] = step
        data, user = pairs[step % len(pairs)]
        obs, chunk, push, core, i, k = data["train"].sample(args.batch, args.goal_dropout)
        z, t = torch.randn_like(chunk), torch.rand(args.batch, device=device)
        seen = obs
        if args.token_drop:  # the flow matching alone sees a share of the scene tokens hidden
            seen = {**obs, "scene_keep": torch.rand(args.batch, task.scene_tokens, device=device) >= args.token_drop}
        loss, fm_parts = task_loss(policy, task, seen, chunk, z, t, args.lambda_id)
        parts["fm"].append(fm_parts["fm"])
        if args.lambda_field:  # the user's commands, through the frozen user module and this task module
            free = hide_goal(obs)
            latent = reverse_flow(policy, free, chunk / policy.action_scale, args.flow_steps, args.flow_steps, adapted)
            bci = bci_command(policy, push, core, draw_sigma(args, args.batch, device), data["train"].noise_at(i, k))
            weight = (active_of(push) & obs["obstacle_mask"].any(1)).float()  # on the obstacle-free layouts T stays I
            field = field_loss(policy, user, task, free, latent, bci, weight)
            parts["field"].append(field.item())
            loss = loss + args.lambda_field * field
        return loss

    @torch.no_grad()
    def validate_entry(data, user):
        sampler = data["val"].sampler
        n = min(2048, 8 * args.batch)
        obs, chunk = sampler.sample(n, args.goal_dropout)
        _, parts = task_loss(policy, task, obs, chunk, torch.randn_like(chunk), torch.rand(n, device=device), 0.0)
        hidden, _ = sampler.sample(n, 1.0)
        given, _ = sampler.sample(n, 0.0)
        val = dict(val_fm=parts["fm"], chunk_collision_hidden=chunk_collision(policy, adapted, hidden, args.flow_steps),
                   chunk_collision_given=chunk_collision(policy, adapted, given, args.flow_steps),
                   chunk_collision_policy_hidden=chunk_collision(policy, None, hidden, args.flow_steps),
                   chunk_collision_policy_given=chunk_collision(policy, None, given, args.flow_steps))
        val.update(expert_check(policy, data["val"], args, user, task))
        return val

    def window_mean(values):
        finite = [v for v in values if np.isfinite(v)]   # as train_loss, a non-finite batch adds nothing
        return float(np.mean(finite)) if finite else None

    def validate():
        val = dict(train_fm=window_mean(parts["fm"]), train_field=window_mean(parts["field"]))
        parts["fm"], parts["field"] = [], []
        per_entry = [validate_entry(data, user) for data, user in pairs]
        val.update({f"{name}_{j}": v for j, vals in enumerate(per_entry) for name, v in vals.items()})
        # an entry without obstacle layouts in its val split lacks the collide_* numbers: average over the entries that have them
        val.update({name: float(np.mean([vals[name] for vals in per_entry if name in vals]))
                    for name in dict.fromkeys(n for vals in per_entry for n in vals)})
        if args.save_every and parts["step"] % args.save_every == 0:   # the module as it is at this step
            save_module(args, "task", task, f"{args.name}_step{parts['step']}", step=parts["step"])
        return val

    save(args, "task", task, train_module(task, args.steps, args.lr, batch_loss, validate, args.log_every))


# ---------------------------------------------------------------- the user module


def train_user(args, policy, model, device, rng):
    """User module from expert labels on the open-field task, with closed-loop recalibration data mixed in
    when `--closed-loop` names an initial module (see the module docstring)."""
    corruption, curl = corruption_spec(args.corruption)
    data = expert_splits(args, None, policy, device, rng, corruption, curl)
    closed = closed_loop_splits(args, policy, model, device, np.random.default_rng(args.seed + 1)) if args.closed_loop else None
    user = UserModule(policy.horizon, step_rank=args.step_rank, cross_step_rank=args.cross_step_rank,
                      cross_step_dims=args.cross_step_dims).to(device)

    def batch_loss(step):
        noise = None
        if closed is None:
            obs, chunk, push, core, i, k = data["train"].sample(args.batch)
            noise = data["train"].noise_at(i, k)
        else:
            obs, chunk, push, core = mixed_batch(data["train"], closed["train"], args.batch, args.closed_loop_mix)
        free = hide_goal(obs)
        latent = reverse_flow(policy, free, chunk / policy.action_scale, args.flow_steps, args.flow_steps)
        bci = bci_command(policy, push, core, draw_sigma(args, args.batch, device), noise)
        return field_loss(policy, user, None, free, latent, bci, active_of(push).float())

    def validate():
        val = expert_check(policy, data["val"], args, user)
        if closed is not None:
            val.update({f"closed_{k}": v for k, v in expert_check(policy, closed["val"], args, user).items()})
        return val

    save(args, "user", user, train_module(user, args.steps, args.lr, batch_loss, validate, args.log_every))


# ---------------------------------------------------------------- main


def save(args, kind, module, log):
    """The module in lerobot's `save_adapter_module` format (kind, config, weights, meta: the arguments and the
    task's obstacle) and its training log, under checkpoints/adapters/<name>/."""
    path, saved = save_module(args, kind, module, args.name)
    with open(adapter_dir(args.name) / "train_log.json", "w") as f:
        json.dump(dict(args=saved, log=log), f, indent=1)
    print(f"saved {path}")


def save_module(args, kind, module, name, **extra):
    """checkpoints/adapters/<name>/adapter.pt in lerobot's `save_adapter_module` format, `extra` added to its meta;
    returns its path and the saved arguments."""
    # the reductions travel in the module's buffers
    saved = {k: v for k, v in vars(args).items() if k not in ("view_reduction", "scene_reduction")}
    reduced = args.token_source in STORED_SOURCES and bool(args.scene_rank)
    center = args.scene_center if args.token_source == "vlm_views" or reduced else None
    store = None if args.scene_store is None else store_name(args.token_source, args.render_size, encoder_digest(args.vlm))
    augmented = {k: getattr(args, k) if kind == "task" else None for k in ("jitter", "jitter_steps", "rigid_motions", "rigid_kept")}
    path = save_adapter_module(module, adapter_dir(name) / "adapter.pt", kind, meta=dict(args=saved, obstacle=args.obstacle, pooled=args.pooled,
                                                              token_source=args.token_source, render_size=args.render_size,
                                                              token_geometry=args.token_geometry,
                                                              teacher=args.teacher, teacher_demos=getattr(args, "teacher_demos", None),
                                                              wrist_span=args.wrist_span, view_stride=args.view_stride,
                                                              scene_rank=args.scene_rank if args.token_source == "vlm_views" else args.scene_rank or None,
                                                              scene_store=store, view_affine=args.view_affine,
                                                              token_drop=args.token_drop, views=getattr(args, "view_info", None),
                                                              scene_center=center, **augmented, **extra))
    return path, saved


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = args.model or config.load_arm("base.yaml")["model"]
    policy = FlowPolicy.load(policy_path(model), device)
    policy.requires_grad_(False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_task(args, policy, device, rng) if args.kind == "task" else train_user(args, policy, model, device, rng)


def dim_indices(text):
    """A comma list of dim indices ('0,1') as a list of ints."""
    try:
        return [int(i) for i in text.split(",") if i.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a comma list of dim indices, got {text!r}") from None


NO_CURL_COPIES = "--rigid-motions mirrors demonstrations and a curl's handedness does not mirror: no curl corruption"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["user", "task"], default="user")
    p.add_argument("--name", required=True)
    p.add_argument("--model", help="policy under checkpoints/policies/ (default: base.yaml's)")
    p.add_argument("--corruption", type=yaml.safe_load,
                   help="the user's corruption, e.g. '{rotation_deg: 60}' or '{curl: 20, tau: 0.1}'")
    p.add_argument("--pooled", help="task: a YAML file `entries: [{obstacle, user_module, corruption}, ...]`: one module "
                   "trained over these contexts (instead of --obstacle, --user-module and --corruption)")
    p.add_argument("--obstacle", type=yaml.safe_load, help="task: the task's obstacle spec")
    p.add_argument("--user-module", help="task: the user's frozen user module, trained with that --corruption")
    p.add_argument("--lambda-field", type=float, default=0.03, help="task: weight of the field loss through the user module")
    p.add_argument("--sigma-min", type=float, default=1.0, help="noise std of the commands is drawn from [sigma_min, sigma_max]")
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--closed-loop", nargs="+", help="user: recalibrate on closed-loop trials of these initial user modules "
                   "(ReFIT-style; several pool their trials)")
    p.add_argument("--closed-loop-trials", type=int, default=2000, help="user: closed-loop trials to run per initial module")
    p.add_argument("--closed-loop-noise", type=float, default=1.0, help="user: the readout's noise std in those trials")
    p.add_argument("--closed-loop-depth", type=int, default=6, help="user: the initial module's reversal depth in them")
    p.add_argument("--closed-loop-arm", default="condition_grid.yaml", help="user: the controller and task they run under")
    p.add_argument("--closed-loop-time-limit", type=float, default=5.0, help="user: their time limit (s)")
    p.add_argument("--closed-loop-mix", type=float, default=0.5, help="user: share of every batch from closed-loop states")
    p.add_argument("--p-free", type=float, default=0.25, help="task: share of layouts without the obstacle")
    p.add_argument("--goal-dropout", type=float, default=0.25, help="task: as the policy was trained")
    p.add_argument("--lambda-id", type=float, default=1e-3, help="task: pull of A, b toward the identity")
    p.add_argument("--token-source", choices=SCENE_SOURCES, default="discs",
                   help="task: what the module reads the obstacles from: the disc parameters, or the rendered scene "
                        "through pi0.5's vision tokens (vlm_image, vlm_prefix; vlm_views: two views per demo frame) or a CNN "
                        "trained with the module (cnn)")
    p.add_argument("--wrist-span", type=float, default=0.5, help="task, vlm_views: the wrist view's side, in screens")
    p.add_argument("--view-stride", type=int, default=2, help="task, vlm_views: a frame every this many demo steps")
    p.add_argument("--scene-rank", type=int, help="task, a vlm_* source: width of the PCA token reduction fitted on the "
                   "train split's frames (vlm_views, default 256) or scenes (vlm_image, vlm_prefix, default full width); "
                   "0: full width, 2048")
    p.add_argument("--scene-center", choices=["yes", "no"], default="yes", help="task, with a --scene-rank reduction: no "
                   "fits it uncentred (the tokens' constant part kept: a zero mean, the second moment's top directions)")
    p.add_argument("--scene-store", help="task, vlm_image / vlm_prefix: a folder (created if missing) whose disk token store "
                   "holds the scenes' tokens, gathered per batch, instead of the device (default: the device)")
    p.add_argument("--view-affine", type=float, default=0.0, help="task, vlm_views: the probability a stored training "
                   "picture is moved (a small rotation and shift, frozen in the frame table)")
    p.add_argument("--token-drop", type=float, default=0.0, help="task, an image source: the share of scene tokens hidden "
                   "from the flow-matching loss, drawn anew every batch")
    p.add_argument("--render-workers", type=int, default=16, help="task, vlm_views: processes rendering the views "
                   "(0: in-process)")
    p.add_argument("--token-geometry", choices=TOKEN_GEOMETRIES, default="absolute",
                   help="task: relative puts each scene token's offset from the cursor into the token (vlm_image, cnn) "
                        "instead of a learned position")
    p.add_argument("--jitter", type=float, default=0.0, help="task: std of a training sample's move off the path (a recovery "
                   "label takes it back out; 0: off)")
    p.add_argument("--jitter-steps", type=int, default=4, help="task: the steps of the chunk that take that move back out")
    p.add_argument("--rigid-motions", type=int, default=0, help="task: up to this many copies of every training demonstration "
                   "under random rigid motions of the screen, their pushes moved with them (0: off)")
    p.add_argument("--teacher", help="task: demonstrations are this parametric task module's oracle_task rollouts (hits only, "
                   "resampled to the policy's step rate) instead of the scripted expert's")
    p.add_argument("--teacher-chunk", type=int, default=500, help="task: layouts per teacher rollout batch")
    p.add_argument("--render-size", type=int, default=SCENE_SIZE, help="task: side of the rendered scene in pixels (cnn: a multiple of 16)")
    p.add_argument("--vlm", default="lerobot/pi05_base", help="task: the pi0.5 checkpoint whose vision tokens vlm_* sources use "
                                                              "(loaded offline from the HF cache)")
    p.add_argument("--d", type=int, default=128, help="task: width")
    p.add_argument("--layers", type=int, default=2, help="task: cross-attention layers")
    p.add_argument("--targets", type=int, nargs=2, default=[4, 4], help="min and max targets per layout")
    p.add_argument("--reach", type=yaml.safe_load, nargs=2, default=[0.35, 0.5], help="min and max reach (null: none)")
    p.add_argument("--step-rank", type=int, help="user: matrices shared by the chunk's steps (default: every step's free)")
    p.add_argument("--cross-step-rank", type=int, default=0, help="user: rank of the coupling between the chunk's steps")
    p.add_argument("--cross-step-dims", type=dim_indices, help="user: the dims the coupling covers, a comma list of "
                   "indices (e.g. 0 or 0,1; default: all)")
    p.add_argument("--n-demos", type=int, default=20000, help="training demonstrations (one reach each)")
    p.add_argument("--demo-seconds", type=float, help="instead of --n-demos, as many demonstrations as fit in this "
                   "much demonstrating, each one's duration up to the goal's selection (`budget_demos`)")
    p.add_argument("--frozen-noise", action="store_true", help="draw the readout's noise once per demonstration step "
                   "(as a recorded calibration has it) instead of anew in every batch")
    p.add_argument("--val-demos", type=int, default=2000)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=0, help="task: also save the module as it is at every this many "
                   "steps, as <name>_step<N> (a multiple of --log-every; 0: off)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.teacher and (args.kind != "task" or args.pooled):
        p.error("--teacher is for a single-context --kind task (user modules calibrate on the open field; no pooled teacher run)")
    if args.teacher_chunk <= 0:
        p.error("--teacher-chunk must be positive")
    if args.teacher and args.demo_seconds is not None:
        p.error("--teacher draws --n-demos hits: a --demo-seconds budget is the scripted expert's")
    if args.token_geometry == "relative" and args.token_source in ("discs", "vlm_prefix"):
        p.error("--token-geometry relative needs screen tokens: --token-source vlm_image or cnn")
    if args.token_source == "vlm_views":
        if args.token_geometry == "relative":
            p.error("vlm_views: nothing derived from the cursor's position enters the scene tokens: no --token-geometry relative")
        if args.teacher:
            p.error("vlm_views trains on the scripted expert's demonstrations: no --teacher")
        if args.pooled:
            p.error("vlm_views is a single-context run: no --pooled")
        if args.jitter > 0 or args.rigid_motions > 0:
            p.error("vlm_views: a frame's pictures show the demonstration's own position: no --jitter or --rigid-motions")
    if args.token_source == "discs" and (args.token_drop > 0 or args.view_affine > 0):
        p.error("--token-drop and --view-affine act on scene tokens: not with --token-source discs")
    if args.view_affine > 0 and args.token_source != "vlm_views":
        p.error("--view-affine moves the stored pictures of --token-source vlm_views")
    if args.wrist_span <= 0:
        p.error("--wrist-span must be positive")
    if args.view_stride < 1:
        p.error("--view-stride must be at least 1")
    if args.scene_rank is not None and not args.token_source.startswith("vlm_"):
        p.error("--scene-rank reduces a vlm_* source's 2048-wide tokens: not with --token-source discs or cnn")
    if args.token_source == "vlm_views" and args.scene_rank is None:
        args.scene_rank = 256
    if args.scene_rank is not None and not 0 <= args.scene_rank <= 2047:
        p.error("--scene-rank must be in 0..2047 (0: full width)")
    if args.scene_center == "no" and not args.scene_rank:
        p.error("--scene-center no fits the token reduction of a vlm_* source: not without a --scene-rank or with 0")
    if args.scene_store is not None and args.token_source not in STORED_SOURCES:
        p.error("--scene-store holds the tokens of --token-source vlm_image or vlm_prefix (vlm_views has its frame table, "
                "discs and cnn no tokens)")
    if args.save_every and (args.kind != "task" or args.save_every < 0 or args.save_every % args.log_every):
        p.error("--save-every is for --kind task and must be a positive multiple of --log-every")
    if not 0 <= args.token_drop < 1:
        p.error("--token-drop must be in [0, 1)")
    if not 0 <= args.view_affine <= 1:
        p.error("--view-affine must be in [0, 1]")
    if args.jitter < 0:
        p.error("--jitter must not be negative")
    if args.jitter_steps < 1:
        p.error("--jitter-steps must be at least 1")
    if args.rigid_motions < 0:
        p.error("--rigid-motions must not be negative")
    if args.kind != "task" and (args.jitter or args.rigid_motions):
        p.error("--jitter and --rigid-motions are for --kind task")
    if args.rigid_motions and args.teacher:
        p.error("--rigid-motions moves the scripted expert's demonstrations: no --teacher")
    if args.pooled:
        if args.kind != "task":
            p.error("--pooled is for --kind task")
        if args.obstacle is not None or args.user_module is not None or args.corruption is not None:
            p.error("--pooled carries its own obstacles, user modules and corruptions: no --obstacle, --user-module or --corruption")
        with open(args.pooled) as f:
            args.pooled = yaml.safe_load(f)["entries"]
        for e in args.pooled:
            if not {"obstacle", "user_module", "corruption"} <= set(e):
                p.error("--pooled: every entry needs obstacle, user_module and corruption")
            if args.rigid_motions and corruption_spec(e["corruption"])[1] is not None:
                p.error(NO_CURL_COPIES)
        args.obstacle = None
        return args
    if args.corruption is None:
        p.error("--corruption is required")
    if args.rigid_motions and corruption_spec(args.corruption)[1] is not None:
        p.error(NO_CURL_COPIES)
    if args.kind == "task" and (args.obstacle is None or args.user_module is None):
        p.error("--kind task needs the task's --obstacle and the user's --user-module")
    if args.frozen_noise and args.closed_loop:
        p.error("--frozen-noise is for calibration on the demonstrations: not with --closed-loop")
    if args.cross_step_dims is not None and not args.cross_step_rank:
        p.error("--cross-step-dims needs --cross-step-rank")
    if args.kind == "user" and args.obstacle is not None:
        p.error("a user module calibrates on the open field: no --obstacle")
    args.obstacle = obstacle_spec(args.obstacle)
    return args


if __name__ == "__main__":
    train(parse_args())
