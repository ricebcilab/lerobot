#!/usr/bin/env python
"""An image task module on its own training layouts, on fresh ones and on its training layouts moved.

    python diag_layouts.py --sweep grid_task_check_image2_quick --context T2 --method oracle_task_rel \\
        --seed 0 --n-demos 300 --p-free 0.25 --fresh-seed 820 --trials 225 --out ../results/sim/diag_layouts

Three sets of layouts of the context's task (the sweep's users x tasks cell of that task with no corruption):
`train`, the layouts the trainer drew for the module's train split (`rebuild_layouts` with the module's --seed,
--n-demos and --p-free; only the rows with the obstacle), `fresh`, as many new ones drawn from --fresh-seed, all
with the obstacle, and `moved`, the `train` rows each under one random rigid motion that keeps it on screen
(`moved_layouts`, drawing from default_rng([seed, 12]), a stream disjoint from the trainer's rigid copies). Each set is cut to its first --trials rows. On every set
`oracle` (the bare policy), `oracle_task` (the parametric module) and --method (the image module) run once at
noise 0.0 with the sweep's seed, the scene arrays cached under --scene-cache. A module that hits on `train` but not
on `fresh` or `moved` has learned its training layouts rather than the obstacle.

Prints a Markdown table of hits / collisions (%) per set and arm, and writes <out>/<context>_<method>.json:
{set: {arm: {"hits": share, "collision": share, "n": trials}}}.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

from cursor_sa import paths as cursor_paths
from cursor_sa.agents import make_agent
from cursor_sa.env import layouts_on_screen, move_layouts, obstacle_spec
from cursor_sa.session import load_checkpoints
from run_sweep import attach_scene, load_sweep, method_arm, run_cell_batch, scene_sources
from train import train_adapter

from lerobot.policies.steering import SceneEncoder
from lerobot.policies.steering.modeling_steering import SCENE_VLM_PATH

REFERENCES = ("oracle", "oracle_task")  # the bare policy and the parametric module, beside the method


def rebuild_layouts(seed, n, p_free, obstacle, targets=(4, 4), reach=(0.35, 0.5)):
    """The trainer's train-split layouts for these arguments: `train_adapter.layouts_for` on the first draws of
    default_rng(seed), as `train` makes them (the defaults are the trainer's --targets and --reach). The obstacle
    rows come first, the share p_free without it last."""
    args = types.SimpleNamespace(reach=list(reach), targets=list(targets))
    layouts, _ = train_adapter.layouts_for(args, obstacle, np.random.default_rng(seed), n, p_free)
    return layouts


def moved_layouts(layouts, rng, shift=0.25, tries=50):
    """Every layout under one random rigid motion (`move_layouts`) that keeps it on screen (`layouts_on_screen`).
    Each round draws, for the rows not yet moved, an angle in [0, 2 pi), a shift in [-shift, shift]^2 and a mirror
    with chance 1/2, in that order, and keeps the motions that fit; after `tries` rounds a row still unmoved keeps
    its original (their count is printed). The rows keep their order; the result carries no scene."""
    out = layouts[np.arange(len(layouts))]  # a copy
    out.scene = None
    todo = np.arange(len(layouts))
    for _ in range(tries):
        if not len(todo):
            break
        angle = rng.uniform(0, 2 * np.pi, len(todo))
        offset = rng.uniform(-shift, shift, (len(todo), 2))
        mirror = rng.random(len(todo)) < 0.5
        moved = move_layouts(layouts[todo], angle, offset, mirror)
        fits = layouts_on_screen(moved)
        rows = todo[fits]
        out.start[rows], out.targets[rows], out.obstacles[rows] = moved.start[fits], moved.targets[fits], moved.obstacles[fits]
        todo = todo[~fits]
    print(f"moved: {len(todo)} of {len(layouts)} layouts found no on-screen motion in {tries} rounds and keep their "
          "original", flush=True)
    return out


def context_cell(sweep, context):
    """The users x tasks cell of the sweep for the task `context` with no corruption."""
    if sweep["grid"]["family"] != "grid":
        raise ValueError(f"{sweep['name']}: the diagnostic needs a users x tasks grid")
    cells = [k for k, c in sweep["grid"]["cells"].items() if c["task"] == context and c["corruption"] is None]
    if len(cells) != 1:
        raise ValueError(f"{sweep['name']}: --context {context} needs exactly one cell of that task with no corruption, "
                         f"found {cells}")
    return cells[0]


def rates(records):
    outcomes = [r["outcome"] for r in records]
    return dict(hits=outcomes.count("hit") / len(outcomes), collision=outcomes.count("collision") / len(outcomes),
                n=len(outcomes))


def table(results, arms):
    lines = ["| set | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for name, row in results.items():
        cells = [f"{100 * row[m]['hits']:.1f} / {100 * row[m]['collision']:.1f}" for m in arms]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", required=True, help="sweep config (bare names are looked up in configs/sweeps/)")
    p.add_argument("--context", required=True, help="the task, e.g. T2 or T4")
    p.add_argument("--method", required=True, help="the sweep's method playing the image module, e.g. oracle_task_rel")
    p.add_argument("--seed", type=int, default=0, help="the module's training --seed")
    p.add_argument("--n-demos", type=int, default=300, help="the module's training --n-demos")
    p.add_argument("--p-free", type=float, default=0.25, help="the module's training --p-free")
    p.add_argument("--fresh-seed", type=int, default=820, help="the seed of the fresh layouts")
    p.add_argument("--trials", type=int, required=True, help="the first this many rows of each set")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--scene-cache", help="folder of the cached scene arrays (default: checkpoints/scene/diag_layouts)")
    p.add_argument("--vlm", default=SCENE_VLM_PATH, help="the pi0.5 checkpoint whose vision tokens vlm_* sources use")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    sweep = load_sweep(args.sweep)
    methods = list(dict.fromkeys([*REFERENCES, args.method]))
    missing = [m for m in methods if m not in sweep["methods"]]
    if missing:
        p.error(f"{sweep['name']} has no methods {missing}")
    cell = context_cell(sweep, args.context)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    obstacle = obstacle_spec(sweep["grid"]["cells"][cell]["obstacle"])
    rebuilt = rebuild_layouts(args.seed, args.n_demos, args.p_free, obstacle)
    train = rebuilt[np.flatnonzero(rebuilt.obstacle_mask.any(1))]
    sets = dict(train=train, fresh=rebuild_layouts(args.fresh_seed, len(train), 0.0, obstacle),
                moved=moved_layouts(train, np.random.default_rng([args.seed, 12])))
    sets = {name: layouts[: args.trials] for name, layouts in sets.items()}

    arms = {m: method_arm(sweep, m, cell) for m in methods}
    policies, adapters = load_checkpoints(arms, device)
    agent = make_agent(sweep["agent"])
    sources = scene_sources(adapters)
    cache_dir = Path(args.scene_cache) if args.scene_cache else cursor_paths.CHECKPOINTS / "scene" / "diag_layouts"
    encoders = []

    def encoder_factory():  # one encoder for the whole run, made at the first vlm_* source to compute or vlm_views module to play
        if not encoders:
            encoders.append(SceneEncoder(args.vlm, device=device))
        return encoders[0]

    results = {}
    try:
        for name, layouts in sets.items():
            attach_scene(layouts, sources, cache_dir, f"{args.context}_{name}", encoder_factory, args.vlm)
            results[name] = {}
            for m, arm in arms.items():
                (records,) = run_cell_batch(sweep, arm, m, layouts, [0.0], agent, policies[arm["model"]], adapters, device,
                                            sweep["seed"], scene_encoder=encoder_factory).values()
                results[name][m] = rates(records)
            print(f"{name}: {len(layouts)} layouts done", flush=True)
    finally:
        for encoder in encoders:
            encoder.close()

    print(f"\n{sweep['name']} {cell} ({args.context}), hits / collisions (%):\n\n{table(results, methods)}\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.context}_{args.method}.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
