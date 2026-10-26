#!/usr/bin/env python
"""Relabel a stratified fraction of a LeRobotDataset's episodes with alternative prompts.

Dataset-level prompt mixing: within every original task string, ``round(frac * n)`` episodes
are drawn (deterministic in ``--seed``) and relabelled with one of ``--prompts``, assigned
round-robin so each paraphrase gets an equal share. The rest keep their original prompt. Use
it to make a prompt that the deployment sends but training never saw in-distribution (e.g. an
object-agnostic "pick something up" instruction next to object-specific ones) without
collecting new data, while the untouched majority keeps language grounding intact.

The source dataset is left untouched: it is copied to ``--dst-root`` (videos included) and the
copy is relabelled in place with ``lerobot.datasets.dataset_tools.modify_tasks`` (the same
operation as ``lerobot-edit-dataset --operation.type modify_tasks``). ``relabel_manifest.json``
in the copy records the plan.

Usage (feeding example; the first prompt is the exact string the deployment graph sends):
  python examples/pi05/feeding_finetune/relabel_prompts.py \
      --src-root $RAW_ROOT/lerobot_v1 --dst-root $RAW_ROOT/lerobot_v1_gp \
      --repo-id rice/feeding_pi05_v1_gp --frac 0.4 --seed 0 \
      --prompts "Reach, grasp and bring to mouth" \
                "Reach, grasp and bring an object to mouth" \
                "Reach, grasp and bring some food to mouth"
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def plan_relabel(
    episode_tasks: Mapping[int, str], frac: float, seed: int, prompts: Sequence[str]
) -> dict[int, str]:
    """Choose which episodes get a new prompt and which paraphrase each gets.

    Stratified per original task string; deterministic in ``seed``; paraphrases balanced
    round-robin over the shuffled selection. Returns ``{episode_index: new_prompt}`` for the
    selected episodes only.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    if not prompts:
        raise ValueError("at least one prompt is required")
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = collections.defaultdict(list)
    for ep, task in sorted(episode_tasks.items()):
        by_task[task].append(ep)

    chosen: list[int] = []
    for task in sorted(by_task):
        eps = sorted(by_task[task])
        k = int(round(frac * len(eps)))
        if k:
            chosen.extend(int(e) for e in rng.choice(eps, size=k, replace=False))
    order = rng.permutation(len(chosen))
    return {chosen[i]: prompts[j % len(prompts)] for j, i in enumerate(order)}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-root", required=True, help="Existing LeRobotDataset root (left untouched).")
    p.add_argument("--dst-root", required=True, help="Where the relabelled copy is written.")
    p.add_argument("--repo-id", required=True, help="repo_id recorded for the copy.")
    p.add_argument("--prompts", nargs="+", required=True, help="New prompts, assigned round-robin.")
    p.add_argument("--frac", type=float, default=0.4, help="Fraction of episodes per original task.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing --dst-root.")
    args = p.parse_args(argv)

    src, dst = Path(args.src_root).expanduser().resolve(), Path(args.dst_root).expanduser().resolve()
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"{src} is not a LeRobotDataset root (no meta/info.json)")
    if dst == src or src in dst.parents:
        raise SystemExit("--dst-root must be a separate directory, not the source or inside it")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    print(f"copying {src} -> {dst} ...")
    shutil.copytree(src, dst)

    from lerobot.datasets.dataset_tools import modify_tasks
    from lerobot.datasets.io_utils import load_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = load_episodes(dst)
    episode_tasks = {int(i): str(episodes[i]["tasks"][0]) for i in range(len(episodes))}
    plan = plan_relabel(episode_tasks, args.frac, args.seed, args.prompts)

    ds = LeRobotDataset(args.repo_id, root=dst)
    modify_tasks(ds, episode_tasks=plan)

    per_task = collections.Counter(episode_tasks[ep] for ep in plan)
    totals = collections.Counter(episode_tasks.values())
    manifest = {
        "src_root": str(src),
        "frac": args.frac,
        "seed": args.seed,
        "prompts": list(args.prompts),
        "n_episodes": len(episode_tasks),
        "n_relabelled": len(plan),
        "relabelled_per_prompt": collections.Counter(plan.values()),
        "relabelled_per_original_task": {t: [per_task[t], totals[t]] for t in sorted(totals)},
        "episodes": {str(ep): prompt for ep, prompt in sorted(plan.items())},
    }
    (dst / "relabel_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nrelabelled {len(plan)}/{len(episode_tasks)} episodes ({len(plan) / len(episode_tasks):.1%})")
    for prompt, n in manifest["relabelled_per_prompt"].items():
        print(f"  {n:5d}  {prompt!r}")
    print("relabelled/total per original task:")
    for t in sorted(totals):
        print(f"  {per_task[t]:3d}/{totals[t]:<4d} {t}")
    print(f"tasks now: {ds.meta.total_tasks}; manifest: {dst / 'relabel_manifest.json'}")


if __name__ == "__main__":
    main()
