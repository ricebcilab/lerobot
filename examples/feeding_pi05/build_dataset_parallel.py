#!/usr/bin/env python
# Copyright 2026 Rice BCI Lab. Apache-2.0.
"""Parallel driver for ``convert_nwb_to_lerobot.py``.

The converter is per-episode overhead bound (~3-4 s/episode is stats + parquet
writing, not video encoding), and it runs single-process, so a 5000-demo build
takes hours while the machine sits idle. This driver shards the seeds across N
worker processes -- each runs the converter on a disjoint seed subset into its
own dataset dir -- then merges the shards into one LeRobotDataset with
``aggregate_datasets``. On a many-core box this is ~Nx faster.

    python build_dataset_parallel.py --raw-root "$RAW_ROOT" --fps 30 --workers 8

Unrecognized flags (e.g. ``--no-success-only``, ``--min-go-seconds 0.5``) are
forwarded verbatim to every converter worker.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CONVERTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_nwb_to_lerobot.py")


def seed_of(path):
    return int(re.search(r"seed(\d+)", os.path.basename(path)).group(1))


def seeds_with_data(raw_root):
    nwb = {seed_of(p) for p in glob.glob(os.path.join(raw_root, "NWB", "*seed*.nwb"))}
    vid = {seed_of(d) for d in glob.glob(os.path.join(raw_root, "videos", "*seed*"))}
    return sorted(nwb & vid)


def split_even(items, n):
    """Split a list into n contiguous, near-equal groups (drops empties)."""
    k, m = divmod(len(items), n)
    out, i = [], 0
    for g in range(n):
        size = k + (1 if g < m else 0)
        out.append(items[i:i + size])
        i += size
    return [g for g in out if g]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", required=True, help="Folder containing NWB/ and videos/.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--workers", type=int, default=8, help="Number of parallel converter processes.")
    p.add_argument("--repo-id", default="rice/feeding_pi05", help="Final merged dataset repo id.")
    p.add_argument("--output-root", default=None, help="Final dataset dir. Default: <raw-root>/lerobot")
    p.add_argument("--shard-root", default=None, help="Where shards are staged. Default: <raw-root>/lerobot_shards")
    p.add_argument("--keep-shards", action="store_true", help="Don't delete shard datasets after merge.")
    args, passthrough = p.parse_known_args()

    raw_root = os.path.abspath(args.raw_root)
    final_root = os.path.abspath(args.output_root or os.path.join(raw_root, "lerobot"))
    shard_base = os.path.abspath(args.shard_root or os.path.join(raw_root, "lerobot_shards"))

    seeds = seeds_with_data(raw_root)
    if not seeds:
        sys.exit(f"No seeds with both NWB and video found under {raw_root}")
    groups = split_even(seeds, min(args.workers, len(seeds)))
    print(f"{len(seeds)} seeds -> {len(groups)} workers: " + " | ".join(",".join(map(str, g)) for g in groups))

    if os.path.exists(shard_base):
        shutil.rmtree(shard_base)
    os.makedirs(shard_base)

    # Launch workers. Each writes to its own dataset dir + log; HDF5 locking off
    # so concurrent read-only NWB opens never block (workers touch disjoint files).
    env = dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE")
    workers = []
    shard_roots, shard_repo_ids = [], []
    for i, g in enumerate(groups):
        root_i = os.path.join(shard_base, f"part{i}")
        repo_i = f"{args.repo_id}_part{i}"
        cmd = [sys.executable, CONVERTER,
               "--raw-root", raw_root, "--fps", str(args.fps),
               "--seeds", ",".join(map(str, g)),
               "--output-root", root_i, "--repo-id", repo_i, "--overwrite", *passthrough]
        log = open(os.path.join(shard_base, f"part{i}.log"), "w")
        workers.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        shard_roots.append(root_i)
        shard_repo_ids.append(repo_i)
        print(f"[worker {i}] seeds {g} -> {root_i}")

    failed = []
    for i, proc, log in workers:
        rc = proc.wait()
        log.close()
        print(f"[worker {i}] exited rc={rc}")
        if rc != 0:
            failed.append(i)
    if failed:
        sys.exit(f"Workers {failed} failed -- see {shard_base}/part*.log")

    print("\nAll workers done. Merging shards ->", final_root)
    from lerobot.datasets.aggregate import aggregate_datasets

    if os.path.exists(final_root):
        shutil.rmtree(final_root)
    aggregate_datasets(
        repo_ids=shard_repo_ids,
        aggr_repo_id=args.repo_id,
        roots=[Path(r) for r in shard_roots],
        aggr_root=Path(final_root),
    )
    print(f"\nDONE. Merged dataset at {final_root}")
    if not args.keep_shards:
        shutil.rmtree(shard_base)
        print("Removed shard staging dir (pass --keep-shards to keep).")


if __name__ == "__main__":
    main()
