#!/usr/bin/env python
# Copyright 2026 Rice BCI Lab. Apache-2.0.
"""Convert the OmniGibson feeding NWB + mp4 collection into a LeRobotDataset for
finetuning pi-0.5.

Input layout (per seed), produced by the patched ``MP4CollectionWrapper``::

    <raw_root>/NWB/*_seed<N>.nwb
    <raw_root>/videos/*_seed<N>/demo_<d>_<cam>.mp4
    <raw_root>/videos/*_seed<N>/demo_<d>_timestamps.npy   # sync clock per frame

Alignment is timestamp-based via the per-frame sidecar (same ``monotonic_ns`` clock
as the NWB action stream, ~4 ms). Per demo we keep only the GO PERIOD
(``go_cue_time -> stop_time``), drop unsuccessful/degenerate trials, and resample
to a fixed control rate. Actions are per-step EEF pose deltas
``[dx,dy,dz,drx,dry,drz, gripper]``; below the native rate (~38 Hz) the six pose
dims are SUMMED per output window (preserves motion speed) and the gripper takes
the window's last value -- so train and deploy at the same ``--fps``.

Output goes to ``<raw_root>/lerobot`` by default (a subfolder; the converter
refuses to write over your raw NWB/ or videos/).

    python convert_nwb_to_lerobot.py --raw-root D:/Robotics/results/pi-finetune --fps 30
"""
import argparse
import glob
import os
import re
import shutil
import sys

import av
import numpy as np

# Camera obs key (in the mp4 filename) -> short feature name used in the dataset.
CAMS = {
    "overhead": "external_overhead_cam_rgb",
    "side": "external_side_cam_rgb",
    "wrist": "kinova_kinova_gen3_bracelet_link_Camera_0_rgb",
}
GRIP = 6  # action dims [0:6] are pose deltas (summed); dim 6 is the gripper (last).
CLOSE_THRESH = -0.02  # gripper action below this counts as "closing"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", default="D:/Robotics/results/pi-finetune",
                   help="Folder containing NWB/ and videos/ (the raw collection).")
    p.add_argument("--output-root", default=None, help="Output dataset dir. Default: <raw-root>/lerobot")
    p.add_argument("--repo-id", default="rice/feeding_pi05", help="Dataset repo id (metadata only).")
    p.add_argument("--task-prompt", default="Reach, grasp and bring to mouth the {food}",
                   help="Language instruction; '{food}' is filled from the trial's text cue.")
    p.add_argument("--fps", type=int, default=30,
                   help="Output control rate (Hz). Native ~38; pose deltas are summed when fps<native.")
    p.add_argument("--min-go-seconds", type=float, default=0.5, help="Drop demos with a shorter go-period.")
    p.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only successful trials (use --no-success-only to include failures).")
    p.add_argument("--seeds", default=None, help="Comma/range of seed indices, e.g. '0,1,2' or '0-9'.")
    p.add_argument("--max-demos-per-seed", type=int, default=None, help="Cap demos per seed (smoke test).")
    p.add_argument("--mid-approach-crops", action=argparse.BooleanOptionalAction, default=False,
                   help="For each kept demo, ALSO emit a cropped episode starting 0.3-1.0 s (uniform) "
                        "before the first gripper close (decorrelates grasp timing from episode time).")
    p.add_argument("--balance-objects", action=argparse.BooleanOptionalAction, default=False,
                   help="Balance the dataset across food categories (the trial text cue): downsample "
                        "over-represented foods and oversample (duplicate) under-represented ones toward "
                        "a per-category target. The plan is GLOBAL over all seeds in --raw-root and "
                        "deterministic per (seed, demo), so parallel shards stay consistent.")
    p.add_argument("--balance-target", default="median",
                   help="Per-category episode target when --balance-objects: 'median' (default), 'min', "
                        "or an integer count.")
    p.add_argument("--balance-max-oversample", type=int, default=4,
                   help="Cap the duplication factor for rare categories (safety against tiny categories).")
    p.add_argument("--overwrite", action="store_true", help="Remove an existing output dataset first.")
    return p.parse_args(argv)


def resolve_seeds(arg):
    if arg is None:
        return None
    if "-" in arg and "," not in arg:
        lo, hi = arg.split("-")
        return set(range(int(lo), int(hi) + 1))
    return {int(x) for x in arg.split(",")}


def seed_index_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def grab(video_path, idxs):
    """Return RGB frames at the given indices (idxs are valid: sidecar len == frame count).

    Uses PyAV rather than OpenCV: the opencv-python wheel's bundled FFmpeg only
    attempts hardware AV1 decode and silently yields zero frames on machines
    without it, whereas PyAV decodes AV1 in software.
    """
    want, store, mx = set(idxs), {}, max(idxs)
    with av.open(video_path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in want:
                store[i] = frame.to_ndarray(format="rgb24")
            if i >= mx:
                break
    if want - store.keys():
        raise LookupError(f"{video_path}: decoded {len(store)} of {len(want)} wanted frames (truncated?)")
    return [store[i] for i in idxs]


def build_episode(seed_dir, demo, fa, proprio, fa_ts, go, stop, fps):
    """(frames_per_cam, actions, states) for one demo's go-period, or None if empty.

    Timestamp-binned at 1/fps: pose deltas summed per bin, gripper = last in bin,
    image+state sampled at each bin start (frame chosen by the sidecar clock).
    """
    side = np.load(os.path.join(seed_dir, f"demo_{demo}_timestamps.npy")).astype(np.float64) / 1e9
    period = 1.0 / fps
    actions, states, idxs = [], [], []
    for t0 in np.arange(go, stop, period):
        js = np.nonzero((fa_ts >= t0) & (fa_ts < t0 + period))[0]
        if len(js):
            act = np.empty(7, np.float32)
            act[:GRIP] = fa[js, :GRIP].sum(0)        # sum pose deltas over the window
            act[GRIP:] = fa[js[-1], GRIP:]           # last gripper
            j0 = int(js[0])
        else:                                        # fps above native: nearest single delta
            j0 = int(np.argmin(np.abs(fa_ts - t0)))
            act = fa[j0].astype(np.float32)
        actions.append(act)
        states.append(proprio[j0])
        idxs.append(int(np.argmin(np.abs(side - t0))))
    if not actions:
        return None
    frames = {n: grab(os.path.join(seed_dir, f"demo_{demo}_{c}.mp4"), idxs) for n, c in CAMS.items()}
    return frames, np.stack(actions), np.stack(states)


def first_close_time(fa, fa_ts, go, stop):
    """Timestamp of the first gripper-close command in the go-period, or None."""
    js = np.nonzero((fa_ts >= go) & (fa_ts < stop) & (fa[:, GRIP] < CLOSE_THRESH))[0]
    return float(fa_ts[js[0]]) if len(js) else None


def _rank_key(seed, demo):
    """Stable pseudo-random sort key per (seed, demo), identical across processes."""
    return float(np.random.default_rng([9, int(seed), int(demo)]).random())


def build_balance_plan(nwbs, vids, args):
    """Map (seed, demo) -> how many times to emit that demo, to balance food categories.

    Built from a GLOBAL scan of every seed's trials table (success + min-go eligibility
    only -- no video/action read), so every parallel shard worker computes the identical
    plan regardless of its --seeds subset. Over-represented categories are downsampled to
    the target (keep the lowest-ranked `target` demos); under-represented categories are
    oversampled by duplicating demos up to the target (capped by --balance-max-oversample).
    """
    from pynwb import NWBHDF5IO
    by_cat = {}  # category -> list of (rank_key, seed, demo)
    for nwb_path in nwbs:
        seed = seed_index_of(nwb_path)
        if seed not in vids:
            continue
        with NWBHDF5IO(nwb_path, "r") as io:
            df = io.read().intervals["trials"].to_dataframe()
        n_demos = len(df) if args.max_demos_per_seed is None else min(args.max_demos_per_seed, len(df))
        for demo in range(n_demos):
            row = df.iloc[demo]
            go, stop = float(row["go_cue_time"]), float(row["stop_time"])
            if (args.success_only and not bool(row["trial_result_result"])) or (stop - go) < args.min_go_seconds:
                continue
            cat = str(row["trial_info_text_cue"])
            by_cat.setdefault(cat, []).append((_rank_key(seed, demo), seed, demo))

    counts = {c: len(v) for c, v in by_cat.items()}
    sizes = sorted(counts.values())
    if args.balance_target == "median":
        target = int(sizes[len(sizes) // 2])
    elif args.balance_target == "min":
        target = int(sizes[0])
    else:
        target = int(args.balance_target)

    plan, summary = {}, {}
    for cat, demos in by_cat.items():
        demos = sorted(demos)  # by rank_key
        n = len(demos)
        if n >= target:  # downsample: keep the first `target` by rank
            for _, seed, demo in demos[:target]:
                plan[(seed, demo)] = 1
            planned = target
        else:  # oversample: base copies for all, +1 for the lowest-ranked remainder
            base = min(target // n, args.balance_max_oversample)
            rem = min(target - base * n, n) if base < args.balance_max_oversample else 0
            for i, (_, seed, demo) in enumerate(demos):
                plan[(seed, demo)] = base + (1 if i < rem else 0)
            planned = base * n + rem
        summary[cat] = (n, planned)
    return plan, target, summary


def save_episode(ds, ep, task):
    frames, actions, states = ep
    for i in range(len(actions)):
        ds.add_frame({"observation.state": states[i], "action": actions[i], "task": task,
                      **{f"observation.images.{n}": frames[n][i] for n in CAMS}})
    ds.save_episode(parallel_encoding=False)


def main(argv=None):
    args = parse_args(argv)
    from pynwb import NWBHDF5IO

    output_root = os.path.abspath(args.output_root or os.path.join(args.raw_root, "lerobot"))
    if output_root == os.path.abspath(args.raw_root) or os.path.isdir(os.path.join(output_root, "NWB")) \
            or glob.glob(os.path.join(output_root, "videos", "*seed*")):
        sys.exit(f"REFUSING: {output_root} holds raw NWB/videos (would be clobbered). Pick another --output-root.")
    if os.path.exists(output_root):
        if not args.overwrite:
            sys.exit(f"{output_root} exists. Pass --overwrite to rebuild.")
        shutil.rmtree(output_root)

    nwbs = sorted(glob.glob(os.path.join(args.raw_root, "NWB", "*seed*.nwb")), key=seed_index_of)
    vids = {seed_index_of(d): d for d in glob.glob(os.path.join(args.raw_root, "videos", "*seed*"))}
    want_seeds = resolve_seeds(args.seeds)

    balance_plan = None
    if args.balance_objects:
        balance_plan, balance_target, balance_summary = build_balance_plan(nwbs, vids, args)
        print(f"balance plan: per-category target={balance_target} (over {len(balance_summary)} foods)")
        for cat in sorted(balance_summary, key=lambda c: -balance_summary[c][0]):
            have, planned = balance_summary[cat]
            print(f"  {cat:16s} eligible={have:4d} -> emit={planned:4d}")

    sample = glob.glob(os.path.join(next(iter(vids.values())), "demo_0_*overhead*.mp4"))[0]
    with av.open(sample) as c:
        H, W = c.streams.video[0].height, c.streams.video[0].width

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": (24,),
                              "names": {"axes": [f"proprio_{i}" for i in range(24)]}},
        "action": {"dtype": "float32", "shape": (7,),
                   "names": {"axes": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]}},
        **{f"observation.images.{n}": {"dtype": "video", "shape": (H, W, 3),
                                       "names": ["height", "width", "channels"]} for n in CAMS},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features,
                               root=output_root, robot_type="kinova_gen3", use_videos=True)

    kept = dropped = crops = balance_skipped = no_close_skipped = 0
    for nwb_path in nwbs:
        seed = seed_index_of(nwb_path)
        if (want_seeds is not None and seed not in want_seeds) or seed not in vids:
            continue
        with NWBHDF5IO(nwb_path, "r") as io:
            nwb = io.read()
            fa = np.array(nwb.acquisition["feeding_action"].data[:], dtype=np.float32)
            proprio = np.array(nwb.acquisition["feeding_robot_proprio"].data[:], dtype=np.float32)
            fa_ts = np.array(nwb.acquisition["feeding_action"].timestamps[:])
            df = nwb.intervals["trials"].to_dataframe()

        n_demos = len(df) if args.max_demos_per_seed is None else min(args.max_demos_per_seed, len(df))
        s_kept = 0
        for demo in range(n_demos):

            row = df.iloc[demo]
            go, stop = float(row["go_cue_time"]), float(row["stop_time"])
            if (args.success_only and not bool(row["trial_result_result"])) or (stop - go) < args.min_go_seconds:
                dropped += 1; continue

            # Object balancing: emit this demo `n_emit` times (0 = drop, >1 = oversample).
            n_emit = balance_plan.get((seed, demo), 0) if balance_plan is not None else 1
            if n_emit == 0:
                balance_skipped += 1; continue
            t_close = first_close_time(fa, fa_ts, go, stop)

            # A real feeding success must close the gripper
            if t_close is None and bool(row["trial_result_result"]):
                print(f"[seed {seed}] demo {demo}: success without gripper close (sim glitch), dropping")
                no_close_skipped += 1; continue
            try:
                ep = build_episode(vids[seed], demo, fa, proprio, fa_ts, go, stop, args.fps)
            except (av.FFmpegError, LookupError) as e:
                print(f"[seed {seed}] demo {demo}: undecodable video, dropping demo ({e})")
                dropped += 1; continue
            if ep is None:
                dropped += 1; continue
            task = args.task_prompt.format(food=str(row["trial_info_text_cue"]))

            ep2 = None
            if args.mid_approach_crops and t_close is not None:
                crop_go = t_close - float(np.random.default_rng([2, seed, demo]).uniform(0.3, 1.0))
                if crop_go > go and (stop - crop_go) >= args.min_go_seconds:
                    try:
                        ep2 = build_episode(vids[seed], demo, fa, proprio, fa_ts, crop_go, stop, args.fps)
                    except (av.FFmpegError, LookupError) as e:
                        print(f"[seed {seed}] demo {demo}: undecodable video, dropping crop ({e})")
                        ep2 = None

            # Emit the (already-decoded) base episode and its crop n_emit times each.
            for _ in range(n_emit):
                save_episode(ds, ep, task)
                kept += 1; s_kept += 1
                if ep2 is not None:
                    save_episode(ds, ep2, task)
                    crops += 1

        print(f"[seed {seed}] kept {s_kept}/{n_demos} demos")

    ds.finalize()
    print(f"\nDONE: {kept} base episodes + {crops} mid-approach crops kept, {dropped} dropped, "
          f"{balance_skipped} balance-subsampled out, {no_close_skipped} no-close sim glitches. "
          f"Dataset at {output_root}")
    print("finalize() computed q01/q99 stats -> pi05 QUANTILE normalization is data-driven.")


if __name__ == "__main__":
    main()
