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

import cv2
import numpy as np

# Camera obs key (in the mp4 filename) -> short feature name used in the dataset.
CAMS = {
    "overhead": "external_overhead_cam_rgb",
    "side": "external_side_cam_rgb",
    "wrist": "kinova_kinova_gen3_bracelet_link_Camera_0_rgb",
}
GRIP = 6  # action dims [0:6] are pose deltas (summed); dim 6 is the gripper (last).


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
    """Return RGB frames at the given indices (idxs are valid: sidecar len == frame count)."""
    cap = cv2.VideoCapture(video_path)
    want, store, i, mx = set(idxs), {}, 0, max(idxs)
    while i <= mx:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            store[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
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

    c = cv2.VideoCapture(glob.glob(os.path.join(next(iter(vids.values())), "demo_0_*overhead*.mp4"))[0])
    H, W = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)); c.release()

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

    kept = dropped = 0
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
            ep = build_episode(vids[seed], demo, fa, proprio, fa_ts, go, stop, args.fps)
            if ep is None:
                dropped += 1; continue
            frames, actions, states = ep
            task = args.task_prompt.format(food=str(row["trial_info_text_cue"]))
            for i in range(len(actions)):
                ds.add_frame({"observation.state": states[i], "action": actions[i], "task": task,
                              **{f"observation.images.{n}": frames[n][i] for n in CAMS}})
            ds.save_episode(parallel_encoding=False)
            kept += 1; s_kept += 1
        print(f"[seed {seed}] kept {s_kept}/{n_demos} demos")

    ds.finalize()
    print(f"\nDONE: {kept} episodes kept, {dropped} dropped. Dataset at {output_root}")
    print("finalize() computed q01/q99 stats -> pi05 QUANTILE normalization is data-driven.")


if __name__ == "__main__":
    main()
