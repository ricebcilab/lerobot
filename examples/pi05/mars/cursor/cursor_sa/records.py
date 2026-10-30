"""Trial records, block logs and run manifests.

A run is one participant's visit under one protocol:

    results/<participant>/<protocol>_s<session>/
        manifest.json               who, what, in which order, with which checkpoints
        block1_<arm>.jsonl          one record per trial, in order (practice first)
        block2_<arm>.jsonl ...

Participants are p<N> for people and sim for the expert agent. Every record
carries the arm's resolved settings, so a file can be analyzed on its own.
"""

import contextlib
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

OUTCOMES = ("hit", "wrong_target", "collision", "timeout", "aborted")
ARM_ALIASES = {"frs_ra_learned": "mars"}  # arm names in runs recorded before the rename -> today's names (files keep theirs)


def outcome_of(env, b):
    if env.success[b]:
        return "hit"
    if env.selected[b] >= 0:
        return "wrong_target"
    if env.collided[b]:
        return "collision"
    return "timeout" if env.done[b] else "aborted"


def arm_fields(arm):
    """The settings of an arm that every trial record repeats."""
    keys = ("mode", "tau", "ic_gain", "n_reverse_steps", "reversal_adapter", "speed_gain", "gain_mode", "flow_steps", "execute",
            "corruption", "input_noise", "target_radius", "min_target_sep", "reach", "dwell_s", "time_limit_s",
            "obstacle", "velocity_tau", "curl", "model")
    out = {k: arm[k] for k in keys}
    out["lambda"] = arm["lam"]  # flow control's written Euler steps (the study's records predate the knob)
    out["targets_range"] = list(arm["targets"])
    out["method"] = arm["method"]
    out["corruption_matrix"] = rounded(arm["corruption_matrix"])
    out["adapter_matrix"] = rounded(arm["adapter_matrix"]) if arm["mode"] == "frs" else None
    return out


def trial_record(outcome, arm, participant, run, protocol, block, arm_name, trial, practice, layout_id, agent):
    """A full trial record: the run context, the trial's outcome fields (from
    session.run_trials) and the arm's settings."""
    return dict(participant=participant, run=run, protocol=protocol, block=block, arm=arm_name, trial=trial,
                practice=practice, layout_id=layout_id, agent=agent, **outcome, **arm_fields(arm))


def summarize(records):
    """Outcome rates and the median time of the hits over a set of trial records."""
    n, hits = len(records), [r["duration_s"] for r in records if r["outcome"] == "hit"]
    rate = lambda outcome: sum(r["outcome"] == outcome for r in records) / n
    return dict(n=n, hits=len(hits), success=rate("hit"), wrong=rate("wrong_target"), collision=rate("collision"),
                timeout=rate("timeout"), time_s=float(np.median(hits)) if hits else None)


def rounded(a, digits=4):
    return None if a is None else np.round(np.asarray(a, dtype=float), digits).tolist()


def append_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(run_dir, manifest):
    with open(Path(run_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)


def read_manifest(run_dir):
    return json.loads((Path(run_dir) / "manifest.json").read_text())


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "."], cwd=cwd, capture_output=True, text=True)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except OSError:
        return None


def environment():
    import torch
    return dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                cuda=torch.cuda.is_available())


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def block_filename(block, arm):
    return f"block{block}_{arm}.jsonl"


def load_run(run_dir):
    """A run's manifest and its records, in block order."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    records = []
    for block, arm in enumerate(manifest["block_order"], 1):
        path = run_dir / block_filename(block, arm)
        if path.exists():
            records.extend(read_records(path))
    for r in records:  # analyses see today's arm names; the files on disk are not touched
        r["arm"] = ARM_ALIASES.get(r["arm"], r["arm"])
    manifest["block_order"] = [ARM_ALIASES.get(a, a) for a in manifest["block_order"]]
    return manifest, records


class Tee:
    """Writes to several streams at once (the console and a log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged(path, append=False):
    """Everything printed inside the block (stdout and stderr) also goes to `path`, the log.txt of the
    output folder it belongs to."""
    with open(path, "a" if append else "w") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            yield log
        finally:
            sys.stdout, sys.stderr = stdout, stderr
