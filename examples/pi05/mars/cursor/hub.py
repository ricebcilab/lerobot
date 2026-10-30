"""The checkpoints on the Hugging Face Hub: the folders under examples/pi05/mars/checkpoints/ that the study and
the grid use, listed in hub_manifest.yaml.

    uv run --extra mars python examples/pi05/mars/cursor/hub.py download                  # fetch them (public repo, no login)
    uv run --extra mars python examples/pi05/mars/cursor/hub.py manifest                  # rewrite hub_manifest.yaml from the configs
    uv run --extra mars python examples/pi05/mars/cursor/hub.py manifest --check          # fail if hub_manifest.yaml is out of date
    uv run --extra mars python examples/pi05/mars/cursor/hub.py upload                    # maintainers: create the repo, upload

The manifest is rebuilt from the configs, never edited by hand:
- study: configs/study.yaml's arms (their policy and adapters);
- grid: every method of configs/sweeps/grid.yaml in every cell (the frozen policy, the retrained policies, the
  seed-0 user and task modules, and the user modules calibrated on the retrained policies);
- grid_seeds: the grid's training seeds 1-4, derived from each seed-0 module of the grid group by putting _s<k> in
  place of its _s0 (user_rot60_expert_n100_s0_on_wall_gap -> user_rot60_expert_n100_s3_on_wall_gap); the retrained
  policies stay one seed.
All five seeds' modules sit side by side in adapters/, every one named with its seed right after its demonstration
count (_n100 / _n300).

`download` fetches those folders from the repo (default ricebcilab/mars-cursor) into examples/pi05/mars/checkpoints/
with huggingface_hub.snapshot_download, which re-fetches any file that differs from the repo's copy; it then checks that
every folder holds the files the loaders open (config.json and model.safetensors of a policy, adapter.pt of an
adapter) and exits 1, naming them, if any are missing.
`upload` creates the public model repo if it is missing and uploads each folder's converted files (config.json and
model.safetensors of a policy, adapter.pt of an adapter, and train_log.json); the *.orig backups and train.out logs
stay local. It refuses to start if a folder is missing or holds a trainer's unconverted checkpoint: run
convert_checkpoints.py first.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT.parent / "checkpoints"  # = cursor_sa.paths.CHECKPOINTS
MANIFEST = ROOT / "hub_manifest.yaml"
REPO = "ricebcilab/mars-cursor"
SEEDS = (1, 2, 3, 4)  # the grid's training seeds besides seed 0 (each module's _s0 -> _s<k>, in adapters/)
GROUPS = ("study", "grid", "grid_seeds")
POLICY_FILES = ("config.json", "model.safetensors", "train_log.json")
ADAPTER_FILES = ("adapter.pt", "train_log.json")
# the files the loaders open (FlowPolicy.load, ReversalAdapter.load / load_adapter_module): a download must bring these
POLICY_REQUIRED = ("config.json", "model.safetensors")
ADAPTER_REQUIRED = ("adapter.pt",)
HEADER = """\
# The checkpoint folders (under examples/pi05/mars/checkpoints/) that the study and the grid use, on the Hub at
# `repo`. Written by `hub.py manifest` from configs/study.yaml, configs/sweeps/grid.yaml and the grid's seeds 1-4;
# do not edit by hand (`hub.py manifest --check` fails when this file and the configs disagree).
"""


def arm_folders(arms):
    """'policies/<name>' and 'adapters/<name>' for every checkpoint the arms load."""
    from cursor_sa import config
    arms = list(arms)
    return {f"policies/{a['model']}" for a in arms} | {f"adapters/{n}" for a in arms for n in config.adapter_names(a)}


def seed_name(name, k):
    """A seed-0 module's name for training seed k, e.g. task_ring_goal_n300_s0_with_rot60 -> task_ring_goal_n300_s2_with_rot60."""
    new, count = re.subn(r"(_n\d+)_s0(?=_|$)", rf"\1_s{k}", name, count=1)
    if not count:
        raise ValueError(f"{name}: no seed-0 tag (_n<N>_s0) to put seed {k} in place of")
    return new


def manifest_folders():
    """{group: sorted folders} rebuilt from the configs."""
    import run_sweep
    from cursor_sa import config
    protocol = config.load_protocol("study.yaml")
    study = arm_folders(config.load_arm(f) for f in protocol["arms"].values())
    sweep = run_sweep.load_sweep("grid")
    grid = arm_folders(run_sweep.method_arm(sweep, m, level) for m in sweep["methods"] for level in sweep["grid"]["levels"])
    seeds = {f"adapters/{seed_name(f.split('/', 1)[1], k)}" for f in grid if f.startswith("adapters/") for k in SEEDS}
    return {"study": sorted(study), "grid": sorted(grid), "grid_seeds": sorted(seeds)}


def manifest_text(folders):
    return HEADER + yaml.safe_dump({"repo": REPO, "folders": folders}, sort_keys=False, width=200)


def read_manifest():
    """{group: folders} as hub_manifest.yaml lists them."""
    return yaml.safe_load(MANIFEST.read_text())["folders"]


def all_folders(folders):
    return sorted({f for group in folders.values() for f in group})


def adapter_converted(path):
    """Whether an adapter.pt is in the save_adapter_module format (convert_checkpoints.py)."""
    import torch
    from convert_checkpoints import converted
    return converted(torch.load(path, map_location="cpu", weights_only=True))


def upload_files(folders):
    """The files to upload, relative to CHECKPOINTS; raises with every problem found if a folder is missing or
    unconverted."""
    files, problems = [], []
    for folder in folders:
        d = CHECKPOINTS / folder
        if folder.startswith("policies/"):
            if (d / "policy.pt").exists():
                problems.append(f"{folder}: policy.pt is unconverted (run convert_checkpoints.py)")
            wanted = POLICY_FILES
        else:
            if (d / "adapter.pt").is_file() and not adapter_converted(d / "adapter.pt"):
                problems.append(f"{folder}: adapter.pt is unconverted (run convert_checkpoints.py)")
            wanted = ADAPTER_FILES
        missing = [f for f in wanted if not (d / f).is_file()]
        if missing:
            problems.append(f"{folder}: missing {missing}")
        files += [f"{folder}/{f}" for f in wanted]
    if problems:
        raise ValueError("cannot upload:\n  " + "\n  ".join(problems))
    return files


def missing_files(folders):
    """'<folder>/<file>' for every file the loaders need that is not under CHECKPOINTS."""
    missing = []
    for folder in folders:
        wanted = POLICY_REQUIRED if folder.startswith("policies/") else ADAPTER_REQUIRED
        missing += [f"{folder}/{f}" for f in wanted if not (CHECKPOINTS / folder / f).is_file()]
    return missing


def cmd_manifest(args):
    folders = manifest_folders()
    n = len(all_folders(folders))
    if args.check:
        if not MANIFEST.is_file() or read_manifest() != folders or yaml.safe_load(MANIFEST.read_text()).get("repo") != REPO:
            print(f"{MANIFEST.name} is out of date: run `hub.py manifest` to rewrite it from the configs")
            return 1
        print(f"{MANIFEST.name} matches the configs ({n} folders)")
        return 0
    MANIFEST.write_text(manifest_text(folders))
    print(f"wrote {MANIFEST} ({n} folders: " + ", ".join(f"{g} {len(folders[g])}" for g in GROUPS) + ")")
    return 0


def cmd_download(args):
    folders = all_folders(read_manifest())
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, repo_type="model", local_dir=str(CHECKPOINTS),
                      allow_patterns=[f"{f}/*" for f in folders])
    missing = missing_files(folders)
    if missing:
        print(f"{len(missing)} files missing after the download from {args.repo}:\n  " + "\n  ".join(missing))
        return 1
    print(f"{len(folders)} checkpoint folders from {args.repo} in {CHECKPOINTS}")
    return 0


def cmd_upload(args):
    folders = all_folders(read_manifest())
    try:
        files = upload_files(folders)
    except ValueError as e:
        print(e)
        return 1
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(CHECKPOINTS), allow_patterns=files,
                      commit_message=f"MARS cursor checkpoints: {len(folders)} folders")
    print(f"uploaded {len(files)} files of {len(folders)} folders to {args.repo}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest", help="rewrite hub_manifest.yaml from the configs")
    m.add_argument("--check", action="store_true", help="only check that hub_manifest.yaml is up to date")
    for name, what in (("download", "fetch the manifest's folders into checkpoints/"),
                       ("upload", "create the Hub repo if missing and upload the manifest's folders")):
        s = sub.add_parser(name, help=what)
        s.add_argument("--repo", default=REPO, help=f"Hub model repo (default {REPO})")
    args = p.parse_args(argv)
    return {"manifest": cmd_manifest, "download": cmd_download, "upload": cmd_upload}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
