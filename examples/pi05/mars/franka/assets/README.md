# The example's assets

Three kinds of file live here, and git tracks none of their contents: `assets/.gitignore` keeps the
downloaded and generated ones out, and this file says where each comes from.

## `droid/norm_stats.json`: the checkpoint's normalization statistics

`lerobot/pi05_droid` needs openpi's DROID statistics to run. The checkpoint does not carry them: its
pre/post-processor JSONs declare `features: {}` and `config.json` has no dataset statistics, so the
policy would run unnormalized and the steering cores would have no action mean or std. openpi keeps
the same numbers beside the DROID checkpoint in its public `openpi-assets` bucket, and the bucket
reads over plain HTTPS without credentials.

Run this from the lerobot checkout:

```bash
curl -sS --create-dirs -o examples/pi05/mars/franka/assets/droid/norm_stats.json \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json
```

`gsutil cp gs://openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json assets/droid/`
does the same with the Google Cloud SDK already installed, and a local OpenPI checkout has the file
at `assets/droid/norm_stats.json`.

The file is about 4.5 kB and 32 wide, the checkpoint's padded state and action width, not a mistake.
The seven joint velocities and the gripper occupy the first eight dims; every dim after them is zero
in `q01` and `q99`. `stats.py` installs the file's `q01`/`q99` on both processor pipelines and
measures that meaningful width, so a file that describes a different action space is refused. The
probe prints the 8-wide mean and std it reads.

## `user/` and `task/`: the Phase B/C checkpoints

The user module and the task module are trained checkpoints, so they are generated here rather than
downloaded from a fixed Hub id. The config schema names them per condition and per task:

- `assets/user/<condition>/user.pt` is the user module that `[modes.mars-u] adapter_user` and the
  composed `[modes.mars]` load;
- `assets/task/<task>/task.pt` is the task module that `[modes.mars-t] adapter_task` and the composed
  `[modes.mars]` load.

Phase A ships neither. `configs/franka.toml` has no `[modes.mars-u]`, `[modes.mars-t]` or
`[modes.mars]` table, and `mars-franka mars-u` names the missing table instead of loading nothing.
The training code arrives in Phase B, and the paths above are where its output goes.
