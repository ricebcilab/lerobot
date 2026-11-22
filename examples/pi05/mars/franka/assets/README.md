# The checkpoint's statistics

`lerobot/pi05_droid` ships without normalization statistics: its processor JSONs list `features: {}`, it has
no `policy_*_step_*.safetensors` beside them, and `config.json` has no dataset statistics. The policy would
run unnormalized, and the steering core would have no action statistics, so FRS could not build a reference.

OpenPI keeps the same numbers in the checkpoint's `assets/droid/norm_stats.json`, in the public
`openpi-assets` bucket. Downloading needs no credentials and no gsutil, because the bucket is readable over
plain HTTPS:

```bash
curl -sS -o examples/pi05/mars/franka/assets/droid/norm_stats.json \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json
```

`gsutil cp gs://openpi-assets/checkpoints/pi05_droid/assets/droid/norm_stats.json assets/droid/` does the
same if the Google Cloud SDK is already installed, and a local OpenPI checkpoint has the file at
`assets/droid/norm_stats.json`. It is 4.5 kB.

The file is 32 wide, which is the checkpoint's action feature width, not a mistake: the seven joint velocities
and the gripper occupy the first eight dims and every dim after them is zero in `q01` and `q99`. `stats.py`
converts the file to LeRobot's shape, installs it on both pipelines, and measures the meaningful width (eight),
so the runner refuses a statistics file that describes a different action space. The runner's `--stats`
defaults to this path, and the file is not committed (`.gitignore`).
