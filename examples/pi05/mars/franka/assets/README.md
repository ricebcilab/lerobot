# The checkpoint's statistics

`lerobot/pi05_droid` ships without normalization statistics: its processor JSONs list `features: {}`, it has
no `policy_*_step_*.safetensors` beside them, and `config.json` has no dataset statistics. The policy would
run unnormalized, and the steering core would have no action statistics, so FRS could not build a reference.

OpenPI keeps the same numbers in the checkpoint's `assets/droid/norm_stats.json`. Copy that file here:

```bash
cp <openpi-checkpoint>/assets/droid/norm_stats.json examples/pi05/mars/franka/assets/droid/norm_stats.json
```

`stats.py` converts it to LeRobot's statistics shape and installs it on both pipelines; the runner's `--stats`
defaults to this path. The file is not committed (`.gitignore`).
