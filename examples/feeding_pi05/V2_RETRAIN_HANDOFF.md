# v2 Retrain Handoff — Feeding π₀.₅ (written 2026-07-10)

Instructions for the next agent working in this repo. The v2 data collection campaign is
complete, verified, and staged; your job is the **conversion + retrain** side. Read
`README.md` in this folder first for the base pipeline; this doc covers only what is new.

Upstream context lives in `/home/user/users/siyuan/robotics/pi-finetune/docs/pi05_improvement_next_steps.md`
(root cause + priorities) — the short version: the v1 policy learned to close the gripper on a
TIME schedule (~2.2 s) instead of on proximity, because demos had constant approach speed.
The v2 data breaks that correlation. Rollout-validated best v1 checkpoint is **015000**, not 30000
— training loss kept dropping while rollout success peaked at 15k. Everything below follows
from those two facts.

## What's staged (all on this machine)

`RAW_ROOT = /home/user/users/siyuan/robotics/pi-finetune/data` — `NWB/` (72 files) +
`videos/` (72 matching dirs), 125 trials each, 9000 trials total:

| seeds | graph | trials | success | notes |
|---|---|---|---|---|
| 0–39 | original OL | 5000 | ~97% | v1 data; constant speed; drinks (ids 7, 8, 16) oversampled 4x |
| 100–123 | varied_speed | 3000 | 94.8% | per-trial speed gain [0.008, 0.03]; first-close p10/med/p90 = 2.1/3.9/8.2 s (v1 was ~2.2 s constant) |
| 200–207 | recovery | 1000 | 96.8% | deliberate miss → reopen → re-approach → grasp (99.3% of trials); some have double-miss |

Quality was verified block-by-block (grasp-timing audit in
`pi-finetune/outputs/rollout_outputs/training_data_audit_v2_{new,all}/`): new-data median
distance-at-first-close is 1.13 cm (matches demos; v1 rollouts closed at 6–12 cm). The
new seeds also oversample previously-failing objects `{1,3,4,10,12,18,19,5:3x, 14:4x, 13:2x}`
with drinks at 1x. Weakest expert object: **14 (strawberry)** — 72% success in varied-speed;
its failed trials are dropped by the success filter, which is fine.

## Task 1 — Build the v2 dataset

`convert_nwb_to_lerobot.py` needs two additions before the full build:

1. **Mid-approach episode crops** (strongest time-decorrelator, applies to old AND new data):
   for each kept demo, ALSO emit a cropped episode starting 0.3–1.0 s (sample uniformly)
   before the first gripper close. First close = first frame where the gripper action
   (`feeding_action[:, 6]` in the NWB) < −0.02. Skip demos where that leaves < `--min-go-seconds`.
   Suggested flag: `--mid-approach-crops` (default off, on for the v2 build).
2. **Old-drink downweighting**: v1 data is 4x-oversampled on drinks, which taught the sloppy
   far-close habit (wide bodies forgive it). Either subsample old-seed (0–39) episodes whose
   target is a drink (ids 7, 8, 16 → categories bottle_of_coke, bottle_of_water, soda_cup)
   to ~1/4 at conversion, or add per-episode weights for the train-time sampler — converter-side
   subsampling is simpler and reproducible; pick that unless you have a reason not to.

Then build (from repo root, `lerobot-pi05` env):

```bash
export RAW_ROOT=/home/user/users/siyuan/robotics/pi-finetune/data
python examples/feeding_pi05/convert_nwb_to_lerobot.py \
    --raw-root "$RAW_ROOT" \
    --output-root "$RAW_ROOT/lerobot_v2" \
    --repo-id rice/feeding_pi05_v2 \
    --fps 30 --overwrite   # + your new flags
```

Keep `--success-only` (the default): it is REQUIRED for the recovery seeds — failed recovery
trials must not enter training. Keep fps 30 and the task prompt unchanged. Smoke-test first
(`--seeds 100,200 --max-demos-per-seed 4`) and eyeball that a recovery episode's gripper
channel shows close → reopen → close, and that crops start mid-approach.

Sanity numbers: ~8700 base episodes post-filter; with crops roughly double that.

## Task 2 — Continue the finetune from 015000

Do NOT start from `lerobot/pi05_base`. Continue from the v1 quality peak:

```bash
--policy.pretrained_path=/home/user/users/siyuan/robotics/pi-finetune/outputs/pi05_feeding/checkpoints/015000/pretrained_model
```

Deltas vs `train_feeding.sh` (which is otherwise the template — expert-only, bf16,
gradient checkpointing, batch 16):

- **Lower peak LR** — v1 used 2.5e-5 cosine→2.5e-6 from base; for continuing, start around
  **1e-5** (cosine to 1e-6).
- **`--save_freq=2500`** (roadmap says ≤5000; v1's 15k-peak was only visible because of 5k
  checkpoints — finer is safer on a shifted mixture).
- `--steps=20000` is plenty; v1 overfit past 15k. `--dataset.repo_id=rice/feeding_pi05_v2`,
  `--dataset.root="$RAW_ROOT/lerobot_v2"`, fresh `--output_dir` (don't clobber v1's
  `outputs/pi05_feeding`).
- Resolution and normalization scheme unchanged; quantile stats are recomputed automatically
  when the new dataset is finalized. v1 pace was 3.85 s/step at batch 16 (~21 h for 20k).

## Task 3 — Hand back for checkpoint selection

Selection is by ROLLOUT metrics, not loss, and runs in the pi-finetune project
(`scripts/rollout/select_checkpoint.py`, 3 trials/checkpoint sweep → 20-trial matched-seed
head-to-head; see Priority 2 of the roadmap doc for the exact command). Success criteria for
the retrain (Priority 3): median distance-at-first-close drops from ~8–10 cm toward 1.3 cm,
first-close TIME spreads out, `close_too_far` stops dominating failures, no-assist success
≥50% and grasp ≥60% over 20+ matched seeds with ≥2 non-drink successes.

## Gotchas

- The demo MP4s are **AV1** (libsvtav1) — system ffmpeg/OpenCV here cannot decode them; PyAV
  (and torchcodec on Linux) can.
- Old-data audit numbers to compare against are in
  `pi-finetune/outputs/rollout_outputs/training_data_audit/summary.json`.
- Frame alignment uses the `demo_*_timestamps.npy` sidecars; they are present in all 72 video
  dirs (verified for the new seeds).
