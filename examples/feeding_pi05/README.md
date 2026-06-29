# Feeding-task π₀.₅ finetuning

Pipeline for finetuning **π₀.₅ (action expert only)** on the OmniGibson Kinova
feeding task, from the OmniGibson NWB + mp4 collection.

- `convert_nwb_to_lerobot.py` — adapter: NWB + mp4 (+ per-frame timestamp sidecar)
  → `LeRobotDataset`. Timestamp-aligned, go-period only, success/min-length filters.
- `train_feeding.sh` — `lerobot-train` launch for pi05.

---

## Step-by-step runbook (fresh workstation)

Assumes Linux/Windows with conda, an NVIDIA GPU, and the raw collection copied to
`$RAW_ROOT` (the folder containing `NWB/` and `videos/`).

### 0. Clone the fork
```bash
git clone https://github.com/ricebcilab/lerobot.git
cd lerobot
```

### 1. Create the env and install
```bash
conda create -y -n lerobot-pi05 python=3.12
conda activate lerobot-pi05
pip install -e ".[pi,training]"   # pi = pi05 deps; training = accelerate + dataset + wandb
pip install pynwb                 # only needed to BUILD the dataset (the conversion step)
```

### 2. Install a CUDA build of torch (the default wheel is CPU-only)
```bash
# cu128 works for Blackwell (RTX 50xx / RTX PRO 6000) and Ada/Ampere too.
pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect True
```

### 3. HuggingFace auth (one-time, for the gated base model + tokenizer)
First, while logged into huggingface.co, click **Agree** on both gated pages:
- https://huggingface.co/google/paligemma-3b-pt-224  (pi05 tokenizer)
- https://huggingface.co/lerobot/pi05_base            (base checkpoint)

Then log in (token from https://huggingface.co/settings/tokens, read scope):
```bash
hf auth login
```

### 4. Build the dataset
```bash
export RAW_ROOT=/path/to/pi-finetune          # folder with NWB/ and videos/
python examples/feeding_pi05/convert_nwb_to_lerobot.py \
    --raw-root "$RAW_ROOT" \
    --fps 30
# -> writes "$RAW_ROOT/lerobot" (a subfolder; never overwrites your raw NWB/videos)
```
Smoke-test first with `--seeds 0 --max-demos-per-seed 4 --overwrite`. Other flags:
`--task-prompt`, `--min-go-seconds`, `--no-success-only`, `--seeds 0-9`, `--overwrite`.

### 5. Finetune (action expert only)
**Single GPU** (expert-only is ~18 GB, fits one RTX 6000 comfortably):
```bash
lerobot-train \
    --dataset.repo_id=rice/feeding_pi05 \
    --dataset.root="$RAW_ROOT/lerobot" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.train_expert_only=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.compile_model=false \
    --batch_size=16 \
    --steps=30000 \
    --save_freq=5000 \
    --job_name=pi05_feeding \
    --output_dir="$RAW_ROOT/outputs/pi05_feeding" \
    --wandb.enable=false
```
Or just edit and run `bash examples/feeding_pi05/train_feeding.sh`.

**Two GPUs** (optional, ~2× faster — expert-only fits per-GPU so DDP is enough):
```bash
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=rice/feeding_pi05 --dataset.root="$RAW_ROOT/lerobot" \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 --batch_size=16 --steps=30000 \
  --output_dir="$RAW_ROOT/outputs/pi05_feeding"
```
Note: effective batch = `batch_size × num_processes`; LeRobot does **not** auto-scale
the LR, so adjust it yourself if you change the GPU count. (Full finetuning instead of
expert-only on 48 GB cards needs FSDP via `accelerate config` — not required here.)

---

## Alternative: skip raw-data transfer via the Hub
Build the dataset once on the machine that has the raw data, push it, then on the
workstation just pull by `repo_id` (no `$RAW_ROOT`, no conversion, no pynwb):
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset("rice/feeding_pi05", root="$RAW_ROOT/lerobot").push_to_hub(private=True)
```
Then drop `--dataset.root` from the train command; it downloads to `HF_LEROBOT_HOME`.

---

## Data facts (verified)
- 5000 trials, 40 seeds × 125 demos, 3 cams @ **224×224**, frames frame-locked.
- Actions = per-step EEF pose deltas `[dx,dy,dz,drx,dry,drz, gripper]`; state =
  24-d proprio; task = `"Reach, grasp and bring to mouth the {food}"` (19 foods),
  set via `--task-prompt`.
- Alignment is by the **timestamp sidecar** (~4 ms to the NWB action clock); no
  reset/terminal frame-offset to manage.
- Below native fps (~38) the six pose-delta dims are **summed** per output window
  (preserves motion speed); gripper takes the window's last value. Train and deploy
  at the **same** `--fps`.
- pi05 QUANTILE normalization is data-driven: `finalize()` writes `q01/q99` stats.

## Gotchas
- 16 GB GPU: keep `train_expert_only=true` + `gradient_checkpointing=true` +
  `dtype=bfloat16`, lower `batch_size` if you OOM.
- `compile_model=false` on Windows (triton/torch.compile is unreliable there).
- On read, LeRobotDataset video decode may need `video_backend="pyav"` on Windows
  (torchcodec can't load ffmpeg libs there); on Linux torchcodec works.
