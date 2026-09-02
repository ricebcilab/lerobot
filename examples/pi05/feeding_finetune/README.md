# Feeding-task π₀.₅ finetuning

Pipeline for finetuning **π₀.₅ (action expert only)** on the OmniGibson Kinova
feeding task, from the OmniGibson NWB + mp4 collection.

- `convert_nwb_to_lerobot.py` — adapter: NWB + mp4 (+ per-frame timestamp sidecar)
  → `LeRobotDataset`. Timestamp-aligned, go-period only, success/min-length filters.
- `train.sh` — `lerobot-train` launch for pi05.

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

- https://huggingface.co/google/paligemma-3b-pt-224 (pi05 tokenizer)
- https://huggingface.co/lerobot/pi05_base (base checkpoint)

Then log in (token from https://huggingface.co/settings/tokens, read scope):

```bash
hf auth login
```

### 4. Build the dataset (object-balanced, pixels-only)

```bash
export RAW_ROOT=/path/to/pi-finetune          # folder with NWB/ and videos/
python examples/pi05/feeding_finetune/build_dataset.py \
    --raw-root "$RAW_ROOT" --fps 30 --workers 12 \
    --output-root "$RAW_ROOT/lerobot_v0" --repo-id rice/feeding_pi05_v0 \
    --balance-objects --mid-approach-crops
# -> writes "$RAW_ROOT/lerobot_v0" (a subfolder; never overwrites your raw NWB/videos)
```

`--balance-objects` equalizes episodes per food category (global, deterministic across
shards; downsample the common, oversample the rare toward the per-category median) so the
policy can't lean on an object-size close cue. `--mid-approach-crops` adds grasp-timing
decorrelation. Single-process build: `convert_nwb_to_lerobot.py` with the same flags.
Smoke-test first with `--seeds 0 --max-demos-per-seed 4 --overwrite`. Other flags:
`--task-prompt`, `--min-go-seconds`, `--no-success-only`, `--balance-target {median,min,N}`.

### 5. Finetune (LoRA on LLM + action expert — the "v1" recipe)

Edit and run `bash examples/pi05/feeding_finetune/train.sh` (canonical config,
fully commented). Key points, validated at 18/20 matched-seed rollouts vs 3/20
for the old expert-only finetune:

- **LoRA r=32/α=64 on the PaliGemma LLM _and_ action-expert attention q/v**
  (`--peft.*` flags; `pip install peft` once). Adapting the LLM fixes language
  grounding — expert-only training cannot. The stock pi05 PEFT defaults adapt
  only the expert and reference stale pi0-era module names; use the explicit
  `--peft.target_modules` regex from the script.
- **Action/time projections fully trained** (`--peft.full_training_modules`):
  the binarized gripper redefines the action space, a full-rank change.
- **LR 10×** the full-FT default (`--policy.optimizer_lr=2.5e-4`), per
  LeRobot's PEFT docs; `train_expert_only=false`, `freeze_vision_encoder=true`.
- Single GPU: only 8.4M params train, ~14.5 GB at batch 16 (~4.8 s/step,
  ~40 h / 30k steps on an RTX 6000 Ada). No DDP/FSDP needed.
- Checkpoints are **adapter-only**: merge with pi-finetune's
  `scripts/merge_lora_checkpoint.py` before rollout/deployment, and select the
  checkpoint by rollout metrics (`scripts/select_checkpoint.py`), not loss.
- Deployment maps the predicted binary gripper state to open/close commands
  with hysteresis (built into brand-rice's `Pi05Agent`).

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
  (preserves motion speed); the gripper is emitted as a latched BINARY state
  (0=open, 1=closed, blips under `--gripper-min-dwell` merged away), sampled as
  the window's last value. Train and deploy
  at the **same** `--fps`.
- pi05 QUANTILE normalization is data-driven: `finalize()` writes `q01/q99` stats.

## Gotchas

- 16 GB GPU: keep `train_expert_only=true` + `gradient_checkpointing=true` +
  `dtype=bfloat16`, lower `batch_size` if you OOM.
- `compile_model=false` on Windows (triton/torch.compile is unreliable there).
- On read, LeRobotDataset video decode may need `video_backend="pyav"` on Windows
  (torchcodec can't load ffmpeg libs there); on Linux torchcodec works.
