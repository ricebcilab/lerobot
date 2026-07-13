#!/usr/bin/env bash
# Finetune pi-0.5 (action expert only) on the OmniGibson feeding dataset.
# Run inside the `lerobot-pi05` conda env after building the dataset with
# convert_nwb_to_lerobot.py (build it object-BALANCED and pixels-only:
# `--balance-objects --mid-approach-crops`). Fresh finetune from lerobot/pi05_base.
#
# Prereqs:
#   1. CUDA torch (already installed: torch 2.11.0+cu128).
#   2. `hf auth login` with a token that has accepted the gated licenses for
#      google/paligemma-3b-pt-224 AND lerobot/pi05_base.
# Checkpoint SELECTION is by rollout metrics, not loss (save_freq is small so the
# rollout peak is resolvable); see scripts/select_checkpoint.py in the pi-finetune repo.
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-D:/Robotics/results/pi-finetune/lerobot}"
REPO_ID="${REPO_ID:-rice/feeding_pi05}"
OUTPUT_DIR="${OUTPUT_DIR:-D:/Robotics/results/pi-finetune/outputs/pi05_feeding}"

lerobot-train \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.compile_model=false \
    --batch_size=16 \
    --steps=30000 \
    --save_freq=2500 \
    --job_name=pi05_feeding \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable=false

# Memory tuning for 16 GB (in rough order to try if you OOM):
#   - lower --batch_size (8 -> 4 -> 2)
#   - keep --policy.train_expert_only=true (frozen VLM => no optimizer state for it)
#   - keep --policy.gradient_checkpointing=true and --policy.dtype=bfloat16
# compile_model is left false: torch.compile/triton is unreliable on Windows.
# Train and DEPLOY at the same control rate you built the dataset with (--fps).
