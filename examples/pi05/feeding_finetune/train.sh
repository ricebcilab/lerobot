#!/usr/bin/env bash
# LoRA-finetune pi-0.5 on the OmniGibson feeding dataset (the "v1" recipe).
# Run inside the `lerobot-pi05` conda env (`pip install peft` once) after
# building the dataset with convert_nwb_to_lerobot.py / build_dataset.py
# (object-BALANCED, pixels-only; the gripper is always binarized).
#
# Recipe notes (validated 2026-08: 18/20 matched-seed rollouts vs 3/20 for the
# old expert-only finetune):
#   - LoRA r=32/alpha=64 on the PaliGemma LLM *and* action-expert attention
#     q/v projections. Adapting the LLM is what fixes language grounding
#     (wrong-target failures); the stock pi05 PEFT defaults adapt only the
#     action expert and reference stale pi0-era module names.
#   - action/time projection layers are FULLY trained (modules_to_save): the
#     binarized gripper redefines the action space, a full-rank change.
#   - LR is 10x the full-finetune default, per LeRobot's PEFT guidance.
#   - train_expert_only=false so PEFT (not the policy flags) governs freezing;
#     SigLIP stays frozen.
#
# Prereqs:
#   1. CUDA torch and `pip install peft` in the lerobot-pi05 env.
#   2. `hf auth login` with a token that has accepted the gated licenses for
#      google/paligemma-3b-pt-224 AND lerobot/pi05_base.
#
# Checkpoint SELECTION is by rollout metrics, not loss (early checkpoints have
# low loss but poor rollouts); see scripts/select_checkpoint.py in pi-finetune.
# Checkpoints are ADAPTER-ONLY: merge with scripts/merge_lora_checkpoint.py
# (pi-finetune) before rollout/deployment. Deployment maps the predicted binary
# gripper state to open/close commands (Pi05Agent in brand-rice, built-in).
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-$HOME/pi-finetune/data_v1/lerobot_v1}"
REPO_ID="${REPO_ID:-rice/feeding_pi05_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/pi-finetune/outputs/pi05_feeding_v1_lora}"

lerobot-train \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.compile_model=false \
    --policy.optimizer_lr=2.5e-4 \
    --policy.scheduler_decay_lr=2.5e-5 \
    --peft.method_type=LORA \
    --peft.r=32 \
    --peft.lora_alpha=64 \
    --peft.target_modules='(.*language_model.*self_attn\.(q|v)_proj|.*gemma_expert.*self_attn\.(q|v)_proj)' \
    --peft.full_training_modules='["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]' \
    --batch_size=16 \
    --steps=30000 \
    --save_freq=2500 \
    --job_name=pi05_feeding_v1_lora \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable=false

# Memory: ~14.5 GB on a single 48 GB card at batch 16 (only 8.4M params train,
# so optimizer state is tiny); lower --batch_size if you OOM on smaller GPUs.
# Wall time: ~4.8 s/step => ~40 h for 30k steps on an RTX 6000 Ada.
