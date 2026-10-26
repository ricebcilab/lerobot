#!/usr/bin/env bash
# LoRA-finetune pi-0.5 on the OmniGibson feeding dataset (the "v1" recipe), or CONTINUE an
# existing adapter checkpoint on a (possibly relabelled) dataset. One script, two modes,
# selected by INIT:
#
#   INIT=lerobot/pi05_base                      (default) fresh LoRA on top of the base model
#   INIT=<run>/checkpoints/NNNNNN/pretrained_model
#        continue that adapter: lerobot loads the base named in adapter_config.json plus the
#        trained adapter (is_trainable=True) and starts a fresh optimizer/LR schedule. This is
#        a NEW run (new dataset/LR/steps allowed), unlike lerobot's --resume, which replays the
#        same run with its saved optimizer state. Use a lower LR than for a fresh adapter.
#
# Everything else is an env override; extra CLI args are passed through to lerobot-train.
#   DATASET_ROOT REPO_ID OUTPUT_DIR JOB_NAME STEPS LR DECAY_LR BATCH_SIZE SAVE_FREQ TRAIN_GPU
#   DRY_RUN=1  prints the resolved command instead of running it.
#
# Examples:
#   bash train.sh                                                   # v1 recipe, 30k steps
#   INIT=$OUT/pi05_feeding_v1_lora/checkpoints/030000/pretrained_model \
#   DATASET_ROOT=$RAW_ROOT/lerobot_v1_gp REPO_ID=rice/feeding_pi05_v1_gp \
#   OUTPUT_DIR=$OUT/pi05_feeding_v1_lora_gp STEPS=10000 LR=1e-4 bash train.sh
#                                                                   # generic-prompt continuation
#
# Recipe notes (validated 2026-08: 18/20 matched-seed rollouts vs 3/20 for the old
# expert-only finetune):
#   - LoRA r=32/alpha=64 on the PaliGemma LLM *and* action-expert attention q/v projections.
#     Adapting the LLM is what fixes language grounding (wrong-target failures); the stock pi05
#     PEFT defaults adapt only the action expert and reference stale pi0-era module names.
#   - action/time projection layers are FULLY trained (modules_to_save): the binarized gripper
#     redefines the action space, a full-rank change.
#   - LR is 10x the full-finetune default, per LeRobot's PEFT guidance.
#   - train_expert_only=false so PEFT (not the policy flags) governs freezing; SigLIP frozen.
#   - warmup/decay auto-scale to STEPS (lerobot scales the pi05 scheduler preset).
#
# Prereqs: CUDA torch + `pip install peft` in the lerobot-pi05 env; `hf auth login` with a token
# that accepted google/paligemma-3b-pt-224 and lerobot/pi05_base (or HF_HUB_OFFLINE=1 when both
# are already in the local hub cache).
#
# lerobot refuses a pre-existing OUTPUT_DIR at start, so redirect the console log elsewhere and
# move it into OUTPUT_DIR once training runs (the open handle follows the rename). The resolved
# config is saved in every checkpoint's pretrained_model/train_config.json.
#
# Checkpoints are ADAPTER-ONLY: merge with pi-finetune's scripts/merge_lora_checkpoint.py before
# rollout/deployment, and SELECT by rollout metrics (scripts/eval_checkpoints.py), not loss.
# Deployment maps the predicted binary gripper state to open/close commands (Pi05Agent).
set -euo pipefail

INIT="${INIT:-lerobot/pi05_base}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/pi-finetune/data_v1/lerobot_v1}"
REPO_ID="${REPO_ID:-rice/feeding_pi05_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/pi-finetune/outputs/pi05_feeding_v1_lora}"
JOB_NAME="${JOB_NAME:-$(basename "$OUTPUT_DIR")}"
STEPS="${STEPS:-30000}"
LR="${LR:-2.5e-4}"
DECAY_LR="${DECAY_LR:-$(awk "BEGIN{print $LR/10}")}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
TRAIN_CMD="${TRAIN_CMD:-lerobot-train}"

if [ -f "$INIT/adapter_config.json" ]; then
    # Continue a PEFT adapter checkpoint; the base model is named in adapter_config.json.
    init_args=(--policy.path="$INIT" --policy.use_peft=true)
else
    init_args=(--policy.type=pi05 --policy.pretrained_path="$INIT")
fi
if [ -n "${TRAIN_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
fi

cmd=("$TRAIN_CMD"
    --dataset.repo_id="$REPO_ID"
    --dataset.root="$DATASET_ROOT"
    "${init_args[@]}"
    --policy.push_to_hub=false
    --policy.train_expert_only=false
    --policy.freeze_vision_encoder=true
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
    --policy.device=cuda
    --policy.compile_model=false
    --policy.optimizer_lr="$LR"
    --policy.scheduler_decay_lr="$DECAY_LR"
    --peft.method_type=LORA
    --peft.r=32
    --peft.lora_alpha=64
    --peft.target_modules='(.*language_model.*self_attn\.(q|v)_proj|.*gemma_expert.*self_attn\.(q|v)_proj)'
    --peft.full_training_modules='["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --save_freq="$SAVE_FREQ"
    --job_name="$JOB_NAME"
    --output_dir="$OUTPUT_DIR"
    --wandb.enable=false
    "$@")

if [ -n "${DRY_RUN:-}" ]; then
    printf '%q ' "${cmd[@]}"; echo
    exit 0
fi
"${cmd[@]}"

# Memory: ~14.5 GB on a single 48 GB card at batch 16 (only 8.4M params train, so optimizer
# state is tiny); lower BATCH_SIZE if you OOM. Wall time: ~4-5 s/step on an RTX 6000 Ada
# (~40 h for 30k steps, ~11 h for a 10k-step continuation).
