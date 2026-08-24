#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

PI05_MODEL_ID="${PI05_MODEL_ID:-lerobot/pi05_libero_finetuned}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial}"
# Use `-` (not `:-`) so callers can intentionally pass an empty value to run all tasks.
LIBERO_TASK_IDS="${LIBERO_TASK_IDS-[0]}"
LIBERO_EPISODES="${LIBERO_EPISODES:-1}"
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-1}"
PI05_ACTION_STEPS="${PI05_ACTION_STEPS:-10}"
PI05_COMPILE_MODEL="${PI05_COMPILE_MODEL:-true}"
LIBERO_OUTPUT_DIR="${LIBERO_OUTPUT_DIR:-${REPO_ROOT}/outputs/pi05_libero_eval}"

TASK_ID_ARGS=()
if [[ -n "${LIBERO_TASK_IDS}" ]]; then
    TASK_ID_ARGS+=("--env.task_ids=${LIBERO_TASK_IDS}")
fi

cd "${REPO_ROOT}"
exec uv run --locked --extra pi --extra libero lerobot-eval \
    --output_dir="${LIBERO_OUTPUT_DIR}" \
    --policy.path="${PI05_MODEL_ID}" \
    --policy.n_action_steps="${PI05_ACTION_STEPS}" \
    --policy.compile_model="${PI05_COMPILE_MODEL}" \
    --env.type=libero \
    --env.task="${LIBERO_TASKS}" \
    "${TASK_ID_ARGS[@]}" \
    --eval.batch_size="${LIBERO_BATCH_SIZE}" \
    --eval.n_episodes="${LIBERO_EPISODES}" \
    --env.max_parallel_tasks=1
