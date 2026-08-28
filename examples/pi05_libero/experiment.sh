#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Shared-autonomy user study on LIBERO: same environment setup as interactive.sh,
# but runs the trials described by config_experiment.yaml and records each one to
# disk. Extra args are forwarded to experiment.py (e.g.
# ./experiment.sh --config my_study.yaml --n-trials 4).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"
exec uv run --locked --extra pi --extra libero python "${SCRIPT_DIR}/experiment.py" "$@"
