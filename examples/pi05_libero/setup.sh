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

cd "${REPO_ROOT}"
uv sync --locked --extra pi --extra libero --no-dev

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"
uv run --locked --extra pi --extra libero \
    python "${SCRIPT_DIR}/configure_libero.py" --config-dir "${LIBERO_CONFIG_PATH}"

echo
echo "Pi0.5 + LIBERO setup complete."
echo "Run the smoke evaluation with: ${SCRIPT_DIR}/eval.sh"
