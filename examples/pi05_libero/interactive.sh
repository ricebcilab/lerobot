#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Interactive LIBERO runner: same environment setup as eval.sh, but drops you
# into a REPL where you type an instruction, the policy runs one rollout, and
# you watch it live in a browser. Extra args are forwarded to interactive.py
# (e.g. ./interactive.sh --suite libero_object --task-id 2 --port 9000).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"
exec uv run --locked --extra pi --extra libero python "${SCRIPT_DIR}/interactive.py" "$@"
