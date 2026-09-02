#!/usr/bin/env bash
# Environment shared by every launcher in this folder. Source it; do not run it.
#
# Keeps LIBERO's config, the Hugging Face hub cache and Matplotlib's cache under
# the repo's ignored .cache/ directory, and selects EGL for headless MuJoCo.
# HF_HOME is left alone so an existing `hf auth login` credential still applies.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/.cache/huggingface/hub}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"

UV_RUN=(uv run --locked --extra pi --extra libero)
