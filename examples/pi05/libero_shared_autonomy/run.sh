#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The one launcher for this folder: runs a command under `uv run` with the pi
# and libero extras, the LIBERO/MuJoCo/cache variables from env.sh, and the
# repo root as working directory.
#
#   run.sh setup                       one-time: sync the extras, configure LIBERO
#   run.sh interactive [flags]         the REPL (interactive.py)
#   run.sh experiment [flags]          scripted trials (experiment.py)
#   run.sh <command> [args...]         anything else in the environment,
#                                      e.g. run.sh lerobot-eval --env.type=libero ...

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "${REPO_ROOT}"

case "${1:-}" in
    setup)
        uv sync --locked --extra pi --extra libero --no-dev
        "${UV_RUN[@]}" python "${SCRIPT_DIR}/configure_libero.py" --config-dir "${LIBERO_CONFIG_PATH}"
        echo
        echo "Pi0.5 + LIBERO setup complete. Try: ${SCRIPT_DIR}/run.sh interactive --config"
        ;;
    interactive)
        shift
        exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/interactive.py" "$@"
        ;;
    experiment)
        shift
        exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/experiment.py" "$@"
        ;;
    "")
        echo "usage: run.sh setup | interactive [flags] | experiment [flags] | <command> [args...]" >&2
        exit 2
        ;;
    *)
        exec "${UV_RUN[@]}" "$@"
        ;;
esac
