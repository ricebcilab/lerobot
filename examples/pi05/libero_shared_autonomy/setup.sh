#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# One-time setup: install the pi + libero extras and configure LIBERO's assets.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

cd "${REPO_ROOT}"
uv sync --locked --extra pi --extra libero --no-dev
"${UV_RUN[@]}" python "${SCRIPT_DIR}/configure_libero.py" --config-dir "${LIBERO_CONFIG_PATH}"

echo
echo "Pi0.5 + LIBERO setup complete."
echo "Run the smoke evaluation with: ${SCRIPT_DIR}/eval.sh"
