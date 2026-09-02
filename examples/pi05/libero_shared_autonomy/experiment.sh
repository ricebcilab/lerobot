#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Shared-autonomy user study on LIBERO: runs the trials described by a config
# file and records each one to disk. Extra args go to experiment.py
# (e.g. ./experiment.sh --config configs/experiment/reverse_flow_full.yaml).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "${REPO_ROOT}"
exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/experiment.py" "$@"
