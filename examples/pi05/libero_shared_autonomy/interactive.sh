#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Interactive LIBERO runner: a REPL where you type an instruction, the policy
# runs one rollout, and you watch it live in a browser. Extra args go to
# interactive.py (e.g. ./interactive.sh --suite libero_object --task-id 2).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "${REPO_ROOT}"
exec "${UV_RUN[@]}" python "${SCRIPT_DIR}/interactive.py" "$@"
