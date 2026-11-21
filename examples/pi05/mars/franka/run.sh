#!/usr/bin/env bash

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The one launcher for this folder: runs a command under `uv run` with the pi and mars extras, from the repo
# root, so the example's scripts find `lerobot` without an install step of their own.
#
#   run.sh probe [flags]                 read the checkpoint and the station, print DECISION.md's inputs
#   run.sh <arm> [flags]                 a block of trials under one arm (runner.py): teleop, floor,
#                                        ceiling, fc, frs, mars_fixed, mars_learned
#   run.sh report [flags]                one or more run directories as a table (report.py)
#   run.sh train [flags]                 a user module from a recorded dataset (train/user_module.py)
#   run.sh verify                        the offline suites this folder is answerable to
#
# c3po is not on PyPI. Install it from its source tree first (`pip install -e path/to/c3po`); every command
# except `verify` imports it.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
UV=(uv run --no-sync --extra pi --extra mars)
ARMS=(teleop floor ceiling fc frs mars_fixed mars_learned)

cd "${REPO_ROOT}"

case "${1:-}" in
    probe)
        shift
        exec "${UV[@]}" python "${SCRIPT_DIR}/probe.py" "$@"
        ;;
    report)
        shift
        exec "${UV[@]}" python "${SCRIPT_DIR}/report.py" "$@"
        ;;
    train)
        shift
        # the trainer reads a recorded LeRobot dataset, which the dataset extra brings
        exec uv run --no-sync --extra pi --extra mars --extra dataset \
            python "${SCRIPT_DIR}/train/user_module.py" "$@"
        ;;
    verify)
        exec uv run --no-sync --extra test pytest tests/examples/pi05_mars_franka tests/policies/steering -q
        ;;
    *)
        for arm in "${ARMS[@]}"; do
            if [[ "${1:-}" == "${arm}" ]]; then
                shift
                exec "${UV[@]}" python "${SCRIPT_DIR}/runner.py" --arm "${arm}" "$@"
            fi
        done
        echo "usage: run.sh probe|report|train|verify|${ARMS[*]} [flags...]" >&2
        exit 2
        ;;
esac
