#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The full shared-autonomy grid with the policy itself as the operator: the
# task-prompted pi0.5 supplies the translation command, corrupted exactly like a
# SpaceMouse would be, while the executing policy only ever sees the arm's
# uninformative prompt. Per task: the three arms swept over depth, plus two
# policy-only anchors (uninformative prompt = floor, real task = ceiling).
#
#   oracle_sweep.sh                      all 10 LIBERO-Goal tasks, 10 matched trials per block
#   TASKS="0 2" DEPTHS=4,10 N_TRIALS=5 oracle_sweep.sh
#   FC_DEPTHS=8,10 FRS_DEPTHS=2,4,6 SEED=0 oracle_sweep.sh --dry-run
#
# Runs are written under $OUT (default outputs/pi05_libero_oracle_sweep); the
# notebook notebooks/analyze_sweep.ipynb reads them after setting ROOT to $OUT.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/run.sh"

OUT="${OUT:-outputs/pi05_libero_oracle_sweep_v2}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
FC_DEPTHS="${FC_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
FRS_DEPTHS="${FRS_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
N_TRIALS="${N_TRIALS:-10}"
PORT="${PORT:-8767}"
SEED="${SEED:-0}"

for task in ${TASKS}; do
    common=(--set "scene.task_ids=[${task}]" --set operator=policy --n-trials "${N_TRIALS}" --seed "${SEED}"
        --output-dir "${OUT}" --port "${PORT}" "$@")
    echo "################ task ${task}: flow control"
    "${RUN}" experiment --config flow_control_rotz20.yaml "${common[@]}" --sweep "control.n_guided_steps=${FC_DEPTHS}"
    echo "################ task ${task}: flow reversal (native)"
    "${RUN}" experiment --config flow_reversal_rotz20.yaml "${common[@]}" --sweep "control.n_reversal_steps=${FRS_DEPTHS}"
    echo "################ task ${task}: flow reversal (adapted)"
    "${RUN}" experiment --config flow_reversal_rotz20_adapted.yaml "${common[@]}" --sweep "control.n_reversal_steps=${FRS_DEPTHS}"
    echo "################ task ${task}: anchors (policy alone)"
    "${RUN}" experiment --config flow_reversal_rotz20.yaml "${common[@]}" --set control.mode=policy --set control.corruption=null
    "${RUN}" experiment --config flow_reversal_rotz20.yaml "${common[@]}" --set control.mode=policy --set control.corruption=null --set prompt=task
done
echo "oracle sweep complete: ${OUT}"
