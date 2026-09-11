#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Task audit for the human pilot: which LIBERO-Goal task separates FRS+F@40
# from FC when each runs at its best depth? The operator is the SpaceMouse-like
# synthetic one (new intent every 20 environment steps, delivered 4 steps late)
# rather than the immediate oracle, since the pilot's operator is a person.
# Per task: FC over its deep optimum, native FRS and FRS+F@40 over their shallow
# one, plus the policy-alone ceiling with the real prompt. Tasks 0/2/3/9 are
# left out: the oracle sweep scores them ~0 for every method.
#
#   task_audit.sh                       tasks 4-8, 20 matched trials per block (~3 h)
#   TASKS="4 8" N_TRIALS=10 task_audit.sh --dry-run
#
# Runs are written under $OUT (default outputs/pi05_libero_task_audit_u20d4).

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${SCRIPT_DIR}/../run.sh"

OUT="${OUT:-outputs/pi05_libero_task_audit_u20d4}"
TASKS="${TASKS:-4 5 6 7 8}"
N_TRIALS="${N_TRIALS:-20}"
SEED="${SEED:-0}"
PORT="${PORT:-8767}"
UPDATE_EVERY="${UPDATE_EVERY:-20}"
DELAY="${DELAY:-4}"
FC_DEPTHS="${FC_DEPTHS:-4,6,8,10}"
FRS_DEPTHS="${FRS_DEPTHS:-1,2,4}"
ADAPTED_DEPTHS="${ADAPTED_DEPTHS:-1,2,4,6}"
ADAPTER='control.reversal_adapter={translation: {rotation_z_deg: 40}, orientation: identity, gripper: identity}'

for task in ${TASKS}; do
    common=(--set "scene.task_ids=[${task}]" --set operator=policy
        --set "policy_operator.update_every_steps=${UPDATE_EVERY}" --set "policy_operator.delay_steps=${DELAY}"
        --n-trials "${N_TRIALS}" --seed "${SEED}" --output-dir "${OUT}" --port "${PORT}" "$@")
    echo "################ task ${task}: flow control"
    "${RUN}" experiment --config flow_control_rotz20.yaml "${common[@]}" --sweep "control.n_guided_steps=${FC_DEPTHS}"
    echo "################ task ${task}: flow reversal (native)"
    "${RUN}" experiment --config flow_reversal_rotz20.yaml "${common[@]}" --sweep "control.n_reversal_steps=${FRS_DEPTHS}"
    echo "################ task ${task}: flow reversal (adapted, F@40)"
    "${RUN}" experiment --config flow_reversal_rotz20_adapted.yaml "${common[@]}" --set "${ADAPTER}" --sweep "control.n_reversal_steps=${ADAPTED_DEPTHS}"
    echo "################ task ${task}: ceiling (policy alone, real prompt)"
    "${RUN}" experiment --config flow_reversal_rotz20.yaml "${common[@]}" --set control.mode=policy --set control.corruption=null --set prompt=task
done
echo "task audit complete: ${OUT}"
