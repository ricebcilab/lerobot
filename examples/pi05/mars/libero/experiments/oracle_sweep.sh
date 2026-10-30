#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The study grid with the policy itself as the operator (the task-prompted pi0.5 supplies the
# command; the executing policy only sees the arm's prompt): per task, the arms in ARMS over their depths.
#
#   oracle_sweep.sh                                   all 10 LIBERO-Goal tasks, 10 matched trials per block
#   TASKS="0 2" DEPTHS=4,10 N_TRIALS=5 oracle_sweep.sh
#   ARMS="fc frs mars ceiling" oracle_sweep.sh --dry-run
#   oracle_sweep.sh --set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4
#
# Extra flags reach every run. Output goes under $OUT; notebooks/analyze.py reads it.

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/arms.sh"

OUT="${OUT:-outputs/pi05_libero_oracle_sweep_v2}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-fc frs mars floor ceiling}"
FC_DEPTHS="${FC_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
FRS_DEPTHS="${FRS_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
FRS_RA_DEPTHS="${FRS_RA_DEPTHS:-${FRS_DEPTHS}}"
N_TRIALS="${N_TRIALS:-10}"
PORT="${PORT:-8767}"
SEED="${SEED:-0}"

for task in ${TASKS}; do
    common=(--set "scene.task_ids=[${task}]" --set operator=policy --n-trials "${N_TRIALS}" --seed "${SEED}"
        --output-dir "${OUT}" --port "${PORT}" "$@")
    for arm in ${ARMS}; do
        echo "################ task ${task}: ${arm}"
        case "${arm}" in
            fc) depths="${FC_DEPTHS}" ;;
            frs) depths="${FRS_DEPTHS}" ;;
            *) depths="${FRS_RA_DEPTHS}" ;;
        esac
        run_arm "${arm}" "${depths}" "${common[@]}"
    done
done
echo "oracle sweep complete: ${OUT}"
