#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The shared-autonomy grid with the policy itself as the operator: the
# task-prompted pi0.5 supplies the translation command, corrupted exactly like a
# SpaceMouse would be, while the executing policy only ever sees the arm's
# uninformative prompt. Per task: the arms in ARMS swept over their depths.
#
#   oracle_sweep.sh                      all 10 LIBERO-Goal tasks, 10 matched trials per block
#   TASKS="0 2" DEPTHS=4,10 N_TRIALS=5 oracle_sweep.sh
#   FC_DEPTHS=8,10 FRS_DEPTHS=2,4,6 SEED=0 oracle_sweep.sh --dry-run
#   ADAPTER_DEG=40 ARMS="fc frs frs_ra ceiling" oracle_sweep.sh   # F@40 instead of the file's F = M, no floor
#
# Extra flags go to every run, e.g. the SpaceMouse-like operator timing:
#   oracle_sweep.sh --set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4
#
# Runs are written under $OUT (default outputs/pi05_libero_oracle_sweep_v2);
# notebooks/analyze.py (load_runs, plot_metric, paired_test) reads them.

set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/arms.sh"

OUT="${OUT:-outputs/pi05_libero_oracle_sweep_v2}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-fc frs frs_ra floor ceiling}"
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
