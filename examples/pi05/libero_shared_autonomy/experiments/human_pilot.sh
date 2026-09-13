#!/usr/bin/env bash
# Human SpaceMouse study: one task, ten matched trials per method/depth.
# Task 4 and the depths (FC 2-10, reversal arms 1-8) were chosen by experiments/task_audit.sh (see README).
#   bash experiments/human_pilot.sh [--dry-run]
#   TASK=4 SEED=4 OUT=outputs/participant_01 ORDER="frs_ra fc frs" bash experiments/human_pilot.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/arms.sh"

TASK="${TASK:-4}"
N_TRIALS="${N_TRIALS:-10}"
SEED="${SEED:-4}"
FC_DEPTHS="${FC_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
FRS_DEPTHS="${FRS_DEPTHS:-${DEPTHS:-1,2,4,6,8}}"
ORDER="${ORDER:-fc frs frs_ra}"
OUT="${OUT:-outputs/pi05_libero_human_pilot_today}"
export ADAPTER_DEG="${ADAPTER_DEG:-40}"

common=(--set "scene.task_ids=[${TASK}]" --n-trials "${N_TRIALS}" --seed "${SEED}"
    --set control.max_steps=600 --output-dir "${OUT}" --port "${PORT:-8773}" "$@")

for arm in ${ORDER}; do
    depths="${FRS_DEPTHS}"
    [ "${arm}" = fc ] && depths="${FC_DEPTHS}"
    run_arm "${arm}" "${depths}" "${common[@]}"
done
echo "Human pilot output: ${OUT}"
