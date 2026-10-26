#!/usr/bin/env bash
# Human SpaceMouse study: one task, matched trials per method and depth (task 4 and the depths
# come from task_audit.sh, see README). MAX_STEPS caps a trial; the pilots up to 5 October 2026 used 600.
#   TASK=4 SEED=4 OUT=outputs/participant_01 ORDER="mars fc frs" bash experiments/human_pilot.sh [--dry-run]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/arms.sh"

TASK="${TASK:-4}"
N_TRIALS="${N_TRIALS:-10}"
SEED="${SEED:-4}"
FC_DEPTHS="${FC_DEPTHS:-${DEPTHS:-2,4,6,8,10}}"
FRS_DEPTHS="${FRS_DEPTHS:-${DEPTHS:-1,2,4,6,8}}"
ORDER="${ORDER:-fc frs mars}"
MAX_STEPS="${MAX_STEPS:-300}"
OUT="${OUT:-outputs/pi05_libero_human_pilot_today}"

common=(--set "scene.task_ids=[${TASK}]" --n-trials "${N_TRIALS}" --seed "${SEED}"
    --set "control.max_steps=${MAX_STEPS}" --output-dir "${OUT}" --port "${PORT:-8773}" "$@")

for arm in ${ORDER}; do
    depths="${FRS_DEPTHS}"
    [ "${arm}" = fc ] && depths="${FC_DEPTHS}"
    run_arm "${arm}" "${depths}" "${common[@]}"
done
echo "Human pilot output: ${OUT}"
