#!/usr/bin/env bash
# Human SpaceMouse study: one task, ten matched trials per method/depth.
# Task 4 and depths 1/2/4/6/8 were chosen by experiments/task_audit.sh (see README).
# bash experiments/human_pilot.sh [--dry-run]
# TASK=4 SEED=4 OUT=outputs/participant_01 bash experiments/human_pilot.sh
set -euo pipefail
STUDY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${STUDY_DIR}/../run.sh"

TASK="${TASK:-4}"
N_TRIALS="${N_TRIALS:-10}"
SEED="${SEED:-4}"
DEPTHS="${DEPTHS:-1,2,4,6,8}"
ORDER="${ORDER:-fc frs adapted}"
OUT="${OUT:-outputs/pi05_libero_human_pilot_today}"

common=(--set operator=human --set "scene.task_ids=[${TASK}]"
    --n-trials "${N_TRIALS}" --seed "${SEED}" --set experiment.task_order=sequential
    --set control.max_steps=600 --output-dir "${OUT}" --port "${PORT:-8773}")

for arm in ${ORDER}; do
    case "${arm}" in
        fc)
            config=flow_control_rotz20.yaml
            extra=(--sweep "control.n_guided_steps=${DEPTHS}")
            ;;
        frs)
            config=flow_reversal_rotz20.yaml
            extra=(--sweep "control.n_reversal_steps=${DEPTHS}")
            ;;
        adapted)
            config=flow_reversal_rotz20_adapted.yaml
            extra=(--sweep "control.n_reversal_steps=${DEPTHS}"
                --set 'control.reversal_adapter={translation: {rotation_z_deg: 40}, orientation: identity, gripper: identity}')
            ;;
        *) echo "Unknown arm: ${arm}; use fc, frs, adapted" >&2; exit 2 ;;
    esac
    "${RUN}" experiment --config "${config}" "${common[@]}" "${extra[@]}" "$@"
done
echo "Human pilot output: ${OUT}"
