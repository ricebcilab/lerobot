#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# The user study (docs/superpowers/specs/2026-11-14-mars-libero-user-study-design.md): one stage's runs, several
# at a time across the GPUs. Every run is `run.sh experiment` on a study_*.yaml condition with the synthetic
# operator; the arm is picked with --set control.mode=..., its depth with --sweep.
#
#   user_study.sh 0        the anchors: floor, ceiling, and FC and FRS on the clean readout, under both operator
#                          timings (the conditions' immediate one and the SpaceMouse-like 20 / 4)
#   user_study.sh prep     before stage 1, on the clean readout: FC's lambda and gain sweeps, the noise level
#                          (FRS at depth 2 over input noise), FRS at depths 1 and 4
#   NOISE=0.3 FC="tau=10,lambda=6 tau=10,lambda=8 tau=50,ic_gain=3" user_study.sh 1
#                          the baselines: FRS over its depths, FC at the settings in FC (each a comma list of
#                          control keys), and override, under the clean readout and the four users at input
#                          noise NOISE
#   user_study.sh prep --dry-run          validate every run without loading a policy
#   GPUS="0" PER_GPU=2 user_study.sh 0    two runs at a time on GPU 0
#
# Extra flags reach every run. Environment: OUT (default outputs/pi05_libero_user_study/stage<N>), GPUS (default
# "0 1"), PER_GPU (runs per GPU at a time, default 3), BASE_PORT (default 8800; each worker takes its own), SEED
# (default 0). A worker takes the next run as it finishes one. Every launch keeps its own record under
# $OUT/logs/<launch time>/: each run's log and code_state.txt (HEAD, the working tree's status and its diff); a
# --dry-run writes nothing under $OUT. A failed run is named and the others continue. notebooks/analyze.py reads
# the run folders of $OUT (one operator timing profile at a time).

set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${HERE}/../run.sh"
REPO_ROOT="$(cd -- "${HERE}/../../../../.." && pwd)"

STAGE="${1:?usage: user_study.sh STAGE [experiment flags...]}"
shift
OUT="${OUT:-outputs/pi05_libero_user_study/stage${STAGE}}"
read -r -a GPU_LIST <<<"${GPUS:-0 1}"
PER_GPU="${PER_GPU:-3}"
BASE_PORT="${BASE_PORT:-8800}"
SEED="${SEED:-0}"
SPACEMOUSE="--set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4"
FC_MODE="--set control.mode=shared_flow_control"

# A job is "<name> <experiment flags>"; no flag holds a space. Longer jobs first: workers take them in order.
JOBS=()
case "${STAGE}" in
    0)
        for suite in object goal; do
            c="study_${suite}.yaml"
            JOBS+=("${suite}_fc_immediate --config ${c} ${FC_MODE} --sweep control.tau=10,25,50")
            JOBS+=("${suite}_frs_immediate --config ${c} --sweep control.n_reversal_steps=2,6,10")
            JOBS+=("${suite}_fc --config ${c} ${FC_MODE} ${SPACEMOUSE} --sweep control.tau=10,25,50")
            JOBS+=("${suite}_frs --config ${c} ${SPACEMOUSE} --sweep control.n_reversal_steps=2,6,10")
        done
        for suite in object goal; do
            c="study_${suite}.yaml"
            JOBS+=("${suite}_floor --config ${c} --set control.mode=policy")
            JOBS+=("${suite}_ceiling --config ${c} --set control.mode=policy --set prompt=task")
        done
        ;;
    prep)
        for suite in object goal; do
            c="study_${suite}.yaml"
            JOBS+=("${suite}_fc_lambda --config ${c} ${FC_MODE} --set control.tau=10 --sweep control.lambda=2,4,6,8,10")
            JOBS+=("${suite}_frs_noise --config ${c} --set control.n_reversal_steps=2 --sweep control.input_noise=0.1,0.2,0.3,0.5")
            JOBS+=("${suite}_fc_gain --config ${c} ${FC_MODE} --sweep control.ic_gain=2,3,5")
            JOBS+=("${suite}_frs_depths --config ${c} --sweep control.n_reversal_steps=1,4")
        done
        ;;
    1)
        : "${NOISE:?stage 1 needs NOISE, the input noise the preparation step chose}"
        : "${FC:?stage 1 needs FC, the flow-control settings the preparation step chose, e.g. \"tau=10,lambda=6 tau=50,ic_gain=3\"}"
        noise="--set control.input_noise=${NOISE}"
        for suite in object goal; do
            for user in clean rotation lowdof lag keypad; do
                c="study_${suite}_${user}.yaml"
                [[ "${user}" == clean ]] && c="study_${suite}.yaml"
                JOBS+=("${suite}_${user}_frs --config ${c} ${noise} --sweep control.n_reversal_steps=1,2,4,6")
            done
        done
        for suite in object goal; do
            for user in clean rotation lowdof lag keypad; do
                c="study_${suite}_${user}.yaml"
                [[ "${user}" == clean ]] && c="study_${suite}.yaml"
                j=0
                for setting in ${FC}; do
                    JOBS+=("${suite}_${user}_fc${j} --config ${c} ${FC_MODE} ${noise} --set control.${setting//,/ --set control.}")
                    j=$((j + 1))
                done
                JOBS+=("${suite}_${user}_override --config ${c} --set control.mode=shared_override ${noise}")
            done
        done
        ;;
    *)
        echo "Stage ${STAGE} is not set up yet; 0, prep and 1 are." >&2
        exit 2
        ;;
esac

run_job() { # slot name flags...: one run on the slot's GPU and port; names a failure without stopping
    local slot=$1 name=$2
    shift 2
    local gpu=${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}
    echo "[slot ${slot}, GPU ${gpu}] ${name}"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" "${RUN}" experiment "$@" \
        --seed "${SEED}" --output-dir "${OUT}" --port "$((BASE_PORT + slot))" "${EXTRA[@]}" \
        >"${LOGS}/${name}.log" 2>&1; then
        echo "FAILED ${name} (see ${LOGS}/${name}.log)"
        return 1
    fi
}

next_job() { # the index of the next job not yet taken (nothing when none is left), one worker at a time
    (
        flock 9
        local i
        i=$(<"${QUEUE}/next")
        if ((i < ${#JOBS[@]})); then
            echo "${i}"
            echo $((i + 1)) >"${QUEUE}/next"
        fi
    ) 9>"${QUEUE}/lock"
}

EXTRA=("$@")
slots=$((${#GPU_LIST[@]} * PER_GPU))
out_dir="${OUT}"
[[ "${out_dir}" == /* ]] || out_dir="${REPO_ROOT}/${out_dir}"
if [[ " ${EXTRA[*]} " == *" --dry-run "* ]]; then
    slots=1 # nothing is loaded: one after the other, and nothing of a finished stage is touched
    LOGS="$(mktemp -d)"
else
    LOGS="${out_dir}/logs/$(date +%Y%m%d_%H%M%S)" # a launch's own record: an earlier one is never overwritten
    mkdir -p "${LOGS}"
    {
        echo "stage ${STAGE}, $(date -Iseconds), HEAD $(git -C "${REPO_ROOT}" rev-parse HEAD)"
        echo "NOISE=${NOISE:-} FC=${FC:-} SEED=${SEED} flags: ${EXTRA[*]:-}"
        git -C "${REPO_ROOT}" status --short -- src/lerobot examples/pi05/mars/libero
        git -C "${REPO_ROOT}" diff HEAD -- src/lerobot examples/pi05/mars/libero
    } >"${LOGS}/code_state.txt"
fi
QUEUE="$(mktemp -d)"
echo 0 >"${QUEUE}/next"
echo "stage ${STAGE}: ${#JOBS[@]} runs, ${slots} at a time, into ${OUT}; logs in ${LOGS}"
pids=()
for ((slot = 0; slot < slots; slot++)); do
    (
        set -f # a job's flags are split on spaces below, never matched against file names
        failed=0
        while i=$(next_job) && [[ -n "${i}" ]]; do
            # shellcheck disable=SC2086  # split on purpose
            run_job "${slot}" ${JOBS[$i]} || failed=1
        done
        exit "${failed}"
    ) &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
rm -rf "${QUEUE}"
echo "stage ${STAGE} complete: ${OUT}$([[ ${status} -eq 0 ]] || echo ' (with failed runs)')"
exit "${status}"
