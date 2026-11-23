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
#   TRANSLATION_NOISE=0.5 user_study.sh gain
#                          FC's gain, chosen once as on the cursor: on the clean readout at translation noise TRANSLATION_NOISE, gains
#                          1.5 (Kao et al.'s on LIBERO), 3, 5, 8 and 12 at tau 10 and at the whole chunk
#   TRANSLATION_NOISE=0.5 FC_GAIN=3 user_study.sh 1
#                          the baselines as their papers define them, each with its one knob tuned as on the cursor,
#                          under the clean readout and the four users at translation noise TRANSLATION_NOISE: FRS at the full reversal
#                          over the pinned prefix (Tang et al.), FC at lambda 1 over tau with a full push written at
#                          FC_GAIN sigma and rotation set to 0 (Kao et al., App. G.1), and teleop (the operator
#                          alone, as on the cursor: its translation, rotation and gripper, no policy; the
#                          rotation and the gripper carry ROTATION_NOISE and GRIPPER_NOISE, default 0.04 and
#                          1.0, the study's levels beside TRANSLATION_NOISE 0.5)
#   TRANSLATION_NOISE=0.2 user_study.sh modules
#                          stage 2, first half: one user module per user (rotation, lowdof, lag, keypad), each
#                          calibrated by train_user_module.py in the LIBERO-Goal scene on the four tasks outside
#                          Goal-6 (0, 2, 3, 9): 20 episodes of the policy under "do something", unsteered, the
#                          synthetic operator watching and pushing or resting by the let-go rule, five start states
#                          per task; that user's readout, noise TRANSLATION_NOISE drawn once per state; 3000 steps,
#                          a fifth of the states held out, into
#                          MODULES/<user>/ (default outputs/pi05_libero_user_study/user_modules). Extra flags
#                          reach the trainer (e.g. --steps 2000); it has no --dry-run
#   TRANSLATION_NOISE=0.5 user_study.sh 2
#                          stage 2, second half: MARS with each user's module on the development split, over the
#                          reversal depth (1, 2, 4, 6, 10 of the policy's 10 steps; DEPTHS picks some) x the
#                          pinned prefix (5, 10, 25, 50), Goal-6 first
#   CONTROL=1 DEPTHS="6 4 2 1" USERS="clean rotation lowdof lag keypad" TRANSLATION_NOISE=0.5 user_study.sh 2
#                          the control for MARS's depths: the same loop without a module (FRS at a partial
#                          reversal), so a gain at a shallow depth can be told from the module's
#   ARMS="frs teleop" TRANSLATION_NOISE=0.5 user_study.sh 1       only those arms (of frs, fc, teleop; fc needs FC_GAIN)
#   ARMS="frs fc teleop floor ceiling" ... user_study.sh 1        with the anchors beside them (the policy alone
#                                     under the arm's prompt and under the scene's own instruction: no user in them)
#   SUITES="goal" ... user_study.sh 1                             only that suite (of object, goal), at any stage
#   USERS="keypad" TRANSLATION_NOISE=0.5 FC_GAIN=3 user_study.sh 1    only those users (of clean, rotation, lowdof, lag, keypad)
#   SPLIT=1 ... user_study.sh 1       a run per knob value instead of one process per sweep: a few cells spread
#                                     over every worker (each run loads the policy)
#   TRANSLATION_NOISE=0.5 FC_GAIN=3 user_study.sh 1 --dry-run     validate every run without loading a policy
#   GPUS="0" PER_GPU=2 user_study.sh 0    two runs at a time on GPU 0
#
# Extra flags reach every run. Environment: OUT (default outputs/pi05_libero_user_study/stage<N>), MODULES (where
# the user modules are written and read, default outputs/pi05_libero_user_study/user_modules), GPUS (default
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
SUITES="${SUITES:-object goal}"
SPACEMOUSE="--set policy_operator.update_every_steps=20 --set policy_operator.delay_steps=4"
FC_MODE="--set control.mode=shared_flow_control"
FC_ROTATION="--set control.zeroed_dims=[rotation]" # Kao et al. on LIBERO: rotation 0 in the steered steps
MODULES="${MODULES:-outputs/pi05_libero_user_study/user_modules}" # the user modules: <user>/user.pt
[[ "${STAGE}" == modules ]] && OUT="${MODULES}" # the trainer's logs go with its modules

# A job is "<name> <experiment flags>"; no flag holds a space. Longer jobs first: workers take them in order.
JOBS=()
add_sweep() { # name key values flags...: one process running a block per value, or with SPLIT=1 a job per value
    local name=$1 key=$2 values=$3 value
    shift 3
    if [[ "${SPLIT:-0}" == 1 ]]; then
        for value in ${values//,/ }; do JOBS+=("${name}_${value} $* --set ${key}=${value}"); done
    else
        JOBS+=("${name} $* --sweep ${key}=${values}")
    fi
}
case "${STAGE}" in
    0)
        for suite in ${SUITES}; do
            c="study_${suite}.yaml"
            JOBS+=("${suite}_fc_immediate --config ${c} ${FC_MODE} --sweep control.tau=10,25,50")
            JOBS+=("${suite}_frs_immediate --config ${c} --sweep control.n_reversal_steps=2,6,10")
            JOBS+=("${suite}_fc --config ${c} ${FC_MODE} ${SPACEMOUSE} --sweep control.tau=10,25,50")
            JOBS+=("${suite}_frs --config ${c} ${SPACEMOUSE} --sweep control.n_reversal_steps=2,6,10")
        done
        for suite in ${SUITES}; do
            c="study_${suite}.yaml"
            JOBS+=("${suite}_floor --config ${c} --set control.mode=policy")
            JOBS+=("${suite}_ceiling --config ${c} --set control.mode=policy --set prompt=task")
        done
        ;;
    gain)
        : "${TRANSLATION_NOISE:?the gain step needs TRANSLATION_NOISE, the translation noise every user carries (the study uses 0.5)}"
        for gain in 1.5 3 5 8 12; do # one run each, so they spread over the workers
            for tau in 10 50; do
                for suite in ${SUITES}; do
                    JOBS+=("${suite}_gain${gain}_tau${tau} --config study_${suite}.yaml ${FC_MODE} ${FC_ROTATION} --set control.translation_noise=${TRANSLATION_NOISE} --set control.ic_gain=${gain} --set control.tau=${tau}")
                done
            done
        done
        ;;
    1)
        : "${TRANSLATION_NOISE:?stage 1 needs TRANSLATION_NOISE, the translation noise every user carries (the study uses 0.5)}"
        translation_noise="--set control.translation_noise=${TRANSLATION_NOISE}"
        ARMS="${ARMS:-frs fc teleop}"
        USERS="${USERS:-clean rotation lowdof lag keypad}"
        if [[ " ${ARMS} " == *" fc "* ]]; then
            : "${FC_GAIN:?stage 1 needs FC_GAIN for the fc arm, the gain the gain step chose}"
        fi
        fc="${FC_MODE} ${FC_ROTATION} --set control.ic_gain=${FC_GAIN:-}"
        # only teleop executes the user's rotation and gripper, so only it carries their noise
        teleop_noise="--set control.rotation_noise=${ROTATION_NOISE:-0.04} --set control.gripper_noise=${GRIPPER_NOISE:-1.0}"
        for arm in fc frs teleop; do # the longer sweeps first
            [[ " ${ARMS} " == *" ${arm} "* ]] || continue
            for suite in ${SUITES}; do
                for user in ${USERS}; do
                    c="study_${suite}_${user}.yaml"
                    [[ "${user}" == clean ]] && c="study_${suite}.yaml"
                    case "${arm}" in
                        fc) add_sweep "${suite}_${user}_fc" control.tau 5,10,20,30,40,50 --config "${c}" ${fc} ${translation_noise} ;;
                        frs) add_sweep "${suite}_${user}_frs" control.pinned_steps 5,10,25,50 --config "${c}" ${translation_noise} ;;
                        teleop) JOBS+=("${suite}_${user}_teleop --config ${c} --set control.mode=teleop ${translation_noise} ${teleop_noise}") ;;
                    esac
                done
            done
        done
        for suite in ${SUITES}; do # the anchors, on request: the policy alone, so no user and no noise in them
            if [[ " ${ARMS} " == *" floor "* ]]; then
                JOBS+=("${suite}_floor --config study_${suite}.yaml --set control.mode=policy")
            fi
            if [[ " ${ARMS} " == *" ceiling "* ]]; then
                JOBS+=("${suite}_ceiling --config study_${suite}.yaml --set control.mode=policy --set prompt=task")
            fi
        done
        ;;
    modules)
        : "${TRANSLATION_NOISE:?the modules stage needs TRANSLATION_NOISE, the noise of the readout they calibrate on}"
        for user in ${USERS:-rotation lowdof lag keypad}; do
            JOBS+=("module_${user} --config ${HERE}/../configs/experiment/study_goal_${user}.yaml --set scene.task_ids=[0,2,3,9] --out ${MODULES}/${user} --episodes 0 --open-episodes 20 --sigma ${TRANSLATION_NOISE} ${TRANSLATION_NOISE} --frozen-noise --steps 3000 --val-frac 0.2 --log-every 300")
        done
        ;;
    2)
        : "${TRANSLATION_NOISE:?stage 2 needs TRANSLATION_NOISE, the translation noise every user carries (the study uses 0.5)}"
        for suite in goal object; do # Goal-6 first: stage 1 found the gap to close there
            [[ " ${SUITES} " == *" ${suite} "* ]] || continue
            for depth in ${DEPTHS:-10 6 4 2 1}; do # the longer runs first
                for user in ${USERS:-rotation lowdof lag keypad}; do
                    c="study_${suite}_${user}.yaml"
                    [[ "${user}" == clean ]] && c="study_${suite}.yaml"
                    if [[ "${CONTROL:-0}" == 1 ]]; then # the same loop without a module: FRS at this depth
                        add_sweep "${suite}_${user}_frs_d${depth}" control.pinned_steps 5,10,25,50 --config "${c}" \
                            --set "control.n_reversal_steps=${depth}" --set "control.translation_noise=${TRANSLATION_NOISE}"
                    else
                        add_sweep "${suite}_${user}_mars_d${depth}" control.pinned_steps 5,10,25,50 --config "${c}" \
                            --set "control.reversal_adapter.learned=${MODULES}/${user}/user.pt" --set "control.n_reversal_steps=${depth}" \
                            --set "control.translation_noise=${TRANSLATION_NOISE}"
                    fi
                done
            done
        done
        ;;
    *)
        echo "Stage ${STAGE} is not set up yet; 0, gain, 1, modules and 2 are." >&2
        exit 2
        ;;
esac

run_job() { # slot name flags...: one run on the slot's GPU and port; names a failure without stopping
    local slot=$1 name=$2
    shift 2
    local gpu=${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}
    echo "[slot ${slot}, GPU ${gpu}] ${name}"
    local command=(experiment "$@" --seed "${SEED}" --output-dir "${OUT}" --port "$((BASE_PORT + slot))")
    if [[ "${STAGE}" == modules ]]; then # the trainer, on its own port (it builds a session as a run does)
        command=(python "${HERE}/../train_user_module.py" "$@" --seed "${SEED}" --set "server.port=$((BASE_PORT + slot))")
    fi
    if ! CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" "${RUN}" "${command[@]}" "${EXTRA[@]}" \
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
        echo "TRANSLATION_NOISE=${TRANSLATION_NOISE:-} ROTATION_NOISE=${ROTATION_NOISE:-0.04} GRIPPER_NOISE=${GRIPPER_NOISE:-1.0} FC_GAIN=${FC_GAIN:-} ARMS=${ARMS:-} USERS=${USERS:-} SUITES=${SUITES} DEPTHS=${DEPTHS:-} CONTROL=${CONTROL:-0} SEED=${SEED} flags: ${EXTRA[*]:-}"
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
