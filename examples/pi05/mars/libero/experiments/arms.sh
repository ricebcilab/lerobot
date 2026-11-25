#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Sourced by the pilot scripts. `run_arm <arm> <depths> [flags...]` runs one arm as a depth sweep (one block per
# depth), or one block for the policy-only anchors. Every arm is configs/experiment/base.yaml plus its settings:
#
#   fc            control.mode=shared_flow_control, the corruption           --sweep control.tau=<depths>
#   frs           the corruption                                             --sweep control.n_reversal_steps=<depths>
#   mars, frs_ra  the corruption and the fixed adapter F = M                 --sweep control.n_reversal_steps=<depths>
#   mars_learned  the corruption and outputs/user_modules/<C>/user.pt        --sweep control.n_reversal_steps=<depths>
#   floor         control.mode=policy: the policy alone, with the arm's prompt
#   ceiling       the same, with the scene's instruction
#
# <C> is CORRUPTION: rotz20 (default, {rotation_z_deg: 20}) or scale ({scale: [2, 1, 1]}); a later
# --set control.corruption=... among the flags replaces it.

ARMS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${ARMS_DIR}/../run.sh"
CORRUPTION="${CORRUPTION:-rotz20}"
case "${CORRUPTION}" in
    rotz20) ARM_CORRUPTION='{rotation_z_deg: 20}' ;;
    scale) ARM_CORRUPTION='{scale: [2, 1, 1]}' ;;
    *)
        echo "Unknown CORRUPTION: ${CORRUPTION}; use rotz20 or scale" >&2
        exit 2
        ;;
esac
ARM_ADAPTER='{translation: corruption, orientation: identity, gripper: identity}'

run_arm() {
    local arm=$1 depths=$2
    shift 2
    local steered=("${RUN}" experiment --config base.yaml --set "control.corruption=${ARM_CORRUPTION}")
    case "${arm}" in
        fc)
            "${steered[@]}" --set control.mode=shared_flow_control "$@" --sweep "control.tau=${depths}"
            ;;
        frs)
            "${steered[@]}" "$@" --sweep "control.n_reversal_steps=${depths}"
            ;;
        mars | frs_ra)
            "${steered[@]}" --set "control.reversal_adapter=${ARM_ADAPTER}" "$@" \
                --sweep "control.n_reversal_steps=${depths}"
            ;;
        mars_learned)
            "${steered[@]}" --set "control.reversal_adapter={learned: outputs/user_modules/${CORRUPTION}/user.pt}" "$@" \
                --sweep "control.n_reversal_steps=${depths}"
            ;;
        floor)
            "${RUN}" experiment --config base.yaml --set control.mode=policy "$@"
            ;;
        ceiling)
            "${RUN}" experiment --config base.yaml --set control.mode=policy "$@" --set prompt=task
            ;;
        *)
            echo "Unknown arm: ${arm}; use fc, frs, mars, mars_learned, floor, ceiling" >&2
            return 2
            ;;
    esac
}
