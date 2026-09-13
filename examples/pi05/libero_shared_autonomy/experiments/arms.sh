#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Sourced by the study scripts: the arms of the shared-autonomy study and how
# each is launched. `run_arm <arm> <depths> [experiment flags...]` runs one
# arm as a depth sweep (one block of n_trials per depth) or, for the
# policy-only anchors, one block:
#
#   fc       flow_control_rotz20.yaml      --sweep control.n_guided_steps=<depths>
#   frs      flow_reversal_rotz20.yaml     --sweep control.n_reversal_steps=<depths>
#   frs_ra   flow_reversal_ra_rotz20.yaml  --sweep control.n_reversal_steps=<depths>
#            with the adapter rotated by ADAPTER_DEG when that is set (else the file's F = M)
#   floor    policy_anchor.yaml            the policy alone with the arm's prompt
#   ceiling  policy_anchor.yaml            the policy alone with the scene's instruction
#
# ARMS_DIR and RUN are exported for the sourcing script.

ARMS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${ARMS_DIR}/../run.sh"

run_arm() {
    local arm=$1 depths=$2
    shift 2
    case "${arm}" in
        fc)
            "${RUN}" experiment --config flow_control_rotz20.yaml "$@" --sweep "control.n_guided_steps=${depths}"
            ;;
        frs)
            "${RUN}" experiment --config flow_reversal_rotz20.yaml "$@" --sweep "control.n_reversal_steps=${depths}"
            ;;
        frs_ra)
            local adapter=()
            [ -n "${ADAPTER_DEG:-}" ] && adapter=(--set "control.reversal_adapter={translation: {rotation_z_deg: ${ADAPTER_DEG}}}")
            "${RUN}" experiment --config flow_reversal_ra_rotz20.yaml "$@" "${adapter[@]}" --sweep "control.n_reversal_steps=${depths}"
            ;;
        floor)
            "${RUN}" experiment --config policy_anchor.yaml "$@"
            ;;
        ceiling)
            "${RUN}" experiment --config policy_anchor.yaml "$@" --set prompt=task
            ;;
        *)
            echo "Unknown arm: ${arm}; use fc, frs, frs_ra, floor, ceiling" >&2
            return 2
            ;;
    esac
}
