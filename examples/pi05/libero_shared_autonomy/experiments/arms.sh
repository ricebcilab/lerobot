#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Sourced by the study scripts. `run_arm <arm> <depths> [flags...]` runs one arm as a depth sweep
# (one block per depth), or one block for the policy-only anchors:
#
#   fc       flow_control_<C>.yaml         --sweep control.n_guided_steps=<depths>
#   frs      flow_reversal_<C>.yaml        --sweep control.n_reversal_steps=<depths>
#   frs_ra   flow_reversal_ra_<C>.yaml     --sweep control.n_reversal_steps=<depths>
#   floor    policy_anchor.yaml            the policy alone, with the arm's prompt
#   ceiling  policy_anchor.yaml            the policy alone, with the scene's instruction
#
# <C> is CORRUPTION: rotz20 (default), scale, or random (a new corruption on every trial).

ARMS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="${ARMS_DIR}/../run.sh"
CORRUPTION="${CORRUPTION:-rotz20}"

run_arm() {
    local arm=$1 depths=$2
    shift 2
    case "${arm}" in
        fc)
            "${RUN}" experiment --config "flow_control_${CORRUPTION}.yaml" "$@" --sweep "control.n_guided_steps=${depths}"
            ;;
        frs)
            "${RUN}" experiment --config "flow_reversal_${CORRUPTION}.yaml" "$@" --sweep "control.n_reversal_steps=${depths}"
            ;;
        frs_ra)
            "${RUN}" experiment --config "flow_reversal_ra_${CORRUPTION}.yaml" "$@" --sweep "control.n_reversal_steps=${depths}"
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
