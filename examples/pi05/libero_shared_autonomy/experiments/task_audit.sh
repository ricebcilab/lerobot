#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Task audit for the human pilot: which LIBERO-Goal task separates FRS-RA (F@40)
# from FC when each runs at its best depth? oracle_sweep.sh with the SpaceMouse-like
# synthetic operator (new intent every 20 environment steps, delivered 4 steps
# late) rather than the immediate oracle, since the pilot's operator is a person:
# FC over its deep optimum, native FRS and FRS-RA over their shallow one, plus the
# ceiling. Tasks 0/2/3/9 are left out: the oracle sweep scores them ~0 for every method.
#
#   task_audit.sh                       tasks 4-8, 20 matched trials per block (~3 h)
#   TASKS="4 8" N_TRIALS=10 task_audit.sh --dry-run
#
# Runs are written under $OUT (default outputs/pi05_libero_task_audit_u20d4).

export OUT="${OUT:-outputs/pi05_libero_task_audit_u20d4}"
export TASKS="${TASKS:-4 5 6 7 8}"
export ARMS="${ARMS:-fc frs frs_ra ceiling}"
export FC_DEPTHS="${FC_DEPTHS:-4,6,8,10}"
export FRS_DEPTHS="${FRS_DEPTHS:-1,2,4}"
export FRS_RA_DEPTHS="${FRS_RA_DEPTHS:-1,2,4,6}"
export N_TRIALS="${N_TRIALS:-20}"
export ADAPTER_DEG="${ADAPTER_DEG:-40}"
exec "$(dirname -- "${BASH_SOURCE[0]}")/oracle_sweep.sh" \
    --set "policy_operator.update_every_steps=${UPDATE_EVERY:-20}" --set "policy_operator.delay_steps=${DELAY:-4}" "$@"
