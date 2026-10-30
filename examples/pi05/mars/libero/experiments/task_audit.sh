#!/usr/bin/env bash

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Which LIBERO-Goal task separates MARS from FC at their best depths? oracle_sweep.sh with the
# SpaceMouse-like operator (new intent every 20 steps, delivered 4 late) on tasks 4-8; tasks 0/2/3/9
# score ~0 for every method and are left out.
#
#   task_audit.sh                       tasks 4-8, 20 matched trials per block (~3 h)
#   TASKS="4 8" N_TRIALS=10 task_audit.sh --dry-run

export OUT="${OUT:-outputs/pi05_libero_task_audit_u20d4}"
export TASKS="${TASKS:-4 5 6 7 8}"
export ARMS="${ARMS:-fc frs mars ceiling}"
export FC_DEPTHS="${FC_DEPTHS:-4,6,8,10}"
export FRS_DEPTHS="${FRS_DEPTHS:-1,2,4}"
export FRS_RA_DEPTHS="${FRS_RA_DEPTHS:-1,2,4,6}"
export N_TRIALS="${N_TRIALS:-20}"
exec "$(dirname -- "${BASH_SOURCE[0]}")/oracle_sweep.sh" \
    --set "policy_operator.update_every_steps=${UPDATE_EVERY:-20}" --set "policy_operator.delay_steps=${DELAY:-4}" "$@"
