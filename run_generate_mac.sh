#!/bin/zsh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -ex

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"eleos"}
CONFIG=${CONFIG:-"eleos_debugmodel"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./outputs/checkpoint/"}
PROMPT=${PROMPT:-"Write a short paragraph regarding the alignment of AI."}

overrides=()
if [ $# -ne 0 ]; then
	for arg in "$@"; do
		if [[ "$arg" == --prompt=* ]]; then
			PROMPT="${arg#--prompt=}"
            if [[ -f "$PROMPT" ]]; then
                PROMPT=$(<"$PROMPT")
            fi
		else
			overrides+=("$arg")
		fi
	done
fi

PYTHON_EXEC="python3"
if [[ -n "$CONDA_PREFIX" ]]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python3"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python3"
fi

if [ "$NGPU" = "1" ]; then
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    MASTER_ADDR="127.0.0.1" \
    MASTER_PORT="29500" \
    GLOO_SOCKET_IFNAME="lo0" \
    WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    $PYTHON_EXEC -m scripts.generate.test_generate \
        --module="${MODULE}" \
        --config="${CONFIG}" \
        --checkpoint="${CHECKPOINT_DIR}" \
        --prompt="${PROMPT}" \
        "${overrides[@]}"
else
    torchrun --standalone \
        --nproc_per_node="${NGPU}" \
        --local-ranks-filter="${LOG_RANK}" \
        -m scripts.generate.test_generate \
        --module="${MODULE}" \
        --config="${CONFIG}" \
        --checkpoint="${CHECKPOINT_DIR}" \
        --prompt="${PROMPT}" \
        "${overrides[@]}"
fi
