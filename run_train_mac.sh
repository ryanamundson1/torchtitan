#!/bin/zsh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0 NGPU=1 ./run_train_mac.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=1 COMM_MODE="fake_backend" ./run_train_mac.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=1 COMM_MODE="local_tensor" ./run_train_mac.sh

NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-${MODEL:-"llama3"}}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-"0.0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [[ -n "$COMM_MODE" ]]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    # Find the correct python executable to bypass macOS zsh path_helper issues
    PYTHON_EXEC="python3"
    if [[ -n "$CONDA_PREFIX" ]]; then
        PYTHON_EXEC="$CONDA_PREFIX/bin/python3"
    elif [[ -n "$VIRTUAL_ENV" ]]; then
        PYTHON_EXEC="$VIRTUAL_ENV/bin/python3"
    fi

    # For single device macbook debugging, torchrun c10d TCPStore often gets permanently hung 
    # trying to do reverse IPv6 DNS lookups. We bypass torchrun completely for NGPU=1
    if [ "$NGPU" = "1" ]; then
        PYTORCH_ALLOC_CONF="expandable_segments:True" \
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
        MASTER_ADDR="127.0.0.1" \
        MASTER_PORT="29500" \
        GLOO_SOCKET_IFNAME="lo0" \
        WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
        $PYTHON_EXEC -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
    else
        PYTORCH_ALLOC_CONF="expandable_segments:True" \
        TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
        MASTER_ADDR="127.0.0.1" \
        MASTER_PORT="29500" \
        GLOO_SOCKET_IFNAME="lo0" \
        torchrun --nproc_per_node=${NGPU} --standalone \
        --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
    fi
fi
