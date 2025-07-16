#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODE="${MODE:-aggregated}"
KIND="${KIND:-dynamo_trtllm}"
TP="${TP:-1}"
DP="${DP:-1}"
PREFILL_TP="${PREFILL_TP:-1}"
PREFILL_DP="${PREFILL_DP:-1}"
DECODE_TP="${DECODE_TP:-1}"
DECODE_DP="${DECODE_DP:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/DeepSeek-R1-FP4}"
INPUT_SEQ_LEN="${INPUT_SEQ_LEN:-3000}"
OUTPUT_SEQ_LEN="${OUTPUT_SEQ_LEN:-150}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64,128,256}"
ARTIFACTS_ROOT_DIR="${ARTIFACTS_ROOT_DIR:-/mnt/artifacts_root}"

bash -x /benchmarks/llm/perf.sh \
 --mode $MODE \
 --deployment-kind $KIND \
 --tensor-parallelism $TP \
 --data-parallelism $DP \
 --prefill-tensor-parallelism $PREFILL_TP \
 --prefill-data-parallelism $PREFILL_DP \
 --decode-tensor-parallelism $DECODE_TP \
 --decode-data-parallelism $DECODE_DP \
 --model $SERVED_MODEL_NAME \
 --input-sequence-length $INPUT_SEQ_LEN \
 --output-sequence-length $OUTPUT_SEQ_LEN \
 --url http://${HEAD_NODE_IP}:8000 \
 --concurrency $CONCURRENCY \
 --artifacts-root-dir $ARTIFACTS_ROOT_DIR