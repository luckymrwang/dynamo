#/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Default values
model="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"
isl=3000
osl=150
url="http://localhost:8000"
concurrency_list="1 2 4 8 16 32 64 128 256"

print_help() {
  echo "Usage: $0 [OPTIONS]

Options:
  --model <MODEL_ID>         Model ID to be used from HuggingFace.
                             Default: ${model}

  --isl <INPUT_SEQ_LENGTH>   Input Sequence Length (ISL).
                             Default: ${isl}

  --osl <OUTPUT_SEQ_LENGTH>  Output Sequence Length (OSL).
                             Default: ${osl}

  --url <URL>                Full URL of the benchmarking endpoint.
                             Default: ${url}

  --concurrency <LEVELS>     Concurrency levels to test.
                             Accepts a single value (e.g., 64) or
                             a comma-separated list (e.g., 1,2,4,8).
                             Default: ${concurrency_list}

  --help                     Show this help message and exit.
"
}

# Function to validate if concurrency values are positive integers
validate_concurrency() {
  for val in "${concurrency_array[@]}"; do
    if ! [[ "$val" =~ ^[0-9]+$ ]] || [ "$val" -le 0 ]; then
      echo "Error: Invalid concurrency value '$val'. Must be a positive integer." >&2
      exit 1
    fi
  done
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --isl) isl="$2"; shift ;;
        --osl) osl="$2"; shift ;;
        --url) url="$2"; shift ;;
        --concurrency) IFS=',' read -r -a concurrency_array <<< "$2"; shift ;;
        --help) print_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; print_help; exit 1 ;;
    esac
    shift
done

# If concurrency_array is not set by user, use default
if [ -z "${concurrency_array+x}" ]; then
    concurrency_array=($concurrency_list)
fi

# Validate concurrency values
validate_concurrency

echo "Running genai-perf with:"
echo "Model: $model"
echo "ISL: $isl"
echo "OSL: $osl"
echo "Concurrency levels: ${concurrency_array[@]}"


for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"

  genai-perf profile \
    --model "${model}" \
    --tokenizer "${model}" \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url "${url}" \
    --synthetic-input-tokens-mean "${isl}" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "${osl}" \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:"${osl}" \
    --extra-inputs min_tokens:"${osl}" \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency "${concurrency}" \
    --request-count $(($concurrency * 10)) \
    --warmup-request-count $(($concurrency * 2)) \
    --num-dataset-entries $(($concurrency * 12)) \
    --random-seed 100 \
    -- \
    -v \
    --max-threads "${concurrency}" \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'

done
