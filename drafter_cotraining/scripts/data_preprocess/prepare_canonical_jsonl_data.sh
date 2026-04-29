#!/usr/bin/env bash
# Convert a canonical .jsonl (one row per conversation, schema:
# {"id", "conversations": [{"role", "content"}, ...]}) into the verl
# drafter-pretrain parquet layout (id + conversations + extra_info).
#
# Example:
#     INPUT=/root/TorchSpec/examples/data/ultrachat_qwen3_8b_eagle3_10k.jsonl \
#     DATA_DIR=$HOME/data/qwen3_8b_eagle3_10k \
#     bash recipe/drafter_cotraining/scripts/data_preprocess/prepare_canonical_jsonl_data.sh
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export INPUT=${INPUT:?"set INPUT=/path/to/canonical.jsonl"}
export DATA_DIR=${DATA_DIR:-"${VERL_HOME}/data/canonical_pretrain"}
export OVERWRITE=${OVERWRITE:-0}
export MAX_SAMPLES=${MAX_SAMPLES:--1}
export TEST_SIZE=${TEST_SIZE:-0.1}
export SEED=${SEED:-42}
PYTHON_BIN=${PYTHON:-python}

mkdir -p "${DATA_DIR}"

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ] || [ "${OVERWRITE}" -eq 1 ]; then
  "${PYTHON_BIN}" -m recipe.drafter_cotraining.scripts.data_preprocess.jsonl_to_parquet \
    --input "${INPUT}" \
    --local_save_dir "${DATA_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}"
fi
