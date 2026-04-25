#!/usr/bin/env bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export DATA_DIR=${DATA_DIR:-"${VERL_HOME}/data/qwen3_8b_eagle3"}
export RAW_JSON=${RAW_JSON:-"${DATA_DIR}/regenerated_complete.json"}
export OVERWRITE=${OVERWRITE:-0}
export MAX_SAMPLES=${MAX_SAMPLES:--1}
export TEST_SIZE=${TEST_SIZE:-0.1}
export SEED=${SEED:-42}

mkdir -p "${DATA_DIR}"

if [ ! -f "${RAW_JSON}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${RAW_JSON}" "https://huggingface.co/datasets/Tengyunw/qwen3_8b_eagle3/resolve/main/regenerated_complete.json?download=true"
fi

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ] || [ "${OVERWRITE}" -eq 1 ]; then
  python "${VERL_HOME}/recipe/drafter_cotraining/scripts/data_preprocess/qwen3_8b_eagle3.py" \
    --cache_path "${RAW_JSON}" \
    --local_save_dir "${DATA_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}"
fi
