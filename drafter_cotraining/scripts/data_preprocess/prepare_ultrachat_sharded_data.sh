#!/usr/bin/env bash
# Ultrachat-style data prep for drafter pretraining.
#
# ============================================================================
# USAGE
# ============================================================================
#
# Step 0 (one-time, you do this yourself) — download the sharded ultrachat
# parquet from HuggingFace, e.g.:
#
#     huggingface-cli download shadowpa0327/qwen3_8b_eagle3-parquet \
#         --repo-type dataset \
#         --local-dir /path/to/ultrachat-data
#
#   (shadowpa0327/qwen3_8b_eagle3-parquet is the sharded form of
#    Tengyunw/qwen3_8b_eagle3 — ultrachat regenerated with Qwen3-8B.)
#
# Step 1 — run this script. Point INPUT_DIR at the dir containing the parquet
# shards (the `data/` subdir of the HF download, by default):
#
#     INPUT_DIR=/path/to/ultrachat-data/data \
#     DATA_DIR=$HOME/data/ultrachat_qwen3_8b_eagle3_10k LIMIT=10000 \
#         bash recipe/drafter_cotraining/scripts/data_preprocess/prepare_ultrachat_sharded_data.sh
#
# Required env:
#   INPUT_DIR   directory holding the sharded parquet shards
#
# Optional env:
#   DATA_DIR        output dir; train.parquet / test.parquet are written here
#                   (default: $VERL_HOME/data/ultrachat_qwen3_8b_eagle3)
#   JSONL_PATH      intermediate canonical JSONL path (default: $DATA_DIR/canonical.jsonl)
#   SHARD_PATTERN   glob for parquet shards inside INPUT_DIR (default: train-*.parquet)
#   LIMIT           rows kept in canonical JSONL; -1 = all (default: 10000)
#   MAX_SAMPLES     rows read by jsonl_to_parquet; -1 = all (default: -1)
#   TEST_SIZE       held-out fraction (or absolute count if >=1, default: 0.1)
#   SEED            shuffle seed (default: 42)
#   OVERWRITE       1 = re-run all stages even if outputs exist (default: 0)
#   PYTHON          python interpreter to use (default: python)
#
# ============================================================================
# PIPELINE
# ============================================================================
#   1. Sharded ultrachat parquet -> canonical JSONL
#      (recipe.drafter_cotraining.scripts.data_preprocess.ultrachat_parquet_to_jsonl)
#   2. Canonical JSONL -> {train,test}.parquet for the drafter trainer
#      (recipe.drafter_cotraining.scripts.data_preprocess.jsonl_to_parquet)
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export DATA_DIR=${DATA_DIR:-"${VERL_HOME}/data/ultrachat_qwen3_8b_eagle3"}
export INPUT_DIR=${INPUT_DIR:?"set INPUT_DIR=/path/to/sharded/parquet/dir (e.g. <repo>/data)"}
export JSONL_PATH=${JSONL_PATH:-"${DATA_DIR}/canonical.jsonl"}
export SHARD_PATTERN=${SHARD_PATTERN:-"train-*.parquet"}
export OVERWRITE=${OVERWRITE:-0}
export LIMIT=${LIMIT:-10000}             # rows kept in canonical JSONL; -1 = all
export MAX_SAMPLES=${MAX_SAMPLES:--1}     # rows read by jsonl_to_parquet; -1 = all
export TEST_SIZE=${TEST_SIZE:-0.1}
export SEED=${SEED:-42}
PYTHON_BIN=${PYTHON:-python}

mkdir -p "${DATA_DIR}"

# 1. Ultrachat parquet -> canonical JSONL (role/content).
if [ ! -f "${JSONL_PATH}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  "${PYTHON_BIN}" -m recipe.drafter_cotraining.scripts.data_preprocess.ultrachat_parquet_to_jsonl \
    --input-dir "${INPUT_DIR}" \
    --output "${JSONL_PATH}" \
    --pattern "${SHARD_PATTERN}" \
    --limit "${LIMIT}"
fi

# 2. Canonical JSONL -> train.parquet / test.parquet.
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ] || [ "${OVERWRITE}" -eq 1 ]; then
  "${PYTHON_BIN}" -m recipe.drafter_cotraining.scripts.data_preprocess.jsonl_to_parquet \
    --input "${JSONL_PATH}" \
    --local_save_dir "${DATA_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}"
fi
