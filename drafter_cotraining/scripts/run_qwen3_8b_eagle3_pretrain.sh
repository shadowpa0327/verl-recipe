#!/usr/bin/env bash
# Qwen3-8B EAGLE3 drafter pretraining with canonical (messages) format.
#
# Data is expected at:
#   $DATA_DIR/canonical/  (output from preprocess_canonical.py)
#
# To preprocess data:
#   python recipe/drafter_cotraining/scripts/preprocess_canonical.py \
#       --input_dir /path/to/raw/data \
#       --output_dir /path/to/canonical

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(cd "$RECIPE_ROOT/../.." && pwd)"

# Dataset/model paths. Override with env vars.
DATA_DIR="${DATA_DIR:-/mnt/hdfs/ccchang_hldy/ultrachat_200k_tenyun_regenerate}"
CANONICAL_DIR="${CANONICAL_DIR:-$DATA_DIR/canonical}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/ccchang_hldy/Qwen3-8B}"
PYTHON_BIN="${PYTHON:-python}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"

# Loss mask mode: all_assistant_turns, last_assistant_only, auto
LOSS_MASK_MODE="${LOSS_MASK_MODE:-all_assistant_turns}"

if ! "$PYTHON_BIN" -c "import hydra, torch" >/dev/null 2>&1 && [ -x "$VERL_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$VERL_ROOT/.venv/bin/python"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
    export PATH="$PYTHON_BIN_DIR:$PATH"
fi

export HYDRA_FULL_ERROR=1

# Check canonical data exists
if [ ! -d "$CANONICAL_DIR" ]; then
    echo "ERROR: Canonical data directory not found: $CANONICAL_DIR"
    echo "Run preprocess_canonical.py first to generate canonical format data."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "WARN: MODEL_PATH is not a local dir; HF download may trigger: $MODEL_PATH"
fi

cd "$VERL_ROOT"

# Build train files list (expand glob to actual files)
mapfile -t TRAIN_FILES < <(ls "$CANONICAL_DIR"/train-*.parquet 2>/dev/null | sort)
if [ ${#TRAIN_FILES[@]} -eq 0 ]; then
    echo "ERROR: No train-*.parquet files found in $CANONICAL_DIR"
    exit 1
fi

echo "Found ${#TRAIN_FILES[@]} training parquet files in $CANONICAL_DIR"

# Build Hydra list syntax: ['file1','file2',...]
TRAIN_FILES_STR=$(printf "'%s'," "${TRAIN_FILES[@]}" | sed 's/,$//')

# Use last train file as eval if no dedicated eval file
EVAL_FILE="${TRAIN_FILES[-1]}"

"$PYTHON_BIN" -m recipe.drafter_cotraining.draft_model_pretrain_trainer \
    --config-name draft_model_pretrain_trainer \
    "data.train_files=[$TRAIN_FILES_STR]" \
    "data.eval_files=['$EVAL_FILE']" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_seq_len=8192 \
    data.loss_mask_mode="$LOSS_MASK_MODE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.drafter.optimizer_config.lr=1e-4 \
    actor_rollout_ref.drafter.optimizer_config.lr_warmup_steps_ratio=0.015 \
    actor_rollout_ref.drafter.optimizer_config.clip_grad=1.0 \
    actor_rollout_ref.drafter.model_config.attention_backend="$ATTENTION_BACKEND" \
    actor_rollout_ref.drafter.model_config.enable_lazy_target=True \
    actor_rollout_ref.drafter.engine_config.micro_batch_size_per_gpu=4 \
    hs_collector.inference.max_model_len=8448 \
    hs_collector.inference.gpu_memory_utilization=0.5 \
    pretrain.val_max_batches=-1 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=500 \
    trainer.total_epochs=5 \
    trainer.project_name='ccc_qwen3_8b_eagle3_pretrain' \
    trainer.experiment_name='qwen3_8b_ultrachat200k_canonical_all_assistant_turns' \
    trainer.val_before_train=false \
    "$@"
