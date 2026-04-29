#!/usr/bin/env bash
# Qwen3-8B EAGLE3 drafter pretraining.
#
# Data is expected at:
#   ~/data/qwen3_8b_eagle3/train.parquet
#   ~/data/qwen3_8b_eagle3/test.parquet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(cd "$RECIPE_ROOT/../.." && pwd)"

# Dataset/model paths. Override with env vars.
DATA_DIR="${DATA_DIR:-$HOME/data/qwen3_8b_eagle3_10k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
PYTHON_BIN="${PYTHON:-python}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"

if ! "$PYTHON_BIN" -c "import hydra, torch" >/dev/null 2>&1 && [ -x "$VERL_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$VERL_ROOT/.venv/bin/python"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
    export PATH="$PYTHON_BIN_DIR:$PATH"
fi

export HYDRA_FULL_ERROR=1

for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: data file missing: $f"
        exit 1
    fi
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "WARN: MODEL_PATH is not a local dir; HF download may trigger: $MODEL_PATH"
fi

cd "$VERL_ROOT"
"$PYTHON_BIN" -m recipe.drafter_cotraining.draft_model_pretrain_trainer \
    --config-name draft_model_pretrain_trainer \
    data.train_files="['$TRAIN_FILE']" \
    data.eval_files="['$VAL_FILE']" \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_seq_length=4096 \
    data.chat_template=qwen \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.drafter.optimizer_config.lr=2.0e-4 \
    actor_rollout_ref.drafter.optimizer_config.lr_warmup_steps_ratio=0.015 \
    actor_rollout_ref.drafter.optimizer_config.clip_grad=0.5 \
    actor_rollout_ref.drafter.model_config.attention_backend="$ATTENTION_BACKEND" \
    hs_collector.inference.max_model_len=6176 \
    hs_collector.inference.gpu_memory_utilization=0.5 \
    pretrain.val_max_batches=-1 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 \
    trainer.val_before_train=false \
    "$@"
