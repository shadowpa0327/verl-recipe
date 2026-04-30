


#!/usr/bin/env bash
# Qwen3-8B EAGLE3 drafter pretraining with draft-vocab pruning.
#
# Differs from run_qwen3_8b_eagle3_pretrain.sh by passing two extra knobs:
#   - actor_rollout_ref.drafter.model_config.local_path        (template JSON
#     declaring draft_vocab_size — REQUIRED so t2d/d2t buffers exist)
#   - actor_rollout_ref.drafter.model_config.vocab_mapping_path (.pt file
#     produced by scripts/data_preprocess/build_vocab_mapping.py)
#
# Build the .pt mapping once before running this script:
#   python -m recipe.drafter_cotraining.scripts.data_preprocess.build_vocab_mapping \
#     --data_files $TRAIN_FILE \
#     --tokenizer Qwen/Qwen3-8B --chat_template qwen --max_length 4096 \
#     --target_vocab_size 151936 --draft_vocab_size 32000 \
#     --output $VOCAB_MAPPING_PATH
#
# Data and template are expected at:
#   ~/data/qwen3_8b_eagle3_10k/{train,test}.parquet
#   $DRAFT_TEMPLATE   (defaults to a /tmp file written below)
#   $VOCAB_MAPPING_PATH
#
# See claude_docs/vocab-pruning.md for the full guide.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(cd "$RECIPE_ROOT/../../.." && pwd)"

# Dataset/model paths. Override with env vars.
DATA_DIR="${DATA_DIR:-/mnt/hdfs/ccchang_hldy/ultrachat_200k_tenyun_regenerate/canonical_80k}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/ccchang_hldy/Qwen3-8B}"
PYTHON_BIN="${PYTHON:-python}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"

# Vocab-pruning knobs. Both must be set; the engine sanity-checks shape.
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
DRAFT_TEMPLATE="${DRAFT_TEMPLATE:-/tmp/qwen3_8b_draft_${DRAFT_VOCAB_SIZE}.json}"
VOCAB_MAPPING_PATH="${VOCAB_MAPPING_PATH:-/mnt/hdfs/ccchang_hldy/ultrachat_200k_tenyun_regenerate/canonical/qwen3_8b_32k.pt}"

EXPERIMENT_NAME="qwen3_8b_ultrachat200k_canonical80k_all_assistant_turns_dv32000_bs64_resume"


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

# Auto-write the draft template JSON if it doesn't exist. The only field
# that matters is draft_vocab_size; everything else is auto-derived from
# the target model's HF AutoConfig in eagle3/draft/auto.py.
if [ ! -f "$DRAFT_TEMPLATE" ]; then
    echo "Writing draft template to $DRAFT_TEMPLATE"
    cat > "$DRAFT_TEMPLATE" <<EOF
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "draft_vocab_size": $DRAFT_VOCAB_SIZE
}
EOF
fi

if [ ! -f "$VOCAB_MAPPING_PATH" ]; then
    echo "ERROR: vocab mapping file missing: $VOCAB_MAPPING_PATH"
    echo "Build it first with:"
    echo "  $PYTHON_BIN -m recipe.drafter_cotraining.scripts.data_preprocess.build_vocab_mapping \\"
    echo "    --data_files $TRAIN_FILE --tokenizer $MODEL_PATH --chat_template qwen \\"
    echo "    --max_length 4096 --target_vocab_size 151936 --draft_vocab_size $DRAFT_VOCAB_SIZE \\"
    echo "    --output $VOCAB_MAPPING_PATH"
    exit 1
fi

cd "$VERL_ROOT"
"$PYTHON_BIN" -m recipe.drafter_cotraining.main_drafter_pretrain \
    --config-name draft_model_pretrain_trainer \
    data.train_files="['$TRAIN_FILE']" \
    data.eval_files="['$VAL_FILE']" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_seq_length=4096 \
    data.chat_template=qwen \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.drafter.optimizer_config.lr=3e-4 \
    actor_rollout_ref.drafter.optimizer_config.lr_warmup_steps_ratio=0.015 \
    actor_rollout_ref.drafter.optimizer_config.clip_grad=1 \
    actor_rollout_ref.drafter.engine_config.micro_batch_size_per_gpu=8 \
    actor_rollout_ref.drafter.model_config.attention_backend="$ATTENTION_BACKEND" \
    actor_rollout_ref.drafter.model_config.local_path="$DRAFT_TEMPLATE" \
    actor_rollout_ref.drafter.model_config.vocab_mapping_path="$VOCAB_MAPPING_PATH" \
    hs_collector.inference.max_model_len=6176 \
    hs_collector.inference.gpu_memory_utilization=0.5 \
    pretrain.val_max_batches=-1 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="./ccc_qwen3_8b_eagle3_pretrain/$EXPERIMENT_NAME" \
    trainer.project_name='ccc_qwen3_8b_eagle3_pretrain' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=false \
    "$@"

