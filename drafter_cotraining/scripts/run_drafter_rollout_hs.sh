#!/usr/bin/env bash
# Micro test: rollout + HS collection only (no drafter/actor training).
#
# Exercises RayDrafterCTPPOTrainer's rollout + HS-collection path via
# scripts/test_drafter_rollout_hs.py. Auto-launches its own mooncake_master.
#
# Override any knob with KEY=VALUE env vars, e.g.:
#   MAX_STEPS=2 BATCH_SIZE=16 ./scripts/run_drafter_rollout_hs.sh
# Extra Hydra overrides flow through "$@":
#   ./scripts/run_drafter_rollout_hs.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(cd "$RECIPE_ROOT/../.." && pwd)"

# ── Defaults (override with env vars) ─────────────────────────────────
VENV_DIR="${VENV_DIR:-$VERL_ROOT/.venv}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/gsm8k/test.parquet}"

# Qwen3-4B has 36 layers; [1, n/2-1, n-4, n-1] = [1, 17, 32, 35].
AUX_LAYER_IDS="${AUX_LAYER_IDS:-[1,17,32,35]}"

BATCH_SIZE="${BATCH_SIZE:-8}"          # must be multiple of rollout.agent.num_workers (default 8)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1024}"
HS_MAX_MODEL_LEN="${HS_MAX_MODEL_LEN:-$ROLLOUT_MAX_MODEL_LEN}"
GPU_MEM="${GPU_MEM:-0.5}"
N_GPUS="${N_GPUS:-2}"

MAX_STEPS="${MAX_STEPS:-5}"

# ── Sanity ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"; exit 1
fi
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: data file missing: $f"; exit 1
    fi
done
if [ ! -d "$MODEL_PATH" ]; then
    echo "WARN: MODEL_PATH not a local dir — HF download will trigger: $MODEL_PATH"
fi

# Kill stale processes from prior crashed runs.
# - mooncake_master: needs to free :50051 / :8090.
# - VLLM::EngineCore / VLLM::Worker: zombie workers re-advertise phantom
#   Mooncake segments to fresh masters and cause batch_put_from to fail
#   intermittently with code=-800 (TRANSFER_FAIL).
pkill -x mooncake_master  2>/dev/null || true
pkill -x VLLM::EngineCore 2>/dev/null || true
pkill -x VLLM::Worker     2>/dev/null || true
sleep 1

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$VERL_ROOT"
python "$SCRIPT_DIR/test_drafter_rollout_hs.py" \
    data.train_files="['$TRAIN_FILE']" \
    data.val_files="['$VAL_FILE']" \
    data.train_batch_size="$BATCH_SIZE" \
    data.val_batch_size="$BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    hs_collector.inference.max_model_len="$HS_MAX_MODEL_LEN" \
    hs_collector.inference.engine_kwargs.vllm.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids="$AUX_LAYER_IDS" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name=drafter_micro_test \
    trainer.experiment_name=rollout_hs_only \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    +micro.max_steps="$MAX_STEPS" \
    "$@"
