# EAGLE3 Draft Model Training

> Co-training (RL + drafter) lives in this recipe too but is currently
> deferred. Use `main_drafter_pretrain.py` for the active path.

Architecture overview: [`docs/architecture-pretrain.md`](./docs/architecture-pretrain.md).
Mooncake setup notes (IPv6 / P2PHANDSHAKE): [`mooncake/README.md`](./mooncake/README.md).

---

## 1. Install

```bash
bash recipe/drafter_cotraining/scripts/utils/install_dependency.sh
source .venv/bin/activate
```

Pins: Python 3.12, PyTorch 2.10 (cu129), vLLM 0.18, flash-attn 2.8.3,
mooncake-transfer-engine. RDMA system libs (`libibverbs-dev`, `libnuma-dev`)
are required for Mooncake.

---

## 2. Data prep — UltraChat (Qwen3-8B target)

Download the sharded UltraChat parquet regenerated with Qwen3-8B:

```bash
huggingface-cli download shadowpa0327/qwen3_8b_eagle3-parquet \
    --repo-type dataset \
    --local-dir /path/to/ultrachat-data
```

Convert shards → canonical JSONL → `train.parquet` / `test.parquet`:

```bash
INPUT_DIR=/path/to/ultrachat-data/data \
DATA_DIR=$HOME/data/qwen3_8b_eagle3_10k \
LIMIT=10000 \
bash recipe/drafter_cotraining/scripts/data_preprocess/prepare_ultrachat_sharded_data.sh
```

Knobs: `LIMIT` rows kept (default 10k, `-1` = all), `TEST_SIZE` held-out fraction
(default 0.1), `OVERWRITE=1` re-runs all stages. Output schema is
`{id: string, conversations: list<{role, content}>}` — the canonical schema
the trainer's collator expects.

If you already have a canonical JSONL from elsewhere, skip the shard step:

```bash
INPUT=/path/to/canonical.jsonl \
DATA_DIR=$HOME/data/canonical_pretrain \
bash recipe/drafter_cotraining/scripts/data_preprocess/prepare_canonical_jsonl_data.sh
```

---

## 3. Launch training

```bash
DATA_DIR=$HOME/data/qwen3_8b_eagle3_10k \
MODEL_PATH=Qwen/Qwen3-8B \
bash recipe/drafter_cotraining/scripts/sft/qwen3_8b_eagle3.sh
```

Defaults: 4 GPUs, batch 16, seq len 4096, 5 epochs, qwen chat template,
lr 2e-4 + 1.5% warmup, FlexAttention backend. Override any of these via env
vars or extra Hydra args appended to the script (e.g. `trainer.n_gpus_per_node=8`).

What the script does:
- Spins up the vLLM HS collector replicas on a slice of the visible GPUs.
- Per macro-step: collator tokenizes a batch → HS collector prefills → tensors
  land in Mooncake → drafter worker fetches per-rank shards → TTT-loop train
  step → optimizer step. Per-token assistant loss mask is supervised across
  every assistant turn.

Checkpoints land in `trainer.default_local_dir` and write both the FSDP
sharded state and an HF-format `huggingface/{config.json, model.safetensors}`
ready for vLLM speculative decoding.

---

## 4. Optional — draft-vocab pruning

Smaller draft `lm_head` (e.g. 32k of the target's 152k vocab) for memory and
throughput. Build the mapping once:

```bash
python -m recipe.drafter_cotraining.scripts.data_preprocess.build_vocab_mapping \
    --data_files $HOME/data/qwen3_8b_eagle3_10k/train.parquet \
    --tokenizer Qwen/Qwen3-8B --chat_template qwen --max_length 4096 \
    --target_vocab_size 151936 --draft_vocab_size 32000 \
    --output $HOME/cache/vocab_mapping/qwen3_8b_32k.pt
```

Then train with the pruned variant:

```bash
DATA_DIR=$HOME/data/qwen3_8b_eagle3_10k \
MODEL_PATH=Qwen/Qwen3-8B \
VOCAB_MAPPING_PATH=$HOME/cache/vocab_mapping/qwen3_8b_32k.pt \
DRAFT_VOCAB_SIZE=32000 \
bash recipe/drafter_cotraining/scripts/sft/qwen3_8b_eagle3_vocab_prune.sh
```

The script auto-writes a draft-architecture template at `$DRAFT_TEMPLATE`
declaring `draft_vocab_size`; the rest of the architecture is auto-derived
from the target model's HF config.

---

## Tests

```bash
pytest recipe/drafter_cotraining/tests/
```

Covers the tokenizer / loss-mask pipeline and the HS-collector ↔ drafter
schema contract. Does not spin up Ray / vLLM / Mooncake.
