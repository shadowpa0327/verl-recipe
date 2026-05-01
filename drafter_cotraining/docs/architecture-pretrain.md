# Drafter Pretrain Trainer — Architecture (compact)

Standalone EAGLE3 draft-model pretraining. Each parquet row is a full
multi-turn conversation; the target model serves only as a frozen
hidden-states source via a colocated vLLM replica.

- Entry:  `main_drafter_pretrain.py` → `trainer/pretrain_trainer.py:run_draft_model_pretrain`
- Config: `config/draft_model_pretrain_trainer.yaml`

---

## Top-level architecture

```
DraftModelPretrainTrainer (driver, CPU)
├── train_dataloader (StatefulDataLoader)
│     ParquetDrafterPretrainDataset → DrafterPretrainCollator
│     parquet rows → tokenize w/ chat-template → assistant loss-mask → DataProto[B, T_pad]
│
├── HSCollectorManager                       ← driver; colocated vLLM replicas
│     wake_up → prefill (max_tokens=1) → MooncakeHiddenStatesConnector.put → sleep
│     emits DataProto in the final schema update_drafter expects
│     (mooncake_keys, shapes, dtypes, seq_lens, loss_masks)
│
└── drafter_wg (RayWorkerGroup, GPU)         ← single role: "drafter"
    └── DrafterPretrainWorker
        └── drafter   (TrainingWorker → FSDPDrafterEngine → Eagle3Model)
                      ├── embed_tokens        (frozen, copied from target)
                      ├── verifier_norm       (frozen, target final RMSNorm)
                      ├── target_lm_head_wt   (frozen, for target distribution)
                      └── draft_model         (trainable: fc + 1 decoder layer + lm_head)
```

---

## Per-step data flow (one macro-step)

```
DrafterPretrainTrainer.fit()
    │
    │  for batch in train_dataloader:                               (driver)
    │      DataProto[B, T_pad] {input_ids, attention_mask, position_ids, loss_mask}
    │                          + non_tensor_batch{uid, seq_lens, loss_masks}
    ▼
drafter_wg.sleep()                                          drafter OFFLOADED
    │   FSDP params + optimizer state → CPU; frees GPU for vLLM collector
    ▼
HSCollectorManager.compute_hidden_states(batch)             collector AWAKE
    │   wake_up → for each sample:
    │     vLLM prefill(seq, max_tokens=1) →
    │     MooncakeHiddenStatesConnector.put(key, hidden_states, last_hs) →
    │     extract kv_transfer_params {mooncake_key, shapes, dtypes}
    │   sleep
    │   returns hs_batch: DataProto.non_tensor_batch
    │     {mooncake_keys, shapes, dtypes, seq_lens, loss_masks}
    ▼
hs_batch.meta_info["mooncake_cfg"] = mooncake_cfg                   (driver)
    │   trainer attaches connection info; no buffering, no schema rewrite —
    │   the collector already emits exactly what update_drafter consumes
    ▼
drafter_wg.wake_up()                                        drafter ON-GPU
    ▼
drafter_wg.update_drafter(hs_batch)                         mesh dispatch
    │   make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter")
    │   np.array_split → chunk[i] → rank i  (pure DP, world = dp)
    ▼
update_drafter(data)  per rank                              (GPU, per rank)
    1. drop samples with empty loss-mask (+ remove Mooncake keys)
    2. total_valid_global = all-reduce SUM(local_valid)
    3. T_pad_macro       = all-reduce MAX(local max seq_len)
       (both: torch.compile-stable divisor / pad)
    4. for mb in micro_batches(micro_batch_size_per_gpu):
         set_requires_gradient_sync(is_last_mb)             # FSDP2 grad-sync gate
         _fetch_drafter_batch_from_mooncake(mb, t_pad=T_pad_macro)
              ├── store.get(key) → ids, hidden_states, last_hs
              ├── align trainer-supplied per-token assistant loss_mask
              ├── store.remove_eagle3_tensors(key)
              └── DataCollatorWithPadding → [B, T_pad, ...]
         engine.prepare_model_inputs(batch)                 # apply verifier_norm
                                                            # to last_hs (vLLM emits pre-norm)
         plosses, _, acces = Eagle3Model(**prepared)        # 7-step TTT
         scale = mb_valid / total_valid_global * dp_size    # FSDP2 mean cancel
         (Σ 0.8^i · ploss_i) · scale → backward()
    5. optimizer_step()  +  lr_scheduler_step()
    6. all-reduce AVG plosses/acces → metrics
    ▼
trainer logs metrics, optionally _validate(), optionally _save_drafter_checkpoint()
```

The **only heavy tensor channel** is Mooncake. The DataProto carried
through Ray is metadata + `loss_masks` only — hidden states never leave
GPU memory until the producer puts them, and go straight back to GPU on
the consumer when `store.get(...)` runs.

---

## Resource layout

```
ResourcePoolManager
└── global_pool : [n_gpus_per_node] * nnodes
       │  mapping = { "drafter" : "global_pool" }
       │
       ├── drafter_wg                  ← FSDP2, world_size = dp_size
       │                                 dp_rank = dist.get_rank()
       │                                 mesh_name = "drafter"
       │
       └── HSCollectorManager replicas ← split by replica_world_size
                                         (= TP × DP × PP of inference cfg)
```

Both groups share the same physical GPUs and time-multiplex via
`drafter_wg.sleep / wake_up` and `HSCollectorManager.sleep / wake_up`.
