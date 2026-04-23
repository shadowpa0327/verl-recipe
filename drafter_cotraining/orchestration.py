"""
Drafter co-training orchestration sketch for RayPPOTrainer.

This file shows HOW the drafter sub-pipeline integrates into
RayPPOTrainer.fit(). It's not meant to run standalone — it documents
the insertion points and call sequence.

See claude_docs/rfc-drafter-trainer-integration.md for the full design.

Integration requires modifying RayPPOTrainer to:
1. Hold a DrafterDataController
2. Call the drafter sub-pipeline after rollout, before actor training
3. Include drafter weights in update_weights()
"""


def drafter_sub_pipeline_sketch(trainer):
    """
    Shows the drafter sub-pipeline that runs inside RayPPOTrainer.fit().

    Insert this AFTER generate_sequences() and BEFORE compute_log_prob().
    The drafter sub-pipeline is a contiguous block:
        rollout → HS collection → dispatch → drafter training → actor training

    GPU timeline per RL step:
        Rollout vLLM (AWAKE)      →  generate_sequences()
        HS Collector vLLM (AWAKE) →  collect_hidden_states()
        Drafter Engine (AWAKE)    →  update_drafter()
        Actor Engine (AWAKE)      →  compute_log_prob() + update_actor()
        Weight Sync               →  actor + drafter → rollout + HS collector
    """

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXISTING: RayPPOTrainer.fit() calls generate_sequences()
    # sequences = actor_rollout_wg.generate_sequences(prompts)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── Phase 1: Rollout → raw_prompts (driver Level 1) ──────
    trainer._drafter_ctrl.push_raw_prompts(sequences)

    # ── Phase 2: raw_prompts → HS collection → sample_pool ───
    raw = trainer._drafter_ctrl.pull_raw_prompts()
    # rollout sleeps, HS collector wakes (handled inside worker)
    sample_metadata = trainer.actor_rollout_wg.collect_hidden_states(raw)
    trainer._drafter_ctrl.push_samples(sample_metadata)

    # ── Dispatch + Drafter training (before actor!) ──────────
    drafter_proto = trainer._drafter_ctrl.drain_as_dataproto()
    if drafter_proto is not None:
        # mesh dispatch splits proto per rank; each worker:
        # unpack → Mooncake.get → train_batch → Mooncake.remove
        trainer.actor_rollout_wg.update_drafter(drafter_proto)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXISTING: RayPPOTrainer.fit() continues with:
    # actor_rollout_wg.compute_log_prob(sequences)
    # compute_advantage(...)
    # actor_rollout_wg.update_actor(sequences)
    # actor_rollout_wg.update_weights()
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def init_drafter_pipeline(trainer):
    """
    Initialize drafter pipeline on RayPPOTrainer.__init__().

    Add to RayPPOTrainer.__init__:
        if config.drafter.enable:
            init_drafter_pipeline(self)
    """
    from recipe.drafter_cotraining.controller import DrafterDataController

    dp_size = trainer.config.trainer.n_gpus_per_node * trainer.config.trainer.nnodes
    trainer._drafter_ctrl = DrafterDataController(dp_size=dp_size)

    # Use ActorRolloutRefDrafterWorker instead of ActorRolloutRefWorker
    # This is configured via the worker class selection in resource_pool setup


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHERE TO MODIFY IN ray_trainer.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 1. RayPPOTrainer.__init__() — add DrafterDataController
#    Location: after self.actor_rollout_wg is created
#    Code: init_drafter_pipeline(self)
#
# 2. RayPPOTrainer.fit() — insert drafter sub-pipeline
#    Location: after generate_sequences() (line ~1380), before compute_reward()
#    Code: drafter_sub_pipeline_sketch(self)
#
# 3. Worker class selection — use ActorRolloutRefDrafterWorker
#    Location: _create_worker_group() or resource_pool config
#    Code: if config.drafter.enable: worker_cls = ActorRolloutRefDrafterWorker
#
# 4. update_weights() — already handled by ActorRolloutRefDrafterWorker.update_weights()
#    The worker class override takes care of syncing drafter → rollout
