"""Tests for Eagle3 loss computation paths.

Verifies that:
1. compiled_forward_kl_loss matches a naive reference implementation.
2. compute_target_p_padded produces correct shapes and valid probabilities
   for both pruning (t2d set) and no-pruning (t2d=None) paths.
"""

import unittest

import torch
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig

from recipe.drafter_cotraining.eagle3.draft.llama3_eagle import LlamaForCausalLMEagle3
from recipe.drafter_cotraining.eagle3.eagle3_model import (
    Eagle3Model,
    LazyTarget,
    PrecomputedTarget,
    compute_lazy_target_padded,
    compute_target_p_padded,
)
from recipe.drafter_cotraining.eagle3.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)


def _reference_forward_kl_loss(hs_flat, target_p_flat, norm_weight, lm_head_weight, norm_eps):
    """Pure-Python reference (no torch.compile) for validation."""
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(target_p_flat * log_p).sum(-1).mean()
    acc = (logits.argmax(-1) == target_p_flat.argmax(-1)).float().mean()
    return loss, acc


def _make_config(H=128, V=256, draft_V=None, num_heads=4, num_kv_heads=2):
    config = LlamaConfig(
        hidden_size=H,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=H * 4,
        max_position_embeddings=1024,
        vocab_size=V,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_scaling=None,
        pretraining_tp=1,
        pad_token_id=0,
    )
    config.draft_vocab_size = draft_V or V
    return config


def _make_model(config, length=3, attention_backend="flex_attention", device="cpu"):
    draft_model = LlamaForCausalLMEagle3(config, attention_backend=attention_backend)
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16)
    model = Eagle3Model(
        draft_model,
        length=length,
        attention_backend=attention_backend,
    )
    model.eval()
    return model


def _make_batch(B, T, H, V, device="cpu"):
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    loss_mask = torch.zeros(B, T, device=device)
    loss_mask[:, T // 4 :] = 1.0
    hidden_states = torch.randn(B, T, H * 3, device=device, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
        "target_hidden_states": target_hidden_states,
    }


class TestCompiledForwardKLLoss(unittest.TestCase):
    """compiled_forward_kl_loss should match the reference implementation."""

    def test_matches_reference(self):
        torch.manual_seed(42)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        raw_logits = F.linear(hs.float(), lm_head_weight.float())
        target_p = F.softmax(raw_logits + torch.randn_like(raw_logits) * 0.5, dim=-1)

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        ref_loss, ref_acc = _reference_forward_kl_loss(
            hs, target_p, norm_weight, lm_head_weight, norm_eps
        )

        torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(acc, ref_acc, atol=1e-3, rtol=1e-3)

    def test_perfect_prediction_equals_entropy(self):
        """When draft == target, cross-entropy loss equals target entropy."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        norm_weight = torch.ones(H, dtype=torch.float32)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        hs = torch.randn(N, H, dtype=torch.float32)
        # The loss function computes H(target, draft) = H(target) + KL(target||draft).
        # When target_p is derived from the same logits, KL ~ 0 so loss ~ H(target).
        variance = hs.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + norm_eps)
        norm_hs = hs * rstd * norm_weight
        logits = F.linear(norm_hs, lm_head_weight)
        target_p = F.softmax(logits, dim=-1)
        expected_entropy = -(target_p * target_p.log()).sum(-1).mean()

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps
        )
        torch.testing.assert_close(loss, expected_entropy, atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(acc.item(), 1.0, places=2)

    def test_loss_non_negative_and_finite(self):
        torch.manual_seed(0)
        N, H, V = 16, 64, 32
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_p = F.softmax(torch.randn(N, V), dim=-1)
        valid_idx = torch.arange(N)

        loss, acc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, 1e-6
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertGreaterEqual(acc.item(), 0.0)
        self.assertLessEqual(acc.item(), 1.0)


class TestComputeTargetPPadded(unittest.TestCase):
    """compute_target_p_padded: shape, dtype, and probability correctness."""

    def test_pruning_shapes_and_position_mask(self):
        torch.manual_seed(0)
        B, T, D = 2, 16, 64
        V_target, V_draft = 128, 32
        length = 3
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.ones(B, T)

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=length,
        )

        self.assertIsInstance(result, PrecomputedTarget)
        self.assertEqual(result.target_p_padded.shape, (B, T + length, V_draft))
        self.assertIsNotNone(result.position_mask)
        self.assertEqual(result.position_mask.shape, (B, T))
        sums = result.target_p_padded[:, :T, :].sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_loss_mask_respected_in_position_mask(self):
        """Masked positions should have position_mask == 0."""
        torch.manual_seed(0)
        B, T, D = 1, 32, 64
        V_target, V_draft = 128, 32
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.zeros(B, T)
        loss_mask[:, T // 2 :] = 1.0

        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True

        result = compute_target_p_padded(
            hs,
            weight,
            t2d=t2d,
            loss_mask=loss_mask,
            length=3,
        )

        self.assertTrue((result.position_mask[:, : T // 2] == 0).all())

    def test_no_pruning_full_vocab(self):
        """When t2d=None, project to V_full, position_mask is None, target_p is bf16."""
        torch.manual_seed(0)
        B, T, D, V = 2, 16, 64, 1000
        length = 7
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V, D, dtype=torch.bfloat16)
        loss_mask = torch.ones(B, T)

        result = compute_target_p_padded(
            hs,
            weight,
            loss_mask=loss_mask,
            length=length,
            t2d=None,
        )
        self.assertEqual(result.target_p_padded.shape, (B, T + length, V))
        self.assertIsNone(result.position_mask)
        self.assertEqual(result.target_p_padded.dtype, torch.bfloat16)
        # Probabilities sum to ~1 along the vocab dim (within bf16 precision).
        sums = result.target_p_padded[:, :T, :].sum(dim=-1).float()
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-2, rtol=1e-2)

    def test_pruning_returns_bf16(self):
        """Pruning path also stores target_p in bf16."""
        torch.manual_seed(0)
        B, T, D, V_target, V_draft = 1, 8, 32, 128, 32
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V_target, D, dtype=torch.bfloat16)
        loss_mask = torch.ones(B, T)
        t2d = torch.zeros(V_target, dtype=torch.bool)
        t2d[:V_draft] = True
        result = compute_target_p_padded(
            hs, weight, loss_mask=loss_mask, length=3, t2d=t2d,
        )
        self.assertEqual(result.target_p_padded.dtype, torch.bfloat16)


class TestRotaryConfigWiring(unittest.TestCase):
    """Model config should fully wire RoPE settings into rotary embeddings."""

    def test_yarn_uses_rope_theta_as_base(self):
        # Note: newer HuggingFace LlamaConfig normalizes rope_theta into
        # rope_scaling dict. We pass rope_theta both at top level and inside
        # rope_scaling to ensure it reaches the rotary embedding.
        config = LlamaConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=262144,
            vocab_size=256,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=50000.0,
            rope_scaling={
                "type": "yarn",
                "rope_theta": 50000.0,
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            pretraining_tp=1,
            pad_token_id=0,
        )
        config.draft_vocab_size = 256

        model = LlamaForCausalLMEagle3(config, attention_backend="flex_attention")
        rotary = model.midlayer.self_attn.rotary_emb

        self.assertEqual(rotary.base, 50000.0)
        self.assertEqual(rotary.original_max_position_embeddings, 4096)
        self.assertEqual(rotary.scaling_factor, 64.0)


def _make_mask_patterns(BT):
    """Return (name, valid_idx) pairs covering diverse masking patterns."""
    patterns = []

    # contiguous first half
    m = torch.zeros(BT, dtype=torch.bool)
    m[: BT // 2] = True
    patterns.append(("first_half", m.nonzero().squeeze(-1)))

    # contiguous second half
    m = torch.zeros(BT, dtype=torch.bool)
    m[BT // 2 :] = True
    patterns.append(("second_half", m.nonzero().squeeze(-1)))

    # every other position (strided)
    m = torch.zeros(BT, dtype=torch.bool)
    m[::2] = True
    patterns.append(("strided", m.nonzero().squeeze(-1)))

    # random sparse (~25%)
    g = torch.Generator().manual_seed(99)
    m = torch.rand(BT, generator=g) < 0.25
    patterns.append(("random_sparse", m.nonzero().squeeze(-1)))

    # single valid position
    patterns.append(("single", torch.tensor([BT // 3])))

    # all valid
    patterns.append(("all", torch.arange(BT)))

    return patterns


class TestValidIdxSubsetting(unittest.TestCase):
    """valid_idx filtering must produce the same loss as manual pre-filtering."""

    BT, H, V = 64, 128, 256

    def _check_forward_kl(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(self.BT, self.V), dim=-1)
        norm_eps = 1e-6

        loss, acc = compiled_forward_kl_loss(
            hs_flat,
            tp_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )

        hs_valid = hs_flat[valid_idx]
        tp_valid = tp_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_ref, acc_ref = compiled_forward_kl_loss(
            hs_valid,
            tp_valid,
            all_idx,
            norm_weight,
            lm_head_weight,
            norm_eps,
        )

        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(acc, acc_ref, atol=1e-5, rtol=1e-5)


# Dynamically generate one test method per mask pattern.
for _name, _vidx in _make_mask_patterns(TestValidIdxSubsetting.BT):

    def _make_kl(vidx=_vidx):
        def test(self):
            self._check_forward_kl(vidx)

        return test

    setattr(TestValidIdxSubsetting, f"test_forward_kl_{_name}", _make_kl())


class TestComputeLazyTargetPadded(unittest.TestCase):
    """compute_lazy_target_padded: shape, dtype, detach correctness."""

    def test_shapes_and_padding(self):
        torch.manual_seed(0)
        B, T, D, V = 2, 16, 64, 512
        length = 7
        hs = torch.randn(B, T, D, dtype=torch.bfloat16)
        weight = torch.randn(V, D, dtype=torch.bfloat16)

        result = compute_lazy_target_padded(
            target_hidden_states=hs,
            target_lm_head_weight=weight,
            length=length,
        )

        self.assertIsInstance(result, LazyTarget)
        self.assertEqual(result.hidden_states_padded.shape, (B, T + length, D))
        self.assertEqual(result.lm_head_weight.shape, (V, D))
        self.assertEqual(result.hidden_states_padded.dtype, torch.bfloat16)
        # Padding region is exactly zero (lazy expects zero-init padding so it
        # round-trips through the kernel cleanly when valid_idx skips it).
        torch.testing.assert_close(
            result.hidden_states_padded[:, T:, :],
            torch.zeros(B, length, D, dtype=torch.bfloat16),
        )

    def test_no_grad_lineage_even_when_inputs_require_grad(self):
        """Even if a future caller passes grad-tracked inputs, the output must
        be detached so the compiled lazy kernel never builds an autograd
        graph through the target side."""
        torch.manual_seed(0)
        B, T, D, V = 1, 8, 32, 64
        # Mark inputs as grad-tracked to simulate a regression where someone
        # forgot to .detach() upstream.
        hs = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(V, D, dtype=torch.float32, requires_grad=True)

        result = compute_lazy_target_padded(
            target_hidden_states=hs,
            target_lm_head_weight=weight,
            length=3,
        )

        self.assertFalse(result.hidden_states_padded.requires_grad)
        self.assertFalse(result.lm_head_weight.requires_grad)
        self.assertIsNone(result.hidden_states_padded.grad_fn)
        self.assertIsNone(result.lm_head_weight.grad_fn)


class TestCompiledForwardKLLossFromHS(unittest.TestCase):
    """compiled_forward_kl_loss_from_hs (lazy kernel) parity tests."""

    def test_matches_precomputed_path_on_identical_inputs(self):
        """Both kernels should produce the same loss when given the same
        (target_hidden_states, target_lm_head_weight) — modulo small float
        noise from bf16 vs fp32 softmax intermediates and one extra cast."""
        torch.manual_seed(13)
        N, H, V = 32, 128, 256
        hs = torch.randn(N, H, dtype=torch.bfloat16)
        ths = torch.randn(N, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        # Build target_p the same way the precomputed factory does (fp32 softmax,
        # then bf16-cast to mirror compute_target_p_padded's storage choice).
        with torch.no_grad():
            target_logits = F.linear(ths, target_lm_head_weight)
            target_p = F.softmax(target_logits.float(), dim=-1).to(torch.bfloat16)

        loss_pc, acc_pc = compiled_forward_kl_loss(
            hs, target_p, valid_idx, norm_weight, lm_head_weight, norm_eps,
        )
        loss_lz, acc_lz = compiled_forward_kl_loss_from_hs(
            hs, ths, valid_idx, norm_weight, lm_head_weight,
            target_lm_head_weight, norm_eps,
        )

        # bf16-rounded tp vs fp32-tp inside the kernel introduces ~1e-3 noise.
        torch.testing.assert_close(loss_lz, loss_pc, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(acc_lz, acc_pc, atol=1e-3, rtol=1e-3)

    def test_target_grad_isolation(self):
        """Backward through the lazy kernel must NOT accumulate grads on the
        verifier weight or hidden states — the .detach() inside the kernel is
        the load-bearing defense for FSDP2 micro-batch accumulation."""
        torch.manual_seed(0)
        N, H, V = 16, 64, 128
        hs = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        ths = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        norm_weight = torch.randn(H, dtype=torch.float32, requires_grad=True)
        lm_head_weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
        target_lm_head_weight = torch.randn(V, H, dtype=torch.float32, requires_grad=True)
        norm_eps = 1e-6
        valid_idx = torch.arange(N)

        loss, _ = compiled_forward_kl_loss_from_hs(
            hs, ths, valid_idx, norm_weight, lm_head_weight,
            target_lm_head_weight, norm_eps,
        )
        loss.backward()

        # Draft side should have grads.
        self.assertIsNotNone(hs.grad)
        self.assertIsNotNone(norm_weight.grad)
        self.assertIsNotNone(lm_head_weight.grad)
        self.assertTrue(torch.isfinite(hs.grad).all())

        # Target side must NOT have grads (or must be all zero) thanks to the
        # in-kernel .detach().
        self.assertTrue(
            ths.grad is None or ths.grad.abs().sum().item() == 0.0,
            f"target hidden states grad should be None/zero, got {ths.grad}",
        )
        self.assertTrue(
            target_lm_head_weight.grad is None
            or target_lm_head_weight.grad.abs().sum().item() == 0.0,
            f"target lm_head grad should be None/zero, got {target_lm_head_weight.grad}",
        )


class TestValidIdxSubsettingLazy(unittest.TestCase):
    """valid_idx filtering must produce the same loss as manual pre-filtering
    under the lazy kernel too — same property as the precomputed kernel."""

    BT, H, V = 64, 128, 256

    def _check_forward_kl_lazy(self, valid_idx):
        torch.manual_seed(7)
        hs_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        ths_flat = torch.randn(self.BT, self.H, dtype=torch.bfloat16)
        norm_weight = torch.randn(self.H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        target_lm_head_weight = torch.randn(self.V, self.H, dtype=torch.bfloat16)
        norm_eps = 1e-6

        loss, acc = compiled_forward_kl_loss_from_hs(
            hs_flat, ths_flat, valid_idx, norm_weight, lm_head_weight,
            target_lm_head_weight, norm_eps,
        )

        hs_valid = hs_flat[valid_idx]
        ths_valid = ths_flat[valid_idx]
        all_idx = torch.arange(hs_valid.shape[0])
        loss_ref, acc_ref = compiled_forward_kl_loss_from_hs(
            hs_valid, ths_valid, all_idx, norm_weight, lm_head_weight,
            target_lm_head_weight, norm_eps,
        )

        torch.testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(acc, acc_ref, atol=1e-4, rtol=1e-4)


# Mirror the precomputed-kernel pattern: one test method per mask shape.
for _name, _vidx in _make_mask_patterns(TestValidIdxSubsettingLazy.BT):

    def _make_lazy(vidx=_vidx):
        def test(self):
            self._check_forward_kl_lazy(vidx)

        return test

    setattr(TestValidIdxSubsettingLazy, f"test_forward_kl_lazy_{_name}", _make_lazy())


class TestValidIdxDynamicShape(unittest.TestCase):
    """Verify compiled loss kernels accept changing valid_idx lengths."""

    def test_compiled_forward_kl_loss_changing_valid_idx_lengths(self):
        """compiled_forward_kl_loss should work with changing valid_idx.shape[0]."""
        torch._dynamo.reset()
        BT, H, V = 4096, 128, 256
        hs_flat = torch.randn(BT, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        tp_flat = F.softmax(torch.randn(BT, V), dim=-1)

        for n in [BT, BT - 1, BT // 2, 1]:
            valid_idx = torch.arange(n)
            torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
            loss, acc = compiled_forward_kl_loss(
                hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, 1e-6
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(acc))

    def test_compiled_forward_kl_loss_from_hs_changing_valid_idx_lengths(self):
        """compiled_forward_kl_loss_from_hs should work with changing valid_idx.shape[0]."""
        torch._dynamo.reset()
        BT, H, V = 4096, 128, 256
        hs_flat = torch.randn(BT, H, dtype=torch.bfloat16)
        ths_flat = torch.randn(BT, H, dtype=torch.bfloat16)
        norm_weight = torch.randn(H, dtype=torch.bfloat16)
        lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)
        target_lm_head_weight = torch.randn(V, H, dtype=torch.bfloat16)

        for n in [BT, BT - 1, BT // 2, 1]:
            valid_idx = torch.arange(n)
            torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
            loss, acc = compiled_forward_kl_loss_from_hs(
                hs_flat, ths_flat, valid_idx, norm_weight,
                lm_head_weight, target_lm_head_weight, 1e-6
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(acc))


class TestEagle3ModelLazyDispatch(unittest.TestCase):
    """End-to-end Eagle3Model.forward should dispatch to the lazy kernel when
    target is a LazyTarget, and produce finite plosses + acces of the right
    length (matches PrecomputedTarget shape contract)."""

    def test_forward_with_lazy_target(self):
        torch.manual_seed(0)
        B, T = 1, 16
        H, V = 128, 256
        length = 3
        config = _make_config(H=H, V=V)
        model = _make_model(config, length=length)

        batch = _make_batch(B, T, H, V)
        target = compute_lazy_target_padded(
            target_hidden_states=batch["target_hidden_states"],
            target_lm_head_weight=torch.randn(V, H, dtype=torch.bfloat16),
            length=length,
        )

        plosses, _, acces = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target=target,
            loss_mask=batch["loss_mask"],
            hidden_states=batch["hidden_states"],
        )

        self.assertEqual(len(plosses), length)
        self.assertEqual(len(acces), length)
        for i, p in enumerate(plosses):
            self.assertTrue(torch.isfinite(p), f"ploss[{i}] non-finite: {p}")


if __name__ == "__main__":
    unittest.main()
