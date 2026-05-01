# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F


@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + RMSNorm + lm_head + Forward KL loss.

    Takes full (B*T, ...) flat tensors and performs index_select inside the
    compiled graph so the compiler can fuse the gather with subsequent ops.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_p_flat: (B*T, V_out) — flattened target probs (detached)
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_out, H) — draft lm_head weight
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_out)

    # Forward KL loss
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(tp * log_p).sum(-1).mean()

    # Accuracy
    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc


@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    target_lm_head_weight,
    norm_eps,
):
    """torch.compile'd lazy variant: build target probs *inside* the graph.

    Differs from ``compiled_forward_kl_loss`` in that the target distribution
    is computed from ``target_hidden_states_flat @ target_lm_head_weight`` per
    call rather than read from a pre-built ``(B*T, V)`` tensor. Saves the
    resident ``target_p_padded`` tensor at the cost of a per-TTT-step full-vocab
    projection. Useful when ``V_full`` is large and the loss-mask is sparse.

    Defensive ``.detach()`` calls on the target inputs are belt-and-suspenders:
    the factory (``compute_lazy_target_padded``) already strips grad lineage,
    but a future caller could regress that. Detaching here guarantees the
    target side never participates in autograd, which is the prerequisite for
    safe micro-batch FSDP gradient accumulation.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_hidden_states_flat: (B*T, D) — flattened verifier hidden states
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_draft, H) — draft lm_head weight
        target_lm_head_weight: (V_full, D) — verifier lm_head weight (frozen)
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx).detach()
    target_w = target_lm_head_weight.detach()

    # Target probs (no grad through the target side by construction).
    tp = F.softmax(F.linear(ths, target_w).float(), dim=-1)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = F.linear(norm_hs, lm_head_weight)  # (N, V_draft)

    # Forward KL loss
    log_p = F.log_softmax(logits.float(), dim=-1)
    loss = -(tp * log_p).sum(-1).mean()

    # Accuracy: argmax in V_draft vs argmax in V_full — only meaningful when
    # V_draft == V_full (no pruning), which is the lazy path's intended regime.
    acc = (logits.argmax(-1) == tp.argmax(-1)).float().mean()

    return loss, acc
