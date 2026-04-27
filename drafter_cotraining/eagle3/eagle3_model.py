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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from recipe.drafter_cotraining.eagle3.ops.loss import (
    compiled_forward_kl_loss,
    compiled_forward_kl_loss_from_hs,
)
def padding(tensor, left=True):
    """Shift tensor by one position along dim=1 with zero padding.

    Mirrors TorchSpec's torchspec.utils.tensor.padding:
    - left=True:  shift right (prepend zero, drop last)
    - left=False: shift left  (drop first, append zero)
    """
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


@dataclass
class PrecomputedTarget:
    """Pre-computed target probabilities for the EAGLE forward-KL loss kernel.

    With vocab pruning (t2d set in compute_target_p_padded):
        target_p_padded shape: (B, T + length, V_draft)
        position_mask: (B, T) — subset of loss_mask, only positions whose
            verifier-argmax token falls in V_draft.
    Without vocab pruning:
        target_p_padded shape: (B, T + length, V_full)
        position_mask: None — loss_mask used directly downstream.

    Stored in bf16 to halve resident memory; loss kernel's `tp * log_p`
    (log_p in fp32) auto-upcasts so loss arithmetic stays fp32.
    """

    target_p_padded: torch.Tensor
    position_mask: Optional[torch.Tensor] = None


@dataclass
class LazyTarget:
    """Deferred target distribution for the EAGLE forward-KL loss kernel.

    Holds the verifier hidden-states and lm_head weight only; the softmax
    over V_full is computed inside ``compiled_forward_kl_loss_from_hs`` per
    TTT step rather than materialized as a (B, T+length, V_full) resident
    tensor. Trades resident memory for per-step compute — see
    ``LazyTarget vs Precomputed Memory Analysis.md`` for the crossover.

    Both fields are detached at factory time (``compute_lazy_target_padded``)
    and again inside the compiled kernel as a belt-and-suspenders defense
    against accidental grad-graph leakage under FSDP micro-batching.

    Note: incompatible with vocab pruning — when ``t2d`` is set, use
    ``PrecomputedTarget`` instead (V_draft is small, so the resident tensor
    is cheap and pruning's whole point is the V_draft projection).
    """

    hidden_states_padded: torch.Tensor  # (B, T + length, D), bf16
    lm_head_weight: torch.Tensor  # (V_full, D), bf16


class Eagle3Model(nn.Module):
    def __init__(
        self,
        draft_model,
        length: int = 7,
        attention_backend="flex_attention",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.gradient_checkpointing = gradient_checkpointing
        self.vocab_pruning = draft_model.vocab_size != draft_model.target_vocab_size

    def can_generate(self) -> bool:
        # Satisfies FSDPCheckpointManager.save_checkpoint, which probes this
        # on the unwrapped module to decide whether to emit a GenerationConfig.
        # Eagle3's TTT forward is not HF-generate compatible — so: no.
        return False

    def _calculate_loss(
        self,
        hidden_states: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        mask: torch.Tensor,
        idx: int,
        seq_length: int,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        norm_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forward-KL loss and accuracy for one TTT step.

        Passes full (B*T, ...) flat views + valid_idx into the compiled
        function so torch.compile can fuse index_select with subsequent ops.

        - PrecomputedTarget: target_p sized over V_draft (pruning) or V_full
          (no pruning); kernel takes a pre-built ``target_p_flat``.
        - LazyTarget: target_p computed inside the compiled graph from
          ``target_hidden_states_flat`` and the (frozen) verifier lm_head;
          saves the resident (B, T+length, V) tensor.
        """
        valid_idx = mask.flatten().nonzero().squeeze(-1)
        if valid_idx.numel() == 0:
            # FSDP requires every trainable param to participate in gradient
            # all-reduce/reduce-scatter.
            total = sum(p.reshape(-1)[0] for p in self.parameters() if p.requires_grad)
            zero = total * 0.0
            return zero, zero.detach()
        # Soft hint to Dynamo that valid_idx.shape[0] may vary across calls.
        # Unlike mark_dynamic, this won't error if Dynamo proves the dimension is constant.
        torch._dynamo.maybe_mark_dynamic(valid_idx, 0)
        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

        if isinstance(target, PrecomputedTarget):
            target_p_step = target.target_p_padded[:, idx : idx + seq_length, :]
            tp_flat = target_p_step.reshape(-1, target_p_step.shape[-1])
            args = (hs_flat, tp_flat, valid_idx, norm_weight, lm_head_weight, norm_eps)
            if self.gradient_checkpointing and self.training:
                return torch_checkpoint(
                    compiled_forward_kl_loss,
                    *args,
                    use_reentrant=False,
                )
            return compiled_forward_kl_loss(*args)

        # LazyTarget — compute the target softmax inside the compiled kernel.
        ths_step = target.hidden_states_padded[:, idx : idx + seq_length, :]
        ths_flat = ths_step.reshape(-1, target.lm_head_weight.shape[-1])
        args = (
            hs_flat,
            ths_flat,
            valid_idx,
            norm_weight,
            lm_head_weight,
            target.lm_head_weight,
            norm_eps,
        )
        if self.gradient_checkpointing and self.training:
            return torch_checkpoint(
                compiled_forward_kl_loss_from_hs,
                *args,
                use_reentrant=False,
            )
        return compiled_forward_kl_loss_from_hs(*args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        norm_weight, lm_head_weight, norm_eps = self.draft_model.get_lm_head_params()

        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # position_mask (vocab pruning) is a subset of loss_mask that further
        # filters to tokens whose target argmax falls in the draft vocab.
        if isinstance(target, PrecomputedTarget) and target.position_mask is not None:
            mask = target.position_mask
        else:
            mask = loss_mask

        plosses = []
        vlosses = []
        acces = []
        cache_keys = None
        cache_values = None

        # Clamp multimodal placeholder IDs (hash-based pad values from SGLang)
        # to valid vocab range before embedding lookup.
        input_ids = input_ids.clamp(min=0, max=self.draft_model.target_vocab_size - 1)

        for idx in range(self.length):
            is_last = idx == self.length - 1

            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                hidden_states_out, cache_keys, cache_values = torch_checkpoint(
                    self.draft_model.backbone,
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_keys,
                    cache_values,
                    True,  # use_cache
                    use_reentrant=False,
                )
            else:
                hidden_states_out, cache_keys, cache_values = self.draft_model.backbone(
                    input_embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_keys=cache_keys,
                    cache_values=cache_values,
                    use_cache=True,
                )

            hidden_states = hidden_states_out

            loss, acc = self._calculate_loss(
                hidden_states=hidden_states,
                target=target,
                mask=mask,
                idx=idx,
                seq_length=seq_length,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
            )
            plosses.append(loss)
            acces.append(acc)

            if not is_last:
                input_ids = padding(input_ids, left=False)
                mask = padding(mask, left=False)
        return plosses, vlosses, acces


@torch.no_grad()
def compute_target_p_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    length: int,
    t2d: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> PrecomputedTarget:
    """Build target probabilities for the EAGLE forward-KL loss kernel.

    With pruning (t2d not None):
        - lm_head projects to V_draft
        - position_mask filters loss_mask further to positions whose
          verifier-argmax token falls in V_draft.
    Without pruning (t2d is None):
        - lm_head projects to V_full
        - position_mask is None; loss_mask is used directly downstream.

    target_p is stored in bf16 to halve resident memory; the loss kernel's
    `tp * log_p` (log_p in fp32) auto-upcasts so loss arithmetic stays fp32.
    """
    target_lm_head_weight = target_lm_head_weight.detach()

    if t2d is not None:
        pruned_weight = target_lm_head_weight[t2d]  # (V_draft, D)

        B, T, _D = target_hidden_states.shape
        loss_mask_bool = loss_mask.bool()
        valid_flat_idx = loss_mask_bool.reshape(-1).nonzero(as_tuple=True)[0]
        valid_hs = target_hidden_states.reshape(-1, _D)[valid_flat_idx]  # (N_valid, D)

        position_mask_flat = torch.zeros(B * T, device=target_hidden_states.device, dtype=torch.float)
        for i in range(0, valid_hs.shape[0], chunk_size):
            chunk_hs = valid_hs[i : i + chunk_size]
            chunk_argmax = F.linear(chunk_hs, target_lm_head_weight).argmax(-1)
            in_draft = t2d[chunk_argmax]
            position_mask_flat[valid_flat_idx[i : i + chunk_size]] = in_draft.float()
        position_mask = position_mask_flat.reshape(B, T)

        target_logits = F.linear(target_hidden_states, pruned_weight)
    else:
        target_logits = F.linear(target_hidden_states, target_lm_head_weight)
        position_mask = None

    target_p = F.softmax(target_logits.float(), dim=-1).to(torch.bfloat16)
    target_p_padded = F.pad(target_p, (0, 0, 0, length), value=0.0)

    return PrecomputedTarget(target_p_padded, position_mask)


@torch.no_grad()
def compute_lazy_target_padded(
    target_hidden_states: torch.Tensor,
    target_lm_head_weight: torch.Tensor,
    length: int,
) -> LazyTarget:
    """Build a LazyTarget that defers softmax to the loss kernel.

    Used for the no-pruning regime to avoid materializing
    ``(B, T + length, V_full)`` resident; per-step transient is
    ``(N_valid, V_full)`` instead.

    Hardened against grad-graph leakage: ``@torch.no_grad()`` plus an
    explicit ``.detach()`` on both the padded HS and the lm_head weight.
    The compiled kernel ``compiled_forward_kl_loss_from_hs`` repeats the
    detach on its target inputs as belt-and-suspenders. Three layers
    matter because lazy is only safe under FSDP micro-batching when the
    target side is fully outside the autograd graph (see hazards #1 and
    #3 in ``Loss Kernel Choice — Lazy vs Precomputed.md``).
    """
    return LazyTarget(
        hidden_states_padded=F.pad(
            target_hidden_states, (0, 0, 0, length), value=0.0
        ).detach(),
        lm_head_weight=target_lm_head_weight.detach(),
    )
