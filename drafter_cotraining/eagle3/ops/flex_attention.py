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
import torch._dynamo as dynamo
import torch._inductor.config as inductor_config
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, or_masks
from transformers.utils import is_torchdynamo_compiling

try:
    dynamo.config.recompile_limit = 128
except AttributeError:
    dynamo.config.cache_size_limit = 128

if "ATEN" not in getattr(inductor_config, "max_autotune_gemm_backends", ""):
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"


class WrappedFlexAttention:
    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_flex_compiled:
            self._compiled_flex_attention = torch.compile(flex_attention)
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


_compiled_create_block_mask = torch.compile(create_block_mask)


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(query, key, value, **kwargs)


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
    **kwargs,
):
    # Compiling create_block_mask replaces the dense (B, H, Q_LEN, KV_LEN) boolean
    # intermediate with block-wise Triton evaluation: ~6000x less peak memory and
    # ~9x faster at training shapes than the default Python path.
    # See claude_docs/eagle3-block-mask-memory-diagnosis.md.
    fn = _compiled_create_block_mask if not is_torchdynamo_compiling() else create_block_mask
    return fn(mask_mod, B, H, Q_LEN, KV_LEN, device, **kwargs)


def generate_eagle3_mask(seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0):
    def causal_mask(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod
