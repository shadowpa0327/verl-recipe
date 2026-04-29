"""Multi-turn conversation tokenization with per-turn assistant loss mask.

Ported from TorchSpec ``torchspec/data/parse.py:_tokenize_with_loss_mask`` and
``torchspec/data/preprocessing.py:preprocess_conversations``. Reduced to the
"general parser" path which is sufficient for Qwen / Llama chat templates.

The flow per row:

1. ``tokenizer.apply_chat_template`` renders the canonical
   ``[{role, content}, ...]`` list to a single string. The tokenizer's Jinja
   template decides where the assistant header / EOT tokens land — we don't
   hand-render anything.
2. ``tokenizer(text, max_length=N, truncation=True, add_special_tokens=False)``
   gives input_ids of length T (≤ N).
3. We regex-find every ``<assistant_header>(content)<eot>`` span in the
   formatted text, then map each character span to a token span by re-encoding
   the text prefix and measuring length difference. ``loss_mask[s_tok:e_tok] = 1``
   for every assistant content token.
4. Truncation past max_length is handled implicitly by ``min(s_tok, T)`` and
   ``min(e_tok, T)`` clamps — partial assistant turns contribute their
   surviving prefix.

Reference: ``/root/TorchSpec/docs/preprocessing_port_spec.md`` section 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _ChatTemplateAnchors:
    """Strings used to find assistant content spans inside the rendered text."""

    assistant_header: str
    end_of_turn_token: str
    system_prompt: str | None = None


# Chat-template anchors keyed by short name, matching TorchSpec's registry.
# Add more entries here as new model families are needed.
_ANCHOR_REGISTRY: dict[str, _ChatTemplateAnchors] = {
    "qwen": _ChatTemplateAnchors(
        assistant_header="<|im_start|>assistant\n",
        end_of_turn_token="<|im_end|>\n",
        system_prompt="You are a helpful assistant.",
    ),
    "qwen3-instruct": _ChatTemplateAnchors(
        # Qwen3-Instruct injects an empty <think>...</think> block right after
        # the assistant header. Anchoring on that ensures the loss mask covers
        # the post-think content (matches TorchSpec's qwen3-instruct entry).
        assistant_header="<|im_start|>assistant\n<think>\n\n</think>\n",
        end_of_turn_token="<|im_end|>\n",
        system_prompt="You are a helpful assistant.",
    ),
    "llama3": _ChatTemplateAnchors(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_of_turn_token="<|eot_id|>",
        system_prompt=(
            "You are a helpful, respectful and honest assistant. Always answer "
            "as helpfully as possible, while being safe."
        ),
    ),
}


def _get_anchors(name: str) -> _ChatTemplateAnchors:
    if name not in _ANCHOR_REGISTRY:
        raise ValueError(
            f"Unknown chat_template name {name!r}. "
            f"Known: {sorted(_ANCHOR_REGISTRY)}"
        )
    return _ANCHOR_REGISTRY[name]


def _ensure_system_prompt(
    conversation: list[dict[str, Any]],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """If the conversation has no system turn and a fallback is configured,
    prepend it. Mirrors TorchSpec ``GeneralParser.format``."""
    if not conversation:
        return conversation
    if conversation[0].get("role") == "system":
        return conversation
    if not system_prompt:
        return conversation
    return [{"role": "system", "content": system_prompt}, *conversation]


def _render_conversation(
    tokenizer,
    conversation: list[dict[str, Any]],
    anchors: _ChatTemplateAnchors,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Apply the tokenizer's chat template (Jinja) to a canonical conversation.

    If ``anchors.system_prompt`` is set and the conversation has no system
    turn, the fallback is prepended (matches TorchSpec).
    """
    kwargs = dict(apply_chat_template_kwargs or {})
    kwargs.setdefault("add_generation_prompt", False)
    msgs = _ensure_system_prompt(conversation, anchors.system_prompt)
    return tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)


def _tokenize_with_assistant_mask(
    tokenizer,
    formatted_text: str,
    anchors: _ChatTemplateAnchors,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize ``formatted_text`` and build a per-turn assistant loss mask.

    Returns:
        input_ids: int64 tensor of shape ``[T]`` with ``T <= max_length``.
        loss_mask: int64 tensor of shape ``[T]``; 1 on every assistant content
            token of every assistant turn, 0 elsewhere.
    """
    encoding = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoding.input_ids[0]
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    pattern = (
        re.escape(anchors.assistant_header)
        + r"([\s\S]*?(?:"
        + re.escape(anchors.end_of_turn_token)
        + "|$))"
    )
    matches = list(re.finditer(pattern, formatted_text))

    for match in matches:
        content_start_char = match.start(1)
        content_end_char = match.end(1)

        prefix_ids = tokenizer.encode(
            formatted_text[:content_start_char], add_special_tokens=False
        )
        full_ids = tokenizer.encode(
            formatted_text[:content_end_char], add_special_tokens=False
        )
        s = min(len(prefix_ids), len(input_ids))
        e = min(len(full_ids), len(input_ids))
        if s < e:
            loss_mask[s:e] = 1

    return input_ids, loss_mask


def build_input_ids_and_loss_mask(
    tokenizer,
    conversation: list[dict[str, Any]],
    chat_template: str,
    max_length: int,
    apply_chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot: render → tokenize → build per-turn assistant loss mask.

    The output is *not* padded; the trainer collator handles padding.
    """
    anchors = _get_anchors(chat_template)
    text = _render_conversation(tokenizer, conversation, anchors, apply_chat_template_kwargs)
    return _tokenize_with_assistant_mask(tokenizer, text, anchors, max_length)
