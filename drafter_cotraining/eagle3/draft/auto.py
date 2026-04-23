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

import copy
import json
import logging
import os
import warnings
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import DeepseekV3Config, LlamaConfig, PretrainedConfig, modeling_utils

from recipe.drafter_cotraining.eagle3.draft.llama3_eagle import LlamaForCausalLMEagle3

logger = logging.getLogger(__name__)


# ── Auto-derive draft architecture from target model ──────────────────────
# Ported from TorchSpec torchspec/config/utils.py:32-149. Source of truth
# for "what shape should the draft be" given a target model — replaces the
# hand-tuned per-target JSON workflow. Override draft-specific fields
# (draft_vocab_size for vocab pruning, non-Llama architectures) by passing
# a template_config_path that the auto-derive overlays on top of.

def _copy_config_value(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return copy.deepcopy(value)


def _normalize_rope_scaling(rope_scaling):
    if rope_scaling is None:
        return None

    normalized = _copy_config_value(rope_scaling)
    if not isinstance(normalized, dict):
        return normalized

    scaling_type = normalized.get("rope_type", normalized.get("type"))
    if scaling_type == "yarn":
        yarn_defaults = {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
        }
        for key, default in yarn_defaults.items():
            if normalized.get(key) is None:
                normalized[key] = default

    return normalized


def generate_draft_model_config(
    target_model_path: str,
    template_config_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """Auto-generate an Eagle3 draft model config from the target model.

    When ``template_config_path`` is provided, that JSON is the starting
    point and architecture fields below are overridden from the target.
    When ``None``, a bare ``{"architectures": ["LlamaForCausalLMEagle3"]}``
    skeleton is the starting point.

    Architecture fields *always* inherited from the target's HF AutoConfig
    (template values, if any, are discarded for these): ``vocab_size``,
    ``hidden_size``, ``num_attention_heads``, ``num_key_value_heads``,
    ``intermediate_size``, ``max_position_embeddings``, ``rope_theta``,
    ``rope_scaling``, ``rms_norm_eps``, ``hidden_act``, ``bos_token_id``,
    ``eos_token_id``, ``torch_dtype``.

    Forced values (Eagle3 invariants): ``num_hidden_layers=1``,
    ``tie_word_embeddings=False``, ``use_cache=True``.

    Defaults: ``draft_vocab_size`` falls back to ``vocab_size`` (no vocab
    pruning) when not explicitly set in the template.

    Returns a dict suitable for ``AutoDraftModelConfig.from_dict``.
    """
    target_config = AutoConfig.from_pretrained(
        target_model_path, cache_dir=cache_dir, trust_remote_code=True
    )

    text_config = getattr(target_config, "text_config", target_config)

    if template_config_path is not None:
        with open(template_config_path) as f:
            draft_config = json.load(f)
    else:
        warnings.warn(
            "No template config provided for draft model. "
            "Auto-generating config entirely from target model. "
            "Provide a template via local_path for vocab pruning, MLA, or other "
            "non-default architectures.",
            stacklevel=2,
        )
        draft_config = {"architectures": ["LlamaForCausalLMEagle3"]}

    draft_config["model_type"] = "llama"

    param_mappings = {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "max_position_embeddings": "max_position_embeddings",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "rms_norm_eps": "rms_norm_eps",
        "hidden_act": "hidden_act",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "torch_dtype": "torch_dtype",
    }

    for target_param, draft_param in param_mappings.items():
        if hasattr(text_config, target_param):
            value = getattr(text_config, target_param)
        elif hasattr(target_config, target_param):
            value = getattr(target_config, target_param)
        else:
            continue
        if target_param == "torch_dtype" and isinstance(value, torch.dtype):
            value = str(value).replace("torch.", "")
        else:
            value = _copy_config_value(value)
        if target_param == "rope_scaling":
            value = _normalize_rope_scaling(value)
        draft_config[draft_param] = value

    draft_config["num_hidden_layers"] = 1
    draft_config["tie_word_embeddings"] = False
    draft_config["use_cache"] = True

    if "draft_vocab_size" not in draft_config:
        draft_config["draft_vocab_size"] = draft_config.get("vocab_size")

    return draft_config


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        # DeepseekV3Config: DeepSeekForCausalLMEagle3,  # deferred
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        _model_cls = cls._model_mapping[type(config)]
        model = _model_cls(config, **config_kwargs)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "DeepSeekForCausalLMEagle3": DeepseekV3Config,
        "Eagle3DeepseekV2ForCausalLM": DeepseekV3Config,
    }

    @classmethod
    def from_dict(cls, config: dict):
        config = dict(config)

        if "tie_word_embeddings" in config:
            logger.info("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        return cls._config_mapping[architecture].from_dict(config)

    @classmethod
    def from_file(cls, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_target(cls, target_model_path: str, template_path: Optional[str] = None):
        """Auto-derive a draft model config from the target model.

        Architecture fields (hidden_size, num_attention_heads, vocab_size,
        rope_*, rms_norm_eps, etc.) inherit from the target's HF AutoConfig.
        ``num_hidden_layers`` is forced to 1 (Eagle3 invariant).
        ``draft_vocab_size`` defaults to ``vocab_size`` (no vocab pruning).

        ``template_path`` is optional and only needed for: vocab pruning
        (set ``draft_vocab_size``), non-Llama draft architectures (set
        ``architectures``), or overriding non-architecture defaults like
        ``attention_bias`` / ``sliding_window`` for non-Llama targets.
        """
        config_dict = generate_draft_model_config(
            target_model_path, template_config_path=template_path
        )
        return cls.from_dict(config_dict)
