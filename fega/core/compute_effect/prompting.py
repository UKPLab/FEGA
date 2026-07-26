from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fega.core.utils import prompt_from_dict
from sae_bench.evals.ravel.instance import Prompt


@dataclass
class AblationSpec:
    """Describe which SAE feature index to zero for each row in a batch."""

    feature_ids: torch.Tensor


def build_prompt_lookup(pairs_path: Path) -> dict[str, dict[str, list[Prompt]]]:
    """Deserialize pairs_full.json into an attr -> role -> prompts lookup."""
    raw = json.loads(Path(pairs_path).read_text())
    lookup: dict[str, dict[str, list[Prompt]]] = {}
    for attr, pair_dict in raw.items():
        lookup[attr] = {}
        for role, prompts in pair_dict.items():
            lookup[attr][role] = [prompt_from_dict(prompt) for prompt in prompts]
    return lookup


def prompt_from_meta(meta: dict[str, Any], tokenizer) -> Prompt:
    """Reconstruct a Prompt from saved activation metadata by re-tokenizing text."""
    text = meta.get("prompt") or meta.get("text")
    if text is None:
        raise ValueError("Missing prompt text in metadata; cannot reconstruct Prompt.")
    toks = tokenizer(
        text,
        return_attention_mask=True,
        add_special_tokens=False,
    )
    input_ids = toks.input_ids
    attn_mask = toks.attention_mask
    final_pos = meta.get("final_entity_token_pos")
    if final_pos is None:
        final_pos = len(input_ids) - 1
    return Prompt(
        text=text,
        template=meta.get("template", ""),
        attribute_type=meta.get("attribute_type", ""),
        attribute_label=meta.get("attribute_label", ""),
        entity_label=meta.get("entity_label", ""),
        context_split=meta.get("context_split", ""),
        entity_split=meta.get("entity_split", ""),
        input_ids=input_ids,
        attention_mask=attn_mask,
        final_entity_token_pos=final_pos,
        attribute_generation=meta.get("attribute_generation"),
        first_generated_token_id=meta.get("first_generated_token_id"),
        is_correct=meta.get("is_correct"),
    )


def resolve_prompt(
    lookup: dict[str, dict[str, list[Prompt]]],
    attribute_label: str,
    pair_role: str,
    pair_index: int,
) -> Prompt | None:
    """Lookup a Prompt by attribute/role/index from a loaded pairs lookup."""
    attr_block = lookup.get(attribute_label)
    if not attr_block:
        return None
    prompts = attr_block.get(pair_role)
    if prompts is None or pair_index >= len(prompts):
        return None
    return prompts[pair_index]


def handle_oom_adjustment(current_bs: int, stats_targets: list[dict[str, Any]]) -> int:
    """Reduce batch size after an OOM and record the retry in caller stats."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    new_bs = max(1, current_bs // 2)
    for stats in stats_targets:
        stats.setdefault("oom_adjustments", []).append(
            {"from": current_bs, "to": new_bs}
        )
    if new_bs == current_bs:
        raise torch.cuda.OutOfMemoryError(
            "Batch size already minimal; cannot recover from OOM."
        )
    return new_bs
