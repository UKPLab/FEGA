from __future__ import annotations

import json
from pathlib import Path

from sae_bench.evals.ravel.instance import Prompt, RAVELFilteredDataset

from fega.core.utils.ravel import (
    ReplayContext,
    build_prompt_pairs,
)


def _write_reference(path: Path, eval_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval_config": eval_config,
                "sae_lens_release_id": "release",
                "sae_lens_id": "sae",
                "sae_cfg_dict": {"hook_layer": 0},
            }
        )
    )


def _prompt(attribute: str, label: str, entity: str = "Paris") -> Prompt:
    return Prompt(
        text=f"{entity} -> {label}",
        template="%s ->",
        attribute_type=attribute,
        attribute_label=label,
        entity_label=entity,
        context_split="train",
        entity_split="train",
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        final_entity_token_pos=-1,
        first_generated_token_id=4,
        attribute_generation=label,
        is_correct=True,
    )


def _prompt_signature(prompt: Prompt) -> tuple:
    return (
        prompt.text,
        prompt.entity_label,
        prompt.attribute_type,
        prompt.attribute_label,
        tuple(prompt.input_ids),
        prompt.final_entity_token_pos,
        prompt.first_generated_token_id,
    )


def _pairs_signature(pairs: dict[str, list[Prompt]]) -> dict[str, list[tuple]]:
    return {
        key: [_prompt_signature(prompt) for prompt in prompts]
        for key, prompts in pairs.items()
    }


def test_build_prompt_pairs_uses_native_ravel_pair_shape(tmp_path: Path):
    reference_json = tmp_path / "ref.json"
    _write_reference(
        reference_json,
        {
            "model_name": "gpt2",
            "num_pairs_per_attribute": 2,
            "random_seed": 3,
            "entity_attribute_selection": {"city": ["Country", "Language"]},
        },
    )
    ctx = ReplayContext.from_file(reference_json)
    dataset = RAVELFilteredDataset(
        prompts=[
            _prompt("Country", "France", "Paris"),
            _prompt("Country", "Germany", "Berlin"),
            _prompt("Country", "Italy", "Rome"),
            _prompt("Language", "French", "Paris"),
            _prompt("Language", "German", "Berlin"),
            _prompt("Language", "Italian", "Rome"),
        ],
        config={},
    )

    pairs = build_prompt_pairs(ctx, dataset, "Country", ["Language"])

    assert set(pairs) == {
        "cause_base_prompts",
        "cause_source_prompts",
        "iso_base_prompts",
        "iso_source_prompts",
    }
    assert len(pairs["cause_base_prompts"]) == 2
    assert len(pairs["cause_source_prompts"]) == 2
    assert len(pairs["iso_base_prompts"]) == 2
    assert len(pairs["iso_source_prompts"]) == 2


def test_build_prompt_pairs_is_independent_of_process_global_rng(tmp_path: Path):
    reference_json = tmp_path / "ref.json"
    _write_reference(
        reference_json,
        {
            "model_name": "gpt2",
            "num_pairs_per_attribute": 4,
            "random_seed": 17,
            "entity_attribute_selection": {"city": ["Country", "Language"]},
        },
    )
    ctx = ReplayContext.from_file(reference_json)
    dataset = RAVELFilteredDataset(
        prompts=[
            _prompt("Country", "France", "Paris"),
            _prompt("Country", "Germany", "Berlin"),
            _prompt("Country", "Italy", "Rome"),
            _prompt("Country", "Spain", "Madrid"),
            _prompt("Country", "Portugal", "Lisbon"),
            _prompt("Country", "Poland", "Warsaw"),
            _prompt("Language", "French", "Paris"),
            _prompt("Language", "German", "Berlin"),
            _prompt("Language", "Italian", "Rome"),
            _prompt("Language", "Spanish", "Madrid"),
            _prompt("Language", "Portuguese", "Lisbon"),
            _prompt("Language", "Polish", "Warsaw"),
        ],
        config={},
    )

    import random

    random.seed(999)
    first = _pairs_signature(build_prompt_pairs(ctx, dataset, "Country", ["Language"]))
    random.seed(12345)
    for _ in range(50):
        random.random()
    second = _pairs_signature(build_prompt_pairs(ctx, dataset, "Country", ["Language"]))

    assert second == first
