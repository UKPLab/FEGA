from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest

from sae_bench.evals.icl_features.generate import curate_balanced_examples
from sae_bench.evals.icl_features.generators import (
    LSCFamily,
    PrOntoQAFamily,
    TTFamily,
    WCFamily,
)
from sae_bench.evals.icl_features.model_filter import CorrectnessResult
from sae_bench.evals.icl_features.schema import (
    balanced_family_quotas,
    validate_dataset,
)


WORDS = [f"word{chr(97 + first)}{chr(97 + second)}" for first in range(20) for second in range(20)]


@dataclass
class FakeFamily:
    family_id: str
    family_index: int
    num_support_slots: int = 1

    def generate(self, candidate_index: int) -> dict[str, Any]:
        answer = "good" if candidate_index % 2 == 0 else "bad"
        return {
            "example_id": f"candidate_{candidate_index}",
            "context_id": self.family_id,
            "query_style": "test",
            "prompt": f"{self.family_id} prompt {candidate_index}",
            "answer": answer,
            "x": "x",
            "y": answer,
            "entity": "entity",
            "source_concept": "source",
            "target_concept": answer,
            "support_example_index": 0,
            "lookup_rule": "source -> answer",
            "induction_prefix": None,
        }


class FakeScorer:
    model_name = "fake"
    resolved_model_name = "fake"
    tokenizer = object()

    def answer_token_ids(self, answer: str) -> list[int]:
        return [1]

    def score(self, rows: list[dict[str, Any]]) -> list[CorrectnessResult]:
        return [
            CorrectnessResult(
                correct=row["answer"] == "good",
                target_first_token_id=1,
                target_token_length=1,
                predicted_first_token_id=1 if row["answer"] == "good" else 2,
                predicted_first_token="good" if row["answer"] == "good" else "bad",
            )
            for row in rows
        ]


def test_balanced_family_quotas_are_exact() -> None:
    quotas = balanced_family_quotas(10, 3)
    assert quotas == [4, 3, 3]
    assert sum(quotas) == 10


def test_curator_fills_exact_model_correct_balanced_quotas() -> None:
    families = [FakeFamily(f"family_{index}", index) for index in range(3)]
    examples, stats = curate_balanced_examples(
        families=families,
        quotas=[3, 2, 2],
        scorer=FakeScorer(),
        candidates_per_family_round=2,
        max_candidates_per_family=20,
        require_single_token_answers=True,
    )
    assert len(examples) == 7
    assert all(row["model_correct_first_token"] for row in examples)
    assert [entry["accepted"] for entry in stats] == [3, 2, 2]
    assert [entry["accepted_by_support_slot"] for entry in stats] == [[3], [2], [2]]
    assert len({row["prompt"] for row in examples}) == len(examples)


def test_task_families_are_deterministic_and_preserve_original_structures() -> None:
    lsc_a = LSCFamily(0, random.Random(1), WORDS, 5, 10)
    lsc_b = LSCFamily(0, random.Random(1), WORDS, 5, 10)
    lsc_row = lsc_a.generate(0)
    assert lsc_row == lsc_b.generate(0)
    tokens = lsc_row["prompt"].split()
    assert tokens[:5] == tokens[-5:]
    assert lsc_row["answer"] == tokens[5]

    wc = WCFamily(0, random.Random(2), WORDS, 3, 2, 2, 50, 7)
    wc_row = wc.generate(1)
    selected_group = wc.groups[1]
    assert wc_row["answer"] == selected_group["label"]
    assert set(selected_group["features"]).issubset(set(wc_row["entity"].split()))

    tt = TTFamily(0, random.Random(3), 5)
    tt_row = tt.generate(0)
    assert tt_row["source_language"] == "en"
    assert tt_row["target_language"] == "de"
    assert tt_row["prompt"].count("German:") == 6

    prontoqa = PrOntoQAFamily(0, random.Random(4), 3, "rule_completion")
    prontoqa_row = prontoqa.generate(0)
    assert prontoqa_row["answer"] not in prontoqa_row["x"]
    assert prontoqa_row["induction_prefix"] in prontoqa_row["x"]
    assert prontoqa_row["lookup_rule"] in prontoqa.context_prompt


def test_dataset_validation_rejects_duplicate_prompts() -> None:
    row = FakeFamily("family_0", 0).generate(0)
    payload = {
        "metadata": {"filtered_example_count": 2, "num_prompt_families": 1},
        "examples": [
            {**row, "example_id": "a", "model_correct_first_token": True},
            {**row, "example_id": "b", "model_correct_first_token": True},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate prompt"):
        validate_dataset(payload, require_model_correct=True)
