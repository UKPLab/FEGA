from __future__ import annotations

from collections import Counter
from typing import Any


REQUIRED_EXAMPLE_FIELDS = (
    "example_id",
    "context_id",
    "query_style",
    "prompt",
    "answer",
    "x",
    "y",
    "entity",
    "source_concept",
    "target_concept",
    "support_example_index",
    "lookup_rule",
    "induction_prefix",
)


def balanced_family_quotas(total_examples: int, num_families: int) -> list[int]:
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if num_families <= 0:
        raise ValueError("num_families must be positive")
    if num_families > total_examples:
        raise ValueError("num_families cannot exceed total_examples")
    base, remainder = divmod(total_examples, num_families)
    return [base + (family_index < remainder) for family_index in range(num_families)]


def validate_dataset(payload: dict[str, Any], *, require_model_correct: bool) -> None:
    metadata = payload.get("metadata")
    examples = payload.get("examples")
    if not isinstance(metadata, dict):
        raise ValueError("Dataset must contain a metadata object")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Dataset must contain a non-empty examples list")

    missing = [
        (index, field)
        for index, row in enumerate(examples)
        for field in REQUIRED_EXAMPLE_FIELDS
        if field not in row
    ]
    if missing:
        index, field = missing[0]
        raise ValueError(f"Example {index} is missing required field {field!r}")

    example_ids = [str(row["example_id"]) for row in examples]
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("example_id values must be unique")

    family_counts = Counter(str(row["context_id"]) for row in examples)
    if max(family_counts.values()) - min(family_counts.values()) > 1:
        raise ValueError("Prompt-family sizes differ by more than one example")
    expected_families = metadata.get("num_prompt_families")
    if expected_families is not None and len(family_counts) != int(expected_families):
        raise ValueError(
            f"metadata.num_prompt_families={expected_families} but found "
            f"{len(family_counts)} families"
        )

    family_prompts: dict[str, set[str]] = {}
    for row in examples:
        family_id = str(row["context_id"])
        prompt = str(row["prompt"])
        prompts = family_prompts.setdefault(family_id, set())
        if prompt in prompts:
            raise ValueError(f"Duplicate prompt retained in family {family_id!r}")
        prompts.add(prompt)

    family_metadata = {
        str(family["context_id"]): family
        for family in payload.get("prompt_families", [])
        if isinstance(family, dict) and "context_id" in family
    }
    if family_metadata:
        for family_id, family_rows in _group_rows_by_family(examples).items():
            expected_slots = int(family_metadata[family_id]["num_support_slots"])
            slot_counts = Counter(
                int(row["support_example_index"]) for row in family_rows
            )
            if set(slot_counts) != set(range(expected_slots)):
                raise ValueError(
                    f"Family {family_id!r} does not cover every support slot"
                )
            if max(slot_counts.values()) - min(slot_counts.values()) > 1:
                raise ValueError(
                    f"Family {family_id!r} support-slot sizes differ by more than one"
                )

    for row in examples:
        if str(row["answer"]) != str(row["y"]):
            raise ValueError(f"answer/y mismatch in {row['example_id']}")
        if require_model_correct and not bool(row.get("model_correct_first_token")):
            raise ValueError(
                f"Model-incorrect row retained in filtered dataset: {row['example_id']}"
            )

    expected = metadata.get("filtered_example_count")
    if expected is not None and int(expected) != len(examples):
        raise ValueError(
            f"metadata.filtered_example_count={expected} but found {len(examples)} rows"
        )


def _group_rows_by_family(
    examples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        grouped.setdefault(str(row["context_id"]), []).append(row)
    return grouped
