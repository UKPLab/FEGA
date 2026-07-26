from __future__ import annotations

from fega.core.data_prep.induction import (
    assign_dense_indices,
    select_induction_contexts,
)


def _record(
    scan_record_id: int,
    source_row_index: int,
    feature_value: float,
    *,
    context_id: int = 0,
    support_example_index: int = 0,
    example_id: str | None = None,
) -> dict:
    return {
        "scan_record_id": scan_record_id,
        "source_row_index": source_row_index,
        "example_id": example_id or f"ex_{source_row_index}",
        "context_id": context_id,
        "support_example_index": support_example_index,
        "prompt": f"prompt {source_row_index}",
        "answer": "A",
        "target_first_token_id": 1,
        "target_token_length": 1,
        "model_correct_first_token": True,
        "feature_activations": {7: feature_value},
    }


def test_induction_selection_is_stable_for_shuffled_scan_records() -> None:
    records = [
        _record(3, 30, 0.7, context_id=1, support_example_index=0),
        _record(1, 10, 0.9, context_id=0, support_example_index=0),
        _record(2, 20, 0.8, context_id=0, support_example_index=1),
        _record(4, 40, 0.6, context_id=1, support_example_index=1),
    ]
    selected_a, stats_a = select_induction_contexts(
        records,
        [7],
        tau_act=0.0,
        max_contexts=3,
        min_contexts=1,
        stratify_by=["context_id", "support_example_index"],
    )
    selected_b, stats_b = select_induction_contexts(
        list(reversed(records)),
        [7],
        tau_act=0.0,
        max_contexts=3,
        min_contexts=1,
        stratify_by=["context_id", "support_example_index"],
    )

    assert [row["source_row_index"] for row in selected_a[7]] == [
        row["source_row_index"] for row in selected_b[7]
    ]
    assert stats_a == stats_b


def test_induction_ties_use_sparse_identity_not_dense_index() -> None:
    records = [
        _record(9, 90, 1.0, context_id=0, support_example_index=0, example_id="b"),
        _record(2, 20, 1.0, context_id=0, support_example_index=0, example_id="a"),
        _record(1, 10, 1.0, context_id=0, support_example_index=0, example_id="a"),
    ]

    selected, _ = select_induction_contexts(
        records,
        [7],
        tau_act=0.0,
        max_contexts=3,
        min_contexts=1,
        stratify_by=["context_id", "support_example_index"],
    )

    assert [
        (row["example_id"], row["scan_record_id"], row["source_row_index"])
        for row in selected[7]
    ] == [("a", 1, 10), ("a", 2, 20), ("b", 9, 90)]


def test_induction_round_robin_respects_max_contexts_and_min_contexts() -> None:
    records = [
        _record(0, 0, 0.9, context_id=0, support_example_index=0),
        _record(1, 1, 0.8, context_id=0, support_example_index=0),
        _record(2, 2, 0.7, context_id=1, support_example_index=0),
        _record(3, 3, 0.6, context_id=1, support_example_index=0),
    ]

    selected, stats = select_induction_contexts(
        records,
        [7],
        tau_act=0.0,
        max_contexts=2,
        min_contexts=1,
        stratify_by=["context_id"],
    )

    assert len(selected[7]) == 2
    assert {row["context_id"] for row in selected[7]} == {0, 1}
    assert stats[7]["capped"] == 2

    dropped, dropped_stats = select_induction_contexts(
        records[:1],
        [7],
        tau_act=0.0,
        max_contexts=2,
        min_contexts=2,
        stratify_by=["context_id"],
    )
    assert dropped[7] == []
    assert dropped_stats[7]["too_rare"] == 1


def test_dense_indices_assigned_after_selected_prompt_union() -> None:
    selected = {
        7: [_record(2, 20, 0.8), _record(1, 10, 0.9)],
        8: [_record(1, 10, 0.5), _record(3, 30, 0.4)],
    }

    dense = assign_dense_indices(selected)

    assert dense == {(1, 10): 0, (2, 20): 1, (3, 30): 2}
