from __future__ import annotations

import numpy as np
import pandas as pd

from sae_bench.evals.icl_features.aggregate_results import benjamini_hochberg
from sae_bench.evals.icl_features.feature_sets import matched_random_feature_sets
from sae_bench.evals.icl_features.iou import jaccard
from sae_bench.evals.icl_features.statistics import (
    empirical_upper_tail,
    exact_mcnemar,
)
from sae_bench.evals.induction_features.analysis import build_feature_metrics_frame


def test_threshold_and_strict_feature_semantics() -> None:
    metrics, summary = build_feature_metrics_frame(
        sae_uid="sae",
        sae_release="release",
        sae_id="id",
        layer=1,
        hook_name="blocks.1.hook_resid_post",
        example_active_counts=np.array([10, 9, 8], dtype=np.int64),
        activation_sum=np.array([10.0, 9.0, 8.0]),
        active_activation_sum=np.array([10.0, 9.0, 8.0]),
        max_activation=np.ones(3),
        slot_active_counts=np.array([[10, 9, 8]], dtype=np.int64),
        slot_totals=np.array([10], dtype=np.int64),
        context_feature_query_counts=np.array(
            [[5, 5, 4], [5, 4, 4]], dtype=np.uint8
        ),
        context_totals=np.array([5, 5], dtype=np.int64),
        analyzed_example_count=10,
        min_example_fraction=0.9,
        min_query_fraction=0.8,
        min_context_fraction=1.0,
    )
    assert summary["candidate_feature_ids"] == [0, 1]
    assert summary["strict_common_feature_ids"] == [0]
    assert metrics["is_strict_common_feature"].tolist() == [True, False, False]


def test_exact_paired_significance_and_empirical_control() -> None:
    result = exact_mcnemar([True] * 10, [True] * 7 + [False] * 3)
    assert result["reference_correct_comparison_wrong"] == 3
    assert abs(result["p_value_one_sided"] - 0.125) < 1e-12
    assert empirical_upper_tail(0.2, [0.1, 0.2, 0.3]) == 0.75


def test_matched_random_sets_are_reproducible_and_exclude_selected() -> None:
    metrics = pd.DataFrame(
        {
            "feature_id": range(20),
            "example_prevalence": np.linspace(0.1, 1.0, 20),
            "mean_activation_when_active": np.linspace(0.2, 2.0, 20),
        }
    )
    first = matched_random_feature_sets(
        metrics=metrics,
        selected_feature_ids=[17, 18, 19],
        trials=3,
        seed=7,
        match_pool_size=5,
    )
    second = matched_random_feature_sets(
        metrics=metrics,
        selected_feature_ids=[17, 18, 19],
        trials=3,
        seed=7,
        match_pool_size=5,
    )
    assert first == second
    assert all(len(values) == 3 for values in first)
    assert all(not set(values) & {17, 18, 19} for values in first)


def test_iou_and_bh_helpers() -> None:
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard(set(), set()) is None
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.04, 0.04]
