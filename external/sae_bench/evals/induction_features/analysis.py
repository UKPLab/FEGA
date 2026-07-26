from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def choose_count_dtype(max_count: int) -> type[np.unsignedinteger[Any]]:
    """Choose the smallest unsigned integer dtype that can hold max_count."""
    if max_count <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_count <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_count <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def compute_context_prevalence_counts(
    context_feature_query_counts: np.ndarray,
    context_totals: np.ndarray,
    min_query_fraction: float,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Count how often each feature appears in any / sufficiently many queries per context."""
    if context_feature_query_counts.ndim != 2:
        raise ValueError("context_feature_query_counts must have shape [n_contexts, d_sae]")
    if context_totals.ndim != 1:
        raise ValueError("context_totals must have shape [n_contexts]")
    if context_feature_query_counts.shape[0] != context_totals.shape[0]:
        raise ValueError("context_feature_query_counts and context_totals must agree on n_contexts")
    if not 0.0 <= min_query_fraction <= 1.0:
        raise ValueError("min_query_fraction must lie in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    valid_context_mask = context_totals > 0
    num_valid_contexts = int(valid_context_mask.sum())
    d_sae = int(context_feature_query_counts.shape[1])

    any_context_counts = np.zeros(d_sae, dtype=np.int64)
    consistent_context_counts = np.zeros(d_sae, dtype=np.int64)

    if num_valid_contexts == 0:
        return any_context_counts, consistent_context_counts, 0

    valid_counts = context_feature_query_counts[valid_context_mask]
    required_query_counts = np.ceil(
        context_totals[valid_context_mask] * min_query_fraction
    ).astype(valid_counts.dtype, copy=False)

    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        chunk = valid_counts[:, start:end]
        any_context_counts[start:end] = np.count_nonzero(chunk > 0, axis=0)
        consistent_context_counts[start:end] = np.count_nonzero(
            chunk >= required_query_counts[:, None],
            axis=0,
        )

    return any_context_counts, consistent_context_counts, num_valid_contexts


def build_feature_metrics_frame(
    *,
    sae_uid: str,
    sae_release: str,
    sae_id: str,
    layer: int,
    hook_name: str,
    example_active_counts: np.ndarray,
    activation_sum: np.ndarray,
    active_activation_sum: np.ndarray,
    max_activation: np.ndarray,
    slot_active_counts: np.ndarray,
    slot_totals: np.ndarray,
    context_feature_query_counts: np.ndarray,
    context_totals: np.ndarray,
    analyzed_example_count: int,
    min_example_fraction: float,
    min_query_fraction: float,
    min_context_fraction: float,
    context_chunk_size: int = 4096,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert streamed feature statistics into a feature-level dataframe and summary."""
    if analyzed_example_count < 0:
        raise ValueError("analyzed_example_count must be non-negative")
    if not 0.0 <= min_example_fraction <= 1.0:
        raise ValueError("min_example_fraction must lie in [0, 1]")
    if not 0.0 <= min_context_fraction <= 1.0:
        raise ValueError("min_context_fraction must lie in [0, 1]")
    if slot_active_counts.ndim != 2:
        raise ValueError("slot_active_counts must have shape [n_slots, d_sae]")
    if slot_totals.ndim != 1:
        raise ValueError("slot_totals must have shape [n_slots]")
    if slot_active_counts.shape[0] != slot_totals.shape[0]:
        raise ValueError("slot_active_counts and slot_totals must agree on n_slots")

    d_sae = int(example_active_counts.shape[0])
    if any(arr.shape[0] != d_sae for arr in (activation_sum, active_activation_sum, max_activation)):
        raise ValueError("All per-feature arrays must have the same length")

    any_context_counts, consistent_context_counts, num_valid_contexts = (
        compute_context_prevalence_counts(
            context_feature_query_counts=context_feature_query_counts,
            context_totals=context_totals,
            min_query_fraction=min_query_fraction,
            chunk_size=context_chunk_size,
        )
    )

    denominator_examples = max(analyzed_example_count, 1)
    denominator_contexts = max(num_valid_contexts, 1)

    example_prevalence = example_active_counts / denominator_examples
    any_context_prevalence = any_context_counts / denominator_contexts
    consistent_context_prevalence = consistent_context_counts / denominator_contexts
    mean_activation = activation_sum / denominator_examples
    mean_activation_when_active = active_activation_sum / np.maximum(example_active_counts, 1)

    candidate_mask = (
        (example_prevalence >= min_example_fraction)
        & (consistent_context_prevalence >= min_context_fraction)
    )
    strict_common_mask = example_active_counts == analyzed_example_count

    data: dict[str, Any] = {
        "feature_uid": [f"{sae_uid}::{feature_id}" for feature_id in range(d_sae)],
        "sae_uid": sae_uid,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "layer": layer,
        "hook_name": hook_name,
        "feature_id": np.arange(d_sae, dtype=np.int64),
        "num_active_examples": example_active_counts,
        "example_prevalence": example_prevalence,
        "num_active_contexts_any": any_context_counts,
        "context_prevalence_any": any_context_prevalence,
        "num_consistent_contexts": consistent_context_counts,
        "consistent_context_prevalence": consistent_context_prevalence,
        "mean_activation": mean_activation,
        "mean_activation_when_active": mean_activation_when_active,
        "max_activation": max_activation,
        "is_candidate_feature": candidate_mask,
        "is_strict_common_feature": strict_common_mask,
    }

    for slot_index in range(slot_active_counts.shape[0]):
        slot_denominator = max(int(slot_totals[slot_index]), 1)
        data[f"slot_{slot_index}_example_prevalence"] = (
            slot_active_counts[slot_index] / slot_denominator
        )

    feature_metrics = pd.DataFrame(data)

    summary = {
        "sae_uid": sae_uid,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "layer": layer,
        "hook_name": hook_name,
        "total_features": d_sae,
        "analyzed_example_count": analyzed_example_count,
        "analyzed_context_count": num_valid_contexts,
        "min_example_fraction": min_example_fraction,
        "min_query_fraction_per_context": min_query_fraction,
        "min_context_fraction": min_context_fraction,
        "candidate_feature_count": int(candidate_mask.sum()),
        "candidate_feature_fraction": float(candidate_mask.mean()) if d_sae > 0 else 0.0,
        "strict_common_feature_count": int(strict_common_mask.sum()),
        "strict_common_feature_fraction": float(strict_common_mask.mean()) if d_sae > 0 else 0.0,
        "any_context_feature_count": int((any_context_counts > 0).sum()),
        "candidate_feature_ids": np.flatnonzero(candidate_mask).tolist(),
        "strict_common_feature_ids": np.flatnonzero(strict_common_mask).tolist(),
    }

    return feature_metrics, summary
