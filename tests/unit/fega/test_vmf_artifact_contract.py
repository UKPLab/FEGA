from __future__ import annotations

import copy
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
import pytest

from fega.core.source_fingerprint import canonical_json_digest
from fega.core.vmf.artifacts import (
    ASSIGNMENT_METRIC_IDENTITY,
    VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION,
    VMF_SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
    VmfArtifactValidationError,
    validate_vmf_scores,
)
from fega.core.vmf.metrics import (
    FITTING_ELIGIBILITY_MIN_ROWS,
    NORMALIZATION_EPSILON,
    PUBLIC_METRIC_KEYS,
    SEED_DERIVATION_VERSION,
    derived_vmf_seed,
    feature_fit_seed,
)


def _valid_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one complete M=2 artifact with two deterministic ARI replicates."""
    # The validator fixture isolates public provenance without running a scientific fit.
    feature_id = 7
    n_valid = 8
    effective_seed = 42
    feature_seed = feature_fit_seed(effective_seed, feature_id)
    feature_ids = [feature_id]
    fingerprint: dict[str, Any] = {
        "schema_version": VMF_SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
        "effect_space": "pre_softcap_logits",
        "source_readout": "final_resid",
        "geometry_metrics_effect_space": "final_resid",
        "vmf_backend_fingerprints": {"authority": "fixture"},
        "backend": "dense_cpu",
        "gpu_device": "cuda:0",
        "candidate_mode_counts": [1, 2],
        "bic_tolerance": 1.0e-9,
        "n_init": 1,
        "max_iter": 20,
        "effective_seed": effective_seed,
        "fitting_eligibility_min_rows": FITTING_ELIGIBILITY_MIN_ROWS,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "assignment_fraction": 0.8,
        "assignment_rounds": 2,
        "seed_derivations": {
            "version": SEED_DERIVATION_VERSION,
            "feature": "base_seed + feature_id * 104729",
            "candidate": "fixture",
            "assignment_subset": "fixture",
            "assignment_refit": "fixture",
            "byte_order": "big",
        },
        "assignment_metric": {
            "identity": ASSIGNMENT_METRIC_IDENTITY,
            "distribution": "scikit-learn",
            "version": version("scikit-learn"),
        },
        "canonical_source": {"digest": "source"},
        "materialization_policy": {"formula": "fixture"},
        "vmf_config_hash": "config",
        "effect_manifest_hash": "manifest",
        "effect_summary_hash": "summary",
        "geometry_metrics_scores_hash": "geometry",
        "feature_ids": feature_ids,
        "feature_inventory_sha256": canonical_json_digest(feature_ids),
    }
    replicates = []
    scores = (0.75, 1.0)
    subset_n = 7
    for replicate_id, score in enumerate(scores):
        subset_seed = derived_vmf_seed(
            feature_seed, 2, replicate_id, "subset"
        )
        replicates.append(
            {
                "replicate_id": replicate_id,
                "subset_seed": subset_seed,
                "refit_seed": derived_vmf_seed(
                    feature_seed, 2, replicate_id, "refit"
                ),
                "subset_indices": np.sort(
                    np.random.default_rng(subset_seed).choice(
                        n_valid, size=subset_n, replace=False
                    )
                ).tolist(),
                "status": "available",
                "adjusted_rand_score": score,
            }
        )
    metrics = {key: None for key in PUBLIC_METRIC_KEYS}
    metrics.update(
        {
            "selected_mode_count": 2,
            "delta_mix": 0.2,
            "mode_mass_min": 0.5,
            "min_mode_c_ray": 0.9,
            "mode_kappa_min": 2.0,
        }
    )
    payload = {
        "phase": "vmf",
        "schema_version": VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "effect_space": "pre_softcap_logits",
        "source_readout": "final_resid",
        "geometry_metrics_effect_space": "final_resid",
        "canonical_source_fingerprint": fingerprint["canonical_source"],
        "fingerprint": fingerprint,
        "features": [
            {
                "feature_id": feature_id,
                "n_valid": n_valid,
                "fit_status": "fitted",
                "model_selection": {
                    "selected_mode_count": 2,
                    "bic_tolerance": 1.0e-9,
                    "candidates": [
                        {
                            "mode_count": 1,
                            "status": "finite",
                            "seed": derived_vmf_seed(
                                feature_seed, 1, -1, "candidate_fit"
                            ),
                            "log_likelihood": -10.0,
                            "bic": 30.0,
                        },
                        {
                            "mode_count": 2,
                            "status": "finite",
                            "seed": derived_vmf_seed(
                                feature_seed, 2, -1, "candidate_fit"
                            ),
                            "log_likelihood": -20.0,
                            "bic": 20.0,
                        },
                    ],
                    "attempted_count": 2,
                    "finite_count": 2,
                    "nonfinite_count": 0,
                    "failed_count": 0,
                },
                "selected_fit": {
                    "weights": [0.5, 0.5],
                    "kappas": [2.0, 3.0],
                    "hard_mode_counts": [4, 4],
                    "hard_assignments": [0, 0, 0, 0, 1, 1, 1, 1],
                },
                "assignment_stability": {
                    "status": "available",
                    "value": sum(scores) / len(scores),
                    "requested_count": 2,
                    "successful_count": 2,
                    "failed_count": 0,
                    "replicates": replicates,
                },
                "metrics": metrics,
            }
        ],
    }
    return payload, fingerprint


def _old_schema(payload: dict[str, Any]) -> None:
    payload.pop("schema_version")


def _fingerprint_drift(payload: dict[str, Any]) -> None:
    payload["fingerprint"]["vmf_backend_fingerprints"] = {"authority": "drift"}


def _missing_feature(payload: dict[str, Any]) -> None:
    payload["features"] = []


def _duplicate_feature(payload: dict[str, Any]) -> None:
    payload["features"].append(copy.deepcopy(payload["features"][0]))


def _missing_candidate(payload: dict[str, Any]) -> None:
    payload["features"][0]["model_selection"]["candidates"].pop()


def _reordered_candidates(payload: dict[str, Any]) -> None:
    payload["features"][0]["model_selection"]["candidates"].reverse()


def _bic_selection_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["model_selection"]["selected_mode_count"] = 1


def _selected_fit_count_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["selected_fit"]["hard_mode_counts"] = [5, 3]


def _replicate_seed_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["replicates"][0][
        "subset_seed"
    ] += 1


def _replicate_subset_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["replicates"][0][
        "subset_indices"
    ][0] = 99


def _replicate_score_missing(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["replicates"][0].pop(
        "adjusted_rand_score"
    )


def _assignment_count_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["successful_count"] = 1


def _assignment_aggregate_drift(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["value"] = 0.0


def _corrupt_metric_type(payload: dict[str, Any]) -> None:
    payload["features"][0]["metrics"]["delta_mix"] = "0.2"


def _metric_fit_mismatch(payload: dict[str, Any]) -> None:
    payload["features"][0]["metrics"]["mode_mass_min"] = 0.4


def _negative_weight(payload: dict[str, Any]) -> None:
    payload["features"][0]["selected_fit"]["weights"][0] = -0.5


def _out_of_range_ari(payload: dict[str, Any]) -> None:
    payload["features"][0]["assignment_stability"]["replicates"][0][
        "adjusted_rand_score"
    ] = 1.01


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_old_schema, "top-level keys mismatch"),
        (_fingerprint_drift, "scientific fingerprint mismatch"),
        (_missing_feature, "feature inventory mismatch"),
        (_duplicate_feature, "feature inventory mismatch"),
        (_missing_candidate, "candidate schedule mismatch"),
        (_reordered_candidates, "candidate schedule mismatch"),
        (_bic_selection_drift, "BIC reselection mismatch"),
        (_selected_fit_count_drift, "selected_fit mode counts mismatch"),
        (_replicate_seed_drift, "assignment subset seed mismatch"),
        (_replicate_subset_drift, "assignment subset mismatch"),
        (_replicate_score_missing, "assignment replicate keys mismatch"),
        (_assignment_count_drift, "assignment successful_count mismatch"),
        (_assignment_aggregate_drift, "assignment aggregate mismatch"),
        (_corrupt_metric_type, "metric delta_mix must be an exact finite number"),
        (_metric_fit_mismatch, "mode_mass_min/selected_fit mismatch"),
        (_negative_weight, "selected_fit weight must be nonnegative"),
        (_out_of_range_ari, r"adjusted Rand score outside \[-1, 1\]"),
    ],
)
def test_vmf_artifact_validation_fails_closed_on_vital_corruption(
    mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Reject every result- or reproducibility-changing public artifact corruption."""
    # Keep expected provenance immutable while mutating only the persisted artifact.
    payload, fingerprint = _valid_artifact()
    expected = copy.deepcopy(fingerprint)
    mutation(payload)

    with pytest.raises(VmfArtifactValidationError, match=message):
        validate_vmf_scores(payload, expected_fingerprint=expected)


def test_vmf_artifact_compatibility_ignores_execution_only_resources() -> None:
    """Reuse identical vMF science across buffer and redundant config metadata drift."""
    # Model the completed eight-worker artifact consumed by four-worker reporting.
    payload, fingerprint = _valid_artifact()
    payload["fingerprint"]["materialization_policy"]["max_vocab_buffers"] = 8
    payload["fingerprint"]["vmf_config_hash"] = "producer-config"
    expected = copy.deepcopy(fingerprint)
    expected["materialization_policy"]["max_vocab_buffers"] = 4
    expected["vmf_config_hash"] = "reporting-config"

    validate_vmf_scores(payload, expected_fingerprint=expected)


def test_vmf_artifact_accepts_complete_candidate_failure_schedule() -> None:
    """Keep a recorded current-version candidate failure as valid provenance."""
    # M=2 fails explicitly; finite M=1 remains the BIC-selected model.
    payload, fingerprint = _valid_artifact()
    feature = payload["features"][0]
    failed = feature["model_selection"]["candidates"][1]
    failed.pop("log_likelihood")
    failed.pop("bic")
    failed["status"] = "fit_failed"
    feature["model_selection"].update(
        {
            "selected_mode_count": 1,
            "finite_count": 1,
            "failed_count": 1,
        }
    )
    feature["metrics"]["selected_mode_count"] = 1
    feature["metrics"]["mode_mass_min"] = 1.0
    feature["selected_fit"] = {
        "weights": [1.0],
        "kappas": [2.0],
        "hard_mode_counts": [8],
        "hard_assignments": [0] * 8,
    }
    feature["assignment_stability"] = {
        "status": "not_applicable",
        "value": None,
        "requested_count": 0,
        "successful_count": 0,
        "failed_count": 0,
        "replicates": [],
    }

    validate_vmf_scores(payload, expected_fingerprint=fingerprint)


def test_vmf_artifact_accepts_complete_unavailable_assignment_evidence() -> None:
    """Keep a failed selected-model refit as valid unavailable mixture evidence."""
    # One required replicate fails, so the ordered aggregate is unavailable, not partial.
    payload, fingerprint = _valid_artifact()
    assignment = payload["features"][0]["assignment_stability"]
    failed = assignment["replicates"][1]
    failed["status"] = "fit_failed"
    failed.pop("adjusted_rand_score")
    assignment.update(
        {
            "status": "unavailable",
            "value": None,
            "successful_count": 1,
            "failed_count": 1,
        }
    )

    validate_vmf_scores(payload, expected_fingerprint=fingerprint)
