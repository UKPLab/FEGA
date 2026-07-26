from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from fega.config_schema import DirectionalMixtureFitConfig, FEGAPipelineConfig
from fega.core.resources import ModelResources
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.vmf.fit import vmf_backend_fingerprints
from fega.core.vmf.metrics import (
    FITTING_ELIGIBILITY_MIN_ROWS,
    NORMALIZATION_EPSILON,
    PUBLIC_METRIC_KEYS,
    SEED_DERIVATION_VERSION,
    derived_vmf_seed,
    feature_fit_seed,
    select_mode_count_from_bic_records,
)
from fega.paths import geometry_metrics_scores_path

VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION = 1
VMF_SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION = 3
ASSIGNMENT_METRIC_IDENTITY = "sklearn.metrics.adjusted_rand_score"
_VOCAB_MATERIALIZATION_CHUNK_SIZE = 16_384


class VmfArtifactValidationError(ValueError):
    """Raised when a standalone vMF artifact cannot be reused safely."""


def load_geometry_metrics_scores(
    config: FEGAPipelineConfig,
    effect_space: str,
    resources: ModelResources | None = None,
) -> dict[str, Any]:
    """Load current geometry_metrics scores used to gate FEGA vMF fitting."""
    path = geometry_metrics_scores_path(config, effect_space)
    if not path.exists():
        raise FileNotFoundError(f"Missing geometry_metrics scores for vmf: {path}")
    if resources is not None:
        cached = resources.get_cached_json(path)
        if cached is not None:
            return cached
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"geometry_metrics scores must be a JSON object: {path}")
    if resources is not None:
        resources.cache_json(path, payload)
    return payload


def write_vmf_scores(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_fingerprint: dict[str, Any],
) -> None:
    """Validate and atomically write the FEGA vMF public score artifact."""
    # Refuse to publish a record that later reporting would reject as incomplete.
    validate_vmf_scores(payload, expected_fingerprint=expected_fingerprint)
    _write_json_atomic(path, payload)


def feature_ids_from_summary(per_feature_summary: dict[str, Any]) -> list[int]:
    """Return the exact ordered feature inventory used by standalone vMF."""
    # Canonicalize by numeric feature key and reject duplicate embedded identities.
    feature_ids: list[int] = []
    seen: set[int] = set()
    for feature_key in sorted(per_feature_summary, key=lambda key: int(key)):
        record = per_feature_summary[feature_key]
        if not isinstance(record, dict):
            raise ValueError(f"Feature {feature_key} summary must be a mapping.")
        feature_id = _coerce_feature_id(record.get("feature_id", feature_key))
        if feature_id in seen:
            raise ValueError(f"Duplicate feature_id in effect_summary: {feature_id}")
        seen.add(feature_id)
        feature_ids.append(feature_id)
    return feature_ids


def vmf_materialization_policy(
    cfg: DirectionalMixtureFitConfig,
) -> dict[str, Any]:
    """Describe the existing bounded scientific coordinate materialization policy."""
    # Scheduling worker count stays outside this independently enforced identity.
    return {
        "formula": "final_resid_direction@canonical_unembedding.T",
        "output_dtype": "float32",
        "vocab_chunk_size": _VOCAB_MATERIALIZATION_CHUNK_SIZE,
        "normalization": "exact_l2_no_epsilon",
        "max_vocab_buffers": int(cfg.max_vocab_buffers),
    }


def build_vmf_scientific_fingerprint(
    *,
    config: FEGAPipelineConfig,
    cfg: DirectionalMixtureFitConfig,
    seed: int,
    inputs_manifest: dict[str, Any],
    inputs_summary: dict[str, Any],
    geometry_metrics_scores: dict[str, Any],
    feature_ids: list[int],
    source_fingerprint: dict[str, Any],
    materialization_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build the one scientific identity shared by checkpoints and public output."""
    # Exclude scheduling-only controls while publishing every fitting/assignment input.
    vmf_config = dict(config.to_dict()["phases"]["vmf"])
    vmf_config.pop("resume", None)
    vmf_config.pop("workers", None)
    vmf_config.pop("checkpoint_flush_features", None)
    candidate_mode_counts = [int(k) for k in sorted(set(cfg.k_values))]
    seed_derivations = {
        "version": SEED_DERIVATION_VERSION,
        "feature": "base_seed + feature_id * 104729",
        "candidate": "sha256(vmf|feature_seed|mode_count|-1|candidate_fit)[:8]",
        "assignment_subset": "sha256(vmf|feature_seed|mode_count|replicate_id|subset)[:8]",
        "assignment_refit": "sha256(vmf|feature_seed|mode_count|replicate_id|refit)[:8]",
        "byte_order": "big",
    }
    return {
        "schema_version": VMF_SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
        "effect_space": cfg.effect_space,
        "source_readout": "final_resid",
        "geometry_metrics_effect_space": "final_resid",
        "vmf_backend_fingerprints": vmf_backend_fingerprints(
            backend=cfg.backend,
            gpu_device=cfg.gpu_device,
        ),
        "backend": cfg.backend,
        "gpu_device": cfg.gpu_device,
        "candidate_mode_counts": candidate_mode_counts,
        "bic_tolerance": float(cfg.bic_tolerance),
        "n_init": int(cfg.n_init),
        "max_iter": int(cfg.max_iter),
        "effective_seed": int(seed),
        "fitting_eligibility_min_rows": FITTING_ELIGIBILITY_MIN_ROWS,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "assignment_fraction": float(cfg.resample_fraction),
        "assignment_rounds": int(cfg.resample_rounds),
        "seed_derivations": seed_derivations,
        "assignment_metric": {
            "identity": ASSIGNMENT_METRIC_IDENTITY,
            "distribution": "scikit-learn",
            "version": version("scikit-learn"),
        },
        "canonical_source": source_fingerprint,
        "materialization_policy": materialization_policy,
        "vmf_config_hash": canonical_json_digest(vmf_config),
        "effect_manifest_hash": canonical_json_digest(inputs_manifest),
        "effect_summary_hash": canonical_json_digest(inputs_summary),
        "geometry_metrics_scores_hash": canonical_json_digest(
            geometry_metrics_scores
        ),
        "feature_ids": feature_ids,
        "feature_inventory_sha256": canonical_json_digest(feature_ids),
    }


def validate_vmf_scores(
    payload: dict[str, Any], *, expected_fingerprint: dict[str, Any]
) -> None:
    """Fail closed on any standalone-vMF schema, schedule, or provenance mismatch."""
    # Validate artifact identity before inspecting feature-local scientific evidence.
    required_top_level = {
        "phase",
        "schema_version",
        "effect_space",
        "source_readout",
        "geometry_metrics_effect_space",
        "canonical_source_fingerprint",
        "fingerprint",
        "features",
    }
    if set(payload) != required_top_level:
        _invalid("top-level keys mismatch")
    if payload.get("schema_version") != VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION:
        _invalid(
            "public schema version mismatch; rerun the standalone vMF phase"
        )
    if payload.get("phase") != "vmf":
        _invalid("phase mismatch")
    if payload.get("effect_space") != "pre_softcap_logits":
        _invalid("effect_space mismatch")
    if payload.get("source_readout") != "final_resid":
        _invalid("source_readout mismatch")
    if payload.get("geometry_metrics_effect_space") != "final_resid":
        _invalid("geometry_metrics_effect_space mismatch")
    if vmf_scientific_compatibility_identity(
        payload.get("fingerprint")
    ) != vmf_scientific_compatibility_identity(expected_fingerprint):
        _invalid("scientific fingerprint mismatch; rerun the standalone vMF phase")
    if payload.get("canonical_source_fingerprint") != expected_fingerprint.get(
        "canonical_source"
    ):
        _invalid("canonical source fingerprint mismatch")
    _validate_fingerprint_contract(expected_fingerprint)

    features = payload.get("features")
    expected_feature_ids = expected_fingerprint["feature_ids"]
    if not isinstance(features, list):
        _invalid("features must be a list")
    observed_ids = [
        _exact_feature_id(feature.get("feature_id"))
        if isinstance(feature, dict)
        else _invalid("feature record must be a mapping")
        for feature in features
    ]
    if observed_ids != expected_feature_ids:
        _invalid("feature inventory mismatch")
    for feature in features:
        _validate_feature_record(feature, expected_fingerprint)


def vmf_scientific_compatibility_identity(
    fingerprint: Any,
) -> Any:
    """Project a vMF fingerprint onto decision-changing scientific identity.

    Buffer count and the redundant whole-config digest describe execution rather
    than the fitted population, method, or decisions. They remain recorded in the
    artifact but do not invalidate reuse.
    """
    # Preserve every explicit scientific field while dropping execution-only metadata.
    if not isinstance(fingerprint, dict):
        return fingerprint
    identity = dict(fingerprint)
    identity.pop("vmf_config_hash", None)
    materialization = identity.get("materialization_policy")
    if isinstance(materialization, dict):
        materialization = dict(materialization)
        materialization.pop("max_vocab_buffers", None)
        identity["materialization_policy"] = materialization
    return identity


def _validate_fingerprint_contract(fingerprint: dict[str, Any]) -> None:
    """Require every public scientific identity field and frozen derivation string."""
    # Exact field names prevent a partial or alternate provenance vocabulary.
    required = {
        "schema_version",
        "effect_space",
        "source_readout",
        "geometry_metrics_effect_space",
        "vmf_backend_fingerprints",
        "backend",
        "gpu_device",
        "candidate_mode_counts",
        "bic_tolerance",
        "n_init",
        "max_iter",
        "effective_seed",
        "fitting_eligibility_min_rows",
        "normalization_epsilon",
        "assignment_fraction",
        "assignment_rounds",
        "seed_derivations",
        "assignment_metric",
        "canonical_source",
        "materialization_policy",
        "vmf_config_hash",
        "effect_manifest_hash",
        "effect_summary_hash",
        "geometry_metrics_scores_hash",
        "feature_ids",
        "feature_inventory_sha256",
    }
    if set(fingerprint) != required:
        _invalid("scientific fingerprint keys mismatch")
    if fingerprint["schema_version"] != VMF_SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION:
        _invalid("scientific fingerprint schema version mismatch")
    if fingerprint["fitting_eligibility_min_rows"] != FITTING_ELIGIBILITY_MIN_ROWS:
        _invalid("fitting eligibility minimum mismatch")
    if fingerprint["normalization_epsilon"] != NORMALIZATION_EPSILON:
        _invalid("normalization epsilon mismatch")
    if fingerprint["seed_derivations"].get("version") != SEED_DERIVATION_VERSION:
        _invalid("seed derivation version mismatch")
    if fingerprint["assignment_metric"] != {
        "identity": ASSIGNMENT_METRIC_IDENTITY,
        "distribution": "scikit-learn",
        "version": version("scikit-learn"),
    }:
        _invalid("assignment metric identity mismatch")
    if fingerprint["feature_inventory_sha256"] != canonical_json_digest(
        fingerprint["feature_ids"]
    ):
        _invalid("feature inventory digest mismatch")


def _validate_feature_record(
    feature: dict[str, Any], fingerprint: dict[str, Any]
) -> None:
    """Validate one complete feature candidate and assignment schedule."""
    # Orthogonal fit-state dimensions must remain explicit and independently checkable.
    required = {
        "feature_id",
        "n_valid",
        "fit_status",
        "model_selection",
        "selected_fit",
        "assignment_stability",
        "metrics",
    }
    feature_id = _exact_feature_id(feature.get("feature_id"))
    if set(feature) != required:
        _invalid(f"feature {feature_id} state keys mismatch")
    n_valid = _nonnegative_int(feature.get("n_valid"), f"feature {feature_id} n_valid")
    fit_status = feature.get("fit_status")
    if fit_status not in {
        "fitted",
        "insufficient_contexts",
        "no_finite_candidate",
        "fit_failed",
    }:
        _invalid(f"feature {feature_id} fit_status invalid")
    metrics = feature.get("metrics")
    if not isinstance(metrics, dict) or tuple(metrics) != PUBLIC_METRIC_KEYS:
        _invalid(f"feature {feature_id} metrics keys mismatch")
    model_selection = feature.get("model_selection")
    if not isinstance(model_selection, dict) or set(model_selection) != {
        "selected_mode_count",
        "bic_tolerance",
        "candidates",
        "attempted_count",
        "finite_count",
        "nonfinite_count",
        "failed_count",
    }:
        _invalid(f"feature {feature_id} model_selection keys mismatch")
    if model_selection["bic_tolerance"] != fingerprint["bic_tolerance"]:
        _invalid(f"feature {feature_id} BIC tolerance mismatch")
    feature_seed = feature_fit_seed(fingerprint["effective_seed"], feature_id)
    feasible_modes = (
        [
            mode
            for mode in fingerprint["candidate_mode_counts"]
            if mode <= n_valid
        ]
        if n_valid >= fingerprint["fitting_eligibility_min_rows"]
        else []
    )
    candidates = model_selection.get("candidates")
    if not isinstance(candidates, list):
        _invalid(f"feature {feature_id} candidates must be a list")
    if [candidate.get("mode_count") for candidate in candidates] != feasible_modes:
        _invalid(f"feature {feature_id} candidate schedule mismatch")
    statuses = []
    for candidate in candidates:
        statuses.append(
            _validate_candidate(candidate, feature_id=feature_id, feature_seed=feature_seed)
        )
    expected_counts = {
        "attempted_count": len(candidates),
        "finite_count": statuses.count("finite"),
        "nonfinite_count": statuses.count("nonfinite"),
        "failed_count": statuses.count("fit_failed"),
    }
    if any(model_selection[key] != value for key, value in expected_counts.items()):
        _invalid(f"feature {feature_id} candidate counts mismatch")

    finite_candidates = [
        candidate for candidate in candidates if candidate["status"] == "finite"
    ]
    selected_mode = model_selection.get("selected_mode_count")
    if finite_candidates:
        selected_mode_int = _positive_int(
            selected_mode, f"feature {feature_id} selected mode count"
        )
        expected_selected = select_mode_count_from_bic_records(
            finite_candidates, tolerance=fingerprint["bic_tolerance"]
        )
        if fit_status != "fitted" or selected_mode_int != expected_selected:
            _invalid(f"feature {feature_id} BIC reselection mismatch")
        if metrics["selected_mode_count"] != selected_mode_int:
            _invalid(f"feature {feature_id} selected mode identity mismatch")
        selected_fit = _validate_selected_fit(
            feature.get("selected_fit"),
            feature_id=feature_id,
            selected_mode=selected_mode_int,
            n_valid=n_valid,
        )
        _validate_fitted_metrics(
            metrics,
            selected_fit=selected_fit,
            selected_mode=selected_mode_int,
            feature_id=feature_id,
        )
    else:
        if selected_mode is not None or feature.get("selected_fit") is not None:
            _invalid(f"feature {feature_id} non-fitted state has selected model")
        if any(metrics[key] is not None for key in PUBLIC_METRIC_KEYS):
            _invalid(f"feature {feature_id} non-fitted metrics must be null")
        expected_status = (
            "insufficient_contexts"
            if n_valid < fingerprint["fitting_eligibility_min_rows"]
            else (
                "fit_failed"
                if candidates and all(status == "fit_failed" for status in statuses)
                else "no_finite_candidate"
            )
        )
        if fit_status != expected_status:
            _invalid(f"feature {feature_id} non-fitted status mismatch")
    _validate_assignment(
        feature.get("assignment_stability"),
        feature_id=feature_id,
        feature_seed=feature_seed,
        selected_mode=selected_mode,
        n_valid=n_valid,
        fingerprint=fingerprint,
    )


def _validate_candidate(
    candidate: Any, *, feature_id: int, feature_seed: int
) -> str:
    """Validate one scheduled candidate record and its deterministic seed."""
    # Candidate failure is valid evidence only when the scheduled record is complete.
    if not isinstance(candidate, dict):
        _invalid(f"feature {feature_id} candidate must be a mapping")
    status = candidate.get("status")
    if status not in {"finite", "nonfinite", "fit_failed"}:
        _invalid(f"feature {feature_id} candidate status invalid")
    required = {"mode_count", "status", "seed"}
    if status == "finite":
        required |= {"log_likelihood", "bic"}
    if set(candidate) != required:
        _invalid(f"feature {feature_id} candidate keys mismatch")
    mode_count = _positive_int(
        candidate.get("mode_count"), f"feature {feature_id} candidate mode_count"
    )
    expected_seed = derived_vmf_seed(
        feature_seed, mode_count, -1, "candidate_fit"
    )
    if candidate.get("seed") != expected_seed:
        _invalid(f"feature {feature_id} candidate seed mismatch")
    if status == "finite":
        _finite_number(
            candidate.get("log_likelihood"),
            f"feature {feature_id} candidate log_likelihood",
        )
        _finite_number(candidate.get("bic"), f"feature {feature_id} candidate BIC")
    return str(status)


def _validate_selected_fit(
    selected_fit: Any, *, feature_id: int, selected_mode: int, n_valid: int
) -> dict[str, Any]:
    """Validate the compact selected-model dimensions needed by assignment evidence."""
    # Preserve parameters and hard assignments without re-fitting or re-scoring them.
    if not isinstance(selected_fit, dict) or set(selected_fit) != {
        "weights",
        "kappas",
        "hard_mode_counts",
        "hard_assignments",
    }:
        _invalid(f"feature {feature_id} selected_fit keys mismatch")
    weights = selected_fit["weights"]
    if not isinstance(weights, list) or len(weights) != selected_mode:
        _invalid(f"feature {feature_id} selected_fit weights mismatch")
    for value in weights:
        _nonnegative_finite_number(
            value, f"feature {feature_id} selected_fit weight"
        )
    kappas = selected_fit["kappas"]
    if kappas is not None:
        if not isinstance(kappas, list) or len(kappas) != selected_mode:
            _invalid(f"feature {feature_id} selected_fit kappas mismatch")
        for value in kappas:
            _nonnegative_finite_number(
                value, f"feature {feature_id} selected_fit kappa"
            )
    mode_counts = selected_fit["hard_mode_counts"]
    if not isinstance(mode_counts, list) or len(mode_counts) != selected_mode:
        _invalid(f"feature {feature_id} selected_fit mode counts mismatch")
    for value in mode_counts:
        _nonnegative_int(value, f"feature {feature_id} selected_fit mode count")
    assignments = selected_fit["hard_assignments"]
    if not isinstance(assignments, list) or len(assignments) != n_valid or any(
        not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < selected_mode
        for value in assignments
    ):
        _invalid(f"feature {feature_id} hard assignments mismatch")
    expected_counts = [assignments.count(mode) for mode in range(selected_mode)]
    if mode_counts != expected_counts:
        _invalid(f"feature {feature_id} selected_fit mode counts mismatch")
    return selected_fit


def _validate_fitted_metrics(
    metrics: dict[str, Any],
    *,
    selected_fit: dict[str, Any],
    selected_mode: int,
    feature_id: int,
) -> None:
    """Require exact finite metric values and reconcile selected-fit minima."""
    metric_mode = _positive_int(
        metrics.get("selected_mode_count"),
        f"feature {feature_id} metric selected_mode_count",
    )
    if metric_mode != selected_mode:
        _invalid(f"feature {feature_id} selected mode identity mismatch")
    for key in ("delta_mix", "mode_mass_min", "min_mode_c_ray", "mode_kappa_min"):
        value = metrics.get(key)
        if value is not None:
            _finite_number(value, f"feature {feature_id} metric {key}")

    expected_mass_min = min(selected_fit["weights"])
    if metrics["mode_mass_min"] != expected_mass_min:
        _invalid(f"feature {feature_id} mode_mass_min/selected_fit mismatch")
    kappas = selected_fit["kappas"]
    expected_kappa_min = None if kappas is None else min(kappas)
    if metrics["mode_kappa_min"] != expected_kappa_min:
        _invalid(f"feature {feature_id} mode_kappa_min/selected_fit mismatch")


def _validate_assignment(
    assignment: Any,
    *,
    feature_id: int,
    feature_seed: int,
    selected_mode: Any,
    n_valid: int,
    fingerprint: dict[str, Any],
) -> None:
    """Reconcile every deterministic assignment replicate and aggregate."""
    # A failed refit is valid unavailable evidence; an incomplete plan is invalid.
    if not isinstance(assignment, dict) or set(assignment) != {
        "status",
        "value",
        "requested_count",
        "successful_count",
        "failed_count",
        "replicates",
    }:
        _invalid(f"feature {feature_id} assignment_stability keys mismatch")
    if selected_mode is None:
        _require_assignment_state(
            assignment,
            feature_id=feature_id,
            status="unavailable",
            value=None,
            requested=0,
            successful=0,
            failed=0,
            replicates=[],
        )
        return
    selected_mode = int(selected_mode)
    if selected_mode == 1:
        _require_assignment_state(
            assignment,
            feature_id=feature_id,
            status="not_applicable",
            value=None,
            requested=0,
            successful=0,
            failed=0,
            replicates=[],
        )
        return
    rounds = int(fingerprint["assignment_rounds"])
    replicates = assignment.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != rounds:
        _invalid(f"feature {feature_id} assignment replicate inventory mismatch")
    subset_n = min(
        n_valid,
        max(
            selected_mode,
            int(math.ceil(float(fingerprint["assignment_fraction"]) * n_valid)),
        ),
    )
    scores: list[float] = []
    for replicate_id, replicate in enumerate(replicates):
        subset_seed = derived_vmf_seed(
            feature_seed, selected_mode, replicate_id, "subset"
        )
        refit_seed = derived_vmf_seed(
            feature_seed, selected_mode, replicate_id, "refit"
        )
        expected_subset = np.sort(
            np.random.default_rng(subset_seed).choice(
                n_valid, size=subset_n, replace=False
            )
        ).tolist()
        if not isinstance(replicate, dict):
            _invalid(f"feature {feature_id} assignment replicate must be a mapping")
        status = replicate.get("status")
        required = {
            "replicate_id",
            "subset_seed",
            "refit_seed",
            "subset_indices",
            "status",
        }
        if status == "available":
            required.add("adjusted_rand_score")
        elif status not in {"fit_failed", "nonfinite"}:
            _invalid(f"feature {feature_id} assignment replicate status invalid")
        if set(replicate) != required:
            _invalid(f"feature {feature_id} assignment replicate keys mismatch")
        if replicate.get("replicate_id") != replicate_id:
            _invalid(f"feature {feature_id} assignment replicate id mismatch")
        if replicate.get("subset_seed") != subset_seed:
            _invalid(f"feature {feature_id} assignment subset seed mismatch")
        if replicate.get("refit_seed") != refit_seed:
            _invalid(f"feature {feature_id} assignment refit seed mismatch")
        if replicate.get("subset_indices") != expected_subset:
            _invalid(f"feature {feature_id} assignment subset mismatch")
        if status == "available":
            scores.append(
                _adjusted_rand_score(
                    replicate.get("adjusted_rand_score"), feature_id=feature_id
                )
            )
    successful = len(scores)
    failed = rounds - successful
    expected_status = "available" if rounds > 0 and failed == 0 else "unavailable"
    expected_value = float(sum(scores) / rounds) if expected_status == "available" else None
    _require_assignment_state(
        assignment,
        feature_id=feature_id,
        status=expected_status,
        value=expected_value,
        requested=rounds,
        successful=successful,
        failed=failed,
        replicates=replicates,
    )


def _require_assignment_state(
    assignment: dict[str, Any],
    *,
    feature_id: int,
    status: str,
    value: float | None,
    requested: int,
    successful: int,
    failed: int,
    replicates: list[Any],
) -> None:
    """Require exact assignment state fields after independent reconciliation."""
    # Exact equality preserves the ordered mean and closed availability vocabulary.
    if assignment.get("status") != status:
        _invalid(f"feature {feature_id} assignment status mismatch")
    actual_value = assignment.get("value")
    if value is None:
        if actual_value is not None:
            _invalid(f"feature {feature_id} assignment aggregate mismatch")
    elif _adjusted_rand_score(actual_value, feature_id=feature_id) != value:
        _invalid(f"feature {feature_id} assignment aggregate mismatch")
    if _nonnegative_int(
        assignment.get("requested_count"),
        f"feature {feature_id} assignment requested_count",
    ) != requested:
        _invalid(f"feature {feature_id} assignment requested_count mismatch")
    if _nonnegative_int(
        assignment.get("successful_count"),
        f"feature {feature_id} assignment successful_count",
    ) != successful:
        _invalid(f"feature {feature_id} assignment successful_count mismatch")
    if _nonnegative_int(
        assignment.get("failed_count"),
        f"feature {feature_id} assignment failed_count",
    ) != failed:
        _invalid(f"feature {feature_id} assignment failed_count mismatch")
    if assignment.get("replicates") != replicates:
        _invalid(f"feature {feature_id} assignment replicates mismatch")


def _coerce_feature_id(value: Any) -> int:
    """Coerce an integer feature identity without accepting booleans."""
    # Boolean JSON values are not feature identifiers despite int subclassing.
    if isinstance(value, bool):
        _invalid(f"feature_id must be an integer, got {value!r}")
    try:
        return int(value)
    except Exception:
        _invalid(f"feature_id must be an integer, got {value!r}")


def _exact_feature_id(value: Any) -> int:
    """Return an exact integer identity from one public feature record."""
    # Public JSON records cannot rely on the summary loader's string-key coercion.
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"feature_id must be an integer, got {value!r}")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    """Return one exact nonnegative integer or fail artifact validation."""
    # Reject truncating floats and boolean values before schedule arithmetic.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    """Return one exact positive integer or fail artifact validation."""
    # Candidate mode counts must be discrete and strictly positive.
    parsed = _nonnegative_int(value, label)
    if parsed <= 0:
        _invalid(f"{label} must be positive")
    return parsed


def _finite_number(value: Any, label: str) -> float:
    """Return one finite non-boolean number from public JSON evidence."""
    # NaN and infinity cannot authenticate BIC or adjusted-Rand evidence.
    if isinstance(value, bool) or not isinstance(value, int | float):
        _invalid(f"{label} must be an exact finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        _invalid(f"{label} must be an exact finite number")
    return parsed


def _nonnegative_finite_number(value: Any, label: str) -> float:
    """Return one exact finite nonnegative public fit parameter."""
    parsed = _finite_number(value, label)
    if parsed < 0.0:
        _invalid(f"{label} must be nonnegative")
    return parsed


def _adjusted_rand_score(value: Any, *, feature_id: int) -> float:
    """Validate the universal adjusted-Rand metric range at the public boundary."""
    parsed = _finite_number(value, f"feature {feature_id} adjusted Rand score")
    if not -1.0 <= parsed <= 1.0:
        _invalid(f"feature {feature_id} adjusted Rand score outside [-1, 1]")
    return parsed


def _invalid(reason: str) -> NoReturn:
    """Raise the typed fail-closed standalone-vMF validation error."""
    # Keep every failure explicit and actionable at the public artifact boundary.
    raise VmfArtifactValidationError(f"invalid standalone vMF artifact: {reason}")


def load_vmf_checkpoint(path: Path) -> dict[str, Any] | None:
    """Load VMF resume checkpoint state, returning None when absent."""
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"vmf checkpoint must be a JSON object: {path}")
    return payload


def write_vmf_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write VMF resume checkpoint state."""
    _write_json_atomic(path, payload)


def delete_vmf_checkpoint(path: Path) -> None:
    """Remove VMF resume checkpoint state if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)
