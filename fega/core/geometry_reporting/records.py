from __future__ import annotations

import copy
import math
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    point_selection_identity,
    resolve_point_selection,
)
from fega.core.source_fingerprint import canonical_json_digest

GEOMETRY_METRIC_KEYS = (
    "r2",
    "c_ray",
    "s_span_1",
    "s_span_2",
    "s_span_3",
    "s_span_4",
    "s_span_8",
    "r_span_pr",
    "r_span_ent",
    "u_span_1",
    "u_span_2",
    "u_span_3",
    "u_span_4",
    "u_span_8",
    "d_span_1",
    "d_span_2",
    "d_span_3",
    "d_span_4",
    "d_span_8",
    "b_axis",
    "e_res",
    "s_res_1",
    "s_res_2",
    "s_res_3",
    "s_res_4",
    "r_ctr_pr",
    "r_ctr_ent",
)
VMF_KEYS = (
    "selected_mode_count",
    "delta_mix",
    "mode_mass_min",
    "min_mode_c_ray",
    "mode_kappa_min",
)
BANNED_KEYS = {
    "label_agreement",
    "selected_k_agreement",
    "low_strength",
    "top_positive_readout_tokens",
    "top_negative_readout_tokens",
    "mode_exemplars",
    "projection_histogram_summary",
    "likely_noise",
    "outlier_sensitive",
    "r2_mean_resultant",
}
ALLOWED_STABILITY_KS = {"1", "2", "3", "4", "8"}


def build_geometry_records(
    inputs: dict[str, Any], config: FEGAPipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join validated point records to raw WP3 evidence without recomputation."""
    # Preserve loader-validated point order and attach only the matching raw evidence.
    del config
    point_records = inputs.get("point_records")
    if not isinstance(point_records, list):
        raise ValueError("validated point records are missing")
    stability_final = _stability_effect_space(
        inputs["payloads"]["stability"], "final_resid"
    )
    stability_per_feature = _mapping(stability_final.get("per_feature"))
    expected_keys = [str(int(record["feature_id"])) for record in point_records]
    if list(stability_per_feature) != expected_keys:
        raise ValueError("stability feature inventory order mismatch")
    records: list[dict[str, Any]] = []
    for point_record in point_records:
        feature_key = str(int(point_record["feature_id"]))
        stability_record = stability_per_feature[feature_key]
        if not isinstance(stability_record, dict):
            raise ValueError(f"stability feature {feature_key} record is invalid")
        raw_evidence = stability_record.get("selected_family_evidence")
        if not isinstance(raw_evidence, dict):
            raise ValueError(
                f"stability feature {feature_key} selected-family evidence is missing"
            )
        record = copy.deepcopy(point_record)
        record.pop("general_stability", None)
        record["selected_family_evidence"] = copy.deepcopy(raw_evidence)
        records.append(record)
    return records, {
        "features_total": len(records),
        "source_paths": inputs["paths"],
    }


def build_point_geometry_records(
    inputs: dict[str, Any], config: FEGAPipelineConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build reporting-owned point records without consuming stability evidence.

    Every record is resolved from canonical compute-effect, geometry metrics, and
    validated standalone-vMF inputs only. Its hash covers the complete point record
    before the hash field itself is attached.
    """
    # Require the loader's exact common inventory rather than unioning partial inputs.
    payloads = inputs["payloads"]
    compute = payloads["compute_effect_final_resid"]
    geometry_metrics = payloads["geometry_metrics_final_resid"]
    vmf_by_feature = _vmf_by_feature(payloads["vmf_pre_softcap_logits"])
    compute_per_feature = _mapping(compute.get("per_feature"))
    geometry_per_feature = _mapping(geometry_metrics.get("per_feature"))
    feature_ids = [int(value) for value in inputs["feature_ids"]]
    expected_keys = [str(value) for value in feature_ids]
    if list(sorted(compute_per_feature, key=lambda key: int(key))) != expected_keys:
        raise ValueError("compute-effect feature inventory mismatch")
    if list(sorted(geometry_per_feature, key=lambda key: int(key))) != expected_keys:
        raise ValueError("geometry-metrics feature inventory mismatch")
    if list(sorted(vmf_by_feature, key=lambda key: int(key))) != expected_keys:
        raise ValueError("standalone-vMF feature inventory mismatch")

    # Construct and resolve each feature in canonical feature order.
    summary = _mapping(compute.get("summary"))
    tau_zero = _finite_float(summary.get("tau_zero"))
    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for feature_id in feature_ids:
        key = str(feature_id)
        record = _build_record(
            key,
            compute_per_feature[key],
            geometry_per_feature[key],
            vmf_by_feature[key],
            None,
            tau_zero=tau_zero,
            eps=config.phases.geometry_reporting.eps,
        )
        for stability_key in (
            "scalar_ci",
            "subspace_stability",
            "centered_residual_subspace_stability",
            "low_context",
            "sample_size_curves",
            "leave_out_sensitivity",
        ):
            record.pop(stability_key, None)
        selection = resolve_point_selection(
            record,
            config.phases.geometry_reporting.threshold_profile,
            vmf_provenance_valid=True,
        )
        record["point_selection"] = point_selection_identity(selection)
        record_hash = canonical_json_digest(record)
        record["point_record_sha256"] = record_hash
        record_hashes[key] = record_hash
        records.append(record)
    inventory = [
        {
            "feature_id": record["feature_id"],
            "point_record_sha256": record_hashes[str(record["feature_id"])],
        }
        for record in records
    ]
    return records, {
        "features_total": len(records),
        "source_paths": inputs["paths"],
        "point_selection_contract_version": POINT_SELECTION_CONTRACT_VERSION,
        "point_record_hashes": record_hashes,
        "point_records_sha256": canonical_json_digest(inventory),
    }


def _build_record(
    feature_key: str,
    compute_record: Any,
    geometry_metrics_record: Any,
    vmf_record: Any,
    stability_record: Any,
    *,
    tau_zero: float | None,
    eps: float,
) -> dict[str, Any]:
    """Merge phase artifacts while preserving independent vMF result dimensions.

    Flattened vMF metrics remain available to established geometry consumers,
    while operational fit, model selection, selected parameters, and assignment
    stability are carried intact for truthful reporting and provenance.
    """
    # Normalize missing phase records before deriving the shared feature identity.
    compute_record = compute_record if isinstance(compute_record, dict) else {}
    geometry_metrics_record = (
        geometry_metrics_record if isinstance(geometry_metrics_record, dict) else {}
    )
    vmf_record = vmf_record if isinstance(vmf_record, dict) else {}
    stability_record = stability_record if isinstance(stability_record, dict) else {}
    feature_id = int(
        compute_record.get("feature_id")
        or geometry_metrics_record.get("feature_id")
        or vmf_record.get("feature_id")
        or stability_record.get("feature_id")
        or feature_key
    )
    loaded_contexts = _finite_float(compute_record.get("loaded_contexts"))
    n_filtered_zero = _finite_float(compute_record.get("skipped_near_zero"))
    zero_denom = max(int(loaded_contexts or 0), 1)
    zero_filter_frac = (
        None if n_filtered_zero is None else float(n_filtered_zero) / float(zero_denom)
    )
    record: dict[str, Any] = {
        "feature_id": feature_id,
        "layer": None,
        "sae_id": None,
        "readout": "final_resid",
        "context_set_id": None,
        "intervention_sign": None,
        "n_valid": _first_float(
            geometry_metrics_record.get("n_valid"), compute_record.get("usable_effects")
        ),
        "n_filtered_zero": _json_number(n_filtered_zero),
        "zero_filter_frac": _json_number(zero_filter_frac),
        "tau_zero": _json_number(tau_zero),
        "m_median": _json_number(compute_record.get("median_magnitude")),
        "m_q10": _json_number(compute_record.get("q10_magnitude")),
        "m_q90": _json_number(compute_record.get("q90_magnitude")),
        "m_cv": _json_number(compute_record.get("cv_magnitude")),
        "eps": float(eps),
        "missingness": {},
        "source_paths": {},
    }
    for key in GEOMETRY_METRIC_KEYS:
        record[key] = _json_number(geometry_metrics_record.get(key))
    for key in VMF_KEYS:
        record[key] = _json_number((vmf_record.get("metrics") or vmf_record).get(key))
    record["fit_status"] = vmf_record.get("fit_status")
    record["model_selection"] = _mapping(vmf_record.get("model_selection"))
    selected_fit = vmf_record.get("selected_fit")
    record["selected_fit"] = selected_fit if isinstance(selected_fit, dict) else None
    record["assignment_stability"] = _mapping(vmf_record.get("assignment_stability"))
    record["scalar_ci"] = _scalar_ci(stability_record)
    _copy_subspace(record, stability_record)
    _copy_centered_residual(record, stability_record)
    record["low_context"] = _low_context_summary(stability_record.get("low_context"))
    record["sample_size_curves"] = _sample_size_summary(
        stability_record.get("sample_size_curves")
    )
    record["leave_out_sensitivity"] = _leave_out_summary(
        stability_record.get("leave_out_sensitivity")
    )
    record["missingness"] = _missingness(
        record, compute_record, geometry_metrics_record, vmf_record
    )
    if compute_record.get("usable_effects") != geometry_metrics_record.get("n_valid"):
        record["missingness"]["n_valid_consistency"] = "mismatch"
    return record


def _vmf_by_feature(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = payload.get("features")
    if isinstance(features, list):
        return {
            str(int(item["feature_id"])): item
            for item in features
            if isinstance(item, dict) and "feature_id" in item
        }
    per_feature = payload.get("per_feature")
    if isinstance(per_feature, dict):
        return {
            str(int(item.get("feature_id", key))): item
            for key, item in per_feature.items()
            if isinstance(item, dict)
        }
    return {}


def _stability_effect_space(payload: dict[str, Any], effect_space: str) -> dict[str, Any]:
    effect_spaces = payload.get("effect_spaces")
    if not isinstance(effect_spaces, dict):
        return {}
    block = effect_spaces.get(effect_space)
    return block if isinstance(block, dict) else {}


def _copy_subspace(record: dict[str, Any], stability_record: dict[str, Any]) -> None:
    block = stability_record.get("subspace_stability")
    if not isinstance(block, dict):
        record["subspace_stability"] = {"status": "not_available", "k": {}}
        return
    sanitized = {
        "status": block.get("status"),
        "n_valid": block.get("n_valid"),
        "numerical_rank": block.get("numerical_rank"),
        "orthonormality_error": _json_number(block.get("orthonormality_error")),
        "k": {},
    }
    per_k = block.get("k")
    if isinstance(per_k, dict):
        for k, entry in per_k.items():
            key = str(k)
            if key not in ALLOWED_STABILITY_KS or not isinstance(entry, dict):
                continue
            angle = _json_number(entry.get("subspace_angle_p90_k"))
            sanitized["k"][key] = {
                "status": entry.get("status"),
                "subspace_angle_p90_k": angle,
            }
            record[f"subspace_angle_p90_{key}"] = angle
    record["subspace_stability"] = sanitized


def _copy_centered_residual(
    record: dict[str, Any], stability_record: dict[str, Any]
) -> None:
    block = stability_record.get("centered_residual_subspace_stability")
    if not isinstance(block, dict):
        record["centered_residual_subspace_stability"] = {
            "status": "not_available",
            "source": "not_available",
            "k": {},
        }
        return
    sanitized = {
        "status": block.get("status"),
        "source": "stability_artifact",
        "n_valid": block.get("n_valid"),
        "numerical_rank": block.get("numerical_rank"),
        "orthonormality_error": _json_number(block.get("orthonormality_error")),
        "k": {},
    }
    per_k = block.get("k")
    if isinstance(per_k, dict):
        for k, entry in per_k.items():
            if not isinstance(entry, dict):
                continue
            sanitized["k"][str(k)] = {
                "status": entry.get("status"),
                "residual_angle_p90_k": _json_number(
                    entry.get("residual_angle_p90_k")
                ),
            }
    record["centered_residual_subspace_stability"] = sanitized


def _scalar_ci(stability_record: dict[str, Any]) -> dict[str, Any]:
    """Preserve every stability interval and its complete replicate evidence."""
    # Read the unified bootstrap interval map rather than projecting only C-ray.
    scalar = stability_record.get("scalar_stability")
    if not isinstance(scalar, dict):
        return {}
    bootstrap = scalar.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return {}
    intervals = bootstrap.get("intervals")
    if isinstance(intervals, dict):
        return _sanitize_value(intervals)
    return {}


def _missingness(
    record: dict[str, Any],
    compute_record: dict[str, Any],
    geometry_metrics_record: dict[str, Any],
    vmf_record: dict[str, Any],
) -> dict[str, Any]:
    missing = {
        "compute_effect_final_resid": not bool(compute_record),
        "geometry_metrics_final_resid": not bool(geometry_metrics_record),
        "vmf_pre_softcap_logits": not bool(vmf_record),
    }
    for key in (
        "c_ray",
        "s_span_1",
        "s_span_2",
        "r_span_pr",
        "u_span_2",
        "d_span_2",
        "e_res",
        "r_ctr_pr",
    ):
        missing[key] = record.get(key) is None
    return missing


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _low_context_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in ("status", "protocol", "n_valid")
        if key in value
    }


def _sample_size_summary(value: Any) -> dict[str, Any]:
    """Preserve the full sample-size plan, margins, crossings, and failures."""
    # Reporting consumers require raw denominators and per-gate evidence without projection.
    if not isinstance(value, dict):
        return {}
    return _sanitize_value(value)


def _leave_out_summary(value: Any) -> dict[str, Any]:
    """Preserve complete leave-one/group profile and change evidence."""
    # Keep immutable identities, failures, labels, selected k, and gate crossings intact.
    if not isinstance(value, dict):
        return {}
    return _sanitize_value(value)


def _spread_or_target_summary(value: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "target_size",
        "subset_count",
        "sampling_mode",
        "median",
        "q10",
        "q90",
        "min",
        "max",
    )
    return {key: value.get(key) for key in allowed if key in value}


def _group_sampling_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: value.get(key)
        for key in ("status", "groups", "usable_groups")
        if key in value
    }
    return _sanitize_value(summary)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if str(key) not in BANNED_KEYS and "residual_angle" not in str(key)
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _json_number(value: Any) -> float | int | None:
    parsed = _finite_float(value)
    if parsed is None:
        return None
    if float(parsed).is_integer():
        return int(parsed)
    return parsed


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
