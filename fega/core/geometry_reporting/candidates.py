from __future__ import annotations

from typing import Any

from .evidence import (
    all_family_anchors_false,
    axis_anchor,
    axis_point_comparisons,
    c_ray_ci_status,
    centered_residual_status,
    directed_ray_point_comparisons,
    finite_float,
    leave_out_status,
    long_tail,
    multimode_anchor,
    one_d_diffuse_condition,
    positive_high_dimensional_or_diffuse,
    ray_anchor,
    residual_anchor,
    residual_point_comparisons,
    sample_size_curve_status,
    span_anchor,
    span_point_comparisons,
    subspace_status,
    subspace_threshold_defined,
)
from .schema import FALLBACK_PRIORITY
from .thresholds import GeometryThresholds


def candidate_labels(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for helper in (
        _directed_ray_candidate,
        _axis_candidate,
        _one_d_diffuse_candidate,
        _multimode_candidate,
        _span_candidate,
        _residual_candidate,
    ):
        candidate = helper(record, thresholds, gate_evidence)
        if candidate is not None:
            candidates.append(candidate)
    family_candidates = [
        candidate
        for candidate in candidates
        if candidate["family"] != "unresolved_high_dimensional_or_diffuse"
    ]
    high_dimensional = _high_dimensional_candidate(
        record, thresholds, bool(family_candidates)
    )
    if high_dimensional is not None:
        candidates.append(high_dimensional)
    return sorted(
        candidates,
        key=lambda item: (
            int(item["priority"]),
            -1 if item.get("selected_k") is None else int(item["selected_k"]),
            str(item["family"]),
        ),
    )


def candidate_primary_k(primary_label: str, candidate: dict[str, Any]) -> int | None:
    if primary_label in {
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
        "residual_lowD_k",
    }:
        selected_k = candidate.get("selected_k")
        return int(selected_k) if selected_k is not None else None
    return None


def _directed_ray_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    if not ray_anchor(record, thresholds):
        return None
    metrics = directed_ray_point_comparisons(record, thresholds)
    failed, missing = _failed_missing_fields(metrics)
    flags = []
    r_span_pr = finite_float(record.get("r_span_pr"))
    boundary_upper = thresholds.tau_r.get(2)
    if (
        r_span_pr is not None
        and boundary_upper is not None
        and thresholds.tau_r_2D <= r_span_pr < boundary_upper
    ):
        flags.append("ray_span_boundary")
    ci_status = c_ray_ci_status(record, thresholds.tau_c_ray, "ge")
    if ci_status == "not_available":
        flags.append("directed_ray_ci_missing")
        missing.append("scalar_ci.c_ray")
    elif ci_status == "unstable":
        flags.append("directed_ray_ci_unstable")
        failed.append("scalar_ci.c_ray")
    details = {
        "gate_decision": gate_evidence["directed_ray"]["decision"],
        "ray_span_boundary": {
            "r_span_pr": r_span_pr,
            "lower": thresholds.tau_r_2D,
            "upper": boundary_upper,
        },
    }
    _add_sample_details(record, flags, details)
    return _candidate(
        "directed_ray",
        selected_k=None,
        confidence=_candidate_confidence(gate_evidence["directed_ray"]),
        anchor_fields=["c_ray", "s_span_1"],
        failed_fields=failed,
        missing_fields=missing,
        flags=flags,
        details=details,
    )


def _axis_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    if not axis_anchor(record, thresholds):
        return None
    metrics = axis_point_comparisons(record, thresholds)
    failed, missing = _failed_missing_fields(metrics)
    flags = []
    ci_status = c_ray_ci_status(record, thresholds.tau_c_ray, "lt")
    if ci_status == "not_available":
        flags.append("axis_ci_missing")
        missing.append("scalar_ci.c_ray")
    elif ci_status == "unstable":
        flags.append("axis_ci_unstable")
        failed.append("scalar_ci.c_ray")
    stability_status = subspace_status(record, 1, thresholds)
    if stability_status == "not_available":
        flags.append("axis_stability_missing")
        missing.append("subspace_stability.k.1")
    elif stability_status == "unstable":
        flags.append("axis_stability_failed")
        failed.append("subspace_stability.k.1")
    details = {"gate_decision": gate_evidence["axis_or_antipodal"]["decision"]}
    _add_sample_details(record, flags, details)
    return _candidate(
        "axis_or_antipodal",
        selected_k=None,
        confidence=_candidate_confidence(gate_evidence["axis_or_antipodal"]),
        anchor_fields=["s_span_1", "c_ray", "b_axis"],
        failed_fields=failed,
        missing_fields=missing,
        flags=flags,
        details=details,
    )


def _one_d_diffuse_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    del gate_evidence
    if not one_d_diffuse_condition(record, thresholds):
        return None
    flags = ["oneD_not_ray_not_axis"]
    failed: list[str] = []
    missing: list[str] = []
    if finite_float(record.get("b_axis")) is None:
        flags.append("b_axis_missing")
        missing.append("b_axis")
    else:
        flags.append("b_axis_low")
        failed.append("b_axis")
    details = {"condition": "strong_1D_not_ray_not_axis"}
    _add_sample_details(record, flags, details)
    return _candidate(
        "oneD_diffuse",
        selected_k=None,
        confidence="candidate",
        anchor_fields=["s_span_1", "c_ray"],
        failed_fields=failed,
        missing_fields=missing,
        flags=flags,
        details=details,
    )


def _multimode_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the preserved multimode fallback from sole-authority gate evidence."""
    # Reuse reporting decisions so candidate construction cannot duplicate thresholds.
    if not multimode_anchor(record, thresholds):
        return None
    evidence = gate_evidence["multi_mode_directional_geometry"]
    gate_values = evidence.get("gate_values") or {}
    field_to_gate = {
        "selected_mode_count": None,
        "delta_mix": "gain",
        "mode_mass_min": "fitted_mass",
        "min_mode_c_ray": "within_mode_ray",
        "assignment_stability": "assignment_stability",
    }
    metrics = {
        field: (
            True
            if gate_name is None and evidence.get("evaluated")
            else (gate_values.get(gate_name) or {}).get("passed")
        )
        for field, gate_name in field_to_gate.items()
    }
    failed, missing = _failed_missing_fields(metrics)
    flags = []
    for field, missing_flag, failed_flag in (
        ("mode_mass_min", "mode_mass_missing", "mode_mass_failed"),
        ("min_mode_c_ray", "mode_c_ray_missing", "mode_c_ray_failed"),
        ("assignment_stability", "assignment_stability_missing", "assignment_stability_failed"),
    ):
        if metrics[field] is None:
            flags.append(missing_flag)
        elif metrics[field] is False:
            flags.append(failed_flag)
    if failed or missing:
        flags.append("multimode_candidate_blocked")
    details = {
        "gate_decision": evidence["decision"],
        "reporting_acceptance": evidence.get("acceptance"),
        "failed_gates": evidence.get("failed_gates", []),
        "unavailable_gates": evidence.get("unavailable_gates", []),
        "mode_metric_status": {
            field: metrics[field]
            for field in ("mode_mass_min", "min_mode_c_ray", "assignment_stability")
        },
    }
    _add_sample_details(record, flags, details)
    return _candidate(
        "multi_mode_directional_geometry",
        selected_k=None,
        confidence=_candidate_confidence(
            gate_evidence["multi_mode_directional_geometry"]
        ),
        anchor_fields=["selected_mode_count", "delta_mix"],
        failed_fields=failed,
        missing_fields=missing,
        flags=flags,
        details=details,
    )


def _span_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    for k in (2, 3, 4, 8):
        if not span_anchor(record, thresholds, k):
            continue
        supported = k in thresholds.tau_r and k in thresholds.tau_p
        label = (
            "global_2D_directional_subspace"
            if k == 2
            else "global_kD_directional_subspace"
        )
        metrics = span_point_comparisons(record, thresholds, k)
        failed, missing = _failed_missing_fields(metrics)
        flags = ["span_selected_k"]
        if not supported:
            flags.append("span_k_unsupported")
        stability_status = subspace_status(record, k, thresholds)
        if not subspace_threshold_defined(thresholds, k):
            flags.append("span_stability_threshold_missing")
            missing.append(f"tau_subspace_angle.{k}")
        elif stability_status == "not_available":
            flags.append("span_stability_missing")
            missing.append(f"subspace_stability.k.{k}")
        elif stability_status == "unstable":
            flags.append("span_stability_failed")
            failed.append(f"subspace_stability.k.{k}")
        if failed or missing:
            flags.append("lowD_candidate_blocked")
        gate = gate_evidence["global_directional_subspace"]
        details = {
            "gate_decision": gate["decision"],
            "supported": supported,
            "lowD_candidate_blocked": {
                "family": label,
                "selected_k": k,
                "nearest_alternative": label,
                "blocked_fields": failed,
                "missing_fields": missing,
                "stability_status": stability_status,
            },
            "span_selected_k": {
                "selected_k": k,
                "family": label,
                "supported": supported,
            },
        }
        _add_sample_details(record, flags, details)
        return _candidate(
            label,
            selected_k=k,
            confidence=_candidate_confidence(_attempt_for_k(gate, k, gate)),
            anchor_fields=[f"s_span_{k}"],
            failed_fields=failed,
            missing_fields=missing,
            flags=flags,
            details=details,
        )
    return None


def _residual_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the smallest residual candidate while preserving overlay evidence."""
    # Fallback primary uses the smallest anchored k, but directed-ray overlays
    # must still see any coexisting supported residual evidence at k >= 2.
    residual_candidates = [
        candidate
        for k in (1, 2, 3, 4)
        if (candidate := _residual_candidate_for_k(record, thresholds, gate_evidence, k))
        is not None
    ]
    if not residual_candidates:
        return None
    selected = residual_candidates[0]
    selected["details"]["directed_ray_residual_overlays"] = [
        candidate["details"]["directed_ray_with_lowD_residual"]
        for candidate in residual_candidates
        if int(candidate["selected_k"]) >= 2
    ]
    return selected


def _residual_candidate_for_k(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    gate_evidence: dict[str, Any],
    k: int,
) -> dict[str, Any] | None:
    """Build one residual candidate for an anchored residual dimension."""
    # Each candidate records support, stability, and blocked/missing fields for k.
    if not residual_anchor(record, thresholds, k):
        return None
    supported = k in thresholds.tau_ctr and k in thresholds.tau_r_ctr
    metrics = residual_point_comparisons(record, thresholds, k)
    failed, missing = _failed_missing_fields(metrics)
    flags = ["residual_selected_k"]
    stability_status = (
        centered_residual_status(record, k, thresholds)
        if supported
        else "not_available"
    )
    if supported and not subspace_threshold_defined(thresholds, k):
        flags.append("residual_stability_threshold_missing")
        missing.append(f"tau_subspace_angle.{k}")
    elif supported and stability_status == "not_available":
        flags.append("residual_stability_missing")
        missing.append(f"centered_residual_subspace_stability.k.{k}")
    elif supported and stability_status == "unstable":
        flags.append("residual_stability_failed")
        failed.append(f"centered_residual_subspace_stability.k.{k}")
    gate = gate_evidence["residual_lowD_k"]
    details = {
        "gate_decision": gate["decision"],
        "supported": supported,
        "residual_selected_k": {
            "selected_k": k,
            "supported": supported,
            "stability_status": stability_status,
        },
        "directed_ray_with_lowD_residual": {
            "residual_k": k,
            "supported": supported,
            "stability_status": stability_status,
            "failed_fields": failed,
            "missing_fields": missing,
        },
    }
    _add_sample_details(record, flags, details)
    return _candidate(
        "residual_lowD_k",
        selected_k=k,
        confidence=_candidate_confidence(_attempt_for_k(gate, k, gate)),
        anchor_fields=["e_res", f"s_res_{k}"],
        failed_fields=failed,
        missing_fields=missing,
        flags=flags,
        details=details,
    )


def _high_dimensional_candidate(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    has_family_candidate: bool,
) -> dict[str, Any] | None:
    if has_family_candidate:
        return None
    if positive_high_dimensional_or_diffuse(record, thresholds):
        flags = ["positive_highD_evidence"]
        details: dict[str, Any] = {"reason": "positive_highD_evidence"}
        _add_sample_details(record, flags, details)
        return _candidate(
            "unresolved_high_dimensional_or_diffuse",
            selected_k=None,
            confidence="unresolved",
            anchor_fields=["r_span_pr"],
            failed_fields=[],
            missing_fields=[],
            flags=flags,
            details=details,
        )
    if long_tail(record, thresholds) and all_family_anchors_false(record, thresholds):
        flags = ["long_tail_spectrum"]
        details = {"reason": "long_tail_spectrum"}
        _add_sample_details(record, flags, details)
        return _candidate(
            "unresolved_high_dimensional_or_diffuse",
            selected_k=None,
            confidence="unresolved",
            anchor_fields=["r_span_ent", "r_span_pr"],
            failed_fields=[],
            missing_fields=[],
            flags=flags,
            details=details,
        )
    return None


def _add_sample_details(
    record: dict[str, Any], flags: list[str], details: dict[str, Any]
) -> None:
    """Add sample-size and leave-out instability details to a candidate."""
    # These global flags can be the reason a strict gate is only a candidate.
    sample_status = sample_size_curve_status(record)
    if sample_status == "unstable":
        flags.append("sample_size_unstable")
        details["sample_size_unstable"] = {
            "status": sample_status,
            "evidence_source": "sample_size_curves",
            "blocked_fields": ["sample_size_curves"],
            "missing_fields": [],
        }
    leave_out = leave_out_status(record)
    if leave_out == "unstable":
        flags.append("leave_out_unstable")
        details["leave_out_unstable"] = {
            "status": leave_out,
            "evidence_source": "leave_out_sensitivity",
            "blocked_fields": ["leave_out_sensitivity"],
            "missing_fields": [],
        }


def _candidate(
    family: str,
    *,
    selected_k: int | None,
    confidence: str,
    anchor_fields: list[str],
    failed_fields: list[str],
    missing_fields: list[str],
    flags: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    normalized_details = dict(details)
    for flag in flags:
        normalized_details.setdefault(
            flag,
            {
                "family": family,
                "selected_k": selected_k,
                "anchor_fields": sorted(dict.fromkeys(anchor_fields)),
                "failed_fields": sorted(dict.fromkeys(failed_fields)),
                "missing_fields": sorted(dict.fromkeys(missing_fields)),
            },
        )
    return {
        "family": family,
        "priority": FALLBACK_PRIORITY[family],
        "selected_k": selected_k,
        "confidence": confidence,
        "anchor_fields": sorted(dict.fromkeys(anchor_fields)),
        "failed_fields": sorted(dict.fromkeys(failed_fields)),
        "missing_fields": sorted(dict.fromkeys(missing_fields)),
        "flags": sorted(dict.fromkeys(flags)),
        "details": normalized_details,
    }


def _candidate_confidence(evidence: dict[str, Any]) -> str:
    decision = str(evidence.get("decision", "not_available"))
    if decision in {"stable", "exploratory"}:
        return decision
    if decision == "not_available":
        return "candidate_missing_evidence"
    return "candidate_blocked"


def _attempt_for_k(gate: Any, k: int, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(gate, dict):
        attempts = gate.get("attempts")
        if isinstance(attempts, dict):
            attempt = attempts.get(str(k))
            if isinstance(attempt, dict):
                return attempt
    return fallback


def _failed_missing_fields(
    metrics: dict[str, bool | None],
) -> tuple[list[str], list[str]]:
    failed = [field for field, value in metrics.items() if value is False]
    missing = [field for field, value in metrics.items() if value is None]
    return failed, missing
