from __future__ import annotations

import copy
from typing import Any

from .candidates import candidate_labels, candidate_primary_k
from .evidence import (
    base_ok,
    base_terminal_reason,
    build_gate_evidence,
    build_point_gate_evidence,
    finite_float,
    ge,
    leave_out_status,
    long_tail,
    low_context_status,
    only_missing_evidence,
    sample_size_curve_status,
)
from .point_selection import point_selection_identity, resolve_point_selection
from .schema import (
    GLOBAL_FLAG_MASK,
    GLOBAL_FLAG_ORDER,
    LABEL_VERSION,
    PRIMARY_LABELS,
    TERMINAL_LABELS,
)
from .selected_family_stability import project_selected_family_stability
from .thresholds import GeometryThresholds, get_threshold_profile

__all__ = [
    "GLOBAL_FLAG_MASK",
    "GLOBAL_FLAG_ORDER",
    "LABEL_VERSION",
    "PRIMARY_LABELS",
    "TERMINAL_LABELS",
    "classify_record",
]

_LEGACY_CANDIDATE_STABILITY_FLAGS = {
    "axis_ci_missing",
    "axis_ci_unstable",
    "axis_stability_failed",
    "axis_stability_missing",
    "directed_ray_ci_missing",
    "directed_ray_ci_unstable",
    "leave_out_unstable",
    "residual_stability_failed",
    "residual_stability_missing",
    "residual_stability_threshold_missing",
    "sample_size_unstable",
    "span_stability_failed",
    "span_stability_missing",
    "span_stability_threshold_missing",
}
_LEGACY_CANDIDATE_STABILITY_PREFIXES = (
    "centered_residual_subspace_stability.",
    "scalar_ci.",
    "subspace_stability.",
    "tau_subspace_angle.",
)
_RETIRED_SELECTED_FAMILY_FLAGS = (
    _LEGACY_CANDIDATE_STABILITY_FLAGS
    - {"leave_out_unstable", "sample_size_unstable"}
)


def classify_record(
    record: dict[str, Any],
    profile: str | GeometryThresholds = "paper",
) -> dict[str, Any]:
    """Classify one point profile, then qualify it with stability evidence."""
    # Deep-copy the merged artifact record so reporting never mutates source evidence.
    thresholds = (
        get_threshold_profile(profile) if isinstance(profile, str) else profile
    )
    result = copy.deepcopy(record)
    locked_identity = result.get("point_selection")
    raw_selected_evidence = result.get("selected_family_evidence")
    locked_wp4_record = isinstance(locked_identity, dict) and isinstance(
        raw_selected_evidence, dict
    )
    gate_evidence = (
        build_point_gate_evidence(result, thresholds)
        if locked_wp4_record
        else build_gate_evidence(result, thresholds)
    )

    flags: list[str] = []
    flag_details: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    strict_gate_label: str | None = None
    label_confidence: str | None = None
    selected_k: int | None = None
    span_selected_k: int | None = None
    residual_selected_k: int | None = None
    terminal_reason: str | None = None

    if not base_ok(result, thresholds):
        primary_label = "insufficient_effect_evidence"
        terminal_reason = base_terminal_reason(result, thresholds)
        label_confidence = "insufficient"
        _add_flag(
            flags, flag_details, terminal_reason, "terminal", reason=terminal_reason
        )
    else:
        candidates = candidate_labels(result, thresholds, gate_evidence)
        if locked_wp4_record:
            candidates = _point_only_candidate_diagnostics(candidates)
        strict = _strict_primary(gate_evidence)
        if strict is not None:
            (
                primary_label,
                gate_key,
                selected_k,
                span_selected_k,
                residual_selected_k,
            ) = strict
            strict_gate_label = primary_label
            label_confidence = _strict_label_confidence(gate_evidence[gate_key])
        else:
            (
                primary_label,
                label_confidence,
                selected_k,
                span_selected_k,
                residual_selected_k,
                terminal_reason,
            ) = _fallback_primary(candidates, gate_evidence, flags, flag_details)
        _add_candidate_flags(flags, flag_details, candidates)
        if primary_label == "directed_ray":
            _add_directed_ray_overlay_flags(flags, flag_details, candidates)

    global_flags = _global_flags(result, thresholds, primary_label)
    for flag in global_flags:
        _add_flag(
            flags,
            flag_details,
            flag,
            "global",
            mask=GLOBAL_FLAG_MASK[flag],
            **_global_flag_details(result, thresholds, flag),
        )

    evidence_status = None
    selected_family_stability: dict[str, Any] | None = None
    if locked_wp4_record:
        # Rerun the reporting-owned point authority and reject any persisted drift.
        point_selection = resolve_point_selection(
            result, thresholds, vmf_provenance_valid=True
        )
        if point_selection_identity(point_selection) != locked_identity:
            raise ValueError("persisted point selection identity mismatch")
        primary_label = point_selection.family
        selected_k = point_selection.selected_k
        span_selected_k = (
            selected_k if primary_label.startswith("global_") else None
        )
        residual_selected_k = (
            selected_k if primary_label == "residual_lowD_k" else None
        )
        strict_gate_label = (
            primary_label if point_selection.mode == "strict" else None
        )
        (
            selected_family_stability,
            label_confidence,
            evidence_status,
            selected_family_flags,
        ) = project_selected_family_stability(
            point_selection_identity(point_selection),
            raw_selected_evidence,
            existing_confidence=label_confidence,
        )
        for selected_family_flag in selected_family_flags:
            _add_flag(
                flags,
                flag_details,
                selected_family_flag,
                "selected_family_stability",
                decision=selected_family_stability["decision"],
                evidence_status=evidence_status,
            )
        # Selected-family reporting never publishes retired qualifier flags.
        for obsolete_flag in _RETIRED_SELECTED_FAMILY_FLAGS:
            flags[:] = [flag for flag in flags if flag != obsolete_flag]
            flag_details.pop(obsolete_flag, None)

    # The v3 public contract has no heterogeneous general-stability alias.
    result.pop("general_stability", None)
    result.pop("selected_family_evidence", None)

    result.update(
        {
            "primary_label": primary_label,
            "secondary_flags": sorted(dict.fromkeys(flags)),
            "flag_details": flag_details,
            "candidate_labels": candidates,
            "strict_gate_label": strict_gate_label,
            "label_confidence": label_confidence,
            "evidence_status": evidence_status,
            "span_selected_k": span_selected_k,
            "residual_selected_k": residual_selected_k,
            "selected_k": selected_k,
            "terminal_reason": terminal_reason,
            "global_flags": global_flags,
            "global_flag_count": len(global_flags),
            "global_flag_mask": "|".join(
                GLOBAL_FLAG_MASK[flag] for flag in global_flags
            ),
            "label_version": LABEL_VERSION,
            "threshold_profile": thresholds.name,
            "gate_evidence": gate_evidence,
        }
    )
    if selected_family_stability is not None:
        result["selected_family_stability"] = selected_family_stability
    return result


def _point_only_candidate_diagnostics(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove retired non-mixture stability inputs from locked point candidates."""
    # Preserve point comparisons, candidate order, and the standalone mixture audit.
    for candidate in candidates:
        if candidate.get("family") == "multi_mode_directional_geometry":
            continue
        failed = _point_only_candidate_fields(candidate.get("failed_fields"))
        missing = _point_only_candidate_fields(candidate.get("missing_fields"))
        flags = [
            flag
            for flag in candidate.get("flags") or []
            if flag not in _LEGACY_CANDIDATE_STABILITY_FLAGS
        ]
        if not failed and not missing:
            flags = [flag for flag in flags if flag != "lowD_candidate_blocked"]
        details = candidate.get("details")
        if isinstance(details, dict):
            for flag in _LEGACY_CANDIDATE_STABILITY_FLAGS:
                details.pop(flag, None)
            if "lowD_candidate_blocked" not in flags:
                details.pop("lowD_candidate_blocked", None)
            for key in ("residual_selected_k", "directed_ray_with_lowD_residual"):
                detail = details.get(key)
                if isinstance(detail, dict):
                    detail.pop("stability_status", None)
            listed = details.get("directed_ray_residual_overlays")
            if isinstance(listed, list):
                for detail in listed:
                    if isinstance(detail, dict):
                        detail.pop("stability_status", None)
                        detail["failed_fields"] = _point_only_candidate_fields(
                            detail.get("failed_fields")
                        )
                        detail["missing_fields"] = _point_only_candidate_fields(
                            detail.get("missing_fields")
                        )
        candidate["failed_fields"] = failed
        candidate["missing_fields"] = missing
        candidate["flags"] = flags
    return candidates


def _point_only_candidate_fields(values: Any) -> list[str]:
    """Keep only full-sample point fields in one candidate diagnostic list."""
    # Legacy stability paths are replaced by the selected-family trace.
    return [
        str(value)
        for value in values or []
        if not str(value).startswith(_LEGACY_CANDIDATE_STABILITY_PREFIXES)
    ]


def _strict_primary(
    gate_evidence: dict[str, Any],
) -> tuple[str, str, int | None, int | None, int | None] | None:
    if gate_evidence["directed_ray"]["decision"] in {"stable", "exploratory"}:
        return ("directed_ray", "directed_ray", None, None, None)
    if gate_evidence["axis_or_antipodal"]["decision"] in {"stable", "exploratory"}:
        return ("axis_or_antipodal", "axis_or_antipodal", None, None, None)
    if gate_evidence["multi_mode_directional_geometry"]["decision"] == "stable":
        return (
            "multi_mode_directional_geometry",
            "multi_mode_directional_geometry",
            None,
            None,
            None,
        )
    if gate_evidence["global_directional_subspace"]["decision"] in {
        "stable",
        "exploratory",
    }:
        selected_k = int(gate_evidence["global_directional_subspace"]["selected_k"])
        label = (
            "global_2D_directional_subspace"
            if selected_k == 2
            else "global_kD_directional_subspace"
        )
        return (label, "global_directional_subspace", selected_k, selected_k, None)
    if gate_evidence["residual_lowD_k"]["decision"] in {"stable", "exploratory"}:
        selected_k = int(gate_evidence["residual_lowD_k"]["selected_k"])
        return ("residual_lowD_k", "residual_lowD_k", selected_k, None, selected_k)
    return None


def _strict_label_confidence(evidence: dict[str, Any]) -> str:
    """Map family-gate evidence to local confidence before stability qualification."""
    # Family evidence remains separate from the record-level stability state.
    if evidence.get("decision") == "exploratory":
        return "exploratory"
    return "accepted"


def _fallback_primary(
    candidates: list[dict[str, Any]],
    gate_evidence: dict[str, Any],
    flags: list[str],
    flag_details: dict[str, Any],
) -> tuple[str, str, int | None, int | None, int | None, str | None]:
    """Choose the first reportable fallback while retaining blocked candidates."""
    # Rejected selected mixtures remain in candidate_labels for audit but cannot
    # become primary through either the family or unresolved fallback branch.
    selectable_candidates = [
        candidate
        for candidate in candidates
        if candidate["family"] != "multi_mode_directional_geometry"
        or (candidate.get("details") or {}).get("reporting_acceptance")
        == "accepted"
    ]
    family_candidates = [
        candidate
        for candidate in selectable_candidates
        if candidate["family"] != "unresolved_high_dimensional_or_diffuse"
    ]
    if only_missing_evidence(gate_evidence) and not family_candidates:
        _add_flag(
            flags,
            flag_details,
            "all_gates_missing",
            "terminal",
            reason="all_gates_missing",
        )
        return (
            "geometry_metrics_unavailable",
            "unavailable",
            None,
            None,
            None,
            "all_gates_missing",
        )
    if family_candidates:
        chosen = family_candidates[0]
        primary = str(chosen["family"])
        selected_k = candidate_primary_k(primary, chosen)
        span_k = selected_k if primary.startswith("global_") else None
        residual_k = selected_k if primary == "residual_lowD_k" else None
        return (
            primary,
            "candidate",
            selected_k,
            span_k,
            residual_k,
            None,
        )
    if selectable_candidates:
        chosen = selectable_candidates[0]
        return (
            str(chosen["family"]),
            "candidate",
            None,
            None,
            None,
            None,
        )
    _add_flag(
        flags,
        flag_details,
        "no_positive_family_evidence",
        "terminal",
        reason="no_positive_family_evidence",
    )
    return (
        "undefined_geometry",
        "undefined",
        None,
        None,
        None,
        "no_positive_family_evidence",
    )


def _add_candidate_flags(
    flags: list[str],
    flag_details: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    for candidate in candidates:
        for flag in candidate.get("flags") or []:
            candidate_details = candidate.get("details") or {}
            flag_specific_details = {}
            if isinstance(candidate_details, dict):
                detail = candidate_details.get(str(flag))
                if isinstance(detail, dict):
                    flag_specific_details = dict(detail)
            details = {
                "selected_k": candidate.get("selected_k"),
                "failed_fields": candidate.get("failed_fields") or [],
                "missing_fields": candidate.get("missing_fields") or [],
            }
            details.update(flag_specific_details)
            _add_flag(
                flags,
                flag_details,
                str(flag),
                str(candidate["family"]),
                **details,
            )


def _add_directed_ray_overlay_flags(
    flags: list[str],
    flag_details: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    span_candidates = [
        candidate
        for candidate in candidates
        if candidate["family"]
        in {"global_2D_directional_subspace", "global_kD_directional_subspace"}
    ]
    residual_details = _directed_ray_residual_overlay_details(candidates)
    if span_candidates:
        span = span_candidates[0]
        _add_flag(
            flags,
            flag_details,
            "lowD_candidate_blocked",
            "directed_ray_overlay",
            reason="strict_directed_ray_priority",
            span_candidate_k=span.get("selected_k"),
            nearest_alternative=span.get("family"),
            failed_fields=span.get("failed_fields") or [],
            missing_fields=span.get("missing_fields") or [],
        )
    if residual_details:
        residual_detail = residual_details[0]
        details = {
            "reason": "residual_candidate_present",
        }
        details.update(residual_detail)
        _add_flag(
            flags,
            flag_details,
            "directed_ray_with_lowD_residual",
            "directed_ray_overlay",
            **details,
        )


def _candidate_selected_k(candidate: dict[str, Any]) -> int | None:
    """Return a normalized selected-k value from a candidate record."""
    # Candidate helpers store JSON-like values, so normalize before comparisons.
    selected_k = candidate.get("selected_k")
    return int(selected_k) if selected_k is not None else None


def _directed_ray_residual_overlay_details(
    candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return residual overlay details for coexisting residual k >= 2 evidence."""
    # Residual fallback can select k=1 while still carrying k>=2 overlay evidence.
    overlays: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("family") != "residual_lowD_k":
            continue
        details = candidate.get("details")
        if not isinstance(details, dict):
            continue
        listed = details.get("directed_ray_residual_overlays")
        if isinstance(listed, list):
            overlays.extend(
                dict(item)
                for item in listed
                if isinstance(item, dict) and int(item.get("residual_k", 0)) >= 2
            )
            continue
        selected_k = _candidate_selected_k(candidate)
        detail = details.get("directed_ray_with_lowD_residual")
        if selected_k is not None and selected_k >= 2 and isinstance(detail, dict):
            overlays.append(dict(detail))
    return sorted(overlays, key=lambda item: int(item.get("residual_k", 0)))


def _add_flag(
    flags: list[str],
    flag_details: dict[str, Any],
    flag: str | None,
    source: str,
    **details: Any,
) -> None:
    if flag is None:
        return
    flags.append(flag)
    existing = flag_details.setdefault(flag, {"sources": []})
    sources = existing.setdefault("sources", [])
    if source not in sources:
        sources.append(source)
    for key, value in details.items():
        if value not in (None, [], {}):
            existing[key] = value


def _global_flags(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    primary_label: str,
) -> list[str]:
    flags = []
    if primary_label not in TERMINAL_LABELS and long_tail(record, thresholds):
        flags.append("long_tail_spectrum")
    if ge(record.get("m_cv"), thresholds.tau_m_cv):
        flags.append("magnitude_unstable")
    if sample_size_curve_status(record) == "unstable":
        flags.append("sample_size_unstable")
    if leave_out_status(record) == "unstable":
        flags.append("leave_out_unstable")
    if _is_exploratory_low_n(record, thresholds):
        flags.append("exploratory_low_n")
    return [flag for flag in GLOBAL_FLAG_ORDER if flag in flags]


def _is_exploratory_low_n(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    """Return whether exploratory status is exactly at the low-n boundary."""
    # Upstream stability has broader exploratory protocols; this flag means n_min.
    if low_context_status(record) != "exploratory":
        return False
    low_context = record.get("low_context")
    low_context_n = (
        low_context.get("n_valid") if isinstance(low_context, dict) else None
    )
    n_valid = finite_float(low_context_n)
    if n_valid is None:
        n_valid = finite_float(record.get("n_valid"))
    return n_valid is not None and int(n_valid) == int(thresholds.n_min)


def _global_flag_details(
    record: dict[str, Any], thresholds: GeometryThresholds, flag: str
) -> dict[str, Any]:
    """Return structured detail metadata for one global visual flag."""
    # Global flags are visual overlays, but their provenance should still be auditable.
    if flag == "long_tail_spectrum":
        r_span_ent = record.get("r_span_ent")
        r_span_pr = record.get("r_span_pr")
        return {
            "field": "r_span_ent/r_span_pr",
            "r_span_ent": r_span_ent,
            "r_span_pr": r_span_pr,
            "threshold": thresholds.tau_longtail,
        }
    if flag == "magnitude_unstable":
        return {
            "field": "m_cv",
            "value": record.get("m_cv"),
            "threshold": thresholds.tau_m_cv,
        }
    if flag == "sample_size_unstable":
        return {
            "status": sample_size_curve_status(record),
            "evidence_source": "sample_size_curves",
            "blocked_fields": ["sample_size_curves"],
            "missing_fields": [],
        }
    if flag == "leave_out_unstable":
        return {
            "status": leave_out_status(record),
            "evidence_source": "leave_out_sensitivity",
            "blocked_fields": ["leave_out_sensitivity"],
            "missing_fields": [],
        }
    if flag == "exploratory_low_n":
        low_context = record.get("low_context")
        low_context_n = (
            low_context.get("n_valid") if isinstance(low_context, dict) else None
        )
        n_valid = finite_float(low_context_n)
        if n_valid is None:
            n_valid = finite_float(record.get("n_valid"))
        return {
            "status": low_context_status(record),
            "evidence_source": "low_context",
            "n_valid": n_valid,
            "threshold": thresholds.n_min,
        }
    return {}
