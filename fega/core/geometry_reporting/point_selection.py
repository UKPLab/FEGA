from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .evidence import (
    all_family_anchors_false,
    axis_anchor,
    base_ok,
    base_terminal_reason,
    build_point_gate_evidence,
    long_tail,
    multimode_anchor,
    one_d_diffuse_condition,
    only_missing_evidence,
    positive_high_dimensional_or_diffuse,
    ray_anchor,
    residual_anchor,
    span_anchor,
)
from .thresholds import GeometryThresholds, get_threshold_profile

POINT_SELECTION_CONTRACT_VERSION = 2

SelectionMode = Literal["strict", "fallback", "terminal"]
MixtureAuditStatus = Literal[
    "accepted", "rejected", "unavailable", "not_applicable"
]


class InvalidVmfProvenanceError(ValueError):
    """Raised when point selection is attempted with invalid standalone vMF state."""


@dataclass(frozen=True)
class MixtureAuditState:
    """Freeze the existing complete mixture gate outcome for point-selection audit."""

    status: MixtureAuditStatus
    acceptance: str | None
    reason: str | None
    failed_gates: tuple[str, ...]
    unavailable_gates: tuple[str, ...]


@dataclass(frozen=True)
class PointSelection:
    """Immutable reporting-owned point family chosen before non-mixture stability."""

    family: str
    selected_k: int | None
    mode: SelectionMode
    point_reason: str
    mixture_audit_state: MixtureAuditState
    contract_version: int = POINT_SELECTION_CONTRACT_VERSION


def point_selection_identity(selection: PointSelection) -> dict[str, Any]:
    """Serialize the complete immutable reporting-owned point selection."""
    # Bind every mixture-audit field so downstream phases never reconstruct state.
    audit = selection.mixture_audit_state
    return {
        "family": selection.family,
        "selected_k": selection.selected_k,
        "mode": selection.mode,
        "point_reason": selection.point_reason,
        "mixture_audit_state": {
            "status": audit.status,
            "acceptance": audit.acceptance,
            "reason": audit.reason,
            "failed_gates": list(audit.failed_gates),
            "unavailable_gates": list(audit.unavailable_gates),
        },
        "contract_version": int(selection.contract_version),
    }


def resolve_point_selection(
    record: dict[str, Any],
    profile: str | GeometryThresholds = "paper",
    *,
    vmf_provenance_valid: bool,
) -> PointSelection:
    """Resolve one family from full-sample point evidence in established priority.

    Non-mixture bootstrap, angle, leave-out, and sample-size evidence is never read.
    Valid standalone vMF mixture evidence is the sole pre-lock exception. Invalid
    artifact provenance raises before any later point-supported family can be used.
    """
    # Fail closed before constructing a rejection-like mixture audit state.
    if not vmf_provenance_valid:
        raise InvalidVmfProvenanceError(
            "standalone vMF provenance is invalid; rerun the vMF phase before point resolution"
        )
    thresholds = (
        get_threshold_profile(profile) if isinstance(profile, str) else profile
    )
    if not base_ok(record, thresholds):
        return PointSelection(
            family="insufficient_effect_evidence",
            selected_k=None,
            mode="terminal",
            point_reason=base_terminal_reason(record, thresholds),
            mixture_audit_state=MixtureAuditState(
                status="not_applicable",
                acceptance="not_evaluated",
                reason="base_terminal",
                failed_gates=(),
                unavailable_gates=(),
            ),
        )
    point_gates = build_point_gate_evidence(record, thresholds)
    mixture_audit = _mixture_audit(point_gates["multi_mode_directional_geometry"])

    strict = _strict_point_selection(point_gates, mixture_audit)
    if strict is not None:
        return strict
    fallback = _point_fallback(record, thresholds, mixture_audit)
    if fallback is not None:
        return fallback
    if only_missing_evidence(point_gates):
        return PointSelection(
            family="geometry_metrics_unavailable",
            selected_k=None,
            mode="terminal",
            point_reason="all_gates_missing",
            mixture_audit_state=mixture_audit,
        )
    return PointSelection(
        family="undefined_geometry",
        selected_k=None,
        mode="terminal",
        point_reason="no_positive_family_evidence",
        mixture_audit_state=mixture_audit,
    )


def _strict_point_selection(
    point_gates: dict[str, Any], mixture_audit: MixtureAuditState
) -> PointSelection | None:
    """Return the first strict point family in the existing reporting order."""
    # Ray and axis retain priority over the standalone mixture exception.
    for gate_key, family in (
        ("directed_ray", "directed_ray"),
        ("axis_or_antipodal", "axis_or_antipodal"),
    ):
        if point_gates[gate_key]["decision"] == "stable":
            return PointSelection(
                family=family,
                selected_k=None,
                mode="strict",
                point_reason=f"strict_point_gate:{gate_key}",
                mixture_audit_state=mixture_audit,
            )
    if mixture_audit.status == "accepted":
        return PointSelection(
            family="multi_mode_directional_geometry",
            selected_k=None,
            mode="strict",
            point_reason="strict_point_gate:multi_mode_directional_geometry",
            mixture_audit_state=mixture_audit,
        )
    span = point_gates["global_directional_subspace"]
    if span["decision"] == "stable":
        selected_k = int(span["selected_k"])
        family = (
            "global_2D_directional_subspace"
            if selected_k == 2
            else "global_kD_directional_subspace"
        )
        return PointSelection(
            family=family,
            selected_k=selected_k,
            mode="strict",
            point_reason="strict_point_gate:global_directional_subspace",
            mixture_audit_state=mixture_audit,
        )
    residual = point_gates["residual_lowD_k"]
    if residual["decision"] == "stable":
        return PointSelection(
            family="residual_lowD_k",
            selected_k=int(residual["selected_k"]),
            mode="strict",
            point_reason="strict_point_gate:residual_lowD_k",
            mixture_audit_state=mixture_audit,
        )
    return None


def _point_fallback(
    record: dict[str, Any],
    thresholds: GeometryThresholds,
    mixture_audit: MixtureAuditState,
) -> PointSelection | None:
    """Return the first existing point fallback without profiling its stability."""
    # Preserve the current fallback order and smallest anchored dimensions exactly.
    if ray_anchor(record, thresholds):
        return _fallback("directed_ray", None, mixture_audit)
    if axis_anchor(record, thresholds):
        return _fallback("axis_or_antipodal", None, mixture_audit)
    if one_d_diffuse_condition(record, thresholds):
        return _fallback("oneD_diffuse", None, mixture_audit)
    if mixture_audit.status == "accepted" and multimode_anchor(record, thresholds):
        return _fallback("multi_mode_directional_geometry", None, mixture_audit)
    for k in (2, 3, 4, 8):
        if span_anchor(record, thresholds, k):
            family = (
                "global_2D_directional_subspace"
                if k == 2
                else "global_kD_directional_subspace"
            )
            return _fallback(family, k, mixture_audit)
    for k in (1, 2, 3, 4):
        if residual_anchor(record, thresholds, k):
            return _fallback("residual_lowD_k", k, mixture_audit)
    if positive_high_dimensional_or_diffuse(record, thresholds) or (
        long_tail(record, thresholds)
        and all_family_anchors_false(record, thresholds)
    ):
        return _fallback(
            "unresolved_high_dimensional_or_diffuse", None, mixture_audit
        )
    return None


def _fallback(
    family: str, selected_k: int | None, mixture_audit: MixtureAuditState
) -> PointSelection:
    """Construct one immutable fallback result with deliberate non-evaluation reason."""
    # The reason is scheduling metadata and does not describe stable evidence.
    return PointSelection(
        family=family,
        selected_k=selected_k,
        mode="fallback",
        point_reason=f"point_fallback:{family}",
        mixture_audit_state=mixture_audit,
    )


def _mixture_audit(evidence: dict[str, Any]) -> MixtureAuditState:
    """Normalize the unchanged mixture gate into a closed immutable audit state."""
    # Unavailable required evidence stays distinct from an observed failed gate.
    failed = tuple(str(item) for item in evidence.get("failed_gates") or ())
    unavailable = tuple(
        str(item) for item in evidence.get("unavailable_gates") or ()
    )
    if evidence.get("acceptance") == "accepted":
        status: MixtureAuditStatus = "accepted"
    elif unavailable:
        status = "unavailable"
    elif evidence.get("evaluated"):
        status = "rejected"
    else:
        status = "not_applicable"
    return MixtureAuditState(
        status=status,
        acceptance=evidence.get("acceptance"),
        reason=evidence.get("reason"),
        failed_gates=failed,
        unavailable_gates=unavailable,
    )
