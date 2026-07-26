from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from fega.core.geometry_reporting.point_selection import (
    InvalidVmfProvenanceError,
    resolve_point_selection,
)
from fega.core.geometry_reporting.thresholds import get_threshold_profile
from fega.core.stability.protocols import signed_gate_margins
from fega.core.stability.schedule import (
    DELIBERATE_NON_EVALUATION_REASON,
    NO_STABILITY_EVIDENCE_REQUEST,
    SELECTED_FAMILY_EVIDENCE_REQUESTS,
    resolve_requested_margin_keys,
)


def _point_record(**overrides: Any) -> dict[str, Any]:
    """Return complete point evidence with no strict or fallback anchor by default."""
    # Finite failed values make undefined versus unavailable tests intentional.
    record: dict[str, Any] = {
        "n_valid": 32,
        "zero_filter_frac": 0.0,
        "c_ray": 0.1,
        "s_span_1": 0.1,
        "b_axis": 0.0,
        "selected_mode_count": 1,
        "delta_mix": 0.0,
        "mode_mass_min": 1.0,
        "min_mode_c_ray": 0.1,
        "mode_kappa_min": 1.0,
        "assignment_stability": {
            "status": "not_applicable",
            "value": None,
        },
        "r_span_pr": 1.0,
        "r_span_ent": 1.0,
        "e_res": 0.05,
        "r_ctr_pr": 1.0,
        "eps": 1.0e-12,
    }
    for k in (2, 3, 4, 8):
        record[f"s_span_{k}"] = 0.1
        record[f"u_span_{k}"] = 0.0
        record[f"d_span_{k}"] = 1.0
    for k in (1, 2, 3, 4):
        record[f"s_res_{k}"] = 0.1
    record.update(overrides)
    return record


def _resolve(**overrides: Any):
    """Resolve one valid-provenance point fixture through the public API."""
    # All fixtures except the explicit invalid case model a validated vMF artifact.
    return resolve_point_selection(
        _point_record(**overrides), vmf_provenance_valid=True
    )


def test_point_priority_and_strict_axis_boundary() -> None:
    """Freeze ray priority and the strict axis c_ray inequality at equality."""
    # A ray at equality outranks every simultaneously passing later family.
    ray = _resolve(
        c_ray=0.8,
        s_span_1=0.8,
        b_axis=0.2,
        selected_mode_count=2,
        delta_mix=0.1,
        mode_mass_min=0.1,
        min_mode_c_ray=0.7,
        mode_kappa_min=1.0,
        assignment_stability={"status": "available", "value": 0.8},
        s_span_2=0.9,
        r_span_pr=1.6,
        u_span_2=0.08,
        d_span_2=0.6,
    )
    axis = _resolve(c_ray=0.799, s_span_1=0.8, b_axis=0.15)

    assert (ray.family, ray.mode, ray.selected_k) == (
        "directed_ray",
        "strict",
        None,
    )
    assert (axis.family, axis.mode, axis.selected_k) == (
        "axis_or_antipodal",
        "strict",
        None,
    )


def test_accepted_rejected_and_unavailable_mixture_paths() -> None:
    """Keep complete mixture acceptance pre-lock and continue rejected states."""
    # Exact-threshold acceptance locks mixture before later point-supported families.
    accepted = _resolve(
        selected_mode_count=2,
        delta_mix=0.1,
        mode_mass_min=0.1,
        min_mode_c_ray=0.7,
        mode_kappa_min=0.0,
        assignment_stability={"status": "available", "value": 0.8},
        s_span_2=0.9,
        r_span_pr=1.6,
        u_span_2=0.08,
        d_span_2=0.6,
    )
    rejected = _resolve(
        selected_mode_count=2,
        delta_mix=0.1,
        mode_mass_min=0.09,
        min_mode_c_ray=0.7,
        mode_kappa_min=1.0,
        assignment_stability={"status": "available", "value": 0.8},
        s_span_2=0.9,
        r_span_pr=1.6,
        u_span_2=0.08,
        d_span_2=0.6,
    )
    unavailable = _resolve(
        selected_mode_count=2,
        delta_mix=0.1,
        mode_mass_min=0.1,
        min_mode_c_ray=0.7,
        mode_kappa_min=1.0,
        assignment_stability={"status": "unavailable", "value": None},
        e_res=0.1,
        s_res_2=0.8,
        r_ctr_pr=1.5,
    )

    assert accepted.family == "multi_mode_directional_geometry"
    assert accepted.mixture_audit_state.status == "accepted"
    assert rejected.family == "global_2D_directional_subspace"
    assert rejected.mixture_audit_state.status == "rejected"
    assert rejected.mixture_audit_state.failed_gates == ("fitted_mass",)
    assert unavailable.family == "residual_lowD_k"
    assert unavailable.mixture_audit_state.status == "unavailable"
    assert unavailable.mixture_audit_state.unavailable_gates == (
        "assignment_stability",
    )


@pytest.mark.parametrize(
    ("selected_k", "requirements"),
    [
        (2, {"r_span_pr": 1.6, "u_span_2": 0.08, "d_span_2": 0.6}),
        (3, {"r_span_pr": 2.3, "u_span_3": 0.05, "d_span_3": 0.6}),
        (4, {"r_span_pr": 3.0, "u_span_4": 0.03, "d_span_4": 0.6}),
        (8, {"r_span_pr": 5.0, "u_span_8": 0.01, "d_span_8": 0.6}),
    ],
)
def test_every_span_k_uses_smallest_passing_point_dimension(
    selected_k: int, requirements: dict[str, float]
) -> None:
    """Freeze every supported span dimension and its inclusive comparisons."""
    # Earlier anchors remain false so each parameter isolates one declared k.
    selection = _resolve(**{f"s_span_{selected_k}": 0.9}, **requirements)

    assert selection.selected_k == selected_k
    assert selection.family == (
        "global_2D_directional_subspace"
        if selected_k == 2
        else "global_kD_directional_subspace"
    )
    assert selection.mode == "strict"


@pytest.mark.parametrize(
    ("selected_k", "r_ctr_pr"), [(2, 1.5), (3, 2.2), (4, 2.9)]
)
def test_every_strict_residual_k_uses_smallest_passing_point_dimension(
    selected_k: int, r_ctr_pr: float
) -> None:
    """Freeze every strict residual dimension while excluding fallback-only k=1."""
    # Earlier strict residual anchors remain false for the isolated selected k.
    selection = _resolve(
        e_res=0.1, **{f"s_res_{selected_k}": 0.8}, r_ctr_pr=r_ctr_pr
    )

    assert (selection.family, selection.selected_k, selection.mode) == (
        "residual_lowD_k",
        selected_k,
        "strict",
    )


def test_minimal_k_changes_and_residual_k1_stays_fallback_only() -> None:
    """Choose a newly passing smaller k and keep residual k=1 descriptive only."""
    # k=4 first passes alone; enabling the complete k=2 gate changes selection to 2.
    span_k4 = _point_record(
        s_span_4=0.9, r_span_pr=3.0, u_span_4=0.03, d_span_4=0.6
    )
    larger = resolve_point_selection(span_k4, vmf_provenance_valid=True)
    span_k4.update(s_span_2=0.9, u_span_2=0.08, d_span_2=0.6)
    smaller = resolve_point_selection(span_k4, vmf_provenance_valid=True)
    residual_k1 = _resolve(e_res=0.1, s_res_1=0.8)

    assert (larger.selected_k, smaller.selected_k) == (4, 2)
    assert (residual_k1.family, residual_k1.selected_k, residual_k1.mode) == (
        "residual_lowD_k",
        1,
        "fallback",
    )


def test_fallback_unresolved_and_terminal_states() -> None:
    """Freeze oneD, unresolved, unavailable, undefined, and base-terminal outcomes."""
    # Each record targets one existing fallback or terminal condition directly.
    one_d = _resolve(c_ray=0.2, s_span_1=0.8, b_axis=0.149)
    unresolved = _resolve()
    unavailable = _resolve(
        c_ray=None,
        s_span_1=None,
        b_axis=None,
        selected_mode_count=None,
        delta_mix=None,
        mode_mass_min=None,
        min_mode_c_ray=None,
        mode_kappa_min=None,
        assignment_stability={},
        r_span_pr=None,
        e_res=None,
        r_ctr_pr=None,
        **{f"s_span_{k}": None for k in (2, 3, 4, 8)},
        **{f"u_span_{k}": None for k in (2, 3, 4, 8)},
        **{f"d_span_{k}": None for k in (2, 3, 4, 8)},
        **{f"s_res_{k}": None for k in (1, 2, 3, 4)},
    )
    undefined = _resolve(e_res=None)
    below_min = _resolve(n_valid=7)
    zero_filtered = _resolve(zero_filter_frac=0.3000001)

    assert (one_d.family, one_d.mode) == ("oneD_diffuse", "fallback")
    assert (unresolved.family, unresolved.mode) == (
        "unresolved_high_dimensional_or_diffuse",
        "fallback",
    )
    assert (unavailable.family, unavailable.point_reason) == (
        "geometry_metrics_unavailable",
        "all_gates_missing",
    )
    assert (undefined.family, undefined.point_reason) == (
        "undefined_geometry",
        "no_positive_family_evidence",
    )
    assert below_min.point_reason == "effect_count_below_min"
    assert zero_filtered.point_reason == "zero_filter_too_high"


def test_invalid_vmf_provenance_blocks_before_later_family_resolution() -> None:
    """Treat invalid standalone provenance as a blocker, never mixture rejection."""
    # A passing span is intentionally present to prove the resolver cannot continue.
    record = _point_record(
        s_span_2=0.9,
        r_span_pr=1.6,
        u_span_2=0.08,
        d_span_2=0.6,
    )

    with pytest.raises(InvalidVmfProvenanceError, match="rerun the vMF phase"):
        resolve_point_selection(record, vmf_provenance_valid=False)


def test_selected_family_evidence_map_is_exact_and_deeply_immutable() -> None:
    """Freeze the approved evidence breadth without generating WP2 schedules."""
    # Axis separates reported k from its required raw k=1 angle.
    axis = SELECTED_FAMILY_EVIDENCE_REQUESTS["axis_or_antipodal"]
    span = SELECTED_FAMILY_EVIDENCE_REQUESTS[
        "global_kD_directional_subspace"
    ]
    residual = SELECTED_FAMILY_EVIDENCE_REQUESTS["residual_lowD_k"]

    assert axis.scalar_bootstrap_metrics == ("c_ray",)
    assert (axis.angle_source, axis.angle_k) == ("raw", 1)
    assert span.strict_k_values == (2, 3, 4, 8)
    assert span.stop_point_gates_at_selected_k is True
    assert residual.strict_k_values == (2, 3, 4)
    assert "fallback_only_residual_k1" in residual.explicitly_omitted
    assert NO_STABILITY_EVIDENCE_REQUEST.no_stability_protocol is True
    assert DELIBERATE_NON_EVALUATION_REASON == (
        "not_evaluated_point_fallback_or_terminal"
    )
    with pytest.raises(TypeError):
        SELECTED_FAMILY_EVIDENCE_REQUESTS["new_family"] = axis  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        axis.angle_k = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("family", "selected_k"),
    [("global_kD_directional_subspace", 3), ("residual_lowD_k", 3)],
)
def test_selected_k_margin_templates_resolve_to_signed_margin_keys(
    family: str, selected_k: int
) -> None:
    """Require every selected-k request key to name an emitted signed margin."""
    request = SELECTED_FAMILY_EVIDENCE_REQUESTS[family]
    requested_keys = resolve_requested_margin_keys(request, selected_k=selected_k)
    available_keys = signed_gate_margins(
        {}, get_threshold_profile("paper"), k_values=[selected_k]
    )

    assert set(requested_keys) <= set(available_keys)
