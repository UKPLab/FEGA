from __future__ import annotations

from fega.core.geometry_reporting.schema import GLOBAL_FLAG_MASK, GLOBAL_FLAG_ORDER

MAP_VECTOR_KEYS = (
    "r2",
    "c_ray",
    "s_span_1",
    "s_span_2",
    "s_span_3",
    "s_span_4",
    "s_span_8",
    "r_span_pr",
    "u_span_2",
    "d_span_2",
    "b_axis",
    "e_res",
    "r_ctr_pr",
    "delta_mix",
    "selected_mode_count",
    "min_mode_c_ray",
    "assignment_stability",
    "n_valid",
    "m_cv",
)
MISSINGNESS_KEYS = tuple(f"{key}_missing" for key in MAP_VECTOR_KEYS)

PRIMARY_LABELS = (
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
    "directed_ray",
    "axis_or_antipodal",
    "oneD_diffuse",
    "multi_mode_directional_geometry",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "residual_lowD_k",
    "unresolved_high_dimensional_or_diffuse",
)

SECONDARY_FLAGS = (
    "long_tail_spectrum",
    "magnitude_unstable",
    "sample_size_unstable",
    "leave_out_unstable",
    "exploratory_low_n",
    "selected_family_unstable",
    "selected_family_evidence_unavailable",
    "selected_family_not_evaluated",
    "ray_span_boundary",
    "directed_ray_ci_unstable",
    "directed_ray_ci_missing",
    "directed_ray_with_lowD_residual",
    "lowD_candidate_blocked",
    "axis_ci_unstable",
    "axis_ci_missing",
    "axis_stability_missing",
    "axis_stability_failed",
    "oneD_not_ray_not_axis",
    "b_axis_low",
    "b_axis_missing",
    "multimode_candidate_blocked",
    "mode_mass_failed",
    "mode_mass_missing",
    "mode_c_ray_failed",
    "mode_c_ray_missing",
    "assignment_stability_failed",
    "assignment_stability_missing",
    "span_selected_k",
    "span_k_unsupported",
    "span_stability_threshold_missing",
    "span_stability_missing",
    "span_stability_failed",
    "residual_selected_k",
    "residual_k_unsupported",
    "residual_stability_threshold_missing",
    "residual_stability_missing",
    "residual_stability_failed",
    "positive_highD_evidence",
    "effect_count_missing",
    "effect_count_below_min",
    "zero_filter_too_high",
    "all_gates_missing",
    "no_positive_family_evidence",
)

GEOMETRY_REPORT_LABELS = (*PRIMARY_LABELS, *SECONDARY_FLAGS)

ATLAS_LABELS = (
    "directed_ray",
    "axis_or_antipodal",
    "oneD_diffuse",
    "multi_mode_directional_geometry",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "residual_lowD_k",
    "unresolved_high_dimensional_or_diffuse",
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
)

LABEL_DISPLAY_NAMES = {
    "insufficient_effect_evidence": "Insufficient effect evidence",
    "geometry_metrics_unavailable": "Geometry metrics unavailable",
    "undefined_geometry": "Undefined geometry",
    "directed_ray": "Directed ray",
    "axis_or_antipodal": "Axis or antipodal",
    "oneD_diffuse": "1D diffuse",
    "multi_mode_directional_geometry": "Multi mode directional geometry",
    "global_2D_directional_subspace": "Global 2D directional subspace",
    "global_kD_directional_subspace": "Global kD directional subspace",
    "residual_lowD_k": "Residual low dimension",
    "unresolved_high_dimensional_or_diffuse": (
        "Unresolved high dimensional or diffuse"
    ),
    "long_tail_spectrum": "Long tail spectrum",
    "magnitude_unstable": "Magnitude unstable",
    "sample_size_unstable": "Sample size unstable",
    "leave_out_unstable": "Leave-out unstable",
    "exploratory_low_n": "Exploratory low n",
    "selected_family_unstable": "Selected-family instability observed",
    "selected_family_evidence_unavailable": "Selected-family evidence unavailable",
    "selected_family_not_evaluated": "Selected-family stability not evaluated",
    "ray_span_boundary": "Ray span boundary",
    "directed_ray_ci_unstable": "Directed ray CI unstable",
    "directed_ray_ci_missing": "Directed ray CI missing",
    "directed_ray_with_lowD_residual": "Directed ray with low-D residual",
    "lowD_candidate_blocked": "Low-D candidate blocked",
    "axis_ci_unstable": "Axis CI unstable",
    "axis_ci_missing": "Axis CI missing",
    "axis_stability_missing": "Axis stability missing",
    "axis_stability_failed": "Axis stability failed",
    "oneD_not_ray_not_axis": "1D diffuse fallback",
    "b_axis_low": "Axis balance low",
    "b_axis_missing": "Axis balance missing",
    "multimode_candidate_blocked": "Multimode candidate blocked",
    "mode_mass_failed": "Mode mass failed",
    "mode_mass_missing": "Mode mass missing",
    "mode_c_ray_failed": "Mode c_ray failed",
    "mode_c_ray_missing": "Mode c_ray missing",
    "assignment_stability_failed": "Assignment stability failed",
    "assignment_stability_missing": "Assignment stability missing",
    "span_selected_k": "Span selected k",
    "span_k_unsupported": "Span k unsupported",
    "span_stability_threshold_missing": "Span stability threshold missing",
    "span_stability_missing": "Span stability missing",
    "span_stability_failed": "Span stability failed",
    "residual_selected_k": "Residual selected k",
    "residual_k_unsupported": "Residual k unsupported",
    "residual_stability_threshold_missing": (
        "Residual stability threshold missing"
    ),
    "residual_stability_missing": "Residual stability missing",
    "residual_stability_failed": "Residual stability failed",
    "positive_highD_evidence": "Positive high-D evidence",
    "effect_count_missing": "Effect count missing",
    "effect_count_below_min": "Effect count below minimum",
    "zero_filter_too_high": "Zero-filter fraction too high",
    "all_gates_missing": "All gates missing",
    "no_positive_family_evidence": "No positive family evidence",
}

LABEL_INTERPRETATIONS = {
    label: "Classifier diagnostic flag emitted for this feature."
    for label in GEOMETRY_REPORT_LABELS
}
LABEL_INTERPRETATIONS.update(
    {
        "insufficient_effect_evidence": (
            "No reliable geometry claim is made because effect evidence is too sparse."
        ),
        "geometry_metrics_unavailable": (
            "Effect evidence exists but geometry metrics are unavailable."
        ),
        "undefined_geometry": (
            "Geometry was attempted, but no positive family evidence was found."
        ),
        "directed_ray": "The feature has one stable directed effect.",
        "axis_or_antipodal": "The feature uses one axis rather than one directed ray.",
        "oneD_diffuse": "The feature is strongly one-dimensional but not ray or axis.",
        "multi_mode_directional_geometry": (
            "The feature has several coherent directional modes."
        ),
        "global_2D_directional_subspace": (
            "The feature effects occupy one meaningfully used plane."
        ),
        "global_kD_directional_subspace": (
            "The feature is low-dimensional but needs more than two dimensions."
        ),
        "residual_lowD_k": (
            "The feature has a low-dimensional residual around the mean effect."
        ),
        "unresolved_high_dimensional_or_diffuse": (
            "Positive high-dimensional or diffuse evidence remains unresolved."
        ),
        "long_tail_spectrum": "A low-energy tail may complicate rank claims.",
        "magnitude_unstable": (
            "The feature keeps the same geometry family but has uneven strength."
        ),
        "sample_size_unstable": "The geometry may depend on sample-size curves.",
        "leave_out_unstable": "The geometry may change under leave-out checks.",
        "exploratory_low_n": "The label is a useful hypothesis, not a strong claim.",
    }
)

CLASS_PALETTE = {
    "directed_ray": "#1f77b4",
    "axis_or_antipodal": "#ff7f0e",
    "oneD_diffuse": "#bcbd22",
    "multi_mode_directional_geometry": "#2ca02c",
    "global_2D_directional_subspace": "#d62728",
    "global_kD_directional_subspace": "#9467bd",
    "residual_lowD_k": "#17becf",
    "unresolved_high_dimensional_or_diffuse": "#8c564b",
    "insufficient_effect_evidence": "#e377c2",
    "geometry_metrics_unavailable": "#7f7f7f",
    "undefined_geometry": "#aec7e8",
}

MARKER_POLICY = {
    "ordinary": "o",
    "global_flags_do_not_change_marker": True,
}
SIZE_POLICY = {
    "formula": "16 + 22 * log1p(m_median)",
    "field": "m_median",
    "fallback": 16.0,
}
OUTLINE_POLICY = {
    "emphasized_flags": [],
    "ordinary_edgecolor": "white",
    "emphasized_edgecolor": "black",
}

GLOBAL_FLAG_PATTERN_POLICY = {
    "long_tail_spectrum": {
        "visual": "dot_fill",
        "layer": 2,
    },
    "sample_size_unstable": {
        "visual": "diagonal_fill",
        "orientation": "forward",
        "layer": 3,
    },
    "leave_out_unstable": {
        "visual": "diagonal_fill",
        "orientation": "reverse",
        "layer": 3,
    },
    "exploratory_low_n": {
        "visual": "cross_line_fill",
        "layer": 4,
    },
}
GLOBAL_FLAG_OVERLAY_POLICY = {
    "atlas_color": "primary_label_only",
    "marker": MARKER_POLICY,
    "outline": OUTLINE_POLICY,
    "overlays": {
        "magnitude_unstable": {
            "visual": "centered_x",
            "color": "black",
            "layer": 5,
        },
    },
    "layer_order": [
        "base_marker",
        "long_tail_spectrum",
        "sample_size_unstable",
        "leave_out_unstable",
        "exploratory_low_n",
        "magnitude_unstable",
    ],
}
BITMASK_RING_POLICY = {
    "slots": {
        "top": {
            "flag": "long_tail_spectrum",
            "code": GLOBAL_FLAG_MASK["long_tail_spectrum"],
        },
        "upper_right": {
            "flag": "sample_size_unstable",
            "code": GLOBAL_FLAG_MASK["sample_size_unstable"],
        },
        "lower_right": {
            "flag": "leave_out_unstable",
            "code": GLOBAL_FLAG_MASK["leave_out_unstable"],
        },
        "bottom": {
            "flag": "exploratory_low_n",
            "code": GLOBAL_FLAG_MASK["exploratory_low_n"],
        },
        "center_glyph": {
            "flag": "magnitude_unstable",
            "code": GLOBAL_FLAG_MASK["magnitude_unstable"],
        },
    }
}

GLOBAL_FLAG_VISUAL_POLICY = {
    "pattern_policy": GLOBAL_FLAG_PATTERN_POLICY,
    "overlay_policy": GLOBAL_FLAG_OVERLAY_POLICY,
    "bitmask_ring": BITMASK_RING_POLICY,
    "flag_order": list(GLOBAL_FLAG_ORDER),
}
