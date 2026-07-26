from __future__ import annotations

# Descriptive version for the persisted geometry-label schema.
LABEL_VERSION = "fega_geometry_labels_v3"

TERMINAL_LABELS = {
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
}

PRIMARY_LABELS = {
    *TERMINAL_LABELS,
    "directed_ray",
    "axis_or_antipodal",
    "oneD_diffuse",
    "multi_mode_directional_geometry",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "residual_lowD_k",
    "unresolved_high_dimensional_or_diffuse",
}

GLOBAL_FLAG_ORDER = (
    "long_tail_spectrum",
    "magnitude_unstable",
    "sample_size_unstable",
    "leave_out_unstable",
    "exploratory_low_n",
)

GLOBAL_FLAG_MASK = {
    "long_tail_spectrum": "LT",
    "magnitude_unstable": "MAG",
    "sample_size_unstable": "SS",
    "leave_out_unstable": "LO",
    "exploratory_low_n": "LOWN",
}

FALLBACK_PRIORITY = {
    "directed_ray": 20,
    "axis_or_antipodal": 30,
    "oneD_diffuse": 40,
    "multi_mode_directional_geometry": 50,
    "global_2D_directional_subspace": 60,
    "global_kD_directional_subspace": 70,
    "residual_lowD_k": 80,
    "unresolved_high_dimensional_or_diffuse": 90,
}
