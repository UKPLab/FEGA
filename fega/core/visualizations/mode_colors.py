"""Cached vMF assignments and deterministic display colors for mixture cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

MODE_COLORS = ("#E83E8C", "#E67E22", "#2A9D8F", "#6C5CE7")


def load_vmf_features(path: Path) -> dict[int, dict[str, Any]]:
    """Index cached vMF feature records without fitting or recomputing assignments."""
    # Keep older runs usable for non-mixture cards when no standalone vMF artifact exists.
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"vMF scores missing `features`: {path}")
    return {
        int(feature["feature_id"]): feature
        for feature in features
        if isinstance(feature, dict) and isinstance(feature.get("feature_id"), int)
    }


def cached_hard_assignments(
    feature: dict[str, Any] | None,
    *,
    expected_rows: int,
) -> tuple[list[int] | None, str | None]:
    """Return fitted hard assignments when they align with the displayed row order."""
    # Require the compact selected fit that generated the published mixture evidence.
    if not isinstance(feature, dict) or feature.get("fit_status") != "fitted":
        return None, "vmf_selected_fit_unavailable"
    selected_fit = feature.get("selected_fit")
    if not isinstance(selected_fit, dict):
        return None, "vmf_selected_fit_unavailable"
    assignments = selected_fit.get("hard_assignments")
    mode_counts = selected_fit.get("hard_mode_counts")
    if not isinstance(assignments, list) or not isinstance(mode_counts, list):
        return None, "vmf_hard_assignments_unavailable"

    # Protect the scientific row-to-color correspondence at the cached-artifact boundary.
    mode_count = len(mode_counts)
    valid = (
        len(assignments) == expected_rows
        and mode_count > 1
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < mode_count
            for value in assignments
        )
    )
    if not valid:
        return None, "vmf_hard_assignments_misaligned"
    return [int(value) for value in assignments], None


def multi_mode_point_colors(
    coordinates: torch.Tensor,
    assignments: list[int],
) -> tuple[list[str], dict[str, Any]]:
    """Color cached modes after a deterministic display-only component ordering."""
    # Order raw vMF labels by their PC1-PC2 centroids because component IDs are arbitrary.
    labels = torch.tensor(assignments, dtype=torch.long)
    raw_modes = sorted(set(assignments))
    if len(raw_modes) > len(MODE_COLORS):
        raise ValueError("vMF mode count exceeds the visualization color palette")
    display_order = sorted(
        raw_modes,
        key=lambda mode: (
            float(coordinates[labels == mode, 0].mean().item()),
            float(coordinates[labels == mode, 1].mean().item()),
            mode,
        ),
    )
    color_by_mode = {
        raw_mode: MODE_COLORS[display_index]
        for display_index, raw_mode in enumerate(display_order)
    }

    # Preserve raw membership while recording the arbitrary label-to-color permutation.
    point_colors = [color_by_mode[mode] for mode in assignments]
    metadata = {
        "source": "vmf/pre_softcap_logits/vmf_scores.json:selected_fit.hard_assignments",
        "display_order_raw_modes": display_order,
        "raw_mode_to_display_color": {
            str(mode): color_by_mode[mode] for mode in raw_modes
        },
        "hard_mode_counts": {str(mode): assignments.count(mode) for mode in raw_modes},
    }
    return point_colors, metadata
