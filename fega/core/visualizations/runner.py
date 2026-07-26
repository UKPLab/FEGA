"""Cached candidate discovery and per-feature visualization generation."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import torch

from fega.core.compute_effect.artifacts import validate_manifest_summary_consistency
from fega.core.geometry_metrics.artifacts import (
    GeometryMetricsInputs,
    iter_feature_blocks,
)
from fega.core.geometry_reporting.artifacts import write_json_atomic
from fega.core.geometry_reporting.map.schema import (
    ATLAS_LABELS,
    CLASS_PALETTE,
    LABEL_DISPLAY_NAMES,
)
from fega.core.visualizations.mode_colors import (
    cached_hard_assignments,
    load_vmf_features,
    multi_mode_point_colors,
)
from fega.core.visualizations.projection import project_directions, surface_coordinates
from fega.core.visualizations.render import (
    AXIS_NEGATIVE_COLOR,
    AXIS_POSITIVE_COLOR,
    AXIS_ZERO_COLOR,
    render_card,
    render_projection_2d,
    render_sphere,
    render_subspace_plane,
)
from fega.core.visualizations.residual_render import (
    RESIDUAL_NEGATIVE_COLOR,
    RESIDUAL_NEUTRAL_COLOR,
    RESIDUAL_POSITIVE_COLOR,
    render_residual_card,
    render_residual_view,
    residual_display_kind,
    residual_point_colors,
)

_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")


def run_visualizations(
    run_dir: Path,
    *,
    top_n: int = 5,
    palette_path: Path | None = None,
    dpi: int = 300,
) -> Path:
    """Render top cached candidates for every reported atlas family.

    The command reads completed FEGA artifacts without loading the model and
    replaces only its own ``visualizations/candidates`` tree and candidate index.
    """
    # Resolve command inputs and load the three scientific artifacts used here.
    run_dir = Path(run_dir).expanduser().resolve()
    if top_n <= 0 or dpi <= 0:
        raise ValueError("top_n and dpi must be positive")
    records_path = run_dir / "geometry_reporting" / "geometry_feature_records.json"
    manifest_path = (
        run_dir / "compute_effect" / "final_resid" / "effect_tensors_manifest.json"
    )
    summary_path = run_dir / "compute_effect" / "final_resid" / "effect_summary.json"
    gram_path = run_dir / "data_prep" / "gram_cache" / "gram.pt"
    vmf_path = run_dir / "vmf" / "pre_softcap_logits" / "vmf_scores.json"
    records_payload = _load_json(records_path, "geometry feature records")
    manifest = _load_json(manifest_path, "final-residual tensor manifest")
    summary = _load_json(summary_path, "final-residual effect summary")
    validate_manifest_summary_consistency(manifest, summary)
    records = records_payload.get("features")
    if not isinstance(records, list):
        raise ValueError(f"geometry feature records missing `features`: {records_path}")
    palette = _load_palette(palette_path)
    candidates = discover_candidates(records, top_n=top_n)
    vmf_features = load_vmf_features(vmf_path)

    # Replace only generator-owned outputs before materializing the selected candidates.
    output_dir = run_dir / "visualizations"
    candidates_dir = output_dir / "candidates"
    index_path = output_dir / "candidates.json"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    if index_path.exists():
        index_path.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Load only selected feature blocks while keeping one source shard live at a time.
    selected_ids = {
        int(candidate["feature_id"])
        for family_candidates in candidates.values()
        for candidate in family_candidates
    }
    per_feature = summary.get("per_feature", {})
    if not isinstance(per_feature, dict):
        raise ValueError(f"effect summary missing `per_feature`: {summary_path}")
    known_ids = {int(feature_id) for feature_id in per_feature}
    inputs = GeometryMetricsInputs(
        effect_space="final_resid",
        artifact_dir=summary_path.parent,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
    )
    blocks = {
        block.feature_id: block
        for block in iter_feature_blocks(
            inputs, exclude_feature_ids=known_ids.difference(selected_ids)
        )
    }
    gram = torch.load(gram_path, map_location="cpu", weights_only=True)
    if not isinstance(gram, torch.Tensor):
        raise ValueError(f"Gram artifact must contain a tensor: {gram_path}")

    # Render each family independently and attach result paths to the candidate index.
    indexed_families: dict[str, list[dict[str, Any]]] = {}
    for family in ATLAS_LABELS:
        indexed_families[family] = []
        for candidate in candidates[family]:
            feature_id = int(candidate["feature_id"])
            rank = int(candidate["rank"])
            candidate_dir = candidates_dir / family / f"rank_{rank:02d}_f{feature_id}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            metrics = _candidate_metrics(candidate, color=palette[family])
            block = blocks.get(feature_id)
            if block is None or block.rows is None or int(block.rows.shape[0]) == 0:
                reason = (
                    "feature_missing_from_effect_summary"
                    if block is None
                    else block.skipped_reason or "no_retained_directions"
                )
                metrics.update(
                    {"visualization_status": "unavailable", "reason": reason}
                )
            else:
                mode_assignments: list[int] | None = None
                mode_reason: str | None = None
                if family == "multi_mode_directional_geometry":
                    mode_assignments, mode_reason = cached_hard_assignments(
                        vmf_features.get(feature_id),
                        expected_rows=int(block.rows.shape[0]),
                    )
                if mode_reason is not None:
                    metrics.update(
                        {"visualization_status": "unavailable", "reason": mode_reason}
                    )
                else:
                    metrics.update(
                        _render_candidate(
                            candidate_dir,
                            candidate,
                            directions=block.rows,
                            gram=gram,
                            color=palette[family],
                            dpi=dpi,
                            run_dir=run_dir,
                            mode_assignments=mode_assignments,
                        )
                    )
            metrics_path = candidate_dir / "metrics.json"
            write_json_atomic(metrics_path, metrics)
            indexed = {
                key: value for key, value in candidate.items() if key != "record"
            }
            indexed.update(
                {
                    "color": palette[family],
                    "visualization_status": metrics["visualization_status"],
                    "reason": metrics.get("reason"),
                    "metrics_path": _relative(metrics_path, run_dir),
                }
            )
            indexed_families[family].append(indexed)

    # Publish one portable index after all per-candidate outputs are complete.
    index_payload = {
        "top_n": top_n,
        "dpi": dpi,
        "palette": palette,
        "source_paths": {
            "geometry_feature_records": _relative(records_path, run_dir),
            "effect_summary": _relative(summary_path, run_dir),
            "effect_tensors_manifest": _relative(manifest_path, run_dir),
            "gram": _relative(gram_path, run_dir),
            "vmf_scores": (_relative(vmf_path, run_dir) if vmf_path.exists() else None),
        },
        "families": indexed_families,
    }
    write_json_atomic(index_path, index_payload)
    return index_path


def discover_candidates(
    records: list[dict[str, Any]], *, top_n: int
) -> dict[str, list[dict[str, Any]]]:
    """Rank each atlas family by valid contexts, effect magnitude, then feature ID."""
    # Group primary labels without filtering on evidence so terminal states stay visible.
    grouped = {family: [] for family in ATLAS_LABELS}
    for record in records:
        if not isinstance(record, dict):
            continue
        family = record.get("primary_label")
        if family in grouped:
            grouped[family].append(record)

    # Apply the declared deterministic order and retain evidence beside every candidate.
    output: dict[str, list[dict[str, Any]]] = {}
    for family, family_records in grouped.items():
        sort_key = (
            _unresolved_candidate_sort_key
            if family == "unresolved_high_dimensional_or_diffuse"
            else _candidate_sort_key
        )
        ordered = sorted(family_records, key=sort_key)[:top_n]
        output[family] = [
            {
                "rank": rank,
                "feature_id": int(record["feature_id"]),
                "primary_label": family,
                "m_median": _finite_or_none(record.get("m_median")),
                "n_valid": record.get("n_valid"),
                "evidence_status": record.get("evidence_status"),
                "label_confidence": record.get("label_confidence"),
                "secondary_flags": record.get("secondary_flags", []),
                "terminal_reason": record.get("terminal_reason"),
                "gate_evidence": record.get("gate_evidence", {}),
                "record": record,
            }
            for rank, record in enumerate(ordered, start=1)
        ]
    return output


def _render_candidate(
    candidate_dir: Path,
    candidate: dict[str, Any],
    *,
    directions: torch.Tensor,
    gram: torch.Tensor,
    color: str,
    dpi: int,
    run_dir: Path,
    mode_assignments: list[int] | None,
) -> dict[str, Any]:
    """Project and render one candidate with faithful and display-only views."""
    # Compute the faithful uncentered coordinates used by both sphere views.
    sphere_projection = project_directions(
        directions, gram, dimensions=3, centered=False
    )
    family = str(candidate["primary_label"])
    record = candidate["record"]

    # Canonicalize only the arbitrary sign of an unsigned axis for consistent display.
    sphere_coordinates = sphere_projection.coordinates
    axis_display_flipped = False
    if family == "axis_or_antipodal":
        sphere_coordinates, axis_display_flipped = _orient_axis_for_display(
            sphere_coordinates
        )
    surface, surface_keep = surface_coordinates(sphere_coordinates)

    # Use enough centered coordinates to expose the selected residual dimension.
    residual_view = family == "residual_lowD_k"
    residual_k = int(record["selected_k"]) if residual_view else None
    if residual_view:
        plane_projection = project_directions(
            directions, gram, dimensions=4, centered=True
        )
        plane_coordinates = plane_projection.coordinates
        plane_title = f"Centered residual k={residual_k} view"
    else:
        plane_projection = sphere_projection
        plane_coordinates = sphere_coordinates[:, :2]
        plane_title = (
            "Best 2D subspace view"
            if family == "global_2D_directional_subspace"
            else (
                "Centered 2D variation"
                if family == "unresolved_high_dimensional_or_diffuse"
                else "2D class view"
            )
        )
    display_plane_coordinates = plane_coordinates
    plane_source_centroid: torch.Tensor | None = None
    if family == "directed_ray":
        display_plane_coordinates, plane_source_centroid = (
            _directed_ray_display_coordinates(plane_coordinates)
        )
    elif family == "unresolved_high_dimensional_or_diffuse":
        plane_source_centroid = plane_coordinates.mean(dim=0)
        display_plane_coordinates = plane_coordinates - plane_source_centroid
    point_colors = (
        _axis_point_colors(sphere_coordinates[:, 0])
        if family == "axis_or_antipodal"
        else None
    )
    residual_color_scale: float | None = None
    if residual_view:
        point_colors, residual_color_scale = residual_point_colors(
            plane_coordinates[:, 0].tolist()
        )
    mode_color_metadata: dict[str, Any] | None = None
    if family == "multi_mode_directional_geometry":
        point_colors, mode_color_metadata = multi_mode_point_colors(
            sphere_coordinates,
            mode_assignments or [],
        )
    surface_point_colors = (
        [
            point_color
            for point_color, keep in zip(
                point_colors, surface_keep.tolist(), strict=True
            )
            if keep
        ]
        if point_colors is not None
        else None
    )
    display_name = LABEL_DISPLAY_NAMES[family]
    title = f"{display_name} — feature {candidate['feature_id']}"
    plane_view_padding = 2.4 if family == "directed_ray" else 1.35
    top_down_mode_view = family == "multi_mode_directional_geometry"
    fitted_line_view = family == "oneD_diffuse"
    unresolved_view = family == "unresolved_high_dimensional_or_diffuse"
    neutral_axis_view = (
        top_down_mode_view or fitted_line_view or residual_view or unresolved_view
    )
    sphere_mean_vector = sphere_coordinates.mean(dim=0) if residual_view else None
    paths = {
        "sphere_ball": candidate_dir / "sphere_ball.png",
        "sphere_surface": candidate_dir / "sphere_surface.png",
        "projection_2d": candidate_dir / "projection_2d.png",
        "card": candidate_dir / "card.png",
    }

    # Write separate figures and the compact card from the same cached coordinates.
    render_sphere(
        paths["sphere_ball"],
        sphere_coordinates.numpy(),
        color=color,
        dpi=dpi,
        point_colors=point_colors,
        mean_vector=(
            sphere_mean_vector.numpy() if sphere_mean_vector is not None else None
        ),
    )
    render_sphere(
        paths["sphere_surface"],
        surface.numpy(),
        color=color,
        dpi=dpi,
        point_colors=surface_point_colors,
    )
    subspace_plane = family == "global_2D_directional_subspace"
    if residual_view and residual_k is not None:
        render_residual_view(
            paths["projection_2d"],
            plane_coordinates.numpy(),
            selected_k=residual_k,
            color=color,
            dpi=dpi,
            point_colors=point_colors or [color] * len(plane_coordinates),
        )
    elif subspace_plane:
        render_subspace_plane(
            paths["projection_2d"],
            sphere_coordinates.numpy(),
            color=color,
            dpi=dpi,
            point_colors=point_colors,
        )
    else:
        render_projection_2d(
            paths["projection_2d"],
            display_plane_coordinates.numpy(),
            color=color,
            dpi=dpi,
            point_colors=point_colors,
            axis_guide=family == "axis_or_antipodal",
            view_padding=plane_view_padding,
            primary_axis_guide=not neutral_axis_view,
            view_limit=1.35 if top_down_mode_view else None,
            mode_assignments=mode_assignments if top_down_mode_view else None,
            fitted_line=fitted_line_view,
        )
    if residual_view and residual_k is not None and sphere_mean_vector is not None:
        render_residual_card(
            paths["card"],
            sphere_coordinates.numpy(),
            plane_coordinates.numpy(),
            selected_k=residual_k,
            color=color,
            title=title,
            plane_title=plane_title,
            footer=_metric_footer(family, record),
            dpi=dpi,
            point_colors=point_colors or [color] * len(plane_coordinates),
            sphere_mean_vector=sphere_mean_vector.numpy(),
        )
    else:
        render_card(
            paths["card"],
            sphere_coordinates.numpy(),
            (
                sphere_coordinates.numpy()
                if subspace_plane
                else display_plane_coordinates.numpy()
            ),
            color=color,
            title=title,
            plane_title=plane_title,
            footer=_metric_footer(family, record),
            dpi=dpi,
            point_colors=point_colors,
            axis_guide=family == "axis_or_antipodal",
            plane_view_padding=plane_view_padding,
            subspace_plane=subspace_plane,
            plane_primary_axis_guide=not neutral_axis_view,
            plane_view_limit=1.35 if top_down_mode_view else None,
            plane_mode_assignments=mode_assignments if top_down_mode_view else None,
            plane_fitted_line=fitted_line_view,
        )
    return {
        "visualization_status": "available",
        "reason": None,
        "projection": {
            "kernel": "D G D^T",
            "sphere_kind": "uncentered_logit_equivalent",
            "projection_2d_kind": (
                "centered_residual" if residual_view else "uncentered_logit_equivalent"
            ),
            "sphere_top_eigenvalues": _top_values(sphere_projection.eigenvalues),
            "sphere_explained_ratios": _top_values(
                sphere_projection.explained_ratios, count=3
            ),
            "sphere_numerical_rank": sphere_projection.numerical_rank,
            "projection_2d_top_eigenvalues": _top_values(plane_projection.eigenvalues),
            "projection_2d_explained_ratios": _top_values(
                plane_projection.explained_ratios,
                count=4 if residual_view else 2,
            ),
            "projection_2d_numerical_rank": plane_projection.numerical_rank,
            "projection_2d_display_transform": (
                "centroid_translation_then_orthogonal_rotation"
                if family == "directed_ray"
                else (
                    "centroid_translation"
                    if family == "unresolved_high_dimensional_or_diffuse"
                    else "none"
                )
            ),
            "projection_2d_display_kind": (
                residual_display_kind(residual_k)
                if residual_view
                else (
                    "leading_3d_with_pc1_pc2_plane_and_orthogonal_feet"
                    if subspace_plane
                    else (
                        "scatter_2d_with_orthogonal_fitted_line"
                        if fitted_line_view
                        else (
                            "scatter_2d_neutral_axes"
                            if unresolved_view
                            else "scatter_2d"
                        )
                    )
                )
            ),
            "projection_2d_source_centroid": (
                [float(value) for value in plane_source_centroid.tolist()]
                if plane_source_centroid is not None
                else None
            ),
            "surface_zero_or_omitted_count": int((~surface_keep).sum().item()),
            "surface_note": "Nonzero projected rows are renormalized for display only.",
            "axis_display_sign_flipped": (
                axis_display_flipped if family == "axis_or_antipodal" else None
            ),
            "mode_coloring": mode_color_metadata,
            "residual_selected_k": residual_k,
            "residual_point_coloring": (
                {
                    "kind": "continuous_centered_residual_pc1",
                    "scale_max_abs": residual_color_scale,
                    "negative_color": RESIDUAL_NEGATIVE_COLOR,
                    "zero_color": RESIDUAL_NEUTRAL_COLOR,
                    "positive_color": RESIDUAL_POSITIVE_COLOR,
                    "note": "Continuous display coordinate, not a cluster assignment.",
                }
                if residual_view
                else None
            ),
            "sphere_mean_arrow": (
                "projected_sample_mean" if sphere_mean_vector is not None else None
            ),
        },
        "image_paths": {name: _relative(path, run_dir) for name, path in paths.items()},
    }


def _orient_axis_for_display(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Place the more populated side of an unsigned first axis on the left."""
    # Apply at most one global PC1 sign flip, which preserves every distance and angle.
    displayed = coordinates.clone()
    first = displayed[:, 0]
    positive_count = int((first > 0.0).sum().item())
    negative_count = int((first < 0.0).sum().item())
    flipped = positive_count > negative_count
    if flipped:
        displayed[:, 0] = -first
    return displayed, flipped


def _directed_ray_display_coordinates(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre and rigidly orient a directed-ray detail view along its local PC1."""
    # Translate to the projected centroid, then rotate without changing separations.
    centroid = coordinates.mean(dim=0)
    centered = coordinates - centroid
    _, _, right_vectors = torch.linalg.svd(centered, full_matrices=False)
    displayed = centered @ right_vectors.T

    # Fix the two arbitrary display-axis signs for deterministic regenerated figures.
    for column_index in range(displayed.shape[1]):
        column = displayed[:, column_index]
        anchor_index = int(torch.argmax(torch.abs(column)).item())
        if float(column[anchor_index].item()) < 0.0:
            displayed[:, column_index] = -column
    return displayed, centroid


def _axis_point_colors(first_coordinates: torch.Tensor) -> list[str]:
    """Map the two signed sides of an unsigned display axis to reference colors."""
    # Keep exact-zero scores neutral because they belong to neither occupied side.
    return [
        AXIS_NEGATIVE_COLOR
        if float(value) < 0.0
        else AXIS_POSITIVE_COLOR
        if float(value) > 0.0
        else AXIS_ZERO_COLOR
        for value in first_coordinates.tolist()
    ]


def _candidate_metrics(candidate: dict[str, Any], *, color: str) -> dict[str, Any]:
    """Build the non-rendering portion of one candidate metrics artifact."""
    # Preserve evidence and only the family-relevant scalar summaries from reporting.
    record = candidate["record"]
    family = str(candidate["primary_label"])
    return {
        "feature_id": int(candidate["feature_id"]),
        "primary_label": family,
        "rank": int(candidate["rank"]),
        "color": color,
        "m_median": candidate.get("m_median"),
        "n_valid": candidate.get("n_valid"),
        "secondary_flags": candidate.get("secondary_flags", []),
        "terminal_reason": candidate.get("terminal_reason"),
        "gate_evidence": candidate.get("gate_evidence", {}),
        "evidence_status": candidate.get("evidence_status"),
        "label_confidence": candidate.get("label_confidence"),
        "missingness": record.get("missingness", {}),
        "family_metrics": _family_metrics(family, record),
    }


def _family_metrics(family: str, record: dict[str, Any]) -> dict[str, Any]:
    """Select concise metrics that explain the reported family."""
    # Keep this as a direct family-to-fields projection rather than a new classifier.
    if family == "directed_ray":
        keys = ("c_ray", "s_span_1", "e_res", "n_valid")
    elif family in {"axis_or_antipodal", "oneD_diffuse"}:
        keys = ("c_ray", "s_span_1", "b_axis", "n_valid")
    elif family == "multi_mode_directional_geometry":
        keys = (
            "selected_mode_count",
            "delta_mix",
            "mode_mass_min",
            "min_mode_c_ray",
            "mode_kappa_min",
            "n_valid",
        )
    elif family == "global_2D_directional_subspace":
        keys = ("s_span_2", "u_span_2", "d_span_2", "r_span_pr", "n_valid")
    elif family == "global_kD_directional_subspace":
        selected_k = record.get("selected_k")
        keys = (
            f"s_span_{selected_k}",
            f"u_span_{selected_k}",
            f"d_span_{selected_k}",
            "r_span_pr",
            "n_valid",
        )
    elif family == "residual_lowD_k":
        selected_k = record.get("selected_k")
        keys = ("e_res", f"s_res_{selected_k}", "r_ctr_pr", "n_valid")
    else:
        keys = (
            "c_ray",
            "s_span_1",
            "r_span_pr",
            "e_res",
            "r_ctr_pr",
            "n_valid",
        )
    metrics = {key: record.get(key) for key in keys}
    metrics["selected_k"] = record.get("selected_k")
    if family == "multi_mode_directional_geometry":
        stability = record.get("assignment_stability")
        metrics["assignment_stability"] = (
            stability.get("value") if isinstance(stability, dict) else stability
        )
    if family in {
        "unresolved_high_dimensional_or_diffuse",
        "insufficient_effect_evidence",
        "geometry_metrics_unavailable",
        "undefined_geometry",
    }:
        flags = record.get("secondary_flags") or []
        metrics["strongest_blocker_or_terminal_reason"] = record.get(
            "terminal_reason"
        ) or (flags[0] if flags else None)
    return metrics


def _metric_footer(family: str, record: dict[str, Any]) -> str:
    """Format one short human-readable family metric footer."""
    # Reuse the selected metrics while keeping titles readable at paper-card scale.
    metrics = _family_metrics(family, record)
    parts = []
    for key, value in metrics.items():
        if value is None:
            continue
        shown = f"{value:.3g}" if isinstance(value, float) else str(value)
        parts.append(f"{key}={shown}")
    reason = record.get("terminal_reason")
    if reason:
        parts.append(f"reason={reason}")
    return "  |  ".join(parts)


def _unresolved_candidate_sort_key(
    record: dict[str, Any],
) -> tuple[bool, float, bool, float, bool, float, bool, float, int]:
    """Prefer well-supported examples that visibly retain broad dimensional spread."""
    # Keep context support first, then favor high rank and weak global coherence.
    n_valid = _finite_or_none(record.get("n_valid"))
    span_rank = _finite_or_none(record.get("r_span_pr"))
    c_ray = _finite_or_none(record.get("c_ray"))
    centered_rank = _finite_or_none(record.get("r_ctr_pr"))
    return (
        n_valid is None,
        -n_valid if n_valid is not None else 0.0,
        span_rank is None,
        -span_rank if span_rank is not None else 0.0,
        c_ray is None,
        c_ray if c_ray is not None else 0.0,
        centered_rank is None,
        -centered_rank if centered_rank is not None else 0.0,
        int(record["feature_id"]),
    )


def _candidate_sort_key(
    record: dict[str, Any],
) -> tuple[bool, float, bool, float, int]:
    """Prefer more valid contexts, then stronger effects, with stable ID ties."""
    # Rank finite context counts and magnitudes descending; missing values sort last.
    n_valid = _finite_or_none(record.get("n_valid"))
    magnitude = _finite_or_none(record.get("m_median"))
    return (
        n_valid is None,
        -n_valid if n_valid is not None else 0.0,
        magnitude is None,
        -magnitude if magnitude is not None else 0.0,
        int(record["feature_id"]),
    )


def _finite_or_none(value: Any) -> float | None:
    """Return one finite float or a portable missing value."""
    # Keep nonfinite ranking inputs out of JSON while preserving their missing order.
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _load_palette(path: Path | None) -> dict[str, str]:
    """Merge one optional partial atlas palette after validating hex colors."""
    # Validate the user-controlled CLI file at its single external boundary.
    palette = dict(CLASS_PALETTE)
    if path is None:
        return palette
    payload = _load_json(Path(path).expanduser().resolve(), "palette")
    unknown = sorted(set(payload).difference(ATLAS_LABELS))
    if unknown:
        raise ValueError(f"palette contains unknown atlas labels: {unknown}")
    invalid = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, str) or _HEX_COLOR.fullmatch(value) is None
    }
    if invalid:
        raise ValueError(f"palette colors must use #RRGGBB: {invalid}")
    palette.update(payload)
    return palette


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """Load one required JSON object from the supplied run boundary."""
    # Surface missing or malformed external artifacts without inventing fallbacks.
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _top_values(values: torch.Tensor, *, count: int = 10) -> list[float]:
    """Convert the leading finite spectral values to portable JSON floats."""
    # Limit diagnostics to the components relevant to inspecting these small panels.
    return [float(value) for value in values[:count].tolist()]


def _relative(path: Path, run_dir: Path) -> str:
    """Return one POSIX path relative to the attribute run directory."""
    # Keep cached output relocatable with its owning FEGA run directory.
    return path.relative_to(run_dir).as_posix()
