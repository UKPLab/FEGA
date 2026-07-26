from __future__ import annotations

from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting.artifacts import write_json_atomic
from fega.core.geometry_reporting.map import embedding as map_embedding
from fega.core.geometry_reporting.map import render as map_render
from fega.core.geometry_reporting.map.rows import feature_rows as _feature_rows
from fega.core.geometry_reporting.map.schema import (
    ATLAS_LABELS,
    BITMASK_RING_POLICY,
    CLASS_PALETTE,
    GEOMETRY_REPORT_LABELS,
    GLOBAL_FLAG_OVERLAY_POLICY,
    GLOBAL_FLAG_PATTERN_POLICY,
    GLOBAL_FLAG_VISUAL_POLICY,
    LABEL_DISPLAY_NAMES,
    LABEL_INTERPRETATIONS,
    MAP_VECTOR_KEYS,
    MARKER_POLICY,
    MISSINGNESS_KEYS,
    OUTLINE_POLICY,
    PRIMARY_LABELS,
    SECONDARY_FLAGS,
    SIZE_POLICY,
)
from fega.core.geometry_reporting.map.stats import write_stats_artifacts
from fega.core.geometry_reporting.map.utils import finite_float as _finite_float
from fega.core.geometry_reporting.map.utils import row_label_counts as _row_label_counts
from fega.paths import (
    geometry_reporting_figures_dir,
    geometry_reporting_map_data_path,
)

__all__ = [
    "ATLAS_LABELS",
    "BITMASK_RING_POLICY",
    "CLASS_PALETTE",
    "GLOBAL_FLAG_OVERLAY_POLICY",
    "GLOBAL_FLAG_PATTERN_POLICY",
    "GLOBAL_FLAG_VISUAL_POLICY",
    "LABEL_DISPLAY_NAMES",
    "LABEL_INTERPRETATIONS",
    "MAP_VECTOR_KEYS",
    "MARKER_POLICY",
    "MISSINGNESS_KEYS",
    "OUTLINE_POLICY",
    "PRIMARY_LABELS",
    "SECONDARY_FLAGS",
    "GEOMETRY_REPORT_LABELS",
    "SIZE_POLICY",
    "_apply_visual_policies",
    "_atlas_rows",
    "_effective_seed",
    "_embed",
    "_finite_float",
    "_fit_umap",
    "_label_plot_order",
    "_preprocess_embedding_matrix",
    "_row_label_counts",
    "_umap_params",
    "_write_scatter",
    "write_geometry_maps",
]

_apply_visual_policies = map_render.apply_visual_policies
_atlas_rows = map_render.atlas_rows
_fit_umap = map_embedding.fit_umap
_label_plot_order = map_render.label_plot_order
_preprocess_embedding_matrix = map_embedding.preprocess_embedding_matrix
_umap_params = map_embedding.umap_params
_effective_seed = map_embedding.effective_seed


def write_geometry_maps(
    config: FEGAPipelineConfig,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = config.phases.geometry_reporting
    figures_dir = geometry_reporting_figures_dir(config)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_figures(figures_dir)
    rows = _feature_rows(records)
    if not rows:
        embedding_metadata = {"method": "none", "reason": "no_features"}
        preprocessing: dict[str, Any] = {}
        stats_path, counts_path = write_stats_artifacts(
            config, records, rows, embedding_metadata
        )
        payload = {
            "phase": "geometry_reporting",
            "schema_version": 2,
            "embedding": embedding_metadata,
            "preprocessing": preprocessing,
            "palette": CLASS_PALETTE,
            "marker_policy": MARKER_POLICY,
            "size_policy": SIZE_POLICY,
            "outline_policy": OUTLINE_POLICY,
            "global_flag_visual_policy": GLOBAL_FLAG_VISUAL_POLICY,
            "features": [],
            "figure_metadata": {
                "atlas": None,
                "atlas_include_insufficient_evidence": (
                    cfg.atlas_include_insufficient_evidence
                ),
                "global_flag_mode": cfg.global_flag_mode,
                "atlas_global_flags_rendered": False,
                "atlas_point_count": 0,
                "atlas_excluded_point_count": 0,
                "atlas_label_counts": {},
                "zooms": {},
                "stats": str(stats_path),
                "counts": str(counts_path),
            },
        }
        write_json_atomic(geometry_reporting_map_data_path(config), payload)
        return payload
    coords, embedding_metadata, preprocessing = _embed(
        rows, embedding=cfg.embedding, seed=cfg.seed
    )
    for row, coord in zip(rows, coords, strict=False):
        row["embedding"] = {"x": float(coord[0]), "y": float(coord[1])}
    _apply_visual_policies(rows, global_flag_mode=cfg.global_flag_mode)
    atlas_rows = _atlas_rows(
        rows,
        include_insufficient_evidence=cfg.atlas_include_insufficient_evidence,
    )

    atlas_path = figures_dir / "geometry_atlas.png"
    _write_scatter(
        atlas_path,
        _atlas_display_rows(atlas_rows),
        title="Geometry Reporting Feature Atlas",
        subtitle=(
            "Diagnostic map: color = FEGA geometry label, "
            "size = median effect strength."
        ),
    )
    zoom_paths = _write_zoom_figures(figures_dir, rows)
    stats_path, counts_path = write_stats_artifacts(
        config, records, rows, embedding_metadata
    )
    payload = {
        "phase": "geometry_reporting",
        "schema_version": 2,
        "embedding": embedding_metadata,
        "preprocessing": preprocessing,
        "palette": CLASS_PALETTE,
        "marker_policy": MARKER_POLICY,
        "size_policy": SIZE_POLICY,
        "outline_policy": OUTLINE_POLICY,
        "global_flag_visual_policy": GLOBAL_FLAG_VISUAL_POLICY,
        "features": rows,
        "figure_metadata": {
            "atlas": str(atlas_path),
            "atlas_include_insufficient_evidence": (
                cfg.atlas_include_insufficient_evidence
            ),
            "global_flag_mode": cfg.global_flag_mode,
            "atlas_global_flags_rendered": False,
            "atlas_point_count": len(atlas_rows),
            "atlas_excluded_point_count": len(rows) - len(atlas_rows),
            "atlas_label_counts": _row_label_counts(atlas_rows, "atlas_label"),
            "zooms": {name: str(path) for name, path in zoom_paths.items()},
            "stats": str(stats_path),
            "counts": str(counts_path),
        },
    }
    write_json_atomic(geometry_reporting_map_data_path(config), payload)
    return payload


def _clear_stale_figures(figures_dir: Path) -> None:
    """Remove old geometry map figures before writing the current zoom set."""
    # Zoom membership is data-dependent, so absent zooms must not leave stale PNGs.
    for path in figures_dir.glob("*.png"):
        path.unlink()


def _atlas_display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return atlas-only row copies with global diagnostic overlays suppressed."""
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        # Keep the persisted row contract intact while making the atlas readable.
        display = dict(row)
        display["global_flags"] = []
        display["global_flag_count"] = 0
        display["global_flag_mask"] = ""
        display["global_flag_visuals"] = {
            "mode": row.get("global_flag_visuals", {}).get(
                "mode", "layered_overlay"
            ),
            "mask": "",
            "count": 0,
            "flags": [],
            "layers": [],
            "bitmask_ring": row.get("global_flag_visuals", {}).get(
                "bitmask_ring", {"slots": [], "active": []}
            )
            | {"active": []},
            "active": [],
        }
        display_rows.append(display)
    return display_rows


def _embed(
    rows: list[dict[str, Any]], *, embedding: str, seed: int | None
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    old_fit_umap = map_embedding.fit_umap
    map_embedding.fit_umap = _fit_umap
    try:
        return map_embedding.embed(rows, embedding=embedding, seed=seed)
    finally:
        map_embedding.fit_umap = old_fit_umap


def _write_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
) -> None:
    map_render.write_scatter(path, rows, title=title, subtitle=subtitle)


def _write_zoom_figures(
    figures_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Path]:
    specs = {
        "directed_ray_map": {"directed_ray"},
        "axis_antipodal_map": {"axis_or_antipodal"},
        "oneD_diffuse_map": {"oneD_diffuse"},
        "multimode_map": {"multi_mode_directional_geometry"},
        "lowD_subspace_map": {
            "global_2D_directional_subspace",
            "global_kD_directional_subspace",
        },
        "residual_lowD_map": {"residual_lowD_k"},
        "unresolved_map": {"unresolved_high_dimensional_or_diffuse"},
    }
    written = _write_label_zooms(figures_dir, rows, specs)
    diagnostic_specs = {
        "span_metric_pass_stability_missing": "span_metric_pass_stability_missing",
        "unresolved_long_tail": "unresolved_long_tail",
        "near_directed_ray_ci_failed": "near_directed_ray_ci_failed",
        "prevented_high_dimensional_fallback": (
            "prevented_high_dimensional_fallback"
        ),
    }
    for name, tag in diagnostic_specs.items():
        selected = [row for row in rows if tag in set(row.get("zoom_tags") or [])]
        if not selected:
            continue
        path = figures_dir / f"{name}.png"
        _write_scatter(
            path,
            selected,
            title=name.replace("_", " ").title(),
            subtitle="Diagnostic zoom reusing atlas coordinates and gate evidence tags.",
        )
        written[name] = path
    return written


def _write_label_zooms(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    specs: dict[str, set[str]],
) -> dict[str, Path]:
    written = {}
    for name, labels in specs.items():
        selected = [row for row in rows if row.get("atlas_label") in labels]
        if not selected:
            continue
        path = figures_dir / f"{name}.png"
        _write_scatter(
            path,
            selected,
            title=name.replace("_", " ").title(),
            subtitle="Zoom view reusing atlas coordinates and FEGA geometry labels.",
        )
        written[name] = path
    return written
