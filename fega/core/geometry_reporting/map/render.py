from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fega.core.geometry_reporting.map.schema import (
    ATLAS_LABELS,
    BITMASK_RING_POLICY,
    CLASS_PALETTE,
    GLOBAL_FLAG_OVERLAY_POLICY,
    GLOBAL_FLAG_PATTERN_POLICY,
    LABEL_DISPLAY_NAMES,
    MARKER_POLICY,
    OUTLINE_POLICY,
    SIZE_POLICY,
)
from fega.core.geometry_reporting.map.utils import finite_float, row_label_counts
from fega.core.geometry_reporting.schema import (
    GLOBAL_FLAG_MASK,
    GLOBAL_FLAG_ORDER,
    TERMINAL_LABELS,
)

GLOBAL_FLAG_MODES = {"layered_overlay", "bitmask_ring"}
GLOBAL_FLAG_LAYER_ORDER = (
    "long_tail_spectrum",
    "sample_size_unstable",
    "leave_out_unstable",
    "exploratory_low_n",
    "magnitude_unstable",
)


def apply_visual_policies(
    rows: list[dict[str, Any]], *, global_flag_mode: str = "layered_overlay"
) -> None:
    """Attach README_v2 atlas visual metadata to each feature-map row."""
    if global_flag_mode not in GLOBAL_FLAG_MODES:
        raise ValueError(
            "`global_flag_mode` must be `layered_overlay` or `bitmask_ring`."
        )
    for row in rows:
        # Keep the marker body reserved for primary-label color semantics.
        flags = [str(flag) for flag in row.get("secondary_flags") or []]
        marker = marker_for_flags(flags)
        outline = "emphasized" if uses_emphasis_outline(flags) else "normal"
        row["atlas_label"] = str(row.get("primary_label"))
        row["marker"] = marker
        row["size"] = effect_size(row.get("m_median"))
        row["outline"] = {
            "style": outline,
            "edgecolor": outline_edgecolor(outline),
            "linewidth": outline_linewidth(outline),
        }
        row["global_flag_visuals"] = global_flag_visuals(
            row, mode=global_flag_mode
        )


def atlas_rows(
    rows: list[dict[str, Any]], *, include_insufficient_evidence: bool
) -> list[dict[str, Any]]:
    if include_insufficient_evidence:
        return rows
    return [
        row
        for row in rows
        if str(row.get("primary_label")) not in TERMINAL_LABELS
    ]


def write_scatter(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
) -> None:
    """Write an atlas scatter using primary-label colors plus flag overlays."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    if rows:
        # Draw primary-label colored marker bodies before additive flag layers.
        for label in label_plot_order(rows):
            color = CLASS_PALETTE.get(label)
            if color is None:
                continue
            label_rows = [row for row in rows if row.get("atlas_label") == label]
            if not label_rows:
                continue
            ax.scatter(
                [row["embedding"]["x"] for row in label_rows],
                [row["embedding"]["y"] for row in label_rows],
                c=color,
                marker=str(MARKER_POLICY["ordinary"]),
                s=[row["size"] for row in label_rows],
                alpha=0.76,
                linewidths=[
                    float(row["outline"]["linewidth"]) for row in label_rows
                ],
                edgecolors=[
                    str(row["outline"]["edgecolor"]) for row in label_rows
                ],
            )
        draw_global_flag_visuals(ax, rows)
        annotate_representatives(ax, rows)
    title_artist = fig.suptitle(title, fontsize=15, y=0.98)
    ax.set_title(subtitle, fontsize=9, loc="left", pad=8)
    ax.set_xlabel("Feature-map coordinate 1")
    ax.set_ylabel("Feature-map coordinate 2")
    ax.grid(True, alpha=0.25)
    class_handles = class_legend_handles(rows)
    flag_handles = global_flag_legend_handles(rows)
    bbox_extra_artists = [title_artist]
    if class_handles:
        class_legend = ax.legend(
            handles=class_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="Primary labels",
            fontsize=8,
            title_fontsize=8,
            frameon=True,
            handletextpad=0.8,
            labelspacing=0.7,
            borderpad=0.5,
        )
        bbox_extra_artists.append(class_legend)
        if flag_handles:
            ax.add_artist(class_legend)
    if flag_handles:
        flag_legend = ax.legend(
            handles=flag_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.42),
            title="Global flags",
            fontsize=8,
            title_fontsize=8,
            frameon=True,
            handletextpad=0.8,
            labelspacing=0.7,
            borderpad=0.5,
        )
        bbox_extra_artists.append(flag_legend)
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 0.94])
    fig.savefig(
        path,
        bbox_inches="tight",
        bbox_extra_artists=tuple(bbox_extra_artists),
    )
    plt.close(fig)


def label_plot_order(rows: list[dict[str, Any]]) -> list[str]:
    counts = row_label_counts(rows, "atlas_label")
    rank = {label: idx for idx, label in enumerate(ATLAS_LABELS)}
    return sorted(
        [label for label in ATLAS_LABELS if counts.get(label, 0) > 0],
        key=lambda label: (-counts[label], rank[label]),
    )


def marker_for_flags(flags: list[str]) -> str:
    """Return the base atlas marker without consuming global-flag semantics."""
    # Global flags are encoded by overlays, not by replacing marker shape.
    del flags
    return str(MARKER_POLICY["ordinary"])


def uses_emphasis_outline(flags: list[str]) -> bool:
    """Return whether non-global diagnostic flags request outline emphasis."""
    # The current V2 policy leaves global flags out of outline styling.
    return any(flag in flags for flag in OUTLINE_POLICY["emphasized_flags"])


def outline_edgecolor(style: str) -> str:
    """Map an outline style to a deterministic edge color."""
    # Emphasis is reserved for non-global diagnostic flags.
    if style == "emphasized":
        return str(OUTLINE_POLICY["emphasized_edgecolor"])
    return str(OUTLINE_POLICY["ordinary_edgecolor"])


def outline_linewidth(style: str) -> float:
    """Map an outline style to a deterministic line width."""
    # Keep ordinary outlines light so overlay glyphs remain legible.
    if style == "emphasized":
        return 1.0
    return 0.4


def effect_size(value: Any) -> float:
    parsed = finite_float(value)
    if parsed is None or parsed < 0.0:
        return float(SIZE_POLICY["fallback"])
    return float(16.0 + 22.0 * math.log1p(parsed))


def global_flag_visuals(
    row: dict[str, Any], *, mode: str = "layered_overlay"
) -> dict[str, Any]:
    """Build complete row-level metadata for layered and bitmask flag visuals."""
    if mode not in GLOBAL_FLAG_MODES:
        raise ValueError("Unknown global flag visual mode.")
    # Keep flag order stable so JSON payloads and legends are deterministic.
    active_flags = set(str(flag) for flag in row.get("global_flags") or [])
    ordered_flags = [flag for flag in GLOBAL_FLAG_ORDER if flag in active_flags]
    layers = [
        _layer_for_flag(flag)
        for flag in GLOBAL_FLAG_LAYER_ORDER
        if flag in active_flags
    ]
    ring_slots = [
        {"slot": slot, "flag": spec["flag"], "code": spec["code"]}
        for slot, spec in BITMASK_RING_POLICY["slots"].items()
    ]
    active_slots = [
        {"slot": slot, "flag": spec["flag"], "code": spec["code"]}
        for slot, spec in BITMASK_RING_POLICY["slots"].items()
        if spec["flag"] in active_flags
    ]
    return {
        "mode": mode,
        "mask": str(row.get("global_flag_mask") or ""),
        "count": int(row.get("global_flag_count") or 0),
        "flags": ordered_flags,
        "layers": layers,
        "bitmask_ring": {"slots": ring_slots, "active": active_slots},
        "active": active_slots if mode == "bitmask_ring" else layers,
    }


def _layer_for_flag(flag: str) -> dict[str, Any]:
    """Return the additive overlay policy for one active global flag."""
    # Pattern and overlay policies are separate in metadata but unified per row.
    if flag in GLOBAL_FLAG_PATTERN_POLICY:
        policy = dict(GLOBAL_FLAG_PATTERN_POLICY[flag])
    else:
        policy = dict(GLOBAL_FLAG_OVERLAY_POLICY["overlays"][flag])
    return {"flag": flag, "code": GLOBAL_FLAG_MASK[flag], **policy}


def draw_global_flag_visuals(ax: Any, rows: list[dict[str, Any]]) -> None:
    """Draw active global-flag visuals after base markers."""
    # Dispatch by row mode so mixed payloads remain renderable in tests.
    layered_rows = [
        row
        for row in rows
        if row.get("global_flag_visuals", {}).get("mode") != "bitmask_ring"
    ]
    ring_rows = [
        row
        for row in rows
        if row.get("global_flag_visuals", {}).get("mode") == "bitmask_ring"
    ]
    if layered_rows:
        _draw_layered_global_flag_visuals(ax, layered_rows)
    if ring_rows:
        _draw_bitmask_ring_visuals(ax, ring_rows)


def _draw_layered_global_flag_visuals(ax: Any, rows: list[dict[str, Any]]) -> None:
    """Draw README_v2 layered flag glyphs in the documented order."""
    # Draw dot fill, line fills, cross fill, then centered X overlay.
    _scatter_flag_rows(
        ax,
        rows,
        "long_tail_spectrum",
        marker="o",
        size_scale=0.018,
        min_size=0.45,
        color="#505050",
        linewidth=0.0,
        alpha=0.14,
    )
    _scatter_flag_rows(
        ax,
        rows,
        "sample_size_unstable",
        marker=_path_marker("forward_diagonal"),
        size_scale=0.68,
        min_size=3.0,
        color="#505050",
        linewidth=0.24,
        alpha=0.18,
    )
    _scatter_flag_rows(
        ax,
        rows,
        "leave_out_unstable",
        marker=_path_marker("reverse_diagonal"),
        size_scale=0.68,
        min_size=3.0,
        color="#505050",
        linewidth=0.24,
        alpha=0.18,
    )
    _scatter_flag_rows(
        ax,
        rows,
        "exploratory_low_n",
        marker=_path_marker("cross"),
        size_scale=0.56,
        min_size=3.0,
        color="#505050",
        linewidth=0.22,
        alpha=0.16,
    )
    _scatter_flag_rows(
        ax,
        rows,
        "magnitude_unstable",
        marker="x",
        size_scale=0.62,
        min_size=4.0,
        color="black",
        linewidth=0.52,
        alpha=0.58,
    )


def _scatter_flag_rows(
    ax: Any,
    rows: list[dict[str, Any]],
    flag: str,
    *,
    marker: Any,
    size_scale: float,
    min_size: float,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    """Scatter one overlay layer for rows carrying ``flag``."""
    # Select by global_flags so secondary non-global names cannot affect styling.
    selected = [row for row in rows if _row_has_global_flag(row, flag)]
    if not selected:
        return
    ax.scatter(
        [row["embedding"]["x"] for row in selected],
        [row["embedding"]["y"] for row in selected],
        marker=marker,
        s=[max(float(row["size"]) * size_scale, min_size) for row in selected],
        c=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=4,
    )


def _path_marker(kind: str) -> Any:
    """Return a line-only marker path for diagonal and cross fills."""
    from matplotlib.path import Path as MplPath

    # Matplotlib path markers let line fills scale with scatter marker size.
    if kind == "forward_diagonal":
        vertices = [(-0.48, -0.48), (0.48, 0.48)]
        codes = [MplPath.MOVETO, MplPath.LINETO]
    elif kind == "reverse_diagonal":
        vertices = [(-0.48, 0.48), (0.48, -0.48)]
        codes = [MplPath.MOVETO, MplPath.LINETO]
    else:
        vertices = [(-0.45, 0.0), (0.45, 0.0), (0.0, -0.45), (0.0, 0.45)]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.MOVETO,
            MplPath.LINETO,
        ]
    return MplPath(vertices, codes)


def _draw_bitmask_ring_visuals(ax: Any, rows: list[dict[str, Any]]) -> None:
    """Draw compact fixed-slot glyphs for dense bitmask-ring mode."""
    offsets = {
        "top": (0.0, 6.5),
        "upper_right": (5.6, 3.4),
        "lower_right": (5.6, -3.4),
        "bottom": (0.0, -6.5),
        "center_glyph": (0.0, 0.0),
    }
    for row in rows:
        # Use point offsets so ring slots stay stable under axis scaling.
        size_pt = math.sqrt(max(float(row.get("size") or 0.0), 1.0))
        fontsize = max(3.0, min(7.0, size_pt * 0.38))
        active = row.get("global_flag_visuals", {}).get("bitmask_ring", {}).get(
            "active", []
        )
        for entry in active:
            slot = str(entry["slot"])
            dx, dy = offsets[slot]
            glyph = "x" if slot == "center_glyph" else "o"
            ax.annotate(
                glyph,
                (row["embedding"]["x"], row["embedding"]["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                fontweight="bold",
                zorder=5,
            )


def _row_has_global_flag(row: dict[str, Any], flag: str) -> bool:
    """Return whether a map row carries one global visual flag."""
    # Only global_flags participate in global-flag visual policy.
    return flag in set(str(item) for item in row.get("global_flags") or [])


def legend_handles(rows: list[dict[str, Any]]) -> list[Any]:
    """Return combined handles for callers that expect one legend list."""
    return class_legend_handles(rows) + global_flag_legend_handles(rows)


def class_legend_handles(rows: list[dict[str, Any]]) -> list[Any]:
    """Return primary-label color legend handles."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="none",
            markerfacecolor=CLASS_PALETTE[label],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            label=LABEL_DISPLAY_NAMES[label],
        )
        for label in label_plot_order(rows)
    ]


def global_flag_legend_handles(rows: list[dict[str, Any]]) -> list[Any]:
    """Return separate global-flag visual legend handles."""
    from matplotlib.lines import Line2D

    style_handles: list[Any] = []
    active_flags = {
        flag
        for row in rows
        for flag in (row.get("global_flags") or [])
    }
    if "long_tail_spectrum" in active_flags:
        style_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=4,
                label=LABEL_DISPLAY_NAMES["long_tail_spectrum"],
            )
        )
    if "sample_size_unstable" in active_flags:
        style_handles.append(
            Line2D(
                [0],
                [0],
                marker=_path_marker("forward_diagonal"),
                linestyle="None",
                color="black",
                markeredgewidth=1.0,
                markersize=8,
                label=LABEL_DISPLAY_NAMES["sample_size_unstable"],
            )
        )
    if "leave_out_unstable" in active_flags:
        style_handles.append(
            Line2D(
                [0],
                [0],
                marker=_path_marker("reverse_diagonal"),
                linestyle="None",
                color="black",
                markeredgewidth=1.0,
                markersize=8,
                label=LABEL_DISPLAY_NAMES["leave_out_unstable"],
            )
        )
    if "exploratory_low_n" in active_flags:
        style_handles.append(
            Line2D(
                [0],
                [0],
                marker=_path_marker("cross"),
                linestyle="None",
                color="black",
                markeredgewidth=1.0,
                markersize=8,
                label=LABEL_DISPLAY_NAMES["exploratory_low_n"],
            )
        )
    if "magnitude_unstable" in active_flags:
        style_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                color="black",
                markersize=8,
                label=LABEL_DISPLAY_NAMES["magnitude_unstable"],
            )
        )
    return style_handles


def annotate_representatives(ax: Any, rows: list[dict[str, Any]]) -> None:
    offsets = [(5, 5), (5, -10), (-24, 5), (-24, -10), (8, 14), (-34, 14), (8, -18)]
    for idx, row in enumerate(annotation_rows(rows)):
        offset = offsets[idx % len(offsets)]
        ax.annotate(
            f"f{int(row['feature_id'])}",
            (row["embedding"]["x"], row["embedding"]["y"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )


def annotation_rows(
    rows: list[dict[str, Any]], max_labels: int = 7
) -> list[dict[str, Any]]:
    by_id = {int(row["feature_id"]): row for row in rows}
    candidates = [
        by_id[feature_id]
        for feature_id in representative_feature_id_list(rows)
        if feature_id in by_id
    ]
    selected: list[dict[str, Any]] = []
    min_distance = max(coord_span(rows, "x"), coord_span(rows, "y"), 1.0) * 0.08
    for row in candidates:
        if len(selected) >= max_labels:
            break
        if all(coord_distance(row, existing) >= min_distance for existing in selected):
            selected.append(row)
    return selected


def representative_feature_id_list(rows: list[dict[str, Any]]) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()

    def add(feature_id: int) -> None:
        if feature_id not in seen:
            selected.append(feature_id)
            seen.add(feature_id)

    for label in ATLAS_LABELS:
        label_rows = [row for row in rows if row.get("atlas_label") == label]
        if label_rows:
            add(int(max_row(label_rows, "m_median")["feature_id"]))
    for key in ("delta_mix", "b_axis"):
        valued = [row for row in rows if finite_float(row["vector"].get(key)) is not None]
        if valued:
            add(int(max_vector_row(valued, key)["feature_id"]))
    residual = [row for row in rows if row.get("primary_label") == "residual_lowD_k"]
    if residual:
        add(int(max_vector_row(residual, "e_res")["feature_id"]))
    low_evidence = [
        row
        for row in rows
        if row.get("primary_label") == "insufficient_effect_evidence"
    ]
    for row in sorted(low_evidence, key=_low_evidence_sort_key)[:2]:
        add(int(row["feature_id"]))
    return selected


def representative_feature_ids(rows: list[dict[str, Any]]) -> set[int]:
    return set(representative_feature_id_list(rows))


def coord_span(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["embedding"][key]) for row in rows if "embedding" in row]
    return 0.0 if not values else max(values) - min(values)


def coord_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(
        float(left["embedding"]["x"]) - float(right["embedding"]["x"]),
        float(left["embedding"]["y"]) - float(right["embedding"]["y"]),
    )


def max_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            finite_float(row.get(key))
            if finite_float(row.get(key)) is not None
            else -math.inf,
            -int(row["feature_id"]),
        ),
    )


def max_vector_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            finite_float(row["vector"].get(key))
            if finite_float(row["vector"].get(key)) is not None
            else -math.inf,
            -int(row["feature_id"]),
        ),
    )


def _low_evidence_sort_key(row: dict[str, Any]) -> tuple[float, int]:
    n_valid = finite_float(row.get("n_valid"))
    return (n_valid if n_valid is not None else math.inf, int(row["feature_id"]))
