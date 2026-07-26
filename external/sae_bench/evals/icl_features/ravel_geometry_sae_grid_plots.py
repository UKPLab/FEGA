from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sae_bench.evals.icl_features.artifact_naming import (
    aggregate_artifact_tag,
    tagged_paths,
)
from sae_bench.evals.icl_features.geometry_plots import _configure_style, _format_axis

from fega.core.geometry_reporting.map.embedding import embed
from fega.core.geometry_reporting.map.schema import CLASS_PALETTE, LABEL_DISPLAY_NAMES
from fega.core.geometry_reporting.schema import TERMINAL_LABELS

DEFAULT_SAE_LABELS = {
    "relu": "ReLU",
    "topk": "TopK",
    "matryoshka": "Matryoshka Batch TopK",
}


def parse_sae_map(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=path/to/geometry_map_data.json")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("SAE label must be non-empty")
    return label, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-style RAVEL geometry figure with one panel per SAE "
            "architecture for a shared model/width."
        )
    )
    parser.add_argument(
        "--sae-map",
        action="append",
        type=parse_sae_map,
        required=True,
        help="LABEL=path/to/ravel/.../geometry_map_data.json; pass once per SAE.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="ravel_geometry_three_saes")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--sae-width", default=None)
    parser.add_argument(
        "--embedding", choices=["auto", "umap", "pca", "tsne"], default="auto"
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["precomputed", "joint"],
        default="precomputed",
        help=(
            "precomputed uses the per-SAE coordinates already stored by FEGA; "
            "joint recomputes one shared embedding across all supplied SAEs."
        ),
    )
    parser.add_argument(
        "--share-axes",
        action="store_true",
        help=(
            "Use one common x/y range across panels. This is automatic for "
            "--coordinate-mode joint."
        ),
    )
    parser.add_argument(
        "--include-insufficient-evidence",
        action="store_true",
        help=(
            "Include terminal FEGA labels such as insufficient_effect_evidence. "
            "By default these rows are hidden to match the native FEGA atlas."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=34.0)
    parser.add_argument("--point-alpha", type=float, default=0.90)
    return parser.parse_args()


def _infer_sae_uid_from_map_path(path: Path) -> str | None:
    parts = path.resolve().parts
    for marker in ("ravel", "city_Country"):
        if marker in parts:
            index = parts.index(marker)
            if index >= 1:
                return parts[index - 1]
    return None


def _load_map_rows(
    path: Path,
    *,
    sae_label: str,
    include_insufficient_evidence: bool,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Geometry map has no features list: {path}")
    rows: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        row = dict(feature)
        row["sae_label"] = sae_label
        row["source_map"] = str(path.resolve())
        row["atlas_label"] = str(row.get("atlas_label") or row.get("primary_label"))
        if (
            not include_insufficient_evidence
            and str(row.get("primary_label")) in TERMINAL_LABELS
        ):
            continue
        rows.append(row)
    return rows


def _attach_joint_embedding(
    rows: list[dict[str, Any]], embedding: str, seed: int
) -> dict[str, Any]:
    if not rows:
        return {
            "embedding": {"method": "none", "reason": "no_features"},
            "preprocessing": {},
        }
    coords, metadata, preprocessing = embed(rows, embedding=embedding, seed=seed)
    for row, coord in zip(rows, coords, strict=True):
        row["joint_x"] = float(coord[0])
        row["joint_y"] = float(coord[1])
    return {"embedding": metadata, "preprocessing": preprocessing}


def _attach_precomputed_coordinates(
    panels: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    missing = []
    for sae_label, panel_rows in panels:
        for row in panel_rows:
            coordinates = row.get("embedding")
            if not isinstance(coordinates, dict):
                missing.append((sae_label, row.get("feature_id")))
                continue
            try:
                row["joint_x"] = float(coordinates["x"])
                row["joint_y"] = float(coordinates["y"])
            except (KeyError, TypeError, ValueError):
                missing.append((sae_label, row.get("feature_id")))
    if missing:
        preview = ", ".join(
            f"{label}:{feature_id}" for label, feature_id in missing[:5]
        )
        raise ValueError(
            "Some RAVEL geometry-map rows are missing persisted FEGA coordinates: "
            f"{preview}"
        )
    return {
        "embedding": {
            "method": "precomputed_per_sae",
            "reason": "using coordinates stored in each FEGA geometry_map_data.json",
        },
        "preprocessing": {},
    }


def _apply_shared_axes(axes: list[Any], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    xs = np.asarray([float(row["joint_x"]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row["joint_y"]) for row in rows], dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return
    x_min, x_max = float(xs[finite].min()), float(xs[finite].max())
    y_min, y_max = float(ys[finite].min()), float(ys[finite].max())
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min, 1.0)
    half_span = 0.55 * span
    x_limits = (center_x - half_span, center_x + half_span)
    y_limits = (center_y - half_span, center_y + half_span)
    x_ticks = np.linspace(x_limits[0], x_limits[1], 5)
    y_ticks = np.linspace(y_limits[0], y_limits[1], 5)
    for ax in axes:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)


def _ordered_labels(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(row["atlas_label"]) for row in rows)
    return sorted(counts, key=lambda label: (-counts[label], label))


def _draw_ravel_rows(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    point_size: float,
    point_alpha: float,
) -> None:
    for label in _ordered_labels(rows):
        selected = [row for row in rows if str(row["atlas_label"]) == label]
        if not selected:
            continue
        ax.scatter(
            [float(row["joint_x"]) for row in selected],
            [float(row["joint_y"]) for row in selected],
            s=point_size,
            c=CLASS_PALETTE.get(label, "#7f7f7f"),
            marker="o",
            alpha=point_alpha,
            edgecolors="#2a2a2a",
            linewidths=0.34,
        )
    _format_axis(ax)


def _legend_handles(rows: list[dict[str, Any]]) -> list[Any]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=CLASS_PALETTE.get(label, "#7f7f7f"),
            markeredgecolor="#222222",
            markeredgewidth=0.9,
            markersize=12,
            label=LABEL_DISPLAY_NAMES.get(label, label),
        )
        for label in _ordered_labels(rows)
    ]


def _add_geometry_legend(fig: Any, rows: list[dict[str, Any]]) -> None:
    handles = _legend_handles(rows)
    if not handles:
        return
    ncol = min(5, max(1, len(handles)))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=ncol,
        frameon=False,
        columnspacing=1.45,
        handletextpad=0.5,
    )


def _write_data_artifacts(
    output_dir: Path,
    basename: str,
    tag: str,
    specs: list[tuple[str, Path]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    coordinate_mode: str,
    share_axes: bool,
    include_insufficient_evidence: bool,
) -> None:
    fields = [
        "sae_label",
        "feature_id",
        "atlas_label",
        "primary_label",
        "joint_x",
        "joint_y",
        "m_median",
        "n_valid",
        "source_map",
    ]
    for path in tagged_paths(output_dir / f"{basename}_plot_data.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})

    labels = sorted(CLASS_PALETTE)
    counts = Counter((str(row["sae_label"]), str(row["atlas_label"])) for row in rows)
    sae_totals = {
        sae_label: sum(
            value
            for (row_sae_label, _), value in counts.items()
            if row_sae_label == sae_label
        )
        for sae_label, _ in specs
    }
    count_rows = []
    for sae_label, _ in specs:
        for label in labels:
            count = counts.get((sae_label, label), 0)
            total = sae_totals.get(sae_label, 0)
            count_rows.append(
                {
                    "sae_label": sae_label,
                    "geometry_category": label,
                    "count": count,
                    "sae_total": total,
                    "fraction": count / total if total else 0.0,
                }
            )
    for path in tagged_paths(output_dir / f"{basename}_category_counts.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "sae_label",
                    "geometry_category",
                    "count",
                    "sae_total",
                    "fraction",
                ],
            )
            writer.writeheader()
            writer.writerows(count_rows)

    metadata_payload = {
        "schema_version": 1,
        "artifact_tag": tag,
        "task": "ravel_city_Country",
        "coordinate_mode": coordinate_mode,
        "share_axes": share_axes,
        "include_insufficient_evidence": include_insufficient_evidence,
        "sae_maps": [{"label": label, "path": str(path)} for label, path in specs],
        "category_counts": count_rows,
        **metadata,
    }
    for path in tagged_paths(output_dir / f"{basename}_metadata.json", tag):
        path.write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")


def write_ravel_sae_grid(
    output_dir: Path,
    specs: list[tuple[str, Path]],
    *,
    basename: str,
    model_name: str | None,
    sae_width: str | None,
    embedding: str,
    coordinate_mode: str,
    share_axes: bool,
    include_insufficient_evidence: bool,
    seed: int,
    point_size: float,
    point_alpha: float,
) -> None:
    panels = [
        (
            label,
            _load_map_rows(
                path,
                sae_label=label,
                include_insufficient_evidence=include_insufficient_evidence,
            ),
        )
        for label, path in specs
    ]
    rows = [row for _, panel_rows in panels for row in panel_rows]
    if coordinate_mode == "joint":
        metadata = _attach_joint_embedding(rows, embedding, seed)
    else:
        metadata = _attach_precomputed_coordinates(panels)
    sae_uids = [
        uid
        for _, path in specs
        if (uid := _infer_sae_uid_from_map_path(path)) is not None
    ]
    tag = aggregate_artifact_tag(
        model_name=model_name,
        sae_uids=sae_uids,
        sae_width=sae_width,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    width = max(5.2 * len(panels), 5.2)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(width, 6.65),
        dpi=300,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    use_shared_axes = share_axes or coordinate_mode == "joint"
    if use_shared_axes:
        _apply_shared_axes(flat_axes, rows)
    for index, (ax, (label, panel_rows)) in enumerate(
        zip(flat_axes, panels, strict=True), start=1
    ):
        _draw_ravel_rows(
            ax,
            panel_rows,
            point_size=point_size,
            point_alpha=point_alpha,
        )
        if use_shared_axes:
            _apply_shared_axes([ax], rows)
        ax.text(
            0.5,
            -0.045,
            f"({chr(96 + index)}) {label}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=19,
            fontweight="normal",
        )

    _add_geometry_legend(fig, rows)
    fig.tight_layout(rect=[0.0, 0.20, 1.0, 1.0], w_pad=0.8)
    for suffix in ("png", "pdf"):
        for path in tagged_paths(output_dir / f"{basename}.{suffix}", tag):
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    _write_data_artifacts(
        output_dir,
        basename,
        tag,
        specs,
        rows,
        metadata,
        coordinate_mode=coordinate_mode,
        share_axes=use_shared_axes,
        include_insufficient_evidence=include_insufficient_evidence,
    )


def main() -> None:
    args = parse_args()
    _configure_style()
    write_ravel_sae_grid(
        args.output_dir,
        args.sae_map,
        basename=args.basename,
        model_name=args.model_name,
        sae_width=args.sae_width,
        embedding=args.embedding,
        coordinate_mode=args.coordinate_mode,
        share_axes=args.share_axes,
        include_insufficient_evidence=args.include_insufficient_evidence,
        seed=args.seed,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
    )


if __name__ == "__main__":
    main()
