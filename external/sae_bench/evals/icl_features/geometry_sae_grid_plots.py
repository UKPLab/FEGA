from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sae_bench.evals.icl_features.artifact_naming import (
    aggregate_artifact_tag,
    tagged_paths,
)
from sae_bench.evals.icl_features.geometry_plots import (
    TASK_EDGE_COLORS,
    TASK_MARKERS,
    TASK_ORDER,
    _add_horizontal_legends,
    _configure_style,
    _format_axis,
)

from fega.core.geometry_reporting.map.embedding import embed
from fega.core.geometry_reporting.map.schema import CLASS_PALETTE


def parse_sae_plot(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=path/to/geometry_plot_data.csv")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("SAE label must be non-empty")
    return label, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a shared three-SAE geometry figure from saved plot data."
    )
    parser.add_argument(
        "--sae-plot",
        action="append",
        type=parse_sae_plot,
        required=True,
        help="LABEL=path/to/geometry_plot_data.csv; pass once per SAE.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", default="geometry_three_saes")
    parser.add_argument(
        "--embedding", choices=["auto", "umap", "pca", "tsne"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_plot_rows(path: Path, *, sae_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    map_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            out = _row_from_source_map(row, map_cache) or dict(row)
            if "vector" not in out:
                if row.get("joint_x") in (None, "") or row.get("joint_y") in (None, ""):
                    continue
                out["joint_x"] = float(row["joint_x"])
                out["joint_y"] = float(row["joint_y"])
            out["atlas_label"] = str(out.get("atlas_label") or out.get("primary_label"))
            out["task"] = str(row.get("task") or out.get("task"))
            out["sae_label"] = sae_label
            out["source_map"] = row.get("source_map") or out.get("source_map")
            rows.append(out)
    return rows


def _row_from_source_map(
    row: dict[str, str], map_cache: dict[Path, dict[int, dict[str, Any]]]
) -> dict[str, Any] | None:
    raw_source = row.get("source_map")
    raw_feature = row.get("feature_id")
    if not raw_source or raw_feature in (None, ""):
        return None
    source_map = Path(raw_source)
    if source_map not in map_cache:
        payload = json.loads(source_map.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list):
            return None
        map_cache[source_map] = {
            int(feature["feature_id"]): dict(feature)
            for feature in features
            if isinstance(feature, dict) and feature.get("feature_id") is not None
        }
    feature = map_cache[source_map].get(int(raw_feature))
    if feature is None:
        return None
    out = dict(feature)
    out["source_map"] = str(source_map)
    return out


def attach_shared_embedding(
    rows: list[dict[str, Any]], embedding: str, seed: int
) -> dict[str, Any]:
    if rows and all("vector" in row for row in rows):
        coords, metadata, preprocessing = embed(rows, embedding=embedding, seed=seed)
        for row, coord in zip(rows, coords, strict=True):
            row["joint_x"] = float(coord[0])
            row["joint_y"] = float(coord[1])
        return {"embedding": metadata, "preprocessing": preprocessing}
    return {
        "embedding": {
            "method": "precomputed_per_sae",
            "reason": "one_or_more_rows_missing_source_vectors",
        },
        "preprocessing": {},
    }


def _model_and_uid_from_plot_path(path: Path) -> tuple[str | None, str | None]:
    # Expected: RESULT_ROOT / MODEL_NAME / SAE_UID / cross_task / geometry_plots / file
    parts = path.resolve().parts
    if len(parts) < 6:
        return None, None
    if parts[-3] == "cross_task":
        return parts[-5], parts[-4]
    if parts[-4] == "cross_task":
        return parts[-6], parts[-5]
    return None, None


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


def _coerce_feature_id(row: dict[str, Any]) -> int | None:
    raw_feature_id = row.get("feature_id")
    if raw_feature_id in (None, ""):
        return None
    try:
        return int(raw_feature_id)
    except (TypeError, ValueError):
        return None


def _common_all_task_feature_ids(rows: list[dict[str, Any]]) -> set[int]:
    features_by_task: dict[str, set[int]] = {task: set() for task in TASK_ORDER}
    for row in rows:
        task = str(row.get("task"))
        if task not in features_by_task:
            continue
        feature_id = _coerce_feature_id(row)
        if feature_id is not None:
            features_by_task[task].add(feature_id)
    if any(not feature_ids for feature_ids in features_by_task.values()):
        return set()
    return set.intersection(*features_by_task.values())


def _mark_common_all_task_rows(
    rows: list[dict[str, Any]],
    common_feature_ids_by_sae: dict[str, set[int]],
) -> None:
    for row in rows:
        sae_label = str(row.get("sae_label"))
        feature_id = _coerce_feature_id(row)
        row["common_all_tasks"] = bool(
            feature_id is not None
            and feature_id in common_feature_ids_by_sae.get(sae_label, set())
        )


def _draw_rows_with_common_emphasis(ax: Any, rows: list[dict[str, Any]]) -> None:
    labels = sorted(
        {str(row["atlas_label"]) for row in rows},
        key=lambda label: (-sum(row["atlas_label"] == label for row in rows), label),
    )
    tasks = [task for task in TASK_ORDER if any(row["task"] == task for row in rows)]

    for is_common in (False, True):
        for label in labels:
            for task in tasks:
                selected = [
                    row
                    for row in rows
                    if row["atlas_label"] == label
                    and row["task"] == task
                    and bool(row.get("common_all_tasks")) is is_common
                ]
                if not selected:
                    continue
                marker = TASK_MARKERS.get(task, "o")
                xs = [float(row["joint_x"]) for row in selected]
                ys = [float(row["joint_y"]) for row in selected]
                if is_common:
                    ax.scatter(
                        xs,
                        ys,
                        s=126,
                        c="none",
                        marker=marker,
                        edgecolors="#1f1f1f",
                        linewidths=1.25,
                        alpha=0.92,
                        zorder=12,
                    )
                ax.scatter(
                    xs,
                    ys,
                    s=92 if is_common else 66,
                    c=CLASS_PALETTE.get(label, "#7f7f7f"),
                    marker=marker,
                    alpha=0.94 if is_common else 0.45,
                    edgecolors=(
                        TASK_EDGE_COLORS.get(task, "#222222")
                        if is_common
                        else "#8a8a8a"
                    ),
                    linewidths=1.05 if is_common else 0.6,
                    zorder=13 if is_common else 5,
                )
    _format_axis(ax)


def write_sae_grid_plot(
    output_dir: Path,
    specs: list[tuple[str, Path]],
    *,
    basename: str,
    embedding: str,
    seed: int,
) -> None:
    panels = [(label, load_plot_rows(path, sae_label=label)) for label, path in specs]
    all_rows = [row for _, rows in panels for row in rows]
    metadata = attach_shared_embedding(all_rows, embedding, seed)
    common_feature_ids_by_sae = {
        label: _common_all_task_feature_ids(rows) for label, rows in panels
    }
    _mark_common_all_task_rows(all_rows, common_feature_ids_by_sae)
    model_names = []
    sae_uids = []
    for _, path in specs:
        model_name, sae_uid = _model_and_uid_from_plot_path(path)
        if model_name:
            model_names.append(model_name)
        if sae_uid:
            sae_uids.append(sae_uid)
    tag = aggregate_artifact_tag(
        model_name=next(iter(dict.fromkeys(model_names)), None),
        sae_uids=sae_uids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    width = max(5.2 * len(panels), 5.2)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(width, 6.7),
        dpi=300,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    _apply_shared_axes(flat_axes, all_rows)
    for index, (ax, (label, rows)) in enumerate(zip(flat_axes, panels, strict=True), start=1):
        _draw_rows_with_common_emphasis(ax, rows)
        _apply_shared_axes([ax], all_rows)
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

    _add_horizontal_legends(fig, all_rows, task_y=0.085, class_y=0.012)
    fig.tight_layout(rect=[0.0, 0.22, 1.0, 1.0], w_pad=0.8)
    for suffix in ("png", "pdf"):
        for path in tagged_paths(output_dir / f"{basename}.{suffix}", tag):
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    fieldnames = [
        "sae_label",
        "task",
        "feature_id",
        "atlas_label",
        "primary_label",
        "joint_x",
        "joint_y",
        "common_all_tasks",
        "source_map",
    ]
    for path in tagged_paths(output_dir / f"{basename}_plot_data.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({field: row.get(field) for field in fieldnames})
    metadata_payload = {
        "schema_version": 1,
        "artifact_tag": tag,
        "sae_plots": [
            {"label": label, "path": str(path)} for label, path in specs
        ],
        "common_all_tasks": {
            "tasks": list(TASK_ORDER),
            "feature_ids_by_sae": {
                label: sorted(feature_ids)
                for label, feature_ids in common_feature_ids_by_sae.items()
            },
            "feature_counts_by_sae": {
                label: len(feature_ids)
                for label, feature_ids in common_feature_ids_by_sae.items()
            },
        },
        **metadata,
    }
    for path in tagged_paths(output_dir / f"{basename}_metadata.json", tag):
        path.write_text(
            json.dumps(metadata_payload, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    _configure_style()
    write_sae_grid_plot(
        args.output_dir,
        args.sae_plot,
        basename=args.basename,
        embedding=args.embedding,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
