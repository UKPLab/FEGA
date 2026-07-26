from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sae_bench.evals.icl_features.artifact_naming import artifact_tag, tagged_paths

from fega.core.geometry_reporting.map.embedding import embed
from fega.core.geometry_reporting.map.schema import CLASS_PALETTE, LABEL_DISPLAY_NAMES

TASK_MARKERS = {"lsc": "o", "wc": "s", "tt": "^", "prontoqa": "P"}
TASK_EDGE_COLORS = {
    "lsc": "#111111",
    "wc": "#009e73",
    "tt": "#005a9c",
    "prontoqa": "#b15928",
}
TASK_ORDER = ("lsc", "wc", "tt", "prontoqa")


def parse_task_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TASK=PATH")
    task, raw_path = value.split("=", 1)
    return task.strip(), Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate and joint paper plots from FEGA geometry map data."
    )
    parser.add_argument(
        "--task-map",
        action="append",
        type=parse_task_path,
        required=True,
        help="TASK=path/to/geometry_map_data.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sae-uid", required=True)
    parser.add_argument(
        "--embedding", choices=["auto", "umap", "pca", "tsne"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default="SAE Feature Effect Geometry Across ICL Tasks")
    return parser.parse_args()


def load_joint_rows(task_maps: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for task, path in task_maps:
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError(f"Geometry map has no features list: {path}")
        for feature in features:
            row = dict(feature)
            row["task"] = task
            row["source_map"] = str(path.resolve())
            row["atlas_label"] = str(
                row.get("atlas_label") or row.get("primary_label")
            )
            rows.append(row)
    return rows


def attach_joint_embedding(
    rows: list[dict[str, Any]], embedding: str, seed: int
) -> dict[str, Any]:
    coords, metadata, preprocessing = embed(rows, embedding=embedding, seed=seed)
    for row, coord in zip(rows, coords, strict=True):
        row["joint_x"] = float(coord[0])
        row["joint_y"] = float(coord[1])
    return {"embedding": metadata, "preprocessing": preprocessing}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_axis(ax: Any) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        left=True,
        bottom=True,
        labelleft=False,
        labelbottom=False,
        length=4.2,
        width=0.9,
        color="#7f7f7f",
    )
    ax.grid(color="#d7d7d7", linewidth=0.7, alpha=0.72)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9f9f9f")
        spine.set_linewidth(0.9)


def _draw_rows(ax: Any, rows: list[dict[str, Any]], *, task_markers: bool) -> None:
    labels = sorted(
        {str(row["atlas_label"]) for row in rows},
        key=lambda label: (-sum(row["atlas_label"] == label for row in rows), label),
    )
    tasks = sorted({str(row["task"]) for row in rows})
    for label in labels:
        for task in tasks:
            selected = [
                row
                for row in rows
                if row["atlas_label"] == label and row["task"] == task
            ]
            if not selected:
                continue
            ax.scatter(
                [row["joint_x"] for row in selected],
                [row["joint_y"] for row in selected],
                s=92,
                c=CLASS_PALETTE.get(label, "#7f7f7f"),
                marker=TASK_MARKERS.get(task, "o") if task_markers else "o",
                alpha=0.90,
                edgecolors=(
                    TASK_EDGE_COLORS.get(task, "#222222")
                    if task_markers
                    else "#222222"
                ),
                linewidths=1.05 if task_markers else 0.8,
            )
    _format_axis(ax)


def _legend_handles(rows: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    from matplotlib.lines import Line2D

    labels = sorted({str(row["atlas_label"]) for row in rows})
    class_handles = [
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
        for label in labels
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS.get(task, "o"),
            linestyle="",
            color=TASK_EDGE_COLORS.get(task, "#222222"),
            markerfacecolor="#f2f2f2",
            markeredgecolor=TASK_EDGE_COLORS.get(task, "#222222"),
            markeredgewidth=1.8,
            markersize=13.5,
            label=task.upper() if task != "prontoqa" else "PrOntoQA",
        )
        for task in [task for task in TASK_ORDER if any(row["task"] == task for row in rows)]
    ]
    return class_handles, task_handles


def _add_horizontal_legends(
    fig: Any,
    rows: list[dict[str, Any]],
    *,
    task_y: float = 0.115,
    class_y: float = 0.035,
) -> None:
    class_handles, task_handles = _legend_handles(rows)
    if task_handles:
        fig.legend(
            handles=task_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, task_y),
            ncol=max(1, len(task_handles)),
            frameon=False,
            columnspacing=2.0,
            handletextpad=0.55,
        )
    if class_handles:
        fig.legend(
            handles=class_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, class_y),
            ncol=max(1, len(class_handles)),
            frameon=False,
            columnspacing=1.45,
            handletextpad=0.5,
        )


def write_combined_plot(
    output_dir: Path, rows: list[dict[str, Any]], title: str, *, tag: str
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.6), dpi=300)
    _draw_rows(ax, rows, task_markers=True)
    _add_horizontal_legends(fig, rows)
    fig.tight_layout(rect=[0.0, 0.19, 1.0, 1.0])
    for suffix in ("png", "pdf"):
        for path in tagged_paths(output_dir / f"geometry_all_tasks.{suffix}", tag):
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_panel_plot(output_dir: Path, rows: list[dict[str, Any]], *, tag: str) -> None:
    tasks = [task for task in TASK_ORDER if any(row["task"] == task for row in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 10.8), dpi=300, sharex=True, sharey=True)
    for ax, task in zip(axes.flat, tasks, strict=False):
        task_rows = [row for row in rows if row["task"] == task]
        _draw_rows(ax, task_rows, task_markers=True)
    for ax in list(axes.flat)[len(tasks) :]:
        ax.set_visible(False)
    _add_horizontal_legends(fig, rows, task_y=0.075, class_y=0.012)
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])
    for suffix in ("png", "pdf"):
        for path in tagged_paths(output_dir / f"geometry_task_panels.{suffix}", tag):
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_individual_plots(output_dir: Path, rows: list[dict[str, Any]], *, tag: str) -> None:
    separate_dir = output_dir / "separate"
    separate_dir.mkdir(parents=True, exist_ok=True)
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=300)
        _draw_rows(ax, task_rows, task_markers=True)
        _add_horizontal_legends(fig, task_rows)
        fig.tight_layout(rect=[0.0, 0.19, 1.0, 1.0])
        for suffix in ("png", "pdf"):
            for path in tagged_paths(separate_dir / f"geometry_{task}.{suffix}", tag):
                fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def write_data_artifacts(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    model_name: str,
    sae_uid: str,
    tasks: list[str],
    tag: str,
) -> None:
    fields = [
        "task",
        "feature_id",
        "atlas_label",
        "primary_label",
        "joint_x",
        "joint_y",
        "m_median",
        "n_valid",
        "source_map",
    ]
    for path in tagged_paths(output_dir / "geometry_plot_data.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fields})
    categories = sorted(CLASS_PALETTE)
    counts = Counter((str(row["task"]), str(row["atlas_label"])) for row in rows)
    task_totals = {
        task: sum(value for (row_task, _), value in counts.items() if row_task == task)
        for task in tasks
    }
    count_rows = [
        {
            "model_name": model_name,
            "sae_uid": sae_uid,
            "task": task,
            "geometry_category": category,
            "count": counts.get((task, category), 0),
            "task_total": task_totals.get(task, 0),
        }
        for task in tasks
        for category in categories
    ]
    for row in count_rows:
        row["fraction"] = (
            row["count"] / row["task_total"] if row["task_total"] else 0.0
        )
    for path in tagged_paths(output_dir / "geometry_category_counts.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "model_name",
                    "sae_uid",
                    "task",
                    "geometry_category",
                    "count",
                    "task_total",
                    "fraction",
                ],
            )
            writer.writeheader()
            writer.writerows(count_rows)
    metadata_payload = {
        "schema_version": 1,
        "model_name": model_name,
        "sae_uid": sae_uid,
        "artifact_tag": tag,
        **metadata,
        "category_counts": count_rows,
    }
    for path in tagged_paths(output_dir / "geometry_plot_metadata.json", tag):
        path.write_text(
            json.dumps(metadata_payload, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    _configure_style()
    tasks = [task for task, _ in args.task_map]
    rows = load_joint_rows(args.task_map)
    tag = artifact_tag(model_name=args.model_name, sae_uid=args.sae_uid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        metadata = attach_joint_embedding(rows, args.embedding, args.seed)
        write_combined_plot(args.output_dir, rows, args.title, tag=tag)
        write_panel_plot(args.output_dir, rows, tag=tag)
        write_individual_plots(args.output_dir, rows, tag=tag)
    else:
        metadata = {
            "embedding": {"method": "none", "reason": "no_features"},
            "preprocessing": {},
        }
    write_data_artifacts(
        args.output_dir,
        rows,
        metadata,
        model_name=args.model_name,
        sae_uid=args.sae_uid,
        tasks=tasks,
        tag=tag,
    )


if __name__ == "__main__":
    main()
