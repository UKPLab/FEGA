from __future__ import annotations

import argparse
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


TASK_DISPLAY_NAMES = {
    "lsc": "LSC",
    "wc": "WC",
    "prontoqa": "PrOntoQA",
    "tt": "TT",
    "ravel": "RAVEL",
}
DEFAULT_TASK_ORDER = ("lsc", "wc", "prontoqa", "tt", "ravel")


def parse_iou_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=path/to/ravel_iou.json")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("SAE label must be non-empty")
    return label, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RAVEL-inclusive 5-task IoU matrices as a three-SAE heatmap row."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/sae_geometry_gemma2b_65k"),
        help="Root containing MODEL_NAME/<SAE_UID>/cross_task/iou outputs.",
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument(
        "--iou-name",
        default="ravel_candidate_iou.json",
        help="RAVEL IoU JSON filename inside each SAE's cross_task/iou directory.",
    )
    parser.add_argument(
        "--sae-iou",
        action="append",
        type=parse_iou_spec,
        default=None,
        help=(
            "Optional explicit LABEL=path/to/ravel_candidate_iou.json. "
            "If omitted, files are discovered from result-root/model-name."
        ),
    )
    parser.add_argument(
        "--task-order",
        nargs="+",
        default=list(DEFAULT_TASK_ORDER),
        help="Task order for rows/columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "RESULT_ROOT/MODEL_NAME/cross_sae/iou_heatmaps."
        ),
    )
    parser.add_argument("--basename", default="ravel_candidate_iou_three_saes")
    parser.add_argument("--cmap", default="Blues")
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sae_label_from_uid(sae_uid: str) -> str:
    lowered = sae_uid.lower()
    if "matryoshka" in lowered:
        return "Matryoshka Batch TopK"
    if "top_k" in lowered or "topk" in lowered:
        return "TopK"
    if "standard" in lowered or "relu" in lowered:
        return "ReLU"
    return sae_uid


def label_sort_key(label: str) -> tuple[int, str]:
    lowered = label.lower()
    if "relu" in lowered:
        return (0, label)
    if "topk" in lowered and "matryoshka" not in lowered:
        return (1, label)
    if "matryoshka" in lowered:
        return (2, label)
    return (99, label)


def discover_iou_specs(result_root: Path, model_name: str, iou_name: str) -> list[tuple[str, Path]]:
    model_root = result_root / model_name
    specs = []
    for path in sorted(model_root.glob(f"*/cross_task/iou/{iou_name}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = sae_label_from_uid(str(payload.get("sae_uid") or path.parts[-4]))
        specs.append((label, path))
    specs.sort(key=lambda item: label_sort_key(item[0]))
    return specs


def matrix_from_payload(payload: dict[str, Any], task_order: list[str]) -> np.ndarray:
    tasks = [str(task) for task in payload["tasks"]]
    raw = np.asarray(payload["matrix"], dtype=np.float64)
    index = {task: idx for idx, task in enumerate(tasks)}
    missing = [task for task in task_order if task not in index]
    if missing:
        raise ValueError(
            f"{payload.get('sae_uid', '<unknown>')}: IoU payload missing tasks {missing}"
        )
    out = np.empty((len(task_order), len(task_order)), dtype=np.float64)
    for row_index, row_task in enumerate(task_order):
        for col_index, col_task in enumerate(task_order):
            out[row_index, col_index] = raw[index[row_task], index[col_task]]
    return out


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    values = matrix.copy()
    values[np.tril_indices_from(values, k=-1)] = np.nan
    return values


def draw_heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    task_order: list[str],
    cmap: Any,
) -> Any:
    masked = np.ma.masked_invalid(upper_triangle(matrix))
    image = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest")
    n_tasks = len(task_order)
    labels = [TASK_DISPLAY_NAMES.get(task, task) for task in task_order]
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    ax.set_box_aspect(1)
    ax.set_xlim(-0.5, n_tasks - 0.5)
    ax.set_ylim(n_tasks - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, n_tasks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_tasks, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#9f9f9f")
        spine.set_linewidth(0.9)

    for row in range(n_tasks):
        for col in range(row, n_tasks):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            color = "white" if value >= 0.55 else "#202020"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=13,
                color=color,
            )
    return image


def write_heatmap_grid(
    specs: list[tuple[str, Path]],
    *,
    task_order: list[str],
    output_dir: Path,
    basename: str,
    cmap_name: str,
    model_name: str,
) -> None:
    if not specs:
        raise ValueError("No RAVEL IoU JSON files were supplied or discovered.")
    payloads = [(label, json.loads(path.read_text(encoding="utf-8")), path) for label, path in specs]
    tag = aggregate_artifact_tag(
        model_name=model_name,
        sae_uids=[str(payload.get("sae_uid") or "") for _, payload, _ in payloads],
    )
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f5f5f5", alpha=1.0)

    fig, axes = plt.subplots(
        1,
        len(payloads),
        figsize=(5.35 * len(payloads) + 0.85, 7.2),
        dpi=300,
        squeeze=False,
        constrained_layout=False,
    )
    images = []
    for index, (ax, (label, payload, _path)) in enumerate(
        zip(axes.flat, payloads, strict=True), start=1
    ):
        matrix = matrix_from_payload(payload, task_order)
        images.append(draw_heatmap(ax, matrix, task_order=task_order, cmap=cmap))
        ax.text(
            0.5,
            -0.24,
            f"({chr(96 + index)}) {label}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=19,
            fontweight="normal",
        )

    cbar_ax = fig.add_axes([0.08, 0.905, 0.84, 0.026])
    cbar = fig.colorbar(images[0], cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=15, length=4, width=0.9)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.245, wspace=0.27)
    for suffix in ("png", "pdf"):
        for path in tagged_paths(output_dir / f"{basename}.{suffix}", tag):
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "artifact_tag": tag,
        "task_order": task_order,
        "color_scale": {"vmin": 0.0, "vmax": 1.0, "cmap": cmap_name},
        "upper_triangle_only": True,
        "iou_files": [
            {"label": label, "path": str(path), "sae_uid": payload.get("sae_uid")}
            for label, payload, path in payloads
        ],
    }
    for path in tagged_paths(output_dir / f"{basename}_metadata.json", tag):
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _configure_style()
    output_dir = args.output_dir or (
        args.result_root / args.model_name / "cross_sae" / "iou_heatmaps"
    )
    specs = args.sae_iou or discover_iou_specs(
        args.result_root, args.model_name, args.iou_name
    )
    specs = sorted(specs, key=lambda item: label_sort_key(item[0]))
    write_heatmap_grid(
        specs,
        task_order=[str(task) for task in args.task_order],
        output_dir=output_dir,
        basename=args.basename,
        cmap_name=args.cmap,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
