from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import artifact_tag, tagged_paths
from sae_bench.evals.icl_features.feature_sets import (
    load_discovery_summary,
    resolve_feature_set,
)

TASK_ORDER = ("lsc", "wc", "prontoqa", "tt", "ravel")


def parse_task_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TASK=PATH")
    task, raw_path = value.split("=", 1)
    return task.strip(), Path(raw_path).expanduser()


def jaccard(left: set[int], right: set[int]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-task feature-set IoU matrices for every shared SAE."
    )
    parser.add_argument(
        "--task-summary",
        action="append",
        type=parse_task_path,
        required=True,
        help="TASK=path/to/discovery/summary.json; pass once per task.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["threshold", "strict"],
        default=["threshold", "strict"],
    )
    return parser.parse_args()


def compute_iou_payload(
    task_summaries: dict[str, dict[str, Any]],
    sae_uid: str,
    feature_set: str,
) -> dict[str, Any]:
    tasks = [
        task for task in TASK_ORDER if task in task_summaries
    ] + sorted(set(task_summaries) - set(TASK_ORDER))
    sets = {
        task: set(resolve_feature_set(task_summaries[task], sae_uid, feature_set))
        for task in tasks
    }
    rows = []
    matrix = []
    for left_task in tasks:
        matrix_row = []
        for right_task in tasks:
            intersection = sets[left_task] & sets[right_task]
            union = sets[left_task] | sets[right_task]
            value = jaccard(sets[left_task], sets[right_task])
            matrix_row.append(value)
            rows.append(
                {
                    "task_a": left_task,
                    "task_b": right_task,
                    "feature_set": feature_set,
                    "count_a": len(sets[left_task]),
                    "count_b": len(sets[right_task]),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "iou": value,
                }
            )
        matrix.append(matrix_row)
    first_summary = task_summaries[tasks[0]]
    feature_entry = first_summary["feature_sets"][sae_uid]
    return {
        "schema_version": 1,
        "model_name": first_summary.get("model_name"),
        "sae_uid": sae_uid,
        "sae_release": feature_entry.get("sae_release"),
        "sae_id": feature_entry.get("sae_id"),
        "feature_set": feature_set,
        "tasks": tasks,
        "feature_ids": {task: sorted(values) for task, values in sets.items()},
        "matrix": matrix,
        "pairs": rows,
    }


def write_iou_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_set = payload["feature_set"]
    tag = artifact_tag(
        model_name=str(payload.get("model_name") or ""),
        sae_uid=str(payload.get("sae_uid") or ""),
    )
    tagged_payload = dict(payload, artifact_tag=tag)
    for path in tagged_paths(output_dir / f"{feature_set}_iou.json", tag):
        path.write_text(
            json.dumps(tagged_payload, indent=2) + "\n",
            encoding="utf-8",
        )
    tasks = payload["tasks"]
    for path in tagged_paths(output_dir / f"{feature_set}_iou_matrix.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["task", *tasks])
            for task, values in zip(tasks, payload["matrix"], strict=True):
                writer.writerow(
                    [
                        task,
                        *[
                            "" if value is None or not math.isfinite(value) else f"{value:.8f}"
                            for value in values
                        ],
                    ]
                )
    for path in tagged_paths(output_dir / f"{feature_set}_iou_pairs.csv", tag):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(payload["pairs"][0]))
            writer.writeheader()
            writer.writerows(payload["pairs"])


def main() -> None:
    args = parse_args()
    task_summaries = {
        task: load_discovery_summary(path) for task, path in args.task_summary
    }
    shared_saes = set.intersection(
        *[set(summary["feature_sets"]) for summary in task_summaries.values()]
    )
    if not shared_saes:
        raise ValueError("No SAE UID is shared across all supplied task summaries")
    for sae_uid in sorted(shared_saes):
        model_name = str(next(iter(task_summaries.values())).get("model_name"))
        output_dir = args.output_root / model_name / sae_uid / "cross_task" / "iou"
        for feature_set in args.feature_sets:
            write_iou_payload(
                output_dir,
                compute_iou_payload(task_summaries, sae_uid, feature_set),
            )


if __name__ == "__main__":
    main()
