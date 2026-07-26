from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import (
    artifact_tag,
    default_discovery_root,
    tagged_path,
)
from sae_bench.evals.icl_features.feature_sets import (
    load_discovery_summary,
    resolve_feature_set,
)
from sae_bench.evals.icl_features.iou import jaccard

TASK_ORDER = ("lsc", "wc", "prontoqa", "tt", "ravel")
RAVEL_FILE_BY_VARIANT = {
    "standard": "standard",
    "topk": "topk",
    "matryoshka": "matryoshka",
}
DRAFT_RAVEL_COUNTS = {
    "standard": 15_295,
    "topk": 446,
    "matryoshka": 1_750,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute candidate-feature IoU between ICL task feature sets and "
            "precomputed RAVEL feature-id files, writing new RAVEL-specific "
            "artifacts without touching existing IoU files."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/sae_geometry_gemma2b_65k"),
    )
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=None,
        help="Discovery inputs; defaults to data/induction_feature_outputs/RESULT_NAME.",
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument(
        "--ravel-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing RAVEL files named standard, topk, matryoshka. "
            "Defaults to RESULT_ROOT/MODEL_NAME/ravel_feature_ids."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["lsc", "wc", "prontoqa", "tt"],
        help="ICL tasks whose discovery summaries should be compared to RAVEL.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["candidate", "threshold", "strict", "strict_common"],
        default="candidate",
        help="Task feature set to compare against RAVEL.",
    )
    parser.add_argument("--kept-threshold", type=float, default=8.0)
    parser.add_argument(
        "--kept-op",
        choices=["gt", "ge"],
        default="ge",
        help="Use kept > threshold (gt) or kept >= threshold (ge) for RAVEL.",
    )
    parser.add_argument(
        "--output-name",
        default="ravel_candidate_iou",
        help=(
            "Basename for new per-SAE output files. The script writes "
            "OUTPUT_NAME.{json,matrix.csv,pairs.csv,counts.csv}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting previously generated RAVEL-specific output files.",
    )
    return parser.parse_args()


def sae_variant(sae_uid: str) -> str:
    lowered = sae_uid.lower()
    if "matryoshka" in lowered:
        return "matryoshka"
    if "top_k" in lowered or "topk" in lowered:
        return "topk"
    if "standard" in lowered or "relu" in lowered:
        return "standard"
    raise ValueError(f"Could not infer SAE variant from UID: {sae_uid}")


def read_ravel_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RAVEL feature file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("feature_stats"), dict
    ):
        raise ValueError(f"{path}: expected JSON object with feature_stats map")
    return payload


def ravel_feature_ids(
    payload: dict[str, Any], *, kept_threshold: float, kept_op: str
) -> list[int]:
    ids = []
    for raw_feature_id, stats in payload["feature_stats"].items():
        if not isinstance(stats, dict):
            continue
        kept = float(stats.get("kept", 0.0))
        keep = kept > kept_threshold if kept_op == "gt" else kept >= kept_threshold
        if keep:
            ids.append(int(raw_feature_id))
    return sorted(set(ids))


def ravel_count_summary(
    payload: dict[str, Any], *, kept_threshold: float
) -> dict[str, Any]:
    stats = payload["feature_stats"]
    return {
        "raw_feature_stats_count": len(stats),
        "features_with_contexts_field": payload.get("features_with_contexts"),
        "total_features_field": payload.get("total_features"),
        "kept_gt_threshold_count": sum(
            float(record.get("kept", 0.0)) > kept_threshold
            for record in stats.values()
            if isinstance(record, dict)
        ),
        "kept_ge_threshold_count": sum(
            float(record.get("kept", 0.0)) >= kept_threshold
            for record in stats.values()
            if isinstance(record, dict)
        ),
    }


def load_task_summaries(
    discovery_model_root: Path, tasks: list[str]
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for task in tasks:
        path = discovery_model_root / task / "summary.json"
        summaries[task] = load_discovery_summary(path)
    return summaries


def compute_payload(
    *,
    result_root: Path,
    model_name: str,
    task_summaries: dict[str, dict[str, Any]],
    sae_uid: str,
    feature_set: str,
    ravel_path: Path,
    ravel_payload: dict[str, Any],
    ravel_ids: list[int],
    kept_threshold: float,
    kept_op: str,
) -> dict[str, Any]:
    variant = sae_variant(sae_uid)
    tasks = [task for task in TASK_ORDER if task in task_summaries] + ["ravel"]
    feature_sets = {
        task: set(resolve_feature_set(task_summaries[task], sae_uid, feature_set))
        for task in task_summaries
    }
    feature_sets["ravel"] = set(ravel_ids)
    rows = []
    matrix = []
    for left_task in tasks:
        matrix_row = []
        for right_task in tasks:
            left = feature_sets[left_task]
            right = feature_sets[right_task]
            value = jaccard(left, right)
            matrix_row.append(value)
            rows.append(
                {
                    "task_a": left_task,
                    "task_b": right_task,
                    "feature_set": feature_set,
                    "count_a": len(left),
                    "count_b": len(right),
                    "intersection_count": len(left & right),
                    "union_count": len(left | right),
                    "iou": value,
                }
            )
        matrix.append(matrix_row)

    first_summary = task_summaries[tasks[0]]
    feature_entry = first_summary["feature_sets"][sae_uid]
    count_summary = ravel_count_summary(ravel_payload, kept_threshold=kept_threshold)
    draft_count = DRAFT_RAVEL_COUNTS.get(variant)
    selected_ravel_count = len(ravel_ids)
    return {
        "schema_version": 1,
        "model_name": model_name,
        "sae_uid": sae_uid,
        "sae_variant": variant,
        "sae_release": feature_entry.get("sae_release"),
        "sae_id": feature_entry.get("sae_id"),
        "feature_set": feature_set,
        "tasks": tasks,
        "ravel": {
            "source_path": str(ravel_path.resolve()),
            "selection_rule": {
                "field": "kept",
                "op": kept_op,
                "threshold": kept_threshold,
            },
            "selected_feature_count": selected_ravel_count,
            "draft_table_count": draft_count,
            "matches_draft_table_count": (
                None if draft_count is None else selected_ravel_count == draft_count
            ),
            **count_summary,
        },
        "feature_ids": {task: sorted(values) for task, values in feature_sets.items()},
        "matrix": matrix,
        "pairs": rows,
        "result_root": str(result_root.resolve()),
    }


def output_paths(output_dir: Path, output_name: str) -> dict[str, Path]:
    return {
        "json": output_dir / f"{output_name}.json",
        "matrix": output_dir / f"{output_name}_matrix.csv",
        "pairs": output_dir / f"{output_name}_pairs.csv",
        "counts": output_dir / f"{output_name}_counts.csv",
    }


def assert_can_write(paths: dict[str, Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing RAVEL IoU outputs; pass --overwrite "
            "to replace them:\n" + "\n".join(str(path) for path in existing)
        )


def write_payload(
    output_dir: Path, output_name: str, payload: dict[str, Any], *, overwrite: bool
) -> None:
    paths = output_paths(output_dir, output_name)
    tag = artifact_tag(
        model_name=str(payload.get("model_name") or ""),
        sae_uid=str(payload.get("sae_uid") or ""),
    )
    tagged_paths_by_name = {
        name: tagged_path(path, tag) for name, path in paths.items()
    }
    all_paths = {
        **{f"canonical_{name}": path for name, path in paths.items()},
        **{f"tagged_{name}": path for name, path in tagged_paths_by_name.items()},
    }
    assert_can_write(all_paths, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, artifact_tag=tag)
    for path in (paths["json"], tagged_paths_by_name["json"]):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    tasks = payload["tasks"]
    for path in (paths["matrix"], tagged_paths_by_name["matrix"]):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["task", *tasks])
            for task, values in zip(tasks, payload["matrix"], strict=True):
                writer.writerow(
                    [
                        task,
                        *[
                            ""
                            if value is None or not math.isfinite(value)
                            else f"{value:.8f}"
                            for value in values
                        ],
                    ]
                )

    for path in (paths["pairs"], tagged_paths_by_name["pairs"]):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(payload["pairs"][0]))
            writer.writeheader()
            writer.writerows(payload["pairs"])

    ravel = payload["ravel"]
    count_row = {
        "model_name": payload["model_name"],
        "sae_uid": payload["sae_uid"],
        "sae_variant": payload["sae_variant"],
        "feature_set": payload["feature_set"],
        "ravel_selected_feature_count": ravel["selected_feature_count"],
        "draft_table_count": ravel["draft_table_count"],
        "matches_draft_table_count": ravel["matches_draft_table_count"],
        "raw_feature_stats_count": ravel["raw_feature_stats_count"],
        "features_with_contexts_field": ravel["features_with_contexts_field"],
        "kept_gt_threshold_count": ravel["kept_gt_threshold_count"],
        "kept_ge_threshold_count": ravel["kept_ge_threshold_count"],
        "kept_op": ravel["selection_rule"]["op"],
        "kept_threshold": ravel["selection_rule"]["threshold"],
        "source_path": ravel["source_path"],
    }
    count_row["artifact_tag"] = tag
    for path in (paths["counts"], tagged_paths_by_name["counts"]):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(count_row))
            writer.writeheader()
            writer.writerow(count_row)


def main() -> None:
    args = parse_args()
    result_root = args.result_root
    model_name = args.model_name
    ravel_dir = args.ravel_dir or (result_root / model_name / "ravel_feature_ids")
    discovery_model_root = (
        args.discovery_root or default_discovery_root(result_root)
    ) / model_name
    task_summaries = load_task_summaries(discovery_model_root, args.tasks)
    shared_saes = set.intersection(
        *[set(summary["feature_sets"]) for summary in task_summaries.values()]
    )
    if not shared_saes:
        raise ValueError("No SAE UID is shared across all supplied task summaries")

    for sae_uid in sorted(shared_saes):
        variant = sae_variant(sae_uid)
        ravel_path = ravel_dir / RAVEL_FILE_BY_VARIANT[variant]
        ravel_payload = read_ravel_file(ravel_path)
        ravel_ids = ravel_feature_ids(
            ravel_payload,
            kept_threshold=args.kept_threshold,
            kept_op=args.kept_op,
        )
        payload = compute_payload(
            result_root=result_root,
            model_name=model_name,
            task_summaries=task_summaries,
            sae_uid=sae_uid,
            feature_set=args.feature_set,
            ravel_path=ravel_path,
            ravel_payload=ravel_payload,
            ravel_ids=ravel_ids,
            kept_threshold=args.kept_threshold,
            kept_op=args.kept_op,
        )
        output_dir = result_root / model_name / sae_uid / "cross_task" / "iou"
        write_payload(
            output_dir,
            args.output_name,
            payload,
            overwrite=args.overwrite,
        )
        ravel = payload["ravel"]
        print(
            f"{variant}: wrote {args.output_name} for {sae_uid} "
            f"(RAVEL count={ravel['selected_feature_count']}, "
            f"draft_count={ravel['draft_table_count']}, "
            f"matches_draft={ravel['matches_draft_table_count']})"
        )


if __name__ == "__main__":
    main()
