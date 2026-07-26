from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import (
    aggregate_artifact_tag,
    default_discovery_root,
    tagged_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready summary tables from SAE Geometry experiment outputs."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=None,
        help="Discovery inputs; defaults to data/induction_feature_outputs/RESULT_NAME.",
    )
    return parser.parse_args()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    total = len(p_values)
    for rank_index in range(total - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * total / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, tag: str | None = None
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    for output_path in tagged_paths(path, tag):
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _infer_tag(rows: list[dict[str, Any]]) -> str:
    model_names = sorted(
        {str(row.get("model_name")) for row in rows if row.get("model_name")}
    )
    sae_uids = sorted({str(row.get("sae_uid")) for row in rows if row.get("sae_uid")})
    return aggregate_artifact_tag(
        model_name=model_names[0] if len(model_names) == 1 else None,
        sae_uids=sae_uids,
    )


def collect_ablation_rows(
    result_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows = []
    random_rows = []
    for path in sorted(result_root.rglob("ablation_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = payload["results"]["selected_ablation"]
        random_summary = payload["results"]["random_control_summary"]
        significance = selected["drop_significance"]
        analysis_filter = payload.get("analysis_filter") or {}
        selected_rows.append(
            {
                "model_name": payload["model_name"],
                "sae_uid": payload["sae_uid"],
                "sae_release": payload["sae_release"],
                "sae_id": payload["sae_id"],
                "task": payload["task"],
                "feature_set": payload["feature_set"],
                "ablation_position": payload["intervention"]["position"],
                "selected_feature_count": payload["selected_feature_count"],
                "baseline_correct_only": analysis_filter.get("baseline_correct_only"),
                "input_example_count": analysis_filter.get("input_example_count"),
                "analysis_example_count": analysis_filter.get("analysis_example_count"),
                "significance_pair_count": analysis_filter.get(
                    "analysis_example_count"
                ),
                "excluded_baseline_incorrect_count": analysis_filter.get(
                    "excluded_baseline_incorrect_count"
                ),
                "full_dataset_baseline_accuracy": analysis_filter.get(
                    "full_dataset_baseline_accuracy"
                ),
                "baseline_accuracy": payload["results"]["baseline"]["accuracy"],
                "ablated_accuracy": selected["accuracy"],
                "accuracy_drop": selected["accuracy_drop"],
                "mcnemar_p_one_sided": significance["p_value_one_sided"],
                "mcnemar_log10_p_one_sided": significance["log10_p_value_one_sided"],
                "mcnemar_p_two_sided": significance["p_value_two_sided"],
                "discordant_baseline_only": significance[
                    "reference_correct_comparison_wrong"
                ],
                "discordant_ablation_only": significance[
                    "reference_wrong_comparison_correct"
                ],
                "random_trials": random_summary["trials"],
                "random_mean_accuracy": random_summary.get("mean_accuracy"),
                "random_sample_std_accuracy": random_summary.get("sample_std_accuracy"),
                "random_mean_accuracy_drop": random_summary["mean_accuracy_drop"],
                "random_sample_std_accuracy_drop": random_summary.get(
                    "sample_std_accuracy_drop"
                ),
                "random_max_accuracy_drop": random_summary["max_accuracy_drop"],
                "random_aggregate_p_one_sided": (
                    random_summary.get("aggregate_accuracy_significance") or {}
                ).get("p_value_one_sided"),
                "empirical_random_p": random_summary[
                    "empirical_p_value_random_drop_at_least_selected"
                ],
                "summary_path": str(path),
            }
        )
        for row in payload["results"]["random_controls"]:
            random_rows.append(
                {
                    "model_name": payload["model_name"],
                    "sae_uid": payload["sae_uid"],
                    "task": payload["task"],
                    "feature_set": payload["feature_set"],
                    "condition": row["condition"],
                    "accuracy": row["accuracy"],
                    "accuracy_drop": row["accuracy_drop"],
                    "drop_p_one_sided": row["drop_significance"]["p_value_one_sided"],
                    "selected_vs_random_p_one_sided": row["selected_vs_random"][
                        "p_value_one_sided"
                    ],
                }
            )
    p_values = [float(row["mcnemar_p_one_sided"]) for row in selected_rows]
    for row, q_value in zip(selected_rows, benjamini_hochberg(p_values), strict=True):
        row["mcnemar_q_bh"] = q_value
    return selected_rows, random_rows


def collect_feature_counts(discovery_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(discovery_root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        feature_sets = payload.get("feature_sets")
        if not isinstance(feature_sets, dict):
            continue
        task = (payload.get("dataset_metadata") or {}).get(
            "task_name"
        ) or path.parent.name
        for sae_uid, feature_set in feature_sets.items():
            rows.append(
                {
                    "model_name": payload.get("model_name"),
                    "sae_uid": sae_uid,
                    "sae_release": feature_set.get("sae_release"),
                    "sae_id": feature_set.get("sae_id"),
                    "task": task,
                    "threshold_feature_count": feature_set.get(
                        "threshold_feature_count",
                        feature_set.get("candidate_feature_count"),
                    ),
                    "strict_feature_count": feature_set.get(
                        "strict_common_feature_count"
                    ),
                    "min_example_fraction": (
                        feature_set.get("selection_thresholds") or {}
                    ).get("min_example_fraction"),
                    "min_query_fraction_per_family": (
                        feature_set.get("selection_thresholds") or {}
                    ).get("min_query_fraction_per_context"),
                    "min_family_fraction": (
                        feature_set.get("selection_thresholds") or {}
                    ).get("min_context_fraction"),
                    "activation_threshold": (
                        feature_set.get("selection_thresholds") or {}
                    ).get("activation_threshold"),
                    "summary_path": str(path),
                }
            )
    return rows


def collect_geometry_counts(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.rglob("geometry_category_counts.csv")):
        if (
            path.parent == result_root / "tables"
            or path.parent.name != "geometry_plots"
        ):
            continue
        with path.open(newline="", encoding="utf-8") as input_file:
            rows.extend(
                dict(row) | {"counts_path": str(path)}
                for row in csv.DictReader(input_file)
            )
    return rows


def main() -> None:
    args = parse_args()
    table_dir = args.result_root / "tables"
    selected_rows, random_rows = collect_ablation_rows(args.result_root)
    feature_rows = collect_feature_counts(
        args.discovery_root or default_discovery_root(args.result_root)
    )
    geometry_rows = collect_geometry_counts(args.result_root)
    tag = _infer_tag([*selected_rows, *feature_rows, *geometry_rows])
    _write_csv(table_dir / "ablation_results.csv", selected_rows, tag=tag)
    _write_csv(table_dir / "random_control_results.csv", random_rows, tag=tag)
    _write_csv(table_dir / "feature_counts.csv", feature_rows, tag=tag)
    _write_csv(table_dir / "geometry_category_counts.csv", geometry_rows, tag=tag)
    manifest = {
        "schema_version": 1,
        "result_root": str(args.result_root.resolve()),
        "artifact_tag": tag,
        "tables": {
            "ablation_results": len(selected_rows),
            "random_control_results": len(random_rows),
            "feature_counts": len(feature_rows),
            "geometry_category_counts": len(geometry_rows),
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR",
            "family": "all selected-feature ablation tests found under result_root",
            "column": "mcnemar_q_bh",
        },
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    for path in tagged_paths(table_dir / "manifest.json", tag):
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
