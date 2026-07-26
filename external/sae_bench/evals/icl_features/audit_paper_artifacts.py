from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sae_bench.evals.icl_features.artifact_naming import default_discovery_root

TASKS = ("lsc", "wc", "tt", "prontoqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail unless all draft-table and draft-figure artifacts exist."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=None,
        help="Discovery inputs; defaults to data/induction_feature_outputs/RESULT_NAME.",
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--sae-uid", action="append", required=True)
    parser.add_argument("--feature-set", default="threshold")
    parser.add_argument("--expected-examples", type=int, default=50_000)
    parser.add_argument("--expected-random-trials", type=int, default=20)
    parser.add_argument("--min-example-fraction", type=float, default=0.9)
    parser.add_argument("--min-query-fraction", type=float, default=0.9)
    parser.add_argument("--min-family-fraction", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require(path: Path, missing: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))


def _matrix_tasks(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as input_file:
        header = next(csv.reader(input_file))
    return header[1:]


def _read_json(path: Path, invalid: list[str]) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        invalid.append(f"{path}: invalid JSON ({exc})")
        return {}
    if not isinstance(payload, dict):
        invalid.append(f"{path}: expected a JSON object")
        return {}
    return payload


def _check_close(
    actual: object,
    expected: float,
    *,
    label: str,
    invalid: list[str],
) -> None:
    try:
        matches = abs(float(actual) - expected) <= 1e-12
    except (TypeError, ValueError):
        matches = False
    if not matches:
        invalid.append(f"{label}: found {actual!r}, expected {expected}")


def main() -> None:
    args = parse_args()
    model_root = args.result_root / args.model_name
    discovery_model_root = (
        args.discovery_root or default_discovery_root(args.result_root)
    ) / args.model_name
    missing: list[str] = []
    checked: list[str] = []

    dataset_report = args.result_root / "preflight" / "datasets.json"
    _require(dataset_report, missing)
    checked.append(str(dataset_report))
    if dataset_report.is_file():
        report = _read_json(dataset_report, missing)
        if report.get("status") != "passed":
            missing.append(f"{dataset_report}: status is not passed")
        if report.get("total_examples") != args.expected_examples * len(TASKS):
            missing.append(
                f"{dataset_report}: total_examples={report.get('total_examples')}, "
                f"expected {args.expected_examples * len(TASKS)}"
            )

    for task in TASKS:
        summary = discovery_model_root / task / "summary.json"
        _require(summary, missing)
        checked.append(str(summary))
        summary_payload = _read_json(summary, missing) if summary.is_file() else {}
        feature_sets = summary_payload.get("feature_sets") or {}
        if set(feature_sets) != set(args.sae_uid):
            missing.append(
                f"{summary}: SAE UIDs do not match the expected three configurations"
            )
        for uid, feature_entry in feature_sets.items():
            thresholds = feature_entry.get("selection_thresholds") or {}
            _check_close(
                thresholds.get("min_example_fraction"),
                args.min_example_fraction,
                label=f"{summary}:{uid}:min_example_fraction",
                invalid=missing,
            )
            _check_close(
                thresholds.get("min_query_fraction_per_context"),
                args.min_query_fraction,
                label=f"{summary}:{uid}:min_query_fraction_per_context",
                invalid=missing,
            )
            _check_close(
                thresholds.get("min_context_fraction"),
                args.min_family_fraction,
                label=f"{summary}:{uid}:min_context_fraction",
                invalid=missing,
            )
        for uid in args.sae_uid:
            task_root = model_root / uid / task
            ablation_root = task_root / "ablation" / args.feature_set
            for filename in (
                "ablation_summary.json",
                "selected_ablation_table.csv",
                "random_ablation_table.csv",
                "random_ablation_aggregate.csv",
                "selected_outcomes.csv.gz",
                "random_outcomes.csv.gz",
            ):
                path = ablation_root / filename
                _require(path, missing)
                checked.append(str(path))
            ablation_summary = ablation_root / "ablation_summary.json"
            if ablation_summary.is_file():
                ablation = _read_json(ablation_summary, missing)
                analysis_filter = ablation.get("analysis_filter") or {}
                baseline = (ablation.get("results") or {}).get("baseline") or {}
                random_summary = (ablation.get("results") or {}).get(
                    "random_control_summary"
                ) or {}
                if analysis_filter.get("baseline_correct_only"):
                    if (
                        analysis_filter.get("input_example_count")
                        != args.expected_examples
                    ):
                        missing.append(
                            f"{ablation_summary}: input example count is "
                            f"{analysis_filter.get('input_example_count')}, expected "
                            f"{args.expected_examples}"
                        )
                    expected_baseline_total = analysis_filter.get(
                        "analysis_example_count"
                    )
                else:
                    expected_baseline_total = args.expected_examples
                if baseline.get("total") != expected_baseline_total:
                    missing.append(
                        f"{ablation_summary}: baseline total is "
                        f"{baseline.get('total')}, expected "
                        f"{expected_baseline_total}"
                    )
                _check_close(
                    baseline.get("accuracy"),
                    1.0,
                    label=f"{ablation_summary}:baseline_accuracy",
                    invalid=missing,
                )
                if random_summary.get("trials") != args.expected_random_trials:
                    missing.append(
                        f"{ablation_summary}: random trials are "
                        f"{random_summary.get('trials')}, expected "
                        f"{args.expected_random_trials}"
                    )
                for field in (
                    "mean_accuracy",
                    "sample_std_accuracy",
                    "mean_accuracy_drop",
                    "sample_std_accuracy_drop",
                    "aggregate_accuracy_significance",
                ):
                    if random_summary.get(field) is None:
                        missing.append(
                            f"{ablation_summary}: random summary missing {field}"
                        )
                aggregate_sig = (
                    random_summary.get("aggregate_accuracy_significance") or {}
                )
                if aggregate_sig.get("p_value_one_sided") is None:
                    missing.append(
                        f"{ablation_summary}: aggregate random p-value is missing"
                    )
            geometry_root = (
                task_root
                / "fega"
                / args.model_name
                / f"{task}.json"
                / f"{task}_pointer_like"
                / "geometry_reporting"
            )
            for filename in (
                "geometry_feature_records.json",
                "geometry_feature_records.csv",
                "geometry_reporting_counts.csv",
                "geometry_map_data.json",
            ):
                path = geometry_root / filename
                _require(path, missing)
                checked.append(str(path))
            map_path = geometry_root / "geometry_map_data.json"
            if map_path.is_file():
                geometry_map = _read_json(map_path, missing)
                features = geometry_map.get("features")
                if isinstance(features, list) and features:
                    atlas_path = geometry_root / "figures/geometry_atlas.png"
                    _require(atlas_path, missing)
                    checked.append(str(atlas_path))

    expected_threshold_tasks = ["lsc", "wc", "prontoqa", "tt"]
    for uid in args.sae_uid:
        cross_root = model_root / uid / "cross_task"
        threshold_matrix = cross_root / "iou" / "threshold_iou_matrix.csv"
        strict_matrix = cross_root / "iou" / "strict_iou_matrix.csv"
        _require(threshold_matrix, missing)
        _require(strict_matrix, missing)
        checked.extend([str(threshold_matrix), str(strict_matrix)])
        if (
            threshold_matrix.is_file()
            and _matrix_tasks(threshold_matrix) != expected_threshold_tasks
        ):
            missing.append(
                f"{threshold_matrix} has tasks {_matrix_tasks(threshold_matrix)}, "
                f"expected {expected_threshold_tasks}"
            )
        if strict_matrix.is_file() and _matrix_tasks(strict_matrix) != [
            "lsc",
            "wc",
            "prontoqa",
            "tt",
        ]:
            missing.append(f"{strict_matrix} is not the expected four-task matrix")
        for plot_root in (
            cross_root / "geometry_plots",
            cross_root / "pointer_only_geometry_plots",
        ):
            for filename in (
                "geometry_all_tasks.png",
                "geometry_all_tasks.pdf",
                "geometry_category_counts.csv",
                "geometry_plot_data.csv",
            ):
                path = plot_root / filename
                _require(path, missing)
                checked.append(str(path))

    for filename in (
        "feature_counts.csv",
        "ablation_results.csv",
        "random_control_results.csv",
        "geometry_category_counts.csv",
        "manifest.json",
    ):
        path = args.result_root / "tables" / filename
        _require(path, missing)
        checked.append(str(path))

    payload = {
        "schema_version": 1,
        "status": "failed" if missing else "passed",
        "result_root": str(args.result_root.resolve()),
        "model_name": args.model_name,
        "sae_uids": args.sae_uid,
        "tasks": list(TASKS),
        "expected_examples_per_task": args.expected_examples,
        "expected_random_trials": args.expected_random_trials,
        "selection_thresholds": {
            "min_example_fraction": args.min_example_fraction,
            "min_query_fraction": args.min_query_fraction,
            "min_family_fraction": args.min_family_fraction,
        },
        "checked_artifact_count": len(checked),
        "missing_or_invalid": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if missing:
        rendered = "\n".join(f"- {value}" for value in missing)
        raise SystemExit(f"Paper artifact audit failed:\n{rendered}")
    print(f"Paper artifact audit passed: {args.output}")


if __name__ == "__main__":
    main()
