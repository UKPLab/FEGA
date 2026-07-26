from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import (
    artifact_tag,
    default_discovery_root,
    tagged_paths,
)

TASK_DISPLAY = {
    "lsc": "LSC",
    "wc": "WC",
    "prontoqa": "PrOntoQA",
    "tt": "TT",
}
DEFAULT_TASK_ORDER = ("lsc", "wc", "prontoqa", "tt")
SAE_BLOCKS = (
    ("relu", "ReLU"),
    ("topk", "TopK"),
    ("matryoshka", "Matryoshka Batch TopK"),
)

GEOMETRY_COLUMNS = (
    ("map", "Map"),
    ("excluded", "Excl."),
    ("unresolved_high_dimensional_or_diffuse", "Unres."),
    ("global_kD_directional_subspace", "G-$k$D"),
    ("residual_lowD_k", "Res.-lowD"),
    ("oneD_diffuse", "1D diff."),
    ("directed_ray", "Ray"),
    ("global_2D_directional_subspace", "G-2D"),
    ("axis_or_antipodal", "Axis"),
    ("multi_mode_directional_geometry", "Multi"),
)
PRIMARY_LABELS = {
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
    "directed_ray",
    "axis_or_antipodal",
    "oneD_diffuse",
    "multi_mode_directional_geometry",
    "global_2D_directional_subspace",
    "global_kD_directional_subspace",
    "residual_lowD_k",
    "unresolved_high_dimensional_or_diffuse",
}
EXCLUDED_LABELS = {
    "insufficient_effect_evidence",
    "geometry_metrics_unavailable",
    "undefined_geometry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a LaTeX table of primary FEGA geometry-label counts."
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
        "--sae-width",
        default="2pow16",
        help="Substring used to select SAE UIDs, e.g. 2pow16.",
    )
    parser.add_argument(
        "--geometry-feature-set",
        default="candidate",
        help="Used only for the caption; FEGA paths are discovered from result files.",
    )
    parser.add_argument(
        "--task-order",
        nargs="+",
        default=list(DEFAULT_TASK_ORDER),
        choices=list(TASK_DISPLAY),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the .txt file that will contain LaTeX table code.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sae_variant(sae_uid: str) -> str | None:
    lowered = sae_uid.lower()
    if "matryoshka" in lowered:
        return "matryoshka"
    if "top_k" in lowered or "topk" in lowered:
        return "topk"
    if "standard" in lowered or "relu" in lowered:
        return "relu"
    return None


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _zero_counts() -> dict[str, int]:
    return {name: 0 for name, _ in GEOMETRY_COLUMNS}


def _finalize_counts(primary_counts: Counter[str]) -> dict[str, int]:
    output = _zero_counts()
    excluded = 0
    mapped = 0
    for label, count in primary_counts.items():
        if label in EXCLUDED_LABELS:
            excluded += int(count)
        elif label in PRIMARY_LABELS:
            mapped += int(count)
        if label in output:
            output[label] += int(count)
    output["excluded"] = excluded
    output["map"] = mapped
    return output


def _counts_from_fega_csv(path: Path) -> dict[str, int]:
    primary = Counter()
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if row.get("kind") == "primary_label":
                primary[str(row["name"])] += int(float(row["count"]))
    return _finalize_counts(primary)


def _counts_from_geometry_records(path: Path) -> dict[str, int]:
    payload = load_json(path)
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(
        summary.get("primary_label_counts"), dict
    ):
        return _finalize_counts(Counter(summary["primary_label_counts"]))
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{path}: missing features list")
    return _finalize_counts(
        Counter(str(feature.get("primary_label")) for feature in features)
    )


def _counts_from_geometry_map(path: Path) -> dict[str, int]:
    payload = load_json(path)
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"{path}: missing features list")
    return _finalize_counts(
        Counter(
            str(feature.get("primary_label") or feature.get("atlas_label"))
            for feature in features
        )
    )


def _counts_from_plot_csv(path: Path, task: str) -> dict[str, int]:
    primary = Counter()
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            if str(row.get("task")) == task:
                primary[str(row.get("primary_label") or row.get("atlas_label"))] += 1
    return _finalize_counts(primary)


def _task_source_candidates(
    result_root: Path, model_name: str, sae_uid: str, task: str
) -> list[tuple[str, Path]]:
    task_root = result_root / model_name / sae_uid / task
    fega_base = (
        task_root / "fega" / model_name / f"{task}.json" / f"{task}_pointer_like"
    )
    return [
        ("fega_counts", fega_base / "geometry_reporting" / "geometry_reporting_counts.csv"),
        ("fega_records", fega_base / "geometry_reporting" / "geometry_feature_records.json"),
        ("fega_map", fega_base / "geometry_reporting" / "geometry_map_data.json"),
        (
            "cross_task_plot",
            result_root
            / model_name
            / sae_uid
            / "cross_task"
            / "geometry_plots"
            / "geometry_plot_data.csv",
        ),
        (
            "pointer_plot",
            result_root
            / model_name
            / sae_uid
            / "cross_task"
            / "pointer_only_geometry_plots"
            / "geometry_plot_data.csv",
        ),
    ]


def load_task_counts(
    result_root: Path,
    discovery_model_root: Path,
    model_name: str,
    sae_uid: str,
    task: str,
    *,
    geometry_feature_set: str,
) -> tuple[dict[str, int], str]:
    for source_kind, path in _task_source_candidates(
        result_root, model_name, sae_uid, task
    ):
        if not path.is_file():
            continue
        if source_kind == "fega_counts":
            return _counts_from_fega_csv(path), str(path)
        if source_kind == "fega_records":
            return _counts_from_geometry_records(path), str(path)
        if source_kind == "fega_map":
            return _counts_from_geometry_map(path), str(path)
        return _counts_from_plot_csv(path, task), str(path)
    if (
        discovered_feature_count(
            discovery_model_root,
            sae_uid,
            task,
            geometry_feature_set=geometry_feature_set,
        )
        == 0
    ):
        return _zero_counts(), "discovery_zero_feature_set"
    raise FileNotFoundError(
        f"No geometry count source found for {sae_uid}/{task}. "
        "Expected geometry_reporting_counts.csv, geometry_feature_records.json, "
        "geometry_map_data.json, or geometry_plot_data.csv."
    )


def discovered_feature_count(
    discovery_model_root: Path,
    sae_uid: str,
    task: str,
    *,
    geometry_feature_set: str,
) -> int | None:
    field = (
        "strict_common_feature_count"
        if geometry_feature_set == "strict_common"
        else f"{geometry_feature_set}_feature_count"
    )
    summary_path = discovery_model_root / task / "summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    feature_sets = summary.get("feature_sets")
    if not isinstance(feature_sets, dict):
        return None
    record = feature_sets.get(sae_uid)
    if not isinstance(record, dict):
        return None
    value = record.get(field)
    if value is None and field == "threshold_feature_count":
        value = record.get("candidate_feature_count")
    if value is None:
        return None
    return int(value)


def discover_sae_uids(
    result_root: Path, model_name: str, *, sae_width: str
) -> dict[str, str]:
    model_root = result_root / model_name
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    selected: dict[str, str] = {}
    for path in sorted(model_root.iterdir()):
        if not path.is_dir():
            continue
        if sae_width and sae_width not in path.name:
            continue
        variant = sae_variant(path.name)
        if variant is None:
            continue
        if variant in selected:
            raise ValueError(
                f"Multiple {variant} SAE directories match width {sae_width!r}: "
                f"{selected[variant]} and {path.name}"
            )
        selected[variant] = path.name
    missing = [label for variant, label in SAE_BLOCKS if variant not in selected]
    if missing:
        raise FileNotFoundError(
            "Missing SAE result directories for: " + ", ".join(missing)
        )
    return selected


def collect_counts(
    result_root: Path,
    discovery_model_root: Path,
    model_name: str,
    *,
    sae_width: str,
    geometry_feature_set: str,
    tasks: list[str],
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[tuple[str, str], str]]:
    sae_uids = discover_sae_uids(result_root, model_name, sae_width=sae_width)
    counts: dict[tuple[str, str], dict[str, int]] = {}
    sources: dict[tuple[str, str], str] = {}
    for variant, _ in SAE_BLOCKS:
        sae_uid = sae_uids[variant]
        for task in tasks:
            task_counts, source = load_task_counts(
                result_root,
                discovery_model_root,
                model_name,
                sae_uid,
                task,
                geometry_feature_set=geometry_feature_set,
            )
            counts[(variant, task)] = task_counts
            sources[(variant, task)] = source
    return counts, sources


def width_display(sae_width: str) -> str:
    if sae_width == "2pow16":
        return "$2^{16}$"
    if sae_width.startswith("2pow"):
        return f"$2^{{{sae_width.removeprefix('2pow')}}}$"
    return sae_width.replace("_", "\\_")


def latex_table(
    counts: dict[tuple[str, str], dict[str, int]],
    *,
    model_name: str,
    sae_width: str,
    geometry_feature_set: str,
    tasks: list[str],
) -> str:
    headers = " & ".join(header for _, header in GEOMETRY_COLUMNS)
    column_spec = "ll" + "r" * len(GEOMETRY_COLUMNS)
    rows = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{3.5pt}",
        f"    \\begin{{tabular}}{{{column_spec}}}",
        "        \\toprule",
        f"        SAE & Task & {headers} \\\\",
        "        \\midrule",
    ]
    width = width_display(sae_width)
    for block_index, (variant, label) in enumerate(SAE_BLOCKS):
        if block_index:
            rows.append("        \\midrule")
        for task_index, task in enumerate(tasks):
            task_counts = counts[(variant, task)]
            sae_cell = f"\\textit{{{label} {width}}}" if task_index == 0 else ""
            value_cells = " & ".join(
                str(task_counts[column]) for column, _ in GEOMETRY_COLUMNS
            )
            rows.append(
                f"        {sae_cell} & {TASK_DISPLAY[task]} & {value_cells} \\\\"
            )
    rows.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\vspace{2mm}",
            (
                "    \\caption{\\textbf{Primary FEGA labels for ICL features.} "
                "Counts show primary FEGA geometry categories for the discovered "
                f"{geometry_feature_set} feature sets across ICL tasks. "
                "\\emph{Map} counts features assigned to a mapped geometry class; "
                "\\emph{Excl.} aggregates insufficient-effect, unavailable, and "
                "undefined geometry labels. Results are for "
                f"{model_name} SAEs at width {sae_width}.}}"
            ),
            "    \\label{tab:algorithmic-geometry-counts}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    tag = artifact_tag(
        model_name=args.model_name,
        sae_width=args.sae_width,
        aggregate=True,
    )
    output_paths = tagged_paths(args.output, tag)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing file(s): "
            + ", ".join(str(path) for path in existing)
            + ". "
            "Pass --overwrite to replace it."
        )
    tasks = [str(task) for task in args.task_order]
    counts, _sources = collect_counts(
        args.result_root,
        (
            (args.discovery_root or default_discovery_root(args.result_root))
            / args.model_name
        ),
        args.model_name,
        sae_width=args.sae_width,
        geometry_feature_set=args.geometry_feature_set,
        tasks=tasks,
    )
    latex = latex_table(
        counts,
        model_name=args.model_name,
        sae_width=args.sae_width,
        geometry_feature_set=args.geometry_feature_set,
        tasks=tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        path.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX geometry-count table: {args.output}")


if __name__ == "__main__":
    main()
