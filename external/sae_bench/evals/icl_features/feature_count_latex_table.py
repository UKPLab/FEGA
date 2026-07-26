from __future__ import annotations

import argparse
import json
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
    "ravel": "RAVEL (City-Country)",
}
TASK_REGIME = {
    "lsc": "Pointer-Like",
    "wc": "Pointer-Like",
    "prontoqa": "Pointer-Like",
    "tt": "Hybrid",
    "ravel": "Value-Like",
}
DEFAULT_ROW_ORDER = ("lsc", "wc", "prontoqa", "tt", "ravel")
SAE_COLUMNS = (
    ("relu", "ReLU SAE"),
    ("topk", "TopK SAE"),
    ("matryoshka", "MatryoshkaBatchTopK SAE"),
)
RAVEL_FILE_BY_VARIANT = {
    "relu": "standard",
    "topk": "topk",
    "matryoshka": "matryoshka",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a LaTeX table of isolated feature counts."
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
        "--feature-set",
        choices=["candidate", "threshold", "strict_common"],
        default="candidate",
        help="ICL feature-set count to report.",
    )
    parser.add_argument(
        "--row-order",
        nargs="+",
        default=list(DEFAULT_ROW_ORDER),
        choices=list(TASK_DISPLAY),
        help="Rows in the output table.",
    )
    parser.add_argument(
        "--ravel-kept-threshold",
        type=float,
        default=8.0,
        help="Threshold applied to RAVEL feature_stats[*].kept.",
    )
    parser.add_argument(
        "--ravel-kept-op",
        choices=["gt", "ge"],
        default="ge",
        help="Use kept > threshold (gt) or kept >= threshold (ge). Draft table uses ge.",
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


def count_field(feature_set: str) -> str:
    if feature_set in {"candidate", "threshold"}:
        return f"{feature_set}_feature_count"
    return "strict_common_feature_count"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def select_sae_counts(
    feature_sets: dict[str, Any], *, sae_width: str, feature_set: str
) -> dict[str, int]:
    field = count_field(feature_set)
    counts: dict[str, int] = {}
    for sae_uid, entry in feature_sets.items():
        if sae_width and sae_width not in sae_uid:
            continue
        variant = sae_variant(sae_uid)
        if variant is None:
            continue
        if variant in counts:
            raise ValueError(
                f"Multiple {variant} SAE entries match width {sae_width!r}"
            )
        value = entry.get(field)
        if value is None and field == "threshold_feature_count":
            value = entry.get("candidate_feature_count")
        if value is None:
            raise ValueError(f"{sae_uid}: missing {field}")
        counts[variant] = int(value)
    missing = [variant for variant, _ in SAE_COLUMNS if variant not in counts]
    if missing:
        raise ValueError(
            f"Missing SAE variants for width {sae_width!r}: {', '.join(missing)}"
        )
    return counts


def load_icl_counts(
    discovery_model_root: Path,
    model_name: str,
    *,
    sae_width: str,
    feature_set: str,
    tasks: list[str],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for task in tasks:
        if task == "ravel":
            continue
        summary_path = discovery_model_root / task / "summary.json"
        summary = load_json(summary_path)
        feature_sets = summary.get("feature_sets")
        if not isinstance(feature_sets, dict):
            raise ValueError(f"{summary_path}: missing feature_sets")
        output[task] = select_sae_counts(
            feature_sets, sae_width=sae_width, feature_set=feature_set
        )
    return output


def ravel_count(path: Path, *, threshold: float, op: str) -> int:
    payload = load_json(path)
    stats = payload.get("feature_stats")
    if not isinstance(stats, dict):
        raise ValueError(f"{path}: missing feature_stats")
    total = 0
    for record in stats.values():
        if not isinstance(record, dict):
            continue
        kept = float(record.get("kept", 0.0))
        keep = kept >= threshold if op == "ge" else kept > threshold
        if keep:
            total += 1
    return total


def load_ravel_counts(
    result_root: Path,
    model_name: str,
    *,
    threshold: float,
    op: str,
) -> dict[str, int]:
    ravel_dir = result_root / model_name / "ravel_feature_ids"
    return {
        variant: ravel_count(
            ravel_dir / filename,
            threshold=threshold,
            op=op,
        )
        for variant, filename in RAVEL_FILE_BY_VARIANT.items()
    }


def fmt_count(value: int) -> str:
    return f"{value:,}"


def latex_table(
    counts: dict[str, dict[str, int]],
    *,
    row_order: list[str],
    model_name: str,
    sae_width: str,
    feature_set: str,
) -> str:
    column_headers = " & ".join(f"\\textbf{{{label}}}" for _, label in SAE_COLUMNS)
    rows = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\begin{tabular}{lcccc}",
        "        \\toprule",
        (
            "        \\textbf{Feature Source} & \\textbf{Regime} & "
            f"{column_headers} \\\\"
        ),
        "        \\midrule",
    ]
    for task in row_order:
        task_counts = counts[task]
        value_cells = " & ".join(
            fmt_count(task_counts[variant]) for variant, _ in SAE_COLUMNS
        )
        rows.append(
            f"        {TASK_DISPLAY[task]} & {TASK_REGIME[task]} & {value_cells} \\\\"
        )
    rows.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\vspace{2mm}",
            (
                "    \\caption{\\textbf{Isolated Feature Counts.} The number of "
                "features successfully identified as value-like (via RAVEL "
                "attribute disentanglement) and pointer-like (via ICL task "
                f"recurrence) for {model_name} SAEs at width {sae_width}. "
                f"ICL counts use the {feature_set} feature set.}}"
            ),
            "    \\label{tab:feature_counts}",
            "\\end{table}",
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
    row_order = [str(task) for task in args.row_order]
    counts = load_icl_counts(
        (
            (args.discovery_root or default_discovery_root(args.result_root))
            / args.model_name
        ),
        args.model_name,
        sae_width=args.sae_width,
        feature_set=args.feature_set,
        tasks=row_order,
    )
    if "ravel" in row_order:
        counts["ravel"] = load_ravel_counts(
            args.result_root,
            args.model_name,
            threshold=args.ravel_kept_threshold,
            op=args.ravel_kept_op,
        )
    latex = latex_table(
        counts,
        row_order=row_order,
        model_name=args.model_name,
        sae_width=args.sae_width,
        feature_set=args.feature_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        path.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX feature-count table: {args.output}")


if __name__ == "__main__":
    main()
