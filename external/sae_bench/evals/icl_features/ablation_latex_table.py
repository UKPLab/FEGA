from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.artifact_naming import artifact_tag, tagged_paths

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a LaTeX table summarizing ICL feature-ablation results."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/sae_geometry_gemma2b_65k"),
    )
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument(
        "--sae-width",
        default="2pow16",
        help="Substring used to select SAE UIDs, e.g. 2pow16.",
    )
    parser.add_argument(
        "--feature-set",
        default="threshold",
        help="Ablation feature-set directory to read, e.g. threshold.",
    )
    parser.add_argument(
        "--task-order",
        nargs="+",
        default=list(DEFAULT_TASK_ORDER),
        choices=list(TASK_DISPLAY),
        help="Task row order inside each SAE block.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the .txt file that will contain LaTeX table code.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def sae_variant(sae_uid: str) -> str | None:
    lowered = sae_uid.lower()
    if "matryoshka" in lowered:
        return "matryoshka"
    if "top_k" in lowered or "topk" in lowered:
        return "topk"
    if "standard" in lowered or "relu" in lowered:
        return "relu"
    return None


def collect_ablation_summaries(
    result_root: Path,
    model_name: str,
    *,
    sae_width: str,
    feature_set: str,
    tasks: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    model_root = result_root / model_name
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(model_root.rglob(f"ablation/{feature_set}/ablation_summary.json")):
        payload = load_json(path)
        if payload.get("model_name") != model_name:
            continue
        task = str(payload.get("task"))
        if task not in tasks:
            continue
        sae_uid = str(payload.get("sae_uid", ""))
        if sae_width and sae_width not in sae_uid:
            continue
        variant = sae_variant(sae_uid)
        if variant is None:
            continue
        key = (variant, task)
        if key in summaries:
            previous = summaries[key].get("_summary_path")
            raise ValueError(
                f"Multiple ablation summaries match {variant}/{task}: "
                f"{previous} and {path}"
            )
        payload["_summary_path"] = str(path)
        summaries[key] = payload
    missing = [
        f"{label}/{TASK_DISPLAY[task]}"
        for variant, label in SAE_BLOCKS
        for task in tasks
        if (variant, task) not in summaries
    ]
    if missing:
        raise FileNotFoundError(
            "Missing ablation summaries for: " + ", ".join(missing)
        )
    return summaries


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def percent(value: Any, digits: int = 1) -> str:
    numeric = _as_float(value)
    if numeric is None or not math.isfinite(numeric):
        return "--"
    return f"{100.0 * numeric:.{digits}f}"


def integer(value: Any) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def p_value(value: Any, *, log10_value: Any | None = None) -> str:
    numeric = _as_float(value)
    log10_numeric = _as_float(log10_value)
    if numeric is None or not math.isfinite(numeric):
        return "--"
    if numeric == 0.0:
        if log10_numeric is not None and math.isfinite(log10_numeric):
            exponent = max(1, min(300, int(math.floor(-log10_numeric))))
            return f"$<10^{{-{exponent}}}$"
        return "$<10^{-300}$"
    if numeric < 1.0e-3:
        exponent = int(math.floor(math.log10(numeric)))
        mantissa = numeric / (10.0**exponent)
        return f"${mantissa:.1f}\\times10^{{{exponent}}}$"
    return f"{numeric:.3f}"


def random_accuracy_cell(summary: dict[str, Any]) -> str:
    mean = summary.get("mean_accuracy")
    std = summary.get("sample_std_accuracy")
    if mean is None:
        return "--"
    if std is None:
        return percent(mean)
    return f"{percent(mean)}$\\pm${percent(std)}"


def ablation_row(payload: dict[str, Any]) -> dict[str, str]:
    selected = payload["results"]["selected_ablation"]
    selected_sig = selected["drop_significance"]
    random_summary = payload["results"]["random_control_summary"]
    random_test = random_summary.get("aggregate_accuracy_significance") or {}
    return {
        "task": TASK_DISPLAY[str(payload["task"])],
        "k": integer(payload.get("selected_feature_count")),
        "target_acc": percent(selected.get("accuracy")),
        "target_p": p_value(
            selected_sig.get("p_value_one_sided"),
            log10_value=selected_sig.get("log10_p_value_one_sided"),
        ),
        "random_acc": random_accuracy_cell(random_summary),
        "random_p": p_value(random_test.get("p_value_one_sided")),
    }


def latex_table(
    summaries: dict[tuple[str, str], dict[str, Any]],
    *,
    model_name: str,
    sae_width: str,
    feature_set: str,
    tasks: list[str],
) -> str:
    rows = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{6pt}",
        "    \\begin{tabular}{llrcccc}",
        "        \\toprule",
        (
            "        \\textbf{SAE} & \\textbf{Task} & \\textbf{$k$} & "
            "\\textbf{Target Acc.} & \\textbf{$p_{\\mathrm{tar}}$} & "
            "\\textbf{Random Acc.} & \\textbf{$p_{\\mathrm{rand}}$} \\\\"
        ),
        "        \\midrule",
    ]
    for block_index, (variant, label) in enumerate(SAE_BLOCKS):
        if block_index:
            rows.append("        \\midrule")
        for task_index, task in enumerate(tasks):
            row = ablation_row(summaries[(variant, task)])
            sae_cell = f"\\textit{{{label}}}" if task_index == 0 else ""
            rows.append(
                "        "
                f"{sae_cell} & {row['task']} & {row['k']} & "
                f"{row['target_acc']} & {row['target_p']} & "
                f"{row['random_acc']} & {row['random_p']} \\\\"
            )
    rows.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\vspace{2mm}",
            (
                "    \\caption{\\textbf{Causal effect of isolated ICL features.} "
                "All accuracies are evaluated on examples answered correctly by "
                "the unablated model, so the baseline accuracy is 100\\% for every "
                "row. Targeted ablations zero the discovered "
                f"{feature_set} feature set; random controls zero matched feature "
                "sets of the same size and report mean$\\pm$standard deviation "
                "across trials. $p_{\\mathrm{tar}}$ is the one-sided exact "
                "paired McNemar test for targeted ablation against the baseline, "
                "and $p_{\\mathrm{rand}}$ is the one-sided aggregate test for "
                "random ablations against the baseline. Results are for "
                f"{model_name} SAEs at width {sae_width}.}}"
            ),
            "    \\label{tab:icl_ablation_results}",
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
    summaries = collect_ablation_summaries(
        args.result_root,
        args.model_name,
        sae_width=args.sae_width,
        feature_set=args.feature_set,
        tasks=tasks,
    )
    latex = latex_table(
        summaries,
        model_name=args.model_name,
        sae_width=args.sae_width,
        feature_set=args.feature_set,
        tasks=tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        path.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX ablation table: {args.output}")


if __name__ == "__main__":
    main()
