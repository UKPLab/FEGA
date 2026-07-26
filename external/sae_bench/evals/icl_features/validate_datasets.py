from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sae_bench.evals.icl_features.schema import validate_dataset


TASKS = ("lsc", "wc", "tt", "prontoqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all curated ICL datasets before a full experiment run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/icl_features"))
    parser.add_argument("--model-name", default="gemma-2-2b")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--expected-examples", type=int, default=50_000)
    parser.add_argument("--expected-families", type=int, default=1_000)
    parser.add_argument("--max-family-size-difference", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--recheck-model-correct",
        action="store_true",
        help="Load Gemma and independently rescore every retained example.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--answer-prefix", default=" ")
    return parser.parse_args()


def _validate_one(
    *,
    path: Path,
    task: str,
    model_name: str,
    expected_examples: int,
    expected_families: int,
    max_family_size_difference: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {task} dataset: {path}. Generate it before the full experiment run."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(payload, require_model_correct=True)
    metadata = payload["metadata"]
    examples = payload["examples"]
    if str(metadata.get("task_name")) != task:
        raise ValueError(
            f"{path}: metadata.task_name={metadata.get('task_name')!r}, expected {task!r}"
        )
    if len(examples) != expected_examples:
        raise ValueError(
            f"{path}: found {len(examples)} examples, expected {expected_examples}"
        )
    family_counts = Counter(str(row["context_id"]) for row in examples)
    if len(family_counts) != expected_families:
        raise ValueError(
            f"{path}: found {len(family_counts)} prompt families, "
            f"expected {expected_families}"
        )
    family_difference = max(family_counts.values()) - min(family_counts.values())
    if family_difference > max_family_size_difference:
        raise ValueError(
            f"{path}: prompt-family size difference is {family_difference}, "
            f"maximum is {max_family_size_difference}"
        )
    model_filter = metadata.get("model_filter") or {}
    if str(model_filter.get("model_name")) != model_name:
        raise ValueError(
            f"{path}: generated for model {model_filter.get('model_name')!r}, "
            f"expected {model_name!r}"
        )
    if not bool(model_filter.get("all_retained_examples_model_correct")):
        raise ValueError(f"{path}: model-correctness metadata is not true")
    if any(
        int(row["target_first_token_id"])
        != int(row["predicted_first_token_id"])
        for row in examples
    ):
        raise ValueError(f"{path}: stored target and prediction IDs disagree")
    report = {
        "task": task,
        "path": str(path.resolve()),
        "examples": len(examples),
        "prompt_families": len(family_counts),
        "queries_per_family_min": min(family_counts.values()),
        "queries_per_family_max": max(family_counts.values()),
        "model_name": model_filter.get("model_name"),
        "resolved_model_name": model_filter.get("resolved_model_name"),
        "stored_model_correct_rows": sum(
            bool(row["model_correct_first_token"]) for row in examples
        ),
        "independent_model_recheck": None,
    }
    return report, examples


def _recheck(
    *,
    reports_and_examples: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    model_name: str,
    device: str,
    dtype: str,
    batch_size: int,
    answer_prefix: str,
) -> None:
    from sae_bench.evals.icl_features.model_filter import GemmaFirstTokenFilter

    scorer = GemmaFirstTokenFilter(
        model_name=model_name,
        device=device,
        dtype_name=dtype,
        batch_size=batch_size,
        answer_prefix=answer_prefix,
    )
    for report, examples in reports_and_examples:
        results = scorer.score(examples)
        incorrect = [
            str(row["example_id"])
            for row, result in zip(examples, results, strict=True)
            if not result.correct
        ]
        report["independent_model_recheck"] = {
            "checked": len(results),
            "correct": len(results) - len(incorrect),
            "incorrect": len(incorrect),
        }
        if incorrect:
            raise ValueError(
                f"{report['task']}: independent model recheck failed for "
                f"{len(incorrect)} examples; first IDs: {incorrect[:10]}"
            )


def main() -> None:
    args = parse_args()
    model_root = args.data_root / args.model_name
    checked = [
        _validate_one(
            path=model_root / f"{task}.json",
            task=task,
            model_name=args.model_name,
            expected_examples=args.expected_examples,
            expected_families=args.expected_families,
            max_family_size_difference=args.max_family_size_difference,
        )
        for task in args.tasks
    ]
    if args.recheck_model_correct:
        _recheck(
            reports_and_examples=checked,
            model_name=args.model_name,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            answer_prefix=args.answer_prefix,
        )
    payload = {
        "schema_version": 1,
        "status": "passed",
        "model_name": args.model_name,
        "expected_examples_per_task": args.expected_examples,
        "expected_prompt_families_per_task": args.expected_families,
        "independent_model_recheck_requested": args.recheck_model_correct,
        "datasets": [report for report, _ in checked],
        "total_examples": sum(report["examples"] for report, _ in checked),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
