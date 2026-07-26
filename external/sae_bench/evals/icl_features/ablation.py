from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.evals.icl_features.artifact_naming import artifact_tag, tagged_paths
from sae_bench.evals.icl_features.feature_sets import (
    load_discovery_summary,
    matched_random_feature_sets,
    per_sae_metrics_path,
    resolve_feature_set,
)
from sae_bench.evals.icl_features.statistics import (
    accuracy_summary,
    empirical_upper_tail,
    exact_mcnemar,
    one_sample_mean_degradation_test,
)
from sae_bench.evals.induction_features.main import (
    LoadedSae,
    SaeReference,
    add_target_token_columns,
    load_sae_reference,
    normalize_examples,
)
from sae_bench.sae_bench_utils.activation_collection import LLM_NAME_TO_DTYPE
from tqdm import tqdm
from transformer_lens import HookedTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure causal accuracy effects of threshold or strict ICL feature sets "
            "using error-preserving SAE zero ablation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--discovery-summary", type=Path, required=True)
    parser.add_argument("--sae-uid", required=True)
    parser.add_argument(
        "--feature-set",
        choices=["threshold", "strict"],
        default="threshold",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--download-saes-dir",
        type=Path,
        default=Path("data/downloaded_saes"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--answer-prefix", default=" ")
    parser.add_argument(
        "--ablation-position",
        choices=["final", "all"],
        default="final",
        help="Primary analysis should use final; all is a broader robustness intervention.",
    )
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-match-pool-size", type=int, default=100)
    parser.add_argument("--expected-examples", type=int, default=50_000)
    parser.add_argument(
        "--allow-imperfect-baseline",
        action="store_true",
        help=(
            "By default fail unless the independent baseline forward pass is 100% "
            "correct on the curated dataset."
        ),
    )
    parser.add_argument(
        "--baseline-correct-only",
        action="store_true",
        help=(
            "Restrict ablation summaries and outcome files to examples answered "
            "correctly by the unablated baseline forward pass."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _reference_from_summary(summary: dict[str, Any], sae_uid: str) -> SaeReference:
    entry = summary["feature_sets"][sae_uid]
    mode = str(summary.get("sae_selection_mode") or "custom_repo")
    return SaeReference(
        source_kind="custom_repo" if mode == "custom_repo" else "sae_lens",
        sae_release=str(entry["sae_release"]),
        sae_id=str(entry["sae_id"]),
        custom_location=str(entry["sae_id"]) if mode == "custom_repo" else None,
        sae_uid=sae_uid,
    )


def _validate_sae_for_intervention(loaded_sae: LoadedSae) -> None:
    hook_head_index = getattr(loaded_sae.sae.cfg, "hook_head_index", None)
    if hook_head_index is not None:
        raise ValueError(
            "ICL feature-set ablation currently supports full residual-stream SAEs, "
            "not single-head SAEs."
        )


def _ablation_hook(
    *,
    sae: Any,
    feature_ids: list[int],
    final_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    position_mode: str,
):
    feature_index = torch.as_tensor(feature_ids, device=sae.device, dtype=torch.long)

    def hook_fn(activations: torch.Tensor, hook: Any) -> torch.Tensor:
        del hook
        if activations.ndim != 3:
            raise ValueError(
                f"Expected residual activations [batch, seq, d_model], got "
                f"{tuple(activations.shape)}"
            )
        modified = activations.clone()
        if position_mode == "final":
            rows = torch.arange(activations.shape[0], device=activations.device)
            source = activations[rows, final_positions].to(
                device=sae.device, dtype=sae.dtype
            )
            encoded = sae.encode(source)
            reconstruction = sae.decode(encoded)
            error = source - reconstruction
            ablated = encoded.clone()
            if feature_index.numel():
                ablated[:, feature_index] = 0.0
            replacement = sae.decode(ablated) + error
            modified[rows, final_positions] = replacement.to(
                device=modified.device, dtype=modified.dtype
            )
            return modified

        valid_mask = attention_mask.to(device=activations.device, dtype=torch.bool)
        source = activations[valid_mask].to(device=sae.device, dtype=sae.dtype)
        encoded = sae.encode(source)
        reconstruction = sae.decode(encoded)
        error = source - reconstruction
        ablated = encoded.clone()
        if feature_index.numel():
            ablated[:, feature_index] = 0.0
        modified[valid_mask] = (sae.decode(ablated) + error).to(
            device=modified.device, dtype=modified.dtype
        )
        return modified

    return hook_fn


@torch.no_grad()
def evaluate_interventions(
    *,
    model: HookedTransformer,
    tokenizer: Any,
    loaded_sae: LoadedSae,
    examples_df: pd.DataFrame,
    conditions: list[tuple[str, list[int]]],
    batch_size: int,
    device: str,
    position_mode: str,
) -> dict[str, Any]:
    prompts = examples_df["prompt"].astype(str).tolist()
    target_ids = examples_df["target_first_token_id"].astype(int).tolist()
    predictions = {
        "baseline": [],
        **{condition_name: [] for condition_name, _ in conditions},
    }
    for start in tqdm(range(0, len(prompts), batch_size), desc="Ablation evaluation"):
        end = min(start + batch_size, len(prompts))
        encoded = tokenizer(
            prompts[start:end],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        final_positions = attention_mask.sum(dim=1) - 1
        row_indices = torch.arange(input_ids.shape[0], device=device)

        baseline_logits = model(
            input_ids,
            return_type="logits",
            prepend_bos=False,
            attention_mask=attention_mask,
        )
        baseline_predictions = (
            baseline_logits[row_indices, final_positions].argmax(dim=-1).cpu().tolist()
        )
        predictions["baseline"].extend(baseline_predictions)
        del baseline_logits

        for condition_name, feature_ids in conditions:
            if not feature_ids:
                predictions[condition_name].extend(baseline_predictions)
                continue
            hook_fn = _ablation_hook(
                sae=loaded_sae.sae,
                feature_ids=feature_ids,
                final_positions=final_positions,
                attention_mask=attention_mask,
                position_mode=position_mode,
            )
            logits = model.run_with_hooks(
                input_ids,
                return_type="logits",
                prepend_bos=False,
                attention_mask=attention_mask,
                fwd_hooks=[(loaded_sae.hook_name, hook_fn)],
            )
            predictions[condition_name].extend(
                logits[row_indices, final_positions].argmax(dim=-1).cpu().tolist()
            )
            del logits

        del input_ids, attention_mask, final_positions

    correct = {
        name: [
            int(predicted) == int(target)
            for predicted, target in zip(values, target_ids, strict=True)
        ]
        for name, values in predictions.items()
    }
    return {
        "target_ids": target_ids,
        "predictions": predictions,
        "correct": correct,
    }


def summarize_ablation(
    *,
    outcomes: dict[str, Any],
    selected_name: str,
    random_names: list[str],
) -> dict[str, Any]:
    correct = outcomes["correct"]
    baseline = correct["baseline"]
    selected = correct[selected_name]
    baseline_summary = accuracy_summary(baseline)
    selected_summary = accuracy_summary(selected)
    selected_drop = baseline_summary["accuracy"] - selected_summary["accuracy"]
    random_results = []
    random_drops = []
    random_accuracies = []
    for random_name in random_names:
        random_correct = correct[random_name]
        random_summary = accuracy_summary(random_correct)
        drop = baseline_summary["accuracy"] - random_summary["accuracy"]
        random_drops.append(drop)
        random_accuracies.append(random_summary["accuracy"])
        random_results.append(
            {
                "condition": random_name,
                **random_summary,
                "accuracy_drop": drop,
                "drop_significance": exact_mcnemar(baseline, random_correct),
                "selected_vs_random": exact_mcnemar(random_correct, selected),
            }
        )
    return {
        "baseline": baseline_summary,
        "selected_ablation": {
            **selected_summary,
            "accuracy_drop": selected_drop,
            "drop_significance": exact_mcnemar(baseline, selected),
        },
        "random_controls": random_results,
        "random_control_summary": {
            "trials": len(random_results),
            "mean_accuracy": (
                sum(random_accuracies) / len(random_accuracies)
                if random_accuracies
                else None
            ),
            "sample_std_accuracy": (
                float(pd.Series(random_accuracies).std(ddof=1))
                if len(random_accuracies) >= 2
                else None
            ),
            "mean_accuracy_drop": (
                sum(random_drops) / len(random_drops) if random_drops else None
            ),
            "sample_std_accuracy_drop": (
                float(pd.Series(random_drops).std(ddof=1))
                if len(random_drops) >= 2
                else None
            ),
            "max_accuracy_drop": max(random_drops) if random_drops else None,
            "aggregate_accuracy_significance": one_sample_mean_degradation_test(
                random_accuracies,
                reference_accuracy=baseline_summary["accuracy"],
            ),
            "empirical_p_value_random_drop_at_least_selected": (
                empirical_upper_tail(selected_drop, random_drops)
                if random_drops
                else None
            ),
        },
    }


def filter_outcomes_to_examples(
    *,
    examples_df: pd.DataFrame,
    outcomes: dict[str, Any],
    keep_mask: list[bool],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(keep_mask) != len(examples_df):
        raise ValueError("keep_mask length must match examples_df")
    kept_indices = [index for index, keep in enumerate(keep_mask) if keep]
    filtered_examples = examples_df.iloc[kept_indices].reset_index(drop=True).copy()
    filtered_outcomes = {
        "target_ids": [outcomes["target_ids"][index] for index in kept_indices],
        "predictions": {
            name: [values[index] for index in kept_indices]
            for name, values in outcomes["predictions"].items()
        },
        "correct": {
            name: [values[index] for index in kept_indices]
            for name, values in outcomes["correct"].items()
        },
    }
    return filtered_examples, filtered_outcomes


def _write_outcomes(
    path: Path,
    examples_df: pd.DataFrame,
    outcomes: dict[str, Any],
    selected_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "example_id",
                "context_id",
                "target_first_token_id",
                "baseline_prediction_id",
                "baseline_correct",
                "selected_prediction_id",
                "selected_correct",
            ]
        )
        for index, row in examples_df.reset_index(drop=True).iterrows():
            writer.writerow(
                [
                    row["example_id"],
                    row["context_id"],
                    outcomes["target_ids"][index],
                    outcomes["predictions"]["baseline"][index],
                    outcomes["correct"]["baseline"][index],
                    outcomes["predictions"][selected_name][index],
                    outcomes["correct"][selected_name][index],
                ]
            )


def _write_random_outcomes(
    path: Path,
    examples_df: pd.DataFrame,
    outcomes: dict[str, Any],
    random_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example_ids = examples_df["example_id"].astype(str).tolist()
    context_ids = examples_df["context_id"].astype(str).tolist()
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "condition",
                "example_id",
                "context_id",
                "target_first_token_id",
                "prediction_id",
                "correct",
            ]
        )
        for condition in random_names:
            for index, example_id in enumerate(example_ids):
                writer.writerow(
                    [
                        condition,
                        example_id,
                        context_ids[index],
                        outcomes["target_ids"][index],
                        outcomes["predictions"][condition][index],
                        outcomes["correct"][condition][index],
                    ]
                )


def _write_triplet_tables(
    *,
    output_dir: Path,
    payload: dict[str, Any],
) -> None:
    tag = artifact_tag(
        model_name=str(payload["model_name"]),
        sae_uid=str(payload["sae_uid"]),
        task=str(payload["task"]),
    )
    selected = payload["results"]["selected_ablation"]
    selected_sig = selected["drop_significance"]
    random_summary = payload["results"]["random_control_summary"]
    random_accuracies = [
        float(row["accuracy"]) for row in payload["results"]["random_controls"]
    ]
    if "mean_accuracy" not in random_summary:
        random_summary["mean_accuracy"] = (
            sum(random_accuracies) / len(random_accuracies)
            if random_accuracies
            else None
        )
        random_summary["sample_std_accuracy"] = (
            float(pd.Series(random_accuracies).std(ddof=1))
            if len(random_accuracies) >= 2
            else None
        )
        random_summary["sample_std_accuracy_drop"] = (
            float(
                pd.Series(
                    [
                        payload["results"]["baseline"]["accuracy"] - accuracy
                        for accuracy in random_accuracies
                    ]
                ).std(ddof=1)
            )
            if len(random_accuracies) >= 2
            else None
        )
        random_summary["aggregate_accuracy_significance"] = (
            one_sample_mean_degradation_test(
                random_accuracies,
                reference_accuracy=payload["results"]["baseline"]["accuracy"],
            )
        )
    common = {
        "task": payload["task"],
        "model_name": payload["model_name"],
        "sae_uid": payload["sae_uid"],
        "sae_release": payload["sae_release"],
        "sae_id": payload["sae_id"],
        "feature_set": payload["feature_set"],
        "ablation_position": payload["intervention"]["position"],
        "selected_feature_count": payload["selected_feature_count"],
        "baseline_correct_only": payload.get("analysis_filter", {}).get(
            "baseline_correct_only"
        ),
        "input_example_count": payload.get("analysis_filter", {}).get(
            "input_example_count"
        ),
        "analysis_example_count": payload.get("analysis_filter", {}).get(
            "analysis_example_count"
        ),
        "significance_pair_count": payload.get("analysis_filter", {}).get(
            "analysis_example_count"
        ),
        "excluded_baseline_incorrect_count": payload.get("analysis_filter", {}).get(
            "excluded_baseline_incorrect_count"
        ),
        "full_dataset_baseline_accuracy": payload.get("analysis_filter", {}).get(
            "full_dataset_baseline_accuracy"
        ),
    }
    selected_row = {
        **common,
        "selected_feature_ids": json.dumps(payload["selected_feature_ids"]),
        "baseline_accuracy": payload["results"]["baseline"]["accuracy"],
        "ablated_accuracy": selected["accuracy"],
        "accuracy_drop": selected["accuracy_drop"],
        "mcnemar_p_one_sided": selected_sig["p_value_one_sided"],
        "mcnemar_log10_p_one_sided": selected_sig["log10_p_value_one_sided"],
        "mcnemar_p_two_sided": selected_sig["p_value_two_sided"],
        "baseline_correct_ablation_wrong": selected_sig[
            "reference_correct_comparison_wrong"
        ],
        "baseline_wrong_ablation_correct": selected_sig[
            "reference_wrong_comparison_correct"
        ],
        "random_trials": random_summary["trials"],
        "random_mean_accuracy": random_summary["mean_accuracy"],
        "random_sample_std_accuracy": random_summary["sample_std_accuracy"],
        "random_mean_accuracy_drop": random_summary["mean_accuracy_drop"],
        "random_sample_std_accuracy_drop": random_summary["sample_std_accuracy_drop"],
        "random_max_accuracy_drop": random_summary["max_accuracy_drop"],
        "random_aggregate_p_one_sided": (
            random_summary["aggregate_accuracy_significance"] or {}
        ).get("p_value_one_sided"),
        "empirical_random_p": random_summary[
            "empirical_p_value_random_drop_at_least_selected"
        ],
    }
    for path in tagged_paths(output_dir / "selected_ablation_table.csv", tag):
        pd.DataFrame([selected_row]).to_csv(path, index=False)

    random_rows = []
    for trial_index, row in enumerate(payload["results"]["random_controls"]):
        drop_sig = row["drop_significance"]
        selected_sig = row["selected_vs_random"]
        random_rows.append(
            {
                **common,
                "random_trial": trial_index,
                "condition": row["condition"],
                "random_accuracy": row["accuracy"],
                "random_accuracy_drop": row["accuracy_drop"],
                "random_drop_p_one_sided": drop_sig["p_value_one_sided"],
                "random_drop_log10_p_one_sided": drop_sig["log10_p_value_one_sided"],
                "random_drop_p_two_sided": drop_sig["p_value_two_sided"],
                "random_correct_selected_wrong": selected_sig[
                    "reference_correct_comparison_wrong"
                ],
                "random_wrong_selected_correct": selected_sig[
                    "reference_wrong_comparison_correct"
                ],
                "selected_drop_greater_than_random_p_one_sided": selected_sig[
                    "p_value_one_sided"
                ],
                "random_feature_ids": json.dumps(
                    payload["random_control"]["feature_sets"][trial_index]
                ),
            }
        )
    for path in tagged_paths(output_dir / "random_ablation_table.csv", tag):
        pd.DataFrame(random_rows).to_csv(path, index=False)
    aggregate_test = random_summary["aggregate_accuracy_significance"] or {}
    aggregate_row = {
        **common,
        "baseline_accuracy": payload["results"]["baseline"]["accuracy"],
        "random_trials": random_summary["trials"],
        "random_mean_accuracy": random_summary["mean_accuracy"],
        "random_sample_std_accuracy": random_summary["sample_std_accuracy"],
        "random_mean_accuracy_drop": random_summary["mean_accuracy_drop"],
        "random_sample_std_accuracy_drop": random_summary["sample_std_accuracy_drop"],
        "random_max_accuracy_drop": random_summary["max_accuracy_drop"],
        "aggregate_test": aggregate_test.get("test"),
        "aggregate_alternative": aggregate_test.get("alternative"),
        "aggregate_t_statistic_for_accuracy_drop": aggregate_test.get(
            "t_statistic_for_accuracy_drop"
        ),
        "aggregate_p_value_one_sided": aggregate_test.get("p_value_one_sided"),
        "selected_drop_empirical_random_p": random_summary[
            "empirical_p_value_random_drop_at_least_selected"
        ],
    }
    for path in tagged_paths(output_dir / "random_ablation_aggregate.csv", tag):
        pd.DataFrame([aggregate_row]).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "ablation_summary.json"
    if summary_path.exists() and not args.overwrite:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        tag = artifact_tag(
            model_name=str(payload["model_name"]),
            sae_uid=str(payload["sae_uid"]),
            task=str(payload["task"]),
        )
        payload = dict(payload, artifact_tag=tag)
        _write_triplet_tables(output_dir=args.output_dir, payload=payload)
        for path in tagged_paths(summary_path, tag):
            path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Skipping completed ablation: {summary_path}")
        return
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    discovery = load_discovery_summary(args.discovery_summary)
    selected_ids = resolve_feature_set(discovery, args.sae_uid, args.feature_set)
    metrics = pd.read_csv(per_sae_metrics_path(discovery, args.sae_uid))
    random_sets = matched_random_feature_sets(
        metrics=metrics,
        selected_feature_ids=selected_ids,
        trials=args.random_trials,
        seed=args.random_seed,
        match_pool_size=args.random_match_pool_size,
    )
    reference = _reference_from_summary(discovery, args.sae_uid)
    dtype_name = args.llm_dtype or LLM_NAME_TO_DTYPE.get(args.model_name, "float32")
    dtype = general_utils.str_to_dtype(dtype_name)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=args.device, dtype=dtype
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded_sae = load_sae_reference(
        reference,
        model_name=args.model_name,
        device=args.device,
        llm_dtype=dtype,
        download_saes_dir=args.download_saes_dir,
    )
    _validate_sae_for_intervention(loaded_sae)

    dataset = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    examples_df, dataset_metadata = normalize_examples(dataset)
    examples_df = add_target_token_columns(
        examples_df, tokenizer, answer_prefix=args.answer_prefix
    )
    if (
        args.expected_examples is not None
        and len(examples_df) != args.expected_examples
    ):
        raise ValueError(
            f"Expected {args.expected_examples} examples, found {len(examples_df)}"
        )
    selected_name = f"{args.feature_set}_features"
    random_names = [f"matched_random_{index:03d}" for index in range(len(random_sets))]
    conditions = [
        (selected_name, selected_ids),
        *zip(random_names, random_sets, strict=True),
    ]
    outcomes = evaluate_interventions(
        model=model,
        tokenizer=tokenizer,
        loaded_sae=loaded_sae,
        examples_df=examples_df,
        conditions=list(conditions),
        batch_size=args.batch_size,
        device=args.device,
        position_mode=args.ablation_position,
    )
    baseline_full_summary = accuracy_summary(outcomes["correct"]["baseline"])
    analysis_examples_df = examples_df
    analysis_outcomes = outcomes
    if args.baseline_correct_only:
        baseline_correct_mask = list(outcomes["correct"]["baseline"])
        if not any(baseline_correct_mask):
            raise ValueError("No examples remain after baseline-correct filtering")
        analysis_examples_df, analysis_outcomes = filter_outcomes_to_examples(
            examples_df=examples_df,
            outcomes=outcomes,
            keep_mask=baseline_correct_mask,
        )
    results = summarize_ablation(
        outcomes=analysis_outcomes,
        selected_name=selected_name,
        random_names=random_names,
    )
    if not args.allow_imperfect_baseline and results["baseline"]["accuracy"] != 1.0:
        raise ValueError(
            "Curated-dataset invariant failed during ablation: baseline accuracy "
            f"was {results['baseline']['accuracy']:.8f}, expected 1.0. Recheck "
            "model, tokenizer, dtype, and dataset provenance."
        )
    payload = {
        "schema_version": 1,
        "task": args.task,
        "model_name": args.model_name,
        "sae_uid": args.sae_uid,
        "sae_release": reference.sae_release,
        "sae_id": reference.sae_id,
        "dataset_path": str(args.dataset_path.resolve()),
        "dataset_metadata": dataset_metadata,
        "discovery_summary": str(args.discovery_summary.resolve()),
        "feature_set": args.feature_set,
        "selected_feature_ids": selected_ids,
        "selected_feature_count": len(selected_ids),
        "analysis_filter": {
            "baseline_correct_only": args.baseline_correct_only,
            "applies_to_selected_and_random_controls": True,
            "significance_tests_use_filtered_pairs": args.baseline_correct_only,
            "input_example_count": int(len(examples_df)),
            "analysis_example_count": int(len(analysis_examples_df)),
            "excluded_baseline_incorrect_count": int(
                len(examples_df) - len(analysis_examples_df)
            ),
            "full_dataset_baseline_accuracy": baseline_full_summary["accuracy"],
            "full_dataset_baseline_correct": baseline_full_summary["correct"],
            "full_dataset_baseline_total": baseline_full_summary["total"],
        },
        "intervention": {
            "type": "zero_ablation",
            "position": args.ablation_position,
            "sae_reconstruction_error_preserved": True,
            "answer_scoring": "first answer token exact argmax",
        },
        "random_control": {
            "trials": args.random_trials,
            "seed": args.random_seed,
            "match_fields": [
                "example_prevalence",
                "log1p(mean_activation_when_active)",
            ],
            "match_pool_size": args.random_match_pool_size,
            "feature_sets": random_sets,
        },
        "results": results,
    }
    tag = artifact_tag(
        model_name=args.model_name,
        sae_uid=args.sae_uid,
        task=args.task,
    )
    payload["artifact_tag"] = tag
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in tagged_paths(summary_path, tag):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_triplet_tables(output_dir=args.output_dir, payload=payload)
    condition_accuracies = pd.DataFrame(
        [
            {
                "condition": "baseline",
                **results["baseline"],
                "accuracy_drop": 0.0,
            },
            {
                "condition": selected_name,
                **{
                    key: value
                    for key, value in results["selected_ablation"].items()
                    if key != "drop_significance"
                },
            },
            *[
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"drop_significance", "selected_vs_random"}
                }
                for row in results["random_controls"]
            ],
        ]
    )
    for path in tagged_paths(args.output_dir / "condition_accuracies.csv", tag):
        condition_accuracies.to_csv(path, index=False)
    _write_outcomes(
        args.output_dir / "selected_outcomes.csv.gz",
        analysis_examples_df,
        analysis_outcomes,
        selected_name,
    )
    if random_names:
        _write_random_outcomes(
            args.output_dir / "random_outcomes.csv.gz",
            analysis_examples_df,
            analysis_outcomes,
            random_names,
        )


if __name__ == "__main__":
    main()
