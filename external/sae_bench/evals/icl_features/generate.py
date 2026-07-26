from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from sae_bench.evals.icl_features.generators import TASKS, build_families
from sae_bench.evals.icl_features.model_filter import (
    CorrectnessResult,
    GemmaFirstTokenFilter,
    load_brown_words,
    single_token_words,
)
from sae_bench.evals.icl_features.schema import (
    balanced_family_quotas,
    validate_dataset,
)


class RowScorer(Protocol):
    model_name: str
    resolved_model_name: str
    tokenizer: Any

    def answer_token_ids(self, answer: str) -> list[int]: ...

    def score(self, rows: list[dict[str, Any]]) -> list[CorrectnessResult]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced, model-correct ICL dataset for SAE pointer-feature "
            "discovery. Prompt families receive equal quotas before feature selection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        choices=["gemma-2-2b", "gemma-2-9b"],
        default="gemma-2-2b",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--answer-prefix", default=" ")
    parser.add_argument("--target-examples", type=int, default=50_000)
    parser.add_argument("--num-families", type=int, default=1_000)
    parser.add_argument(
        "--family-pool-multiplier",
        type=int,
        default=1,
        help=(
            "Build this many times more candidate prompt families than the final "
            "dataset needs and keep the first families that satisfy their balanced "
            "model-correct quota. This preserves the final family balance while "
            "making generation robust to low-acceptance random families."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates-per-family-round", type=int, default=8)
    parser.add_argument("--max-candidates-per-family", type=int, default=2_000)
    parser.add_argument(
        "--allow-multitoken-answers",
        action="store_true",
        help="Keep model-correct rows whose answer has more than one token.",
    )
    parser.add_argument("--word-pool-size", type=int, default=1_000)
    parser.add_argument("--pattern-length", type=int, default=5)
    parser.add_argument("--random-gap-length", type=int, default=10)
    parser.add_argument("--wc-demo-per-group", type=int, default=3)
    parser.add_argument("--wc-features-per-group", type=int, default=2)
    parser.add_argument("--wc-groups", type=int, default=2)
    parser.add_argument("--wc-random-pool", type=int, default=50)
    parser.add_argument("--wc-distractors", type=int, default=7)
    parser.add_argument("--tt-demos", type=int, default=5)
    parser.add_argument("--prontoqa-shots", type=int, default=3)
    parser.add_argument(
        "--prontoqa-query-style",
        choices=["rule_completion", "entity_completion"],
        default="rule_completion",
    )
    return parser.parse_args()


def curate_balanced_examples(
    *,
    families: list[Any],
    quotas: list[int],
    scorer: RowScorer,
    candidates_per_family_round: int,
    max_candidates_per_family: int,
    require_single_token_answers: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(families) != len(quotas):
        raise ValueError("families and quotas must have the same length")
    accepted: list[list[dict[str, Any]]] = [[] for _ in families]
    slot_quotas = [
        balanced_family_quotas(quota, int(getattr(family, "num_support_slots", 1)))
        for family, quota in zip(families, quotas, strict=True)
    ]
    accepted_by_slot = [
        [0] * len(family_slot_quotas) for family_slot_quotas in slot_quotas
    ]
    attempts = [0] * len(families)
    correct_counts = [0] * len(families)
    multitoken_rejections = [0] * len(families)
    duplicate_prompt_rejections = [0] * len(families)
    filled_slot_rejections = [0] * len(families)
    seen_prompts: list[set[str]] = [set() for _ in families]

    while any(len(rows) < quota for rows, quota in zip(accepted, quotas, strict=True)):
        candidates = []
        candidate_family_indices = []
        for family_index, (family, quota) in enumerate(
            zip(families, quotas, strict=True)
        ):
            if len(accepted[family_index]) >= quota:
                continue
            remaining_attempts = max_candidates_per_family - attempts[family_index]
            if remaining_attempts <= 0:
                continue
            round_count = min(candidates_per_family_round, remaining_attempts)
            for _ in range(round_count):
                candidate_index = attempts[family_index]
                attempts[family_index] += 1
                candidate = family.generate(candidate_index)
                prompt = str(candidate["prompt"])
                if prompt in seen_prompts[family_index]:
                    duplicate_prompt_rejections[family_index] += 1
                    continue
                seen_prompts[family_index].add(prompt)
                candidates.append(candidate)
                candidate_family_indices.append(family_index)

        if not candidates and any(
            len(accepted[index]) < quotas[index]
            and attempts[index] < max_candidates_per_family
            for index in range(len(families))
        ):
            continue
        if not candidates:
            failures = [
                {
                    "context_id": families[index].family_id,
                    "quota": quotas[index],
                    "accepted": len(accepted[index]),
                    "attempts": attempts[index],
                }
                for index in range(len(families))
                if len(accepted[index]) < quotas[index]
            ]
            raise RuntimeError(
                "Unable to fill model-correct family quotas before the candidate "
                f"limit. First failures: {failures[:10]}"
            )

        results = scorer.score(candidates)
        for row, family_index, result in zip(
            candidates, candidate_family_indices, results, strict=True
        ):
            if result.correct:
                correct_counts[family_index] += 1
            if not result.correct:
                continue
            if require_single_token_answers and result.target_token_length != 1:
                multitoken_rejections[family_index] += 1
                continue
            if len(accepted[family_index]) >= quotas[family_index]:
                continue
            support_slot = int(row.get("support_example_index") or 0)
            if not 0 <= support_slot < len(slot_quotas[family_index]):
                raise ValueError(
                    f"Family {families[family_index].family_id!r} generated invalid "
                    f"support slot {support_slot}"
                )
            if (
                accepted_by_slot[family_index][support_slot]
                >= slot_quotas[family_index][support_slot]
            ):
                filled_slot_rejections[family_index] += 1
                continue
            accepted_rank = len(accepted[family_index])
            row.update(
                {
                    "example_id": (
                        f"{families[family_index].family_id}_query_{accepted_rank:05d}"
                    ),
                    "family_query_index": accepted_rank,
                    "target_first_token_id": result.target_first_token_id,
                    "target_token_length": result.target_token_length,
                    "model_correct_first_token": True,
                    "predicted_first_token_id": result.predicted_first_token_id,
                    "predicted_first_token": result.predicted_first_token,
                }
            )
            accepted[family_index].append(row)
            accepted_by_slot[family_index][support_slot] += 1

    examples = [row for family_rows in accepted for row in family_rows]
    stats = []
    for index, family in enumerate(families):
        stats.append(
            {
                "context_id": family.family_id,
                "quota": quotas[index],
                "accepted": len(accepted[index]),
                "support_slot_quotas": slot_quotas[index],
                "accepted_by_support_slot": accepted_by_slot[index],
                "candidates_scored": attempts[index],
                "model_correct_candidates": correct_counts[index],
                "multitoken_correct_rejected": multitoken_rejections[index],
                "duplicate_prompts_rejected": duplicate_prompt_rejections[index],
                "filled_support_slot_rejected": filled_slot_rejections[index],
                "model_correct_acceptance_rate": (
                    correct_counts[index] / attempts[index] if attempts[index] else 0.0
                ),
            }
        )
    return examples, stats


def curate_balanced_examples_with_family_backfill(
    *,
    families: list[Any],
    final_family_count: int,
    final_quotas: list[int],
    scorer: RowScorer,
    candidates_per_family_round: int,
    max_candidates_per_family: int,
    require_single_token_answers: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    if final_family_count <= 0:
        raise ValueError("final_family_count must be positive")
    if len(final_quotas) != final_family_count:
        raise ValueError("final_quotas must have one entry per final family")
    selected_examples: list[dict[str, Any]] = []
    selected_stats: list[dict[str, Any]] = []
    selected_families: list[Any] = []
    rejected_families: list[dict[str, Any]] = []

    for family in families:
        if len(selected_families) >= final_family_count:
            break
        quota = final_quotas[len(selected_families)]
        try:
            family_examples, family_stats = curate_balanced_examples(
                families=[family],
                quotas=[quota],
                scorer=scorer,
                candidates_per_family_round=candidates_per_family_round,
                max_candidates_per_family=max_candidates_per_family,
                require_single_token_answers=require_single_token_answers,
            )
        except RuntimeError as exc:
            rejected_families.append(
                {
                    "context_id": family.family_id,
                    "quota": quota,
                    "reason": str(exc),
                }
            )
            if len(rejected_families) <= 10 or len(rejected_families) % 50 == 0:
                print(
                    "Rejected "
                    f"{family.family_id}; selected "
                    f"{len(selected_families)}/{final_family_count}, "
                    f"scanned {len(selected_families) + len(rejected_families)}.",
                    flush=True,
                )
            continue
        selected_examples.extend(family_examples)
        selected_stats.extend(family_stats)
        selected_families.append(family)
        if len(selected_families) % 50 == 0:
            print(
                "Selected "
                f"{len(selected_families)}/{final_family_count} prompt families "
                f"after scanning {len(selected_families) + len(rejected_families)}.",
                flush=True,
            )

    if len(selected_families) < final_family_count:
        raise RuntimeError(
            "Unable to fill the requested number of balanced model-correct prompt "
            f"families. Requested {final_family_count}, selected "
            f"{len(selected_families)}, rejected {len(rejected_families)}. "
            f"First rejected families: {rejected_families[:10]}"
        )
    return selected_examples, selected_stats, selected_families, rejected_families


def build_dataset(args: argparse.Namespace, scorer: RowScorer) -> dict[str, Any]:
    if args.candidates_per_family_round <= 0:
        raise ValueError("--candidates-per-family-round must be positive")
    if args.max_candidates_per_family <= 0:
        raise ValueError("--max-candidates-per-family must be positive")
    if args.family_pool_multiplier <= 0:
        raise ValueError("--family-pool-multiplier must be positive")

    if args.task in {"lsc", "wc"}:
        words = single_token_words(
            scorer.tokenizer,
            load_brown_words(),
            limit=args.word_pool_size,
        )
        word_source = "brown_corpus_single_token_words"
    else:
        words = []
        word_source = None

    candidate_family_count = args.num_families * args.family_pool_multiplier
    families = build_families(
        task=args.task,
        num_families=candidate_family_count,
        seed=args.seed,
        words=words,
        pattern_length=args.pattern_length,
        random_gap_length=args.random_gap_length,
        wc_demo_per_group=args.wc_demo_per_group,
        wc_features_per_group=args.wc_features_per_group,
        wc_groups=args.wc_groups,
        wc_random_pool=args.wc_random_pool,
        wc_distractors=args.wc_distractors,
        tt_demos=args.tt_demos,
        prontoqa_shots=args.prontoqa_shots,
        prontoqa_query_style=args.prontoqa_query_style,
    )
    quotas = balanced_family_quotas(args.target_examples, args.num_families)
    require_single_token_answers = (
        args.task != "prontoqa" and not args.allow_multitoken_answers
    )
    rejected_family_stats: list[dict[str, Any]] = []
    if args.family_pool_multiplier == 1:
        examples, family_filter_stats = curate_balanced_examples(
            families=families,
            quotas=quotas,
            scorer=scorer,
            candidates_per_family_round=args.candidates_per_family_round,
            max_candidates_per_family=args.max_candidates_per_family,
            require_single_token_answers=require_single_token_answers,
        )
        selected_families = families
    else:
        (
            examples,
            family_filter_stats,
            selected_families,
            rejected_family_stats,
        ) = curate_balanced_examples_with_family_backfill(
            families=families,
            final_family_count=args.num_families,
            final_quotas=quotas,
            scorer=scorer,
            candidates_per_family_round=args.candidates_per_family_round,
            max_candidates_per_family=args.max_candidates_per_family,
            require_single_token_answers=require_single_token_answers,
        )
    family_counts = Counter(str(row["context_id"]) for row in examples)
    metadata = {
        "dataset_name": f"sae_geometry_{args.task}",
        "schema_version": 2,
        "task_name": args.task,
        "layout": "flat",
        "answer_format": (
            "single_token"
            if require_single_token_answers
            else "first_token_correct_multitoken_allowed"
        ),
        "chain_of_thought_included": False,
        "seed": args.seed,
        "filtered_example_count": len(examples),
        "num_contexts": len(selected_families),
        "num_prompt_families": len(selected_families),
        "family_balance_max_difference": (
            max(family_counts.values()) - min(family_counts.values())
        ),
        "support_slot_balance_max_difference": 1,
        "all_prompts_unique_within_family": True,
        "queries_per_context_min": min(family_counts.values()),
        "queries_per_context_max": max(family_counts.values()),
        "prompt_family_definition": {
            "lsc": "fixed repeated prefix P*; target and random gap vary by query",
            "wc": "fixed prompt-local feature-to-label mapping and demonstrations; query class and distractors vary",
            "tt": "fixed language direction and demonstration set; held-out translation query varies",
            "prontoqa": "fixed prompt-local source-to-target support rules; entity and reused support rule vary",
        }[args.task],
        "model_filter": {
            "required": True,
            "model_name": scorer.model_name,
            "resolved_model_name": scorer.resolved_model_name,
            "criterion": "argmax next token equals first answer token",
            "answer_prefix": args.answer_prefix,
            "all_retained_examples_model_correct": True,
            "single_token_answers_required": require_single_token_answers,
        },
        "feature_selection_contract": {
            "active_if_post_encode_activation_exceeds": 0.0,
            "min_example_fraction": 0.9,
            "min_query_fraction_per_context": 0.9,
            "min_context_fraction": 0.9,
            "strict_feature_rule": "active in every analyzed example",
            "token_position": "final_prompt_token_before_first_answer_token",
        },
        "generation_config": {
            "target_examples": args.target_examples,
            "num_families": args.num_families,
            "candidate_family_count": candidate_family_count,
            "family_pool_multiplier": args.family_pool_multiplier,
            "selected_candidate_family_count": len(selected_families),
            "rejected_candidate_family_count": len(rejected_family_stats),
            "candidates_per_family_round": args.candidates_per_family_round,
            "max_candidates_per_family": args.max_candidates_per_family,
            "word_source": word_source,
            "word_pool_size": len(words),
            "pattern_length": args.pattern_length,
            "random_gap_length": args.random_gap_length,
            "wc_demo_per_group": args.wc_demo_per_group,
            "wc_features_per_group": args.wc_features_per_group,
            "wc_groups": args.wc_groups,
            "wc_random_pool": args.wc_random_pool,
            "wc_distractors": args.wc_distractors,
            "tt_demos": args.tt_demos,
            "tt_directions": ["en->de", "en->fr", "en->es", "en->it"],
            "prontoqa_shots": args.prontoqa_shots,
            "prontoqa_query_style": args.prontoqa_query_style,
        },
        "mechanistic_notes": {
            "lsc": [
                "Queries use the original P* T R* P* literal-copying structure.",
                "Each family fixes P* while targets and random gaps vary.",
            ],
            "wc": [
                "Queries use the original random-token feature/label classification structure.",
                "Each family fixes the prompt-local feature-to-label mapping and demonstrations.",
            ],
            "tt": [
                "Queries use the original five-shot token-translation structure.",
                "Each family fixes demonstrations and language direction.",
            ],
            "prontoqa": [
                "Each family contains prompt-local random source-to-target mappings.",
                "Targets do not appear in final query text.",
                "Source and target concepts use disjoint fictional concept families.",
            ],
        }[args.task],
    }
    payload = {
        "metadata": metadata,
        "prompt_families": [family.metadata() for family in selected_families],
        "family_filter_stats": family_filter_stats,
        "rejected_family_filter_stats": rejected_family_stats,
        "examples": examples,
    }
    validate_dataset(payload, require_model_correct=True)
    return payload


def main() -> None:
    args = parse_args()
    scorer = GemmaFirstTokenFilter(
        model_name=args.model_name,
        device=args.device,
        dtype_name=args.dtype,
        batch_size=args.batch_size,
        answer_prefix=args.answer_prefix,
    )
    dataset = build_dataset(args, scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")


if __name__ == "__main__":
    main()
