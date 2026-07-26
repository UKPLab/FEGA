#!/usr/bin/env python3
"""Generate PrOntoQA-style induction datasets with one-word answers.

This script intentionally does not use the stock `run_experiment.py` generator:
the stock generator makes the relevant rule explicit in the test question and is
optimized for true/false reasoning plus chain-of-thought. For mechanistic
induction-head probes, we instead want prompt-local random mappings that are
recoverable only from the in-context demonstrations.

Each context contains `shots` demonstrations of the form:

    Q: Max is a wumpus. Every wumpus is a dumpus. Max is a
    A: dumpus

Each query then reuses one of the demonstrated source concepts while hiding the
target concept. The default query style mirrors the user's proposed task:

    Q: Laro is a wumpus. Laro is a
    A: dumpus

For a stronger induction-head signal, use `--query-style rule_completion`, which
creates an exact repeated prefix immediately before the answer:

    Q: Laro is a wumpus. Every wumpus is a
    A: dumpus
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SUPPORT_ENTITY_NAMES = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"]

# These are the fictional PrOntoQA concept families used by `run_experiment.py`
# when `--disjoint-concept-names` is enabled.
FICTIONAL_CONCEPT_FAMILIES: List[List[str]] = [
    ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus"],
    ["timple", "yimple", "starple", "shumple", "zhomple", "remple", "fomple", "fimple", "worple", "sorple"],
    ["tergit", "gergit", "stergit", "kergit", "shergit", "pergit", "bongit", "orgit", "welgit", "jelgit"],
    ["felper", "dolper", "sarper", "irper", "chorper", "parper", "arper", "lemper", "hilper", "gomper"],
    ["dalpist", "umpist", "rifpist", "storpist", "shalpist", "yerpist", "ilpist", "boompist", "scrompist", "phorpist"],
    ["prilpant", "gwompant", "urpant", "grimpant", "shilpant", "zhorpant", "rorpant", "dropant", "lerpant", "quimpant"],
    ["zilpor", "frompor", "stirpor", "porpor", "kurpor", "shampor", "werpor", "zhimpor", "yempor", "jempor"],
    ["folpee", "drompee", "delpee", "lompee", "wolpee", "gorpee", "shimpee", "rimpee", "twimpee", "serpee"],
    ["daumpin", "thorpin", "borpin", "rofpin", "bempin", "dulpin", "harpin", "lirpin", "yompin", "stopin"],
]

USABLE_CONCEPT_FAMILIES: List[List[str]] = [
    [concept for concept in family if concept[0] not in "aeiou"]
    for family in FICTIONAL_CONCEPT_FAMILIES
]
FICTIONAL_CONCEPTS = [concept for family in USABLE_CONCEPT_FAMILIES for concept in family]

NAME_ONSETS = [
    "b", "br", "c", "cl", "d", "dr", "f", "fl", "g", "gr", "j", "k", "kr",
    "l", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z",
]
NAME_VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ia", "ou"]
NAME_CODAS = ["", "l", "m", "n", "r", "s", "th", "v", "x", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prontoqa-1/prontoqa_induction_dataset.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--num-contexts",
        type=int,
        default=1000,
        help="Number of distinct 3-shot contexts to generate.",
    )
    parser.add_argument(
        "--queries-per-context",
        type=int,
        default=100,
        help="Number of final questions to generate for each context.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=3,
        help="Number of ICL examples per context.",
    )
    parser.add_argument(
        "--query-style",
        choices=["entity_completion", "rule_completion"],
        default="entity_completion",
        help=(
            "Final-question template. "
            "`entity_completion` matches the user's proposed format; "
            "`rule_completion` gives a cleaner induction-head copy signal."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=["grouped", "flat"],
        default="grouped",
        help=(
            "`grouped` stores queries under each 3-shot context; "
            "`flat` stores one fully materialized prompt per example."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation.",
    )
    parser.add_argument(
        "--allow-same-family",
        action="store_true",
        help=(
            "By default, source and target concepts are drawn from different "
            "PrOntoQA families to avoid easy phonological analogies."
        ),
    )
    return parser.parse_args()


def make_name(rng: random.Random, used: set[str]) -> str:
    while True:
        syllable_count = 2
        pieces = []
        for _ in range(syllable_count):
            pieces.append(rng.choice(NAME_ONSETS) + rng.choice(NAME_VOWELS) + rng.choice(NAME_CODAS))
        name = "".join(pieces).capitalize()
        if 4 <= len(name) <= 8 and name not in used and name.lower() not in FICTIONAL_CONCEPTS:
            used.add(name)
            return name


def sample_pairs(rng: random.Random, shots: int, allow_same_family: bool) -> List[Tuple[str, str]]:
    if allow_same_family:
        chosen = rng.sample(FICTIONAL_CONCEPTS, shots * 2)
        return [(chosen[i], chosen[shots + i]) for i in range(shots)]

    if shots * 2 > len(USABLE_CONCEPT_FAMILIES):
        raise ValueError(
            "Cannot keep source and target families disjoint: "
            f"need {shots * 2} families but only {len(USABLE_CONCEPT_FAMILIES)} exist."
        )

    family_indices = list(range(len(USABLE_CONCEPT_FAMILIES)))
    source_family_indices = rng.sample(family_indices, shots)
    remaining_families = [idx for idx in family_indices if idx not in source_family_indices]
    target_family_indices = rng.sample(remaining_families, shots)

    sources = [rng.choice(USABLE_CONCEPT_FAMILIES[idx]) for idx in source_family_indices]
    targets = [rng.choice(USABLE_CONCEPT_FAMILIES[idx]) for idx in target_family_indices]
    return list(zip(sources, targets))


def build_support_x(entity: str, source: str, target: str) -> str:
    return f"{entity} is a {source}. Every {source} is a {target}. {entity} is a"


def build_query_x(entity: str, source: str, query_style: str) -> str:
    if query_style == "entity_completion":
        return f"{entity} is a {source}. {entity} is a"
    if query_style == "rule_completion":
        return f"{entity} is a {source}. Every {source} is a"
    raise ValueError(f"Unsupported query style: {query_style}")


def make_balanced_support_indices(rng: random.Random, shots: int, query_count: int) -> List[int]:
    indices = list(range(shots)) * (query_count // shots)
    remainder = query_count % shots
    if remainder:
        indices.extend(rng.sample(list(range(shots)), remainder))
    rng.shuffle(indices)
    return indices


def build_context(
    rng: random.Random,
    context_index: int,
    shots: int,
    queries_per_context: int,
    query_style: str,
    allow_same_family: bool,
) -> Dict[str, object]:
    context_id = f"ctx_{context_index:05d}"
    used_names: set[str] = set()

    support_names = rng.sample(SUPPORT_ENTITY_NAMES, shots)
    used_names.update(support_names)
    pairs = sample_pairs(rng, shots, allow_same_family)

    icl_examples: List[Dict[str, object]] = []
    for shot_index, ((source, target), entity) in enumerate(zip(pairs, support_names)):
        x = build_support_x(entity, source, target)
        y = target
        icl_examples.append(
            {
                "example_id": f"{context_id}_shot_{shot_index:02d}",
                "entity": entity,
                "source_concept": source,
                "target_concept": target,
                "x": x,
                "y": y,
                "full_qa": f"Q: {x}\nA: {y}",
                "rule": f"Every {source} is a {target}.",
            }
        )

    context_prompt = "\n\n".join(example["full_qa"] for example in icl_examples)
    support_indices = make_balanced_support_indices(rng, shots, queries_per_context)

    queries: List[Dict[str, object]] = []
    for query_index, support_example_index in enumerate(support_indices):
        support_example = icl_examples[support_example_index]
        entity = make_name(rng, used_names)
        x = build_query_x(entity, support_example["source_concept"], query_style)
        answer = support_example["target_concept"]
        prompt = f"{context_prompt}\n\nQ: {x}\nA:"
        queries.append(
            {
                "query_id": f"{context_id}_query_{query_index:03d}",
                "entity": entity,
                "source_concept": support_example["source_concept"],
                "target_concept": answer,
                "x": x,
                "y": answer,
                "answer": answer,
                "prompt": prompt,
                "support_example_index": support_example_index,
                "lookup_rule": support_example["rule"],
                "induction_prefix": (
                    f"Every {support_example['source_concept']} is a"
                    if query_style == "rule_completion"
                    else None
                ),
            }
        )

    context = {
        "context_id": context_id,
        "query_style": query_style,
        "icl_examples": icl_examples,
        "context_prompt": context_prompt,
        "queries": queries,
    }
    validate_context(context)
    return context


def validate_context(context: Dict[str, object]) -> None:
    icl_examples = context["icl_examples"]
    queries = context["queries"]
    query_style = context["query_style"]

    assert isinstance(icl_examples, list)
    assert isinstance(queries, list)
    assert isinstance(query_style, str)

    source_to_target = {example["source_concept"]: example["target_concept"] for example in icl_examples}
    assert len(source_to_target) == len(icl_examples)
    assert len(set(source_to_target.values())) == len(icl_examples)

    support_entities = {example["entity"] for example in icl_examples}

    for query in queries:
        src = query["source_concept"]
        tgt = query["target_concept"]
        support_index = query["support_example_index"]
        support_example = icl_examples[support_index]

        assert query["answer"] == query["y"] == tgt
        assert source_to_target[src] == tgt
        assert support_example["source_concept"] == src
        assert support_example["target_concept"] == tgt
        assert query["entity"] not in support_entities
        assert tgt not in query["x"]
        assert sum(example["source_concept"] == src for example in icl_examples) == 1

        if query_style == "rule_completion":
            prefix = query["induction_prefix"]
            support_surface = f"{support_example['x']} {support_example['y']}"
            assert prefix is not None
            assert prefix in query["x"]
            assert f"{prefix} {tgt}" in support_surface


def generate_dataset(args: argparse.Namespace) -> Dict[str, object]:
    if args.shots <= 0:
        raise ValueError("--shots must be positive.")
    if args.queries_per_context <= 0:
        raise ValueError("--queries-per-context must be positive.")
    if args.num_contexts <= 0:
        raise ValueError("--num-contexts must be positive.")

    rng = random.Random(args.seed)
    contexts = [
        build_context(
            rng=rng,
            context_index=context_index,
            shots=args.shots,
            queries_per_context=args.queries_per_context,
            query_style=args.query_style,
            allow_same_family=args.allow_same_family,
        )
        for context_index in range(args.num_contexts)
    ]

    metadata = {
        "dataset_name": "prontoqa_induction_dataset",
        "schema_version": 1,
        "num_contexts": args.num_contexts,
        "queries_per_context": args.queries_per_context,
        "shots": args.shots,
        "query_style": args.query_style,
        "layout": args.layout,
        "answer_format": "one_word",
        "chain_of_thought_included": False,
        "seed": args.seed,
        "concept_pool": "prontoqa_fictional",
        "distinct_source_target_families": not args.allow_same_family,
        "mechanistic_notes": [
            "Each prompt contains prompt-local random source->target mappings.",
            "Each query source concept appears in exactly one support example.",
            "Targets never appear in the final query text.",
            "Answers are single-word PrOntoQA nonce concepts.",
            "Vowel-initial nonce concepts are filtered out so the fixed 'is a <answer>' surface form stays grammatical.",
            "rule_completion is the cleanest exact-prefix induction-head condition.",
        ],
    }

    if args.layout == "grouped":
        return {"metadata": metadata, "contexts": contexts}

    examples = []
    for context in contexts:
        for query in context["queries"]:
            examples.append(
                {
                    "example_id": query["query_id"],
                    "context_id": context["context_id"],
                    "query_style": context["query_style"],
                    "prompt": query["prompt"],
                    "answer": query["answer"],
                    "x": query["x"],
                    "y": query["y"],
                    "entity": query["entity"],
                    "source_concept": query["source_concept"],
                    "target_concept": query["target_concept"],
                    "support_example_index": query["support_example_index"],
                    "lookup_rule": query["lookup_rule"],
                    "induction_prefix": query["induction_prefix"],
                }
            )

    return {"metadata": metadata, "examples": examples}


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(dataset, f, indent=2)
        else:
            json.dump(dataset, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
