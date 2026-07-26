from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Protocol

from sae_bench.evals.icl_features.translations import translations


TASKS = ("lsc", "wc", "tt", "prontoqa")
TT_DIRECTIONS = (("en", "de"), ("en", "fr"), ("en", "es"), ("en", "it"))
LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
}

SUPPORT_ENTITY_NAMES = [
    "Fae",
    "Rex",
    "Sally",
    "Max",
    "Alex",
    "Sam",
    "Polly",
    "Stella",
    "Wren",
]
FICTIONAL_CONCEPT_FAMILIES = [
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
USABLE_CONCEPT_FAMILIES = [
    [concept for concept in family if concept[0] not in "aeiou"]
    for family in FICTIONAL_CONCEPT_FAMILIES
]
FICTIONAL_CONCEPTS = {
    concept for family in USABLE_CONCEPT_FAMILIES for concept in family
}
NAME_ONSETS = [
    "b", "br", "c", "cl", "d", "dr", "f", "fl", "g", "gr", "j", "k", "kr",
    "l", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z",
]
NAME_VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ia", "ou"]
NAME_CODAS = ["", "l", "m", "n", "r", "s", "th", "v", "x", "z"]


class FamilyGenerator(Protocol):
    family_id: str
    family_index: int

    def generate(self, candidate_index: int) -> dict[str, Any]: ...

    def metadata(self) -> dict[str, Any]: ...


def _family_seed(seed: int, task: str, family_index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task}:{family_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _base_example(
    *,
    task: str,
    family_id: str,
    family_index: int,
    candidate_index: int,
    query_style: str,
    prompt: str,
    answer: str,
    x: str,
    entity: str,
    source_concept: str,
    target_concept: str,
    support_example_index: int,
    lookup_rule: str,
    induction_prefix: str | None,
) -> dict[str, Any]:
    return {
        "example_id": f"{family_id}_candidate_{candidate_index:06d}",
        "context_id": family_id,
        "query_style": query_style,
        "prompt": prompt,
        "answer": answer,
        "x": x,
        "y": answer,
        "entity": entity,
        "source_concept": source_concept,
        "target_concept": target_concept,
        "support_example_index": int(support_example_index),
        "lookup_rule": lookup_rule,
        "induction_prefix": induction_prefix,
        "task_name": task,
        "prompt_family_id": family_id,
        "prompt_family_index": family_index,
        "candidate_index": candidate_index,
        "source_language": None,
        "target_language": None,
    }


@dataclass
class LSCFamily:
    family_index: int
    rng: random.Random
    words: list[str]
    pattern_length: int
    random_gap_length: int

    def __post_init__(self) -> None:
        self.family_id = f"lsc_family_{self.family_index:05d}"
        self.num_support_slots = 1
        self.pattern = self.rng.sample(self.words, self.pattern_length)

    def generate(self, candidate_index: int) -> dict[str, Any]:
        excluded = set(self.pattern)
        choices = [word for word in self.words if word not in excluded]
        target = self.rng.choice(choices)
        gap_choices = [word for word in choices if word != target]
        random_gap = self.rng.sample(gap_choices, self.random_gap_length)
        tokens = self.pattern + [target] + random_gap + self.pattern
        prompt = " " + " ".join(tokens)
        pattern_text = " ".join(self.pattern)
        return _base_example(
            task="lsc",
            family_id=self.family_id,
            family_index=self.family_index,
            candidate_index=candidate_index,
            query_style="literal_sequence_copying",
            prompt=prompt,
            answer=target,
            x=prompt,
            entity=pattern_text,
            source_concept=pattern_text,
            target_concept=target,
            support_example_index=0,
            lookup_rule="P* T R* P* -> T",
            induction_prefix=pattern_text,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "context_id": self.family_id,
            "task_name": "lsc",
            "num_support_slots": self.num_support_slots,
            "pattern": self.pattern,
            "pattern_length": self.pattern_length,
            "random_gap_length": self.random_gap_length,
        }


@dataclass
class WCFamily:
    family_index: int
    rng: random.Random
    words: list[str]
    n_demo_per_group: int
    n_features_per_group: int
    n_groups: int
    n_random_pool: int
    n_distractors: int

    def __post_init__(self) -> None:
        self.family_id = f"wc_family_{self.family_index:05d}"
        self.num_support_slots = self.n_groups
        required = self.n_groups * (self.n_features_per_group + 1) + self.n_random_pool
        if required > len(self.words):
            raise ValueError(
                f"WC family requires {required} distinct words, only {len(self.words)} available"
            )
        sampled = self.rng.sample(self.words, required)
        cursor = 0
        self.groups: list[dict[str, Any]] = []
        for group_index in range(self.n_groups):
            label = sampled[cursor]
            cursor += 1
            features = sampled[cursor : cursor + self.n_features_per_group]
            cursor += self.n_features_per_group
            self.groups.append(
                {"group_index": group_index, "label": label, "features": features}
            )
        self.distractor_pool = sampled[cursor:]
        demos = []
        for group in self.groups:
            for _ in range(self.n_demo_per_group):
                tokens = self.rng.choices(self.distractor_pool, k=self.n_distractors)
                tokens += list(group["features"])
                self.rng.shuffle(tokens)
                demos.append(f"{' '.join(tokens)}\n {group['label']}\n")
        self.rng.shuffle(demos)
        self.context_prompt = "".join(demos)

    def generate(self, candidate_index: int) -> dict[str, Any]:
        group = self.groups[candidate_index % self.n_groups]
        tokens = self.rng.choices(self.distractor_pool, k=self.n_distractors)
        tokens += list(group["features"])
        self.rng.shuffle(tokens)
        x = " ".join(tokens) + "\n"
        prompt = self.context_prompt + x
        return _base_example(
            task="wc",
            family_id=self.family_id,
            family_index=self.family_index,
            candidate_index=candidate_index,
            query_style="word_content",
            prompt=prompt,
            answer=str(group["label"]),
            x=x,
            entity=" ".join(tokens),
            source_concept="|".join(group["features"]),
            target_concept=str(group["label"]),
            support_example_index=int(group["group_index"]),
            lookup_rule=f"{' + '.join(group['features'])} -> {group['label']}",
            induction_prefix=None,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "context_id": self.family_id,
            "task_name": "wc",
            "num_support_slots": self.num_support_slots,
            "groups": self.groups,
            "n_demo_per_group": self.n_demo_per_group,
            "n_features_per_group": self.n_features_per_group,
            "n_groups": self.n_groups,
            "n_distractors": self.n_distractors,
        }


@dataclass
class TTFamily:
    family_index: int
    rng: random.Random
    n_demos: int

    def __post_init__(self) -> None:
        self.family_id = f"tt_family_{self.family_index:05d}"
        self.num_support_slots = 1
        self.source_language, self.target_language = TT_DIRECTIONS[
            self.family_index % len(TT_DIRECTIONS)
        ]
        pairs = list(
            zip(
                translations[self.source_language],
                translations[self.target_language],
            )
        )
        unique_pairs = list(dict.fromkeys(pairs))
        self.demo_pairs = self.rng.sample(unique_pairs, self.n_demos)
        demo_set = set(self.demo_pairs)
        self.query_pairs = [pair for pair in unique_pairs if pair not in demo_set]
        source_name = LANGUAGE_NAMES[self.source_language]
        target_name = LANGUAGE_NAMES[self.target_language]
        self.context_prompt = (
            f"Translate {source_name} words into {target_name}.\n"
            + "\n".join(
                f"{source_name}: {source}\n{target_name}: {target}"
                for source, target in self.demo_pairs
            )
            + "\n"
        )

    def generate(self, candidate_index: int) -> dict[str, Any]:
        source, target = self.query_pairs[candidate_index % len(self.query_pairs)]
        source_name = LANGUAGE_NAMES[self.source_language]
        target_name = LANGUAGE_NAMES[self.target_language]
        prompt = f" {self.context_prompt}{source_name}: {source}\n{target_name}:"
        return _base_example(
            task="tt",
            family_id=self.family_id,
            family_index=self.family_index,
            candidate_index=candidate_index,
            query_style="token_translation",
            prompt=prompt,
            answer=target,
            x=f"{source_name}: {source}\n{target_name}:",
            entity=source,
            source_concept=source,
            target_concept=target,
            support_example_index=0,
            lookup_rule=f"{source_name}->{target_name}",
            induction_prefix=f"{target_name}:",
        ) | {
            "source_language": self.source_language,
            "target_language": self.target_language,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "context_id": self.family_id,
            "task_name": "tt",
            "num_support_slots": self.num_support_slots,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "demonstrations": [
                {"source": source, "target": target}
                for source, target in self.demo_pairs
            ],
            "n_demos": self.n_demos,
        }


def _make_name(rng: random.Random, used: set[str]) -> str:
    while True:
        name = "".join(
            rng.choice(NAME_ONSETS) + rng.choice(NAME_VOWELS) + rng.choice(NAME_CODAS)
            for _ in range(2)
        ).capitalize()
        if 4 <= len(name) <= 8 and name not in used and name.lower() not in FICTIONAL_CONCEPTS:
            used.add(name)
            return name


def _sample_prontoqa_pairs(
    rng: random.Random, shots: int
) -> list[tuple[str, str]]:
    if shots * 2 > len(USABLE_CONCEPT_FAMILIES):
        raise ValueError("PrOntoQA shots exceed the disjoint concept-family capacity")
    family_indices = list(range(len(USABLE_CONCEPT_FAMILIES)))
    source_indices = rng.sample(family_indices, shots)
    remaining = [index for index in family_indices if index not in source_indices]
    target_indices = rng.sample(remaining, shots)
    return [
        (
            rng.choice(USABLE_CONCEPT_FAMILIES[source_index]),
            rng.choice(USABLE_CONCEPT_FAMILIES[target_index]),
        )
        for source_index, target_index in zip(
            source_indices, target_indices, strict=True
        )
    ]


@dataclass
class PrOntoQAFamily:
    family_index: int
    rng: random.Random
    shots: int
    query_style: str

    def __post_init__(self) -> None:
        self.family_id = f"prontoqa_family_{self.family_index:05d}"
        self.num_support_slots = self.shots
        self.used_names: set[str] = set()
        support_names = self.rng.sample(SUPPORT_ENTITY_NAMES, self.shots)
        self.used_names.update(support_names)
        pairs = _sample_prontoqa_pairs(self.rng, self.shots)
        self.icl_examples = []
        for shot_index, ((source, target), entity) in enumerate(
            zip(pairs, support_names, strict=True)
        ):
            x = f"{entity} is a {source}. Every {source} is a {target}. {entity} is a"
            self.icl_examples.append(
                {
                    "example_id": f"{self.family_id}_shot_{shot_index:02d}",
                    "entity": entity,
                    "source_concept": source,
                    "target_concept": target,
                    "x": x,
                    "y": target,
                    "full_qa": f"Q: {x}\nA: {target}",
                    "rule": f"Every {source} is a {target}.",
                }
            )
        self.context_prompt = "\n\n".join(
            str(example["full_qa"]) for example in self.icl_examples
        )

    def generate(self, candidate_index: int) -> dict[str, Any]:
        support_index = candidate_index % self.shots
        support = self.icl_examples[support_index]
        entity = _make_name(self.rng, self.used_names)
        source = str(support["source_concept"])
        target = str(support["target_concept"])
        if self.query_style == "rule_completion":
            x = f"{entity} is a {source}. Every {source} is a"
            induction_prefix = f"Every {source} is a"
        elif self.query_style == "entity_completion":
            x = f"{entity} is a {source}. {entity} is a"
            induction_prefix = None
        else:
            raise ValueError(f"Unsupported PrOntoQA query style: {self.query_style}")
        prompt = f"{self.context_prompt}\n\nQ: {x}\nA:"
        return _base_example(
            task="prontoqa",
            family_id=self.family_id,
            family_index=self.family_index,
            candidate_index=candidate_index,
            query_style=self.query_style,
            prompt=prompt,
            answer=target,
            x=x,
            entity=entity,
            source_concept=source,
            target_concept=target,
            support_example_index=support_index,
            lookup_rule=str(support["rule"]),
            induction_prefix=induction_prefix,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "context_id": self.family_id,
            "task_name": "prontoqa",
            "num_support_slots": self.num_support_slots,
            "query_style": self.query_style,
            "shots": self.shots,
            "icl_examples": self.icl_examples,
            "context_prompt": self.context_prompt,
        }


def build_families(
    *,
    task: str,
    num_families: int,
    seed: int,
    words: list[str],
    pattern_length: int,
    random_gap_length: int,
    wc_demo_per_group: int,
    wc_features_per_group: int,
    wc_groups: int,
    wc_random_pool: int,
    wc_distractors: int,
    tt_demos: int,
    prontoqa_shots: int,
    prontoqa_query_style: str,
) -> list[FamilyGenerator]:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {TASKS}")
    families: list[FamilyGenerator] = []
    for family_index in range(num_families):
        rng = random.Random(_family_seed(seed, task, family_index))
        if task == "lsc":
            family: FamilyGenerator = LSCFamily(
                family_index, rng, words, pattern_length, random_gap_length
            )
        elif task == "wc":
            family = WCFamily(
                family_index,
                rng,
                words,
                wc_demo_per_group,
                wc_features_per_group,
                wc_groups,
                wc_random_pool,
                wc_distractors,
            )
        elif task == "tt":
            family = TTFamily(family_index, rng, tt_demos)
        else:
            family = PrOntoQAFamily(
                family_index, rng, prontoqa_shots, prontoqa_query_style
            )
        families.append(family)
    return families
