from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

ScoreRecord = Dict[str, float | None]
PromptKey = Tuple[str, str | None, str | None, str | None]


@dataclass
class ScoreMaps:
    by_pair_index: Dict[int, ScoreRecord]
    by_prompt: Dict[PromptKey, ScoreRecord]
    stats: Dict[str, int]


def normalize_prompt(text: str) -> str:
    """Normalize prompt text for robust joining."""
    return " ".join(str(text).split())


def _coerce_score(value) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score == -1:
        return None
    return score


def _coerce_index(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_prompt(raw: Dict) -> str | None:
    for key in ("base_prompt", "prompt", "base_text"):
        if raw.get(key):
            return str(raw[key])
    return None


def _coerce_label(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _has_all_labels(
    entity_label: str | None, attribute_type: str | None, attribute_label: str | None
) -> bool:
    return (
        entity_label is not None
        and attribute_type is not None
        and attribute_label is not None
    )


def _prompt_key(
    prompt: str,
    entity_label: str | None,
    attribute_type: str | None,
    attribute_label: str | None,
) -> PromptKey:
    return (
        normalize_prompt(prompt),
        entity_label,
        attribute_type,
        attribute_label,
    )


def _find_attr_block(payload: Dict, entity: str, attribute: str) -> Dict:
    unstructured = payload.get("eval_result_unstructured", {})
    entity_key = f"{entity}_results"
    attr_key = f"{entity}_{attribute}"
    direct = unstructured.get(entity_key, {})
    if isinstance(direct, dict) and attr_key in direct:
        return direct.get(attr_key) or {}
    per_class = unstructured.get("per_class", {}).get(entity_key, {})
    if isinstance(per_class, dict) and attr_key in per_class:
        return per_class.get(attr_key) or {}
    return {}


def _update_accum(accum: Dict, key, cause: float | None, iso: float | None) -> None:
    if cause is None and iso is None:
        return
    entry = accum.setdefault(
        key,
        {"cause_sum": 0.0, "cause_count": 0, "iso_sum": 0.0, "iso_count": 0},
    )
    if cause is not None:
        entry["cause_sum"] += float(cause)
        entry["cause_count"] += 1
    if iso is not None:
        entry["iso_sum"] += float(iso)
        entry["iso_count"] += 1


def _finalize_accum(accum: Dict) -> Dict:
    result: Dict = {}
    for key, entry in accum.items():
        cause_mean = (
            entry["cause_sum"] / entry["cause_count"]
            if entry["cause_count"]
            else None
        )
        iso_mean = (
            entry["iso_sum"] / entry["iso_count"] if entry["iso_count"] else None
        )
        result[key] = {"cause_score": cause_mean, "isolation_score": iso_mean}
    return result


def extract_instance_scores_from_payload(
    payload: Dict[str, Any], entity: str, attribute: str
) -> ScoreMaps:
    attr_block = _find_attr_block(payload, entity, attribute)
    raw_cause_instances = attr_block.get("cause_instances")
    raw_isolation_instances = attr_block.get("isolation_instances")

    stats = {
        "cause_instances": 0,
        "isolation_instances": 0,
        "instances_missing_prompt": 0,
        "instances_missing_pair_index": 0,
        "instances_missing_scores": 0,
        "instances_not_list": 0,
        "pair_index_duplicates": 0,
        "prompt_duplicates": 0,
    }

    cause_instances = (
        raw_cause_instances if isinstance(raw_cause_instances, list) else []
    )
    isolation_instances = (
        raw_isolation_instances if isinstance(raw_isolation_instances, list) else []
    )
    if raw_cause_instances is not None and not isinstance(raw_cause_instances, list):
        stats["instances_not_list"] += 1
    if raw_isolation_instances is not None and not isinstance(
        raw_isolation_instances, list
    ):
        stats["instances_not_list"] += 1
    stats["cause_instances"] = len(cause_instances)
    stats["isolation_instances"] = len(isolation_instances)

    pair_accum: Dict[int, Dict] = {}
    prompt_accum: Dict[PromptKey, Dict] = {}

    def process_instances(instances) -> None:
        for inst in instances:
            cause_score = _coerce_score(inst.get("cause_score"))
            iso_score = _coerce_score(inst.get("isolation_score"))
            if cause_score is None and iso_score is None:
                stats["instances_missing_scores"] += 1
                continue
            idx = _coerce_index(
                inst.get("pair_id") or inst.get("pair_index") or inst.get("index")
            )
            if idx is not None:
                _update_accum(pair_accum, idx, cause_score, iso_score)
            else:
                stats["instances_missing_pair_index"] += 1
            prompt = _extract_prompt(inst)
            if prompt:
                entity_label = _coerce_label(
                    inst.get("base_entity_label") or inst.get("entity_label")
                )
                attribute_type = _coerce_label(
                    inst.get("base_attribute_type") or inst.get("attribute_type")
                )
                attribute_label = _coerce_label(
                    inst.get("base_attribute_label") or inst.get("attribute_label")
                )
                if _has_all_labels(entity_label, attribute_type, attribute_label):
                    key = _prompt_key(
                        prompt, entity_label, attribute_type, attribute_label
                    )
                else:
                    key = _prompt_key(prompt, None, None, None)
                _update_accum(prompt_accum, key, cause_score, iso_score)
            else:
                stats["instances_missing_prompt"] += 1

    process_instances(cause_instances)
    process_instances(isolation_instances)

    by_pair_index = _finalize_accum(pair_accum)
    by_prompt = _finalize_accum(prompt_accum)

    stats["pair_index_duplicates"] = sum(
        1
        for entry in pair_accum.values()
        if entry["cause_count"] + entry["iso_count"] > 1
    )
    stats["prompt_duplicates"] = sum(
        1
        for entry in prompt_accum.values()
        if entry["cause_count"] + entry["iso_count"] > 1
    )
    stats["pair_index_keys"] = len(by_pair_index)
    stats["prompt_keys"] = len(by_prompt)

    return ScoreMaps(
        by_pair_index=by_pair_index, by_prompt=by_prompt, stats=stats
    )


def extract_instance_scores(
    reference_json: Path, entity: str, attribute: str
) -> ScoreMaps:
    payload = json.loads(Path(reference_json).read_text())
    return extract_instance_scores_from_payload(payload, entity, attribute)


def annotate_contexts_with_scores(
    contexts: Dict[int, list[Dict]], score_maps: ScoreMaps
) -> Dict[str, int]:
    stats = {
        "contexts_total": 0,
        "contexts_with_any_score": 0,
        "contexts_with_cause_score": 0,
        "contexts_with_isolation_score": 0,
        "contexts_missing_scores": 0,
        "contexts_matched_pair_index": 0,
        "contexts_matched_prompt": 0,
    }

    use_pair = bool(score_maps.by_pair_index)
    use_prompt = bool(score_maps.by_prompt)

    for ctxs in contexts.values():
        for rec in ctxs:
            stats["contexts_total"] += 1
            cause_score = None
            isolation_score = None
            matched = False

            if use_pair:
                idx = _coerce_index(rec.get("pair_index"))
                if idx is not None:
                    scores = score_maps.by_pair_index.get(idx)
                    if scores is not None:
                        matched = True
                        stats["contexts_matched_pair_index"] += 1
                        cause_score = scores.get("cause_score")
                        isolation_score = scores.get("isolation_score")

            if not matched and use_prompt:
                prompt = rec.get("prompt")
                if prompt:
                    entity_label = _coerce_label(rec.get("entity_label"))
                    attribute_type = _coerce_label(rec.get("attribute_type"))
                    attribute_label = _coerce_label(rec.get("attribute_label"))
                    if _has_all_labels(entity_label, attribute_type, attribute_label):
                        key = _prompt_key(
                            prompt, entity_label, attribute_type, attribute_label
                        )
                    else:
                        key = _prompt_key(prompt, None, None, None)
                    scores = score_maps.by_prompt.get(key)
                    if scores is not None:
                        matched = True
                        stats["contexts_matched_prompt"] += 1
                        cause_score = scores.get("cause_score")
                        isolation_score = scores.get("isolation_score")

            if cause_score is not None or isolation_score is not None:
                stats["contexts_with_any_score"] += 1
            else:
                stats["contexts_missing_scores"] += 1
            if cause_score is not None:
                stats["contexts_with_cause_score"] += 1
            if isolation_score is not None:
                stats["contexts_with_isolation_score"] += 1

            rec["cause_score"] = cause_score
            rec["isolation_score"] = isolation_score

    return stats
