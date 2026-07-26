from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting.artifacts import write_json_atomic
from fega.core.geometry_reporting.classifier import GLOBAL_FLAG_ORDER, TERMINAL_LABELS
from fega.paths import (
    geometry_reporting_gate_diagnostics_json_path,
    geometry_reporting_gate_diagnostics_md_path,
)

GATE_KEYS = (
    "directed_ray",
    "axis_or_antipodal",
    "multi_mode_directional_geometry",
    "global_directional_subspace",
    "residual_lowD_k",
)
DECISIONS = ("stable", "exploratory", "unstable", "not_available")
SPAN_KS = ("2", "3", "4", "8")


def write_gate_diagnostics(
    config: FEGAPipelineConfig, records: list[dict[str, Any]]
) -> dict[str, Path]:
    payload = build_gate_diagnostics(records)
    json_path = write_json_atomic(
        geometry_reporting_gate_diagnostics_json_path(config), payload
    )
    md_path = geometry_reporting_gate_diagnostics_md_path(config)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_diagnostics_markdown(payload))
    return {"json": json_path, "markdown": md_path}


def build_gate_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    gate_decisions = {
        gate: _count_gate_field(records, gate, "decision", DECISIONS)
        for gate in GATE_KEYS
    }
    top_blockers = {
        gate: _top_blockers(records, gate)
        for gate in GATE_KEYS
    }
    metric_pass_stability_missing = {
        gate: _metric_pass_stability_missing(records, gate)
        for gate in GATE_KEYS
    }
    zoom_sets = _zoom_feature_sets(records)
    return {
        "phase": "geometry_reporting",
        "schema_version": 1,
        "features_total": len(records),
        "primary_label_counts": _record_field_counts(records, "primary_label"),
        "strict_gate_label_counts": _record_field_counts(records, "strict_gate_label"),
        "candidate_label_counts": _candidate_family_counts(records),
        "terminal_reason_counts": _nonempty_record_field_counts(
            records, "terminal_reason"
        ),
        "global_flag_counts": _global_flag_counts(records),
        "global_flag_mask_counts": _record_field_counts(records, "global_flag_mask"),
        "global_flag_count_distribution": _record_field_counts(
            records, "global_flag_count"
        ),
        "failed_field_counts": _candidate_field_counts(records, "failed_fields"),
        "missing_field_counts": _candidate_field_counts(records, "missing_fields"),
        "prevented_high_dimensional_fallback_count": (
            _prevented_high_dimensional_fallback_count(records)
        ),
        "gate_decision_counts": gate_decisions,
        "top_blockers": top_blockers,
        "span_attempts_by_k": _span_attempts_by_k(records),
        "metric_pass_stability_missing": metric_pass_stability_missing,
        "unresolved_breakdown": _unresolved_breakdown(records),
        "feature_sets": zoom_sets,
    }


def gate_summary_for_record(record: dict[str, Any]) -> dict[str, Any]:
    gates = _gates(record)
    span_attempts = _attempts(gates.get("global_directional_subspace"))
    return {
        "decisions": {
            gate: _field(gates.get(gate), "decision")
            for gate in GATE_KEYS
        },
        "span_attempt_decisions": {
            k: _field(span_attempts.get(k), "decision")
            for k in SPAN_KS
        },
        "tooltip_fields": {
            "directed_ray.decision": _field(gates.get("directed_ray"), "decision"),
            "span.k2.decision": _field(span_attempts.get("2"), "decision"),
            "residual.decision": _field(gates.get("residual_lowD_k"), "decision"),
            "strict_gate_label": str(record.get("strict_gate_label")),
            "terminal_reason": str(record.get("terminal_reason")),
            "global_flag_mask": str(record.get("global_flag_mask") or ""),
        },
    }


def strongest_blocker(record: dict[str, Any]) -> str:
    primary_label = str(record.get("primary_label"))
    if primary_label not in TERMINAL_LABELS | {
        "unresolved_high_dimensional_or_diffuse"
    }:
        return "classified"
    if primary_label in TERMINAL_LABELS:
        reason = record.get("terminal_reason")
        return str(reason) if reason is not None else primary_label
    if _has_span_metric_pass_stability_missing(record):
        return "span_metric_pass_stability_missing"
    if _has_near_directed_ray_ci_failed(record):
        return "near_directed_ray_ci_failed"
    if "long_tail_spectrum" in set(record.get("secondary_flags") or []):
        return "unresolved_long_tail"
    gates = _gates(record)
    for gate in GATE_KEYS:
        blockers = _blocked_reasons(gates.get(gate))
        if blockers:
            return f"{gate}:{blockers[0]}"
    return "unresolved_no_specific_blocker"


def zoom_tags_for_record(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if _has_span_metric_pass_stability_missing(record):
        tags.append("span_metric_pass_stability_missing")
    if (
        record.get("primary_label") == "unresolved_high_dimensional_or_diffuse"
        and "long_tail_spectrum" in set(record.get("secondary_flags") or [])
    ):
        tags.append("unresolved_long_tail")
    if _has_near_directed_ray_ci_failed(record):
        tags.append("near_directed_ray_ci_failed")
    if record.get("primary_label") == "oneD_diffuse":
        tags.append("oneD_diffuse")
    if _prevented_high_dimensional_fallback(record):
        tags.append("prevented_high_dimensional_fallback")
    return tags


def _gates(record: dict[str, Any]) -> dict[str, Any]:
    gates = record.get("gate_evidence")
    return gates if isinstance(gates, dict) else {}


def _attempts(gate: Any) -> dict[str, Any]:
    if not isinstance(gate, dict):
        return {}
    attempts = gate.get("attempts")
    return attempts if isinstance(attempts, dict) else {}


def _field(block: Any, name: str) -> str:
    if not isinstance(block, dict):
        return "not_available"
    value = block.get(name)
    return str(value) if value is not None else "not_available"


def _blocked_reasons(block: Any) -> list[str]:
    if not isinstance(block, dict):
        return []
    reasons = block.get("blocked_reasons")
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons if reason is not None]


def _count_gate_field(
    records: list[dict[str, Any]],
    gate: str,
    field: str,
    known_values: tuple[str, ...] = (),
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[_field(_gates(record).get(gate), field)] += 1
    return _ordered_counts(counts, known_values)


def _ordered_counts(
    counts: Counter[str], known_values: tuple[str, ...] = ()
) -> dict[str, int]:
    ordered = {value: int(counts.get(value, 0)) for value in known_values}
    for value, count in sorted(counts.items()):
        if value not in ordered:
            ordered[value] = int(count)
    return ordered


def _record_field_counts(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(field)
        key = "none" if value is None or value == "" else str(value)
        counts[key] += 1
    return dict(sorted((key, int(value)) for key, value in counts.items()))


def _nonempty_record_field_counts(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        value = record.get(field)
        if value is None or value == "":
            continue
        counts[str(value)] += 1
    return dict(sorted((key, int(value)) for key, value in counts.items()))


def _candidate_family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for candidate in _candidates(record):
            counts[str(candidate.get("family"))] += 1
    return dict(sorted((key, int(value)) for key, value in counts.items()))


def _candidate_field_counts(
    records: list[dict[str, Any]], field: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for candidate in _candidates(record):
            values = candidate.get(field)
            if not isinstance(values, list):
                continue
            for value in values:
                counts[str(value)] += 1
    return dict(sorted((key, int(value)) for key, value in counts.items()))


def _global_flag_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        for flag in record.get("global_flags") or []:
            counts[str(flag)] += 1
    ordered = {flag: int(counts.get(flag, 0)) for flag in GLOBAL_FLAG_ORDER}
    for flag, count in sorted(counts.items()):
        if flag not in ordered:
            ordered[flag] = int(count)
    return ordered


def _candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = record.get("candidate_labels")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _top_blockers(
    records: list[dict[str, Any]], gate: str
) -> list[dict[str, int | str]]:
    counts: Counter[str] = Counter()
    for record in records:
        for reason in _blocked_reasons(_gates(record).get(gate)):
            counts[reason] += 1
    return [
        {"reason": reason, "count": int(count)}
        for reason, count in counts.most_common()
    ]


def _metric_pass_stability_missing(records: list[dict[str, Any]], gate: str) -> int:
    count = 0
    for record in records:
        evidence = _gates(record).get(gate)
        if (
            isinstance(evidence, dict)
            and evidence.get("metric_status") == "stable"
            and evidence.get("subspace_status") == "not_available"
        ):
            count += 1
    return count


def _span_attempts_by_k(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in SPAN_KS:
        decision_counts: Counter[str] = Counter()
        metric_counts: Counter[str] = Counter()
        subspace_counts: Counter[str] = Counter()
        sample_counts: Counter[str] = Counter()
        metric_pass_missing = 0
        attempted = 0
        for record in records:
            attempts = _attempts(_gates(record).get("global_directional_subspace"))
            attempt = attempts.get(k)
            if not isinstance(attempt, dict):
                continue
            attempted += 1
            decision_counts[_field(attempt, "decision")] += 1
            metric_counts[_field(attempt, "metric_status")] += 1
            subspace_counts[_field(attempt, "subspace_status")] += 1
            sample_counts[_field(attempt, "sample_size_status")] += 1
            if (
                attempt.get("metric_status") == "stable"
                and attempt.get("subspace_status") == "not_available"
            ):
                metric_pass_missing += 1
        result[k] = {
            "attempted": attempted,
            "decision_counts": _ordered_counts(decision_counts, DECISIONS),
            "metric_status_counts": _ordered_counts(metric_counts),
            "subspace_status_counts": _ordered_counts(subspace_counts),
            "sample_size_status_counts": _ordered_counts(sample_counts),
            "metric_pass_stability_missing": metric_pass_missing,
        }
    return result


def _unresolved_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("primary_label") == "unresolved_high_dimensional_or_diffuse":
            counts[strongest_blocker(record)] += 1
    return dict(sorted(counts.items()))


def _prevented_high_dimensional_fallback_count(
    records: list[dict[str, Any]],
) -> int:
    return sum(1 for record in records if _prevented_high_dimensional_fallback(record))


def _prevented_high_dimensional_fallback(record: dict[str, Any]) -> bool:
    primary_label = record.get("primary_label")
    if primary_label in TERMINAL_LABELS or (
        primary_label == "unresolved_high_dimensional_or_diffuse"
    ):
        return False
    if record.get("strict_gate_label") is not None:
        return False
    return any(
        candidate.get("family") == primary_label for candidate in _candidates(record)
    )


def _zoom_feature_sets(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    feature_sets = {
        "span_metric_pass_stability_missing": [],
        "unresolved_long_tail": [],
        "near_directed_ray_ci_failed": [],
        "oneD_diffuse": [],
        "prevented_high_dimensional_fallback": [],
    }
    for record in records:
        feature_id = record.get("feature_id")
        if feature_id is None:
            continue
        for tag in zoom_tags_for_record(record):
            feature_sets[tag].append(int(feature_id))
    return feature_sets


def _has_span_metric_pass_stability_missing(record: dict[str, Any]) -> bool:
    attempts = _attempts(_gates(record).get("global_directional_subspace"))
    return any(
        isinstance(attempt, dict)
        and attempt.get("metric_status") == "stable"
        and attempt.get("subspace_status") == "not_available"
        for attempt in attempts.values()
    )


def _has_near_directed_ray_ci_failed(record: dict[str, Any]) -> bool:
    directed = _gates(record).get("directed_ray")
    return (
        isinstance(directed, dict)
        and directed.get("metric_status") == "stable"
        and directed.get("scalar_ci_status") == "unstable"
    )


def _diagnostics_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Geometry Gate Diagnostics",
        "",
        f"- features_total: {payload['features_total']}",
        "- prevented_high_dimensional_fallback_count: "
        f"{payload['prevented_high_dimensional_fallback_count']}",
        "",
        "## Label And Candidate Counts",
        "",
        f"- primary_label_counts: {payload['primary_label_counts']}",
        f"- strict_gate_label_counts: {payload['strict_gate_label_counts']}",
        f"- candidate_label_counts: {payload['candidate_label_counts']}",
        f"- terminal_reason_counts: {payload['terminal_reason_counts']}",
        "",
        "## Global Flags",
        "",
        f"- global_flag_counts: {payload['global_flag_counts']}",
        f"- global_flag_mask_counts: {payload['global_flag_mask_counts']}",
        f"- global_flag_count_distribution: "
        f"{payload['global_flag_count_distribution']}",
        "",
        "## Candidate Blocker Fields",
        "",
        f"- failed_field_counts: {payload['failed_field_counts']}",
        f"- missing_field_counts: {payload['missing_field_counts']}",
        "",
        "## Gate Decision Counts",
        "",
    ]
    for gate, counts in payload["gate_decision_counts"].items():
        rendered = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(f"- {gate}: {rendered}")
    lines.extend(["", "## Top Blockers", ""])
    for gate, blockers in payload["top_blockers"].items():
        if blockers:
            rendered = ", ".join(
                f"{item['reason']}={item['count']}" for item in blockers[:8]
            )
        else:
            rendered = "none"
        lines.append(f"- {gate}: {rendered}")
    lines.extend(["", "## Span Attempts By K", ""])
    for k, block in payload["span_attempts_by_k"].items():
        lines.append(
            f"- k={k}: attempted={block['attempted']}, "
            f"metric_pass_stability_missing={block['metric_pass_stability_missing']}, "
            f"decisions={block['decision_counts']}"
        )
    lines.extend(["", "## Metric-Pass But Stability-Missing", ""])
    for gate, count in payload["metric_pass_stability_missing"].items():
        lines.append(f"- {gate}: {count}")
    lines.extend(["", "## Unresolved Breakdown", ""])
    unresolved = payload["unresolved_breakdown"]
    if unresolved:
        for reason, count in unresolved.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Zoom Feature Sets", ""])
    for name, feature_ids in payload["feature_sets"].items():
        rendered = ", ".join(f"f{feature_id}" for feature_id in feature_ids[:50])
        if len(feature_ids) > 50:
            rendered += f", ... ({len(feature_ids)} total)"
        lines.append(f"- {name}: {rendered or 'none'}")
    lines.append("")
    return "\n".join(lines)
