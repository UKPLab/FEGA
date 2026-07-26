from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting.map.render import representative_feature_ids
from fega.core.geometry_reporting.map.schema import (
    GEOMETRY_REPORT_LABELS,
    LABEL_INTERPRETATIONS,
    PRIMARY_LABELS,
    SECONDARY_FLAGS,
)
from fega.core.geometry_reporting.map.utils import (
    format_optional_float,
    median_record_value,
)
from fega.paths import geometry_reporting_counts_path, geometry_reporting_stats_path


def write_stats_artifacts(
    config: FEGAPipelineConfig,
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    embedding_metadata: dict[str, Any],
) -> tuple[Path, Path]:
    stats_path = geometry_reporting_stats_path(config)
    counts_path = geometry_reporting_counts_path(config)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    counts = aggregate_counts(records)
    write_counts_csv(counts_path, counts)
    stats_path.write_text(
        stats_markdown(config, records, rows, counts, embedding_metadata)
    )
    return stats_path, counts_path


def aggregate_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(records)
    buckets = {
        "primary_label": {},
        "secondary_flag": {},
        "selected_k": {},
        "primary_selected_k": {},
        "span_selected_k": {},
        "residual_selected_k": {},
        "terminal_reason": {},
        "candidate_label": {},
        "global_flag": {},
        "global_flag_mask": {},
        "global_flag_count": {},
    }
    for record in records:
        label = str(record.get("primary_label"))
        _inc(buckets["primary_label"], label)
        for flag in record.get("secondary_flags") or []:
            _inc(buckets["secondary_flag"], str(flag))
        k_value = record.get("selected_k")
        if k_value is not None:
            _inc(buckets["selected_k"], str(k_value))
            _inc(buckets["primary_selected_k"], str(k_value))
        _add_selected_k_counts(record, label, k_value, buckets)
        terminal_reason = record.get("terminal_reason")
        if terminal_reason is not None:
            _inc(buckets["terminal_reason"], str(terminal_reason))
        for candidate in record.get("candidate_labels") or []:
            if isinstance(candidate, dict):
                _inc(buckets["candidate_label"], str(candidate.get("family")))
        for flag in record.get("global_flags") or []:
            _inc(buckets["global_flag"], str(flag))
        _inc(buckets["global_flag_mask"], str(record.get("global_flag_mask") or "none"))
        _inc(buckets["global_flag_count"], str(int(record.get("global_flag_count") or 0)))
    rows: list[dict[str, Any]] = []
    for kind, mapping in buckets.items():
        for name, count in sorted(mapping.items()):
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "count": count,
                    "fraction": 0.0 if total == 0 else count / total,
                }
            )
    return rows


def write_counts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "name", "count", "fraction"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "kind": row["kind"],
                    "name": row["name"],
                    "count": row["count"],
                    "fraction": f"{float(row['fraction']):.12g}",
                }
            )


def stats_markdown(
    config: FEGAPipelineConfig,
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    counts: list[dict[str, Any]],
    embedding_metadata: dict[str, Any],
) -> str:
    total = len(records)
    primary_counts = counts_by_kind(counts, "primary_label")
    secondary_counts = counts_by_kind(counts, "secondary_flag")
    representatives = sorted(representative_feature_ids(rows))
    sections = [
        "# Geometry Reporting Statistics",
        "",
        "## Run Metadata",
        "",
        f"- threshold_profile: {config.phases.geometry_reporting.threshold_profile}",
        f"- embedding_requested: {config.phases.geometry_reporting.embedding}",
        f"- embedding_resolved: {embedding_metadata.get('method')}",
        f"- total_features: {total}",
        "",
        "## Primary Label Counts",
        "",
        *(_defined_count_lines(primary_counts, PRIMARY_LABELS, total) or ["- none"]),
        "",
        "## Secondary Flag Counts",
        "",
        *(_defined_count_lines(secondary_counts, SECONDARY_FLAGS, total) or ["- none"]),
        "",
        "## FEGA Geometry Label Interpretations",
        "",
        *_label_interpretation_lines(primary_counts, secondary_counts, total),
    ]
    for title, kind in (
        ("Selected-K Counts", "selected_k"),
        ("Primary Selected-K Counts", "primary_selected_k"),
        ("Span Selected-K Counts", "span_selected_k"),
        ("Residual Selected-K Counts", "residual_selected_k"),
        ("Terminal Reason Counts", "terminal_reason"),
        ("Candidate Label Counts", "candidate_label"),
        ("Global Flag Counts", "global_flag"),
        ("Global Flag Mask Counts", "global_flag_mask"),
    ):
        sections.extend(["", f"## {title}", "", *(count_lines(counts, kind) or ["- none"])])
    sections.extend(
        [
            "",
            "## Medians",
            "",
            f"- median n_valid: {format_optional_float(median_record_value(records, 'n_valid'))}",
            f"- median m_median: {format_optional_float(median_record_value(records, 'm_median'))}",
            "",
            "## Representative Feature IDs",
            "",
            "- " + ", ".join(f"f{feature_id}" for feature_id in representatives)
            if representatives
            else "- none",
            "",
        ]
    )
    return "\n".join(sections)


def counts_by_kind(counts: list[dict[str, Any]], kind: str) -> dict[str, int]:
    return {
        str(row["name"]): int(row["count"])
        for row in counts
        if row["kind"] == kind
    }


def count_lines(counts: list[dict[str, Any]], kind: str) -> list[str]:
    return [
        f"- {row['name']}: {row['count']} ({float(row['fraction']) * 100:.1f}%)"
        for row in counts
        if row["kind"] == kind
    ]


def _add_selected_k_counts(
    record: dict[str, Any],
    label: str,
    k_value: Any,
    buckets: dict[str, dict[str, int]],
) -> None:
    span_k = record.get("span_selected_k")
    if span_k is None and label in {
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
    }:
        span_k = k_value
    if span_k is not None:
        _inc(buckets["span_selected_k"], str(span_k))
    residual_k = record.get("residual_selected_k")
    if residual_k is None and label == "residual_lowD_k":
        residual_k = k_value
    if residual_k is not None:
        _inc(buckets["residual_selected_k"], str(residual_k))


def _defined_count_lines(
    counts: dict[str, int], labels: tuple[str, ...], total: int
) -> list[str]:
    return [
        f"- {label}: {counts.get(label, 0)} ({_percentage(counts.get(label, 0), total)})"
        for label in labels
    ]


def _label_interpretation_lines(
    primary_counts: dict[str, int], secondary_counts: dict[str, int], total: int
) -> list[str]:
    lines = []
    for label in GEOMETRY_REPORT_LABELS:
        kind = "primary label" if label in PRIMARY_LABELS else "secondary flag"
        count = (
            primary_counts.get(label, 0)
            if label in PRIMARY_LABELS
            else secondary_counts.get(label, 0)
        )
        lines.append(
            f"- `{label}` ({kind}, {count}, {_percentage(count, total)}): "
            f"{LABEL_INTERPRETATIONS[label]}"
        )
    return lines


def _inc(bucket: dict[str, int], key: str) -> None:
    bucket[key] = bucket.get(key, 0) + 1


def _percentage(count: int, total: int) -> str:
    return f"{0.0 if total == 0 else (count / total) * 100:.1f}%"
