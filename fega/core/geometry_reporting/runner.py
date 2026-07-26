from __future__ import annotations

import csv
import json
import logging
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting.artifacts import (
    load_geometry_inputs,
    write_json_atomic,
)
from fega.core.geometry_reporting.classifier import LABEL_VERSION, classify_record
from fega.core.geometry_reporting.diagnostics import write_gate_diagnostics
from fega.core.geometry_reporting.maps import write_geometry_maps
from fega.core.geometry_reporting.records import build_geometry_records
from fega.core.resources import ModelResources
from fega.paths import (
    geometry_reporting_gate_diagnostics_json_path,
    geometry_reporting_gate_diagnostics_md_path,
    geometry_reporting_records_csv_path,
    geometry_reporting_records_path,
)

_logger = logging.getLogger(__name__)
_SCHEMA_VERSION = 2


def run_geometry_reporting(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    """Run FEGA geometry classification over cached artifacts."""
    inputs = load_geometry_inputs(config, resources)
    raw_records, summary = build_geometry_records(inputs, config)
    classified = [
        classify_record(record, config.phases.geometry_reporting.threshold_profile)
        for record in raw_records
    ]
    payload = {
        "phase": "geometry_reporting",
        "schema_version": _SCHEMA_VERSION,
        "canonical_source_fingerprint": inputs[
            "canonical_source_fingerprint"
        ],
        "label_version": LABEL_VERSION,
        "threshold_profile": config.phases.geometry_reporting.threshold_profile,
        "config": config.to_dict()["phases"]["geometry_reporting"],
        "source_paths": summary["source_paths"],
        "summary": {
            "features_total": summary["features_total"],
            "primary_label_counts": _label_counts(classified),
            "terminal_reason_counts": _terminal_reason_counts(classified),
            "global_flag_counts": _global_flag_counts(classified),
        },
        "diagnostics_paths": {
            "gate_diagnostics_json": str(
                geometry_reporting_gate_diagnostics_json_path(config)
            ),
            "gate_diagnostics_markdown": str(
                geometry_reporting_gate_diagnostics_md_path(config)
            ),
        },
        "features": classified,
    }
    output_path = write_json_atomic(geometry_reporting_records_path(config), payload)
    write_gate_diagnostics(config, classified)
    if config.phases.geometry_reporting.write_csv:
        _write_records_csv(geometry_reporting_records_csv_path(config), classified)
    if config.phases.geometry_reporting.map_enabled:
        write_geometry_maps(config, classified)
    _logger.info("geometry_reporting complete: path=%s", output_path)


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        label = str(record.get("primary_label"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _terminal_reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = record.get("terminal_reason")
        if reason is None:
            continue
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _global_flag_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for flag in record.get("global_flags") or []:
            key = str(flag)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _write_records_csv(path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_id",
        "primary_label",
        "selected_k",
        "span_selected_k",
        "residual_selected_k",
        "strict_gate_label",
        "label_confidence",
        "evidence_status",
        "selected_family_decision",
        "terminal_reason",
        "secondary_flags",
        "global_flags",
        "global_flag_count",
        "global_flag_mask",
        "candidate_labels",
        "threshold_profile",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "feature_id": record.get("feature_id"),
                    "primary_label": record.get("primary_label"),
                    "selected_k": record.get("selected_k"),
                    "span_selected_k": record.get("span_selected_k"),
                    "residual_selected_k": record.get("residual_selected_k"),
                    "strict_gate_label": record.get("strict_gate_label"),
                    "label_confidence": record.get("label_confidence"),
                    "evidence_status": record.get("evidence_status"),
                    "selected_family_decision": (
                        record.get("selected_family_stability") or {}
                    ).get("decision"),
                    "terminal_reason": record.get("terminal_reason"),
                    "secondary_flags": ",".join(record.get("secondary_flags") or []),
                    "global_flags": ",".join(record.get("global_flags") or []),
                    "global_flag_count": record.get("global_flag_count"),
                    "global_flag_mask": record.get("global_flag_mask"),
                    "candidate_labels": json.dumps(
                        record.get("candidate_labels") or [], sort_keys=True
                    ),
                    "threshold_profile": record.get("threshold_profile"),
                }
            )
