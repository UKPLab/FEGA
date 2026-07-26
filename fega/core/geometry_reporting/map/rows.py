from __future__ import annotations

from typing import Any

from fega.core.geometry_reporting.diagnostics import (
    gate_summary_for_record,
    strongest_blocker,
    zoom_tags_for_record,
)
from fega.core.geometry_reporting.map.schema import MAP_VECTOR_KEYS
from fega.core.geometry_reporting.map.utils import finite_float


def feature_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build map rows solely from canonical classified final-residual records."""
    # Ignore non-scientific caller context and source every vector coordinate from the record.
    rows = []
    for record in records:
        vector = {}
        missingness = {}
        for key in MAP_VECTOR_KEYS:
            value = record.get(key)
            if key == "assignment_stability" and isinstance(value, dict):
                value = value.get("value")
            parsed = finite_float(value)
            vector[key] = parsed
            missingness[key] = parsed is None
        rows.append(
            {
                "feature_id": int(record["feature_id"]),
                "primary_label": record.get("primary_label"),
                "selected_k": record.get("selected_k"),
                "span_selected_k": record.get("span_selected_k"),
                "residual_selected_k": record.get("residual_selected_k"),
                "secondary_flags": list(record.get("secondary_flags") or []),
                "flag_details": record.get("flag_details") or {},
                "candidate_labels": list(record.get("candidate_labels") or []),
                "strict_gate_label": record.get("strict_gate_label"),
                "label_confidence": record.get("label_confidence"),
                "evidence_status": record.get("evidence_status"),
                "selected_family_decision": (
                    record.get("selected_family_stability") or {}
                ).get("decision"),
                "terminal_reason": record.get("terminal_reason"),
                "global_flags": list(record.get("global_flags") or []),
                "global_flag_count": int(record.get("global_flag_count") or 0),
                "global_flag_mask": str(record.get("global_flag_mask") or ""),
                "n_valid": finite_float(record.get("n_valid")),
                "m_median": finite_float(record.get("m_median")),
                "gate_summary": gate_summary_for_record(record),
                "strongest_blocker": strongest_blocker(record),
                "zoom_tags": zoom_tags_for_record(record),
                "vector": vector,
                "missingness": missingness,
            }
        )
    return rows
