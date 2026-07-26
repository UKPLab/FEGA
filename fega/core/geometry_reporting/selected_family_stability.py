from __future__ import annotations

import copy
from typing import Any


def project_selected_family_stability(
    point_selection: dict[str, Any],
    evidence: dict[str, Any],
    *,
    existing_confidence: str | None,
) -> tuple[dict[str, Any], str | None, str, tuple[str, ...]]:
    """Project raw WP3 evidence through the approved closed WP4 state table.

    This function interprets only persisted counters and the existing low-context
    qualification. It does not recompute a margin, threshold, family, dimension,
    or protocol result.
    """
    # Aggregate raw denominators once while preserving every protocol below the trace.
    protocols = evidence.get("protocols")
    protocol_counters = evidence.get("protocol_counters")
    required_ids = evidence.get("required_protocol_ids")
    if not isinstance(protocols, dict) or not isinstance(protocol_counters, dict):
        raise ValueError("selected-family protocol evidence is missing")
    if not isinstance(required_ids, list) or not all(
        isinstance(item, str) for item in required_ids
    ):
        raise ValueError("selected-family required protocol IDs are missing")
    totals = {
        key: sum(
            int(counters.get(key, 0))
            for counters in protocol_counters.values()
            if isinstance(counters, dict)
        )
        for key in ("requested", "valid", "failed", "non_applicable", "skipped")
    }
    no_work_reason = evidence.get("no_work_reason")
    assignment_reuse = required_ids == ["standalone_assignment_reuse"]
    missing_required = any(protocol_id not in protocols for protocol_id in required_ids)
    completed_instability = int(evidence.get("completed_instability_count", 0))
    required_failures = int(evidence.get("required_failure_count", 0))
    unavailable = bool(
        required_failures
        or totals["failed"]
        or totals["skipped"]
        or missing_required
    )

    # Apply the exact precedence table: observed instability outranks unavailability.
    flags: tuple[str, ...]
    if no_work_reason is not None:
        decision = "not_evaluated"
        confidence = existing_confidence
        evidence_status = "not_evaluated"
        flags = ("selected_family_not_evaluated",)
    elif assignment_reuse:
        decision = "stable"
        confidence = "accepted"
        evidence_status = "available"
        flags = ()
    elif completed_instability > 0:
        decision = "unstable"
        confidence = "unstable"
        evidence_status = "unavailable" if unavailable else "available"
        flags = (
            ("selected_family_unstable", "selected_family_evidence_unavailable")
            if unavailable
            else ("selected_family_unstable",)
        )
    elif unavailable:
        decision = "unavailable"
        confidence = None
        evidence_status = "unavailable"
        flags = ("selected_family_evidence_unavailable",)
    else:
        low_context = protocols.get("low_context_qualification")
        exploratory = (
            isinstance(low_context, dict)
            and low_context.get("status") == "exploratory"
        )
        decision = "stable"
        confidence = "exploratory" if exploratory else "accepted"
        evidence_status = "available"
        flags = ()

    # Retain raw scientific evidence once in canonical JSON plus a compact projection.
    trace = {
        "family": point_selection["family"],
        "selected_k": point_selection["selected_k"],
        "selection_mode": point_selection["mode"],
        "point_reason": point_selection["point_reason"],
        "point_selection_contract_version": point_selection["contract_version"],
        "decision": decision,
        "evidence_status": evidence_status,
        "required_protocol_ids": list(required_ids),
        "requested_count": totals["requested"],
        "valid_count": totals["valid"],
        "failed_count": totals["failed"],
        "non_applicable_count": totals["non_applicable"],
        "skipped_count": totals["skipped"],
        "completed_instability_count": completed_instability,
        "required_failure_count": required_failures,
        "point_margins": copy.deepcopy(evidence.get("point_margins")),
        "executed_plan_digests": {
            protocol_id: protocol.get("plan_digest")
            for protocol_id, protocol in protocols.items()
            if isinstance(protocol, dict) and protocol.get("plan_digest") is not None
        },
        "protocols": copy.deepcopy(protocols),
        "no_work_reason": no_work_reason,
    }
    return trace, confidence, evidence_status, flags
