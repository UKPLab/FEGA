from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fega.core.geometry_metrics.metrics import (
    c_ray_pairwise_final_resid,
    effective_rank_from_spectrum,
    span_spectrum_final_resid,
)
from fega.core.geometry_reporting.thresholds import (
    get_threshold_profile,
    subspace_angle_threshold,
)
from fega.core.stability import runner as stability_runner
from fega.core.stability.metrics import (
    final_resid_unit_rows,
    scheduled_principal_angle_stability,
)
from fega.core.stability.protocols import signed_gate_margins
from fega.core.stability.sampling import build_subset_plan, low_context_protocol
from fega.core.stability.schedule import AngleResamplePlan

STRICT_K = (2, 3)
MARGIN_KEYS = (
    "s_span_2",
    "r_span_pr_k2",
    "u_span_2",
    "d_span_2",
    "s_span_3",
    "r_span_pr_k3",
    "u_span_3",
    "d_span_3",
)


def _config() -> SimpleNamespace:
    """Build the exact numerical controls consumed by selected-family execution."""
    # Keep the oracle independent of pipeline loading and schedule construction.
    metric = SimpleNamespace(eps=1.0e-12)
    return SimpleNamespace(
        phases=SimpleNamespace(
            geometry_metrics=SimpleNamespace(
                c_ray=metric, span=metric, resid=metric, effective_rank=metric
            ),
            stability=SimpleNamespace(
                scalar=SimpleNamespace(ci_quantiles=(0.25, 0.75)),
                subspace=SimpleNamespace(angle_p90_quantile=0.9, eig_floor=1.0e-8),
            ),
        )
    )


def _counts(**updates: int) -> dict[str, int]:
    """Return the closed protocol counter vocabulary without runner helpers."""
    # Preserve zero-valued fields so equality detects omitted failure accounting.
    counts = dict(requested=0, valid=0, failed=0, non_applicable=0, skipped=0)
    counts.update({key: int(value) for key, value in updates.items()})
    return counts


def _digest(plans: tuple[object, ...]) -> str:
    """Hash frozen plan identities in their declared order."""
    # Reproduce the artifact contract directly rather than using schedule utilities.
    return hashlib.sha256("".join(plan.digest for plan in plans).encode()).hexdigest()


def _passes(key: str, value: float) -> bool:
    """Apply the signed-margin boundary used by the reporting gates."""
    # The strict c-ray less-than gate is not present in this subspace fixture.
    return value > 0.0 if key == "c_ray_lt" else value >= 0.0


def _bootstrap_plan(replicate_id: int, indices: tuple[int, ...]) -> SimpleNamespace:
    """Construct one frozen bootstrap plan identity without schedule utilities."""
    # Hash the same canonical identity fields while preserving duplicate observations.
    identity = {
        "global_seed": 119,
        "feature_id": 23,
        "protocol": "bootstrap",
        "target_or_group_identity": "4",
        "replicate_id": int(replicate_id),
        "purpose": "all_scalars",
        "indices": tuple(sorted(indices)),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SimpleNamespace(**identity, digest=digest)


def _bootstrap_oracle(
    rows: torch.Tensor,
    gram: torch.Tensor,
    plans: tuple[SimpleNamespace, ...],
    *,
    point_estimate: float,
    threshold: float,
) -> dict[str, object]:
    """Build exact retained c-ray evidence directly from frozen plans."""
    # Evaluate every plan with the metric authority, then aggregate finite values locally.
    replicates = []
    finite_values = []
    for plan in plans:
        index = torch.tensor(plan.indices, dtype=torch.long)
        value = c_ray_pairwise_final_resid(
            rows.index_select(0, index), gram, eps=1.0e-12
        ).c_ray
        scalar = float(value) if value is not None and math.isfinite(value) else None
        if scalar is not None:
            finite_values.append(scalar)
        replicates.append(
            {
                "plan_digest": plan.digest,
                "seed": plan.global_seed,
                "replicate_id": plan.replicate_id,
                "protocol": plan.protocol,
                "target_or_group_identity": plan.target_or_group_identity,
                "status": "valid" if scalar is not None else "failed",
                "metrics": {"c_ray": scalar},
            }
        )
    failed = len(plans) - len(finite_values)
    low = float(np.quantile(finite_values, 0.25)) if finite_values else None
    high = float(np.quantile(finite_values, 0.75)) if finite_values else None
    status = (
        "unavailable"
        if failed
        else ("stable" if low is not None and low >= threshold else "unstable")
    )
    counters = _counts(
        requested=len(plans), valid=len(finite_values), failed=failed
    )
    return {
        "counters": counters,
        "requested_count": len(plans),
        "valid_count": len(finite_values),
        "failed_count": failed,
        "non_applicable_count": 0,
        "skipped_count": 0,
        "replicates": replicates,
        "plan_digest": _digest(plans),
        "metric": "c_ray",
        "estimate": float(point_estimate),
        "ci_low": low,
        "ci_high": high,
        "quantiles": [0.25, 0.75],
        "comparison": "ge",
        "threshold": float(threshold),
        "status": status,
        "instability_count": int(status == "unstable"),
        "required_failure_count": failed,
    }


def _subset_evidence(
    rows: torch.Tensor, gram: torch.Tensor, thresholds: object
) -> dict[str, object]:
    """Evaluate one frozen subset through geometry and margin authorities directly."""
    # Compute only the selected span family and the two approved candidate dimensions.
    span = span_spectrum_final_resid(rows, gram, k_values=STRICT_K, eps=1.0e-12)
    rank = effective_rank_from_spectrum(span.eigenvalues, eps=1.0e-12)
    metrics: dict[str, float | None] = {"r_span_pr": rank.r_pr}
    for key, value in span.s_span.items():
        metrics[f"s_span_{key}"] = value
    for key, value in span.u_span.items():
        metrics[f"u_span_{key}"] = value
    for key, value in span.d_span.items():
        metrics[f"d_span_{key}"] = value
    all_margins = signed_gate_margins(metrics, thresholds, k_values=STRICT_K)
    margins = {key: all_margins.get(key) for key in MARGIN_KEYS}
    missing = [key for key in MARGIN_KEYS if margins[key] is None]
    derived = None
    for k in STRICT_K:
        keys = (f"s_span_{k}", f"r_span_pr_k{k}", f"u_span_{k}", f"d_span_{k}")
        if all(margins[key] is not None for key in keys) and all(
            _passes(key, float(margins[key])) for key in keys
        ):
            derived = k
            break
    locked = MARGIN_KEYS[-4:]
    locked_failed = any(
        margins[key] is not None and not _passes(key, float(margins[key]))
        for key in locked
    )
    locked_missing = any(margins[key] is None for key in locked)
    supported = False if locked_failed else (None if locked_missing else True)
    return {
        "metrics": metrics,
        "margins": margins,
        "family_supported": supported,
        "derived_subset_k": derived,
        "selected_k_mismatch": derived is not None and derived != 3,
        "missing_required_margins": missing,
    }


def _subset_block(
    rows: torch.Tensor,
    gram: torch.Tensor,
    plans: tuple[object, ...],
    point_margins: dict[str, float | None],
    thresholds: object,
) -> dict[str, object]:
    """Evaluate and aggregate frozen subset plans without production composition."""
    # Retain every replicate before independently counting crossings and mismatch.
    replicates = []
    crossings = {key: 0 for key in point_margins}
    instability = mismatch_count = unavailable = 0
    for plan in plans:
        index = torch.tensor(plan.indices, dtype=torch.long)
        evidence = _subset_evidence(rows.index_select(0, index), gram, thresholds)
        replicate = {
            "plan_digest": plan.digest,
            "seed": plan.global_seed,
            "replicate_id": plan.replicate_id,
            "protocol": plan.protocol,
            "target_or_group_identity": plan.target_or_group_identity,
            "status": "valid",
            **evidence,
        }
        replicates.append(replicate)
        unavailable += int(bool(evidence["missing_required_margins"]))
        crossed = False
        for key, point in point_margins.items():
            subset = evidence["margins"][key]
            if (
                point is not None
                and subset is not None
                and _passes(key, point) != _passes(key, subset)
            ):
                crossings[key] += 1
                crossed = True
        mismatch = bool(evidence["selected_k_mismatch"])
        mismatch_count += int(mismatch)
        instability += int(crossed or mismatch)
    counters = _counts(requested=len(plans), valid=len(plans))
    status = "unstable" if instability else ("unavailable" if unavailable else "stable")
    return {
        "counters": counters,
        "requested_count": len(plans),
        "valid_count": len(plans),
        "failed_count": 0,
        "non_applicable_count": 0,
        "skipped_count": 0,
        "replicates": replicates,
        "plan_digest": _digest(plans),
        "status": status,
        "instability_count": instability,
        "selected_k_mismatch_count": mismatch_count,
        "required_margin_unavailable_count": unavailable,
        "gate_crossing_count": sum(crossings.values()),
        "gate_crossing_counts": crossings,
        "required_failure_count": unavailable,
    }


def test_complete_selected_family_record_matches_independent_oracle() -> None:
    """Match all retained protocols without schedule, resolver, aggregator, or writer calls."""
    # Freeze one complete strict-family request with real angle and subset authorities.
    config = _config()
    thresholds = get_threshold_profile("paper")
    generator = torch.Generator().manual_seed(917)
    raw_rows = torch.randn(36, 8, generator=generator)
    gram = torch.diag(torch.linspace(0.5, 1.5, 8))
    unit_rows, valid_counts, _ = final_resid_unit_rows(raw_rows, gram, eps=1.0e-12)
    point_metrics = _subset_evidence(unit_rows, gram, thresholds)["metrics"]
    point_record = dict(point_metrics)
    point_margins_all = signed_gate_margins(point_record, thresholds, k_values=STRICT_K)
    point_margins = {key: point_margins_all[key] for key in MARGIN_KEYS}
    leave = (
        build_subset_plan(
            global_seed=91,
            feature_id=17,
            protocol="leave_one_out",
            target_or_group_identity="0",
            replicate_id=0,
            purpose="profile_and_margins",
            indices=range(1, 36),
        ),
    )
    sample = (
        build_subset_plan(
            global_seed=92,
            feature_id=17,
            protocol="sample_size",
            target_or_group_identity="32",
            replicate_id=0,
            purpose="profile_and_margins",
            indices=range(32),
        ),
    )
    angle_plans = (
        AngleResamplePlan(
            feature_id=17,
            protocol="subspace_resample",
            source="raw",
            angle_k=3,
            seed=93,
            replicate_id=0,
            indices=tuple(range(32)),
            digest="angle-plan",
        ),
    )
    schedule = SimpleNamespace(
        feature_id=17,
        family="global_kD_directional_subspace",
        selection_mode="strict",
        reported_selected_k=3,
        point_selection=SimpleNamespace(point_reason="oracle_fixture"),
        schedule_digest="schedule-oracle",
        point_record_sha256="point-oracle",
        no_work_reason=None,
        reuse_standalone_assignment=False,
        scalar_metrics=(),
        bootstrap=(),
        angle_source="raw",
        angle_k=3,
        raw_angle_plans=angle_plans,
        residual_angle_plans=(),
        leave_out=leave,
        sample_size=sample,
        evaluated_strict_k_values=STRICT_K,
        margin_keys=MARGIN_KEYS,
        evidence_request=SimpleNamespace(low_context_qualification=True),
    )
    item = stability_runner._ScheduledFeature(
        schedule, point_record, raw_rows, unit_rows, gram, valid_counts
    )

    # Build the expected complete evidence solely from frozen plans and authorities.
    angle = scheduled_principal_angle_stability(
        raw_rows,
        gram,
        plans=angle_plans,
        source="raw",
        k=3,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )
    angle_decision = (
        "stable"
        if angle["angle_p90_deg"] <= subspace_angle_threshold(thresholds, 3)
        else "unstable"
    )
    angle = {
        **angle,
        "decision": angle_decision,
        "instability_count": int(angle_decision == "unstable"),
        "required_failure_count": 0,
    }
    leave_block = _subset_block(unit_rows, gram, leave, point_margins, thresholds)
    sample_block = _subset_block(unit_rows, gram, sample, point_margins, thresholds)
    bootstrap = {
        "status": "not_applicable",
        "plan_digest": hashlib.sha256(b"").hexdigest(),
        "replicates": [],
        "counters": _counts(non_applicable=1),
        "instability_count": 0,
        "required_failure_count": 0,
    }
    low = low_context_protocol(36)
    low_context = {
        **low,
        "reason": None,
        "observed_n_valid": 36,
        "observed_numerical_rank": angle["numerical_rank"],
        "required_n_valid": 32,
        "required_k": 3,
        "counters": _counts(requested=1, valid=1),
    }
    protocols = {
        "low_context_qualification": low_context,
        "bootstrap": bootstrap,
        "angle": angle,
        "leave_out": leave_block,
        "sample_size": sample_block,
    }
    expected = {
        "required_protocol_ids": [
            "low_context_qualification",
            "angle",
            "leave_out",
            "sample_size",
        ],
        "no_work_reason": None,
        "completed_instability_count": sum(
            int(protocols[key]["instability_count"])
            for key in ("bootstrap", "angle", "leave_out", "sample_size")
        ),
        "required_failure_count": sum(
            int(protocols[key]["required_failure_count"])
            for key in ("bootstrap", "angle", "leave_out", "sample_size")
        ),
        "point_margins": point_margins,
        "protocols": protocols,
        "protocol_counters": {key: value["counters"] for key, value in protocols.items()},
    }

    actual = stability_runner._execute_one(item, config, thresholds)
    assert actual.record["selected_family_evidence"] == expected


@pytest.mark.parametrize(
    "plan_indices",
    [
        ((0, 1, 2, 3), (0, 0, 1, 1)),
        ((0, 1, 2, 3), (0,)),
    ],
)
def test_c_ray_bootstrap_matches_independent_frozen_plan_oracle(
    plan_indices: tuple[tuple[int, ...], ...],
) -> None:
    """Match finite and retained-missing scalar bootstrap evidence exactly."""
    # Compare production execution to independently identified plans and aggregation.
    config = _config()
    thresholds = get_threshold_profile("paper")
    root_half = math.sqrt(0.5)
    rows = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [root_half, root_half]]
    )
    gram = torch.eye(2)
    point_estimate = 0.37
    plans = tuple(
        _bootstrap_plan(replicate_id, indices)
        for replicate_id, indices in enumerate(plan_indices)
    )
    schedule = SimpleNamespace(
        family="directed_ray", scalar_metrics=("c_ray",), bootstrap=plans
    )
    item = stability_runner._ScheduledFeature(
        schedule,
        {"c_ray": point_estimate},
        rows,
        rows,
        gram,
        {"n_valid": 4},
    )

    expected = _bootstrap_oracle(
        rows,
        gram,
        plans,
        point_estimate=point_estimate,
        threshold=float(thresholds.tau_c_ray),
    )
    actual = stability_runner._execute_bootstrap(item, config, thresholds)

    assert actual == expected
    assert actual["estimate"] == point_estimate
    if len(plan_indices[-1]) == 1:
        assert actual["status"] == "unavailable"
        assert actual["required_failure_count"] == 1
        assert actual["counters"] == _counts(requested=2, valid=1, failed=1)
        assert actual["valid_count"] == 1
        assert actual["failed_count"] == 1
        assert actual["replicates"][-1]["status"] == "failed"
        assert actual["replicates"][-1]["metrics"]["c_ray"] is None
        assert actual["requested_count"] == sum(
            actual["counters"][key]
            for key in ("valid", "failed", "non_applicable", "skipped")
        )
    else:
        assert actual["ci_low"] != actual["ci_high"]


def test_residual_low_context_preserves_raw_nonfailure_status() -> None:
    """Keep reachable residual insufficient-context evidence raw and non-failing."""
    # Exercise the n=8..15 centered-residual path and its separate qualification fact.
    config = _config()
    thresholds = get_threshold_profile("paper")
    rows = torch.eye(4).repeat(3, 1)[:10]
    gram = torch.eye(4)
    unit_rows, valid_counts, _ = final_resid_unit_rows(
        rows, gram, eps=1.0e-12
    )
    schedule = SimpleNamespace(
        angle_source="centered_residual",
        angle_k=2,
        raw_angle_plans=(),
        residual_angle_plans=(),
    )
    item = stability_runner._ScheduledFeature(
        schedule,
        {},
        rows,
        unit_rows,
        gram,
        valid_counts,
    )

    angle = stability_runner._execute_angle(item, config, thresholds)
    qualification = stability_runner._execute_low_context_qualification(item, angle)

    assert angle["status"] == "insufficient_contexts"
    assert angle["decision"] == "insufficient_contexts"
    assert angle["required_failure_count"] == 0
    assert qualification["status"] == "exploratory"
    assert qualification["observed_n_valid"] == 10
