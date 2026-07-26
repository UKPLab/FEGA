"""Deterministic selected-family subset evaluation and aggregation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fega.core.geometry_reporting.thresholds import (
    get_threshold_profile,
    subspace_angle_threshold,
)
from fega.core.stability.sampling import SubsetPlan, build_subset_plan

SubsetEvaluator = Callable[[SubsetPlan], dict[str, Any]]


def signed_gate_margins(
    metrics: Mapping[str, Any],
    thresholds: Any,
    *,
    k_values: Sequence[int],
    assignment_stability: Mapping[str, Any] | None = None,
) -> dict[str, float | None]:
    """Return positive-on-pass margins for the existing reporting comparisons."""
    # Encode every inequality direction once so subset execution never redefines a gate.
    margins = {
        "c_ray_ge": _margin(metrics.get("c_ray"), thresholds.tau_c_ray, 1),
        "c_ray_lt": _margin(metrics.get("c_ray"), thresholds.tau_c_ray, -1),
        "s_span_1_axis": _margin(metrics.get("s_span_1"), thresholds.tau_axis, 1),
        "b_axis": _margin(metrics.get("b_axis"), thresholds.tau_b_axis, 1),
        "delta_mix": _margin(metrics.get("delta_mix"), thresholds.tau_mix, 1),
        "mode_mass_min": _margin(
            metrics.get("mode_mass_min"), thresholds.tau_mode_mass, 1
        ),
        "min_mode_c_ray": _margin(
            metrics.get("min_mode_c_ray"), thresholds.tau_mode_c_ray, 1
        ),
        "assignment_stability": _margin(
            (
                assignment_stability.get("value")
                if assignment_stability is not None
                else None
            ),
            thresholds.tau_assignment_stability,
            1,
        ),
        "e_res": _margin(metrics.get("e_res"), thresholds.tau_res, 1),
    }
    for k in sorted(set(int(value) for value in k_values)):
        margins[f"s_span_{k}"] = _margin(
            metrics.get(f"s_span_{k}"), thresholds.tau_span_k, 1
        )
        if k > 1 and k in thresholds.tau_r:
            margins[f"r_span_pr_k{k}"] = _margin(
                metrics.get("r_span_pr"), thresholds.tau_r[k], 1
            )
        if k in thresholds.tau_p:
            margins[f"u_span_{k}"] = _margin(
                metrics.get(f"u_span_{k}"), thresholds.tau_p[k], 1
            )
        margins[f"d_span_{k}"] = _margin(
            metrics.get(f"d_span_{k}"), thresholds.tau_gap_k, -1
        )
        if k in thresholds.tau_ctr:
            margins[f"s_res_{k}"] = _margin(
                metrics.get(f"s_res_{k}"), thresholds.tau_ctr[k], 1
            )
        if k in thresholds.tau_r_ctr:
            margins[f"r_ctr_pr_k{k}"] = _margin(
                metrics.get("r_ctr_pr"), thresholds.tau_r_ctr[k], 1
            )
    return margins


def evaluate_subset_plans(
    plans: Sequence[SubsetPlan], evaluator: SubsetEvaluator
) -> dict[str, Any]:
    """Evaluate immutable subset plans in order while retaining every outcome."""
    # Catch approved numerical failures only; programming defects still fail the phase.
    replicates: list[dict[str, Any]] = []
    for plan in plans:
        try:
            result = evaluator(plan)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            result = {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
        replicates.append(
            {
                "plan_digest": plan.digest,
                "seed": int(plan.global_seed),
                "replicate_id": int(plan.replicate_id),
                "protocol": plan.protocol,
                "target_or_group_identity": plan.target_or_group_identity,
                **result,
            }
        )
    counts = _replicate_counters(replicates, requested=len(plans))
    return {
        "counters": counts,
        "requested_count": counts["requested"],
        "valid_count": counts["valid"],
        "failed_count": counts["failed"],
        "non_applicable_count": counts["non_applicable"],
        "skipped_count": counts["skipped"],
        "replicates": replicates,
        "plan_digest": _plan_digest(plans),
    }


def locked_family_result(
    family: str,
    *,
    selected_k: int | None,
    strict_k_values: Sequence[int],
    margins: Mapping[str, float | None],
) -> dict[str, Any]:
    """Derive family support and exact locked-k agreement from retained margins."""
    # Keep finite support, minimal-k, mismatch, and missingness as orthogonal facts.
    required = _required_family_margin_keys(family, strict_k_values)
    missing = [key for key in required if margins.get(key) is None]
    if family in {"directed_ray", "axis_or_antipodal"}:
        failed = any(
            margins.get(key) is not None
            and not _margin_passes(key, margins[key])
            for key in required
        )
        supported = False if failed else (None if missing else True)
        return {
            "family_supported": supported,
            "derived_subset_k": None,
            "selected_k_mismatch": False,
            "missing_required_margins": missing,
        }

    # Retain the smallest complete pass even when other candidates remain unavailable.
    derived: int | None = None
    for k in strict_k_values:
        candidate_keys = _candidate_margin_keys(family, int(k))
        if all(margins.get(key) is not None for key in candidate_keys) and all(
            _margin_passes(key, margins[key]) for key in candidate_keys
        ):
            derived = int(k)
            break
    if selected_k is None:
        raise ValueError(f"strict family {family} requires selected_k")
    locked_keys = _candidate_margin_keys(family, int(selected_k))
    locked_failed = any(
        margins.get(key) is not None and not _margin_passes(key, margins[key])
        for key in locked_keys
    )
    locked_missing = any(margins.get(key) is None for key in locked_keys)
    supported = False if locked_failed else (None if locked_missing else True)
    return {
        "family_supported": supported,
        "derived_subset_k": derived,
        "selected_k_mismatch": derived is not None and derived != selected_k,
        "missing_required_margins": missing,
    }


def aggregate_subset_protocol(
    *,
    point_margins: Mapping[str, float | None],
    block: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate family-local crossings, locked-k mismatch, and unavailability."""
    # Retain finite mismatch and missing margins as orthogonal scientific outcomes.
    crossings = {key: 0 for key in point_margins}
    instability_count = 0
    selected_k_mismatch_count = 0
    unavailable_count = 0
    for replicate in block.get("replicates", []):
        if replicate.get("status") != "valid":
            continue
        missing = list(replicate.get("missing_required_margins", []))
        if missing:
            unavailable_count += 1
        replicate_crossing = False
        subset_margins = replicate.get("margins", {})
        for key, point_margin in point_margins.items():
            subset_margin = subset_margins.get(key)
            if point_margin is None or subset_margin is None:
                continue
            if _margin_passes(key, point_margin) != _margin_passes(
                key, subset_margin
            ):
                crossings[key] += 1
                replicate_crossing = True
        mismatch = bool(replicate.get("selected_k_mismatch"))
        selected_k_mismatch_count += int(mismatch)
        instability_count += int(mismatch or replicate_crossing)
    counters = dict(block.get("counters", {}))
    failed = int(counters.get("failed", 0))
    requested = int(counters.get("requested", 0))
    status = (
        "unstable"
        if instability_count
        else (
            "unavailable"
            if failed or unavailable_count
            else ("not_applicable" if requested == 0 else "stable")
        )
    )
    return {
        **dict(block),
        "status": status,
        "instability_count": int(instability_count),
        "selected_k_mismatch_count": int(selected_k_mismatch_count),
        "required_margin_unavailable_count": int(unavailable_count),
        "gate_crossing_count": int(sum(crossings.values())),
        "gate_crossing_counts": crossings,
        "required_failure_count": int(failed + unavailable_count),
    }


def aggregate_c_ray_bootstrap(
    *,
    family: str,
    point_estimate: Any,
    block: Mapping[str, Any],
    quantiles: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Aggregate the metric-local c_ray interval for ray or axis only."""
    # Retain successful quantiles while incomplete schedules remain unavailable.
    values = [
        float(item["metrics"]["c_ray"])
        for item in block.get("replicates", [])
        if item.get("status") == "valid"
        and isinstance(item.get("metrics"), Mapping)
        and _finite_number(item["metrics"].get("c_ray")) is not None
    ]
    counters = dict(block.get("counters", {}))
    requested = int(counters.get("requested", 0))
    failed = int(counters.get("failed", 0)) + max(
        0, int(counters.get("valid", 0)) - len(values)
    )
    low = float(np.quantile(values, float(quantiles[0]))) if values else None
    high = float(np.quantile(values, float(quantiles[1]))) if values else None
    if requested == 0:
        decision = "not_applicable"
    elif failed or len(values) != requested:
        decision = "unavailable"
    elif family == "directed_ray":
        decision = "stable" if low is not None and low >= float(threshold) else "unstable"
    elif family == "axis_or_antipodal":
        decision = "stable" if high is not None and high < float(threshold) else "unstable"
    else:
        raise ValueError(f"c_ray bootstrap is not retained for family {family}")
    return {
        **dict(block),
        "metric": "c_ray",
        "estimate": _finite_number(point_estimate),
        "ci_low": low,
        "ci_high": high,
        "quantiles": [float(quantiles[0]), float(quantiles[1])],
        "comparison": "ge" if family == "directed_ray" else "lt",
        "threshold": float(threshold),
        "status": decision,
        "instability_count": int(decision == "unstable"),
        "required_failure_count": int(failed),
    }


def angle_stability_decision(
    angle: float | None, k: int, thresholds: Any | None = None
) -> str:
    """Apply the existing inclusive selected-k principal-angle boundary."""
    # Delegate 30/35-degree selection to the reporting threshold authority.
    value = _finite_number(angle)
    if value is None:
        return "unavailable"
    profile = get_threshold_profile("paper") if thresholds is None else thresholds
    limit = subspace_angle_threshold(profile, int(k))
    return "stable" if value <= float(limit) else "unstable"


def bootstrap_plans(
    *, seed: int, feature_id: int, n_rows: int, rounds: int
) -> list[SubsetPlan]:
    """Predeclare full-size with-replacement c_ray bootstrap plans."""
    # Draw every bootstrap from identity-derived replicate RNGs.
    if n_rows <= 0 or rounds <= 0:
        return []
    return [
        _random_plan(
            seed,
            feature_id,
            "bootstrap",
            str(n_rows),
            replicate,
            "all_scalars",
            n_rows,
            n_rows,
            replace=True,
        )
        for replicate in range(rounds)
    ]


def leave_out_plans(
    *,
    seed: int,
    feature_id: int,
    n_rows: int,
    group_labels: Sequence[str | None] | None,
) -> list[SubsetPlan]:
    """Predeclare leave-one and complete leave-group-out plans."""
    # Missing group metadata suppresses only group plans; leave-one remains complete.
    plans = [
        build_subset_plan(
            global_seed=seed,
            feature_id=feature_id,
            protocol="leave_one_out",
            target_or_group_identity=str(omitted),
            replicate_id=omitted,
            purpose="profile_and_margins",
            indices=[index for index in range(n_rows) if index != omitted],
        )
        for omitted in range(n_rows)
    ]
    if group_labels is None:
        return plans
    groups = sorted({str(label) for label in group_labels if label is not None})
    for replicate, group in enumerate(groups):
        keep = [
            index
            for index, label in enumerate(group_labels)
            if str(label) != group
        ]
        plans.append(
            build_subset_plan(
                global_seed=seed,
                feature_id=feature_id,
                protocol="leave_group_out",
                target_or_group_identity=group,
                replicate_id=replicate,
                purpose="profile_and_margins",
                indices=keep,
            )
        )
    return plans


def sample_size_plans(
    *, seed: int, feature_id: int, n_rows: int, targets: Sequence[int], rounds: int
) -> list[SubsetPlan]:
    """Predeclare deterministic sample-size plans for every feasible target."""
    # Use independent plan-identity RNGs so worker scheduling cannot perturb subsets.
    plans: list[SubsetPlan] = []
    for target in sorted({int(value) for value in targets if 0 < int(value) <= n_rows}):
        target_rounds = 1 if target == n_rows else rounds
        for replicate in range(target_rounds):
            plans.append(
                _random_plan(
                    seed,
                    feature_id,
                    "sample_size",
                    str(target),
                    replicate,
                    "gate_margins",
                    n_rows,
                    target,
                )
            )
    return plans


def _random_plan(
    seed: int,
    feature_id: int,
    protocol: str,
    target: str,
    replicate: int,
    purpose: str,
    n_rows: int,
    subset_n: int,
    replace: bool = False,
) -> SubsetPlan:
    """Create one plan from its complete scientific identity."""
    # Hash a provisional identity before drawing sorted indices with fixed multiplicity.
    provisional = build_subset_plan(
        global_seed=seed,
        feature_id=feature_id,
        protocol=protocol,
        target_or_group_identity=target,
        replicate_id=replicate,
        purpose=purpose,
        indices=[],
    )
    rng = np.random.default_rng(int(provisional.digest[:16], 16))
    indices = np.sort(rng.choice(n_rows, size=subset_n, replace=replace)).tolist()
    return build_subset_plan(
        global_seed=seed,
        feature_id=feature_id,
        protocol=protocol,
        target_or_group_identity=target,
        replicate_id=replicate,
        purpose=purpose,
        indices=indices,
    )


def _required_family_margin_keys(
    family: str, strict_k_values: Sequence[int]
) -> tuple[str, ...]:
    """Return the approved complete margin set for one locked family."""
    # Keep the table mechanical and reject any family outside the selected map.
    if family == "directed_ray":
        return ("c_ray_ge", "s_span_1_axis")
    if family == "axis_or_antipodal":
        return ("c_ray_lt", "s_span_1_axis", "b_axis")
    keys: list[str] = []
    for k in strict_k_values:
        keys.extend(_candidate_margin_keys(family, int(k)))
    if not keys:
        raise ValueError(f"unsupported selected family margins: {family}")
    return tuple(dict.fromkeys(keys))


def _candidate_margin_keys(family: str, k: int) -> tuple[str, ...]:
    """Return one strict candidate's exact existing gate-margin names."""
    # Span and residual candidates intentionally use different effective-rank sources.
    if family in {
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
    }:
        return (
            f"s_span_{k}",
            f"r_span_pr_k{k}",
            f"u_span_{k}",
            f"d_span_{k}",
        )
    if family == "residual_lowD_k":
        return ("e_res", f"s_res_{k}", f"r_ctr_pr_k{k}")
    raise ValueError(f"family {family} has no strict-k candidate margins")


def _margin(value: Any, threshold: float, direction: int) -> float | None:
    """Compute a finite signed threshold margin with positive meaning pass."""
    # Reject booleans and non-finite values rather than fabricating a gate side.
    parsed = _finite_number(value)
    if parsed is None:
        return None
    return float(direction) * (parsed - float(threshold))


def _margin_passes(name: str, margin: Any) -> bool:
    """Apply inclusive gates except the strict axis-side c_ray complement."""
    # Equality fails only the established c_ray < tau comparison.
    return float(margin) > 0.0 if name.endswith("_lt") else float(margin) >= 0.0


def _finite_number(value: Any) -> float | None:
    """Return one finite non-boolean numeric value."""
    # Scientific missingness remains None instead of an implicit threshold side.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _replicate_counters(
    replicates: Sequence[Mapping[str, Any]], *, requested: int
) -> dict[str, int]:
    """Count every requested replicate in the closed outcome vocabulary."""
    # Unknown statuses are retained as non-applicable rather than dropped.
    valid = sum(item.get("status") == "valid" for item in replicates)
    failed = sum(item.get("status") == "failed" for item in replicates)
    skipped = sum(item.get("status") == "skipped" for item in replicates)
    non_applicable = int(requested) - valid - failed - skipped
    return {
        "requested": int(requested),
        "valid": int(valid),
        "failed": int(failed),
        "non_applicable": int(non_applicable),
        "skipped": int(skipped),
    }


def _plan_digest(plans: Sequence[SubsetPlan]) -> str:
    """Hash ordered per-replicate identities into one protocol identity."""
    # The empty plan has the standard empty SHA256 and remains deterministic.
    joined = "".join(plan.digest for plan in plans)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()
