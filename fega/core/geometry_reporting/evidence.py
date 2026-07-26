from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .thresholds import GeometryThresholds, subspace_angle_threshold


def build_gate_evidence(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, Any]:
    return {
        "directed_ray": directed_ray_gate(record, thresholds),
        "axis_or_antipodal": axis_gate(record, thresholds),
        "multi_mode_directional_geometry": multimode_gate(record, thresholds),
        "global_directional_subspace": span_gate(record, thresholds),
        "residual_lowD_k": residual_gate(record, thresholds),
    }


def build_point_gate_evidence(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, Any]:
    """Build full-sample point gates without reading non-mixture stability evidence.

    The comparisons and candidate dimensions are the existing reporting authority.
    This projection removes only bootstrap, angle, leave-out, and sample-size inputs
    so point selection can precede selected-family stability qualification.
    """
    span = _ordered_point_attempts(
        (
            (k, span_point_comparisons(record, thresholds, k))
            for k in (2, 3, 4, 8)
            if k in thresholds.tau_r and k in thresholds.tau_p
        ),
        reason="no_passing_span_k",
    )
    residual = _ordered_point_attempts(
        (
            (k, residual_point_comparisons(record, thresholds, k))
            for k in (2, 3, 4)
        ),
        reason="no_passing_centered_residual_k",
    )
    # Keep family keys aligned with the established reporting gate vocabulary.
    return {
        "directed_ray": point_gate_result(
            directed_ray_point_comparisons(record, thresholds)
        ),
        "axis_or_antipodal": point_gate_result(
            axis_point_comparisons(record, thresholds)
        ),
        "multi_mode_directional_geometry": multimode_gate(record, thresholds),
        "global_directional_subspace": span,
        "residual_lowD_k": residual,
    }


def base_ok(record: dict[str, Any], thresholds: GeometryThresholds) -> bool:
    return not insufficient_base(record, thresholds)


def insufficient_base(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    n_valid = finite_float(record.get("n_valid"))
    if n_valid is None or n_valid < thresholds.n_min:
        return True
    zero_filter_frac = finite_float(record.get("zero_filter_frac"))
    return (
        zero_filter_frac is not None
        and zero_filter_frac > thresholds.tau_zero_filter_frac
    )


def base_terminal_reason(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> str:
    n_valid = finite_float(record.get("n_valid"))
    if n_valid is None:
        return "effect_count_missing"
    if n_valid < thresholds.n_min:
        return "effect_count_below_min"
    return "zero_filter_too_high"


def directed_ray_point_comparisons(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, bool | None]:
    """Return the existing directed-ray point comparisons from one authority."""
    return {
        "c_ray": ge(record.get("c_ray"), thresholds.tau_c_ray),
        "s_span_1": ge(record.get("s_span_1"), thresholds.tau_axis),
    }


def axis_point_comparisons(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, bool | None]:
    """Return the existing axis point comparisons from one authority."""
    return {
        "s_span_1": ge(record.get("s_span_1"), thresholds.tau_axis),
        "c_ray": lt(record.get("c_ray"), thresholds.tau_c_ray),
        "b_axis": ge(record.get("b_axis"), thresholds.tau_b_axis),
    }


def span_point_comparisons(
    record: dict[str, Any], thresholds: GeometryThresholds, k: int
) -> dict[str, bool | None]:
    """Return the existing span point comparisons for one candidate dimension."""
    tau_r = thresholds.tau_r.get(k)
    tau_p = thresholds.tau_p.get(k)
    return {
        f"s_span_{k}": ge(record.get(f"s_span_{k}"), thresholds.tau_span_k),
        "r_span_pr": (
            ge(record.get("r_span_pr"), tau_r) if tau_r is not None else None
        ),
        f"u_span_{k}": (
            ge(record.get(f"u_span_{k}"), tau_p) if tau_p is not None else None
        ),
        f"d_span_{k}": le(record.get(f"d_span_{k}"), thresholds.tau_gap_k),
    }


def residual_point_comparisons(
    record: dict[str, Any], thresholds: GeometryThresholds, k: int
) -> dict[str, bool | None]:
    """Return the existing residual point comparisons for one candidate dimension."""
    tau_ctr = thresholds.tau_ctr.get(k)
    tau_r_ctr = thresholds.tau_r_ctr.get(k)
    return {
        "e_res": ge(record.get("e_res"), thresholds.tau_res),
        f"s_res_{k}": (
            ge(record.get(f"s_res_{k}"), tau_ctr) if tau_ctr is not None else None
        ),
        "r_ctr_pr": (
            ge(record.get("r_ctr_pr"), tau_r_ctr)
            if tau_r_ctr is not None
            else None
        ),
    }


def ray_anchor(record: dict[str, Any], thresholds: GeometryThresholds) -> bool:
    return all(
        value is True
        for value in directed_ray_point_comparisons(record, thresholds).values()
    )


def axis_anchor(record: dict[str, Any], thresholds: GeometryThresholds) -> bool:
    return all(
        value is True for value in axis_point_comparisons(record, thresholds).values()
    )


def one_d_diffuse_condition(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    comparisons = axis_point_comparisons(record, thresholds)
    return (
        base_ok(record, thresholds)
        and comparisons["s_span_1"] is True
        and comparisons["c_ray"] is True
        and comparisons["b_axis"] is not True
    )


def multimode_anchor(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    """Return whether selected multimode evidence is sufficient for audit candidacy."""
    # Anchor on selected M>1 plus gain and any passing mode metric; reporting gates
    # independently decide whether the preserved candidate can become primary.
    mode_metrics = (
        ge(record.get("mode_mass_min"), thresholds.tau_mode_mass),
        ge(record.get("min_mode_c_ray"), thresholds.tau_mode_c_ray),
        ge(
            assignment_stability_value(record),
            thresholds.tau_assignment_stability,
        ),
    )
    return (
        gt(record.get("selected_mode_count"), 1.0) is True
        and ge(record.get("delta_mix"), thresholds.tau_mix) is True
        and any(value is True for value in mode_metrics)
    )


def span_anchor(
    record: dict[str, Any], thresholds: GeometryThresholds, k: int
) -> bool:
    return span_point_comparisons(record, thresholds, k)[f"s_span_{k}"] is True


def residual_anchor(
    record: dict[str, Any], thresholds: GeometryThresholds, k: int
) -> bool:
    if k not in thresholds.tau_ctr:
        return False
    comparisons = residual_point_comparisons(record, thresholds, k)
    return (
        comparisons["e_res"] is True and comparisons[f"s_res_{k}"] is True
    )


def positive_high_dimensional_or_diffuse(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    s_span_1 = finite_float(record.get("s_span_1"))
    e_res = finite_float(record.get("e_res"))
    selected_mode_count = finite_float(record.get("selected_mode_count"))
    delta_mix = finite_float(record.get("delta_mix"))
    finite_spans = [
        value
        for value in (
            finite_float(record.get(f"s_span_{k}")) for k in (2, 3, 4, 8)
        )
        if value is not None
    ]
    return (
        base_ok(record, thresholds)
        and all_family_anchors_false(record, thresholds)
        and s_span_1 is not None
        and s_span_1 < thresholds.tau_axis
        and e_res is not None
        and e_res < thresholds.tau_res
        and bool(finite_spans)
        and all(value < thresholds.tau_span_k for value in finite_spans)
        and selected_mode_count is not None
        and delta_mix is not None
        and (selected_mode_count <= 1 or delta_mix < thresholds.tau_mix)
    )


def all_family_anchors_false(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> bool:
    return not (
        ray_anchor(record, thresholds)
        or axis_anchor(record, thresholds)
        or one_d_diffuse_condition(record, thresholds)
        or multimode_anchor(record, thresholds)
        or any(span_anchor(record, thresholds, k) for k in (2, 3, 4, 8))
        or any(residual_anchor(record, thresholds, k) for k in (1, 2, 3, 4))
    )


def subspace_threshold_defined(thresholds: GeometryThresholds, k: int) -> bool:
    return str(k) in thresholds.tau_subspace_angle or (
        k != 1 and "k" in thresholds.tau_subspace_angle
    )


def directed_ray_gate(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, Any]:
    return gate_result(
        directed_ray_point_comparisons(record, thresholds),
        scalar_ci_status=c_ray_ci_status(record, thresholds.tau_c_ray, "ge"),
        subspace_status="not_required",
        sample_size_status=sample_size_status(record),
        sample_size_curve_status=sample_size_curve_status(record),
        leave_out_status=leave_out_status(record),
        low_context_status=low_context_status(record),
    )


def axis_gate(record: dict[str, Any], thresholds: GeometryThresholds) -> dict[str, Any]:
    return gate_result(
        axis_point_comparisons(record, thresholds),
        scalar_ci_status=c_ray_ci_status(record, thresholds.tau_c_ray, "lt"),
        subspace_status=subspace_status(record, 1, thresholds),
        sample_size_status=sample_size_status(record),
        sample_size_curve_status=sample_size_curve_status(record),
        leave_out_status=leave_out_status(record),
        low_context_status=low_context_status(record),
    )


def multimode_gate(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, Any]:
    """Apply paper multimode thresholds once without changing selected mode count.

    Operational fit and BIC selection remain independent. Only a fitted selected
    model with M>1 is evaluated; all other states retain explicit non-evaluation
    reasons. Missing required evidence rejects an evaluated mixture and is kept
    distinct from a numerically observed failed gate.
    """
    # Reject non-evaluable operational states before reading any scientific gate.
    fit_status = record.get("fit_status")
    selected_m = finite_float(record.get("selected_mode_count"))
    if fit_status not in {None, "fitted"}:
        return _multimode_not_evaluated(f"fit_status_{fit_status}")
    if selected_m is None or selected_m <= 1:
        return _multimode_not_evaluated("selected_mode_count_not_multimode")

    assignment = record.get("assignment_stability")
    stability_status = assignment.get("status") if isinstance(assignment, dict) else None
    stability_value = assignment_stability_value(record)
    raw_gates = {
        "gain": (
            finite_float(record.get("delta_mix")),
            thresholds.tau_mix,
            ge(record.get("delta_mix"), thresholds.tau_mix),
        ),
        "fitted_mass": (
            finite_float(record.get("mode_mass_min")),
            thresholds.tau_mode_mass,
            ge(record.get("mode_mass_min"), thresholds.tau_mode_mass),
        ),
        "within_mode_ray": (
            finite_float(record.get("min_mode_c_ray")),
            thresholds.tau_mode_c_ray,
            ge(record.get("min_mode_c_ray"), thresholds.tau_mode_c_ray),
        ),
        "finite_concentration": (
            finite_float(record.get("mode_kappa_min")),
            None,
            finite_float(record.get("mode_kappa_min")) is not None,
        ),
        "assignment_stability": (
            stability_value,
            thresholds.tau_assignment_stability,
            (
                ge(stability_value, thresholds.tau_assignment_stability)
                if stability_status in {None, "available"}
                else None
            ),
        ),
    }
    gate_values = {
        name: {"value": value, "threshold": threshold, "passed": passed}
        for name, (value, threshold, passed) in raw_gates.items()
    }
    failed = [name for name, (_, _, passed) in raw_gates.items() if passed is False]
    unavailable = [name for name, (_, _, passed) in raw_gates.items() if passed is None]
    accepted = not failed and not unavailable
    reasons = list(failed)
    reasons.extend(
        "stability_unavailable" if name == "assignment_stability" else f"{name}_unavailable"
        for name in unavailable
    )
    return {
        "evaluated": True,
        "acceptance": "accepted" if accepted else "rejected",
        "reason": None,
        "failed_gates": failed,
        "unavailable_gates": unavailable,
        "gate_values": gate_values,
        "metric_status": "stable" if accepted else "unstable",
        "scalar_ci_status": "not_required",
        "subspace_status": "not_required",
        "sample_size_status": "not_required",
        "sample_size_curve_status": sample_size_curve_status(record),
        "leave_out_status": leave_out_status(record),
        "low_context_status": low_context_status(record),
        "decision": "stable" if accepted else "unstable",
        "blocked_reasons": reasons,
    }


def _multimode_not_evaluated(reason: str) -> dict[str, Any]:
    """Build an explicit non-evaluated multimode reporting state."""
    # Emit the current evidence keys expected by classifier consumers.
    return {
        "evaluated": False,
        "acceptance": None,
        "reason": reason,
        "failed_gates": [],
        "unavailable_gates": [],
        "gate_values": {},
        "metric_status": "not_available",
        "scalar_ci_status": "not_required",
        "subspace_status": "not_required",
        "sample_size_status": "not_required",
        "sample_size_curve_status": "stable",
        "leave_out_status": "stable",
        "low_context_status": "stable",
        "decision": "not_available",
        "blocked_reasons": [reason],
    }


def span_gate(record: dict[str, Any], thresholds: GeometryThresholds) -> dict[str, Any]:
    attempts: dict[str, Any] = {}
    missing_only = True
    for k in (2, 3, 4, 8):
        if k not in thresholds.tau_r or k not in thresholds.tau_p:
            continue
        subspace = subspace_status(record, k, thresholds)
        sample = sample_size_status(record)
        entry = gate_result(
            span_point_comparisons(record, thresholds, k),
            scalar_ci_status="not_required",
            subspace_status=subspace,
            sample_size_status=sample,
            sample_size_curve_status=sample_size_curve_status(record),
            leave_out_status=leave_out_status(record),
            low_context_status=low_context_status(record),
        )
        attempts[str(k)] = entry
        if entry["decision"] in {"stable", "exploratory"}:
            return {
                **entry,
                "scalar_ci_status": "not_required",
                "subspace_status": subspace,
                "selected_k": k,
                "attempts": attempts,
                "blocked_reasons": [],
            }
        missing_only = missing_only and entry["decision"] == "not_available"
    decision = "not_available" if missing_only else "unstable"
    return _aggregate_gate(record, decision, attempts, "no_passing_span_k")


def residual_gate(
    record: dict[str, Any], thresholds: GeometryThresholds
) -> dict[str, Any]:
    attempts: dict[str, Any] = {}
    for k in (2, 3, 4):
        entry = gate_result(
            residual_point_comparisons(record, thresholds, k),
            scalar_ci_status="not_required",
            subspace_status=centered_residual_status(record, k, thresholds),
            sample_size_status=sample_size_status(record),
            sample_size_curve_status=sample_size_curve_status(record),
            leave_out_status=leave_out_status(record),
            low_context_status=low_context_status(record),
        )
        attempts[str(k)] = entry
        if entry["decision"] in {"stable", "exploratory"}:
            return {**entry, "selected_k": k, "attempts": attempts}
    missing_only = bool(attempts) and all(
        entry.get("decision") == "not_available" for entry in attempts.values()
    )
    decision = "not_available" if not attempts or missing_only else "unstable"
    return _aggregate_gate(
        record, decision, attempts, "no_passing_centered_residual_k"
    )


def _ordered_point_attempts(
    comparisons_by_k: Iterable[tuple[int, dict[str, bool | None]]], *, reason: str
) -> dict[str, Any]:
    """Select the first passing dimension from shared ordered comparisons."""
    attempts: dict[str, Any] = {}
    for k, comparisons in comparisons_by_k:
        entry = point_gate_result(comparisons)
        attempts[str(k)] = entry
        if entry["decision"] == "stable":
            return {**entry, "selected_k": k, "attempts": attempts}
    return _point_attempts_result(attempts, reason)


def point_gate_result(metrics: dict[str, bool | None]) -> dict[str, Any]:
    """Return the closed point-only decision for one existing metric set."""
    # Reuse metrics_status so missing evidence retains its established precedence.
    decision = metrics_status(metrics)
    return {
        "metric_status": decision,
        "decision": decision,
        "metrics": metrics,
        "blocked_reasons": metric_blockers(metrics),
    }


def _point_attempts_result(
    attempts: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Aggregate ordered point attempts without consulting stability evidence."""
    # All-missing is distinct from a completed failed point comparison.
    missing_only = bool(attempts) and all(
        entry.get("decision") == "not_available" for entry in attempts.values()
    )
    decision = "not_available" if missing_only else "unstable"
    return {
        "metric_status": decision,
        "decision": decision,
        "selected_k": None,
        "attempts": attempts,
        "blocked_reasons": [reason],
    }


def _aggregate_gate(
    record: dict[str, Any], decision: str, attempts: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "metric_status": decision,
        "scalar_ci_status": "not_required",
        "subspace_status": decision,
        "sample_size_status": sample_size_status(record),
        "sample_size_curve_status": sample_size_curve_status(record),
        "leave_out_status": leave_out_status(record),
        "low_context_status": low_context_status(record),
        "decision": decision,
        "selected_k": None,
        "attempts": attempts,
        "blocked_reasons": [reason],
    }


def gate_result(
    metrics: dict[str, bool | None],
    *,
    scalar_ci_status: str,
    subspace_status: str,
    sample_size_status: str,
    sample_size_curve_status: str,
    leave_out_status: str,
    low_context_status: str,
) -> dict[str, Any]:
    metric_status = metrics_status(metrics)
    statuses = [metric_status, scalar_ci_status, subspace_status, sample_size_status]
    if "not_available" in statuses:
        decision = "not_available"
    elif "unstable" in statuses:
        decision = "unstable"
    elif metric_status == "stable" and "exploratory" in statuses:
        decision = "exploratory"
    elif metric_status == "stable":
        decision = "stable"
    else:
        decision = "unstable"
    return {
        "metric_status": metric_status,
        "scalar_ci_status": scalar_ci_status,
        "subspace_status": subspace_status,
        "sample_size_status": sample_size_status,
        "sample_size_curve_status": sample_size_curve_status,
        "leave_out_status": leave_out_status,
        "low_context_status": low_context_status,
        "decision": decision,
        "blocked_reasons": blocked_reasons(
            metrics, scalar_ci_status, subspace_status
        ),
    }


def metrics_status(metrics: dict[str, bool | None]) -> str:
    if any(value is None for value in metrics.values()):
        return "not_available"
    if any(value is False for value in metrics.values()):
        return "unstable"
    return "stable"


def metric_blockers(metrics: dict[str, bool | None]) -> list[str]:
    return [name for name, value in metrics.items() if value is not True]


def blocked_reasons(
    metrics: dict[str, bool | None], scalar_ci_status: str, subspace_status: str
) -> list[str]:
    reasons = metric_blockers(metrics)
    if scalar_ci_status in {"unstable", "not_available"}:
        reasons.append(f"scalar_ci_{scalar_ci_status}")
    if subspace_status in {"unstable", "not_available"}:
        reasons.append(f"subspace_{subspace_status}")
    return reasons


def c_ray_ci_status(record: dict[str, Any], threshold: float, direction: str) -> str:
    """Interpret C-ray interval evidence without treating incomplete replicates as complete."""
    # Required replicate failure blocks the interval even when successful quantiles exist.
    ci = (record.get("scalar_ci") or {}).get("c_ray")
    if not isinstance(ci, dict):
        return "not_available"
    if ci.get("evidence_status") == "unavailable":
        return "not_available"
    low = finite_float(ci.get("ci_low"))
    high = finite_float(ci.get("ci_high"))
    if low is None or high is None:
        return "not_available"
    if direction == "ge":
        if low >= threshold:
            return "stable"
        return "unstable"
    if high < threshold:
        return "stable"
    return "unstable"


def subspace_status(
    record: dict[str, Any], k: int, thresholds: GeometryThresholds
) -> str:
    block = record.get("subspace_stability")
    if not isinstance(block, dict):
        return "not_available"
    per_k = block.get("k")
    entry = per_k.get(str(k)) if isinstance(per_k, dict) else None
    if not isinstance(entry, dict):
        return "not_available"
    status = entry.get("status")
    angle = finite_float(entry.get("subspace_angle_p90_k"))
    if status == "exploratory":
        return "exploratory"
    if status != "ok" or angle is None:
        return "not_available"
    return "stable" if angle <= subspace_angle_threshold(thresholds, k) else "unstable"


def centered_residual_status(
    record: dict[str, Any], k: int, thresholds: GeometryThresholds
) -> str:
    block = record.get("centered_residual_subspace_stability")
    if not isinstance(block, dict) or block.get("source") != "stability_artifact":
        return "not_available"
    per_k = block.get("k")
    entry = per_k.get(str(k)) if isinstance(per_k, dict) else None
    if not isinstance(entry, dict):
        return "not_available"
    status = entry.get("status")
    angle = finite_float(entry.get("residual_angle_p90_k"))
    if status == "exploratory":
        return "exploratory"
    if status != "ok" or angle is None:
        return "not_available"
    return "stable" if angle <= subspace_angle_threshold(thresholds, k) else "unstable"


def sample_size_status(record: dict[str, Any]) -> str:
    """Combine leave/sample evidence while preserving unavailable separately."""
    # Completed instability outranks another protocol's failure under the orthogonal contract.
    if sample_size_curve_status(record) == "unstable":
        return "unstable"
    if leave_out_status(record) == "unstable":
        return "unstable"
    if sample_size_curve_status(record) == "not_available":
        return "not_available"
    if leave_out_status(record) == "not_available":
        return "not_available"
    return low_context_status(record)


def sample_size_curve_status(record: dict[str, Any]) -> str:
    return _stability_block_status(record.get("sample_size_curves"))


def leave_out_status(record: dict[str, Any]) -> str:
    return _stability_block_status(record.get("leave_out_sensitivity"))


def _stability_block_status(value: Any) -> str:
    """Normalize raw stability block states for geometry evidence consumers."""
    # Keep computational unavailability distinct from stable and unstable science.
    if not isinstance(value, dict) or not value:
        return "not_available"
    status = value.get("geometry_status") or value.get("decision") or value.get("status")
    requested = value.get("requested_count")
    if isinstance(requested, int) and requested <= 0:
        return "stable" if status == "not_applicable" else "not_available"
    if status in {"unstable", "changed", "crossed_boundary"}:
        return "unstable"
    if status in {"unavailable", "failed", "missing_input"}:
        return "not_available"
    return "stable"


def low_context_status(record: dict[str, Any]) -> str:
    low_context = record.get("low_context")
    if isinstance(low_context, dict) and low_context.get("status") == "exploratory":
        return "exploratory"
    return "stable"


def only_missing_evidence(gate_evidence: dict[str, Any]) -> bool:
    return bool(gate_evidence) and all(
        evidence.get("decision") == "not_available"
        for evidence in gate_evidence.values()
    )


def long_tail(record: dict[str, Any], thresholds: GeometryThresholds) -> bool:
    ent = finite_float(record.get("r_span_ent"))
    pr = finite_float(record.get("r_span_pr"))
    eps = finite_float(record.get("eps")) or 1.0e-12
    if ent is None or pr is None:
        return False
    return ent / (pr + eps) >= thresholds.tau_longtail


def ge(value: Any, threshold: float) -> bool | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed >= threshold


def gt(value: Any, threshold: float) -> bool | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed > threshold


def lt(value: Any, threshold: float) -> bool | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed < threshold


def le(value: Any, threshold: float) -> bool | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed <= threshold


def assignment_stability_value(record: dict[str, Any]) -> float | None:
    """Read the published assignment-stability value from its structured state."""
    # Keep the structured vMF result as the sole public source of this metric.
    assignment = record.get("assignment_stability")
    if not isinstance(assignment, dict):
        return None
    return finite_float(assignment.get("value"))


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
