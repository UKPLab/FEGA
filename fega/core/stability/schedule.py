from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from fega.core.geometry_reporting.point_selection import PointSelection
from fega.core.geometry_reporting.point_selection import (
    point_selection_identity as _point_selection_identity,
)
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.stability.protocols import (
    bootstrap_plans,
    leave_out_plans,
    sample_size_plans,
)
from fega.core.stability.sampling import (
    SubsetPlan,
    derive_seed,
    subspace_resample_indices,
)

AngleSource = Literal["none", "raw", "centered_residual"]
AngleDimension = Literal["selected_k"] | int | None
SELECTED_FAMILY_SCHEDULE_VERSION = 1


@dataclass(frozen=True)
class SelectedFamilyEvidenceRequest:
    """Immutable evidence breadth for one strict point-selected family.

    This WP1 value object freezes the approved schedule map only. It does not
    generate subsets or execute stability; those later steps must consume this
    contract without widening its metric, family, or dimension breadth.
    """

    full_sample_point_requirements: tuple[str, ...]
    scalar_bootstrap_metrics: tuple[str, ...]
    angle_source: AngleSource
    angle_k: AngleDimension
    leave_sample_margins: tuple[str, ...]
    strict_k_values: tuple[int, ...]
    stop_point_gates_at_selected_k: bool
    low_context_qualification: bool
    reuse_assignment_stability: bool
    no_stability_protocol: bool
    explicitly_omitted: tuple[str, ...]


_RAY = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=("c_ray", "s_span_1"),
    scalar_bootstrap_metrics=("c_ray",),
    angle_source="none",
    angle_k=None,
    leave_sample_margins=("c_ray_ge", "s_span_1_axis"),
    strict_k_values=(),
    stop_point_gates_at_selected_k=False,
    low_context_qualification=True,
    reuse_assignment_stability=False,
    no_stability_protocol=False,
    explicitly_omitted=(
        "vmf",
        "all_angles",
        "other_scalar_intervals",
        "other_family_margins",
    ),
)

_AXIS = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=("s_span_1", "c_ray", "b_axis"),
    scalar_bootstrap_metrics=("c_ray",),
    angle_source="raw",
    angle_k=1,
    leave_sample_margins=("c_ray_lt", "s_span_1_axis", "b_axis"),
    strict_k_values=(),
    stop_point_gates_at_selected_k=False,
    low_context_qualification=True,
    reuse_assignment_stability=False,
    no_stability_protocol=False,
    explicitly_omitted=(
        "vmf",
        "angles_away_from_k1",
        "b_axis_bootstrap_interval",
        "s_span_1_bootstrap_interval",
        "other_family_margins",
    ),
)

_MIXTURE = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=(
        "selected_mode_count",
        "delta_mix",
        "mode_mass_min",
        "min_mode_c_ray",
        "mode_kappa_min",
        "assignment_stability",
    ),
    scalar_bootstrap_metrics=(),
    angle_source="none",
    angle_k=None,
    leave_sample_margins=(),
    strict_k_values=(),
    stop_point_gates_at_selected_k=False,
    low_context_qualification=False,
    reuse_assignment_stability=True,
    no_stability_protocol=True,
    explicitly_omitted=("all_stability_phase_fitting_and_resampling",),
)

_SPAN = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=(
        "s_span_{k}",
        "u_span_{k}",
        "d_span_{k}",
        "r_span_pr",
    ),
    scalar_bootstrap_metrics=(),
    angle_source="raw",
    angle_k="selected_k",
    leave_sample_margins=(
        "s_span_{k}",
        "r_span_pr_k{k}",
        "u_span_{k}",
        "d_span_{k}",
    ),
    strict_k_values=(2, 3, 4, 8),
    stop_point_gates_at_selected_k=True,
    low_context_qualification=True,
    reuse_assignment_stability=False,
    no_stability_protocol=False,
    explicitly_omitted=(
        "vmf",
        "angles_away_from_selected_k",
        "point_gates_above_selected_k",
        "scalar_bootstrap_intervals",
        "other_family_margins",
    ),
)

_RESIDUAL = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=("e_res", "s_res_{k}", "r_ctr_pr"),
    scalar_bootstrap_metrics=(),
    angle_source="centered_residual",
    angle_k="selected_k",
    leave_sample_margins=("e_res", "s_res_{k}", "r_ctr_pr_k{k}"),
    strict_k_values=(2, 3, 4),
    stop_point_gates_at_selected_k=True,
    low_context_qualification=True,
    reuse_assignment_stability=False,
    no_stability_protocol=False,
    explicitly_omitted=(
        "vmf",
        "angles_away_from_selected_k",
        "point_gates_above_selected_k",
        "fallback_only_residual_k1",
        "scalar_bootstrap_intervals",
        "other_family_margins",
    ),
)


SELECTED_FAMILY_EVIDENCE_REQUESTS: Mapping[
    str, SelectedFamilyEvidenceRequest
] = MappingProxyType(
    {
        "directed_ray": _RAY,
        "axis_or_antipodal": _AXIS,
        "multi_mode_directional_geometry": _MIXTURE,
        "global_2D_directional_subspace": _SPAN,
        "global_kD_directional_subspace": _SPAN,
        "residual_lowD_k": _RESIDUAL,
    }
)


NO_STABILITY_EVIDENCE_REQUEST = SelectedFamilyEvidenceRequest(
    full_sample_point_requirements=(),
    scalar_bootstrap_metrics=(),
    angle_source="none",
    angle_k=None,
    leave_sample_margins=(),
    strict_k_values=(),
    stop_point_gates_at_selected_k=False,
    low_context_qualification=False,
    reuse_assignment_stability=False,
    no_stability_protocol=True,
    explicitly_omitted=(
        "all_bootstrap",
        "all_angles",
        "all_leave_out",
        "all_sample_size",
        "all_vmf",
    ),
)

DELIBERATE_NON_EVALUATION_REASON = (
    "not_evaluated_point_fallback_or_terminal"
)


def resolve_requested_margin_keys(
    request: SelectedFamilyEvidenceRequest, *, selected_k: int | None
) -> tuple[str, ...]:
    """Resolve contract key templates without generating or executing a schedule."""
    requires_k = any("{k}" in key for key in request.leave_sample_margins)
    if not requires_k:
        return request.leave_sample_margins
    if selected_k is None or selected_k not in request.strict_k_values:
        raise ValueError("selected_k must be a declared strict dimension")
    return tuple(key.format(k=selected_k) for key in request.leave_sample_margins)


@dataclass(frozen=True)
class SelectedFamilySchedule:
    """Immutable pre-execution schedule for one locked point-family selection.

    The schedule contains only existing generator outputs and dimension choices.
    Execution and aggregation deliberately remain outside this WP2 construction.
    """

    feature_id: int
    point_selection: PointSelection
    evidence_request: SelectedFamilyEvidenceRequest
    family: str
    selection_mode: str
    reported_selected_k: int | None
    angle_source: AngleSource
    angle_k: int | None
    evaluated_strict_k_values: tuple[int, ...]
    scalar_metrics: tuple[str, ...]
    margin_keys: tuple[str, ...]
    bootstrap: tuple[SubsetPlan, ...]
    leave_out: tuple[SubsetPlan, ...]
    sample_size: tuple[SubsetPlan, ...]
    raw_angle_plans: tuple[AngleResamplePlan, ...]
    residual_angle_plans: tuple[AngleResamplePlan, ...]
    feature_seed: int
    reuse_standalone_assignment: bool
    request_count: int
    no_work_reason: str | None
    point_record_sha256: str
    schedule_digest: str
    version: int = SELECTED_FAMILY_SCHEDULE_VERSION


@dataclass(frozen=True)
class AngleResamplePlan:
    """One content-addressed existing sequential angle-resample draw."""

    feature_id: int
    protocol: str
    source: AngleSource
    angle_k: int
    seed: int
    replicate_id: int
    indices: tuple[int, ...]
    digest: str


def build_selected_family_schedule(
    *,
    selection: PointSelection,
    feature_id: int,
    point_record_sha256: str,
    base_seed: int,
    effect_space: str,
    n_rows: int,
    group_labels: Sequence[str | None] | None,
    stability_config: Any,
) -> SelectedFamilySchedule:
    """Build one frozen selected-family schedule using existing plan authorities.

    Seed arithmetic, RNG streams, plan order, multiplicity, protocol names, and
    purposes come only from the existing sampling and protocol generators.
    """
    # Validate the complete locked family evidence before constructing any plan.
    family = str(selection.family)
    request = SELECTED_FAMILY_EVIDENCE_REQUESTS.get(family)
    strict = selection.mode == "strict"
    if strict and request is None:
        raise ValueError(f"unsupported strict selected family: {family}")
    _validate_selected_dimension(selection)
    _validate_selected_mixture_audit(selection)
    feature_seed = derive_seed(
        int(base_seed), feature_id=int(feature_id), effect_space=str(effect_space)
    )

    # Fallbacks and terminals retain an explicit trace with every plan empty.
    if not strict:
        return _finish_schedule(
            feature_id=feature_id,
            selection=selection,
            point_record_sha256=point_record_sha256,
            feature_seed=feature_seed,
            request=NO_STABILITY_EVIDENCE_REQUEST,
            strict_k_values=(),
            margin_keys=(),
            bootstrap=(),
            leave_out=(),
            sample_size=(),
            raw_angle_plans=(),
            residual_angle_plans=(),
            request_count=0,
            no_work_reason=DELIBERATE_NON_EVALUATION_REASON,
        )
    assert request is not None

    # Mixture reuses its validated standalone assignment and schedules no work.
    if family == "multi_mode_directional_geometry":
        return _finish_schedule(
            feature_id=feature_id,
            selection=selection,
            point_record_sha256=point_record_sha256,
            feature_seed=feature_seed,
            request=request,
            strict_k_values=(),
            margin_keys=(),
            bootstrap=(),
            leave_out=(),
            sample_size=(),
            raw_angle_plans=(),
            residual_angle_plans=(),
            request_count=1,
            no_work_reason=None,
        )

    # Reuse the current scalar, leave-out, and sample-size generators verbatim.
    n_rows = int(n_rows)
    scalar_cfg = stability_config.scalar
    leave_cfg = stability_config.leave_out
    sample_cfg = stability_config.sample_size
    subspace_cfg = stability_config.subspace
    bootstrap = tuple(
        bootstrap_plans(
            seed=feature_seed,
            feature_id=int(feature_id),
            n_rows=n_rows,
            rounds=(
                int(scalar_cfg.bootstrap_rounds)
                if scalar_cfg.enabled and request.scalar_bootstrap_metrics
                else 0
            ),
        )
    )
    leave = tuple(
        leave_out_plans(
            seed=feature_seed + 17,
            feature_id=int(feature_id),
            n_rows=n_rows,
            group_labels=group_labels,
        )
        if leave_cfg.enabled
        else []
    )
    sample = tuple(
        sample_size_plans(
            seed=feature_seed + 31,
            feature_id=int(feature_id),
            n_rows=n_rows,
            targets=sample_cfg.target_sizes,
            rounds=(
                int(sample_cfg.strong_subset_rounds)
                if n_rows >= 32
                else min(
                    int(sample_cfg.subset_rounds),
                    int(sample_cfg.max_enumerated_subsets),
                )
            ),
        )
        if sample_cfg.enabled and n_rows >= 16
        else []
    )

    # Reuse the existing sequential angle RNG with the frozen family seed offsets.
    raw_angle_plans: tuple[AngleResamplePlan, ...] = ()
    residual_angle_plans: tuple[AngleResamplePlan, ...] = ()
    angle_k = _resolved_angle_k(request, selection.selected_k)
    if n_rows >= 32 and subspace_cfg.enabled and request.angle_source == "raw":
        assert angle_k is not None
        raw_angle_plans = _angle_plans(
            n_rows,
            float(subspace_cfg.resample_fraction),
            int(subspace_cfg.resample_rounds),
            feature_seed + 47,
            feature_id=int(feature_id),
            angle_k=angle_k,
            protocol="principal_angle",
            source="raw",
        )
    if (
        n_rows >= 32
        and subspace_cfg.enabled
        and request.angle_source == "centered_residual"
    ):
        assert angle_k is not None
        residual_angle_plans = _angle_plans(
            n_rows,
            float(subspace_cfg.resample_fraction),
            int(subspace_cfg.resample_rounds),
            feature_seed + 53,
            feature_id=int(feature_id),
            angle_k=angle_k,
            protocol="centered_residual_principal_angle",
            source="centered_residual",
        )
    strict_k_values = _strict_k_values_through(request, selection.selected_k)
    margin_keys = _margin_keys_through(request, strict_k_values)
    return _finish_schedule(
        feature_id=feature_id,
        selection=selection,
        point_record_sha256=point_record_sha256,
        feature_seed=feature_seed,
        request=request,
        strict_k_values=strict_k_values,
        margin_keys=margin_keys,
        bootstrap=bootstrap,
        leave_out=leave,
        sample_size=sample,
        raw_angle_plans=raw_angle_plans,
        residual_angle_plans=residual_angle_plans,
        request_count=1,
        no_work_reason=None,
    )


def selected_family_schedule_identity(
    schedule: SelectedFamilySchedule,
) -> dict[str, Any]:
    """Return the canonical JSON identity covered by ``schedule_digest``."""
    # Serialize every plan field, including duplicate bootstrap indices and digest.
    return {
        "version": int(schedule.version),
        "feature_id": int(schedule.feature_id),
        "point_selection": _point_selection_identity(schedule.point_selection),
        "evidence_request": _evidence_request_identity(schedule.evidence_request),
        "family": schedule.family,
        "selection_mode": schedule.selection_mode,
        "reported_selected_k": schedule.reported_selected_k,
        "angle_source": schedule.angle_source,
        "angle_k": schedule.angle_k,
        "evaluated_strict_k_values": list(schedule.evaluated_strict_k_values),
        "scalar_metrics": list(schedule.scalar_metrics),
        "margin_keys": list(schedule.margin_keys),
        "bootstrap": [_subset_plan_identity(plan) for plan in schedule.bootstrap],
        "leave_out": [_subset_plan_identity(plan) for plan in schedule.leave_out],
        "sample_size": [_subset_plan_identity(plan) for plan in schedule.sample_size],
        "raw_angle_plans": [
            _angle_plan_identity(plan) for plan in schedule.raw_angle_plans
        ],
        "residual_angle_plans": [
            _angle_plan_identity(plan) for plan in schedule.residual_angle_plans
        ],
        "feature_seed": int(schedule.feature_seed),
        "reuse_standalone_assignment": schedule.reuse_standalone_assignment,
        "request_count": int(schedule.request_count),
        "no_work_reason": schedule.no_work_reason,
        "point_record_sha256": schedule.point_record_sha256,
    }


def _evidence_request_identity(
    request: SelectedFamilyEvidenceRequest,
) -> dict[str, Any]:
    """Serialize the complete immutable selected-family evidence request."""
    # Preserve every omission and no-work flag as part of the schedule contract.
    return {
        "full_sample_point_requirements": list(
            request.full_sample_point_requirements
        ),
        "scalar_bootstrap_metrics": list(request.scalar_bootstrap_metrics),
        "angle_source": request.angle_source,
        "angle_k": request.angle_k,
        "leave_sample_margins": list(request.leave_sample_margins),
        "strict_k_values": list(request.strict_k_values),
        "stop_point_gates_at_selected_k": request.stop_point_gates_at_selected_k,
        "low_context_qualification": request.low_context_qualification,
        "reuse_assignment_stability": request.reuse_assignment_stability,
        "no_stability_protocol": request.no_stability_protocol,
        "explicitly_omitted": list(request.explicitly_omitted),
    }


def _angle_plan_identity(plan: AngleResamplePlan) -> dict[str, Any]:
    """Serialize one explicit angle plan including feature and resolved dimension."""
    # Bind the feature and angle k so identical index draws cannot cross schedules.
    return {
        "feature_id": int(plan.feature_id),
        "protocol": plan.protocol,
        "source": plan.source,
        "angle_k": int(plan.angle_k),
        "seed": int(plan.seed),
        "replicate_id": int(plan.replicate_id),
        "indices": list(plan.indices),
        "digest": plan.digest,
    }


def _subset_plan_identity(plan: SubsetPlan) -> dict[str, Any]:
    """Serialize one existing subset plan without changing its identity."""
    # Keep duplicate indices and the generator-produced digest verbatim.
    return {
        "global_seed": int(plan.global_seed),
        "feature_id": int(plan.feature_id),
        "protocol": plan.protocol,
        "target_or_group_identity": plan.target_or_group_identity,
        "replicate_id": int(plan.replicate_id),
        "purpose": plan.purpose,
        "indices": list(plan.indices),
        "digest": plan.digest,
    }


def _finish_schedule(
    *,
    feature_id: int,
    selection: PointSelection,
    point_record_sha256: str,
    feature_seed: int,
    request: SelectedFamilyEvidenceRequest,
    strict_k_values: tuple[int, ...],
    margin_keys: tuple[str, ...],
    bootstrap: tuple[SubsetPlan, ...],
    leave_out: tuple[SubsetPlan, ...],
    sample_size: tuple[SubsetPlan, ...],
    raw_angle_plans: tuple[AngleResamplePlan, ...],
    residual_angle_plans: tuple[AngleResamplePlan, ...],
    request_count: int,
    no_work_reason: str | None,
) -> SelectedFamilySchedule:
    """Construct one immutable schedule and bind its canonical digest."""
    # Build once without a digest, then hash the exact public identity projection.
    angle_k = _resolved_angle_k(request, selection.selected_k)
    provisional = SelectedFamilySchedule(
        feature_id=int(feature_id),
        point_selection=selection,
        evidence_request=request,
        family=str(selection.family),
        selection_mode=str(selection.mode),
        reported_selected_k=selection.selected_k,
        angle_source=request.angle_source,
        angle_k=angle_k,
        evaluated_strict_k_values=strict_k_values,
        scalar_metrics=request.scalar_bootstrap_metrics,
        margin_keys=margin_keys,
        bootstrap=bootstrap,
        leave_out=leave_out,
        sample_size=sample_size,
        raw_angle_plans=raw_angle_plans,
        residual_angle_plans=residual_angle_plans,
        feature_seed=int(feature_seed),
        reuse_standalone_assignment=request.reuse_assignment_stability,
        request_count=int(request_count),
        no_work_reason=no_work_reason,
        point_record_sha256=str(point_record_sha256),
        schedule_digest="",
    )
    return SelectedFamilySchedule(
        **{
            **provisional.__dict__,
            "schedule_digest": canonical_json_digest(
                selected_family_schedule_identity(provisional)
            ),
        }
    )


def _validate_selected_dimension(selection: PointSelection) -> None:
    """Reject family/dimension combinations outside the frozen point contract."""
    # Axis/ray/mixture report null; span and strict residual report supported k.
    family = selection.family
    selected_k = selection.selected_k
    if family in {
        "directed_ray",
        "axis_or_antipodal",
        "multi_mode_directional_geometry",
    } and selected_k is not None:
        raise ValueError(f"{family} must report selected_k=null")
    if family == "global_2D_directional_subspace" and selected_k != 2:
        raise ValueError("global_2D_directional_subspace must use selected_k=2")
    if family == "global_kD_directional_subspace" and selected_k not in (3, 4, 8):
        raise ValueError(
            "global_kD_directional_subspace must use selected_k in 3, 4, 8"
        )
    if family == "residual_lowD_k" and selection.mode == "strict":
        if selected_k not in (2, 3, 4):
            raise ValueError("strict residual_lowD_k must use selected_k in 2, 3, 4")


def _validate_selected_mixture_audit(selection: PointSelection) -> None:
    """Reject any mixture family not backed by complete accepted point evidence."""
    # Reporting permits strict or fallback mixture only after the same audit accepts.
    if selection.family != "multi_mode_directional_geometry":
        return
    audit = selection.mixture_audit_state
    if (
        audit.status != "accepted"
        or audit.acceptance != "accepted"
        or audit.failed_gates
        or audit.unavailable_gates
    ):
        raise ValueError(
            "multi_mode_directional_geometry requires an accepted mixture audit"
        )


def _resolved_angle_k(
    request: SelectedFamilyEvidenceRequest, selected_k: int | None
) -> int | None:
    """Resolve the retained protocol dimension independently of plan availability."""
    # Low-row early returns suppress draws, not the locked angle dimension metadata.
    return selected_k if request.angle_k == "selected_k" else request.angle_k


def _strict_k_values_through(
    request: SelectedFamilyEvidenceRequest, selected_k: int | None
) -> tuple[int, ...]:
    """Return strict candidates in frozen order through the locked dimension."""
    # Ray and axis have no candidate-k loop; span/residual stop exactly at k-star.
    if not request.strict_k_values:
        return ()
    if selected_k not in request.strict_k_values:
        raise ValueError("selected_k must be a declared strict dimension")
    end = request.strict_k_values.index(int(selected_k)) + 1
    return request.strict_k_values[:end]


def _margin_keys_through(
    request: SelectedFamilyEvidenceRequest, strict_k_values: tuple[int, ...]
) -> tuple[str, ...]:
    """Expand family-local margin templates through the locked strict k."""
    # Preserve static keys once and expand k-local keys in ascending candidate order.
    static = tuple(key for key in request.leave_sample_margins if "{k}" not in key)
    templated = tuple(key for key in request.leave_sample_margins if "{k}" in key)
    expanded: list[str] = list(static)
    for k in strict_k_values:
        expanded.extend(key.format(k=k) for key in templated)
    return tuple(expanded)


def _angle_plans(
    n_rows: int,
    fraction: float,
    rounds: int,
    seed: int,
    *,
    feature_id: int,
    angle_k: int,
    protocol: str,
    source: AngleSource,
) -> tuple[AngleResamplePlan, ...]:
    """Freeze existing sequential angle draws with complete protocol identity."""
    # Delegate every draw/order choice, then address the resulting immutable plans.
    plans: list[AngleResamplePlan] = []
    for replicate, values in enumerate(
        subspace_resample_indices(n_rows, fraction, rounds, seed)
    ):
        indices = tuple(int(index) for index in values.tolist())
        identity = {
            "feature_id": int(feature_id),
            "protocol": protocol,
            "source": source,
            "angle_k": int(angle_k),
            "seed": int(seed),
            "replicate_id": int(replicate),
            "indices": list(indices),
        }
        plans.append(
            AngleResamplePlan(
                feature_id=int(feature_id),
                protocol=protocol,
                source=source,
                angle_k=int(angle_k),
                seed=int(seed),
                replicate_id=int(replicate),
                indices=indices,
                digest=canonical_json_digest(identity),
            )
        )
    return tuple(plans)
