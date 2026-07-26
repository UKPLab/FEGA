from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_metrics.metrics import (
    c_ray_pairwise_final_resid,
    centered_residual_spectrum_final_resid,
    effective_rank_from_spectrum,
    span_spectrum_final_resid,
)
from fega.core.geometry_reporting.artifacts import (
    load_build_and_write_point_geometry_records,
    load_point_geometry_records,
    point_geometry_records_path,
)
from fega.core.geometry_reporting.thresholds import get_threshold_profile
from fega.core.source_fingerprint import canonical_source_fingerprint
from fega.core.stability.artifacts import (
    STABILITY_PUBLIC_SCHEMA_VERSION,
    StabilityFeatureBlock,
    StabilityInputs,
    build_selected_family_checkpoint_fingerprint,
    build_selected_family_checkpoint_payload,
    delete_stability_checkpoint,
    iter_stability_feature_descriptors,
    load_selected_family_checkpoint,
    load_stability_feature_block,
    load_stability_inputs,
    scientific_stability_config,
    selected_family_required_protocol_ids,
    source_paths,
    write_stability_checkpoint,
    write_stability_scores,
)
from fega.core.stability.metrics import (
    final_resid_unit_rows,
    scheduled_principal_angle_stability,
)
from fega.core.stability.protocols import (
    aggregate_c_ray_bootstrap,
    aggregate_subset_protocol,
    angle_stability_decision,
    evaluate_subset_plans,
    locked_family_result,
    signed_gate_margins,
)
from fega.core.stability.sampling import low_context_protocol
from fega.core.stability.schedule import (
    SelectedFamilySchedule,
    build_selected_family_schedule,
)
from fega.paths import stability_checkpoint_path

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScheduledFeature:
    """Bind one authoritative point lock to its canonical filtered row inventory."""

    schedule: SelectedFamilySchedule
    point_record: dict[str, Any]
    raw_rows: torch.Tensor
    unit_rows: torch.Tensor
    gram: torch.Tensor
    valid_counts: dict[str, int]


@dataclass(frozen=True)
class _ExecutionResult:
    """Separate one scientific feature record from non-scientific wall telemetry."""

    record: dict[str, Any]
    protocol_seconds: dict[str, float]


def run_stability(
    config: FEGAPipelineConfig, resources: Any | None = None
) -> None:
    """Execute immutable selected-family stability schedules from final-residual rows.

    ``resources`` is accepted only for loader caching and type compatibility. Stability
    never creates model resources, materializes vocabulary coordinates, constructs a
    factor, fits vMF, reselects BIC, refits assignments, or classifies a subset.
    """
    # Load or create the reporting-owned point artifact, then validate it durably.
    started = time.perf_counter()
    cfg = config.phases.stability
    if cfg.effect_space != "final_resid":
        raise ValueError("stability requires effect_space='final_resid'.")
    point_bundle = _load_or_build_point_bundle(config, resources)
    inputs = load_stability_inputs(config, "final_resid", resources)
    base_seed = int(cfg.seed if cfg.seed is not None else config.seed.global_)
    schedules = _build_schedules(
        config,
        inputs,
        point_bundle,
        base_seed=base_seed,
    )
    fingerprint = build_selected_family_checkpoint_fingerprint(
        config=config,
        point_bundle=point_bundle,
        schedules=schedules,
    )

    # Reuse only an exact current checkpoint when resume is enabled.
    checkpoint_path = stability_checkpoint_path(config)
    loaded = load_selected_family_checkpoint(
        checkpoint_path,
        expected_fingerprint=fingerprint,
        expected_schedules=schedules,
    )
    candidate_records = list(loaded.records) if cfg.resume else []
    completed, incomplete = _partition_current_records(
        schedules, candidate_records
    )
    work = _iter_incomplete_scheduled_features(
        config,
        inputs,
        point_bundle,
        schedules=schedules,
        incomplete_feature_ids={int(item.feature_id) for item in incomplete},
    )
    records, protocol_seconds = _execute_scheduled_features(
        work,
        config=config,
        fingerprint=fingerprint,
        schedules=schedules,
        initial_records=completed,
    )

    # Serialize scientific records canonically; keep timing only in execution telemetry.
    records.sort(key=lambda item: int(item["feature_id"]))
    per_feature = {str(int(item["feature_id"])): item for item in records}
    source_fingerprint = canonical_source_fingerprint(
        inputs.geometry_metrics_inputs.manifest,
        inputs.geometry_metrics_inputs.summary,
    )
    payload = {
        "phase": "stability",
        "schema_version": STABILITY_PUBLIC_SCHEMA_VERSION,
        "canonical_source_fingerprint": source_fingerprint,
        "fingerprint": fingerprint,
        "config": scientific_stability_config(config),
        "source_paths": {"final_resid": source_paths(inputs)},
        "execution": {
            "workers": int(cfg.workers),
            "resume": bool(cfg.resume),
            "checkpoint_flush_features": int(cfg.checkpoint_flush_features),
            "checkpoint_load": {
                "status": loaded.status,
                "rejection_reason": loaded.rejection_reason,
                "records_reused": len(completed),
            },
            "telemetry": {
                "wall_seconds": float(time.perf_counter() - started),
                "features_executed": len(incomplete),
                "features_reused": len(completed),
                "family_protocol_wall_seconds": dict(sorted(protocol_seconds.items())),
            },
        },
        "summary": {
            "requested_effect_spaces": ["final_resid"],
            "usable_effect_spaces": 1,
            "features_total": len(schedules),
            "family_protocol_counters": _family_protocol_counters(records),
        },
        "effect_spaces": {
            "final_resid": {
                "status": "ok",
                "summary": {
                    "features_total": len(schedules),
                    "features_scored": len(records),
                    "features_skipped": 0,
                },
                "source_paths": source_paths(inputs),
                "per_feature": per_feature,
            }
        },
    }
    write_stability_scores(config, payload)
    delete_stability_checkpoint(config)
    _logger.info("selected-family stability complete: features=%d", len(records))


def _load_or_build_point_bundle(
    config: FEGAPipelineConfig, resources: Any | None
) -> dict[str, Any]:
    """Load the durable point authority, building it once only when absent."""
    # A present invalid artifact fails closed; only literal absence invokes its owner.
    path = point_geometry_records_path(config)
    if not path.exists():
        load_build_and_write_point_geometry_records(config, resources)
    return load_point_geometry_records(config, resources)


def _build_schedules(
    config: FEGAPipelineConfig,
    inputs: StabilityInputs,
    point_bundle: Mapping[str, Any],
    *,
    base_seed: int,
) -> list[SelectedFamilySchedule]:
    """Freeze the complete schedule inventory without retaining feature tensors."""
    # Require identical point, selection, hash, and compute-effect feature inventories.
    point_records = point_bundle.get("point_records")
    point_selections = point_bundle.get("point_selections")
    point_hashes = point_bundle.get("point_record_hashes")
    if not isinstance(point_records, list) or not isinstance(point_selections, Mapping):
        raise ValueError("validated point bundle is missing records or selections")
    if not isinstance(point_hashes, Mapping):
        raise ValueError("validated point bundle is missing point record hashes")
    # Derive each schedule from one transient block, then release its row tensors.
    schedules: list[SelectedFamilySchedule] = []
    eps = float(config.phases.geometry_metrics.c_ray.eps)
    for point_record, block in _iter_authoritative_blocks(point_records, inputs):
        feature_id = int(point_record["feature_id"])
        raw_source = _required_block_rows(block, inputs.gram)
        _unit_rows, counts, valid_indices = final_resid_unit_rows(
            raw_source, inputs.gram, eps=eps
        )
        point_n_valid = _exact_count(point_record.get("n_valid"), feature_id)
        if point_n_valid != int(counts["n_valid"]):
            raise ValueError(
                f"feature {feature_id} point n_valid mismatch: "
                f"point={point_n_valid} rows={counts['n_valid']}"
            )
        selection = point_selections.get(feature_id)
        point_hash = point_hashes.get(str(feature_id))
        if selection is None or not isinstance(point_hash, str) or not point_hash:
            raise ValueError(f"feature {feature_id} point authority is incomplete")
        labels = _filtered_group_labels(block.group_labels, valid_indices)
        schedule = build_selected_family_schedule(
            selection=selection,
            feature_id=feature_id,
            point_record_sha256=point_hash,
            base_seed=int(base_seed),
            effect_space="final_resid",
            n_rows=int(counts["n_valid"]),
            group_labels=labels,
            stability_config=config.phases.stability,
        )
        schedules.append(schedule)
        del _unit_rows, raw_source, block
    return schedules


def _iter_authoritative_blocks(
    point_records: Sequence[dict[str, Any]], inputs: StabilityInputs
) -> Iterator[tuple[dict[str, Any], StabilityFeatureBlock]]:
    """Yield point records and row blocks in one validated canonical inventory."""
    # Validate every position while keeping only the current loader block live.
    blocks = iter(_iter_bounded_feature_blocks(inputs))
    for point_record in point_records:
        try:
            block = next(blocks)
        except StopIteration as exc:
            raise ValueError(
                "point and final-residual feature inventory length mismatch"
            ) from exc
        point_id = int(point_record["feature_id"])
        if int(block.feature_id) != point_id:
            raise ValueError("point and final-residual feature inventory order mismatch")
        yield point_record, block
        del block
    try:
        next(blocks)
    except StopIteration:
        return
    raise ValueError("point and final-residual feature inventory length mismatch")


def _iter_bounded_feature_blocks(
    inputs: StabilityInputs,
) -> Iterator[StabilityFeatureBlock]:
    """Load canonical feature blocks while retaining at most one source shard."""
    # Clear prior shard payloads at each shard boundary before loading the next block.
    shard_cache: dict[str, dict[str, Any]] = {}
    current_shard: str | None = None
    first = True
    for descriptor in iter_stability_feature_descriptors(inputs):
        if first or descriptor.tensor_shard != current_shard:
            shard_cache.clear()
            current_shard = descriptor.tensor_shard
            first = False
        yield load_stability_feature_block(
            inputs, descriptor, shard_cache=shard_cache
        )


def _iter_incomplete_scheduled_features(
    config: FEGAPipelineConfig,
    inputs: StabilityInputs,
    point_bundle: Mapping[str, Any],
    *,
    schedules: Sequence[SelectedFamilySchedule],
    incomplete_feature_ids: set[int],
) -> Iterator[_ScheduledFeature]:
    """Revalidate the frozen inventory and prepare only incomplete features lazily."""
    # Recompute the exact valid-row authority while retaining no completed feature tensor.
    point_records = point_bundle.get("point_records")
    if not isinstance(point_records, list) or len(point_records) != len(schedules):
        raise ValueError("validated point bundle and schedule inventory mismatch")
    eps = float(config.phases.geometry_metrics.c_ray.eps)
    for position, (point_record, block) in enumerate(
        _iter_authoritative_blocks(point_records, inputs)
    ):
        schedule = schedules[position]
        feature_id = int(point_record["feature_id"])
        if int(schedule.feature_id) != feature_id:
            raise ValueError("point and frozen schedule inventory order mismatch")
        raw_source = _required_block_rows(block, inputs.gram)
        unit_rows, counts, valid_indices = final_resid_unit_rows(
            raw_source, inputs.gram, eps=eps
        )
        point_n_valid = _exact_count(point_record.get("n_valid"), feature_id)
        if point_n_valid != int(counts["n_valid"]):
            raise ValueError(
                f"feature {feature_id} point n_valid mismatch: "
                f"point={point_n_valid} rows={counts['n_valid']}"
            )
        if feature_id not in incomplete_feature_ids:
            del unit_rows, raw_source, block
            continue
        raw_valid = raw_source.index_select(
            0, torch.as_tensor(valid_indices, dtype=torch.long)
        )
        yield _ScheduledFeature(
            schedule=schedule,
            point_record=point_record,
            raw_rows=raw_valid,
            unit_rows=unit_rows,
            gram=inputs.gram,
            valid_counts=counts,
        )
        del raw_valid, unit_rows, raw_source, block


def _partition_current_records(
    schedules: Sequence[SelectedFamilySchedule],
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[SelectedFamilySchedule]]:
    """Partition exact current records from incomplete schedules in canonical order."""
    # Never reuse a feature record whose schedule identity is stale or duplicated.
    schedule_by_id = {int(item.feature_id): item for item in schedules}
    current: dict[int, dict[str, Any]] = {}
    for record in records:
        feature_id = int(record.get("feature_id", -1))
        schedule = schedule_by_id.get(feature_id)
        if (
            schedule is not None
            and record.get("schedule_digest") == schedule.schedule_digest
            and feature_id not in current
        ):
            current[feature_id] = record
    ordered_schedules = sorted(schedules, key=lambda item: int(item.feature_id))
    completed = [
        current[int(schedule.feature_id)]
        for schedule in ordered_schedules
        if int(schedule.feature_id) in current
    ]
    incomplete = [
        schedule
        for schedule in ordered_schedules
        if int(schedule.feature_id) not in current
    ]
    return completed, incomplete


def _execute_scheduled_features(
    work: Iterable[_ScheduledFeature],
    *,
    config: FEGAPipelineConfig,
    fingerprint: Mapping[str, Any],
    schedules: Sequence[SelectedFamilySchedule],
    initial_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Execute only incomplete schedules and atomically checkpoint ordered records."""
    # Consume results canonically through a worker-sized window of live feature tensors.
    records = list(initial_records)
    protocol_seconds: dict[str, float] = {}
    flush_every = max(1, int(config.phases.stability.checkpoint_flush_features))
    since_flush = 0

    def execute(item: _ScheduledFeature) -> _ExecutionResult:
        """Execute one immutable feature schedule with current configuration."""
        # Thresholds are immutable and resolved independently inside each thread.
        return _execute_one(
            item,
            config,
            get_threshold_profile(config.phases.geometry_reporting.threshold_profile),
        )

    try:
        for result in _iter_ordered_execution_results(
            work, execute, workers=int(config.phases.stability.workers)
        ):
            records.append(result.record)
            since_flush += 1
            for key, value in result.protocol_seconds.items():
                protocol_seconds[key] = protocol_seconds.get(key, 0.0) + float(value)
            if since_flush >= flush_every:
                _write_current_checkpoint(config, fingerprint, records, schedules)
                since_flush = 0
    except BaseException:
        if since_flush:
            _write_current_checkpoint(config, fingerprint, records, schedules)
        raise
    if since_flush or not stability_checkpoint_path(config).exists():
        _write_current_checkpoint(config, fingerprint, records, schedules)
    return records, protocol_seconds


def _iter_ordered_execution_results(
    work: Iterable[_ScheduledFeature],
    execute: Callable[[_ScheduledFeature], _ExecutionResult],
    *,
    workers: int,
) -> Iterator[_ExecutionResult]:
    """Execute lazily with at most one prepared feature resident per worker."""
    # Sequential execution consumes one item; parallel execution preserves submit order.
    if workers < 1:
        raise ValueError("stability workers must be positive")
    if workers == 1:
        yield from map(execute, work)
        return
    items = iter(work)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        for _ in range(workers):
            try:
                pending.append(executor.submit(execute, next(items)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(execute, next(items)))
            except StopIteration:
                pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _execute_one(
    item: _ScheduledFeature, config: FEGAPipelineConfig, thresholds: Any
) -> _ExecutionResult:
    """Execute exactly one selected-family schedule without relabeling or widening."""
    # Bind the locked identity before evaluating any scheduled protocol.
    schedule = item.schedule
    record = {
        "feature_id": int(schedule.feature_id),
        "family": schedule.family,
        "selection_mode": schedule.selection_mode,
        "selected_k": schedule.reported_selected_k,
        "point_reason": schedule.point_selection.point_reason,
        "schedule_digest": schedule.schedule_digest,
        "point_record_sha256": schedule.point_record_sha256,
        "n_valid": int(item.valid_counts["n_valid"]),
    }
    if schedule.no_work_reason is not None:
        no_work = {
            "status": "not_evaluated",
            "reason": schedule.no_work_reason,
            "plan_digest": _empty_digest(),
            "counters": _counts(non_applicable=1),
        }
        record["selected_family_evidence"] = {
            "required_protocol_ids": selected_family_required_protocol_ids(schedule),
            "no_work_reason": schedule.no_work_reason,
            "protocols": {"deliberate_non_evaluation": no_work},
            "protocol_counters": {
                "deliberate_non_evaluation": no_work["counters"]
            },
        }
        return _ExecutionResult(record, {})
    if schedule.reuse_standalone_assignment:
        timings: dict[str, float] = {}
        reused = _timed(
            timings,
            schedule.family,
            "standalone_assignment_reuse",
            lambda: _reuse_standalone_assignment(item.point_record),
        )
        record["selected_family_evidence"] = {
            "required_protocol_ids": selected_family_required_protocol_ids(schedule),
            "no_work_reason": None,
            "protocols": {"standalone_assignment_reuse": reused},
            "protocol_counters": {
                "standalone_assignment_reuse": reused["counters"]
            },
        }
        return _ExecutionResult(record, timings)

    # Evaluate each retained protocol against only the selected family's evidence map.
    timings: dict[str, float] = {}
    point_margins = _point_margins(item.point_record, schedule, thresholds)
    bootstrap = _timed(
        timings,
        schedule.family,
        "bootstrap",
        lambda: _execute_bootstrap(item, config, thresholds),
    )
    angle = _timed(
        timings,
        schedule.family,
        "angle",
        lambda: _execute_angle(item, config, thresholds),
    )
    leave_out = _timed(
        timings,
        schedule.family,
        "leave_out",
        lambda: _execute_subset_block(item, schedule.leave_out, config, thresholds),
    )
    sample_size = _timed(
        timings,
        schedule.family,
        "sample_size",
        lambda: _execute_subset_block(item, schedule.sample_size, config, thresholds),
    )
    leave_summary = aggregate_subset_protocol(
        point_margins=point_margins, block=leave_out
    )
    sample_summary = aggregate_subset_protocol(
        point_margins=point_margins, block=sample_size
    )
    low_context = _timed(
        timings,
        schedule.family,
        "low_context_qualification",
        lambda: _execute_low_context_qualification(item, angle),
    )
    instability = sum(
        int(block.get("instability_count", 0))
        for block in (bootstrap, angle, leave_summary, sample_summary)
    )
    failures = sum(
        int(block.get("required_failure_count", 0))
        for block in (bootstrap, angle, leave_summary, sample_summary)
    )
    protocols = {
        "low_context_qualification": low_context,
        "bootstrap": bootstrap,
        "angle": angle,
        "leave_out": leave_summary,
        "sample_size": sample_summary,
    }
    record["selected_family_evidence"] = {
        "required_protocol_ids": selected_family_required_protocol_ids(schedule),
        "no_work_reason": None,
        "completed_instability_count": int(instability),
        "required_failure_count": int(failures),
        "point_margins": point_margins,
        "protocols": protocols,
        "protocol_counters": {
            key: dict(value.get("counters", _counts()))
            for key, value in protocols.items()
        },
    }
    return _ExecutionResult(record, timings)


def _execute_low_context_qualification(
    item: _ScheduledFeature, angle: Mapping[str, Any]
) -> dict[str, Any]:
    """Record the existing low-context authority without mapping reporting semantics."""
    # Persist only observed qualification inputs and the authority's raw protocol result.
    schedule = item.schedule
    n_valid = int(item.valid_counts["n_valid"])
    result = dict(low_context_protocol(n_valid))
    return {
        **result,
        "reason": None if result["status"] == "ok" else result["status"],
        "observed_n_valid": n_valid,
        "observed_numerical_rank": angle.get("numerical_rank"),
        "required_n_valid": 32,
        "required_k": schedule.angle_k,
        "counters": _counts(requested=1, valid=1),
    }


def _execute_bootstrap(
    item: _ScheduledFeature, config: FEGAPipelineConfig, thresholds: Any
) -> dict[str, Any]:
    """Execute metric-local c_ray bootstrap only for selected ray or axis."""
    # The immutable schedule is the sole source of requested replicate indices.
    schedule = item.schedule
    if not schedule.scalar_metrics:
        return {
            "status": "not_applicable",
            "plan_digest": _empty_digest(),
            "replicates": [],
            "counters": _counts(non_applicable=1),
            "instability_count": 0,
            "required_failure_count": 0,
        }

    def evaluate(plan: Any) -> dict[str, Any]:
        """Compute only c_ray for one scheduled with-replacement subset."""
        # Duplicate bootstrap indices are preserved by tensor index selection.
        rows = _subset_rows(item.unit_rows, plan.indices)
        value = c_ray_pairwise_final_resid(
            rows, item_gram, eps=float(config.phases.geometry_metrics.c_ray.eps)
        ).c_ray
        scalar = _json_scalar(value)
        return {
            "status": "valid" if scalar is not None else "failed",
            "metrics": {"c_ray": scalar},
        }

    item_gram = _gram_for(item)
    block = evaluate_subset_plans(schedule.bootstrap, evaluate)
    return aggregate_c_ray_bootstrap(
        family=schedule.family,
        point_estimate=item.point_record.get("c_ray"),
        block=block,
        quantiles=config.phases.stability.scalar.ci_quantiles,
        threshold=thresholds.tau_c_ray,
    )


def _execute_angle(
    item: _ScheduledFeature, config: FEGAPipelineConfig, thresholds: Any
) -> dict[str, Any]:
    """Execute one raw or centered-residual angle at the locked schedule k."""
    # Plans and the one selected dimension are fixed before execution.
    schedule = item.schedule
    if schedule.angle_source == "none" or schedule.angle_k is None:
        return {
            "status": "not_applicable",
            "plan_digest": _empty_digest(),
            "replicates": [],
            "counters": _counts(non_applicable=1),
            "instability_count": 0,
            "required_failure_count": 0,
        }
    plans = (
        schedule.raw_angle_plans
        if schedule.angle_source == "raw"
        else schedule.residual_angle_plans
    )
    result = scheduled_principal_angle_stability(
        item.raw_rows,
        _gram_for(item),
        plans=plans,
        source=schedule.angle_source,
        k=int(schedule.angle_k),
        angle_quantile=float(config.phases.stability.subspace.angle_p90_quantile),
        eig_floor=float(config.phases.stability.subspace.eig_floor),
    )
    numerical_status = str(result["status"])
    if numerical_status == "ok":
        decision = angle_stability_decision(
            result.get("angle_p90_deg"), int(schedule.angle_k), thresholds
        )
    elif numerical_status in {
        "exploratory",
        "insufficient_contexts",
        "not_applicable",
    }:
        decision = numerical_status
    else:
        decision = "unavailable"
    return {
        **result,
        "decision": decision,
        "instability_count": int(decision == "unstable"),
        "required_failure_count": int(decision == "unavailable"),
    }


def _execute_subset_block(
    item: _ScheduledFeature,
    plans: Sequence[Any],
    config: FEGAPipelineConfig,
    thresholds: Any,
) -> dict[str, Any]:
    """Execute family-local margins for one immutable leave/sample plan block."""
    # Each subset evaluates no family or dimension outside the locked request.
    schedule = item.schedule

    def evaluate(plan: Any) -> dict[str, Any]:
        """Compute one subset's retained metrics, margins, and locked support."""
        # Index the canonical unit rows by the exact predeclared subset identity.
        evidence = _family_subset_evidence(
            config,
            _subset_rows(item.unit_rows, plan.indices),
            _gram_for(item),
            family=schedule.family,
            selected_k=schedule.reported_selected_k,
            strict_k_values=schedule.evaluated_strict_k_values,
            margin_keys=schedule.margin_keys,
            thresholds=thresholds,
        )
        return {"status": "valid", **evidence}

    return evaluate_subset_plans(plans, evaluate)


def _family_subset_evidence(
    config: Any,
    rows: torch.Tensor,
    gram: torch.Tensor,
    *,
    family: str,
    selected_k: int | None,
    strict_k_values: Sequence[int],
    margin_keys: Sequence[str],
    thresholds: Any,
) -> dict[str, Any]:
    """Compute only one locked family's retained geometry metrics and margins."""
    # Select exactly one approved metric family; no universal profile is constructed.
    metrics: dict[str, Any] = {}
    if family in {"directed_ray", "axis_or_antipodal"}:
        ray = c_ray_pairwise_final_resid(
            rows, gram, eps=float(config.phases.geometry_metrics.c_ray.eps)
        )
        span = span_spectrum_final_resid(
            rows,
            gram,
            k_values=(1,),
            eps=float(config.phases.geometry_metrics.span.eps),
        )
        metrics.update({"c_ray": ray.c_ray, "s_span_1": span.s_span[1]})
        if family == "axis_or_antipodal":
            metrics["b_axis"] = span.b_axis
    elif family in {
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
    }:
        span = span_spectrum_final_resid(
            rows,
            gram,
            k_values=strict_k_values,
            eps=float(config.phases.geometry_metrics.span.eps),
        )
        rank = effective_rank_from_spectrum(
            span.eigenvalues,
            eps=float(config.phases.geometry_metrics.effective_rank.eps),
        )
        metrics["r_span_pr"] = rank.r_pr
        for key, value in span.s_span.items():
            metrics[f"s_span_{key}"] = value
        for key, value in span.u_span.items():
            metrics[f"u_span_{key}"] = value
        for key, value in span.d_span.items():
            metrics[f"d_span_{key}"] = value
    elif family == "residual_lowD_k":
        residual = centered_residual_spectrum_final_resid(
            rows,
            gram,
            k_values=strict_k_values,
            eps=float(config.phases.geometry_metrics.resid.eps),
        )
        rank = effective_rank_from_spectrum(
            residual.eigenvalues,
            eps=float(config.phases.geometry_metrics.effective_rank.eps),
        )
        metrics.update({"e_res": residual.e_res, "r_ctr_pr": rank.r_pr})
        for key, value in residual.s_res.items():
            metrics[f"s_res_{key}"] = value
    else:
        raise ValueError(f"unsupported executable selected family: {family}")
    safe_metrics = {key: _json_scalar(value) for key, value in metrics.items()}
    all_margins = signed_gate_margins(
        safe_metrics,
        thresholds,
        k_values=strict_k_values or (1,),
    )
    margins = {key: all_margins.get(key) for key in margin_keys}
    locked = _locked_family_result(
        family,
        selected_k=selected_k,
        strict_k_values=strict_k_values,
        margins=margins,
    )
    return {"metrics": safe_metrics, "margins": margins, **locked}


def _locked_family_result(
    family: str,
    *,
    selected_k: int | None,
    strict_k_values: Sequence[int],
    margins: Mapping[str, float | None],
) -> dict[str, Any]:
    """Expose the protocol authority for focused locked-family behavior tests."""
    # Keep production and verification on the same exact mismatch/unavailability policy.
    return locked_family_result(
        family,
        selected_k=selected_k,
        strict_k_values=strict_k_values,
        margins=margins,
    )


def _reuse_standalone_assignment(point_record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one complete authoritative standalone assignment-stability record."""
    # Accepted mixture schedules may reuse the record exactly once and perform zero work.
    assignment = point_record.get("assignment_stability")
    if not isinstance(assignment, Mapping) or assignment.get("status") != "available":
        raise ValueError("accepted mixture assignment stability is unavailable")
    return {
        "status": "reused",
        "assignment_stability": copy.deepcopy(dict(assignment)),
        "counters": _counts(requested=1, valid=1),
        "plan_digest": _empty_digest(),
    }


def _point_margins(
    point_record: Mapping[str, Any],
    schedule: SelectedFamilySchedule,
    thresholds: Any,
) -> dict[str, float | None]:
    """Project the authoritative point record onto the schedule's exact margins."""
    # Never recompute point selection or add unrequested margin keys.
    all_margins = signed_gate_margins(
        point_record,
        thresholds,
        k_values=schedule.evaluated_strict_k_values or (1,),
    )
    margins = {key: all_margins.get(key) for key in schedule.margin_keys}
    missing = [key for key, value in margins.items() if value is None]
    if missing:
        raise ValueError(
            f"feature {schedule.feature_id} point margins unavailable: {missing}"
        )
    return margins


def _required_block_rows(
    block: StabilityFeatureBlock, gram: torch.Tensor
) -> torch.Tensor:
    """Return one rank-2 block or an empty row matrix for a skipped source feature."""
    # Point n_valid reconciliation decides whether an empty source is authoritative.
    if block.rows is not None:
        return block.rows
    return torch.empty((0, int(gram.shape[0])), dtype=torch.float32)


def _filtered_group_labels(
    labels: list[str | None] | None, valid_indices: Sequence[int]
) -> list[str | None] | None:
    """Apply the exact scientific valid-row mask to optional group labels."""
    # Retain None positions because schedule identity depends on complete row alignment.
    if labels is None:
        return None
    filtered = [labels[int(index)] for index in valid_indices]
    return filtered if any(label is not None for label in filtered) else None


def _gram_for(item: _ScheduledFeature) -> torch.Tensor:
    """Return the canonical Gram bound to one prepared feature."""
    # Every subset and angle path consumes the same loader-returned tensor authority.
    return item.gram


def _subset_rows(rows: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Index canonical rows while preserving duplicate bootstrap multiplicity."""
    # Tensor index selection follows the immutable sorted plan exactly.
    index = torch.as_tensor(tuple(int(value) for value in indices), dtype=torch.long)
    return rows.index_select(0, index)


def _write_current_checkpoint(
    config: FEGAPipelineConfig,
    fingerprint: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    schedules: Sequence[SelectedFamilySchedule],
) -> None:
    """Atomically write the current canonical completed-record prefix or subset."""
    # The artifact helper sorts and hashes records against every current schedule.
    payload = build_selected_family_checkpoint_payload(
        fingerprint=fingerprint,
        records=records,
        schedules=schedules,
    )
    write_stability_checkpoint(config, payload)


def _family_protocol_counters(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate deterministic counters by locked family and protocol."""
    # Wall time is intentionally absent from this scientific summary.
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for record in records:
        family = str(record["family"])
        family_summary = summary.setdefault(family, {})
        stability = record.get("selected_family_evidence")
        counters = (
            stability.get("protocol_counters", {})
            if isinstance(stability, Mapping)
            else {}
        )
        for protocol, values in counters.items():
            target = family_summary.setdefault(str(protocol), _counts())
            for key in target:
                target[key] += int(values.get(key, 0))
    return {
        family: {protocol: values for protocol, values in sorted(block.items())}
        for family, block in sorted(summary.items())
    }


def _timed(
    timings: dict[str, float],
    family: str,
    protocol: str,
    operation: Any,
) -> dict[str, Any]:
    """Measure one protocol outside its scientific result block."""
    # Telemetry keys are family-local and excluded from checkpoint/output equality checks.
    started = time.perf_counter()
    result = operation()
    timings[f"{family}:{protocol}"] = float(time.perf_counter() - started)
    return result


def _exact_count(value: Any, feature_id: int) -> int:
    """Parse an exact nonnegative integer point count without lossy coercion."""
    # Point/row disagreement is an authority failure, never a scheduling fallback.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"feature {feature_id} point n_valid is not an integer")
    parsed = int(value)
    if float(value) != float(parsed) or parsed < 0:
        raise ValueError(f"feature {feature_id} point n_valid is not an integer")
    return parsed


def _json_scalar(value: Any) -> float | int | str | bool | None:
    """Convert one geometry scalar to a finite JSON value."""
    # Missing and non-finite evidence stay unavailable instead of becoming a gate side.
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _counts(
    *,
    requested: int = 0,
    valid: int = 0,
    failed: int = 0,
    non_applicable: int = 0,
    skipped: int = 0,
) -> dict[str, int]:
    """Return the closed deterministic protocol counter vocabulary."""
    # Emit zero-valued keys so clean/resume comparisons never depend on omission.
    return {
        "requested": int(requested),
        "valid": int(valid),
        "failed": int(failed),
        "non_applicable": int(non_applicable),
        "skipped": int(skipped),
    }


def _empty_digest() -> str:
    """Return the canonical empty ordered-plan digest."""
    # Match the schedule/protocol SHA256 convention without importing another helper.
    import hashlib

    return hashlib.sha256(b"").hexdigest()
