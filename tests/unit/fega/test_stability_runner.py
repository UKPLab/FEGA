from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fega.core.geometry_metrics.metrics import (
    c_ray_pairwise_final_resid,
    centered_residual_spectrum_final_resid,
    effective_rank_from_spectrum,
    span_spectrum_final_resid,
)
from fega.core.geometry_reporting.thresholds import get_threshold_profile
from fega.core.stability import runner as stability_runner
from fega.core.stability.protocols import signed_gate_margins


def _metric_config() -> SimpleNamespace:
    """Build only the immutable geometry settings used by retained-evidence tests."""
    # Keep the fixture independent of schedule and artifact construction.
    return SimpleNamespace(
        phases=SimpleNamespace(
            geometry_metrics=SimpleNamespace(
                c_ray=SimpleNamespace(eps=1.0e-12),
                span=SimpleNamespace(eps=1.0e-12),
                resid=SimpleNamespace(eps=1.0e-12),
                effective_rank=SimpleNamespace(eps=1.0e-12),
            )
        )
    )


def _retained_oracle(
    rows: torch.Tensor,
    gram: torch.Tensor,
    *,
    family: str,
    strict_k_values: tuple[int, ...],
    margin_keys: tuple[str, ...],
) -> dict[str, object]:
    """Evaluate retained evidence without schedule, runner aggregation, or writers."""
    # Call geometry and gate authorities directly, then mechanically filter margins.
    thresholds = get_threshold_profile("paper")
    metrics: dict[str, object] = {}
    if family in {"directed_ray", "axis_or_antipodal"}:
        c_ray = c_ray_pairwise_final_resid(rows, gram, eps=1.0e-12)
        metrics["c_ray"] = c_ray.c_ray
    if family in {
        "directed_ray",
        "axis_or_antipodal",
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
    }:
        span_k = strict_k_values or (1,)
        span = span_spectrum_final_resid(rows, gram, k_values=span_k, eps=1.0e-12)
        if strict_k_values:
            rank = effective_rank_from_spectrum(span.eigenvalues, eps=1.0e-12)
            metrics["r_span_pr"] = rank.r_pr
            for key, value in span.s_span.items():
                metrics[f"s_span_{key}"] = value
            for key, value in span.u_span.items():
                metrics[f"u_span_{key}"] = value
            for key, value in span.d_span.items():
                metrics[f"d_span_{key}"] = value
        else:
            metrics["s_span_1"] = span.s_span[1]
            if family == "axis_or_antipodal":
                metrics["b_axis"] = span.b_axis
    if family == "residual_lowD_k":
        residual = centered_residual_spectrum_final_resid(
            rows, gram, k_values=strict_k_values, eps=1.0e-12
        )
        rank = effective_rank_from_spectrum(residual.eigenvalues, eps=1.0e-12)
        metrics.update({"e_res": residual.e_res, "r_ctr_pr": rank.r_pr})
        for key, value in residual.s_res.items():
            metrics[f"s_res_{key}"] = value
    margins = signed_gate_margins(
        metrics, thresholds, k_values=strict_k_values or (1,)
    )
    return {
        "metrics": metrics,
        "margins": {key: margins.get(key) for key in margin_keys},
    }


@pytest.mark.parametrize(
    ("family", "strict_k_values", "margin_keys"),
    [
        ("directed_ray", (), ("c_ray_ge", "s_span_1_axis")),
        ("axis_or_antipodal", (), ("c_ray_lt", "s_span_1_axis", "b_axis")),
        (
            "global_kD_directional_subspace",
            (2, 3),
            (
                "s_span_2",
                "r_span_pr_k2",
                "u_span_2",
                "d_span_2",
                "s_span_3",
                "r_span_pr_k3",
                "u_span_3",
                "d_span_3",
            ),
        ),
        (
            "residual_lowD_k",
            (2, 3),
            ("e_res", "s_res_2", "r_ctr_pr_k2", "s_res_3", "r_ctr_pr_k3"),
        ),
    ],
)
def test_family_subset_matches_independent_retained_evidence_oracle(
    family: str,
    strict_k_values: tuple[int, ...],
    margin_keys: tuple[str, ...],
) -> None:
    """Match direct retained evidence while evaluating no unselected family or k."""
    # Compare production composition with an oracle that knows no schedule or runner state.
    generator = torch.Generator().manual_seed(914)
    rows = torch.randn(36, 8, generator=generator)
    gram = torch.diag(torch.linspace(0.5, 1.5, 8))
    expected = _retained_oracle(
        rows,
        gram,
        family=family,
        strict_k_values=strict_k_values,
        margin_keys=margin_keys,
    )

    actual = stability_runner._family_subset_evidence(
        _metric_config(),
        rows,
        gram,
        family=family,
        selected_k=(strict_k_values[-1] if strict_k_values else None),
        strict_k_values=strict_k_values,
        margin_keys=margin_keys,
        thresholds=get_threshold_profile("paper"),
    )

    assert actual["metrics"] == pytest.approx(expected["metrics"])
    assert actual["margins"] == pytest.approx(expected["margins"])
    assert set(actual["margins"]) == set(margin_keys)


def test_locked_k_mismatch_and_missing_margin_remain_distinct() -> None:
    """Retain finite mismatch as instability and missing required evidence as unavailable."""
    # Exercise the policy independently of metric arithmetic.
    smaller_passes = {
        "s_span_2": 0.1,
        "r_span_pr_k2": 0.1,
        "u_span_2": 0.1,
        "d_span_2": 0.1,
        "s_span_3": 0.1,
        "r_span_pr_k3": 0.1,
        "u_span_3": 0.1,
        "d_span_3": 0.1,
    }
    mismatch = stability_runner._locked_family_result(
        "global_kD_directional_subspace",
        selected_k=3,
        strict_k_values=(2, 3),
        margins=smaller_passes,
    )
    missing = dict(smaller_passes)
    missing["r_span_pr_k2"] = None
    unavailable = stability_runner._locked_family_result(
        "global_kD_directional_subspace",
        selected_k=3,
        strict_k_values=(2, 3),
        margins=missing,
    )

    assert mismatch == {
        "family_supported": True,
        "derived_subset_k": 2,
        "selected_k_mismatch": True,
        "missing_required_margins": [],
    }
    assert unavailable["family_supported"] is True
    assert unavailable["selected_k_mismatch"] is False
    assert unavailable["derived_subset_k"] == 3
    assert unavailable["missing_required_margins"] == ["r_span_pr_k2"]


def test_locked_family_preserves_finite_results_alongside_missingness() -> None:
    """Keep support, minimal-k mismatch, and missing margins as orthogonal facts."""
    # Exercise the architect-approved coexistence cases without metric arithmetic.
    smaller_pass_locked_missing = {
        "s_span_2": 0.1,
        "r_span_pr_k2": 0.1,
        "u_span_2": 0.1,
        "d_span_2": 0.1,
        "s_span_3": None,
        "r_span_pr_k3": 0.1,
        "u_span_3": 0.1,
        "d_span_3": 0.1,
    }
    result = stability_runner._locked_family_result(
        "global_kD_directional_subspace",
        selected_k=3,
        strict_k_values=(2, 3),
        margins=smaller_pass_locked_missing,
    )
    assert result == {
        "family_supported": None,
        "derived_subset_k": 2,
        "selected_k_mismatch": True,
        "missing_required_margins": ["s_span_3"],
    }

    earlier_missing_later_pass = {
        "s_span_2": None,
        "r_span_pr_k2": 0.1,
        "u_span_2": 0.1,
        "d_span_2": 0.1,
        "s_span_3": 0.1,
        "r_span_pr_k3": 0.1,
        "u_span_3": 0.1,
        "d_span_3": 0.1,
        "s_span_4": None,
        "r_span_pr_k4": 0.1,
        "u_span_4": 0.1,
        "d_span_4": 0.1,
    }
    result = stability_runner._locked_family_result(
        "global_kD_directional_subspace",
        selected_k=4,
        strict_k_values=(2, 3, 4),
        margins=earlier_missing_later_pass,
    )
    assert result == {
        "family_supported": None,
        "derived_subset_k": 3,
        "selected_k_mismatch": True,
        "missing_required_margins": ["s_span_2", "s_span_4"],
    }

    earlier_missing_locked_failure = dict(smaller_pass_locked_missing)
    earlier_missing_locked_failure.update(
        {
            "s_span_2": None,
            "s_span_3": -0.1,
        }
    )
    result = stability_runner._locked_family_result(
        "global_kD_directional_subspace",
        selected_k=3,
        strict_k_values=(2, 3),
        margins=earlier_missing_locked_failure,
    )
    assert result == {
        "family_supported": False,
        "derived_subset_k": None,
        "selected_k_mismatch": False,
        "missing_required_margins": ["s_span_2"],
    }


def test_residual_global_gate_and_ray_failure_remain_decisive_when_companions_missing() -> None:
    """Never turn a finite family-gate failure into unknown support."""
    # Global residual and ray gates remain decisive beside independent missing evidence.
    residual = stability_runner._locked_family_result(
        "residual_lowD_k",
        selected_k=2,
        strict_k_values=(2,),
        margins={"e_res": -0.1, "s_res_2": None, "r_ctr_pr_k2": 0.1},
    )
    ray = stability_runner._locked_family_result(
        "directed_ray",
        selected_k=None,
        strict_k_values=(),
        margins={"c_ray_ge": -0.1, "s_span_1_axis": None},
    )

    assert residual == {
        "family_supported": False,
        "derived_subset_k": None,
        "selected_k_mismatch": False,
        "missing_required_margins": ["s_res_2"],
    }
    assert ray == {
        "family_supported": False,
        "derived_subset_k": None,
        "selected_k_mismatch": False,
        "missing_required_margins": ["s_span_1_axis"],
    }


def test_subset_aggregation_counts_crossing_and_mismatch_beside_unavailability() -> None:
    """Count finite instability without discarding simultaneous missing evidence."""
    # One replicate deliberately contains a crossing, k mismatch, and missing companion.
    result = stability_runner.aggregate_subset_protocol(
        point_margins={"c_ray_ge": 0.1, "s_span_1_axis": 0.1},
        block={
            "counters": {
                "requested": 1,
                "valid": 1,
                "failed": 0,
                "non_applicable": 0,
                "skipped": 0,
            },
            "replicates": [
                {
                    "status": "valid",
                    "margins": {"c_ray_ge": -0.1, "s_span_1_axis": None},
                    "selected_k_mismatch": True,
                    "missing_required_margins": ["s_span_1_axis"],
                }
            ],
        },
    )

    assert result["status"] == "unstable"
    assert result["instability_count"] == 1
    assert result["selected_k_mismatch_count"] == 1
    assert result["required_margin_unavailable_count"] == 1
    assert result["gate_crossing_counts"]["c_ray_ge"] == 1
    assert result["required_failure_count"] == 1


def test_mixture_reuses_one_authoritative_assignment_without_work() -> None:
    """Copy accepted standalone assignment evidence exactly once without resampling."""
    # The reuse path must not derive, refit, or aggregate assignment evidence.
    assignment = {
        "status": "available",
        "value": 0.93,
        "requested_count": 8,
        "valid_count": 8,
        "failed_count": 0,
        "replicates": [{"replicate_id": 0, "adjusted_rand_score": 0.93}],
    }
    record = {"assignment_stability": assignment}

    reused = stability_runner._reuse_standalone_assignment(record)

    assert reused["assignment_stability"] == assignment
    assert reused["assignment_stability"] is not assignment
    assert reused["counters"] == {
        "requested": 1,
        "valid": 1,
        "failed": 0,
        "non_applicable": 0,
        "skipped": 0,
    }


def test_missing_point_artifact_builds_once_but_corrupt_present_artifact_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build a missing point artifact once while never repairing a present invalid one."""
    # Isolate point ownership from all upstream artifact details.
    point_path = tmp_path / "geometry_point_records.json"
    calls = {"build": 0, "load": 0}

    def build(*_args, **_kwargs):
        """Materialize the missing reporting-owned point artifact once."""
        # Record ownership and create only the path needed by the loader fixture.
        calls["build"] += 1
        point_path.write_text("{}")

    def load(*_args, **_kwargs):
        """Return a valid bundle or surface a present corrupt artifact."""
        # Never convert corrupt present content into another build request.
        calls["load"] += 1
        if point_path.read_text() == "corrupt":
            raise ValueError("point artifact corrupt")
        return {"point_records": []}

    monkeypatch.setattr(stability_runner, "point_geometry_records_path", lambda *_: point_path)
    monkeypatch.setattr(
        stability_runner, "load_build_and_write_point_geometry_records", build
    )
    monkeypatch.setattr(stability_runner, "load_point_geometry_records", load)

    assert stability_runner._load_or_build_point_bundle(object(), None) == {
        "point_records": []
    }
    assert calls == {"build": 1, "load": 1}

    point_path.write_text("corrupt")
    with pytest.raises(ValueError, match="point artifact corrupt"):
        stability_runner._load_or_build_point_bundle(object(), None)
    assert calls == {"build": 1, "load": 2}


def test_stability_runner_has_zero_vmf_model_factor_or_materializer_reachability() -> None:
    """Keep every removed universal-path dependency unreachable from stability."""
    # Static source reachability makes accidental imports and hidden fallbacks visible.
    source = inspect.getsource(stability_runner)
    forbidden = (
        "score_vmf_feature",
        "FeatureFactor",
        "ModelResources(",
        "_BoundedLinearMaterializer",
        "_validated_canonical_unembedding",
        "validate_gpu_execution_workers",
        "classify_record",
    )

    assert all(name not in source for name in forbidden)


def test_incomplete_schedule_order_is_canonical_and_never_reuses_stale_records() -> None:
    """Execute only exact current incomplete schedules in canonical feature order."""
    # A record with the wrong schedule digest is stale even when its feature ID matches.
    schedules = [
        SimpleNamespace(feature_id=3, schedule_digest="three"),
        SimpleNamespace(feature_id=1, schedule_digest="one"),
        SimpleNamespace(feature_id=2, schedule_digest="two"),
    ]
    records = [
        {"feature_id": 1, "schedule_digest": "one"},
        {"feature_id": 2, "schedule_digest": "stale"},
    ]

    completed, incomplete = stability_runner._partition_current_records(
        schedules, records
    )

    assert [record["feature_id"] for record in completed] == [1]
    assert [schedule.feature_id for schedule in incomplete] == [2, 3]


def test_sequential_and_parallel_execution_have_identical_scientific_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep worker count out of feature order, evidence, and checkpoint records."""
    # Replace numerical work only; exercise the production ordered executor and flush path.
    work = [SimpleNamespace(schedule=SimpleNamespace(feature_id=value)) for value in (1, 2, 3)]
    schedules = [
        SimpleNamespace(feature_id=value, schedule_digest=f"schedule-{value}")
        for value in (1, 2, 3)
    ]
    checkpoint = tmp_path / "checkpoint.json"
    writes: list[list[dict[str, object]]] = []

    def execute(item, *_args):
        """Return one deterministic feature result independent of worker scheduling."""
        # Make feature identity the entire scientific result for this executor test.
        feature_id = item.schedule.feature_id
        return stability_runner._ExecutionResult(
            {
                "feature_id": feature_id,
                "schedule_digest": f"schedule-{feature_id}",
                "family": "directed_ray",
            },
            {"directed_ray:bootstrap": float(feature_id)},
        )

    def write(_config, _fingerprint, records, _schedules):
        """Capture each ordered atomic-checkpoint request."""
        # Touch the fixture path so the final-flush branch observes an existing file.
        writes.append([dict(record) for record in records])
        checkpoint.write_text("checkpoint")

    monkeypatch.setattr(stability_runner, "_execute_one", execute)
    monkeypatch.setattr(stability_runner, "_write_current_checkpoint", write)
    monkeypatch.setattr(stability_runner, "stability_checkpoint_path", lambda *_: checkpoint)

    def config(workers: int):
        """Build the minimal worker/checkpoint configuration surface."""
        # Keep every scientific control identical while changing only worker count.
        return SimpleNamespace(
            phases=SimpleNamespace(
                stability=SimpleNamespace(
                    workers=workers, checkpoint_flush_features=1
                ),
                geometry_reporting=SimpleNamespace(threshold_profile="paper"),
            )
        )

    sequential, _ = stability_runner._execute_scheduled_features(
        work,
        config=config(1),
        fingerprint={},
        schedules=schedules,
        initial_records=[],
    )
    checkpoint.unlink()
    parallel, _ = stability_runner._execute_scheduled_features(
        work,
        config=config(3),
        fingerprint={},
        schedules=schedules,
        initial_records=[],
    )

    assert parallel == sequential
    assert [record["feature_id"] for record in parallel] == [1, 2, 3]


def test_parallel_execution_does_not_eagerly_prepare_the_complete_inventory() -> None:
    """Bound parallel feature preparation to the configured worker window."""
    # A first result must be observable before the source generator advances past workers.
    consumed: list[int] = []

    def work():
        """Expose every source-generator advance to the bounded-execution assertion."""
        # Ten items make eager Executor.map-style consumption unambiguously visible.
        for feature_id in range(10):
            consumed.append(feature_id)
            yield SimpleNamespace(schedule=SimpleNamespace(feature_id=feature_id))

    def execute(item):
        """Return one immediate result without influencing source consumption."""
        # Keep numerical work absent so the test isolates executor buffering behavior.
        return stability_runner._ExecutionResult(
            {"feature_id": item.schedule.feature_id}, {}
        )

    results = stability_runner._iter_ordered_execution_results(
        work(), execute, workers=2
    )
    first = next(results)

    assert first.record["feature_id"] == 0
    assert consumed == [0, 1]
    results.close()


def test_interrupted_resume_equals_clean_and_preserves_completed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume only incomplete current schedules and reproduce the clean record sequence."""
    # Fail after two flushed records, then feed that exact prefix through partitioning.
    schedules = [
        SimpleNamespace(feature_id=value, schedule_digest=f"schedule-{value}")
        for value in (1, 2, 3, 4)
    ]
    work = [SimpleNamespace(schedule=schedule) for schedule in schedules]
    checkpoint = tmp_path / "checkpoint.json"
    last_records: list[dict[str, object]] = []
    fail_once = {3}

    def execute(item, *_args):
        """Interrupt feature three once and deterministically score every other call."""
        # The second invocation simulates resuming after the external interruption clears.
        feature_id = item.schedule.feature_id
        if feature_id in fail_once:
            fail_once.remove(feature_id)
            raise FloatingPointError("forced interruption")
        return stability_runner._ExecutionResult(
            {
                "feature_id": feature_id,
                "schedule_digest": f"schedule-{feature_id}",
                "family": "directed_ray",
            },
            {},
        )

    def write(_config, _fingerprint, records, _schedules):
        """Persist the latest completed ordered record sequence in memory."""
        # Mirror atomic replacement by replacing the whole captured list.
        last_records[:] = [dict(record) for record in records]
        checkpoint.write_text("checkpoint")

    config = SimpleNamespace(
        phases=SimpleNamespace(
            stability=SimpleNamespace(workers=1, checkpoint_flush_features=1),
            geometry_reporting=SimpleNamespace(threshold_profile="paper"),
        )
    )
    monkeypatch.setattr(stability_runner, "_execute_one", execute)
    monkeypatch.setattr(stability_runner, "_write_current_checkpoint", write)
    monkeypatch.setattr(stability_runner, "stability_checkpoint_path", lambda *_: checkpoint)

    with pytest.raises(FloatingPointError, match="forced interruption"):
        stability_runner._execute_scheduled_features(
            work,
            config=config,
            fingerprint={},
            schedules=schedules,
            initial_records=[],
        )
    assert [record["feature_id"] for record in last_records] == [1, 2]

    completed, incomplete = stability_runner._partition_current_records(
        schedules, last_records
    )
    resumed_work = [
        SimpleNamespace(schedule=schedule) for schedule in incomplete
    ]
    resumed, _ = stability_runner._execute_scheduled_features(
        resumed_work,
        config=config,
        fingerprint={},
        schedules=schedules,
        initial_records=completed,
    )
    checkpoint.unlink()
    clean, _ = stability_runner._execute_scheduled_features(
        work,
        config=config,
        fingerprint={},
        schedules=schedules,
        initial_records=[],
    )

    assert resumed == clean
