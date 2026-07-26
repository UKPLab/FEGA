from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from fega.config_schema import FEGAPipelineConfig, StabilityConfig
from fega.core.geometry_reporting import artifacts as reporting_artifacts
from fega.core.geometry_reporting import records as reporting_records
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    MixtureAuditState,
    MixtureAuditStatus,
    PointSelection,
    SelectionMode,
    point_selection_identity,
)
from fega.core.source_fingerprint import (
    canonical_json_digest,
    canonical_source_fingerprint,
)
from fega.core.stability import artifacts as stability_artifacts
from fega.core.stability.artifacts import (
    build_selected_family_checkpoint_fingerprint,
    build_selected_family_checkpoint_payload,
    load_selected_family_checkpoint,
    selected_family_required_protocol_ids,
)
from fega.core.stability.protocols import (
    bootstrap_plans,
    leave_out_plans,
    sample_size_plans,
)
from fega.core.stability.sampling import derive_seed, subspace_resample_indices
from fega.core.stability.schedule import (
    DELIBERATE_NON_EVALUATION_REASON,
    build_selected_family_schedule,
)


def _selection(
    family: str, selected_k: int | None, *, mode: SelectionMode = "strict"
) -> PointSelection:
    """Return one immutable locked point selection for schedule fixtures."""
    # Mixture uses accepted assignment evidence; other families are not applicable.
    mixture_status: MixtureAuditStatus = (
        "accepted"
        if family == "multi_mode_directional_geometry"
        else "not_applicable"
    )
    return PointSelection(
        family=family,
        selected_k=selected_k,
        mode=mode,
        point_reason=f"fixture:{family}",
        mixture_audit_state=MixtureAuditState(
            status=mixture_status,
            acceptance=("accepted" if mixture_status == "accepted" else None),
            reason="fixture",
            failed_gates=(),
            unavailable_gates=(),
        ),
    )


def _schedule(
    family: str = "directed_ray",
    selected_k: int | None = None,
    *,
    mode: SelectionMode = "strict",
    feature_id: int = 7,
    config: StabilityConfig | None = None,
    n_rows: int = 32,
):
    """Build one deterministic schedule with nontrivial subset plans."""
    # Bind the schedule to the same complete durable point record used by fixtures.
    selection = _selection(family, selected_k, mode=mode)
    point_record = {
        "feature_id": int(feature_id),
        "point_selection": point_selection_identity(selection),
    }
    return build_selected_family_schedule(
        selection=selection,
        feature_id=feature_id,
        point_record_sha256=canonical_json_digest(point_record),
        base_seed=42,
        effect_space="final_resid",
        n_rows=n_rows,
        group_labels=["a"] * (n_rows // 2) + ["b"] * (n_rows - n_rows // 2),
        stability_config=config or StabilityConfig(),
    )


def _raw_selected_family_evidence(schedule) -> dict[str, object]:
    """Return structurally complete raw checkpoint evidence for one schedule mode."""
    # Retain plan identities and denominator counts without duplicating result semantics.
    if schedule.no_work_reason is not None:
        counters = _protocol_counts(non_applicable=1)
        protocol = {
            "status": "not_evaluated",
            "plan_digest": hashlib.sha256(b"").hexdigest(),
            "counters": counters,
        }
        return {
            "required_protocol_ids": selected_family_required_protocol_ids(schedule),
            "no_work_reason": schedule.no_work_reason,
            "protocols": {"deliberate_non_evaluation": protocol},
            "protocol_counters": {"deliberate_non_evaluation": counters},
        }
    if schedule.reuse_standalone_assignment:
        counters = _protocol_counts(requested=1, valid=1)
        protocol = {
            "status": "reused",
            "plan_digest": hashlib.sha256(b"").hexdigest(),
            "counters": counters,
        }
        return {
            "required_protocol_ids": selected_family_required_protocol_ids(schedule),
            "no_work_reason": None,
            "protocols": {"standalone_assignment_reuse": protocol},
            "protocol_counters": {"standalone_assignment_reuse": counters},
        }
    angle_plans = (
        schedule.raw_angle_plans
        if schedule.angle_source == "raw"
        else schedule.residual_angle_plans
    )
    low_counters = _protocol_counts(requested=1, valid=1)
    protocols = {
        "low_context_qualification": {"status": "ok", "counters": low_counters},
        "bootstrap": _planned_protocol(
            schedule.bootstrap, empty_non_applicable=not schedule.scalar_metrics
        ),
        "angle": _planned_protocol(angle_plans, empty_non_applicable=True),
        "leave_out": _planned_protocol(
            schedule.leave_out, empty_non_applicable=False
        ),
        "sample_size": _planned_protocol(
            schedule.sample_size, empty_non_applicable=False
        ),
    }
    return {
        "required_protocol_ids": selected_family_required_protocol_ids(schedule),
        "no_work_reason": None,
        "protocols": protocols,
        "protocol_counters": {
            key: dict(protocol["counters"]) for key, protocol in protocols.items()
        },
    }


def _protocol_counts(
    *, requested: int = 0, valid: int = 0, non_applicable: int = 0
) -> dict[str, int]:
    """Return the complete denominator vocabulary emitted by the runner."""
    # Fixtures keep explicit zeros because omitted denominator states are ambiguous.
    return {
        "requested": requested,
        "valid": valid,
        "failed": 0,
        "non_applicable": non_applicable,
        "skipped": 0,
    }


def _planned_protocol(
    plans, *, empty_non_applicable: bool
) -> dict[str, object]:
    """Return a minimal protocol block bound to ordered immutable plan identities."""
    # Scientific values are irrelevant to checkpoint structural validation.
    counters = (
        _protocol_counts(requested=len(plans), valid=len(plans))
        if plans
        else _protocol_counts(non_applicable=int(empty_non_applicable))
    )
    joined = "".join(plan.digest for plan in plans)
    return {
        "status": "fixture",
        "plan_digest": hashlib.sha256(joined.encode("ascii")).hexdigest(),
        "replicates": [{"plan_digest": plan.digest} for plan in plans],
        "counters": counters,
    }


def _checkpoint_record(schedule) -> dict[str, object]:
    """Return the exact structural record emitted around raw WP3 evidence."""
    # Locked point and schedule identity remain explicit across checkpoint resume.
    return {
        "feature_id": schedule.feature_id,
        "family": schedule.family,
        "selection_mode": schedule.selection_mode,
        "selected_k": schedule.reported_selected_k,
        "point_reason": schedule.point_selection.point_reason,
        "schedule_digest": schedule.schedule_digest,
        "point_record_sha256": schedule.point_record_sha256,
        "n_valid": 32,
        "selected_family_evidence": _raw_selected_family_evidence(schedule),
    }


@pytest.mark.parametrize(
    ("family", "selected_k", "strict_ks", "angle_source", "angle_k"),
    [
        ("directed_ray", None, (), "none", None),
        ("axis_or_antipodal", None, (), "raw", 1),
        ("global_2D_directional_subspace", 2, (2,), "raw", 2),
        ("global_kD_directional_subspace", 3, (2, 3), "raw", 3),
        ("global_kD_directional_subspace", 4, (2, 3, 4), "raw", 4),
        ("global_kD_directional_subspace", 8, (2, 3, 4, 8), "raw", 8),
        ("residual_lowD_k", 2, (2,), "centered_residual", 2),
        ("residual_lowD_k", 3, (2, 3), "centered_residual", 3),
        ("residual_lowD_k", 4, (2, 3, 4), "centered_residual", 4),
    ],
)
def test_selected_family_golden_schedule_breadth(
    family: str,
    selected_k: int | None,
    strict_ks: tuple[int, ...],
    angle_source: str,
    angle_k: int | None,
) -> None:
    """Freeze every strict family, span/residual k, and axis dimension split."""
    # Inspect only construction breadth; execution remains a later work package.
    schedule = _schedule(family, selected_k)
    assert schedule.request_count == 1
    assert schedule.family == family
    assert schedule.reported_selected_k == selected_k
    assert schedule.evaluated_strict_k_values == strict_ks
    assert schedule.angle_source == angle_source
    assert schedule.angle_k == angle_k
    assert schedule.point_selection.point_reason == f"fixture:{family}"
    assert schedule.evidence_request.full_sample_point_requirements
    if angle_source == "raw":
        assert schedule.raw_angle_plans
        assert schedule.residual_angle_plans == ()
    elif angle_source == "centered_residual":
        assert schedule.residual_angle_plans
        assert schedule.raw_angle_plans == ()
    else:
        assert schedule.raw_angle_plans == schedule.residual_angle_plans == ()


@pytest.mark.parametrize("n_rows", [7, 8, 15, 16, 31])
@pytest.mark.parametrize(
    ("family", "selected_k", "angle_k"),
    [
        ("axis_or_antipodal", None, 1),
        ("global_2D_directional_subspace", 2, 2),
        ("residual_lowD_k", 3, 3),
    ],
)
def test_angle_schedule_preserves_dimension_but_skips_rows_below_32(
    n_rows: int, family: str, selected_k: int | None, angle_k: int
) -> None:
    """Match existing principal-angle early returns across every low-row band."""
    # Construction retains the selected protocol dimension but emits no angle draws.
    schedule = _schedule(family, selected_k, n_rows=n_rows)
    assert schedule.angle_k == angle_k
    assert schedule.raw_angle_plans == ()
    assert schedule.residual_angle_plans == ()


@pytest.mark.parametrize(
    ("family", "selected_k", "message"),
    [
        ("global_2D_directional_subspace", 3, "must use selected_k=2"),
        ("global_2D_directional_subspace", 4, "must use selected_k=2"),
        ("global_kD_directional_subspace", 2, "must use selected_k in 3, 4, 8"),
        ("global_kD_directional_subspace", 1, "must use selected_k in 3, 4, 8"),
        ("global_kD_directional_subspace", 9, "must use selected_k in 3, 4, 8"),
        ("residual_lowD_k", 1, "must use selected_k in 2, 3, 4"),
        ("residual_lowD_k", 8, "must use selected_k in 2, 3, 4"),
    ],
)
def test_strict_family_dimension_contract_rejects_mislabeled_pairs(
    family: str, selected_k: int, message: str
) -> None:
    """Reject family labels whose selected dimension contradicts point semantics."""
    # A malformed durable point selection must fail before any subset plan is made.
    with pytest.raises(ValueError, match=message):
        _schedule(family, selected_k)


def test_selected_family_plan_identity_matches_existing_generators() -> None:
    """Preserve existing seeds, plan order, bootstrap multiplicity, and angle RNG."""
    # Compare complete immutable values against each established generator directly.
    config = StabilityConfig()
    schedule = _schedule("axis_or_antipodal", None, config=config)
    feature_seed = derive_seed(42, feature_id=7, effect_space="final_resid")
    assert schedule.bootstrap == tuple(
        bootstrap_plans(
            seed=feature_seed,
            feature_id=7,
            n_rows=32,
            rounds=config.scalar.bootstrap_rounds,
        )
    )
    assert schedule.leave_out == tuple(
        leave_out_plans(
            seed=feature_seed + 17,
            feature_id=7,
            n_rows=32,
            group_labels=["a"] * 16 + ["b"] * 16,
        )
    )
    assert schedule.sample_size == tuple(
        sample_size_plans(
            seed=feature_seed + 31,
            feature_id=7,
            n_rows=32,
            targets=config.sample_size.target_sizes,
            rounds=config.sample_size.strong_subset_rounds,
        )
    )
    expected_angle = subspace_resample_indices(
        32,
        config.subspace.resample_fraction,
        config.subspace.resample_rounds,
        feature_seed + 47,
    )
    assert [plan.indices for plan in schedule.raw_angle_plans] == [
        tuple(int(value) for value in indices.tolist())
        for indices in expected_angle
    ]
    assert {plan.seed for plan in schedule.raw_angle_plans} == {feature_seed + 47}
    assert [plan.replicate_id for plan in schedule.raw_angle_plans] == list(
        range(config.subspace.resample_rounds)
    )
    assert {plan.feature_id for plan in schedule.raw_angle_plans} == {7}
    assert {plan.angle_k for plan in schedule.raw_angle_plans} == {1}
    first_angle = schedule.raw_angle_plans[0]
    assert first_angle.digest == canonical_json_digest(
        {
            "feature_id": 7,
            "protocol": "principal_angle",
            "source": "raw",
            "angle_k": 1,
            "seed": feature_seed + 47,
            "replicate_id": 0,
            "indices": list(first_angle.indices),
        }
    )


def test_mixture_reuse_and_fallback_non_evaluation_are_distinct() -> None:
    """Keep standalone assignment reuse separate from deliberate no-work traces."""
    # Both omit stability plans, but only strict mixture carries assignment reuse.
    mixture = _schedule("multi_mode_directional_geometry", None)
    fallback = _schedule("residual_lowD_k", 1, mode="fallback")
    for schedule in (mixture, fallback):
        assert schedule.bootstrap == ()
        assert schedule.leave_out == ()
        assert schedule.sample_size == ()
        assert schedule.raw_angle_plans == ()
        assert schedule.residual_angle_plans == ()
    assert mixture.request_count == 1
    assert mixture.reuse_standalone_assignment is True
    assert mixture.no_work_reason is None
    assert fallback.request_count == 0
    assert fallback.reuse_standalone_assignment is False
    assert fallback.no_work_reason == DELIBERATE_NON_EVALUATION_REASON


@pytest.mark.parametrize(
    ("mode", "audit"),
    [
        (
            "strict",
            MixtureAuditState("rejected", "rejected", "failed", ("delta_mix",), ()),
        ),
        (
            "strict",
            MixtureAuditState(
                "unavailable", None, "missing", (), ("assignment_stability",)
            ),
        ),
        (
            "strict",
            MixtureAuditState("not_applicable", None, "not_evaluated", (), ()),
        ),
        (
            "fallback",
            MixtureAuditState("rejected", "rejected", "failed", ("delta_mix",), ()),
        ),
    ],
)
def test_mixture_schedule_requires_complete_accepted_audit(
    mode: SelectionMode, audit: MixtureAuditState
) -> None:
    """Reject mixture schedules unless reporting supplied complete accepted evidence."""
    # Strict and fallback mixture labels both depend on the same accepted point audit.
    selection = PointSelection(
        family="multi_mode_directional_geometry",
        selected_k=None,
        mode=mode,
        point_reason="invalid-mixture-fixture",
        mixture_audit_state=audit,
    )
    with pytest.raises(ValueError, match="accepted mixture audit"):
        build_selected_family_schedule(
            selection=selection,
            feature_id=7,
            point_record_sha256="record",
            base_seed=42,
            effect_space="final_resid",
            n_rows=32,
            group_labels=["a"] * 16 + ["b"] * 16,
            stability_config=StabilityConfig(),
        )


def test_schedule_repeat_and_worker_identity_are_invariant() -> None:
    """Keep the schedule independent of repetition and worker-count controls."""
    # Worker count is never read by construction; repeated complete identities match.
    first_config = StabilityConfig(workers=1)
    second_config = StabilityConfig(workers=8)
    first = _schedule("global_kD_directional_subspace", 4, config=first_config)
    repeated = _schedule("global_kD_directional_subspace", 4, config=first_config)
    worker_changed = _schedule(
        "global_kD_directional_subspace", 4, config=second_config
    )
    assert first == repeated
    assert first.schedule_digest == worker_changed.schedule_digest


def _pipeline_config(*, workers: int = 1) -> FEGAPipelineConfig:
    """Return one complete pipeline config for authoritative fingerprint tests."""
    # Only the scheduling worker count varies; all scientific controls stay fixed.
    config = FEGAPipelineConfig(
        reference_json=Path("reference.json"),
        output_root=Path("results"),
        device="cpu",
        entity_attribute_selection={"city": ["Country"]},
    )
    config.phases.stability.workers = workers
    config.phases.stability.resume = True
    return config


def _validated_point_bundle(schedule) -> dict:
    """Return the complete non-placeholder output shape of the point loader."""
    # Every required source, artifact, schedule, and record identity is populated.
    feature_id = int(schedule.feature_id)
    unhashed_point_record = {
        "feature_id": feature_id,
        "point_selection": point_selection_identity(schedule.point_selection),
    }
    point_record_hash = canonical_json_digest(unhashed_point_record)
    if point_record_hash != schedule.point_record_sha256:
        raise ValueError("fixture schedule does not match its durable point record")
    point_record = {
        **unhashed_point_record,
        "point_record_sha256": point_record_hash,
    }
    point_inventory_digest = canonical_json_digest(
        [
            {
                "feature_id": feature_id,
                "point_record_sha256": point_record_hash,
            }
        ]
    )
    scientific_fingerprint = {
        "schema_version": 3,
        "effect_space": "pre_softcap_logits",
        "source_readout": "final_resid",
        "vmf_backend_fingerprints": {"dense_cpu": {"sha256": "backend"}},
        "candidate_mode_counts": [1, 2, 3, 4],
        "assignment_fraction": 0.8,
        "assignment_rounds": 8,
        "seed_derivations": {"version": 1, "feature": "feature-rule"},
        "assignment_metric": {
            "identity": "sklearn.metrics.adjusted_rand_score",
            "distribution": "scikit-learn",
            "version": "test",
        },
        "feature_ids": [feature_id],
        "feature_inventory_sha256": "vmf-inventory",
    }
    return {
        "feature_ids": [feature_id],
        "source_identity": {
            "canonical_source": {
                "schema_version": 2,
                "algorithm": "sha256",
                "digest": "source",
                "components": {"manifest_sha256": "manifest"},
            },
            "manifest": {
                "path": "effect_tensors_manifest.json",
                "canonical_digest": "manifest-canonical",
                "file_sha256": "manifest-file",
            },
            "summary": {
                "path": "effect_summary.json",
                "canonical_digest": "summary-canonical",
                "file_sha256": "summary-file",
            },
            "tensor_shards": [
                {"path": "effect_tensors_00000.pt", "file_sha256": "shard-file"}
            ],
            "gram": {
                "path": "gram.pt",
                "file_sha256": "gram-file",
                "metadata": {"gram_sha256": "gram-metadata"},
            },
        },
        "input_artifact_hashes": {
            "geometry_metrics": {
                "canonical_digest": "geometry-canonical",
                "file_sha256": "geometry-file",
            },
            "standalone_vmf": {
                "canonical_digest": "vmf-canonical",
                "file_sha256": "vmf-file",
            },
        },
        "standalone_vmf_identity": {
            "public_schema_version": 1,
            "scientific_fingerprint": scientific_fingerprint,
            "artifact": {
                "canonical_digest": "vmf-canonical",
                "file_sha256": "vmf-file",
            },
            "feature_ids": [feature_id],
            "candidate_schedule": [1, 2, 3, 4],
            "assignment_schedule": {
                "fraction": 0.8,
                "rounds": 8,
                "seed_derivations": {"version": 1, "feature": "feature-rule"},
                "metric": scientific_fingerprint["assignment_metric"],
            },
        },
        "point_artifact_identity": {
            "canonical_digest": "point-canonical",
            "file_sha256": "point-file",
        },
        "point_record_hashes": {str(feature_id): schedule.point_record_sha256},
        "point_records_sha256": point_inventory_digest,
        "point_records": [point_record],
        "point_selections": {feature_id: schedule.point_selection},
    }


def _fingerprint(schedule, *, workers: int = 1, point_bundle: dict | None = None):
    """Build one fingerprint only through the production-authoritative builder."""
    # The builder derives config, label, kernel, versions, and threads itself.
    return build_selected_family_checkpoint_fingerprint(
        config=_pipeline_config(workers=workers),
        point_bundle=(
            _validated_point_bundle(schedule) if point_bundle is None else point_bundle
        ),
        schedules=[schedule],
    )


def test_checkpoint_fingerprint_excludes_worker_count_and_obsolete_vmf_state() -> None:
    """Bind selected-family science without old stability-side vMF admission state."""
    # Compare complete fingerprints while asserting forbidden identities are absent.
    schedule = _schedule()
    first = _fingerprint(schedule, workers=1)
    second = _fingerprint(schedule, workers=8)
    assert first == second
    assert "workers" not in first["retained_stability_config"]
    assert "vmf_refit_config_hash" not in first
    assert "stability_factor_live_admission" not in first
    assert first["checkpoint_schema_version"] == (
        stability_artifacts.SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION
    )
    assert first["stability_public_schema_version"] == (
        stability_artifacts.STABILITY_PUBLIC_SCHEMA_VERSION
    )
    assert first["locked_features"][0]["selection_mode"] == "strict"
    assert first["geometry_metrics_identity"]["file_sha256"] == "geometry-file"


def test_checkpoint_rejects_family_substitution_with_reused_point_hash() -> None:
    """Reject an internally valid schedule that substitutes the authoritative family."""
    # Reusing the ray record hash must not authorize an axis schedule for that feature.
    ray = _schedule()
    bundle = _validated_point_bundle(ray)
    axis = build_selected_family_schedule(
        selection=_selection("axis_or_antipodal", None),
        feature_id=ray.feature_id,
        point_record_sha256=ray.point_record_sha256,
        base_seed=42,
        effect_space="final_resid",
        n_rows=32,
        group_labels=["a"] * 16 + ["b"] * 16,
        stability_config=StabilityConfig(),
    )

    with pytest.raises(ValueError, match="schedule point selection mismatch"):
        _fingerprint(axis, point_bundle=bundle)


@pytest.mark.parametrize(
    "selection_change",
    [
        pytest.param(
            lambda selection: replace(
                selection, point_reason="substituted-point-reason"
            ),
            id="point-reason",
        ),
        pytest.param(
            lambda selection: replace(
                selection,
                mixture_audit_state=replace(
                    selection.mixture_audit_state,
                    reason="substituted-mixture-audit-reason",
                ),
            ),
            id="mixture-audit",
        ),
    ],
)
def test_checkpoint_rejects_complete_point_selection_substitution(
    selection_change: Callable[[PointSelection], PointSelection],
) -> None:
    """Bind point reason and complete mixture audit, not only family, mode, and k."""
    # Flat schedule fields remain valid so only full PointSelection comparison rejects.
    schedule = _schedule()
    bundle = _validated_point_bundle(schedule)
    substituted = replace(
        schedule,
        point_selection=selection_change(schedule.point_selection),
    )

    with pytest.raises(ValueError, match="schedule point selection mismatch"):
        _fingerprint(substituted, point_bundle=bundle)


def test_checkpoint_fingerprint_drifts_for_authoritative_bundle_components() -> None:
    """Bind every required point-loader provenance component without placeholders."""
    # Mutate each independent component and require a different scientific digest.
    schedule = _schedule()
    baseline_bundle = _validated_point_bundle(schedule)
    baseline = _fingerprint(schedule, point_bundle=baseline_bundle)["digest"]
    mutations = {
        "canonical_source": lambda value: value["source_identity"][
            "canonical_source"
        ].__setitem__("digest", "changed"),
        "manifest_canonical": lambda value: value["source_identity"][
            "manifest"
        ].__setitem__("canonical_digest", "changed"),
        "manifest_file": lambda value: value["source_identity"]["manifest"].__setitem__(
            "file_sha256", "changed"
        ),
        "summary_canonical": lambda value: value["source_identity"][
            "summary"
        ].__setitem__("canonical_digest", "changed"),
        "summary_file": lambda value: value["source_identity"]["summary"].__setitem__(
            "file_sha256", "changed"
        ),
        "tensor_shards": lambda value: value["source_identity"]["tensor_shards"][
            0
        ].__setitem__("file_sha256", "changed"),
        "gram_file": lambda value: value["source_identity"]["gram"].__setitem__(
            "file_sha256", "changed"
        ),
        "gram_metadata": lambda value: value["source_identity"]["gram"][
            "metadata"
        ].__setitem__("gram_sha256", "changed"),
        "geometry_canonical": lambda value: value["input_artifact_hashes"][
            "geometry_metrics"
        ].__setitem__("canonical_digest", "changed"),
        "geometry_file": lambda value: value["input_artifact_hashes"][
            "geometry_metrics"
        ].__setitem__("file_sha256", "changed"),
        "point_artifact_canonical": lambda value: value[
            "point_artifact_identity"
        ].__setitem__("canonical_digest", "changed"),
        "point_artifact_file": lambda value: value[
            "point_artifact_identity"
        ].__setitem__("file_sha256", "changed"),
        "vmf_schema": lambda value: value["standalone_vmf_identity"].__setitem__(
            "public_schema_version", 99
        ),
        "vmf_fingerprint": lambda value: value["standalone_vmf_identity"][
            "scientific_fingerprint"
        ].__setitem__("feature_inventory_sha256", "changed"),
        "vmf_artifact_canonical": lambda value: (
            value["standalone_vmf_identity"]["artifact"].__setitem__(
                "canonical_digest", "changed"
            ),
            value["input_artifact_hashes"]["standalone_vmf"].__setitem__(
                "canonical_digest", "changed"
            ),
        ),
        "vmf_artifact_file": lambda value: (
            value["standalone_vmf_identity"]["artifact"].__setitem__(
                "file_sha256", "changed"
            ),
            value["input_artifact_hashes"]["standalone_vmf"].__setitem__(
                "file_sha256", "changed"
            ),
        ),
        "vmf_candidates": lambda value: (
            value["standalone_vmf_identity"].__setitem__(
                "candidate_schedule", [1, 3]
            ),
            value["standalone_vmf_identity"]["scientific_fingerprint"].__setitem__(
                "candidate_mode_counts", [1, 3]
            ),
        ),
        "vmf_assignment": lambda value: (
            value["standalone_vmf_identity"]["assignment_schedule"].__setitem__(
                "rounds", 9
            ),
            value["standalone_vmf_identity"]["scientific_fingerprint"].__setitem__(
                "assignment_rounds", 9
            ),
        ),
    }
    for name, mutate in mutations.items():
        changed = copy.deepcopy(baseline_bundle)
        mutate(changed)
        assert _fingerprint(schedule, point_bundle=changed)["digest"] != baseline, name

    changed_record = {
        "feature_id": schedule.feature_id,
        "point_selection": point_selection_identity(schedule.point_selection),
        "point_reason_source": "changed-authoritative-record",
    }
    changed_record_hash = canonical_json_digest(changed_record)
    changed_schedule = build_selected_family_schedule(
        selection=schedule.point_selection,
        feature_id=schedule.feature_id,
        point_record_sha256=changed_record_hash,
        base_seed=42,
        effect_space="final_resid",
        n_rows=32,
        group_labels=["a"] * 16 + ["b"] * 16,
        stability_config=StabilityConfig(),
    )
    changed_bundle = _validated_point_bundle(schedule)
    changed_bundle["point_records"] = [
        {**changed_record, "point_record_sha256": changed_record_hash}
    ]
    changed_bundle["point_record_hashes"] = {
        str(schedule.feature_id): changed_record_hash
    }
    changed_bundle["point_records_sha256"] = canonical_json_digest(
        [
            {
                "feature_id": schedule.feature_id,
                "point_record_sha256": changed_record_hash,
            }
        ]
    )
    assert (
        _fingerprint(changed_schedule, point_bundle=changed_bundle)["digest"]
        != baseline
    )


def test_checkpoint_fingerprint_drifts_for_scientific_config_and_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind threshold, retained stability configuration, and label semantics."""
    # Vary only components that can change selected-family evidence or interpretation.
    schedule = _schedule()
    bundle = _validated_point_bundle(schedule)
    label_version = stability_artifacts.LABEL_VERSION
    baseline = _fingerprint(schedule, point_bundle=bundle)["digest"]

    config = _pipeline_config()
    config.phases.geometry_reporting.threshold_profile = "changed-profile"
    assert build_selected_family_checkpoint_fingerprint(
        config=config, point_bundle=bundle, schedules=[schedule]
    )["digest"] != baseline
    config = _pipeline_config()
    config.phases.stability.scalar.bootstrap_rounds += 1
    assert build_selected_family_checkpoint_fingerprint(
        config=config, point_bundle=bundle, schedules=[schedule]
    )["digest"] != baseline

    monkeypatch.setattr(stability_artifacts, "LABEL_VERSION", "changed-label")
    assert _fingerprint(schedule, point_bundle=bundle)["digest"] != baseline
    monkeypatch.setattr(stability_artifacts, "LABEL_VERSION", label_version)


@pytest.mark.parametrize(
    "malform",
    [
        lambda value: value.__setitem__("source_identity", {}),
        lambda value: value["source_identity"].__setitem__("tensor_shards", []),
        lambda value: value.__setitem__("standalone_vmf_identity", {}),
        lambda value: value.__setitem__("point_artifact_identity", {}),
        lambda value: value.__setitem__("point_record_hashes", {}),
        lambda value: value.__setitem__("point_records_sha256", "changed"),
        lambda value: value.__setitem__("point_records", []),
        lambda value: value.__setitem__("point_selections", {}),
        lambda value: value.pop("point_records"),
        lambda value: value.pop("point_selections"),
        lambda value: value["point_selections"].__setitem__(
            7,
            replace(value["point_selections"][7], point_reason="changed-authority"),
        ),
        lambda value: value["point_records"][0]["point_selection"].__setitem__(
            "point_reason", "changed-without-rehash"
        ),
        lambda value: value["point_records"][0].__setitem__(
            "changed_without_rehash", True
        ),
    ],
)
def test_checkpoint_fingerprint_rejects_empty_or_malformed_authority(malform) -> None:
    """Reject caller-shaped placeholders instead of hashing incomplete provenance."""
    # Production construction accepts only a complete validated point-loader bundle.
    schedule = _schedule()
    bundle = _validated_point_bundle(schedule)
    malform(bundle)
    with pytest.raises(ValueError, match="identity|inventory|hash"):
        _fingerprint(schedule, point_bundle=bundle)


def test_checkpoint_exact_reuse_and_fail_closed_rejections(tmp_path: Path) -> None:
    """Reuse exact ordered records and reject whole stale/corrupt checkpoint states."""
    # Write one exact checkpoint, then mutate orthogonal contract dimensions.
    schedule = _schedule()
    fingerprint = _fingerprint(schedule)
    arbitrary_record = _checkpoint_record(schedule)
    arbitrary_record.pop("selected_family_evidence")
    arbitrary_record["x"] = 1
    with pytest.raises(ValueError, match="evidence contract mismatch"):
        build_selected_family_checkpoint_payload(
            fingerprint=fingerprint,
            records=[arbitrary_record],
            schedules=[schedule],
        )
    with pytest.raises(ValueError, match="evidence contract mismatch"):
        build_selected_family_checkpoint_payload(
            fingerprint=fingerprint,
            records=[{**arbitrary_record, "selected_family_evidence": {}}],
            schedules=[schedule],
        )
    record = _checkpoint_record(schedule)
    payload = build_selected_family_checkpoint_payload(
        fingerprint=fingerprint, records=[record], schedules=[schedule]
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload))
    reused = load_selected_family_checkpoint(
        path,
        expected_fingerprint=fingerprint,
        expected_schedules=[schedule],
    )
    assert reused.status == "reused"
    assert reused.rejection_reason is None
    assert reused.records == (record,)

    # Coherently rehashed semantic drift must still fail the schedule-specific contract.
    evidence_mutations = [
        lambda value: value["required_protocol_ids"].append("extra"),
        lambda value: value["required_protocol_ids"].pop(),
        lambda value: (
            value["protocols"].__setitem__(
                "extra", {"status": "fixture", "counters": {}}
            ),
            value["protocol_counters"].__setitem__("extra", {}),
        ),
        lambda value: (
            value["protocols"].pop("angle"),
            value["protocol_counters"].pop("angle"),
        ),
        lambda value: value.__setitem__("no_work_reason", "wrong"),
        lambda value: value["protocols"]["angle"].__setitem__(
            "counters", {"requested": 1}
        ),
        lambda value: (
            value["protocol_counters"]["bootstrap"].__setitem__("failed", -1),
            value["protocols"]["bootstrap"]["counters"].__setitem__(
                "failed", -1
            ),
        ),
        lambda value: value["protocols"]["bootstrap"]["replicates"].pop(),
    ]
    for mutate_evidence in evidence_mutations:
        changed = copy.deepcopy(payload)
        mutate_evidence(changed["features"][0]["record"]["selected_family_evidence"])
        changed["features"][0]["record_sha256"] = canonical_json_digest(
            changed["features"][0]["record"]
        )
        path.write_text(json.dumps(changed))
        rejected = load_selected_family_checkpoint(
            path,
            expected_fingerprint=fingerprint,
            expected_schedules=[schedule],
        )
        assert rejected.rejection_reason == "checkpoint_record_evidence_mismatch"

    mutations = {
        "checkpoint_schema_version_mismatch": lambda value: value.__setitem__(
            "checkpoint_schema_version", 0
        ),
        "checkpoint_fingerprint_mismatch": lambda value: value[
            "fingerprint"
        ]["retained_stability_config"].__setitem__("seed", 123),
        "checkpoint_duplicate_feature_id": lambda value: value["features"].append(
            copy.deepcopy(value["features"][0])
        ),
        "checkpoint_record_hash_mismatch": lambda value: value["features"][
            0
        ].__setitem__("record_sha256", "stale"),
        "checkpoint_schedule_digest_mismatch": lambda value: value["features"][
            0
        ].__setitem__("schedule_digest", "stale"),
        "checkpoint_record_selection_mismatch": lambda value: (
            value["features"][0]["record"].__setitem__("family", "wrong"),
            value["features"][0].__setitem__(
                "record_sha256",
                canonical_json_digest(value["features"][0]["record"]),
            ),
        ),
        "checkpoint_record_evidence_mismatch": lambda value: (
            value["features"][0]["record"].pop("selected_family_evidence"),
            value["features"][0].__setitem__(
                "record_sha256",
                canonical_json_digest(value["features"][0]["record"]),
            ),
        ),
    }
    for expected_reason, mutate in mutations.items():
        changed = copy.deepcopy(payload)
        mutate(changed)
        path.write_text(json.dumps(changed))
        result = load_selected_family_checkpoint(
            path,
            expected_fingerprint=fingerprint,
            expected_schedules=[schedule],
        )
        assert result.status == "rejected"
        assert result.rejection_reason == expected_reason
        assert result.records == ()
    path.write_text("{")
    corrupt = load_selected_family_checkpoint(
        path, expected_fingerprint=fingerprint, expected_schedules=[schedule]
    )
    assert corrupt == type(corrupt)("rejected", "corrupt_checkpoint", ())

    path.unlink()
    missing = load_selected_family_checkpoint(
        path, expected_fingerprint=fingerprint, expected_schedules=[schedule]
    )
    assert missing == type(missing)("missing", None, ())


@pytest.mark.parametrize("feature_id", ["7", 7.0, True])
def test_checkpoint_rejects_non_exact_integer_feature_ids(feature_id: object) -> None:
    """Reject feature IDs whose JSON type would drift across clean and resumed runs."""
    # Scientific feature identity is an exact integer, never a coercible scalar.
    schedule = _schedule()
    record = _checkpoint_record(schedule)
    record["feature_id"] = feature_id

    with pytest.raises(ValueError, match="feature_id must be an integer"):
        build_selected_family_checkpoint_payload(
            fingerprint=_fingerprint(schedule),
            records=[record],
            schedules=[schedule],
        )


@pytest.mark.parametrize(
    ("family", "selected_k", "mode"),
    [
        ("directed_ray", None, "strict"),
        ("multi_mode_directional_geometry", None, "strict"),
        ("residual_lowD_k", 1, "fallback"),
    ],
)
def test_checkpoint_reuses_exact_real_mode_evidence(
    tmp_path: Path,
    family: str,
    selected_k: int | None,
    mode: SelectionMode,
) -> None:
    """Reuse exact normal, assignment-reuse, and deliberate-no-work evidence modes."""
    # Build each mode through the real schedule authority before checkpoint validation.
    schedule = _schedule(family, selected_k, mode=mode)
    fingerprint = _fingerprint(schedule)
    record = _checkpoint_record(schedule)
    payload = build_selected_family_checkpoint_payload(
        fingerprint=fingerprint, records=[record], schedules=[schedule]
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload))

    reused = load_selected_family_checkpoint(
        path,
        expected_fingerprint=fingerprint,
        expected_schedules=[schedule],
    )

    assert reused.status == "reused"
    assert reused.records == (record,)


def test_checkpoint_schedule_drift_rejects_all_records(tmp_path: Path) -> None:
    """Reject matching bytes when the current selected-family schedule has drifted."""
    # Keep the old fingerprint argument fixed to isolate schedule-digest validation.
    old = _schedule()
    current = build_selected_family_schedule(
        selection=old.point_selection,
        feature_id=old.feature_id,
        point_record_sha256="changed-point-record",
        base_seed=42,
        effect_space="final_resid",
        n_rows=32,
        group_labels=["a"] * 16 + ["b"] * 16,
        stability_config=StabilityConfig(),
    )
    fingerprint = _fingerprint(old)
    record = _checkpoint_record(old)
    payload = build_selected_family_checkpoint_payload(
        fingerprint=fingerprint, records=[record], schedules=[old]
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(payload))
    result = load_selected_family_checkpoint(
        path,
        expected_fingerprint=fingerprint,
        expected_schedules=[current],
    )
    assert result.status == "rejected"
    assert result.rejection_reason == "checkpoint_schedule_digest_mismatch"
    assert result.records == ()


def test_fingerprint_rejects_stale_point_selection_contract() -> None:
    """Reject mixed or stale point-selection contracts instead of taking a maximum."""
    # Contract equality is required for every schedule before checkpoint hashing.
    schedule = _schedule()
    stale = replace(
        schedule,
        point_selection=replace(schedule.point_selection, contract_version=0),
    )
    with pytest.raises(ValueError, match="point-selection version mismatch"):
        _fingerprint(stale, point_bundle=_validated_point_bundle(schedule))


def test_schedule_construction_cannot_reach_vmf_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep construction unable to fit, select BIC, or resample assignments."""
    # Any accidental call into either public vMF computation boundary fails loudly.
    from fega.core.vmf import fit as vmf_fit
    from fega.core.vmf import metrics as vmf_metrics

    def forbidden(*args, **kwargs):
        """Fail if schedule construction crosses into standalone-vMF computation."""
        # Arguments are intentionally ignored because every call is forbidden.
        del args, kwargs
        raise AssertionError("selected-family construction reached vMF computation")

    monkeypatch.setattr(vmf_fit, "fit_vmf_candidate", forbidden)
    monkeypatch.setattr(vmf_metrics, "score_vmf_feature", forbidden)
    schedule = _schedule("multi_mode_directional_geometry", None)
    assert schedule.reuse_standalone_assignment is True


def _source_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict, dict]:
    """Return one canonical source and matching geometry/vMF shells."""
    # Persist every file whose exact bytes enter the authoritative source identity.
    shard_path = tmp_path / "effect_tensors_00000.pt"
    gram_path = tmp_path / "gram.pt"
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    shard_path.write_bytes(b"shard")
    gram_path.write_bytes(b"gram")
    manifest = {
        "effect_space": "final_resid",
        "inputs": {"gram_path": str(gram_path)},
        "gram_metadata": {
            "checkpoint_identity": "test-model",
            "readout_name": "final_resid",
            "hidden_width": 2,
            "gram_dtype": "float32",
            "construction_recipe": "test",
            "unembedding_fingerprint": "readout",
            "unembedding_dtype": "torch.float32",
            "unembedding_shape": [2, 2],
            "gram_shape": [2, 2],
            "gram_sha256": "a" * 64,
        },
        "shards": [{"path": str(shard_path)}],
    }
    summary = {
        "per_feature": {
            "1": {
                "feature_id": 1,
                "candidate_identity": [],
                "retained_mask": [],
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest))
    summary_path.write_text(json.dumps(summary))
    source = canonical_source_fingerprint(manifest, summary)
    inputs = SimpleNamespace(
        manifest=manifest,
        summary=summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
        artifact_dir=tmp_path,
    )
    geometry = {
        "canonical_source_fingerprint": source,
        "per_feature": {"1": {"feature_id": 1}},
    }
    vmf = {
        "schema_version": 1,
        "canonical_source_fingerprint": source,
        "features": [{"feature_id": 1}],
    }
    return inputs, geometry, vmf


def test_point_loader_rejects_unversioned_vmf_and_accepts_current_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give legacy standalone-vMF artifacts a precise regeneration reason."""
    # Patch only heavyweight fingerprint validation; inventory and schema stay real.
    inputs, geometry, vmf = _source_fixture(tmp_path)
    geometry_path = tmp_path / "geometry.json"
    vmf_path = tmp_path / "vmf.json"
    geometry_path.write_text(json.dumps(geometry))
    vmf_path.write_text(
        json.dumps(
            {"canonical_source_fingerprint": vmf["canonical_source_fingerprint"]}
        )
    )
    monkeypatch.setattr(
        reporting_artifacts, "load_geometry_metrics_inputs", lambda *args: inputs
    )
    monkeypatch.setattr(
        reporting_artifacts, "resolve_final_resid_gram", lambda *args: None
    )
    monkeypatch.setattr(
        reporting_artifacts,
        "geometry_metrics_scores_path",
        lambda *args: geometry_path,
    )
    monkeypatch.setattr(reporting_artifacts, "vmf_scores_path", lambda *args: vmf_path)
    expected_vmf_fingerprint = {
        "schema_version": 3,
        "canonical_source": vmf["canonical_source_fingerprint"],
        "candidate_mode_counts": [1, 2, 3, 4],
        "assignment_fraction": 0.8,
        "assignment_rounds": 8,
        "seed_derivations": {"version": 1},
        "assignment_metric": {"identity": "adjusted_rand_score"},
        "feature_ids": [1],
    }
    monkeypatch.setattr(
        reporting_artifacts,
        "build_vmf_scientific_fingerprint",
        lambda **kwargs: expected_vmf_fingerprint,
    )
    monkeypatch.setattr(
        reporting_artifacts, "validate_vmf_scores", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        reporting_artifacts,
        "vmf_materialization_policy",
        lambda cfg: {"formula": "validated-test-materialization"},
    )
    config = SimpleNamespace(
        phases=SimpleNamespace(vmf=SimpleNamespace(seed=None)),
        seed=SimpleNamespace(global_=42),
    )
    with pytest.raises(
        reporting_artifacts.StandaloneVmfRegenerationRequiredError,
        match="reason=unversioned_standalone_vmf_artifact",
    ):
        reporting_artifacts.load_point_geometry_inputs(config)
    vmf["fingerprint"] = expected_vmf_fingerprint
    vmf_path.write_text(json.dumps(vmf))
    loaded = reporting_artifacts.load_point_geometry_inputs(config)
    assert loaded["feature_ids"] == [1]
    assert loaded["payloads"]["vmf_pre_softcap_logits"] == vmf
    assert set(loaded["input_artifact_hashes"]["geometry_metrics"]) == {
        "canonical_digest",
        "file_sha256",
    }
    assert loaded["source_identity"]["manifest"]["file_sha256"]
    assert loaded["source_identity"]["tensor_shards"][0]["path"] == str(
        tmp_path / "effect_tensors_00000.pt"
    )
    assert loaded["standalone_vmf_identity"]["candidate_schedule"] == [1, 2, 3, 4]
    assert loaded["vmf_scientific_fingerprint"] == expected_vmf_fingerprint


def test_point_artifact_writer_loader_and_fail_closed_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-trip the durable point contract and reject every identity corruption."""
    # Build two ordered complete selections so ordering and aggregate hashes are real.
    schedules = [
        _schedule(feature_id=1),
        _schedule("global_2D_directional_subspace", 2, feature_id=2),
    ]
    records: list[dict] = []
    record_hashes: dict[str, str] = {}
    for schedule in schedules:
        record = {
            "feature_id": schedule.feature_id,
            "point_selection": point_selection_identity(schedule.point_selection),
            "point_reason_source": "validated_point_inputs",
        }
        record_hash = canonical_json_digest(record)
        record["point_record_sha256"] = record_hash
        records.append(record)
        record_hashes[str(schedule.feature_id)] = record_hash
    inventory = [
        {
            "feature_id": record["feature_id"],
            "point_record_sha256": record["point_record_sha256"],
        }
        for record in records
    ]

    bundle = _validated_point_bundle(schedules[0])
    bundle["feature_ids"] = [1, 2]
    bundle["standalone_vmf_identity"]["feature_ids"] = [1, 2]
    bundle["standalone_vmf_identity"]["scientific_fingerprint"]["feature_ids"] = [
        1,
        2,
    ]
    inputs = {
        "paths": {
            "compute_effect_final_resid_manifest": "manifest.json",
            "compute_effect_final_resid": "summary.json",
            "geometry_metrics_final_resid": "geometry.json",
            "vmf_pre_softcap_logits": "vmf.json",
        },
        "payloads": {
            "compute_effect_final_resid": {"per_feature": {"1": 1, "2": 2}},
            "geometry_metrics_final_resid": {"per_feature": {"1": 1, "2": 2}},
            "vmf_pre_softcap_logits": {"features": [1, 2]},
        },
        "canonical_source_fingerprint": bundle["source_identity"][
            "canonical_source"
        ],
        "source_identity": bundle["source_identity"],
        "vmf_scientific_fingerprint": bundle["standalone_vmf_identity"][
            "scientific_fingerprint"
        ],
        "standalone_vmf_identity": bundle["standalone_vmf_identity"],
        "feature_ids": [1, 2],
        "input_artifact_hashes": bundle["input_artifact_hashes"],
    }
    summary = {
        "features_total": 2,
        "source_paths": inputs["paths"],
        "point_selection_contract_version": POINT_SELECTION_CONTRACT_VERSION,
        "point_record_hashes": record_hashes,
        "point_records_sha256": canonical_json_digest(inventory),
    }
    point_path = tmp_path / "geometry_point_records.json"
    monkeypatch.setattr(
        reporting_artifacts, "load_point_geometry_inputs", lambda *_args: inputs
    )
    monkeypatch.setattr(
        reporting_records,
        "build_point_geometry_records",
        lambda *_args: (copy.deepcopy(records), copy.deepcopy(summary)),
    )
    monkeypatch.setattr(
        reporting_artifacts, "point_geometry_records_path", lambda *_args: point_path
    )

    config = _pipeline_config()
    written = reporting_artifacts.load_build_and_write_point_geometry_records(config)
    loaded = reporting_artifacts.load_point_geometry_records(config)
    assert loaded["point_records"] == records
    assert loaded["point_record_hashes"] == record_hashes
    assert loaded["point_artifact_identity"] == written["point_artifact_identity"]
    assert loaded["point_selections"][1] == schedules[0].point_selection
    pristine = json.loads(point_path.read_text())

    corruptions = {
        "phase mismatch": lambda value: value.__setitem__("phase", "wrong"),
        "schema version mismatch": lambda value: value.__setitem__(
            "schema_version", 0
        ),
        "top-level keys mismatch": lambda value: value.__setitem__("extra", True),
        "source identity mismatch": lambda value: value["source_identity"].__setitem__(
            "manifest", {"changed": True}
        ),
        "standalone vMF identity mismatch": lambda value: value[
            "standalone_vmf_identity"
        ].__setitem__("candidate_schedule", [1]),
        "input artifact identity mismatch": lambda value: value[
            "input_artifact_hashes"
        ].__setitem__("geometry_metrics", {"changed": True}),
        "feature inventory order mismatch": lambda value: value["features"].reverse(),
        "point record hash mismatch": lambda value: value["features"][0].__setitem__(
            "point_record_sha256", "changed"
        ),
        "point record inventory hash mismatch": lambda value: value.__setitem__(
            "point_records_sha256", "changed"
        ),
        "point_selection fields mismatch": lambda value: value["features"][0][
            "point_selection"
        ].pop("point_reason"),
        "point-selection contract version mismatch": lambda value: value[
            "features"
        ][0]["point_selection"].__setitem__("contract_version", 0),
    }
    for expected, corrupt in corruptions.items():
        payload = copy.deepcopy(pristine)
        corrupt(payload)
        point_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=expected):
            reporting_artifacts.load_point_geometry_records(config)
