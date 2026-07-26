from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting import artifacts as geometry_artifacts
from fega.core.geometry_reporting import runner as reporting_runner
from fega.core.geometry_reporting.classifier import (
    GLOBAL_FLAG_MASK,
    GLOBAL_FLAG_ORDER,
    LABEL_VERSION,
    classify_record,
)
from fega.core.geometry_reporting.map.rows import feature_rows
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    point_selection_identity,
    resolve_point_selection,
)
from fega.core.geometry_reporting.schema import FALLBACK_PRIORITY
from fega.core.geometry_reporting.thresholds import get_threshold_profile
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.stability.artifacts import (
    SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION,
    SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
    STABILITY_PUBLIC_SCHEMA_VERSION,
)
from fega.core.stability.schedule import SELECTED_FAMILY_SCHEDULE_VERSION
from fega.paths import geometry_reporting_records_path

BANNED_KEYS = {
    "label_agreement",
    "selected_k_agreement",
    "low_strength",
    "top_positive_readout_tokens",
    "top_negative_readout_tokens",
    "mode_exemplars",
    "projection_histogram_summary",
    "likely_noise",
    "outlier_sensitive",
    "r2_mean_resultant",
}


def _base_classifier_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "feature_id": 1,
        "n_valid": 32,
        "zero_filter_frac": 0.0,
        "c_ray": 0.1,
        "s_span_1": 0.1,
        "s_span_2": 0.1,
        "r_span_pr": 1.0,
        "u_span_2": 0.01,
        "d_span_2": 1.0,
        "b_axis": 0.0,
        "m_cv": 0.0,
        "scalar_ci": {"c_ray": {"ci_low": 0.05, "ci_high": 0.2}},
        "subspace_stability": {
            "status": "ok",
            "k": {
                "1": {"status": "ok", "subspace_angle_p90_k": 10.0},
                "2": {"status": "ok", "subspace_angle_p90_k": 10.0},
            },
        },
        "centered_residual_subspace_stability": {
            "source": "stability_artifact",
            "status": "ok",
            "k": {"2": {"status": "ok", "residual_angle_p90_k": 10.0}},
        },
        "sample_size_curves": {
            "requested_count": 1,
            "valid_count": 1,
            "failed_count": 0,
            "geometry_status": "stable",
        },
        "leave_out_sensitivity": {
            "requested_count": 1,
            "valid_count": 1,
            "failed_count": 0,
            "geometry_status": "stable",
        },
        "eps": 1.0e-12,
    }
    record.update(overrides)
    return record


def _assert_v3_schema(record: dict[str, Any]) -> None:
    assert record["label_version"] == LABEL_VERSION
    assert record["threshold_profile"] == "paper"
    assert record["label_confidence"] in {
        None,
        "accepted",
        "exploratory",
        "unstable",
        "candidate",
        "insufficient",
        "unavailable",
        "undefined",
    }
    assert isinstance(record["flag_details"], dict)
    assert isinstance(record["candidate_labels"], list)
    assert record["global_flags"] == [
        flag for flag in GLOBAL_FLAG_ORDER if flag in record["secondary_flags"]
    ]
    assert record["global_flag_count"] == len(record["global_flags"])
    assert record["global_flag_mask"] == "|".join(
        GLOBAL_FLAG_MASK[flag] for flag in record["global_flags"]
    )
    for candidate in record["candidate_labels"]:
        assert {
            "family",
            "priority",
            "selected_k",
            "confidence",
            "anchor_fields",
            "failed_fields",
            "missing_fields",
            "flags",
            "details",
        } <= set(candidate)


def _candidate_for(
    record: dict[str, Any], family: str
) -> dict[str, Any] | None:
    for candidate in record["candidate_labels"]:
        if candidate["family"] == family:
            return candidate
    return None


def test_threshold_profiles_match_paper_defaults() -> None:
    paper = get_threshold_profile("paper")

    assert paper.n_min == 8
    assert paper.tau_zero_filter_frac == pytest.approx(0.30)
    assert paper.tau_c_ray == pytest.approx(0.80)
    assert paper.tau_axis == pytest.approx(0.80)
    assert paper.tau_r_2D == pytest.approx(1.45)
    assert paper.tau_b_axis == pytest.approx(0.15)
    assert paper.tau_m_cv == pytest.approx(1.00)
    assert paper.tau_span_k == pytest.approx(0.90)
    assert paper.tau_r[2] == pytest.approx(1.60)
    assert paper.tau_r[3] == pytest.approx(2.30)
    assert paper.tau_r[4] == pytest.approx(3.00)
    assert paper.tau_r[8] == pytest.approx(5.00)
    assert paper.tau_p[2] == pytest.approx(0.08)
    assert paper.tau_p[3] == pytest.approx(0.05)
    assert paper.tau_p[4] == pytest.approx(0.03)
    assert paper.tau_p[8] == pytest.approx(0.01)
    assert paper.tau_gap_k == pytest.approx(0.60)
    assert paper.tau_subspace_angle["1"] == pytest.approx(30.0)
    assert paper.tau_subspace_angle["2"] == pytest.approx(30.0)
    assert paper.tau_subspace_angle["k"] == pytest.approx(35.0)
    assert paper.tau_mix == pytest.approx(0.10)
    assert paper.tau_mode_mass == pytest.approx(0.10)
    assert paper.tau_mode_c_ray == pytest.approx(0.70)
    assert paper.tau_assignment_stability == pytest.approx(0.80)
    assert paper.tau_res == pytest.approx(0.10)
    assert paper.tau_ctr[1] == pytest.approx(0.80)
    assert paper.tau_ctr[4] == pytest.approx(0.80)
    assert paper.tau_r_ctr[2] == pytest.approx(1.50)
    assert paper.tau_r_ctr[3] == pytest.approx(2.20)
    assert paper.tau_r_ctr[4] == pytest.approx(2.90)
    assert 1 not in paper.tau_r_ctr
    assert paper.tau_longtail == pytest.approx(1.50)

    with pytest.raises(ValueError, match="Unknown geometry_reporting threshold profile"):
        get_threshold_profile("exploratory")
    with pytest.raises(ValueError, match="Unknown geometry_reporting threshold profile"):
        get_threshold_profile("stronger")


def _selected_family_evidence(
    *, instability: int = 0, failures: int = 0, low_context: str = "ok"
) -> dict[str, Any]:
    """Build one complete raw WP3 ray-evidence record for reporting tests."""
    # Keep every raw count explicit so the test exercises reporting projection only.
    counts = {
        "requested": 1,
        "valid": 1,
        "failed": 0,
        "non_applicable": 0,
        "skipped": 0,
    }
    protocols = {
        "low_context_qualification": {
            "status": low_context,
            "plan_digest": "low-context",
            "counters": dict(counts),
        },
        "bootstrap": {
            "status": "ok",
            "plan_digest": "bootstrap",
            "counters": dict(counts),
        },
        "leave_out": {
            "status": "ok",
            "plan_digest": "leave-out",
            "counters": dict(counts),
        },
        "sample_size": {
            "status": "failed" if failures else "ok",
            "plan_digest": "sample-size",
            "counters": dict(counts, valid=0, failed=1) if failures else dict(counts),
        },
    }
    return {
        "required_protocol_ids": list(protocols),
        "no_work_reason": None,
        "completed_instability_count": instability,
        "required_failure_count": failures,
        "point_margins": {"c_ray_ge": 0.1, "s_span_1_axis": 0.1},
        "protocols": protocols,
        "protocol_counters": {
            key: dict(value["counters"]) for key, value in protocols.items()
        },
    }


@pytest.mark.parametrize(
    ("instability", "failures", "low_context", "confidence", "evidence_status"),
    [
        (1, 0, "ok", "unstable", "available"),
        (1, 1, "ok", "unstable", "unavailable"),
        (0, 0, "ok", "accepted", "available"),
        (0, 0, "exploratory", "exploratory", "available"),
        (0, 1, "exploratory", None, "unavailable"),
    ],
)
def test_selected_family_reporting_maps_completed_evidence_without_relabeling(
    instability: int,
    failures: int,
    low_context: str,
    confidence: str | None,
    evidence_status: str,
) -> None:
    """Map the five executable rows of the closed WP4 state table exactly."""
    # Lock a point-selected ray before supplying raw selected-family evidence.
    record = _base_classifier_record(c_ray=0.9, s_span_1=0.9)
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    record["point_selection"] = point_selection_identity(selection)
    record["selected_family_evidence"] = _selected_family_evidence(
        instability=instability,
        failures=failures,
        low_context=low_context,
    )

    classified = classify_record(record)

    assert classified["primary_label"] == "directed_ray"
    assert classified["selected_k"] is None
    assert classified["label_confidence"] == confidence
    assert classified["evidence_status"] == evidence_status
    assert "general_stability" not in classified
    assert classified["selected_family_stability"]["family"] == "directed_ray"


@pytest.mark.parametrize(
    ("overrides", "family", "gate_key"),
    [
        ({"c_ray": 0.9, "s_span_1": 0.9}, "directed_ray", "directed_ray"),
        (
            {"c_ray": 0.2, "s_span_1": 0.9, "b_axis": 0.3},
            "axis_or_antipodal",
            "axis_or_antipodal",
        ),
        (
            {"s_span_2": 0.92, "r_span_pr": 1.8, "u_span_2": 0.1, "d_span_2": 0.4},
            "global_2D_directional_subspace",
            "global_directional_subspace",
        ),
        (
            {"e_res": 0.2, "s_res_2": 0.9, "r_ctr_pr": 1.6},
            "residual_lowD_k",
            "residual_lowD_k",
        ),
    ],
)
def test_locked_selected_family_diagnostics_are_point_only(
    overrides: dict[str, float], family: str, gate_key: str
) -> None:
    """Accepted WP4 evidence must not expose retired stability contradictions."""
    # Model the production point artifact, which intentionally has no legacy evidence.
    record = _base_classifier_record(**overrides)
    for key in (
        "scalar_ci",
        "subspace_stability",
        "centered_residual_subspace_stability",
        "sample_size_curves",
        "leave_out_sensitivity",
    ):
        record.pop(key, None)
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    assert selection.family == family
    record["point_selection"] = point_selection_identity(selection)
    record["selected_family_evidence"] = _selected_family_evidence()

    classified = classify_record(record)

    obsolete = {
        "directed_ray_ci_missing",
        "axis_ci_missing",
        "axis_stability_missing",
        "span_stability_missing",
        "residual_stability_missing",
        "lowD_candidate_blocked",
    }
    assert classified["label_confidence"] == "accepted"
    assert classified["gate_evidence"][gate_key]["decision"] == "stable"
    assert obsolete.isdisjoint(classified["secondary_flags"])
    selected_candidate = _candidate_for(classified, family)
    assert selected_candidate is not None
    assert selected_candidate["confidence"] == "stable"
    assert obsolete.isdisjoint(selected_candidate["flags"])


def test_locked_ray_preserves_truthful_point_blocked_span_audit() -> None:
    """Keep a point-metric-blocked span candidate beside accepted ray stability."""
    # The span anchor is present, but its full-sample participation ratio fails.
    record = _base_classifier_record(
        c_ray=0.9,
        s_span_1=0.9,
        s_span_2=0.92,
        r_span_pr=1.0,
        u_span_2=0.1,
        d_span_2=0.4,
    )
    for key in (
        "scalar_ci",
        "subspace_stability",
        "centered_residual_subspace_stability",
        "sample_size_curves",
        "leave_out_sensitivity",
    ):
        record.pop(key, None)
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    record["point_selection"] = point_selection_identity(selection)
    record["selected_family_evidence"] = _selected_family_evidence()

    classified = classify_record(record)

    assert classified["primary_label"] == "directed_ray"
    assert classified["label_confidence"] == "accepted"
    assert classified["evidence_status"] == "available"
    assert "lowD_candidate_blocked" in classified["secondary_flags"]
    assert classified["flag_details"]["lowD_candidate_blocked"]["reason"] == (
        "strict_directed_ray_priority"
    )
    assert classified["flag_details"]["lowD_candidate_blocked"][
        "failed_fields"
    ] == ["r_span_pr"]


def test_selected_family_reporting_maps_mixture_and_deliberate_no_work_rows() -> None:
    """Cover accepted mixture reuse and preserved fallback confidence exactly."""
    # Reuse complete assignment evidence for the strict mixture row.
    mixture = _base_classifier_record(
        fit_status="fitted",
        selected_mode_count=2,
        model_selection={"selected_mode_count": 2},
        delta_mix=0.2,
        mode_mass_min=0.2,
        min_mode_c_ray=0.8,
        mode_kappa_min=10.0,
        assignment_stability={"status": "available", "value": 0.9},
    )
    mixture_selection = resolve_point_selection(
        mixture, vmf_provenance_valid=True
    )
    mixture["point_selection"] = point_selection_identity(mixture_selection)
    reuse_counts = {
        "requested": 1,
        "valid": 1,
        "failed": 0,
        "non_applicable": 0,
        "skipped": 0,
    }
    mixture["selected_family_evidence"] = {
        "required_protocol_ids": ["standalone_assignment_reuse"],
        "no_work_reason": None,
        "protocols": {
            "standalone_assignment_reuse": {
                "status": "reused",
                "assignment_stability": mixture["assignment_stability"],
                "plan_digest": "assignment",
                "counters": reuse_counts,
            }
        },
        "protocol_counters": {"standalone_assignment_reuse": reuse_counts},
    }
    classified_mixture = classify_record(mixture)
    assert classified_mixture["primary_label"] == "multi_mode_directional_geometry"
    assert classified_mixture["label_confidence"] == "accepted"
    assert classified_mixture["evidence_status"] == "available"

    # Preserve the existing one-dimensional fallback confidence without profiling it.
    fallback = _base_classifier_record(c_ray=0.2, s_span_1=0.9, b_axis=0.05)
    fallback_selection = resolve_point_selection(
        fallback, vmf_provenance_valid=True
    )
    fallback["point_selection"] = point_selection_identity(fallback_selection)
    no_work_counts = {
        "requested": 0,
        "valid": 0,
        "failed": 0,
        "non_applicable": 1,
        "skipped": 0,
    }
    fallback["selected_family_evidence"] = {
        "required_protocol_ids": ["deliberate_non_evaluation"],
        "no_work_reason": "not_evaluated_point_fallback_or_terminal",
        "protocols": {
            "deliberate_non_evaluation": {
                "status": "not_evaluated",
                "reason": "not_evaluated_point_fallback_or_terminal",
                "plan_digest": "none",
                "counters": no_work_counts,
            }
        },
        "protocol_counters": {"deliberate_non_evaluation": no_work_counts},
    }
    classified_fallback = classify_record(fallback)
    assert classified_fallback["primary_label"] == "oneD_diffuse"
    assert classified_fallback["label_confidence"] == "candidate"
    assert classified_fallback["evidence_status"] == "not_evaluated"
    assert "selected_family_not_evaluated" in classified_fallback["secondary_flags"]

    # Preserve terminal confidence under the same deliberate no-work contract.
    terminal = _base_classifier_record(n_valid=3)
    terminal_selection = resolve_point_selection(
        terminal, vmf_provenance_valid=True
    )
    terminal["point_selection"] = point_selection_identity(terminal_selection)
    terminal["selected_family_evidence"] = fallback["selected_family_evidence"]
    classified_terminal = classify_record(terminal)
    assert classified_terminal["primary_label"] == "insufficient_effect_evidence"
    assert classified_terminal["label_confidence"] == "insufficient"
    assert classified_terminal["evidence_status"] == "not_evaluated"


def test_missing_or_skipped_selected_family_evidence_is_unavailable() -> None:
    """Never translate a skipped required protocol into stable evidence."""
    # Keep the point lock valid while making one requested protocol incomplete.
    record = _base_classifier_record(c_ray=0.9, s_span_1=0.9)
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    record["point_selection"] = point_selection_identity(selection)
    evidence = _selected_family_evidence()
    skipped = evidence["protocol_counters"]["sample_size"]
    skipped.update({"requested": 1, "valid": 0, "skipped": 1})
    evidence["protocols"]["sample_size"]["counters"] = dict(skipped)
    record["selected_family_evidence"] = evidence

    classified = classify_record(record)

    assert classified["label_confidence"] is None
    assert classified["evidence_status"] == "unavailable"
    assert "selected_family_evidence_unavailable" in classified["secondary_flags"]


@pytest.mark.parametrize(
    ("overrides", "family"),
    [
        (
            {"s_span_2": 0.92, "r_span_pr": 1.8, "u_span_2": 0.1, "d_span_2": 0.4},
            "global_2D_directional_subspace",
        ),
        (
            {"e_res": 0.2, "s_res_2": 0.9, "r_ctr_pr": 1.6},
            "residual_lowD_k",
        ),
    ],
)
def test_selected_family_mismatch_evidence_cannot_change_locked_k(
    overrides: dict[str, float], family: str
) -> None:
    """Keep span and residual k locked when a subset selects another dimension."""
    # Record a completed mismatch as instability without invoking family reselection.
    record = _base_classifier_record(**overrides)
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    assert selection.family == family
    assert selection.selected_k == 2
    record["point_selection"] = point_selection_identity(selection)
    evidence = _selected_family_evidence(instability=1)
    evidence["protocols"]["leave_out"]["derived_selected_k"] = 3
    record["selected_family_evidence"] = evidence

    classified = classify_record(record)

    assert classified["primary_label"] == family
    assert classified["selected_k"] == 2
    assert classified["label_confidence"] == "unstable"


def test_reporting_uses_current_point_lock_provenance_and_public_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run one production-shaped report and reject a drifted WP3 point lock."""
    # Freeze one durable point record and bind the current stability fingerprint to it.
    config = FEGAPipelineConfig(
        reference_json=tmp_path / "reference.json",
        output_root=tmp_path / "results",
        device="cpu",
        entity_attribute_selection={"city": ["Country"]},
    )
    config.phases.geometry_reporting.map_enabled = False
    point_record = _base_classifier_record(c_ray=0.9, s_span_1=0.9)
    for key in (
        "scalar_ci",
        "subspace_stability",
        "centered_residual_subspace_stability",
        "sample_size_curves",
        "leave_out_sensitivity",
    ):
        point_record.pop(key, None)
    selection = resolve_point_selection(point_record, vmf_provenance_valid=True)
    point_record["point_selection"] = point_selection_identity(selection)
    point_hash = canonical_json_digest(point_record)
    point_record["point_record_sha256"] = point_hash
    point_inventory = canonical_json_digest(
        [{"feature_id": 1, "point_record_sha256": point_hash}]
    )
    point_identity = {"canonical_digest": "point", "file_sha256": "point-file"}
    point_bundle = {
        "feature_ids": [1],
        "canonical_source_fingerprint": {"digest": "source"},
        "source_identity": {"canonical_source": {"digest": "source"}},
        "input_artifact_hashes": {"geometry_metrics": {"file_sha256": "geometry"}},
        "standalone_vmf_identity": {"artifact": {"file_sha256": "vmf"}},
        "point_artifact_identity": point_identity,
        "point_record_hashes": {"1": point_hash},
        "point_records_sha256": point_inventory,
        "point_records": [point_record],
        "point_selections": {1: selection},
        "paths": {"geometry_point_records": "point.json"},
        "payloads": {},
    }
    stability_record = {
        "feature_id": 1,
        "family": selection.family,
        "selection_mode": selection.mode,
        "selected_k": selection.selected_k,
        "point_reason": selection.point_reason,
        "schedule_digest": "schedule",
        "point_record_sha256": point_hash,
        "n_valid": 32,
        "selected_family_evidence": _selected_family_evidence(),
    }
    fingerprint = {
        "fingerprint_version": SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION,
        "checkpoint_schema_version": SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
        "stability_public_schema_version": STABILITY_PUBLIC_SCHEMA_VERSION,
        "point_artifact_identity": point_identity,
        "point_record_hashes": {"1": point_hash},
        "point_records_sha256": point_inventory,
        "label_version": LABEL_VERSION,
        "point_selection_version": POINT_SELECTION_CONTRACT_VERSION,
        "selected_family_schedule_version": SELECTED_FAMILY_SCHEDULE_VERSION,
        "retained_stability_config": {
            key: value
            for key, value in config.to_dict()["phases"]["stability"].items()
            if key not in {"workers", "resume", "checkpoint_flush_features"}
        },
        "threshold_profile": "paper",
        "locked_features": [
            {
                "feature_id": 1,
                "family": selection.family,
                "selection_mode": selection.mode,
                "selected_k": selection.selected_k,
                "schedule_digest": "schedule",
            }
        ],
    }
    fingerprint["digest"] = canonical_json_digest(fingerprint)
    stability_path = tmp_path / "stability.json"
    stability_payload = {
        "phase": "stability",
        "schema_version": STABILITY_PUBLIC_SCHEMA_VERSION,
        "canonical_source_fingerprint": point_bundle[
            "canonical_source_fingerprint"
        ],
        "fingerprint": fingerprint,
        "config": copy.deepcopy(fingerprint["retained_stability_config"]),
        "effect_spaces": {
            "final_resid": {"per_feature": {"1": stability_record}}
        },
    }
    stability_path.write_text(json.dumps(stability_payload))
    monkeypatch.setattr(
        geometry_artifacts, "load_point_geometry_records", lambda *_args: point_bundle
    )
    monkeypatch.setattr(
        geometry_artifacts, "stability_scores_path", lambda *_args: stability_path
    )

    inputs = geometry_artifacts.load_geometry_inputs(config)
    config.phases.stability.scalar.bootstrap_rounds += 1
    inputs = geometry_artifacts.load_geometry_inputs(config)
    monkeypatch.setattr(reporting_runner, "load_geometry_inputs", lambda *_args: inputs)
    monkeypatch.setattr(reporting_runner, "write_gate_diagnostics", lambda *_args: None)
    reporting_runner.run_geometry_reporting(config)
    payload = json.loads(geometry_reporting_records_path(config).read_text())
    assert payload["schema_version"] == 2
    assert payload["label_version"] == "fega_geometry_labels_v3"
    assert [record["feature_id"] for record in payload["features"]] == [1]
    assert "selected_family_stability" in payload["features"][0]
    assert "general_stability" not in payload["features"][0]

    for field, drifted in (
        (
            "fingerprint_version",
            SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION + 1,
        ),
        (
            "selected_family_schedule_version",
            SELECTED_FAMILY_SCHEDULE_VERSION + 1,
        ),
    ):
        drifted_payload = copy.deepcopy(stability_payload)
        drifted_payload["fingerprint"][field] = drifted
        drifted_payload["fingerprint"]["digest"] = canonical_json_digest(
            {
                key: value
                for key, value in drifted_payload["fingerprint"].items()
                if key != "digest"
            }
        )
        stability_path.write_text(json.dumps(drifted_payload))
        with pytest.raises(ValueError, match=field):
            geometry_artifacts.load_geometry_inputs(config)

    drifted_payload = copy.deepcopy(stability_payload)
    drifted_payload["fingerprint"]["retained_stability_config"]["seed"] = 123
    drifted_payload["fingerprint"]["digest"] = canonical_json_digest(
        {
            key: value
            for key, value in drifted_payload["fingerprint"].items()
            if key != "digest"
        }
    )
    stability_path.write_text(json.dumps(drifted_payload))
    with pytest.raises(ValueError, match="retained_stability_config"):
        geometry_artifacts.load_geometry_inputs(config)

    stability_path.write_text(json.dumps(stability_payload))
    stability_payload["effect_spaces"]["final_resid"]["per_feature"]["1"][
        "selected_k"
    ] = 2
    stability_path.write_text(json.dumps(stability_payload))
    with pytest.raises(ValueError, match="locked selection mismatch"):
        geometry_artifacts.load_geometry_inputs(config)


def test_geometry_map_vector_uses_canonical_final_resid_r2() -> None:
    """Source map r2 from the canonical classified record without logits input."""
    # Build the map row through the one-input scientific API and inspect r2.
    record = _base_classifier_record(r2=0.625)

    rows = feature_rows([record])

    assert rows[0]["vector"]["r2"] == pytest.approx(0.625)
    assert rows[0]["missingness"]["r2"] is False



def test_classifier_primary_labels_schema_and_strict_order() -> None:
    insufficient = classify_record(_base_classifier_record(n_valid=3))
    _assert_v3_schema(insufficient)
    assert insufficient["primary_label"] == "insufficient_effect_evidence"
    assert insufficient["terminal_reason"] == "effect_count_below_min"
    assert insufficient["label_confidence"] == "insufficient"

    directed = classify_record(
        _base_classifier_record(
            n_valid=8,
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
            m_cv=1.2,
        )
    )
    _assert_v3_schema(directed)
    assert directed["primary_label"] == "directed_ray"
    assert directed["strict_gate_label"] == "directed_ray"
    assert directed["label_confidence"] == "accepted"
    assert directed["global_flags"] == ["magnitude_unstable"]
    assert directed["global_flag_mask"] == "MAG"

    axis = classify_record(
        _base_classifier_record(
            s_span_1=0.9,
            c_ray=0.2,
            b_axis=0.3,
            scalar_ci={"c_ray": {"ci_low": 0.1, "ci_high": 0.3}},
        )
    )
    _assert_v3_schema(axis)
    assert axis["primary_label"] == "axis_or_antipodal"
    assert axis["strict_gate_label"] == "axis_or_antipodal"

    multimode = classify_record(
        _base_classifier_record(
            selected_mode_count=2,
            delta_mix=0.2,
            mode_mass_min=0.2,
            min_mode_c_ray=0.8,
            mode_kappa_min=10.0,
            assignment_stability={"status": "available", "value": 0.9},
        )
    )
    _assert_v3_schema(multimode)
    assert multimode["primary_label"] == "multi_mode_directional_geometry"
    assert multimode["strict_gate_label"] == "multi_mode_directional_geometry"

    span = classify_record(
        _base_classifier_record(
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
        )
    )
    _assert_v3_schema(span)
    assert span["primary_label"] == "global_2D_directional_subspace"
    assert span["strict_gate_label"] == "global_2D_directional_subspace"
    assert span["selected_k"] == 2
    assert span["span_selected_k"] == 2

    residual = classify_record(
        _base_classifier_record(e_res=0.2, s_res_2=0.9, r_ctr_pr=1.6)
    )
    _assert_v3_schema(residual)
    assert residual["primary_label"] == "residual_lowD_k"
    assert residual["strict_gate_label"] == "residual_lowD_k"
    assert residual["selected_k"] == 2
    assert residual["residual_selected_k"] == 2
    assert "residual_lowD_k" not in residual["secondary_flags"]


def test_rejected_selected_m2_retains_identity_and_explicit_reporting_gates() -> None:
    """Keep BIC-selected M=2 immutable when paper reporting gates reject it."""
    # Supply a fitted mixture with failed mass and unavailable assignment evidence.
    record = _base_classifier_record(
        fit_status="fitted",
        selected_mode_count=2,
        model_selection={"selected_mode_count": 2},
        delta_mix=0.2,
        mode_mass_min=0.05,
        min_mode_c_ray=0.8,
        mode_kappa_min=10.0,
        assignment_stability={
            "status": "unavailable",
            "value": None,
            "requested_count": 3,
            "successful_count": 2,
            "failed_count": 1,
            "replicates": [],
        },
    )
    selection = resolve_point_selection(record, vmf_provenance_valid=True)
    record["point_selection"] = point_selection_identity(selection)
    no_work_counts = {
        "requested": 0,
        "valid": 0,
        "failed": 0,
        "non_applicable": 1,
        "skipped": 0,
    }
    record["selected_family_evidence"] = {
        "required_protocol_ids": ["deliberate_non_evaluation"],
        "no_work_reason": "not_evaluated_point_fallback_or_terminal",
        "protocols": {
            "deliberate_non_evaluation": {
                "status": "not_evaluated",
                "reason": "not_evaluated_point_fallback_or_terminal",
                "plan_digest": "none",
                "counters": no_work_counts,
            }
        },
        "protocol_counters": {"deliberate_non_evaluation": no_work_counts},
    }
    classified = classify_record(record)

    gate = classified["gate_evidence"]["multi_mode_directional_geometry"]
    assert record["selected_mode_count"] == 2
    assert gate["evaluated"] is True
    assert gate["acceptance"] == "rejected"
    assert gate["failed_gates"] == ["fitted_mass"]
    assert gate["unavailable_gates"] == ["assignment_stability"]
    assert classified["primary_label"] != "multi_mode_directional_geometry"
    candidate = _candidate_for(classified, "multi_mode_directional_geometry")
    assert candidate is not None
    assert candidate["confidence"] == "candidate_blocked"
    assert candidate["details"]["reporting_acceptance"] == "rejected"
    assert candidate["details"]["failed_gates"] == ["fitted_mass"]
    assert candidate["details"]["unavailable_gates"] == [
        "assignment_stability"
    ]
    mixture_audit = classified["point_selection"]["mixture_audit_state"]
    assert mixture_audit["status"] == "unavailable"
    assert mixture_audit["acceptance"] == "rejected"
    assert classified["evidence_status"] == "not_evaluated"
    assert "stability_unavailable" in gate["blocked_reasons"]


def test_absent_required_stability_protocols_cannot_accept_a_strict_family() -> None:
    """Require missing protocol evidence to block otherwise strict point gates."""
    # Remove both required resampling blocks from an otherwise accepted directed ray.
    classified = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            scalar_ci={"c_ray": {"ci_low": 0.85, "ci_high": 0.95}},
            sample_size_curves={},
            leave_out_sensitivity={},
        )
    )

    assert classified["strict_gate_label"] is None
    assert classified["label_confidence"] == "candidate"
    assert classified["gate_evidence"]["directed_ray"]["sample_size_status"] == (
        "not_available"
    )


def test_classifier_v3_candidate_fallbacks_and_blockers() -> None:
    """Preserve candidate fallback families and their evidence blockers."""
    # Exercise missing-CI ray fallback away from the participation-rank boundary.
    ray_boundary = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.8,
            scalar_ci={},
        )
    )
    _assert_v3_schema(ray_boundary)
    assert ray_boundary["primary_label"] == "directed_ray"
    assert ray_boundary["strict_gate_label"] is None
    assert ray_boundary["label_confidence"] == "candidate"
    assert "directed_ray_ci_missing" in ray_boundary["secondary_flags"]

    ray_not_boundary = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.8,
            scalar_ci={},
        )
    )
    assert ray_not_boundary["primary_label"] == "directed_ray"
    assert "directed_ray_ci_missing" in ray_not_boundary["secondary_flags"]
    assert "ray_span_boundary" not in ray_not_boundary["secondary_flags"]

    ray_ci_unstable = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            scalar_ci={"c_ray": {"ci_low": 0.7, "ci_high": 0.95}},
        )
    )
    assert ray_ci_unstable["primary_label"] == "directed_ray"
    assert ray_ci_unstable["label_confidence"] == "candidate"
    assert "directed_ray_ci_unstable" in ray_ci_unstable["secondary_flags"]

    axis_missing_stability = classify_record(
        _base_classifier_record(
            s_span_1=0.9,
            c_ray=0.2,
            b_axis=0.3,
            scalar_ci={"c_ray": {"ci_low": 0.1, "ci_high": 0.3}},
            subspace_stability={"status": "ok", "k": {}},
        )
    )
    assert axis_missing_stability["primary_label"] == "axis_or_antipodal"
    assert axis_missing_stability["label_confidence"] == "candidate"
    assert "axis_stability_missing" in axis_missing_stability["secondary_flags"]

    one_d = classify_record(
        _base_classifier_record(
            s_span_1=0.9,
            c_ray=0.2,
            b_axis=0.05,
            scalar_ci={"c_ray": {"ci_low": 0.1, "ci_high": 0.3}},
        )
    )
    assert one_d["primary_label"] == "oneD_diffuse"
    assert {"oneD_not_ray_not_axis", "b_axis_low"} <= set(
        one_d["secondary_flags"]
    )

    multimode_blocked = classify_record(
        _base_classifier_record(
            selected_mode_count=2,
            delta_mix=0.2,
            mode_mass_min=0.05,
            min_mode_c_ray=0.8,
            assignment_stability={"status": "available", "value": 0.9},
        )
    )
    assert multimode_blocked["primary_label"] != "multi_mode_directional_geometry"
    blocked_candidate = _candidate_for(
        multimode_blocked, "multi_mode_directional_geometry"
    )
    assert blocked_candidate is not None
    assert blocked_candidate["details"]["reporting_acceptance"] == "rejected"

    multimode_without_passing_mode_metric = classify_record(
        _base_classifier_record(
            selected_mode_count=2,
            delta_mix=0.2,
            mode_mass_min=0.05,
            min_mode_c_ray=0.6,
            assignment_stability={"status": "available", "value": 0.7},
        )
    )
    assert multimode_without_passing_mode_metric["primary_label"] != (
        "multi_mode_directional_geometry"
    )
    assert (
        _candidate_for(
            multimode_without_passing_mode_metric,
            "multi_mode_directional_geometry",
        )
        is None
    )

    span_missing_metric = classify_record(
        _base_classifier_record(
            s_span_2=0.92,
            r_span_pr=None,
            u_span_2=0.1,
            d_span_2=0.4,
        )
    )
    assert span_missing_metric["primary_label"] == "global_2D_directional_subspace"
    assert span_missing_metric["span_selected_k"] == 2
    assert span_missing_metric["label_confidence"] == "candidate"
    assert "lowD_candidate_blocked" in span_missing_metric["secondary_flags"]
    span_candidate = _candidate_for(
        span_missing_metric, "global_2D_directional_subspace"
    )
    assert span_candidate is not None
    assert "r_span_pr" in span_candidate["missing_fields"]

    residual_k1 = classify_record(
        _base_classifier_record(e_res=0.2, s_res_1=0.9, r_ctr_pr=1.0)
    )
    assert residual_k1["primary_label"] == "residual_lowD_k"
    assert residual_k1["selected_k"] == 1
    assert residual_k1["residual_selected_k"] == 1
    assert residual_k1["strict_gate_label"] is None

    residual_k4 = classify_record(
        _base_classifier_record(
            e_res=0.2,
            s_res_4=0.9,
            r_ctr_pr=3.0,
            centered_residual_subspace_stability={
                "source": "stability_artifact",
                "status": "ok",
                "k": {"4": {"status": "ok", "residual_angle_p90_k": 10.0}},
            },
        )
    )
    assert residual_k4["primary_label"] == "residual_lowD_k"
    assert residual_k4["strict_gate_label"] == "residual_lowD_k"
    assert residual_k4["selected_k"] == 4
    assert residual_k4["residual_selected_k"] == 4
    assert "residual_k_unsupported" not in residual_k4["secondary_flags"]

    missing_centered = classify_record(
        _base_classifier_record(
            e_res=0.2,
            s_res_2=0.9,
            r_ctr_pr=1.6,
            centered_residual_subspace_stability={
                "source": "stability_artifact",
                "status": "ok",
                "k": {},
            },
        )
    )
    assert missing_centered["primary_label"] == "residual_lowD_k"
    assert missing_centered["label_confidence"] == "candidate"
    assert "residual_stability_missing" in missing_centered["secondary_flags"]
    assert (
        missing_centered["gate_evidence"]["residual_lowD_k"]["decision"]
        == "not_available"
    )


@pytest.mark.parametrize(
    ("one_d_overrides", "expected_leading_family"),
    [
        ({"c_ray": 0.2, "b_axis": 0.3}, "axis_or_antipodal"),
        ({"c_ray": 0.2, "b_axis": 0.05}, "oneD_diffuse"),
    ],
)
def test_d6_fallback_priority_prefers_one_d_before_downstream_candidates(
    one_d_overrides: dict[str, Any], expected_leading_family: str
) -> None:
    """Lock schema order and prove each non-ray 1D family wins downstream ties."""
    # Combine one 1D anchor with compatible multimode, span, and residual candidates.
    classified = classify_record(
        _base_classifier_record(
            s_span_1=0.9,
            r_span_pr=None,
            scalar_ci={},
            selected_mode_count=2,
            delta_mix=0.2,
            mode_mass_min=0.05,
            min_mode_c_ray=0.8,
            assignment_stability={"status": "available", "value": 0.9},
            s_span_2=0.92,
            u_span_2=0.01,
            d_span_2=0.4,
            e_res=0.2,
            s_res_1=0.9,
            **one_d_overrides,
        )
    )

    assert classified["strict_gate_label"] is None
    assert classified["primary_label"] == expected_leading_family
    assert classified["label_confidence"] == "candidate"
    assert [candidate["family"] for candidate in classified["candidate_labels"]] == [
        expected_leading_family,
        "multi_mode_directional_geometry",
        "global_2D_directional_subspace",
        "residual_lowD_k",
    ]
    ordered_families = [
        family
        for family, _ in sorted(
            FALLBACK_PRIORITY.items(), key=lambda item: item[1]
        )
    ]
    semantic_names = {
        "oneD_diffuse": "one_dimensional_directional",
        "unresolved_high_dimensional_or_diffuse": "unresolved",
    }
    assert [semantic_names.get(family, family) for family in ordered_families] == [
        "directed_ray",
        "axis_or_antipodal",
        "one_dimensional_directional",
        "multi_mode_directional_geometry",
        "global_2D_directional_subspace",
        "global_kD_directional_subspace",
        "residual_lowD_k",
        "unresolved",
    ]


@pytest.mark.parametrize(
    ("overrides", "expected_family", "expected_k", "candidate_confidence"),
    [
        (
            {
                "s_span_2": 0.92,
                "r_span_pr": None,
                "u_span_2": 0.1,
                "d_span_2": 0.4,
            },
            "global_2D_directional_subspace",
            2,
            "candidate_missing_evidence",
        ),
        (
            {"e_res": 0.2, "s_res_1": 0.9, "r_ctr_pr": 1.0},
            "residual_lowD_k",
            1,
            "candidate_missing_evidence",
        ),
    ],
)
def test_d6_partial_family_evidence_remains_candidate_non_strict(
    overrides: dict[str, Any],
    expected_family: str,
    expected_k: int | None,
    candidate_confidence: str,
) -> None:
    """Preserve partial family evidence as a candidate rather than strict science."""
    # Classify one isolated partial anchor for each preserved fallback family.
    classified = classify_record(_base_classifier_record(**overrides))

    assert classified["primary_label"] == expected_family
    assert classified["strict_gate_label"] is None
    assert classified["label_confidence"] == "candidate"
    assert classified["selected_k"] == expected_k
    candidate = _candidate_for(classified, expected_family)
    assert candidate is not None
    assert candidate["confidence"] == candidate_confidence


def test_d6_isolated_rejected_m2_is_blocked_candidate_not_primary() -> None:
    """Preserve rejected selected M2 evidence without promoting it to primary."""
    # Isolate a partial multimode anchor whose failed mass gate blocks reporting.
    classified = classify_record(
        _base_classifier_record(
            selected_mode_count=2,
            delta_mix=0.2,
            mode_mass_min=0.05,
            min_mode_c_ray=0.8,
            mode_kappa_min=10.0,
            assignment_stability={"status": "available", "value": 0.9},
        )
    )

    assert classified["primary_label"] == "undefined_geometry"
    assert classified["strict_gate_label"] is None
    candidate = _candidate_for(classified, "multi_mode_directional_geometry")
    assert candidate is not None
    assert candidate["confidence"] == "candidate_blocked"
    assert candidate["details"]["reporting_acceptance"] == "rejected"
    assert candidate["details"]["failed_gates"] == ["fitted_mass"]


def test_rank_boundary_is_a_ray_flag_not_a_strict_gate() -> None:
    """Require rank-boundary rays to stay strict while rank remains only a flag."""
    # Place an otherwise accepted directed ray exactly inside the rank boundary band.
    classified = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.5,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )

    assert classified["primary_label"] == "directed_ray"
    assert classified["strict_gate_label"] == "directed_ray"
    assert classified["label_confidence"] == "accepted"
    assert "ray_span_boundary" in classified["secondary_flags"]
    ray_gate = classified["gate_evidence"]["directed_ray"]
    assert "r_span_pr" not in ray_gate.get("blocked_reasons", [])
    ray_candidate = _candidate_for(classified, "directed_ray")
    assert ray_candidate is not None
    assert "r_span_pr" not in ray_candidate["failed_fields"]
    assert "r_span_pr" not in ray_candidate["missing_fields"]


def test_residual_k1_stays_descriptive_without_an_unsupported_flag() -> None:
    """Keep the k=1 fallback descriptive without implying a missing strict family."""
    # Supply only favorable k=1 residual diagnostics to isolate non-strict fallback intent.
    classified = classify_record(
        _base_classifier_record(e_res=0.2, s_res_1=0.9, r_ctr_pr=1.0)
    )

    assert classified["primary_label"] == "residual_lowD_k"
    assert classified["strict_gate_label"] is None
    assert classified["label_confidence"] == "candidate"
    assert classified["selected_k"] == 1
    assert classified["residual_selected_k"] == 1
    assert "residual_k_unsupported" not in classified["secondary_flags"]


def test_strict_residual_selects_the_smallest_passing_supported_k() -> None:
    """Select strict residual k=2 before simultaneously passing k=3 and k=4."""
    # Supply complete stable evidence at every strict residual dimension.
    classified = classify_record(
        _base_classifier_record(
            e_res=0.2,
            s_res_2=0.9,
            s_res_3=0.9,
            s_res_4=0.9,
            r_ctr_pr=3.0,
            centered_residual_subspace_stability={
                "source": "stability_artifact",
                "status": "ok",
                "k": {
                    str(k): {"status": "ok", "residual_angle_p90_k": 10.0}
                    for k in (2, 3, 4)
                },
            },
        )
    )

    assert classified["strict_gate_label"] == "residual_lowD_k"
    assert classified["selected_k"] == 2
    assert classified["residual_selected_k"] == 2
    assert list(classified["gate_evidence"]["residual_lowD_k"]["attempts"]) == [
        "2"
    ]


def test_classifier_v3_strict_priority_and_family_anchors() -> None:
    """Preserve strict family priority, anchors, and candidate overlays."""
    # Exercise strict-family collisions and descriptive candidate overlays.
    ray_with_span_anchor = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            s_span_2=0.92,
            r_span_pr=1.0,
            u_span_2=0.1,
            d_span_2=0.4,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )
    assert ray_with_span_anchor["primary_label"] == "directed_ray"
    assert ray_with_span_anchor["strict_gate_label"] == "directed_ray"
    assert "lowD_candidate_blocked" in ray_with_span_anchor["secondary_flags"]
    assert ray_with_span_anchor["flag_details"]["lowD_candidate_blocked"][
        "span_candidate_k"
    ] == 2
    assert ray_with_span_anchor["flag_details"]["lowD_candidate_blocked"][
        "nearest_alternative"
    ] == "global_2D_directional_subspace"
    assert (
        _candidate_for(ray_with_span_anchor, "global_2D_directional_subspace")
        is not None
    )

    ray_with_residual = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            e_res=0.2,
            s_res_2=0.9,
            r_ctr_pr=1.6,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )
    assert ray_with_residual["primary_label"] == "directed_ray"
    assert "directed_ray_with_lowD_residual" in ray_with_residual[
        "secondary_flags"
    ]
    assert ray_with_residual["flag_details"]["directed_ray_with_lowD_residual"][
        "residual_k"
    ] == 2
    assert ray_with_residual["flag_details"]["directed_ray_with_lowD_residual"][
        "supported"
    ] is True
    assert ray_with_residual["flag_details"]["directed_ray_with_lowD_residual"][
        "stability_status"
    ] == "stable"
    assert _candidate_for(ray_with_residual, "residual_lowD_k") is not None

    ray_with_residual_k1 = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            e_res=0.2,
            s_res_1=0.9,
            r_ctr_pr=1.0,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )
    assert ray_with_residual_k1["primary_label"] == "directed_ray"
    assert _candidate_for(ray_with_residual_k1, "residual_lowD_k") is not None
    assert (
        "directed_ray_with_lowD_residual"
        not in ray_with_residual_k1["secondary_flags"]
    )
    assert "lowD_candidate_blocked" not in ray_with_residual_k1["secondary_flags"]

    ray_with_residual_k1_and_k2 = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            e_res=0.2,
            s_res_1=0.9,
            s_res_2=0.9,
            r_ctr_pr=1.6,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )
    assert ray_with_residual_k1_and_k2["primary_label"] == "directed_ray"
    assert "directed_ray_with_lowD_residual" in ray_with_residual_k1_and_k2[
        "secondary_flags"
    ]
    assert ray_with_residual_k1_and_k2["flag_details"][
        "directed_ray_with_lowD_residual"
    ]["residual_k"] == 2

    ray_anchor_span_strict = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
            scalar_ci={},
        )
    )
    assert (
        ray_anchor_span_strict["primary_label"]
        == "global_2D_directional_subspace"
    )
    assert (
        ray_anchor_span_strict["strict_gate_label"]
        == "global_2D_directional_subspace"
    )
    assert _candidate_for(ray_anchor_span_strict, "directed_ray") is not None

    ray_anchor_residual_strict = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            e_res=0.2,
            s_res_2=0.9,
            r_ctr_pr=1.6,
            scalar_ci={},
        )
    )
    assert ray_anchor_residual_strict["primary_label"] == "residual_lowD_k"
    assert ray_anchor_residual_strict["strict_gate_label"] == "residual_lowD_k"
    assert _candidate_for(ray_anchor_residual_strict, "directed_ray") is not None

    span_missing_stability = classify_record(
        _base_classifier_record(
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
            subspace_stability={"status": "ok", "k": {}},
        )
    )
    assert (
        span_missing_stability["primary_label"]
        == "global_2D_directional_subspace"
    )
    assert "span_stability_missing" in span_missing_stability["secondary_flags"]

    non_directed_magnitude = classify_record(
        _base_classifier_record(
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
            m_cv=1.2,
        )
    )
    assert (
        non_directed_magnitude["primary_label"]
        == "global_2D_directional_subspace"
    )
    assert "magnitude_unstable" in non_directed_magnitude["secondary_flags"]


def test_semantic_classifier_projection_preserves_decisions() -> None:
    """Freeze curated copied metrics and semantic classifier decisions."""
    # Project selected source numerics, decisions, ordered candidates, and salient flags.
    records = [
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        ),
        _base_classifier_record(
            s_span_2=0.92,
            r_span_pr=None,
            u_span_2=0.1,
            d_span_2=0.4,
        ),
        _base_classifier_record(),
    ]
    projection = [
        {
            "numeric": {
                key: classified[key]
                for key in (
                    "feature_id",
                    "n_valid",
                    "zero_filter_frac",
                    "c_ray",
                    "s_span_1",
                )
            },
            "classification": (
                classified["primary_label"],
                classified["strict_gate_label"],
                classified["label_confidence"],
                classified["selected_k"],
            ),
            "candidate_families": [
                candidate["family"] for candidate in classified["candidate_labels"]
            ],
            "secondary_flags": classified["secondary_flags"],
        }
        for classified in (classify_record(record) for record in records)
    ]

    assert projection == [
        {
            "numeric": {
                "feature_id": 1,
                "n_valid": 32,
                "zero_filter_frac": 0.0,
                "c_ray": 0.9,
                "s_span_1": 0.9,
            },
            "classification": ("directed_ray", "directed_ray", "accepted", None),
            "candidate_families": ["directed_ray"],
            "secondary_flags": [],
        },
        {
            "numeric": {
                "feature_id": 1,
                "n_valid": 32,
                "zero_filter_frac": 0.0,
                "c_ray": 0.1,
                "s_span_1": 0.1,
            },
            "classification": ("global_2D_directional_subspace", None, "candidate", 2),
            "candidate_families": ["global_2D_directional_subspace"],
            "secondary_flags": ["lowD_candidate_blocked", "span_selected_k"],
        },
        {
            "numeric": {
                "feature_id": 1,
                "n_valid": 32,
                "zero_filter_frac": 0.0,
                "c_ray": 0.1,
                "s_span_1": 0.1,
            },
            "classification": ("undefined_geometry", None, "undefined", None),
            "candidate_families": [],
            "secondary_flags": ["no_positive_family_evidence"],
        },
    ]


def test_classifier_v3_terminal_high_dimensional_and_global_flags() -> None:
    high_dimensional = classify_record(
        _base_classifier_record(
            s_span_1=0.1,
            s_span_2=0.2,
            s_span_3=0.3,
            s_span_4=0.4,
            s_span_8=0.5,
            e_res=0.0,
            selected_mode_count=1,
            delta_mix=0.0,
        )
    )
    _assert_v3_schema(high_dimensional)
    assert (
        high_dimensional["primary_label"]
        == "unresolved_high_dimensional_or_diffuse"
    )
    assert high_dimensional["label_confidence"] == "candidate"
    assert "positive_highD_evidence" in high_dimensional["secondary_flags"]

    missing_high_d_fields = classify_record(
        _base_classifier_record(s_span_1=0.1, s_span_2=0.2)
    )
    assert missing_high_d_fields["primary_label"] == "undefined_geometry"

    long_tail = classify_record(
        _base_classifier_record(r_span_pr=2.0, r_span_ent=4.0)
    )
    assert long_tail["primary_label"] == "unresolved_high_dimensional_or_diffuse"
    assert "long_tail_spectrum" in long_tail["secondary_flags"]
    assert long_tail["global_flags"] == ["long_tail_spectrum"]

    anchored_long_tail = classify_record(
        _base_classifier_record(
            c_ray=0.90,
            s_span_1=0.90,
            r_span_pr=1.0,
            r_span_ent=2.0,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
        )
    )
    assert anchored_long_tail["primary_label"] == "directed_ray"
    assert "long_tail_spectrum" in anchored_long_tail["secondary_flags"]

    all_missing = classify_record(
        _base_classifier_record(
            c_ray=None,
            s_span_1=None,
            s_span_2=None,
            r_span_pr=None,
            u_span_2=None,
            d_span_2=None,
            b_axis=None,
            scalar_ci={},
        )
    )
    assert all_missing["primary_label"] == "geometry_metrics_unavailable"
    assert all_missing["terminal_reason"] == "all_gates_missing"
    assert all_missing["label_confidence"] == "unavailable"
    assert "all_gates_missing" in all_missing["secondary_flags"]
    assert "no_positive_family_evidence" not in all_missing["secondary_flags"]

    undefined = classify_record(_base_classifier_record())
    assert undefined["primary_label"] == "undefined_geometry"
    assert undefined["terminal_reason"] == "no_positive_family_evidence"
    assert undefined["label_confidence"] == "undefined"

    all_global_flags = classify_record(
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            r_span_pr=1.0,
            r_span_ent=2.0,
            scalar_ci={"c_ray": {"ci_low": 0.86, "ci_high": 0.95}},
            m_cv=1.2,
            sample_size_curves={"status": "unstable"},
            leave_out_sensitivity={"status": "unstable"},
            low_context={"status": "exploratory", "n_valid": 8},
        )
    )
    assert all_global_flags["primary_label"] == "directed_ray"
    assert all_global_flags["global_flags"] == list(GLOBAL_FLAG_ORDER)
    assert all_global_flags["global_flag_count"] == 5
    assert all_global_flags["global_flag_mask"] == "LT|MAG|SS|LO|LOWN"
    assert all_global_flags["flag_details"]["sample_size_unstable"][
        "blocked_fields"
    ] == ["sample_size_curves"]
    assert set(all_global_flags["flag_details"]["sample_size_unstable"]["sources"]) == {
        "directed_ray",
        "global",
    }
    assert all_global_flags["flag_details"]["leave_out_unstable"][
        "blocked_fields"
    ] == ["leave_out_sensitivity"]

    anchored_records = [
        _base_classifier_record(
            c_ray=0.9,
            s_span_1=0.9,
            scalar_ci={"c_ray": {"ci_low": 0.7, "ci_high": 0.95}},
        ),
        _base_classifier_record(s_span_1=0.9, c_ray=0.2, b_axis=0.3),
        _base_classifier_record(s_span_1=0.9, c_ray=0.2, b_axis=0.05),
        _base_classifier_record(selected_mode_count=2, delta_mix=0.2, mode_mass_min=0.05),
        _base_classifier_record(s_span_2=0.92, r_span_pr=None),
        _base_classifier_record(e_res=0.2, s_res_1=0.9),
    ]
    for anchored in anchored_records:
        classified = classify_record(anchored)
        assert classified["primary_label"] != (
            "unresolved_high_dimensional_or_diffuse"
        )


def test_classifier_uses_low_context_per_k_span_stability_as_exploratory() -> None:
    span = classify_record(
        _base_classifier_record(
            n_valid=8,
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
            low_context={
                "status": "exploratory",
                "protocol": "leave_out_sensitivity",
                "n_valid": 8,
            },
            subspace_stability={
                "status": "exploratory",
                "k": {
                    "2": {
                        "status": "exploratory",
                        "subspace_angle_p90_k": None,
                    }
                },
            },
        )
    )

    assert span["primary_label"] == "global_2D_directional_subspace"
    assert span["selected_k"] == 2
    assert span["span_selected_k"] == 2
    assert "exploratory_low_n" in span["secondary_flags"]
    assert (
        span["gate_evidence"]["global_directional_subspace"]["subspace_status"]
        == "exploratory"
    )
    assert (
        span["gate_evidence"]["global_directional_subspace"]["sample_size_status"]
        == "exploratory"
    )


def test_exploratory_low_n_flag_is_limited_to_n_min_boundary() -> None:
    above_boundary = classify_record(
        _base_classifier_record(
            n_valid=12,
            s_span_2=0.92,
            r_span_pr=1.8,
            u_span_2=0.1,
            d_span_2=0.4,
            low_context={
                "status": "exploratory",
                "protocol": "leave_out_sensitivity",
                "n_valid": 12,
            },
            subspace_stability={
                "status": "exploratory",
                "k": {
                    "2": {
                        "status": "exploratory",
                        "subspace_angle_p90_k": None,
                    }
                },
            },
        )
    )

    assert above_boundary["primary_label"] == "global_2D_directional_subspace"
    assert "exploratory_low_n" not in above_boundary["secondary_flags"]
    assert "exploratory_low_n" not in above_boundary["global_flags"]


def _contains_banned_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in BANNED_KEYS for key in value):
            return True
        return any(_contains_banned_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_banned_key(item) for item in value)
    return False


def _gate_decisions_allowed(value: Any) -> bool:
    allowed = {"stable", "exploratory", "unstable", "not_available"}
    if isinstance(value, dict):
        if "decision" in value and value["decision"] not in allowed:
            return False
        return all(_gate_decisions_allowed(item) for item in value.values())
    if isinstance(value, list):
        return all(_gate_decisions_allowed(item) for item in value)
    return True
