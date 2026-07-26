from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_metrics.artifacts import (
    load_geometry_metrics_inputs,
    resolve_final_resid_gram,
)
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    MixtureAuditState,
    PointSelection,
    point_selection_identity,
)
from fega.core.geometry_reporting.schema import LABEL_VERSION
from fega.core.resources import ModelResources
from fega.core.source_fingerprint import (
    canonical_json_digest,
    canonical_source_fingerprint,
    require_canonical_source_fingerprint,
)
from fega.core.vmf.artifacts import (
    VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION,
    build_vmf_scientific_fingerprint,
    feature_ids_from_summary,
    validate_vmf_scores,
    vmf_materialization_policy,
)
from fega.paths import (
    geometry_metrics_scores_path,
    geometry_reporting_records_path,
    stability_scores_path,
    vmf_scores_path,
)

_REQUIRED_STABILITY_SCHEMA_VERSION = 3
POINT_GEOMETRY_ARTIFACT_SCHEMA_VERSION = 1
_POINT_GEOMETRY_ARTIFACT_FIELDS = {
    "phase",
    "schema_version",
    "canonical_source_fingerprint",
    "source_identity",
    "vmf_scientific_fingerprint",
    "standalone_vmf_identity",
    "input_artifact_hashes",
    "threshold_profile",
    "point_selection_contract_version",
    "point_records_sha256",
    "source_paths",
    "features",
}


class StandaloneVmfRegenerationRequiredError(ValueError):
    """Raised when point construction requires a fresh standalone-vMF artifact."""

    def __init__(self, reason: str, path: Path) -> None:
        """Store one precise fail-closed regeneration reason and artifact path."""
        # Keep the reason machine-readable while making the remediation explicit.
        self.reason = str(reason)
        self.path = path
        super().__init__(
            "standalone vMF regeneration required: "
            f"reason={self.reason}; rerun the standalone vMF phase: {path}"
        )


def load_point_geometry_inputs(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> dict[str, Any]:
    """Load and validate pre-stability point inputs without a stability artifact.

    Canonical compute-effect, geometry metrics, and the complete standalone-vMF
    contract are validated before any point record can be built. Invalid vMF
    provenance is a regeneration error, never mixture-rejection evidence.
    """
    # Load the canonical final-residual source and validate its exact Gram bytes.
    source_inputs = load_geometry_metrics_inputs(config, "final_resid", resources)
    resolve_final_resid_gram(source_inputs, resources)
    source_fingerprint = canonical_source_fingerprint(
        source_inputs.manifest, source_inputs.summary
    )
    geometry_path = geometry_metrics_scores_path(config, "final_resid")
    vmf_path = vmf_scores_path(config, "pre_softcap_logits")
    geometry_payload = _load_json_mapping(
        geometry_path, "geometry_metrics final_resid scores"
    )
    require_canonical_source_fingerprint(
        geometry_payload, source_fingerprint, artifact_label="geometry_metrics"
    )

    # Give missing and unversioned standalone artifacts precise regeneration reasons.
    if not vmf_path.exists():
        raise StandaloneVmfRegenerationRequiredError(
            "missing_standalone_vmf_artifact", vmf_path
        )
    try:
        vmf_payload = _load_json_mapping(vmf_path, "standalone vMF scores")
    except (OSError, ValueError) as exc:
        raise StandaloneVmfRegenerationRequiredError(
            f"invalid_standalone_vmf_artifact:{exc}", vmf_path
        ) from exc
    if "schema_version" not in vmf_payload:
        raise StandaloneVmfRegenerationRequiredError(
            "unversioned_standalone_vmf_artifact", vmf_path
        )
    if vmf_payload.get("schema_version") != VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION:
        raise StandaloneVmfRegenerationRequiredError(
            "standalone_vmf_schema_version_mismatch", vmf_path
        )
    try:
        require_canonical_source_fingerprint(
            vmf_payload, source_fingerprint, artifact_label="vMF"
        )
        vmf_cfg = config.phases.vmf
        effective_seed = (
            vmf_cfg.seed if vmf_cfg.seed is not None else config.seed.global_
        )
        feature_ids = feature_ids_from_summary(source_inputs.summary["per_feature"])
        expected_vmf_fingerprint = build_vmf_scientific_fingerprint(
            config=config,
            cfg=vmf_cfg,
            seed=int(effective_seed),
            inputs_manifest=source_inputs.manifest,
            inputs_summary=source_inputs.summary,
            geometry_metrics_scores=geometry_payload,
            feature_ids=feature_ids,
            source_fingerprint=source_fingerprint,
            materialization_policy=vmf_materialization_policy(vmf_cfg),
        )
        validate_vmf_scores(
            vmf_payload, expected_fingerprint=expected_vmf_fingerprint
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StandaloneVmfRegenerationRequiredError(
            f"invalid_standalone_vmf_artifact:{exc}", vmf_path
        ) from exc
    _validate_geometry_feature_inventory(feature_ids, geometry_payload)
    try:
        _validate_vmf_feature_inventory(feature_ids, vmf_payload)
    except ValueError as exc:
        raise StandaloneVmfRegenerationRequiredError(
            f"invalid_standalone_vmf_artifact:{exc}", vmf_path
        ) from exc
    paths = {
        "compute_effect_final_resid_manifest": source_inputs.manifest_path,
        "compute_effect_final_resid": source_inputs.summary_path,
        "geometry_metrics_final_resid": geometry_path,
        "vmf_pre_softcap_logits": vmf_path,
    }
    input_artifact_hashes = {
        "geometry_metrics": {
            "canonical_digest": canonical_json_digest(geometry_payload),
            "file_sha256": _file_sha256(geometry_path),
        },
        "standalone_vmf": {
            "canonical_digest": canonical_json_digest(vmf_payload),
            "file_sha256": _file_sha256(vmf_path),
        },
    }
    recorded_vmf_fingerprint = vmf_payload["fingerprint"]
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "payloads": {
            "compute_effect_final_resid": source_inputs.summary,
            "geometry_metrics_final_resid": geometry_payload,
            "vmf_pre_softcap_logits": vmf_payload,
        },
        "canonical_source_fingerprint": source_fingerprint,
        "source_identity": _build_source_identity(
            source_inputs, source_fingerprint=source_fingerprint
        ),
        "vmf_scientific_fingerprint": recorded_vmf_fingerprint,
        "standalone_vmf_identity": _build_standalone_vmf_identity(
            vmf_payload,
            expected_fingerprint=recorded_vmf_fingerprint,
            feature_ids=feature_ids,
            artifact_identity=input_artifact_hashes["standalone_vmf"],
        ),
        "feature_ids": feature_ids,
        "input_artifact_hashes": input_artifact_hashes,
    }


def load_build_and_write_point_geometry_records(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> dict[str, Any]:
    """Build and persist the reporting-owned point artifact before stability.

    The returned bundle carries the ordered point records and both per-record and
    aggregate hashes consumed by selected-family scheduling and checkpoints.
    """
    # Import lazily to keep artifact loading independent of record construction.
    from fega.core.geometry_reporting.records import build_point_geometry_records

    inputs = load_point_geometry_inputs(config, resources)
    records, summary = build_point_geometry_records(inputs, config)
    payload = {
        "phase": "geometry_point_selection",
        "schema_version": POINT_GEOMETRY_ARTIFACT_SCHEMA_VERSION,
        "canonical_source_fingerprint": inputs["canonical_source_fingerprint"],
        "source_identity": inputs["source_identity"],
        "vmf_scientific_fingerprint": inputs["vmf_scientific_fingerprint"],
        "standalone_vmf_identity": inputs["standalone_vmf_identity"],
        "input_artifact_hashes": inputs["input_artifact_hashes"],
        "threshold_profile": config.phases.geometry_reporting.threshold_profile,
        "point_selection_contract_version": summary[
            "point_selection_contract_version"
        ],
        "point_records_sha256": summary["point_records_sha256"],
        "source_paths": summary["source_paths"],
        "features": records,
    }
    path = point_geometry_records_path(config)
    write_json_atomic(path, payload)
    point_artifact_identity = {
        "canonical_digest": canonical_json_digest(payload),
        "file_sha256": _file_sha256(path),
    }
    return {
        **inputs,
        "point_records_path": str(path),
        "point_artifact_sha256": point_artifact_identity["canonical_digest"],
        "point_artifact_identity": point_artifact_identity,
        "point_records_sha256": summary["point_records_sha256"],
        "point_record_hashes": summary["point_record_hashes"],
        "point_records": records,
    }


def load_point_geometry_records(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> dict[str, Any]:
    """Load the durable point-selection artifact against authoritative live inputs.

    No schema translation, partial inventory reuse, or point-selection inference is
    allowed. The artifact must exactly match the currently validated source, geometry,
    standalone-vMF, threshold, record-hash, and selection contracts.
    """
    # Revalidate every upstream authority before trusting the durable point artifact.
    inputs = load_point_geometry_inputs(config, resources)
    path = point_geometry_records_path(config)
    payload = _load_json_mapping(path, "geometry point records")
    if set(payload) != _POINT_GEOMETRY_ARTIFACT_FIELDS:
        raise ValueError(f"point artifact top-level keys mismatch: {path}")
    if payload.get("phase") != "geometry_point_selection":
        raise ValueError(f"point artifact phase mismatch: {path}")
    if payload.get("schema_version") != POINT_GEOMETRY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"point artifact schema version mismatch: {path}")
    expected_identities = {
        "canonical_source_fingerprint": inputs["canonical_source_fingerprint"],
        "source_identity": inputs["source_identity"],
        "vmf_scientific_fingerprint": inputs["vmf_scientific_fingerprint"],
        "standalone_vmf_identity": inputs["standalone_vmf_identity"],
        "input_artifact_hashes": inputs["input_artifact_hashes"],
        "source_paths": inputs["paths"],
    }
    identity_labels = {
        "canonical_source_fingerprint": "canonical source identity",
        "source_identity": "source identity",
        "vmf_scientific_fingerprint": "standalone vMF fingerprint",
        "standalone_vmf_identity": "standalone vMF identity",
        "input_artifact_hashes": "input artifact identity",
        "source_paths": "source paths",
    }
    for key, expected in expected_identities.items():
        if payload.get(key) != expected:
            label = identity_labels[key]
            raise ValueError(f"point artifact {label} mismatch: {path}")
    if payload.get("threshold_profile") != (
        config.phases.geometry_reporting.threshold_profile
    ):
        raise ValueError(f"point artifact threshold profile mismatch: {path}")
    if payload.get("point_selection_contract_version") != (
        POINT_SELECTION_CONTRACT_VERSION
    ):
        raise ValueError(
            f"point artifact point-selection contract version mismatch: {path}"
        )

    # Validate ordered records, self-hashes, aggregate hash, and complete selections.
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"point artifact features must be a list: {path}")
    expected_feature_ids = [int(value) for value in inputs["feature_ids"]]
    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    selections: dict[int, PointSelection] = {}
    actual_feature_ids: list[int] = []
    for position, raw_record in enumerate(features):
        if not isinstance(raw_record, dict):
            raise ValueError(f"point artifact feature {position} must be a mapping")
        feature_id = _coerce_inventory_feature_id(
            raw_record.get("feature_id"), "point artifact"
        )
        actual_feature_ids.append(feature_id)
        selection = _point_selection_from_identity(
            raw_record.get("point_selection"), feature_id=feature_id
        )
        stored_hash = raw_record.get("point_record_sha256")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise ValueError(
                f"point artifact feature {feature_id} point record hash mismatch"
            )
        unhashed_record = dict(raw_record)
        unhashed_record.pop("point_record_sha256")
        if canonical_json_digest(unhashed_record) != stored_hash:
            raise ValueError(
                f"point artifact feature {feature_id} point record hash mismatch"
            )
        record_hashes[str(feature_id)] = stored_hash
        selections[feature_id] = selection
        records.append(raw_record)
    if actual_feature_ids != expected_feature_ids:
        raise ValueError(f"point artifact feature inventory order mismatch: {path}")
    inventory = [
        {
            "feature_id": feature_id,
            "point_record_sha256": record_hashes[str(feature_id)],
        }
        for feature_id in actual_feature_ids
    ]
    if payload.get("point_records_sha256") != canonical_json_digest(inventory):
        raise ValueError(f"point artifact point record inventory hash mismatch: {path}")
    point_artifact_identity = {
        "canonical_digest": canonical_json_digest(payload),
        "file_sha256": _file_sha256(path),
    }
    return {
        **inputs,
        "point_records_path": str(path),
        "point_artifact_sha256": point_artifact_identity["canonical_digest"],
        "point_artifact_identity": point_artifact_identity,
        "point_records_sha256": payload["point_records_sha256"],
        "point_record_hashes": record_hashes,
        "point_records": records,
        "point_selections": selections,
    }


def point_geometry_records_path(config: FEGAPipelineConfig) -> Path:
    """Return the pre-stability reporting-owned point artifact path."""
    # Co-locate the pre-stability contract with final reporting artifacts.
    return geometry_reporting_records_path(config).with_name(
        "geometry_point_records.json"
    )


def _validate_geometry_feature_inventory(
    feature_ids: list[int], geometry_metrics_payload: dict[str, Any]
) -> None:
    """Require the exact canonical feature inventory in geometry metrics."""
    # Keep geometry provenance errors distinct from standalone-vMF regeneration.
    geometry_per_feature = geometry_metrics_payload.get("per_feature")
    if not isinstance(geometry_per_feature, dict):
        raise ValueError("geometry_metrics per_feature inventory is missing")
    geometry_ids = _ordered_per_feature_ids(geometry_per_feature, "geometry_metrics")
    if geometry_ids != feature_ids:
        raise ValueError("geometry_metrics feature inventory mismatch")


def _validate_vmf_feature_inventory(
    feature_ids: list[int], vmf_payload: dict[str, Any]
) -> None:
    """Require the exact canonical feature inventory in standalone vMF."""
    # A vMF inventory mismatch is a precise fail-closed regeneration condition.
    vmf_features = vmf_payload.get("features")
    if not isinstance(vmf_features, list):
        raise ValueError("standalone vMF feature inventory is missing")
    vmf_ids = [
        _exact_inventory_feature_id(record, "standalone vMF")
        for record in vmf_features
    ]
    if vmf_ids != feature_ids:
        raise ValueError("standalone vMF feature inventory mismatch")


def _build_source_identity(
    source_inputs: Any, *, source_fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Bind canonical source documents, shard bytes, and exact Gram provenance."""
    # Preserve declared paths and file bytes rather than accepting digest placeholders.
    shards = source_inputs.manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("source manifest shards must be a list")
    shard_identities: list[dict[str, str]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"source manifest shards[{index}] must be a mapping")
        raw_path = shard.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"source manifest shards[{index}].path is invalid")
        path = Path(raw_path)
        shard_identities.append(
            {"path": str(path), "file_sha256": _file_sha256(path)}
        )
    shard_identities.sort(key=lambda item: item["path"])
    gram_raw = source_inputs.manifest.get("inputs", {}).get("gram_path")
    if not isinstance(gram_raw, str) or not gram_raw:
        raise ValueError("source manifest inputs.gram_path is required")
    gram_metadata = source_inputs.manifest.get("gram_metadata")
    if not isinstance(gram_metadata, dict) or not gram_metadata:
        raise ValueError("source manifest gram_metadata is required")
    gram_path = Path(gram_raw)
    return {
        "canonical_source": source_fingerprint,
        "manifest": {
            "path": str(source_inputs.manifest_path),
            "canonical_digest": canonical_json_digest(source_inputs.manifest),
            "file_sha256": _file_sha256(source_inputs.manifest_path),
        },
        "summary": {
            "path": str(source_inputs.summary_path),
            "canonical_digest": canonical_json_digest(source_inputs.summary),
            "file_sha256": _file_sha256(source_inputs.summary_path),
        },
        "tensor_shards": shard_identities,
        "gram": {
            "path": str(gram_path),
            "file_sha256": _file_sha256(gram_path),
            "metadata": gram_metadata,
        },
    }


def _build_standalone_vmf_identity(
    payload: dict[str, Any],
    *,
    expected_fingerprint: dict[str, Any],
    feature_ids: list[int],
    artifact_identity: dict[str, str],
) -> dict[str, Any]:
    """Project the fully validated standalone-vMF schedule and artifact authority."""
    # Candidate order and assignment protocol are copied from the validated fingerprint.
    candidate_schedule = expected_fingerprint.get("candidate_mode_counts")
    seed_derivations = expected_fingerprint.get("seed_derivations")
    assignment_metric = expected_fingerprint.get("assignment_metric")
    if not isinstance(candidate_schedule, list) or not candidate_schedule:
        raise ValueError("validated standalone vMF candidate schedule is missing")
    if not isinstance(seed_derivations, dict) or not seed_derivations:
        raise ValueError("validated standalone vMF seed derivations are missing")
    if not isinstance(assignment_metric, dict) or not assignment_metric:
        raise ValueError("validated standalone vMF assignment metric is missing")
    assignment_fraction = expected_fingerprint.get("assignment_fraction")
    assignment_rounds = expected_fingerprint.get("assignment_rounds")
    if not isinstance(assignment_fraction, int | float) or isinstance(
        assignment_fraction, bool
    ):
        raise ValueError("validated standalone vMF assignment fraction is missing")
    if not isinstance(assignment_rounds, int) or isinstance(assignment_rounds, bool):
        raise ValueError("validated standalone vMF assignment rounds are missing")
    return {
        "public_schema_version": payload["schema_version"],
        "scientific_fingerprint": expected_fingerprint,
        "artifact": artifact_identity,
        "feature_ids": list(feature_ids),
        "candidate_schedule": [int(value) for value in candidate_schedule],
        "assignment_schedule": {
            "fraction": float(assignment_fraction),
            "rounds": int(assignment_rounds),
            "seed_derivations": seed_derivations,
            "metric": assignment_metric,
        },
    }


def _point_selection_from_identity(
    value: Any, *, feature_id: int
) -> PointSelection:
    """Validate and reconstruct one complete durable point-selection identity."""
    # Reject incomplete, extended, stale, or type-coerced selection records.
    selection_fields = {
        "family",
        "selected_k",
        "mode",
        "point_reason",
        "mixture_audit_state",
        "contract_version",
    }
    if not isinstance(value, dict) or set(value) != selection_fields:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection fields mismatch"
        )
    audit_value = value.get("mixture_audit_state")
    audit_fields = {
        "status",
        "acceptance",
        "reason",
        "failed_gates",
        "unavailable_gates",
    }
    if not isinstance(audit_value, dict) or set(audit_value) != audit_fields:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection fields mismatch"
        )
    if value.get("contract_version") != POINT_SELECTION_CONTRACT_VERSION:
        raise ValueError(
            "point artifact point-selection contract version mismatch: "
            f"feature {feature_id}"
        )
    selected_k = value.get("selected_k")
    if selected_k is not None and (
        not isinstance(selected_k, int) or isinstance(selected_k, bool)
    ):
        raise ValueError(
            f"point artifact feature {feature_id} point_selection selected_k invalid"
        )
    failed_gates = audit_value.get("failed_gates")
    unavailable_gates = audit_value.get("unavailable_gates")
    if not isinstance(failed_gates, list) or not all(
        isinstance(item, str) for item in failed_gates
    ):
        raise ValueError(
            f"point artifact feature {feature_id} point_selection failed_gates invalid"
        )
    if not isinstance(unavailable_gates, list) or not all(
        isinstance(item, str) for item in unavailable_gates
    ):
        raise ValueError(
            "point artifact feature "
            f"{feature_id} point_selection unavailable_gates invalid"
        )
    if value.get("mode") not in {"strict", "fallback", "terminal"}:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection mode invalid"
        )
    if audit_value.get("status") not in {
        "accepted",
        "rejected",
        "unavailable",
        "not_applicable",
    }:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection audit status invalid"
        )
    for audit_key in ("acceptance", "reason"):
        audit_item = audit_value.get(audit_key)
        if audit_item is not None and not isinstance(audit_item, str):
            raise ValueError(
                "point artifact feature "
                f"{feature_id} point_selection {audit_key} invalid"
            )
    if not isinstance(value.get("family"), str) or not value["family"]:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection family invalid"
        )
    if not isinstance(value.get("point_reason"), str) or not value["point_reason"]:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection reason invalid"
        )
    selection = PointSelection(
        family=value["family"],
        selected_k=selected_k,
        mode=value["mode"],
        point_reason=value["point_reason"],
        mixture_audit_state=MixtureAuditState(
            status=audit_value["status"],
            acceptance=audit_value["acceptance"],
            reason=audit_value["reason"],
            failed_gates=tuple(failed_gates),
            unavailable_gates=tuple(unavailable_gates),
        ),
        contract_version=POINT_SELECTION_CONTRACT_VERSION,
    )
    if point_selection_identity(selection) != value:
        raise ValueError(
            f"point artifact feature {feature_id} point_selection fields mismatch"
        )
    return selection


def _file_sha256(path: Path) -> str:
    """Hash one original JSON artifact exactly as persisted on disk."""
    # Stream the file so point provenance binds bytes as well as canonical JSON.
    digest = hashlib.sha256()
    with open(path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_per_feature_ids(
    per_feature: dict[str, Any], artifact_label: str
) -> list[int]:
    """Return numeric-key-ordered embedded feature identities without duplicates."""
    # Preserve the canonical numeric key order used by upstream summaries.
    result: list[int] = []
    for key in sorted(per_feature, key=lambda value: int(value)):
        record = per_feature[key]
        if not isinstance(record, dict):
            raise ValueError(f"{artifact_label} feature {key} must be a mapping")
        result.append(
            _coerce_inventory_feature_id(record.get("feature_id", key), artifact_label)
        )
    if len(set(result)) != len(result):
        raise ValueError(f"{artifact_label} feature inventory contains duplicates")
    return result


def _exact_inventory_feature_id(record: Any, artifact_label: str) -> int:
    """Read one required feature identity from a list-style artifact record."""
    # List records may not infer identity from position.
    if not isinstance(record, dict) or "feature_id" not in record:
        raise ValueError(f"{artifact_label} feature record missing feature_id")
    return _coerce_inventory_feature_id(record["feature_id"], artifact_label)


def _coerce_inventory_feature_id(value: Any, artifact_label: str) -> int:
    """Coerce one non-boolean exact integer feature identity."""
    # Reject lossy numeric coercions such as booleans and fractional floats.
    if isinstance(value, bool):
        raise ValueError(f"{artifact_label} feature_id must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{artifact_label} feature_id must be an integer") from exc
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{artifact_label} feature_id must be an integer")
    return parsed


def load_geometry_inputs(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> dict[str, Any]:
    """Load the durable point authority and matching selected-family result.

    Reporting reuses the same point loader as scheduling, then verifies that the
    complete WP3 result preserves its ordered inventory, coupled versions,
    provenance, and per-feature lock before any public output is written.
    """
    # Reuse the reporting-owned point validator rather than rebuilding its authority.
    point_bundle = load_point_geometry_records(config, resources)
    stability_path = stability_scores_path(config)
    stability_payload = _load_json_mapping(stability_path, "stability scores")
    _validate_stability_schema(stability_payload, stability_path)
    require_canonical_source_fingerprint(
        stability_payload,
        point_bundle["canonical_source_fingerprint"],
        artifact_label="stability",
    )
    _validate_selected_family_stability(
        stability_payload,
        point_bundle=point_bundle,
        threshold_profile=config.phases.geometry_reporting.threshold_profile,
        path=stability_path,
    )
    return {
        **point_bundle,
        "paths": {**point_bundle["paths"], "stability": str(stability_path)},
        "payloads": {
            **point_bundle["payloads"],
            "stability": stability_payload,
        },
    }


def _validate_selected_family_stability(
    payload: dict[str, Any],
    *,
    point_bundle: dict[str, Any],
    threshold_profile: str,
    path: Path,
) -> None:
    """Verify WP3 provenance and point locks without reinterpreting its science."""
    # Bind only decision-changing versions, identities, order, and immutable locks.
    from fega.core.stability.artifacts import (
        SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION,
        SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
    )
    from fega.core.stability.schedule import SELECTED_FAMILY_SCHEDULE_VERSION

    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"stability fingerprint missing: {path}")
    unhashed_fingerprint = dict(fingerprint)
    stored_digest = unhashed_fingerprint.pop("digest", None)
    if stored_digest != canonical_json_digest(unhashed_fingerprint):
        raise ValueError(f"stability fingerprint digest mismatch: {path}")
    retained_stability_config = payload.get("config")
    if not isinstance(retained_stability_config, dict):
        raise ValueError(f"stability config missing: {path}")
    expected_fingerprint_fields = {
        "fingerprint_version": SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION,
        "checkpoint_schema_version": SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
        "stability_public_schema_version": _REQUIRED_STABILITY_SCHEMA_VERSION,
        "point_artifact_identity": point_bundle["point_artifact_identity"],
        "point_record_hashes": point_bundle["point_record_hashes"],
        "point_records_sha256": point_bundle["point_records_sha256"],
        "label_version": LABEL_VERSION,
        "point_selection_version": POINT_SELECTION_CONTRACT_VERSION,
        "selected_family_schedule_version": SELECTED_FAMILY_SCHEDULE_VERSION,
        "retained_stability_config": retained_stability_config,
        "threshold_profile": threshold_profile,
    }
    for key, expected in expected_fingerprint_fields.items():
        if fingerprint.get(key) != expected:
            raise ValueError(f"stability fingerprint {key} mismatch: {path}")

    # Require exact canonical feature order and compare every persisted lock field.
    expected_ids = [int(value) for value in point_bundle["feature_ids"]]
    effect_spaces = payload.get("effect_spaces")
    final_resid = (
        effect_spaces.get("final_resid") if isinstance(effect_spaces, dict) else None
    )
    per_feature = (
        final_resid.get("per_feature") if isinstance(final_resid, dict) else None
    )
    if not isinstance(per_feature, dict) or list(per_feature) != [
        str(value) for value in expected_ids
    ]:
        raise ValueError(f"stability feature inventory order mismatch: {path}")
    locked_features = fingerprint.get("locked_features")
    if not isinstance(locked_features, list) or len(locked_features) != len(
        expected_ids
    ):
        raise ValueError(f"stability locked feature inventory mismatch: {path}")
    selections = point_bundle["point_selections"]
    hashes = point_bundle["point_record_hashes"]
    for position, feature_id in enumerate(expected_ids):
        record = per_feature[str(feature_id)]
        locked = locked_features[position]
        selection = selections[feature_id]
        if not isinstance(record, dict) or not isinstance(locked, dict):
            raise ValueError(f"stability feature {feature_id} lock is invalid: {path}")
        expected_lock = {
            "feature_id": feature_id,
            "family": selection.family,
            "selection_mode": selection.mode,
            "selected_k": selection.selected_k,
            "point_reason": selection.point_reason,
            "point_record_sha256": hashes[str(feature_id)],
        }
        if any(record.get(key) != value for key, value in expected_lock.items()):
            raise ValueError(
                f"stability feature {feature_id} locked selection mismatch: {path}"
            )
        fingerprint_lock = {
            "feature_id": feature_id,
            "family": selection.family,
            "selection_mode": selection.mode,
            "selected_k": selection.selected_k,
            "schedule_digest": record.get("schedule_digest"),
        }
        if locked != fingerprint_lock:
            raise ValueError(
                f"stability feature {feature_id} fingerprint lock mismatch: {path}"
            )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)
    return path


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        with open(path) as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_stability_schema(payload: dict[str, Any], path: Path) -> None:
    if payload.get("phase") != "stability":
        raise ValueError(f"stability artifact phase mismatch: {path}")
    schema_version = payload.get("schema_version")
    if schema_version != _REQUIRED_STABILITY_SCHEMA_VERSION:
        raise ValueError(
            "stability schema version "
            f"{_REQUIRED_STABILITY_SCHEMA_VERSION} required for geometry_reporting; "
            f"got {schema_version!r}: {path}"
        )
    if not isinstance(payload.get("effect_spaces"), dict):
        raise ValueError(f"stability schema missing effect_spaces mapping: {path}")
