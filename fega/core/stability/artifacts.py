from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.common import require_single_entity_attr
from fega.core.geometry_metrics.artifacts import (
    GeometryMetricsInputs,
    load_geometry_metrics_inputs,
    resolve_final_resid_gram,
)
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    PointSelection,
    point_selection_identity,
)
from fega.core.geometry_reporting.schema import LABEL_VERSION
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.stability.schedule import (
    SELECTED_FAMILY_SCHEDULE_VERSION,
    SelectedFamilySchedule,
)
from fega.core.utils import ChunkProcessor
from fega.paths import (
    data_prep_pairs_path,
    data_prep_select_dir,
    stability_checkpoint_path,
    stability_scores_path,
)

SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION = 4
SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION = 1
STABILITY_PUBLIC_SCHEMA_VERSION = 3
_PROTOCOL_COUNTER_KEYS = {
    "requested",
    "valid",
    "failed",
    "non_applicable",
    "skipped",
}
_CHECKPOINT_RECORD_KEYS = {
    "feature_id",
    "family",
    "selection_mode",
    "selected_k",
    "point_reason",
    "schedule_digest",
    "point_record_sha256",
    "n_valid",
    "selected_family_evidence",
}


def scientific_stability_config(config: FEGAPipelineConfig) -> dict[str, Any]:
    """Return only stability controls that can change scientific results."""
    # Worker count, resume mode, and flush cadence are execution-only controls.
    result = dict(config.to_dict()["phases"]["stability"])
    for key in ("workers", "resume", "checkpoint_flush_features"):
        result.pop(key, None)
    return result


@dataclass(frozen=True)
class SelectedFamilyCheckpointLoadResult:
    """Fail-closed result of loading a selected-family construction checkpoint."""

    status: Literal["missing", "reused", "rejected"]
    rejection_reason: str | None
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StabilityGroupLookup:
    """Group metadata keyed by complete prompt identity or canonical context index."""

    context_labels: dict[int, str | None]
    pair_labels: dict[tuple[str, str, int], str | None] | None
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class StabilityInputs:
    effect_space: str
    geometry_metrics_inputs: GeometryMetricsInputs
    gram: torch.Tensor
    group_lookup: StabilityGroupLookup


@dataclass(frozen=True)
class StabilityFeatureBlock:
    feature_id: int
    source_summary: dict[str, Any]
    tensor_shard: str | None
    rows: torch.Tensor | None
    context_indices: list[int]
    pair_indices: list[int]
    attribute_labels: list[str]
    pair_roles: list[str]
    group_labels: list[str | None] | None
    skipped_reason: str | None


@dataclass(frozen=True)
class StabilityFeatureDescriptor:
    feature_id: int
    feature_key: str
    source_summary: dict[str, Any]
    tensor_shard: str | None
    skipped_reason: str | None


def load_stability_inputs(
    config: FEGAPipelineConfig,
    effect_space: str,
    resources: Any | None = None,
) -> StabilityInputs:
    """Load compute-effect artifacts plus stability-only row metadata."""
    # Stability has one scientific source and rejects diagnostic readout spaces.
    if effect_space != "final_resid":
        raise ValueError("stability supports only effect_space='final_resid'.")
    geometry_metrics_inputs = load_geometry_metrics_inputs(config, effect_space, resources)
    gram = resolve_final_resid_gram(geometry_metrics_inputs, resources)
    return StabilityInputs(
        effect_space=effect_space,
        geometry_metrics_inputs=geometry_metrics_inputs,
        gram=gram,
        group_lookup=_load_group_lookup(config, geometry_metrics_inputs),
    )


def iter_stability_feature_blocks(
    inputs: StabilityInputs,
    *,
    exclude_feature_ids: set[int] | None = None,
) -> Iterator[StabilityFeatureBlock]:
    """Yield feature row blocks with context/pair identity preserved."""
    shard_cache: dict[str, dict[str, Any]] = {}
    for descriptor in iter_stability_feature_descriptors(
        inputs, exclude_feature_ids=exclude_feature_ids
    ):
        yield load_stability_feature_block(inputs, descriptor, shard_cache=shard_cache)


def iter_stability_feature_descriptors(
    inputs: StabilityInputs,
    *,
    exclude_feature_ids: set[int] | None = None,
) -> Iterator[StabilityFeatureDescriptor]:
    """Yield compact feature descriptors without loading tensor rows."""
    base = inputs.geometry_metrics_inputs
    per_feature = base.summary.get("per_feature")
    if not isinstance(per_feature, dict):
        raise ValueError(f"effect_summary missing `per_feature`: {base.summary_path}")
    excluded = exclude_feature_ids or set()

    for feature_key in sorted(per_feature, key=lambda key: int(key)):
        record = per_feature[feature_key]
        if not isinstance(record, dict):
            raise ValueError(f"Feature {feature_key} summary must be a mapping.")
        feature_id = int(record.get("feature_id", feature_key))
        if feature_id in excluded:
            continue
        shard_name = record.get("tensor_shard")
        if shard_name is None:
            yield StabilityFeatureDescriptor(
                feature_id=feature_id,
                feature_key=str(feature_key),
                source_summary=record,
                tensor_shard=None,
                skipped_reason=record.get("skipped_reason") or "missing_tensor_shard",
            )
            continue

        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError(f"Feature {feature_key} has invalid tensor_shard.")
        yield StabilityFeatureDescriptor(
            feature_id=feature_id,
            feature_key=str(feature_key),
            source_summary=record,
            tensor_shard=shard_name,
            skipped_reason=None,
        )


def load_stability_feature_block(
    inputs: StabilityInputs,
    descriptor: StabilityFeatureDescriptor,
    *,
    shard_cache: dict[str, dict[str, Any]] | None = None,
) -> StabilityFeatureBlock:
    """Load one feature block from a compact descriptor."""
    # Slice directions and complete prompt identity from the same canonical shard rows.
    record = descriptor.source_summary
    if descriptor.tensor_shard is None:
        return StabilityFeatureBlock(
            feature_id=descriptor.feature_id,
            source_summary=record,
            tensor_shard=None,
            rows=None,
            context_indices=[],
            pair_indices=[],
            attribute_labels=[],
            pair_roles=[],
            group_labels=None,
            skipped_reason=descriptor.skipped_reason or "missing_tensor_shard",
        )

    value_key = "direction"
    cache = shard_cache if shard_cache is not None else {}
    current_payload = cache.get(descriptor.tensor_shard)
    if current_payload is None:
        current_payload = _load_shard(
            inputs.geometry_metrics_inputs.artifact_dir / descriptor.tensor_shard,
            value_key,
        )
        cache[descriptor.tensor_shard] = current_payload

    row_start = _required_int(
        record.get("row_start"), f"{descriptor.feature_key}.row_start"
    )
    row_end = _required_int(record.get("row_end"), f"{descriptor.feature_key}.row_end")
    if row_end < row_start:
        raise ValueError(f"Feature {descriptor.feature_key} has invalid row range.")

    values = current_payload[value_key]
    if row_end > int(values.shape[0]):
        raise ValueError(
            f"Feature {descriptor.feature_key} row range exceeds shard rows "
            f"in {descriptor.tensor_shard}."
        )
    rows = values[row_start:row_end].detach().cpu()
    if rows.dtype != torch.float32:
        rows = rows.to(dtype=torch.float32)
    context_indices = _slice_int_tensor(
        current_payload, "context_indices", row_start, row_end
    )
    pair_indices = _slice_int_tensor(current_payload, "pair_indices", row_start, row_end)
    attribute_labels = _slice_string_list(
        current_payload, "attribute_labels", row_start, row_end
    )
    pair_roles = _slice_string_list(current_payload, "pair_roles", row_start, row_end)
    return StabilityFeatureBlock(
        feature_id=descriptor.feature_id,
        source_summary=record,
        tensor_shard=descriptor.tensor_shard,
        rows=rows,
        context_indices=context_indices,
        pair_indices=pair_indices,
        attribute_labels=attribute_labels,
        pair_roles=pair_roles,
        group_labels=_group_labels_for_pairs(
            context_indices,
            pair_indices,
            attribute_labels,
            pair_roles,
            inputs.group_lookup,
        ),
        skipped_reason=None,
    )


def source_paths(inputs: StabilityInputs) -> dict[str, str]:
    base = inputs.geometry_metrics_inputs
    paths = {
        "manifest": str(base.manifest_path),
        "summary": str(base.summary_path),
    }
    if inputs.gram is not None:
        gram_raw = base.manifest.get("inputs", {}).get("gram_path")
        if gram_raw:
            paths["gram"] = str(gram_raw)
    if inputs.group_lookup.source_paths:
        paths["group_metadata"] = ",".join(inputs.group_lookup.source_paths)
    return paths


def write_stability_scores(config: FEGAPipelineConfig, payload: dict[str, Any]) -> Path:
    path = stability_scores_path(config)
    _write_json_atomic(path, payload)
    return path


def load_stability_checkpoint(config: FEGAPipelineConfig) -> dict[str, Any] | None:
    path = stability_checkpoint_path(config)
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"stability checkpoint must be a JSON object: {path}")
    return payload


def write_stability_checkpoint(
    config: FEGAPipelineConfig, payload: dict[str, Any]
) -> None:
    _write_json_atomic(stability_checkpoint_path(config), payload)


def delete_stability_checkpoint(config: FEGAPipelineConfig) -> None:
    try:
        stability_checkpoint_path(config).unlink()
    except FileNotFoundError:
        return


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _load_shard(path: Path, value_key: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Shard must contain a tensor mapping: {path}")
    if value_key not in payload:
        raise ValueError(f"Shard {path} missing required `{value_key}` tensor.")
    values = payload[value_key]
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"Shard {path} `{value_key}` must be a rank-2 tensor.")
    return payload


def _slice_int_tensor(
    payload: dict[str, Any], key: str, row_start: int, row_end: int
) -> list[int]:
    values = payload.get(key)
    if not isinstance(values, torch.Tensor):
        return []
    if row_end > int(values.shape[0]):
        raise ValueError(f"Shard metadata `{key}` is shorter than value rows.")
    return [int(v) for v in values[row_start:row_end].detach().cpu().tolist()]


def _group_labels_for_pairs(
    context_indices: list[int],
    pair_indices: list[int],
    attribute_labels: list[str],
    pair_roles: list[str],
    lookup: StabilityGroupLookup,
) -> list[str | None] | None:
    """Resolve groups without allowing local pair-index collisions across prompt families."""
    # Prefer the complete prompt identity; context metadata remains an independent fallback.
    if not context_indices and not pair_indices:
        return None
    labels: list[str | None] = []
    row_count = max(len(context_indices), len(pair_indices))
    for pos in range(row_count):
        row = context_indices[pos] if pos < len(context_indices) else -1
        pair_idx = pair_indices[pos] if pos < len(pair_indices) else -1
        label = None
        if (
            lookup.pair_labels is not None
            and pos < len(attribute_labels)
            and pos < len(pair_roles)
        ):
            label = lookup.pair_labels.get(
                (attribute_labels[pos], pair_roles[pos], pair_idx)
            )
        if label is None:
            label = lookup.context_labels.get(row)
        labels.append(label)
    return labels if any(label is not None for label in labels) else None


def _load_group_lookup(
    config: FEGAPipelineConfig, inputs: GeometryMetricsInputs
) -> StabilityGroupLookup:
    manifest_inputs = inputs.manifest.get("inputs") or {}
    context_labels: dict[int, str | None] = {}
    pair_labels: dict[tuple[str, str, int], str | None] | None = None
    sources: list[str] = []

    contexts_path = _metadata_path(
        manifest_inputs.get("contexts_path"),
        default=_default_contexts_path(config),
    )
    if contexts_path is not None:
        labels = _load_feature_context_labels(contexts_path)
        if labels:
            context_labels.update(labels)
        sources.append(str(contexts_path))

    pairs_path = _metadata_path(
        manifest_inputs.get("pairs_path"),
        default=_default_pairs_path(config),
    )
    if pairs_path is not None:
        pair_labels = _load_pair_group_lookup(pairs_path)
        sources.append(str(pairs_path))

    activation_labels = _load_activation_group_labels(
        manifest_inputs.get("manifest_path"), manifest_inputs.get("activations_dir")
    )
    if activation_labels:
        for idx, label in activation_labels.items():
            context_labels.setdefault(idx, label)
    manifest_path = _metadata_path(manifest_inputs.get("manifest_path"), default=None)
    activations_dir = _metadata_path(
        manifest_inputs.get("activations_dir"), default=None
    )
    if manifest_path is not None and activations_dir is not None:
        sources.append(str(manifest_path))

    return StabilityGroupLookup(
        context_labels=context_labels,
        pair_labels=pair_labels,
        source_paths=tuple(dict.fromkeys(sources)),
    )


def _metadata_path(raw: Any, *, default: Path | None) -> Path | None:
    if raw is not None:
        return Path(raw)
    if default is not None and default.exists():
        return default
    return None


def _default_contexts_path(config: FEGAPipelineConfig) -> Path | None:
    try:
        entity, attr = require_single_entity_attr(config)
    except ValueError:
        return None
    return data_prep_select_dir(config, entity, attr) / "feature_contexts.json"


def _default_pairs_path(config: FEGAPipelineConfig) -> Path | None:
    try:
        entity, attr = require_single_entity_attr(config)
    except ValueError:
        return None
    return data_prep_pairs_path(config, entity, attr)


def _load_feature_context_labels(path: Path) -> dict[int, str | None]:
    """Load optional context groups while rejecting corrupt present metadata."""
    # A selected path is present, so read or JSON failures must remain visible.
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid group metadata at {path}: {exc}") from exc
    labels: dict[int, str | None] = {}
    for record in _flatten_prompt_records(payload):
        if not isinstance(record, dict):
            continue
        idx = record.get("index")
        if idx is None:
            continue
        label = _extract_group_label(record)
        if label is not None:
            labels[int(idx)] = label
    return labels


def _load_pair_group_lookup(
    path: Path,
) -> dict[tuple[str, str, int], str | None] | None:
    """Load group labels using the complete persisted prompt identity."""
    # Flatten only after preserving the enclosing attribute and list-role names.
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid group metadata at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Group metadata must be a JSON object: {path}")
    labels: dict[tuple[str, str, int], str | None] = {}
    for attribute_label, role_blocks in payload.items():
        if not isinstance(role_blocks, dict):
            continue
        for raw_role, prompts in role_blocks.items():
            if not isinstance(prompts, list):
                continue
            pair_role = str(raw_role).removesuffix("_prompts")
            for pair_index, prompt in enumerate(prompts):
                if isinstance(prompt, dict):
                    labels[(str(attribute_label), pair_role, pair_index)] = (
                        _extract_group_label(prompt)
                    )
    return labels if any(label is not None for label in labels.values()) else None


def _slice_string_list(
    payload: dict[str, Any], key: str, row_start: int, row_end: int
) -> list[str]:
    """Slice required string row metadata from a tensor shard."""
    # Treat absent metadata as unavailable so only group protocols are disabled.
    values = payload.get(key)
    if not isinstance(values, list) or row_end > len(values):
        return []
    return [str(value) for value in values[row_start:row_end]]


def _load_activation_group_labels(
    manifest_raw: Any, activations_dir_raw: Any
) -> dict[int, str | None]:
    if manifest_raw is None and activations_dir_raw is None:
        return {}
    if not manifest_raw or not activations_dir_raw:
        declared_path = manifest_raw or activations_dir_raw
        raise ValueError(
            "Incomplete activation group metadata declaration at "
            f"{declared_path}: both manifest_path and activations_dir are required."
        )
    manifest_path = Path(manifest_raw)
    activations_dir = Path(activations_dir_raw)
    if not manifest_path.exists():
        raise ValueError(
            f"Invalid group metadata at {manifest_path}: path does not exist."
        )
    if not activations_dir.exists():
        raise ValueError(
            f"Invalid group metadata at {activations_dir}: path does not exist."
        )
    labels: dict[int, str | None] = {}
    try:
        for _tensor_path, meta_path in ChunkProcessor.stream(
            manifest_path, activations_dir
        ):
            try:
                with open(meta_path) as f:
                    for line in f:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            continue
                        idx = record.get("index")
                        if idx is None:
                            continue
                        label = _extract_group_label(record)
                        if label is not None:
                            labels[int(idx)] = label
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid group metadata at {meta_path}: {exc}"
                ) from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid group metadata from {manifest_path}: {exc}"
        ) from exc
    return labels


def _extract_group_label(record: dict[str, Any]) -> str | None:
    """Return a group identity containing both metadata dimension and value."""
    # Prevent equal values from different grouping dimensions from collapsing.
    for key in (
        "context_split",
        "entity_split",
        "group",
        "split",
        "pair_role",
        "attribute_label",
        "entity_label",
        "attribute_type",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return f"{key}={value}"
    return None


def _flatten_prompt_records(payload: Any) -> list[Any]:
    records: list[Any] = []
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return records
    for value in payload.values():
        if isinstance(value, list):
            records.extend(value)
        elif isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, list):
                    records.extend(nested)
    return records


def _load_json(
    path: Path, resources: Any | None, *, label: str
) -> dict[str, Any]:
    if resources is not None:
        cached = resources.get_cached_json(path)
        if cached is not None:
            return cached
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    if resources is not None:
        resources.cache_json(path, payload)
    return payload


def _required_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"`{label}` must be an integer, got {value!r}.")
    return value


def build_selected_family_checkpoint_fingerprint(
    *,
    config: FEGAPipelineConfig,
    point_bundle: Mapping[str, Any],
    schedules: Sequence[SelectedFamilySchedule],
) -> dict[str, Any]:
    """Build checkpoint provenance from validated artifacts and live authorities.

    Callers provide no free-form identity fragments. Source, geometry, point, and vMF
    identities must come from ``load_point_geometry_records``; threshold, label, and
    retained stability controls are derived here from the production configuration.
    """
    # Validate the selected-family lock before reading any durable provenance.
    ordered_schedules = sorted(schedules, key=lambda item: int(item.feature_id))
    feature_ids = [int(schedule.feature_id) for schedule in ordered_schedules]
    if not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError(
            "selected-family schedule feature inventory is empty or duplicate"
        )
    if any(
        schedule.point_selection.contract_version
        != POINT_SELECTION_CONTRACT_VERSION
        for schedule in ordered_schedules
    ):
        raise ValueError("selected-family schedule point-selection version mismatch")
    if any(
        schedule.version != SELECTED_FAMILY_SCHEDULE_VERSION
        for schedule in ordered_schedules
    ):
        raise ValueError("selected-family schedule version mismatch")
    if any(
        schedule.family != schedule.point_selection.family
        or schedule.selection_mode != schedule.point_selection.mode
        or schedule.reported_selected_k != schedule.point_selection.selected_k
        for schedule in ordered_schedules
    ):
        raise ValueError("selected-family schedule lock mismatch")

    # Accept only the complete authoritative point-loader identity contract.
    provenance = _validated_point_bundle_identity(
        point_bundle, schedules=ordered_schedules, feature_ids=feature_ids
    )
    retained_config = scientific_stability_config(config)
    if not retained_config:
        raise ValueError("retained stability config identity is missing")
    threshold_profile = config.phases.geometry_reporting.threshold_profile
    if not isinstance(threshold_profile, str) or not threshold_profile:
        raise ValueError("threshold profile identity is missing")
    if not isinstance(LABEL_VERSION, str) or not LABEL_VERSION:
        raise ValueError("label version identity is missing")
    identity = {
        "fingerprint_version": SELECTED_FAMILY_CHECKPOINT_FINGERPRINT_VERSION,
        "checkpoint_schema_version": SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
        "stability_public_schema_version": STABILITY_PUBLIC_SCHEMA_VERSION,
        "source_identity": provenance["source_identity"],
        "geometry_metrics_identity": provenance["geometry_metrics_identity"],
        "point_artifact_identity": provenance["point_artifact_identity"],
        "point_record_hashes": provenance["point_record_hashes"],
        "point_records_sha256": provenance["point_records_sha256"],
        "standalone_vmf_identity": provenance["standalone_vmf_identity"],
        "threshold_profile": threshold_profile,
        "label_version": LABEL_VERSION,
        "point_selection_version": POINT_SELECTION_CONTRACT_VERSION,
        "selected_family_schedule_version": SELECTED_FAMILY_SCHEDULE_VERSION,
        "retained_stability_config": retained_config,
        "locked_features": [
            {
                "feature_id": int(schedule.feature_id),
                "family": schedule.family,
                "selection_mode": schedule.selection_mode,
                "selected_k": schedule.reported_selected_k,
                "schedule_digest": schedule.schedule_digest,
            }
            for schedule in ordered_schedules
        ],
    }
    return {**identity, "digest": canonical_json_digest(identity)}


def _validated_point_bundle_identity(
    point_bundle: Mapping[str, Any],
    *,
    schedules: Sequence[SelectedFamilySchedule],
    feature_ids: list[int],
) -> dict[str, Any]:
    """Validate the exact provenance projection returned by the point loader."""
    # Reject placeholder mappings and cross-component schedule or inventory drift.
    source = _required_identity_mapping(
        point_bundle.get("source_identity"),
        {
            "canonical_source",
            "manifest",
            "summary",
            "tensor_shards",
            "gram",
        },
        "source identity",
    )
    canonical_source = _required_identity_mapping(
        source["canonical_source"],
        {"schema_version", "algorithm", "digest", "components"},
        "canonical source identity",
    )
    _require_nonempty_string_fields(
        canonical_source, ("algorithm", "digest"), "canonical source identity"
    )
    _required_nonempty_mapping(
        canonical_source["components"], "canonical source component identity"
    )
    for key in ("manifest", "summary"):
        document = _required_identity_mapping(
            source[key],
            {"path", "canonical_digest", "file_sha256"},
            f"source {key} identity",
        )
        _require_nonempty_string_fields(
            document,
            ("path", "canonical_digest", "file_sha256"),
            f"source {key} identity",
        )
    shards = source["tensor_shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("source tensor shard identity inventory is empty or malformed")
    for index, shard_value in enumerate(shards):
        shard = _required_identity_mapping(
            shard_value,
            {"path", "file_sha256"},
            f"source tensor shard identity {index}",
        )
        _require_nonempty_string_fields(
            shard, ("path", "file_sha256"), f"source tensor shard identity {index}"
        )
    gram = _required_identity_mapping(
        source["gram"], {"path", "file_sha256", "metadata"}, "source Gram identity"
    )
    _require_nonempty_string_fields(
        gram, ("path", "file_sha256"), "source Gram identity"
    )
    _required_nonempty_mapping(gram["metadata"], "source Gram metadata identity")

    input_hashes = _required_identity_mapping(
        point_bundle.get("input_artifact_hashes"),
        {"geometry_metrics", "standalone_vmf"},
        "input artifact identity",
    )
    for key in ("geometry_metrics", "standalone_vmf"):
        artifact = _required_identity_mapping(
            input_hashes[key],
            {"canonical_digest", "file_sha256"},
            f"{key} artifact identity",
        )
        _require_nonempty_string_fields(
            artifact, ("canonical_digest", "file_sha256"), f"{key} artifact identity"
        )
    point_artifact = _required_identity_mapping(
        point_bundle.get("point_artifact_identity"),
        {"canonical_digest", "file_sha256"},
        "point artifact identity",
    )
    _require_nonempty_string_fields(
        point_artifact, ("canonical_digest", "file_sha256"), "point artifact identity"
    )
    record_hashes, records_digest = _validated_durable_point_authority(
        point_bundle,
        schedules=schedules,
        feature_ids=feature_ids,
    )
    bundle_feature_ids = point_bundle.get("feature_ids")
    if bundle_feature_ids != feature_ids:
        raise ValueError("point bundle feature inventory mismatch")

    vmf = _required_identity_mapping(
        point_bundle.get("standalone_vmf_identity"),
        {
            "public_schema_version",
            "scientific_fingerprint",
            "artifact",
            "feature_ids",
            "candidate_schedule",
            "assignment_schedule",
        },
        "standalone vMF identity",
    )
    if not isinstance(vmf["public_schema_version"], int) or isinstance(
        vmf["public_schema_version"], bool
    ):
        raise ValueError("standalone vMF public schema identity is malformed")
    scientific = _required_nonempty_mapping(
        vmf["scientific_fingerprint"], "standalone vMF scientific fingerprint"
    )
    required_scientific_fields = {
        "schema_version",
        "candidate_mode_counts",
        "assignment_fraction",
        "assignment_rounds",
        "seed_derivations",
        "assignment_metric",
        "feature_ids",
        "feature_inventory_sha256",
    }
    if not required_scientific_fields.issubset(scientific):
        raise ValueError("standalone vMF scientific fingerprint fields mismatch")
    vmf_artifact = _required_identity_mapping(
        vmf["artifact"],
        {"canonical_digest", "file_sha256"},
        "standalone vMF artifact identity",
    )
    if dict(vmf_artifact) != dict(input_hashes["standalone_vmf"]):
        raise ValueError("standalone vMF artifact identity mismatch")
    if (
        vmf["feature_ids"] != feature_ids
        or scientific.get("feature_ids") != feature_ids
    ):
        raise ValueError("standalone vMF feature inventory mismatch")
    candidates = vmf["candidate_schedule"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in candidates
        )
    ):
        raise ValueError("standalone vMF candidate schedule identity is malformed")
    if candidates != scientific.get("candidate_mode_counts"):
        raise ValueError("standalone vMF candidate schedule identity mismatch")
    assignment = _required_identity_mapping(
        vmf["assignment_schedule"],
        {"fraction", "rounds", "seed_derivations", "metric"},
        "standalone vMF assignment schedule identity",
    )
    _required_nonempty_mapping(
        assignment["seed_derivations"], "standalone vMF seed derivation identity"
    )
    _required_nonempty_mapping(
        assignment["metric"], "standalone vMF assignment metric identity"
    )
    if not isinstance(assignment["fraction"], int | float) or isinstance(
        assignment["fraction"], bool
    ):
        raise ValueError("standalone vMF assignment fraction identity is malformed")
    if not isinstance(assignment["rounds"], int) or isinstance(
        assignment["rounds"], bool
    ):
        raise ValueError("standalone vMF assignment rounds identity is malformed")
    expected_assignment = {
        "fraction": scientific.get("assignment_fraction"),
        "rounds": scientific.get("assignment_rounds"),
        "seed_derivations": scientific.get("seed_derivations"),
        "metric": scientific.get("assignment_metric"),
    }
    if dict(assignment) != expected_assignment:
        raise ValueError("standalone vMF assignment schedule identity mismatch")
    return {
        "source_identity": dict(source),
        "geometry_metrics_identity": dict(input_hashes["geometry_metrics"]),
        "point_artifact_identity": dict(point_artifact),
        "point_record_hashes": dict(record_hashes),
        "point_records_sha256": records_digest,
        "standalone_vmf_identity": dict(vmf),
    }


def _validated_durable_point_authority(
    point_bundle: Mapping[str, Any],
    *,
    schedules: Sequence[SelectedFamilySchedule],
    feature_ids: list[int],
) -> tuple[dict[str, str], str]:
    """Revalidate durable point records and reconstructed selections per schedule."""
    # Loader-returned records and objects jointly own the complete point-selection lock.
    records = point_bundle.get("point_records")
    if (
        not isinstance(records, list)
        or not records
        or len(records) != len(feature_ids)
    ):
        raise ValueError("point record authority inventory mismatch")
    selections = point_bundle.get("point_selections")
    if (
        not isinstance(selections, Mapping)
        or not selections
        or list(selections) != feature_ids
    ):
        raise ValueError("point selection authority inventory mismatch")

    authoritative_hashes: dict[str, str] = {}
    inventory: list[dict[str, Any]] = []
    for feature_id, schedule, record_value in zip(
        feature_ids, schedules, records, strict=True
    ):
        if not isinstance(record_value, Mapping) or not record_value:
            raise ValueError(f"point record authority {feature_id} is malformed")
        record = dict(record_value)
        record_feature_id = record.get("feature_id")
        if (
            not isinstance(record_feature_id, int)
            or isinstance(record_feature_id, bool)
            or record_feature_id != feature_id
        ):
            raise ValueError("point record authority inventory mismatch")
        authoritative_selection = selections[feature_id]
        if not isinstance(authoritative_selection, PointSelection):
            raise ValueError(f"point selection authority {feature_id} is malformed")
        authoritative_identity = point_selection_identity(authoritative_selection)
        record_selection = record.get("point_selection")
        if not isinstance(record_selection, Mapping) or (
            dict(record_selection) != authoritative_identity
        ):
            raise ValueError("point record selection identity mismatch")
        if (
            schedule.point_selection != authoritative_selection
            or point_selection_identity(schedule.point_selection)
            != authoritative_identity
        ):
            raise ValueError("selected-family schedule point selection mismatch")

        stored_hash = record.get("point_record_sha256")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise ValueError("point record hash inventory mismatch")
        unhashed_record = dict(record)
        unhashed_record.pop("point_record_sha256")
        try:
            observed_hash = canonical_json_digest(unhashed_record)
        except (TypeError, ValueError) as error:
            raise ValueError("point record hash inventory mismatch") from error
        if observed_hash != stored_hash or schedule.point_record_sha256 != stored_hash:
            raise ValueError("point record hash inventory mismatch")
        authoritative_hashes[str(feature_id)] = stored_hash
        inventory.append(
            {
                "feature_id": feature_id,
                "point_record_sha256": stored_hash,
            }
        )

    provided_hashes = point_bundle.get("point_record_hashes")
    if not isinstance(provided_hashes, Mapping) or (
        dict(provided_hashes) != authoritative_hashes
    ):
        raise ValueError("point record hash inventory mismatch")
    records_digest = point_bundle.get("point_records_sha256")
    if not isinstance(records_digest, str) or not records_digest:
        raise ValueError("point record inventory hash is missing")
    if records_digest != canonical_json_digest(inventory):
        raise ValueError("point record inventory hash mismatch")
    return authoritative_hashes, records_digest


def _required_identity_mapping(
    value: Any, required_fields: set[str], label: str
) -> Mapping[str, Any]:
    """Require one nonempty exact-key identity mapping."""
    # Exact fields prevent silent provenance extensions from escaping the digest review.
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is empty or malformed")
    if set(value) != required_fields:
        raise ValueError(f"{label} fields mismatch")
    return value


def _required_nonempty_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Require one mapping whose authority is not an empty placeholder."""
    # Placeholder dictionaries do not identify any artifact, schedule, or runtime.
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} is empty or malformed")
    return value


def _require_nonempty_string_fields(
    value: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    """Require exact persisted paths and hashes to be present and nonempty."""
    # Reject ``None`` and empty strings before canonical fingerprinting.
    if any(
        not isinstance(value.get(field), str) or not value[field]
        for field in fields
    ):
        raise ValueError(f"{label} contains an empty or malformed path/hash")


def build_selected_family_checkpoint_payload(
    *,
    fingerprint: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    schedules: Sequence[SelectedFamilySchedule],
) -> dict[str, Any]:
    """Build one ordered checkpoint payload without translating record content."""
    # Require each completed record to bind the current selected-family schedule.
    schedule_by_id = _schedule_by_feature_id(schedules)
    ordered_records = sorted(records, key=lambda item: int(item["feature_id"]))
    seen: set[int] = set()
    features: list[dict[str, Any]] = []
    for record in ordered_records:
        feature_id = _checkpoint_feature_id(record)
        if feature_id in seen:
            raise ValueError(f"duplicate checkpoint feature_id {feature_id}")
        seen.add(feature_id)
        schedule = schedule_by_id.get(feature_id)
        if schedule is None:
            raise ValueError(f"unexpected checkpoint feature_id {feature_id}")
        if record.get("schedule_digest") != schedule.schedule_digest:
            raise ValueError(f"feature {feature_id} schedule digest mismatch")
        if not _record_matches_schedule(record, schedule):
            raise ValueError(f"feature {feature_id} locked selection mismatch")
        if not _record_has_selected_family_evidence(record, schedule):
            raise ValueError(
                f"feature {feature_id} selected-family evidence contract mismatch"
            )
        features.append(
            {
                "record": record,
                "record_sha256": canonical_json_digest(record),
                "schedule_digest": schedule.schedule_digest,
            }
        )
    return {
        "checkpoint_schema_version": SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION,
        "phase": "selected_family_stability",
        "fingerprint": dict(fingerprint),
        "features": features,
    }


def load_selected_family_checkpoint(
    path: Path,
    *,
    expected_fingerprint: Mapping[str, Any],
    expected_schedules: Sequence[SelectedFamilySchedule],
) -> SelectedFamilyCheckpointLoadResult:
    """Load an exact current checkpoint or return a fresh empty fail-closed state.

    Missing, old, corrupt, mismatched, duplicate, record-drifted, and
    schedule-drifted inputs are never translated, merged, partially reused, or
    deleted by this loader.
    """
    # A missing checkpoint is a normal fresh state, not a rejection.
    if not path.exists():
        return SelectedFamilyCheckpointLoadResult("missing", None, ())
    try:
        with open(path) as checkpoint_file:
            payload = json.load(checkpoint_file)
    except (OSError, ValueError):
        return _rejected_checkpoint("corrupt_checkpoint")
    if not isinstance(payload, dict):
        return _rejected_checkpoint("checkpoint_not_mapping")
    if set(payload) != {
        "checkpoint_schema_version",
        "phase",
        "fingerprint",
        "features",
    }:
        return _rejected_checkpoint("checkpoint_fields_mismatch")
    if payload.get("checkpoint_schema_version") != (
        SELECTED_FAMILY_CHECKPOINT_SCHEMA_VERSION
    ):
        return _rejected_checkpoint("checkpoint_schema_version_mismatch")
    if payload.get("phase") != "selected_family_stability":
        return _rejected_checkpoint("checkpoint_phase_mismatch")
    if payload.get("fingerprint") != dict(expected_fingerprint):
        return _rejected_checkpoint("checkpoint_fingerprint_mismatch")
    features = payload.get("features")
    if not isinstance(features, list):
        return _rejected_checkpoint("checkpoint_features_not_list")
    schedule_by_id = _schedule_by_feature_id(expected_schedules)
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    previous_feature_id: int | None = None
    for wrapper in features:
        if not isinstance(wrapper, dict) or set(wrapper) != {
            "record",
            "record_sha256",
            "schedule_digest",
        }:
            return _rejected_checkpoint("checkpoint_record_wrapper_mismatch")
        record = wrapper.get("record")
        if not isinstance(record, dict):
            return _rejected_checkpoint("checkpoint_record_not_mapping")
        try:
            feature_id = _checkpoint_feature_id(record)
            record_sha256 = canonical_json_digest(record)
        except (TypeError, ValueError):
            return _rejected_checkpoint("checkpoint_record_invalid")
        if feature_id in seen:
            return _rejected_checkpoint("checkpoint_duplicate_feature_id")
        if previous_feature_id is not None and feature_id <= previous_feature_id:
            return _rejected_checkpoint("checkpoint_record_order_mismatch")
        seen.add(feature_id)
        previous_feature_id = feature_id
        schedule = schedule_by_id.get(feature_id)
        if schedule is None:
            return _rejected_checkpoint("checkpoint_unexpected_feature_id")
        if wrapper.get("record_sha256") != record_sha256:
            return _rejected_checkpoint("checkpoint_record_hash_mismatch")
        if (
            wrapper.get("schedule_digest") != schedule.schedule_digest
            or record.get("schedule_digest") != schedule.schedule_digest
        ):
            return _rejected_checkpoint("checkpoint_schedule_digest_mismatch")
        if not _record_matches_schedule(record, schedule):
            return _rejected_checkpoint("checkpoint_record_selection_mismatch")
        if not _record_has_selected_family_evidence(record, schedule):
            return _rejected_checkpoint("checkpoint_record_evidence_mismatch")
        records.append(record)
    return SelectedFamilyCheckpointLoadResult("reused", None, tuple(records))


def _schedule_by_feature_id(
    schedules: Sequence[SelectedFamilySchedule],
) -> dict[int, SelectedFamilySchedule]:
    """Index expected schedules while rejecting duplicate feature identities."""
    # Duplicate expected schedules make checkpoint identity ambiguous.
    result: dict[int, SelectedFamilySchedule] = {}
    for schedule in schedules:
        feature_id = int(schedule.feature_id)
        if feature_id in result:
            raise ValueError(f"duplicate expected schedule feature_id {feature_id}")
        result[feature_id] = schedule
    return result


def _checkpoint_feature_id(record: Mapping[str, Any]) -> int:
    """Read one exact non-boolean integer checkpoint feature identity."""
    # Reject coercible strings/floats before clean and resumed records can diverge.
    value = record.get("feature_id")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("checkpoint feature_id must be an integer")
    return value


def _record_matches_schedule(
    record: Mapping[str, Any], schedule: SelectedFamilySchedule
) -> bool:
    """Require checkpoint output to retain the complete locked selection tuple."""
    # Prevent a matching schedule digest from masking mislabeled record semantics.
    return (
        record.get("family") == schedule.family
        and record.get("selection_mode") == schedule.selection_mode
        and record.get("selected_k") == schedule.reported_selected_k
    )


def selected_family_required_protocol_ids(
    schedule: SelectedFamilySchedule,
) -> list[str]:
    """Return exact required protocol IDs for one selected-family schedule mode."""
    # Keep runner serialization and checkpoint validation on one mode-aware authority.
    if schedule.no_work_reason is not None:
        return ["deliberate_non_evaluation"]
    if schedule.reuse_standalone_assignment:
        return ["standalone_assignment_reuse"]
    required: list[str] = []
    if schedule.evidence_request.low_context_qualification:
        required.append("low_context_qualification")
    if schedule.scalar_metrics:
        required.append("bootstrap")
    if schedule.angle_source != "none":
        required.append("angle")
    if schedule.leave_out:
        required.append("leave_out")
    if schedule.sample_size:
        required.append("sample_size")
    return required


def _record_has_selected_family_evidence(
    record: Mapping[str, Any], schedule: SelectedFamilySchedule
) -> bool:
    """Require structural schedule identity and truthful protocol denominators."""
    # Scientific values are hash-protected; do not rederive runner semantics here.
    if (
        set(record) != _CHECKPOINT_RECORD_KEYS
        or record.get("point_reason") != schedule.point_selection.point_reason
        or record.get("point_record_sha256") != schedule.point_record_sha256
    ):
        return False
    evidence = record.get("selected_family_evidence")
    if not isinstance(evidence, Mapping):
        return False
    required = selected_family_required_protocol_ids(schedule)
    if schedule.no_work_reason is not None:
        protocol_keys = {"deliberate_non_evaluation"}
        expected_reason = schedule.no_work_reason
    elif schedule.reuse_standalone_assignment:
        protocol_keys = {"standalone_assignment_reuse"}
        expected_reason = None
    else:
        protocol_keys = {
            "low_context_qualification",
            "bootstrap",
            "angle",
            "leave_out",
            "sample_size",
        }
        expected_reason = None
    protocols = evidence.get("protocols")
    counters = evidence.get("protocol_counters")
    if (
        evidence.get("required_protocol_ids") != required
        or evidence.get("no_work_reason") != expected_reason
        or not isinstance(protocols, Mapping)
        or set(protocols) != protocol_keys
        or not isinstance(counters, Mapping)
        or set(counters) != protocol_keys
    ):
        return False
    if not all(
        isinstance(protocols[key], Mapping)
        and _complete_protocol_counters(counters[key])
        and protocols[key].get("counters") == counters[key]
        for key in protocol_keys
    ):
        return False
    if schedule.no_work_reason is not None:
        return (
            protocols["deliberate_non_evaluation"].get("plan_digest")
            == hashlib.sha256(b"").hexdigest()
            and counters["deliberate_non_evaluation"]
            == _fixed_counters(non_applicable=1)
        )
    if schedule.reuse_standalone_assignment:
        return (
            protocols["standalone_assignment_reuse"].get("plan_digest")
            == hashlib.sha256(b"").hexdigest()
            and counters["standalone_assignment_reuse"]
            == _fixed_counters(requested=1, valid=1)
        )
    plans_by_protocol = {
        "bootstrap": (schedule.bootstrap, not schedule.scalar_metrics),
        "angle": (
            (
                schedule.raw_angle_plans
                if schedule.angle_source == "raw"
                else schedule.residual_angle_plans
            ),
            True,
        ),
        "leave_out": (schedule.leave_out, False),
        "sample_size": (schedule.sample_size, False),
    }
    return (
        counters["low_context_qualification"]
        == _fixed_counters(requested=1, valid=1)
        and all(
            _planned_protocol_identity(
                protocols[key], plans, empty_non_applicable=empty_non_applicable
            )
            for key, (plans, empty_non_applicable) in plans_by_protocol.items()
        )
    )


def _planned_protocol_identity(
    block: Mapping[str, Any],
    plans: Sequence[Any],
    *,
    empty_non_applicable: bool,
) -> bool:
    """Match retained replicate identities and counts to one immutable plan list."""
    # Validate schedule coverage without judging any scientific result value.
    replicates = block.get("replicates")
    counters = block.get("counters")
    if not isinstance(replicates, list) or not _complete_protocol_counters(counters):
        return False
    if plans:
        expected_counts = len(plans)
        if (
            counters["requested"] != expected_counts
            or sum(counters[key] for key in _PROTOCOL_COUNTER_KEYS - {"requested"})
            != expected_counts
        ):
            return False
    elif counters != _fixed_counters(non_applicable=int(empty_non_applicable)):
        return False
    joined = "".join(str(plan.digest) for plan in plans)
    return (
        len(replicates) == len(plans)
        and block.get("plan_digest")
        == hashlib.sha256(joined.encode("ascii")).hexdigest()
        and all(
            isinstance(replicate, Mapping)
            and replicate.get("plan_digest") == plan.digest
            for replicate, plan in zip(replicates, plans, strict=True)
        )
    )


def _complete_protocol_counters(value: Any) -> bool:
    """Require the exact nonnegative integer protocol counter vocabulary."""
    # Counts are scientific denominators, so booleans and omitted zeros are invalid.
    return (
        isinstance(value, Mapping)
        and set(value) == _PROTOCOL_COUNTER_KEYS
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value.values()
        )
    )


def _fixed_counters(
    *, requested: int = 0, valid: int = 0, non_applicable: int = 0
) -> dict[str, int]:
    """Return the complete counter map for fixed non-replicate protocol events."""
    # Fixed records still retain every zero-valued denominator field.
    return {
        "requested": requested,
        "valid": valid,
        "failed": 0,
        "non_applicable": non_applicable,
        "skipped": 0,
    }


def _rejected_checkpoint(reason: str) -> SelectedFamilyCheckpointLoadResult:
    """Return a rejected checkpoint with an exact reason and fresh empty state."""
    # Every rejection discards the entire candidate payload without deletion.
    return SelectedFamilyCheckpointLoadResult("rejected", reason, ())
