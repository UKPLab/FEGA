from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.compute_effect.artifacts import validate_manifest_summary_consistency
from fega.core.data_prep.gram_cache import GRAM_REQUIRED_METADATA, gram_fingerprint
from fega.core.resources import ModelResources
from fega.paths import effect_summary_path, effect_tensors_manifest_path


@dataclass(frozen=True)
class GeometryMetricsInputs:
    effect_space: str
    artifact_dir: Path
    manifest_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FeatureEffectBlock:
    feature_id: int
    source_summary: dict[str, Any]
    tensor_shard: str | None
    rows: torch.Tensor | None
    skipped_reason: str | None


def load_geometry_metrics_inputs(
    config: FEGAPipelineConfig,
    effect_space: str,
    resources: ModelResources | None = None,
) -> GeometryMetricsInputs:
    """Load and validate compute_effect manifest + summary for one effect space."""
    manifest_path = effect_tensors_manifest_path(config, effect_space)
    summary_path = effect_summary_path(config, effect_space)
    manifest = _load_json(manifest_path, resources, label="compute_effect manifest")
    summary = _load_json(summary_path, resources, label="compute_effect summary")

    _validate_effect_space_contract(effect_space, manifest, manifest_path)
    validate_manifest_summary_consistency(manifest, summary)
    _validate_final_resid_metadata(manifest, summary, manifest_path)
    artifact_dir = manifest_path.parent
    _validate_referenced_shards(artifact_dir, manifest, summary)
    return GeometryMetricsInputs(
        effect_space=effect_space,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        manifest=manifest,
        summary=summary,
    )


def iter_feature_blocks(
    inputs: GeometryMetricsInputs, *, exclude_feature_ids: set[int] | None = None
) -> Iterator[FeatureEffectBlock]:
    """Yield feature row blocks, keeping at most one shard payload live."""
    per_feature = inputs.summary.get("per_feature")
    if not isinstance(per_feature, dict):
        raise ValueError(f"effect_summary missing `per_feature`: {inputs.summary_path}")
    excluded = exclude_feature_ids or set()

    value_key = "direction"
    current_shard_name: str | None = None
    current_payload: dict[str, Any] | None = None

    for feature_key in sorted(per_feature, key=lambda key: int(key)):
        record = per_feature[feature_key]
        if not isinstance(record, dict):
            raise ValueError(f"Feature {feature_key} summary must be a mapping.")
        feature_id = int(record.get("feature_id", feature_key))
        if feature_id in excluded:
            continue
        shard_name = record.get("tensor_shard")
        if shard_name is None:
            yield FeatureEffectBlock(
                feature_id=feature_id,
                source_summary=record,
                tensor_shard=None,
                rows=None,
                skipped_reason=record.get("skipped_reason") or "missing_tensor_shard",
            )
            continue

        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError(f"Feature {feature_key} has invalid tensor_shard.")
        if current_shard_name != shard_name:
            current_payload = _load_shard(inputs.artifact_dir / shard_name, value_key)
            current_shard_name = shard_name
        if current_payload is None:
            raise ValueError(f"Shard payload unexpectedly unavailable: {shard_name}")
        row_start = _required_int(record.get("row_start"), f"{feature_key}.row_start")
        row_end = _required_int(record.get("row_end"), f"{feature_key}.row_end")
        if row_end < row_start:
            raise ValueError(f"Feature {feature_key} has invalid row range.")
        values = current_payload[value_key]
        if row_end > int(values.shape[0]):
            raise ValueError(
                f"Feature {feature_key} row range exceeds shard rows in {shard_name}."
            )
        feature_ids = current_payload.get("feature_ids")
        if not isinstance(feature_ids, torch.Tensor):
            raise ValueError(f"Shard {shard_name} missing feature_ids tensor.")
        matches = (feature_ids == feature_id).nonzero(as_tuple=False).flatten().tolist()
        if len(matches) != 1:
            raise ValueError(f"Shard {shard_name} has ambiguous feature identity.")
        group_idx = matches[0]
        candidate_identity = current_payload.get("candidate_identity", [])[group_idx]
        retained_mask = current_payload.get("retained_mask", [])[group_idx]
        if candidate_identity != record.get("candidate_identity"):
            raise ValueError("Shard/summary candidate identity mismatch.")
        if retained_mask != record.get("retained_mask"):
            raise ValueError("Shard/summary retained mask mismatch.")
        retained_identity = [
            identity
            for identity, keep in zip(candidate_identity, retained_mask)
            if keep
        ]
        shard_identity = [
            {
                "attribute_label": current_payload["attribute_labels"][row],
                "pair_role": current_payload["pair_roles"][row],
                "pair_index": int(current_payload["pair_indices"][row].item()),
            }
            for row in range(row_start, row_end)
        ]
        if retained_identity != shard_identity:
            raise ValueError("Retained shard identity is not the declared mask result.")
        yield FeatureEffectBlock(
            feature_id=feature_id,
            source_summary=record,
            tensor_shard=shard_name,
            rows=values[row_start:row_end].detach().cpu().to(dtype=torch.float32),
            skipped_reason=None,
        )


def resolve_final_resid_gram(
    inputs: GeometryMetricsInputs, resources: ModelResources | None = None
) -> torch.Tensor:
    """Resolve the residual Gram through shared compute_effect cache, then path."""
    # Resolve only after metadata has passed the exact manifest/summary contract.
    gram_raw = inputs.manifest.get("inputs", {}).get("gram_path")
    if not gram_raw:
        raise ValueError(
            f"final_resid manifest missing inputs.gram_path: {inputs.manifest_path}"
        )
    gram_path = Path(gram_raw)
    key = _cache_key(gram_path)
    if resources is not None:
        cache = getattr(resources, "_compute_effect_gram_cache", None)
        if isinstance(cache, dict) and key in cache:
            gram = cache[key].detach().cpu()
            _validate_loaded_gram(gram, inputs)
            return gram
    if not gram_path.exists():
        raise FileNotFoundError(f"Missing residual Gram tensor: {gram_path}")
    gram = torch.load(gram_path, map_location="cpu")
    _validate_loaded_gram(gram, inputs)
    if resources is not None:
        cache = getattr(resources, "_compute_effect_gram_cache", None)
        if cache is None:
            cache = {}
            setattr(resources, "_compute_effect_gram_cache", cache)
        cache[key] = gram
    return gram


def write_geometry_metrics_scores(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the geometry_metrics JSON score artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _load_json(
    path: Path, resources: ModelResources | None, *, label: str
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
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


def _validate_effect_space_contract(
    effect_space: str, manifest: dict[str, Any], manifest_path: Path
) -> None:
    if effect_space != "final_resid":
        raise ValueError("geometry_metrics supports only effect_space='final_resid'.")
    expected_metric_space = "residual_gram"
    if manifest.get("effect_space") != effect_space:
        raise ValueError(
            f"Manifest effect_space mismatch in {manifest_path}: "
            f"expected {effect_space!r}, got {manifest.get('effect_space')!r}."
        )
    if manifest.get("metric_space") != expected_metric_space:
        raise ValueError(
            f"Manifest metric_space mismatch in {manifest_path}: "
            f"expected {expected_metric_space!r}, got {manifest.get('metric_space')!r}."
        )


def _validate_final_resid_metadata(
    manifest: dict[str, Any], summary: dict[str, Any], manifest_path: Path
) -> None:
    """Require identical complete Gram/readout provenance before downstream use."""
    # Compare required fields exactly; missing or drifted metadata is fatal.
    manifest_meta = manifest.get("gram_metadata")
    summary_meta = summary.get("gram_metadata")
    if not isinstance(manifest_meta, dict) or not isinstance(summary_meta, dict):
        raise ValueError(f"Missing final_resid Gram metadata: {manifest_path}")
    missing = [key for key in GRAM_REQUIRED_METADATA if key not in manifest_meta]
    if missing:
        raise ValueError(f"Missing final_resid Gram metadata fields: {missing}")
    expected = {key: manifest_meta[key] for key in GRAM_REQUIRED_METADATA}
    if summary_meta != expected:
        raise ValueError("final_resid manifest/summary Gram metadata mismatch.")
    gram_meta_raw = manifest.get("inputs", {}).get("gram_meta_path")
    if not gram_meta_raw:
        raise ValueError("final_resid manifest missing inputs.gram_meta_path.")
    gram_meta_path = Path(gram_meta_raw)
    if not gram_meta_path.exists():
        raise FileNotFoundError(f"Missing Gram metadata artifact: {gram_meta_path}")
    with open(gram_meta_path) as f:
        gram_meta_artifact = json.load(f)
    if not isinstance(gram_meta_artifact, dict):
        raise ValueError("Gram metadata artifact must be a JSON object.")
    artifact_missing = [
        key for key in GRAM_REQUIRED_METADATA if key not in gram_meta_artifact
    ]
    if artifact_missing:
        raise ValueError(f"Gram metadata artifact missing fields: {artifact_missing}")
    artifact_expected = {
        key: gram_meta_artifact[key] for key in GRAM_REQUIRED_METADATA
    }
    if artifact_expected != expected:
        raise ValueError("Gram metadata artifact/manifest mismatch.")
    hidden_width = int(expected["hidden_width"])
    if manifest.get("vector_size") != hidden_width:
        raise ValueError("final_resid vector_size does not match Gram hidden width.")


def _validate_loaded_gram(gram: torch.Tensor, inputs: GeometryMetricsInputs) -> None:
    """Check exact loaded or cached Gram bytes before any feature rows are scored."""
    # Bind every load boundary to declared shape, dtype, and cryptographic digest.
    metadata = inputs.manifest["gram_metadata"]
    hidden_width = int(metadata["hidden_width"])
    if gram.ndim != 2 or tuple(gram.shape) != (hidden_width, hidden_width):
        raise ValueError(
            "Residual Gram tensor shape does not match fingerprint metadata."
        )
    if str(gram.dtype) != f"torch.{metadata['gram_dtype']}":
        raise ValueError("Residual Gram tensor dtype does not match metadata.")
    if gram_fingerprint(gram) != metadata["gram_sha256"]:
        raise ValueError("Residual Gram tensor SHA-256 does not match metadata.")


def _validate_referenced_shards(
    artifact_dir: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> None:
    manifest_shards = {
        Path(str(shard.get("path"))).name for shard in manifest.get("shards", [])
    }
    for feature_key, record in summary.get("per_feature", {}).items():
        if not isinstance(record, dict):
            raise ValueError(f"Feature {feature_key} summary must be a mapping.")
        shard = record.get("tensor_shard")
        if shard is None:
            continue
        if not isinstance(shard, str) or not shard:
            raise ValueError(f"Feature {feature_key} has invalid tensor_shard.")
        if manifest_shards and shard not in manifest_shards:
            raise ValueError(
                f"Feature {feature_key} references shard {shard!r} absent from manifest."
            )
        shard_path = artifact_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing compute_effect shard: {shard_path}")


def _load_shard(path: Path, value_key: str) -> dict[str, Any]:
    """Load one effect shard and require scientific identity/mask tensors."""
    # Fail before scoring when retained tensors cannot be traced to one mask.
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Shard must contain a tensor mapping: {path}")
    if value_key not in payload:
        raise ValueError(f"Shard {path} missing required `{value_key}` tensor.")
    required_identity_keys = {
        "feature_ids",
        "pair_indices",
        "attribute_labels",
        "pair_roles",
        "candidate_identity",
        "retained_mask",
    }
    missing = sorted(required_identity_keys.difference(payload))
    if missing:
        raise ValueError(f"Shard {path} missing identity/mask keys: {missing}.")
    values = payload[value_key]
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"Shard {path} `{value_key}` must be a rank-2 tensor.")
    if values.dtype != torch.float32:
        payload[value_key] = values.to(dtype=torch.float32)
    return payload


def _required_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"`{label}` must be an integer, got {value!r}.")
    return value


def _cache_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)
