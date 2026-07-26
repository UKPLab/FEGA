from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fega.core.data_prep.gram_cache import GRAM_REQUIRED_METADATA

CANONICAL_SOURCE_FINGERPRINT_SCHEMA_VERSION = 2


def canonical_source_fingerprint(
    manifest: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Return the canonical final-residual source fingerprint shared by phases."""
    # Bind loaded source documents, ordered identity, and exact Gram/readout metadata.
    if manifest.get("effect_space") != "final_resid":
        raise ValueError("Canonical source manifest must use final_resid.")
    metadata = manifest.get("gram_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Canonical source manifest is missing Gram metadata.")
    missing = [key for key in GRAM_REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Canonical source Gram metadata missing fields: {missing}.")
    ordered_retained_identity = _ordered_retained_identity(summary)
    components = {
        "manifest_sha256": canonical_json_digest(manifest),
        "summary_sha256": canonical_json_digest(summary),
        "ordered_retained_identity_sha256": canonical_json_digest(
            ordered_retained_identity
        ),
        "tensor_shards_sha256": _tensor_shards_digest(manifest),
        "gram_readout_metadata": {
            key: metadata[key] for key in GRAM_REQUIRED_METADATA
        },
    }
    return {
        "schema_version": CANONICAL_SOURCE_FINGERPRINT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "digest": canonical_json_digest(components),
        "components": components,
    }


def require_canonical_source_fingerprint(
    payload: dict[str, Any],
    expected: dict[str, Any],
    *,
    artifact_label: str,
) -> None:
    """Reject an artifact whose canonical source differs from the loaded source."""
    # Compare the complete shared contract before any downstream scoring or reporting.
    actual = payload.get("canonical_source_fingerprint")
    if actual != expected:
        raise ValueError(
            f"{artifact_label} canonical source fingerprint mismatch."
        )


def _ordered_retained_identity(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Project each feature's declared mask into one deterministic retained identity."""
    # Preserve numeric feature order and the source order within every retained row list.
    per_feature = summary.get("per_feature")
    if not isinstance(per_feature, dict):
        raise ValueError("Canonical source summary is missing per_feature.")
    ordered: list[dict[str, Any]] = []
    for feature_key in sorted(per_feature, key=lambda key: int(key)):
        record = per_feature[feature_key]
        if not isinstance(record, dict):
            raise ValueError(
                f"Canonical source feature {feature_key} must be a mapping."
            )
        identities = record.get("candidate_identity")
        retained_mask = record.get("retained_mask")
        if not isinstance(identities, list) or not isinstance(retained_mask, list):
            raise ValueError(
                f"Canonical source feature {feature_key} is missing identity/mask."
            )
        if len(identities) != len(retained_mask):
            raise ValueError(
                f"Canonical source feature {feature_key} identity/mask length mismatch."
            )
        ordered.append(
            {
                "feature_id": int(record.get("feature_id", feature_key)),
                "retained_identity": [
                    identity
                    for identity, retained in zip(
                        identities, retained_mask, strict=True
                    )
                    if retained
                ],
            }
        )
    return ordered


def _tensor_shards_digest(manifest: dict[str, Any]) -> str:
    """Hash sorted manifest shard paths together with their complete file bytes.

    Every declared ``shards[*].path`` is opened directly and streamed into
    SHA-256 so same-path content changes invalidate the canonical source before
    any downstream scoring or checkpoint reuse.
    """
    # Validate each declared path, stream its bytes, then hash the sorted records.
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        raise ValueError("Canonical source manifest shards must be a list.")
    records: list[dict[str, str]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(
                f"Canonical source manifest shards[{index}] must be a mapping."
            )
        raw_path = shard.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(
                f"Canonical source manifest shards[{index}].path is invalid: "
                f"{raw_path!r}."
            )
        path = Path(raw_path)
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as shard_file:
                for chunk in iter(lambda: shard_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError as exc:
            raise ValueError(
                f"Canonical source shard is invalid or missing: {path}: {exc}"
            ) from exc
        records.append({"path": raw_path, "sha256": hasher.hexdigest()})
    records.sort(key=lambda record: (record["path"], record["sha256"]))
    return canonical_json_digest(records)


def canonical_json_digest(value: Any) -> str:
    """Hash one JSON-compatible value with canonical key and separator ordering."""
    # Use a single deterministic encoding for every component and the final contract.
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
