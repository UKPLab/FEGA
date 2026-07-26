from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.common import require_single_entity_attr
from fega.core.compute_effect.artifacts import (
    EffectArtifactWriter,
    summarize_magnitudes,
    validate_manifest_summary_consistency,
)
from fega.core.compute_effect.effects import (
    EffectInputLoader,
    build_effect_context_records,
    compute_feature_effect_rows,
)
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    GRAM_REQUIRED_METADATA,
    canonical_unembedding,
    gram_fingerprint,
    unembedding_fingerprint,
)
from fega.core.positioning import build_positioning_metadata
from fega.core.resources import ModelResources
from fega.paths import (
    compute_effect_dir,
    compute_effect_readout_dir,
    data_prep_activations_dir,
    data_prep_pairs_path,
    data_prep_select_dir,
    effect_summary_path,
    effect_tensors_manifest_path,
    gram_cache_meta_path,
    gram_cache_tensor_path,
)

_logger = logging.getLogger(__name__)


def run_compute_effect(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    """Compute reusable effect tensor artifacts for configured readouts."""
    entity, attr = require_single_entity_attr(config)
    effect_cfg = config.phases.compute_effect
    requested_config_readouts = list(config.phases.data_prep.readouts)
    activations_dir = data_prep_activations_dir(config, entity, attr)
    manifest_path = activations_dir / "activations_manifest.json"
    contexts_path = data_prep_select_dir(config, entity, attr) / "feature_contexts.json"
    pairs_path = data_prep_pairs_path(config, entity, attr)
    output_root = compute_effect_dir(config, entity, attr)
    output_root.mkdir(parents=True, exist_ok=True)

    loader = EffectInputLoader(
        resources=resources,
        cache_max_chunks=effect_cfg.cache_max_chunks,
        cache_max_bytes=effect_cfg.cache_max_bytes,
    )
    manifest, requested_readouts = loader.validate_manifest_readouts(
        manifest_path, requested_config_readouts
    )
    source_positioning = manifest["positioning"]
    positioning_schema_version = int(source_positioning["schema_version"])
    pad_token_id = int(source_positioning["pad_token_id"])
    compute_positioning = build_positioning_metadata(
        pad_token_id=pad_token_id,
        positioning_schema_version=positioning_schema_version,
        batch_size_provenance={
            "phase": "compute_effect",
            "configured_batch_size": effect_cfg.batch_size,
            "oom_adjustments_recorded_per_feature": True,
        },
        source_data_prep_positioning=source_positioning,
    )
    contexts_by_feature = loader.load_contexts(contexts_path)
    gram: torch.Tensor | None = None
    gram_meta: dict[str, Any] | None = None
    hidden_size: int | None = None
    if "final_resid" in requested_readouts:
        gram_cpu, gram_meta = loader.load_gram(
            gram_cache_tensor_path(config), gram_cache_meta_path(config)
        )
        _validate_gram(gram_cpu, gram_meta)
        gram = _gram_to_device(gram_cpu, config.device)
        hidden_size = int(gram.shape[0])

    run_resources = resources or ModelResources(config)
    model, tokenizer, sae = run_resources.get_model_and_sae()
    model.eval()
    sae.eval()
    if gram_meta is not None:
        _validate_gram_readout(model, gram_meta)
    prompt_lookup = (
        loader.load_prompt_lookup(pairs_path) if pairs_path.exists() else None
    )

    writers = {
        readout: EffectArtifactWriter(
            compute_effect_readout_dir(config, readout, entity, attr),
            shard_size=effect_cfg.effect_shard_size,
            include_magnitude_direction=(readout == "final_resid"),
        )
        for readout in requested_readouts
    }
    per_feature_by_readout: dict[str, dict[str, dict[str, Any]]] = {
        readout: {} for readout in requested_readouts
    }
    skipped_features_by_readout: dict[str, list[dict[str, Any]]] = {
        readout: [] for readout in requested_readouts
    }
    readout_widths: dict[str, int] = {}
    min_coverage = int(effect_cfg.min_coverage)
    batch_size = int(effect_cfg.batch_size or 1)

    for feature_id in sorted(contexts_by_feature):
        raw_contexts = contexts_by_feature.get(feature_id, [])
        target_indices = [
            int(record["index"])
            for record in raw_contexts
            if record.get("index") is not None
        ]
        example_bank, _ = loader.load_examples(
            manifest_path=manifest_path,
            activations_dir=activations_dir,
            target_indices=target_indices,
            requested_readouts=requested_readouts,
        )
        _update_readout_widths(readout_widths, example_bank, requested_readouts)
        if "final_resid" in requested_readouts:
            _validate_examples_against_gram(example_bank, hidden_size, manifest_path)
        records, record_stats = build_effect_context_records(
            feature_id=feature_id,
            raw_contexts=raw_contexts,
            example_bank=example_bank,
            prompt_lookup=prompt_lookup,
            tokenizer=tokenizer,
            default_attr=attr,
        )
        rows_by_readout, compute_stats_by_readout = compute_feature_effect_rows(
            feature_id=feature_id,
            context_records=records,
            example_bank=example_bank,
            model=model,
            sae=sae,
            requested_readouts=requested_readouts,
            gram=gram,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            positioning_schema_version=positioning_schema_version,
            normalization_eps=effect_cfg.normalization_eps,
            tau_zero=effect_cfg.tau_zero,
        )
        for readout in requested_readouts:
            rows = rows_by_readout[readout]
            counters = {**record_stats, **compute_stats_by_readout[readout]}
            pointer = None
            if len(rows) >= min_coverage:
                pointer = writers[readout].add_feature_rows(
                    feature_id,
                    rows,
                    candidate_identity=list(counters["candidate_identity"]),
                    retained_mask=list(counters["retained_mask"]),
                )
            feature_summary = _build_feature_summary(
                readout=readout,
                feature_id=feature_id,
                requested_contexts=len(raw_contexts),
                loaded_contexts=len(records),
                rows=rows,
                counters=counters,
                pointer=pointer,
                min_coverage=min_coverage,
            )
            per_feature_by_readout[readout][str(feature_id)] = feature_summary
            if pointer is None:
                skipped_features_by_readout[readout].append(
                    {
                        "feature_id": feature_id,
                        "skipped_reason": feature_summary["skipped_reason"],
                    }
                )

    for readout, writer in writers.items():
        writer.flush()
        summary = _build_summary(
            readout=readout,
            entity=entity,
            attr=attr,
            per_feature=per_feature_by_readout[readout],
            skipped_features=skipped_features_by_readout[readout],
            writer=writer,
            min_coverage=min_coverage,
            normalization_eps=effect_cfg.normalization_eps,
            tau_zero=effect_cfg.tau_zero,
            manifest_path=manifest_path,
            contexts_path=contexts_path,
            positioning=compute_positioning,
            gram_path=gram_cache_tensor_path(config)
            if readout == "final_resid"
            else None,
            gram_metadata=gram_meta if readout == "final_resid" else None,
        )
        readout_dir = compute_effect_readout_dir(config, readout, entity, attr)
        manifest_payload = _build_manifest(
            readout=readout,
            vector_size=_readout_vector_size(readout, hidden_size, readout_widths),
            min_coverage=min_coverage,
            normalization_eps=effect_cfg.normalization_eps,
            tau_zero=effect_cfg.tau_zero,
            manifest_path=manifest_path,
            contexts_path=contexts_path,
            pairs_path=pairs_path,
            activations_dir=activations_dir,
            gram_path=gram_cache_tensor_path(config)
            if readout == "final_resid"
            else None,
            gram_meta_path=gram_cache_meta_path(config)
            if readout == "final_resid"
            else None,
            summary_path=effect_summary_path(config, readout),
            output_dir=readout_dir,
            summary=summary,
            writer=writer,
            source_manifest=manifest,
            positioning=compute_positioning,
            gram_metadata=gram_meta if readout == "final_resid" else None,
        )
        validate_manifest_summary_consistency(manifest_payload, summary)
        writer.write_summary(effect_summary_path(config, readout), summary)
        writer.write_manifest(
            effect_tensors_manifest_path(config, readout), manifest_payload
        )
    _logger.info(
        "compute_effect complete: readouts=%s, dir=%s",
        ",".join(requested_readouts),
        output_root,
    )


def _validate_gram(gram: torch.Tensor, gram_meta: dict[str, Any]) -> None:
    """Validate required Gram metadata and exact tensor dimensions."""
    # Check tensor rank first, then require the complete fail-closed metadata set.
    if gram.dim() != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(
            f"Gram tensor must be square rank-2, got shape {tuple(gram.shape)}."
        )
    missing = [key for key in GRAM_REQUIRED_METADATA if key not in gram_meta]
    if missing:
        raise ValueError(f"Gram metadata missing required fields: {missing}.")
    if gram_meta["readout_name"] != "final_resid":
        raise ValueError("Gram metadata readout_name must be 'final_resid'.")
    if gram_meta["construction_recipe"] != GRAM_CONSTRUCTION_RECIPE:
        raise ValueError("Gram metadata construction_recipe mismatch.")
    meta_hidden = int(gram_meta["hidden_width"])
    if meta_hidden != int(gram.shape[0]):
        raise ValueError(
            "Gram metadata hidden_width="
            f"{meta_hidden} does not match tensor shape {tuple(gram.shape)}."
        )
    if list(gram_meta["gram_shape"]) != list(gram.shape):
        raise ValueError("Gram metadata gram_shape does not match tensor shape.")
    expected_dtype = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(gram_meta["gram_dtype"])
    if expected_dtype is None or gram.dtype != expected_dtype:
        raise ValueError("Gram metadata gram_dtype does not match tensor dtype.")
    if gram_fingerprint(gram) != gram_meta["gram_sha256"]:
        raise ValueError("Gram tensor SHA-256 does not match metadata.")


def _validate_gram_readout(model: Any, gram_meta: dict[str, Any]) -> None:
    """Bind loaded Gram metadata to the exact model readout before scoring."""
    # Recompute the canonical readout digest rather than trusting summary fields.
    unembedding = canonical_unembedding(model)
    model_config = getattr(model, "config", None)
    checkpoint_identity = (
        getattr(model_config, "name_or_path", None)
        or getattr(model_config, "_name_or_path", None)
        or getattr(model, "name_or_path", None)
    )
    if checkpoint_identity != gram_meta["checkpoint_identity"]:
        raise ValueError("Gram checkpoint identity mismatch.")
    if list(unembedding.shape) != list(gram_meta["unembedding_shape"]):
        raise ValueError("Gram metadata unembedding_shape mismatch.")
    if str(unembedding.dtype) != gram_meta["unembedding_dtype"]:
        raise ValueError("Gram metadata unembedding_dtype mismatch.")
    if unembedding_fingerprint(unembedding) != gram_meta["unembedding_fingerprint"]:
        raise ValueError("Gram/readout fingerprint mismatch.")


def _gram_to_device(gram: torch.Tensor, device: str) -> torch.Tensor:
    try:
        return gram.to(device=torch.device(device))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Unable to keep Gram cache resident on device {device!r}; "
            "reduce hidden size, use a device with more memory, or regenerate a "
            "valid Gram cache."
        ) from exc


def _validate_examples_against_gram(
    example_bank: dict[int, dict[str, Any]],
    hidden_size: int | None,
    manifest_path: Path,
) -> None:
    if hidden_size is None:
        raise ValueError("Gram hidden_size is required for final_resid validation.")
    for idx, example in example_bank.items():
        final_resid = _example_readout(example, "final_resid")
        if final_resid.numel() != hidden_size:
            raise ValueError(
                "Gram shape is incompatible with final_resid width: "
                f"index {idx} has width {final_resid.numel()}, Gram width {hidden_size} "
                f"(manifest {manifest_path})."
            )


def _build_feature_summary(
    *,
    readout: str,
    feature_id: int,
    requested_contexts: int,
    loaded_contexts: int,
    rows: list[dict[str, Any]],
    counters: dict[str, Any],
    pointer: dict[str, Any] | None,
    min_coverage: int,
) -> dict[str, Any]:
    usable = len(rows)
    record: dict[str, Any] = {
        "feature_id": feature_id,
        "readout_name": readout,
        "requested_contexts": requested_contexts,
        "loaded_contexts": loaded_contexts,
        "usable_effects": usable,
        "n_j": usable,
        "skipped_near_zero": int(counters.get("skipped_near_zero", 0)),
        "skipped_zero_norm": int(counters.get("skipped_zero_norm", 0)),
        "skipped_nonfinite": int(counters.get("skipped_nonfinite", 0)),
        "skipped_numerical_failure": int(
            counters.get("skipped_numerical_failure", 0)
        ),
        "candidate_identity": list(counters.get("candidate_identity", [])),
        "retained_mask": list(counters.get("retained_mask", [])),
        "skipped_missing_example": int(counters.get("skipped_missing_example", 0)),
        "skipped_invalid": int(counters.get("skipped_invalid", 0)),
        "skipped_missing_prompt": int(counters.get("skipped_missing_prompt", 0)),
        "oom_adjustments": counters.get("oom_adjustments", []),
    }
    if readout == "final_resid":
        magnitudes = [float(row["magnitude"]) for row in rows]
        record.update(summarize_magnitudes(magnitudes))
        record["max_unit_gram_norm_error"] = max(
            (float(row["unit_gram_norm_error"]) for row in rows), default=None
        )
    if pointer is None:
        record.update(
            {
                "skipped_reason": f"below_min_coverage_{min_coverage}",
                "tensor_shard": None,
                "row_start": None,
                "row_end": None,
            }
        )
    else:
        record.update(pointer)
    return record


def _build_summary(
    *,
    readout: str,
    entity: str,
    attr: str,
    per_feature: dict[str, dict[str, Any]],
    skipped_features: list[dict[str, Any]],
    writer: EffectArtifactWriter,
    min_coverage: int,
    normalization_eps: float,
    tau_zero: float,
    manifest_path: Path,
    contexts_path: Path,
    positioning: dict[str, Any],
    gram_path: Path | None,
    gram_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    features_with_effects = sum(
        1 for record in per_feature.values() if record.get("tensor_shard") is not None
    )
    summary = {
        "summary": {
            "readout_name": readout,
            "entity_class": entity,
            "attribute": attr,
            "features_total": len(per_feature),
            "features_with_effects": features_with_effects,
            "features_skipped": len(skipped_features),
            "total_effect_rows": writer.total_rows,
            "shard_count": len(writer.shard_records),
            "min_coverage": min_coverage,
            "normalization_eps": normalization_eps,
            "tau_zero": tau_zero,
            "manifest_path": str(manifest_path),
            "contexts_path": str(contexts_path),
        },
        "positioning": positioning,
        "per_feature": per_feature,
        "skipped_features": skipped_features,
    }
    if gram_path is not None:
        summary["summary"]["gram_path"] = str(gram_path)
    if gram_metadata is not None:
        summary["gram_metadata"] = {
            key: gram_metadata[key] for key in GRAM_REQUIRED_METADATA
        }
    return summary


def _build_manifest(
    *,
    readout: str,
    vector_size: int | None,
    min_coverage: int,
    normalization_eps: float,
    tau_zero: float,
    manifest_path: Path,
    contexts_path: Path,
    pairs_path: Path,
    activations_dir: Path,
    gram_path: Path | None,
    gram_meta_path: Path | None,
    summary_path: Path,
    output_dir: Path,
    summary: dict[str, Any],
    writer: EffectArtifactWriter,
    source_manifest: dict[str, Any],
    positioning: dict[str, Any],
    gram_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = _readout_manifest_metadata(readout)
    inputs = {
        "activations_dir": str(activations_dir),
        "manifest_path": str(manifest_path),
        "contexts_path": str(contexts_path),
        "pairs_path": str(pairs_path),
        "source_total_records": source_manifest.get("total_records"),
        "source_chunk_count": source_manifest.get("chunk_count"),
    }
    if gram_path is not None:
        inputs["gram_path"] = str(gram_path)
    if gram_meta_path is not None:
        inputs["gram_meta_path"] = str(gram_meta_path)
    payload = {
        "schema_version": 1,
        **metadata,
        "vector_size": vector_size,
        "dtype": "float32",
        "normalization_eps": normalization_eps,
        "tau_zero": tau_zero,
        "min_coverage": min_coverage,
        "inputs": inputs,
        "outputs": {
            "compute_effect_dir": str(output_dir),
            "effect_summary_path": str(summary_path),
        },
        "counts": {
            "features_total": summary["summary"]["features_total"],
            "features_with_effects": summary["summary"]["features_with_effects"],
            "features_skipped": summary["summary"]["features_skipped"],
            "total_effect_rows": writer.total_rows,
            "shard_count": len(writer.shard_records),
        },
        "positioning": positioning,
        "shards": writer.shard_records,
    }
    if readout == "final_resid":
        payload["hidden_size"] = vector_size
        if gram_metadata is None:
            raise ValueError("final_resid manifest requires Gram metadata.")
        payload["gram_metadata"] = {
            key: gram_metadata[key] for key in GRAM_REQUIRED_METADATA
        }
    return payload


def _readout_manifest_metadata(readout: str) -> dict[str, str]:
    if readout == "final_resid":
        return {
            "readout_name": "final_resid",
            "effect_space": "final_resid",
            "metric_space": "residual_gram",
        }
    raise ValueError(f"Unsupported compute_effect readout: {readout!r}.")


def _update_readout_widths(
    widths: dict[str, int],
    example_bank: dict[int, dict[str, Any]],
    requested_readouts: list[str],
) -> None:
    for example in example_bank.values():
        for readout in requested_readouts:
            if readout in widths:
                continue
            tensor = _example_readout(example, readout)
            if tensor.dim() != 1:
                raise ValueError(
                    f"Expected rank-1 {readout} readout, got {tuple(tensor.shape)}."
                )
            widths[readout] = int(tensor.numel())


def _readout_vector_size(
    readout: str, hidden_size: int | None, widths: dict[str, int]
) -> int | None:
    if readout == "final_resid":
        return hidden_size
    return widths.get(readout)


def _example_readout(example: dict[str, Any], readout: str) -> torch.Tensor:
    readouts = example.get("readouts")
    if isinstance(readouts, dict) and readout in readouts:
        return readouts[readout]
    return example[readout]
