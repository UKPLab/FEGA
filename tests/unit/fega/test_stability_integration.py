from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core import resources
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    gram_fingerprint,
    unembedding_fingerprint,
)
from fega.core.geometry_reporting import artifacts as reporting_artifacts
from fega.core.geometry_reporting import classifier
from fega.core.geometry_reporting import records as reporting_records
from fega.core.geometry_reporting.point_selection import (
    POINT_SELECTION_CONTRACT_VERSION,
    MixtureAuditState,
    PointSelection,
    point_selection_identity,
)
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.stability import runner as stability_runner
from fega.core.stability.artifacts import (
    build_selected_family_checkpoint_fingerprint,
    build_selected_family_checkpoint_payload,
    load_stability_inputs,
    write_stability_checkpoint,
)
from fega.core.vmf import factor_reuse
from fega.core.vmf import metrics as vmf_metrics
from fega.core.vmf import runner as vmf_runner
from fega.paths import (
    effect_summary_path,
    effect_tensors_manifest_path,
    gram_cache_dir,
    gram_cache_meta_path,
    gram_cache_tensor_path,
    stability_scores_path,
)


def _config(tmp_path: Path, *, workers: int, resume: bool) -> FEGAPipelineConfig:
    """Build one small production-shaped stability configuration."""
    # Keep all scientific controls fixed while varying execution-only resume/workers.
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"eval_config": {"model_name": "gpt2"}}))
    config = FEGAPipelineConfig(
        reference_json=reference,
        output_root=tmp_path / "output",
        device="cpu",
        entity_attribute_selection={"city": ["Country"]},
    )
    config.phases.stability.workers = workers
    config.phases.stability.resume = resume
    config.phases.stability.checkpoint_flush_features = 1
    config.phases.stability.scalar.bootstrap_rounds = 2
    config.phases.stability.subspace.resample_rounds = 1
    config.phases.stability.sample_size.target_sizes = [8]
    config.phases.stability.sample_size.subset_rounds = 1
    config.phases.stability.sample_size.strong_subset_rounds = 1
    return config


def _write_compute_effect_artifacts(
    config: FEGAPipelineConfig, feature_rows: dict[int, torch.Tensor]
) -> None:
    """Write the real compute-effect and Gram contracts consumed by stability."""
    # Pack ordered feature rows into one canonical shard with complete row identities.
    artifact_dir = effect_tensors_manifest_path(config, "final_resid").parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_ids = sorted(feature_rows)
    chunks = [
        feature_rows[feature_id].to(dtype=torch.float32)
        for feature_id in feature_ids
    ]
    rows = torch.cat(chunks, dim=0)
    offsets = [0]
    for chunk in chunks:
        offsets.append(offsets[-1] + int(chunk.shape[0]))
    identities = [
        [
            {
                "attribute_label": "Country",
                "pair_role": "cause_base",
                "pair_index": row,
            }
            for row in range(int(feature_rows[feature_id].shape[0]))
        ]
        for feature_id in feature_ids
    ]
    retained = [[True] * len(values) for values in identities]
    shard = artifact_dir / "effect_tensors_00000.pt"
    torch.save(
        {
            "feature_ids": torch.tensor(feature_ids, dtype=torch.long),
            "row_offsets": torch.tensor(offsets, dtype=torch.long),
            "context_indices": torch.arange(rows.shape[0], dtype=torch.long),
            "pair_indices": torch.arange(rows.shape[0], dtype=torch.long),
            "attribute_labels": ["Country"] * int(rows.shape[0]),
            "pair_roles": ["cause_base"] * int(rows.shape[0]),
            "candidate_identity": identities,
            "retained_mask": retained,
            "direction": rows,
            "delta": rows,
            "magnitude": torch.ones(rows.shape[0]),
        },
        shard,
    )

    # Bind an identity Gram with the same real metadata and byte checks as production.
    width = int(rows.shape[1])
    unembedding = torch.eye(width, dtype=torch.float32)
    gram = torch.eye(width, dtype=torch.float32)
    gram_metadata = {
        "checkpoint_identity": "integration-test-model",
        "readout_name": "final_resid",
        "hidden_width": width,
        "gram_dtype": "float32",
        "construction_recipe": GRAM_CONSTRUCTION_RECIPE,
        "unembedding_fingerprint": unembedding_fingerprint(unembedding),
        "unembedding_dtype": str(unembedding.dtype),
        "unembedding_shape": list(unembedding.shape),
        "gram_shape": [width, width],
        "gram_sha256": gram_fingerprint(gram),
    }
    gram_cache_dir(config).mkdir(parents=True, exist_ok=True)
    torch.save(gram, gram_cache_tensor_path(config))
    gram_cache_meta_path(config).write_text(json.dumps(gram_metadata))
    manifest = {
        "schema_version": 1,
        "effect_space": "final_resid",
        "metric_space": "residual_gram",
        "vector_size": width,
        "inputs": {
            "gram_path": str(gram_cache_tensor_path(config)),
            "gram_meta_path": str(gram_cache_meta_path(config)),
        },
        "counts": {"total_effect_rows": int(rows.shape[0]), "shard_count": 1},
        "shards": [{"path": str(shard), "rows": int(rows.shape[0])}],
        "gram_metadata": gram_metadata,
    }
    effect_tensors_manifest_path(config, "final_resid").write_text(json.dumps(manifest))
    effect_summary_path(config, "final_resid").write_text(
        json.dumps(
            {
                "summary": {
                    "readout_name": "final_resid",
                    "total_effect_rows": int(rows.shape[0]),
                    "shard_count": 1,
                },
                "gram_metadata": gram_metadata,
                "per_feature": {
                    str(feature_id): {
                        "feature_id": feature_id,
                        "usable_effects": int(chunks[position].shape[0]),
                        "tensor_shard": shard.name,
                        "row_start": offsets[position],
                        "row_end": offsets[position + 1],
                        "candidate_identity": identities[position],
                        "retained_mask": retained[position],
                    }
                    for position, feature_id in enumerate(feature_ids)
                },
            }
        )
    )


def _point_records(feature_ids: list[int], *, n_valid: int) -> tuple[list[dict], dict]:
    """Build complete strict-ray point records and their durable hash summary."""
    # The reporting-owned selection is explicit; schedules must consume it unchanged.
    selection = PointSelection(
        family="directed_ray",
        selected_k=None,
        mode="strict",
        point_reason="ray_anchor",
        mixture_audit_state=MixtureAuditState(
            status="not_applicable",
            acceptance="not_evaluated",
            reason="ray_precedence",
            failed_gates=(),
            unavailable_gates=(),
        ),
    )
    records: list[dict] = []
    hashes: dict[str, str] = {}
    for feature_id in feature_ids:
        record = {
            "feature_id": feature_id,
            "n_valid": n_valid,
            "c_ray": 0.95,
            "s_span_1": 0.95,
            "point_selection": point_selection_identity(selection),
        }
        digest = canonical_json_digest(record)
        record["point_record_sha256"] = digest
        hashes[str(feature_id)] = digest
        records.append(record)
    inventory = [
        {
            "feature_id": record["feature_id"],
            "point_record_sha256": record["point_record_sha256"],
        }
        for record in records
    ]
    return records, {
        "features_total": len(records),
        "source_paths": {
            "compute_effect_final_resid_manifest": "manifest.json",
            "compute_effect_final_resid": "summary.json",
            "geometry_metrics_final_resid": "geometry.json",
            "vmf_pre_softcap_logits": "vmf.json",
        },
        "point_selection_contract_version": POINT_SELECTION_CONTRACT_VERSION,
        "point_record_hashes": hashes,
        "point_records_sha256": canonical_json_digest(inventory),
    }


def _point_inputs(feature_ids: list[int]) -> dict:
    """Provide complete validated upstream identities to the real point loader."""
    # Use non-placeholder closed identity shapes required by the checkpoint authority.
    scientific = {
        "schema_version": 3,
        "candidate_mode_counts": [1, 2, 3, 4],
        "assignment_fraction": 0.8,
        "assignment_rounds": 8,
        "seed_derivations": {"version": 1},
        "assignment_metric": {"identity": "adjusted_rand_score"},
        "feature_ids": feature_ids,
        "feature_inventory_sha256": "vmf-inventory",
    }
    artifact = {"canonical_digest": "vmf-canonical", "file_sha256": "vmf-file"}
    paths = {
        "compute_effect_final_resid_manifest": "manifest.json",
        "compute_effect_final_resid": "summary.json",
        "geometry_metrics_final_resid": "geometry.json",
        "vmf_pre_softcap_logits": "vmf.json",
    }
    return {
        "paths": paths,
        "payloads": {},
        "canonical_source_fingerprint": {
            "schema_version": 2,
            "algorithm": "sha256",
            "digest": "source",
            "components": {"manifest_sha256": "manifest"},
        },
        "source_identity": {
            "canonical_source": {
                "schema_version": 2,
                "algorithm": "sha256",
                "digest": "source",
                "components": {"manifest_sha256": "manifest"},
            },
            "manifest": {
                "path": "manifest.json",
                "canonical_digest": "manifest-canonical",
                "file_sha256": "manifest-file",
            },
            "summary": {
                "path": "summary.json",
                "canonical_digest": "summary-canonical",
                "file_sha256": "summary-file",
            },
            "tensor_shards": [{"path": "shard.pt", "file_sha256": "shard-file"}],
            "gram": {
                "path": "gram.pt",
                "file_sha256": "gram-file",
                "metadata": {"gram_sha256": "gram-metadata"},
            },
        },
        "vmf_scientific_fingerprint": scientific,
        "standalone_vmf_identity": {
            "public_schema_version": 1,
            "scientific_fingerprint": scientific,
            "artifact": artifact,
            "feature_ids": feature_ids,
            "candidate_schedule": [1, 2, 3, 4],
            "assignment_schedule": {
                "fraction": 0.8,
                "rounds": 8,
                "seed_derivations": {"version": 1},
                "metric": {"identity": "adjusted_rand_score"},
            },
        },
        "feature_ids": feature_ids,
        "input_artifact_hashes": {
            "geometry_metrics": {
                "canonical_digest": "geometry-canonical",
                "file_sha256": "geometry-file",
            },
            "standalone_vmf": artifact,
        },
    }


def _scientific_payload(config: FEGAPipelineConfig) -> dict:
    """Read the public artifact while excluding non-scientific execution telemetry."""
    # Worker, resume, checkpoint reuse, and wall timing cannot affect science equality.
    payload = json.loads(stability_scores_path(config).read_text())
    payload.pop("execution", None)
    return payload


def test_real_selected_family_run_is_deterministic_resumable_and_dependency_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise point loading through bounded execution and real checkpoint reuse."""
    # Build real source files and a durable reporting-owned point authority first.
    config = _config(tmp_path, workers=1, resume=False)
    generator = torch.Generator().manual_seed(701)
    rows = {
        feature_id: torch.randn(8, 4, generator=generator)
        for feature_id in (1, 2)
    }
    _write_compute_effect_artifacts(config, rows)
    records, summary = _point_records([1, 2], n_valid=8)
    inputs = _point_inputs([1, 2])
    monkeypatch.setattr(
        reporting_artifacts, "load_point_geometry_inputs", lambda *_args: inputs
    )
    monkeypatch.setattr(
        reporting_records,
        "build_point_geometry_records",
        lambda *_args: (copy.deepcopy(records), copy.deepcopy(summary)),
    )
    reporting_artifacts.load_build_and_write_point_geometry_records(config)

    # Make every forbidden model/vMF/factor/classification boundary fail if reached.
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        """Return a sentinel that records and rejects one forbidden dependency call."""
        # Stability must finish every real run without crossing these boundaries.
        def fail(*_args, **_kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"forbidden stability dependency reached: {name}")

        return fail

    monkeypatch.setattr(vmf_metrics, "score_vmf_feature", forbidden("vmf_score"))
    monkeypatch.setattr(vmf_metrics, "select_by_bic", forbidden("bic_selection"))
    monkeypatch.setattr(
        vmf_metrics, "assignment_stability", forbidden("assignment_refit")
    )
    monkeypatch.setattr(classifier, "classify_record", forbidden("classification"))
    monkeypatch.setattr(
        resources.ModelResources, "__init__", forbidden("model_resources")
    )
    monkeypatch.setattr(
        vmf_runner._BoundedLinearMaterializer,
        "__init__",
        forbidden("materialization"),
    )
    monkeypatch.setattr(
        factor_reuse.FeatureFactor,
        "build",
        classmethod(forbidden("factor_creation")),
    )

    # Run sequentially, then construct a real one-record checkpoint for resume.
    stability_runner.run_stability(config)
    sequential = _scientific_payload(config)
    point_bundle = reporting_artifacts.load_point_geometry_records(config)
    stability_inputs = load_stability_inputs(config, "final_resid")
    schedules = stability_runner._build_schedules(
        config,
        stability_inputs,
        point_bundle,
        base_seed=int(
            config.phases.stability.seed
            if config.phases.stability.seed is not None
            else config.seed.global_
        ),
    )
    fingerprint = build_selected_family_checkpoint_fingerprint(
        config=config, point_bundle=point_bundle, schedules=schedules
    )
    first_record = sequential["effect_spaces"]["final_resid"]["per_feature"]["1"]
    checkpoint = build_selected_family_checkpoint_payload(
        fingerprint=fingerprint,
        records=[first_record],
        schedules=schedules,
    )
    write_stability_checkpoint(config, checkpoint)

    config.phases.stability.workers = 2
    config.phases.stability.resume = True
    stability_runner.run_stability(config)
    resumed = _scientific_payload(config)
    resumed_execution = json.loads(stability_scores_path(config).read_text())[
        "execution"
    ]
    assert resumed_execution["checkpoint_load"]["records_reused"] == 1

    # A clean bounded parallel run must reproduce both sequential and resumed science.
    config.phases.stability.resume = False
    stability_runner.run_stability(config)
    parallel = _scientific_payload(config)
    assert sequential == resumed == parallel
    assert forbidden_calls == []
    for record in parallel["effect_spaces"]["final_resid"]["per_feature"].values():
        evidence = record["selected_family_evidence"]
        assert "decision" not in evidence
        assert evidence["protocols"]["low_context_qualification"]["status"] == (
            "exploratory"
        )
