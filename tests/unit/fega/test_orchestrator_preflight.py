from __future__ import annotations

import json
from pathlib import Path

import pytest

import fega.orchestrator as orchestrator
from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_reporting import artifacts as reporting_artifacts


def _write_config(cfg_path: Path, reference_json: Path, output_root: Path) -> None:
    cfg_path.write_text(
        "\n".join(
            [
                f"reference_json: {reference_json}",
                f"output_root: {output_root}",
                "device: cpu",
                "entity_attribute_selection:",
                "  city: ['Country']",
            ]
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("phase", "expected_prefix"),
    [
        ("data_prep", "Missing MDBM checkpoint:"),
        ("compute_effect", "Missing inputs for compute_effect:"),
        ("geometry_metrics", "Missing inputs for geometry_metrics:"),
        ("vmf", "Missing inputs for vmf:"),
        ("stability", "Missing inputs for stability:"),
        ("geometry_reporting", "Missing inputs for geometry_reporting:"),
    ],
)
def test_phase_prerequisites_report_missing_scientific_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_prefix: str,
) -> None:
    reference_json = tmp_path / "ref.json"
    reference_json.write_text(json.dumps({"eval_config": {"model_name": "gpt2"}}))
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    monkeypatch.setattr(
        orchestrator,
        "resolve_mdbm_path",
        lambda *_args, **_kwargs: tmp_path / "missing_mdbm.pt",
    )

    reason = orchestrator._check_prerequisites(phase, config)

    assert reason is not None
    assert reason.startswith(expected_prefix)


def test_run_phase_dispatches_all_phases_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_json = tmp_path / "ref.json"
    reference_json.write_text(json.dumps({"eval_config": {"model_name": "gpt2"}}))
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    phases = [
        "data_prep",
        "compute_effect",
        "geometry_metrics",
        "vmf",
        "stability",
        "geometry_reporting",
    ]
    calls: list[str] = []
    for phase in phases:
        monkeypatch.setattr(
            orchestrator,
            f"run_{phase}",
            lambda _config, _resources, name=phase: calls.append(name),
        )

    for phase in phases:
        orchestrator._run_phase(phase, config, resources=None)

    assert calls == phases
    with pytest.raises(ValueError, match="Unknown phase"):
        orchestrator._run_phase("unsupported_phase", config, resources=None)


def test_scientific_readiness_uses_final_resid_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_json = tmp_path / "ref.json"
    reference_json.write_text(json.dumps({"eval_config": {"model_name": "gpt2"}}))
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)

    final_manifest = orchestrator.effect_tensors_manifest_path(config, "final_resid")
    final_summary = orchestrator.effect_summary_path(config, "final_resid")
    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    shard_path = final_manifest.parent / "effect_tensors_00000.pt"
    shard_path.write_bytes(b"canonical")
    gram_path = orchestrator.gram_cache_tensor_path(config)
    gram_path.parent.mkdir(parents=True, exist_ok=True)
    gram_path.write_bytes(b"gram")
    final_manifest.write_text(
        json.dumps(
            {
                "effect_space": "final_resid",
                "metric_space": "residual_gram",
                "inputs": {"gram_path": str(gram_path)},
                "shards": [{"path": str(shard_path)}],
            }
        )
    )
    final_summary.write_text(
        json.dumps(
            {
                "per_feature": {
                    "1": {
                        "feature_id": 1,
                        "tensor_shard": shard_path.name,
                    }
                }
            }
        )
    )
    geometry_metrics_path = orchestrator.geometry_metrics_scores_path(config, "final_resid")
    geometry_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_metrics_path.write_text(json.dumps({"per_feature": {"1": {}}}))
    monkeypatch.setattr(
        reporting_artifacts, "load_point_geometry_inputs", lambda _config: {}
    )

    assert orchestrator._check_vmf_inputs(config) is None
    assert orchestrator._check_stability_inputs(config) is None


def test_stability_preflight_surfaces_point_provenance_and_accepts_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface precise legacy-vMF regeneration and accept current point inputs."""
    # Isolate the post-source shared validator while preserving its public error text.
    reference_json = tmp_path / "ref.json"
    reference_json.write_text(json.dumps({"eval_config": {"model_name": "gpt2"}}))
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    monkeypatch.setattr(
        orchestrator, "_check_stability_effect_space", lambda *_args: None
    )
    failure = reporting_artifacts.StandaloneVmfRegenerationRequiredError(
        "unversioned_standalone_vmf_artifact", tmp_path / "vmf_scores.json"
    )

    def reject(_config: FEGAPipelineConfig) -> dict[str, object]:
        """Raise the shared loader's precise legacy-artifact failure."""
        # Preflight must expose this message without translating the reason.
        raise failure

    monkeypatch.setattr(reporting_artifacts, "load_point_geometry_inputs", reject)
    reason = orchestrator._check_stability_inputs(config)
    assert reason is not None
    assert "reason=unversioned_standalone_vmf_artifact" in reason

    monkeypatch.setattr(
        reporting_artifacts, "load_point_geometry_inputs", lambda _config: {}
    )
    assert orchestrator._check_stability_inputs(config) is None
