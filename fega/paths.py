from __future__ import annotations

from pathlib import Path

from fega.config_schema import FEGAPipelineConfig


def _single_entity_attr(config: FEGAPipelineConfig) -> tuple[str, str]:
    if len(config.entity_attribute_selection) != 1:
        raise ValueError("Expected exactly one entity in selection.")
    entity, attrs = next(iter(config.entity_attribute_selection.items()))
    if len(attrs) != 1:
        raise ValueError("Expected exactly one attribute in selection.")
    return entity, attrs[0]


def run_root(config: FEGAPipelineConfig) -> Path:
    """Root directory for a pipeline run derived from the reference JSON."""
    ref = config.reference_json
    ref_group = ref.parent.name
    ref_basename = ref.name
    return config.output_root / ref_group / ref_basename


def attr_root(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Per-attribute root under the run root."""
    ent, at = (entity, attr) if entity and attr else _single_entity_attr(config)
    return run_root(config) / f"{ent}_{at}"


def data_prep_dir(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Root directory for data-prep-owned collection, selection, and Gram artifacts."""
    entity, attr = (entity, attr) if entity and attr else _single_entity_attr(config)
    return attr_root(config, entity, attr) / "data_prep"


def data_prep_collect_dir(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Directory for data-prep collection outputs."""
    return data_prep_dir(config, entity, attr) / "collect"


def data_prep_activations_dir(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Directory for activation chunks and manifest written by data-prep collection."""
    return data_prep_collect_dir(config, entity, attr) / "activations"


def data_prep_pairs_path(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Canonical prompt-pair artifact written by data-prep collection."""
    return data_prep_collect_dir(config, entity, attr) / "pairs_full.json"


def data_prep_select_dir(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Directory for data-prep feature/context selection outputs."""
    return data_prep_dir(config, entity, attr) / "select"


def gram_cache_dir(config: FEGAPipelineConfig) -> Path:
    """Directory for dense residual-space Gram cache artifacts."""
    return data_prep_dir(config) / "gram_cache"


def gram_cache_tensor_path(config: FEGAPipelineConfig) -> Path:
    """Path for the dense Gram tensor."""
    return gram_cache_dir(config) / "gram.pt"


def gram_cache_meta_path(config: FEGAPipelineConfig) -> Path:
    """Path for Gram cache metadata."""
    return gram_cache_dir(config) / "gram_meta.json"


def compute_effect_dir(
    config: FEGAPipelineConfig, entity: str | None = None, attr: str | None = None
) -> Path:
    """Root directory for reusable effect tensor artifacts."""
    entity, attr = (entity, attr) if entity and attr else _single_entity_attr(config)
    return attr_root(config, entity, attr) / "compute_effect"


def compute_effect_readout_dir(
    config: FEGAPipelineConfig,
    readout: str,
    entity: str | None = None,
    attr: str | None = None,
) -> Path:
    """Directory for one readout-native compute_effect artifact set."""
    if not isinstance(readout, str) or not readout:
        raise ValueError("readout must be a non-empty string.")
    if readout in {".", ".."} or any(sep in readout for sep in ("/", "\\")):
        raise ValueError(f"Invalid readout name: {readout!r}")
    return compute_effect_dir(config, entity, attr) / readout


def _compute_effect_artifact_dir(
    config: FEGAPipelineConfig, readout: str | None
) -> Path:
    return (
        compute_effect_readout_dir(config, readout)
        if readout
        else compute_effect_dir(config)
    )


def effect_tensors_manifest_path(
    config: FEGAPipelineConfig, readout: str | None = None
) -> Path:
    """Path for the compute_effect tensor manifest."""
    return (
        _compute_effect_artifact_dir(config, readout) / "effect_tensors_manifest.json"
    )


def effect_summary_path(config: FEGAPipelineConfig, readout: str | None = None) -> Path:
    """Path for the compute_effect summary artifact."""
    return _compute_effect_artifact_dir(config, readout) / "effect_summary.json"


def effect_tensor_shard_path(
    config: FEGAPipelineConfig, shard_idx: int, readout: str | None = None
) -> Path:
    """Path for one compute_effect tensor shard."""
    return (
        _compute_effect_artifact_dir(config, readout)
        / f"effect_tensors_{shard_idx:05d}.pt"
    )


def _validate_metric(metric: str) -> str:
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string.")
    if metric in {".", ".."} or any(sep in metric for sep in ("/", "\\")):
        raise ValueError(f"Invalid metric name: {metric!r}")
    if not metric.replace("_", "").isalnum():
        raise ValueError(f"Invalid metric name: {metric!r}")
    return metric


def geometry_metrics_dir(config: FEGAPipelineConfig, effect_space: str | None = None) -> Path:
    """Directory for directional-concentration outputs in one effect space."""
    entity, attr = _single_entity_attr(config)
    space = effect_space or config.phases.geometry_metrics.effect_space
    metric = _validate_metric(space)
    return attr_root(config, entity, attr) / "geometry_metrics" / metric


def geometry_metrics_scores_path(
    config: FEGAPipelineConfig, effect_space: str | None = None
) -> Path:
    """Path for geometry_metrics per-feature score output."""
    return geometry_metrics_dir(config, effect_space) / "geometry_metrics_scores.json"


def vmf_dir(config: FEGAPipelineConfig, effect_space: str | None = None) -> Path:
    """Directory for vMF multimodality outputs in one effect space."""
    entity, attr = _single_entity_attr(config)
    space = effect_space or config.phases.vmf.effect_space
    metric = _validate_metric(space)
    return attr_root(config, entity, attr) / "vmf" / metric


def vmf_scores_path(
    config: FEGAPipelineConfig, effect_space: str | None = None
) -> Path:
    """Path for vMF per-feature score output."""
    return vmf_dir(config, effect_space) / "vmf_scores.json"


def vmf_checkpoint_path(
    config: FEGAPipelineConfig, effect_space: str | None = None
) -> Path:
    """Path for vMF partial resume checkpoint state."""
    return vmf_dir(config, effect_space) / "vmf_checkpoint.json"


def stability_dir(config: FEGAPipelineConfig) -> Path:
    """Directory for FEGA Section 12 stability diagnostics."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "stability"


def stability_scores_path(config: FEGAPipelineConfig) -> Path:
    """Path for the combined stability diagnostic artifact."""
    return stability_dir(config) / "stability_scores.json"


def stability_checkpoint_path(config: FEGAPipelineConfig) -> Path:
    """Path for stability partial resume checkpoint state."""
    return stability_dir(config) / "stability_checkpoint.json"


def geometry_reporting_dir(config: FEGAPipelineConfig) -> Path:
    """Directory for FEGA geometry-classification artifacts."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "geometry_reporting"


def geometry_reporting_records_path(config: FEGAPipelineConfig) -> Path:
    """Path for per-feature FEGA geometry-classification records."""
    return geometry_reporting_dir(config) / "geometry_feature_records.json"


def geometry_reporting_records_csv_path(config: FEGAPipelineConfig) -> Path:
    """Optional CSV mirror for FEGA geometry-classification records."""
    return geometry_reporting_dir(config) / "geometry_feature_records.csv"


def geometry_reporting_map_data_path(config: FEGAPipelineConfig) -> Path:
    """Path for FEGA diagnostic feature-map data."""
    return geometry_reporting_dir(config) / "geometry_map_data.json"


def geometry_reporting_stats_path(config: FEGAPipelineConfig) -> Path:
    """Path for the human-readable FEGA class statistics report."""
    return geometry_reporting_dir(config) / "geometry_reporting_stats.md"


def geometry_reporting_counts_path(config: FEGAPipelineConfig) -> Path:
    """Path for compact FEGA aggregate class/count rows."""
    return geometry_reporting_dir(config) / "geometry_reporting_counts.csv"


def geometry_reporting_gate_diagnostics_json_path(config: FEGAPipelineConfig) -> Path:
    """Path for FEGA gate-failure diagnostics as structured JSON."""
    return geometry_reporting_dir(config) / "geometry_gate_diagnostics.json"


def geometry_reporting_gate_diagnostics_md_path(config: FEGAPipelineConfig) -> Path:
    """Path for FEGA gate-failure diagnostics as a Markdown report."""
    return geometry_reporting_dir(config) / "geometry_gate_diagnostics.md"


def geometry_reporting_figures_dir(config: FEGAPipelineConfig) -> Path:
    """Directory for FEGA diagnostic feature-map figures."""
    return geometry_reporting_dir(config) / "figures"


def logs_dir(config: FEGAPipelineConfig) -> Path:
    """Directory for run logs."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "logs"


def run_metadata_path(config: FEGAPipelineConfig) -> Path:
    """Path to run_metadata.json under the run root."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "run_metadata.json"


def run_status_path(config: FEGAPipelineConfig) -> Path:
    """Path to run_status.json under the run root."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "run_status.json"


def config_used_path(config: FEGAPipelineConfig) -> Path:
    """Path to a copy of the resolved pipeline config."""
    entity, attr = _single_entity_attr(config)
    return attr_root(config, entity, attr) / "config_used.yaml"
