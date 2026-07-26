from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataPrepConfig:
    enabled: bool = True
    batch_size: int | None = None
    save_chunk_size: int | None = 512
    single_file: bool = False
    limit: int | None = None
    tau_act: float = 0.0
    max_contexts: int = 32
    min_contexts: int = 8
    readouts: list[str] = field(default_factory=lambda: ["final_resid"])
    gram_cache: bool = False
    gram_cache_dtype: str = "float64"


@dataclass
class ComputeEffectConfig:
    enabled: bool = False
    batch_size: int | None = 8
    min_coverage: int = 8
    normalization_eps: float = 1e-12
    tau_zero: float = 1e-12
    effect_shard_size: int = 4096
    cache_max_chunks: int = 2
    cache_max_bytes: int = 1073741824


@dataclass
class GeometryMetricsCRayConfig:
    enabled: bool = True
    method: str = "pairwise"
    store_r2: bool = True
    eps: float = 1e-12


@dataclass
class GeometryMetricsSpanConfig:
    enabled: bool = True
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])
    eps: float = 1e-12


@dataclass
class GeometryMetricsResidConfig:
    enabled: bool = False
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float = 1e-12


@dataclass
class GeometryMetricsEffectiveRankConfig:
    enabled: bool = False
    eps: float = 1e-12


@dataclass
class GeometryMetricsConfig:
    enabled: bool = False
    effect_space: str = "final_resid"
    c_ray: GeometryMetricsCRayConfig = field(
        default_factory=GeometryMetricsCRayConfig
    )
    span: GeometryMetricsSpanConfig = field(default_factory=GeometryMetricsSpanConfig)
    resid: GeometryMetricsResidConfig = field(default_factory=GeometryMetricsResidConfig)
    effective_rank: GeometryMetricsEffectiveRankConfig = field(
        default_factory=GeometryMetricsEffectiveRankConfig
    )


@dataclass
class DirectionalMixtureFitConfig:
    enabled: bool = False
    effect_space: str = "pre_softcap_logits"
    backend: str = "dense_cpu"
    gpu_device: str = "cuda:0"
    workers: int = 1
    max_vocab_buffers: int = 1
    resume: bool = False
    checkpoint_flush_features: int = 256
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    bic_tolerance: float = 1.0e-9
    resample_fraction: float = 0.80
    resample_rounds: int = 8
    seed: int | None = None
    n_init: int = 4
    max_iter: int = 200


@dataclass
class StabilityScalarConfig:
    enabled: bool = True
    bootstrap_rounds: int = 200
    ci_quantiles: list[float] = field(default_factory=lambda: [0.025, 0.975])


@dataclass
class StabilitySubspaceConfig:
    enabled: bool = True
    k_values: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    resample_fraction: float = 0.75
    resample_rounds: int = 20
    angle_p90_quantile: float = 0.90
    eig_floor: float = 1.0e-8


@dataclass
class StabilitySampleSizeConfig:
    enabled: bool = True
    target_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    subset_rounds: int = 20
    strong_subset_rounds: int = 50
    max_enumerated_subsets: int = 20


@dataclass
class StabilityLeaveOutConfig:
    enabled: bool = True
    max_leave_two_out_pairs: int = 20
    min_group_count: int = 2
    min_group_size: int = 2


@dataclass
class StabilityConfig:
    enabled: bool = False
    effect_space: str = "final_resid"
    workers: int = 1
    resume: bool = False
    checkpoint_flush_features: int = 256
    seed: int | None = None
    scalar: StabilityScalarConfig = field(default_factory=StabilityScalarConfig)
    subspace: StabilitySubspaceConfig = field(default_factory=StabilitySubspaceConfig)
    sample_size: StabilitySampleSizeConfig = field(
        default_factory=StabilitySampleSizeConfig
    )
    leave_out: StabilityLeaveOutConfig = field(default_factory=StabilityLeaveOutConfig)


@dataclass
class GeometryReportingConfig:
    enabled: bool = False
    threshold_profile: str = "paper"
    map_enabled: bool = True
    atlas_include_insufficient_evidence: bool = True
    global_flag_mode: str = "layered_overlay"
    embedding: str = "auto"
    seed: int | None = None
    eps: float = 1.0e-12
    write_csv: bool = False


@dataclass
class SeedConfig:
    global_: int = 42
    selection_seed: int | None = None


@dataclass
class InductionConfig:
    summary_json: Path
    feature_set: str = "candidate"
    explicit_feature_ids: list[int] | None = None
    sae_uid: str | None = None
    require_model_correct: bool | None = None
    answer_prefix: str | None = None
    source_activation_threshold: float | None = None
    max_source_contexts: int | None = None
    max_source_examples: int | None = None
    stratify_by: list[str] = field(
        default_factory=lambda: ["context_id", "support_example_index"]
    )
    materialize_only_selected: bool = True


@dataclass
class PhasesConfig:
    data_prep: DataPrepConfig = field(default_factory=DataPrepConfig)
    compute_effect: ComputeEffectConfig = field(default_factory=ComputeEffectConfig)
    geometry_metrics: GeometryMetricsConfig = field(default_factory=GeometryMetricsConfig)
    vmf: DirectionalMixtureFitConfig = field(default_factory=DirectionalMixtureFitConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    geometry_reporting: GeometryReportingConfig = field(default_factory=GeometryReportingConfig)


@dataclass
class FEGAPipelineConfig:
    reference_json: Path
    output_root: Path
    device: str
    entity_attribute_selection: dict[str, list[str]]
    source_kind: str = "ravel"
    run_id: str | None = None
    cache_dir: Path | None = None
    download_saes_dir: Path | None = None
    sae_repo_id: str | None = None
    sae_source: str = "auto"
    local_sae_checkpoint_path: Path | None = None
    local_sae_resolved_config_path: Path | None = None
    llm_batch_size_override: int | None = None
    mdbm_root: Path | None = None
    mdbm_weight_path: Path | None = None
    seed: SeedConfig = field(default_factory=SeedConfig)
    reuse_model_across_phases: bool = False
    phases: PhasesConfig = field(default_factory=PhasesConfig)
    induction: InductionConfig | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> FEGAPipelineConfig:
        """Load and validate a pipeline config from YAML."""
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found at {cfg_path}")
        raw = _load_raw_config(cfg_path)
        base_dir = cfg_path.parent

        source_kind = _coerce_source_kind(raw.get("source_kind", "ravel"))
        reference_json = _require_path(raw, "reference_json", base_dir, must_exist=True)
        # output_root may be relative to the current working directory (common in scripts).
        output_root = _require_path(raw, "output_root", base_dir, prefer_cwd=True)
        _validate_output_root(output_root)
        device = raw.get("device")
        if not device:
            raise ValueError("`device` must be provided in the pipeline config.")

        cache_dir = _optional_path(raw.get("cache_dir"), base_dir)
        download_saes_dir = _optional_path(
            raw.get("download_saes_dir"), base_dir, prefer_cwd=True
        )
        sae_repo_id = raw.get("sae_repo_id")
        sae_source = _coerce_sae_source(raw.get("sae_source", "auto"))
        local_sae_checkpoint_path = _optional_path(
            raw.get("local_sae_checkpoint_path"), base_dir
        )
        local_sae_resolved_config_path = _optional_path(
            raw.get("local_sae_resolved_config_path"), base_dir
        )
        llm_batch_size_override = _coerce_optional_positive_int(
            raw.get("llm_batch_size_override"), "llm_batch_size_override"
        )
        mdbm_root = _optional_path(raw.get("mdbm_root"), base_dir)
        mdbm_weight_path = _optional_path(raw.get("mdbm_weight_path"), base_dir)
        run_id = raw.get("run_id")

        _validate_sae_source_paths(
            sae_source=sae_source,
            local_checkpoint_path=local_sae_checkpoint_path,
            local_resolved_config_path=local_sae_resolved_config_path,
        )

        eas = raw.get("entity_attribute_selection")
        validated_eas = _validate_entity_attribute_selection(eas)

        seed_cfg = _build_seed_config(raw.get("seed", {}))

        phases_cfg = _build_phases_config(raw.get("phases", {}))
        induction_cfg = _build_induction_config(
            raw.get("induction"), base_dir, source_kind=source_kind
        )
        if source_kind == "induction":
            _validate_induction_namespace(validated_eas)
            assert induction_cfg is not None
            _validate_induction_summary_config(induction_cfg)

        cfg = cls(
            reference_json=reference_json,
            output_root=output_root,
            device=device,
            entity_attribute_selection=validated_eas,
            source_kind=source_kind,
            run_id=run_id,
            cache_dir=cache_dir,
            download_saes_dir=download_saes_dir,
            sae_repo_id=sae_repo_id,
            sae_source=sae_source,
            local_sae_checkpoint_path=local_sae_checkpoint_path,
            local_sae_resolved_config_path=local_sae_resolved_config_path,
            llm_batch_size_override=llm_batch_size_override,
            mdbm_root=mdbm_root,
            mdbm_weight_path=mdbm_weight_path,
            seed=seed_cfg,
            reuse_model_across_phases=bool(raw.get("reuse_model_across_phases", False)),
            phases=phases_cfg,
            induction=induction_cfg,
        )
        # Stash source path for downstream provenance.
        setattr(cfg, "_config_path", cfg_path)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a plain Python dict (paths as strings)."""
        return _convert_paths(asdict(self))


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in {".yml", ".yaml"}:
        raise ValueError(
            "Pipeline config must be a YAML file with .yml or .yaml extension."
        )
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configs.")
    text = path.read_text()
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("Pipeline config must be a mapping at the top level.")
    return dict(loaded)


def _optional_path(
    value: Any, base_dir: Path, *, prefer_cwd: bool = False
) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, base_dir, prefer_cwd=prefer_cwd)


def _require_path(
    raw: Mapping[str, Any],
    key: str,
    base_dir: Path,
    *,
    must_exist: bool = False,
    prefer_cwd: bool = False,
) -> Path:
    value = raw.get(key)
    if value is None:
        raise ValueError(f"Missing required config key `{key}`.")
    path = _optional_path(value, base_dir, prefer_cwd=prefer_cwd)  # type: ignore[arg-type]
    assert path is not None
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path for `{key}` does not exist: {path}")
    return path


def _validate_entity_attribute_selection(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            "`entity_attribute_selection` must be a mapping of entity to attributes."
        )
    if len(raw) != 1:
        raise ValueError(
            "`entity_attribute_selection` must contain exactly one entity for now."
        )
    entity, attrs = next(iter(raw.items()))
    if not isinstance(entity, str) or not isinstance(attrs, list) or not attrs:
        raise ValueError(
            "`entity_attribute_selection` must map entity name to a non-empty list of attributes."
        )
    attr_list = []
    for attr in attrs:
        if not isinstance(attr, str) or not attr:
            raise ValueError("Attributes must be non-empty strings.")
        attr_list.append(attr)
    return {entity: attr_list}


def _coerce_source_kind(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("`source_kind` must be `ravel` or `induction`.")
    source_kind = value.strip().lower()
    if source_kind not in {"ravel", "induction"}:
        raise ValueError("`source_kind` must be `ravel` or `induction`.")
    return source_kind


def _build_induction_config(
    raw_section: Any, base_dir: Path, *, source_kind: str
) -> InductionConfig | None:
    if source_kind != "induction":
        if raw_section is not None:
            raise ValueError(
                "`induction` config is only valid with `source_kind: induction`."
            )
        return None
    if raw_section is None:
        raise ValueError("`induction.summary_json` is required for induction source.")
    if not isinstance(raw_section, Mapping):
        raise ValueError("`induction` must be a mapping.")

    raw = dict(raw_section)
    legacy_keys = {"max_contexts", "max_examples"}
    present_legacy = sorted(legacy_keys.intersection(raw))
    if present_legacy:
        rendered = ", ".join(f"`induction.{key}`" for key in present_legacy)
        raise ValueError(
            f"{rendered} are not accepted. Use `induction.max_source_contexts` / "
            "`induction.max_source_examples` for source scan caps, and "
            "`phases.data_prep.max_contexts` / `phases.data_prep.min_contexts` for "
            "selected-context caps."
        )
    forbidden_namespace = {"namespace_entity", "namespace_attribute"}
    present_namespace = sorted(forbidden_namespace.intersection(raw))
    if present_namespace:
        rendered = ", ".join(f"`induction.{key}`" for key in present_namespace)
        raise ValueError(
            f"{rendered} are not accepted. Use `entity_attribute_selection` as the "
            "sole induction namespace source."
        )
    _reject_unknown_keys(raw, InductionConfig, "induction")

    summary_json = _require_path(raw, "summary_json", base_dir, must_exist=True)
    feature_set = str(raw.get("feature_set", InductionConfig.feature_set)).strip()
    if feature_set not in {"candidate", "strict_common", "explicit"}:
        raise ValueError(
            "`induction.feature_set` must be `candidate`, `strict_common`, or `explicit`."
        )
    explicit_feature_ids = _coerce_optional_feature_ids(
        raw.get("explicit_feature_ids"), feature_set=feature_set
    )
    require_model_correct = _coerce_optional_bool(
        raw.get("require_model_correct"), "induction.require_model_correct"
    )
    answer_prefix = raw.get("answer_prefix")
    if answer_prefix is not None and not isinstance(answer_prefix, str):
        raise ValueError("`induction.answer_prefix` must be a string when provided.")
    source_activation_threshold = _coerce_optional_nonnegative_float(
        raw.get("source_activation_threshold"),
        "induction.source_activation_threshold",
    )
    max_source_contexts = _coerce_optional_positive_int(
        raw.get("max_source_contexts"), "induction.max_source_contexts"
    )
    max_source_examples = _coerce_optional_positive_int(
        raw.get("max_source_examples"), "induction.max_source_examples"
    )
    stratify_by = _coerce_nonempty_string_list(
        raw.get("stratify_by", ["context_id", "support_example_index"]),
        "induction.stratify_by",
    )
    materialize_only_selected = _coerce_bool(
        raw.get("materialize_only_selected", True),
        "induction.materialize_only_selected",
    )
    if not materialize_only_selected:
        raise ValueError(
            "`induction.materialize_only_selected` must be true; induction data prep "
            "materializes only the selected prompt union."
        )
    sae_uid = raw.get("sae_uid")
    if sae_uid is not None and (not isinstance(sae_uid, str) or not sae_uid.strip()):
        raise ValueError(
            "`induction.sae_uid` must be a non-empty string when provided."
        )

    return InductionConfig(
        summary_json=summary_json,
        feature_set=feature_set,
        explicit_feature_ids=explicit_feature_ids,
        sae_uid=sae_uid.strip() if isinstance(sae_uid, str) else None,
        require_model_correct=require_model_correct,
        answer_prefix=answer_prefix,
        source_activation_threshold=source_activation_threshold,
        max_source_contexts=max_source_contexts,
        max_source_examples=max_source_examples,
        stratify_by=stratify_by,
        materialize_only_selected=materialize_only_selected,
    )


def _validate_induction_namespace(eas: dict[str, list[str]]) -> None:
    if len(eas) != 1:
        raise ValueError(
            "Induction configs require exactly one `entity_attribute_selection` entity."
        )
    _, attrs = next(iter(eas.items()))
    if len(attrs) != 1:
        raise ValueError(
            "Induction configs require exactly one selected attribute in "
            "`entity_attribute_selection`."
        )


def _validate_induction_summary_config(induction: InductionConfig) -> None:
    try:
        with open(induction.summary_json) as f:
            summary = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Could not read induction summary JSON: {induction.summary_json} ({exc})"
        ) from exc
    feature_sets = summary.get("feature_sets")
    if not isinstance(feature_sets, Mapping) or not feature_sets:
        raise ValueError(
            "Induction summary must contain a non-empty `feature_sets` map."
        )
    if len(feature_sets) > 1 and induction.sae_uid is None:
        raise ValueError(
            "`induction.sae_uid` is required when summary_json contains multiple SAE "
            "feature sets."
        )
    if induction.sae_uid is not None and induction.sae_uid not in feature_sets:
        raise ValueError(
            f"`induction.sae_uid` {induction.sae_uid!r} not found in summary feature_sets."
        )


def _build_seed_config(raw_seed: Any) -> SeedConfig:
    raw_seed = raw_seed or {}
    if not isinstance(raw_seed, Mapping):
        raise ValueError("`seed` must be a mapping if provided.")
    normalized = dict(raw_seed)
    if "global" in normalized and "global_" not in normalized:
        normalized["global_"] = normalized.pop("global")
    _reject_unknown_keys(normalized, SeedConfig, "seed")
    global_seed = _coerce_seed(
        normalized.get("global_", SeedConfig.global_), "seed.global", allow_none=False
    )
    selection_seed = _coerce_seed(
        normalized.get("selection_seed"), "seed.selection_seed", allow_none=True
    )
    assert global_seed is not None
    return SeedConfig(global_=global_seed, selection_seed=selection_seed)


def _build_phases_config(raw_phases: Any) -> PhasesConfig:
    raw_phases = raw_phases or {}
    if not isinstance(raw_phases, Mapping):
        raise ValueError("`phases` must be a mapping if provided.")
    _reject_unknown_keys(raw_phases, PhasesConfig, "phases")

    data_prep_cfg = _build_data_prep_config(raw_phases.get("data_prep"))
    compute_effect_cfg = _build_compute_effect_config(raw_phases.get("compute_effect"))
    geometry_metrics_cfg = _build_geometry_metrics_config(raw_phases.get("geometry_metrics"))
    vmf_cfg = _build_vmf_config(raw_phases.get("vmf"))
    stability_cfg = _build_stability_config(raw_phases.get("stability"))
    geometry_reporting_cfg = _build_geometry_reporting_config(raw_phases.get("geometry_reporting"))
    _validate_compute_effect_prereq_config(data_prep_cfg, compute_effect_cfg)
    _validate_downstream_readout_dependencies(
        data_prep_cfg=data_prep_cfg,
        compute_effect_cfg=compute_effect_cfg,
        geometry_metrics_cfg=geometry_metrics_cfg,
        vmf_cfg=vmf_cfg,
        stability_cfg=stability_cfg,
    )

    return PhasesConfig(
        data_prep=data_prep_cfg,
        compute_effect=compute_effect_cfg,
        geometry_metrics=geometry_metrics_cfg,
        vmf=vmf_cfg,
        stability=stability_cfg,
        geometry_reporting=geometry_reporting_cfg,
    )


def _build_data_prep_config(raw_section: Any) -> DataPrepConfig:
    cfg = _build_section(raw_section, DataPrepConfig)
    if cfg.batch_size is not None:
        cfg.batch_size = _coerce_positive_int(
            cfg.batch_size, "phases.data_prep.batch_size"
        )
    cfg.readouts = _coerce_readouts(cfg.readouts)
    cfg.gram_cache_dtype = _coerce_gram_cache_dtype(cfg.gram_cache_dtype)
    if cfg.gram_cache and "final_resid" not in cfg.readouts:
        raise ValueError(
            "`phases.data_prep.gram_cache` requires `final_resid` in "
            "`phases.data_prep.readouts`."
        )
    return cfg


def _build_compute_effect_config(raw_section: Any) -> ComputeEffectConfig:
    cfg = _build_section(raw_section, ComputeEffectConfig)
    cfg.min_coverage = _coerce_positive_int(
        cfg.min_coverage, "phases.compute_effect.min_coverage"
    )
    cfg.effect_shard_size = _coerce_positive_int(
        cfg.effect_shard_size, "phases.compute_effect.effect_shard_size"
    )
    cfg.cache_max_chunks = _coerce_nonnegative_int(
        cfg.cache_max_chunks, "phases.compute_effect.cache_max_chunks"
    )
    cfg.cache_max_bytes = _coerce_nonnegative_int(
        cfg.cache_max_bytes, "phases.compute_effect.cache_max_bytes"
    )
    cfg.normalization_eps = _coerce_positive_float(
        cfg.normalization_eps, "phases.compute_effect.normalization_eps"
    )
    cfg.tau_zero = _coerce_nonnegative_float(
        cfg.tau_zero, "phases.compute_effect.tau_zero"
    )
    if cfg.batch_size is not None:
        cfg.batch_size = _coerce_positive_int(
            cfg.batch_size, "phases.compute_effect.batch_size"
        )
    return cfg


def _build_geometry_metrics_config(raw_section: Any) -> GeometryMetricsConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryMetricsConfig.")
    _reject_unknown_keys(raw, GeometryMetricsConfig, "phases.geometry_metrics")

    c_ray_cfg = _build_geometry_metrics_c_ray_config(raw.get("c_ray"))
    span_cfg = _build_geometry_metrics_span_config(raw.get("span"))
    resid_cfg = _build_geometry_metrics_resid_config(raw.get("resid"))
    effective_rank_cfg = _build_geometry_metrics_effective_rank_config(raw.get("effective_rank"))
    cfg = GeometryMetricsConfig(
        enabled=bool(raw.get("enabled", GeometryMetricsConfig.enabled)),
        effect_space=raw.get("effect_space", GeometryMetricsConfig.effect_space),
        c_ray=c_ray_cfg,
        span=span_cfg,
        resid=resid_cfg,
        effective_rank=effective_rank_cfg,
    )
    if cfg.effect_space != "final_resid":
        raise ValueError("`phases.geometry_metrics.effect_space` must be `final_resid`.")
    if cfg.effective_rank.enabled:
        if cfg.effect_space != "final_resid":
            raise ValueError(
                "`phases.geometry_metrics.effective_rank.enabled` requires "
                "`effect_space: final_resid`."
            )
        if not cfg.span.enabled:
            raise ValueError(
                "`phases.geometry_metrics.effective_rank.enabled` requires "
                "`phases.geometry_metrics.span.enabled: true`."
            )
        if not cfg.resid.enabled:
            raise ValueError(
                "`phases.geometry_metrics.effective_rank.enabled` requires "
                "`phases.geometry_metrics.resid.enabled: true`."
            )
    return cfg


def _build_geometry_metrics_c_ray_config(raw_section: Any) -> GeometryMetricsCRayConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryMetricsCRayConfig.")
    _reject_unknown_keys(raw, GeometryMetricsCRayConfig, "phases.geometry_metrics.c_ray")
    cfg = GeometryMetricsCRayConfig(**raw)
    if cfg.method not in {"pairwise", "fast_formula"}:
        raise ValueError(
            "`phases.geometry_metrics.c_ray.method` must be `pairwise` or `fast_formula`."
        )
    cfg.eps = _coerce_positive_float(cfg.eps, "phases.geometry_metrics.c_ray.eps")
    cfg.enabled = bool(cfg.enabled)
    cfg.store_r2 = bool(cfg.store_r2)
    return cfg


def _build_geometry_metrics_span_config(raw_section: Any) -> GeometryMetricsSpanConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryMetricsSpanConfig.")
    _reject_unknown_keys(raw, GeometryMetricsSpanConfig, "phases.geometry_metrics.span")
    cfg = GeometryMetricsSpanConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.eps = _coerce_positive_float(cfg.eps, "phases.geometry_metrics.span.eps")
    cfg.k_values = _coerce_unique_positive_ints(
        cfg.k_values, "phases.geometry_metrics.span.k_values"
    )
    return cfg


def _build_geometry_metrics_resid_config(raw_section: Any) -> GeometryMetricsResidConfig:
    """Build centered-residual diagnostics for the supported FEGA k subset."""
    # Parse generic positive integers first, then enforce the residual-only boundary.
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryMetricsResidConfig.")
    _reject_unknown_keys(raw, GeometryMetricsResidConfig, "phases.geometry_metrics.resid")
    cfg = GeometryMetricsResidConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.eps = _coerce_positive_float(cfg.eps, "phases.geometry_metrics.resid.eps")
    cfg.k_values = _coerce_unique_positive_ints(
        cfg.k_values, "phases.geometry_metrics.resid.k_values"
    )
    if any(k not in {1, 2, 3, 4} for k in cfg.k_values):
        raise ValueError(
            "`phases.geometry_metrics.resid.k_values` must be a positive unique subset "
            "of [1, 2, 3, 4]."
        )
    return cfg


def _build_geometry_metrics_effective_rank_config(
    raw_section: Any,
) -> GeometryMetricsEffectiveRankConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryMetricsEffectiveRankConfig.")
    _reject_unknown_keys(
        raw, GeometryMetricsEffectiveRankConfig, "phases.geometry_metrics.effective_rank"
    )
    cfg = GeometryMetricsEffectiveRankConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.eps = _coerce_positive_float(cfg.eps, "phases.geometry_metrics.effective_rank.eps")
    return cfg


def _build_vmf_config(raw_section: Any) -> DirectionalMixtureFitConfig:
    """Build the operational vMF fitting configuration without reporting gates.

    The retained fields control resources, candidate fitting, numerical BIC tie
    handling, and deterministic assignment resampling. Scientific acceptance
    thresholds belong exclusively to geometry reporting.
    """
    # Reject removed fit-stage fields before coercing the retained controls.
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section DirectionalMixtureFitConfig.")
    _reject_unknown_keys(raw, DirectionalMixtureFitConfig, "phases.vmf")
    cfg = DirectionalMixtureFitConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    if cfg.effect_space != "pre_softcap_logits":
        raise ValueError(
            "`phases.vmf.effect_space` must be `pre_softcap_logits`."
        )
    if isinstance(cfg.workers, bool) or isinstance(cfg.workers, float):
        raise ValueError(
            f"`phases.vmf.workers` must be an integer > 0, got {cfg.workers!r}."
        )
    cfg.workers = _coerce_positive_int(cfg.workers, "phases.vmf.workers")
    if isinstance(cfg.max_vocab_buffers, bool) or isinstance(
        cfg.max_vocab_buffers, float
    ):
        raise ValueError(
            "`phases.vmf.max_vocab_buffers` must be an integer > 0, "
            f"got {cfg.max_vocab_buffers!r}."
        )
    cfg.max_vocab_buffers = _coerce_positive_int(
        cfg.max_vocab_buffers, "phases.vmf.max_vocab_buffers"
    )
    cfg.resume = _coerce_bool(cfg.resume, "phases.vmf.resume")
    if isinstance(cfg.checkpoint_flush_features, bool) or isinstance(
        cfg.checkpoint_flush_features, float
    ):
        raise ValueError(
            "`phases.vmf.checkpoint_flush_features` must be an integer > 0, "
            f"got {cfg.checkpoint_flush_features!r}."
        )
    cfg.checkpoint_flush_features = _coerce_positive_int(
        cfg.checkpoint_flush_features, "phases.vmf.checkpoint_flush_features"
    )
    cfg.k_values = _coerce_unique_positive_ints(cfg.k_values, "phases.vmf.k_values")
    if any(k not in {1, 2, 3, 4} for k in cfg.k_values):
        raise ValueError(
            "`phases.vmf.k_values` must be a positive unique subset of [1, 2, 3, 4]."
        )
    cfg.bic_tolerance = _coerce_nonnegative_float(
        cfg.bic_tolerance, "phases.vmf.bic_tolerance"
    )
    cfg.resample_fraction = _coerce_unit_interval_float(
        cfg.resample_fraction, "phases.vmf.resample_fraction", allow_zero=False
    )
    cfg.resample_rounds = _coerce_nonnegative_int(
        cfg.resample_rounds, "phases.vmf.resample_rounds"
    )
    cfg.seed = _coerce_seed(cfg.seed, "phases.vmf.seed", allow_none=True)
    cfg.n_init = _coerce_positive_int(cfg.n_init, "phases.vmf.n_init")
    cfg.max_iter = _coerce_positive_int(cfg.max_iter, "phases.vmf.max_iter")
    return cfg


def _build_stability_config(raw_section: Any) -> StabilityConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section StabilityConfig.")
    _reject_unknown_keys(raw, StabilityConfig, "phases.stability")

    cfg = StabilityConfig(
        enabled=bool(raw.get("enabled", StabilityConfig.enabled)),
        effect_space=raw.get("effect_space", StabilityConfig.effect_space),
        workers=raw.get("workers", StabilityConfig.workers),
        resume=raw.get("resume", StabilityConfig.resume),
        checkpoint_flush_features=raw.get(
            "checkpoint_flush_features", StabilityConfig.checkpoint_flush_features
        ),
        seed=raw.get("seed"),
        scalar=_build_stability_scalar_config(raw.get("scalar")),
        subspace=_build_stability_subspace_config(raw.get("subspace")),
        sample_size=_build_stability_sample_size_config(raw.get("sample_size")),
        leave_out=_build_stability_leave_out_config(raw.get("leave_out")),
    )
    if cfg.effect_space != "final_resid":
        raise ValueError("`phases.stability.effect_space` must be `final_resid`.")
    if isinstance(cfg.workers, bool) or isinstance(cfg.workers, float):
        raise ValueError(
            "`phases.stability.workers` must be an integer > 0, "
            f"got {cfg.workers!r}."
        )
    cfg.workers = _coerce_positive_int(cfg.workers, "phases.stability.workers")
    cfg.resume = _coerce_bool(cfg.resume, "phases.stability.resume")
    if isinstance(cfg.checkpoint_flush_features, bool) or isinstance(
        cfg.checkpoint_flush_features, float
    ):
        raise ValueError(
            "`phases.stability.checkpoint_flush_features` must be an integer > 0, "
            f"got {cfg.checkpoint_flush_features!r}."
        )
    cfg.checkpoint_flush_features = _coerce_positive_int(
        cfg.checkpoint_flush_features,
        "phases.stability.checkpoint_flush_features",
    )
    cfg.seed = _coerce_seed(cfg.seed, "phases.stability.seed", allow_none=True)
    return cfg


def _build_stability_scalar_config(raw_section: Any) -> StabilityScalarConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section StabilityScalarConfig.")
    _reject_unknown_keys(raw, StabilityScalarConfig, "phases.stability.scalar")
    cfg = StabilityScalarConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.bootstrap_rounds = _coerce_nonnegative_int(
        cfg.bootstrap_rounds, "phases.stability.scalar.bootstrap_rounds"
    )
    cfg.ci_quantiles = _coerce_ordered_quantile_pair(
        cfg.ci_quantiles, "phases.stability.scalar.ci_quantiles"
    )
    return cfg


def _build_stability_subspace_config(raw_section: Any) -> StabilitySubspaceConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section StabilitySubspaceConfig.")
    _reject_unknown_keys(raw, StabilitySubspaceConfig, "phases.stability.subspace")
    cfg = StabilitySubspaceConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.k_values = _coerce_unique_positive_ints(
        cfg.k_values, "phases.stability.subspace.k_values"
    )
    cfg.resample_fraction = _coerce_unit_interval_float(
        cfg.resample_fraction,
        "phases.stability.subspace.resample_fraction",
        allow_zero=False,
    )
    cfg.resample_rounds = _coerce_nonnegative_int(
        cfg.resample_rounds, "phases.stability.subspace.resample_rounds"
    )
    cfg.angle_p90_quantile = _coerce_unit_interval_float(
        cfg.angle_p90_quantile,
        "phases.stability.subspace.angle_p90_quantile",
        allow_zero=False,
    )
    cfg.eig_floor = _coerce_positive_float(
        cfg.eig_floor, "phases.stability.subspace.eig_floor"
    )
    return cfg


def _build_stability_sample_size_config(
    raw_section: Any,
) -> StabilitySampleSizeConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section StabilitySampleSizeConfig.")
    _reject_unknown_keys(raw, StabilitySampleSizeConfig, "phases.stability.sample_size")
    cfg = StabilitySampleSizeConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.target_sizes = _coerce_unique_positive_ints(
        cfg.target_sizes, "phases.stability.sample_size.target_sizes"
    )
    cfg.subset_rounds = _coerce_nonnegative_int(
        cfg.subset_rounds, "phases.stability.sample_size.subset_rounds"
    )
    cfg.strong_subset_rounds = _coerce_nonnegative_int(
        cfg.strong_subset_rounds,
        "phases.stability.sample_size.strong_subset_rounds",
    )
    cfg.max_enumerated_subsets = _coerce_nonnegative_int(
        cfg.max_enumerated_subsets,
        "phases.stability.sample_size.max_enumerated_subsets",
    )
    return cfg


def _build_stability_leave_out_config(raw_section: Any) -> StabilityLeaveOutConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section StabilityLeaveOutConfig.")
    _reject_unknown_keys(raw, StabilityLeaveOutConfig, "phases.stability.leave_out")
    cfg = StabilityLeaveOutConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    cfg.max_leave_two_out_pairs = _coerce_nonnegative_int(
        cfg.max_leave_two_out_pairs,
        "phases.stability.leave_out.max_leave_two_out_pairs",
    )
    cfg.min_group_count = _coerce_positive_int(
        cfg.min_group_count, "phases.stability.leave_out.min_group_count"
    )
    cfg.min_group_size = _coerce_positive_int(
        cfg.min_group_size, "phases.stability.leave_out.min_group_size"
    )
    return cfg


def _build_geometry_reporting_config(raw_section: Any) -> GeometryReportingConfig:
    raw = raw_section or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Expected mapping for section GeometryReportingConfig.")
    _reject_unknown_keys(raw, GeometryReportingConfig, "phases.geometry_reporting")
    cfg = GeometryReportingConfig(**raw)
    cfg.enabled = bool(cfg.enabled)
    if cfg.threshold_profile != "paper":
        raise ValueError(
            "`phases.geometry_reporting.threshold_profile` must be `paper`."
        )
    if cfg.embedding not in {"auto", "umap", "pca", "tsne"}:
        raise ValueError(
            "`phases.geometry_reporting.embedding` must be `auto`, `umap`, `pca`, or `tsne`."
        )
    if cfg.global_flag_mode not in {"layered_overlay", "bitmask_ring"}:
        raise ValueError(
            "`phases.geometry_reporting.global_flag_mode` must be `layered_overlay` "
            "or `bitmask_ring`."
        )
    cfg.seed = _coerce_seed(cfg.seed, "phases.geometry_reporting.seed", allow_none=True)
    cfg.eps = _coerce_positive_float(cfg.eps, "phases.geometry_reporting.eps")
    cfg.map_enabled = bool(cfg.map_enabled)
    cfg.atlas_include_insufficient_evidence = bool(
        cfg.atlas_include_insufficient_evidence
    )
    cfg.write_csv = bool(cfg.write_csv)
    return cfg


def _validate_compute_effect_prereq_config(
    data_prep_cfg: DataPrepConfig, compute_effect_cfg: ComputeEffectConfig
) -> None:
    if not compute_effect_cfg.enabled:
        return
    if "final_resid" in data_prep_cfg.readouts and not data_prep_cfg.gram_cache:
        raise ValueError(
            "`phases.compute_effect.enabled` requires "
            "`phases.data_prep.gram_cache: true` when `final_resid` is included in "
            "`phases.data_prep.readouts`; set it and rerun data_prep."
        )


def _validate_downstream_readout_dependencies(
    *,
    data_prep_cfg: DataPrepConfig,
    compute_effect_cfg: ComputeEffectConfig,
    geometry_metrics_cfg: GeometryMetricsConfig,
    vmf_cfg: DirectionalMixtureFitConfig,
    stability_cfg: StabilityConfig,
) -> None:
    readouts = set(data_prep_cfg.readouts)
    required: dict[str, list[str]] = {}
    if geometry_metrics_cfg.enabled:
        required.setdefault(geometry_metrics_cfg.effect_space, []).append(
            "phases.geometry_metrics.effect_space"
        )
    if vmf_cfg.enabled:
        required.setdefault("final_resid", []).append(
            "phases.vmf.effect_space (source_readout final_resid)"
        )
    if stability_cfg.enabled:
        required.setdefault(stability_cfg.effect_space, []).append(
            "phases.stability.effect_space"
        )

    missing = sorted(space for space in required if space not in readouts)
    if missing:
        details = "; ".join(
            f"`{space}` required by {', '.join(required[space])}" for space in missing
        )
        raise ValueError(
            "Enabled downstream phases require matching `phases.data_prep.readouts`: "
            f"{details}."
        )
    if (
        compute_effect_cfg.enabled
        and "final_resid" in required
        and not data_prep_cfg.gram_cache
    ):
        raise ValueError(
            "Enabled downstream final_resid consumers with "
            "`phases.compute_effect.enabled: true` require "
            "`phases.data_prep.gram_cache: true`."
        )


def _build_section(raw_section: Any, section_cls):
    if raw_section is None:
        return section_cls()
    if not isinstance(raw_section, Mapping):
        raise ValueError(f"Expected mapping for section {section_cls.__name__}.")
    allowed = {f.name for f in fields(section_cls)}
    filtered = {k: v for k, v in raw_section.items() if k in allowed}
    return section_cls(**filtered)


def _reject_unknown_keys(
    raw_section: Mapping[str, Any], section_cls, path: str
) -> None:
    allowed = {f.name for f in fields(section_cls)}
    unknown = sorted(str(k) for k in raw_section if k not in allowed)
    if unknown:
        rendered = ", ".join(f"`{path}.{key}`" for key in unknown)
        raise ValueError(f"Unknown config key(s): {rendered}.")


def _coerce_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(
            f"`{field_name}` must be an integer > 0, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {value!r}.")
    return parsed


def _coerce_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"`{field_name}` must be a boolean, got {value!r}.")
    return value


def _coerce_nonnegative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(
            f"`{field_name}` must be an integer >= 0, got {value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(f"`{field_name}` must be >= 0, got {value!r}.")
    return parsed


def _coerce_unique_positive_ints(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{field_name}` must be a non-empty list of integers > 0.")
    parsed: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"`{field_name}` must contain integers, got {item!r}.")
        integer = item
        if integer <= 0:
            raise ValueError(f"`{field_name}` values must be > 0, got {item!r}.")
        if integer in seen:
            raise ValueError(f"`{field_name}` values must be unique, got {item!r}.")
        parsed.append(integer)
        seen.add(integer)
    return parsed


def _coerce_optional_feature_ids(value: Any, *, feature_set: str) -> list[int] | None:
    if feature_set != "explicit":
        if value is None:
            return None
        parsed = _coerce_unique_nonnegative_ints(
            value, "induction.explicit_feature_ids"
        )
        return parsed
    if value is None:
        raise ValueError(
            "`induction.feature_set: explicit` requires non-empty "
            "`induction.explicit_feature_ids`."
        )
    return _coerce_unique_nonnegative_ints(value, "induction.explicit_feature_ids")


def _coerce_unique_nonnegative_ints(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{field_name}` must be a non-empty list of integers >= 0.")
    parsed: list[int] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"`{field_name}` must contain integers, got {item!r}.")
        if item < 0:
            raise ValueError(f"`{field_name}` values must be >= 0, got {item!r}.")
        if item in seen:
            raise ValueError(f"`{field_name}` values must be unique, got {item!r}.")
        parsed.append(item)
        seen.add(item)
    return parsed


def _coerce_optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _coerce_bool(value, field_name)


def _coerce_optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_nonnegative_float(value, field_name)


def _coerce_nonempty_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{field_name}` must be a non-empty list of strings.")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"`{field_name}` must contain non-empty strings.")
        cleaned = item.strip()
        if cleaned not in out:
            out.append(cleaned)
    return out


def _coerce_ordered_quantile_pair(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"`{field_name}` must be a two-value list in [0, 1].")
    parsed = [_coerce_unit_interval_float(item, field_name) for item in value]
    if parsed[0] >= parsed[1]:
        raise ValueError(f"`{field_name}` must be ordered low < high.")
    return parsed


def _coerce_positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"`{field_name}` must be a float > 0, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {value!r}.")
    return parsed


def _coerce_nonnegative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(
            f"`{field_name}` must be a float >= 0, got {value!r}."
        ) from exc
    if parsed < 0:
        raise ValueError(f"`{field_name}` must be >= 0, got {value!r}.")
    return parsed


def _coerce_unit_interval_float(
    value: Any, field_name: str, *, allow_zero: bool = True
) -> float:
    parsed = _coerce_nonnegative_float(value, field_name)
    if parsed > 1.0 or (parsed == 0.0 and not allow_zero):
        lower = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"`{field_name}` must be {lower} and <= 1, got {value!r}.")
    return parsed


def _coerce_readouts(value: Any) -> list[str]:
    supported = {"final_resid"}
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "`phases.data_prep.readouts` must contain non-empty strings."
                )
            items.append(item.strip())
    else:
        raise ValueError(
            "`phases.data_prep.readouts` must be a non-empty list of strings."
        )
    if not items:
        raise ValueError("`phases.data_prep.readouts` must be non-empty.")
    normalized: list[str] = []
    for item in items:
        if item not in supported:
            raise ValueError(
                f"Unsupported data_prep readout {item!r}. Supported: {', '.join(sorted(supported))}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def _coerce_gram_cache_dtype(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "`phases.data_prep.gram_cache_dtype` must be a non-empty string."
        )
    normalized = value.strip().lower()
    supported = {"float64", "float32", "float16", "bfloat16"}
    if normalized not in supported:
        raise ValueError(
            f"Unsupported gram_cache_dtype {value!r}. Supported: {', '.join(sorted(supported))}"
        )
    return normalized


def _convert_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_paths(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _validate_output_root(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"`output_root` must be a directory path, got file: {path}")
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _coerce_seed(value: Any, field_name: str, *, allow_none: bool) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"`{field_name}` must be provided and non-null.")
    try:
        return int(value)
    except Exception:
        raise ValueError(f"`{field_name}` must be an integer, got {value!r}.")


def _coerce_sae_source(value: Any) -> str:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError(
            "`sae_source` must be one of: auto, sae_lens, local_checkpoint."
        )
    normalized = value.strip().lower()
    allowed = {"auto", "sae_lens", "local_checkpoint"}
    if normalized not in allowed:
        raise ValueError(
            f"`sae_source` must be one of {sorted(allowed)}, got {value!r}."
        )
    return normalized


def _coerce_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(
            f"`{field_name}` must be an integer > 0, got {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(f"`{field_name}` must be > 0, got {value!r}.")
    return parsed


def _validate_sae_source_paths(
    *,
    sae_source: str,
    local_checkpoint_path: Path | None,
    local_resolved_config_path: Path | None,
) -> None:
    if sae_source == "local_checkpoint" and local_checkpoint_path is None:
        raise ValueError(
            "`local_sae_checkpoint_path` is required when `sae_source` is `local_checkpoint`."
        )
    if local_checkpoint_path is not None and not local_checkpoint_path.exists():
        raise FileNotFoundError(
            f"`local_sae_checkpoint_path` does not exist: {local_checkpoint_path}"
        )
    if (
        local_resolved_config_path is not None
        and not local_resolved_config_path.exists()
    ):
        raise FileNotFoundError(
            "`local_sae_resolved_config_path` does not exist: "
            f"{local_resolved_config_path}"
        )


def _resolve_path(value: Any, base_dir: Path, *, prefer_cwd: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        base = Path.cwd() if prefer_cwd else base_dir
        path = base / path
    return path.resolve()
