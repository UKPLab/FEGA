from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fega.config_schema import FEGAPipelineConfig
from fega.core.data_prep import run_data_prep
from fega.core.geometry_metrics import run_geometry_metrics
from fega.core.resources import ModelResources, resolve_mdbm_path
from fega.paths import (
    config_used_path,
    data_prep_activations_dir,
    data_prep_select_dir,
    effect_summary_path,
    effect_tensors_manifest_path,
    geometry_metrics_scores_path,
    gram_cache_meta_path,
    gram_cache_tensor_path,
    run_metadata_path,
    run_root,
    run_status_path,
)
from fega.run_metadata import build_base_metadata, write_run_metadata

PHASES: list[str] = [
    "data_prep",
    "compute_effect",
    "geometry_metrics",
    "vmf",
    "stability",
    "geometry_reporting",
]

_logger = logging.getLogger(__name__)


def run_compute_effect(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    from fega.core.compute_effect import run_compute_effect as _run_compute_effect

    _run_compute_effect(config, resources)


def run_vmf(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    from fega.core.vmf import run_vmf as _run_vmf

    _run_vmf(config, resources)


def run_stability(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    from fega.core.stability import run_stability as _run_stability

    _run_stability(config, resources)


def run_geometry_reporting(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    from fega.core.geometry_reporting import (
        run_geometry_reporting as _run_geometry_reporting,
    )

    _run_geometry_reporting(config, resources)


def resolve_phases(requested: list[str] | None) -> list[str]:
    """Normalize and validate the requested phases."""
    if requested is None or requested == ["all"]:
        return list(PHASES)
    normalized = []
    for name in requested:
        if name not in PHASES:
            raise ValueError(
                f"Unknown phase `{name}`. Valid phases: {', '.join(PHASES)}"
            )
        if name not in normalized:
            normalized.append(name)
    # Enforce global ordering
    return [phase for phase in PHASES if phase in normalized]


def run_pipeline(
    config: FEGAPipelineConfig,
    phases: list[str] | None = None,
    fail_fast: bool = True,
    *,
    resume: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the configured FEGA pipeline."""
    # Initialize minimal logging if caller did not configure it.
    if not _logger.handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )

    # Prepare run root and persist resolved config + metadata.
    root = run_root(config)
    root.mkdir(parents=True, exist_ok=True)
    _logger.info("Run root: %s", root)
    _check_static_inputs(config)

    _write_resolved_config(config)
    cfg_src = config_path or getattr(config, "_config_path", None)
    if cfg_src is None:
        raise ValueError(
            "config_path is required for provenance; call run_pipeline with config_path "
            "or load configs via FEGAPipelineConfig.from_file."
        )
    _write_base_metadata(config, Path(cfg_src))

    # Load any prior status for resume/skip logic and initialize tracking.
    existing_status = _load_existing_status(config)
    run_status = _initialize_status(existing_status)
    _write_run_status(config, run_status)

    # Decide which phases to run, applying config enablement and resume rules.
    phases_to_run = resolve_phases(_coerce_phase_arg(phases))
    explicit_phases = phases is not None and phases != ["all"]
    enabled_map = _phase_enabled_map(config)
    phases_to_run = _apply_config_enablement(
        phases_to_run, enabled_map, explicit_phases
    )
    phases_to_run = _apply_resume_filter(
        phases_to_run, run_status, resume, explicit_phases
    )
    _mark_skipped_phases(run_status, phases_to_run, enabled_map, explicit_phases)
    _write_run_status(config, run_status)
    _logger.info(
        "Phases to run: %s", ", ".join(phases_to_run) if phases_to_run else "(none)"
    )

    use_local_sae = config.sae_source == "local_checkpoint" or (
        config.sae_source == "auto" and config.local_sae_checkpoint_path is not None
    )
    resources = (
        ModelResources(config)
        if config.reuse_model_across_phases or use_local_sae
        else None
    )

    # Execute phases sequentially with per-phase status updates.
    for phase in phases_to_run:
        missing_reason = _check_prerequisites(phase, config)
        if missing_reason:
            run_status["phases"][phase]["status"] = "skipped"
            run_status["phases"][phase]["reason"] = "missing_inputs"
            run_status["phases"][phase]["error"] = missing_reason
            run_status["last_updated"] = _now_iso()
            _write_run_status(config, run_status)
            _logger.warning(
                "Skipping phase %s due to missing inputs: %s", phase, missing_reason
            )
            continue
        run_status["phases"][phase]["status"] = "running"
        run_status["phases"][phase].pop("error", None)
        run_status["last_updated"] = _now_iso()
        _write_run_status(config, run_status)
        try:
            _run_phase(phase, config, resources)
        except Exception as exc:
            run_status["phases"][phase]["status"] = "failed"
            run_status["phases"][phase]["error"] = str(exc)
            run_status["last_updated"] = _now_iso()
            _write_run_status(config, run_status)
            _logger.exception("Phase %s failed", phase)
            if fail_fast:
                raise
        else:
            run_status["phases"][phase]["status"] = "success"
            run_status["phases"][phase].pop("error", None)
            run_status["last_updated"] = _now_iso()
            _write_run_status(config, run_status)

    run_status["status"] = _summarize_overall_status(run_status, enabled_map)
    run_status["last_updated"] = _now_iso()
    _write_run_status(config, run_status)


def _coerce_phase_arg(phases: list[str] | str | None) -> list[str] | None:
    if phases is None or phases == ["all"]:
        return None
    if isinstance(phases, str):
        return [phases]
    return phases


def _apply_config_enablement(
    phases: list[str], enabled: dict[str, bool], explicit: bool
) -> list[str]:
    """Filter out phases disabled in config unless user explicitly listed them."""
    if explicit:
        return phases
    filtered = []
    for phase in phases:
        if enabled.get(phase, True):
            filtered.append(phase)
        else:
            _logger.info("Skip phase %s (disabled in config)", phase)
    return filtered


def _apply_resume_filter(
    phases: list[str],
    run_status: dict[str, Any],
    resume: bool,
    explicit: bool,
) -> list[str]:
    """Skip phases already succeeded when resume is requested and phases not explicitly listed."""
    if not resume:
        return phases
    if explicit:
        return phases
    filtered = []
    for phase in phases:
        prior_status = run_status["phases"].get(phase, {}).get("status")
        if prior_status == "success":
            _logger.info("Resume enabled: skip already successful phase %s", phase)
            continue
        filtered.append(phase)
    return filtered


def _check_prerequisites(phase: str, config: FEGAPipelineConfig) -> str | None:
    """Lightweight existence checks to avoid slow failures downstream."""
    entity, attr = _require_single_entity_attr(config)
    activations_dir = data_prep_activations_dir(config, entity, attr)
    manifest = activations_dir / "activations_manifest.json"

    if phase == "data_prep":
        if config.source_kind == "induction":
            return _check_induction_data_prep_inputs(config)
        return _check_mdbm_path(config, entity, attr)

    contexts_path, _ = _try_resolve_context_paths(config)

    if phase == "compute_effect":
        missing = [str(p) for p in (activations_dir, manifest) if not p.exists()]
        if contexts_path is None or not contexts_path.exists():
            missing.append(str(data_prep_select_dir(config) / "feature_contexts.json"))
        if "final_resid" in config.phases.data_prep.readouts:
            for p in (gram_cache_tensor_path(config), gram_cache_meta_path(config)):
                if not p.exists():
                    missing.append(str(p))
        if missing:
            return f"Missing inputs for {phase}: {', '.join(missing)}"
        return _check_compute_effect_manifest(
            manifest, config.phases.data_prep.readouts
        )

    if phase == "geometry_metrics":
        return _check_geometry_metrics_inputs(config)

    if phase == "vmf":
        return _check_vmf_inputs(config)

    if phase == "stability":
        return _check_stability_inputs(config)

    if phase == "geometry_reporting":
        return _check_geometry_reporting_inputs(config)

    return None


def _check_static_inputs(config: FEGAPipelineConfig) -> None:
    """Validate static run inputs before entering phase execution."""
    with open(config.reference_json, "rb"):
        pass
    if config.source_kind == "induction":
        if config.induction is None:
            raise ValueError(
                "`induction.summary_json` is required for induction source."
            )
        with open(config.induction.summary_json, "rb"):
            pass
    probe = run_root(config) / ".write_probe"
    with open(probe, "wb"):
        pass
    probe.unlink(missing_ok=True)

    use_local_sae = config.sae_source == "local_checkpoint" or (
        config.sae_source == "auto" and config.local_sae_checkpoint_path is not None
    )
    if use_local_sae:
        checkpoint = config.local_sae_checkpoint_path
        if checkpoint is None:
            raise ValueError(
                "`local_sae_checkpoint_path` is required when `sae_source` is `local_checkpoint`."
            )
        if not checkpoint.exists() or not checkpoint.is_file():
            raise FileNotFoundError(f"Local SAE checkpoint not found: {checkpoint}")
        with open(checkpoint, "rb"):
            pass


def _check_mdbm_path(config: FEGAPipelineConfig, entity: str, attr: str) -> str | None:
    """Return a missing-input message when the resolved MDBM path is unavailable."""
    weight_path = resolve_mdbm_path(config, entity, attr)
    if not weight_path.exists():
        return f"Missing MDBM checkpoint: {weight_path}"
    return None


def _check_induction_data_prep_inputs(config: FEGAPipelineConfig) -> str | None:
    missing = []
    if not config.reference_json.exists():
        missing.append(str(config.reference_json))
    if config.induction is None:
        missing.append("induction.summary_json")
    elif not config.induction.summary_json.exists():
        missing.append(str(config.induction.summary_json))
    if missing:
        return f"Missing inputs for data_prep: {', '.join(missing)}"
    return None


def _mark_skipped_phases(
    run_status: dict[str, Any],
    phases_to_run: list[str],
    enabled_map: dict[str, bool],
    explicit: bool,
) -> None:
    """Mark phases that will not run so status reflects intentional skips."""
    planned = set(phases_to_run)
    for phase in PHASES:
        if phase in planned:
            continue
        entry = run_status["phases"].setdefault(phase, {"status": "pending"})
        if entry.get("status") == "success":
            continue
        if not explicit and not enabled_map.get(phase, True):
            entry["status"] = "skipped"
            entry["reason"] = "disabled"
        else:
            entry["status"] = "skipped"
            entry["reason"] = "not_requested"


def _phase_enabled_map(config: FEGAPipelineConfig) -> dict[str, bool]:
    return {
        "data_prep": config.phases.data_prep.enabled,
        "compute_effect": config.phases.compute_effect.enabled,
        "geometry_metrics": config.phases.geometry_metrics.enabled,
        "vmf": config.phases.vmf.enabled,
        "stability": config.phases.stability.enabled,
        "geometry_reporting": config.phases.geometry_reporting.enabled,
    }


def _write_resolved_config(config: FEGAPipelineConfig) -> None:
    """Persist the resolved config snapshot under the run root for reproducibility."""
    cfg_path = config_used_path(config)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f)


def _write_base_metadata(config: FEGAPipelineConfig, config_path: Path) -> None:
    """Capture environment and seed info in run_metadata.json."""
    meta = build_base_metadata(config_path, config)
    meta.global_seed = config.seed.global_
    meta.stage_seeds = {
        "data_prep": config.seed.selection_seed or config.seed.global_,
        "compute_effect": config.seed.global_,
        "geometry_metrics": config.seed.global_,
        "vmf": config.phases.vmf.seed
        if config.phases.vmf.seed is not None
        else config.seed.global_,
        "stability": config.phases.stability.seed
        if config.phases.stability.seed is not None
        else config.seed.global_,
        "geometry_reporting": config.phases.geometry_reporting.seed
        if config.phases.geometry_reporting.seed is not None
        else config.seed.global_,
    }
    write_run_metadata(run_metadata_path(config), meta)


def _initialize_status(existing: dict[str, Any] | None) -> dict[str, Any]:
    """Start a fresh run_status structure, optionally seeded from prior status."""
    phases_state = {name: {"status": "pending"} for name in PHASES}
    if existing and isinstance(existing.get("phases"), dict):
        for name, state in existing["phases"].items():
            if name in phases_state:
                phases_state[name] = dict(state)
    return {
        "phases": phases_state,
        "last_updated": _now_iso(),
        "status": existing.get("status") if existing else "pending",
    }


def _summarize_overall_status(
    run_status: dict[str, Any], enabled_map: dict[str, bool]
) -> str:
    """Aggregate per-phase outcomes into a run-level status."""
    any_enabled = False
    for phase in PHASES:
        enabled = enabled_map.get(phase, True)
        state = run_status["phases"].get(phase, {}).get("status")
        if not enabled:
            continue
        any_enabled = True
        if state == "failed":
            return "failed"
        if state in {"pending", "running", "skipped"}:
            return "partial"
    return "success" if any_enabled else "pending"


def _write_run_status(config: FEGAPipelineConfig, status: dict[str, Any]) -> None:
    """Persist run_status.json under the run root."""
    path = run_status_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def _load_existing_status(config: FEGAPipelineConfig) -> dict[str, Any] | None:
    """Load previous run_status.json if present."""
    path = run_status_path(config)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _run_phase(
    name: str, config: FEGAPipelineConfig, resources: ModelResources | None
) -> None:
    """Dispatch to the phase-specific runner."""
    if name == "data_prep":
        run_data_prep(config, resources)
    elif name == "compute_effect":
        run_compute_effect(config, resources)
    elif name == "geometry_metrics":
        run_geometry_metrics(config, resources)
    elif name == "vmf":
        run_vmf(config, resources)
    elif name == "stability":
        run_stability(config, resources)
    elif name == "geometry_reporting":
        run_geometry_reporting(config, resources)
    else:
        raise ValueError(f"Unknown phase: {name}")


def _require_single_entity_attr(config: FEGAPipelineConfig) -> tuple[str, str]:
    """Assert config currently contains exactly one entity and one attribute."""
    if len(config.entity_attribute_selection) != 1:
        raise ValueError("Pipeline currently supports exactly one entity selection.")
    entity, attrs = next(iter(config.entity_attribute_selection.items()))
    if len(attrs) != 1:
        raise ValueError("Pipeline currently supports exactly one attribute selection.")
    return entity, attrs[0]


def _try_resolve_context_paths(
    config: FEGAPipelineConfig,
) -> tuple[Path | None, Path | None]:
    """Best-effort context resolution without raising; used for prereq checks."""
    preferred = data_prep_select_dir(config) / "feature_contexts.json"
    preferred_summary = data_prep_select_dir(config) / "feature_contexts_summary.json"
    if preferred.exists():
        return preferred, preferred_summary if preferred_summary.exists() else None
    return None, None


def _check_compute_effect_manifest(
    manifest_path: Path, requested_readouts: list[str]
) -> str | None:
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as exc:
        return (
            "Could not read activation manifest for compute_effect: "
            f"{manifest_path} ({exc})"
        )
    readouts = set(manifest.get("readouts") or [])
    tensor_keys = set(manifest.get("tensor_keys") or [])
    missing = [
        readout
        for readout in requested_readouts
        if readout not in readouts and readout not in tensor_keys
    ]
    if missing:
        rendered = ", ".join(f"`{readout}`" for readout in missing)
        return (
            "Missing inputs for compute_effect: activation manifest does not include "
            f"{rendered}; set `phases.data_prep.readouts` to include the missing "
            "readout(s) and rerun data_prep."
        )
    return None


def _check_geometry_metrics_inputs(config: FEGAPipelineConfig) -> str | None:
    effect_space = config.phases.geometry_metrics.effect_space
    manifest_path = effect_tensors_manifest_path(config, effect_space)
    summary_path = effect_summary_path(config, effect_space)
    missing = [str(p) for p in (manifest_path, summary_path) if not p.exists()]
    if missing:
        return f"Missing inputs for geometry_metrics: {', '.join(missing)}"
    try:
        with open(manifest_path) as f:
            effect_manifest = json.load(f)
        with open(summary_path) as f:
            effect_summary = json.load(f)
    except Exception as exc:
        return (
            "Could not read compute_effect artifacts for geometry_metrics: "
            f"{manifest_path}, {summary_path} ({exc})"
        )
    if effect_manifest.get("effect_space") != effect_space:
        return (
            "Missing inputs for geometry_metrics: manifest effect_space "
            f"{effect_manifest.get('effect_space')!r} does not match {effect_space!r}"
        )
    shard_names = {
        Path(str(shard.get("path"))).name for shard in effect_manifest.get("shards", [])
    }
    for feature_id, record in (effect_summary.get("per_feature") or {}).items():
        if not isinstance(record, dict):
            return f"Missing inputs for geometry_metrics: invalid feature summary {feature_id}"
        shard = record.get("tensor_shard")
        if shard is None:
            continue
        if shard_names and shard not in shard_names:
            return (
                "Missing inputs for geometry_metrics: feature "
                f"{feature_id} references shard {shard!r} absent from manifest"
            )
        shard_path = manifest_path.parent / str(shard)
        if not shard_path.exists():
            missing.append(str(shard_path))
    return f"Missing inputs for geometry_metrics: {', '.join(missing)}" if missing else None


def _check_vmf_inputs(config: FEGAPipelineConfig) -> str | None:
    source_readout = "final_resid"
    manifest_path = effect_tensors_manifest_path(config, source_readout)
    summary_path = effect_summary_path(config, source_readout)
    geometry_metrics_path = geometry_metrics_scores_path(config, source_readout)
    missing = [
        str(p) for p in (manifest_path, summary_path, geometry_metrics_path) if not p.exists()
    ]
    if missing:
        return f"Missing inputs for vmf: {', '.join(missing)}"
    try:
        with open(manifest_path) as f:
            effect_manifest = json.load(f)
        with open(summary_path) as f:
            effect_summary = json.load(f)
        with open(geometry_metrics_path) as f:
            geometry_metrics_scores = json.load(f)
    except Exception as exc:
        return (
            "Could not read inputs for vmf: "
            f"{manifest_path}, {summary_path}, {geometry_metrics_path} ({exc})"
        )
    if effect_manifest.get("effect_space") != source_readout:
        return (
            "Missing inputs for vmf: manifest effect_space "
            f"{effect_manifest.get('effect_space')!r} does not match "
            f"{source_readout!r}"
        )
    if not isinstance(effect_summary.get("per_feature"), dict):
        return f"Missing inputs for vmf: invalid effect summary {summary_path}"
    if not isinstance(geometry_metrics_scores.get("per_feature"), dict):
        return f"Missing inputs for vmf: invalid geometry_metrics scores {geometry_metrics_path}"
    shard_names = {
        Path(str(shard.get("path"))).name for shard in effect_manifest.get("shards", [])
    }
    for feature_id, record in (effect_summary.get("per_feature") or {}).items():
        if not isinstance(record, dict):
            return f"Missing inputs for vmf: invalid feature summary {feature_id}"
        shard = record.get("tensor_shard")
        if shard is None:
            continue
        if shard_names and shard not in shard_names:
            return (
                "Missing inputs for vmf: feature "
                f"{feature_id} references shard {shard!r} absent from manifest"
            )
        shard_path = manifest_path.parent / str(shard)
        if not shard_path.exists():
            missing.append(str(shard_path))
    return f"Missing inputs for vmf: {', '.join(missing)}" if missing else None


def _check_stability_inputs(config: FEGAPipelineConfig) -> str | None:
    """Require canonical point inputs through the reporting-owned validator."""
    # Preserve source-first phase order before validating geometry and standalone vMF.
    effect_space = config.phases.stability.effect_space
    reason = _check_stability_effect_space(config, effect_space)
    if reason is not None:
        return f"Missing inputs for stability: {effect_space}: {reason}"
    from fega.core.geometry_reporting.artifacts import load_point_geometry_inputs

    try:
        load_point_geometry_inputs(config)
    except (OSError, ValueError) as exc:
        return f"Missing inputs for stability: {effect_space}: {exc}"
    return None


def _check_stability_effect_space(
    config: FEGAPipelineConfig, effect_space: str
) -> str | None:
    manifest_path = effect_tensors_manifest_path(config, effect_space)
    summary_path = effect_summary_path(config, effect_space)
    missing = [str(p) for p in (manifest_path, summary_path) if not p.exists()]
    if missing:
        return ", ".join(missing)
    try:
        with open(manifest_path) as f:
            effect_manifest = json.load(f)
        with open(summary_path) as f:
            effect_summary = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"could not read {manifest_path}, {summary_path} ({exc})"
    if effect_manifest.get("effect_space") != effect_space:
        return (
            "manifest effect_space "
            f"{effect_manifest.get('effect_space')!r} does not match {effect_space!r}"
        )
    expected_metric_space = "residual_gram" if effect_space == "final_resid" else None
    if effect_manifest.get("metric_space") != expected_metric_space:
        return (
            "manifest metric_space "
            f"{effect_manifest.get('metric_space')!r} does not match "
            f"{expected_metric_space!r}"
        )
    if not isinstance(effect_summary.get("per_feature"), dict):
        return f"invalid effect summary {summary_path}"
    shard_names = {
        Path(str(shard.get("path"))).name for shard in effect_manifest.get("shards", [])
    }
    for feature_id, record in (effect_summary.get("per_feature") or {}).items():
        if not isinstance(record, dict):
            return f"invalid feature summary {feature_id}"
        shard = record.get("tensor_shard")
        if shard is None:
            continue
        if shard_names and shard not in shard_names:
            return f"feature {feature_id} references absent shard {shard!r}"
        shard_path = manifest_path.parent / str(shard)
        if not shard_path.exists():
            missing.append(str(shard_path))
    gram_raw = (effect_manifest.get("inputs") or {}).get("gram_path")
    gram_path = Path(gram_raw) if gram_raw else gram_cache_tensor_path(config)
    if not gram_path.exists():
        missing.append(str(gram_path))
    return ", ".join(missing) if missing else None


def _check_geometry_reporting_inputs(config: FEGAPipelineConfig) -> str | None:
    """Require the same point-lock and provenance contract as final reporting."""
    # Invoke the production loader so existence-only checks cannot admit stale science.
    from fega.core.geometry_reporting.artifacts import load_geometry_inputs

    try:
        load_geometry_inputs(config)
    except (OSError, ValueError) as exc:
        return f"Missing inputs for geometry_reporting: {exc}"
    return None


def _now_iso() -> str:
    """UTC timestamp helper for status files."""
    return datetime.now(timezone.utc).isoformat()
