from __future__ import annotations

import logging
from typing import Any

from fega.config_schema import FEGAPipelineConfig
from fega.core.geometry_metrics.artifacts import (
    FeatureEffectBlock,
    iter_feature_blocks,
    load_geometry_metrics_inputs,
    resolve_final_resid_gram,
    write_geometry_metrics_scores,
)
from fega.core.geometry_metrics.metrics import (
    CenteredResidualSpectrumResult,
    CRayResult,
    EffectiveRankResult,
    SpanSpectrumResult,
    c_ray_fast_final_resid,
    c_ray_pairwise_final_resid,
    centered_residual_spectrum_final_resid,
    effective_rank_from_spectrum,
    span_spectrum_final_resid,
)
from fega.core.resources import ModelResources
from fega.core.source_fingerprint import canonical_source_fingerprint
from fega.paths import geometry_metrics_scores_path

_logger = logging.getLogger(__name__)
_NO_EFFECTIVE_RANK = EffectiveRankResult(r_ent=None, r_pr=None)


def run_geometry_metrics(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    """Score directional concentration from reusable compute-effect artifacts.

    Centered-residual diagnostics are restricted at this runtime boundary as a
    defense against callers that mutate an already validated config dataclass.
    """
    # Validate direct-call invariants before loading or scoring any scientific input.
    cfg = config.phases.geometry_metrics
    if cfg.effect_space != "final_resid":
        raise ValueError("geometry_metrics requires effect_space='final_resid'.")
    if any(k not in {1, 2, 3, 4} for k in cfg.resid.k_values):
        raise ValueError(
            "phases.geometry_metrics.resid.k_values must be a subset of [1, 2, 3, 4]."
        )
    effect_space = "final_resid"
    inputs = load_geometry_metrics_inputs(config, effect_space, resources)
    gram = resolve_final_resid_gram(inputs, resources)
    source_fingerprint = canonical_source_fingerprint(
        inputs.manifest, inputs.summary
    )

    per_feature: dict[str, dict[str, Any]] = {}
    features_total = 0
    features_scored = 0
    features_skipped = 0

    for block in iter_feature_blocks(inputs):
        features_total += 1
        record = _score_block(block)
        if block.skipped_reason is None and block.rows is not None:
            span: SpanSpectrumResult | None = None
            resid: CenteredResidualSpectrumResult | None = None
            if cfg.c_ray.enabled:
                if cfg.c_ray.method == "fast_formula":
                    c_ray = c_ray_fast_final_resid(
                        block.rows, gram, eps=cfg.c_ray.eps
                    )
                else:
                    c_ray = c_ray_pairwise_final_resid(
                        block.rows, gram, eps=cfg.c_ray.eps
                    )
                _record_c_ray(record, c_ray, cfg.c_ray.store_r2)
            if cfg.span.enabled:
                span = span_spectrum_final_resid(
                    block.rows,
                    gram,
                    k_values=cfg.span.k_values,
                    eps=cfg.span.eps,
                )
                _record_span(record, span)
            if cfg.resid.enabled:
                resid = centered_residual_spectrum_final_resid(
                    block.rows,
                    gram,
                    k_values=cfg.resid.k_values,
                    eps=cfg.resid.eps,
                )
                _record_resid(record, resid)
            if cfg.effective_rank.enabled:
                if span is None or resid is None:
                    raise ValueError(
                        "effective rank requires span and residual spectra."
                    )
                span_rank = effective_rank_from_spectrum(
                    span.eigenvalues, eps=cfg.effective_rank.eps
                )
                ctr_rank = effective_rank_from_spectrum(
                    resid.eigenvalues, eps=cfg.effective_rank.eps
                )
                _record_effective_rank(record, span_rank, ctr_rank)
            if record.get("skipped_reason") is None:
                features_scored += 1
            else:
                features_skipped += 1
        else:
            if cfg.effective_rank.enabled:
                _record_effective_rank(record, _NO_EFFECTIVE_RANK, _NO_EFFECTIVE_RANK)
            features_skipped += 1
        per_feature[str(block.feature_id)] = record

    output_path = geometry_metrics_scores_path(config, effect_space)
    payload = {
        "canonical_source_fingerprint": source_fingerprint,
        "summary": {
            "effect_space": effect_space,
            "source_manifest_path": str(inputs.manifest_path),
            "source_summary_path": str(inputs.summary_path),
            "output_path": str(output_path),
            "features_total": features_total,
            "features_scored": features_scored,
            "features_skipped": features_skipped,
            "source_counts": inputs.manifest.get("counts", {}),
            "c_ray": {
                "enabled": cfg.c_ray.enabled,
                "method": cfg.c_ray.method,
                "store_r2": cfg.c_ray.store_r2,
                "eps": cfg.c_ray.eps,
            },
            "span": {
                "enabled": cfg.span.enabled,
                "k_values": cfg.span.k_values,
                "eps": cfg.span.eps,
            },
            "resid": {
                "enabled": cfg.resid.enabled,
                "k_values": cfg.resid.k_values,
                "eps": cfg.resid.eps,
            },
        },
        "per_feature": per_feature,
    }
    if cfg.effective_rank.enabled:
        payload["summary"]["effective_rank"] = {
            "enabled": True,
            "eps": cfg.effective_rank.eps,
        }
    write_geometry_metrics_scores(output_path, payload)
    _logger.info(
        "geometry_metrics complete: effect_space=%s path=%s", effect_space, output_path
    )


def _score_block(block: FeatureEffectBlock) -> dict[str, Any]:
    source = block.source_summary
    record: dict[str, Any] = {
        "feature_id": block.feature_id,
        "source_tensor_shard": block.tensor_shard,
        "source_row_start": source.get("row_start"),
        "source_row_end": source.get("row_end"),
        "source_usable_effects": source.get("usable_effects"),
        "n_valid": 0,
    }
    if block.skipped_reason is not None:
        record["skipped_reason"] = block.skipped_reason
    return record


def _record_c_ray(
    record: dict[str, Any], result: CRayResult, store_r2: bool
) -> None:
    record.update(
        {
            "n_total": result.n_total,
            "n_valid": result.n_valid,
            "skipped_nonfinite": result.skipped_nonfinite,
            "skipped_zero_norm": result.skipped_zero_norm,
            "s_norm": result.s_norm,
            "s_norm_sq": result.s_norm_sq,
        }
    )
    if result.c_ray is not None:
        record["c_ray"] = result.c_ray
    if store_r2 and result.r2 is not None:
        record["r2"] = result.r2
    if result.c_ray is None and record.get("skipped_reason") is None:
        record["skipped_reason"] = "below_min_valid_effects_2"


def _record_span(record: dict[str, Any], result: SpanSpectrumResult) -> None:
    record.setdefault("n_total", result.n_total)
    record["n_valid"] = max(int(record.get("n_valid") or 0), result.n_valid)
    record["span_n_valid"] = result.n_valid
    record["span_skipped_nonfinite"] = result.skipped_nonfinite
    record["span_skipped_zero_norm"] = result.skipped_zero_norm
    record["span_eigenvalues"] = result.eigenvalues
    for k, value in result.s_span.items():
        record[f"s_span_{k}"] = value
    for k, value in result.u_span.items():
        record[f"u_span_{k}"] = value
    for k, value in result.d_span.items():
        record[f"d_span_{k}"] = value
    record["b_axis"] = result.b_axis
    if result.n_valid == 0 and record.get("skipped_reason") is None:
        record["skipped_reason"] = "span_no_valid_effects"


def _record_resid(
    record: dict[str, Any], result: CenteredResidualSpectrumResult
) -> None:
    record.setdefault("n_total", result.n_total)
    record["n_valid"] = max(int(record.get("n_valid") or 0), result.n_valid)
    record["resid_n_valid"] = result.n_valid
    record["resid_skipped_nonfinite"] = result.skipped_nonfinite
    record["resid_skipped_zero_norm"] = result.skipped_zero_norm
    record["resid_eigenvalues"] = result.eigenvalues
    record["e_res"] = result.e_res
    for k, value in result.s_res.items():
        record[f"s_res_{k}"] = value
    if result.n_valid == 0 and record.get("skipped_reason") is None:
        record["skipped_reason"] = "resid_no_valid_effects"


def _record_effective_rank(
    record: dict[str, Any],
    span: EffectiveRankResult,
    centered_residual: EffectiveRankResult,
) -> None:
    record["r_span_ent"] = span.r_ent
    record["r_span_pr"] = span.r_pr
    record["r_ctr_ent"] = centered_residual.r_ent
    record["r_ctr_pr"] = centered_residual.r_pr
