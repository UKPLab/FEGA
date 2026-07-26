from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from fega.config_schema import DirectionalMixtureFitConfig, FEGAPipelineConfig
from fega.core.data_prep.gram_cache import (
    canonical_unembedding,
    unembedding_fingerprint,
)
from fega.core.geometry_metrics.artifacts import (
    FeatureEffectBlock,
    GeometryMetricsInputs,
    iter_feature_blocks,
    load_geometry_metrics_inputs,
    resolve_final_resid_gram,
)
from fega.core.resources import ModelResources
from fega.core.source_fingerprint import (
    canonical_json_digest,
    canonical_source_fingerprint,
    require_canonical_source_fingerprint,
)
from fega.core.vmf.artifacts import (
    VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION,
    build_vmf_scientific_fingerprint,
    delete_vmf_checkpoint,
    feature_ids_from_summary,
    load_geometry_metrics_scores,
    load_vmf_checkpoint,
    vmf_materialization_policy,
    vmf_scientific_compatibility_identity,
    write_vmf_checkpoint,
    write_vmf_scores,
)
from fega.core.vmf.metrics import (
    PUBLIC_METRIC_KEYS,
    feature_fit_seed,
    score_vmf_feature,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor_gpu import (
    validate_gpu_execution_workers,
)
from fega.paths import vmf_checkpoint_path, vmf_scores_path

_logger = logging.getLogger(__name__)
_monotonic = time.monotonic
_VOCAB_MATERIALIZATION_CHUNK_SIZE = 16_384


def run_vmf(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> None:
    """Fit vMF models from bounded linear coordinates of canonical residual rows."""
    # Load only the canonical source, then bind it to the actual model readout.
    cfg = config.phases.vmf
    validate_gpu_execution_workers(cfg.backend, cfg.workers)
    if cfg.effect_space != "pre_softcap_logits":
        raise ValueError("vmf fitting requires effect_space='pre_softcap_logits'.")
    model_resources = resources or ModelResources(config)
    inputs = load_geometry_metrics_inputs(config, "final_resid", model_resources)
    source_fingerprint = canonical_source_fingerprint(
        inputs.manifest, inputs.summary
    )
    geometry_metrics_scores = load_geometry_metrics_scores(config, "final_resid", model_resources)
    require_canonical_source_fingerprint(
        geometry_metrics_scores,
        source_fingerprint,
        artifact_label="geometry_metrics",
    )
    unembedding = _validated_canonical_unembedding(
        config, inputs, model_resources
    )
    if not isinstance(geometry_metrics_scores.get("per_feature"), dict):
        raise ValueError("geometry_metrics scores for vmf must contain `per_feature`.")

    seed = cfg.seed if cfg.seed is not None else config.seed.global_
    per_feature_summary = inputs.summary.get("per_feature")
    if not isinstance(per_feature_summary, dict):
        raise ValueError(f"effect_summary missing `per_feature`: {inputs.summary_path}")

    feature_ids = feature_ids_from_summary(per_feature_summary)
    materialization_policy = vmf_materialization_policy(cfg)
    scientific_fingerprint = build_vmf_scientific_fingerprint(
        config=config,
        cfg=cfg,
        seed=int(seed),
        inputs_manifest=inputs.manifest,
        inputs_summary=inputs.summary,
        geometry_metrics_scores=geometry_metrics_scores,
        feature_ids=feature_ids,
        source_fingerprint=source_fingerprint,
        materialization_policy=materialization_policy,
    )
    materializer = _BoundedLinearMaterializer(
        unembedding=unembedding,
        max_buffers=cfg.max_vocab_buffers,
    )
    checkpoint_path = vmf_checkpoint_path(config, "pre_softcap_logits")
    checkpoint = _VmfCheckpointState.load(
        enabled=cfg.resume,
        path=checkpoint_path,
        fingerprint=scientific_fingerprint,
        expected_feature_ids=set(feature_ids),
        workers=cfg.workers,
        flush_every_features=cfg.checkpoint_flush_features,
    )
    if cfg.resume and checkpoint.completed_count:
        _logger.info(
            "vmf resume: checkpoint=%s completed=%d remaining=%d",
            checkpoint_path,
            checkpoint.completed_count,
            max(0, len(feature_ids) - checkpoint.completed_count),
        )

    progress = _VmfProgressLogger(
        total_features=len(per_feature_summary),
        workers=cfg.workers,
        initial_processed=checkpoint.completed_count,
    )
    progress.start()

    blocks = iter_feature_blocks(
        inputs, exclude_feature_ids=checkpoint.completed_ids
    )
    if cfg.workers == 1:
        computed_features = _score_feature_blocks_sequential(
            blocks,
            cfg,
            int(seed),
            materializer,
            progress,
            checkpoint,
        )
    else:
        computed_features = _score_feature_blocks_parallel(
            blocks,
            cfg,
            int(seed),
            materializer,
            workers=cfg.workers,
            progress=progress,
            checkpoint=checkpoint,
        )
    features = checkpoint.features() if cfg.resume else computed_features
    features.sort(key=lambda record: int(record["feature_id"]))

    output_path = vmf_scores_path(config, "pre_softcap_logits")
    payload = {
        "phase": "vmf",
        "schema_version": VMF_PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "effect_space": "pre_softcap_logits",
        "source_readout": "final_resid",
        "geometry_metrics_effect_space": "final_resid",
        "canonical_source_fingerprint": source_fingerprint,
        "fingerprint": scientific_fingerprint,
        "features": features,
    }
    write_vmf_scores(
        output_path, payload, expected_fingerprint=scientific_fingerprint
    )
    delete_vmf_checkpoint(checkpoint_path)
    progress.finish()
    _logger.info("vmf complete: path=%s", output_path)


def _score_feature_blocks_sequential(
    blocks: Iterable[FeatureEffectBlock],
    cfg: DirectionalMixtureFitConfig,
    seed: int,
    materializer: _BoundedLinearMaterializer,
    progress: _VmfProgressLogger,
    checkpoint: _VmfCheckpointState,
) -> list[dict[str, Any]]:
    """Score feature blocks serially while checkpointing complete feature states."""
    # Preserve source order during computation and canonicalize only at artifact output.
    features: list[dict[str, Any]] = []
    for block in blocks:
        feature = _score_feature_block(block, cfg, seed, materializer)
        features.append(feature)
        checkpoint.record(feature)
        progress.advance()
    return features


def _score_feature_blocks_parallel(
    blocks: Iterable[FeatureEffectBlock],
    cfg: DirectionalMixtureFitConfig,
    seed: int,
    materializer: _BoundedLinearMaterializer,
    *,
    workers: int,
    progress: _VmfProgressLogger,
    checkpoint: _VmfCheckpointState,
) -> list[dict[str, Any]]:
    """Score independent feature states concurrently without schedule-derived inputs."""
    # Bound queued futures while every feature seed remains a pure feature-id function.
    features: list[dict[str, Any]] = []
    pending: set[Future[dict[str, Any]]] = set()
    max_pending = workers * 2

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for block in blocks:
            pending.add(
                executor.submit(
                    _score_feature_block,
                    block,
                    cfg,
                    seed,
                    materializer,
                )
            )
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                _record_completed_futures(done, features, checkpoint, progress)

        if pending:
            done, _ = wait(pending)
            _record_completed_futures(done, features, checkpoint, progress)

    features.sort(key=lambda record: int(record["feature_id"]))
    return features


def _score_feature_block(
    block: FeatureEffectBlock,
    cfg: DirectionalMixtureFitConfig,
    seed: int,
    materializer: _BoundedLinearMaterializer,
) -> dict[str, Any]:
    """Materialize, score, and preserve one complete independent feature state."""
    # Hold the dedicated permit until scoring releases the only derived vocab buffer.
    with materializer.coordinates(block) as coordinates:
        result = score_vmf_feature(
            coordinates,
            cfg,
            seed=_feature_fit_seed(seed, block.feature_id),
        )
    metrics = {key: result.metrics[key] for key in PUBLIC_METRIC_KEYS}
    return {
        "feature_id": block.feature_id,
        "n_valid": result.n_valid,
        "fit_status": result.fit_status,
        "model_selection": result.model_selection,
        "selected_fit": result.selected_fit,
        "assignment_stability": result.assignment_stability,
        "metrics": metrics,
    }


def _feature_fit_seed(base_seed: int, feature_id: int) -> int:
    """Return the published deterministic vMF fit seed for one feature.

    The feature identifier multiplier is part of the production reproducibility
    contract and must be shared by published fits and identical full-cloud
    stability refits.
    """
    # Derive the feature-local fit seed from the single published arithmetic rule.
    return feature_fit_seed(base_seed, feature_id)


def _record_completed_futures(
    futures: Iterable[Future[dict[str, Any]]],
    features: list[dict[str, Any]],
    checkpoint: _VmfCheckpointState,
    progress: _VmfProgressLogger,
) -> None:
    first_error: Exception | None = None
    for future in futures:
        try:
            feature = future.result()
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        features.append(feature)
        checkpoint.record(feature)
        progress.advance()
    if first_error is not None:
        raise first_error


class _BoundedLinearMaterializer:
    """Limit simultaneous feature-by-vocabulary tensors independently of workers."""

    def __init__(self, *, unembedding: torch.Tensor, max_buffers: int) -> None:
        """Bind one canonical readout and a dedicated vocabulary-buffer permit pool."""
        # Keep the model-owned readout by reference; only derived coordinates are bounded.
        self.unembedding = unembedding
        self._permits = threading.BoundedSemaphore(int(max_buffers))

    @contextmanager
    def coordinates(self, block: FeatureEffectBlock):
        """Yield one normalized linear block and release it before returning the permit."""
        # Skipped features never allocate a vocabulary-sized tensor.
        if block.skipped_reason is not None or block.rows is None:
            yield None
            return
        self._permits.acquire()
        coordinates: torch.Tensor | None = None
        try:
            coordinates = materialize_linear_coordinates(
                block.rows, self.unembedding
            )
            yield coordinates
        finally:
            del coordinates
            self._permits.release()


def materialize_linear_coordinates(
    final_resid_direction: torch.Tensor,
    unembedding: torch.Tensor,
) -> torch.Tensor:
    """Return exact unit linear coordinates ``direction @ W_U.T`` on CPU.

    The input rows are the canonical retained final-residual directions. The
    returned feature-by-vocabulary tensor is deliberately ephemeral and must
    be released by the caller immediately after vMF scoring.
    """
    # Fill one CPU output buffer from bounded device chunks, then normalize in place.
    if final_resid_direction.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("final_resid directions and W_U must be rank-2 tensors.")
    if final_resid_direction.shape[1] != unembedding.shape[1]:
        raise ValueError("final_resid hidden width does not match canonical W_U.")
    device_rows = final_resid_direction.to(
        device=unembedding.device, dtype=torch.float32
    )
    coordinates = torch.empty(
        (final_resid_direction.shape[0], unembedding.shape[0]),
        dtype=torch.float32,
        device="cpu",
    )
    for start in range(0, unembedding.shape[0], _VOCAB_MATERIALIZATION_CHUNK_SIZE):
        end = min(start + _VOCAB_MATERIALIZATION_CHUNK_SIZE, unembedding.shape[0])
        readout_chunk = unembedding[start:end].to(dtype=torch.float32)
        coordinate_chunk = device_rows @ readout_chunk.T
        coordinates[:, start:end].copy_(coordinate_chunk.detach().cpu())
        del coordinate_chunk, readout_chunk
    norms = torch.linalg.vector_norm(coordinates, dim=1)
    if not torch.isfinite(norms).all() or bool(torch.any(norms <= 0).item()):
        raise ValueError("Materialized linear coordinates contain invalid norms.")
    coordinates.div_(norms[:, None])
    return coordinates


def _validated_canonical_unembedding(
    config: FEGAPipelineConfig,
    inputs: GeometryMetricsInputs,
    resources: ModelResources,
) -> torch.Tensor:
    """Bind slice-1 Gram provenance to the actual model output embeddings."""
    # Resolve the actual model and require every recorded readout identity field.
    model, _, _ = resources.get_model_and_sae()
    unembedding = canonical_unembedding(model)
    metadata = inputs.manifest.get("gram_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("final_resid manifest missing Gram/readout metadata.")
    model_config = getattr(model, "config", None)
    checkpoint_identity = (
        getattr(model_config, "name_or_path", None)
        or getattr(model_config, "_name_or_path", None)
        or getattr(model, "name_or_path", None)
        or getattr(config, "sae_repo_id", None)
    )
    if checkpoint_identity != metadata.get("checkpoint_identity"):
        raise ValueError("vMF model checkpoint identity does not match Gram metadata.")
    if metadata.get("readout_name") != "final_resid":
        raise ValueError("vMF Gram readout_name must be 'final_resid'.")
    if list(unembedding.shape) != list(metadata.get("unembedding_shape") or []):
        raise ValueError("vMF canonical W_U shape does not match Gram metadata.")
    if str(unembedding.dtype) != metadata.get("unembedding_dtype"):
        raise ValueError("vMF canonical W_U dtype does not match Gram metadata.")
    if unembedding_fingerprint(unembedding) != metadata.get(
        "unembedding_fingerprint"
    ):
        raise ValueError("vMF canonical W_U fingerprint does not match Gram metadata.")
    gram = resolve_final_resid_gram(inputs, resources)
    del gram
    return unembedding


def _coerce_feature_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"feature_id must be an integer, got {value!r}.")
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"feature_id must be an integer, got {value!r}.") from exc


class _VmfCheckpointState:
    def __init__(
        self,
        *,
        enabled: bool,
        path: Path,
        fingerprint: dict[str, Any],
        expected_feature_ids: set[int],
        workers: int,
        flush_every_features: int,
        features_by_id: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.path = path
        self.fingerprint = fingerprint
        self.expected_feature_ids = expected_feature_ids
        self.workers = workers
        self.flush_every_features = max(1, int(flush_every_features))
        self._features_by_id = dict(features_by_id or {})
        self._pending_since_flush = 0

    @classmethod
    def load(
        cls,
        *,
        enabled: bool,
        path: Path,
        fingerprint: dict[str, Any],
        expected_feature_ids: set[int],
        workers: int,
        flush_every_features: int,
    ) -> _VmfCheckpointState:
        if not enabled:
            return cls(
                enabled=False,
                path=path,
                fingerprint=fingerprint,
                expected_feature_ids=expected_feature_ids,
                workers=workers,
                flush_every_features=flush_every_features,
            )
        try:
            payload = load_vmf_checkpoint(path)
        except Exception as exc:
            _logger.warning(
                "vmf resume checkpoint ignored: path=%s reason=%s", path, exc
            )
            payload = None
        features_by_id: dict[int, dict[str, Any]] = {}
        if payload is not None:
            if vmf_scientific_compatibility_identity(
                payload.get("fingerprint")
            ) != vmf_scientific_compatibility_identity(fingerprint):
                _logger.info(
                    "vmf resume checkpoint ignored: path=%s reason=fingerprint_mismatch",
                    path,
                )
            else:
                try:
                    features_by_id = _validate_checkpoint_payload(
                        payload, expected_feature_ids
                    )
                except ValueError as exc:
                    _logger.warning(
                        "vmf resume checkpoint ignored: path=%s reason=%s", path, exc
                    )
        return cls(
            enabled=True,
            path=path,
            fingerprint=fingerprint,
            expected_feature_ids=expected_feature_ids,
            workers=workers,
            flush_every_features=flush_every_features,
            features_by_id=features_by_id,
        )

    @property
    def completed_ids(self) -> set[int]:
        return set(self._features_by_id)

    @property
    def completed_count(self) -> int:
        return len(self._features_by_id)

    def features(self) -> list[dict[str, Any]]:
        return [
            self._features_by_id[feature_id]
            for feature_id in sorted(self._features_by_id)
        ]

    def record(self, feature: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = _validate_checkpoint_feature_record(
            feature, self.expected_feature_ids
        )
        self._features_by_id[int(record["feature_id"])] = record
        self._pending_since_flush += 1
        if self._pending_since_flush >= self.flush_every_features:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self._pending_since_flush <= 0:
            return
        write_vmf_checkpoint(self.path, self._payload())
        self._pending_since_flush = 0

    def _payload(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": 2,
            "phase": "vmf",
            "effect_space": self.fingerprint["effect_space"],
            "source_readout": self.fingerprint["source_readout"],
            "workers": self.workers,
            "flush_every_features": self.flush_every_features,
            "fingerprint": self.fingerprint,
            "features": [
                {
                    "record": record,
                    "record_sha256": canonical_json_digest(record),
                }
                for record in self.features()
            ],
        }


def _validate_checkpoint_payload(
    payload: dict[str, Any], expected_feature_ids: set[int]
) -> dict[int, dict[str, Any]]:
    """Validate checkpoint metadata and every complete independent feature state."""
    # Reject partial or duplicate feature states before allowing resume reuse.
    if payload.get("checkpoint_schema_version") != 2:
        raise ValueError("checkpoint_schema_version mismatch")
    if payload.get("phase") != "vmf":
        raise ValueError("phase mismatch")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("features must be a list")
    features_by_id: dict[int, dict[str, Any]] = {}
    for feature in features:
        if not isinstance(feature, dict) or set(feature) != {
            "record",
            "record_sha256",
        }:
            raise ValueError("feature checkpoint wrapper keys mismatch")
        record = feature["record"]
        try:
            record_sha256 = canonical_json_digest(record)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature checkpoint record is not canonical JSON") from exc
        if feature["record_sha256"] != record_sha256:
            raise ValueError("feature checkpoint record_sha256 mismatch")
        record = _validate_checkpoint_feature_record(record, expected_feature_ids)
        feature_id = int(record["feature_id"])
        if feature_id in features_by_id:
            raise ValueError(f"duplicate feature_id {feature_id}")
        features_by_id[feature_id] = record
    return features_by_id


def _validate_checkpoint_feature_record(
    feature: Any, expected_feature_ids: set[int]
) -> dict[str, Any]:
    """Validate exact checkpoint state dimensions rather than flattened metrics alone."""
    # Require every orthogonal fit-state dimension and preserve it without projection.
    if not isinstance(feature, dict):
        raise ValueError("feature record must be a mapping")
    feature_id = _coerce_feature_id(feature.get("feature_id"))
    if feature_id not in expected_feature_ids:
        raise ValueError(f"unexpected feature_id {feature_id}")
    metrics = feature.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"feature {feature_id} metrics must be a mapping")
    if set(metrics) != set(PUBLIC_METRIC_KEYS):
        raise ValueError(f"feature {feature_id} metrics keys mismatch")
    required = {
        "feature_id",
        "n_valid",
        "fit_status",
        "model_selection",
        "selected_fit",
        "assignment_stability",
        "metrics",
    }
    if set(feature) != required:
        raise ValueError(f"feature {feature_id} state keys mismatch")
    if feature.get("fit_status") not in {
        "fitted",
        "insufficient_contexts",
        "no_finite_candidate",
        "fit_failed",
    }:
        raise ValueError(f"feature {feature_id} fit_status invalid")
    model_selection = feature.get("model_selection")
    if not isinstance(model_selection, dict):
        raise ValueError(f"feature {feature_id} model_selection must be a mapping")
    if set(model_selection) != {
        "selected_mode_count",
        "bic_tolerance",
        "candidates",
        "attempted_count",
        "finite_count",
        "nonfinite_count",
        "failed_count",
    }:
        raise ValueError(f"feature {feature_id} model_selection keys mismatch")
    assignment_stability = feature.get("assignment_stability")
    if not isinstance(assignment_stability, dict):
        raise ValueError(f"feature {feature_id} assignment_stability must be a mapping")
    if set(assignment_stability) != {
        "status",
        "value",
        "requested_count",
        "successful_count",
        "failed_count",
        "replicates",
    }:
        raise ValueError(f"feature {feature_id} assignment_stability keys mismatch")
    if feature["fit_status"] == "fitted":
        selected_fit = feature.get("selected_fit")
        if not isinstance(selected_fit, dict) or set(selected_fit) != {
            "weights",
            "kappas",
            "hard_mode_counts",
            "hard_assignments",
        }:
            raise ValueError(f"feature {feature_id} selected_fit keys mismatch")
        if metrics["selected_mode_count"] != model_selection["selected_mode_count"]:
            raise ValueError(f"feature {feature_id} selected mode identity mismatch")
    elif feature.get("selected_fit") is not None or metrics["selected_mode_count"] is not None:
        raise ValueError(f"feature {feature_id} non-fitted state has selected model")
    return {
        "feature_id": feature_id,
        "n_valid": int(feature["n_valid"]),
        "fit_status": feature["fit_status"],
        "model_selection": model_selection,
        "selected_fit": feature["selected_fit"],
        "assignment_stability": assignment_stability,
        "metrics": {key: metrics[key] for key in PUBLIC_METRIC_KEYS},
    }


class _VmfProgressLogger:
    def __init__(
        self,
        *,
        total_features: int,
        workers: int,
        initial_processed: int = 0,
        log_interval_seconds: float = 60.0,
    ) -> None:
        self.total_features = max(0, int(total_features))
        self.workers = workers
        self.log_interval_seconds = log_interval_seconds
        self.processed = max(0, int(initial_processed))
        self._started_at = 0.0
        self._last_progress_at = 0.0

    def start(self) -> None:
        now = _monotonic()
        self._started_at = now
        self._last_progress_at = now
        _logger.info(
            "vmf start: features_total=%d workers=%d",
            self.total_features,
            self.workers,
        )

    def advance(self, count: int = 1) -> None:
        self.processed += count
        now = _monotonic()
        if now - self._last_progress_at < self.log_interval_seconds:
            return
        self._last_progress_at = now
        self._log("vmf progress", now)

    def finish(self) -> None:
        self._log("vmf progress complete", _monotonic())

    def _log(self, label: str, now: float) -> None:
        elapsed_seconds = max(0.0, now - self._started_at)
        rate_per_minute = self._rate_per_minute(elapsed_seconds)
        _logger.info(
            (
                "%s: processed=%d/%d percent=%.1f elapsed=%s eta=%s "
                "features_per_min=%.2f workers=%d"
            ),
            label,
            self.processed,
            self.total_features,
            self._percent(),
            _format_duration(elapsed_seconds),
            _format_duration(self._eta_seconds(rate_per_minute)),
            rate_per_minute,
            self.workers,
        )

    def _percent(self) -> float:
        if self.total_features == 0:
            return 100.0
        return min(100.0, (float(self.processed) / float(self.total_features)) * 100.0)

    def _rate_per_minute(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0.0 or self.processed <= 0:
            return 0.0
        return (float(self.processed) / elapsed_seconds) * 60.0

    def _eta_seconds(self, rate_per_minute: float) -> float | None:
        if self.total_features == 0 or self.processed >= self.total_features:
            return 0.0
        if self.processed <= 0 or rate_per_minute <= 0.0:
            return None
        remaining = max(0, self.total_features - self.processed)
        return float(remaining) / (rate_per_minute / 60.0)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
