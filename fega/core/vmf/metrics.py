from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, overload

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from fega.config_schema import DirectionalMixtureFitConfig
from fega.core.geometry_metrics.metrics import normalize_logit_deltas
from fega.core.vmf.factor_reuse import FeatureFactorView
from fega.core.vmf.fit import (
    VmfCandidate,
    VmfFit,
    finalize_vmf_candidate,
    fit_vmf_candidate,
    fit_vmf_mixture,
    production_factor_policy,
    rerun_vmf_candidate_dense,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor import (
    bic_decision_is_ambiguous,
)

PUBLIC_METRIC_KEYS = (
    "selected_mode_count",
    "delta_mix",
    "mode_mass_min",
    "min_mode_c_ray",
    "mode_kappa_min",
)
FITTING_ELIGIBILITY_MIN_ROWS = 8
NORMALIZATION_EPSILON = 1.0e-12
SEED_DERIVATION_VERSION = 1

FitFunction = Callable[[torch.Tensor, int, DirectionalMixtureFitConfig, int], VmfFit]
InternalFit = VmfFit | VmfCandidate


@dataclass(frozen=True)
class VmfFeatureResult:
    """Carry orthogonal operational, selection, fit, and stability dimensions."""

    metrics: dict[str, float | int | None]
    n_valid: int
    fit_status: str
    model_selection: dict[str, Any]
    selected_fit: dict[str, Any] | None
    assignment_stability: dict[str, Any]


def empty_metrics() -> dict[str, float | int | None]:
    """Return raw reporting inputs for a feature with no selected model."""
    # Keep every flattened consumer field explicit without fabricating a mode count.
    return {
        "selected_mode_count": None,
        "delta_mix": None,
        "mode_mass_min": None,
        "min_mode_c_ray": None,
        "mode_kappa_min": None,
    }


def default_fit_fn(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
    """Run one fixed-mode fit with the configuration's complete init budget."""
    # Delegate initialization selection to the validated numerical fit authority.
    return fit_vmf_mixture(
        rows,
        k=k,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        seed=seed,
        backend=cfg.backend,
        gpu_device=cfg.gpu_device,
    )


def _fit_with_optional_trace(
    rows: torch.Tensor,
    k: int,
    cfg: DirectionalMixtureFitConfig,
    seed: int,
    fit_fn: FitFunction,
    trace: dict[str, Any] | None,
    prepared: FeatureFactorView | None = None,
) -> InternalFit:
    """Return a center-free production candidate without widening test doubles."""
    # Custom fit functions retain the frozen four-argument compatibility contract.
    if fit_fn is default_fit_fn:
        if prepared is None:
            return fit_vmf_candidate(
                rows,
                k=k,
                n_init=cfg.n_init,
                max_iter=cfg.max_iter,
                seed=seed,
                trace=trace,
                backend=cfg.backend,
                gpu_device=cfg.gpu_device,
            )
        return fit_vmf_candidate(
            rows,
            k=k,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter,
            seed=seed,
            trace=trace,
            prepared=prepared,
            backend=cfg.backend,
            gpu_device=cfg.gpu_device,
        )
    return fit_fn(rows, k, cfg, seed)


def score_vmf_feature(
    rows: torch.Tensor | None,
    cfg: DirectionalMixtureFitConfig,
    *,
    seed: int,
    fit_fn: FitFunction = default_fit_fn,
    assignment_stability_enabled: bool = True,
    trace: dict[str, Any] | None = None,
    prepared: FeatureFactorView | None = None,
) -> VmfFeatureResult:
    """Fit every feasible mode count, select finite BIC, and build reporting state.

    Fitting eligibility is operational and fixed. No directional-concentration
    metrics or scientific thresholds participate in fitting or model selection.
    Each configured feasible mode count is attempted independently so one model
    failure cannot suppress later candidates.
    """
    # Reuse prepared row identity when present, otherwise normalize canonically.
    started = perf_counter()
    if trace is not None:
        trace.clear()
        trace.update(
            {
                "backend": "dense_cpu",
                "seed": int(seed),
                "n_rows_input": 0 if rows is None else int(rows.shape[0]),
                "ambient_dim": None if rows is None else int(rows.shape[1]),
                "candidate_order": [int(k) for k in sorted(set(cfg.k_values))],
                "assignment_stability_enabled": bool(assignment_stability_enabled),
            }
        )
    if rows is None:
        result = _nonfitted_result(0, "insufficient_contexts", (), cfg)
        _finalize_feature_trace(trace, result, [], None, None, started, started)
        return result
    if prepared is None:
        unit_rows, counts = normalize_logit_deltas(
            rows, eps=NORMALIZATION_EPSILON
        )
    else:
        unit_rows = prepared.unit_rows
        counts = {
            "n_total": int(unit_rows.shape[0]),
            "n_valid": int(unit_rows.shape[0]),
            "skipped_nonfinite": 0,
            "skipped_zero_norm": 0,
        }
    normalized_at = perf_counter()
    n_valid = int(counts["n_valid"])
    if n_valid < FITTING_ELIGIBILITY_MIN_ROWS:
        result = _nonfitted_result(n_valid, "insufficient_contexts", (), cfg)
        _finalize_feature_trace(
            trace, result, [], None, None, started, normalized_at
        )
        return result

    evidence: list[dict[str, Any]] = []
    finite_fits: list[InternalFit] = []
    candidate_traces: list[dict[str, Any]] = []
    for k in sorted(set(cfg.k_values)):
        if k > n_valid:
            continue
        candidate_seed = derived_vmf_seed(seed, k, -1, "candidate_fit")
        fit_trace: dict[str, Any] | None = {} if trace is not None else None
        candidate_trace: dict[str, Any] = {
            "mode_count": int(k),
            "seed": int(candidate_seed),
        }
        try:
            fit = _fit_with_optional_trace(
                unit_rows,
                k,
                cfg,
                candidate_seed,
                fit_fn,
                fit_trace,
                prepared,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            evidence.append(_candidate_evidence(k, "fit_failed", candidate_seed))
            candidate_trace.update(
                {
                    "status": "fit_failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            if fit_trace:
                candidate_trace["fit_trace"] = fit_trace
            candidate_traces.append(candidate_trace)
            continue
        likelihood = _finite_float(fit.log_likelihood)
        bic = (
            _finite_float(bic_score(fit, n_valid=n_valid, dim=int(unit_rows.shape[1])))
            if likelihood is not None
            else None
        )
        if likelihood is None or bic is None:
            evidence.append(_candidate_evidence(k, "nonfinite", candidate_seed))
            candidate_trace.update({"status": "nonfinite"})
            if fit_trace:
                candidate_trace["fit_trace"] = fit_trace
            candidate_traces.append(candidate_trace)
            continue
        evidence.append(
            _candidate_evidence(k, "finite", candidate_seed, likelihood, bic)
        )
        finite_fits.append(fit)
        candidate_trace.update(
            {
                "status": "finite",
                "log_likelihood": float(likelihood),
                "bic": float(bic),
            }
        )
        if fit_trace:
            candidate_trace["fit_trace"] = fit_trace
        candidate_traces.append(candidate_trace)

    candidates_at = perf_counter()

    if not finite_fits:
        status = (
            "fit_failed"
            if evidence and all(item["status"] == "fit_failed" for item in evidence)
            else "no_finite_candidate"
        )
        result = _nonfitted_result(n_valid, status, tuple(evidence), cfg)
        _finalize_feature_trace(
            trace,
            result,
            candidate_traces,
            None,
            None,
            started,
            normalized_at,
            candidates_at=candidates_at,
        )
        return result

    selection_trace: dict[str, Any] | None = {} if trace is not None else None
    finite_fits = _resolve_ambiguous_factor_bic(
        finite_fits,
        unit_rows,
        evidence,
        candidate_traces,
        n_valid=n_valid,
        dim=int(unit_rows.shape[1]),
        tolerance=cfg.bic_tolerance,
    )
    selected_internal = select_by_bic(
        finite_fits,
        n_valid=n_valid,
        dim=int(unit_rows.shape[1]),
        tolerance=cfg.bic_tolerance,
        trace=selection_trace,
    )
    selected = (
        finalize_vmf_candidate(selected_internal, unit_rows)
        if isinstance(selected_internal, VmfCandidate)
        else selected_internal
    )
    selected_at = perf_counter()
    assignment_trace: dict[str, Any] | None = {} if trace is not None else None
    stability = (
        assignment_stability(
            unit_rows,
            np.asarray(selected.labels, dtype=np.int64),
            selected.k,
            cfg,
            seed=seed,
            fit_fn=fit_fn,
            trace=assignment_trace,
            prepared=prepared,
        )
        if assignment_stability_enabled
        else _stability_result("not_evaluated", None, 0, 0, 0, [])
    )
    stability_at = perf_counter()
    metrics = metrics_from_fit(unit_rows, selected, stability)
    result = VmfFeatureResult(
        metrics=metrics,
        n_valid=n_valid,
        fit_status="fitted",
        model_selection=_model_selection(selected.k, cfg, tuple(evidence)),
        selected_fit=_selected_fit(selected),
        assignment_stability=stability,
    )
    _finalize_feature_trace(
        trace,
        result,
        candidate_traces,
        selection_trace,
        assignment_trace,
        started,
        normalized_at,
        candidates_at=candidates_at,
        selected_at=selected_at,
        stability_at=stability_at,
    )
    return result


def _resolve_ambiguous_factor_bic(
    candidates: list[InternalFit],
    unit_rows: torch.Tensor,
    evidence: list[dict[str, Any]],
    candidate_traces: list[dict[str, Any]],
    *,
    n_valid: int,
    dim: int,
    tolerance: float,
) -> list[InternalFit]:
    """Rerun every factor candidate densely when cross-mode BIC is ambiguous.

    BIC selection is one joint decision over the finite candidate set. If any
    factor-involved comparison enters its accepted calibration band, every
    factor member of that decision is restarted from its own original seed and
    fixed budget.
    """
    # Skip dense-only and unambiguous selections without changing their ordering.
    if not _factor_bic_is_ambiguous(
        candidates, n_valid=n_valid, dim=dim, tolerance=tolerance
    ):
        return candidates
    resolved: list[InternalFit] = []
    for fit in candidates:
        if not isinstance(fit, VmfCandidate) or fit.backend == "dense_cpu":
            resolved.append(fit)
            continue
        dense_trace: dict[str, Any] = {}
        dense = rerun_vmf_candidate_dense(
            fit,
            unit_rows,
            reason="bic_selection_ambiguity",
            trace=dense_trace,
        )
        resolved.append(dense)
        dense_bic = bic_score(dense, n_valid=n_valid, dim=dim)
        for record in evidence:
            if record.get("mode_count") == fit.k and record.get("status") == "finite":
                record.update(
                    {
                        "log_likelihood": float(dense.log_likelihood),
                        "bic": float(dense_bic),
                    }
                )
                break
        for record in candidate_traces:
            if record.get("mode_count") == fit.k and record.get("status") == "finite":
                if "fit_trace" in record:
                    record["pre_bic_factor_trace"] = record["fit_trace"]
                record.update(
                    {
                        "log_likelihood": float(dense.log_likelihood),
                        "bic": float(dense_bic),
                        "fit_trace": dense_trace,
                        "bic_resolution": "dense_cpu",
                    }
                )
                break
    return resolved


def _factor_bic_is_ambiguous(
    candidates: list[InternalFit],
    *,
    n_valid: int,
    dim: int,
    tolerance: float,
) -> bool:
    """Evaluate real cross-candidate comparisons against the calibrated BIC band."""
    # Mirror the frozen selection loop while excluding its inert self-comparison.
    finite = [
        fit
        for fit in candidates
        if _finite_float(fit.log_likelihood) is not None
        and _finite_float(bic_score(fit, n_valid=n_valid, dim=dim)) is not None
    ]
    if len(finite) < 2:
        return False
    policy = production_factor_policy()
    best = min(finite, key=lambda fit: fit.k)
    best_score = bic_score(best, n_valid=n_valid, dim=dim)
    for fit in sorted(finite, key=lambda item: item.k):
        if fit is best:
            continue
        score = bic_score(fit, n_valid=n_valid, dim=dim)
        factor_involved = (
            isinstance(fit, VmfCandidate) and fit.backend != "dense_cpu"
        ) or (
            isinstance(best, VmfCandidate) and best.backend != "dense_cpu"
        )
        if factor_involved and bic_decision_is_ambiguous(
            score, best_score, tolerance, policy
        ):
            return True
        if score < best_score - tolerance:
            best = fit
            best_score = score
    return False


def select_by_bic(
    candidates: list[InternalFit],
    *,
    n_valid: int,
    dim: int,
    tolerance: float = 1.0e-9,
    trace: dict[str, Any] | None = None,
) -> InternalFit:
    """Select the smallest finite BIC with smaller-M tolerance tie breaking."""
    # Filter non-finite candidates defensively, then compare in mode-count order.
    finite = [
        fit
        for fit in candidates
        if _finite_float(fit.log_likelihood) is not None
        and _finite_float(bic_score(fit, n_valid=n_valid, dim=dim)) is not None
    ]
    if not finite:
        raise ValueError("Cannot select vMF model without a finite BIC candidate.")
    selected_mode_count, comparisons = select_mode_count_from_bic_records(
        [
            {
                "mode_count": int(fit.k),
                "bic": bic_score(fit, n_valid=n_valid, dim=dim),
            }
            for fit in finite
        ],
        tolerance=tolerance,
        return_comparisons=True,
    )
    best = next(fit for fit in finite if int(fit.k) == selected_mode_count)
    best_score = bic_score(best, n_valid=n_valid, dim=dim)
    if trace is not None:
        trace.clear()
        trace.update(
            {
                "operator": "candidate_bic < incumbent_bic - tolerance",
                "tolerance": float(tolerance),
                "comparisons": comparisons,
                "selected_mode_count": int(best.k),
                "selected_bic": float(best_score),
            }
        )
    return best


@overload
def select_mode_count_from_bic_records(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
    return_comparisons: Literal[False] = False,
) -> int: ...


@overload
def select_mode_count_from_bic_records(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
    return_comparisons: Literal[True],
) -> tuple[int, list[dict[str, Any]]]: ...


def select_mode_count_from_bic_records(
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
    return_comparisons: bool = False,
) -> int | tuple[int, list[dict[str, Any]]]:
    """Reselect the persisted finite BIC candidate with production tie semantics.

    Validation uses this same comparison authority as live selection. The helper
    consumes already-computed BIC values and therefore performs no fit, likelihood,
    parameter-count, or backend computation.
    """
    # Smaller mode count is the initial incumbent and wins tolerance ties.
    finite = []
    for candidate in candidates:
        mode_count = int(candidate["mode_count"])
        bic = _finite_float(candidate.get("bic"))
        if bic is not None:
            finite.append((mode_count, bic))
    if not finite:
        raise ValueError("Cannot select vMF model without a finite BIC candidate.")
    best_mode, best_score = min(finite, key=lambda item: item[0])
    comparisons: list[dict[str, Any]] = []
    for mode_count, score in sorted(finite):
        threshold = best_score - tolerance
        improved = score < threshold
        comparisons.append(
            {
                "mode_count": mode_count,
                "bic": float(score),
                "incumbent_mode_count": best_mode,
                "incumbent_bic": float(best_score),
                "strict_threshold": float(threshold),
                "strict_improvement": bool(improved),
            }
        )
        if improved:
            best_mode = mode_count
            best_score = score
    if return_comparisons:
        return best_mode, comparisons
    return best_mode


def bic_score(fit: InternalFit, *, n_valid: int, dim: int) -> float:
    """Compute the paper's full-likelihood vMF mixture BIC."""
    # Count directional means, concentrations, and simplex weights exactly once.
    param_count = fit.k * (dim - 1) + fit.k + (fit.k - 1)
    return -2.0 * float(fit.log_likelihood) + float(param_count) * math.log(n_valid)


def metrics_from_fit(
    unit_rows: torch.Tensor,
    fit: VmfFit,
    stability: dict[str, Any],
) -> dict[str, float | int | None]:
    """Derive raw reporting inputs from one immutable BIC-selected fit.

    Fitted mixture weights drive mass and mixture gain. Hard assignments are
    used only to determine whether each within-mode ray statistic has at least
    two observations and is therefore mathematically defined.
    """
    # Evaluate selected-fit diagnostics without applying any acceptance threshold.
    labels = np.asarray(fit.labels, dtype=np.int64)
    weights = np.asarray(fit.weights, dtype=np.float64)
    global_c_ray = c_ray_unit_rows(unit_rows)
    mode_c_rays: list[float] = []
    all_defined = True
    for mode in range(fit.k):
        mask = labels == mode
        if int(mask.sum()) < 2:
            all_defined = False
            continue
        c_ray = c_ray_unit_rows(unit_rows[torch.from_numpy(mask)])
        if c_ray is None:
            all_defined = False
            continue
        mode_c_rays.append(c_ray)
    min_mode_c_ray = (
        min(mode_c_rays) if all_defined and len(mode_c_rays) == fit.k else None
    )
    valid_weights = weights.size == fit.k and np.all(np.isfinite(weights))
    mode_mass_min = float(np.min(weights)) if valid_weights and weights.size else None
    delta_mix = None
    if (
        global_c_ray is not None
        and all_defined
        and len(mode_c_rays) == fit.k
        and valid_weights
    ):
        delta_mix = float(np.dot(weights, np.asarray(mode_c_rays))) - global_c_ray
    return {
        "selected_mode_count": int(fit.k),
        "delta_mix": _json_float(delta_mix),
        "mode_mass_min": _json_float(mode_mass_min),
        "min_mode_c_ray": _json_float(min_mode_c_ray),
        "mode_kappa_min": _json_float(_kappa_min(fit.kappas)),
    }


def assignment_stability(
    unit_rows: torch.Tensor,
    full_labels: np.ndarray,
    k: int,
    cfg: DirectionalMixtureFitConfig,
    *,
    seed: int,
    fit_fn: FitFunction,
    trace: dict[str, Any] | None = None,
    prepared: FeatureFactorView | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed, schedule-independent assignment-resampling plan.

    Every subset and refit seed is derived independently with SHA-256 from the
    feature seed, selected mode count, replicate id, and role. Subsets are
    predeclared before any refit, sorted, never replaced, and reduced in stable
    replicate order. Any required failed or non-finite refit makes the aggregate
    unavailable rather than averaging only successful replicates.
    """
    # Handle the single-mode case before declaring a multimode resampling schedule.
    if k <= 1:
        if trace is not None:
            trace.clear()
            trace.update(
                {
                    "requested_count": 0,
                    "successful_count": 0,
                    "failed_count": 0,
                    "status": "not_applicable",
                    "value": None,
                    "replicates": [],
                }
            )
        return _stability_result("not_applicable", None, 0, 0, 0, [])
    n_valid = int(unit_rows.shape[0])
    subset_n = min(n_valid, max(k, int(math.ceil(cfg.resample_fraction * n_valid))))
    plans: list[tuple[int, int, int, list[int]]] = []
    for replicate_id in range(cfg.resample_rounds):
        subset_seed = derived_vmf_seed(seed, k, replicate_id, "subset")
        refit_seed = derived_vmf_seed(seed, k, replicate_id, "refit")
        rng = np.random.default_rng(subset_seed)
        subset = np.sort(rng.choice(n_valid, size=subset_n, replace=False)).tolist()
        plans.append((replicate_id, subset_seed, refit_seed, subset))

    scores: list[float] = []
    replicate_evidence: list[dict[str, Any]] = []
    replicate_traces: list[dict[str, Any]] = []
    for replicate_id, subset_seed, refit_seed, subset in plans:
        subset_rows = unit_rows[torch.as_tensor(subset, dtype=torch.long)]
        status = "available"
        fit_trace: dict[str, Any] | None = {} if trace is not None else None
        subset_prepared = (
            prepared.select(subset)
            if prepared is not None and fit_fn is default_fit_fn
            else None
        )
        adjusted_rand: float | None = None
        try:
            refit = _fit_with_optional_trace(
                subset_rows,
                k,
                cfg,
                refit_seed,
                fit_fn,
                fit_trace,
                subset_prepared,
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            status = "fit_failed"
            error_type = type(error).__name__
            error_message = str(error)
        else:
            error_type = None
            error_message = None
            try:
                refit_bic = bic_score(
                    refit,
                    n_valid=len(subset),
                    dim=int(unit_rows.shape[1]),
                )
                if not math.isfinite(float(refit.log_likelihood)) or not math.isfinite(
                    refit_bic
                ):
                    status = "nonfinite"
                else:
                    adjusted_rand = float(
                        adjusted_rand_score(
                            full_labels[np.asarray(subset, dtype=np.int64)],
                            np.asarray(refit.labels, dtype=np.int64),
                        )
                    )
                    scores.append(adjusted_rand)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                status = "fit_failed"
                error_type = type(error).__name__
                error_message = str(error)
        replicate_record = {
            "replicate_id": replicate_id,
            "subset_seed": subset_seed,
            "refit_seed": refit_seed,
            "subset_indices": subset,
            "status": status,
        }
        if status == "available":
            replicate_record["adjusted_rand_score"] = adjusted_rand
        replicate_evidence.append(replicate_record)
        if trace is not None:
            replicate_trace: dict[str, Any] = {
                "replicate_id": int(replicate_id),
                "subset_seed": int(subset_seed),
                "refit_seed": int(refit_seed),
                "subset_indices": [int(value) for value in subset],
                "status": status,
            }
            if error_type is not None:
                replicate_trace.update(
                    {"error_type": error_type, "error_message": error_message}
                )
            if fit_trace:
                replicate_trace["fit_trace"] = fit_trace
            replicate_traces.append(replicate_trace)
    requested = len(plans)
    successful = len(scores)
    failed = requested - successful
    status = "available" if requested > 0 and failed == 0 else "unavailable"
    value = float(sum(scores) / requested) if status == "available" else None
    if trace is not None:
        trace.clear()
        trace.update(
            {
                "requested_count": int(requested),
                "successful_count": int(successful),
                "failed_count": int(failed),
                "status": status,
                "value": _json_float(value),
                "replicates": replicate_traces,
            }
        )
    return _stability_result(
        status, value, requested, successful, failed, replicate_evidence
    )


def _finalize_feature_trace(
    trace: dict[str, Any] | None,
    result: VmfFeatureResult,
    candidate_traces: list[dict[str, Any]],
    selection_trace: dict[str, Any] | None,
    assignment_trace: dict[str, Any] | None,
    started: float,
    normalized_at: float,
    *,
    candidates_at: float | None = None,
    selected_at: float | None = None,
    stability_at: float | None = None,
) -> None:
    """Attach non-scientific telemetry and every dense decision input to a trace."""
    # Keep timings outside the canonical result while preserving exact decision payloads.
    if trace is None:
        return
    candidates_at = normalized_at if candidates_at is None else candidates_at
    selected_at = candidates_at if selected_at is None else selected_at
    stability_at = selected_at if stability_at is None else stability_at
    finalized_at = perf_counter()
    nested_fit_traces = [
        candidate["fit_trace"]
        for candidate in candidate_traces
        if isinstance(candidate.get("fit_trace"), dict)
    ]
    if isinstance(assignment_trace, dict):
        nested_fit_traces.extend(
            replicate["fit_trace"]
            for replicate in assignment_trace.get("replicates", [])
            if isinstance(replicate.get("fit_trace"), dict)
        )
    finite_candidates = sum(
        candidate.get("status") == "finite" for candidate in candidate_traces
    )
    selection_bic_evaluations = (
        2 * finite_candidates + 1 if selection_trace is not None else 0
    )
    trace.update(
        {
            "n_valid": int(result.n_valid),
            "fit_status": result.fit_status,
            "candidates": candidate_traces,
            "bic_selection": selection_trace,
            "model_selection": result.model_selection,
            "selected_fit": result.selected_fit,
            "assignment_stability": result.assignment_stability,
            "assignment_trace": assignment_trace,
            "reporting_inputs": result.metrics,
            "workload": {
                "candidate_fits_attempted": len(candidate_traces),
                "candidate_fits_finite": finite_candidates,
                "candidate_fits_nonfinite": sum(
                    candidate.get("status") == "nonfinite"
                    for candidate in candidate_traces
                ),
                "candidate_fits_failed": sum(
                    candidate.get("status") == "fit_failed"
                    for candidate in candidate_traces
                ),
                "assignment_refits_requested": int(
                    result.assignment_stability["requested_count"]
                ),
                "assignment_refits_successful": int(
                    result.assignment_stability["successful_count"]
                ),
                "assignment_refits_failed": int(
                    result.assignment_stability["failed_count"]
                ),
                "initialization_attempts": sum(
                    int(item.get("workload", {}).get("initialization_attempts", 0))
                    for item in nested_fit_traces
                ),
                "iterations": sum(
                    int(item.get("workload", {}).get("iterations", 0))
                    for item in nested_fit_traces
                ),
                "kmeans_selections": sum(
                    int(item.get("workload", {}).get("kmeans_selections", 0))
                    for item in nested_fit_traces
                ),
                "fallback_components": sum(
                    int(item.get("workload", {}).get("fallback_components", 0))
                    for item in nested_fit_traces
                ),
                "bic_evaluations": finite_candidates
                + selection_bic_evaluations
                + int(result.assignment_stability["successful_count"]),
            },
            "timings_seconds": {
                "normalization": float(normalized_at - started),
                "candidate_fits": float(candidates_at - normalized_at),
                "bic_selection": float(selected_at - candidates_at),
                "assignment_stability": float(stability_at - selected_at),
                "reporting": float(finalized_at - stability_at),
                "total": float(finalized_at - started),
            },
        }
    )


def c_ray_unit_rows(unit_rows: torch.Tensor) -> float | None:
    """Compute mean pairwise cosine for already normalized rows."""
    # Use the exact sum-vector identity and clamp only floating roundoff.
    n_valid = int(unit_rows.shape[0])
    if n_valid < 2:
        return None
    s_norm_sq = float(unit_rows.sum(dim=0).square().sum(dtype=torch.float64).item())
    value = (s_norm_sq - float(n_valid)) / float(n_valid * (n_valid - 1))
    return max(-1.0, min(1.0, value))


def _nonfitted_result(
    n_valid: int,
    status: str,
    evidence: tuple[dict[str, Any], ...],
    cfg: DirectionalMixtureFitConfig,
) -> VmfFeatureResult:
    """Build a truthful non-fitted feature state without a sentinel model."""
    # Preserve attempted candidate evidence while leaving selection and fit absent.
    counts = _candidate_counts(evidence)
    return VmfFeatureResult(
        metrics=empty_metrics(),
        n_valid=n_valid,
        fit_status=status,
        model_selection={
            "selected_mode_count": None,
            "bic_tolerance": float(cfg.bic_tolerance),
            "candidates": list(evidence),
            **counts,
        },
        selected_fit=None,
        assignment_stability=_stability_result(
            "unavailable", None, 0, 0, 0, []
        ),
    )


def _candidate_evidence(
    mode_count: int,
    status: str,
    seed: int,
    log_likelihood: float | None = None,
    bic: float | None = None,
) -> dict[str, Any]:
    """Serialize one candidate with finite numbers only when selectable."""
    # Omit invalid numerical values so emitted JSON never contains NaN or infinity.
    record: dict[str, Any] = {
        "mode_count": mode_count,
        "status": status,
        "seed": seed,
    }
    if status == "finite":
        if log_likelihood is None or bic is None:
            raise ValueError("Finite vMF candidate evidence requires likelihood and BIC.")
        record["log_likelihood"] = float(log_likelihood)
        record["bic"] = float(bic)
    return record


def _model_selection(
    selected_mode_count: int,
    cfg: DirectionalMixtureFitConfig,
    evidence: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Serialize immutable BIC selection and complete attempt counts."""
    # Keep ordered evidence and derived counts adjacent to the selected identity.
    return {
        "selected_mode_count": selected_mode_count,
        "bic_tolerance": float(cfg.bic_tolerance),
        "candidates": list(evidence),
        **_candidate_counts(evidence),
    }


def _candidate_counts(evidence: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """Count attempted finite, non-finite, and failed model candidates."""
    # Derive all counts from serialized evidence to prevent state divergence.
    return {
        "attempted_count": len(evidence),
        "finite_count": sum(item["status"] == "finite" for item in evidence),
        "nonfinite_count": sum(item["status"] == "nonfinite" for item in evidence),
        "failed_count": sum(item["status"] == "fit_failed" for item in evidence),
    }


def _selected_fit(fit: VmfFit) -> dict[str, Any]:
    """Preserve only the BIC-selected parameters and hard assignments."""
    # Serialize compact selected-fit evidence without candidate responsibilities.
    labels = np.asarray(fit.labels, dtype=np.int64)
    return {
        "weights": [float(value) for value in np.asarray(fit.weights).tolist()],
        "kappas": (
            None
            if fit.kappas is None
            else [float(value) for value in np.asarray(fit.kappas).tolist()]
        ),
        "hard_mode_counts": [
            int(value) for value in np.bincount(labels, minlength=fit.k).tolist()
        ],
        "hard_assignments": [int(value) for value in labels.tolist()],
    }


def _stability_result(
    status: str,
    value: float | None,
    requested: int,
    successful: int,
    failed: int,
    replicates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete assignment-stability state with reproducibility evidence."""
    # Keep availability, aggregate, counts, and fixed replicate plan orthogonal.
    return {
        "status": status,
        "value": _json_float(value),
        "requested_count": requested,
        "successful_count": successful,
        "failed_count": failed,
        "replicates": replicates,
    }


def feature_fit_seed(base_seed: int, feature_id: int) -> int:
    """Return the existing published feature-local vMF fit seed."""
    # Preserve the established arithmetic identity used by standalone and stability fits.
    return int(base_seed) + int(feature_id) * 104729


def derived_vmf_seed(
    feature_seed: int,
    selected_k: int,
    replicate_id: int,
    role: str,
) -> int:
    """Derive a schedule-independent seed from canonical stable identifiers."""
    # Hash a typed delimiter-separated identity; the existing fit adapter owns folding.
    identity = f"vmf|{int(feature_seed)}|{int(selected_k)}|{int(replicate_id)}|{role}"
    return int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
    )


def _kappa_min(kappas: np.ndarray | None) -> float | None:
    """Return the minimum concentration only when every value is finite."""
    # Treat missing, empty, or non-finite concentration evidence as unavailable.
    if kappas is None:
        return None
    values = np.asarray(kappas, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(np.min(values))


def _finite_float(value: Any) -> float | None:
    """Coerce a scalar to finite float or return unavailable."""
    # Reject non-numeric and non-finite values before model selection or JSON output.
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_float(value: Any) -> float | None:
    """Return a JSON-safe finite float while preserving unavailable values."""
    # Reuse finite coercion so no raw NaN or infinity reaches artifacts.
    return _finite_float(value)
