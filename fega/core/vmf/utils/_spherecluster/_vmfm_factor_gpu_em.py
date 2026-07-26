"""Deterministic PyTorch/CUDA mechanics for the float64 factor vMF backend."""

from __future__ import annotations

import math
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

from ._vmf_numerics import log_vmf_normalizer_plus_kappa
from ._vmfm_factor import (
    MAX_CONCENTRATION,
    DenseRerunRequired,
    FactorCoordinates,
    FactorFit,
)
from ._vmfm_factor_em import (
    _factor_tolerance,
    _guard_comparison,
    _guard_gap,
    _validated_force_weights,
)
from ._vmfm_factor_gpu import GPU_BACKEND_NAME, GpuFactorAmbiguityPolicy

_EPS = np.finfo(np.float64).eps


def fit_factor_movmf_gpu_impl(
    factor: FactorCoordinates,
    z: torch.Tensor,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: GpuFactorAmbiguityPolicy,
    posterior_type: str,
    force_weights: np.ndarray | None,
    init: str,
    tol: float,
) -> FactorFit:
    """Fit one fixed-mode candidate with CPU-frozen ordering and CUDA arithmetic."""
    # Match the accepted CPU validation and seed schedule before any initialization.
    if init != "k-means++":
        raise DenseRerunRequired("unsupported_factor_initialization")
    if posterior_type not in {"soft", "hard"}:
        raise ValueError("posterior_type must be 'soft' or 'hard'")
    if n_init <= 0 or max_iter <= 0:
        raise ValueError("n_init and max_iter must be positive")
    n_rows = int(z.shape[0])
    if n_clusters <= 0 or n_clusters > n_rows:
        raise ValueError("n_clusters must be between one and the row count")
    weights_override = _validated_force_weights(force_weights, n_clusters)
    scaled_tolerance = _factor_tolerance(factor.z, factor.ambient_dim, tol)
    rng = check_random_state(int(random_state))
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_init)
    attempts: list[FactorFit] = []
    initialization_seconds = 0.0
    em_seconds = 0.0
    normalizer_seconds = 0.0
    for init_index, seed in enumerate(seeds):
        attempt, timing = _fit_one_initialization(
            factor,
            z,
            n_clusters=n_clusters,
            max_iter=max_iter,
            seed=int(seed),
            policy=policy,
            posterior_type=posterior_type,
            force_weights=weights_override,
            tolerance=scaled_tolerance,
            init_index=init_index,
        )
        attempts.append(attempt)
        initialization_seconds += timing["initialization"]
        em_seconds += timing["em"]
        normalizer_seconds += timing["normalizer"]
    selection_events: list[Mapping[str, Any]] = []
    if len(attempts) > 1:
        ordered = sorted(
            attempts, key=lambda item: (-item.log_likelihood, item.selected_init_index)
        )
        _guard_gap(
            policy,
            "initialization_likelihood",
            ordered[0].log_likelihood - ordered[1].log_likelihood,
            1.0e-8,
            selection_events,
            context={
                "candidate_indices": [
                    ordered[0].selected_init_index,
                    ordered[1].selected_init_index,
                ]
            },
        )
    best = attempts[0]
    for attempt in attempts[1:]:
        if attempt.log_likelihood > best.log_likelihood:
            best = attempt
    trace = {
        "backend": GPU_BACKEND_NAME,
        "ambient_dim": int(factor.ambient_dim),
        "factor_dim": int(z.shape[1]),
        "source_sha256": factor.source_sha256,
        "mode_count": int(n_clusters),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "scaled_tolerance": float(scaled_tolerance),
        "initializations": [dict(attempt.trace) for attempt in attempts],
        "selected_init_index": int(best.selected_init_index),
        "selected_log_likelihood": float(best.log_likelihood),
        "selection_guard_events": selection_events,
        "timings_seconds": {
            "initialization": float(initialization_seconds),
            "em": float(em_seconds),
            "normalizer": float(normalizer_seconds),
        },
    }
    return _copy_fit_with_trace(best, trace)


def _fit_one_initialization(
    factor: FactorCoordinates,
    z: torch.Tensor,
    *,
    n_clusters: int,
    max_iter: int,
    seed: int,
    policy: GpuFactorAmbiguityPolicy,
    posterior_type: str,
    force_weights: np.ndarray | None,
    tolerance: float,
    init_index: int,
) -> tuple[FactorFit, dict[str, float]]:
    """Run one ordered CUDA initialization and EM sequence."""
    # Keep random draws on NumPy's accepted RandomState while computing geometry on GPU.
    events: list[Mapping[str, Any]] = []
    torch.cuda.synchronize(z.device)
    started = perf_counter()
    centers, initialization = _factor_kmeans_plus_plus_gpu(
        z, n_clusters, check_random_state(seed), policy, events
    )
    torch.cuda.synchronize(z.device)
    initialized_at = perf_counter()
    weights = (
        torch.full(
            (n_clusters,),
            1.0 / n_clusters,
            dtype=torch.float64,
            device=z.device,
        )
        if force_weights is None
        else torch.as_tensor(force_weights, dtype=torch.float64, device=z.device)
    )
    kappas = torch.ones(n_clusters, dtype=torch.float64, device=z.device)
    iterations: list[Mapping[str, Any]] = []
    coefficients = torch.zeros(
        (n_clusters, int(z.shape[0])), dtype=torch.float64, device=z.device
    )
    normalizer_seconds = 0.0
    posterior = torch.empty((n_clusters, int(z.shape[0])), device=z.device)
    for iteration in range(max_iter):
        previous = centers.clone()
        posterior, score_gaps, elapsed_normalizer = _factor_expectation_gpu(
            factor, z, centers, weights, kappas, posterior_type
        )
        normalizer_seconds += elapsed_normalizer
        for row_index, gap in enumerate(score_gaps):
            _guard_gap(
                policy,
                "posterior_score",
                gap,
                1.0e-10,
                events,
                context={"iteration": iteration, "row_index": row_index},
            )
        centers, coefficients, weights, kappas, component_trace = (
            _factor_maximization_gpu(
                factor,
                z,
                posterior,
                force_weights,
                policy,
                events,
                iteration,
            )
        )
        shift = float(torch.sum((previous - centers) ** 2).item())
        _guard_comparison(
            policy,
            "convergence",
            shift,
            tolerance,
            1.0e-12,
            events,
            context={"iteration": iteration},
        )
        iterations.append(
            {
                "iteration": int(iteration),
                "posterior": posterior.detach().cpu().numpy().tolist(),
                "center_shift": shift,
                "tolerance": float(tolerance),
                "components": component_trace,
            }
        )
        if shift <= tolerance:
            break
    labels = torch.argmax(posterior, dim=0)
    inertia = _factor_inertia_gpu(z, centers, labels)
    likelihood, elapsed_normalizer = _factor_mixture_log_likelihood_gpu(
        factor, z, centers, weights, kappas
    )
    normalizer_seconds += elapsed_normalizer
    torch.cuda.synchronize(z.device)
    finished = perf_counter()
    trace = {
        "init_index": int(init_index),
        "seed": int(seed),
        "initialization": initialization,
        "iterations": iterations,
        "iteration_count": len(iterations),
        "converged": bool(iterations and iterations[-1]["center_shift"] <= tolerance),
        "labels": labels.detach().cpu().numpy().tolist(),
        "inertia": float(inertia),
        "log_likelihood": float(likelihood),
        "guard_events": events,
    }
    fit = FactorFit(
        centers=centers.detach().cpu().numpy(),
        center_coefficients=coefficients.detach().cpu().numpy(),
        labels=labels.detach().cpu().numpy().astype(np.int64, copy=False),
        responsibilities=posterior.detach().cpu().numpy(),
        weights=weights.detach().cpu().numpy(),
        kappas=kappas.detach().cpu().numpy(),
        inertia=float(inertia),
        log_likelihood=float(likelihood),
        selected_init_index=int(init_index),
        trace=trace,
    )
    return fit, {
        "initialization": float(initialized_at - started),
        "em": float(finished - initialized_at),
        "normalizer": float(normalizer_seconds),
    }


def _factor_kmeans_plus_plus_gpu(
    z: torch.Tensor,
    n_clusters: int,
    random_state: np.random.RandomState,
    policy: GpuFactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Mirror accepted k-means++ draws while evaluating distances on CUDA."""
    # Preserve the first-center and local-trial NumPy draw order exactly.
    n_rows, factor_dim = (int(z.shape[0]), int(z.shape[1]))
    n_local_trials = 2 + int(np.log(n_clusters))
    centers = torch.empty(
        (n_clusters, factor_dim), dtype=torch.float64, device=z.device
    )
    first = int(random_state.randint(n_rows))
    selected = [first]
    centers[0] = z[first] / torch.linalg.vector_norm(z[first])
    closest = torch.clamp(
        torch.sum((centers[0].unsqueeze(0) - z) ** 2, dim=1), min=0.0
    )
    current_potential = float(torch.sum(closest).item())
    selections: list[Mapping[str, Any]] = []
    for center_position in range(1, n_clusters):
        if current_potential <= 0.0 or not math.isfinite(current_potential):
            raise DenseRerunRequired("kmeans_zero_or_nonfinite_potential")
        _guard_unselected_distances_gpu(
            z, centers[:center_position], selected, policy, events, center_position
        )
        random_values = random_state.random_sample(n_local_trials) * current_potential
        closest_cpu = closest.detach().cpu().numpy()
        cumulative = stable_cumsum(closest_cpu)
        raw_candidate_ids = np.searchsorted(cumulative, random_values)
        if np.any(raw_candidate_ids >= closest_cpu.size):
            raise DenseRerunRequired(
                "kmeans_candidate_index_out_of_range",
                {"center_position": center_position},
            )
        for random_value, candidate_id in zip(random_values, raw_candidate_ids):
            lower = float(cumulative[candidate_id - 1]) if candidate_id > 0 else 0.0
            upper = float(cumulative[candidate_id])
            _guard_gap(
                policy,
                "kmeans_mass",
                min(abs(float(random_value) - lower), abs(upper - float(random_value))),
                1.0e-12 * max(1.0, current_potential),
                events,
                context={"center_position": center_position},
                spacing_value=abs(float(random_value)),
            )
        candidate_ids = raw_candidate_ids.copy()
        np.clip(candidate_ids, None, closest_cpu.size - 1, out=candidate_ids)
        candidate_index = torch.as_tensor(candidate_ids, dtype=torch.long, device=z.device)
        distances = torch.sum(
            (z.index_select(0, candidate_index).unsqueeze(1) - z.unsqueeze(0)) ** 2,
            dim=2,
        )
        distances = torch.minimum(closest.unsqueeze(0), torch.clamp(distances, min=0.0))
        candidate_potentials = torch.sum(distances, dim=1).detach().cpu().numpy()
        if not np.all(np.isfinite(candidate_potentials)):
            raise DenseRerunRequired("kmeans_nonfinite_candidate_potential")
        if candidate_potentials.size > 1:
            ordered = np.sort(candidate_potentials)
            _guard_gap(
                policy,
                "kmeans_potential",
                float(ordered[1] - ordered[0]),
                1.0e-10,
                events,
                context={"center_position": center_position},
            )
        best_local = int(np.argmin(candidate_potentials))
        current_potential = float(candidate_potentials[best_local])
        closest = distances[best_local]
        best_candidate = int(candidate_ids[best_local])
        selected.append(best_candidate)
        centers[center_position] = z[best_candidate] / torch.linalg.vector_norm(
            z[best_candidate]
        )
        selections.append(
            {
                "center_position": int(center_position),
                "sampling_potential": float(cumulative[-1]),
                "random_values": random_values.tolist(),
                "candidate_indices": candidate_ids.tolist(),
                "candidate_potentials": candidate_potentials.tolist(),
                "selected_index": best_candidate,
            }
        )
    return centers, {"first_center_index": first, "selections": selections}


def _factor_expectation_gpu(
    factor: FactorCoordinates,
    z: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    kappas: torch.Tensor,
    posterior_type: str,
) -> tuple[torch.Tensor, list[float], float]:
    """Compute factor posterior scores on CUDA with authoritative normalizers."""
    # Keep high-dimensional special-function authority on CPU and tensor algebra on GPU.
    n_rows = int(z.shape[0])
    n_clusters = int(centers.shape[0])
    normalizer_started = perf_counter()
    normalizers = [
        log_vmf_normalizer_plus_kappa(factor.ambient_dim, float(kappas[k].item()))
        for k in range(n_clusters)
    ]
    normalizer_seconds = perf_counter() - normalizer_started
    logged_weights = torch.full_like(weights, -torch.inf)
    positive = weights > 0.0
    logged_weights[positive] = torch.log(weights[positive])
    alignments = z @ centers.T
    weighted_logs = torch.empty(
        (n_clusters, n_rows), dtype=torch.float64, device=z.device
    )
    for component in range(n_clusters):
        weighted_logs[component] = (
            logged_weights[component]
            + float(normalizers[component])
            + kappas[component] * (alignments[:, component] - 1.0)
        )
    posterior = torch.zeros_like(weighted_logs)
    gaps: list[float] = []
    for row_index in range(n_rows):
        scores = weighted_logs[:, row_index]
        scores_cpu = scores.detach().cpu().numpy()
        if not np.all(np.isfinite(scores_cpu) | np.isneginf(scores_cpu)):
            raise DenseRerunRequired("nonfinite_factor_posterior_score")
        finite_scores = np.sort(scores_cpu[np.isfinite(scores_cpu)])
        gaps.append(
            math.inf
            if finite_scores.size < 2
            else float(finite_scores[-1] - finite_scores[-2])
        )
        if posterior_type == "soft":
            posterior[:, row_index] = torch.exp(scores - torch.logsumexp(scores, dim=0))
        else:
            posterior[int(torch.argmax(scores).item()), row_index] = 1.0
    return posterior, gaps, float(normalizer_seconds)


def _factor_maximization_gpu(
    factor: FactorCoordinates,
    z: torch.Tensor,
    posterior: torch.Tensor,
    force_weights: np.ndarray | None,
    policy: GpuFactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Mapping[str, Any]]]:
    """Apply accepted M-step formulas with fixed-shape CUDA reductions."""
    # Preserve component and row loop order; only vector arithmetic and reduction move.
    n_clusters, n_rows = (int(posterior.shape[0]), int(posterior.shape[1]))
    centers = torch.zeros(
        (n_clusters, int(z.shape[1])), dtype=torch.float64, device=z.device
    )
    coefficients = torch.zeros(
        (n_clusters, n_rows), dtype=torch.float64, device=z.device
    )
    weights = (
        torch.zeros(n_clusters, dtype=torch.float64, device=z.device)
        if force_weights is None
        else torch.as_tensor(force_weights, dtype=torch.float64, device=z.device).clone()
    )
    kappas = torch.zeros(n_clusters, dtype=torch.float64, device=z.device)
    component_trace: list[Mapping[str, Any]] = []
    for component in range(n_clusters):
        if force_weights is None:
            weights[component] = torch.mean(posterior[component])
        scaled = z.clone()
        for row_index in range(n_rows):
            scaled[row_index] *= posterior[component, row_index]
        resultant = torch.sum(scaled, dim=0)
        resultant_norm = float(torch.linalg.vector_norm(resultant).item())
        weight = float(weights[component].item())
        if (
            not math.isfinite(weight)
            or weight <= _EPS
            or not math.isfinite(resultant_norm)
            or resultant_norm <= 1.0e-8
        ):
            raise DenseRerunRequired(
                "inherited_resultant_fallback",
                {
                    "iteration": int(iteration),
                    "component": int(component),
                    "weight": weight,
                    "resultant_norm": resultant_norm,
                },
            )
        _guard_comparison(
            policy,
            "resultant",
            resultant_norm,
            1.0e-8,
            1.0e-12,
            events,
            context={"iteration": iteration, "component": component},
        )
        centers[component] = resultant / resultant_norm
        coefficients[component] = posterior[component] / resultant_norm
        rbar = resultant_norm / (n_rows * weight)
        if not math.isfinite(rbar):
            raise DenseRerunRequired(
                "nonfinite_resultant_ratio",
                {
                    "iteration": int(iteration),
                    "component": int(component),
                    "weight": weight,
                    "resultant_norm": resultant_norm,
                },
            )
        if rbar >= 1.0 - 1.0e-10:
            concentration = MAX_CONCENTRATION
        else:
            rbar = max(0.0, float(rbar))
            concentration = (rbar * factor.ambient_dim - rbar**3.0) / (
                1.0 - rbar**2.0
            )
        kappas[component] = float(concentration)
        component_trace.append(
            {
                "component": int(component),
                "weight": weight,
                "resultant_norm": resultant_norm,
                "rbar": float(rbar),
                "concentration": float(concentration),
            }
        )
    return centers, coefficients, weights, kappas, component_trace


def _guard_unselected_distances_gpu(
    z: torch.Tensor,
    centers: torch.Tensor,
    selected: list[int],
    policy: GpuFactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
    center_position: int,
) -> None:
    """Guard CUDA distances that can alter k-means++ sampling mass."""
    # Exclude exact selected-center zeros before measuring the nearest live boundary.
    mask = torch.ones(int(z.shape[0]), dtype=torch.bool, device=z.device)
    mask[torch.as_tensor(selected, dtype=torch.long, device=z.device)] = False
    if not bool(torch.any(mask).item()):
        raise DenseRerunRequired("kmeans_no_unselected_rows")
    raw = (
        torch.sum(centers * centers, dim=1).unsqueeze(1)
        + torch.sum(z * z, dim=1).unsqueeze(0)
        - 2.0 * centers @ z.T
    )[:, mask]
    minimum_raw = float(torch.min(raw).item())
    if minimum_raw < 0.0:
        raise DenseRerunRequired(
            "kmeans_negative_distance_before_clipping",
            {"center_position": int(center_position), "minimum_raw": minimum_raw},
        )
    _guard_gap(
        policy,
        "kmeans_distance",
        minimum_raw,
        1.0e-12,
        events,
        context={"center_position": center_position},
    )


def _factor_mixture_log_likelihood_gpu(
    factor: FactorCoordinates,
    z: torch.Tensor,
    centers: torch.Tensor,
    weights: torch.Tensor,
    kappas: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate the accepted stable likelihood with CUDA factor alignments."""
    # Use the same plus-kappa normalizer form as dense and CPU factor authorities.
    started = perf_counter()
    normalizers = [
        log_vmf_normalizer_plus_kappa(factor.ambient_dim, float(kappa.item()))
        for kappa in kappas
    ]
    normalizer_seconds = perf_counter() - started
    logged_weights = torch.full_like(weights, -torch.inf)
    positive = weights > 0.0
    logged_weights[positive] = torch.log(weights[positive])
    alignments = z @ centers.T
    scores = torch.empty(
        (int(centers.shape[0]), int(z.shape[0])),
        dtype=torch.float64,
        device=z.device,
    )
    for component in range(int(centers.shape[0])):
        scores[component] = (
            logged_weights[component]
            + float(normalizers[component])
            + kappas[component] * (alignments[:, component] - 1.0)
        )
    likelihood = float(torch.sum(torch.logsumexp(scores, dim=0)).item())
    return likelihood, float(normalizer_seconds)


def _factor_inertia_gpu(
    z: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor
) -> float:
    """Accumulate inherited cosine inertia in stable row order on CUDA."""
    # Retain the explicit row loop so label indexing cannot change reduction shape.
    inertia = torch.zeros(int(z.shape[0]), dtype=torch.float64, device=z.device)
    for row_index in range(int(z.shape[0])):
        inertia[row_index] = 1.0 - torch.dot(z[row_index], centers[labels[row_index]])
    return float(torch.sum(inertia).item())


def _copy_fit_with_trace(fit: FactorFit, trace: Mapping[str, Any]) -> FactorFit:
    """Replace only aggregate trace data on an immutable fit record."""
    # Scientific arrays and selected initialization remain byte-identical.
    return FactorFit(
        centers=fit.centers,
        center_coefficients=fit.center_coefficients,
        labels=fit.labels,
        responsibilities=fit.responsibilities,
        weights=fit.weights,
        kappas=fit.kappas,
        inertia=fit.inertia,
        log_likelihood=fit.log_likelihood,
        selected_init_index=fit.selected_init_index,
        trace=dict(trace),
    )
