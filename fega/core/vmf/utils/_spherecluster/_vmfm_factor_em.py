"""EM mechanics for the standalone exact vMF factor backend."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import squared_norm, stable_cumsum

from ._vmf_numerics import log_vmf_normalizer_plus_kappa
from ._vmfm_factor import (
    MAX_CONCENTRATION,
    DenseRerunRequired,
    FactorAmbiguityPolicy,
    FactorCoordinates,
    FactorFit,
    _band,
    factor_mixture_log_likelihood,
)

_EPS = np.finfo(np.float64).eps


def fit_factor_movmf_impl(
    factor: FactorCoordinates,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: FactorAmbiguityPolicy,
    posterior_type: str,
    force_weights: np.ndarray | None,
    init: str,
    tol: float,
) -> FactorFit:
    """Fit one fixed-M factor candidate with frozen dense ordering semantics."""
    # Validate controls before deriving the same ordered initialization seeds.
    if init != "k-means++":
        raise DenseRerunRequired("unsupported_factor_initialization")
    if posterior_type not in {"soft", "hard"}:
        raise ValueError("posterior_type must be 'soft' or 'hard'")
    if n_init <= 0 or max_iter <= 0:
        raise ValueError("n_init and max_iter must be positive")
    z = np.asarray(factor.z, dtype=np.float64)
    n_rows = z.shape[0]
    if n_clusters <= 0 or n_clusters > n_rows:
        raise ValueError("n_clusters must be between one and the row count")
    weights_override = _validated_force_weights(force_weights, n_clusters)
    scaled_tolerance = _factor_tolerance(z, factor.ambient_dim, tol)
    rng = check_random_state(int(random_state))
    seeds = rng.randint(np.iinfo(np.int32).max, size=n_init)
    attempts: list[FactorFit] = []
    attempt_traces: list[Mapping[str, Any]] = []
    for init_index, seed in enumerate(seeds):
        attempt = _fit_one_initialization(
            factor,
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
        attempt_traces.append(attempt.trace)
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
        "backend": factor.backend,
        "ambient_dim": int(factor.ambient_dim),
        "factor_dim": int(z.shape[1]),
        "source_sha256": factor.source_sha256,
        "mode_count": int(n_clusters),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "scaled_tolerance": float(scaled_tolerance),
        "initializations": attempt_traces,
        "selected_init_index": int(best.selected_init_index),
        "selected_log_likelihood": float(best.log_likelihood),
        "selection_guard_events": selection_events,
    }
    return FactorFit(
        centers=best.centers,
        center_coefficients=best.center_coefficients,
        labels=best.labels,
        responsibilities=best.responsibilities,
        weights=best.weights,
        kappas=best.kappas,
        inertia=best.inertia,
        log_likelihood=best.log_likelihood,
        selected_init_index=best.selected_init_index,
        trace=trace,
    )


def _fit_one_initialization(
    factor: FactorCoordinates,
    *,
    n_clusters: int,
    max_iter: int,
    seed: int,
    policy: FactorAmbiguityPolicy,
    posterior_type: str,
    force_weights: np.ndarray | None,
    tolerance: float,
    init_index: int,
) -> FactorFit:
    """Run one ordered factor EM initialization with dense-routing guards."""
    # Initialize only from observation rows so every center remains in the span.
    events: list[Mapping[str, Any]] = []
    centers, initialization = _factor_kmeans_plus_plus(
        factor.z, n_clusters, check_random_state(seed), policy, events
    )
    weights = (
        np.ones(n_clusters, dtype=np.float64) / n_clusters
        if force_weights is None
        else force_weights.copy()
    )
    kappas = np.ones(n_clusters, dtype=np.float64)
    iterations: list[Mapping[str, Any]] = []
    coefficients = np.zeros((n_clusters, factor.z.shape[0]), dtype=np.float64)
    for iteration in range(max_iter):
        previous = centers.copy()
        posterior, score_gaps = _factor_expectation(
            factor, centers, weights, kappas, posterior_type
        )
        for row_index, gap in enumerate(score_gaps):
            _guard_gap(
                policy,
                "posterior_score",
                gap,
                1.0e-10,
                events,
                context={"iteration": iteration, "row_index": row_index},
            )
        centers, coefficients, weights, kappas, component_trace = _factor_maximization(
            factor, posterior, force_weights, policy, events, iteration
        )
        shift = float(squared_norm(previous - centers))
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
                "posterior": np.asarray(posterior, dtype=np.float64).tolist(),
                "center_shift": shift,
                "tolerance": float(tolerance),
                "components": component_trace,
            }
        )
        if shift <= tolerance:
            break
    labels = np.asarray(np.argmax(posterior, axis=0), dtype=np.int64)
    inertia = _factor_inertia(factor.z, centers, labels)
    likelihood = factor_mixture_log_likelihood(factor, centers, weights, kappas)
    trace = {
        "init_index": int(init_index),
        "seed": int(seed),
        "initialization": initialization,
        "iterations": iterations,
        "iteration_count": len(iterations),
        "converged": bool(iterations and iterations[-1]["center_shift"] <= tolerance),
        "labels": labels.tolist(),
        "inertia": float(inertia),
        "log_likelihood": float(likelihood),
        "guard_events": events,
    }
    return FactorFit(
        centers=centers,
        center_coefficients=coefficients,
        labels=labels,
        responsibilities=posterior,
        weights=weights,
        kappas=kappas,
        inertia=float(inertia),
        log_likelihood=float(likelihood),
        selected_init_index=int(init_index),
        trace=trace,
    )


def _factor_kmeans_plus_plus(
    z: np.ndarray,
    n_clusters: int,
    random_state: np.random.RandomState,
    policy: FactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Mirror vendored k-means++ in exact factor coordinates and trace branches."""
    # Preserve first-center and local-trial random draw ordering exactly.
    n_rows, factor_dim = z.shape
    n_local_trials = 2 + int(np.log(n_clusters))
    centers = np.empty((n_clusters, factor_dim), dtype=np.float64)
    first = int(random_state.randint(n_rows))
    selected = [first]
    centers[0] = z[first]
    centers[0] /= np.linalg.norm(centers[0])
    closest = euclidean_distances(
        centers[0, np.newaxis], z, Y_norm_squared=np.ones(n_rows), squared=True
    )
    current_potential = float(closest.sum())
    selections: list[Mapping[str, Any]] = []
    for center_position in range(1, n_clusters):
        if current_potential <= 0.0 or not math.isfinite(current_potential):
            raise DenseRerunRequired("kmeans_zero_or_nonfinite_potential")
        _guard_unselected_distances(
            z, centers[:center_position], selected, policy, events, center_position
        )
        random_values = random_state.random_sample(n_local_trials) * current_potential
        cumulative = stable_cumsum(closest)
        raw_candidate_ids = np.searchsorted(cumulative, random_values)
        if np.any(raw_candidate_ids >= closest.size):
            raise DenseRerunRequired(
                "kmeans_candidate_index_out_of_range",
                {"center_position": center_position},
            )
        for random_value, candidate_id in zip(random_values, raw_candidate_ids):
            lower = float(cumulative[candidate_id - 1]) if candidate_id > 0 else 0.0
            upper = float(cumulative[candidate_id])
            boundary_gap = min(
                abs(float(random_value) - lower),
                abs(upper - float(random_value)),
            )
            analytical = 1.0e-12 * max(1.0, current_potential)
            _guard_gap(
                policy,
                "kmeans_mass",
                boundary_gap,
                analytical,
                events,
                context={"center_position": center_position},
                spacing_value=abs(float(random_value)),
            )
        candidate_ids = raw_candidate_ids.copy()
        np.clip(candidate_ids, None, closest.size - 1, out=candidate_ids)
        distances = euclidean_distances(
            z[candidate_ids], z, Y_norm_squared=np.ones(n_rows), squared=True
        )
        np.minimum(closest, distances, out=distances)
        candidate_potentials = distances.sum(axis=1)
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
        centers[center_position] = z[best_candidate]
        centers[center_position] /= np.linalg.norm(centers[center_position])
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


def _factor_expectation(
    factor: FactorCoordinates,
    centers: np.ndarray,
    weights: np.ndarray,
    kappas: np.ndarray,
    posterior_type: str,
) -> tuple[np.ndarray, list[float]]:
    """Compute responsibilities with ambient normalizers and factor alignments."""
    # Reproduce the dense component-then-row posterior evaluation order.
    n_rows = factor.z.shape[0]
    n_clusters = centers.shape[0]
    weighted_logs = np.empty((n_clusters, n_rows), dtype=np.float64)
    logged_weights = np.full(n_clusters, -np.inf, dtype=np.float64)
    positive = weights > 0.0
    logged_weights[positive] = np.log(weights[positive])
    alignments = factor.z @ centers.T
    for component in range(n_clusters):
        weighted_logs[component] = (
            logged_weights[component]
            + log_vmf_normalizer_plus_kappa(
                factor.ambient_dim, float(kappas[component])
            )
            + float(kappas[component]) * (alignments[:, component] - 1.0)
        )
    posterior = np.zeros((n_clusters, n_rows), dtype=np.float64)
    gaps: list[float] = []
    for row_index in range(n_rows):
        scores = weighted_logs[:, row_index]
        if not np.all(np.isfinite(scores) | np.isneginf(scores)):
            raise DenseRerunRequired("nonfinite_factor_posterior_score")
        finite_scores = np.sort(scores[np.isfinite(scores)])
        gaps.append(
            math.inf
            if finite_scores.size < 2
            else float(finite_scores[-1] - finite_scores[-2])
        )
        if posterior_type == "soft":
            posterior[:, row_index] = np.exp(scores - logsumexp(scores))
        else:
            posterior[int(np.argmax(scores)), row_index] = 1.0
    return posterior, gaps


def _factor_maximization(
    factor: FactorCoordinates,
    posterior: np.ndarray,
    force_weights: np.ndarray | None,
    policy: FactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    """Update factor centers and row coefficients with ambient concentrations."""
    # Follow the dense row-scaling and summation order before normalization.
    n_clusters, n_rows = posterior.shape
    centers = np.zeros((n_clusters, factor.z.shape[1]), dtype=np.float64)
    coefficients = np.zeros((n_clusters, n_rows), dtype=np.float64)
    weights = (
        np.zeros(n_clusters, dtype=np.float64)
        if force_weights is None
        else force_weights.copy()
    )
    kappas = np.zeros(n_clusters, dtype=np.float64)
    component_trace: list[Mapping[str, Any]] = []
    for component in range(n_clusters):
        if force_weights is None:
            weights[component] = np.mean(posterior[component])
        scaled = factor.z.copy()
        for row_index in range(n_rows):
            scaled[row_index] *= posterior[component, row_index]
        resultant = scaled.sum(axis=0)
        resultant_norm = float(np.linalg.norm(resultant))
        if (
            not math.isfinite(float(weights[component]))
            or weights[component] <= _EPS
            or not math.isfinite(resultant_norm)
            or resultant_norm <= 1.0e-8
        ):
            raise DenseRerunRequired(
                "inherited_resultant_fallback",
                {
                    "iteration": int(iteration),
                    "component": int(component),
                    "weight": float(weights[component]),
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
        rbar = resultant_norm / (n_rows * weights[component])
        if not math.isfinite(float(rbar)):
            raise DenseRerunRequired(
                "nonfinite_resultant_ratio",
                {
                    "iteration": int(iteration),
                    "component": int(component),
                    "weight": float(weights[component]),
                    "resultant_norm": resultant_norm,
                },
            )
        elif rbar >= 1.0 - 1.0e-10:
            kappas[component] = MAX_CONCENTRATION
        else:
            rbar = max(0.0, float(rbar))
            kappas[component] = (
                rbar * factor.ambient_dim - rbar**3.0
            ) / (1.0 - rbar**2.0)
        component_trace.append(
            {
                "component": int(component),
                "weight": float(weights[component]),
                "resultant_norm": resultant_norm,
                "rbar": float(rbar),
                "concentration": float(kappas[component]),
            }
        )
    return centers, coefficients, weights, kappas, component_trace


def _guard_unselected_distances(
    z: np.ndarray,
    centers: np.ndarray,
    selected: list[int],
    policy: FactorAmbiguityPolicy,
    events: list[Mapping[str, Any]],
    center_position: int,
) -> None:
    """Guard only off-selected distances that can change k-means++ mass."""
    # Exclude the mathematically exact selected-center zeros from ambiguity.
    mask = np.ones(z.shape[0], dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = False
    if not np.any(mask):
        raise DenseRerunRequired("kmeans_no_unselected_rows")
    raw = (
        np.sum(centers * centers, axis=1)[:, None]
        + np.sum(z * z, axis=1)[None, :]
        - 2.0 * centers @ z.T
    )[:, mask]
    if np.any(raw < 0.0):
        raise DenseRerunRequired(
            "kmeans_negative_distance_before_clipping",
            {"center_position": int(center_position), "minimum_raw": float(raw.min())},
        )
    minimum = float(np.min(raw))
    _guard_gap(
        policy,
        "kmeans_distance",
        minimum,
        1.0e-12,
        events,
        context={"center_position": center_position},
    )


def _guard_comparison(
    policy: FactorAmbiguityPolicy,
    surface: str,
    value: float,
    boundary: float,
    analytical: float,
    events: list[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
) -> None:
    """Record and optionally route a scalar decision near its frozen boundary."""
    # Use the preregistered common band without modifying the comparison itself.
    margin = abs(float(value) - float(boundary))
    band = _band(value, boundary, analytical, policy.observed(surface))
    event = {
        "surface": surface,
        "value": float(value),
        "boundary": float(boundary),
        "margin": margin,
        "band": band,
        **context,
    }
    events.append(event)
    if policy.enforce and margin <= band:
        raise DenseRerunRequired(f"ambiguous_{surface}", event)


def _guard_gap(
    policy: FactorAmbiguityPolicy,
    surface: str,
    gap: float,
    analytical: float,
    events: list[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    spacing_value: float | None = None,
) -> None:
    """Record and optionally route a non-negative tie or boundary gap."""
    # Treat infinity as safe only when fewer than two finite choices exist.
    if math.isinf(float(gap)) and float(gap) > 0.0:
        return
    if not math.isfinite(float(gap)) or gap < 0.0:
        raise DenseRerunRequired(f"invalid_{surface}_gap")
    spacing_source = float(gap if spacing_value is None else spacing_value)
    band = max(
        float(analytical),
        256.0 * abs(float(np.spacing(abs(spacing_source)))),
        8.0 * policy.observed(surface),
    )
    event = {"surface": surface, "gap": float(gap), "band": band, **context}
    events.append(event)
    if policy.enforce and gap <= band:
        raise DenseRerunRequired(f"ambiguous_{surface}", event)


def _factor_tolerance(z: np.ndarray, ambient_dim: int, tol: float) -> float:
    """Re-express dense mean coordinate variance without using factor rank."""
    # Isometry preserves total variance; the frozen dense rule divides by ambient D.
    variances = np.var(z, axis=0)
    return float(np.sum(variances) * float(tol) / float(ambient_dim))


def _factor_inertia(z: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Compute the inherited cosine inertia from factor dot products."""
    # Preserve the dense per-row accumulation order.
    inertia = np.zeros(z.shape[0], dtype=np.float64)
    for row_index in range(z.shape[0]):
        inertia[row_index] = 1.0 - z[row_index].dot(centers[labels[row_index]].T)
    return float(np.sum(inertia))


def _validated_force_weights(
    force_weights: np.ndarray | None, n_clusters: int
) -> np.ndarray | None:
    """Validate and normalize the inherited optional fixed mixture weights."""
    # Match dense validation before any factor EM work starts.
    if force_weights is None:
        return None
    weights = np.asarray(force_weights, dtype=np.float64)
    if weights.shape != (n_clusters,) or not np.all(np.isfinite(weights)):
        raise ValueError("force_weights must be a finite vector of length K")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("force_weights must contain non-negative positive mass")
    return weights / weights.sum()
