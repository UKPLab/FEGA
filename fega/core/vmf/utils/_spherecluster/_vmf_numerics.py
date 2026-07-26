"""Standalone vMF normalization and full mixture likelihood utilities."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.special import gammaln, i0e, ive, logsumexp

__all__ = [
    "log_vmf_normalizer",
    "log_vmf_normalizer_plus_kappa",
    "vmf_mixture_log_likelihood",
]


_MAX_VALIDATED_DIMENSION = 256_000
_SATURATION_RBAR = 1.0 - 1.0e-10
_MAX_CONCENTRATION = 1.0e10
_QUADRATURE_RELATIVE_ERROR_LIMIT = 1.0e-11


def log_vmf_normalizer(dim: int, kappa: float) -> float:
    """Return the paper's order-aware ``log C_dim(kappa)``.

    The exact zero-concentration limit is used at ``kappa == 0``. Positive
    concentrations first use SciPy's exponentially scaled Bessel function.
    When that result underflows, adaptive quadrature evaluates an exact,
    mode-centered spherical integral without constructing a Bessel value.
    """
    # Evaluate the cancellation-safe shifted term before subtracting concentration.
    kappa = _validate_normalizer_inputs(dim, kappa)
    result = log_vmf_normalizer_plus_kappa(dim, kappa) - kappa
    if not math.isfinite(result):
        raise FloatingPointError("vMF log normalizer was not finite.")
    return float(result)


def log_vmf_normalizer_plus_kappa(dim: int, kappa: float) -> float:
    """Return ``log C_dim(kappa) + kappa`` over the validated dense domain.

    The shifted form prevents cancellation when a vMF log density adds
    ``kappa * dot`` to a normalizer whose leading term is ``-kappa``. SciPy's
    scaled Bessel function remains the primary route. When it underflows,
    SciPy's QUADPACK wrapper evaluates the exact spherical ``0F1`` integral
    after a mode-centered hyperbolic transform. Both paths avoid subtracting
    concentration-sized terms in the final shifted result.
    """
    # Validate coverage before evaluating the zero limit or an upstream library.
    kappa = _validate_normalizer_inputs(dim, kappa)
    half_dim = dim / 2.0
    zero_limit = float(gammaln(half_dim) - math.log(2.0) - half_dim * math.log(math.pi))
    if not math.isfinite(zero_limit):
        raise FloatingPointError("vMF zero-concentration limit was not finite.")
    if kappa == 0.0:
        return zero_limit

    # Prefer SciPy's exponentially scaled Bessel value when it is finite.
    nu = half_dim - 1.0
    scaled = float(i0e(kappa) if nu == 0.0 else ive(nu, kappa))
    if math.isfinite(scaled) and scaled > 0.0:
        result = (
            nu * math.log(kappa) - half_dim * math.log(2.0 * math.pi) - math.log(scaled)
        )
    else:
        result = zero_limit + _quadrature_kappa_minus_log_h(dim, kappa)
    if not math.isfinite(result):
        raise FloatingPointError("vMF shifted log normalizer was not finite.")
    return float(result)


def _validate_normalizer_inputs(dim: int, kappa: float) -> float:
    """Validate the paper normalizer's declared production-reachable domain.

    Dimensions cover the supported logit spheres through Gemma's 256,000-token
    vocabulary. Concentrations cover both the inherited ``1e10`` saturation
    value and the largest float64 value reachable immediately below that branch.
    """
    # Reject values outside the independently validated dense-oracle coverage.
    if isinstance(dim, bool) or not isinstance(dim, int | np.integer) or dim < 2:
        raise ValueError("vMF dimension must be an integer of at least two.")
    if dim > _MAX_VALIDATED_DIMENSION:
        raise ValueError(
            f"vMF dimension {dim} exceeds validated coverage "
            f"{_MAX_VALIDATED_DIMENSION}."
        )
    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("vMF concentration must be finite and non-negative.")
    if kappa > _max_validated_kappa(int(dim)):
        raise ValueError(
            f"vMF concentration {kappa} exceeds validated coverage for dim={dim}."
        )
    return kappa


def _max_validated_kappa(dim: int) -> float:
    """Return the largest concentration reachable by the inherited update rule.

    The value immediately below the frozen saturation comparison is evaluated
    with the frozen Banerjee approximation; the explicit saturation value is
    retained when it is larger for low ambient dimensions.
    """
    # Mirror the inherited branch boundary without introducing a new fit threshold.
    rbar = math.nextafter(_SATURATION_RBAR, 0.0)
    unsaturated = (rbar * float(dim) - rbar**3.0) / (1.0 - rbar**2.0)
    return max(_MAX_CONCENTRATION, float(unsaturated))


def _quadrature_kappa_minus_log_h(dim: int, kappa: float) -> float:
    """Return ``kappa - log(0F1(; dim/2; kappa**2/4))`` exactly.

    The spherical integral is transformed with ``t=tanh(y)``, centered at its
    analytic mode, and scaled by ``sqrt(dim-1)``. This keeps the integrand broad
    and order-one throughout the validated domain while SciPy's adaptive
    QUADPACK wrapper supplies an explicit integration error estimate.
    """
    # Construct the mode and normalization of the exact transformed integral.
    coefficient = float(dim - 1)
    mode_tanh = 2.0 * kappa / (math.hypot(coefficient, 2.0 * kappa) + coefficient)
    root_coefficient = math.sqrt(coefficient)

    def scaled_integrand(scaled_offset: float) -> float:
        """Evaluate the exact mode-centered integrand in scaled coordinates."""
        # Apply hyperbolic addition identities before concentration multiplication.
        offset = scaled_offset / root_coefficient
        offset_tanh = math.tanh(offset)
        denominator = 1.0 + mode_tanh * offset_tanh
        magnitude = abs(offset)
        log_cosh = magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)
        exponent = coefficient * (
            mode_tanh * offset_tanh / denominator - log_cosh - math.log(denominator)
        )
        return math.exp(exponent) / root_coefficient

    integral, absolute_error = quad(
        scaled_integrand,
        -math.inf,
        math.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
        limit=300,
    )
    relative_error = absolute_error / integral if integral > 0.0 else math.inf
    if (
        not math.isfinite(integral)
        or integral <= 0.0
        or not math.isfinite(relative_error)
        or relative_error > _QUADRATURE_RELATIVE_ERROR_LIMIT
    ):
        raise FloatingPointError(
            "exact vMF normalizer quadrature did not meet its error contract."
        )

    # Assemble the shifted log-H value without concentration-scale subtraction.
    half_dim = dim / 2.0
    log_density_constant = float(
        gammaln(half_dim) - 0.5 * math.log(math.pi) - gammaln(half_dim - 0.5)
    )
    kappa_minus_mode = coefficient * mode_tanh / (1.0 + mode_tanh)
    kappa_minus_mode += 0.5 * coefficient * math.log(kappa / (coefficient * mode_tanh))
    return kappa_minus_mode - log_density_constant - math.log(integral)


def vmf_mixture_log_likelihood(
    x: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    kappas: np.ndarray,
) -> float:
    """Return the full finite vMF mixture log likelihood for all observations.

    Inputs define an ``N x D`` sample matrix, ``K x D`` centers, non-negative
    mixture weights, and non-negative concentrations. Zero-weight components
    remain impossible rather than receiving artificial mass.
    """
    # Materialize and validate the complete mixture before evaluating components.
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    kappas = np.asarray(kappas, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("vMF observations must be a non-empty 2D array.")
    if centers.ndim != 2 or centers.shape[1] != x.shape[1]:
        raise ValueError("vMF centers must have shape (K, D) matching observations.")
    n_components = centers.shape[0]
    if weights.shape != (n_components,) or kappas.shape != (n_components,):
        raise ValueError("vMF weights and concentrations must each have shape (K,).")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(centers)):
        raise ValueError("vMF observations and centers must be finite.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("vMF weights must be finite and non-negative.")
    weight_total = float(math.fsum(float(weight) for weight in weights))
    if not math.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("vMF weights must have finite positive total mass.")
    if not np.all(np.isfinite(kappas)) or np.any(kappas < 0.0):
        raise ValueError("vMF concentrations must be finite and non-negative.")

    # Assemble every component density with cancellation-safe shifted normalizers.
    normalized_weights = weights / weight_total
    log_weights = np.full(n_components, -np.inf, dtype=np.float64)
    positive = normalized_weights > 0.0
    log_weights[positive] = np.log(normalized_weights[positive])
    dots = x @ centers.T
    component_logs = np.empty((x.shape[0], n_components), dtype=np.float64)
    for index, kappa in enumerate(kappas):
        component_logs[:, index] = (
            log_weights[index]
            + log_vmf_normalizer_plus_kappa(x.shape[1], float(kappa))
            + float(kappa) * (dots[:, index] - 1.0)
        )
    likelihood = float(
        math.fsum(float(value) for value in logsumexp(component_logs, axis=1))
    )
    if not math.isfinite(likelihood):
        raise FloatingPointError("vMF mixture log likelihood was not finite.")
    return likelihood
