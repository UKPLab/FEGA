"""Exact no-truncation factor-coordinate vMF fitting and dense routing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import __version__ as _SCIPY_VERSION
from scipy.special import logsumexp
from threadpoolctl import __version__ as _THREADPOOLCTL_VERSION
from threadpoolctl import threadpool_info

from ._vmf_numerics import log_vmf_normalizer_plus_kappa

MAX_CONCENTRATION = 1.0e10
_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny
_MIN_ROWS = 2
_MAX_ROWS = 64
_MAX_AMBIENT_DIMENSION = 256_000
_CPU_FINGERPRINT_KEYS = frozenset(
    {
        "python_version",
        "python_implementation",
        "numpy",
        "scipy",
        "threadpoolctl",
        "blas_name",
        "blas_version",
        "blas_configuration",
        "blas_thread_pools",
        "thread_environment",
    }
)
_FACTOR_SOURCE_FINGERPRINT_KEYS = frozenset(
    {"factor_source_sha256", "factor_em_source_sha256"}
)
_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REQUIRED_SURFACES = frozenset(
    {
        "bic",
        "convergence",
        "initialization_likelihood",
        "kmeans_distance",
        "kmeans_mass",
        "kmeans_potential",
        "posterior_score",
        "resultant",
    }
)


class FactorIneligible(ValueError):
    """Report that an unmodified input cannot use the requested factor path."""


class DenseRerunRequired(RuntimeError):
    """Request a whole-fit corrected-dense rerun from the original input."""

    def __init__(self, reason: str, event: Mapping[str, Any] | None = None) -> None:
        """Retain a stable routing reason and the triggering decision evidence."""
        # Preserve the first trigger so callers can bind it to the dense rerun.
        super().__init__(reason)
        self.reason = reason
        self.event = dict(event or {})


@dataclass(frozen=True)
class FactorCoordinates:
    """Store all row-span coordinates without retaining the ambient QR basis."""

    z: np.ndarray
    ambient_dim: int
    source_sha256: str
    backend: str
    diagnostics: Mapping[str, float]
    cpu_fingerprint: Mapping[str, str]
    source_fingerprint: Mapping[str, str]


@dataclass(frozen=True)
class FactorFit:
    """Store one fixed-M optimized fit entirely in factor and row coefficients."""

    centers: np.ndarray
    center_coefficients: np.ndarray
    labels: np.ndarray
    responsibilities: np.ndarray
    weights: np.ndarray
    kappas: np.ndarray
    inertia: float
    log_likelihood: float
    selected_init_index: int
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class RoutedFactorFit:
    """Return either a factor fit or the authoritative whole-fit dense result."""

    backend: str
    factor_fit: FactorFit | None
    dense_result: tuple[np.ndarray, ...] | None
    route_reason: str | None
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class FactorAmbiguityPolicy:
    """Hold preregistered observed error envelopes for routing-only guards."""

    observed_errors: Mapping[str, float]
    expected_cpu_fingerprint: Mapping[str, str] | None = None
    expected_source_fingerprint: Mapping[str, str] | None = None
    enforce: bool = True

    def __post_init__(self) -> None:
        """Reject absent or invalid evidence before a path can be promoted."""
        # Require complete evidence only when decisions are allowed to survive.
        missing = _REQUIRED_SURFACES.difference(self.observed_errors)
        if self.enforce and missing:
            raise ValueError(
                "factor ambiguity evidence is missing: " + ", ".join(sorted(missing))
            )
        for surface, value in self.observed_errors.items():
            numeric = float(value)
            if surface not in _REQUIRED_SURFACES:
                raise ValueError(f"unknown factor ambiguity surface: {surface}")
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"observed factor error for {surface} must be finite and non-negative"
                )
        if self.enforce:
            fingerprint = dict(self.expected_cpu_fingerprint or {})
            if set(fingerprint) != _CPU_FINGERPRINT_KEYS:
                raise ValueError(
                    "factor CPU fingerprint must contain exactly: "
                    + ", ".join(sorted(_CPU_FINGERPRINT_KEYS))
                )
            if any(not isinstance(value, str) or not value for value in fingerprint.values()):
                raise ValueError("factor CPU fingerprint values must be non-empty strings")
            source_fingerprint = dict(self.expected_source_fingerprint or {})
            if set(source_fingerprint) != _FACTOR_SOURCE_FINGERPRINT_KEYS:
                raise ValueError("factor promotion requires both accepted source hashes")
            if any(
                not isinstance(value, str) or len(value) != 64
                for value in source_fingerprint.values()
            ):
                raise ValueError("factor source fingerprints must be SHA-256 hex digests")

    @classmethod
    def shadow(cls) -> FactorAmbiguityPolicy:
        """Create a non-promotable policy that records but does not route bands."""
        # Zero placeholders are diagnostic only because enforcement is disabled.
        return cls(
            {},
            expected_cpu_fingerprint=None,
            expected_source_fingerprint=None,
            enforce=False,
        )

    def observed(self, surface: str) -> float:
        """Return one validated empirical envelope or zero in shadow mode."""
        # Promotion construction already proves that every required key exists.
        return float(self.observed_errors.get(surface, 0.0))

    def require_current_cpu_fingerprint(self) -> Mapping[str, str]:
        """Return the current fingerprint or reject an uncalibrated CPU backend."""
        # Promotion is valid only for the exact NumPy, SciPy, and BLAS build recorded.
        current = current_cpu_numerical_fingerprint()
        if self.enforce and current != dict(self.expected_cpu_fingerprint or {}):
            raise FactorIneligible("CPU numerical fingerprint mismatch")
        return current

    def require_current_source_fingerprint(self) -> Mapping[str, str]:
        """Return current factor source hashes or reject uncalibrated code bytes."""
        # Promotion belongs to the exact wrapper and EM implementation calibrated.
        current = factor_backend_source_fingerprint()
        if self.enforce and current != dict(self.expected_source_fingerprint or {}):
            raise FactorIneligible("factor source fingerprint mismatch")
        return current


def current_cpu_numerical_fingerprint() -> dict[str, str]:
    """Describe the exact Python, numerical, and thread runtime owning calibration."""
    # Read the locked Python and NumPy build identities used by factor arithmetic.
    config = getattr(np.__config__, "CONFIG", {})
    build_dependencies = config.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {})
    # Normalize loaded BLAS pools without embedding installation-specific paths.
    blas_pools = sorted(
        (
            {
                "architecture": str(pool.get("architecture", "unknown")),
                "internal_api": str(pool.get("internal_api", "unknown")),
                "num_threads": int(pool.get("num_threads", 0)),
                "prefix": str(pool.get("prefix", "unknown")),
                "threading_layer": str(pool.get("threading_layer", "unknown")),
                "user_api": str(pool.get("user_api", "unknown")),
                "version": str(pool.get("version", "unknown")),
            }
            for pool in threadpool_info()
            if pool.get("user_api") == "blas"
        ),
        key=lambda pool: (
            pool["prefix"],
            pool["version"],
            pool["threading_layer"],
            pool["architecture"],
        ),
    )
    # Bind explicit thread controls, including the distinction between unset and set.
    thread_environment = {
        key: os.environ.get(key, "<unset>") for key in _THREAD_ENVIRONMENT_KEYS
    }
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": str(np.__version__),
        "scipy": str(_SCIPY_VERSION),
        "threadpoolctl": str(_THREADPOOLCTL_VERSION),
        "blas_name": str(blas.get("name", "unknown")),
        "blas_version": str(blas.get("version", "unknown")),
        "blas_configuration": str(blas.get("openblas configuration", "unknown")),
        "blas_thread_pools": json.dumps(
            blas_pools, sort_keys=True, separators=(",", ":")
        ),
        "thread_environment": json.dumps(
            thread_environment, sort_keys=True, separators=(",", ":")
        ),
    }


def factor_backend_source_fingerprint() -> dict[str, str]:
    """Hash both CPU factor implementation files that own promoted decisions."""
    # Bind admission to exact public wrapper and EM source bytes.
    source = Path(__file__).resolve()
    return {
        "factor_source_sha256": _file_sha256(source),
        "factor_em_source_sha256": _file_sha256(
            source.with_name("_vmfm_factor_em.py")
        ),
    }


@cache
def _file_sha256(path: Path) -> str:
    """Return one implementation file's SHA-256 without retaining its bytes."""
    # Stream source files so admission adds bounded memory overhead.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factor_from_explicit_rows(rows: np.ndarray) -> FactorCoordinates:
    """Build ``Y.T = Q T`` and return ``Z = T.T`` with all N coordinates."""
    # Validate the exact row-ordered normalized source before economy QR.
    y = _validated_rows(rows)
    n_rows, ambient_dim = y.shape
    if ambient_dim < n_rows:
        raise FactorIneligible(
            "explicit factor requires ambient dimension at least the row count"
        )
    q, t = np.linalg.qr(y.T, mode="reduced")
    z = np.asarray(t.T, dtype=np.float64)
    reconstruction = _relative_residual(y.T, q @ t)
    orthogonality = float(
        np.linalg.norm(q.T @ q - np.eye(n_rows), ord="fro") / n_rows
    )
    gram_error = float(np.max(np.abs(z @ z.T - y @ y.T)))
    diagnostics = {
        "reconstruction_relative_error": reconstruction,
        "orthogonality_error_per_row": orthogonality,
        "gram_max_absolute_error": gram_error,
        "normalized_row_max_error": float(
            np.max(np.abs(np.linalg.norm(y, axis=1) - 1.0))
        ),
    }
    if reconstruction > 1.0e-10:
        raise FactorIneligible("explicit QR reconstruction residual exceeded 1e-10")
    if orthogonality > 1.0e-10:
        raise FactorIneligible("explicit QR orthogonality residual exceeded 1e-10")
    if gram_error > 1.0e-10:
        raise FactorIneligible("explicit factor Gram residual exceeded 1e-10")
    return FactorCoordinates(
        z=z,
        ambient_dim=ambient_dim,
        source_sha256=_array_sha256(y),
        backend="factor_cpu_explicit_y",
        diagnostics=diagnostics,
        cpu_fingerprint=current_cpu_numerical_fingerprint(),
        source_fingerprint=factor_backend_source_fingerprint(),
    )


def factor_from_hidden_gram(
    gram: np.ndarray, explicit_rows: np.ndarray
) -> FactorCoordinates:
    """Build an unmodified Cholesky factor only after every hidden-G check."""
    # Bind the hidden matrix to the retained explicit provider before Cholesky.
    y = _validated_rows(explicit_rows)
    raw_gram = np.asarray(gram)
    if raw_gram.dtype != np.dtype(np.float64):
        raise FactorIneligible("hidden Gram must use float64 without coercion")
    g = np.asarray(raw_gram, dtype=np.float64)
    n_rows, ambient_dim = y.shape
    if g.shape != (n_rows, n_rows) or not np.all(np.isfinite(g)):
        raise FactorIneligible("hidden Gram must be a finite square row Gram")
    scale = max(1.0, float(np.max(np.abs(g))))
    symmetry_error = float(np.max(np.abs(g - g.T)) / scale)
    if symmetry_error > 1.0e-12:
        raise FactorIneligible("hidden Gram symmetry error exceeded 1e-12")
    source_gram_error = float(np.max(np.abs(g - y @ y.T)))
    if source_gram_error > 1.0e-10:
        raise FactorIneligible("hidden Gram did not match the explicit source")
    try:
        z = np.linalg.cholesky(g)
    except np.linalg.LinAlgError as error:
        raise FactorIneligible("hidden Gram Cholesky failed without repair") from error
    residual = _relative_residual(g, z @ z.T)
    condition = float(np.linalg.cond(z, p=2))
    gram_error = float(np.max(np.abs(z @ z.T - g)))
    if residual > 1.0e-10:
        raise FactorIneligible("hidden Gram Cholesky residual exceeded 1e-10")
    if not math.isfinite(condition) or condition > 1.0e8:
        raise FactorIneligible("hidden Gram factor condition exceeded 1e8")
    if gram_error > 1.0e-10:
        raise FactorIneligible("hidden factor Gram residual exceeded 1e-10")
    return FactorCoordinates(
        z=np.asarray(z, dtype=np.float64),
        ambient_dim=ambient_dim,
        source_sha256=_array_sha256(y),
        backend="factor_cpu_hidden_g",
        diagnostics={
            "symmetry_relative_error": symmetry_error,
            "source_gram_max_absolute_error": source_gram_error,
            "cholesky_relative_residual": residual,
            "cholesky_condition_2": condition,
            "gram_max_absolute_error": gram_error,
        },
        cpu_fingerprint=current_cpu_numerical_fingerprint(),
        source_fingerprint=factor_backend_source_fingerprint(),
    )


def build_cpu_factor(
    rows: np.ndarray, gram: np.ndarray | None = None
) -> tuple[FactorCoordinates, Mapping[str, Any]]:
    """Apply hidden-G then explicit-Y ownership without repairing either input."""
    # A hidden rejection selects explicit rows; only explicit rejection selects dense.
    construction_trace: dict[str, Any] = {"hidden_requested": gram is not None}
    if gram is not None:
        try:
            factor = factor_from_hidden_gram(gram, rows)
            construction_trace["selected_backend"] = factor.backend
            construction_trace["diagnostics"] = dict(factor.diagnostics)
            return factor, construction_trace
        except FactorIneligible as error:
            construction_trace["hidden_rejection"] = str(error)
    factor = factor_from_explicit_rows(rows)
    construction_trace["selected_backend"] = factor.backend
    construction_trace["diagnostics"] = dict(factor.diagnostics)
    return factor, construction_trace


def fit_factor_movmf(
    factor: FactorCoordinates,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: FactorAmbiguityPolicy,
    posterior_type: str = "soft",
    force_weights: np.ndarray | None = None,
    init: str = "k-means++",
    tol: float = 1.0e-6,
) -> FactorFit:
    """Fit one fixed-M factor candidate with frozen dense ordering semantics."""
    # Reject a factor built or fitted outside the exact calibrated CPU environment.
    current_fingerprint = policy.require_current_cpu_fingerprint()
    current_source = policy.require_current_source_fingerprint()
    if dict(factor.cpu_fingerprint) != dict(current_fingerprint):
        raise FactorIneligible("factor construction fingerprint mismatch")
    if dict(factor.source_fingerprint) != dict(current_source):
        raise FactorIneligible("factor construction source fingerprint mismatch")

    # Keep EM mechanics in a sibling so each installed patch file remains bounded.
    from ._vmfm_factor_em import fit_factor_movmf_impl

    return fit_factor_movmf_impl(
        factor,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        policy=policy,
        posterior_type=posterior_type,
        force_weights=force_weights,
        init=init,
        tol=tol,
    )


def fit_factor_or_dense(
    rows: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: FactorAmbiguityPolicy,
    gram: np.ndarray | None = None,
    posterior_type: str = "soft",
    force_weights: np.ndarray | None = None,
    init: str = "k-means++",
    tol: float = 1.0e-6,
) -> RoutedFactorFit:
    """Use a factor only when eligible and rerun every guarded fit densely."""
    # Retain the original provider before factor eligibility checks for dense routing.
    y = np.array(np.asarray(rows), order="C", copy=True)
    factor_trace: dict[str, Any] = {}
    try:
        policy.require_current_cpu_fingerprint()
        policy.require_current_source_fingerprint()
        factor, construction = build_cpu_factor(y, gram=gram)
        factor_trace["construction"] = construction
        factor_fit = fit_factor_movmf(
            factor,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=int(random_state),
            policy=policy,
            posterior_type=posterior_type,
            force_weights=force_weights,
            init=init,
            tol=tol,
        )
        factor_trace["fit"] = dict(factor_fit.trace)
        return RoutedFactorFit(
            backend=factor.backend,
            factor_fit=factor_fit,
            dense_result=None,
            route_reason=None,
            trace=factor_trace,
        )
    except (FactorIneligible, DenseRerunRequired, FloatingPointError) as error:
        # Restart the full fixed-M budget from the original rows and seed.
        from ._vmfm import movMF

        dense_trace: dict[str, Any] = {}
        dense_result = movMF(
            y,
            n_clusters,
            posterior_type=posterior_type,
            force_weights=force_weights,
            n_init=n_init,
            n_jobs=1,
            max_iter=max_iter,
            init=init,
            random_state=int(random_state),
            tol=tol,
            copy_x=True,
            trace=dense_trace,
        )
        if isinstance(error, DenseRerunRequired):
            reason = error.reason
        elif isinstance(error, FloatingPointError):
            reason = f"factor_floating_point_error: {error}"
        else:
            reason = str(error)
        factor_trace["route"] = {
            "reason": reason,
            "event": dict(error.event) if isinstance(error, DenseRerunRequired) else {},
            "original_seed": int(random_state),
            "original_n_init": int(n_init),
            "original_max_iter": int(max_iter),
            "source_sha256": _array_sha256(y),
        }
        factor_trace["dense_trace"] = dense_trace
        return RoutedFactorFit(
            backend="dense_cpu",
            factor_fit=None,
            dense_result=dense_result,
            route_reason=reason,
            trace=factor_trace,
        )


def bic_decision_is_ambiguous(
    candidate_bic: float,
    incumbent_bic: float,
    tolerance: float,
    policy: FactorAmbiguityPolicy,
) -> bool:
    """Guard the frozen ``candidate < incumbent - tolerance`` comparison."""
    # Measure distance to the existing boundary without changing its outcome.
    boundary = float(incumbent_bic) - float(tolerance)
    margin = abs(float(candidate_bic) - boundary)
    band = _band(
        float(candidate_bic), boundary, 2.56e-6, policy.observed("bic")
    )
    return margin <= band


def factor_mixture_log_likelihood(
    factor: FactorCoordinates,
    centers: np.ndarray,
    weights: np.ndarray,
    kappas: np.ndarray,
) -> float:
    """Evaluate the submitted ambient-dimension likelihood from factor dots."""
    # Apply the dense authority's shifted-normalizer assembly to factor alignments.
    z = np.asarray(factor.z, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    kappas = np.asarray(kappas, dtype=np.float64)
    n_components = centers.shape[0]
    if centers.ndim != 2 or centers.shape[1] != z.shape[1]:
        raise ValueError("factor centers must have shape (K, factor_dim)")
    if weights.shape != (n_components,) or kappas.shape != (n_components,):
        raise ValueError("factor weights and kappas must each have shape (K,)")
    if (
        not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(kappas))
    ):
        raise DenseRerunRequired("nonfinite_factor_likelihood_input")
    if np.any(weights < 0.0) or np.any(kappas < 0.0):
        raise ValueError("factor weights and kappas must be non-negative")
    total = float(math.fsum(float(value) for value in weights))
    if not math.isfinite(total):
        raise DenseRerunRequired("nonfinite_factor_weight_total")
    if total <= 0.0:
        raise ValueError("factor weights must contain positive mass")
    normalized_weights = weights / total
    log_weights = np.full(n_components, -np.inf, dtype=np.float64)
    positive = normalized_weights > 0.0
    log_weights[positive] = np.log(normalized_weights[positive])
    dots = z @ centers.T
    component_logs = np.empty((z.shape[0], n_components), dtype=np.float64)
    for index, kappa in enumerate(kappas):
        component_logs[:, index] = (
            log_weights[index]
            + log_vmf_normalizer_plus_kappa(factor.ambient_dim, float(kappa))
            + float(kappa) * (dots[:, index] - 1.0)
        )
    likelihood = float(
        math.fsum(float(value) for value in logsumexp(component_logs, axis=1))
    )
    if not math.isfinite(likelihood):
        raise FloatingPointError("factor vMF mixture likelihood was not finite")
    return likelihood


def _band(value: float, boundary: float, analytical: float, observed: float) -> float:
    """Return the accepted common ambiguity band for factor decisions."""
    # Combine analytical, local-spacing, and empirical allowances conservatively.
    return max(
        float(analytical),
        256.0 * _EPS * max(1.0, abs(float(value)), abs(float(boundary))),
        8.0 * float(observed),
    )


def _validated_rows(rows: np.ndarray) -> np.ndarray:
    """Return a copied finite float64 matrix inside explicit-factor eligibility."""
    # Accept only the normalized row provider approved by the factor contract.
    raw = np.asarray(rows)
    if raw.dtype != np.dtype(np.float64):
        raise FactorIneligible("explicit rows must use float64 without coercion")
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim != 2:
        raise FactorIneligible("explicit rows must be an N x D matrix")
    if not _MIN_ROWS <= y.shape[0] <= _MAX_ROWS:
        raise FactorIneligible("explicit row count must be between 2 and 64")
    if not 2 <= y.shape[1] <= _MAX_AMBIENT_DIMENSION:
        raise FactorIneligible("explicit ambient dimension must be between 2 and 256000")
    if not np.all(np.isfinite(y)):
        raise FactorIneligible("explicit rows must be finite")
    row_norms = np.linalg.norm(y, axis=1)
    if np.any(row_norms <= 0.0):
        raise FactorIneligible("explicit rows must have positive norms")
    if float(np.max(np.abs(row_norms - 1.0))) > 1.0e-12:
        raise FactorIneligible("explicit normalized-row error exceeded 1e-12")
    return np.array(y, dtype=np.float64, order="C", copy=True)


def _relative_residual(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return a finite Frobenius residual with a non-zero denominator."""
    # Use the preregistered residual definition without repairing either matrix.
    numerator = float(np.linalg.norm(reference - reconstructed, ord="fro"))
    denominator = max(float(np.linalg.norm(reference, ord="fro")), _TINY)
    return numerator / denominator


def _array_sha256(array: np.ndarray) -> str:
    """Bind row order, shape, dtype, and exact float64 bytes deterministically."""
    # Hash canonical metadata before contiguous source bytes.
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
