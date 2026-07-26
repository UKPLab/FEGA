from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from fega.core.geometry_metrics.metrics import normalize_logit_deltas
from fega.core.vmf.utils._spherecluster._vmfm_factor import (
    DenseRerunRequired,
    FactorAmbiguityPolicy,
    FactorCoordinates,
    FactorIneligible,
    RoutedFactorFit,
    build_cpu_factor,
    fit_factor_movmf,
)

_NORMALIZATION_EPSILON = 1.0e-12
_STABILITY_SOURCE_GRAM_ATOL = 1.0e-10


@dataclass(frozen=True)
class FeatureFactorView:
    """Expose one ordered subset of an immutable feature-owned CPU factor.

    ``absolute_indices`` retain the root row identity, including repeated values.
    The view carries the exact normalized ambient provider required for truthful
    dense routing and selected-final center reconstruction, while ``factor`` is
    only ordered indexing into the single root QR result.
    """

    _owner: FeatureFactor
    absolute_indices: tuple[int, ...]
    unit_rows: torch.Tensor
    normalized_rows: np.ndarray
    factor: FactorCoordinates | None
    subspace_rows: torch.Tensor | None
    factor_unavailable_reason: str | None
    subspace_unavailable_reason: str | None

    @property
    def construction_trace(self) -> MappingProxyType:
        """Expose the sole root construction trace to candidate telemetry."""
        # Every view reports the same root build rather than inventing subset builds.
        return self._owner.construction_trace

    def select(self, relative_indices: Sequence[int]) -> FeatureFactorView:
        """Create a nested ordered view without rebuilding or deduplicating rows."""
        # Compose assignment-resample indices with the parent subset identity.
        selected = _validated_indices(relative_indices, len(self.absolute_indices))
        absolute = tuple(self.absolute_indices[index] for index in selected)
        return self._owner._view_from_absolute(absolute)


@dataclass(frozen=True)
class FeatureFactor:
    """Own one immutable normalized provider and at most one CPU factor build."""

    _unit_rows: torch.Tensor
    _normalized_rows: np.ndarray
    _factor: FactorCoordinates | None
    _row_scales: np.ndarray
    _subspace_rows: torch.Tensor | None
    _construction_trace: MappingProxyType
    _factor_unavailable_reason: str | None
    _subspace_unavailable_reason: str | None

    @classmethod
    def build(
        cls,
        rows: torch.Tensor,
        *,
        row_scales: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        stability_rows: torch.Tensor | None = None,
        stability_gram: torch.Tensor | None = None,
    ) -> FeatureFactor:
        """Build one vMF factor and source-gate its optional stability coordinates.

        Factor ineligibility is retained as a feature-level dense-routing reason;
        it is not retried by candidate, bootstrap, assignment, or subspace views.
        When the canonical stability source is supplied, factor-coordinate reuse
        is retained only if its raw Gram matches that source under the same
        ``1e-10`` source-identity tolerance used by hidden factor construction.
        Unexpected construction and source-validation failures remain loud.
        """
        # Freeze the score-level rows before constructing their exact float64 provider.
        unit_rows, counts = normalize_logit_deltas(rows, eps=_NORMALIZATION_EPSILON)
        if int(counts["n_valid"]) != int(counts["n_total"]):
            raise ValueError("feature factor input must contain only finite nonzero rows")
        provider = normalized_numpy_rows(unit_rows)
        provider.setflags(write=False)
        scales = _validated_row_scales(row_scales, int(unit_rows.shape[0]))
        stability_kernel = _validated_stability_kernel(
            stability_rows,
            stability_gram,
            row_count=int(unit_rows.shape[0]),
        )
        construction: dict[str, Any] = {}
        factor: FactorCoordinates | None = None
        unavailable_reason: str | None = None
        try:
            factor, trace = build_cpu_factor(provider)
            construction.update(trace)
            _freeze_factor(factor)
        except FactorIneligible as error:
            unavailable_reason = str(error)
            construction.update(
                {"selected_backend": "dense_cpu", "rejection": unavailable_reason}
            )
        subspace_rows = None
        subspace_unavailable_reason = unavailable_reason
        if factor is not None:
            # Restore raw-coordinate norms once at the immutable feature boundary.
            scaled = np.array(factor.z, dtype=np.float64, order="C", copy=True)
            scaled *= scales[:, None]
            subspace_rows = torch.from_numpy(scaled)
            subspace_unavailable_reason = None
            if stability_kernel is not None:
                # Apply the existing unit-Gram source check before raw subspace reuse.
                factor_kernel = factor.z @ factor.z.T
                stability_norms = np.sqrt(np.diag(stability_kernel))
                normalized_stability_kernel = stability_kernel / (
                    stability_norms[:, None] * stability_norms[None, :]
                )
                source_error = float(
                    np.max(np.abs(factor_kernel - normalized_stability_kernel))
                )
                construction["stability_source_gram_max_absolute_error"] = source_error
                construction["stability_raw_gram_max_absolute_error"] = float(
                    np.max(np.abs(scaled @ scaled.T - stability_kernel))
                )
                if source_error > _STABILITY_SOURCE_GRAM_ATOL:
                    subspace_unavailable_reason = (
                        "factor stability source Gram mismatch exceeded 1e-10"
                    )
                    subspace_rows = None
                    construction.update(
                        {
                            "stability_selected_backend": "dense_cpu",
                            "stability_rejection": subspace_unavailable_reason,
                        }
                    )
                else:
                    construction["stability_selected_backend"] = "factor_cpu"
        elif stability_kernel is not None:
            # Make factor construction rejection visible to stability telemetry too.
            construction.update(
                {
                    "stability_selected_backend": "dense_cpu",
                    "stability_rejection": subspace_unavailable_reason,
                }
            )
        return cls(
            _unit_rows=unit_rows,
            _normalized_rows=provider,
            _factor=factor,
            _row_scales=scales,
            _subspace_rows=subspace_rows,
            _construction_trace=MappingProxyType(construction),
            _factor_unavailable_reason=unavailable_reason,
            _subspace_unavailable_reason=subspace_unavailable_reason,
        )

    @property
    def full_view(self) -> FeatureFactorView:
        """Return the full row-ordered provider through the same view contract."""
        # Use absolute root indices so point scoring and nested subsets share identity.
        return self._view_from_absolute(tuple(range(int(self._unit_rows.shape[0]))))

    @property
    def construction_trace(self) -> MappingProxyType:
        """Expose immutable root construction diagnostics for backend telemetry."""
        # Return the frozen mapping created at the sole construction boundary.
        return self._construction_trace

    def view(self, indices: Sequence[int]) -> FeatureFactorView:
        """Return an ordered root subset while preserving repeated indices exactly."""
        # Validate positions without sorting or applying uniqueness operations.
        absolute = _validated_indices(indices, int(self._unit_rows.shape[0]))
        return self._view_from_absolute(absolute)

    def _view_from_absolute(
        self, absolute_indices: tuple[int, ...]
    ) -> FeatureFactorView:
        """Materialize one bounded provider and factor-coordinate subset."""
        # Select every representation with the same ordered absolute index tensor.
        is_full = absolute_indices == tuple(range(int(self._unit_rows.shape[0])))
        index = torch.as_tensor(absolute_indices, dtype=torch.long)
        unit_rows = (
            self._unit_rows
            if is_full
            else self._unit_rows.index_select(0, index)
        )
        numpy_index = np.asarray(absolute_indices, dtype=np.int64)
        provider = (
            self._normalized_rows
            if is_full
            else np.ascontiguousarray(
                self._normalized_rows[numpy_index], dtype=np.float64
            )
        )
        provider.setflags(write=False)
        factor = None
        if self._factor is not None:
            z = (
                self._factor.z
                if is_full
                else np.ascontiguousarray(
                    self._factor.z[numpy_index], dtype=np.float64
                )
            )
            z.setflags(write=False)
            factor = FactorCoordinates(
                z=z,
                ambient_dim=self._factor.ambient_dim,
                source_sha256=array_sha256(provider),
                backend=self._factor.backend,
                diagnostics=self._factor.diagnostics,
                cpu_fingerprint=self._factor.cpu_fingerprint,
                source_fingerprint=self._factor.source_fingerprint,
            )
        subspace_rows = None
        if self._subspace_rows is not None:
            # Index the source-gated root coordinates without reconstructing them.
            subspace_rows = (
                self._subspace_rows
                if is_full
                else self._subspace_rows.index_select(0, index)
            )
        return FeatureFactorView(
            _owner=self,
            absolute_indices=absolute_indices,
            unit_rows=unit_rows,
            normalized_rows=provider,
            factor=factor,
            subspace_rows=subspace_rows,
            factor_unavailable_reason=self._factor_unavailable_reason,
            subspace_unavailable_reason=self._subspace_unavailable_reason,
        )


def fit_prepared_factor_or_dense(
    prepared: FeatureFactorView,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: FactorAmbiguityPolicy,
) -> RoutedFactorFit:
    """Fit a prebuilt ordered factor view or restart the whole fit dense.

    This is the feature-owned counterpart of the accepted standalone router. It
    preserves the same caught surfaces, seed/budget, dense source, and provenance
    while deliberately omitting factor construction from the candidate boundary.
    """
    # Retain the immutable view provider; dense authority copies only if routing occurs.
    source = prepared.normalized_rows
    trace: dict[str, Any] = {"construction": dict(prepared.construction_trace)}
    try:
        if prepared.factor is None:
            raise FactorIneligible(
                prepared.factor_unavailable_reason or "feature_factor_unavailable"
            )
        fit = fit_factor_movmf(
            prepared.factor,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=int(random_state),
            policy=policy,
            posterior_type="soft",
            init="k-means++",
        )
        trace["fit"] = dict(fit.trace)
        return RoutedFactorFit(
            backend=prepared.factor.backend,
            factor_fit=fit,
            dense_result=None,
            route_reason=None,
            trace=trace,
        )
    except (FactorIneligible, DenseRerunRequired, FloatingPointError) as error:
        # Restart the corrected dense authority from the exact subset provider.
        from fega.core.vmf.utils._spherecluster._vmfm import movMF

        dense_trace: dict[str, Any] = {}
        dense_result = movMF(
            source,
            n_clusters,
            posterior_type="soft",
            n_init=n_init,
            n_jobs=1,
            max_iter=max_iter,
            init="k-means++",
            random_state=int(random_state),
            copy_x=True,
            trace=dense_trace,
        )
        reason = _factor_route_reason(error)
        trace["route"] = {
            "reason": reason,
            "event": dict(error.event)
            if isinstance(error, DenseRerunRequired)
            else {},
            "original_seed": int(random_state),
            "original_n_init": int(n_init),
            "original_max_iter": int(max_iter),
            "source_sha256": array_sha256(source),
        }
        trace["dense_trace"] = dense_trace
        return RoutedFactorFit(
            backend="dense_cpu",
            factor_fit=None,
            dense_result=dense_result,
            route_reason=reason,
            trace=trace,
        )


def fit_prepared_gpu_or_dense(
    prepared: FeatureFactorView,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: Any,
    device: str,
) -> RoutedFactorFit:
    """Fit one prepared factor on pinned CUDA or restart corrected dense CPU."""
    # Preserve the same original provider and complete-fit fallback as the CPU router.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    source = prepared.normalized_rows
    trace: dict[str, Any] = {"construction": dict(prepared.construction_trace)}
    try:
        if prepared.factor is None:
            raise FactorIneligible(
                prepared.factor_unavailable_reason or "feature_factor_unavailable"
            )
        fit = gpu_backend.fit_factor_movmf_gpu(
            prepared.factor,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=int(random_state),
            policy=policy,
            device=device,
            posterior_type="soft",
            init="k-means++",
        )
        trace["fit"] = dict(fit.trace)
        return RoutedFactorFit(
            backend=gpu_backend.GPU_BACKEND_NAME,
            factor_fit=fit,
            dense_result=None,
            route_reason=None,
            trace=trace,
        )
    except (FactorIneligible, DenseRerunRequired, FloatingPointError) as error:
        # Route directly to dense CPU without attempting another optimized backend.
        from fega.core.vmf.utils._spherecluster._vmfm import movMF

        dense_trace: dict[str, Any] = {}
        dense_result = movMF(
            source,
            n_clusters,
            posterior_type="soft",
            n_init=n_init,
            n_jobs=1,
            max_iter=max_iter,
            init="k-means++",
            random_state=int(random_state),
            copy_x=True,
            trace=dense_trace,
        )
        reason = _factor_route_reason(error)
        trace["route"] = {
            "reason": reason,
            "event": dict(error.event)
            if isinstance(error, DenseRerunRequired)
            else {},
            "original_seed": int(random_state),
            "original_n_init": int(n_init),
            "original_max_iter": int(max_iter),
            "source_sha256": array_sha256(source),
            "requested_backend": gpu_backend.GPU_BACKEND_NAME,
        }
        trace["dense_trace"] = dense_trace
        return RoutedFactorFit(
            backend="dense_cpu",
            factor_fit=None,
            dense_result=dense_result,
            route_reason=reason,
            trace=trace,
        )


def normalized_numpy_rows(rows: torch.Tensor) -> np.ndarray:
    """Return the exact float64 normalized provider used by every vMF backend."""
    # Preserve the existing CPU conversion and explicit row renormalization contract.
    values = rows.detach().cpu().to(dtype=torch.float64).numpy()
    if values.ndim != 2:
        raise ValueError("vMF input must be a 2D array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError("vMF input contains non-finite row norms.")
    if np.any(norms <= 0.0):
        raise ValueError("vMF input contains zero-norm rows.")
    return np.asarray(values / norms, dtype=np.float64)


def array_sha256(array: np.ndarray) -> str:
    """Bind row order, shape, dtype, and exact normalized provider bytes."""
    # Match the standalone factor and selected-final source identity byte-for-byte.
    canonical = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _validated_indices(indices: Sequence[int], row_count: int) -> tuple[int, ...]:
    """Validate ordered row positions without altering multiplicity or order."""
    # Convert each index once and reject Python's otherwise valid negative indexing.
    ordered = tuple(int(index) for index in indices)
    if any(index < 0 or index >= row_count for index in ordered):
        raise IndexError("feature factor view index is outside the parent row range")
    return ordered


def _validated_row_scales(
    scales: torch.Tensor | np.ndarray | Sequence[float] | None, row_count: int
) -> np.ndarray:
    """Freeze positive raw-coordinate norms used only by stability subspaces."""
    # Default to unit scaling for already normalized standalone feature rows.
    if scales is None:
        values = np.ones(row_count, dtype=np.float64)
    elif isinstance(scales, torch.Tensor):
        values = scales.detach().cpu().to(dtype=torch.float64).numpy().copy()
    else:
        values = np.asarray(scales, dtype=np.float64).copy()
    if values.shape != (row_count,):
        raise ValueError("feature factor row scales must have shape (n_rows,)")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("feature factor row scales must be finite and positive")
    values.setflags(write=False)
    return values


def _validated_stability_kernel(
    rows: torch.Tensor | None,
    gram: torch.Tensor | None,
    *,
    row_count: int,
) -> np.ndarray | None:
    """Return the canonical raw stability Gram or reject an incomplete source."""
    # Require the canonical rows and readout Gram together so identity is auditable.
    if (rows is None) != (gram is None):
        raise ValueError("stability rows and Gram must be supplied together")
    if rows is None or gram is None:
        return None
    values = rows.detach().cpu().to(dtype=torch.float64).numpy()
    metric = gram.detach().cpu().to(dtype=torch.float64).numpy()
    if values.ndim != 2 or int(values.shape[0]) != row_count:
        raise ValueError("stability rows must have shape (n_rows, source_width)")
    source_width = int(values.shape[1])
    if metric.shape != (source_width, source_width):
        raise ValueError("stability Gram must match the stability row width")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(metric)):
        raise ValueError("stability rows and Gram must be finite")
    # Materialize only the bounded row Gram needed for exact source eligibility.
    kernel = np.asarray(values @ metric @ values.T, dtype=np.float64)
    if np.any(np.diag(kernel) <= 0.0):
        raise ValueError("stability source rows must have positive Gram norms")
    return kernel


def _freeze_factor(factor: FactorCoordinates) -> None:
    """Prevent accidental mutation of the root coordinate matrix after construction."""
    # NumPy arrays remain mutable inside frozen dataclasses unless explicitly sealed.
    factor.z.setflags(write=False)


def _factor_route_reason(
    error: FactorIneligible | DenseRerunRequired | FloatingPointError,
) -> str:
    """Preserve the accepted whole-fit dense routing reason vocabulary."""
    # Match the standalone routed boundary for every approved factor rejection.
    if isinstance(error, DenseRerunRequired):
        return error.reason
    if isinstance(error, FloatingPointError):
        return f"factor_floating_point_error: {error}"
    return str(error)
