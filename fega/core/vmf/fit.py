from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from fega.core.vmf.backend_policy import (
    BackendPolicyManifestError,
    load_backend_policy_manifest,
)
from fega.core.vmf.factor_reuse import (
    FeatureFactor,
    FeatureFactorView,
    array_sha256,
    fit_prepared_factor_or_dense,
    fit_prepared_gpu_or_dense,
    normalized_numpy_rows,
)
from fega.core.vmf.utils._spherecluster._vmf_numerics import (
    vmf_mixture_log_likelihood,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor import (
    FactorAmbiguityPolicy,
    FactorFit,
    FactorIneligible,
    fit_factor_or_dense,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor_gpu import (
    GPU_BACKEND_NAME,
    GpuFactorAmbiguityPolicy,
)


@dataclass(frozen=True)
class VmfFit:
    """Expose the selected-final compatibility record used by FEGA consumers."""

    k: int
    labels: np.ndarray
    responsibilities: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    kappas: np.ndarray | None
    log_likelihood: float
    model: Any | None = None


@dataclass(frozen=True)
class VmfCandidate:
    """Carry one fixed-mode fit without an ambient center or estimator object.

    Factor candidates retain row coefficients for selected-only reconstruction.
    Dense candidates retain only deterministic rerun controls, so inherited
    ambient fallback behavior is recreated only if that candidate is selected.
    """

    k: int
    labels: np.ndarray
    responsibilities: np.ndarray
    weights: np.ndarray
    kappas: np.ndarray
    log_likelihood: float
    center_coefficients: np.ndarray | None
    ambient_dim: int
    source_sha256: str
    backend: str
    route_reason: str | None
    seed: int
    n_init: int
    max_iter: int
    trace: Mapping[str, Any]


def fit_vmf_mixture(
    rows: torch.Tensor,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    trace: dict[str, Any] | None = None,
    backend: str = "factor_cpu",
    gpu_device: str = "cuda:0",
) -> VmfFit:
    """Fit one fixed mode count and construct its selected-final public record.

    The shared backend first produces an internal center-free candidate. This
    direct compatibility entry point treats that sole candidate as selected and
    therefore performs the one permitted ambient-center finalization.
    """
    # Keep direct callers on the same routing contract as feature-level selection.
    candidate = fit_vmf_candidate(
        rows,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
        trace=trace,
        backend=backend,
        gpu_device=gpu_device,
    )
    return finalize_vmf_candidate(candidate, rows)


def fit_vmf_candidate(
    rows: torch.Tensor,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    trace: dict[str, Any] | None = None,
    prepared: FeatureFactorView | None = None,
    backend: str = "factor_cpu",
    gpu_device: str = "cuda:0",
) -> VmfCandidate:
    """Fit one fixed mode count through an explicit production backend.

    The returned record deliberately omits ambient centers and model objects.
    """
    # Reuse a feature-owned provider when supplied; standalone calls normalize locally.
    started = perf_counter()
    x = prepared.normalized_rows if prepared is not None else _normalized_numpy_rows(rows)
    normalized_at = perf_counter()
    effective_seed = int(seed) % (2**32)
    if backend not in {"dense_cpu", "factor_cpu", GPU_BACKEND_NAME}:
        raise ValueError(f"unknown vMF backend: {backend}")
    if backend == "dense_cpu":
        candidate = _fit_dense_candidate(
            x,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=effective_seed,
            route_reason="explicit_dense_cpu_backend",
            retain_trace=trace is not None,
        )
    elif backend == "factor_cpu":
        candidate = (
            _fit_prepared_candidate(
                prepared,
                k=k,
                n_init=n_init,
                max_iter=max_iter,
                seed=effective_seed,
                retain_trace=trace is not None,
            )
            if prepared is not None
            else _fit_routed_candidate(
                x,
                k=k,
                n_init=n_init,
                max_iter=max_iter,
                seed=effective_seed,
                retain_trace=trace is not None,
            )
        )
    else:
        gpu_prepared = prepared or FeatureFactor.build(rows).full_view
        candidate = _fit_prepared_gpu_candidate(
            gpu_prepared,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=effective_seed,
            retain_trace=trace is not None,
            device=gpu_device,
        )
    fitted_at = perf_counter()
    _write_candidate_trace(
        trace,
        candidate,
        normalization_seconds=normalized_at - started,
        fit_seconds=fitted_at - normalized_at,
        finalization_seconds=0.0,
    )
    return candidate


def rerun_vmf_candidate_dense(
    candidate: VmfCandidate,
    rows: torch.Tensor,
    *,
    reason: str,
    trace: dict[str, Any] | None = None,
) -> VmfCandidate:
    """Restart one factor candidate through corrected dense CPU from its source.

    The original normalized row identity, seed, initialization count, and
    iteration budget are verified before the complete fixed-mode rerun.
    """
    # Reject a mismatched provider rather than silently rerunning different science.
    x = _normalized_numpy_rows(rows)
    _require_candidate_source(candidate, x)
    dense = _fit_dense_candidate(
        x,
        k=candidate.k,
        n_init=candidate.n_init,
        max_iter=candidate.max_iter,
        seed=candidate.seed,
        route_reason=reason,
        retain_trace=trace is not None,
    )
    _write_candidate_trace(
        trace,
        dense,
        normalization_seconds=0.0,
        fit_seconds=0.0,
        finalization_seconds=0.0,
    )
    return dense


def finalize_vmf_candidate(candidate: VmfCandidate, rows: torch.Tensor) -> VmfFit:
    """Construct the sole selected-final ambient compatibility record.

    Factor centers are reconstructed from audited row coefficients and the same
    normalized provider. Dense candidates are deterministically rerun through
    the estimator so its optional compatibility object and any inherited ambient
    fallback are created only for the selected candidate.
    """
    # Bind finalization to the candidate's exact source before creating any center.
    x = _normalized_numpy_rows(rows)
    _require_candidate_source(candidate, x)
    if candidate.backend != "dense_cpu":
        coefficients = candidate.center_coefficients
        if coefficients is None:
            raise ValueError("factor candidate is missing center coefficients")
        centers = _renormalize_unit_rows(
            np.asarray(coefficients, dtype=np.float64) @ x
        )
        return VmfFit(
            k=candidate.k,
            labels=candidate.labels,
            responsibilities=candidate.responsibilities,
            centers=centers,
            weights=candidate.weights,
            kappas=candidate.kappas,
            log_likelihood=candidate.log_likelihood,
            model=None,
        )
    return _finalize_dense_candidate(candidate, x)


def production_factor_policy() -> FactorAmbiguityPolicy:
    """Return the tracked, self-validating CPU factor calibration policy."""
    # Load public policy data rather than embedding generated evidence in code.
    policy = load_backend_policy_manifest()["cpu_factor"]
    try:
        return FactorAmbiguityPolicy(
            dict(policy["observed_error_envelopes"]),
            expected_cpu_fingerprint=dict(policy["cpu_numerical_fingerprint"]),
            expected_source_fingerprint=dict(policy["source_fingerprint"]),
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise BackendPolicyManifestError(
            "vMF CPU backend policy construction is invalid"
        ) from error


def production_gpu_factor_policy() -> GpuFactorAmbiguityPolicy:
    """Return the tracked, self-validating GPU factor calibration policy."""
    # Load every accepted inherited CPU and native GPU identity from one manifest.
    policy = load_backend_policy_manifest()["gpu_factor"]
    try:
        return GpuFactorAmbiguityPolicy(
            dict(policy["observed_error_envelopes"]),
            expected_gpu_fingerprint=dict(policy["gpu_numerical_fingerprint"]),
            expected_cpu_fingerprint=dict(policy["factor_cpu_numerical_fingerprint"]),
            expected_source_fingerprint=dict(policy["source_fingerprint"]),
            expected_factor_source_fingerprint=dict(
                policy["factor_source_fingerprint"]
            ),
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise BackendPolicyManifestError(
            "vMF GPU backend policy construction is invalid"
        ) from error


def vmf_backend_fingerprints(
    *, backend: str = "dense_cpu", gpu_device: str = "cuda:0"
) -> dict[str, Any]:
    """Fingerprint every numerical/backend identity that can change a checkpoint."""
    # Bind runtime reuse to source bytes, policy, and backend-scoped live admission.
    spherecluster = Path(__file__).resolve().parent / "utils" / "_spherecluster"
    oracle = _fingerprinted(
        {
            "backend": "dense_cpu",
            "vmfm_sha256": _file_sha256(spherecluster / "_vmfm.py"),
            "numerics_sha256": _file_sha256(spherecluster / "_vmf_numerics.py"),
        }
    )
    factor = _fingerprinted(
        {
            "backends": ["factor_cpu_explicit_y", "factor_cpu_hidden_g"],
            "factor_sha256": _file_sha256(spherecluster / "_vmfm_factor.py"),
            "factor_em_sha256": _file_sha256(
                spherecluster / "_vmfm_factor_em.py"
            ),
        }
    )
    gpu_factor = _fingerprinted(
        {
            "backend": GPU_BACKEND_NAME,
            "gpu_source_sha256": _file_sha256(
                spherecluster / "_vmfm_factor_gpu.py"
            ),
            "gpu_em_source_sha256": _file_sha256(
                spherecluster / "_vmfm_factor_gpu_em.py"
            ),
        }
    )
    initialization = _fingerprinted(
        {
            "name": "k-means++",
            "seed_fold": "int(seed) % 2**32",
            "fixed_budget": True,
            "dense_source_sha256": oracle["vmfm_sha256"],
            "factor_source_sha256": factor["factor_em_sha256"],
        }
    )
    try:
        policy_manifest = load_backend_policy_manifest()
    except BackendPolicyManifestError as error:
        # Dense checkpoints remain usable while strict backends fail closed.
        policy_manifest = _policy_error_identity(error)
        calibration = _fingerprinted(dict(policy_manifest))
        gpu_calibration = _fingerprinted(dict(policy_manifest))
        validated_domain = _fingerprinted(dict(policy_manifest))
        gpu_domain = _fingerprinted(dict(policy_manifest))
    else:
        calibration = _fingerprinted(dict(policy_manifest["cpu_factor"]))
        gpu_calibration = _fingerprinted(dict(policy_manifest["gpu_factor"]))
        validated_domain = _fingerprinted(
            dict(policy_manifest["cpu_factor"]["validated_domain"])
        )
        gpu_domain = _fingerprinted(
            dict(policy_manifest["gpu_factor"]["validated_domain"])
        )
    backend_component = _fingerprinted(
        {
            "schema_version": 1,
            "dense_route": "whole_fixed_mode_original_seed_and_budget",
            "selected_final_center_only": True,
            "fit_adapter_sha256": _file_sha256(Path(__file__).resolve()),
            "factor_reuse_adapter_sha256": _file_sha256(
                Path(__file__).resolve().with_name("factor_reuse.py")
            ),
        }
    )
    return {
        "oracle": oracle,
        "factor": factor,
        "gpu_factor": gpu_factor,
        "initialization": initialization,
        "backend": backend_component,
        "calibration": calibration,
        "gpu_calibration": gpu_calibration,
        "validated_domain": validated_domain,
        "gpu_domain": gpu_domain,
        "policy_manifest": _fingerprinted(policy_manifest),
        "live_admission": vmf_backend_live_admission(
            backend=backend, gpu_device=gpu_device
        ),
    }


def vmf_backend_live_admission(
    *, backend: str, gpu_device: str = "cuda:0"
) -> dict[str, Any]:
    """Fingerprint the live numerical identity admitted for one backend route."""
    # Expose one reusable checkpoint component without rehashing static source files.
    return _fingerprinted(
        _live_backend_admission(backend=backend, gpu_device=gpu_device)
    )


def _live_backend_admission(*, backend: str, gpu_device: str) -> dict[str, Any]:
    """Return the exact live identity controlling this checkpoint's backend route."""
    # Dense authority is portable and must not probe optional optimized runtimes.
    if backend == "dense_cpu":
        return {"backend": backend, "accepted": True, "route": "dense_authority"}
    if backend not in {"factor_cpu", GPU_BACKEND_NAME}:
        raise ValueError(f"unknown vMF backend: {backend}")
    observed: dict[str, Any] = {"backend": backend, "gpu_device": gpu_device}
    try:
        from fega.core.vmf.utils._spherecluster import _vmfm_factor as cpu_backend

        if backend == "factor_cpu":
            policy = production_factor_policy()
            current_cpu = cpu_backend.current_cpu_numerical_fingerprint()
            observed["cpu_numerical_fingerprint"] = current_cpu
            if current_cpu != dict(policy.expected_cpu_fingerprint or {}):
                raise FactorIneligible("CPU numerical fingerprint mismatch")
            current_source = cpu_backend.factor_backend_source_fingerprint()
            observed["source_fingerprint"] = current_source
            if current_source != dict(policy.expected_source_fingerprint or {}):
                raise FactorIneligible("factor source fingerprint mismatch")
        else:
            from fega.core.vmf.utils._spherecluster import (
                _vmfm_factor_gpu as gpu_backend,
            )

            policy = production_gpu_factor_policy()
            current_cpu = cpu_backend.current_cpu_numerical_fingerprint()
            observed["factor_cpu_numerical_fingerprint"] = current_cpu
            if current_cpu != dict(policy.expected_cpu_fingerprint or {}):
                raise FactorIneligible("CPU numerical fingerprint mismatch")
            current_factor_source = cpu_backend.factor_backend_source_fingerprint()
            observed["factor_source_fingerprint"] = current_factor_source
            if current_factor_source != dict(
                policy.expected_factor_source_fingerprint or {}
            ):
                raise FactorIneligible("factor source fingerprint mismatch")
            current_gpu_source = gpu_backend.gpu_backend_source_fingerprint()
            observed["gpu_source_fingerprint"] = current_gpu_source
            if current_gpu_source != dict(policy.expected_source_fingerprint or {}):
                raise FactorIneligible("GPU source fingerprint mismatch")
            current_gpu = gpu_backend.gpu_backend_fingerprint(gpu_device)
            observed["gpu_numerical_fingerprint"] = current_gpu
            if current_gpu != dict(policy.expected_gpu_fingerprint or {}):
                raise FactorIneligible("GPU numerical fingerprint mismatch")
    except (BackendPolicyManifestError, FactorIneligible, ValueError) as error:
        observed.update(
            {
                "accepted": False,
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        )
        return observed
    observed["accepted"] = True
    return observed


def _policy_error_identity(error: BackendPolicyManifestError) -> dict[str, Any]:
    """Normalize an unavailable policy into a deterministic checkpoint identity."""
    # Include the underlying parser/filesystem class without exposing private evidence.
    cause = error.__cause__
    return {
        "available": False,
        "error_type": type(error).__name__,
        "cause_type": type(cause).__name__ if cause is not None else None,
        "reason": str(error),
    }


def _fit_routed_candidate(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    retain_trace: bool,
) -> VmfCandidate:
    """Convert the standalone routed result into a center-free candidate."""
    # Let the reviewed factor boundary own eligibility and complete dense routing.
    try:
        policy = production_factor_policy()
    except (BackendPolicyManifestError, ValueError) as error:
        return _fit_dense_candidate(
            x,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            route_reason=_policy_route_reason(error),
            retain_trace=retain_trace,
        )
    routed = fit_factor_or_dense(
        x,
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        policy=policy,
        posterior_type="soft",
        init="k-means++",
    )
    if routed.factor_fit is not None:
        return _candidate_from_factor_fit(
            routed.factor_fit,
            backend=routed.backend,
            ambient_dim=int(x.shape[1]),
            source_sha256=str(routed.factor_fit.trace["source_sha256"]),
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            fit_trace=routed.trace if retain_trace else {},
        )
    if routed.dense_result is None:
        raise RuntimeError("routed vMF fit returned neither factor nor dense state")
    return _candidate_from_dense_result(
        x,
        routed.dense_result,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
        route_reason=routed.route_reason,
        fit_trace=routed.trace if retain_trace else {},
    )


def _fit_prepared_candidate(
    prepared: FeatureFactorView,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    retain_trace: bool,
) -> VmfCandidate:
    """Fit one candidate from a feature-owned factor or route the whole fit dense."""
    # Delegate the accepted caught surfaces and dense restart to the reuse boundary.
    try:
        policy = production_factor_policy()
    except (BackendPolicyManifestError, ValueError) as error:
        return _fit_dense_candidate(
            prepared.normalized_rows,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            route_reason=_policy_route_reason(error),
            retain_trace=retain_trace,
        )
    routed = fit_prepared_factor_or_dense(
        prepared,
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        policy=policy,
    )
    if routed.factor_fit is not None:
        factor = prepared.factor
        if factor is None:
            raise RuntimeError("prepared factor fit is missing its source factor")
        return _candidate_from_factor_fit(
            routed.factor_fit,
            backend=routed.backend,
            ambient_dim=factor.ambient_dim,
            source_sha256=factor.source_sha256,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            fit_trace=routed.trace if retain_trace else {},
        )
    if routed.dense_result is None:
        raise RuntimeError("prepared vMF route returned neither factor nor dense state")
    return _candidate_from_dense_result(
        prepared.normalized_rows,
        routed.dense_result,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
        route_reason=routed.route_reason,
        fit_trace=routed.trace if retain_trace else {},
    )


def _fit_prepared_gpu_candidate(
    prepared: FeatureFactorView,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    retain_trace: bool,
    device: str,
) -> VmfCandidate:
    """Fit one prepared candidate on the explicit pinned CUDA backend."""
    # GPU rejection routes directly to corrected dense CPU from the same provider.
    try:
        policy = production_gpu_factor_policy()
    except (BackendPolicyManifestError, ValueError) as error:
        return _fit_dense_candidate(
            prepared.normalized_rows,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            route_reason=_policy_route_reason(error),
            retain_trace=retain_trace,
        )
    routed = fit_prepared_gpu_or_dense(
        prepared,
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        policy=policy,
        device=device,
    )
    if routed.factor_fit is not None:
        factor = prepared.factor
        if factor is None:
            raise RuntimeError("prepared GPU fit is missing its source factor")
        return _candidate_from_factor_fit(
            routed.factor_fit,
            backend=routed.backend,
            ambient_dim=factor.ambient_dim,
            source_sha256=factor.source_sha256,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
            fit_trace=routed.trace if retain_trace else {},
        )
    if routed.dense_result is None:
        raise RuntimeError("prepared GPU route returned neither factor nor dense state")
    return _candidate_from_dense_result(
        prepared.normalized_rows,
        routed.dense_result,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
        route_reason=routed.route_reason,
        fit_trace=routed.trace if retain_trace else {},
    )


def _policy_route_reason(error: Exception) -> str:
    """Describe a strict-policy failure that caused a whole-fit dense rerun."""
    # Preserve the public failure class while keeping dense routing deterministic.
    return f"optimized backend policy unavailable: {type(error).__name__}: {error}"


def _candidate_from_factor_fit(
    fit: FactorFit,
    *,
    backend: str,
    ambient_dim: int,
    source_sha256: str,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    fit_trace: Mapping[str, Any],
) -> VmfCandidate:
    """Convert a factor fit into the approved center-free candidate record."""
    # Retain row coefficients and discrete decisions without an ambient center.
    return VmfCandidate(
        k=k,
        labels=np.asarray(fit.labels, dtype=np.int64),
        responsibilities=np.asarray(fit.responsibilities, dtype=np.float64),
        weights=np.asarray(fit.weights, dtype=np.float64),
        kappas=np.asarray(fit.kappas, dtype=np.float64),
        log_likelihood=float(fit.log_likelihood),
        center_coefficients=np.asarray(fit.center_coefficients, dtype=np.float64),
        ambient_dim=int(ambient_dim),
        source_sha256=str(source_sha256),
        backend=backend,
        route_reason=None,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        trace=dict(fit_trace),
    )


def _fit_dense_candidate(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    route_reason: str,
    retain_trace: bool,
) -> VmfCandidate:
    """Run corrected dense CPU and immediately discard its ambient centers."""
    # Execute the inherited fixed-mode budget before retaining center-free state.
    from fega.core.vmf.utils._spherecluster._vmfm import movMF

    dense_trace: dict[str, Any] | None = {} if retain_trace else None
    result = movMF(
        x,
        k,
        posterior_type="soft",
        n_init=n_init,
        n_jobs=1,
        max_iter=max_iter,
        init="k-means++",
        random_state=seed,
        copy_x=True,
        trace=dense_trace,
    )
    return _candidate_from_dense_result(
        x,
        result,
        k=k,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
        route_reason=route_reason,
        fit_trace=dense_trace or {},
    )


def _candidate_from_dense_result(
    x: np.ndarray,
    result: tuple[np.ndarray, ...],
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    route_reason: str | None,
    fit_trace: Mapping[str, Any],
) -> VmfCandidate:
    """Strip ambient dense output down to the approved internal state."""
    # Recompute the public full likelihood once before dropping dense centers.
    centers, labels, _inertia, weights, kappas, responsibilities = result
    normalized_centers = _renormalize_unit_rows(
        np.asarray(centers, dtype=np.float64)
    )
    likelihood = vmf_mixture_log_likelihood(
        x,
        normalized_centers,
        np.asarray(weights, dtype=np.float64),
        np.asarray(kappas, dtype=np.float64),
    )
    return VmfCandidate(
        k=k,
        labels=np.asarray(labels, dtype=np.int64),
        responsibilities=np.asarray(responsibilities, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        kappas=np.asarray(kappas, dtype=np.float64),
        log_likelihood=float(likelihood),
        center_coefficients=None,
        ambient_dim=int(x.shape[1]),
        source_sha256=array_sha256(x),
        backend="dense_cpu",
        route_reason=route_reason,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        trace=dict(fit_trace),
    )


def _finalize_dense_candidate(candidate: VmfCandidate, x: np.ndarray) -> VmfFit:
    """Rerun the selected dense candidate and verify deterministic identity."""
    # Construct the optional estimator only at the selected-final compatibility edge.
    from fega.core.vmf.utils._spherecluster._vmfm import VonMisesFisherMixture

    model = VonMisesFisherMixture(
        n_clusters=candidate.k,
        posterior_type="soft",
        n_init=candidate.n_init,
        max_iter=candidate.max_iter,
        random_state=candidate.seed,
        init="k-means++",
        normalize=False,
    )
    model.fit(x)
    responsibilities = np.asarray(model.posterior_, dtype=np.float64)
    labels = np.asarray(np.argmax(responsibilities, axis=0), dtype=np.int64)
    centers = _renormalize_unit_rows(
        np.asarray(model.cluster_centers_, dtype=np.float64)
    )
    weights = np.asarray(model.weights_, dtype=np.float64)
    kappas = np.asarray(model.concentrations_, dtype=np.float64)
    likelihood = vmf_mixture_log_likelihood(x, centers, weights, kappas)
    if not (
        np.array_equal(labels, candidate.labels)
        and np.array_equal(responsibilities, candidate.responsibilities)
        and np.array_equal(weights, candidate.weights)
        and np.array_equal(kappas, candidate.kappas)
        and float(likelihood) == candidate.log_likelihood
    ):
        raise RuntimeError("selected dense vMF rerun was not deterministic")
    return VmfFit(
        k=candidate.k,
        labels=labels,
        responsibilities=responsibilities,
        centers=centers,
        weights=weights,
        kappas=kappas,
        log_likelihood=float(likelihood),
        model=model,
    )


def _write_candidate_trace(
    target: dict[str, Any] | None,
    candidate: VmfCandidate,
    *,
    normalization_seconds: float,
    fit_seconds: float,
    finalization_seconds: float,
) -> None:
    """Expose one backend-neutral trace while retaining routed evidence."""
    # Flatten the selected backend trace so the dense observational API stays stable.
    if target is None:
        return
    target.clear()
    source = dict(candidate.trace)
    if candidate.backend == "dense_cpu" and "dense_trace" in source:
        primary = dict(source["dense_trace"])
    elif candidate.backend != "dense_cpu" and "fit" in source:
        primary = dict(source["fit"])
    else:
        primary = source
    backend_timings = primary.get("timings_seconds")
    target.update(primary)
    target.update(
        {
            "backend": candidate.backend,
            "seed": candidate.seed,
            "n_rows": int(candidate.responsibilities.shape[1]),
            "ambient_dim": candidate.ambient_dim,
            "mode_count": candidate.k,
            "n_init": candidate.n_init,
            "max_iter": candidate.max_iter,
            "selected_log_likelihood": candidate.log_likelihood,
            "route_reason": candidate.route_reason,
            "source_sha256": candidate.source_sha256,
            "timings_seconds": {
                "normalization": float(normalization_seconds),
                "fit": float(fit_seconds),
                "finalization": float(finalization_seconds),
            },
        }
    )
    if isinstance(backend_timings, Mapping):
        target["backend_timings_seconds"] = dict(backend_timings)
    if "route" in source:
        target["routing"] = dict(source["route"])


def _normalized_numpy_rows(rows: torch.Tensor) -> np.ndarray:
    """Materialize the exact float64 normalized provider for all backend decisions."""
    # Preserve the existing CPU conversion and explicit row renormalization contract.
    return normalized_numpy_rows(rows)


def _require_candidate_source(candidate: VmfCandidate, x: np.ndarray) -> None:
    """Reject finalization or rerouting from a different row provider."""
    # Check ambient dimension and exact ordered normalized bytes together.
    if int(x.shape[1]) != candidate.ambient_dim:
        raise ValueError("vMF candidate ambient dimension does not match its source")
    if array_sha256(x) != candidate.source_sha256:
        raise ValueError("vMF candidate source fingerprint mismatch")


def _renormalize_unit_rows(x: np.ndarray) -> np.ndarray:
    """Return finite float64 rows projected back onto the unit sphere.

    A two-dimensional non-empty array with strictly positive row norms is
    required so downstream vMF density evaluation cannot receive invalid rays.
    """
    # Validate row geometry before applying one explicit normalization.
    if x.ndim != 2:
        raise ValueError("vMF input must be a 2D array.")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)):
        raise ValueError("vMF input contains non-finite row norms.")
    if np.any(norms <= 0.0):
        raise ValueError("vMF input contains zero-norm rows.")
    return np.asarray(x / norms, dtype=np.float64)


def _fingerprinted(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a strict canonical SHA-256 digest to one identity component."""
    # Hash only finite JSON primitives so checkpoint equality is portable.
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def _file_sha256(path: Path) -> str:
    """Hash one runtime source file without importing repository tooling."""
    # Read the exact package bytes that own the active numerical implementation.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
