"""Pinned deterministic float64 CUDA backend for exact factor-coordinate vMF."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from ._vmfm_factor import (
    _REQUIRED_SURFACES,
    DenseRerunRequired,
    FactorCoordinates,
    FactorFit,
    FactorIneligible,
    current_cpu_numerical_fingerprint,
    factor_backend_source_fingerprint,
)

GPU_BACKEND_NAME = "factor_gpu_explicit_y_float64"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_GPU_SOURCE_FINGERPRINT_KEYS = frozenset(
    {"gpu_source_sha256", "gpu_em_source_sha256"}
)


@dataclass(frozen=True)
class GpuFactorAmbiguityPolicy:
    """Hold routing envelopes and every accepted CPU/GPU backend identity."""

    observed_errors: Mapping[str, float]
    expected_gpu_fingerprint: Mapping[str, Any] | None = None
    expected_cpu_fingerprint: Mapping[str, str] | None = None
    expected_source_fingerprint: Mapping[str, str] | None = None
    expected_factor_source_fingerprint: Mapping[str, str] | None = None
    enforce: bool = True

    def __post_init__(self) -> None:
        """Reject incomplete promotion evidence before CUDA can own decisions."""
        # Shadow validation may omit envelopes because it cannot promote outcomes.
        missing = _REQUIRED_SURFACES.difference(self.observed_errors)
        if self.enforce and missing:
            raise ValueError(
                "GPU ambiguity evidence is missing: " + ", ".join(sorted(missing))
            )
        for surface, value in self.observed_errors.items():
            numeric = float(value)
            if surface not in _REQUIRED_SURFACES:
                raise ValueError(f"unknown GPU ambiguity surface: {surface}")
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"observed GPU error for {surface} must be finite and non-negative"
                )
        if self.enforce:
            if not self.expected_gpu_fingerprint:
                raise ValueError("GPU promotion requires an accepted device fingerprint")
            cpu_fingerprint = dict(self.expected_cpu_fingerprint or {})
            if set(cpu_fingerprint) != set(current_cpu_numerical_fingerprint()):
                raise ValueError("GPU promotion requires the accepted CPU fingerprint")
            if any(
                not isinstance(value, str) or not value
                for value in cpu_fingerprint.values()
            ):
                raise ValueError("GPU CPU fingerprint values must be non-empty strings")
            source_fingerprint = dict(self.expected_source_fingerprint or {})
            if set(source_fingerprint) != _GPU_SOURCE_FINGERPRINT_KEYS:
                raise ValueError("GPU promotion requires both accepted source hashes")
            if any(
                not isinstance(value, str) or len(value) != 64
                for value in source_fingerprint.values()
            ):
                raise ValueError("GPU source fingerprints must be SHA-256 hex digests")
            factor_source_fingerprint = dict(
                self.expected_factor_source_fingerprint or {}
            )
            if set(factor_source_fingerprint) != set(
                factor_backend_source_fingerprint()
            ):
                raise ValueError("GPU promotion requires accepted CPU factor sources")
            if any(
                not isinstance(value, str) or len(value) != 64
                for value in factor_source_fingerprint.values()
            ):
                raise ValueError(
                    "GPU factor source fingerprints must be SHA-256 hex digests"
                )

    @classmethod
    def shadow(cls) -> GpuFactorAmbiguityPolicy:
        """Create a diagnostic policy that records guards without promoting them."""
        # Empty errors do not affect decisions because enforcement is disabled.
        return cls(
            {},
            expected_gpu_fingerprint=None,
            expected_cpu_fingerprint=None,
            expected_source_fingerprint=None,
            expected_factor_source_fingerprint=None,
            enforce=False,
        )

    def observed(self, surface: str) -> float:
        """Return one calibrated error envelope or zero in shadow execution."""
        # Promotion construction guarantees complete keys when enforcement is active.
        return float(self.observed_errors.get(surface, 0.0))

    def require_current_gpu_fingerprint(self, device: str | torch.device) -> dict[str, Any]:
        """Return the pinned manifest or reject an unavailable or drifted backend."""
        # Configure and inspect exactly the execution surface used by this fit.
        current = gpu_backend_fingerprint(device)
        if self.enforce and current != dict(self.expected_gpu_fingerprint or {}):
            raise FactorIneligible("GPU numerical fingerprint mismatch")
        return current

    def require_current_cpu_fingerprint(self) -> dict[str, str]:
        """Return the CPU identity accepted to construct exact QR coordinates."""
        # GPU promotion inherits the calibrated NumPy/SciPy/BLAS factor boundary.
        current = current_cpu_numerical_fingerprint()
        if self.enforce and current != dict(self.expected_cpu_fingerprint or {}):
            raise FactorIneligible("CPU numerical fingerprint mismatch")
        return current

    def require_current_source_fingerprint(self) -> dict[str, str]:
        """Return current GPU wrapper hashes or reject uncalibrated source bytes."""
        # A code change cannot retain promotion merely because CUDA settings match.
        current = gpu_backend_source_fingerprint()
        if self.enforce and current != dict(self.expected_source_fingerprint or {}):
            raise FactorIneligible("GPU source fingerprint mismatch")
        return current

    def require_current_factor_source_fingerprint(self) -> dict[str, str]:
        """Return inherited CPU factor hashes or reject uncalibrated source bytes."""
        # GPU coordinates are valid only under the exact CPU factor implementation.
        current = factor_backend_source_fingerprint()
        if self.enforce and current != dict(
            self.expected_factor_source_fingerprint or {}
        ):
            raise FactorIneligible("factor source fingerprint mismatch")
        return current


def gpu_backend_source_fingerprint() -> dict[str, str]:
    """Hash both standalone GPU implementation files that own promoted decisions."""
    # Bind admission to exact wrapper and EM bytes without importing FEGA modules.
    source = Path(__file__).resolve()
    return {
        "gpu_source_sha256": _file_sha256(source),
        "gpu_em_source_sha256": _file_sha256(source.with_name("_vmfm_factor_gpu_em.py")),
    }


@cache
def _file_sha256(path: Path) -> str:
    """Return one implementation file's SHA-256 without retaining its bytes."""
    # Stream bounded source files so fingerprinting has no scientific side effects.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gpu_execution_workers(backend: str, workers: int) -> None:
    """Require one feature worker for the single-stream promoted CUDA backend."""
    # CPU execution keeps its existing scheduling contract.
    if backend == GPU_BACKEND_NAME and int(workers) != 1:
        raise ValueError("GPU factor backend requires exactly one feature worker")


def configure_deterministic_gpu(device: str | torch.device) -> torch.device:
    """Pin deterministic PyTorch/CUDA controls before factor arithmetic starts."""
    # Fail before allocating fit state when CUDA or the requested device is absent.
    if not torch.cuda.is_available():
        raise FactorIneligible("CUDA is unavailable")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise FactorIneligible("GPU factor backend requires a CUDA device")
    index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise FactorIneligible("requested CUDA device is unavailable")
    resolved = torch.device("cuda", index)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return resolved


def gpu_backend_manifest(device: str | torch.device) -> dict[str, Any]:
    """Describe the complete deterministic float64 CUDA execution surface."""
    # Read settings only after the common configurator pins every mutable control.
    resolved = configure_deterministic_gpu(device)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "backend": GPU_BACKEND_NAME,
        "dtype": "float64",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "nvidia_driver_version": _nvidia_driver_version(int(resolved.index)),
        "torch_version": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "device_index": int(resolved.index),
        "device_name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "stream_count": 1,
        "synchronization_policy": "before_and_after_timed_regions",
        "reduction_order": "fixed_tensor_shape_single_stream",
        "feature_sharding": "ordered_feature_ids_single_worker",
    }


@cache
def _nvidia_driver_version(device_index: int) -> str:
    """Return the exact NVIDIA driver serving one admitted CUDA device."""
    # Query the public driver tool because this PyTorch build exposes no driver API.
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(device_index),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FactorIneligible("NVIDIA driver fingerprint is unavailable") from error
    # Fail closed on empty or multi-line output rather than guessing device identity.
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(versions) != 1:
        raise FactorIneligible("NVIDIA driver fingerprint is ambiguous")
    return versions[0]


def gpu_backend_fingerprint(device: str | torch.device) -> dict[str, Any]:
    """Return the exact manifest fields that delimit GPU promotion eligibility."""
    # The full deterministic manifest is decision-relevant for this narrow backend.
    return gpu_backend_manifest(device)


def fit_factor_movmf_gpu(
    factor: FactorCoordinates,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int,
    policy: GpuFactorAmbiguityPolicy,
    device: str | torch.device,
    posterior_type: str = "soft",
    force_weights: np.ndarray | None = None,
    init: str = "k-means++",
    tol: float = 1.0e-4,
) -> FactorFit:
    """Fit one fixed-mode factor candidate on a pinned float64 CUDA backend."""
    # Validate domain and fingerprint before transferring the immutable coordinates.
    z = np.asarray(factor.z)
    if z.dtype != np.dtype(np.float64):
        raise FactorIneligible("GPU factor coordinates must use float64 without coercion")
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise FactorIneligible("GPU factor coordinates must be a finite matrix")
    if not 2 <= int(z.shape[0]) <= 64:
        raise FactorIneligible("GPU factor row count is outside calibrated [2, 64]")
    if not 2 <= int(factor.ambient_dim) <= 256_000:
        raise FactorIneligible(
            "GPU factor ambient dimension is outside calibrated [2, 256000]"
        )
    source_fingerprint = policy.require_current_source_fingerprint()
    cpu_fingerprint = policy.require_current_cpu_fingerprint()
    factor_source_fingerprint = policy.require_current_factor_source_fingerprint()
    if dict(factor.cpu_fingerprint) != dict(cpu_fingerprint):
        raise FactorIneligible("factor construction fingerprint mismatch")
    if dict(factor.source_fingerprint) != dict(factor_source_fingerprint):
        raise FactorIneligible("factor construction source fingerprint mismatch")
    resolved = configure_deterministic_gpu(device)
    fingerprint = policy.require_current_gpu_fingerprint(resolved)
    torch.cuda.synchronize(resolved)
    torch.cuda.reset_peak_memory_stats(resolved)
    started = perf_counter()
    z_device = torch.as_tensor(np.array(z, copy=True), dtype=torch.float64, device=resolved)
    torch.cuda.synchronize(resolved)
    transferred_at = perf_counter()

    # Keep the mechanics in a separate standalone module so this public boundary stays small.
    from ._vmfm_factor_gpu_em import fit_factor_movmf_gpu_impl

    try:
        with torch.inference_mode():
            fit = fit_factor_movmf_gpu_impl(
                factor,
                z_device,
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
    except DenseRerunRequired as error:
        # Synchronize routed attempts so their transfer and CUDA work remain auditable.
        torch.cuda.synchronize(resolved)
        failed_at = perf_counter()
        event = dict(error.event)
        event["backend_timings_seconds"] = {
            "transfer": float(transferred_at - started),
            "gpu_attempt": float(failed_at - transferred_at),
            "end_to_end": float(failed_at - started),
        }
        event["peak_device_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(resolved)
        )
        raise DenseRerunRequired(error.reason, event) from error
    torch.cuda.synchronize(resolved)
    finished = perf_counter()
    trace = dict(fit.trace)
    trace["backend"] = GPU_BACKEND_NAME
    trace["gpu_fingerprint"] = fingerprint
    trace["gpu_source_fingerprint"] = source_fingerprint
    trace["factor_cpu_fingerprint"] = cpu_fingerprint
    trace["factor_source_fingerprint"] = factor_source_fingerprint
    trace["timings_seconds"] = {
        **dict(trace.get("timings_seconds", {})),
        "transfer": float(transferred_at - started),
        "gpu_fit": float(finished - transferred_at),
        "end_to_end": float(finished - started),
    }
    trace["peak_device_memory_bytes"] = int(torch.cuda.max_memory_allocated(resolved))
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
        trace=trace,
    )
