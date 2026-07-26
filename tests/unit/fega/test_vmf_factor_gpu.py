from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fega.config_schema import DirectionalMixtureFitConfig
from fega.core.vmf.factor_reuse import FeatureFactor, fit_prepared_gpu_or_dense
from fega.core.vmf.fit import (
    fit_vmf_candidate,
    production_gpu_factor_policy,
    vmf_backend_fingerprints,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor import (
    DenseRerunRequired,
    FactorAmbiguityPolicy,
    FactorIneligible,
    fit_factor_movmf,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor_gpu import (
    GPU_BACKEND_NAME,
    GpuFactorAmbiguityPolicy,
    fit_factor_movmf_gpu,
    gpu_backend_manifest,
    validate_gpu_execution_workers,
)


def _separated_rows() -> torch.Tensor:
    """Return a deterministic two-mode cloud far from every decision boundary."""
    # Keep the fixture small while exercising multi-component initialization and EM.
    rows = torch.zeros((12, 64), dtype=torch.float32)
    rows[:6, 0] = 1.0
    rows[6:, 1] = 1.0
    rows[:, 2] = torch.linspace(-0.02, 0.02, 12)
    return rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True)


def test_gpu_backend_unavailable_restarts_complete_dense_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route an unavailable GPU from the original provider, seed, and full budget."""
    # Force only the GPU admission boundary to fail; dense authority remains real.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    rows = _separated_rows()
    prepared = FeatureFactor.build(rows).full_view

    def unavailable(*_args, **_kwargs):
        """Simulate a missing promoted CUDA fingerprint before any fit state exists."""
        # Use the production ineligibility type so the routed boundary owns fallback.
        from fega.core.vmf.utils._spherecluster._vmfm_factor import FactorIneligible

        raise FactorIneligible("pinned CUDA backend unavailable")

    monkeypatch.setattr(gpu_backend, "fit_factor_movmf_gpu", unavailable)
    routed = fit_prepared_gpu_or_dense(
        prepared,
        n_clusters=2,
        n_init=1,
        max_iter=8,
        random_state=17,
        policy=GpuFactorAmbiguityPolicy.shadow(),
        device="cuda:0",
    )

    assert routed.backend == "dense_cpu"
    assert routed.factor_fit is None
    assert routed.dense_result is not None
    assert routed.route_reason == "pinned CUDA backend unavailable"
    assert routed.trace["route"] == {
        "reason": "pinned CUDA backend unavailable",
        "event": {},
        "original_seed": 17,
        "original_n_init": 1,
        "original_max_iter": 8,
        "source_sha256": prepared.factor.source_sha256,
        "requested_backend": GPU_BACKEND_NAME,
    }


def test_gpu_rejects_factor_built_under_unaccepted_cpu_fingerprint() -> None:
    """Route dense before CUDA when QR coordinates carry a drifted CPU identity."""
    # GPU eligibility inherits the accepted CPU factor that constructs its coordinates.
    prepared = FeatureFactor.build(_separated_rows()).full_view
    assert prepared.factor is not None
    drifted_factor = replace(
        prepared.factor,
        cpu_fingerprint={**prepared.factor.cpu_fingerprint, "numpy": "drifted"},
    )
    drifted_view = replace(prepared, factor=drifted_factor)

    routed = fit_prepared_gpu_or_dense(
        drifted_view,
        n_clusters=2,
        n_init=1,
        max_iter=8,
        random_state=17,
        policy=production_gpu_factor_policy(),
        device="cuda:0",
    )

    assert routed.backend == "dense_cpu"
    assert routed.factor_fit is None
    assert routed.dense_result is not None
    assert routed.route_reason == "factor construction fingerprint mismatch"
    assert routed.trace["route"]["requested_backend"] == GPU_BACKEND_NAME


def test_gpu_rejects_current_cpu_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route dense when the live NumPy/SciPy/BLAS identity leaves calibration."""
    # Reject before CUDA even when the cached factor still carries its original stamp.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    prepared = FeatureFactor.build(_separated_rows()).full_view
    assert prepared.factor is not None
    drifted = {**prepared.factor.cpu_fingerprint, "numpy": "drifted"}
    monkeypatch.setattr(gpu_backend, "current_cpu_numerical_fingerprint", lambda: drifted)

    routed = fit_prepared_gpu_or_dense(
        prepared,
        n_clusters=2,
        n_init=1,
        max_iter=8,
        random_state=17,
        policy=production_gpu_factor_policy(),
        device="cuda:0",
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "CPU numerical fingerprint mismatch"
    assert routed.trace["route"]["requested_backend"] == GPU_BACKEND_NAME


def test_gpu_rejects_uncalibrated_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route dense when either promoted GPU implementation source hash drifts."""
    # Source identity is independent of a matching CUDA device fingerprint.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    prepared = FeatureFactor.build(_separated_rows()).full_view
    monkeypatch.setattr(
        gpu_backend,
        "gpu_backend_source_fingerprint",
        lambda: {
            "gpu_source_sha256": "0" * 64,
            "gpu_em_source_sha256": "1" * 64,
        },
    )

    routed = fit_prepared_gpu_or_dense(
        prepared,
        n_clusters=2,
        n_init=1,
        max_iter=8,
        random_state=17,
        policy=production_gpu_factor_policy(),
        device="cuda:0",
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "GPU source fingerprint mismatch"
    assert routed.trace["route"]["requested_backend"] == GPU_BACKEND_NAME


def test_gpu_rejects_inherited_cpu_factor_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route dense when the CPU implementation owning GPU coordinates changes."""
    # Keep the native GPU sources valid while drifting the inherited factor source.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    prepared = FeatureFactor.build(_separated_rows()).full_view
    monkeypatch.setattr(
        gpu_backend,
        "factor_backend_source_fingerprint",
        lambda: {
            "factor_source_sha256": "0" * 64,
            "factor_em_source_sha256": "1" * 64,
        },
    )

    routed = fit_prepared_gpu_or_dense(
        prepared,
        n_clusters=2,
        n_init=1,
        max_iter=8,
        random_state=17,
        policy=production_gpu_factor_policy(),
        device="cuda:0",
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "factor source fingerprint mismatch"
    assert routed.trace["route"]["requested_backend"] == GPU_BACKEND_NAME


def test_gpu_execution_rejects_parallel_feature_workers() -> None:
    """Keep the promoted backend on one deterministic stream and worker lane."""
    # CPU routing stays unrestricted; only explicit GPU execution owns this constraint.
    validate_gpu_execution_workers("factor_cpu", 16)
    validate_gpu_execution_workers(GPU_BACKEND_NAME, 1)
    with pytest.raises(ValueError, match="exactly one feature worker"):
        validate_gpu_execution_workers(GPU_BACKEND_NAME, 2)


def test_gpu_driver_query_failure_is_optimized_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when the public NVIDIA driver identity cannot be queried."""
    # Clear the cached real driver before simulating a missing system boundary.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    gpu_backend._nvidia_driver_version.cache_clear()

    def unavailable_driver(*_args, **_kwargs):
        """Simulate an installation without a usable nvidia-smi boundary."""
        # Raise the same operating-system error produced by a missing executable.
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_backend.subprocess, "run", unavailable_driver)

    with pytest.raises(FactorIneligible, match="driver fingerprint is unavailable"):
        gpu_backend._nvidia_driver_version(0)

    gpu_backend._nvidia_driver_version.cache_clear()


def test_vmf_backend_fingerprint_binds_gpu_sources_and_calibration() -> None:
    """Invalidate checkpoints when CUDA mechanics or calibration identity changes."""
    # Require source, backend domain, and calibration hashes in the common fingerprint.
    fingerprint = vmf_backend_fingerprints(
        backend=GPU_BACKEND_NAME,
        gpu_device="cuda:0",
    )

    assert fingerprint["gpu_factor"]["backend"] == GPU_BACKEND_NAME
    assert len(fingerprint["gpu_factor"]["gpu_source_sha256"]) == 64
    assert len(fingerprint["gpu_factor"]["gpu_em_source_sha256"]) == 64
    assert fingerprint["gpu_calibration"]["promoted"] is True
    assert fingerprint["gpu_calibration"]["source_fingerprint"] == {
        "gpu_source_sha256": fingerprint["gpu_factor"]["gpu_source_sha256"],
        "gpu_em_source_sha256": fingerprint["gpu_factor"]["gpu_em_source_sha256"],
    }
    assert fingerprint["gpu_calibration"]["factor_source_fingerprint"] == {
        "factor_source_sha256": fingerprint["factor"]["factor_sha256"],
        "factor_em_source_sha256": fingerprint["factor"]["factor_em_sha256"],
    }
    assert fingerprint["gpu_calibration"]["factor_cpu_numerical_fingerprint"] == (
        fingerprint["calibration"]["cpu_numerical_fingerprint"]
    )
    assert fingerprint["gpu_calibration"]["gpu_numerical_fingerprint"][
        "python_version"
    ] == "3.10.19"
    assert fingerprint["gpu_calibration"]["gpu_numerical_fingerprint"][
        "nvidia_driver_version"
    ] == "590.48.01"
    assert len(fingerprint["policy_manifest"]["sha256"]) == 64
    assert fingerprint["live_admission"]["accepted"] is True
    assert fingerprint["gpu_domain"]["dtype"] == "float64"


def test_gpu_checkpoint_identity_records_exact_device_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalidate GPU checkpoints when the live device manifest changes."""
    # Hold inherited CPU/source identities fixed and drift one GPU runtime field.
    from fega.core.vmf import fit as vmf_fit
    from fega.core.vmf.utils._spherecluster import _vmfm_factor as cpu_backend
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu as gpu_backend

    policy = production_gpu_factor_policy()
    expected_cpu = dict(policy.expected_cpu_fingerprint or {})
    expected_factor_source = dict(policy.expected_factor_source_fingerprint or {})
    expected_gpu_source = dict(policy.expected_source_fingerprint or {})
    expected_gpu = dict(policy.expected_gpu_fingerprint or {})
    monkeypatch.setattr(
        cpu_backend, "current_cpu_numerical_fingerprint", lambda: expected_cpu
    )
    monkeypatch.setattr(
        cpu_backend,
        "factor_backend_source_fingerprint",
        lambda: expected_factor_source,
    )
    monkeypatch.setattr(
        gpu_backend, "gpu_backend_source_fingerprint", lambda: expected_gpu_source
    )
    monkeypatch.setattr(
        gpu_backend,
        "gpu_backend_fingerprint",
        lambda _device: {**expected_gpu, "nvidia_driver_version": "drifted"},
    )

    fingerprint = vmf_fit.vmf_backend_fingerprints(
        backend=GPU_BACKEND_NAME,
        gpu_device="cuda:0",
    )

    assert fingerprint["live_admission"]["accepted"] is False
    assert fingerprint["live_admission"]["gpu_numerical_fingerprint"][
        "nvidia_driver_version"
    ] == "drifted"


def test_dense_cpu_is_an_explicit_comparable_backend() -> None:
    """Expose corrected dense CPU directly for the production benchmark contract."""
    # Request the authority backend without disabling or passing through factor routing.
    candidate = fit_vmf_candidate(
        _separated_rows(),
        k=2,
        n_init=1,
        max_iter=12,
        seed=23,
        backend="dense_cpu",
    )

    assert candidate.backend == "dense_cpu"
    assert candidate.route_reason == "explicit_dense_cpu_backend"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend validation")
def test_gpu_manifest_pins_deterministic_float64_execution() -> None:
    """Record every decision-relevant CUDA setting required by the promoted domain."""
    # Configure first, then inspect the same device fingerprint used for eligibility.
    manifest = gpu_backend_manifest("cuda:0")

    assert manifest["backend"] == GPU_BACKEND_NAME
    assert manifest["dtype"] == "float64"
    assert manifest["deterministic_algorithms"] is True
    assert manifest["tf32_matmul"] is False
    assert manifest["tf32_cudnn"] is False
    assert manifest["matmul_precision"] == "highest"
    assert manifest["cublas_workspace_config"] == ":4096:8"
    assert manifest["stream_count"] == 1
    assert manifest["synchronization_policy"] == "before_and_after_timed_regions"
    assert manifest["reduction_order"] == "fixed_tensor_shape_single_stream"
    assert "NVIDIA A100" in manifest["device_name"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend validation")
def test_gpu_dense_route_preserves_synchronized_attempt_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain transfer and failed-attempt telemetry when GPU routes dense."""
    # Force the standalone implementation to route after coordinates reach CUDA.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor_gpu_em as gpu_em

    factor = FeatureFactor.build(_separated_rows()).full_view.factor
    assert factor is not None

    def route_dense(*_args, **_kwargs):
        """Raise one calibrated route with its original trigger evidence."""
        # The public GPU boundary must augment rather than replace this event.
        raise DenseRerunRequired(
            "ambiguous_convergence", {"surface": "convergence"}
        )

    monkeypatch.setattr(gpu_em, "fit_factor_movmf_gpu_impl", route_dense)

    with pytest.raises(DenseRerunRequired) as captured:
        fit_factor_movmf_gpu(
            factor,
            n_clusters=2,
            n_init=1,
            max_iter=4,
            random_state=17,
            policy=GpuFactorAmbiguityPolicy.shadow(),
            device="cuda:0",
        )

    assert captured.value.event["surface"] == "convergence"
    assert set(captured.value.event["backend_timings_seconds"]) == {
        "transfer",
        "gpu_attempt",
        "end_to_end",
    }
    assert captured.value.event["backend_timings_seconds"]["transfer"] >= 0.0
    assert captured.value.event["backend_timings_seconds"]["gpu_attempt"] >= 0.0
    assert captured.value.event["peak_device_memory_bytes"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend validation")
def test_gpu_factor_is_repeatable_and_matches_cpu_scientific_decisions() -> None:
    """Repeat one pinned backend bitwise and match CPU labels and selected init."""
    # Share the accepted CPU factor so the comparison isolates only EM arithmetic.
    feature = FeatureFactor.build(_separated_rows())
    factor = feature.full_view.factor
    assert factor is not None
    cpu = fit_factor_movmf(
        factor,
        n_clusters=2,
        n_init=1,
        max_iter=12,
        random_state=23,
        policy=FactorAmbiguityPolicy.shadow(),
    )
    first = fit_factor_movmf_gpu(
        factor,
        n_clusters=2,
        n_init=1,
        max_iter=12,
        random_state=23,
        policy=GpuFactorAmbiguityPolicy.shadow(),
        device="cuda:0",
    )
    second = fit_factor_movmf_gpu(
        factor,
        n_clusters=2,
        n_init=1,
        max_iter=12,
        random_state=23,
        policy=GpuFactorAmbiguityPolicy.shadow(),
        device="cuda:0",
    )

    assert np.array_equal(first.labels, cpu.labels)
    assert first.selected_init_index == cpu.selected_init_index
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.responsibilities, second.responsibilities)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.kappas, second.kappas)
    assert np.array_equal(first.center_coefficients, second.center_coefficients)
    assert first.log_likelihood == second.log_likelihood


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend validation")
def test_promoted_gpu_backend_routes_safe_case_on_pinned_device() -> None:
    """Exercise the production GPU policy through the public fixed-mode adapter."""
    # A separated case is outside calibrated ambiguity bands and must stay on CUDA.
    trace: dict[str, object] = {}
    candidate = fit_vmf_candidate(
        _separated_rows(),
        k=2,
        n_init=1,
        max_iter=12,
        seed=23,
        trace=trace,
        backend=GPU_BACKEND_NAME,
        gpu_device="cuda:0",
    )

    assert candidate.backend == GPU_BACKEND_NAME
    assert candidate.route_reason is None
    assert trace["backend"] == GPU_BACKEND_NAME
    assert trace["route_reason"] is None
    assert set(trace["backend_timings_seconds"]) == {
        "initialization",
        "em",
        "normalizer",
        "transfer",
        "gpu_fit",
        "end_to_end",
    }


def test_directional_mixture_config_defaults_to_benchmark_winning_dense_cpu() -> None:
    """Default production runs to dense CPU after optimized routes lose closure."""
    # Factor CPU and GPU remain explicit choices without imposing routed overhead.
    assert DirectionalMixtureFitConfig().backend == "dense_cpu"
