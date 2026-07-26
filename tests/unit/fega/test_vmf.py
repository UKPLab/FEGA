from __future__ import annotations

import json
import logging
import math
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fega.core.vmf.fit as vmf_fit
import fega.core.vmf.metrics as vmf_metrics
import fega.core.vmf.runner as vmf_runner
from fega.config_schema import DirectionalMixtureFitConfig, FEGAPipelineConfig
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    gram_fingerprint,
    unembedding_fingerprint,
)
from fega.core.source_fingerprint import (
    canonical_json_digest,
    canonical_source_fingerprint,
)
from fega.core.vmf.backend_policy import (
    BackendPolicyManifestError,
    load_backend_policy_manifest,
    validate_backend_policy_manifest,
)
from fega.core.vmf.fit import (
    VmfCandidate,
    VmfFit,
    finalize_vmf_candidate,
    fit_vmf_candidate,
    fit_vmf_mixture,
    vmf_backend_fingerprints,
)
from fega.core.vmf.metrics import (
    PUBLIC_METRIC_KEYS,
    bic_score,
    c_ray_unit_rows,
    derived_vmf_seed,
    feature_fit_seed,
    score_vmf_feature,
    select_by_bic,
)
from fega.core.vmf.utils._spherecluster import _vmf_numerics
from fega.core.vmf.utils._spherecluster._vmf_numerics import (
    log_vmf_normalizer,
    log_vmf_normalizer_plus_kappa,
    vmf_mixture_log_likelihood,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor_gpu import GPU_BACKEND_NAME
from fega.paths import (
    effect_summary_path,
    effect_tensors_manifest_path,
    geometry_metrics_scores_path,
    gram_cache_meta_path,
    gram_cache_tensor_path,
    vmf_checkpoint_path,
    vmf_scores_path,
)


def _cfg(**overrides) -> DirectionalMixtureFitConfig:
    """Build the retained operational vMF configuration for focused tests."""
    # Keep scientific reporting thresholds out of fit-stage fixtures.
    values = {
        "enabled": True,
        "effect_space": "pre_softcap_logits",
        "k_values": [1, 2],
        "bic_tolerance": 1.0e-9,
        "resample_fraction": 0.8,
        "resample_rounds": 2,
        "n_init": 1,
        "max_iter": 20,
    }
    values.update(overrides)
    return DirectionalMixtureFitConfig(**values)


def _two_mode_rows() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10, dtype=torch.float32)


def _single_ray_rows() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0]] * 12, dtype=torch.float32)


def _factor_candidate_rows() -> torch.Tensor:
    """Create an unambiguous calibrated-domain cloud with ambient width above N."""
    # Separate two modes while retaining nontrivial row variation for factor EM.
    rng = np.random.RandomState(4)
    first = np.zeros(64, dtype=np.float64)
    second = np.zeros(64, dtype=np.float64)
    first[0] = 4.0
    second[1] = 4.0
    rows = np.vstack(
        [
            rng.normal(first, 0.2, size=(6, 64)),
            rng.normal(second, 0.2, size=(6, 64)),
        ]
    )
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return torch.from_numpy(rows)


def _directional_fit(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
    del cfg, seed
    x = rows.detach().cpu().numpy()
    if k == 1:
        labels = np.zeros(x.shape[0], dtype=np.int64)
        centers = np.asarray([[1.0, 0.0]], dtype=np.float64)
        weights = np.asarray([1.0], dtype=np.float64)
        kappas = np.asarray([5.0], dtype=np.float64)
        ll = -10.0
    else:
        labels = np.asarray(x[:, 1] > x[:, 0], dtype=np.int64)
        centers = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        weights = counts / counts.sum()
        kappas = np.full(k, 10.0, dtype=np.float64)
        ll = 100.0
    responsibilities = np.zeros((k, x.shape[0]), dtype=np.float64)
    responsibilities[labels, np.arange(x.shape[0])] = 1.0
    return VmfFit(
        k=k,
        labels=labels,
        responsibilities=responsibilities,
        centers=centers,
        weights=weights,
        kappas=kappas,
        log_likelihood=ll,
    )


def _fit_record(k: int, ll: float) -> VmfFit:
    labels = np.zeros(4, dtype=np.int64)
    return VmfFit(
        k=k,
        labels=labels,
        responsibilities=np.ones((k, 4), dtype=np.float64) / float(k),
        centers=np.eye(k, 3, dtype=np.float64),
        weights=np.ones(k, dtype=np.float64) / float(k),
        kappas=np.ones(k, dtype=np.float64),
        log_likelihood=ll,
    )


def test_single_ray_is_fit_independently_of_geometry_metrics_ambiguity() -> None:
    """Require every eligible canonical cloud to enter operational fitting."""
    # Fit a coherent ray without supplying any directional-concentration metrics.
    result = score_vmf_feature(
        _single_ray_rows(),
        _cfg(),
        seed=1,
        fit_fn=_directional_fit,
    )

    assert result.fit_status == "fitted"
    assert result.metrics["selected_mode_count"] == 2


def test_two_coherent_modes_emit_accepted_multimode_metrics() -> None:
    result = score_vmf_feature(
        _two_mode_rows(),
        _cfg(),
        seed=1,
        fit_fn=_directional_fit,
    )

    metrics = result.metrics
    assert set(metrics) == {
        "selected_mode_count",
        "delta_mix",
        "mode_mass_min",
        "min_mode_c_ray",
        "mode_kappa_min",
    }
    assert metrics["selected_mode_count"] == 2
    assert metrics["delta_mix"] == pytest.approx(0.526315789, abs=1.0e-6)
    assert metrics["mode_mass_min"] == pytest.approx(0.5)
    assert metrics["min_mode_c_ray"] == pytest.approx(1.0)
    assert result.assignment_stability["value"] == pytest.approx(1.0)
    assert metrics["mode_kappa_min"] == pytest.approx(10.0)


def test_tiny_mode_keeps_computed_reporting_metrics() -> None:
    """Preserve tiny-mode diagnostics without constraining BIC model identity."""
    # Exercise reporting gates without constraining the BIC-selected mode count.
    rows = torch.tensor([[1.0, 0.0]] * 9 + [[0.0, 1.0]], dtype=torch.float32)
    result = score_vmf_feature(
        rows,
        _cfg(resample_rounds=0),
        seed=1,
        fit_fn=_directional_fit,
    )

    assert result.metrics["mode_mass_min"] == pytest.approx(0.1)
    assert result.metrics["min_mode_c_ray"] is None
    assert result.metrics["mode_kappa_min"] == pytest.approx(10.0)


def test_impossible_k_values_are_not_fit() -> None:
    """Skip only mode counts that exceed the available observation count."""
    # Include one sample-count-feasible fit and one mathematically impossible fit.
    attempted: list[int] = []

    def record_fit(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
        attempted.append(k)
        return _directional_fit(rows, k, cfg, seed)

    score_vmf_feature(
        torch.tensor([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 3, dtype=torch.float32),
        _cfg(k_values=[1, 9]),
        seed=1,
        fit_fn=record_fit,
    )

    assert attempted == [1]


def test_every_feasible_mode_count_is_attempted_before_support_filtering() -> None:
    """Require BIC candidates to be attempted before reporting-support gates."""
    attempted: list[int] = []

    # Return finite fits for every requested M while making M=1 the BIC winner.
    def record_fit(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
        del cfg, seed
        attempted.append(k)
        n_rows = int(rows.shape[0])
        labels = np.zeros(n_rows, dtype=np.int64)
        responsibilities = np.zeros((k, n_rows), dtype=np.float64)
        responsibilities[0, :] = 1.0
        return VmfFit(
            k=k,
            labels=labels,
            responsibilities=responsibilities,
            centers=np.tile(np.asarray([[1.0, 0.0]]), (k, 1)),
            weights=np.asarray([1.0] + [0.0] * (k - 1), dtype=np.float64),
            kappas=np.ones(k, dtype=np.float64),
            log_likelihood=-100.0 * float(k),
        )

    score_vmf_feature(
        torch.tensor([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 3, dtype=torch.float32),
        _cfg(k_values=[1, 2, 3, 4]),
        seed=1,
        fit_fn=record_fit,
    )

    assert attempted == [1, 2, 3, 4]


def test_bic_selected_mode_count_survives_reporting_gate_failure() -> None:
    """Keep the BIC-selected mode count when reporting-only gates fail."""
    # Make M=2 win BIC while its tiny second mode fails mass and definedness gates.
    rows = torch.tensor([[1.0, 0.0]] * 9 + [[0.0, 1.0]], dtype=torch.float32)
    result = score_vmf_feature(
        rows,
        _cfg(resample_rounds=0),
        seed=1,
        fit_fn=_directional_fit,
    )

    assert result.metrics["selected_mode_count"] == 2
    assert result.metrics["mode_mass_min"] == pytest.approx(0.1)
    assert result.metrics["min_mode_c_ray"] is None


def test_bic_selection_chooses_lowest_bic_and_ties_smaller_k() -> None:
    n_valid = 10
    dim = 3
    k1 = _fit_record(1, ll=0.0)
    k2 = _fit_record(2, ll=100.0)

    assert select_by_bic([k1, k2], n_valid=n_valid, dim=dim).k == 2

    tie_ll = ((7 - 3) * math.log(n_valid)) / 2.0
    tied_k2 = _fit_record(2, ll=tie_ll)
    assert bic_score(k1, n_valid=n_valid, dim=dim) == pytest.approx(
        bic_score(tied_k2, n_valid=n_valid, dim=dim)
    )
    trace = {}
    assert (
        select_by_bic(
            [tied_k2, k1], n_valid=n_valid, dim=dim, trace=trace
        ).k
        == 1
    )
    assert trace["operator"] == "candidate_bic < incumbent_bic - tolerance"
    assert trace["selected_mode_count"] == 1
    assert trace["comparisons"][1]["strict_improvement"] is False

    nonfinite = _fit_record(3, ll=float("nan"))
    assert select_by_bic([nonfinite, k2], n_valid=n_valid, dim=dim).k == 2


def test_factor_bic_ambiguity_restarts_each_factor_candidate_densely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a cross-mode factor BIC boundary only with complete dense reruns."""
    # Choose likelihoods that make the inherited k=1 and k=2 BIC values equal.
    n_valid = 12
    dim = 64
    k1_params = 64
    k2_params = 129
    k2_likelihood = 0.5 * (k2_params - k1_params) * math.log(n_valid)

    def candidate(k: int, likelihood: float) -> VmfCandidate:
        """Build a minimal center-free factor candidate at the BIC boundary."""
        # Supply only fields consumed by selection and the dense-rerun contract.
        labels = np.zeros(n_valid, dtype=np.int64)
        return VmfCandidate(
            k=k,
            labels=labels,
            responsibilities=np.ones((k, n_valid), dtype=np.float64) / k,
            weights=np.ones(k, dtype=np.float64) / k,
            kappas=np.ones(k, dtype=np.float64),
            log_likelihood=likelihood,
            center_coefficients=np.ones((k, n_valid), dtype=np.float64),
            ambient_dim=dim,
            source_sha256="source",
            backend="factor_cpu_explicit_y",
            route_reason=None,
            seed=k,
            n_init=1,
            max_iter=5,
            trace={},
        )

    candidates = [candidate(1, 0.0), candidate(2, k2_likelihood)]
    rerun_modes: list[int] = []

    def dense_rerun(
        fit: VmfCandidate,
        rows: torch.Tensor,
        *,
        reason: str,
        trace: dict[str, object] | None = None,
    ) -> VmfCandidate:
        """Record the complete fixed-mode resolution selected by the BIC guard."""
        # Preserve scientific values while exposing a resolved dense backend.
        del rows
        rerun_modes.append(fit.k)
        assert reason == "bic_selection_ambiguity"
        if trace is not None:
            trace.update({"backend": "dense_cpu"})
        return replace(
            fit,
            center_coefficients=None,
            backend="dense_cpu",
            route_reason=reason,
        )

    monkeypatch.setattr(vmf_metrics, "rerun_vmf_candidate_dense", dense_rerun)
    evidence = [
        {
            "mode_count": fit.k,
            "status": "finite",
            "log_likelihood": fit.log_likelihood,
        }
        for fit in candidates
    ]
    traces = [
        {"mode_count": fit.k, "status": "finite", "fit_trace": {}}
        for fit in candidates
    ]

    resolved = vmf_metrics._resolve_ambiguous_factor_bic(
        candidates,
        torch.zeros((n_valid, dim), dtype=torch.float64),
        evidence,
        traces,
        n_valid=n_valid,
        dim=dim,
        tolerance=1.0e-9,
    )

    assert rerun_modes == [1, 2]
    assert all(fit.backend == "dense_cpu" for fit in resolved)
    assert all(record["bic_resolution"] == "dense_cpu" for record in traces)


def test_operational_states_are_explicit_and_never_use_sentinel_modes() -> None:
    """Distinguish insufficient, non-finite, failed, and fitted operational states."""
    # Drive each state with the smallest direct fit behavior and inspect selection identity.
    insufficient = score_vmf_feature(
        torch.tensor([[1.0, 0.0]] * 7), _cfg(), seed=1, fit_fn=_directional_fit
    )

    def nonfinite_fit(
        rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        """Return a structurally valid fit whose likelihood is unusable for BIC."""
        # Reuse deterministic parameters and invalidate only candidate likelihood.
        fit = _directional_fit(rows, k, cfg, seed)
        return VmfFit(**{**fit.__dict__, "log_likelihood": float("nan")})

    def failed_fit(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
        """Represent an operational estimator failure for every attempted mode count."""
        # Raise independently for each candidate so the scorer records every failure.
        del rows, k, cfg, seed
        raise FloatingPointError("fit failed")

    nonfinite = score_vmf_feature(
        _two_mode_rows(), _cfg(), seed=1, fit_fn=nonfinite_fit
    )
    failure_trace = {}
    failed = score_vmf_feature(
        _two_mode_rows(),
        _cfg(),
        seed=1,
        fit_fn=failed_fit,
        trace=failure_trace,
    )
    fitted = score_vmf_feature(
        _two_mode_rows(), _cfg(resample_rounds=0), seed=1, fit_fn=_directional_fit
    )

    assert [
        insufficient.fit_status,
        nonfinite.fit_status,
        failed.fit_status,
        fitted.fit_status,
    ] == ["insufficient_contexts", "no_finite_candidate", "fit_failed", "fitted"]
    assert insufficient.metrics["selected_mode_count"] is None
    assert nonfinite.model_selection["selected_mode_count"] is None
    assert failed.model_selection["failed_count"] == 2
    assert fitted.model_selection["selected_mode_count"] == 2
    assert failure_trace["fit_status"] == "fit_failed"
    assert failure_trace["workload"]["candidate_fits_failed"] == 2
    assert [item["error_type"] for item in failure_trace["candidates"]] == [
        "FloatingPointError",
        "FloatingPointError",
    ]


def test_unexpected_candidate_fit_error_propagates() -> None:
    """Propagate unexpected scorer failures instead of converting them to fit evidence."""
    # Raise an unexpected error from the first candidate fit.
    def unexpected_fit(
        rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        """Raise an unexpected candidate-scoring failure."""
        # Discard the fit inputs before surfacing the programming error.
        del rows, k, cfg, seed
        raise RuntimeError("unexpected scorer failure")

    with pytest.raises(RuntimeError, match="unexpected scorer failure"):
        score_vmf_feature(_two_mode_rows(), _cfg(), seed=1, fit_fn=unexpected_fit)


def test_fitted_weights_drive_mass_and_delta_while_hard_counts_define_rays() -> None:
    """Separate soft fitted mass from hard-assignment definedness and frequencies."""
    # Give equal hard counts unequal fitted weights and keep both hard modes defined.
    rows = torch.tensor(
        [[1.0, 0.0]] * 4
        + [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )

    def unequal_weight_fit(
        fit_rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        """Return an M=2 winner with weights deliberately unlike hard frequencies."""
        # Preserve four hard observations per mode while assigning fitted mass 0.75/0.25.
        del cfg, seed
        n_rows = int(fit_rows.shape[0])
        labels = np.zeros(n_rows, dtype=np.int64) if k == 1 else np.asarray([0] * 4 + [1] * 4)
        responsibilities = np.zeros((k, n_rows), dtype=np.float64)
        responsibilities[labels, np.arange(n_rows)] = 1.0
        return VmfFit(
            k=k,
            labels=labels,
            responsibilities=responsibilities,
            centers=np.tile(np.asarray([[1.0, 0.0]]), (k, 1)),
            weights=np.asarray([1.0]) if k == 1 else np.asarray([0.75, 0.25]),
            kappas=np.ones(k),
            log_likelihood=-100.0 if k == 1 else 100.0,
        )

    result = score_vmf_feature(
        rows, _cfg(resample_rounds=0), seed=1, fit_fn=unequal_weight_fit
    )

    expected = 0.75 * 1.0 + 0.25 * (1.0 / 3.0) - c_ray_unit_rows(rows)
    assert result.metrics["mode_mass_min"] == pytest.approx(0.25)
    assert result.metrics["delta_mix"] == pytest.approx(expected)
    assert result.selected_fit["hard_mode_counts"] == [4, 4]


def test_fixed_assignment_plan_reports_failed_required_refit_without_replacement() -> None:
    """Make one fixed required replicate fail and preserve the complete unavailable plan."""
    # Count subset refits and fail exactly the second predeclared replicate.
    refit_calls = 0
    seen_n_init: list[int] = []

    def partly_failing_fit(
        rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        """Fit candidates normally, then fail one deterministic subset refit."""
        # Distinguish full-cloud candidates from fixed-size resampling calls.
        nonlocal refit_calls
        seen_n_init.append(cfg.n_init)
        if int(rows.shape[0]) < 20 and k == 2:
            refit_calls += 1
            if refit_calls == 2:
                raise FloatingPointError("required refit failed")
        return _directional_fit(rows, k, cfg, seed)

    result = score_vmf_feature(
        _two_mode_rows(),
        _cfg(resample_rounds=3, n_init=4),
        seed=9,
        fit_fn=partly_failing_fit,
    )

    stability = result.assignment_stability
    assert stability["status"] == "unavailable"
    assert stability["value"] is None
    assert (stability["requested_count"], stability["successful_count"], stability["failed_count"]) == (3, 2, 1)
    assert len(stability["replicates"]) == 3
    assert len({item["subset_seed"] for item in stability["replicates"]}) == 3
    assert all(item["subset_indices"] == sorted(item["subset_indices"]) for item in stability["replicates"])
    assert [
        "adjusted_rand_score" in item for item in stability["replicates"]
    ].count(True) == 2
    assert all(
        "refit_seed" in item and "refit_seed_base" not in item
        for item in stability["replicates"]
    )
    assert all(value == 4 for value in seen_n_init)


def test_unexpected_assignment_refit_error_propagates() -> None:
    """Propagate an unexpected error from the first assignment subset refit."""
    # Fit full-cloud candidates normally and fail only the first smaller subset call.
    subset_calls = 0

    def unexpected_refit(
        rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        """Raise an unexpected failure only during the first subset refit."""
        # Distinguish candidate fits from subset refits by their row counts.
        nonlocal subset_calls
        if int(rows.shape[0]) < 20:
            subset_calls += 1
            if subset_calls == 1:
                raise RuntimeError("unexpected refit failure")
        return _directional_fit(rows, k, cfg, seed)

    with pytest.raises(RuntimeError, match="unexpected refit failure"):
        score_vmf_feature(
            _two_mode_rows(), _cfg(), seed=9, fit_fn=unexpected_refit
        )


def test_delta_mix_recomputes_global_c_ray_from_logit_rows() -> None:
    rows = _two_mode_rows()
    result = score_vmf_feature(
        rows,
        _cfg(),
        seed=1,
        fit_fn=_directional_fit,
    )

    expected = 1.0 - c_ray_unit_rows(rows)
    assert result.metrics["delta_mix"] == pytest.approx(expected)
    assert result.metrics["delta_mix"] != pytest.approx(1.0 - (-999.0))


def test_rejected_mode_size_below_two_sets_min_mode_c_ray_null() -> None:
    """Keep undefined within-mode c_ray independent of selected M."""
    # Build a singleton mode whose within-mode c_ray cannot be reported.
    rows = torch.tensor([[1.0, 0.0]] * 9 + [[0.0, 1.0]], dtype=torch.float32)

    result = score_vmf_feature(
        rows,
        _cfg(resample_rounds=0),
        seed=1,
        fit_fn=_directional_fit,
    )

    assert result.metrics["min_mode_c_ray"] is None


def test_missing_or_unreliable_kappa_sets_mode_kappa_min_null() -> None:
    """Keep missing concentration diagnostics null without fixing selected M."""
    # Strip fitted kappas so reporting cannot produce a reliable minimum.
    def no_kappa_fit(rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int) -> VmfFit:
        fit = _directional_fit(rows, k, cfg, seed)
        return VmfFit(
            k=fit.k,
            labels=fit.labels,
            responsibilities=fit.responsibilities,
            centers=fit.centers,
            weights=fit.weights,
            kappas=None,
            log_likelihood=fit.log_likelihood,
        )

    result = score_vmf_feature(
        _two_mode_rows(),
        _cfg(),
        seed=1,
        fit_fn=no_kappa_fit,
    )

    assert result.metrics["mode_kappa_min"] is None


def test_same_seed_produces_identical_stability_and_result() -> None:
    first = score_vmf_feature(
        _two_mode_rows(),
        _cfg(),
        seed=7,
        fit_fn=_directional_fit,
    )
    second = score_vmf_feature(
        _two_mode_rows(),
        _cfg(),
        seed=7,
        fit_fn=_directional_fit,
    )

    assert first.metrics == second.metrics


def test_large_feature_seed_derives_distinct_sha_seed_bases() -> None:
    """Keep per-mode SHA seed bases distinct for large feature-derived seeds."""
    # Record full seed bases at the fit-function boundary before adapter folding.
    seeds: list[int] = []

    def record_seed_fit(
        rows: torch.Tensor, k: int, cfg: DirectionalMixtureFitConfig, seed: int
    ) -> VmfFit:
        seeds.append(seed)
        return _directional_fit(rows, k, cfg, seed)

    large_induction_seed = 42 + 64637 * 104729
    score_vmf_feature(
        _two_mode_rows(),
        _cfg(resample_rounds=0),
        seed=large_induction_seed,
        fit_fn=record_seed_fit,
    )

    assert len(seeds) == 2
    assert all(0 <= seed < 2**64 for seed in seeds)
    assert seeds[0] != seeds[1]


def test_local_copied_spherecluster_fit_smoke() -> None:
    rows = _two_mode_rows()
    fit = fit_vmf_mixture(rows, k=2, n_init=1, max_iter=5, seed=0)

    assert fit.k == 2
    assert fit.labels.shape == (20,)
    assert fit.responsibilities.shape == (2, 20)
    assert np.isfinite(fit.log_likelihood)


def test_shared_backend_keeps_factor_candidate_center_free_until_finalization() -> None:
    """Construct an ambient center only for the selected-final compatibility fit."""
    # Request the production factor backend explicitly and verify its lifecycle.
    rows = _factor_candidate_rows()
    candidate = fit_vmf_candidate(
        rows,
        k=2,
        n_init=1,
        max_iter=30,
        seed=7,
        backend="factor_cpu",
    )

    assert isinstance(candidate, VmfCandidate)
    assert candidate.backend == "factor_cpu_explicit_y"
    assert candidate.center_coefficients is not None
    assert not hasattr(candidate, "centers")
    assert not hasattr(candidate, "model")

    selected = finalize_vmf_candidate(candidate, rows)

    assert selected.centers.shape == (2, 64)
    assert selected.model is None
    assert np.allclose(np.linalg.norm(selected.centers, axis=1), 1.0)


def test_selected_dense_candidate_alone_constructs_compatibility_model() -> None:
    """Discard dense candidate centers and recreate its model only after selection."""
    # Use d < N so the compatibility path remains corrected dense CPU.
    rows = _two_mode_rows()
    candidate = fit_vmf_candidate(rows, k=2, n_init=1, max_iter=5, seed=0)

    assert candidate.backend == "dense_cpu"
    assert candidate.center_coefficients is None
    assert not hasattr(candidate, "centers")
    assert not hasattr(candidate, "model")

    selected = finalize_vmf_candidate(candidate, rows)

    assert selected.centers.shape == (2, 2)
    assert selected.model is not None


def test_feature_selection_finalizes_once_after_candidate_and_bootstrap_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep model candidates and assignment bootstraps center-free until selection."""
    # Replace only backend mechanics so this test isolates the lifecycle boundary.
    candidate_calls: list[int] = []
    finalized: list[VmfCandidate] = []

    def candidate_fit(
        rows: torch.Tensor,
        *,
        k: int,
        n_init: int,
        max_iter: int,
        seed: int,
        trace: dict[str, object] | None = None,
        backend: str = "dense_cpu",
        gpu_device: str = "cuda:0",
    ) -> VmfCandidate:
        """Return the public test fit as a center-free internal candidate."""
        # Mirror the production backend interface while isolating center finalization.
        del gpu_device
        public = _directional_fit(rows, k, _cfg(), seed)
        candidate_calls.append(int(rows.shape[0]))
        if trace is not None:
            trace.update({"backend": backend})
        return VmfCandidate(
            k=public.k,
            labels=public.labels,
            responsibilities=public.responsibilities,
            weights=public.weights,
            kappas=np.asarray(public.kappas, dtype=np.float64),
            log_likelihood=public.log_likelihood,
            center_coefficients=np.ones((k, rows.shape[0]), dtype=np.float64),
            ambient_dim=int(rows.shape[1]),
            source_sha256="test-source",
            backend=backend,
            route_reason=None,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            trace={},
        )

    def finalize_candidate(candidate: VmfCandidate, rows: torch.Tensor) -> VmfFit:
        """Record the one selected-final conversion and supply its public fixture."""
        # Finalization must occur once and only after BIC has selected its mode count.
        finalized.append(candidate)
        return _directional_fit(rows, candidate.k, _cfg(), candidate.seed)

    monkeypatch.setattr(vmf_metrics, "fit_vmf_candidate", candidate_fit)
    monkeypatch.setattr(vmf_metrics, "finalize_vmf_candidate", finalize_candidate)

    result = score_vmf_feature(
        _two_mode_rows(),
        _cfg(k_values=[2], resample_rounds=2),
        seed=9,
    )

    assert result.fit_status == "fitted"
    assert candidate_calls == [20, 16, 16]
    assert len(finalized) == 1
    assert finalized[0].k == 2


def test_vmf_backend_fingerprint_binds_every_resume_identity() -> None:
    """Bind checkpoints to oracle, factor, initialization, and calibration state."""
    # Require CPU and GPU identity classes plus their canonical component digests.
    fingerprint = vmf_backend_fingerprints()

    assert set(fingerprint) == {
        "oracle",
        "factor",
        "gpu_factor",
        "initialization",
        "backend",
        "calibration",
        "gpu_calibration",
        "validated_domain",
        "gpu_domain",
        "policy_manifest",
        "live_admission",
    }
    assert fingerprint["calibration"]["source_fingerprint"] == {
        "factor_source_sha256": fingerprint["factor"]["factor_sha256"],
        "factor_em_source_sha256": fingerprint["factor"]["factor_em_sha256"],
    }
    assert fingerprint["calibration"]["cpu_numerical_fingerprint"][
        "python_version"
    ] == "3.10.19"
    assert all(
        len(component["sha256"]) == 64 for component in fingerprint.values()
    )
    assert fingerprint["live_admission"]["route"] == "dense_authority"


def test_factor_checkpoint_identity_records_exact_live_cpu_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalidate optimized checkpoints when the live CPU identity changes."""
    # Mutate one observed field and require both state and component digest to change.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor as cpu_backend

    accepted = vmf_backend_fingerprints(backend="factor_cpu")
    current = cpu_backend.current_cpu_numerical_fingerprint()
    monkeypatch.setattr(
        cpu_backend,
        "current_cpu_numerical_fingerprint",
        lambda: {**current, "numpy": "drifted"},
    )

    drifted = vmf_backend_fingerprints(backend="factor_cpu")

    assert drifted["live_admission"]["accepted"] is False
    assert drifted["live_admission"]["cpu_numerical_fingerprint"]["numpy"] == (
        "drifted"
    )
    assert drifted["live_admission"]["sha256"] != accepted["live_admission"][
        "sha256"
    ]


@pytest.mark.parametrize(
    ("backend", "failure"),
    [
        ("factor_cpu", "cannot load vMF backend policy"),
        (GPU_BACKEND_NAME, "vMF backend policy digest mismatch"),
    ],
)
def test_policy_failure_whole_fit_reruns_corrected_dense(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    failure: str,
) -> None:
    """Route missing or corrupt optimized authority through the dense fit contract."""
    # Compare the complete candidate with a direct corrected-dense run at one seed/budget.
    rows = _factor_candidate_rows()
    expected = fit_vmf_candidate(
        rows,
        k=2,
        n_init=1,
        max_iter=8,
        seed=41,
        backend="dense_cpu",
    )

    def unavailable_policy() -> dict[str, object]:
        """Simulate an installed manifest that cannot provide optimized authority."""
        # Use the public loader error so the adapter owns whole-fit routing.
        raise BackendPolicyManifestError(failure)

    monkeypatch.setattr(vmf_fit, "load_backend_policy_manifest", unavailable_policy)
    actual = fit_vmf_candidate(
        rows,
        k=2,
        n_init=1,
        max_iter=8,
        seed=41,
        backend=backend,
    )

    assert actual.backend == "dense_cpu"
    assert actual.route_reason is not None
    assert actual.route_reason.startswith("optimized backend policy unavailable")
    assert actual.log_likelihood == expected.log_likelihood
    np.testing.assert_array_equal(actual.labels, expected.labels)
    np.testing.assert_allclose(
        actual.responsibilities,
        expected.responsibilities,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.weights, expected.weights)
    np.testing.assert_allclose(actual.kappas, expected.kappas, rtol=3.0e-15, atol=0.0)


@pytest.mark.parametrize(
    ("backend", "policy_section"),
    [("factor_cpu", "cpu_factor"), (GPU_BACKEND_NAME, "gpu_factor")],
)
def test_construction_invalid_policy_whole_fit_reruns_corrected_dense(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    policy_section: str,
) -> None:
    """Normalize self-consistent but unconstructable policy data into dense routing."""
    # Preserve a valid outer digest while making one routed ambiguity value nonnumeric.
    policy = load_backend_policy_manifest()
    policy[policy_section]["observed_error_envelopes"]["bic"] = None
    invalid_policy = validate_backend_policy_manifest(
        {
            "schema_version": 1,
            "policy_sha256": canonical_json_digest(policy),
            "policy": policy,
        }
    )
    monkeypatch.setattr(
        vmf_fit,
        "load_backend_policy_manifest",
        lambda: invalid_policy,
    )
    rows = _factor_candidate_rows()
    expected = fit_vmf_candidate(
        rows,
        k=2,
        n_init=1,
        max_iter=8,
        seed=43,
        backend="dense_cpu",
    )

    actual = fit_vmf_candidate(
        rows,
        k=2,
        n_init=1,
        max_iter=8,
        seed=43,
        backend=backend,
    )

    assert actual.backend == "dense_cpu"
    assert actual.route_reason is not None
    assert "BackendPolicyManifestError" in actual.route_reason
    assert actual.log_likelihood == expected.log_likelihood
    np.testing.assert_array_equal(actual.labels, expected.labels)
    np.testing.assert_allclose(
        actual.responsibilities,
        expected.responsibilities,
        rtol=2.0e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.weights, expected.weights)
    np.testing.assert_allclose(actual.kappas, expected.kappas, rtol=3.0e-15, atol=0.0)


def test_vmf_resume_rejects_mixed_backend_calibration_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Reject a checkpoint written by a different backend/calibration mixture."""
    # Change only the accepted calibration digest inside an otherwise equal key.
    current = {
        "effect_space": "pre_softcap_logits",
        "source_readout": "final_resid",
        "vmf_backend_fingerprints": vmf_backend_fingerprints(),
    }
    stale = json.loads(json.dumps(current))
    stale["vmf_backend_fingerprints"]["calibration"]["sha256"] = "0" * 64
    path = tmp_path / "vmf_checkpoint.json"
    path.write_text(json.dumps({"fingerprint": stale, "features": [{"stale": True}]}))

    with caplog.at_level(logging.INFO):
        state = vmf_runner._VmfCheckpointState.load(
            enabled=True,
            path=path,
            fingerprint=current,
            expected_feature_ids={1},
            workers=1,
            flush_every_features=1,
        )

    assert state.completed_count == 0
    assert "reason=fingerprint_mismatch" in caplog.text


def test_dense_fit_trace_records_decisions_without_changing_scientific_fit() -> None:
    """Freeze seeded dense decisions and workload counts before factor optimization."""
    # Compare traced and untraced runs, then require every dense decision layer.
    rows = _two_mode_rows()
    expected = fit_vmf_mixture(rows, k=2, n_init=2, max_iter=5, seed=31)
    trace = {}
    actual = fit_vmf_mixture(
        rows,
        k=2,
        n_init=2,
        max_iter=5,
        seed=31,
        trace=trace,
    )

    np.testing.assert_array_equal(actual.labels, expected.labels)
    np.testing.assert_array_equal(actual.responsibilities, expected.responsibilities)
    np.testing.assert_array_equal(actual.centers, expected.centers)
    np.testing.assert_array_equal(actual.weights, expected.weights)
    np.testing.assert_array_equal(actual.kappas, expected.kappas)
    assert actual.log_likelihood == expected.log_likelihood
    assert trace["backend"] == "dense_cpu"
    assert trace["selected_init_index"] in (0, 1)
    assert len(trace["initializations"]) == 2
    assert trace["workload"]["initialization_attempts"] == 2
    assert trace["workload"]["iterations"] > 0
    assert trace["workload"]["kmeans_selections"] == 2
    assert all(item["iterations"] for item in trace["initializations"])
    assert all(
        "posterior" in iteration and "maximization" in iteration
        for item in trace["initializations"]
        for iteration in item["iterations"]
    )


def test_dense_feature_trace_covers_bic_stability_and_reporting_inputs() -> None:
    """Trace every feature-level decision without changing serialized science."""
    # Use the deterministic fit double so this test isolates scorer instrumentation.
    trace = {}
    result = score_vmf_feature(
        _two_mode_rows(),
        _cfg(resample_rounds=2),
        seed=17,
        fit_fn=_directional_fit,
        trace=trace,
    )

    assert trace["fit_status"] == result.fit_status == "fitted"
    assert trace["model_selection"] == result.model_selection
    assert trace["selected_fit"] == result.selected_fit
    assert trace["assignment_stability"] == result.assignment_stability
    assert trace["reporting_inputs"] == result.metrics
    assert trace["bic_selection"]["selected_mode_count"] == 2
    assert trace["workload"] == {
        "candidate_fits_attempted": 2,
        "candidate_fits_finite": 2,
        "candidate_fits_nonfinite": 0,
        "candidate_fits_failed": 0,
        "assignment_refits_requested": 2,
        "assignment_refits_successful": 2,
        "assignment_refits_failed": 0,
        "initialization_attempts": 0,
        "iterations": 0,
        "kmeans_selections": 0,
        "fallback_components": 0,
        "bic_evaluations": 9,
    }


def test_dense_fallback_trace_is_exactly_observational() -> None:
    """Expose the inherited empty-component fallback without changing its arrays."""
    # Give one component all mass and the other none to activate exactly one fallback.
    from fega.core.vmf.utils._spherecluster._vmfm import _maximization

    rows = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    posterior = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    expected = _maximization(rows, posterior)
    trace = {}
    actual = _maximization(rows, posterior, trace=trace)

    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert [item["fallback"] for item in trace["components"]] == [False, True]
    assert trace["components"][1]["concentration"] == 0.0


def test_dense_weighted_component_sums_are_bit_exact_and_vectorized() -> None:
    """Dense M-step accumulation must preserve bytes while removing row dispatch."""
    # Use production n/M and a bounded ambient width with the same memory access pattern.
    from fega.core.vmf.utils._spherecluster._vmfm import (
        _dense_weighted_component_sums,
    )

    rng = np.random.default_rng(20260718)
    rows = rng.standard_normal((64, 8192), dtype=np.float64)
    posterior = rng.random((4, 64), dtype=np.float64)
    posterior /= posterior.sum(axis=0, keepdims=True)

    def legacy() -> np.ndarray:
        """Reproduce the inherited per-row dense accumulation exactly."""
        # Retain the old loop only inside this regression/performance reference.
        centers = np.zeros((posterior.shape[0], rows.shape[1]), dtype=np.float64)
        for component in range(posterior.shape[0]):
            scaled = rows.copy()
            for example in range(rows.shape[0]):
                scaled[example, :] *= posterior[component, example]
            centers[component, :] = scaled.sum(axis=0)
        return centers

    expected = legacy()
    actual = _dense_weighted_component_sums(rows, posterior)
    np.testing.assert_array_equal(actual, expected)

    legacy_times = []
    vectorized_times = []
    for _ in range(3):
        started = time.perf_counter()
        legacy()
        legacy_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        _dense_weighted_component_sums(rows, posterior)
        vectorized_times.append(time.perf_counter() - started)
    assert min(vectorized_times) < 0.6 * min(legacy_times)


def test_vmf_normalizer_uses_exact_kappa_zero_limit() -> None:
    """Lock the exact uniform-circle normalizer at concentration zero."""
    # Use a committed value whose binary float equals the analytic zero limit.
    assert log_vmf_normalizer(2, 0.0) == -1.8378770664093453


@pytest.mark.parametrize(
    ("dim", "kappa", "expected", "absolute_tolerance"),
    [
        (3, 1.0, -2.6924636085404864, 2.0e-14),
        (3, 1.0e10, -9_999_999_978.812025, 2.0e-6),
        (256_000, 1.0, 1_230_721.4699999804, 1.0e-8),
        (256_000, 10_000.0, 1_230_526.3062100879, 1.0e-8),
        (256_000, 507_123.004573869, 955_165.6871137724, 1.0e-8),
        (256_000, 1.0e10, -9_997_287_949.120268, 2.0e-6),
        (
            256_000,
            1_279_993_472_886_641.5,
            -1_279_993_468_669_345.5,
            0.25,
        ),
    ],
)
def test_vmf_normalizer_matches_committed_high_precision_constants(
    dim: int,
    kappa: float,
    expected: float,
    absolute_tolerance: float,
) -> None:
    """Compare representative small and actual-width dimensions to references."""
    # Keep reference values literal so the implementation cannot generate its oracle.
    assert log_vmf_normalizer(dim, kappa) == pytest.approx(
        expected, abs=absolute_tolerance, rel=0.0
    )


@pytest.mark.parametrize(
    ("dim", "kappa", "expected"),
    [
        (2, 0.0, -1.8378770664093455),
        (3, 1.0, -1.6924636085404864),
        (3, 1.0e10, 21.18797386353111),
        (50_257, 1.0, 200_698.19434291698),
        (50_257, 1.0e10, 532_411.4388123726),
        (256_000, 0.0, 1_230_721.4700019336),
        (256_000, 1.0, 1_230_722.4699999804),
        (256_000, 10_000.0, 1_240_526.306210088),
        (256_000, 507_123.004573869, 1_462_288.6916876414),
        (256_000, 1.0e10, 2_712_050.8797322506),
        (256_000, 1_279_993_472_886_641.5, 4_217_296.077439653),
    ],
)
def test_vmf_shifted_normalizer_meets_absolute_production_contract(
    dim: int, kappa: float, expected: float
) -> None:
    """Lock the absolute error used to derive likelihood and BIC allowances."""
    # Compare the exact production likelihood representation with zero relative slack.
    actual = log_vmf_normalizer_plus_kappa(dim, kappa)
    assert abs(actual - expected) <= 1.0e-8


def test_vmf_normalizer_hyp0f1_overlaps_ive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the standard hyp0f1 identity and compare with a finite ive result."""
    # Record the library path, then force ive to report the documented zero failure.
    expected = log_vmf_normalizer(3, 1.0)
    monkeypatch.setattr(_vmf_numerics, "ive", lambda nu, kappa: 0.0)

    assert log_vmf_normalizer(3, 1.0) == pytest.approx(
        expected, abs=1.0e-12, rel=0.0
    )


def test_vmf_shifted_normalizer_preserves_extreme_density_precision() -> None:
    """Keep the saturated-kappa likelihood finite without subtracting two 1e15 terms."""
    # Compare the cancellation-safe shifted value to an independent literal reference.
    kappa = 1_279_993_472_886_641.5
    assert log_vmf_normalizer_plus_kappa(256_000, kappa) == pytest.approx(
        4_217_296.077439653,
        abs=1.0e-8,
        rel=0.0,
    )
    observation = np.zeros((1, 256_000), dtype=np.float64)
    observation[0, 0] = 1.0
    likelihood = vmf_mixture_log_likelihood(
        observation,
        observation.copy(),
        np.asarray([1.0], dtype=np.float64),
        np.asarray([kappa], dtype=np.float64),
    )
    assert likelihood == pytest.approx(
        4_217_296.077439653, abs=1.0e-8, rel=0.0
    )


@pytest.mark.parametrize(
    ("dim", "kappa", "message"),
    [
        (1, 1.0, "at least two"),
        (256_001, 1.0, "exceeds validated coverage"),
        (256_000, 1.3e15, "exceeds validated coverage"),
    ],
)
def test_vmf_normalizer_fails_outside_validated_dense_domain(
    dim: int, kappa: float, message: str
) -> None:
    """Prevent the corrected dense path from claiming authority beyond its corpus."""
    # Exercise each declared dimension and concentration boundary explicitly.
    with pytest.raises(ValueError, match=message):
        log_vmf_normalizer(dim, kappa)


def test_vmf_normalizer_fails_closed_when_library_paths_are_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a normalizer when every upstream library path is non-finite."""
    # Force the scaled Bessel and exact quadrature paths to fail their contracts.
    monkeypatch.setattr(_vmf_numerics, "ive", lambda nu, kappa: 0.0)
    monkeypatch.setattr(
        _vmf_numerics,
        "quad",
        lambda *args, **kwargs: (math.nan, math.nan),
    )

    with pytest.raises(FloatingPointError, match="error contract"):
        log_vmf_normalizer(3, 1.0)


def test_movmf_attempts_full_budget_and_selects_largest_finite_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select by finite full likelihood despite failures and lower winning inertia."""
    # Encode initialization identity in kappa so the likelihood oracle controls ranking.
    import fega.core.vmf.utils._spherecluster._vmfm as vmfm

    seeds = np.random.RandomState(17).randint(np.iinfo(np.int32).max, size=4)
    seed_to_index = {int(seed): index for index, seed in enumerate(seeds)}
    attempted: list[int] = []

    def fake_movmf(X, n_clusters, **kwargs):
        """Return deterministic candidates while making initialization zero raise."""
        # Record the per-initialization seed passed by the scheduling wrapper.
        del X, n_clusters
        index = seed_to_index[int(kwargs["random_state"])]
        attempted.append(index)
        if index == 0:
            raise FloatingPointError("deliberate initialization failure")
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray([float(index)], dtype=np.float64),
            np.asarray([[1.0, 1.0]], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            0.0 if index == 2 else 10.0,
        )

    def fake_likelihood(X, centers, weights, kappas):
        """Make init one non-finite and init three the finite likelihood winner."""
        # Rank solely by encoded initialization identity, independent of inertia.
        del X, centers, weights
        index = int(kappas[0])
        return {1: float("nan"), 2: 5.0, 3: 10.0}[index]

    monkeypatch.setattr(vmfm, "_movMF", fake_movmf)
    monkeypatch.setattr(vmfm, "vmf_mixture_log_likelihood", fake_likelihood)
    result = vmfm.movMF(
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        1,
        n_init=4,
        n_jobs=1,
        random_state=17,
    )

    assert attempted == [0, 1, 2, 3]
    assert result[2] == 10.0
    assert result[4] == pytest.approx([3.0])


def test_movmf_propagates_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unexpected implementation failures visible through public movMF."""
    # Raise outside the numerical failure contract of one initialization attempt.
    import fega.core.vmf.utils._spherecluster._vmfm as vmfm

    def fail_movmf(X, n_clusters, **kwargs):
        """Raise the unexpected implementation failure under test."""
        # Ignore fit inputs because propagation is the only behavior under test.
        del X, n_clusters, kwargs
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(vmfm, "_movMF", fail_movmf)

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        vmfm.movMF(
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            1,
            n_init=1,
            n_jobs=1,
            random_state=17,
        )


def test_movmf_all_initialization_failures_retain_dense_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every attempted dense initialization visible when no fit survives."""
    # Fail inside the dense initialization wrapper after deterministic seeds exist.
    import fega.core.vmf.utils._spherecluster._vmfm as vmfm

    def fail_movmf(X, n_clusters, **kwargs):
        """Raise one declared numerical failure for each actual initialization."""
        # Ignore fit values because trace preservation is the behavior under test.
        del X, n_clusters, kwargs
        raise FloatingPointError("deliberate dense initialization failure")

    monkeypatch.setattr(vmfm, "_movMF", fail_movmf)
    trace = {}
    with pytest.raises(FloatingPointError, match="No initialization produced"):
        vmfm.movMF(
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            1,
            n_init=2,
            n_jobs=1,
            random_state=17,
            trace=trace,
        )

    assert trace["status"] == "failed"
    assert trace["selected_init_index"] is None
    assert trace["selected_log_likelihood"] is None
    assert trace["workload"]["initialization_attempts"] == 2
    assert [item["status"] for item in trace["initializations"]] == [
        "failed",
        "failed",
    ]
    assert all(
        "deliberate dense initialization failure" in item["error"]
        for item in trace["initializations"]
    )


def test_movmf_likelihood_tie_selects_smaller_initialization_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve equal finite full likelihoods by deterministic initialization order."""
    # Encode initialization order in kappa and give every candidate the same score.
    import fega.core.vmf.utils._spherecluster._vmfm as vmfm

    seeds = np.random.RandomState(23).randint(np.iinfo(np.int32).max, size=3)
    seed_to_index = {int(seed): index for index, seed in enumerate(seeds)}

    def fake_movmf(X, n_clusters, **kwargs):
        """Return a successful candidate carrying its deterministic init index."""
        # Map the prederived seed back to its stable initialization position.
        del X, n_clusters
        index = seed_to_index[int(kwargs["random_state"])]
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray([float(index)], dtype=np.float64),
            np.asarray([[1.0, 1.0]], dtype=np.float64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            float(index),
        )

    def tied_likelihood(X, centers, weights, kappas):
        """Return the same finite full likelihood for every initialization."""
        # Ignore candidate values so initialization index is the only tie-break key.
        del X, centers, weights, kappas
        return 7.0

    monkeypatch.setattr(vmfm, "_movMF", fake_movmf)
    monkeypatch.setattr(vmfm, "vmf_mixture_log_likelihood", tied_likelihood)
    result = vmfm.movMF(
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        1,
        n_init=3,
        n_jobs=1,
        random_state=23,
    )

    assert result[2] == 0.0
    assert result[4] == pytest.approx([0.0])


def test_fitted_vmf_mixture_log_likelihood_matches_training_value() -> None:
    """Score the fitted training rows through the estimator likelihood API."""
    # Fit one finite component and compare the public method with its stored score.
    from fega.core.vmf.utils._spherecluster._vmfm import VonMisesFisherMixture

    x = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    model = VonMisesFisherMixture(
        n_clusters=1,
        n_init=1,
        max_iter=5,
        random_state=0,
    ).fit(x)

    assert model.log_likelihood(x) == pytest.approx(model.log_likelihood_)


def test_local_copied_spherecluster_accepts_large_seed() -> None:
    rows = _two_mode_rows()
    large_induction_seed = 42 + 64637 * 104729 + 2 + 7

    first = fit_vmf_mixture(rows, k=2, n_init=1, max_iter=5, seed=large_induction_seed)
    second = fit_vmf_mixture(rows, k=2, n_init=1, max_iter=5, seed=large_induction_seed)

    assert first.k == 2
    assert first.labels.shape == (20,)
    assert np.array_equal(first.labels, second.labels)
    assert np.isfinite(first.log_likelihood)


def test_local_copied_spherecluster_handles_empty_component_without_warning() -> None:
    from fega.core.vmf.utils._spherecluster._vmfm import _expectation, _maximization

    x = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    posterior = np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        centers, weights, concentrations = _maximization(x, posterior)

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert np.all(np.isfinite(centers))
    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(concentrations))
    assert np.allclose(weights, [1.0, 0.0])
    assert np.isclose(np.linalg.norm(centers[1]), 1.0)
    assert concentrations[1] == 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        next_posterior = _expectation(x, centers, weights, concentrations)

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert np.all(np.isfinite(next_posterior))
    assert np.allclose(next_posterior[1], 0.0)


def test_local_copied_spherecluster_rejects_invalid_force_weights() -> None:
    from fega.core.vmf.utils._spherecluster._vmfm import VonMisesFisherMixture

    model = VonMisesFisherMixture(n_clusters=2, force_weights=[float("nan"), 1.0])
    with pytest.raises(ValueError, match="force_weights must be finite"):
        model._check_force_weights()

    model = VonMisesFisherMixture(n_clusters=2, force_weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="force_weights must be non-negative"):
        model._check_force_weights()

    model = VonMisesFisherMixture(n_clusters=2, force_weights=[2.0, 2.0])
    model._check_force_weights()
    assert np.allclose(model.force_weights, [0.5, 0.5])


def _write_reference_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval_config": {
                    "model_name": "gpt2",
                    "llm_dtype": "float32",
                    "entity_attribute_selection": {"city": ["Country"]},
                }
            }
        )
    )


def _write_config(
    tmp_path: Path,
    *,
    workers: int | None = None,
    max_vocab_buffers: int | None = None,
    resume: bool | None = None,
    checkpoint_flush_features: int | None = None,
) -> FEGAPipelineConfig:
    reference_json = tmp_path / "refs" / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"reference_json: {reference_json}",
                f"output_root: {tmp_path / 'out'}",
                "device: cpu",
                "entity_attribute_selection:",
                "  city: ['Country']",
                "phases:",
                "  geometry_metrics:",
                "    effect_space: final_resid",
                "  vmf:",
                "    enabled: true",
                *(["    workers: " + str(workers)] if workers is not None else []),
                *(
                    ["    max_vocab_buffers: " + str(max_vocab_buffers)]
                    if max_vocab_buffers is not None
                    else []
                ),
                *(
                    ["    resume: " + ("true" if resume else "false")]
                    if resume is not None
                    else []
                ),
                *(
                    ["    checkpoint_flush_features: " + str(checkpoint_flush_features)]
                    if checkpoint_flush_features is not None
                    else []
                ),
                "    bic_tolerance: 1.0e-9",
                "    resample_rounds: 2",
                "    resample_fraction: 0.8",
                "    n_init: 1",
                "    max_iter: 20",
            ]
        )
        + "\n"
    )
    return FEGAPipelineConfig.from_file(cfg_path)


def _write_effect_artifacts(
    config: FEGAPipelineConfig,
    rows: torch.Tensor,
    *,
    unembedding: torch.Tensor | None = None,
) -> None:
    """Write one canonical final-residual fixture with optional explicit W_U."""
    # Delegate to the multi-feature writer so provenance stays identical.
    _write_multi_feature_effect_artifacts(
        config, {1: rows}, unembedding=unembedding
    )


def _write_multi_feature_effect_artifacts(
    config: FEGAPipelineConfig,
    features: dict[int, torch.Tensor],
    *,
    unembedding: torch.Tensor | None = None,
) -> None:
    """Write slice-1-compatible rows, identity, mask, Gram, and readout metadata."""
    # Assemble one ordered shard and bind it to the exact fixture unembedding.
    artifact_dir = effect_tensors_manifest_path(config, "final_resid").parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shard_path = artifact_dir / "effect_tensors_00000.pt"
    rows_by_feature = {
        feature_id: rows.to(dtype=torch.float32)
        for feature_id, rows in sorted(features.items())
    }
    combined_rows = torch.cat(list(rows_by_feature.values()), dim=0)
    hidden_width = int(combined_rows.shape[1])
    unembedding = (
        torch.eye(hidden_width, dtype=torch.float32)
        if unembedding is None
        else unembedding.to(dtype=torch.float32)
    )
    if unembedding.ndim != 2 or unembedding.shape[1] != hidden_width:
        raise ValueError("test unembedding must have shape [vocab, hidden_width]")
    setattr(config, "_test_unembedding", unembedding)
    row_count = int(combined_rows.shape[0])
    per_feature = {}
    candidate_identity: list[list[dict[str, object]]] = []
    retained_masks: list[list[bool]] = []
    attribute_labels: list[str] = []
    pair_roles: list[str] = []
    pair_indices: list[int] = []
    row_start = 0
    for feature_id, rows in rows_by_feature.items():
        row_end = row_start + int(rows.shape[0])
        identities = [
            {
                "attribute_label": "Country",
                "pair_role": "cause_base",
                "pair_index": index,
            }
            for index in range(int(rows.shape[0]))
        ]
        mask = [True] * len(identities)
        candidate_identity.append(identities)
        retained_masks.append(mask)
        attribute_labels.extend("Country" for _ in identities)
        pair_roles.extend("cause_base" for _ in identities)
        pair_indices.extend(range(len(identities)))
        per_feature[str(feature_id)] = {
            "feature_id": feature_id,
            "usable_effects": int(rows.shape[0]),
            "tensor_shard": "effect_tensors_00000.pt",
            "row_start": row_start,
            "row_end": row_end,
            "candidate_identity": identities,
            "retained_mask": mask,
        }
        row_start = row_end
    torch.save(
        {
            "feature_ids": torch.tensor(sorted(rows_by_feature), dtype=torch.long),
            "direction": combined_rows,
            "attribute_labels": attribute_labels,
            "pair_roles": pair_roles,
            "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
            "candidate_identity": candidate_identity,
            "retained_mask": retained_masks,
        },
        shard_path,
    )
    gram_path = gram_cache_tensor_path(config)
    gram_meta_path = gram_cache_meta_path(config)
    gram_path.parent.mkdir(parents=True, exist_ok=True)
    gram = unembedding.T @ unembedding
    torch.save(gram, gram_path)
    gram_metadata = {
        "checkpoint_identity": "test-model",
        "readout_name": "final_resid",
        "hidden_width": hidden_width,
        "gram_dtype": "float32",
        "construction_recipe": GRAM_CONSTRUCTION_RECIPE,
        "unembedding_fingerprint": unembedding_fingerprint(unembedding),
        "unembedding_dtype": str(unembedding.dtype),
        "unembedding_shape": list(unembedding.shape),
        "gram_shape": [hidden_width, hidden_width],
        "gram_sha256": gram_fingerprint(gram),
    }
    gram_meta_path.write_text(json.dumps(gram_metadata))
    effect_summary_path(config, "final_resid").write_text(
        json.dumps(
            {
                "summary": {
                    "readout_name": "final_resid",
                    "features_total": len(rows_by_feature),
                    "features_with_effects": len(rows_by_feature),
                    "features_skipped": 0,
                    "total_effect_rows": row_count,
                    "shard_count": 1,
                },
                "gram_metadata": gram_metadata,
                "per_feature": per_feature,
                "skipped_features": [],
            }
        )
    )
    effect_tensors_manifest_path(config, "final_resid").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "readout_name": "final_resid",
                "effect_space": "final_resid",
                "metric_space": "residual_gram",
                "vector_size": hidden_width,
                "dtype": "float32",
                "inputs": {
                    "gram_path": str(gram_path),
                    "gram_meta_path": str(gram_meta_path),
                },
                "gram_metadata": gram_metadata,
                "outputs": {
                    "effect_summary_path": str(
                        effect_summary_path(config, "final_resid")
                    )
                },
                "counts": {
                    "features_total": len(rows_by_feature),
                    "features_with_effects": len(rows_by_feature),
                    "features_skipped": 0,
                    "total_effect_rows": row_count,
                    "shard_count": 1,
                },
                "shards": [
                    {
                        "shard": 0,
                        "path": str(shard_path),
                        "rows": row_count,
                        "feature_ids": sorted(rows_by_feature),
                        "row_start": 0,
                        "row_end": row_count,
                    }
                ],
            }
        )
    )


class _FakeResources:
    """Provide the exact test output embedding without loading a real model."""

    def __init__(self, unembedding: torch.Tensor) -> None:
        """Create an in-memory resource cache bound to one canonical W_U."""
        # Mirror the small subset of ModelResources used by the vMF input path.
        self._json_cache: dict[str, object] = {}
        self._compute_effect_gram_cache: dict[str, torch.Tensor] = {}
        output_embeddings = SimpleNamespace(weight=unembedding)
        self.model = SimpleNamespace(
            config=SimpleNamespace(name_or_path="test-model"),
            get_output_embeddings=lambda: output_embeddings,
        )

    def get_model_and_sae(self):
        """Return the test model in the production resource tuple shape."""
        # Tokenizer and SAE are unused by downstream materialization.
        return self.model, None, None

    def get_cached_json(self, path: Path):
        """Return a cached JSON payload when one has already been loaded."""
        # Key by the same string convention as ModelResources.
        return self._json_cache.get(str(path))

    def cache_json(self, path: Path, payload: object) -> None:
        """Cache one JSON payload for repeated downstream reads."""
        # Preserve exact objects to match production cache semantics.
        self._json_cache[str(path)] = payload


def run_vmf(config: FEGAPipelineConfig) -> None:
    """Run vMF with the exact unembedding used by the synthetic Gram fixture."""
    # Require the fixture writer to establish canonical readout provenance first.
    unembedding = getattr(config, "_test_unembedding")
    vmf_runner.run_vmf(config, _FakeResources(unembedding))


def _write_geometry_metrics_scores(
    config: FEGAPipelineConfig, effect_space: str, *, r2: float = 0.1
) -> str:
    return _write_geometry_metrics_scores_for_features(config, effect_space, {1: r2})


def _write_geometry_metrics_scores_for_features(
    config: FEGAPipelineConfig, effect_space: str, feature_r2: dict[int, float]
) -> str:
    path = geometry_metrics_scores_path(config, effect_space)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "canonical_source_fingerprint": canonical_source_fingerprint(
            json.loads(
                effect_tensors_manifest_path(config, "final_resid").read_text()
            ),
            json.loads(effect_summary_path(config, "final_resid").read_text()),
        ),
        "summary": {"effect_space": effect_space},
        "per_feature": {
            str(feature_id): {"feature_id": feature_id, "r2": r2}
            for feature_id, r2 in sorted(feature_r2.items())
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text)
    return text


def _fake_vmf_feature_record(
    feature_id: int, *, bic_tolerance: float = 1.0e-9
) -> dict[str, object]:
    """Return one complete fitted feature state for checkpoint runner fixtures."""
    # Keep orthogonal state dimensions explicit so checkpoint validation is exercised.
    feature_seed = feature_fit_seed(42, feature_id)
    candidates = [
        {
            "mode_count": mode_count,
            "status": "finite",
            "seed": derived_vmf_seed(
                feature_seed, mode_count, -1, "candidate_fit"
            ),
            "log_likelihood": -1.0,
            "bic": float(mode_count + 1),
        }
        for mode_count in (1, 2, 3, 4)
    ]
    return {
        "feature_id": feature_id,
        "n_valid": 20,
        "fit_status": "fitted",
        "model_selection": {
            "selected_mode_count": 1,
            "bic_tolerance": bic_tolerance,
            "candidates": candidates,
            "attempted_count": 4,
            "finite_count": 4,
            "nonfinite_count": 0,
            "failed_count": 0,
        },
        "selected_fit": {
            "weights": [1.0],
            "kappas": [5.0 + float(feature_id)],
            "hard_mode_counts": [20],
            "hard_assignments": [0] * 20,
        },
        "assignment_stability": {
            "status": "not_applicable",
            "value": None,
            "requested_count": 0,
            "successful_count": 0,
            "failed_count": 0,
            "replicates": [],
        },
        "metrics": {
            "selected_mode_count": 1,
            "delta_mix": float(feature_id) / 10.0,
            "mode_mass_min": 1.0,
            "min_mode_c_ray": 0.9,
            "mode_kappa_min": 5.0 + float(feature_id),
        },
    }


def _patch_runner_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_candidate(
        rows: torch.Tensor,
        *,
        k: int,
        n_init: int,
        max_iter: int,
        seed: int,
        trace: dict[str, object] | None = None,
        backend: str = "dense_cpu",
        gpu_device: str = "cuda:0",
    ) -> VmfCandidate:
        """Return deterministic center-free state for runner-only tests."""
        # Mirror production routing inputs while keeping numerical runtime isolated.
        del gpu_device
        public = _directional_fit(
            rows, k, _cfg(n_init=n_init, max_iter=max_iter), seed
        )
        if trace is not None:
            trace.update({"backend": backend})
        return VmfCandidate(
            k=k,
            labels=public.labels,
            responsibilities=public.responsibilities,
            weights=public.weights,
            kappas=np.asarray(public.kappas, dtype=np.float64),
            log_likelihood=public.log_likelihood,
            center_coefficients=None,
            ambient_dim=int(rows.shape[1]),
            source_sha256="runner-test-source",
            backend=backend,
            route_reason="runner_test_double",
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            trace={},
        )

    def fake_finalize(candidate: VmfCandidate, rows: torch.Tensor) -> VmfFit:
        """Create the selected public fixture after runner candidate selection."""
        # Preserve existing artifact expectations at the compatibility boundary.
        return _directional_fit(
            rows,
            candidate.k,
            _cfg(n_init=candidate.n_init, max_iter=candidate.max_iter),
            candidate.seed,
        )

    monkeypatch.setattr(vmf_metrics, "fit_vmf_candidate", fake_candidate)
    monkeypatch.setattr(vmf_metrics, "finalize_vmf_candidate", fake_finalize)


def test_vmf_materializes_linear_coordinates_from_final_resid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require runner scoring to receive exact ephemeral coordinates from W_U."""
    # Use a nontrivial tall readout whose Gram preserves the canonical row norms.
    config = _write_config(tmp_path)
    unembedding = torch.tensor(
        [[2.0**-0.5, 0.0], [2.0**-0.5, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    _write_effect_artifacts(config, _two_mode_rows(), unembedding=unembedding)
    _write_geometry_metrics_scores(config, "final_resid", r2=0.1)
    _patch_runner_fit(monkeypatch)
    captured: list[torch.Tensor] = []
    original_score = vmf_runner.score_vmf_feature

    def capture_score(rows, *args, **kwargs):
        captured.append(rows.clone())
        return original_score(rows, *args, **kwargs)

    monkeypatch.setattr(vmf_runner, "score_vmf_feature", capture_score)

    run_vmf(config)

    payload = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())
    geometry_metrics_payload = json.loads(
        geometry_metrics_scores_path(config, "final_resid").read_text()
    )
    assert payload["effect_space"] == "pre_softcap_logits"
    assert payload["source_readout"] == "final_resid"
    assert payload["geometry_metrics_effect_space"] == "final_resid"
    expected = _two_mode_rows() @ unembedding.T
    expected.div_(torch.linalg.vector_norm(expected, dim=1, keepdim=True))
    assert torch.equal(captured[0], expected)
    source_fingerprint = payload["canonical_source_fingerprint"]
    assert geometry_metrics_payload["canonical_source_fingerprint"] == source_fingerprint
    assert payload["fingerprint"]["canonical_source"] == source_fingerprint
    assert source_fingerprint["components"]["gram_readout_metadata"][
        "unembedding_fingerprint"
    ] == (
        unembedding_fingerprint(unembedding)
    )
    assert payload["fingerprint"]["materialization_policy"] == {
        "formula": "final_resid_direction@canonical_unembedding.T",
        "output_dtype": "float32",
        "vocab_chunk_size": 16_384,
        "normalization": "exact_l2_no_epsilon",
        "max_vocab_buffers": 1,
    }


def test_vmf_artifact_schema_has_exact_public_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path)
    _write_effect_artifacts(config, _two_mode_rows())
    _write_geometry_metrics_scores(config, "final_resid", r2=0.1)
    _patch_runner_fit(monkeypatch)

    run_vmf(config)

    payload = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())
    assert set(payload) == {
        "phase",
        "schema_version",
        "effect_space",
        "source_readout",
        "geometry_metrics_effect_space",
        "canonical_source_fingerprint",
        "fingerprint",
        "features",
    }
    assert payload["phase"] == "vmf"
    assert payload["schema_version"] == 1
    assert payload["fingerprint"]["schema_version"] == 3
    assert payload["fingerprint"]["feature_ids"] == [1]
    assert payload["fingerprint"]["candidate_mode_counts"] == [1, 2, 3, 4]
    assert payload["fingerprint"]["assignment_metric"]["identity"] == (
        "sklearn.metrics.adjusted_rand_score"
    )
    feature = payload["features"][0]
    assert set(feature) == {
        "feature_id",
        "n_valid",
        "fit_status",
        "model_selection",
        "selected_fit",
        "assignment_stability",
        "metrics",
    }
    assert tuple(feature["metrics"]) == PUBLIC_METRIC_KEYS
    assert "status" not in feature
    assert "provenance" not in feature
    assert "vmf_fit_k_raw" not in feature["metrics"]


def test_vmf_rejects_geometry_metrics_source_mismatch_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail before feature scoring when geometry_metrics was built from another source."""
    # Corrupt only the upstream source identity while leaving all score rows intact.
    config = _write_config(tmp_path)
    _write_effect_artifacts(config, _two_mode_rows())
    _write_geometry_metrics_scores(config, "final_resid", r2=0.1)
    path = geometry_metrics_scores_path(config, "final_resid")
    payload = json.loads(path.read_text())
    payload["canonical_source_fingerprint"]["digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        vmf_runner,
        "score_vmf_feature",
        lambda *args, **kwargs: pytest.fail("scoring must not start"),
    )

    with pytest.raises(ValueError, match="canonical source fingerprint mismatch"):
        run_vmf(config)


def test_vmf_resume_false_leaves_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, resume=False)
    _write_effect_artifacts(config, _two_mode_rows())
    _write_geometry_metrics_scores(config, "final_resid", r2=0.1)
    _patch_runner_fit(monkeypatch)

    run_vmf(config)

    assert vmf_scores_path(config, "pre_softcap_logits").exists()
    assert not vmf_checkpoint_path(config, "pre_softcap_logits").exists()


def test_vmf_resume_skips_checkpointed_features_and_matches_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path / "resume", resume=True, checkpoint_flush_features=2
    )
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
            3: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(
        config, "final_resid", {1: 0.1, 2: 0.1, 3: 0.1}
    )

    first_run_calls: list[int] = []

    def fail_after_first_checkpoint_batch(block, cfg, seed, materializer):
        """Return two complete states, then simulate interruption before the third."""
        # Ignore scoring inputs because this fixture targets checkpoint batching only.
        del cfg, seed, materializer
        first_run_calls.append(block.feature_id)
        if len(first_run_calls) > 2:
            raise RuntimeError("simulated cancellation")
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(
        vmf_runner, "_score_feature_block", fail_after_first_checkpoint_batch
    )

    with pytest.raises(RuntimeError, match="simulated cancellation"):
        run_vmf(config)

    checkpoint_path = vmf_checkpoint_path(config, "pre_softcap_logits")
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["effect_space"] == "pre_softcap_logits"
    assert checkpoint["source_readout"] == "final_resid"
    assert [entry["record"]["feature_id"] for entry in checkpoint["features"]] == [1, 2]

    resumed_calls: list[int] = []

    def score_missing_feature(block, cfg, seed, materializer):
        """Record and return the sole feature absent from the resumed checkpoint."""
        # Ignore scoring inputs while preserving the resumed feature identity.
        del cfg, seed, materializer
        resumed_calls.append(block.feature_id)
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_missing_feature)
    config.phases.vmf.checkpoint_flush_features = 1
    config.phases.vmf.max_vocab_buffers += 1
    run_vmf(config)
    resumed = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())

    assert resumed_calls == [3]
    assert [record["feature_id"] for record in resumed["features"]] == [1, 2, 3]
    assert not checkpoint_path.exists()

    clean_config = _write_config(tmp_path / "clean", resume=True)
    _write_multi_feature_effect_artifacts(
        clean_config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
            3: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(
        clean_config, "final_resid", {1: 0.1, 2: 0.1, 3: 0.1}
    )
    clean_calls: list[int] = []

    def score_clean_feature(block, cfg, seed, materializer):
        """Record every feature in the clean-run checkpoint comparison."""
        # Ignore scoring inputs while emitting the same complete fixture state.
        del cfg, seed, materializer
        clean_calls.append(block.feature_id)
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_clean_feature)
    run_vmf(clean_config)
    clean = json.loads(
        vmf_scores_path(clean_config, "pre_softcap_logits").read_text()
    )

    assert clean_calls == [1, 2, 3]
    assert resumed["features"] == clean["features"]
    assert {
        key: resumed[key]
        for key in ("phase", "effect_space", "geometry_metrics_effect_space")
    } == {
        key: clean[key]
        for key in ("phase", "effect_space", "geometry_metrics_effect_space")
    }
    resumed_policy = dict(resumed["fingerprint"]["materialization_policy"])
    clean_policy = dict(clean["fingerprint"]["materialization_policy"])
    resumed_policy.pop("max_vocab_buffers")
    clean_policy.pop("max_vocab_buffers")
    assert resumed_policy == clean_policy
    resumed_source = resumed["fingerprint"]["canonical_source"]
    clean_source = clean["fingerprint"]["canonical_source"]
    expected_sources = (
        canonical_source_fingerprint(
            json.loads(
                effect_tensors_manifest_path(config, "final_resid").read_text()
            ),
            json.loads(effect_summary_path(config, "final_resid").read_text()),
        ),
        canonical_source_fingerprint(
            json.loads(
                effect_tensors_manifest_path(clean_config, "final_resid").read_text()
            ),
            json.loads(effect_summary_path(clean_config, "final_resid").read_text()),
        ),
    )
    for payload, source, expected_source in zip(
        (resumed, clean), (resumed_source, clean_source), expected_sources, strict=True
    ):
        assert source == payload["canonical_source_fingerprint"] == expected_source
        assert set(source) == {"schema_version", "algorithm", "digest", "components"}
        assert source["schema_version"] == 2
        assert source["algorithm"] == "sha256"
        assert set(source["components"]) == {
            "manifest_sha256",
            "summary_sha256",
            "ordered_retained_identity_sha256",
            "tensor_shards_sha256",
            "gram_readout_metadata",
        }
        for key in (
            "digest",
            "manifest_sha256",
            "summary_sha256",
            "ordered_retained_identity_sha256",
            "tensor_shards_sha256",
        ):
            value = source[key] if key == "digest" else source["components"][key]
            assert isinstance(value, str)
            assert len(value) == 64
            assert all(character in "0123456789abcdef" for character in value)


def test_vmf_resume_does_not_write_partial_checkpoint_batch_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, resume=True, checkpoint_flush_features=2)
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(config, "final_resid", {1: 0.1, 2: 0.1})
    calls: list[int] = []

    def fail_after_one_completed_feature(block, cfg, seed, materializer):
        """Simulate cancellation before a configured checkpoint batch is complete."""
        # Ignore scoring inputs and fail only after one complete feature record.
        del cfg, seed, materializer
        calls.append(block.feature_id)
        if len(calls) > 1:
            raise RuntimeError("simulated cancellation")
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(
        vmf_runner, "_score_feature_block", fail_after_one_completed_feature
    )

    with pytest.raises(RuntimeError, match="simulated cancellation"):
        run_vmf(config)

    assert calls == [1, 2]
    assert not vmf_checkpoint_path(config, "pre_softcap_logits").exists()


def test_vmf_resume_honors_configured_checkpoint_flush_feature_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, resume=True, checkpoint_flush_features=2)
    feature_rows = {feature_id: _two_mode_rows() for feature_id in range(1, 6)}
    _write_multi_feature_effect_artifacts(config, feature_rows)
    _write_geometry_metrics_scores_for_features(
        config, "final_resid", {feature_id: 0.1 for feature_id in feature_rows}
    )
    writes: list[list[int]] = []
    original_write_checkpoint = vmf_runner.write_vmf_checkpoint

    def record_checkpoint_write(path, payload):
        writes.append([entry["record"]["feature_id"] for entry in payload["features"]])
        original_write_checkpoint(path, payload)

    def score_feature(block, cfg, seed, materializer):
        """Return complete deterministic states for checkpoint cadence inspection."""
        # Ignore scoring inputs because write cadence is the isolated behavior.
        del cfg, seed, materializer
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "write_vmf_checkpoint", record_checkpoint_write)
    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_feature)

    run_vmf(config)
    payload = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())

    assert writes == [[1, 2], [1, 2, 3, 4]]
    assert [record["feature_id"] for record in payload["features"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert not vmf_checkpoint_path(config, "pre_softcap_logits").exists()


def test_vmf_resume_ignores_stale_checkpoint_when_fingerprint_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, resume=True, checkpoint_flush_features=1)
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(config, "final_resid", {1: 0.1, 2: 0.1})

    first_run_calls: list[int] = []

    def fail_after_first_feature(block, cfg, seed, materializer):
        """Write one stale feature state before simulating interrupted execution."""
        # Ignore scoring inputs and fail after the first fingerprinted feature.
        del cfg, seed, materializer
        first_run_calls.append(block.feature_id)
        if len(first_run_calls) > 1:
            raise RuntimeError("simulated cancellation")
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", fail_after_first_feature)
    with pytest.raises(RuntimeError, match="simulated cancellation"):
        run_vmf(config)
    assert vmf_checkpoint_path(config, "pre_softcap_logits").exists()

    config.phases.vmf.bic_tolerance = 1.0e-8
    recomputed_calls: list[int] = []

    def score_recomputed_feature(block, cfg, seed, materializer):
        """Record all states recomputed after the scientific fingerprint changes."""
        # Ignore scoring inputs while tracking stale-checkpoint invalidation.
        del seed, materializer
        recomputed_calls.append(block.feature_id)
        return _fake_vmf_feature_record(
            block.feature_id, bic_tolerance=cfg.bic_tolerance
        )

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_recomputed_feature)
    run_vmf(config)

    assert recomputed_calls == [1, 2]
    assert not vmf_checkpoint_path(config, "pre_softcap_logits").exists()


def test_vmf_resume_recomputes_checkpoint_when_tensor_shard_bytes_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a checkpoint when same-path, same-shape compute-effect bytes drift."""
    # Persist one completed feature before mutating only the existing direction tensor.
    config = _write_config(tmp_path, resume=True, checkpoint_flush_features=1)
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(
        config, "final_resid", {1: 0.1, 2: 0.1}
    )
    first_run_calls: list[int] = []

    def stop_after_checkpointed_feature(block, cfg, seed, materializer):
        """Flush one valid entry, then stop before the second feature completes."""
        # Preserve a valid old-input checkpoint whose reuse would now be scientifically stale.
        del cfg, seed, materializer
        first_run_calls.append(block.feature_id)
        if len(first_run_calls) > 1:
            raise RuntimeError("stop after checkpoint flush")
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(
        vmf_runner, "_score_feature_block", stop_after_checkpointed_feature
    )
    with pytest.raises(RuntimeError, match="stop after checkpoint flush"):
        run_vmf(config)

    checkpoint_path = vmf_checkpoint_path(config, "pre_softcap_logits")
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text())
    assert [entry["record"]["feature_id"] for entry in checkpoint["features"]] == [1]
    shard_path = (
        effect_tensors_manifest_path(config, "final_resid").parent
        / "effect_tensors_00000.pt"
    )
    shard = torch.load(shard_path, map_location="cpu")
    shard["direction"][0, 0] += 0.5
    torch.save(shard, shard_path)
    _write_geometry_metrics_scores_for_features(
        config, "final_resid", {1: 0.1, 2: 0.1}
    )

    recomputed_calls: list[int] = []

    def score_recomputed_feature(block, cfg, seed, materializer):
        """Record every feature recomputed after tensor-byte fingerprint drift."""
        # A stale completed entry cannot represent the newly fingerprinted source bytes.
        del cfg, seed, materializer
        recomputed_calls.append(block.feature_id)
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_recomputed_feature)
    run_vmf(config)

    assert recomputed_calls == [1, 2]
    assert not checkpoint_path.exists()


def test_vmf_resume_ignores_corrupt_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, resume=True, max_vocab_buffers=1)
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(config, "final_resid", {1: 0.1, 2: 0.1})
    checkpoint_path = vmf_checkpoint_path(config, "pre_softcap_logits")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{")
    calls: list[int] = []

    def score_feature(block, cfg, seed, materializer):
        """Return complete states after a corrupt checkpoint is rejected."""
        # Ignore scoring inputs while tracking full recomputation.
        del cfg, seed, materializer
        calls.append(block.feature_id)
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_feature)

    run_vmf(config)

    assert calls == [1, 2]
    assert vmf_scores_path(config, "pre_softcap_logits").exists()
    assert not checkpoint_path.exists()


def test_vmf_resume_recomputes_same_fingerprint_semantically_corrupt_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a completed feature whose scientific record changed after checkpointing."""
    # Create one valid checkpoint entry, then interrupt before the second feature.
    config = _write_config(tmp_path, resume=True, checkpoint_flush_features=1)
    _write_multi_feature_effect_artifacts(
        config,
        {
            1: _two_mode_rows(),
            2: _two_mode_rows(),
        },
    )
    _write_geometry_metrics_scores_for_features(config, "final_resid", {1: 0.1, 2: 0.1})
    first_run_calls: list[int] = []

    def stop_after_checkpointed_feature(block, cfg, seed, materializer):
        """Persist the first complete feature before simulating interruption."""
        # Keep the generated checkpoint scientifically valid before mutation.
        del cfg, seed, materializer
        first_run_calls.append(block.feature_id)
        if len(first_run_calls) > 1:
            raise RuntimeError("stop after checkpoint flush")
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(
        vmf_runner, "_score_feature_block", stop_after_checkpointed_feature
    )
    with pytest.raises(RuntimeError, match="stop after checkpoint flush"):
        run_vmf(config)

    checkpoint_path = vmf_checkpoint_path(config, "pre_softcap_logits")
    checkpoint = json.loads(checkpoint_path.read_text())
    fingerprint = checkpoint["fingerprint"]
    feature_entry = checkpoint["features"][0]
    feature_record = feature_entry.get("record", feature_entry)
    feature_record["selected_fit"]["weights"] = [0.25]
    checkpoint_path.write_text(json.dumps(checkpoint))
    assert checkpoint["fingerprint"] == fingerprint

    recomputed_calls: list[int] = []

    def score_recomputed_feature(block, cfg, seed, materializer):
        """Record recomputation after the same-input checkpoint record is rejected."""
        # Return the uncorrupted scientific record for every affected feature.
        del cfg, seed, materializer
        recomputed_calls.append(block.feature_id)
        return _fake_vmf_feature_record(block.feature_id)

    monkeypatch.setattr(vmf_runner, "_score_feature_block", score_recomputed_feature)
    run_vmf(config)

    payload = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())
    assert 1 in recomputed_calls
    assert payload["features"][0]["selected_fit"]["weights"] == [1.0]


def test_vmf_parallel_workers_match_sequential_public_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config = _write_config(tmp_path, resume=True)
    _write_multi_feature_effect_artifacts(
        config,
        {
            2: _two_mode_rows(),
            1: torch.tensor(
                [[1.0, 0.0]] * 8 + [[0.0, 1.0]] * 12,
                dtype=torch.float32,
            ),
        },
    )
    _write_geometry_metrics_scores_for_features(config, "final_resid", {1: 0.1, 2: 0.1})
    _patch_runner_fit(monkeypatch)
    active_materializations = 0
    peak_materializations = 0
    materialization_lock = threading.Lock()
    original_coordinates = vmf_runner._BoundedLinearMaterializer.coordinates

    @contextmanager
    def tracked_coordinates(self, block):
        nonlocal active_materializations, peak_materializations
        with original_coordinates(self, block) as coordinates:
            if coordinates is None:
                yield None
                return
            with materialization_lock:
                active_materializations += 1
                peak_materializations = max(
                    peak_materializations, active_materializations
                )
            try:
                yield coordinates
            finally:
                with materialization_lock:
                    active_materializations -= 1

    monkeypatch.setattr(
        vmf_runner._BoundedLinearMaterializer,
        "coordinates",
        tracked_coordinates,
    )
    caplog.set_level(logging.INFO, logger="fega.core.vmf.runner")

    config.phases.vmf.workers = 1
    run_vmf(config)
    sequential = json.loads(
        vmf_scores_path(config, "pre_softcap_logits").read_text()
    )

    config.phases.vmf.workers = 4
    run_vmf(config)
    parallel = json.loads(vmf_scores_path(config, "pre_softcap_logits").read_text())

    assert {k: v for k, v in parallel.items() if k != "features"} == {
        k: v for k, v in sequential.items() if k != "features"
    }
    assert [record["feature_id"] for record in parallel["features"]] == [1, 2]
    assert [record["feature_id"] for record in sequential["features"]] == [1, 2]
    assert parallel["features"] == sequential["features"]
    assert peak_materializations <= config.phases.vmf.max_vocab_buffers
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "fega.core.vmf.runner"
    ]
    assert any(
        message == "vmf start: features_total=2 workers=4" for message in messages
    )
    assert any(
        message.startswith("vmf progress complete: processed=2/2")
        and "workers=4" in message
        for message in messages
    )


def _assert_public_metrics_close(
    sequential_features: list[dict[str, object]],
    parallel_features: list[dict[str, object]],
) -> None:
    sequential_by_id = {
        int(record["feature_id"]): record["metrics"] for record in sequential_features
    }
    parallel_by_id = {
        int(record["feature_id"]): record["metrics"] for record in parallel_features
    }
    assert parallel_by_id.keys() == sequential_by_id.keys()
    for feature_id, sequential_metrics in sequential_by_id.items():
        assert isinstance(sequential_metrics, dict)
        parallel_metrics = parallel_by_id[feature_id]
        assert isinstance(parallel_metrics, dict)
        assert tuple(sequential_metrics) == PUBLIC_METRIC_KEYS
        assert tuple(parallel_metrics) == PUBLIC_METRIC_KEYS
        for key in PUBLIC_METRIC_KEYS:
            sequential_value = sequential_metrics[key]
            parallel_value = parallel_metrics[key]
            if sequential_value is None or parallel_value is None:
                assert parallel_value is sequential_value
            elif isinstance(sequential_value, int) and isinstance(parallel_value, int):
                assert parallel_value == sequential_value
            else:
                assert parallel_value == pytest.approx(
                    sequential_value, rel=1.0e-9, abs=1.0e-12
                )
