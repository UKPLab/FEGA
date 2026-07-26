from __future__ import annotations

import numpy as np
import pytest

from fega.core.vmf.utils._spherecluster import _vmfm as dense_backend
from fega.core.vmf.utils._spherecluster import _vmfm_factor_em as factor_em
from fega.core.vmf.utils._spherecluster._vmf_numerics import (
    vmf_mixture_log_likelihood,
)
from fega.core.vmf.utils._spherecluster._vmfm import movMF
from fega.core.vmf.utils._spherecluster._vmfm_factor import (
    DenseRerunRequired,
    FactorAmbiguityPolicy,
    FactorIneligible,
    bic_decision_is_ambiguous,
    build_cpu_factor,
    current_cpu_numerical_fingerprint,
    factor_backend_source_fingerprint,
    factor_from_explicit_rows,
    factor_from_hidden_gram,
    fit_factor_movmf,
    fit_factor_or_dense,
)
from fega.core.vmf.utils._spherecluster._vmfm_factor_em import (
    _factor_maximization,
    _guard_comparison,
    _guard_gap,
)


def _promotion_policy() -> FactorAmbiguityPolicy:
    """Return complete zero-observation evidence for controlled test inputs."""
    # Tests exercise analytical bands; calibration supplies nonzero envelopes later.
    return FactorAmbiguityPolicy(
        {
            "bic": 0.0,
            "convergence": 0.0,
            "initialization_likelihood": 0.0,
            "kmeans_distance": 0.0,
            "kmeans_mass": 0.0,
            "kmeans_potential": 0.0,
            "posterior_score": 0.0,
            "resultant": 0.0,
        },
        expected_cpu_fingerprint=current_cpu_numerical_fingerprint(),
        expected_source_fingerprint=factor_backend_source_fingerprint(),
    )


def _two_mode_rows(seed: int = 4, *, n_rows: int = 12, dim: int = 20) -> np.ndarray:
    """Create a well-separated unit-row cloud with ambient dimension above N."""
    # Keep the fixture away from all preregistered ambiguity surfaces.
    rng = np.random.RandomState(seed)
    first = np.zeros(dim, dtype=np.float64)
    second = np.zeros(dim, dtype=np.float64)
    first[0] = 4.0
    second[1] = 4.0
    half = n_rows // 2
    rows = np.vstack(
        [
            rng.normal(first, 0.2, size=(half, dim)),
            rng.normal(second, 0.2, size=(n_rows - half, dim)),
        ]
    )
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_explicit_factor_retains_all_coordinates_without_q() -> None:
    """Require exact Gram preservation even when the row cloud is rank deficient."""
    # Duplicate two rows so truncation would be tempting but remains forbidden.
    rows = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    factor = factor_from_explicit_rows(rows)

    assert factor.z.shape == (4, 4)
    assert not hasattr(factor, "q")
    assert np.max(np.abs(factor.z @ factor.z.T - rows @ rows.T)) <= 1.0e-10
    assert factor.ambient_dim == 6


def test_hidden_gram_uses_unmodified_cholesky_or_explicit_rows() -> None:
    """Forbid repair while retaining the approved hidden-G to explicit-Y cascade."""
    # A full-rank Gram is eligible; a singular one must fall back to exact QR.
    rows = _two_mode_rows(n_rows=8, dim=16)
    hidden, hidden_trace = build_cpu_factor(rows, gram=rows @ rows.T)
    singular_rows = rows.copy()
    singular_rows[1] = singular_rows[0]
    explicit, explicit_trace = build_cpu_factor(
        singular_rows, gram=singular_rows @ singular_rows.T
    )

    assert hidden.backend == "factor_cpu_hidden_g"
    assert hidden_trace["selected_backend"] == "factor_cpu_hidden_g"
    assert explicit.backend == "factor_cpu_explicit_y"
    assert "Cholesky failed without repair" in explicit_trace["hidden_rejection"]
    assert np.max(
        np.abs(explicit.z @ explicit.z.T - singular_rows @ singular_rows.T)
    ) <= 1.0e-10


def test_factor_construction_enforces_the_frozen_cpu_domain() -> None:
    """Permit only float64 clouds inside the calibrated row and ambient bounds."""
    # Exercise both inclusive boundaries and every neighboring excluded domain.
    lower = np.eye(2, dtype=np.float64)
    upper_rows = np.eye(64, dtype=np.float64)
    upper_dim = np.zeros((2, 256_000), dtype=np.float64)
    upper_dim[0, 0] = 1.0
    upper_dim[1, 1] = 1.0
    assert factor_from_explicit_rows(lower).z.shape == (2, 2)
    assert factor_from_explicit_rows(upper_rows).z.shape == (64, 64)
    assert factor_from_explicit_rows(upper_dim).ambient_dim == 256_000

    with pytest.raises(FactorIneligible, match="row count"):
        factor_from_explicit_rows(lower[:1])
    with pytest.raises(FactorIneligible, match="row count"):
        factor_from_explicit_rows(np.eye(65, dtype=np.float64))
    with pytest.raises(FactorIneligible, match="ambient dimension"):
        factor_from_explicit_rows(np.zeros((2, 256_001), dtype=np.float64))
    with pytest.raises(FactorIneligible, match="float64"):
        factor_from_explicit_rows(lower.astype(np.float32))


def test_dtype_and_cpu_fingerprint_mismatch_route_complete_dense_fit() -> None:
    """Keep uncalibrated dtypes and numerical builds outside factor promotion."""
    # Both mismatches must select dense before constructing a factor candidate.
    rows = _two_mode_rows()
    dtype_routed = fit_factor_or_dense(
        rows.astype(np.float32),
        n_clusters=2,
        n_init=1,
        max_iter=30,
        random_state=7,
        policy=_promotion_policy(),
    )
    mismatch = current_cpu_numerical_fingerprint()
    mismatch["numpy"] = "unreviewed"
    fingerprint_routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=1,
        max_iter=30,
        random_state=7,
        policy=FactorAmbiguityPolicy(
            _promotion_policy().observed_errors,
            expected_cpu_fingerprint=mismatch,
            expected_source_fingerprint=factor_backend_source_fingerprint(),
        ),
    )

    assert dtype_routed.backend == "dense_cpu"
    assert "float64" in str(dtype_routed.route_reason)
    assert fingerprint_routed.backend == "dense_cpu"
    assert fingerprint_routed.route_reason == "CPU numerical fingerprint mismatch"


def test_cpu_factor_source_drift_routes_complete_dense_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject optimized decisions after either calibrated factor source changes."""
    # Build the accepted policy first, then simulate source drift at admission.
    from fega.core.vmf.utils._spherecluster import _vmfm_factor as factor_backend

    rows = _two_mode_rows()
    policy = _promotion_policy()
    monkeypatch.setattr(
        factor_backend,
        "factor_backend_source_fingerprint",
        lambda: {
            "factor_source_sha256": "0" * 64,
            "factor_em_source_sha256": "1" * 64,
        },
    )

    routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=1,
        max_iter=30,
        random_state=7,
        policy=policy,
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "factor source fingerprint mismatch"


def test_nonfinite_input_attempts_dense_from_the_original_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route factor-ineligible nonfinite input without sanitizing the dense source."""
    # Replace dense computation only to observe the exact NaN-bearing provider.
    rows = _two_mode_rows()
    rows[0, 0] = np.nan
    captured: dict[str, np.ndarray] = {}

    def dense_sentinel(source: np.ndarray, *_args: object, **_kwargs: object) -> tuple[np.ndarray, ...]:
        """Capture the dense input and return an inert six-field result."""
        # Preserve the source exactly so the assertion detects any repair or normalization.
        captured["source"] = np.array(source, copy=True)
        return tuple(np.empty(0, dtype=np.float64) for _ in range(6))

    monkeypatch.setattr(dense_backend, "movMF", dense_sentinel)
    routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=1,
        max_iter=3,
        random_state=5,
        policy=_promotion_policy(),
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "explicit rows must be finite"
    assert np.array_equal(captured["source"], rows, equal_nan=True)


def test_factor_floating_point_failure_restarts_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert a nonfinite factor-result failure into a complete dense restart."""
    # Force the terminal factor likelihood surface after otherwise valid EM work.
    rows = _two_mode_rows(seed=15)

    def fail_likelihood(*_args: object, **_kwargs: object) -> float:
        """Represent a nonfinite terminal factor result."""
        # Raise the numerical failure caught by the whole-fit routing boundary.
        raise FloatingPointError("forced nonfinite factor result")

    monkeypatch.setattr(factor_em, "factor_mixture_log_likelihood", fail_likelihood)
    routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=1,
        max_iter=30,
        random_state=7,
        policy=_promotion_policy(),
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == (
        "factor_floating_point_error: forced nonfinite factor result"
    )
    assert routed.trace["route"]["original_seed"] == 7
    assert routed.trace["route"]["original_n_init"] == 1


def test_hidden_gram_rejects_each_unrepaired_eligibility_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce finite, symmetry, source, residual, and conditioning boundaries."""
    # Check each gate directly so hidden-G cannot silently repair invalid input.
    rows = _two_mode_rows(n_rows=8, dim=16)
    gram = rows @ rows.T
    exact_cholesky = np.linalg.cholesky(gram)
    nonfinite = gram.copy()
    nonfinite[0, 0] = np.nan
    asymmetric = gram.copy()
    asymmetric[0, 1] += 1.0e-3
    mismatched = gram.copy()
    mismatched[0, 0] += 1.0e-3
    with pytest.raises(FactorIneligible, match="finite square"):
        factor_from_hidden_gram(nonfinite, rows)
    with pytest.raises(FactorIneligible, match="symmetry"):
        factor_from_hidden_gram(asymmetric, rows)
    with pytest.raises(FactorIneligible, match="explicit source"):
        factor_from_hidden_gram(mismatched, rows)

    monkeypatch.setattr(np.linalg, "cholesky", lambda _: np.eye(rows.shape[0]))
    with pytest.raises(FactorIneligible, match="Cholesky residual"):
        factor_from_hidden_gram(gram, rows)
    monkeypatch.setattr(np.linalg, "cholesky", lambda _: exact_cholesky)
    monkeypatch.setattr(np.linalg, "cond", lambda *_args, **_kwargs: 1.0e9)
    with pytest.raises(FactorIneligible, match="condition"):
        factor_from_hidden_gram(gram, rows)


def test_factor_fit_matches_dense_and_reconstructs_only_selected_centers() -> None:
    """Preserve fixed-M scientific payloads under the exact row-span isometry."""
    # Compare one unambiguous initialization and reconstruct through row coefficients.
    rows = _two_mode_rows()
    factor = factor_from_explicit_rows(rows)
    optimized = fit_factor_movmf(
        factor,
        n_clusters=2,
        n_init=1,
        max_iter=30,
        random_state=7,
        policy=_promotion_policy(),
    )
    dense = movMF(
        rows,
        2,
        posterior_type="soft",
        n_init=1,
        n_jobs=1,
        max_iter=30,
        init="k-means++",
        random_state=7,
        tol=1.0e-6,
    )
    dense_centers, dense_labels, dense_inertia, dense_weights, dense_kappas, dense_posterior = dense
    reconstructed = optimized.center_coefficients @ rows
    dense_likelihood = vmf_mixture_log_likelihood(
        rows, dense_centers, dense_weights, dense_kappas
    )

    assert optimized.labels == pytest.approx(dense_labels, rel=0.0, abs=0.0)
    assert optimized.weights == pytest.approx(dense_weights, rel=0.0, abs=1.0e-15)
    assert optimized.kappas == pytest.approx(dense_kappas, rel=0.0, abs=1.0e-9)
    assert optimized.responsibilities == pytest.approx(
        dense_posterior, rel=0.0, abs=1.0e-14
    )
    assert reconstructed == pytest.approx(dense_centers, rel=0.0, abs=1.0e-14)
    assert optimized.inertia == pytest.approx(dense_inertia, rel=0.0, abs=1.0e-14)
    assert optimized.log_likelihood == pytest.approx(
        dense_likelihood, rel=0.0, abs=1.0e-10
    )
    assert optimized.trace["ambient_dim"] == rows.shape[1]
    assert optimized.trace["factor_dim"] == rows.shape[0]


def test_factor_fit_is_bitwise_repeatable_for_one_pinned_backend() -> None:
    """Make backend-scoped deterministic payloads independent of repeated calls."""
    # Rebuild and rerun from the same rows and seed to cover factorization and EM.
    rows = _two_mode_rows(seed=11)
    results = [
        fit_factor_movmf(
            factor_from_explicit_rows(rows),
            n_clusters=2,
            n_init=1,
            max_iter=30,
            random_state=31,
            policy=_promotion_policy(),
        )
        for _ in range(2)
    ]

    for field in (
        "centers",
        "center_coefficients",
        "labels",
        "responsibilities",
        "weights",
        "kappas",
    ):
        assert np.array_equal(getattr(results[0], field), getattr(results[1], field))
    assert results[0].inertia == results[1].inertia
    assert results[0].log_likelihood == results[1].log_likelihood
    assert results[0].trace == results[1].trace


def test_ambiguous_factor_fit_restarts_complete_dense_budget() -> None:
    """Prove routing restarts from original rows, seed, and initialization budget."""
    # Identical rows force zero k-means++ potential before factor EM can continue.
    rows = np.zeros((4, 8), dtype=np.float64)
    rows[:, 0] = 1.0
    routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=2,
        max_iter=3,
        random_state=19,
        policy=_promotion_policy(),
    )
    direct_trace: dict[str, object] = {}
    direct = movMF(
        rows,
        2,
        posterior_type="soft",
        n_init=2,
        n_jobs=1,
        max_iter=3,
        init="k-means++",
        random_state=19,
        tol=1.0e-6,
        trace=direct_trace,
    )

    assert routed.backend == "dense_cpu"
    assert routed.factor_fit is None
    assert routed.route_reason == "kmeans_zero_or_nonfinite_potential"
    assert routed.trace["route"]["original_seed"] == 19
    assert routed.trace["route"]["original_n_init"] == 2
    assert routed.trace["route"]["original_max_iter"] == 3
    assert routed.trace["dense_trace"] == direct_trace
    assert routed.dense_result is not None
    for routed_value, direct_value in zip(routed.dense_result, direct):
        assert np.array_equal(routed_value, direct_value)


def test_initialization_winner_tie_restarts_complete_dense_fit() -> None:
    """Route an empirically tied likelihood winner before exposing factor output."""
    # Both starts converge to the same well-separated optimum and guarded tie.
    rows = _two_mode_rows(seed=9)
    routed = fit_factor_or_dense(
        rows,
        n_clusters=2,
        n_init=2,
        max_iter=30,
        random_state=13,
        policy=_promotion_policy(),
    )

    assert routed.backend == "dense_cpu"
    assert routed.route_reason == "ambiguous_initialization_likelihood"
    assert routed.trace["route"]["original_seed"] == 13
    assert routed.trace["route"]["original_n_init"] == 2


def test_actual_resultant_fallback_never_synthesizes_factor_center() -> None:
    """Route the inherited out-of-span fallback rather than replacing its basis."""
    # Equal responsibility on antipodal rows has an exact zero resultant.
    rows = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=np.float64
    )
    factor = factor_from_explicit_rows(rows)
    posterior = np.asarray([[0.5, 0.5]], dtype=np.float64)

    with pytest.raises(DenseRerunRequired, match="inherited_resultant_fallback"):
        _factor_maximization(
            factor,
            posterior,
            None,
            _promotion_policy(),
            [],
            0,
        )


def test_bic_guard_uses_pairwise_normalizer_allowance() -> None:
    """Use the reviewed pairwise BIC allowance without changing BIC selection."""
    # A near-boundary candidate routes dense while a clearly separated one survives.
    policy = _promotion_policy()
    incumbent = 100.0
    tolerance = 1.0e-9

    assert bic_decision_is_ambiguous(
        incumbent - tolerance + 1.0e-7, incumbent, tolerance, policy
    )
    assert not bic_decision_is_ambiguous(
        incumbent - tolerance - 1.0e-4, incumbent, tolerance, policy
    )


@pytest.mark.parametrize(
    ("surface", "gap", "analytical"),
    [
        ("posterior_score", 0.0, 1.0e-10),
        ("kmeans_mass", 1.0e-13, 1.0e-12),
        ("kmeans_potential", 0.0, 1.0e-10),
    ],
)
def test_guarded_tie_surfaces_require_dense_rerun(
    surface: str, gap: float, analytical: float
) -> None:
    """Route exact or near ties instead of relying on factor tie ordering."""
    # Exercise the common guard used at each declared discrete score surface.
    with pytest.raises(DenseRerunRequired, match=f"ambiguous_{surface}"):
        _guard_gap(
            _promotion_policy(),
            surface,
            gap,
            analytical,
            [],
            context={},
        )


def test_convergence_boundary_requires_dense_rerun() -> None:
    """Route a center-shift comparison inside the frozen convergence band."""
    # The comparison remains unchanged; only its result source becomes dense.
    with pytest.raises(DenseRerunRequired, match="ambiguous_convergence"):
        _guard_comparison(
            _promotion_policy(),
            "convergence",
            1.0e-6 + 5.0e-13,
            1.0e-6,
            1.0e-12,
            [],
            context={},
        )
