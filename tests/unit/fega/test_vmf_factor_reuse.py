from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fega.config_schema import DirectionalMixtureFitConfig
from fega.core.vmf import factor_reuse
from fega.core.vmf.factor_reuse import FeatureFactor
from fega.core.vmf.metrics import score_vmf_feature


def _unit_rows(*, count: int = 8, width: int = 12) -> torch.Tensor:
    """Return deterministic full-rank unit rows for standalone-vMF factor tests."""
    # Normalize one fixed random draw so candidate evidence is reproducible.
    generator = torch.Generator().manual_seed(917)
    rows = torch.randn(count, width, generator=generator, dtype=torch.float32)
    return rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True)


def test_feature_factor_builds_once_and_preserves_repeated_ordered_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build one standalone-vMF root QR and preserve duplicate ordered views."""
    # Count the only operation allowed to construct a CPU factor for the feature.
    original_build = factor_reuse.build_cpu_factor
    build_count = 0

    def counted_build(rows: np.ndarray, gram: np.ndarray | None = None):
        """Record root construction while delegating exact factor mathematics."""
        # Increment before delegation so even a rejected construction is observable.
        nonlocal build_count
        build_count += 1
        return original_build(rows, gram=gram)

    monkeypatch.setattr(factor_reuse, "build_cpu_factor", counted_build)
    rows = _unit_rows()
    scales = torch.arange(1, 9, dtype=torch.float64)

    feature = FeatureFactor.build(rows, row_scales=scales)
    repeated = feature.view([6, 1, 6, 0])
    nested = repeated.select([2, 0, 2, 1])

    assert build_count == 1
    assert repeated.absolute_indices == (6, 1, 6, 0)
    assert nested.absolute_indices == (6, 6, 6, 1)
    assert torch.equal(repeated.unit_rows, feature.full_view.unit_rows[[6, 1, 6, 0]])
    assert repeated.factor is not None
    assert feature.full_view.factor is not None
    np.testing.assert_array_equal(
        repeated.factor.z,
        feature.full_view.factor.z[[6, 1, 6, 0]],
    )
    np.testing.assert_allclose(
        repeated.factor.z @ repeated.factor.z.T,
        repeated.normalized_rows @ repeated.normalized_rows.T,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        repeated.subspace_rows.numpy(),
        repeated.factor.z * scales[[6, 1, 6, 0]].numpy()[:, None],
        rtol=0.0,
        atol=0.0,
    )


def test_prepared_candidates_and_assignment_reuse_root_without_scientific_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match fresh-factor vMF decisions while forbidding prepared-path QR rebuild."""
    # Use two separated modes so candidate selection and assignment refit are exercised.
    rows = torch.zeros((12, 64), dtype=torch.float32)
    rows[:6, 0] = 1.0
    rows[6:, 1] = 1.0
    rows[:, 2] = torch.linspace(-0.02, 0.02, 12)
    config = DirectionalMixtureFitConfig(
        enabled=True,
        k_values=[1, 2],
        n_init=1,
        max_iter=8,
        resample_fraction=0.8,
        resample_rounds=1,
    )
    expected = score_vmf_feature(rows, config, seed=29)
    feature = FeatureFactor.build(rows)

    def reject_rebuild(*_args, **_kwargs):
        """Fail if a candidate or assignment subset constructs another QR."""
        # The sole permitted build already occurred at standalone-vMF ownership.
        raise AssertionError("prepared vMF path rebuilt a factor")

    monkeypatch.setattr(factor_reuse, "build_cpu_factor", reject_rebuild)
    actual = score_vmf_feature(
        feature.full_view.unit_rows,
        config,
        seed=29,
        prepared=feature.full_view,
    )

    assert actual.fit_status == expected.fit_status
    assert actual.metrics == expected.metrics
    assert actual.model_selection == expected.model_selection
    assert actual.assignment_stability == expected.assignment_stability


def test_ineligible_root_routes_dense_once_from_immutable_view_provider() -> None:
    """Preserve corrected-dense vMF results when the sole feature factor is ineligible."""
    # Use more rows than dimensions to force the explicit-QR eligibility boundary.
    rows = _unit_rows(count=8, width=4)
    config = DirectionalMixtureFitConfig(
        enabled=True,
        backend="factor_cpu",
        k_values=[1],
        n_init=1,
        max_iter=4,
        resample_rounds=0,
    )
    expected = score_vmf_feature(
        rows,
        replace(config, backend="dense_cpu"),
        seed=37,
    )
    feature = FeatureFactor.build(rows)
    view = feature.full_view
    trace: dict[str, object] = {}

    assert view.factor is None
    actual = score_vmf_feature(
        view.unit_rows,
        config,
        seed=37,
        prepared=view,
        trace=trace,
    )

    assert actual.fit_status == expected.fit_status
    assert actual.metrics == expected.metrics
    assert actual.model_selection == expected.model_selection
    assert trace["candidates"][0]["fit_trace"]["routing"]["reason"] == (
        "explicit factor requires ambient dimension at least the row count"
    )
