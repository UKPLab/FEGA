from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fega.core.geometry_reporting.thresholds import (
    get_threshold_profile,
    subspace_angle_threshold,
)
from fega.core.stability.metrics import (
    basis_orthonormality_error,
    final_resid_unit_rows,
    g_orthonormal_basis,
    principal_angles_degrees,
    scheduled_principal_angle_stability,
)


def _angle_plan(replicate_id: int, indices: range | tuple[int, ...]) -> SimpleNamespace:
    """Build the minimal immutable plan surface consumed by dense angle execution."""
    # Keep plan identity explicit without importing the production schedule builder.
    return SimpleNamespace(
        replicate_id=replicate_id,
        seed=71,
        digest=f"plan-{replicate_id}",
        indices=tuple(indices),
    )


def test_dense_angle_uses_only_scheduled_indices_and_one_locked_k() -> None:
    """Execute exactly the supplied index plans and expose every retained angle."""
    # Two distinct rank-preserving subsets make scheduled-index use observable.
    rows = torch.randn(32, 4, generator=torch.Generator().manual_seed(22))
    plans = (
        _angle_plan(0, range(24)),
        _angle_plan(1, tuple(range(8, 32))),
    )

    result = scheduled_principal_angle_stability(
        rows,
        torch.eye(4),
        plans=plans,
        source="raw",
        k=2,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )

    assert result["k"] == 2
    assert result["source"] == "raw"
    assert [item["plan_digest"] for item in result["replicates"]] == [
        "plan-0",
        "plan-1",
    ]
    assert result["counters"] == {
        "requested": 2,
        "valid": 2,
        "failed": 0,
        "non_applicable": 0,
        "skipped": 0,
    }
    assert result["angle_p90_deg"] == pytest.approx(
        np.quantile(
            [item["max_angle_deg"] for item in result["replicates"]], 0.9
        )
    )


def test_centered_low_context_retains_status_without_raw_rank_override() -> None:
    """Preserve the Section 6 raw-versus-centered low-context asymmetry."""
    # Rank-one rows make the raw k=2 override differ from centered residual status.
    rows = torch.tensor([[1.0, 0.0, 0.0]] * 16)

    raw = scheduled_principal_angle_stability(
        rows,
        torch.eye(3),
        plans=(),
        source="raw",
        k=2,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )
    centered = scheduled_principal_angle_stability(
        rows,
        torch.eye(3),
        plans=(),
        source="centered_residual",
        k=2,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )

    assert raw["status"] == "insufficient_rank"
    assert centered["status"] == "exploratory"


def test_material_negative_eigenvalue_boundary_is_strict() -> None:
    """Accept exactly -1e-5 for clamping and reject the next smaller value."""
    # Identity rows expose the supplied Gram eigenvalues directly.
    accepted = g_orthonormal_basis(
        torch.eye(2),
        torch.diag(torch.tensor([1.0, -1.0e-5], dtype=torch.float64)),
        eig_floor=1.0e-8,
    )
    smaller = math.nextafter(-1.0e-5, -math.inf)

    assert accepted["rank"] == 1
    with pytest.raises(ValueError, match="materially negative eigenvalue"):
        g_orthonormal_basis(
            torch.eye(2),
            torch.diag(torch.tensor([1.0, smaller], dtype=torch.float64)),
            eig_floor=1.0e-8,
        )


def test_eigenvalue_equal_to_floor_is_excluded() -> None:
    """Exclude equality at eig_floor instead of silently treating it as rank."""
    # The induced kernel is diagonal, so its two values are exact test inputs.
    result = g_orthonormal_basis(
        torch.eye(2),
        torch.diag(torch.tensor([1.0, 1.0e-8], dtype=torch.float64)),
        eig_floor=1.0e-8,
    )

    assert result["rank"] == 1


def test_principal_angles_clip_overlap_before_arccos() -> None:
    """Clamp roundoff outside the cosine domain before applying arccos."""
    # A minimally over-unit overlap must remain a zero-degree angle.
    basis_a = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    basis_b = torch.tensor([[1.0 + 1.0e-10], [0.0]], dtype=torch.float64)

    assert principal_angles_degrees(basis_a, basis_b, torch.eye(2), 1) == (
        pytest.approx([0.0])
    )


@pytest.mark.parametrize(
    ("k", "angle", "expected"),
    [
        (1, 30.0, "stable"),
        (1, math.nextafter(30.0, math.inf), "unstable"),
        (2, 30.0, "stable"),
        (2, math.nextafter(30.0, math.inf), "unstable"),
        (3, 35.0, "stable"),
        (3, math.nextafter(35.0, math.inf), "unstable"),
    ],
)
def test_angle_threshold_boundaries_are_inclusive_only_at_limit(
    k: int, angle: float, expected: str
) -> None:
    """Use 30/35-degree boundaries with equality stable and the next value unstable."""
    # Test the decision boundary separately from the dense numerical estimator.
    assert stability_runner_angle_decision(angle, k) == expected


def stability_runner_angle_decision(angle: float, k: int) -> str:
    """Call the production angle boundary without importing runner aggregation."""
    # Import locally so this metrics test remains independent of runner initialization.
    from fega.core.stability.protocols import angle_stability_decision

    return angle_stability_decision(angle, k)


def test_dense_basis_is_gram_orthonormal_after_kernel_symmetrization() -> None:
    """Retain the dense induced-kernel basis and Gram orthonormality contract."""
    # A nonidentity Gram exercises the dense path rather than a Euclidean shortcut.
    rows = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    gram = torch.diag(torch.tensor([2.0, 3.0]))

    basis = g_orthonormal_basis(rows, gram, eig_floor=1.0e-8)

    assert basis["rank"] == 2
    assert basis_orthonormality_error(basis["basis"], gram) == pytest.approx(0.0)


def test_dense_row_norm_filter_preserves_negative_and_eps_boundaries() -> None:
    """Round only tiny negative norms and skip every value at or below eps squared."""
    # Diagonal Gram entries make each row's x^T G x value explicit.
    eps_sq = 1.0e-24
    rows = torch.eye(4, dtype=torch.float64)
    gram = torch.diag(
        torch.tensor(
            [
                math.nextafter(-1.0e-7, 0.0),
                -1.0e-7,
                eps_sq,
                math.nextafter(eps_sq, math.inf),
            ],
            dtype=torch.float64,
        )
    )

    valid, counts, indices = final_resid_unit_rows(rows, gram, eps=1.0e-12)

    assert indices == [3]
    assert valid.shape == (1, 4)
    assert counts == {
        "n_total": 4,
        "n_valid": 1,
        "skipped_nonfinite": 0,
        "skipped_zero_norm": 3,
    }


def test_full_rank_and_subset_rank_failures_remain_unavailable() -> None:
    """Reject locked k when either the full cloud or a required subset lacks rank."""
    # The full-rank case fails immediately; the second case fails on its scheduled subset.
    rank_one = torch.tensor([[1.0, 0.0]] * 32)
    plan = (_angle_plan(0, range(24)),)
    full_failure = scheduled_principal_angle_stability(
        rank_one,
        torch.eye(2),
        plans=plan,
        source="raw",
        k=2,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )
    mixed = torch.tensor([[1.0, 0.0]] * 24 + [[0.0, 1.0]] * 8)
    subset_failure = scheduled_principal_angle_stability(
        mixed,
        torch.eye(2),
        plans=plan,
        source="raw",
        k=2,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )

    assert full_failure["status"] == "insufficient_rank"
    assert full_failure["angle_p90_deg"] is None
    assert subset_failure["status"] == "unavailable"
    assert subset_failure["replicates"][0]["failure"] == "insufficient_rank"
    assert subset_failure["angle_p90_deg"] is None


def _equivalence_rows(k: int, *, above_threshold: bool) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Construct an exact zero- or 45-degree selected-k subset with at least 32 rows."""
    # Symmetric row pairs make raw and centered-residual sources share the same geometry.
    width = int(k) + 1
    pair_repeats = max(4, math.ceil(16 / int(k)))
    rows: list[torch.Tensor] = []
    canonical_indices: list[list[int]] = []
    for axis in range(int(k)):
        indices: list[int] = []
        vector = torch.zeros(width, dtype=torch.float64)
        vector[axis] = 1.0
        for _ in range(pair_repeats):
            indices.extend((len(rows), len(rows) + 1))
            rows.extend((vector.clone(), -vector.clone()))
        canonical_indices.append(indices)
    rotated = torch.zeros(width, dtype=torch.float64)
    rotated[int(k) - 1] = math.sqrt(0.5)
    rotated[int(k)] = math.sqrt(0.5)
    reflected = rotated.clone()
    reflected[int(k)] *= -1.0
    rotated_indices: list[int] = []
    for vector in (rotated, reflected):
        rotated_indices.extend((len(rows), len(rows) + 1))
        rows.extend((vector.clone(), -vector.clone()))
    retained_axes = canonical_indices[:-1] if above_threshold else canonical_indices
    subset = [index for indices in retained_axes for index in indices[:2]]
    if above_threshold:
        subset.extend(rotated_indices[:2])
    return torch.stack(rows).to(dtype=torch.float32), tuple(subset)


@pytest.mark.parametrize(
    ("source", "k"),
    [
        ("raw", 1),
        ("raw", 2),
        ("raw", 3),
        ("raw", 4),
        ("raw", 8),
        ("centered_residual", 2),
        ("centered_residual", 3),
        ("centered_residual", 4),
    ],
)
@pytest.mark.parametrize("above_threshold", [False, True])
def test_dense_angle_matches_factor_oracle_across_locked_sources_and_threshold_sides(
    source: str, k: int, above_threshold: bool
) -> None:
    """Block dense/factor drift across every supported selected-k threshold class."""
    # Factor coordinates remain verification-only and are never a production fallback.
    rows, indices = _equivalence_rows(k, above_threshold=above_threshold)
    width = int(rows.shape[1])
    unembedding = torch.diag(torch.linspace(0.75, 1.25, width, dtype=torch.float64))
    gram = unembedding @ unembedding.T
    plans = (_angle_plan(0, indices),)
    dense = scheduled_principal_angle_stability(
        rows,
        gram,
        plans=plans,
        source=source,
        k=k,
        angle_quantile=0.9,
        eig_floor=1.0e-8,
    )

    factor_rows = rows.to(dtype=torch.float64) @ unembedding.to(dtype=torch.float64)
    identity = torch.eye(width, dtype=torch.float64)
    full_rows = (
        factor_rows - factor_rows.mean(dim=0, keepdim=True)
        if source == "centered_residual"
        else factor_rows
    )
    full = g_orthonormal_basis(full_rows, identity, eig_floor=1.0e-8)
    maxima: list[float] = []
    for plan in plans:
        subset = factor_rows.index_select(
            0, torch.as_tensor(plan.indices, dtype=torch.long)
        )
        if source == "centered_residual":
            subset = subset - subset.mean(dim=0, keepdim=True)
        basis = g_orthonormal_basis(subset, identity, eig_floor=1.0e-8)
        maxima.append(
            max(principal_angles_degrees(full["basis"], basis["basis"], identity, k))
        )
    factor_p90 = float(np.quantile(maxima, 0.9))
    threshold = subspace_angle_threshold(get_threshold_profile("paper"), k)

    assert dense["angle_p90_deg"] == pytest.approx(factor_p90, abs=1.0e-4)
    assert (dense["angle_p90_deg"] <= threshold) == (factor_p90 <= threshold)
    assert (factor_p90 > threshold) is above_threshold
