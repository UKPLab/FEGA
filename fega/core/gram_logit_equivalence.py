"""Validation helpers for hidden-to-logit Gram equivalence evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from fega.core.compute_effect.artifacts import gram_magnitude
from fega.core.geometry_metrics.metrics import (
    c_ray_pairwise_final_resid,
    centered_residual_spectrum_final_resid,
    span_spectrum_final_resid,
)
from fega.core.stability.metrics import (
    final_resid_unit_kernel,
    logits_unit_kernel,
)

FLOAT32_EQUIVALENCE_TOLERANCES: dict[str, dict[str, float]] = {
    "reconstruction": {"rtol": 1e-6, "atol": 1e-6},
    "norms": {"rtol": 1e-5, "atol": 1e-6},
    "inner_products": {"rtol": 1e-5, "atol": 1e-5},
    "cosines": {"rtol": 1e-5, "atol": 1e-5},
    "c_ray": {"rtol": 1e-4, "atol": 1e-4},
    "s_span_1": {"rtol": 1e-4, "atol": 1e-4},
    "s_res_1": {"rtol": 1e-4, "atol": 1e-4},
}


def _resolved_tolerances(
    overrides: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Return independent per-check tolerances with validated named overrides.

    Overrides may replace either ``rtol`` or ``atol`` for any named required
    check while retaining the other approved default.
    """
    # Copy the defaults before applying caller-provided named values.
    resolved = {
        name: values.copy()
        for name, values in FLOAT32_EQUIVALENCE_TOLERANCES.items()
    }
    if overrides is None:
        return resolved
    for name, values in overrides.items():
        if name not in resolved:
            raise ValueError(f"Unknown tolerance name: {name}")
        for field, value in values.items():
            if field not in {"rtol", "atol"}:
                raise ValueError(f"Unknown tolerance field for {name}: {field}")
            resolved[name][field] = float(value)
    return resolved


def _absolute_error_summary(actual: Tensor, expected: Tensor) -> dict[str, float]:
    """Summarize elementwise absolute error with maximum, median, and q99.

    The tensors are flattened after subtraction so the summary is independent
    of whether the comparison concerns vectors, kernels, or scalar profiles.
    """
    # Compute the requested distributional summary from finite tensor errors.
    errors = (actual - expected).abs().reshape(-1).to(dtype=torch.float64)
    if errors.numel() == 0:
        return {"max": 0.0, "median": 0.0, "q99": 0.0}
    return {
        "max": float(errors.max().item()),
        "median": float(errors.median().item()),
        "q99": float(torch.quantile(errors, 0.99).item()),
    }


def _comparison(
    actual: Tensor,
    expected: Tensor,
    tolerance: Mapping[str, float],
) -> dict[str, Any]:
    """Build the standard result record for one numerical equivalence check.

    A check passes only when every element satisfies the named relative and
    absolute tolerance, while the error summary remains purely absolute.
    """
    # Compare floating tensors in float64 without changing the source tensors.
    comparison_actual = actual
    comparison_expected = expected
    if actual.is_floating_point() and expected.is_floating_point():
        comparison_actual = actual.to(dtype=torch.float64)
        comparison_expected = expected.to(dtype=torch.float64)

    # Evaluate allclose and package the common tolerance and error fields.
    return {
        "passed": bool(
            torch.allclose(
                comparison_actual,
                comparison_expected,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
            )
        ),
        "tolerance": dict(tolerance),
        "absolute_error": _absolute_error_summary(
            comparison_actual, comparison_expected
        ),
    }


def _profile_scalar(profile: Any, name: str) -> Tensor:
    """Read one paper scalar from a validation profile or result object.

    The profile's scalar values are converted to tensors solely to share the
    same comparison and absolute-error reporting path as the matrix checks.
    """
    # Read the named scalar through the common comparison path.
    value = profile[name] if isinstance(profile, Mapping) else getattr(profile, name)
    return value if isinstance(value, Tensor) else torch.as_tensor(value)


def _canonical_geometry_profile(
    directions: Tensor, gram: Tensor, *, eps: float
) -> dict[str, float | None]:
    """Read the three validation scalars from canonical geometry authorities."""
    # Evaluate the unit directions through the production geometry metric APIs.
    c_ray = c_ray_pairwise_final_resid(directions, gram, eps=eps)
    span = span_spectrum_final_resid(directions, gram, k_values=[1], eps=eps)
    residual = centered_residual_spectrum_final_resid(
        directions, gram, k_values=[1], eps=eps
    )
    return {
        "c_ray": c_ray.c_ray,
        "s_span_1": span.s_span[1],
        "s_res_1": residual.s_res[1],
    }


def _explicit_geometry_profile(
    unit_kernel: Tensor, *, eps: float
) -> dict[str, float | None]:
    """Compute the three paper scalars directly from an explicit unit kernel.

    The explicit logit kernel is PSD by construction. Its spectra are clamped
    only at zero to remove negative roundoff after float64 symmetrization.
    """
    # Normalize the small row kernel representation before spectral identities.
    kernel = unit_kernel.detach().cpu().to(dtype=torch.float64)
    kernel = (kernel + kernel.T) / 2.0
    n_rows = int(kernel.shape[0])
    if n_rows < 2:
        return {"c_ray": None, "s_span_1": None, "s_res_1": None}

    # Compute the off-diagonal mean and leading uncentered spectral mass.
    off_diagonal_sum = kernel.sum() - torch.trace(kernel)
    c_ray = off_diagonal_sum / float(n_rows * (n_rows - 1))
    span_eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.0)
    s_span_1 = span_eigenvalues[-1] / (span_eigenvalues.sum() + eps)

    # Center with the sole row-sized identity and compute residual spectral mass.
    identity = torch.eye(n_rows, dtype=torch.float64)
    centering = identity - torch.ones_like(kernel) / float(n_rows)
    centered_kernel = centering @ kernel @ centering
    centered_kernel = (centered_kernel + centered_kernel.T) / 2.0
    residual_eigenvalues = torch.linalg.eigvalsh(centered_kernel).clamp_min(0.0)
    s_res_1 = residual_eigenvalues[-1] / (residual_eigenvalues.sum() + eps)
    return {
        "c_ray": float(c_ray.item()),
        "s_span_1": float(s_span_1.item()),
        "s_res_1": float(s_res_1.item()),
    }


def _reject_indefinite_gram(gram: Tensor) -> None:
    """Reject a supplied hidden-space Gram with any negative eigenvalue."""
    # Use the symmetric input-Gram spectrum directly without a tolerance or clamp.
    matrix = gram.detach().cpu().to(dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    if eigenvalues.numel() and torch.any(eigenvalues < 0):
        raise ValueError(
            "Hidden-space Gram has negative eigenvalue "
            f"{float(eigenvalues.min().item()):.6g}."
        )


def evaluate_gram_logit_equivalence(
    *,
    hidden_deltas: Tensor,
    explicit_logit_deltas: Tensor,
    unembedding: Tensor,
    gram: Tensor,
    returned_model_output_deltas: Tensor,
    expected_source_fingerprint: Any,
    observed_source_fingerprint: Any,
    tolerances: Mapping[str, Mapping[str, float]] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate Gram/logit equivalence and the returned-model-output diagnostic.

    ``unembedding`` is vocab-by-hidden, so linear logits are reconstructed as
    ``hidden_deltas @ unembedding.T``. ``gram`` is hidden-by-hidden and is used
    for hidden-space norm and inner-product equivalence. Fingerprint mismatch
    raises immediately. The returned-model-output comparison is diagnostic only
    and cannot affect the overall seven-check status.
    """
    # Reject incompatible bundles before performing any numerical validation.
    if expected_source_fingerprint != observed_source_fingerprint:
        raise ValueError(
            "Gram/logit equivalence source fingerprint mismatch: "
            f"expected {expected_source_fingerprint!r}, "
            f"observed {observed_source_fingerprint!r}"
        )
    resolved = _resolved_tolerances(tolerances)
    reconstructed = hidden_deltas @ unembedding.T
    direct_norms = torch.linalg.vector_norm(explicit_logit_deltas, dim=-1)
    hidden_norms = gram_magnitude(hidden_deltas, gram)
    _reject_indefinite_gram(gram)
    hidden_inner_products = hidden_deltas @ gram @ hidden_deltas.T
    direct_inner_products = explicit_logit_deltas @ explicit_logit_deltas.T

    checks: dict[str, dict[str, Any]] = {
        "reconstruction": _comparison(
            reconstructed, explicit_logit_deltas, resolved["reconstruction"]
        ),
        "norms": _comparison(direct_norms, hidden_norms, resolved["norms"]),
        "inner_products": _comparison(
            direct_inner_products,
            hidden_inner_products,
            resolved["inner_products"],
        ),
    }

    direct_kernel, _, direct_rows = logits_unit_kernel(explicit_logit_deltas, eps=eps)
    hidden_kernel, _, hidden_unit_rows, hidden_rows = final_resid_unit_kernel(
        hidden_deltas, gram, eps=eps
    )
    if direct_rows != hidden_rows:
        reason = (
            "Normalized kernels have different valid-row indices: "
            f"logits={direct_rows}, hidden_gram={hidden_rows}"
        )
        checks["cosines"] = {
            "passed": False,
            "tolerance": dict(resolved["cosines"]),
            "reason": reason,
        }
        for name in ("c_ray", "s_span_1", "s_res_1"):
            checks[name] = {
                "passed": False,
                "tolerance": dict(resolved[name]),
                "reason": reason,
            }
    else:
        checks["cosines"] = _comparison(
            direct_kernel, hidden_kernel, resolved["cosines"]
        )
        direct_profile = _explicit_geometry_profile(direct_kernel, eps=eps)
        hidden_profile = _canonical_geometry_profile(
            hidden_unit_rows, gram.detach().cpu().to(dtype=torch.float64), eps=eps
        )
        for name in ("c_ray", "s_span_1", "s_res_1"):
            checks[name] = _comparison(
                _profile_scalar(direct_profile, name),
                _profile_scalar(hidden_profile, name),
                resolved[name],
            )

    diagnostic = _comparison(
        returned_model_output_deltas,
        reconstructed,
        resolved["reconstruction"],
    )
    return {
        "status": "pass"
        if all(check["passed"] for check in checks.values())
        else "fail",
        "fingerprints": {
            "expected": expected_source_fingerprint,
            "observed": observed_source_fingerprint,
            "match": True,
        },
        "checks": checks,
        "diagnostics": {
            "returned_model_output_delta_vs_linear": {
                "equivalent": diagnostic["passed"],
                "tolerance": diagnostic["tolerance"],
                "absolute_error": diagnostic["absolute_error"],
            }
        },
    }


def evaluate_grouped_gram_logit_equivalence(
    *,
    feature_groups: Mapping[int | str, Mapping[str, Any]],
    unembedding: Tensor,
    gram: Tensor,
    expected_source_fingerprint: Any,
    observed_source_fingerprint: Any,
    tolerances: Mapping[str, Mapping[str, float]] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate each feature cloud independently in numeric feature order.

    Every group supplies its persisted hidden deltas, explicit linear-logit
    deltas, returned-output diagnostic deltas, and complete ordered row
    identities. Shared readout tensors, fingerprints, and tolerances are passed
    unchanged to the existing single-cloud mathematical evaluator.
    """
    # Normalize numeric feature identifiers and reject ambiguous or empty groups.
    if not isinstance(feature_groups, Mapping) or not feature_groups:
        raise ValueError("feature_groups must be a non-empty mapping")
    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_feature_id, group in feature_groups.items():
        try:
            feature_id = int(raw_feature_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature ID must be numeric: {raw_feature_id!r}") from exc
        if str(feature_id) != str(raw_feature_id):
            raise ValueError(f"feature ID must use canonical numeric form: {raw_feature_id!r}")
        if feature_id in normalized:
            raise ValueError(f"duplicate numeric feature ID: {feature_id}")
        if not isinstance(group, Mapping):
            raise TypeError(f"feature group {feature_id} must be a mapping")
        normalized[feature_id] = group

    # Validate row-aligned group fields before invoking the mathematical unit.
    required = {
        "hidden_deltas",
        "explicit_logit_deltas",
        "returned_model_output_deltas",
        "row_identities",
    }
    per_feature: dict[str, Any] = {}
    for feature_id in sorted(normalized):
        group = normalized[feature_id]
        missing = sorted(required.difference(group))
        if missing:
            raise KeyError(
                f"feature group {feature_id} is missing required keys: {', '.join(missing)}"
            )
        tensors = [
            group["hidden_deltas"],
            group["explicit_logit_deltas"],
            group["returned_model_output_deltas"],
        ]
        if any(not isinstance(tensor, Tensor) or tensor.ndim != 2 for tensor in tensors):
            raise ValueError(f"feature group {feature_id} tensors must be rank-2")
        row_count = int(tensors[0].shape[0])
        if row_count == 0 or any(int(tensor.shape[0]) != row_count for tensor in tensors):
            raise ValueError(f"feature group {feature_id} must have aligned non-empty rows")
        identities = group["row_identities"]
        if (
            not isinstance(identities, Sequence)
            or isinstance(identities, str | bytes)
            or len(identities) != row_count
            or any(not isinstance(identity, Mapping) for identity in identities)
        ):
            raise ValueError(
                f"feature group {feature_id} row identities must align with tensor rows"
            )
        evaluation = evaluate_gram_logit_equivalence(
            hidden_deltas=tensors[0],
            explicit_logit_deltas=tensors[1],
            returned_model_output_deltas=tensors[2],
            unembedding=unembedding,
            gram=gram,
            expected_source_fingerprint=expected_source_fingerprint,
            observed_source_fingerprint=observed_source_fingerprint,
            tolerances=tolerances,
            eps=eps,
        )
        per_feature[str(feature_id)] = {
            "feature_id": feature_id,
            "row_identities": [dict(identity) for identity in identities],
            **evaluation,
        }

    # Aggregate only the independently evaluated feature outcomes.
    return {
        "status": "pass"
        if all(result["status"] == "pass" for result in per_feature.values())
        else "fail",
        "observed_feature_ids": [int(feature_id) for feature_id in per_feature],
        "per_feature": per_feature,
    }
