from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from fega.core.geometry_metrics.metrics import normalize_logit_deltas


@torch.no_grad()
def logits_unit_kernel(
    rows: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, dict[str, int], list[int]]:
    """Build the diagnostic logit-space unit kernel used by equivalence checks."""
    # Retain the existing normalization authority and ordered valid-row inventory.
    unit_rows, counts = normalize_logit_deltas(rows, eps=eps)
    return _kernel_from_unit_rows(unit_rows), counts, _valid_logit_indices(rows, eps)


@torch.no_grad()
def final_resid_unit_kernel(
    rows: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, dict[str, int], torch.Tensor, list[int]]:
    """Build the canonical dense-Gram unit kernel and exact valid-row inventory."""
    # Filter and normalize once so downstream artifact comparisons share row identity.
    valid_rows, counts, valid_indices = final_resid_unit_rows(rows, gram, eps=eps)
    return (
        _kernel_from_unit_rows(valid_rows, gram),
        counts,
        valid_rows,
        valid_indices,
    )


@torch.no_grad()
def scheduled_principal_angle_stability(
    rows: torch.Tensor,
    gram: torch.Tensor,
    *,
    plans: Sequence[Any],
    source: str,
    k: int,
    angle_quantile: float,
    eig_floor: float,
) -> dict[str, Any]:
    """Execute one dense-Gram principal angle at one locked k on scheduled indices.

    Raw and centered-residual paths retain the established row filter, induced-kernel
    eigensolver, rank policy, singular-value clipping, and NumPy linear quantile. The
    caller supplies the complete immutable index schedule; no RNG or extra dimension
    can be introduced here.
    """
    # Validate the locked source and derive the canonical raw valid-row inventory.
    if source not in {"raw", "centered_residual"}:
        raise ValueError(f"unsupported principal-angle source: {source}")
    if int(k) <= 0:
        raise ValueError("principal-angle k must be positive")
    valid_rows, counts, _ = _valid_final_resid_rows(rows, gram, eps=1.0e-12)
    n_valid = int(valid_rows.shape[0])
    basis_rows = _center_rows(valid_rows) if source == "centered_residual" else valid_rows
    full_basis = g_orthonormal_basis(basis_rows, gram, eig_floor=eig_floor)
    full_rank = int(full_basis["rank"])
    common = {
        "source": source,
        "k": int(k),
        "n_valid": n_valid,
        "numerical_rank": full_rank,
        "orthonormality_error": _json_float(
            basis_orthonormality_error(full_basis["basis"], gram)
        ),
        "plan_digest": _plan_digest(plans),
    }

    # Preserve the approved low-context asymmetry without fabricating angle values.
    if n_valid < 32:
        if plans:
            raise ValueError("low-context angle schedule must be empty")
        base_status = "insufficient_contexts" if n_valid < 8 else "exploratory"
        status = (
            "insufficient_rank"
            if source == "raw" and int(k) > full_rank
            else (
                "insufficient_contexts"
                if source == "centered_residual" and n_valid < 16
                else base_status
            )
        )
        return {
            **common,
            "status": status,
            "angle_p90_deg": None,
            "replicates": [],
            "counters": _counters(non_applicable=1),
        }

    # Missing enabled work is explicit non-applicability; rank failure is unavailable.
    if not plans:
        return {
            **common,
            "status": "not_applicable",
            "angle_p90_deg": None,
            "replicates": [],
            "counters": _counters(non_applicable=1),
        }
    if int(k) > full_rank:
        first, *rest = list(plans)
        return {
            **common,
            "status": "insufficient_rank",
            "angle_p90_deg": None,
            "replicates": [
                _failed_angle_record(first, "insufficient_rank"),
                *[
                    _skipped_angle_record(plan, "prior_insufficient_rank")
                    for plan in rest
                ],
            ],
            "counters": _counters(
                requested=len(plans), failed=1, skipped=max(0, len(plans) - 1)
            ),
        }

    # Execute every scheduled subset until the established sticky rank failure occurs.
    replicates: list[dict[str, Any]] = []
    max_angles: list[float] = []
    mean_angles: list[float] = []
    failed = 0
    skipped = 0
    for position, plan in enumerate(plans):
        try:
            indices = tuple(int(index) for index in plan.indices)
            if not indices or min(indices) < 0 or max(indices) >= n_valid:
                raise ValueError("principal-angle plan index out of range")
            index = torch.as_tensor(indices, dtype=torch.long)
            subset_rows = valid_rows.index_select(0, index)
            if source == "centered_residual":
                subset_rows = _center_rows(subset_rows)
            subset_basis = g_orthonormal_basis(
                subset_rows, gram, eig_floor=eig_floor
            )
            subset_rank = int(subset_basis["rank"])
            if int(k) > subset_rank:
                replicates.append(_failed_angle_record(plan, "insufficient_rank"))
                failed = 1
                for later in plans[position + 1 :]:
                    replicates.append(
                        _skipped_angle_record(later, "prior_insufficient_rank")
                    )
                    skipped += 1
                break
            angles = principal_angles_degrees(
                full_basis["basis"], subset_basis["basis"], gram, int(k)
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            replicates.append(
                _failed_angle_record(plan, f"{type(exc).__name__}:{exc}")
            )
            failed = 1
            for later in plans[position + 1 :]:
                replicates.append(_skipped_angle_record(later, "prior_angle_failure"))
                skipped += 1
            break
        mean_angle = float(np.mean(angles))
        max_angle = float(np.max(angles))
        mean_angles.append(mean_angle)
        max_angles.append(max_angle)
        replicates.append(
            {
                "plan_digest": str(plan.digest),
                "seed": int(plan.seed),
                "replicate_id": int(plan.replicate_id),
                "status": "valid",
                "mean_angle_deg": mean_angle,
                "max_angle_deg": max_angle,
            }
        )
    valid_count = len(max_angles)
    return {
        **common,
        "status": "unavailable" if failed else "ok",
        "angle_p90_deg": _json_float(_quantile(max_angles, angle_quantile)),
        "mean_angle_deg_median": _json_float(_quantile(mean_angles, 0.5)),
        "max_angle_deg_median": _json_float(_quantile(max_angles, 0.5)),
        "replicates": replicates,
        "counters": _counters(
            requested=len(plans),
            valid=valid_count,
            failed=failed,
            skipped=skipped,
        ),
    }


def g_orthonormal_basis(
    rows: torch.Tensor, gram: torch.Tensor, *, eig_floor: float
) -> dict[str, Any]:
    """Construct the established dense-Gram orthonormal row-subspace basis."""
    # Form and symmetrize only the induced row kernel; the canonical Gram is immutable.
    if rows.ndim != 2:
        raise ValueError("rows must be rank-2.")
    rows64 = rows.detach().cpu().to(dtype=torch.float64)
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    kernel = rows64 @ gram64 @ rows64.T
    kernel = (kernel + kernel.T) / 2.0
    eigvals_raw, eigvecs_raw = torch.linalg.eigh(kernel)
    order = torch.argsort(eigvals_raw, descending=True)
    eigvals = eigvals_raw[order]
    eigvecs = eigvecs_raw[:, order]
    if eigvals.numel():
        min_eval = float(eigvals.min().item())
        if min_eval < -1.0e-5:
            raise ValueError(
                "Subspace Gram matrix has materially negative eigenvalue "
                f"{min_eval:.6g}."
            )
        eigvals = torch.clamp(eigvals, min=0.0)
    keep = eigvals > float(eig_floor)
    if not bool(keep.any().item()):
        return {
            "basis": torch.empty((int(rows.shape[1]), 0), dtype=torch.float64),
            "eigenvalues": [],
            "rank": 0,
        }
    kept_vals = eigvals[keep]
    kept_vecs = eigvecs[:, keep]
    basis = rows64.T @ kept_vecs @ torch.diag(torch.rsqrt(kept_vals))
    return {
        "basis": basis,
        "eigenvalues": [float(value) for value in kept_vals.tolist()],
        "rank": int(kept_vals.numel()),
    }


def principal_angles_degrees(
    basis_a: torch.Tensor,
    basis_b: torch.Tensor,
    gram: torch.Tensor,
    k: int,
) -> list[float]:
    """Return the first-k dense-Gram principal angles in degrees."""
    # Clamp overlap singular values before arccos to absorb numerical domain roundoff.
    if int(k) <= 0:
        raise ValueError("k must be positive.")
    if basis_a.shape[1] < int(k) or basis_b.shape[1] < int(k):
        raise ValueError("Cannot compute principal angles when k exceeds basis rank.")
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    overlap = basis_a[:, : int(k)].T @ gram64 @ basis_b[:, : int(k)]
    singular = torch.linalg.svdvals(overlap)
    singular = torch.clamp(singular, min=-1.0, max=1.0)
    return [
        float(math.degrees(math.acos(float(value))))
        for value in singular.tolist()
    ]


def basis_orthonormality_error(
    basis: torch.Tensor, gram: torch.Tensor
) -> float | None:
    """Measure dense-Gram basis orthonormality in the infinity norm."""
    # Empty bases have no meaningful orthonormality residual.
    if basis.numel() == 0:
        return None
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    identity = basis.T @ gram64 @ basis
    difference = identity - torch.eye(int(identity.shape[0]), dtype=torch.float64)
    return float(torch.linalg.matrix_norm(difference, ord=float("inf")).item())


def group_sampling_status(
    labels: Sequence[str | None] | None, *, min_group_count: int, min_group_size: int
) -> dict[str, Any]:
    """Describe whether filtered group labels support complete group-out schedules."""
    # Count only concrete labels and retain all observed group denominators.
    if not labels:
        return {"status": "group_sampling_unavailable", "groups": {}}
    counts: dict[str, int] = {}
    for label in labels:
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    usable = {
        label: count for label, count in counts.items() if count >= min_group_size
    }
    if len(usable) < min_group_count:
        return {"status": "group_sampling_unavailable", "groups": counts}
    return {"status": "ok", "groups": counts, "usable_groups": usable}


def final_resid_unit_rows(
    rows: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, dict[str, int], list[int]]:
    """Filter and unit-normalize final-residual rows under the canonical Gram."""
    # Preserve the exact valid-row mask before normalizing retained rows in CPU float64.
    valid, counts, valid_indices = _valid_final_resid_rows(rows, gram, eps=eps)
    if int(valid.shape[0]) == 0:
        return valid, counts, valid_indices
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    unit_rows = []
    for row in valid.to(dtype=torch.float64):
        norm_sq = float((row @ gram64 @ row).item())
        norm_sq = 0.0 if -1.0e-7 < norm_sq < 0.0 else norm_sq
        unit_rows.append((row / math.sqrt(norm_sq)).to(dtype=torch.float32))
    return torch.stack(unit_rows), counts, valid_indices


def _valid_final_resid_rows(
    rows: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, dict[str, int], list[int]]:
    """Apply the frozen finite and dense-Gram norm validity mask."""
    # Materially negative norms are skipped by the same <= eps^2 boundary as zero rows.
    if rows.ndim != 2:
        raise ValueError(f"final_resid rows must be rank-2, got {tuple(rows.shape)}.")
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    if gram64.ndim != 2 or gram64.shape[0] != gram64.shape[1]:
        raise ValueError(f"Gram tensor must be square rank-2, got {tuple(gram.shape)}.")
    if int(gram64.shape[0]) != int(rows.shape[1]):
        raise ValueError(
            f"Gram width {gram64.shape[0]} does not match row width {rows.shape[1]}."
        )
    valid: list[torch.Tensor] = []
    valid_indices: list[int] = []
    skipped_nonfinite = 0
    skipped_zero_norm = 0
    for row_index, row in enumerate(rows.detach().cpu().to(dtype=torch.float64)):
        if not torch.isfinite(row).all().item():
            skipped_nonfinite += 1
            continue
        norm_sq = float((row @ gram64 @ row).item())
        norm_sq = 0.0 if -1.0e-7 < norm_sq < 0.0 else norm_sq
        if not math.isfinite(norm_sq):
            skipped_nonfinite += 1
            continue
        if norm_sq <= float(eps) * float(eps):
            skipped_zero_norm += 1
            continue
        valid.append(row.to(dtype=torch.float32))
        valid_indices.append(row_index)
    stacked = (
        torch.stack(valid)
        if valid
        else torch.empty((0, int(rows.shape[1])), dtype=torch.float32)
    )
    return (
        stacked,
        {
            "n_total": int(rows.shape[0]),
            "n_valid": len(valid),
            "skipped_nonfinite": skipped_nonfinite,
            "skipped_zero_norm": skipped_zero_norm,
        },
        valid_indices,
    )


def _failed_angle_record(plan: Any, reason: str) -> dict[str, Any]:
    """Build one retained failed scheduled-angle record."""
    # Preserve plan identity while separating numerical failure from instability.
    return {
        "plan_digest": str(plan.digest),
        "seed": int(plan.seed),
        "replicate_id": int(plan.replicate_id),
        "status": "failed",
        "failure": str(reason),
    }


def _skipped_angle_record(plan: Any, reason: str) -> dict[str, Any]:
    """Build one retained sticky-unavailability angle record."""
    # A prior rank/numerical failure makes later scheduled draws explicitly skipped.
    return {
        "plan_digest": str(plan.digest),
        "seed": int(plan.seed),
        "replicate_id": int(plan.replicate_id),
        "status": "skipped",
        "reason": str(reason),
    }


def _counters(
    *,
    requested: int = 0,
    valid: int = 0,
    failed: int = 0,
    non_applicable: int = 0,
    skipped: int = 0,
) -> dict[str, int]:
    """Return the closed deterministic protocol counter vocabulary."""
    # Emit every key even at zero so equality never depends on omitted counters.
    return {
        "requested": int(requested),
        "valid": int(valid),
        "failed": int(failed),
        "non_applicable": int(non_applicable),
        "skipped": int(skipped),
    }


def _plan_digest(plans: Sequence[Any]) -> str:
    """Hash ordered scheduled-angle identities into one protocol digest."""
    # Preserve plan order and return the empty SHA256 for deliberate no-work schedules.
    joined = "".join(str(plan.digest) for plan in plans)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _kernel_from_unit_rows(
    unit_rows: torch.Tensor, gram: torch.Tensor | None = None
) -> torch.Tensor:
    """Build and symmetrize a unit-row diagnostic kernel."""
    # The optional Euclidean branch is retained only for logit equivalence diagnostics.
    if int(unit_rows.shape[0]) == 0:
        return torch.empty((0, 0), dtype=torch.float64)
    rows64 = unit_rows.to(dtype=torch.float64)
    kernel = (
        rows64 @ rows64.T
        if gram is None
        else rows64 @ gram.to(dtype=torch.float64) @ rows64.T
    )
    return (kernel + kernel.T) / 2.0


def _center_rows(rows: torch.Tensor) -> torch.Tensor:
    """Center rows in CPU float64 before the dense residual angle kernel."""
    # Preserve the existing centered-residual numerical dtype boundary.
    if int(rows.shape[0]) == 0:
        return rows.detach().cpu().to(dtype=torch.float32)
    rows64 = rows.detach().cpu().to(dtype=torch.float64)
    return (rows64 - rows64.mean(dim=0, keepdim=True)).to(dtype=torch.float32)


def _valid_logit_indices(rows: torch.Tensor, eps: float) -> list[int]:
    """Return the ordered finite nonzero logit-row inventory."""
    # Match normalize_logit_deltas validity without reconstructing its output values.
    if rows.ndim != 2:
        raise ValueError(f"logit rows must be rank-2, got {tuple(rows.shape)}.")
    indices: list[int] = []
    for row_index, row in enumerate(rows.detach().cpu().to(dtype=torch.float32)):
        if not torch.isfinite(row).all().item():
            continue
        norm = torch.linalg.vector_norm(row)
        if torch.isfinite(norm).item() and float(norm) > float(eps):
            indices.append(row_index)
    return indices


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    """Return NumPy's default-linear finite quantile."""
    # Exclude non-finite diagnostics without changing the requested quantile estimator.
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.quantile(np.asarray(finite, dtype=np.float64), float(quantile)))


def _json_float(value: float | None) -> float | None:
    """Convert one finite scientific scalar to its JSON representation."""
    # Non-finite numerical diagnostics remain explicit unavailable values.
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)
