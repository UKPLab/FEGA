from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CRayResult:
    c_ray: float | None
    r2: float | None
    n_total: int
    n_valid: int
    skipped_nonfinite: int
    skipped_zero_norm: int
    s_norm: float
    s_norm_sq: float
    principal_direction: torch.Tensor | None = None


@dataclass(frozen=True)
class SpanSpectrumResult:
    n_total: int
    n_valid: int
    skipped_nonfinite: int
    skipped_zero_norm: int
    eigenvalues: list[float]
    s_span: dict[int, float | None]
    u_span: dict[int, float | None]
    d_span: dict[int, float | None]
    b_axis: float | None


@dataclass(frozen=True)
class CenteredResidualSpectrumResult:
    n_total: int
    n_valid: int
    skipped_nonfinite: int
    skipped_zero_norm: int
    eigenvalues: list[float]
    e_res: float | None
    s_res: dict[int, float | None]


@dataclass(frozen=True)
class EffectiveRankResult:
    r_ent: float | None
    r_pr: float | None


def normalize_logit_deltas(
    deltas: torch.Tensor | Sequence[torch.Tensor], *, eps: float
) -> tuple[torch.Tensor, dict[str, int]]:
    """Validate and Euclidean-normalize raw logit deltas row-wise."""
    eps_f = _validate_eps(eps, "eps")
    rows = _as_rows(deltas, name="delta")
    valid: list[torch.Tensor] = []
    skipped_nonfinite = 0
    skipped_zero_norm = 0
    width: int | None = None
    output_dtype: torch.dtype | None = None
    for row in rows:
        if row.ndim != 1:
            raise ValueError(f"Each delta must be 1D, got shape {tuple(row.shape)}.")
        if row.device.type != "cpu":
            raise ValueError(f"Each delta must be a CPU tensor (got {row.device}).")
        if row.dtype not in {torch.float32, torch.float64}:
            raise ValueError(
                "Each delta must be torch.float32 or torch.float64 "
                f"(got {row.dtype})."
            )
        if output_dtype is None:
            output_dtype = row.dtype
        elif row.dtype != output_dtype:
            raise ValueError("All deltas must share the same dtype.")
        if width is None:
            width = int(row.numel())
        elif int(row.numel()) != width:
            raise ValueError(
                f"All deltas must share the same shape; got width {width} vs {row.numel()}."
            )
        if not torch.isfinite(row).all().item():
            skipped_nonfinite += 1
            continue
        norm = torch.linalg.vector_norm(row)
        if not torch.isfinite(norm).item():
            skipped_nonfinite += 1
            continue
        norm_f = float(norm)
        if norm_f <= eps_f:
            skipped_zero_norm += 1
            continue
        valid.append(row.to(device="cpu", dtype=output_dtype).mul(1.0 / norm_f))
    if valid:
        normalized = torch.stack(valid)
    else:
        normalized = torch.empty(
            (0, width or 0), dtype=output_dtype or torch.float32
        )
    return normalized, {
        "n_total": len(rows),
        "n_valid": len(valid),
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_zero_norm": skipped_zero_norm,
    }


@torch.no_grad()
def c_ray_pairwise_final_resid(
    directions: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> CRayResult:
    """Compute pairwise c_ray with the residual Gram."""
    valid, counts = _valid_final_resid_rows(directions, gram, eps=eps)
    n_valid = int(counts["n_valid"])
    if n_valid == 0:
        return CRayResult(
            c_ray=None,
            r2=None,
            s_norm=0.0,
            s_norm_sq=0.0,
            principal_direction=None,
            **counts,
        )

    gram_f = _validate_gram(gram, width=valid.shape[1])
    s_norm_sq, s_norm = _g_norm_sq_and_norm(valid.sum(dim=0), gram_f)
    r2 = s_norm_sq / float(n_valid * n_valid)
    if n_valid < 2:
        return CRayResult(
            c_ray=None,
            r2=r2,
            s_norm=s_norm,
            s_norm_sq=s_norm_sq,
            principal_direction=None,
            **counts,
        )

    k_matrix = valid @ gram_f @ valid.T
    offdiag_sum = float((k_matrix.sum() - torch.trace(k_matrix)).item())
    value = offdiag_sum / float(n_valid * (n_valid - 1))
    principal_direction = (
        valid.sum(dim=0).div(s_norm).to(dtype=torch.float32) if s_norm > 0.0 else None
    )
    return CRayResult(
        c_ray=value,
        r2=r2,
        s_norm=s_norm,
        s_norm_sq=s_norm_sq,
        principal_direction=principal_direction,
        **counts,
    )


@torch.no_grad()
def c_ray_fast_final_resid(
    directions: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> CRayResult:
    """Compute the sum-vector c_ray approximation with the residual Gram."""
    valid, counts = _valid_final_resid_rows(directions, gram, eps=eps)
    n_valid = int(counts["n_valid"])
    if n_valid == 0:
        return CRayResult(
            c_ray=None,
            r2=None,
            s_norm=0.0,
            s_norm_sq=0.0,
            principal_direction=None,
            **counts,
        )

    gram_f = _validate_gram(gram, width=valid.shape[1])
    s_vec = valid.sum(dim=0)
    s_norm_sq, s_norm = _g_norm_sq_and_norm(s_vec, gram_f)
    r2 = s_norm_sq / float(n_valid * n_valid)
    if n_valid < 2:
        return CRayResult(
            c_ray=None,
            r2=r2,
            s_norm=s_norm,
            s_norm_sq=s_norm_sq,
            principal_direction=None,
            **counts,
        )

    value = (s_norm_sq - float(n_valid)) / float(n_valid * (n_valid - 1))
    principal_direction = (
        s_vec.div(s_norm).to(dtype=torch.float32) if s_norm > 0.0 else None
    )
    return CRayResult(
        c_ray=value,
        r2=r2,
        s_norm=s_norm,
        s_norm_sq=s_norm_sq,
        principal_direction=principal_direction,
        **counts,
    )


@torch.no_grad()
def span_spectrum_final_resid(
    directions: torch.Tensor,
    gram: torch.Tensor,
    *,
    k_values: Sequence[int],
    eps: float,
) -> SpanSpectrumResult:
    """Compute span diagnostics with dual-spectrum ambient zero completion.

    The eigensolver returns the complete ``n_valid`` dual spectrum.  Eigenvalue
    indices beyond that spectrum are nevertheless known zeros while they remain
    inside the ambient direction width; only indices beyond that width are
    unavailable.
    """
    # Validate inputs and retain the ambient width needed to classify spectral tails.
    eps_f = _validate_eps(eps, "eps")
    valid, counts = _valid_final_resid_rows(directions, gram, eps=eps_f)
    k_list = [int(k) for k in k_values]
    if int(counts["n_valid"]) == 0:
        return SpanSpectrumResult(
            eigenvalues=[],
            s_span={k: None for k in k_list},
            u_span={k: None for k in k_list},
            d_span={k: None for k in k_list},
            b_axis=None,
            **counts,
        )

    gram_f = _validate_gram(gram, width=valid.shape[1]).to(dtype=torch.float64)
    valid64 = valid.to(dtype=torch.float64)
    k_matrix = valid64 @ gram_f @ valid64.T
    k_matrix = (k_matrix + k_matrix.T) / 2.0
    eigvals_raw, eigvecs_raw = torch.linalg.eigh(k_matrix)
    eigvals = eigvals_raw.to(dtype=torch.float64)
    sort_idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs_raw[:, sort_idx].to(dtype=torch.float64)
    if eigvals.numel():
        min_eval = float(eigvals.min().item())
        if min_eval < -1.0e-5:
            raise ValueError(
                "Span Gram matrix has materially negative eigenvalue "
                f"{min_eval:.6g}; cannot compute directional span."
            )
        eigvals = torch.clamp(eigvals, min=0.0)

    total = float(eigvals.sum().item())
    s_span: dict[int, float | None] = {}
    u_span: dict[int, float | None] = {}
    d_span: dict[int, float | None] = {}
    denom = total + eps_f
    n_eigvals = int(eigvals.numel())
    ambient_width = int(valid.shape[1])
    for k in k_list:
        take = min(k, n_eigvals)
        s_span[k] = float(eigvals[:take].sum().item() / denom) if denom > 0 else None
        idx = k - 1
        if idx < 0 or idx >= ambient_width:
            u_span[k] = None
            d_span[k] = None
            continue

        lambda_k = float(eigvals[idx].item()) if idx < n_eigvals else 0.0
        u_span[k] = float(lambda_k / denom)
        next_idx = idx + 1
        if next_idx >= ambient_width:
            d_span[k] = None
        else:
            lambda_next = (
                float(eigvals[next_idx].item()) if next_idx < n_eigvals else 0.0
            )
            d_span[k] = float(lambda_next / (lambda_k + eps_f))

    b_axis: float | None = None
    if n_eigvals > 0:
        lambda_1 = float(eigvals[0].item())
        if lambda_1 > eps_f:
            q_1 = eigvecs[:, 0]
            projections = (
                k_matrix.to(dtype=torch.float64) @ q_1
            ) / math.sqrt(lambda_1 + eps_f)
            n_valid = float(counts["n_valid"])
            neg_frac = float((projections < 0.0).sum().item()) / n_valid
            pos_frac = float((projections > 0.0).sum().item()) / n_valid
            b_axis = min(neg_frac, pos_frac)
    return SpanSpectrumResult(
        eigenvalues=[float(v) for v in eigvals.tolist()],
        s_span=s_span,
        u_span=u_span,
        d_span=d_span,
        b_axis=b_axis,
        **counts,
    )


@torch.no_grad()
def centered_residual_spectrum_final_resid(
    directions: torch.Tensor,
    gram: torch.Tensor,
    *,
    k_values: Sequence[int],
    eps: float,
) -> CenteredResidualSpectrumResult:
    """Compute the centered residual spectrum for supported FEGA cutoffs.

    Centered-residual concentration is defined only for ``k`` in
    ``{1, 2, 3, 4}``.  This metric API validates that contract before filtering
    rows so direct callers cannot request unsupported fields, including when
    every input row would otherwise be skipped.
    """
    # Reject unsupported centered-residual cutoffs at the public metric boundary.
    eps_f = _validate_eps(eps, "eps")
    k_list = [int(k) for k in k_values]
    invalid_k_values = [k for k in k_list if k not in {1, 2, 3, 4}]
    if invalid_k_values:
        raise ValueError(
            "Centered residual spectrum k_values must contain only "
            f"{{1, 2, 3, 4}}; got unsupported values {invalid_k_values}."
        )

    # Filter unusable rows only after the caller's requested cutoffs are valid.
    valid, counts = _valid_final_resid_rows(directions, gram, eps=eps_f)
    if int(counts["n_valid"]) == 0:
        return CenteredResidualSpectrumResult(
            eigenvalues=[],
            e_res=None,
            s_res={k: None for k in k_list},
            **counts,
        )

    gram_f = _validate_gram(gram, width=valid.shape[1]).to(dtype=torch.float64)
    valid64 = valid.to(dtype=torch.float64)
    k_matrix = valid64 @ gram_f @ valid64.T
    n = int(k_matrix.shape[0])
    ones_n = torch.ones((n, 1), dtype=torch.float64)
    h_matrix = torch.eye(n, dtype=torch.float64) - (ones_n @ ones_n.T) / float(n)
    k_ctr = h_matrix @ k_matrix @ h_matrix
    k_ctr = (k_ctr + k_ctr.T) / 2.0

    eigvals = torch.linalg.eigvalsh(k_ctr).to(dtype=torch.float64)
    eigvals = torch.sort(eigvals, descending=True).values
    if eigvals.numel():
        min_eval = float(eigvals.min().item())
        if min_eval < -1.0e-5:
            raise ValueError(
                "Centered residual Gram matrix has materially negative eigenvalue "
                f"{min_eval:.6g}; cannot compute centered residual spectrum."
            )
        eigvals = torch.clamp(eigvals, min=0.0)

    total = float(eigvals.sum().item())
    denom = total + eps_f
    s_res: dict[int, float | None] = {}
    for k in k_list:
        take = min(k, int(eigvals.numel()))
        s_res[k] = float(eigvals[:take].sum().item() / denom) if denom > 0 else None

    trace_k = float(torch.trace(k_matrix).item())
    trace_ctr = float(torch.trace(k_ctr).item())
    e_res = trace_ctr / (trace_k + eps_f)
    return CenteredResidualSpectrumResult(
        eigenvalues=[float(v) for v in eigvals.tolist()],
        e_res=float(e_res),
        s_res=s_res,
        **counts,
    )


def effective_rank_from_spectrum(
    spectrum: Sequence[float], *, eps: float
) -> EffectiveRankResult:
    """Compute literal visible-epsilon effective ranks for a nonnegative spectrum.

    Any finite spectrum with positive total mass is evaluated, including totals
    smaller than ``eps``.  The literal weights remain
    ``p_i = s_i / (sum(s) + eps)``.  Participation rank uses the direct source
    formula for ordinary inputs and a scaled logarithmic equivalent only when
    every floating-point ``p_i**2`` underflows; an exact reciprocal beyond the
    representable float range is reported as infinity.  Empty and all-zero
    spectra have no defined effective rank and return null diagnostics.
    """
    # Validate every value before deciding whether the spectrum carries positive mass.
    eps_f = _validate_eps(eps, "eps")
    values = [float(value) for value in spectrum]
    if not values:
        return EffectiveRankResult(r_ent=None, r_pr=None)

    total = 0.0
    max_value = 0.0
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Effective-rank spectrum values must be finite.")
        if value < 0.0:
            raise ValueError("Effective-rank spectrum values must be nonnegative.")
        total += value
        max_value = max(max_value, value)
    if max_value == 0.0:
        return EffectiveRankResult(r_ent=None, r_pr=None)

    # Apply the source formulas directly to visible-epsilon normalized weights.
    entropy = 0.0
    denom = total + eps_f
    scaled_denom = (
        math.fsum(candidate / max_value for candidate in values)
        + eps_f / max_value
        if not math.isfinite(denom)
        else None
    )
    squared_probability_sum = 0.0
    probabilities: list[float] = []
    for value in values:
        if math.isfinite(denom):
            p_value = value / denom
        else:
            assert scaled_denom is not None
            p_value = (value / max_value) / scaled_denom
        probabilities.append(p_value)
        entropy += p_value * math.log(p_value + eps_f)
        squared_probability_sum += p_value * p_value

    # Recover the reciprocal through scale/log space if all squared weights underflow.
    if squared_probability_sum > 0.0:
        reciprocal_overflows = squared_probability_sum < (
            1.0 / float.fromhex("0x1.fffffffffffffp+1023")
        )
        r_pr = math.inf if reciprocal_overflows else 1.0 / squared_probability_sum
    else:
        max_probability = max(probabilities)
        if max_probability == 0.0:
            r_pr = math.inf
        else:
            scaled_square_sum = math.fsum(
                (p_value / max_probability) ** 2 for p_value in probabilities
            )
            log_r_pr = -2.0 * math.log(max_probability) - math.log(
                scaled_square_sum
            )
            max_float_log = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
            r_pr = math.inf if log_r_pr > max_float_log else math.exp(log_r_pr)

    return EffectiveRankResult(
        r_ent=float(math.exp(-entropy)),
        r_pr=float(r_pr),
    )


def _valid_final_resid_rows(
    directions: torch.Tensor, gram: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, dict[str, int]]:
    eps_f = _validate_eps(eps, "eps")
    rows = _as_rows(directions, name="direction")
    if not rows:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            {
                "n_total": 0,
                "n_valid": 0,
                "skipped_nonfinite": 0,
                "skipped_zero_norm": 0,
            },
        )
    width = int(rows[0].numel())
    gram_f = _validate_gram(gram, width=width)
    valid: list[torch.Tensor] = []
    skipped_nonfinite = 0
    skipped_zero_norm = 0
    for row in rows:
        _validate_cpu_float32_row(row, "direction")
        if int(row.numel()) != width:
            raise ValueError(
                f"All directions must share the same shape; got width {width} vs {row.numel()}."
            )
        if not torch.isfinite(row).all().item():
            skipped_nonfinite += 1
            continue
        norm_sq, norm = _g_norm_sq_and_norm(row, gram_f)
        if not math.isfinite(norm_sq) or not math.isfinite(norm):
            skipped_nonfinite += 1
            continue
        if norm <= eps_f:
            skipped_zero_norm += 1
            continue
        valid.append(row.to(device="cpu", dtype=gram_f.dtype))
    if valid:
        stacked = torch.stack(valid)
    else:
        stacked = torch.empty((0, width), dtype=gram_f.dtype)
    return stacked, {
        "n_total": len(rows),
        "n_valid": len(valid),
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_zero_norm": skipped_zero_norm,
    }


def _as_rows(
    values: torch.Tensor | Sequence[torch.Tensor], *, name: str
) -> list[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        if values.ndim == 1:
            return [values]
        if values.ndim == 2:
            return [row for row in values]
        raise ValueError(f"`{name}` rows must be rank-1 or rank-2, got {values.ndim}.")
    return list(values)


def _validate_cpu_float32_row(row: torch.Tensor, name: str) -> None:
    if row.ndim != 1:
        raise ValueError(f"Each {name} must be 1D, got shape {tuple(row.shape)}.")
    if row.device.type != "cpu":
        raise ValueError(f"Each {name} must be a CPU tensor (got {row.device}).")
    if row.dtype != torch.float32:
        raise ValueError(f"Each {name} must be torch.float32 (got {row.dtype}).")


def _validate_gram(gram: torch.Tensor, *, width: int) -> torch.Tensor:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"Gram tensor must be square rank-2, got {tuple(gram.shape)}.")
    if int(gram.shape[0]) != int(width):
        raise ValueError(
            f"Gram width {gram.shape[0]} does not match direction width {width}."
        )
    if gram.device.type != "cpu":
        gram = gram.to(device="cpu")
    if gram.dtype == torch.float64:
        return gram
    return gram.to(dtype=torch.float32)


def _g_norm_sq_and_norm(row: torch.Tensor, gram: torch.Tensor) -> tuple[float, float]:
    row_f = row.to(dtype=gram.dtype)
    norm_sq = float((row_f @ gram @ row_f).item())
    if norm_sq < 0.0 and norm_sq > -1.0e-7:
        norm_sq = 0.0
    norm = math.sqrt(norm_sq) if norm_sq >= 0.0 else float("nan")
    return norm_sq, norm


def _validate_eps(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"`{name}` must be a float > 0, got {value!r}.") from exc
    if parsed <= 0.0:
        raise ValueError(f"`{name}` must be > 0, got {value!r}.")
    return parsed
