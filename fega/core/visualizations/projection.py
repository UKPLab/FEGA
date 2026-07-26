"""Gram-equivalent spectral coordinates for cached residual directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralProjection:
    """One deterministic spectral embedding of a Gram-space kernel."""

    coordinates: torch.Tensor
    kernel: torch.Tensor
    eigenvalues: torch.Tensor
    explained_ratios: torch.Tensor
    numerical_rank: int
    centered: bool


@torch.no_grad()
def project_directions(
    directions: torch.Tensor,
    gram: torch.Tensor,
    *,
    dimensions: int,
    centered: bool = False,
) -> SpectralProjection:
    """Project directions from the exact kernel ``D G D^T``.

    Coordinates are uncentered unless ``centered`` is requested. Eigenvector
    signs are fixed from the first maximum-magnitude coordinate so repeated
    runs do not introduce arbitrary axis reflections.
    """
    # Establish the matrix dimensions required by the Gram-equivalent projection.
    if directions.ndim != 2 or directions.shape[0] == 0:
        raise ValueError("directions must be a non-empty rank-2 tensor")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    width = int(directions.shape[1])
    if gram.ndim != 2 or tuple(gram.shape) != (width, width):
        raise ValueError("Gram shape must match the direction width")

    # Form and symmetrize the exact logit-equivalent kernel in float64.
    directions64 = directions.detach().cpu().to(dtype=torch.float64)
    gram64 = gram.detach().cpu().to(dtype=torch.float64)
    kernel = directions64 @ gram64 @ directions64.T
    kernel = (kernel + kernel.T) / 2.0
    if centered:
        row_mean = kernel.mean(dim=1, keepdim=True)
        kernel = kernel - row_mean - row_mean.T + kernel.mean()
        kernel = (kernel + kernel.T) / 2.0
    if not torch.isfinite(kernel).all():
        raise ValueError("projection kernel contains non-finite values")

    # Match geometry metrics by rejecting material negative spectrum and clamping roundoff.
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    minimum = float(eigenvalues.min().item())
    if minimum < -1.0e-5:
        raise ValueError(
            f"projection kernel has a materially negative eigenvalue {minimum:.6g}"
        )
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # Convert the leading eigensystem to coordinates and fix each free axis sign.
    take = min(dimensions, int(eigenvalues.numel()))
    coordinates = eigenvectors[:, :take] * torch.sqrt(eigenvalues[:take])
    for column_index in range(take):
        column = coordinates[:, column_index]
        anchor_index = int(torch.argmax(torch.abs(column)).item())
        if float(column[anchor_index].item()) < 0.0:
            coordinates[:, column_index] = -column
    if take < dimensions:
        padding = torch.zeros(
            (coordinates.shape[0], dimensions - take), dtype=torch.float64
        )
        coordinates = torch.cat((coordinates, padding), dim=1)

    # Report the effective numerical support and per-axis share of kernel trace.
    total = float(eigenvalues.sum().item())
    ratios = eigenvalues / total if total > 0.0 else torch.zeros_like(eigenvalues)
    largest = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
    rank_tolerance = max(1.0e-12, largest * 1.0e-12)
    numerical_rank = int((eigenvalues > rank_tolerance).sum().item())
    return SpectralProjection(
        coordinates=coordinates,
        kernel=kernel,
        eigenvalues=eigenvalues,
        explained_ratios=ratios,
        numerical_rank=numerical_rank,
        centered=centered,
    )


def surface_coordinates(
    coordinates: torch.Tensor, *, tolerance: float = 1.0e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalize nonzero projected rows for the display-only sphere surface."""
    # Keep the omission mask explicit so faithful and display-only panels cannot mix.
    norms = torch.linalg.vector_norm(coordinates, dim=1)
    keep = norms > tolerance
    surface = coordinates[keep] / norms[keep, None]
    return surface, keep
