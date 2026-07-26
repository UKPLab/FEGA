import numpy as np
import torch


def unit_normalize_rows_np(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalize rows of X, masking rows with non-finite or tiny norms."""
    norms = np.linalg.norm(X, axis=1)
    ok_mask = np.isfinite(norms) & (norms > eps)
    X_unit = np.zeros_like(X)
    if np.any(ok_mask):
        denom = np.clip(norms[ok_mask], a_min=eps, a_max=None)
        X_unit[ok_mask] = X[ok_mask] / denom[:, None]
    return X_unit, ok_mask


def unit_normalize_rows_torch(
    X: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows of X, masking rows with non-finite or tiny norms."""
    norms = torch.linalg.norm(X, dim=1)
    ok_mask = torch.isfinite(norms) & (norms > eps)
    X_unit = torch.zeros_like(X)
    if torch.any(ok_mask):
        denom = norms[ok_mask].clamp_min(eps).unsqueeze(1)
        X_unit[ok_mask] = X[ok_mask] / denom
    return X_unit, ok_mask
