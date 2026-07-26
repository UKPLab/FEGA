import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path):
    """Create directory if missing (no error if it exists)."""
    os.makedirs(path, exist_ok=True)


def resolve_path(
    base: Path, candidate: str | Path, secondary_base: Path | None = None
) -> Path:
    """Resolve a candidate path relative to base/secondary_base, or raise if missing."""
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return candidate_path

    primary = (base / candidate_path).resolve()
    if primary.exists():
        return primary

    if secondary_base is not None:
        secondary = (secondary_base / candidate_path).resolve()
        if secondary.exists():
            return secondary

    tried = [str(primary)]
    if secondary_base is not None:
        tried.append(str((secondary_base / candidate_path).resolve()))
    raise FileNotFoundError(
        f"Could not resolve path for {candidate} (tried {', '.join(tried)})"
    )


def seed_everything(seed):
    """Seed python/random, numpy, and torch (including CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_json(obj: Any):
    """Recursively convert common array/tensor/path types into JSON-friendly objects."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return to_json(obj.detach().cpu().tolist())
    return obj
