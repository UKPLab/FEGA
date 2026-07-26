from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubsetPlan:
    """Immutable, content-addressed description of one stability evaluation subset."""

    global_seed: int
    feature_id: int
    protocol: str
    target_or_group_identity: str
    replicate_id: int
    purpose: str
    indices: tuple[int, ...]
    digest: str


def build_subset_plan(
    *,
    global_seed: int,
    feature_id: int,
    protocol: str,
    target_or_group_identity: str,
    replicate_id: int,
    purpose: str,
    indices: Sequence[int],
) -> SubsetPlan:
    """Create a sorted plan whose SHA256 retains bootstrap index multiplicity."""
    # Canonicalize caller order while retaining duplicate bootstrap observations.
    sorted_indices = tuple(sorted(int(index) for index in indices))
    identity = {
        "global_seed": int(global_seed),
        "feature_id": int(feature_id),
        "protocol": str(protocol),
        "target_or_group_identity": str(target_or_group_identity),
        "replicate_id": int(replicate_id),
        "purpose": str(purpose),
        "indices": sorted_indices,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SubsetPlan(**identity, digest=digest)


def derive_seed(
    base_seed: int, *, feature_id: int, effect_space: str, salt: int = 0
) -> int:
    """Derive a deterministic stable feature seed for a feature/effect-space pair."""
    # Preserve the existing stable arithmetic mapping used by current artifacts.
    space_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(effect_space))
    value = int(base_seed) + int(feature_id) * 104729 + space_offset * 37 + int(salt)
    return int(value % (2**31 - 1))


def subspace_resample_indices(
    n_valid: int, fraction: float, rounds: int, seed: int
) -> list[np.ndarray]:
    if n_valid <= 0 or rounds <= 0:
        return []
    subset_n = max(1, int(math.ceil(float(fraction) * n_valid)))
    subset_n = min(n_valid, subset_n)
    rng = np.random.default_rng(seed)
    return [
        np.sort(rng.choice(n_valid, size=subset_n, replace=False))
        for _ in range(rounds)
    ]


def low_context_protocol(n_valid: int) -> dict[str, str | int]:
    if n_valid < 8:
        return {
            "status": "insufficient_contexts",
            "protocol": "descriptive",
            "n_valid": int(n_valid),
        }
    if n_valid < 16:
        return {
            "status": "exploratory",
            "protocol": "leave_out_sensitivity",
            "n_valid": int(n_valid),
        }
    if n_valid < 32:
        return {
            "status": "exploratory",
            "protocol": "exploratory_subsampling",
            "n_valid": int(n_valid),
        }
    return {"status": "ok", "protocol": "principal_angle", "n_valid": int(n_valid)}
