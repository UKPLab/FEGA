from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_discovery_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("feature_sets"), dict):
        raise ValueError(f"Discovery summary has no feature_sets map: {path}")
    payload["_summary_path"] = str(path.resolve())
    return payload


def resolve_feature_set(
    summary: dict[str, Any], sae_uid: str, feature_set: str
) -> list[int]:
    try:
        entry = summary["feature_sets"][sae_uid]
    except KeyError as exc:
        raise ValueError(
            f"SAE UID {sae_uid!r} not present in discovery summary"
        ) from exc
    field = {
        "threshold": "threshold_feature_ids",
        "candidate": "candidate_feature_ids",
        "strict": "strict_common_feature_ids",
        "strict_common": "strict_common_feature_ids",
    }.get(feature_set)
    if field is None:
        raise ValueError(f"Unknown feature set {feature_set!r}")
    values = entry.get(field)
    if values is None and field == "threshold_feature_ids":
        values = entry.get("candidate_feature_ids")
    if not isinstance(values, list):
        raise ValueError(f"Feature set field {field!r} is missing for {sae_uid}")
    return sorted({int(value) for value in values})


def per_sae_metrics_path(summary: dict[str, Any], sae_uid: str) -> Path:
    filename = f"{sae_uid}_feature_metrics.csv"
    candidates: list[Path] = []
    summary_path_value = summary.get("_summary_path")
    if summary_path_value:
        summary_dir = Path(summary_path_value).parent
        candidates.append(summary_dir / "per_sae" / filename)
    else:
        summary_dir = None

    output_dir = Path(summary["output_dir"])
    if not output_dir.is_absolute() and summary_dir is not None:
        candidates.append(summary_dir / output_dir / "per_sae" / filename)
    candidates.append(output_dir / "per_sae" / filename)

    unique_candidates = list(dict.fromkeys(candidates))
    for path in unique_candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Per-SAE feature metrics not found; checked: {checked}")


def matched_random_feature_sets(
    *,
    metrics: pd.DataFrame,
    selected_feature_ids: list[int],
    trials: int,
    seed: int,
    match_pool_size: int,
) -> list[list[int]]:
    """Match controls by prevalence and active magnitude using randomized neighbors."""
    if trials <= 0:
        return []
    if not selected_feature_ids:
        return [[] for _ in range(trials)]
    indexed = metrics.set_index("feature_id", drop=False)
    missing = sorted(set(selected_feature_ids) - set(indexed.index.astype(int)))
    if missing:
        raise ValueError(f"Selected feature IDs missing from metrics: {missing[:10]}")
    selected = indexed.loc[selected_feature_ids]
    excluded = set(selected_feature_ids)
    pool = metrics[~metrics["feature_id"].astype(int).isin(excluded)].copy()
    if len(pool) < len(selected_feature_ids):
        raise ValueError("Not enough non-selected SAE features for random controls")

    prevalence = pool["example_prevalence"].to_numpy(dtype=np.float64)
    magnitude = np.log1p(
        pool["mean_activation_when_active"].clip(lower=0).to_numpy(dtype=np.float64)
    )
    prevalence_scale = max(float(np.std(prevalence)), 1e-6)
    magnitude_scale = max(float(np.std(magnitude)), 1e-6)
    pool_ids = pool["feature_id"].to_numpy(dtype=np.int64)
    selected_targets = [
        (
            float(row["example_prevalence"]),
            math_log1p_nonnegative(float(row["mean_activation_when_active"])),
        )
        for _, row in selected.iterrows()
    ]

    output: list[list[int]] = []
    for trial_index in range(trials):
        rng = random.Random(seed + trial_index)
        available = np.ones(len(pool_ids), dtype=bool)
        chosen: list[int] = []
        order = list(range(len(selected_targets)))
        rng.shuffle(order)
        for target_index in order:
            target_prevalence, target_magnitude = selected_targets[target_index]
            distances = ((prevalence - target_prevalence) / prevalence_scale) ** 2 + (
                (magnitude - target_magnitude) / magnitude_scale
            ) ** 2
            candidate_indices = np.flatnonzero(available)
            ranked = candidate_indices[np.argsort(distances[candidate_indices])]
            neighborhood = ranked[: max(1, min(match_pool_size, len(ranked)))]
            selected_index = int(rng.choice(neighborhood.tolist()))
            available[selected_index] = False
            chosen.append(int(pool_ids[selected_index]))
        output.append(sorted(chosen))
    return output


def math_log1p_nonnegative(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))
