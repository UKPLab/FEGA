from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

from fega.core.geometry_reporting.map.schema import MAP_VECTOR_KEYS, MISSINGNESS_KEYS


def embed(
    rows: list[dict[str, Any]], *, embedding: str, seed: int | None
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    matrix, preprocessing = preprocess_embedding_matrix(rows)
    metadata: dict[str, Any] = {
        "requested": embedding,
        "method": None,
        "seed": seed,
        "effective_seed": effective_seed(seed),
    }
    if not rows:
        metadata["method"] = "empty"
        return np.empty((0, 2), dtype=np.float64), metadata, preprocessing
    if len(rows) == 1:
        metadata["method"] = "single_point"
        return np.zeros((1, 2), dtype=np.float64), metadata, preprocessing
    if embedding in {"auto", "umap"}:
        params = umap_params(len(rows), seed)
        coords = fit_umap(matrix, params)
        metadata["method"] = "umap"
        metadata["umap_params"] = params
        return coords.astype(np.float64, copy=False), metadata, preprocessing
    if embedding == "tsne" and len(rows) >= 4:
        perplexity = max(2, min(30, (len(rows) - 1) // 3))
        if perplexity < len(rows):
            coords = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="random",
                learning_rate="auto",
                random_state=seed,
            ).fit_transform(matrix)
            metadata["method"] = "tsne"
            metadata["tsne_params"] = {
                "n_components": 2,
                "perplexity": perplexity,
                "init": "random",
                "learning_rate": "auto",
                "random_state": seed,
            }
            return coords.astype(np.float64, copy=False), metadata, preprocessing
    n_components = min(2, matrix.shape[0], matrix.shape[1])
    coords = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    if n_components == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    metadata["method"] = "pca"
    metadata["pca_params"] = {
        "n_components": int(n_components),
        "random_state": seed,
    }
    return coords.astype(np.float64, copy=False), metadata, preprocessing


def preprocess_embedding_matrix(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(
        [
            [
                np.nan if row["vector"][key] is None else float(row["vector"][key])
                for key in MAP_VECTOR_KEYS
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if raw.size == 0:
        missing = np.zeros((len(rows), len(MAP_VECTOR_KEYS)), dtype=bool)
        raw = np.empty((len(rows), len(MAP_VECTOR_KEYS)), dtype=np.float64)
    else:
        missing = ~np.isfinite(raw)
    imputed = raw.copy()
    fill_values: dict[str, float] = {}
    all_missing_fields: list[str] = []
    for idx, key in enumerate(MAP_VECTOR_KEYS):
        observed = raw[~missing[:, idx], idx]
        if observed.size:
            fill = float(np.median(observed))
        else:
            fill = 0.0
            all_missing_fields.append(key)
        fill_values[key] = fill
        imputed[missing[:, idx], idx] = fill

    if len(rows):
        scaler = RobustScaler()
        scaled = scaler.fit_transform(imputed)
        centers = [float(value) for value in scaler.center_]
        scales = [float(value) for value in scaler.scale_]
        q75, q25 = np.percentile(imputed, [75, 25], axis=0)
        degenerate_fields = [
            key
            for key, value in zip(MAP_VECTOR_KEYS, q75 - q25, strict=True)
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
    else:
        scaled = imputed
        centers = [0.0 for _ in MAP_VECTOR_KEYS]
        scales = [1.0 for _ in MAP_VECTOR_KEYS]
        degenerate_fields = list(MAP_VECTOR_KEYS)

    matrix = np.column_stack([scaled, missing.astype(np.float64)])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("geometry_reporting map preprocessing produced non-finite values")
    return matrix, {
        "metric_fields": list(MAP_VECTOR_KEYS),
        "missingness_fields": list(MISSINGNESS_KEYS),
        "imputation": {
            "method": "median",
            "fill_values": fill_values,
            "all_missing_fields": all_missing_fields,
        },
        "scaling": {
            "method": "robust",
            "center": dict(zip(MAP_VECTOR_KEYS, centers, strict=True)),
            "scale": dict(zip(MAP_VECTOR_KEYS, scales, strict=True)),
            "degenerate_fields": degenerate_fields,
        },
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }


def fit_umap(matrix: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "umap-learn is required for geometry_reporting UMAP feature maps. "
            "Install dependency `umap-learn>=0.5.6,<0.6`."
        ) from exc
    coords = UMAP(**params).fit_transform(matrix)
    if not np.all(np.isfinite(coords)):
        raise ValueError("UMAP produced non-finite geometry_reporting coordinates")
    return coords


def umap_params(row_count: int, seed: int | None) -> dict[str, Any]:
    resolved = effective_seed(seed)
    return {
        "n_components": 2,
        "n_neighbors": int(min(15, max(2, row_count - 1))),
        "min_dist": 0.1,
        "metric": "euclidean",
        "init": "random",
        "random_state": resolved,
        "transform_seed": resolved,
    }


def effective_seed(seed: int | None) -> int:
    return 0 if seed is None else int(seed)
