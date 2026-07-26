from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.paths import gram_cache_meta_path, gram_cache_tensor_path

GRAM_CONSTRUCTION_RECIPE = (
    "float64(canonical_unembedding).T@float64(canonical_unembedding); "
    "stored_as_configured_gram_cache_dtype"
)
GRAM_REQUIRED_METADATA = (
    "checkpoint_identity",
    "readout_name",
    "hidden_width",
    "gram_dtype",
    "construction_recipe",
    "unembedding_fingerprint",
    "unembedding_dtype",
    "unembedding_shape",
    "gram_shape",
    "gram_sha256",
)


def canonical_unembedding(model: Any) -> torch.Tensor:
    """Return the output embedding matrix in `[vocab, hidden]` convention."""
    output_embeddings = None
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        weight = output_embeddings.weight
    elif hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        weight = model.lm_head.weight
    else:
        raise ValueError("Could not resolve model output embedding weights.")
    if weight.dim() != 2:
        raise ValueError(
            f"Expected rank-2 unembedding weight, got shape {tuple(weight.shape)}."
        )
    return weight.detach()


def unembedding_fingerprint(unembedding: torch.Tensor) -> str:
    """Hash exact canonical unembedding bytes together with dtype and shape."""
    # Canonicalize only device/layout; preserve every stored value and dtype bit.
    tensor = unembedding.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw_bytes = tensor.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def gram_fingerprint(gram: torch.Tensor) -> str:
    """Hash exact persisted Gram bytes together with dtype and shape."""
    # Preserve the stored dtype and every value bit while canonicalizing layout.
    tensor = gram.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw_bytes = tensor.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def write_gram_cache(
    model: Any,
    config: FEGAPipelineConfig,
    *,
    final_resid_width: int | None = None,
) -> Path | None:
    """Materialize a fingerprint-bound dense residual-space Gram cache."""
    # Stop immediately when this run does not request the shared Gram cache.
    data_prep = config.phases.data_prep
    if not data_prep.gram_cache:
        return None

    unembedding = canonical_unembedding(model)
    vocab, hidden = unembedding.shape
    if final_resid_width is not None and final_resid_width != hidden:
        raise ValueError(
            "final_resid hidden width does not match unembedding width: "
            f"{final_resid_width} != {hidden}"
        )

    compute_device = _safe_compute_device(config.device)
    dtype = _torch_dtype(data_prep.gram_cache_dtype)
    w_u = unembedding.to(device=compute_device, dtype=torch.float64)
    gram = (w_u.T @ w_u).to(dtype=dtype).cpu()
    if gram.shape != (hidden, hidden):
        raise ValueError(
            f"Expected Gram shape {(hidden, hidden)}, got {tuple(gram.shape)}."
        )

    tensor_path = gram_cache_tensor_path(config)
    meta_path = gram_cache_meta_path(config)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gram, tensor_path)

    summary_tensor = gram
    checkpoint_identity = _model_name(model, config)
    if not checkpoint_identity:
        raise ValueError("Could not resolve checkpoint identity for Gram metadata.")
    meta = {
        "model_name": checkpoint_identity,
        "checkpoint_identity": checkpoint_identity,
        "readout_name": "final_resid",
        "hidden_width": int(hidden),
        "gram_dtype": data_prep.gram_cache_dtype,
        "construction_recipe": GRAM_CONSTRUCTION_RECIPE,
        "unembedding_fingerprint": unembedding_fingerprint(unembedding),
        "unembedding_dtype": str(unembedding.dtype),
        "unembedding_shape": [int(vocab), int(hidden)],
        "gram_shape": [int(gram.shape[0]), int(gram.shape[1])],
        "gram_sha256": gram_fingerprint(gram),
        "dtype": data_prep.gram_cache_dtype,
        "compute_device": str(compute_device),
        "tensor_path": str(tensor_path),
        "summary": {
            "sum": float(summary_tensor.sum().item()),
            "mean": float(summary_tensor.mean().item()),
            "std": float(summary_tensor.std(unbiased=False).item()),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported gram cache dtype: {name}") from exc


def _safe_compute_device(configured: str) -> torch.device:
    requested = torch.device(configured)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return requested


def _model_name(model: Any, config: FEGAPipelineConfig) -> str | None:
    model_config = getattr(model, "config", None)
    return (
        getattr(model_config, "name_or_path", None)
        or getattr(model_config, "_name_or_path", None)
        or getattr(model, "name_or_path", None)
        or getattr(config, "sae_repo_id", None)
    )
