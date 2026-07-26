from __future__ import annotations

import logging
import json
from pathlib import Path

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.common import require_single_entity_attr
from fega.core.data_prep.collection import _collect_data_prep_artifacts
from fega.core.data_prep.gram_cache import write_gram_cache
from fega.core.data_prep.induction import run_induction_data_prep
from fega.core.data_prep.selection import _run_context_selection
from fega.core.resources import ModelResources
from fega.core.utils import ChunkProcessor
from fega.paths import data_prep_activations_dir

_logger = logging.getLogger(__name__)


def run_data_prep(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> Path:
    """Run collection, context selection, and optional Gram cache as one phase."""
    data_prep = config.phases.data_prep
    if config.source_kind == "ravel":
        manifest_path = _collect_data_prep_artifacts(config, resources)
        contexts_path = _run_context_selection(config, resources)
    elif config.source_kind == "induction":
        manifest_path, contexts_path = run_induction_data_prep(config, resources)
    else:
        raise ValueError(f"Unsupported source_kind: {config.source_kind!r}")

    if data_prep.gram_cache:
        manifest = json.loads(Path(manifest_path).read_text())
        if int(manifest.get("total_records") or 0) == 0:
            _logger.info("Skipping Gram cache because data prep produced zero records.")
            return contexts_path
        gram_resources = resources or ModelResources(config)
        model, _, _ = gram_resources.get_model_and_sae()
        entity, attr = require_single_entity_attr(config)
        activations_dir = data_prep_activations_dir(config, entity, attr)
        final_resid_width = _infer_final_resid_width(manifest_path, activations_dir)
        meta_path = write_gram_cache(model, config, final_resid_width=final_resid_width)
        _logger.info("Wrote Gram cache metadata to %s", meta_path)

    return contexts_path


def _infer_final_resid_width(manifest_path: Path, activations_dir: Path) -> int:
    """Read the first chunk that contains final_resid and return its hidden width."""
    for tensors_path, _ in ChunkProcessor.stream(manifest_path, activations_dir):
        payload = torch.load(tensors_path, map_location="cpu")
        final_resid = payload.get("final_resid")
        if final_resid is None:
            continue
        if final_resid.dim() < 1:
            raise ValueError(
                f"final_resid must have a hidden dimension in {tensors_path}."
            )
        return int(final_resid.shape[-1])
    raise ValueError(
        "Gram cache requires activation chunks with a final_resid readout, "
        f"but none were found via {manifest_path}."
    )
