import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers.modeling_outputs import ModelOutput

import sae_bench.sae_bench_utils.activation_collection as activation_collection
from fega.config_schema import FEGAPipelineConfig
from fega.core.common import patched_attr, require_single_entity_attr
from fega.core.config import FEGAConfig
from fega.core.positioning import (
    POSITIONING_SCHEMA_VERSION,
    build_padded_prompt_batch,
    build_positioning_metadata,
)
from fega.core.resources import ModelResources
from fega.core.utils import (
    ChunkProcessor,
    RunPaths,
    load_model_and_sae,
    prompt_from_dict,
    prompt_to_dict,
    setup_run,
)
from fega.core.utils.ravel import (
    ReplayContext,
    build_prompt_pairs,
    filtered_dataset_path,
    load_or_create_filtered_dataset,
)
from fega.paths import data_prep_dir
from sae_bench.evals.ravel.instance import Prompt, RAVELFilteredDataset


@dataclass
class Stats:
    # Collected run-level counters/metadata recorded in collection_meta.json.
    dataset_path: str
    dataset_source: str
    limit: int | None
    batch_size_start: int | None
    batch_size_min: int | None
    skipped_nan: int = 0
    total_prompts_seen: int = 0
    pairs_reloaded: bool = False
    pairs_reload_error: str | None = None
    pairs_reloaded_from: str | None = None
    pairs_generated_for: list[str] = field(default_factory=list)
    oom_errors: int = 0
    collected_roles: list[str] = field(default_factory=lambda: ["cause_base_prompts"])
    truncated_by_limit: bool = False
    saved_records: int = 0
    batch_size_adjustments: list[dict] = field(default_factory=list)
    chunk_count: int | None = None
    chunk_size: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollectionMeta:
    # Container for the summary written to collection_meta.json.
    total_records: int
    pairs_counts: dict[str, dict[str, int]]
    limit: int | None
    entity_attribute_selection: dict[str, list[str]]
    reference_json: str
    pairs_full_path: str
    manifest_path: str
    positioning: dict[str, Any]
    stats: Stats

    def as_dict(self) -> dict:
        data = asdict(self)
        data["stats"] = self.stats.as_dict()
        return data


def load_filtered_dataset_if_cached(
    ctx: ReplayContext,
    entity_class: str,
    tokenizer,
    model,
    selected_attributes: list[str],
) -> tuple[RAVELFilteredDataset, Path, str]:
    """Prefer cached filtered dataset on disk; otherwise rebuild with RAVEL helpers."""
    ds_path = filtered_dataset_path(ctx, entity_class)
    if ds_path.exists():
        print(f"Use RAVELFilteredDataset to load from path: {ds_path}")
        return RAVELFilteredDataset.load(str(ds_path)), ds_path, "loaded"
    dataset = load_or_create_filtered_dataset(
        ctx,
        entity_class,
        tokenizer=tokenizer,
        model=model,
        force_recompute=False,
        selected_attributes=selected_attributes,
    )
    return dataset, ds_path, "recomputed"


def load_or_build_pairs(
    fega_cfg: FEGAConfig,
    ctx: ReplayContext,
    dataset: RAVELFilteredDataset,
    entity_class: str,
    attrs: list[str],
    model,
    stats: Stats,
    pairs_cache_path: Path,
) -> dict[str, dict[str, list[Prompt]]]:
    """Reload pairs from common locations; otherwise generate native RAVEL pairs."""
    candidate_pairs = [
        Path(fega_cfg.eval_config.artifact_dir) / f"{entity_class}_pairs_full.json",
        Path(fega_cfg.reference_json).parent / f"{entity_class}_pairs_full.json",
        pairs_cache_path,
    ]
    expected_pair_len = fega_cfg.eval_config.num_pairs_per_attribute
    for cand in candidate_pairs:
        if not cand.exists():
            continue
        try:
            with open(cand) as f:
                raw = json.load(f)
            loaded_pairs: dict[str, dict[str, list[Prompt]]] = {}
            for attr, pair_dict in raw.items():
                loaded_pairs[attr] = {}
                for key, prompts in pair_dict.items():
                    loaded_pairs[attr][key] = [prompt_from_dict(p) for p in prompts]
            valid_counts = True
            if expected_pair_len is not None:
                for pair_dict in loaded_pairs.values():
                    cause_len = len(pair_dict.get("cause_base_prompts", []))
                    if cause_len and cause_len != expected_pair_len:
                        valid_counts = False
                        break
            if valid_counts:
                stats.pairs_reloaded = True
                stats.pairs_reloaded_from = str(cand)
                return loaded_pairs
            stats.pairs_reload_error = f"pair count mismatch in {cand}"
        except Exception as exc:
            stats.pairs_reload_error = f"failed to load pairs_full.json at {cand}"
            print(
                f"Warning: failed to reload pairs_full.json at {cand}; regenerating pairs. Error: {exc}"
            )

    built_pairs: dict[str, dict[str, list[Prompt]]] = {}
    for cause_attr in attrs:
        iso_attrs = [a for a in attrs if a != cause_attr]
        generated = build_prompt_pairs(
            ctx,
            dataset,
            cause_attr,
            iso_attrs,
        )
        built_pairs[cause_attr] = generated
        stats.pairs_generated_for.append(cause_attr)
    return built_pairs


def serialize_pairs(
    stored_pairs: dict[str, dict[str, list[Prompt]]], paths: list[Path]
) -> dict[str, dict[str, list[dict]]]:
    """Serialize Prompt objects to dicts and write to provided paths."""
    serializable: dict[str, dict[str, list[dict]]] = {}
    for attr, pair_dict in stored_pairs.items():
        serializable[attr] = {}
        for key, prompts in pair_dict.items():
            serializable[attr][key] = [prompt_to_dict(p) for p in prompts]
    for path in paths:
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
    return serializable


def run_sae_reconstruction(
    model,
    sae,
    tokens: torch.Tensor,
    attn: torch.Tensor,
    target_positions: list[int],
    *,
    position_ids: torch.Tensor,
    readouts: list[str] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Reconstruct SAE rows and capture final residuals at target positions."""
    # Resolve the sole pipeline readout and install its output-embedding hook.
    requested = list(dict.fromkeys(readouts or ["final_resid"]))
    unsupported = set(requested).difference({"final_resid"})
    if unsupported:
        raise ValueError(f"Unsupported data-prep readouts: {sorted(unsupported)}.")
    include_final_resid = "final_resid" in requested
    layer = sae.cfg.hook_layer
    captured_x: list[torch.Tensor] = [None] * tokens.shape[0]  # type: ignore
    captured_z: list[torch.Tensor] = [None] * tokens.shape[0]  # type: ignore
    hook_handle = None
    readout_hook_handle = None
    captured_readout_inputs: list[torch.Tensor] = []

    def replace_resid(module, inputs, outputs):
        if isinstance(outputs, ModelOutput):
            hidden_states = outputs[0]
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        modified = hidden_states.clone()
        for b_idx, pos in enumerate(target_positions):
            x_val = hidden_states[b_idx, pos, :].detach()
            captured_x[b_idx] = x_val
            z_val = sae.encode(x_val.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype))
            captured_z[b_idx] = z_val.detach()
            x_hat = sae.decode(z_val).to(modified.dtype)
            modified[b_idx, pos, :] = x_hat
        if isinstance(outputs, ModelOutput):
            outputs_tuple = outputs.to_tuple()
            new_outputs = type(outputs)(*(modified, *outputs_tuple[1:]))
        elif isinstance(outputs, tuple):
            new_outputs = (modified, *outputs[1:])
        else:
            new_outputs = modified
        return new_outputs

    hook_handle = activation_collection.get_module(model, layer).register_forward_hook(
        replace_resid
    )
    if include_final_resid:
        output_embeddings = model.get_output_embeddings()
        if not isinstance(output_embeddings, torch.nn.Module):
            raise RuntimeError("Model output embeddings must be a torch module.")

        def capture_readout_input(module, inputs):
            """Capture the exact positional tensor passed into the linear readout."""
            # Fail closed later unless the module receives one rank-3 tensor once.
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Output embedding input is not one tensor.")
            captured_readout_inputs.append(inputs[0].detach())

        readout_hook_handle = output_embeddings.register_forward_pre_hook(
            capture_readout_input
        )
    try:
        model(
            input_ids=tokens,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if readout_hook_handle is not None:
            readout_hook_handle.remove()
    if any(x is None or z is None for x, z in zip(captured_x, captured_z)):
        raise RuntimeError(
            "Hook did not capture target layer output for all batch items."
        )
    reps = []
    zs = []
    readout_tensors: dict[str, list[torch.Tensor]] = {
        name: [] for name in requested if name == "final_resid"
    }
    if include_final_resid and len(captured_readout_inputs) != 1:
        raise RuntimeError(
            "Expected exactly one output-embedding input capture, got "
            f"{len(captured_readout_inputs)}."
        )
    captured_final_resid = captured_readout_inputs[0] if include_final_resid else None
    if captured_final_resid is not None and captured_final_resid.dim() != 3:
        raise RuntimeError(
            "Output-embedding input must be rank-3, got "
            f"{tuple(captured_final_resid.shape)}."
        )
    for b_idx, pos in enumerate(target_positions):
        reps.append(captured_x[b_idx])
        zs.append(captured_z[b_idx])
        if "final_resid" in readout_tensors:
            readout_tensors["final_resid"].append(
                captured_final_resid[b_idx, pos, :].detach()
            )
    return reps, zs, readout_tensors


def collect_activations(
    fega_cfg: FEGAConfig,
    cache_dir: str | None = None,
    limit: int | None = None,
    download_location: str | None = None,
    sae_repo_id: str | None = None,
    readouts: list[str] | None = None,
) -> Path:
    """Collect SAE activations/codes and requested readouts into data-prep artifacts."""
    # Default standalone collection to the sole pipeline readout.
    requested_readouts = list(readouts or ["final_resid"])
    unsupported = set(requested_readouts).difference({"final_resid"})
    if unsupported:
        raise ValueError(f"Unsupported data-prep readouts: {sorted(unsupported)}.")
    # Load replay context and model/SAE/tokenizer.
    ctx = ReplayContext.from_file(fega_cfg.reference_json)
    ctx.eval_config = fega_cfg.eval_config

    model, tokenizer, sae = load_model_and_sae(
        fega_cfg.eval_config,
        fega_cfg.device,
        cache_dir=cache_dir,
        sae_release_id=ctx.sae_lens_release_id,
        sae_id_override=ctx.sae_lens_id,
        download_location=download_location,
        sae_repo_id=sae_repo_id,
        sae_cfg_dict=ctx.sae_cfg_dict,
    )
    model.eval()
    sae.eval()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    pad_token_source = "tokenizer.pad_token_id"
    if pad_token_id is None:
        pad_token_id = 0
        pad_token_source = "fallback_legacy_zero"
    pad_token_id = int(pad_token_id)

    # Validate entity/attribute selection (exactly one entity).
    if len(fega_cfg.eval_config.entity_attribute_selection) != 1:
        raise ValueError("FEGA data_prep expects exactly one entity in selection.")
    entity_class = list(fega_cfg.eval_config.entity_attribute_selection.keys())[0]
    attrs = fega_cfg.eval_config.entity_attribute_selection[entity_class]
    if len(attrs) == 0:
        raise ValueError("No attributes provided in selection.")
    run_paths: RunPaths = setup_run(fega_cfg, entity_class, attrs)

    dataset, ds_path, dataset_source = load_filtered_dataset_if_cached(
        ctx,
        entity_class,
        tokenizer,
        model,
        fega_cfg.eval_config.entity_attribute_selection[entity_class],
    )

    # Track per-run stats and counts.
    all_pairs: dict[str, dict[str, int]] = {}
    stats = Stats(
        dataset_path=str(ds_path),
        dataset_source=dataset_source,
        limit=limit,
        batch_size_start=fega_cfg.eval_config.llm_batch_size,
        batch_size_min=fega_cfg.eval_config.llm_batch_size,
    )
    activations_dir = run_paths.activations_dir
    collect_dir = run_paths.collect_dir
    out_dir = activations_dir
    run_pairs_path = run_paths.run_pairs_path
    manifest_path = run_paths.manifest_path
    meta_chunk_tpl = run_paths.meta_tpl
    tensor_chunk_tpl = run_paths.tensor_tpl
    chunk_size = None if fega_cfg.single_file else (fega_cfg.save_chunk_size or 512)

    # Buffer activations/meta and manage chunk manifest entries.
    chunk_processor = ChunkProcessor(
        out_dir,
        chunk_size,
        tensor_chunk_tpl,
        meta_chunk_tpl,
        fega_cfg.single_file,
    )
    pairs_cache_path = run_paths.pairs_cache_path

    stored_pairs = load_or_build_pairs(
        fega_cfg,
        ctx,
        dataset,
        entity_class,
        attrs,
        model,
        stats,
        pairs_cache_path,
    )

    for cause_attr in attrs:
        iso_attrs: list[str] = [a for a in attrs if a != cause_attr]

        if stored_pairs and cause_attr in stored_pairs:
            pair_info = stored_pairs[cause_attr]
        else:
            # generate pairs once using RAVEL logic and save for reuse
            generated = build_prompt_pairs(
                ctx,
                dataset,
                cause_attr,
                iso_attrs,
            )
            stored_pairs = stored_pairs or {}
            stored_pairs[cause_attr] = generated
            pair_info = generated
            stats.pairs_generated_for.append(cause_attr)

        cause_base_prompts = pair_info.get("cause_base_prompts", [])

        all_pairs[cause_attr] = {
            "cause_base_prompts": len(pair_info.get("cause_base_prompts", [])),
            "cause_source_prompts": len(pair_info.get("cause_source_prompts", [])),
            "iso_base_prompts": len(pair_info.get("iso_base_prompts", [])),
            "iso_source_prompts": len(pair_info.get("iso_source_prompts", [])),
        }

        remaining = cause_base_prompts if limit is None else cause_base_prompts[:limit]
        current_batch_size = fega_cfg.eval_config.llm_batch_size or 1
        stats.batch_size_start = current_batch_size
        stats.batch_size_min = current_batch_size

        start_idx = 0
        while start_idx < len(remaining):
            batch_slice = remaining[start_idx : start_idx + current_batch_size]
            if not batch_slice:
                break
            batch_chunk = sorted(
                list(enumerate(batch_slice, start=start_idx)),
                key=lambda kv: len(kv[1].input_ids),
            )
            try:
                prompt_batch = build_padded_prompt_batch(
                    [prompt for _, prompt in batch_chunk],
                    device=model.device,
                    pad_token_id=pad_token_id,
                    original_indices=[orig_idx for orig_idx, _ in batch_chunk],
                    positioning_schema_version=POSITIONING_SCHEMA_VERSION,
                )
                with torch.no_grad():
                    reps, zs, batch_readouts = run_sae_reconstruction(
                        model,
                        sae,
                        prompt_batch.input_ids,
                        prompt_batch.attention_mask,
                        prompt_batch.target_positions,
                        position_ids=prompt_batch.position_ids,
                        readouts=requested_readouts,
                    )
            except torch.cuda.OutOfMemoryError:
                stats.oom_errors += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = max(1, current_batch_size // 2)
                stats.batch_size_adjustments.append(
                    {"from": current_batch_size, "to": new_bs, "reason": "oom"}
                )
                current_batch_size = new_bs
                stats.batch_size_min = min(stats.batch_size_min, current_batch_size)
                # Retry the same start_idx with the smaller batch size
                continue

            stats.total_prompts_seen += len(batch_chunk)
            batch_records = []
            for local_idx, (orig_idx, prompt) in enumerate(batch_chunk):
                rep = reps[local_idx]
                z = zs[local_idx]
                row_readouts = {
                    key: values[local_idx] for key, values in batch_readouts.items()
                }

                def _is_bad(t: torch.Tensor) -> bool:
                    return torch.isnan(t).any().item() or torch.isinf(t).any().item()

                if (
                    _is_bad(rep)
                    or _is_bad(z)
                    or any(_is_bad(tensor) for tensor in row_readouts.values())
                ):
                    stats.skipped_nan += 1
                    continue

                meta_rec = {
                    "pair_index": orig_idx,
                    "pair_role": "cause_base_prompts",
                    "prompt": prompt.text,
                    "entity_label": prompt.entity_label,
                    "attribute_type": prompt.attribute_type,
                    "attribute_label": prompt.attribute_label,
                    "cause_attribute": cause_attr,
                    "final_entity_token_pos": prompt.final_entity_token_pos,
                    "first_generated_token_id": prompt.first_generated_token_id,
                }
                meta_rec.update(prompt_batch.row_metadata(local_idx))
                batch_records.append((orig_idx, rep, z, row_readouts, meta_rec))

            for _, rep, z, row_readouts, meta_rec in sorted(
                batch_records, key=lambda item: item[0]
            ):
                idx = stats.saved_records
                meta_rec["index"] = idx
                chunk_processor.add(rep, z, None, meta_rec, readouts=row_readouts)
                stats.saved_records += 1
            start_idx += current_batch_size

    # Flush remaining buffered tensors/meta and write manifest
    chunk_processor.flush()
    chunk_count = len(chunk_processor.manifest_entries)

    # Write activations manifest describing chunk files.
    positioning = build_positioning_metadata(
        pad_token_id=pad_token_id,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        batch_size_provenance={
            "phase": "data_prep",
            "configured_batch_size": fega_cfg.eval_config.llm_batch_size,
            "batch_size_start": stats.batch_size_start,
            "batch_size_min": stats.batch_size_min,
            "oom_adjustments": stats.batch_size_adjustments,
        },
    )
    positioning["pad_token_source"] = pad_token_source
    manifest = {
        "total_records": stats.saved_records,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "single_file": fega_cfg.single_file,
        "tensor_keys": ["index", "x", "z", *requested_readouts],
        "readouts": requested_readouts,
        "positioning": positioning,
        "chunks": chunk_processor.manifest_entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    stats.chunk_count = chunk_count
    stats.chunk_size = chunk_size

    # Persist the canonical prompt-pair artifact for downstream phases.
    if stored_pairs:
        paths = [run_pairs_path]
        if pairs_cache_path != run_pairs_path and not pairs_cache_path.exists():
            paths.append(pairs_cache_path)
        serialize_pairs(stored_pairs, paths)
    meta = CollectionMeta(
        total_records=stats.saved_records,
        pairs_counts=all_pairs,
        limit=limit,
        entity_attribute_selection=fega_cfg.eval_config.entity_attribute_selection,
        reference_json=str(fega_cfg.reference_json),
        pairs_full_path=str(run_pairs_path),
        manifest_path=str(manifest_path),
        positioning=positioning,
        stats=stats,
    )

    # Save full run meta for reproducibility.
    with open(collect_dir / "collection_meta.json", "w") as f:
        json.dump(meta.as_dict(), f, indent=2)

    return manifest_path


def _collect_data_prep_artifacts(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> Path:
    """Run the collection stage with `data_prep` as the artifact base directory."""
    entity, attr = require_single_entity_attr(config)
    base_dir = data_prep_dir(config, entity, attr)
    base_dir.mkdir(parents=True, exist_ok=True)
    data_prep = config.phases.data_prep
    cfg = FEGAConfig.from_reference(
        config.reference_json,
        device=config.device,
        output_dir=base_dir,
        entity_attribute_selection=config.entity_attribute_selection,
        save_chunk_size=data_prep.save_chunk_size,
        single_file=data_prep.single_file,
        random_seed=config.seed.global_,
        llm_batch_size_override=(
            data_prep.batch_size
            if data_prep.batch_size is not None
            else config.llm_batch_size_override
        ),
    )

    def _reuse_model_and_sae(*args, **kwargs):
        return (
            resources.get_model_and_sae()
            if resources
            else load_model_and_sae(*args, **kwargs)
        )

    cache_dir = str(config.cache_dir) if config.cache_dir else None
    download_location = (
        str(config.download_saes_dir) if config.download_saes_dir else None
    )
    sae_repo_id = config.sae_repo_id

    if resources:
        with patched_attr(
            sys.modules[__name__],
            "load_model_and_sae",
            _reuse_model_and_sae,
        ):
            return collect_activations(
                cfg,
                cache_dir=cache_dir,
                limit=data_prep.limit,
                download_location=download_location,
                sae_repo_id=sae_repo_id,
                readouts=data_prep.readouts,
            )

    return collect_activations(
        cfg,
        cache_dir=cache_dir,
        limit=data_prep.limit,
        download_location=download_location,
        sae_repo_id=sae_repo_id,
        readouts=data_prep.readouts,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Internal data-prep helper: collect activations/codes for FEGA."
    )
    p.add_argument(
        "--reference_json", required=True, help="Path to replay reference JSON."
    )
    p.add_argument(
        "--output_dir",
        default="results/data_prep",
        help="Where to store collected activations.",
    )
    p.add_argument(
        "--device", default=None, help="Device for model/SAE (e.g., cuda:0, cpu)."
    )
    p.add_argument(
        "--cache_dir", default=None, help="HF cache dir for model/tokenizer."
    )
    p.add_argument(
        "--entity_attribute_selection",
        default='{"city":["Country"]}',
        help="JSON string for selection override.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total prompts collected (optional).",
    )
    p.add_argument(
        "--save_chunk_size",
        type=int,
        default=512,
        help="Number of examples per activation chunk (ignored if --single_file).",
    )
    p.add_argument(
        "--single_file",
        action="store_true",
        help="If set, save activations into a single tensor/meta file (not recommended for large runs).",
    )
    p.add_argument(
        "--download_location",
        default=None,
        help="Local cache dir for SAE checkpoints/configs.",
    )
    p.add_argument(
        "--sae_repo_id",
        default=None,
        help="HF repo id for dictionary-learning SAEs (fallback if sae_lens release is unavailable).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/order.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    selection = (
        json.loads(args.entity_attribute_selection)
        if args.entity_attribute_selection
        else None
    )
    cfg = FEGAConfig.from_reference(
        args.reference_json,
        device=args.device or "cuda:0",
        output_dir=args.output_dir,
        entity_attribute_selection=selection,
        save_chunk_size=args.save_chunk_size,
        single_file=args.single_file,
        random_seed=args.seed,
    )
    out_path = collect_activations(
        cfg,
        cache_dir=args.cache_dir,
        limit=args.limit,
        download_location=args.download_location,
        sae_repo_id=args.sae_repo_id,
    )
    print(f"Wrote activations to {out_path}")


if __name__ == "__main__":
    main()
