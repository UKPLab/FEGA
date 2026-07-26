from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from fega.config_schema import FEGAPipelineConfig, InductionConfig
from fega.core.common import require_single_entity_attr, selection_seed
from fega.core.data_prep.collection import run_sae_reconstruction
from fega.core.positioning import (
    POSITIONING_SCHEMA_VERSION,
    build_padded_prompt_batch,
    build_positioning_metadata,
)
from fega.core.utils import ChunkProcessor, ensure_dir, prompt_to_dict
from fega.paths import (
    data_prep_activations_dir,
    data_prep_collect_dir,
    data_prep_dir,
    data_prep_pairs_path,
    data_prep_select_dir,
)
from sae_bench.evals.ravel.instance import Prompt
from sae_bench.sae_bench_utils import activation_collection

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedInductionFeatureSet:
    sae_uid: str
    sae_release: str
    sae_id: str
    layer: int
    hook_name: str
    selected_feature_ids: list[int]
    feature_set_name: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ResolvedModelSaeSpec:
    eval_config: Any
    sae_release_id: str
    sae_id_override: str
    sae_repo_id: str | None
    sae_cfg_dict: dict[str, Any] | None
    feature_set: ResolvedInductionFeatureSet
    summary: dict[str, Any]


def load_induction_summary(path: Path) -> dict[str, Any]:
    with open(path) as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"Induction summary must be a JSON object: {path}")
    return summary


def resolve_induction_feature_set(
    config: FEGAPipelineConfig,
) -> tuple[dict[str, Any], ResolvedInductionFeatureSet]:
    induction = _require_induction_config(config)
    summary = load_induction_summary(induction.summary_json)
    feature_sets = summary.get("feature_sets")
    if not isinstance(feature_sets, Mapping) or not feature_sets:
        raise ValueError(
            "Induction summary must contain a non-empty `feature_sets` map."
        )
    sae_uid = induction.sae_uid
    if sae_uid is None:
        if len(feature_sets) != 1:
            raise ValueError(
                "`induction.sae_uid` is required when summary has multiple SAE entries."
            )
        sae_uid = str(next(iter(feature_sets)))
    if sae_uid not in feature_sets:
        raise ValueError(f"SAE uid {sae_uid!r} not present in induction summary.")
    raw_set = dict(feature_sets[sae_uid])
    feature_ids = _resolve_feature_ids(induction, raw_set)
    return summary, ResolvedInductionFeatureSet(
        sae_uid=sae_uid,
        sae_release=str(raw_set["sae_release"]),
        sae_id=str(raw_set["sae_id"]),
        layer=int(raw_set["layer"]),
        hook_name=str(raw_set["hook_name"]),
        selected_feature_ids=feature_ids,
        feature_set_name=induction.feature_set,
        raw=raw_set,
    )


def resolve_induction_model_sae_spec(
    config: FEGAPipelineConfig,
) -> ResolvedModelSaeSpec:
    summary, feature_set = resolve_induction_feature_set(config)
    model_name = str(summary.get("model_name") or "")
    llm_dtype = str(summary.get("llm_dtype") or "float32")
    if not model_name:
        raise ValueError("Induction summary is missing `model_name`.")
    eval_config = SimpleNamespace(
        model_name=model_name,
        llm_dtype=llm_dtype,
        llm_batch_size=(
            config.phases.data_prep.batch_size
            if config.phases.data_prep.batch_size is not None
            else config.llm_batch_size_override
        ),
    )
    sae_selection_mode = str(summary.get("sae_selection_mode") or "").strip()
    sae_cfg_dict: dict[str, Any] | None
    if sae_selection_mode == "custom_repo":
        sae_repo_id = config.sae_repo_id or feature_set.sae_release
        repo_base = sae_repo_id.split("/")[-1]
        release_id = f"{repo_base}_{feature_set.sae_id.replace('/', '_')}"
        sae_id_override = "custom_sae"
        total_features = feature_set.raw.get("total_features")
        sae_cfg_dict = {
            "model_name": model_name,
            "hook_layer": feature_set.layer,
            "hook_name": feature_set.hook_name,
            "dtype": llm_dtype,
        }
        if total_features is not None:
            sae_cfg_dict["d_sae"] = int(total_features)
    else:
        sae_repo_id = config.sae_repo_id
        release_id = feature_set.sae_release
        sae_id_override = feature_set.sae_id
        sae_cfg_dict = None
    return ResolvedModelSaeSpec(
        eval_config=eval_config,
        sae_release_id=release_id,
        sae_id_override=sae_id_override,
        sae_repo_id=sae_repo_id,
        sae_cfg_dict=sae_cfg_dict,
        feature_set=feature_set,
        summary=summary,
    )


def run_induction_data_prep(
    config: FEGAPipelineConfig, resources: Any | None = None
) -> tuple[Path, Path]:
    """Collect and select induction prompts into the existing FEGA artifact contract."""
    induction = _require_induction_config(config)
    summary, feature_set = resolve_induction_feature_set(config)
    model_resources = resources
    if model_resources is None:
        from fega.core.resources import ModelResources

        model_resources = ModelResources(config)
    model, tokenizer, sae = model_resources.get_model_and_sae()
    if model is None or tokenizer is None or sae is None:
        raise RuntimeError(
            "ModelResources returned an incomplete model/tokenizer/SAE bundle."
        )
    model.eval()
    sae.eval()

    examples = load_induction_examples(config.reference_json)
    scan_examples = prepare_scan_examples(
        examples,
        tokenizer,
        induction=induction,
        summary=summary,
        seed=selection_seed(config),
        limit=config.phases.data_prep.limit,
    )
    scan_records, scan_stats = scan_induction_records(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        examples=scan_examples,
        feature_ids=feature_set.selected_feature_ids,
        require_model_correct=_resolve_require_model_correct(induction, summary),
        batch_size=_effective_batch_size(config),
        device=config.device,
    )
    selected_by_feature, selection_stats = select_induction_contexts(
        scan_records,
        feature_set.selected_feature_ids,
        tau_act=config.phases.data_prep.tau_act,
        max_contexts=config.phases.data_prep.max_contexts,
        min_contexts=config.phases.data_prep.min_contexts,
        stratify_by=induction.stratify_by,
    )
    dense_rows = materialize_selected_induction_rows(
        config=config,
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        selected_by_feature=selected_by_feature,
        feature_set=feature_set,
        summary=summary,
        scan_stats=scan_stats,
        selection_stats=selection_stats,
    )
    manifest_path, contexts_path = write_induction_artifacts(
        config=config,
        tokenizer=tokenizer,
        dense_rows=dense_rows,
        selected_by_feature=selected_by_feature,
        feature_set=feature_set,
        summary=summary,
        scan_stats=scan_stats,
        selection_stats=selection_stats,
    )
    return manifest_path, contexts_path


def load_induction_examples(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(dataset_path).read_text())
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if "examples" in payload:
        raw_examples = list(payload["examples"])
    elif "contexts" in payload:
        raw_examples = []
        for context in payload["contexts"]:
            context_id = context["context_id"]
            query_style = context.get("query_style")
            for query in context["queries"]:
                normalized = dict(query)
                normalized.setdefault("example_id", query.get("query_id"))
                normalized["context_id"] = context_id
                normalized.setdefault("query_style", query_style)
                raw_examples.append(normalized)
    else:
        raise ValueError(
            "Induction dataset JSON must contain either `examples` or `contexts`."
        )
    examples: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(raw_examples):
        if "context_id" not in row or "prompt" not in row or "answer" not in row:
            raise ValueError(
                "Induction examples require `context_id`, `prompt`, and `answer`."
            )
        normalized = dict(row)
        normalized.setdefault("example_id", f"example_{source_row_index:07d}")
        normalized["support_example_index"] = int(
            normalized.get("support_example_index") or 0
        )
        normalized["source_row_index"] = source_row_index
        normalized["dataset_metadata"] = metadata
        examples.append(normalized)
    return examples


def prepare_scan_examples(
    examples: list[dict[str, Any]],
    tokenizer,
    *,
    induction: InductionConfig,
    summary: dict[str, Any],
    seed: int | None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    sampled = _apply_source_caps(
        examples,
        max_contexts=induction.max_source_contexts,
        max_examples=induction.max_source_examples,
        seed=seed if seed is not None else 0,
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("phases.data_prep.limit must be positive when provided.")
        sampled = sampled[:limit]
    answer_prefix = _resolve_answer_prefix(induction, summary)
    single_token_only = bool((summary.get("filtering") or {}).get("single_token_only"))
    out: list[dict[str, Any]] = []
    token_cache: dict[str, list[int]] = {}
    for row in sampled:
        answer = str(row["answer"])
        if answer not in token_cache:
            token_ids = tokenizer.encode(
                f"{answer_prefix}{answer}", add_special_tokens=False
            )
            if not token_ids:
                raise ValueError(f"Tokenizer produced no target tokens for {answer!r}.")
            token_cache[answer] = [int(token) for token in token_ids]
        target_tokens = token_cache[answer]
        if single_token_only and len(target_tokens) != 1:
            continue
        augmented = dict(row)
        augmented["target_first_token_id"] = int(target_tokens[0])
        augmented["target_token_length"] = len(target_tokens)
        out.append(augmented)
    return out


@torch.no_grad()
def scan_induction_records(
    *,
    model,
    tokenizer,
    sae,
    examples: list[dict[str, Any]],
    feature_ids: list[int],
    require_model_correct: bool,
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pad_token_id, pad_token_source = _pad_token(tokenizer)
    records: list[dict[str, Any]] = []
    stats = {
        "source_examples": len(examples),
        "kept_after_correctness": 0,
        "dropped_model_incorrect": 0,
        "pad_token_id": pad_token_id,
        "pad_token_source": pad_token_source,
    }
    for start in range(0, len(examples), batch_size):
        batch_rows = examples[start : start + batch_size]
        prompts = [
            _prompt_from_induction_row(row, tokenizer, "induction", "rule_completion")
            for row in batch_rows
        ]
        prompt_batch = build_padded_prompt_batch(
            prompts,
            device=device,
            pad_token_id=pad_token_id,
            original_indices=[int(row["source_row_index"]) for row in batch_rows],
            positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        )
        feature_activations, predicted_ids = _run_base_scan_forward(
            model=model,
            sae=sae,
            tokens=prompt_batch.input_ids,
            attn=prompt_batch.attention_mask,
            target_positions=prompt_batch.target_positions,
            position_ids=prompt_batch.position_ids,
            feature_ids=feature_ids,
        )
        for local_idx, row in enumerate(batch_rows):
            target_id = int(row["target_first_token_id"])
            pred_id = predicted_ids[local_idx]
            model_correct = pred_id == target_id
            if require_model_correct and not model_correct:
                stats["dropped_model_incorrect"] += 1
                continue
            selected_activations = feature_activations[local_idx]
            record = {
                "scan_record_id": len(records),
                "source_row_index": int(row["source_row_index"]),
                "example_id": str(row.get("example_id")),
                "context_id": row.get("context_id"),
                "support_example_index": int(row.get("support_example_index") or 0),
                "prompt": row["prompt"],
                "answer": row["answer"],
                "target_first_token_id": target_id,
                "target_token_length": int(row["target_token_length"]),
                "model_correct_first_token": model_correct,
                "predicted_first_token_id": pred_id,
                "feature_activations": selected_activations,
                "source_concept": row.get("source_concept"),
                "target_concept": row.get("target_concept"),
                "lookup_rule": row.get("lookup_rule"),
                "induction_prefix": row.get("induction_prefix"),
                "x": row.get("x"),
                "y": row.get("y"),
                "task_name": row.get("task_name"),
                "prompt_family_id": row.get("prompt_family_id"),
                "prompt_family_index": row.get("prompt_family_index"),
                "family_query_index": row.get("family_query_index"),
                "source_language": row.get("source_language"),
                "target_language": row.get("target_language"),
            }
            record.update(prompt_batch.row_metadata(local_idx))
            records.append(record)
            stats["kept_after_correctness"] += 1
    return records, stats


@torch.no_grad()
def _run_base_scan_forward(
    *,
    model,
    sae,
    tokens: torch.Tensor,
    attn: torch.Tensor,
    target_positions: list[int],
    position_ids: torch.Tensor,
    feature_ids: list[int],
) -> tuple[list[dict[int, float]], list[int]]:
    layer = int(getattr(sae.cfg, "hook_layer"))
    captured_x: list[torch.Tensor | None] = [None] * tokens.shape[0]

    def capture_resid(module, inputs, outputs):
        to_tuple = getattr(outputs, "to_tuple", None)
        if callable(to_tuple):
            hidden_states = cast(Any, to_tuple())[0]
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        for b_idx, pos in enumerate(target_positions):
            captured_x[b_idx] = hidden_states[b_idx, pos, :].detach()
        return outputs

    hook_handle = activation_collection.get_module(model, layer).register_forward_hook(
        capture_resid
    )
    try:
        model_out = model(
            input_ids=tokens,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )
        logits_out = model_out.logits
    finally:
        hook_handle.remove()
    if any(x is None for x in captured_x):
        raise RuntimeError("Hook did not capture induction scan activations.")
    batch_indices = torch.arange(tokens.shape[0], device=logits_out.device)
    position_tensor = torch.tensor(
        target_positions, device=logits_out.device, dtype=torch.long
    )
    predicted_ids = (
        logits_out[batch_indices, position_tensor, :].argmax(dim=-1).detach().cpu()
    )
    if not feature_ids:
        return [{} for _ in range(tokens.shape[0])], [
            int(pred_id) for pred_id in predicted_ids.tolist()
        ]

    x_batch = torch.stack([cast(torch.Tensor, x) for x in captured_x])
    z_batch = sae.encode(x_batch.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype))
    feature_id_tensor = torch.tensor(
        feature_ids, device=z_batch.device, dtype=torch.long
    )
    selected_z = z_batch.index_select(dim=-1, index=feature_id_tensor).detach().cpu()
    feature_activations = []
    for row_idx in range(selected_z.shape[0]):
        feature_activations.append(
            {
                int(feature_id): float(selected_z[row_idx, col_idx].item())
                for col_idx, feature_id in enumerate(feature_ids)
            }
        )
    return feature_activations, [int(pred_id) for pred_id in predicted_ids.tolist()]


def select_induction_contexts(
    scan_records: list[dict[str, Any]],
    feature_ids: list[int],
    *,
    tau_act: float,
    max_contexts: int,
    min_contexts: int,
    stratify_by: list[str],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    stats: dict[int, dict[str, Any]] = {}
    for feature_id in feature_ids:
        candidates = []
        skipped_tau = 0
        for record in scan_records:
            activations = record.get("feature_activations") or {}
            z_value = activations.get(feature_id)
            if z_value is None:
                z_value = activations.get(str(feature_id))
            if z_value is None:
                continue
            z_float = float(z_value)
            if z_float > tau_act:
                rec = dict(record)
                rec["feature_id"] = int(feature_id)
                rec["z"] = z_float
                candidates.append(rec)
            else:
                skipped_tau += 1
        ordered = _round_robin_select(candidates, max_contexts, stratify_by)
        too_rare = len(ordered) if len(ordered) < min_contexts else 0
        selected[int(feature_id)] = [] if too_rare else ordered
        stats[int(feature_id)] = {
            "candidates": len(candidates),
            "selected": 0 if too_rare else len(ordered),
            "skipped_tau": skipped_tau,
            "capped": max(0, len(candidates) - len(ordered)),
            "too_rare": too_rare,
        }
    return selected, stats


def assign_dense_indices(
    selected_by_feature: dict[int, list[dict[str, Any]]],
) -> dict[tuple[int, int], int]:
    source_records: dict[tuple[int, int], dict[str, Any]] = {}
    for records in selected_by_feature.values():
        for record in records:
            key = _source_key(record)
            source_records[key] = record
    ordered_keys = sorted(source_records, key=lambda key: (key[0], key[1]))
    return {key: dense_idx for dense_idx, key in enumerate(ordered_keys)}


@torch.no_grad()
def materialize_selected_induction_rows(
    *,
    config: FEGAPipelineConfig,
    model,
    tokenizer,
    sae,
    selected_by_feature: dict[int, list[dict[str, Any]]],
    feature_set: ResolvedInductionFeatureSet,
    summary: dict[str, Any],
    scan_stats: dict[str, Any],
    selection_stats: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    entity, attr = require_single_entity_attr(config)
    dense_lookup = assign_dense_indices(selected_by_feature)
    records_by_key = {
        _source_key(record): record
        for records in selected_by_feature.values()
        for record in records
    }
    selected_z_by_source = _selected_z_by_source(selected_by_feature)
    ordered_records = [
        records_by_key[key]
        for key, _ in sorted(dense_lookup.items(), key=lambda kv: kv[1])
    ]
    if not ordered_records:
        return []
    pad_token_id, _ = _pad_token(tokenizer)
    batch_size = _effective_batch_size(config)
    dense_rows: list[dict[str, Any]] = []
    for start in range(0, len(ordered_records), batch_size):
        batch_records = ordered_records[start : start + batch_size]
        prompts = [
            _prompt_from_induction_row(record, tokenizer, entity, attr)
            for record in batch_records
        ]
        prompt_batch = build_padded_prompt_batch(
            prompts,
            device=next(model.parameters()).device,
            pad_token_id=pad_token_id,
            original_indices=[
                int(record["source_row_index"]) for record in batch_records
            ],
            positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        )
        reps, zs, readouts = run_sae_reconstruction(
            model,
            sae,
            prompt_batch.input_ids,
            prompt_batch.attention_mask,
            prompt_batch.target_positions,
            position_ids=prompt_batch.position_ids,
            readouts=config.phases.data_prep.readouts,
        )
        for local_idx, record in enumerate(batch_records):
            dense_idx = dense_lookup[_source_key(record)]
            aligned_z = _align_selected_feature_activations(
                zs[local_idx], record, selected_z_by_source
            )
            dense_rows.append(
                _build_dense_row(
                    dense_idx=dense_idx,
                    record=record,
                    prompt=prompts[local_idx],
                    x_tensor=reps[local_idx],
                    z_tensor=aligned_z,
                    readouts={
                        name: values[local_idx] for name, values in readouts.items()
                    },
                    row_positioning=prompt_batch.row_metadata(local_idx),
                    entity=entity,
                    attr=attr,
                    sae_uid=feature_set.sae_uid,
                )
            )
    dense_rows.sort(key=lambda row: int(row["index"]))
    _assert_scan_materialization_parity(dense_rows, selected_by_feature)
    return dense_rows


def _selected_z_by_source(
    selected_by_feature: dict[int, list[dict[str, Any]]],
) -> dict[tuple[int, int], dict[int, float]]:
    selected: dict[tuple[int, int], dict[int, float]] = {}
    for feature_id, records in selected_by_feature.items():
        fid = int(feature_id)
        for record in records:
            key = _source_key(record)
            if "z" not in record:
                raise ValueError(
                    "Selected induction context is missing scan activation `z`."
                )
            z_value = float(record["z"])
            feature_values = selected.setdefault(key, {})
            previous = feature_values.get(fid)
            if previous is not None and previous != z_value:
                raise ValueError(
                    "Selected induction contexts disagree on scan activation for "
                    f"feature {fid} and source key {key}: {previous} vs {z_value}."
                )
            feature_values[fid] = z_value
    return selected


def _align_selected_feature_activations(
    z_tensor: torch.Tensor,
    record: dict[str, Any],
    selected_z_by_source: dict[tuple[int, int], dict[int, float]],
) -> torch.Tensor:
    aligned = z_tensor.detach().cpu().clone()
    selected_values = selected_z_by_source.get(_source_key(record), {})
    for feature_id, scan_z in sorted(selected_values.items()):
        if feature_id < 0 or feature_id >= aligned.numel():
            raise ValueError(
                "Selected induction feature id is outside the materialized SAE "
                f"activation vector: feature {feature_id}, width {aligned.numel()}."
            )
        aligned[feature_id] = float(scan_z)
    return aligned


def write_induction_artifacts(
    *,
    config: FEGAPipelineConfig,
    tokenizer,
    dense_rows: list[dict[str, Any]],
    selected_by_feature: dict[int, list[dict[str, Any]]],
    feature_set: ResolvedInductionFeatureSet,
    summary: dict[str, Any],
    scan_stats: dict[str, Any],
    selection_stats: dict[int, dict[str, Any]],
) -> tuple[Path, Path]:
    entity, attr = require_single_entity_attr(config)
    base_dir = data_prep_dir(config, entity, attr)
    collect_dir = data_prep_collect_dir(config, entity, attr)
    activations_dir = data_prep_activations_dir(config, entity, attr)
    select_dir = data_prep_select_dir(config, entity, attr)
    ensure_dir(base_dir)
    ensure_dir(collect_dir)
    ensure_dir(activations_dir)
    ensure_dir(select_dir)

    chunk_size = (
        None
        if config.phases.data_prep.single_file
        else (config.phases.data_prep.save_chunk_size or 512)
    )
    chunk_processor = ChunkProcessor(
        activations_dir,
        chunk_size,
        "activations_tensors_{:04d}.pt",
        "activations_meta_{:04d}.jsonl",
        config.phases.data_prep.single_file,
    )
    pairs_prompts: list[Prompt] = []
    for row in dense_rows:
        meta = dict(row["meta"])
        row_readouts = dict(row["readouts"])
        chunk_processor.add(
            row["x"],
            row["z"],
            None,
            meta,
            readouts=row_readouts,
        )
        pairs_prompts.append(row["prompt_obj"])
    chunk_processor.flush()

    pad_token_id, pad_token_source = _pad_token(tokenizer)
    positioning = build_positioning_metadata(
        pad_token_id=pad_token_id,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        batch_size_provenance={
            "phase": "data_prep",
            "source_kind": "induction",
            "configured_batch_size": _effective_batch_size(config),
        },
    )
    positioning["pad_token_source"] = pad_token_source
    manifest_path = activations_dir / "activations_manifest.json"
    manifest = {
        "source_kind": "induction",
        "source_summary_json": str(_require_induction_config(config).summary_json),
        "total_records": len(dense_rows),
        "chunk_size": chunk_size,
        "chunk_count": len(chunk_processor.manifest_entries),
        "single_file": config.phases.data_prep.single_file,
        "tensor_keys": ["index", "x", "z", *config.phases.data_prep.readouts],
        "readouts": config.phases.data_prep.readouts,
        "positioning": positioning,
        "chunks": chunk_processor.manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    pairs_path = data_prep_pairs_path(config, entity, attr)
    pairs_payload = {
        attr: {
            "cause_base_prompts": [prompt_to_dict(prompt) for prompt in pairs_prompts]
        }
    }
    pairs_path.write_text(json.dumps(pairs_payload, indent=2))

    dense_lookup = assign_dense_indices(selected_by_feature)
    contexts = _dense_feature_contexts(
        selected_by_feature,
        dense_lookup,
        entity,
        attr,
        feature_set.sae_uid,
    )
    contexts_path = select_dir / "feature_contexts.json"
    contexts_path.write_text(json.dumps(contexts, indent=2))

    dropped_totals = {
        "skipped_tau": sum(
            int(v.get("skipped_tau", 0)) for v in selection_stats.values()
        ),
        "capped": sum(int(v.get("capped", 0)) for v in selection_stats.values()),
        "too_rare": sum(1 for v in selection_stats.values() if v.get("too_rare", 0)),
    }
    induction = _require_induction_config(config)
    summary_path = select_dir / "feature_contexts_summary.json"
    summary_payload = {
        "source_kind": "induction",
        "source_summary_json": str(induction.summary_json),
        "sae_uid": feature_set.sae_uid,
        "feature_set": feature_set.feature_set_name,
        "target_features": feature_set.selected_feature_ids,
        "total_features": len(feature_set.selected_feature_ids),
        "features_with_contexts": sum(1 for rows in contexts.values() if rows),
        "tau_act": config.phases.data_prep.tau_act,
        "source_activation_threshold": _resolve_source_activation_threshold(
            induction, summary
        ),
        "max_contexts": config.phases.data_prep.max_contexts,
        "min_contexts": config.phases.data_prep.min_contexts,
        "max_source_contexts": induction.max_source_contexts,
        "max_source_examples": induction.max_source_examples,
        "entity_class": entity,
        "cause_attribute": attr,
        "manifest_path": str(manifest_path),
        "feature_stats": {str(k): v for k, v in selection_stats.items()},
        "dropped_totals": dropped_totals,
        "scan_stats": scan_stats,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    collection_meta = {
        "source_kind": "induction",
        "total_records": len(dense_rows),
        "entity_attribute_selection": config.entity_attribute_selection,
        "reference_json": str(config.reference_json),
        "summary_json": str(induction.summary_json),
        "pairs_full_path": str(pairs_path),
        "manifest_path": str(manifest_path),
        "positioning": positioning,
        "dataset_metadata": summary.get("dataset_metadata"),
        "filtering": summary.get("filtering"),
        "activation_definition": summary.get("activation_definition"),
        "sae_uid": feature_set.sae_uid,
        "selected_feature_ids": feature_set.selected_feature_ids,
        "source_activation_threshold": _resolve_source_activation_threshold(
            induction, summary
        ),
        "tau_act": config.phases.data_prep.tau_act,
        "scan_stats": scan_stats,
        "selection_stats": {str(k): v for k, v in selection_stats.items()},
    }
    (collect_dir / "collection_meta.json").write_text(
        json.dumps(collection_meta, indent=2)
    )
    return manifest_path, contexts_path


def _dense_feature_contexts(
    selected_by_feature: dict[int, list[dict[str, Any]]],
    dense_lookup: dict[tuple[int, int], int],
    entity: str,
    attr: str,
    sae_uid: str,
) -> dict[int, list[dict[str, Any]]]:
    contexts: dict[int, list[dict[str, Any]]] = {}
    for feature_id, records in selected_by_feature.items():
        out_records = []
        for record in records:
            dense_idx = dense_lookup[_source_key(record)]
            out = {
                "index": dense_idx,
                "pair_index": dense_idx,
                "pair_role": "cause_base_prompts",
                "attribute_label": attr,
                "attribute_type": attr,
                "entity_label": entity,
                "prompt": record["prompt"],
                "z": float(record["z"]),
                "source_kind": "induction",
                "sae_uid": sae_uid,
                "feature_id": int(feature_id),
                "scan_record_id": int(record["scan_record_id"]),
                "source_row_index": int(record["source_row_index"]),
                "example_id": record.get("example_id"),
                "context_id": record.get("context_id"),
                "support_example_index": int(record.get("support_example_index") or 0),
                "source_concept": record.get("source_concept"),
                "target_concept": record.get("target_concept"),
                "lookup_rule": record.get("lookup_rule"),
                "induction_prefix": record.get("induction_prefix"),
                "answer": record.get("answer"),
                "task_name": record.get("task_name"),
                "prompt_family_id": record.get("prompt_family_id"),
                "prompt_family_index": record.get("prompt_family_index"),
                "family_query_index": record.get("family_query_index"),
                "source_language": record.get("source_language"),
                "target_language": record.get("target_language"),
                "target_first_token_id": int(record["target_first_token_id"]),
                "model_correct_first_token": bool(record["model_correct_first_token"]),
            }
            out_records.append(out)
        contexts[int(feature_id)] = out_records
    return contexts


def _build_dense_row(
    *,
    dense_idx: int,
    record: dict[str, Any],
    prompt: Prompt,
    x_tensor: torch.Tensor,
    z_tensor: torch.Tensor,
    readouts: dict[str, torch.Tensor],
    row_positioning: dict[str, Any],
    entity: str,
    attr: str,
    sae_uid: str,
) -> dict[str, Any]:
    meta = {
        "index": dense_idx,
        "pair_index": dense_idx,
        "pair_role": "cause_base_prompts",
        "prompt": record["prompt"],
        "entity_label": entity,
        "attribute_type": attr,
        "attribute_label": attr,
        "cause_attribute": attr,
        "final_entity_token_pos": -1,
        "first_generated_token_id": int(record["target_first_token_id"]),
        "source_kind": "induction",
        "sae_uid": sae_uid,
        "scan_record_id": int(record["scan_record_id"]),
        "source_row_index": int(record["source_row_index"]),
        "example_id": record.get("example_id"),
        "context_id": record.get("context_id"),
        "support_example_index": int(record.get("support_example_index") or 0),
        "source_concept": record.get("source_concept"),
        "target_concept": record.get("target_concept"),
        "lookup_rule": record.get("lookup_rule"),
        "induction_prefix": record.get("induction_prefix"),
        "answer": record.get("answer"),
        "target_first_token_id": int(record["target_first_token_id"]),
        "target_token_length": int(record["target_token_length"]),
        "model_correct_first_token": bool(record["model_correct_first_token"]),
        "task_name": record.get("task_name"),
        "prompt_family_id": record.get("prompt_family_id"),
        "prompt_family_index": record.get("prompt_family_index"),
        "family_query_index": record.get("family_query_index"),
        "source_language": record.get("source_language"),
        "target_language": record.get("target_language"),
    }
    meta.update(row_positioning)
    return {
        "index": dense_idx,
        "x": x_tensor.detach().cpu(),
        "z": z_tensor.detach().cpu(),
        "readouts": {k: v.detach().cpu() for k, v in readouts.items()},
        "meta": meta,
        "prompt_obj": prompt,
    }


def _assert_scan_materialization_parity(
    dense_rows: list[dict[str, Any]],
    selected_by_feature: dict[int, list[dict[str, Any]]],
    *,
    atol: float = 1.0e-5,
) -> None:
    z_by_dense = {int(row["index"]): row["z"] for row in dense_rows}
    dense_lookup = assign_dense_indices(selected_by_feature)
    for feature_id, records in selected_by_feature.items():
        for record in records:
            dense_idx = dense_lookup[_source_key(record)]
            materialized = float(z_by_dense[dense_idx][int(feature_id)].item())
            if abs(materialized - float(record["z"])) > atol:
                raise ValueError(
                    "Materialized induction z activation does not match scan "
                    f"activation for feature {feature_id}, dense index {dense_idx}: "
                    f"{materialized} vs {record['z']}."
                )


def _round_robin_select(
    candidates: list[dict[str, Any]], max_contexts: int, stratify_by: list[str]
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in candidates:
        key = tuple(_sort_value(record.get(field)) for field in stratify_by)
        buckets.setdefault(key, []).append(record)
    for key in buckets:
        buckets[key].sort(key=_candidate_sort_key)
    selected: list[dict[str, Any]] = []
    bucket_keys = sorted(buckets)
    positions = {key: 0 for key in bucket_keys}
    while len(selected) < max_contexts:
        progressed = False
        for key in bucket_keys:
            pos = positions[key]
            if pos >= len(buckets[key]):
                continue
            selected.append(buckets[key][pos])
            positions[key] += 1
            progressed = True
            if len(selected) >= max_contexts:
                break
        if not progressed:
            break
    selected.sort(key=_candidate_sort_key)
    return selected[:max_contexts]


def _candidate_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(record["z"]),
        _sort_value(record.get("context_id")),
        _sort_value(record.get("support_example_index")),
        _sort_value(record.get("example_id")),
        int(record.get("scan_record_id", 0)),
        int(record.get("source_row_index", 0)),
    )


def _sort_value(value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    try:
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return (0, int(value))
    except Exception:
        pass
    return (1, str(value))


def _source_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        int(record.get("scan_record_id", 0)),
        int(record.get("source_row_index", 0)),
    )


def _resolve_feature_ids(
    induction: InductionConfig, feature_set: dict[str, Any]
) -> list[int]:
    if induction.feature_set == "candidate":
        key = "candidate_feature_ids"
    elif induction.feature_set == "strict_common":
        key = "strict_common_feature_ids"
    else:
        assert induction.explicit_feature_ids is not None
        return [int(feature_id) for feature_id in induction.explicit_feature_ids]
    raw_ids = feature_set.get(key)
    if not isinstance(raw_ids, list):
        raise ValueError(f"Induction summary feature set is missing `{key}`.")
    return [int(feature_id) for feature_id in raw_ids]


def _require_induction_config(config: FEGAPipelineConfig) -> InductionConfig:
    if config.source_kind != "induction" or config.induction is None:
        raise ValueError("Induction data prep requires `source_kind: induction`.")
    return config.induction


def _resolve_answer_prefix(induction: InductionConfig, summary: dict[str, Any]) -> str:
    if induction.answer_prefix is not None:
        return induction.answer_prefix
    return str((summary.get("filtering") or {}).get("answer_prefix", " "))


def _resolve_require_model_correct(
    induction: InductionConfig, summary: dict[str, Any]
) -> bool:
    if induction.require_model_correct is not None:
        return bool(induction.require_model_correct)
    return bool((summary.get("filtering") or {}).get("require_model_correct", False))


def _resolve_source_activation_threshold(
    induction: InductionConfig, summary: dict[str, Any]
) -> float | None:
    if induction.source_activation_threshold is not None:
        return induction.source_activation_threshold
    activation_definition = summary.get("activation_definition") or {}
    value = activation_definition.get("active_if_post_encode_activation_exceeds")
    return None if value is None else float(value)


def _apply_source_caps(
    examples: list[dict[str, Any]],
    *,
    max_contexts: int | None,
    max_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled = list(examples)
    if max_contexts is not None:
        if max_contexts <= 0:
            raise ValueError("induction.max_source_contexts must be positive.")
        unique_contexts = sorted(
            {row["context_id"] for row in sampled}, key=_sort_value
        )
        if max_contexts < len(unique_contexts):
            chosen = set(rng.sample(unique_contexts, max_contexts))
            sampled = [row for row in sampled if row["context_id"] in chosen]
    sampled = sorted(sampled, key=lambda row: int(row["source_row_index"]))
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("induction.max_source_examples must be positive.")
        if max_examples < len(sampled):
            chosen_indices = set(rng.sample(range(len(sampled)), max_examples))
            sampled = [row for idx, row in enumerate(sampled) if idx in chosen_indices]
    return sorted(sampled, key=lambda row: int(row["source_row_index"]))


def _prompt_from_induction_row(
    row: dict[str, Any], tokenizer, entity: str, attr: str
) -> Prompt:
    text = str(row["prompt"])
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    return Prompt(
        text=text,
        template="induction",
        attribute_type=attr,
        attribute_label=attr,
        entity_label=entity,
        context_split=str(row.get("context_id", "")),
        entity_split=str(row.get("support_example_index", "")),
        input_ids=[int(token_id) for token_id in input_ids],
        attention_mask=[1] * len(input_ids),
        final_entity_token_pos=-1,
        attribute_generation=str(row.get("answer", "")),
        first_generated_token_id=int(row["target_first_token_id"]),
        is_correct=bool(row.get("model_correct_first_token", True)),
    )


def _pad_token(tokenizer) -> tuple[int, str]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return 0, "fallback_legacy_zero"
    return int(pad_token_id), "tokenizer.pad_token_id"


def _effective_batch_size(config: FEGAPipelineConfig) -> int:
    value = config.phases.data_prep.batch_size or config.llm_batch_size_override or 1
    return max(1, int(value))
