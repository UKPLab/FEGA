from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sae_bench.evals.ravel.instance import Prompt
from sae_bench.sae_bench_utils import activation_collection
from transformers.modeling_outputs import ModelOutput

from fega.core.compute_effect.artifacts import effect_direction
from fega.core.compute_effect.prompting import (
    AblationSpec,
    build_prompt_lookup,
    handle_oom_adjustment,
    prompt_from_meta,
    resolve_prompt,
)
from fega.core.positioning import (
    build_padded_prompt_batch,
    require_compatible_positioning,
)
from fega.core.resources import ModelResources
from fega.core.utils import ChunkProcessor

SUPPORTED_EFFECT_READOUTS = ("final_resid",)


@dataclass
class EffectContextRecord:
    index: int
    pair_index: int
    pair_role: str
    attribute_label: str
    feature_activation: float
    prompt: Prompt


class EffectInputLoader:
    """Load compute_effect inputs with best-effort JSON, Gram, and chunk caching."""

    def __init__(
        self,
        *,
        resources: ModelResources | None = None,
        cache_max_chunks: int = 2,
        cache_max_bytes: int = 1073741824,
    ) -> None:
        self.resources = resources
        self.cache_max_chunks = int(cache_max_chunks)
        self.cache_max_bytes = int(cache_max_bytes)
        self._local_json_cache: dict[str, Any] = {}
        self._local_gram_cache: dict[str, torch.Tensor] = {}
        self._local_tensor_cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._local_tensor_cache_bytes = 0
        self.load_counts = {"json": 0, "gram": 0, "chunks": 0}

    def load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        return self._load_json(Path(manifest_path))

    def validate_manifest_readouts(
        self, manifest_path: Path, requested_readouts: list[str] | None = None
    ) -> tuple[dict[str, Any], list[str]]:
        manifest = self.load_manifest(manifest_path)
        requested = list(dict.fromkeys(requested_readouts or ["final_resid"]))
        unsupported = [
            readout for readout in requested if readout not in SUPPORTED_EFFECT_READOUTS
        ]
        if unsupported:
            raise ValueError(
                "compute_effect supports only readouts "
                f"{', '.join(SUPPORTED_EFFECT_READOUTS)}; got {unsupported}."
            )
        readouts = set(manifest.get("readouts") or [])
        tensor_keys = set(manifest.get("tensor_keys") or [])
        missing = [
            readout
            for readout in requested
            if readout not in readouts and readout not in tensor_keys
        ]
        if missing:
            rendered = ", ".join(f"`{readout}`" for readout in missing)
            raise ValueError(
                "compute_effect requires data_prep activations with configured "
                f"readout(s) {rendered}; manifest {manifest_path} does not advertise "
                "them. Rerun data_prep with matching `phases.data_prep.readouts`."
            )
        require_compatible_positioning(
            manifest.get("positioning"),
            artifact_name=f"data_prep manifest {manifest_path}",
        )
        return manifest, requested

    def load_contexts(self, contexts_path: Path) -> dict[int, list[dict[str, Any]]]:
        raw = self._load_json(Path(contexts_path))
        if not isinstance(raw, dict):
            raise ValueError(f"feature contexts must be a mapping: {contexts_path}")
        return {
            int(feature_id): list(records or []) for feature_id, records in raw.items()
        }

    def load_prompt_lookup(
        self, pairs_path: Path
    ) -> dict[str, dict[str, list[Prompt]]]:
        cached = self._get_cached_json(Path(pairs_path))
        if cached is not None:
            return cached
        lookup = build_prompt_lookup(Path(pairs_path))
        self._cache_json(Path(pairs_path), lookup)
        self.load_counts["json"] += 1
        return lookup

    def load_gram(
        self, gram_path: Path, gram_meta_path: Path
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        gram_path = Path(gram_path)
        gram_meta = self._load_json(Path(gram_meta_path))
        cached = self._get_cached_gram(gram_path)
        if cached is not None:
            return cached, gram_meta
        gram = torch.load(gram_path, map_location="cpu")
        self.load_counts["gram"] += 1
        self._cache_gram(gram_path, gram)
        return gram, gram_meta

    def load_activation_chunk(
        self,
        tensors_path: Path,
        meta_path: Path,
        *,
        requested_readouts: list[str] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tensors_path = Path(tensors_path)
        meta_path = Path(meta_path)
        cache_key = self._cache_key(tensors_path)
        cached = self._get_cached_tensor(tensors_path)
        if cached is not None:
            self._validate_activation_chunk(
                cached[0], cached[1], tensors_path, requested_readouts
            )
            return cached
        tensors = torch.load(tensors_path, map_location="cpu")
        with open(meta_path) as mf:
            meta_lines = [json.loads(line) for line in mf]
        self._validate_activation_chunk(
            tensors, meta_lines, tensors_path, requested_readouts
        )
        payload = (tensors, meta_lines)
        self.load_counts["chunks"] += 1
        self._cache_tensor(cache_key, payload, self._estimate_bytes(tensors))
        return payload

    def load_examples(
        self,
        *,
        manifest_path: Path,
        activations_dir: Path,
        target_indices: list[int],
        requested_readouts: list[str],
    ) -> tuple[dict[int, dict[str, Any]], set[int]]:
        """Load only z, requested readouts, and metadata needed by target indices."""
        examples: dict[int, dict[str, Any]] = {}
        missing = {int(idx) for idx in target_indices}
        if not missing:
            return examples, missing
        for tensors_path, meta_path in ChunkProcessor.stream(
            manifest_path, activations_dir
        ):
            tensors, meta_lines = self.load_activation_chunk(
                tensors_path,
                meta_path,
                requested_readouts=requested_readouts,
            )
            idx_list = tensors.get("index")
            idx_values = (
                idx_list.tolist() if isinstance(idx_list, torch.Tensor) else idx_list
            )
            idx_map = {int(idx): row for row, idx in enumerate(idx_values)}
            hit_now = [idx for idx in list(missing) if idx in idx_map]
            for idx in hit_now:
                row = idx_map[idx]
                readout_payload = {
                    readout: tensors[readout][row].detach().cpu().clone()
                    for readout in requested_readouts
                }
                example = {
                    "z": tensors["z"][row].detach().cpu().clone(),
                    "readouts": readout_payload,
                    "meta": dict(meta_lines[row]),
                }
                example.update(readout_payload)
                examples[idx] = example
            missing.difference_update(hit_now)
            if not missing:
                break
        return examples, missing

    def _load_json(self, path: Path) -> Any:
        cached = self._get_cached_json(path)
        if cached is not None:
            return cached
        with open(path) as f:
            payload = json.load(f)
        self._cache_json(path, payload)
        self.load_counts["json"] += 1
        return payload

    def _get_cached_json(self, path: Path) -> Any | None:
        if self.resources is not None:
            cached = self.resources.get_cached_json(path)
            if cached is not None:
                return cached
        return self._local_json_cache.get(self._cache_key(path))

    def _cache_json(self, path: Path, payload: Any) -> None:
        if self.resources is not None:
            self.resources.cache_json(path, payload)
        self._local_json_cache[self._cache_key(path)] = payload

    def _get_cached_gram(self, path: Path) -> torch.Tensor | None:
        key = self._cache_key(path)
        if self.resources is not None:
            cache = getattr(self.resources, "_compute_effect_gram_cache", None)
            if isinstance(cache, dict) and key in cache:
                return cache[key]
        return self._local_gram_cache.get(key)

    def _cache_gram(self, path: Path, gram: torch.Tensor) -> None:
        key = self._cache_key(path)
        if self.resources is not None:
            cache = getattr(self.resources, "_compute_effect_gram_cache", None)
            if cache is None:
                cache = {}
                setattr(self.resources, "_compute_effect_gram_cache", cache)
            cache[key] = gram
        self._local_gram_cache[key] = gram

    def _get_cached_tensor(self, path: Path | str) -> Any | None:
        cache = self._tensor_cache()
        key = self._cache_key(path)
        if key not in cache:
            return None
        payload, size = cache.pop(key)
        cache[key] = (payload, size)
        return payload

    def _cache_tensor(self, path: Path | str, payload: Any, size: int) -> None:
        if self.cache_max_chunks <= 0 or self.cache_max_bytes <= 0:
            return
        cache = self._tensor_cache()
        key = self._cache_key(path)
        if key in cache:
            _, old_size = cache.pop(key)
            self._set_tensor_cache_bytes(self._tensor_cache_bytes() - old_size)
        cache[key] = (payload, size)
        self._set_tensor_cache_bytes(self._tensor_cache_bytes() + size)
        while (
            len(cache) > self.cache_max_chunks
            or self._tensor_cache_bytes() > self.cache_max_bytes
        ):
            _, (_, evicted_size) = cache.popitem(last=False)
            self._set_tensor_cache_bytes(self._tensor_cache_bytes() - evicted_size)

    def _tensor_cache(self) -> OrderedDict[str, tuple[Any, int]]:
        if self.resources is None:
            return self._local_tensor_cache
        cache = getattr(self.resources, "_compute_effect_tensor_cache", None)
        if cache is None:
            cache = OrderedDict()
            setattr(self.resources, "_compute_effect_tensor_cache", cache)
            setattr(self.resources, "_compute_effect_tensor_cache_bytes", 0)
        return cache

    def _tensor_cache_bytes(self) -> int:
        if self.resources is None:
            return self._local_tensor_cache_bytes
        return int(getattr(self.resources, "_compute_effect_tensor_cache_bytes", 0))

    def _set_tensor_cache_bytes(self, value: int) -> None:
        if self.resources is None:
            self._local_tensor_cache_bytes = max(0, int(value))
        else:
            setattr(
                self.resources, "_compute_effect_tensor_cache_bytes", max(0, int(value))
            )

    @staticmethod
    def _cache_key(path: Path | str) -> str:
        try:
            return str(Path(path).resolve())
        except OSError:
            return str(path)

    @classmethod
    def _estimate_bytes(cls, payload: Any) -> int:
        if isinstance(payload, torch.Tensor):
            return payload.numel() * payload.element_size()
        if isinstance(payload, dict):
            return sum(cls._estimate_bytes(value) for value in payload.values())
        if isinstance(payload, (list, tuple)):
            return sum(cls._estimate_bytes(value) for value in payload)
        if isinstance(payload, (int, float)):
            return 8
        return 0

    @staticmethod
    def _validate_activation_chunk(
        tensors: dict[str, Any],
        meta_lines: list[dict[str, Any]],
        tensors_path: Path,
        requested_readouts: list[str] | None,
    ) -> None:
        readouts = list(dict.fromkeys(requested_readouts or []))
        required = {"index", "z", *readouts}
        missing = sorted(required.difference(tensors))
        if missing:
            raise ValueError(
                f"Missing tensor keys {missing} in {tensors_path}. Rerun data_prep "
                "with matching `phases.data_prep.readouts` so compute_effect can "
                "load the configured readout(s)."
            )
        z = tensors["z"]
        if z.dim() != 2:
            raise ValueError(
                f"Expected rank-2 z in {tensors_path}, got {tuple(z.shape)}."
            )
        row_counts = {"z": z.shape[0], "meta": len(meta_lines)}
        for readout in readouts:
            tensor = tensors[readout]
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                raise ValueError(
                    f"Expected rank-2 {readout} in {tensors_path}, "
                    f"got {tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor)}."
                )
            row_counts[readout] = tensor.shape[0]
        if any(count != z.shape[0] for count in row_counts.values()):
            raise ValueError(
                "Activation chunk row mismatch in "
                f"{tensors_path}: "
                + ", ".join(f"{name}={count}" for name, count in row_counts.items())
                + "."
            )


def build_effect_context_records(
    *,
    feature_id: int,
    raw_contexts: list[dict[str, Any]],
    example_bank: dict[int, dict[str, Any]],
    prompt_lookup: dict[str, dict[str, list[Prompt]]] | None,
    tokenizer,
    default_attr: str,
) -> tuple[list[EffectContextRecord], dict[str, int]]:
    """Resolve contexts while preserving canonical prompt and value provenance.

    RAVEL prompt lookup is keyed by the raw attribute type (or the configured
    default), while the raw attribute label remains the context's value identity.
    A present lookup fails closed on misses; metadata reconstruction is reserved
    for induction-style inputs that have no prompt lookup.
    """
    # Resolve each source context without crossing the lookup/fallback boundary.
    records: list[EffectContextRecord] = []
    stats = {
        "skipped_missing_example": 0,
        "skipped_missing_prompt": 0,
        "skipped_invalid": 0,
    }
    for raw in raw_contexts:
        idx = raw.get("index")
        if idx is None:
            stats["skipped_invalid"] += 1
            continue
        idx = int(idx)
        example = example_bank.get(idx)
        if example is None:
            stats["skipped_missing_example"] += 1
            continue
        pair_idx = raw.get("pair_index")
        pair_role = raw.get("pair_role")
        attr_label = raw.get("attribute_label") or default_attr
        lookup_attr = raw.get("attribute_type") or default_attr
        if pair_idx is None or not pair_role or not attr_label:
            stats["skipped_invalid"] += 1
            continue
        prompt = None
        if prompt_lookup is not None:
            prompt = resolve_prompt(
                prompt_lookup, lookup_attr, pair_role, int(pair_idx)
            )
        else:
            try:
                prompt = prompt_from_meta(example["meta"], tokenizer)
            except Exception:
                prompt = None
        if prompt is None:
            stats["skipped_missing_prompt"] += 1
            continue
        feature_activation = raw.get("z")
        if feature_activation is None:
            z = example["z"]
            if int(feature_id) >= int(z.numel()):
                stats["skipped_invalid"] += 1
                continue
            feature_activation = float(z[int(feature_id)].item())
        records.append(
            EffectContextRecord(
                index=idx,
                pair_index=int(pair_idx),
                pair_role=str(pair_role),
                attribute_label=str(attr_label),
                feature_activation=float(feature_activation),
                prompt=prompt,
            )
        )
    return records, stats


@torch.no_grad()
def run_ablation_readouts_batch(
    model,
    sae,
    tokens: torch.Tensor,
    attn: torch.Tensor,
    target_positions: list[int],
    z_batch: torch.Tensor,
    ablation_spec: AblationSpec,
    *,
    position_ids: torch.Tensor,
    requested_readouts: list[str],
) -> dict[str, list[torch.Tensor]]:
    """Ablate SAE features and capture exact LM-head inputs plus diagnostics."""
    # Validate positioning and prepare the ablated SAE reconstruction.
    if position_ids.shape != tokens.shape:
        raise ValueError(
            f"position_ids shape {tuple(position_ids.shape)} != tokens shape {tuple(tokens.shape)}."
        )
    first_param = next(model.parameters())
    model_device = first_param.device
    model_dtype = first_param.dtype
    requested = list(dict.fromkeys(requested_readouts))
    unsupported = [
        readout for readout in requested if readout not in SUPPORTED_EFFECT_READOUTS
    ]
    if unsupported:
        raise ValueError(f"Unsupported compute_effect readout(s): {unsupported}.")
    z_ablate = z_batch.clone()
    feature_ids = ablation_spec.feature_ids.to(device=z_ablate.device, dtype=torch.long)
    if feature_ids.numel() != z_ablate.shape[0]:
        raise ValueError("feature_ids must have one entry per batch row.")
    row_ids = torch.arange(z_ablate.shape[0], device=z_ablate.device)
    z_ablate[row_ids, feature_ids] = 0.0
    decoded = sae.decode(z_ablate).to(device=model_device, dtype=model_dtype)

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
            modified[b_idx, pos, :] = decoded[b_idx].to(dtype=modified.dtype)
        if isinstance(outputs, ModelOutput):
            outputs_tuple = outputs.to_tuple()
            new_outputs = type(outputs)(*(modified, *outputs_tuple[1:]))
        elif isinstance(outputs, tuple):
            new_outputs = (modified, *outputs[1:])
        else:
            new_outputs = modified
        return new_outputs

    layer = sae.cfg.hook_layer
    hook_handle = activation_collection.get_module(model, layer).register_forward_hook(
        replace_resid
    )
    if "final_resid" in requested:
        output_embeddings = model.get_output_embeddings()
        if not isinstance(output_embeddings, torch.nn.Module):
            raise RuntimeError("Model output embeddings must be a torch module.")

        def capture_readout_input(module, inputs):
            """Capture the tensor supplied directly to the output embedding."""
            # Require a single tensor argument and validate capture count below.
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
    captured_final_resid = None
    if "final_resid" in requested:
        if len(captured_readout_inputs) != 1:
            raise RuntimeError(
                "Expected exactly one output-embedding input capture, got "
                f"{len(captured_readout_inputs)}."
            )
        captured_final_resid = captured_readout_inputs[0]
        if captured_final_resid.dim() != 3:
            raise RuntimeError(
                "Output-embedding input must be rank-3, got "
                f"{tuple(captured_final_resid.shape)}."
            )
    out: dict[str, list[torch.Tensor]] = {readout: [] for readout in requested}
    for b_idx, pos in enumerate(target_positions):
        if "final_resid" in out:
            out["final_resid"].append(
                captured_final_resid[b_idx, pos, :]
                .detach()
                .cpu()
                .to(dtype=torch.float32)
            )
    return out


def compute_feature_effect_rows(
    *,
    feature_id: int,
    context_records: list[EffectContextRecord],
    example_bank: dict[int, dict[str, Any]],
    model,
    sae,
    requested_readouts: list[str],
    gram: torch.Tensor | None,
    batch_size: int,
    pad_token_id: int,
    positioning_schema_version: int,
    normalization_eps: float,
    tau_zero: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Compute retained effect rows for one feature, grouped by readout."""
    requested = list(dict.fromkeys(requested_readouts))
    rows_by_readout: dict[str, list[dict[str, Any]]] = {
        readout: [] for readout in requested
    }
    stats_by_readout: dict[str, dict[str, Any]] = {
        readout: _empty_compute_stats() for readout in requested
    }
    if "final_resid" in requested and gram is None:
        raise ValueError("final_resid compute_effect rows require a Gram tensor.")
    if not context_records:
        return rows_by_readout, stats_by_readout
    sae_device, sae_dtype = _sae_device_dtype(sae)
    model_device = next(model.parameters()).device
    # Preserve source order so candidate identity and its mask are canonical.
    prepared = list(context_records)
    current_bs = max(1, int(batch_size))
    idx = 0
    while idx < len(prepared):
        chunk = prepared[idx : idx + current_bs]
        (
            tokens,
            attn,
            position_ids,
            target_positions,
            z_batch,
            base_readouts,
        ) = _prep_readout_batch(
            chunk,
            example_bank,
            requested,
            model_device,
            sae_device,
            sae_dtype,
            pad_token_id=pad_token_id,
            positioning_schema_version=positioning_schema_version,
        )
        ablation_spec = AblationSpec(
            feature_ids=torch.full(
                (len(chunk),), int(feature_id), dtype=torch.long, device=sae_device
            )
        )
        try:
            ablated = run_ablation_readouts_batch(
                model=model,
                sae=sae,
                tokens=tokens,
                attn=attn,
                target_positions=target_positions,
                z_batch=z_batch,
                ablation_spec=ablation_spec,
                position_ids=position_ids,
                requested_readouts=requested,
            )
        except torch.cuda.OutOfMemoryError:
            current_bs = handle_oom_adjustment(
                current_bs, list(stats_by_readout.values())
            )
            continue
        chunk_mask = [True] * len(chunk)
        if "final_resid" in requested:
            final_rows, chunk_mask = _build_final_resid_rows(
                records=chunk,
                bases=base_readouts["final_resid"],
                ablated=ablated["final_resid"],
                gram=gram,
                tau_zero=tau_zero,
                stats=stats_by_readout["final_resid"],
            )
            rows_by_readout["final_resid"].extend(final_rows)
        identities = [_record_identity(record) for record in chunk]
        for readout in requested:
            stats_by_readout[readout]["candidate_identity"].extend(identities)
            stats_by_readout[readout]["retained_mask"].extend(chunk_mask)
        idx += current_bs
    return rows_by_readout, stats_by_readout


def _prep_readout_batch(
    records: list[EffectContextRecord],
    example_bank: dict[int, dict[str, Any]],
    requested_readouts: list[str],
    model_device: torch.device,
    sae_device: torch.device,
    sae_dtype: torch.dtype,
    *,
    pad_token_id: int,
    positioning_schema_version: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor,
    dict[str, list[torch.Tensor]],
]:
    prompts = [record.prompt for record in records]
    prompt_batch = build_padded_prompt_batch(
        prompts,
        device=model_device,
        pad_token_id=pad_token_id,
        original_indices=[record.index for record in records],
        positioning_schema_version=positioning_schema_version,
    )
    z_batch = torch.stack([example_bank[record.index]["z"] for record in records]).to(
        device=sae_device, dtype=sae_dtype
    )
    base_readouts = {
        readout: [
            _example_readout(example_bank[record.index], readout) for record in records
        ]
        for readout in requested_readouts
    }
    return (
        prompt_batch.input_ids,
        prompt_batch.attention_mask,
        prompt_batch.position_ids,
        prompt_batch.target_positions,
        z_batch,
        base_readouts,
    )


def _build_final_resid_rows(
    *,
    records: list[EffectContextRecord],
    bases: list[torch.Tensor],
    ablated: list[torch.Tensor],
    gram: torch.Tensor | None,
    tau_zero: float,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[bool]]:
    """Filter and exactly Gram-normalize final-residual candidate rows."""
    # Evaluate every candidate in source order and retain one authoritative mask.
    if gram is None:
        raise ValueError("final_resid rows require a Gram tensor.")
    retained_mask: list[bool] = []
    rows: list[dict[str, Any]] = []
    for record, base_raw, abl_raw in zip(records, bases, ablated):
        base = base_raw.detach().cpu().to(dtype=torch.float32)
        abl = abl_raw.detach().cpu().to(dtype=torch.float32)
        if base.dim() != 1 or abl.dim() != 1:
            raise ValueError(
                "Expected rank-1 final_resid readouts, got "
                f"base={tuple(base.shape)}, ablated={tuple(abl.shape)}."
            )
        if base.shape != abl.shape:
            raise ValueError(
                "final_resid readout shape mismatch: "
                f"base={tuple(base.shape)}, ablated={tuple(abl.shape)}."
            )
        if not torch.isfinite(base).all() or not torch.isfinite(abl).all():
            stats["skipped_nonfinite"] += 1
            stats["skipped_invalid"] += 1
            retained_mask.append(False)
            continue
        delta = abl - base
        if not torch.isfinite(delta).all():
            stats["skipped_nonfinite"] += 1
            stats["skipped_invalid"] += 1
            retained_mask.append(False)
            continue
        if delta.numel() != int(gram.shape[0]):
            raise ValueError(
                "Gram shape is incompatible with final_resid width: "
                f"delta width {delta.numel()}, Gram width {gram.shape[0]}."
            )
        compute_dtype = (
            torch.float64 if gram.dtype == torch.float64 else torch.float32
        )
        delta_device = delta.to(device=gram.device, dtype=compute_dtype)
        gram_compute = gram.to(device=gram.device, dtype=compute_dtype)
        q = torch.sum((delta_device @ gram_compute) * delta_device)
        if not torch.isfinite(q):
            stats["skipped_nonfinite"] += 1
            stats["skipped_invalid"] += 1
            retained_mask.append(False)
            continue
        if float(q.item()) < 0.0:
            delta64 = delta.to(device=gram.device, dtype=torch.float64)
            gram64 = gram.to(device=gram.device, dtype=torch.float64)
            q = torch.sum((delta64 @ gram64) * delta64)
            if not torch.isfinite(q) or float(q.item()) < 0.0:
                stats["skipped_numerical_failure"] += 1
                stats["skipped_invalid"] += 1
                retained_mask.append(False)
                continue
        magnitude_value = float(torch.sqrt(q).detach().cpu().item())
        if magnitude_value <= float(tau_zero):
            stats["skipped_near_zero"] += 1
            retained_mask.append(False)
            continue
        direction = effect_direction(delta, torch.sqrt(q))
        if not torch.isfinite(direction).all():
            stats["skipped_nonfinite"] += 1
            stats["skipped_invalid"] += 1
            retained_mask.append(False)
            continue
        persisted_direction = direction.detach().cpu().to(dtype=torch.float32)
        unit_q = torch.sum(
            (
                persisted_direction.to(device=gram.device, dtype=torch.float64)
                @ gram.to(device=gram.device, dtype=torch.float64)
            )
            * persisted_direction.to(device=gram.device, dtype=torch.float64)
        )
        rows.append(
            {
                "context_index": record.index,
                "pair_index": record.pair_index,
                "pair_role": record.pair_role,
                "attribute_label": record.attribute_label,
                "feature_activation": record.feature_activation,
                "delta": delta,
                "magnitude": magnitude_value,
                "direction": persisted_direction,
                "unit_gram_norm_error": abs(float(unit_q.item()) - 1.0),
            }
        )
        retained_mask.append(True)
        stats["kept"] += 1
    return rows, retained_mask


def _empty_compute_stats() -> dict[str, Any]:
    return {
        "kept": 0,
        "skipped_near_zero": 0,
        "skipped_zero_norm": 0,
        "skipped_nonfinite": 0,
        "skipped_invalid": 0,
        "skipped_numerical_failure": 0,
        "candidate_identity": [],
        "retained_mask": [],
        "oom_adjustments": [],
    }


def _record_identity(record: EffectContextRecord) -> dict[str, Any]:
    """Return the complete scientific identity for one ordered candidate row."""
    # Serialize the three fields required to replay the canonical mask.
    return {
        "attribute_label": record.attribute_label,
        "pair_role": record.pair_role,
        "pair_index": record.pair_index,
    }


def _example_readout(example: dict[str, Any], readout: str) -> torch.Tensor:
    readouts = example.get("readouts")
    if isinstance(readouts, dict) and readout in readouts:
        return readouts[readout]
    return example[readout]


def _sae_device_dtype(sae) -> tuple[torch.device, torch.dtype]:
    if hasattr(sae, "W_dec"):
        return sae.W_dec.device, sae.W_dec.dtype
    first_param = next(sae.parameters())
    return first_param.device, first_param.dtype
