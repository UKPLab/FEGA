import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.common import require_single_entity_attr, selection_seed
from fega.core.ravel_scores import (
    annotate_contexts_with_scores,
    extract_instance_scores_from_payload,
)
from fega.core.resources import ModelResources
from fega.core.utils import ChunkProcessor, ensure_dir, load_mdbm_mask
from fega.core.utils.ravel import ReplayContext, mdbm_weight_path
from fega.paths import (
    data_prep_activations_dir,
    data_prep_collect_dir,
    data_prep_select_dir,
)

_logger = logging.getLogger(__name__)


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _prompt_digest(rec: dict) -> str:
    parts = [
        rec.get("prompt", ""),
        rec.get("entity_label", ""),
        rec.get("attribute_type", ""),
        rec.get("attribute_label", ""),
    ]
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def _stable_prompt_identity(rec: dict) -> tuple[int, str, int]:
    missing_pair_index = 2**63 - 1
    pair_index = _coerce_int(rec.get("pair_index"), missing_pair_index)
    index = _coerce_int(rec.get("index"), missing_pair_index)
    return (pair_index, _prompt_digest(rec), index)


def _selection_sort_key(kv: tuple[float, dict]) -> tuple[float, tuple[int, str, int]]:
    z_val, rec = kv
    return (-z_val, _stable_prompt_identity(rec))


def _stratified_topk(
    records: list[tuple[float, dict]], max_contexts: int, seed: int | None = None
) -> list[dict]:
    """Bucket-aware top-k: keep diverse contexts across (entity_label, attribute_type) buckets."""
    rng = random.Random(seed) if seed is not None else None
    buckets: dict[tuple[str, str], list[tuple[float, dict]]] = defaultdict(list)
    for z_val, rec in records:
        key = (rec.get("entity_label", ""), rec.get("attribute_type", ""))
        buckets[key].append((z_val, rec))

    # Sort each bucket by z descending (shuffle first if seeded to randomize tie-breaking)
    for key, recs in buckets.items():
        if rng:
            rng.shuffle(recs)
        recs.sort(key=_selection_sort_key)

    selected: list[tuple[float, dict]] = []

    # Take top-1 from each bucket
    bucket_order = list(buckets.keys())
    if rng:
        rng.shuffle(bucket_order)
    else:
        bucket_order.sort()
    for key in bucket_order:
        if buckets[key]:
            selected.append(buckets[key][0])

    # Round-robin over remaining entries
    pointers = {key: 1 for key in buckets}  # next index per bucket
    bucket_keys = list(bucket_order)
    idx = 0
    max_per_bucket = max(1, max_contexts // max(len(bucket_keys), 1))
    while len(selected) < max_contexts and bucket_keys:
        key = bucket_keys[idx % len(bucket_keys)]
        if pointers[key] < len(buckets[key]) and pointers[key] < max_per_bucket:
            selected.append(buckets[key][pointers[key]])
            pointers[key] += 1
        else:
            bucket_keys.remove(key)
            if not bucket_keys:
                break
            idx -= 1  # stay on current position after removal
        idx += 1

    # If still underfilled, backfill proportionally by remaining bucket sizes
    if len(selected) < max_contexts:
        remaining_per_bucket: dict[tuple[str, str], list[tuple[float, dict]]] = {}
        for key in buckets:
            remaining_per_bucket[key] = buckets[key][pointers[key] :]
        total_remaining = sum(len(v) for v in remaining_per_bucket.values())
        if total_remaining > 0:
            # allocate slots proportionally, but at least 1 if bucket has remaining
            slots_left = max_contexts - len(selected)
            allocations = {}
            for key, items in remaining_per_bucket.items():
                if not items:
                    continue
                alloc = max(1, int((len(items) / total_remaining) * slots_left))
                allocations[key] = min(alloc, len(items))
            # round-robin through allocated sets for determinism
            keys_cycle = list(allocations.keys())
            if rng:
                rng.shuffle(keys_cycle)
            else:
                keys_cycle.sort()
            ptrs = {k: 0 for k in keys_cycle}
            idx_rr = 0
            while len(selected) < max_contexts and keys_cycle:
                key = keys_cycle[idx_rr % len(keys_cycle)]
                if ptrs[key] < allocations[key]:
                    selected.append(remaining_per_bucket[key][ptrs[key]])
                    ptrs[key] += 1
                else:
                    keys_cycle.remove(key)
                    if not keys_cycle:
                        break
                    idx_rr -= 1
                idx_rr += 1

    selected = sorted(selected, key=_selection_sort_key)[:max_contexts]
    return [rec for _, rec in selected]


def _filter_by_tau(
    feature_ctx: dict[int, list[tuple[float, dict]]],
    feature_stats: dict[int, dict[str, int]],
    z_chunk: torch.Tensor,
    meta_lines: list[dict],
    target_set: set[int],
    tau: float,
    max_contexts: int,
):
    for row_idx in range(z_chunk.shape[0]):
        meta = meta_lines[row_idx]
        z_row = z_chunk[row_idx]
        for j in target_set:
            z_val = z_row[j].item()
            if z_val > tau:
                record = {
                    "index": meta.get("index", row_idx),
                    "pair_index": meta.get("pair_index"),
                    "pair_role": meta.get("pair_role"),
                    "attribute_label": meta.get("attribute_label"),
                    "attribute_type": meta.get("attribute_type"),
                    "entity_label": meta.get("entity_label"),
                    "prompt": meta.get("prompt"),
                    "z": float(z_val),
                }
                if record["pair_index"] is None or record["pair_role"] is None:
                    feature_stats[j]["missing_pair_meta"] += 1
                    continue
                feature_ctx[j].append((record["z"], record))
                feature_stats[j]["kept"] = min(len(feature_ctx[j]), max_contexts)
            else:
                feature_stats[j]["skipped_tau"] += 1


def _enforce_min_max(
    feature_ctx: dict[int, list[tuple[float, dict]]],
    feature_stats: dict[int, dict[str, int]],
    max_contexts: int,
    min_contexts: int,
    seed: int | None,
) -> dict[int, list[dict]]:
    contexts: dict[int, list[dict]] = {}
    for j, recs in feature_ctx.items():
        if not recs:
            contexts[j] = []
            continue
        contexts[j] = _stratified_topk(recs, max_contexts, seed=seed)
        if len(recs) > max_contexts:
            feature_stats[j]["capped"] += len(recs) - max_contexts
        if not contexts[j]:
            print(
                f"Warning: no contexts selected for feature {j} (kept={feature_stats[j]['kept']}, skipped_tau={feature_stats[j]['skipped_tau']}, missing_meta={feature_stats[j]['missing_pair_meta']})"
            )
        if len(contexts[j]) < min_contexts:
            feature_stats[j]["too_rare"] = len(contexts[j])
            contexts[j] = []
    return contexts


def select_contexts(
    manifest_path: Path,
    activations_dir: Path,
    target_features: list[int],
    tau_act: float,
    max_contexts: int,
    min_contexts: int,
    seed: int | None = None,
) -> tuple[dict[int, list[dict]], dict[int, dict[str, int]]]:
    """Load activation chunks, filter by `tau_act`, and return contexts plus counters."""
    feature_ctx: dict[int, list[tuple[float, dict]]] = {j: [] for j in target_features}
    feature_stats: dict[int, dict[str, int]] = {
        j: {
            "kept": 0,
            "skipped_tau": 0,
            "capped": 0,
            "missing_pair_meta": 0,
            "too_rare": 0,
        }
        for j in target_features
    }
    target_set = set(target_features)

    for tensors_path, meta_path in ChunkProcessor.stream(
        manifest_path, activations_dir
    ):
        tensors = torch.load(tensors_path, map_location="cpu")
        z_chunk: torch.Tensor = tensors["z"]
        if z_chunk.dim() != 2:
            raise ValueError(
                f"Expected z chunk to be rank-2, got shape {tuple(z_chunk.shape)} from {tensors_path}"
            )
        with open(meta_path) as mf:
            meta_lines = [json.loads(line) for line in mf]
        if len(meta_lines) != z_chunk.shape[0]:
            raise ValueError(
                f"Meta length {len(meta_lines)} != z rows {z_chunk.shape[0]} for chunk {tensors_path}"
            )
        z_width = z_chunk.shape[1]
        if any(j >= z_width for j in target_set):
            raise ValueError(
                f"Mask feature index exceeds z width: max feature {max(target_set)} vs z dim {z_width} (chunk {tensors_path})"
            )
        _filter_by_tau(
            feature_ctx,
            feature_stats,
            z_chunk,
            meta_lines,
            target_set,
            tau_act,
            max_contexts,
        )

    contexts = _enforce_min_max(
        feature_ctx, feature_stats, max_contexts, min_contexts, seed
    )
    return contexts, feature_stats


def _run_context_selection(
    config: FEGAPipelineConfig, resources: ModelResources | None = None
) -> Path:
    """Select MDBM feature contexts from data-prep activation artifacts."""
    entity, attr = require_single_entity_attr(config)
    data_prep = config.phases.data_prep
    collect_dir = data_prep_collect_dir(config, entity, attr)
    activations_dir = data_prep_activations_dir(config, entity, attr)
    manifest_path = activations_dir / "activations_manifest.json"
    pairs_path = collect_dir / "pairs_full.json"
    missing: list[str] = [
        str(p) for p in (activations_dir, manifest_path, pairs_path) if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required selection inputs missing:\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    ctx = ReplayContext.from_file(config.reference_json)
    weight_path = (
        config.mdbm_weight_path
        if config.mdbm_weight_path
        else mdbm_weight_path(
            ctx,
            entity,
            attr,
            ctx.reference_path,
            mdbm_root_override=config.mdbm_root,
        )
    )
    if not weight_path.exists():
        raise FileNotFoundError(f"MDBM weights not found at {weight_path}")
    mask = (
        resources.get_mdbm_mask(entity, attr, weight_path)
        if resources
        else load_mdbm_mask(weight_path)
    )
    if hasattr(mask, "detach"):
        mask = mask.detach()
    target_features = [idx for idx, val in enumerate(mask) if float(val) > 0]

    contexts, feature_stats = select_contexts(
        manifest_path,
        activations_dir,
        target_features,
        tau_act=data_prep.tau_act,
        max_contexts=data_prep.max_contexts,
        min_contexts=data_prep.min_contexts,
        seed=selection_seed(config),
    )
    score_maps = extract_instance_scores_from_payload(ctx.raw_reference, entity, attr)
    score_coverage = annotate_contexts_with_scores(contexts, score_maps)
    score_summary = {"instances": score_maps.stats, "contexts": score_coverage}
    out_dir = data_prep_select_dir(config, entity, attr)
    ensure_dir(out_dir)
    contexts_path = out_dir / "feature_contexts.json"
    summary_path = out_dir / "feature_contexts_summary.json"
    _write_outputs(
        contexts_path,
        summary_path,
        contexts,
        feature_stats,
        entity,
        attr,
        config,
        target_features,
        weight_path,
        manifest_path,
        score_summary,
    )
    _logger.info(
        "Selected contexts for %d/%d features (tau_act=%s, max=%s, min=%s)",
        sum(1 for ctxs in contexts.values() if ctxs),
        len(target_features),
        data_prep.tau_act,
        data_prep.max_contexts,
        data_prep.min_contexts,
    )
    return contexts_path


def _write_outputs(
    contexts_path: Path,
    summary_path: Path,
    contexts: dict[int, list[dict]],
    feature_stats: dict[int, dict[str, int]],
    entity: str,
    attribute: str,
    config: FEGAPipelineConfig,
    target_features: list[int],
    weight_path: Path,
    manifest_path: Path,
    score_summary: dict | None = None,
) -> None:
    """Write canonical data-prep selection outputs and summary counters."""
    data_prep = config.phases.data_prep
    with open(contexts_path, "w") as f:
        json.dump(contexts, f, indent=2)
    dropped_totals = {
        "skipped_tau": sum(stats["skipped_tau"] for stats in feature_stats.values()),
        "capped": sum(stats["capped"] for stats in feature_stats.values()),
        "missing_pair_meta": sum(
            stats.get("missing_pair_meta", 0) for stats in feature_stats.values()
        ),
        "too_rare": sum(
            1 for stats in feature_stats.values() if stats.get("too_rare", 0) > 0
        ),
    }
    too_rare_features = [
        j for j, stats in feature_stats.items() if stats.get("too_rare", 0) > 0
    ]
    summary = {
        "total_features": len(target_features),
        "features_with_contexts": sum(1 for v in contexts.values() if v),
        "tau_act": data_prep.tau_act,
        "max_contexts": data_prep.max_contexts,
        "min_contexts": data_prep.min_contexts,
        "entity_class": entity,
        "cause_attribute": attribute,
        "manifest_path": str(manifest_path),
        "weight_path": str(weight_path),
        "feature_stats": feature_stats,
        "dropped_totals": dropped_totals,
        "too_rare_features": too_rare_features,
    }
    if score_summary:
        summary["score_coverage"] = score_summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def choose_features(args):
    """Resolve selection config, load MDBM mask, pick contexts, and write outputs."""
    ctx = ReplayContext.from_file(args.reference_json)
    effective_cfg = ctx.eval_config
    if args.entity_attribute_selection:
        effective_cfg.entity_attribute_selection = json.loads(
            args.entity_attribute_selection
        )
    if len(effective_cfg.entity_attribute_selection) != 1:
        raise ValueError(
            "Feature/context selection expects exactly one entity selection."
        )
    entity_class = list(effective_cfg.entity_attribute_selection.keys())[0]
    cause_attr = effective_cfg.entity_attribute_selection[entity_class][0]

    weight_path = (
        Path(args.weight_path)
        if args.weight_path
        else mdbm_weight_path(
            ctx,
            entity_class,
            cause_attr,
            ctx.reference_path,
            mdbm_root_override=Path(args.mdbm_root) if args.mdbm_root else None,
        )
    )
    mask = load_mdbm_mask(weight_path)
    target_features = [idx for idx, val in enumerate(mask) if val.item() > 0]

    activations_dir = Path(args.activations_dir)
    manifest_path = Path(args.manifest or activations_dir / "activations_manifest.json")
    ensure_dir(activations_dir)
    pairs_path = activations_dir.parent / "pairs_full.json"
    missing = []
    if not manifest_path.exists():
        missing.append(
            f"manifest not found at {manifest_path} (expected from data_prep collection artifacts or --manifest override)"
        )
    if not pairs_path.exists():
        missing.append(
            f"pairs_full.json not found at {pairs_path} (expected from data_prep collection artifacts)"
        )
    if missing:
        raise FileNotFoundError(
            "Required selection inputs missing:\n"
            + "\n".join(f"- {m}" for m in missing)
        )
    contexts, feature_stats = select_contexts(
        manifest_path,
        activations_dir,
        target_features,
        tau_act=args.tau,
        max_contexts=args.max_contexts,
        min_contexts=args.min_contexts,
        seed=args.selection_seed,
    )
    score_maps = extract_instance_scores_from_payload(
        ctx.raw_reference, entity_class, cause_attr
    )
    score_coverage = annotate_contexts_with_scores(contexts, score_maps)
    score_summary = {"instances": score_maps.stats, "contexts": score_coverage}

    out_dir = activations_dir.parent.parent / "select"
    ensure_dir(out_dir)
    contexts_path = out_dir / "feature_contexts.json"
    summary_path = out_dir / "feature_contexts_summary.json"

    with open(contexts_path, "w") as f:
        json.dump(contexts, f, indent=2)
    dropped_totals = {
        "skipped_tau": sum(stats["skipped_tau"] for stats in feature_stats.values()),
        "capped": sum(stats["capped"] for stats in feature_stats.values()),
        "missing_pair_meta": sum(
            stats.get("missing_pair_meta", 0) for stats in feature_stats.values()
        ),
        "too_rare": sum(
            1 for stats in feature_stats.values() if stats.get("too_rare", 0) > 0
        ),
    }
    too_rare_features = [
        j for j, stats in feature_stats.items() if stats.get("too_rare", 0) > 0
    ]
    summary = {
        "total_features": len(target_features),
        "features_with_contexts": sum(1 for v in contexts.values() if v),
        "tau_act": args.tau,
        "max_contexts": args.max_contexts,
        "min_contexts": args.min_contexts,
        "entity_class": entity_class,
        "cause_attribute": cause_attr,
        "manifest_path": str(manifest_path),
        "weight_path": str(weight_path),
        "feature_stats": feature_stats,
        "dropped_totals": dropped_totals,
        "too_rare_features": too_rare_features,
    }
    if score_summary:
        summary["score_coverage"] = score_summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote contexts to {contexts_path}")
    return contexts_path


def parse_args():
    """CLI for internal feature/context artifact selection."""
    p = argparse.ArgumentParser(
        description="Internal data-prep helper: choose SAE features and contexts."
    )
    p.add_argument(
        "--reference_json", required=True, help="Reference JSON from replay."
    )
    p.add_argument(
        "--activations_dir",
        required=True,
        help="Directory containing activations manifest/tensors.",
    )
    p.add_argument(
        "--manifest", default=None, help="Path to activations_manifest.json (optional)."
    )
    p.add_argument(
        "--mdbm_root", default=None, help="Override for MDBM root directory."
    )
    p.add_argument(
        "--weight_path",
        default=None,
        help="Explicit path to MDBM checkpoint (overrides mdbm_root lookup).",
    )
    p.add_argument(
        "--entity_attribute_selection",
        default=None,
        help='JSON string to override entity selection, e.g. {"city":["Country"]}',
    )
    p.add_argument(
        "--tau", type=float, default=0.0, help="Threshold on z_j to consider a context."
    )
    p.add_argument(
        "--max_contexts", type=int, default=32, help="Max contexts to keep per feature."
    )
    p.add_argument(
        "--min_contexts",
        type=int,
        default=8,
        help="Minimum contexts required per feature; below this features are dropped as too rare.",
    )
    p.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Seed for stratified top-k selection to make bucket sampling reproducible.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    choose_features(args)


if __name__ == "__main__":
    main()
