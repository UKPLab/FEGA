import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sae_bench.evals.ravel.instance as ravel_instance
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.instance import Prompt, RAVELFilteredDataset, RAVELInstance
from sae_bench.evals.ravel.intervention import get_prompt_pairs
from sae_bench.evals.ravel.main import LLM_NAME_MAP


@dataclass
class ReplayContext:
    """FEGA-owned reader for saved RAVEL reference metadata."""

    reference_path: Path
    raw_reference: dict[str, Any]
    eval_config: RAVELEvalConfig
    sae_lens_release_id: str | None = None
    sae_lens_id: str | None = None
    sae_cfg_dict: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, reference_json: str | Path) -> "ReplayContext":
        reference_path = Path(reference_json)
        with open(reference_path) as f:
            raw_reference: dict[str, Any] = json.load(f)
        eval_config = RAVELEvalConfig(**raw_reference.get("eval_config", {}))
        return cls(
            reference_path=reference_path,
            raw_reference=raw_reference,
            eval_config=eval_config,
            sae_lens_release_id=raw_reference.get("sae_lens_release_id"),
            sae_lens_id=raw_reference.get("sae_lens_id"),
            sae_cfg_dict=raw_reference.get("sae_cfg_dict"),
        )


def filtered_dataset_path(ctx: ReplayContext, entity_class: str) -> Path:
    """Return the native RAVEL filtered-dataset path for an entity class."""
    config = ctx.eval_config
    filename = ravel_instance.get_instance_name(
        entity_class,
        config.model_name,
        config.full_dataset_downsample,
        config.top_n_entities,
    )
    roots = _recorded_dir_candidates(ctx.reference_path, Path(config.artifact_dir))
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return roots[0] / filename


def load_or_create_filtered_dataset(
    ctx: ReplayContext,
    entity_class: str,
    *,
    tokenizer,
    model,
    force_recompute: bool = False,
    selected_attributes: list[str] | None = None,
) -> RAVELFilteredDataset:
    """Load a cached RAVEL filtered dataset or rebuild it through native RAVEL."""
    config = ctx.eval_config
    dataset_path = filtered_dataset_path(ctx, entity_class)
    if dataset_path.exists() and not force_recompute:
        return RAVELFilteredDataset.load(str(dataset_path))

    orig_batch_size = config.llm_batch_size
    config.llm_batch_size = orig_batch_size * 8
    try:
        artifact_dir = _recorded_dir_candidates(
            ctx.reference_path, Path(config.artifact_dir)
        )[0]
        full_dataset = RAVELInstance.create_from_files(
            config=config,
            entity_type=entity_class,
            data_dir=str(artifact_dir),
            tokenizer=tokenizer,
            model=model,
            model_name=config.model_name,
            attribute_types=(
                selected_attributes or config.entity_attribute_selection[entity_class]
            ),
            downsample=config.full_dataset_downsample,
        )
    finally:
        config.llm_batch_size = orig_batch_size

    return full_dataset.create_and_save_filtered_dataset(
        artifact_dir=str(artifact_dir),
        top_n_entities=config.top_n_entities,
    )


def _resolve_recorded_path(reference_path: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    candidates = _recorded_dir_candidates(reference_path, path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _recorded_dir_candidates(reference_path: Path, path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]

    canonical = path
    if path.parts and path.parts[0] in {"artifacts", "mdbm"}:
        canonical = Path("data") / path

    base = _find_reference_base(reference_path, path)
    candidates = [canonical]
    if canonical != path:
        candidates.append(path)
    if base is not None:
        candidates = [base / candidate for candidate in candidates]
    return candidates


def build_prompt_pairs(
    ctx: ReplayContext,
    dataset: RAVELFilteredDataset,
    cause_attribute: str,
    iso_attributes: list[str],
) -> dict[str, list[Prompt]]:
    """Build the prompt-pair dictionary FEGA persists from native RAVEL pairs."""
    config = ctx.eval_config
    rng_state = random.getstate()
    try:
        # Native RAVEL prompt sampling uses module-global random. Scope that RNG
        # use to the replay seed so unrelated callers cannot perturb data_prep.
        if config.random_seed is not None:
            random.seed(config.random_seed)
        cause_base_prompts, cause_source_prompts = get_prompt_pairs(
            dataset=dataset,
            base_attribute=cause_attribute,
            source_attribute=cause_attribute,
            n_interventions=config.num_pairs_per_attribute,
        )

        iso_base_prompts: list[Prompt] = []
        iso_source_prompts: list[Prompt] = []
        for iso_attr in iso_attributes:
            attr_base_prompts, attr_source_prompts = get_prompt_pairs(
                dataset=dataset,
                base_attribute=iso_attr,
                source_attribute=iso_attr,
                n_interventions=config.num_pairs_per_attribute,
            )
            iso_base_prompts.extend(attr_base_prompts)
            iso_source_prompts.extend(attr_source_prompts)

        combined = list(zip(iso_base_prompts, iso_source_prompts))
        if combined:
            random.shuffle(combined)
            iso_base_prompts, iso_source_prompts = map(list, zip(*combined))
    finally:
        random.setstate(rng_state)

    cause_length = len(cause_base_prompts)
    return {
        "cause_base_prompts": cause_base_prompts,
        "cause_source_prompts": cause_source_prompts,
        "iso_base_prompts": list(iso_base_prompts[:cause_length]),
        "iso_source_prompts": list(iso_source_prompts[:cause_length]),
    }


def mdbm_weight_path(
    ctx: ReplayContext,
    entity_class: str,
    cause_attribute: str,
    reference_path: str | Path | None = None,
    *,
    mdbm_root_override: str | Path | None = None,
) -> Path:
    """Resolve native RAVEL MDBM weights without relying on process cwd."""
    candidates = _mdbm_weight_candidates(
        ctx,
        entity_class,
        cause_attribute,
        Path(reference_path) if reference_path is not None else ctx.reference_path,
        Path(mdbm_root_override) if mdbm_root_override is not None else None,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _mdbm_weight_candidates(
    ctx: ReplayContext,
    entity_class: str,
    cause_attribute: str,
    reference_path: Path,
    mdbm_root_override: Path | None,
) -> list[Path]:
    config = ctx.eval_config
    root = Path(mdbm_root_override) if mdbm_root_override else Path(config.mdbm_dir)
    filename = (
        f"{entity_class}_{cause_attribute}_downsampled-"
        f"{config.full_dataset_downsample}_top-{config.top_n_entities}.pt"
    )

    base = _find_reference_base(reference_path, root)
    if mdbm_root_override is not None:
        resolved_root = root if root.is_absolute() or base is None else base / root
        resolved_roots = [resolved_root]
        recorded_roots = _recorded_dir_candidates(reference_path, Path(config.mdbm_dir))
        if recorded_roots and resolved_root == recorded_roots[0]:
            resolved_roots.extend(recorded_roots[1:])
    else:
        resolved_roots = _recorded_dir_candidates(reference_path, root)

    subpaths: list[Path] = [Path(reference_path.name)]
    if reference_path.is_absolute() and base is not None:
        subpaths.append(reference_path.relative_to(base))
    elif reference_path.is_absolute():
        pass
    else:
        subpaths.append(reference_path)

    candidates: list[Path] = []
    for resolved_root in resolved_roots:
        for subpath in subpaths:
            candidates.append(resolved_root / subpath / filename)

    native_dir = Path(os.path.join(str(root), str(reference_path)))
    candidates.append(native_dir / filename)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _find_reference_base(reference_path: Path, mdbm_root: Path) -> Path | None:
    if not reference_path.is_absolute():
        return None

    relative_root_exists = not mdbm_root.is_absolute()
    for ancestor in reference_path.parents:
        if relative_root_exists and (ancestor / mdbm_root).exists():
            return ancestor
        if (ancestor / ".git").exists():
            return ancestor
        if (ancestor / "external" / "sae_bench").is_dir() and (
            ancestor / "fega"
        ).is_dir():
            return ancestor
    return None


def load_filtered_dataset(
    eval_config, entity_class: str, reference_json: Path | None = None
):
    """
    Try to load the filtered dataset path recorded in the reference JSON if available,
    otherwise fall back to computed instance name.
    """
    artifact_dir = Path(eval_config.artifact_dir)
    if reference_json and not artifact_dir.is_absolute():
        artifact_dir = _resolve_recorded_path(Path(reference_json), artifact_dir)

    if reference_json:
        # Expect filtered dataset alongside artifacts; try recorded path
        cand = (
            artifact_dir
            / f"{entity_class}_{LLM_NAME_MAP.get(eval_config.model_name, eval_config.model_name).replace('/', '--')}_downsampled-{eval_config.full_dataset_downsample}_top-{eval_config.top_n_entities}-entities_filtered_dataset.json"
        )
        if cand.exists():
            return ravel_instance.RAVELFilteredDataset.load(str(cand))

    path = artifact_dir / ravel_instance.get_instance_name(
        entity_class,
        LLM_NAME_MAP.get(eval_config.model_name, eval_config.model_name),
        eval_config.full_dataset_downsample,
        eval_config.top_n_entities,
    )
    return ravel_instance.RAVELFilteredDataset.load(str(path))


def load_pairs_from_replay(reference_json: str | Path, entity_class: str):
    """Load stored pairs from the replay run if available."""
    pairs_path = Path("data/artifacts/ravel") / f"{entity_class}_pairs.json"
    if pairs_path.exists():
        import json

        with open(pairs_path) as f:
            return json.load(f)
    return None


def prompt_to_dict(p: Prompt) -> dict:
    """Convert a Prompt into a plain dict with JSON-friendly fields."""

    def _maybe_list(x):
        return x.tolist() if hasattr(x, "tolist") else x

    return {
        "text": p.text,
        "template": p.template,
        "attribute_type": p.attribute_type,
        "attribute_label": p.attribute_label,
        "entity_label": p.entity_label,
        "context_split": p.context_split,
        "entity_split": p.entity_split,
        "input_ids": _maybe_list(p.input_ids),
        "attention_mask": (
            _maybe_list(p.attention_mask) if p.attention_mask is not None else None
        ),
        "final_entity_token_pos": p.final_entity_token_pos,
        "first_generated_token_id": p.first_generated_token_id,
        "attribute_generation": getattr(p, "attribute_generation", None),
        "is_correct": getattr(p, "is_correct", None),
    }


def prompt_from_dict(d: dict) -> Prompt:
    """Reconstruct a Prompt from a dict (inverse of prompt_to_dict)."""
    return Prompt(
        text=d["text"],
        template=d["template"],
        attribute_type=d["attribute_type"],
        attribute_label=d["attribute_label"],
        entity_label=d["entity_label"],
        context_split=d["context_split"],
        entity_split=d["entity_split"],
        input_ids=d["input_ids"],
        attention_mask=d.get("attention_mask"),
        final_entity_token_pos=d.get("final_entity_token_pos"),
        attribute_generation=d.get("attribute_generation"),
        first_generated_token_id=d.get("first_generated_token_id"),
        is_correct=d.get("is_correct"),
    )
