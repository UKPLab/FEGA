"""Write durable per-feature FEGA Gram/logit equivalence evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from sae_bench.sae_bench_utils import activation_collection
from transformers.modeling_outputs import ModelOutput

from fega.config_schema import FEGAPipelineConfig
from fega.core.common import require_single_entity_attr
from fega.core.compute_effect.effects import (
    EffectInputLoader,
    _prep_readout_batch,
    _sae_device_dtype,
    build_effect_context_records,
    compute_feature_effect_rows,
    run_ablation_readouts_batch,
)
from fega.core.compute_effect.prompting import AblationSpec
from fega.core.compute_effect.runner import _validate_gram, _validate_gram_readout
from fega.core.data_prep.gram_cache import (
    canonical_unembedding,
    gram_fingerprint,
    unembedding_fingerprint,
)
from fega.core.gram_logit_equivalence import (
    FLOAT32_EQUIVALENCE_TOLERANCES,
    evaluate_grouped_gram_logit_equivalence,
)
from fega.core.resources import ModelResources
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.utils import ChunkProcessor
from fega.paths import (
    data_prep_activations_dir,
    data_prep_pairs_path,
    data_prep_select_dir,
    gram_cache_meta_path,
    gram_cache_tensor_path,
)

PLANNED_EQUIVALENCE_SELECTION: dict[str, Any] = {
    "model": {
        "repo_id": "google/gemma-2-2b",
        "revision": "c5ebcd40d208330abc697524c919956e692655cf",
        "dtype": "bfloat16",
    },
    "sae": {
        "repo_id": "canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
        "release": (
            "saebench_gemma-2-2b_width-2pow16_date-0107_"
            "gemma-2-2b_standard_new_width-2pow16_date-0107_"
            "resid_post_layer_12_trainer_0"
        ),
    },
    "task": "city/Country",
    "seed": 42,
    "feature_selection": {
        "rule": "lowest_numeric_feature_ids_after_canonical_retained_mask_filtering",
        "count": 8,
    },
    "row_selection": {
        "rule": "first_retained_rows_per_feature_in_canonical_source_order",
        "count_per_feature": 8,
    },
    "readout": {"name": "final_resid", "location": "actual_lm_head_input"},
    "diagnostic": "returned_model_output_delta_post_softcap",
    "gram_dtype": "float64",
    "evaluation_dtype": "float64",
    "span_rank": 1,
    "residual_rank": 1,
}


def canonical_equivalence_manifest_fingerprint(
    selection: dict[str, Any], expected_source_fingerprint: Any
) -> dict[str, Any]:
    """Bind the frozen evidence selection to its canonical source fingerprints."""
    # Hash one canonical JSON representation of the complete binding.
    encoded = json.dumps(
        {
            "selection": selection,
            "expected_source_fingerprint": expected_source_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    """Parse the required live config and output paths."""
    # Accept only the executable's frozen real-model evaluation interface.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def git_provenance() -> dict[str, Any]:
    """Bind validation evidence to the exact relevant dirty source tree."""
    # Hash the complete tracked binary diff plus every relevant untracked file.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    untracked_output = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "fega",
            "scripts",
            "tests",
            "external",
            "README.md",
            "todo.md",
        ],
        check=True,
        capture_output=True,
    ).stdout
    untracked_files_sha256 = {
        raw_path.decode("utf-8"): _file_sha256(Path(raw_path.decode("utf-8")))
        for raw_path in sorted(filter(None, untracked_output.split(b"\0")))
    }
    binding = {
        "head": head,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_files_sha256": untracked_files_sha256,
    }
    return {
        **binding,
        "source_tree_sha256": canonical_json_digest(binding),
        "dirty": bool(tracked_diff or untracked_files_sha256),
    }


def environment_provenance() -> dict[str, Any]:
    """Collect runtime and scheduler facts needed to reproduce validation."""
    # Snapshot the executing environment without changing scientific artifacts.
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
    }


def _file_sha256(path: Path) -> str:
    """Hash the exact bytes of one required source artifact.

    The validation binding must change for whitespace or ordering changes in the
    activation manifest or feature-context JSON, not only semantic JSON changes.
    """
    # Stream the file so provenance hashing does not duplicate large JSON in memory.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_input_fingerprint(
    activation_manifest_path: Path,
    activations_dir: Path,
    feature_contexts_path: Path,
    pairs_path: Path,
) -> dict[str, Any]:
    """Bind the exact bytes of every canonical scientific input used by R12.

    The activation manifest is hashed directly, while its referenced tensor and
    metadata shards are hashed in manifest order through the production stream
    resolver. Feature contexts are always required; the optional full-pairs
    source is represented explicitly as ``None`` when it is absent.
    """
    # Preserve manifest order while hashing every resolved activation input byte.
    activation_chunks = [
        {
            "tensors_sha256": _file_sha256(tensor_path),
            "meta_sha256": _file_sha256(meta_path),
        }
        for tensor_path, meta_path in ChunkProcessor.stream(
            activation_manifest_path, activations_dir
        )
    ]
    return {
        "activation_manifest_sha256": _file_sha256(activation_manifest_path),
        "activation_chunks": activation_chunks,
        "feature_contexts_sha256": _file_sha256(feature_contexts_path),
        "pairs_full_sha256": _file_sha256(pairs_path) if pairs_path.exists() else None,
    }


def select_bounded_feature_rows(
    contexts_by_feature: dict[int, list[dict[str, Any]]],
    compute_candidate: Callable[[int, list[dict[str, Any]]], dict[str, Any]],
    *,
    feature_count: int = 8,
    row_count: int = 8,
) -> list[dict[str, Any]]:
    """Compute candidates in numeric order and stop at the bounded frozen sample.

    ``compute_candidate`` returns the production in-memory row bundle for one
    feature. Features with fewer than ``row_count`` retained rows are skipped;
    eligible bundles are truncated to their first canonical retained rows.
    """
    # Traverse numeric feature IDs once and stop immediately at the frozen count.
    selected: list[dict[str, Any]] = []
    for feature_id in sorted(contexts_by_feature):
        bundle = compute_candidate(feature_id, contexts_by_feature[feature_id])
        if int(bundle.get("feature_id", feature_id)) != feature_id:
            raise ValueError(f"computed feature ID mismatch for {feature_id}")
        rows = bundle.get("rows")
        if not isinstance(rows, list):
            raise TypeError(f"computed rows for feature {feature_id} must be a list")
        if len(rows) < row_count:
            continue
        bundle["rows"] = rows[:row_count]
        selected.append(bundle)
        if len(selected) == feature_count:
            break
    if len(selected) != feature_count:
        raise ValueError(
            f"expected {feature_count} eligible features, found {len(selected)}"
        )
    return selected


def _selected_context_records(
    records: list[Any], row_identities: list[dict[str, Any]]
) -> list[Any]:
    """Recover computed context records in canonical retained-row order.

    Full context and pair identity is matched so the separate live capture uses
    exactly the same eight examples as the production effect-row computation.
    """
    # Index complete normalized identities before replaying retained row order.
    indexed = {
        (
            int(record.index),
            str(record.attribute_label),
            str(record.pair_role),
            int(record.pair_index),
        ): record
        for record in records
    }
    selected = []
    for row in row_identities:
        key = (
            int(row["context_index"]),
            str(row["attribute_label"]),
            str(row["pair_role"]),
            int(row["pair_index"]),
        )
        if key not in indexed:
            raise ValueError(
                f"retained effect row is absent from computed contexts: {key}"
            )
        selected.append(indexed[key])
    return selected


def _returned_target_outputs(output: Any, target_positions: list[int]) -> torch.Tensor:
    """Extract returned target-position output vectors from one model forward."""
    # Accept the standard ModelOutput/tuple contract while requiring rank-3 outputs.
    values = output.logits if hasattr(output, "logits") else output[0]
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise RuntimeError("model forward did not return rank-3 output logits")
    return torch.stack(
        [values[row, position, :] for row, position in enumerate(target_positions)]
    ).detach().to(torch.float32)


@torch.no_grad()
def _live_base_readouts(
    model: Any,
    sae: Any,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_positions: list[int],
    z_batch: torch.Tensor,
) -> torch.Tensor:
    """Replay the stored SAE reconstruction for returned-output diagnostics.

    This validation-local replay decodes the exact stored codes prepared from the
    selected canonical prompts, then substitutes them into the same token batch
    used by the ablated replay. Its returned logits are diagnostic only and never
    replace the production in-memory hidden deltas used for scientific checks.
    """
    # Decode the stored base codes and observe one otherwise identical model replay.
    returned: list[torch.Tensor] = []
    model_param = next(model.parameters())
    decoded = sae.decode(z_batch).to(
        device=model_param.device, dtype=model_param.dtype
    )

    def replace_resid(module, inputs, outputs):
        """Substitute stored SAE reconstructions at selected target positions."""
        # Preserve the model's output container while replacing only target rows.
        del module, inputs
        if isinstance(outputs, ModelOutput):
            hidden_states = outputs[0]
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        modified = hidden_states.clone()
        for row, position in enumerate(target_positions):
            modified[row, position, :] = decoded[row].to(dtype=modified.dtype)
        if isinstance(outputs, ModelOutput):
            outputs_tuple = outputs.to_tuple()
            return type(outputs)(*(modified, *outputs_tuple[1:]))
        if isinstance(outputs, tuple):
            return (modified, *outputs[1:])
        return modified

    def capture_returned(module, inputs, output):
        """Capture diagnostic target outputs at the outer model boundary."""
        # Keep returned outputs separate from canonical scientific tensors.
        del module, inputs
        returned.append(_returned_target_outputs(output, target_positions))

    resid_handle = activation_collection.get_module(
        model, sae.cfg.hook_layer
    ).register_forward_hook(replace_resid)
    output_handle = model.register_forward_hook(capture_returned)
    try:
        model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )
    finally:
        resid_handle.remove()
        output_handle.remove()
    if len(returned) != 1:
        raise RuntimeError("expected exactly one base returned-output capture")
    return returned[0]


def _live_ablated_readouts(
    *,
    model: Any,
    sae: Any,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_positions: list[int],
    z_batch: torch.Tensor,
    feature_id: int,
) -> torch.Tensor:
    """Replay canonical ablation and return only diagnostic model outputs.

    The replay uses the same prepared token batch, positioning tensors, target
    positions, and stored SAE codes as the validation-local base replay. Scientific
    hidden deltas remain the production rows computed before this diagnostic pass.
    """
    # Observe the returned output without requesting an unused linear-head capture.
    returned: list[torch.Tensor] = []

    def capture_returned(module, inputs, output):
        """Capture diagnostic target outputs from the ablated model forward."""
        # Keep this hook observational so production ablation behavior is unchanged.
        del module, inputs
        returned.append(_returned_target_outputs(output, target_positions))

    handle = model.register_forward_hook(capture_returned)
    try:
        run_ablation_readouts_batch(
            model=model,
            sae=sae,
            tokens=tokens,
            attn=attention_mask,
            target_positions=target_positions,
            z_batch=z_batch,
            ablation_spec=AblationSpec(
                feature_ids=torch.full(
                    (len(target_positions),),
                    feature_id,
                    dtype=torch.long,
                    device=z_batch.device,
                )
            ),
            position_ids=position_ids,
            requested_readouts=[],
        )
    finally:
        handle.remove()
    if len(returned) != 1:
        raise RuntimeError("expected exactly one ablated returned-output capture")
    return returned[0]


def _validate_frozen_config(config: FEGAPipelineConfig, resources: ModelResources) -> None:
    """Require the config-owned task, seed, SAE, and readout frozen by the plan."""
    # Check only existing config/reference fields; no validation-only public field is added.
    entity, attribute = require_single_entity_attr(config)
    if f"{entity}/{attribute}" != PLANNED_EQUIVALENCE_SELECTION["task"]:
        raise ValueError("config task does not match frozen validation selection")
    if int(config.seed.global_) != PLANNED_EQUIVALENCE_SELECTION["seed"]:
        raise ValueError("config seed does not match frozen validation selection")
    if list(config.phases.data_prep.readouts) != ["final_resid"]:
        raise ValueError("config readout does not match frozen validation selection")
    if (
        config.phases.data_prep.gram_cache_dtype
        != PLANNED_EQUIVALENCE_SELECTION["gram_dtype"]
    ):
        raise ValueError("config Gram dtype does not match frozen validation selection")
    if (
        resources._ensure_eval_config().llm_dtype
        != PLANNED_EQUIVALENCE_SELECTION["model"]["dtype"]
    ):
        raise ValueError("config model dtype does not match frozen validation selection")
    if config.sae_repo_id != PLANNED_EQUIVALENCE_SELECTION["sae"]["repo_id"]:
        raise ValueError("config SAE repo does not match frozen validation selection")
    replay = resources._ensure_replay_context()
    if replay.sae_lens_release_id != PLANNED_EQUIVALENCE_SELECTION["sae"]["release"]:
        raise ValueError("config SAE release does not match frozen validation selection")


def _validate_loaded_model(model: Any) -> None:
    """Require the exact frozen checkpoint identity and resolved commit hash."""
    # Fail closed when Hugging Face does not expose the resolved pinned revision.
    model_config = getattr(model, "config", None)
    identity = getattr(model_config, "name_or_path", None) or getattr(
        model_config, "_name_or_path", None
    )
    if identity != PLANNED_EQUIVALENCE_SELECTION["model"]["repo_id"]:
        raise ValueError("loaded model repo does not match frozen validation selection")
    commit = getattr(model_config, "_commit_hash", None)
    if commit != PLANNED_EQUIVALENCE_SELECTION["model"]["revision"]:
        raise ValueError(
            f"loaded model commit mismatch: expected frozen revision, observed {commit!r}"
        )
    if next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("loaded model dtype does not match frozen validation selection")


def _config_evaluation(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build frozen real-model evidence with bounded production row computation.

    Existing data-prep activations, feature contexts, and Gram artifacts are read,
    but no compute-effect phase, shard, summary, or manifest is required or written.
    """
    # Load the frozen config and canonical data-prep inputs without running a phase.
    config = FEGAPipelineConfig.from_file(config_path)
    revision = PLANNED_EQUIVALENCE_SELECTION["model"]["revision"]
    resources = ModelResources(config, model_revision=revision)
    _validate_frozen_config(config, resources)
    effect_config = config.phases.compute_effect
    loader = EffectInputLoader(
        resources=resources,
        cache_max_chunks=effect_config.cache_max_chunks,
        cache_max_bytes=effect_config.cache_max_bytes,
    )
    entity, attribute = require_single_entity_attr(config)
    activations_dir = data_prep_activations_dir(config, entity, attribute)
    activation_manifest_path = activations_dir / "activations_manifest.json"
    activation_manifest, _ = loader.validate_manifest_readouts(
        activation_manifest_path, ["final_resid"]
    )
    feature_contexts_path = (
        data_prep_select_dir(config, entity, attribute) / "feature_contexts.json"
    )
    contexts = loader.load_contexts(feature_contexts_path)
    pairs_path = data_prep_pairs_path(config, entity, attribute)
    prompt_lookup = (
        loader.load_prompt_lookup(pairs_path) if pairs_path.exists() else None
    )
    gram, gram_meta = loader.load_gram(
        gram_cache_tensor_path(config), gram_cache_meta_path(config)
    )
    _validate_gram(gram, gram_meta)
    if gram_meta["gram_dtype"] != PLANNED_EQUIVALENCE_SELECTION["gram_dtype"]:
        raise ValueError("loaded Gram dtype does not match frozen validation selection")

    # Load and validate the pinned live readout before any feature metric work.
    model, tokenizer, sae = resources.get_model_and_sae()
    model.eval()
    sae.eval()
    _validate_loaded_model(model)
    _validate_gram_readout(model, gram_meta)
    unembedding = canonical_unembedding(model).to(dtype=torch.float32).to(
        dtype=torch.float64
    )
    source_fingerprint = _scientific_input_fingerprint(
        activation_manifest_path=activation_manifest_path,
        activations_dir=activations_dir,
        feature_contexts_path=feature_contexts_path,
        pairs_path=pairs_path,
    )
    observed_fingerprint = {
        "canonical_source": source_fingerprint,
        "unembedding_fingerprint": unembedding_fingerprint(
            canonical_unembedding(model)
        ),
        "gram_sha256": gram_fingerprint(gram),
    }
    expected_fingerprint = {
        "canonical_source": source_fingerprint,
        "unembedding_fingerprint": gram_meta["unembedding_fingerprint"],
        "gram_sha256": gram_meta["gram_sha256"],
    }
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "loaded Gram/readout fingerprints do not match model artifacts"
        )
    gram_device = gram.to(device=unembedding.device, dtype=torch.float64)
    evaluator_unembedding = unembedding.detach().to(
        device="cpu", dtype=torch.float64
    )
    evaluator_gram = gram.detach().to(device="cpu", dtype=torch.float64)

    def compute_candidate(
        feature_id: int, raw_contexts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute one feature's canonical production rows entirely in memory.

        The returned bundle retains only inputs needed for the later independent
        live base/ablated capture; no production effect artifact is materialized.
        """
        # Load this feature's activation rows and resolve their canonical prompts.
        target_indices = [
            int(record["index"])
            for record in raw_contexts
            if record.get("index") is not None
        ]
        example_bank, _ = loader.load_examples(
            manifest_path=activation_manifest_path,
            activations_dir=activations_dir,
            target_indices=target_indices,
            requested_readouts=["final_resid"],
        )
        records, _ = build_effect_context_records(
            feature_id=feature_id,
            raw_contexts=raw_contexts,
            example_bank=example_bank,
            prompt_lookup=prompt_lookup,
            tokenizer=tokenizer,
            default_attr=attribute,
        )
        rows_by_readout, _ = compute_feature_effect_rows(
            feature_id=feature_id,
            context_records=records,
            example_bank=example_bank,
            model=model,
            sae=sae,
            requested_readouts=["final_resid"],
            gram=gram_device,
            batch_size=int(effect_config.batch_size or 1),
            pad_token_id=int(activation_manifest["positioning"]["pad_token_id"]),
            positioning_schema_version=int(
                activation_manifest["positioning"]["schema_version"]
            ),
            normalization_eps=effect_config.normalization_eps,
            tau_zero=effect_config.tau_zero,
        )
        return {
            "feature_id": feature_id,
            "rows": rows_by_readout["final_resid"],
            "records": records,
            "example_bank": example_bank,
        }

    # Stop production row computation as soon as the frozen bounded sample exists.
    selected = select_bounded_feature_rows(contexts, compute_candidate)
    per_feature: dict[str, Any] = {}
    observed_ids: list[int] = []
    positioning = activation_manifest["positioning"]
    for feature_bundle in selected:
        feature_id = int(feature_bundle["feature_id"])
        rows = feature_bundle["rows"]
        canonical_delta = torch.stack(
            [row["delta"].detach().cpu().to(torch.float32) for row in rows]
        ).to(dtype=torch.float64)
        row_identities = [
            {
                "context_index": int(row["context_index"]),
                "attribute_label": str(row["attribute_label"]),
                "pair_role": str(row["pair_role"]),
                "pair_index": int(row["pair_index"]),
            }
            for row in rows
        ]
        records = _selected_context_records(feature_bundle["records"], row_identities)
        example_bank = feature_bundle["example_bank"]
        sae_device, sae_dtype = _sae_device_dtype(sae)
        (
            tokens,
            attention_mask,
            position_ids,
            target_positions,
            z_batch,
            _,
        ) = _prep_readout_batch(
            records,
            example_bank,
            ["final_resid"],
            next(model.parameters()).device,
            sae_device,
            sae_dtype,
            pad_token_id=int(positioning["pad_token_id"]),
            positioning_schema_version=int(positioning["schema_version"]),
        )
        base_returned = _live_base_readouts(
            model,
            sae,
            tokens,
            attention_mask,
            position_ids,
            target_positions,
            z_batch,
        )
        ablated_returned = _live_ablated_readouts(
            model=model,
            sae=sae,
            tokens=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            target_positions=target_positions,
            z_batch=z_batch,
            feature_id=feature_id,
        )
        explicit_delta = (
            canonical_delta.to(unembedding.device) @ unembedding.T
        ).to(
            device="cpu", dtype=torch.float64
        )
        returned_delta = (
            ablated_returned.to(unembedding.device)
            - base_returned.to(unembedding.device)
        ).to(
            device="cpu", dtype=torch.float64
        )
        grouped = evaluate_grouped_gram_logit_equivalence(
            feature_groups={
                feature_id: {
                    "hidden_deltas": canonical_delta,
                    "explicit_logit_deltas": explicit_delta,
                    "returned_model_output_deltas": returned_delta,
                    "row_identities": row_identities,
                }
            },
            unembedding=evaluator_unembedding,
            gram=evaluator_gram,
            expected_source_fingerprint=expected_fingerprint,
            observed_source_fingerprint=observed_fingerprint,
            tolerances=FLOAT32_EQUIVALENCE_TOLERANCES,
        )
        per_feature[str(feature_id)] = grouped["per_feature"][str(feature_id)]
        observed_ids.append(feature_id)
        del explicit_delta, returned_delta, base_returned, ablated_returned
        del canonical_delta, grouped

    evaluation = {
        "status": "pass"
        if all(result["status"] == "pass" for result in per_feature.values())
        else "fail",
        "observed_feature_ids": observed_ids,
        "per_feature": per_feature,
    }
    evidence = {
        "source": "config",
        "config_path": str(config_path),
        "model_revision": revision,
        "activation_manifest_path": str(activation_manifest_path),
        "feature_contexts_path": str(feature_contexts_path),
        "gram_path": str(gram_cache_tensor_path(config)),
        "gram_metadata_path": str(gram_cache_meta_path(config)),
        "canonical_manifest_fingerprint": canonical_equivalence_manifest_fingerprint(
            PLANNED_EQUIVALENCE_SELECTION, expected_fingerprint
        ),
    }
    return evaluation, evidence


def main() -> int:
    """Write the durable JSON record and return its protocol outcome code."""
    # Evaluate the frozen live config before writing its complete evidence record.
    args = parse_args()
    evaluation, evidence = _config_evaluation(args.config)
    record = {
        "schema_version": 2,
        "protocol": "fega_gram_logit_equivalence",
        "selection": PLANNED_EQUIVALENCE_SELECTION,
        "command": shlex.join(sys.argv),
        "git": git_provenance(),
        "environment": environment_provenance(),
        "tolerances": {
            "profile": "float64_evaluation_with_frozen_float32_tolerances",
            **FLOAT32_EQUIVALENCE_TOLERANCES,
        },
        "evidence": evidence,
        **evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
