from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    get_all_hf_repo_autoencoders,
    load_dictionary_learning_sae,
)
from sae_bench.evals.induction_features.analysis import (
    build_feature_metrics_frame,
    choose_count_dtype,
)
from sae_bench.sae_bench_utils.activation_collection import LLM_NAME_TO_DTYPE
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from tqdm import tqdm
from transformer_lens import HookedTransformer

DEFAULT_EXCLUDE_KEYWORDS = ["checkpoints"]
HEAD_DIM_HOOK_SUBSTRINGS = ("hook_q", "hook_k", "hook_v", "hook_z")


def _relative_to_output(path: Path, output_dir: Path) -> str:
    return os.path.relpath(
        path.expanduser().resolve(), output_dir.expanduser().resolve()
    )


@dataclass(slots=True)
class SaeReference:
    source_kind: str
    sae_release: str
    sae_id: str
    custom_location: str | None
    sae_uid: str


@dataclass(slots=True)
class LoadedSae:
    reference: SaeReference
    sae: Any
    hook_name: str
    layer: int
    d_sae: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which SAE features are commonly active across prompt families "
            "at the final pre-answer token. Supports the shared SAE Geometry ICL "
            "schema used by LSC, WC, TT, and PrOntoQA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to a flat or grouped induction dataset JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/induction_feature_outputs"),
        help="Directory in which CSV summaries, JSON summaries, and plots will be written.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="TransformerLens model name used both for prompts and for the selected SAEs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which to run the base model and SAE encoders.",
    )
    parser.add_argument(
        "--download-saes-dir",
        type=Path,
        default=Path("data/downloaded_saes"),
        help="Directory for downloaded custom SAE checkpoints.",
    )
    parser.add_argument(
        "--llm-dtype",
        type=str,
        default=None,
        help="Base model / SAE dtype. If omitted, use SAEBench defaults for known models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Prompt batch size for feature extraction.",
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="A feature counts as active when its post-encode activation exceeds this threshold.",
    )
    parser.add_argument(
        "--min-example-fraction",
        type=float,
        default=0.9,
        help=(
            "A candidate feature must be active in at least this fraction of all "
            "analyzed examples."
        ),
    )
    parser.add_argument(
        "--min-query-fraction-per-context",
        type=float,
        default=0.9,
        help=(
            "Within a context, a feature is considered consistently active if it is active in at least "
            "this fraction of that context's analyzed queries."
        ),
    )
    parser.add_argument(
        "--min-context-fraction",
        type=float,
        default=0.9,
        help=(
            "A feature is marked as a candidate induction feature if it is consistently active in at least "
            "this fraction of analyzed contexts."
        ),
    )
    parser.add_argument(
        "--answer-prefix",
        type=str,
        default=" ",
        help="Prefix prepended before answers when computing the target first token ID.",
    )
    parser.add_argument(
        "--single-token-only",
        action="store_true",
        help=(
            "Restrict analysis to examples whose answer text tokenizes to a single token "
            "after applying --answer-prefix."
        ),
    )
    parser.add_argument(
        "--require-model-correct",
        action="store_true",
        help=(
            "Only analyze examples for which the base model predicts the correct first answer token. "
            "This is slower because it requires a full forward pass to logits."
        ),
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Optional cap on the number of contexts to analyze, sampled uniformly at random.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of examples to analyze after context sampling.",
    )
    parser.add_argument(
        "--max-saes",
        type=int,
        default=None,
        help="Optional cap on the number of SAEs to analyze after selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for dataset subsampling.",
    )
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=25,
        help="Number of top features to show in the ranking plot.",
    )
    parser.add_argument(
        "--context-chunk-size",
        type=int,
        default=4096,
        help="Feature chunk size used when summarizing context prevalence.",
    )
    parser.add_argument(
        "--expected-examples",
        type=int,
        default=None,
        help="Fail unless this many examples remain after token filtering.",
    )
    parser.add_argument(
        "--max-family-size-difference",
        type=int,
        default=None,
        help=(
            "Fail if the largest and smallest prompt families differ by more than "
            "this many retained examples."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--repo-id",
        type=str,
        help=(
            "Hugging Face repo containing dictionary-learning SAEs in the layout expected by "
            "sae_bench.custom_saes.run_all_evals_dictionary_learning_saes."
        ),
    )
    source_group.add_argument(
        "--sae-regex-pattern",
        type=str,
        help="Regex used to select SAE Lens releases.",
    )

    parser.add_argument(
        "--sae-block-pattern",
        type=str,
        default=None,
        help="Regex used to select SAE Lens SAE IDs. Required with --sae-regex-pattern.",
    )
    parser.add_argument(
        "--sae-location",
        action="append",
        default=None,
        help="Exact custom SAE location inside --repo-id. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-keyword",
        action="append",
        default=None,
        help="Keyword filter for custom SAE locations inside --repo-id. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=None,
        help="Keyword exclusion filter for custom SAE locations inside --repo-id. May be passed multiple times.",
    )

    return parser.parse_args()


def normalize_examples(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = payload.get("metadata", {})

    if "examples" in payload:
        raw_examples = payload["examples"]
    elif "contexts" in payload:
        raw_examples = []
        for context in payload["contexts"]:
            context_id = context["context_id"]
            query_style = context["query_style"]
            for query in context["queries"]:
                normalized = dict(query)
                normalized.setdefault("example_id", query.get("query_id"))
                normalized["context_id"] = context_id
                normalized.setdefault("query_style", query_style)
                raw_examples.append(normalized)
    else:
        raise ValueError(
            "Dataset JSON must contain either an 'examples' key or a 'contexts' key"
        )

    dataframe = pd.DataFrame(raw_examples)
    required_columns = {"context_id", "prompt", "answer"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if "example_id" not in dataframe.columns:
        dataframe["example_id"] = [
            f"example_{idx:07d}" for idx in range(len(dataframe))
        ]
    if "support_example_index" not in dataframe.columns:
        dataframe["support_example_index"] = 0

    dataframe["support_example_index"] = (
        dataframe["support_example_index"].fillna(0).astype(int)
    )
    dataframe = dataframe.reset_index(drop=True)
    return dataframe, metadata


def subsample_examples(
    examples_df: pd.DataFrame,
    *,
    max_contexts: int | None,
    max_examples: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled = examples_df

    if max_contexts is not None:
        unique_contexts = sorted(sampled["context_id"].unique().tolist())
        if max_contexts <= 0:
            raise ValueError("--max-contexts must be positive when provided")
        if max_contexts < len(unique_contexts):
            chosen_contexts = set(rng.sample(unique_contexts, max_contexts))
            sampled = sampled[sampled["context_id"].isin(chosen_contexts)].copy()

    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("--max-examples must be positive when provided")
        if max_examples < len(sampled):
            indices = rng.sample(range(len(sampled)), max_examples)
            sampled = sampled.iloc[sorted(indices)].copy()

    return sampled.reset_index(drop=True)


def add_target_token_columns(
    examples_df: pd.DataFrame,
    tokenizer: Any,
    *,
    answer_prefix: str,
) -> pd.DataFrame:
    token_cache: dict[str, list[int]] = {}
    target_first_token_ids: list[int] = []
    target_token_lengths: list[int] = []

    for answer in examples_df["answer"].tolist():
        if answer not in token_cache:
            token_ids = tokenizer.encode(
                f"{answer_prefix}{answer}",
                add_special_tokens=False,
            )
            if len(token_ids) == 0:
                raise ValueError(
                    f"Tokenizer produced zero tokens for answer {answer!r}"
                )
            token_cache[answer] = token_ids

        cached_token_ids = token_cache[answer]
        target_first_token_ids.append(int(cached_token_ids[0]))
        target_token_lengths.append(len(cached_token_ids))

    augmented = examples_df.copy()
    augmented["target_first_token_id"] = target_first_token_ids
    augmented["target_token_length"] = target_token_lengths
    return augmented


def make_sae_uid(left: str, right: str) -> str:
    return f"{left.replace('/', '_')}__{right.replace('/', '_')}"


def resolve_sae_references(args: argparse.Namespace) -> list[SaeReference]:
    references: list[SaeReference]

    if args.repo_id is not None:
        if args.sae_location:
            locations = args.sae_location
        else:
            locations = get_all_hf_repo_autoencoders(
                args.repo_id, download_location=str(args.download_saes_dir)
            )
            locations = general_utils.filter_keywords(
                locations,
                exclude_keywords=args.exclude_keyword or DEFAULT_EXCLUDE_KEYWORDS,
                include_keywords=args.include_keyword or [],
            )

        references = [
            SaeReference(
                source_kind="custom_repo",
                sae_release=args.repo_id,
                sae_id=location,
                custom_location=location,
                sae_uid=make_sae_uid(args.repo_id, location),
            )
            for location in locations
        ]
    else:
        if args.sae_block_pattern is None:
            raise ValueError("--sae-block-pattern is required with --sae-regex-pattern")

        selected_saes = get_saes_from_regex(
            args.sae_regex_pattern, args.sae_block_pattern
        )
        references = [
            SaeReference(
                source_kind="sae_lens",
                sae_release=release,
                sae_id=sae_id,
                custom_location=None,
                sae_uid=make_sae_uid(release, sae_id),
            )
            for release, sae_id in selected_saes
        ]

    if args.max_saes is not None:
        if args.max_saes <= 0:
            raise ValueError("--max-saes must be positive when provided")
        references = references[: args.max_saes]

    if not references:
        raise ValueError("No SAEs matched the requested selection")

    return references


def load_sae_reference(
    reference: SaeReference,
    *,
    model_name: str,
    device: str,
    llm_dtype: torch.dtype,
    download_saes_dir: Path,
) -> LoadedSae:
    if reference.source_kind == "custom_repo":
        sae = load_dictionary_learning_sae(
            repo_id=reference.sae_release,
            location=reference.sae_id,
            model_name=model_name,
            device=device,
            dtype=llm_dtype,
            download_location=str(download_saes_dir),
        )
    else:
        _, sae, _ = general_utils.load_and_format_sae(
            reference.sae_release,
            reference.sae_id,
            device,
        )
        sae = sae.to(device=device, dtype=llm_dtype)

    sae.eval()
    hook_name = sae.cfg.hook_name
    layer = int(sae.cfg.hook_layer)
    d_sae = int(sae.W_dec.shape[0])
    return LoadedSae(
        reference=reference,
        sae=sae,
        hook_name=hook_name,
        layer=layer,
        d_sae=d_sae,
    )


def maybe_disable_hook_z_reshaping(sae: Any) -> bool | None:
    hook_name = getattr(sae.cfg, "hook_name", "")
    if "hook_z" not in hook_name:
        return None

    previous_mode = getattr(sae, "hook_z_reshaping_mode", None)
    if previous_mode is not None and hasattr(
        sae, "turn_off_forward_pass_hook_z_reshaping"
    ):
        sae.turn_off_forward_pass_hook_z_reshaping()
    return previous_mode


def restore_hook_z_reshaping(sae: Any, previous_mode: bool | None) -> None:
    if previous_mode is None:
        return
    if previous_mode and hasattr(sae, "turn_on_forward_pass_hook_z_reshaping"):
        sae.turn_on_forward_pass_hook_z_reshaping()
    elif not previous_mode and hasattr(sae, "turn_off_forward_pass_hook_z_reshaping"):
        sae.turn_off_forward_pass_hook_z_reshaping()


def standardize_cached_activations(
    sae: Any, hook_activations: torch.Tensor
) -> torch.Tensor:
    hook_name = sae.cfg.hook_name
    hook_head_index = getattr(sae.cfg, "hook_head_index", None)

    if hook_head_index is not None:
        return hook_activations[:, :, hook_head_index]
    if hook_activations.ndim == 4 and any(
        substring in hook_name for substring in HEAD_DIM_HOOK_SUBSTRINGS
    ):
        return hook_activations.flatten(-2, -1)
    if hook_activations.ndim != 3:
        raise ValueError(
            f"Unsupported cached activation shape {tuple(hook_activations.shape)} for hook {hook_name}"
        )
    return hook_activations


def append_frame(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def maybe_write_empty_csv(header_frame: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        header_frame.to_csv(path, index=False)


def short_feature_label(row: pd.Series) -> str:
    sae_id = str(row["sae_id"])
    if len(sae_id) > 28:
        sae_id = sae_id[:25] + "..."
    return f"L{int(row['layer'])} F{int(row['feature_id'])} | {sae_id}"


@torch.no_grad()
def analyze_single_sae(
    *,
    model: HookedTransformer,
    tokenizer: Any,
    examples_df: pd.DataFrame,
    loaded_sae: LoadedSae,
    args: argparse.Namespace,
    per_sae_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context_ids = sorted(examples_df["context_id"].unique().tolist())
    context_to_index = {
        context_id: index for index, context_id in enumerate(context_ids)
    }
    context_indices = (
        examples_df["context_id"].map(context_to_index).to_numpy(dtype=np.int32)
    )
    support_indices = examples_df["support_example_index"].to_numpy(dtype=np.int32)
    target_first_token_ids = examples_df["target_first_token_id"].to_numpy(
        dtype=np.int64
    )
    prompts = examples_df["prompt"].tolist()

    num_contexts = len(context_ids)
    num_slots = int(support_indices.max()) + 1 if len(support_indices) > 0 else 1
    max_context_size = int(examples_df.groupby("context_id").size().max())
    count_dtype = choose_count_dtype(max_context_size)

    context_feature_query_counts = np.zeros(
        (num_contexts, loaded_sae.d_sae),
        dtype=count_dtype,
    )
    example_active_counts = np.zeros(loaded_sae.d_sae, dtype=np.int64)
    activation_sum = np.zeros(loaded_sae.d_sae, dtype=np.float64)
    active_activation_sum = np.zeros(loaded_sae.d_sae, dtype=np.float64)
    max_activation = np.zeros(loaded_sae.d_sae, dtype=np.float32)
    slot_active_counts = np.zeros((num_slots, loaded_sae.d_sae), dtype=np.int64)

    if args.require_model_correct:
        context_totals = np.zeros(num_contexts, dtype=np.int32)
        slot_totals = np.zeros(num_slots, dtype=np.int32)
    else:
        context_totals = np.bincount(context_indices, minlength=num_contexts).astype(
            np.int32
        )
        slot_totals = np.bincount(support_indices, minlength=num_slots).astype(np.int32)

    analyzed_example_count = 0
    logits_required = bool(args.require_model_correct)
    stop_at_layer = None if logits_required else loaded_sae.layer + 1
    previous_hook_z_mode = maybe_disable_hook_z_reshaping(loaded_sae.sae)

    try:
        for batch_start in tqdm(
            range(0, len(prompts), args.batch_size),
            desc=f"Analyzing {loaded_sae.reference.sae_uid}",
        ):
            batch_end = min(batch_start + args.batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            batch_encoding = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = batch_encoding["input_ids"]
            attention_mask = batch_encoding["attention_mask"]
            if int(input_ids.shape[1]) > int(model.cfg.n_ctx):
                raise ValueError(
                    f"Prompt length {input_ids.shape[1]} exceeds model context window {model.cfg.n_ctx} "
                    f"for SAE {loaded_sae.reference.sae_uid}"
                )

            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            final_positions = attention_mask.sum(dim=1) - 1

            model_output, cache = model.run_with_cache(
                input_ids,
                prepend_bos=False,
                names_filter=[loaded_sae.hook_name],
                stop_at_layer=stop_at_layer,
            )

            if logits_required:
                batch_target_ids = torch.as_tensor(
                    target_first_token_ids[batch_start:batch_end],
                    device=model_output.device,
                )
                final_logits = model_output[
                    torch.arange(model_output.shape[0], device=model_output.device),
                    final_positions,
                ]
                analyzed_mask = final_logits.argmax(dim=-1).eq(batch_target_ids)
            else:
                analyzed_mask = torch.ones(
                    input_ids.shape[0],
                    device=input_ids.device,
                    dtype=torch.bool,
                )

            analyzed_rows = analyzed_mask.nonzero(as_tuple=False).squeeze(-1)
            if analyzed_rows.numel() == 0:
                del model_output, cache, input_ids, attention_mask, final_positions
                continue

            standardized_acts = standardize_cached_activations(
                loaded_sae.sae,
                cache[loaded_sae.hook_name],
            )
            final_token_acts = standardized_acts[
                torch.arange(
                    standardized_acts.shape[0], device=standardized_acts.device
                ),
                final_positions,
            ]
            final_token_acts = final_token_acts[analyzed_mask].to(
                device=loaded_sae.sae.device,
                dtype=loaded_sae.sae.dtype,
            )

            sae_activations = loaded_sae.sae.encode(final_token_acts)
            if sae_activations.ndim != 2:
                raise ValueError(
                    f"Expected SAE activations to have shape [batch, d_sae], got {tuple(sae_activations.shape)}"
                )

            active_mask = sae_activations > args.activation_threshold
            masked_activations = torch.where(
                active_mask,
                sae_activations,
                torch.zeros_like(sae_activations),
            )

            example_active_counts += active_mask.sum(dim=0).cpu().numpy()
            activation_sum += sae_activations.sum(dim=0).float().cpu().numpy()
            active_activation_sum += masked_activations.sum(dim=0).float().cpu().numpy()
            max_activation = np.maximum(
                max_activation,
                sae_activations.max(dim=0).values.float().cpu().numpy(),
            )

            analyzed_rows_np = analyzed_rows.cpu().numpy()
            batch_context_indices = context_indices[batch_start:batch_end][
                analyzed_rows_np
            ]
            batch_support_indices = support_indices[batch_start:batch_end][
                analyzed_rows_np
            ]

            if args.require_model_correct:
                np.add.at(context_totals, batch_context_indices, 1)
                np.add.at(slot_totals, batch_support_indices, 1)

            unique_slots = np.unique(batch_support_indices)
            for slot in unique_slots:
                slot_mask_np = batch_support_indices == slot
                slot_mask = torch.as_tensor(slot_mask_np, device=active_mask.device)
                slot_active_counts[int(slot)] += (
                    active_mask[slot_mask].sum(dim=0).cpu().numpy()
                )

            nonzero_rows_and_cols = active_mask.nonzero(as_tuple=False)
            if nonzero_rows_and_cols.numel() > 0:
                active_rows = nonzero_rows_and_cols[:, 0].cpu().numpy()
                active_cols = nonzero_rows_and_cols[:, 1].cpu().numpy()
                np.add.at(
                    context_feature_query_counts,
                    (batch_context_indices[active_rows], active_cols),
                    1,
                )

            analyzed_example_count += int(analyzed_rows.numel())

            del (
                model_output,
                cache,
                input_ids,
                attention_mask,
                final_positions,
                standardized_acts,
                final_token_acts,
                sae_activations,
                active_mask,
                masked_activations,
            )
    finally:
        restore_hook_z_reshaping(loaded_sae.sae, previous_hook_z_mode)

    feature_metrics_df, sae_summary = build_feature_metrics_frame(
        sae_uid=loaded_sae.reference.sae_uid,
        sae_release=loaded_sae.reference.sae_release,
        sae_id=loaded_sae.reference.sae_id,
        layer=loaded_sae.layer,
        hook_name=loaded_sae.hook_name,
        example_active_counts=example_active_counts,
        activation_sum=activation_sum,
        active_activation_sum=active_activation_sum,
        max_activation=max_activation,
        slot_active_counts=slot_active_counts,
        slot_totals=slot_totals,
        context_feature_query_counts=context_feature_query_counts,
        context_totals=context_totals,
        analyzed_example_count=analyzed_example_count,
        min_example_fraction=args.min_example_fraction,
        min_query_fraction=args.min_query_fraction_per_context,
        min_context_fraction=args.min_context_fraction,
        context_chunk_size=args.context_chunk_size,
    )

    feature_metrics_df = feature_metrics_df.sort_values(
        ["consistent_context_prevalence", "mean_activation_when_active", "feature_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    per_sae_feature_path = (
        per_sae_dir / f"{loaded_sae.reference.sae_uid}_feature_metrics.csv"
    )
    feature_metrics_df.to_csv(per_sae_feature_path, index=False)

    per_sae_candidate_path = (
        per_sae_dir / f"{loaded_sae.reference.sae_uid}_candidate_features.csv"
    )
    candidate_df = feature_metrics_df[feature_metrics_df["is_candidate_feature"]].copy()
    candidate_df.to_csv(per_sae_candidate_path, index=False)

    sae_summary["per_sae_feature_metrics_path"] = _relative_to_output(
        per_sae_feature_path, per_sae_dir.parent
    )
    sae_summary["per_sae_candidate_features_path"] = _relative_to_output(
        per_sae_candidate_path, per_sae_dir.parent
    )
    return feature_metrics_df, sae_summary


def plot_layer_distribution(layer_summary_df: pd.DataFrame, plots_dir: Path) -> None:
    if layer_summary_df.empty:
        return

    ordered = layer_summary_df.sort_values("layer")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].bar(ordered["layer"], ordered["candidate_feature_count"], color="#2369bd")
    axes[0].set_ylabel("Candidate features")
    axes[0].set_title("Candidate induction-feature count by layer")

    axes[1].bar(
        ordered["layer"],
        100.0 * ordered["candidate_feature_fraction"],
        color="#4aa377",
    )
    axes[1].set_ylabel("% of layer features")
    axes[1].set_xlabel("Layer")
    axes[1].set_title("Candidate induction-feature fraction by layer")

    fig.tight_layout()
    fig.savefig(
        plots_dir / "candidate_features_by_layer.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_feature_prevalence_overview(
    feature_metrics_df: pd.DataFrame, plots_dir: Path
) -> None:
    if feature_metrics_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(
        feature_metrics_df["consistent_context_prevalence"],
        bins=40,
        color="#2369bd",
        alpha=0.9,
    )
    axes[0].set_xlabel("Consistent context prevalence")
    axes[0].set_ylabel("Feature count")
    axes[0].set_title("How broadly features recur across contexts")

    hexbin = axes[1].hexbin(
        feature_metrics_df["consistent_context_prevalence"],
        feature_metrics_df["mean_activation_when_active"].clip(lower=0.0),
        gridsize=45,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axes[1].set_xlabel("Consistent context prevalence")
    axes[1].set_ylabel("Mean activation when active")
    axes[1].set_title("Feature strength vs context commonality")
    fig.colorbar(hexbin, ax=axes[1], label="log10(count)")

    fig.tight_layout()
    fig.savefig(
        plots_dir / "feature_prevalence_overview.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_top_features(
    feature_metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    plots_dir: Path,
    *,
    top_n: int,
) -> None:
    if feature_metrics_df.empty or top_n <= 0:
        return

    top_source_df = candidate_df if not candidate_df.empty else feature_metrics_df
    top_df = top_source_df.sort_values(
        ["consistent_context_prevalence", "mean_activation_when_active", "feature_id"],
        ascending=[False, False, True],
    ).head(top_n)

    labels = [short_feature_label(row) for _, row in top_df.iloc[::-1].iterrows()]
    values = top_df["consistent_context_prevalence"].iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(top_df))))
    ax.barh(range(len(top_df)), values, color="#d9822b")
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Consistent context prevalence")
    ax.set_title(
        "Top candidate induction features"
        if not candidate_df.empty
        else "Top features by context prevalence"
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "top_induction_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_layer_summary(sae_summary_df: pd.DataFrame) -> pd.DataFrame:
    if sae_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "total_features",
                "candidate_feature_count",
                "strict_common_feature_count",
                "candidate_feature_fraction",
                "strict_common_feature_fraction",
            ]
        )

    grouped = (
        sae_summary_df.groupby("layer", as_index=False)[
            [
                "total_features",
                "candidate_feature_count",
                "strict_common_feature_count",
            ]
        ]
        .sum()
        .sort_values("layer")
        .reset_index(drop=True)
    )
    grouped["candidate_feature_fraction"] = (
        grouped["candidate_feature_count"] / grouped["total_features"]
    )
    grouped["strict_common_feature_fraction"] = (
        grouped["strict_common_feature_count"] / grouped["total_features"]
    )
    return grouped


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sae_regex_pattern is not None and args.sae_block_pattern is None:
        raise ValueError("--sae-block-pattern is required with --sae-regex-pattern")

    llm_dtype_name = args.llm_dtype or LLM_NAME_TO_DTYPE.get(args.model_name, "float32")
    llm_dtype = general_utils.str_to_dtype(llm_dtype_name)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sae_dir = output_dir / "per_sae"
    plots_dir = output_dir / "plots"
    per_sae_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with args.dataset_path.open("r", encoding="utf-8") as dataset_file:
        dataset_payload = json.load(dataset_file)
    all_examples_df, dataset_metadata = normalize_examples(dataset_payload)
    sampled_examples_df = subsample_examples(
        all_examples_df,
        max_contexts=args.max_contexts,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=llm_dtype,
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_examples_df = add_target_token_columns(
        sampled_examples_df,
        tokenizer,
        answer_prefix=args.answer_prefix,
    )

    single_token_examples_df = tokenized_examples_df
    if args.single_token_only:
        single_token_examples_df = tokenized_examples_df[
            tokenized_examples_df["target_token_length"] == 1
        ].copy()

    if single_token_examples_df.empty:
        raise ValueError("No examples remain after token-length filtering")
    if (
        args.expected_examples is not None
        and len(single_token_examples_df) != args.expected_examples
    ):
        raise ValueError(
            f"Expected {args.expected_examples} retained examples, found "
            f"{len(single_token_examples_df)}"
        )
    family_sizes = single_token_examples_df.groupby("context_id").size()
    if args.max_family_size_difference is not None:
        family_difference = int(family_sizes.max() - family_sizes.min())
        if family_difference > args.max_family_size_difference:
            raise ValueError(
                "Prompt-family imbalance exceeds the requested limit: "
                f"difference={family_difference}, "
                f"limit={args.max_family_size_difference}"
            )

    sae_references = resolve_sae_references(args)

    combined_feature_metrics_path = output_dir / "feature_metrics.csv"
    combined_candidate_features_path = output_dir / "candidate_features.csv"
    if combined_feature_metrics_path.exists():
        combined_feature_metrics_path.unlink()
    if combined_candidate_features_path.exists():
        combined_candidate_features_path.unlink()

    sae_summary_rows: list[dict[str, Any]] = []
    feature_sets: dict[str, Any] = {}

    for reference in sae_references:
        loaded_sae = load_sae_reference(
            reference,
            model_name=args.model_name,
            device=args.device,
            llm_dtype=llm_dtype,
            download_saes_dir=args.download_saes_dir,
        )

        feature_metrics_df, sae_summary = analyze_single_sae(
            model=model,
            tokenizer=tokenizer,
            examples_df=single_token_examples_df,
            loaded_sae=loaded_sae,
            args=args,
            per_sae_dir=per_sae_dir,
        )
        if (
            args.require_model_correct
            and args.expected_examples is not None
            and int(sae_summary["analyzed_example_count"]) != args.expected_examples
        ):
            raise ValueError(
                "Independent model-correctness recheck changed the retained dataset "
                f"for SAE {reference.sae_uid}: expected {args.expected_examples}, "
                f"analyzed {sae_summary['analyzed_example_count']}"
            )

        candidate_df = feature_metrics_df[
            feature_metrics_df["is_candidate_feature"]
        ].copy()
        append_frame(feature_metrics_df, combined_feature_metrics_path)
        if candidate_df.empty:
            maybe_write_empty_csv(candidate_df, combined_candidate_features_path)
        else:
            append_frame(candidate_df, combined_candidate_features_path)

        feature_sets[reference.sae_uid] = {
            "sae_release": reference.sae_release,
            "sae_id": reference.sae_id,
            "layer": sae_summary["layer"],
            "hook_name": sae_summary["hook_name"],
            "total_features": sae_summary["total_features"],
            "candidate_feature_ids": sae_summary["candidate_feature_ids"],
            "threshold_feature_ids": sae_summary["candidate_feature_ids"],
            "strict_common_feature_ids": sae_summary["strict_common_feature_ids"],
            "candidate_feature_count": sae_summary["candidate_feature_count"],
            "threshold_feature_count": sae_summary["candidate_feature_count"],
            "strict_common_feature_count": sae_summary["strict_common_feature_count"],
            "selection_thresholds": {
                "min_example_fraction": args.min_example_fraction,
                "min_query_fraction_per_context": (args.min_query_fraction_per_context),
                "min_context_fraction": args.min_context_fraction,
                "activation_threshold": args.activation_threshold,
            },
        }
        per_sae_feature_set_path = per_sae_dir / f"{reference.sae_uid}_feature_set.json"
        per_sae_feature_set_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_name": args.model_name,
                    "sae_uid": reference.sae_uid,
                    "dataset_path": _relative_to_output(args.dataset_path, output_dir),
                    "task_name": dataset_metadata.get("task_name"),
                    "feature_set": feature_sets[reference.sae_uid],
                    "strict_feature_rule": "active in every analyzed example",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        sae_summary_rows.append(sae_summary)

        del loaded_sae.sae, loaded_sae, feature_metrics_df, candidate_df
        gc.collect()
        if torch.cuda.is_available() and "cuda" in str(args.device):
            torch.cuda.empty_cache()

    sae_summary_df = pd.DataFrame(sae_summary_rows).sort_values(
        ["layer", "candidate_feature_count", "sae_uid"],
        ascending=[True, False, True],
    )
    layer_summary_df = build_layer_summary(sae_summary_df)

    sae_summary_df.to_csv(output_dir / "sae_summary.csv", index=False)
    layer_summary_df.to_csv(output_dir / "layer_summary.csv", index=False)

    combined_feature_metrics_df = pd.read_csv(combined_feature_metrics_path)
    if combined_candidate_features_path.exists():
        candidate_features_df = pd.read_csv(combined_candidate_features_path)
    else:
        candidate_features_df = combined_feature_metrics_df.head(0).copy()

    plot_layer_distribution(layer_summary_df, plots_dir)
    plot_feature_prevalence_overview(combined_feature_metrics_df, plots_dir)
    plot_top_features(
        combined_feature_metrics_df,
        candidate_features_df,
        plots_dir,
        top_n=args.plot_top_n,
    )

    overall_total_features = int(sae_summary_df["total_features"].sum())
    overall_candidate_features = int(sae_summary_df["candidate_feature_count"].sum())
    overall_strict_common_features = int(
        sae_summary_df["strict_common_feature_count"].sum()
    )

    summary_payload = {
        "dataset_path": _relative_to_output(args.dataset_path, output_dir),
        "output_dir": ".",
        "model_name": args.model_name,
        "device": args.device,
        "llm_dtype": llm_dtype_name,
        "sae_selection_mode": "custom_repo" if args.repo_id is not None else "sae_lens",
        "num_selected_saes": len(sae_references),
        "dataset_metadata": dataset_metadata,
        "filtering": {
            "max_contexts": args.max_contexts,
            "max_examples": args.max_examples,
            "single_token_only": args.single_token_only,
            "answer_prefix": args.answer_prefix,
            "require_model_correct": args.require_model_correct,
        },
        "activation_definition": {
            "token_position": "final_prompt_token_before_first_answer_token",
            "active_if_post_encode_activation_exceeds": args.activation_threshold,
            "common_feature_rule": {
                "min_example_fraction": args.min_example_fraction,
                "min_query_fraction_per_context": args.min_query_fraction_per_context,
                "min_context_fraction": args.min_context_fraction,
            },
            "strict_feature_rule": "active in every analyzed example",
        },
        "dataset_sizes": {
            "loaded_examples": int(len(all_examples_df)),
            "sampled_examples_before_token_filter": int(len(sampled_examples_df)),
            "examples_after_token_filter": int(len(single_token_examples_df)),
            "contexts_after_token_filter": int(
                single_token_examples_df["context_id"].nunique()
            ),
            "min_examples_per_context": int(family_sizes.min()),
            "max_examples_per_context": int(family_sizes.max()),
        },
        "overall_feature_counts": {
            "total_features": overall_total_features,
            "candidate_feature_count": overall_candidate_features,
            "candidate_feature_fraction": (
                overall_candidate_features / overall_total_features
                if overall_total_features > 0
                else 0.0
            ),
            "strict_common_feature_count": overall_strict_common_features,
            "strict_common_feature_fraction": (
                overall_strict_common_features / overall_total_features
                if overall_total_features > 0
                else 0.0
            ),
        },
        "feature_sets": feature_sets,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)
        summary_file.write("\n")


if __name__ == "__main__":
    main()
