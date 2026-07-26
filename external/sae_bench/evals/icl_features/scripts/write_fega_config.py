from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from sae_bench.evals.icl_features.feature_sets import (
    load_discovery_summary,
    per_sae_metrics_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a task-specific FEGA geometry config from the checked template."
    )
    parser.add_argument(
        "--task", choices=["lsc", "wc", "tt", "prontoqa"], required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sae-repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("fega/config/induction/gemma2b_relu_trainer5.yaml"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/sae_geometry")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download-saes-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-set", choices=["candidate", "strict_common"], default="candidate"
    )
    parser.add_argument("--sae-uid", default=None)
    parser.add_argument(
        "--smoke-profile",
        action="store_true",
        help=(
            "Use a bounded FEGA configuration and explicit high-prevalence features. "
            "This is for installation/path validation, not scientific results."
        ),
    )
    parser.add_argument("--smoke-max-features", type=int, default=2)
    return parser.parse_args()


def _smoke_feature_ids(
    summary: dict,
    *,
    sae_uid: str,
    maximum: int,
) -> list[int]:
    if maximum <= 0:
        raise ValueError("--smoke-max-features must be positive")
    entry = summary["feature_sets"][sae_uid]
    candidate_ids = [int(value) for value in entry.get("candidate_feature_ids", [])]
    if len(candidate_ids) >= maximum:
        return candidate_ids[:maximum]

    metrics_path = per_sae_metrics_path(summary, sae_uid)
    metrics = pd.read_csv(metrics_path)
    ranking_columns = [
        column
        for column in (
            "consistent_context_prevalence",
            "example_prevalence",
            "mean_activation_when_active",
        )
        if column in metrics.columns
    ]
    if not ranking_columns:
        raise ValueError(f"No prevalence columns found in {metrics_path}")
    ranked = metrics.sort_values(
        ranking_columns,
        ascending=[False] * len(ranking_columns),
        kind="stable",
    )
    ranked_ids = ranked["feature_id"].astype(int).tolist()
    return list(dict.fromkeys([*candidate_ids, *ranked_ids]))[:maximum]


def _apply_smoke_profile(config: dict) -> None:
    induction = config["induction"]
    induction["max_source_contexts"] = 4
    induction["max_source_examples"] = 32

    phases = config["phases"]
    phases["data_prep"].update(
        {
            "batch_size": 8,
            "save_chunk_size": 32,
            "max_contexts": 4,
            "min_contexts": 2,
        }
    )
    phases["compute_effect"].update(
        {
            "batch_size": 32,
            "min_coverage": 2,
            "effect_shard_size": 32,
            "cache_max_chunks": 2,
            "cache_max_bytes": 1073741824,
        }
    )
    phases["geometry_metrics"]["span"]["k_values"] = [1, 2]
    phases["geometry_metrics"]["resid"]["k_values"] = [1, 2]
    phases["vmf"].update(
        {
            "effect_space": "pre_softcap_logits",
            "workers": 1,
            "checkpoint_flush_features": 1,
            "k_values": [1, 2],
            "n_min": 2,
            "min_mode_count": 1,
            "resample_rounds": 1,
            "n_init": 1,
            "max_iter": 30,
        }
    )
    phases["stability"].update(
        {
            "workers": 1,
            "checkpoint_flush_features": 1,
        }
    )
    phases["stability"]["scalar"]["bootstrap_rounds"] = 4
    phases["stability"]["subspace"].update(
        {
            "k_values": [1, 2],
            "resample_rounds": 2,
        }
    )
    phases["stability"]["sample_size"].update(
        {
            "target_sizes": [2, 4],
            "subset_rounds": 2,
            "strong_subset_rounds": 2,
            "max_enumerated_subsets": 2,
        }
    )
    phases["stability"]["leave_out"].update(
        {
            "max_leave_two_out_pairs": 2,
            "min_group_count": 1,
            "min_group_size": 1,
        }
    )
    phases["geometry_reporting"].update(
        {
            "atlas_include_insufficient_evidence": True,
            "embedding": "pca",
        }
    )


def _config_relative(path: Path, config_path: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), config_path.parent.resolve())


def _working_directory_relative(path: Path) -> str:
    path = path.expanduser()
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["run_id"] = f"{args.task}_pointer_geometry_{args.feature_set}"
    config["source_kind"] = "induction"
    config["reference_json"] = _config_relative(args.dataset, args.output)
    config["output_root"] = _working_directory_relative(args.output_root)
    config["device"] = args.device
    config["sae_repo_id"] = args.sae_repo_id
    if args.download_saes_dir is not None:
        config["download_saes_dir"] = _working_directory_relative(
            args.download_saes_dir
        )
    config["entity_attribute_selection"] = {args.task: ["pointer_like"]}
    config["phases"]["data_prep"]["readouts"] = ["final_resid"]
    config["phases"]["vmf"]["effect_space"] = "pre_softcap_logits"
    config["induction"]["summary_json"] = _config_relative(args.summary, args.output)
    config["induction"]["feature_set"] = args.feature_set
    summary = load_discovery_summary(args.summary)
    feature_sets = summary.get("feature_sets") or {}
    if args.sae_uid is not None:
        sae_uid = args.sae_uid
    elif len(feature_sets) == 1:
        sae_uid = next(iter(feature_sets))
    else:
        raise ValueError(
            "The discovery summary contains multiple SAE feature sets; pass --sae-uid."
        )
    config["induction"]["sae_uid"] = sae_uid
    config["induction"]["require_model_correct"] = True
    config["induction"]["stratify_by"] = [
        "context_id",
        "support_example_index",
    ]
    config["phases"]["geometry_reporting"]["threshold_profile"] = "paper"
    if args.smoke_profile:
        config["run_id"] = f"{args.task}_pointer_geometry_smoke"
        config["induction"]["feature_set"] = "explicit"
        config["induction"]["explicit_feature_ids"] = _smoke_feature_ids(
            summary,
            sae_uid=sae_uid,
            maximum=args.smoke_max_features,
        )
        _apply_smoke_profile(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
