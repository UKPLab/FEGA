from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import sae_bench.evals.ravel.main as ravel
import sae_bench.sae_bench_utils.general_utils as general_utils
import yaml
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    load_dictionary_learning_sae,
)
from sae_bench.evals.icl_features.artifact_naming import artifact_tag, tagged_paths
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

import fega

DEFAULT_RAVEL_ATTRIBUTES = ("Country", "Continent", "Language")
REPO_ROOT = Path.cwd().resolve()
DEFAULT_FEGA_CONFIG = (
    Path(fega.__file__).resolve().parent
    / "config"
    / "induction"
    / "gemma2b_relu_trainer5.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run existing RAVEL value-feature discovery and FEGA geometry for one "
            "or more SAEs, without changing RAVEL/FEGA logic."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--source",
        choices=["custom_repo", "sae_lens"],
        required=True,
        help="custom_repo for dictionary-learning checkpoints; sae_lens for Gemma Scope.",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--sae-location",
        action="append",
        default=[],
        help="Custom SAE location inside --repo-id. Pass once per SAE.",
    )
    parser.add_argument("--sae-release", default=None)
    parser.add_argument(
        "--sae-id",
        action="append",
        default=[],
        help="SAELens SAE id. Pass once per SAE.",
    )
    parser.add_argument("--sae-regex-pattern", default=None)
    parser.add_argument("--sae-block-pattern", default=None)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["ravel", "fega", "summarize"],
        choices=["ravel", "fega", "summarize"],
    )
    parser.add_argument("--force-rerun-ravel", action="store_true")
    parser.add_argument("--resume-fega", action="store_true", default=True)
    parser.add_argument("--no-resume-fega", dest="resume_fega", action="store_false")
    parser.add_argument("--fega-phases", default=None)
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--llm-dtype", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--entity", default="city")
    parser.add_argument("--attribute", default="Country")
    parser.add_argument(
        "--ravel-attributes",
        nargs="+",
        default=list(DEFAULT_RAVEL_ATTRIBUTES),
        help=(
            "Attributes used by the RAVEL run. Keep at least two attributes so "
            "isolation examples are defined; FEGA still selects --attribute only."
        ),
    )
    parser.add_argument("--top-n-entities", type=int, default=500)
    parser.add_argument("--top-n-templates", type=int, default=90)
    parser.add_argument("--num-pairs-per-attribute", type=int, default=5000)
    parser.add_argument("--train-test-split", type=float, default=0.7)
    parser.add_argument("--kept-threshold", type=float, default=8.0)
    parser.add_argument("--kept-op", choices=["gt", "ge"], default="ge")
    parser.add_argument(
        "--download-saes-dir",
        type=Path,
        default=Path("data/downloaded_saes"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--fega-base-config",
        type=Path,
        default=DEFAULT_FEGA_CONFIG,
    )
    return parser.parse_args()


def _rel(path: Path) -> str:
    path = _repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _config_rel(path: Path, config_path: Path) -> str:
    return os.path.relpath(_repo_path(path), start=config_path.parent)


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _sae_uid(release: str, sae_id: str) -> str:
    return f"{_sanitize(release)}__{_sanitize(sae_id)}"


def _custom_release(repo_id: str, location: str) -> str:
    return f"{repo_id.split('/')[-1]}_{location.replace('/', '_')}"


def _selected_saes(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if args.source == "custom_repo":
        if not args.repo_id or not args.sae_location:
            raise ValueError(
                "--source custom_repo requires --repo-id and --sae-location"
            )
        return [(args.repo_id, location) for location in args.sae_location]

    if args.sae_release and args.sae_id:
        return [(args.sae_release, sae_id) for sae_id in args.sae_id]
    if args.sae_regex_pattern and args.sae_block_pattern:
        return get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    raise ValueError(
        "--source sae_lens requires either --sae-release/--sae-id or "
        "--sae-regex-pattern/--sae-block-pattern"
    )


def _ravel_config(
    args: argparse.Namespace, artifact_dir: Path, mdbm_dir: Path
) -> RAVELEvalConfig:
    config = RAVELEvalConfig(
        model_name=args.model_name,
        random_seed=args.random_seed,
        entity_attribute_selection={args.entity: list(args.ravel_attributes)},
        top_n_entities=args.top_n_entities,
        top_n_templates=args.top_n_templates,
        num_pairs_per_attribute=args.num_pairs_per_attribute,
        train_test_split=args.train_test_split,
        artifact_dir=_rel(artifact_dir),
        mdbm_dir=_rel(mdbm_dir),
    )
    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    return config


def _ravel_result_path(output_folder: Path, release: str, sae_id: str) -> Path:
    return Path(general_utils.get_results_filepath(str(output_folder), release, sae_id))


def _mdbm_weight_path(
    mdbm_dir: Path,
    reference_json: Path,
    entity: str,
    attribute: str,
    config: RAVELEvalConfig,
) -> Path:
    filename = (
        f"{entity}_{attribute}_downsampled-"
        f"{config.full_dataset_downsample}_top-{config.top_n_entities}.pt"
    )
    return mdbm_dir / reference_json.name / filename


def _run_ravel_for_sae(
    args: argparse.Namespace,
    *,
    release_or_repo: str,
    sae_location_or_id: str,
    output_folder: Path,
    artifact_dir: Path,
    mdbm_dir: Path,
) -> tuple[str, str, Path, Path]:
    config = _ravel_config(args, artifact_dir, mdbm_dir)
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    output_folder_arg = Path(_rel(output_folder))
    if args.source == "custom_repo":
        repo_id = release_or_repo
        location = sae_location_or_id
        sae = load_dictionary_learning_sae(
            repo_id=repo_id,
            location=location,
            model_name=args.model_name,
            device=args.device,
            dtype=llm_dtype,
            download_location=str(args.download_saes_dir),
        )
        release_for_ravel = _custom_release(repo_id, location)
        sae_id_for_ravel = "custom_sae"
        selected = [(release_for_ravel, sae)]
    else:
        release_for_ravel = release_or_repo
        sae_id_for_ravel = sae_location_or_id
        selected = [(release_for_ravel, sae_id_for_ravel)]

    reference_path = _ravel_result_path(
        output_folder_arg, release_for_ravel, sae_id_for_ravel
    )
    if "ravel" in args.stages:
        ravel.run_eval(
            config,
            selected,
            args.device,
            str(output_folder_arg),
            force_rerun=args.force_rerun_ravel,
            artifacts_path=_rel(artifact_dir.parent),
        )

    weight_path = _mdbm_weight_path(
        mdbm_dir,
        reference_path,
        args.entity,
        args.attribute,
        config,
    )
    return release_for_ravel, sae_id_for_ravel, reference_path, weight_path


def _write_fega_config(
    args: argparse.Namespace,
    *,
    reference_json: Path,
    weight_path: Path,
    output_root: Path,
    mdbm_root: Path,
    config_path: Path,
    custom_repo_id: str | None,
) -> None:
    config = yaml.safe_load(args.fega_base_config.read_text(encoding="utf-8"))
    config.pop("induction", None)
    config["run_id"] = f"ravel_{args.entity}_{args.attribute}_value_geometry"
    config["source_kind"] = "ravel"
    config["reference_json"] = _config_rel(reference_json, config_path)
    config["output_root"] = _rel(output_root)
    config["device"] = args.device
    if args.cache_dir is not None:
        config["cache_dir"] = str(args.cache_dir.expanduser())
    config["download_saes_dir"] = _rel(args.download_saes_dir)
    config["sae_repo_id"] = custom_repo_id
    config["mdbm_root"] = _config_rel(mdbm_root, config_path)
    config["mdbm_weight_path"] = _config_rel(weight_path, config_path)
    config["entity_attribute_selection"] = {args.entity: [args.attribute]}
    config["reuse_model_across_phases"] = True
    phases = config["phases"]
    phases["data_prep"]["enabled"] = True
    phases["data_prep"]["max_contexts"] = 64
    phases["data_prep"]["min_contexts"] = 8
    phases["data_prep"]["readouts"] = ["final_resid"]
    phases["data_prep"]["gram_cache"] = True
    phases["compute_effect"]["enabled"] = True
    phases["geometry_metrics"]["enabled"] = True
    phases["vmf"]["enabled"] = True
    phases["vmf"]["effect_space"] = "pre_softcap_logits"
    phases["stability"]["enabled"] = True
    phases["geometry_reporting"]["enabled"] = True
    phases["geometry_reporting"]["write_csv"] = True
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_fega(config_path: Path, args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "fega.cli", "run", "--config", str(config_path)]
    if args.resume_fega:
        cmd.append("--resume")
    if args.fega_phases:
        cmd.extend(["--phases", args.fega_phases])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _selected_ids(
    feature_stats: dict[str, Any], *, threshold: float, op: str
) -> list[int]:
    ids: list[int] = []
    for raw_feature_id, stats in feature_stats.items():
        kept = float(stats.get("kept", 0.0))
        keep = kept > threshold if op == "gt" else kept >= threshold
        if keep:
            ids.append(int(raw_feature_id))
    return sorted(set(ids))


def _summarize(
    args: argparse.Namespace,
    *,
    uid: str,
    release: str,
    sae_id: str,
    source_release: str,
    source_sae_id: str,
    source_kind: str,
    reference_json: Path,
    weight_path: Path,
    fega_output_root: Path,
    summary_path: Path,
    output_dir: Path,
) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    tag = artifact_tag(model_name=args.model_name, sae_uid=uid)
    ids = _selected_ids(
        payload["feature_stats"],
        threshold=args.kept_threshold,
        op=args.kept_op,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_ids_path = output_dir / "feature_ids.json"
    for path in tagged_paths(feature_ids_path, tag):
        path.write_text(
            json.dumps({"artifact_tag": tag, "feature_ids": ids}, indent=2) + "\n"
        )

    value_summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "task": "ravel",
        "entity": args.entity,
        "attribute": args.attribute,
        "sae_uid": uid,
        "artifact_tag": tag,
        "sae_release": release,
        "sae_id": sae_id,
        "source_kind": source_kind,
        "source_release": source_release,
        "source_sae_id": source_sae_id,
        "reference_json": _rel(reference_json),
        "mdbm_weight_path": _rel(weight_path),
        "fega_output_root": _rel(fega_output_root),
        "selection_rule": {
            "field": "kept",
            "op": args.kept_op,
            "threshold": args.kept_threshold,
        },
        "value_like_feature_ids": ids,
        "value_like_feature_count": len(ids),
        "context_summary": payload,
    }
    for path in tagged_paths(output_dir / "value_feature_summary.json", tag):
        path.write_text(json.dumps(value_summary, indent=2) + "\n", encoding="utf-8")
    discovery_summary = {
        "schema_version": 1,
        "model_name": args.model_name,
        "task": "ravel",
        "feature_sets": {
            uid: {
                "sae_release": release,
                "sae_id": sae_id,
                "threshold_feature_ids": ids,
                "threshold_feature_count": len(ids),
                "candidate_feature_ids": ids,
                "candidate_feature_count": len(ids),
                "selection_method": "RAVEL MDBM positive mask filtered by selected contexts",
                "selection_rule": value_summary["selection_rule"],
                "source_path": _rel(output_dir / "value_feature_summary.json"),
            }
        },
    }
    for path in tagged_paths(output_dir / "ravel_discovery_summary.json", tag):
        path.write_text(
            json.dumps(discovery_summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Wrote {len(ids)} RAVEL value-like feature ids to {feature_ids_path}")


def main() -> None:
    args = parse_args()
    args.result_root = _repo_path(args.result_root)
    args.data_root = _repo_path(args.data_root)
    args.download_saes_dir = _repo_path(args.download_saes_dir)
    args.fega_base_config = _repo_path(args.fega_base_config)
    if args.attribute not in args.ravel_attributes:
        raise ValueError("--attribute must be present in --ravel-attributes")
    if len(args.ravel_attributes) < 2:
        raise ValueError("--ravel-attributes must contain at least two attributes")

    model_root = args.result_root / args.model_name
    ravel_output_folder = args.data_root / "instance_eval_results" / "ravel"
    artifact_dir = args.data_root / "artifacts" / "ravel"
    mdbm_dir = args.data_root / "mdbm" / "ravel"

    for release_or_repo, sae_location_or_id in _selected_saes(args):
        if sae_location_or_id is None:
            raise ValueError("Internal error: missing SAE location/id")

        source_release = release_or_repo
        source_sae_id = sae_location_or_id
        release_for_ravel, sae_id_for_ravel, reference_json, weight_path = (
            _run_ravel_for_sae(
                args,
                release_or_repo=release_or_repo,
                sae_location_or_id=sae_location_or_id,
                output_folder=ravel_output_folder,
                artifact_dir=artifact_dir,
                mdbm_dir=mdbm_dir,
            )
        )
        uid = _sae_uid(release_for_ravel, sae_id_for_ravel)
        sae_root = model_root / uid / "ravel" / f"{args.entity}_{args.attribute}"
        fega_output_root = sae_root / "fega"
        config_path = sae_root / "fega_config.yaml"

        if "fega" in args.stages:
            _write_fega_config(
                args,
                reference_json=reference_json,
                weight_path=weight_path,
                output_root=fega_output_root,
                mdbm_root=mdbm_dir,
                config_path=config_path,
                custom_repo_id=args.repo_id if args.source == "custom_repo" else None,
            )
            _run_fega(config_path, args)

        if "summarize" in args.stages:
            geometry_run = (
                fega_output_root
                / reference_json.parent.name
                / reference_json.name
                / f"{args.entity}_{args.attribute}"
            )
            summary_path = (
                geometry_run / "data_prep" / "select" / "feature_contexts_summary.json"
            )
            if not summary_path.exists():
                raise FileNotFoundError(
                    f"FEGA context summary not found: {summary_path}. "
                    "Run with --stages fega summarize, or check the FEGA output root."
                )
            _summarize(
                args,
                uid=uid,
                release=release_for_ravel,
                sae_id=sae_id_for_ravel,
                source_release=source_release,
                source_sae_id=source_sae_id,
                source_kind=args.source,
                reference_json=reference_json,
                weight_path=weight_path,
                fega_output_root=fega_output_root,
                summary_path=summary_path,
                output_dir=sae_root / "value_features",
            )


if __name__ == "__main__":
    main()
