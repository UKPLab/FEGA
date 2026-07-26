from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from sae_bench.evals.icl_features import (
    ablation,
    audit_paper_artifacts,
    geometry_plots,
    ravel_feature_summary,
    ravel_geometry_sae_grid_plots,
)
from sae_bench.evals.icl_features.aggregate_results import collect_feature_counts
from sae_bench.evals.icl_features.artifact_naming import default_discovery_root
from sae_bench.evals.icl_features.feature_sets import (
    load_discovery_summary,
    per_sae_metrics_path,
)
from sae_bench.evals.icl_features.geometry_counts_latex_table import (
    _task_source_candidates,
    load_task_counts,
)

REPO_ROOT = Path(__file__).parents[4]


def test_ablation_accepts_downloaded_sae_data_directory(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ablation",
            "--task",
            "lsc",
            "--dataset-path",
            "dataset.json",
            "--discovery-summary",
            "summary.json",
            "--sae-uid",
            "sae",
            "--output-dir",
            "results/run",
            "--model-name",
            "gemma-2-2b",
            "--download-saes-dir",
            "cache/custom-saes",
        ],
    )

    assert ablation.parse_args().download_saes_dir == Path("cache/custom-saes")


def test_ablation_entrypoints_propagate_downloaded_sae_directory() -> None:
    scripts = [
        "run_icl_pipeline.sh",
        "run_saebench_gemma2b_width2pow16_smoke_test.sh",
        "run_gemmascope_width_sweep.sh",
    ]
    script_root = REPO_ROOT / "external/sae_bench/evals/icl_features/scripts"

    for name in scripts:
        source = (script_root / name).read_text(encoding="utf-8")
        assert '--download-saes-dir "${download_saes_dir}"' in source
        assert 'DOWNLOAD_SAES_DIR="${download_saes_dir}"' in source


def test_default_discovery_root_is_data_scoped() -> None:
    assert default_discovery_root(Path("results/sae_geometry_public")) == Path(
        "data/induction_feature_outputs/sae_geometry_public"
    )


def test_checked_in_induction_inputs_resolve_under_data() -> None:
    config_path = REPO_ROOT / "fega/config/induction/gemma2b_relu_trainer5.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert Path(os.path.normpath(config_path.parent / config["reference_json"])) == (
        REPO_ROOT / "data/prontoqa-1/prontoqa_induction_rule.json"
    )
    assert Path(
        os.path.normpath(config_path.parent / config["induction"]["summary_json"])
    ) == (
        REPO_ROOT / "data/induction_feature_outputs/gemma2b_relu_trainer5/summary.json"
    )
    assert config["download_saes_dir"] == "data/downloaded_saes"
    assert config["output_root"].startswith("results/")
    assert not (REPO_ROOT / "prontoqa-1").exists()
    assert not (REPO_ROOT / "induction_feature_outputs").exists()


def test_moved_discovery_summary_resolves_adjacent_metrics(tmp_path: Path) -> None:
    summary_path = tmp_path / "data/induction_feature_outputs/run/lsc/summary.json"
    metrics_path = summary_path.parent / "per_sae/sae_feature_metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text("feature_id,example_prevalence\n1,1.0\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "output_dir": "results/removed/discovery/lsc",
                "feature_sets": {"sae": {"candidate_feature_ids": [1]}},
            }
        ),
        encoding="utf-8",
    )

    summary = load_discovery_summary(summary_path)

    assert per_sae_metrics_path(summary, "sae") == metrics_path


def test_pipeline_routes_discovery_and_downloads_to_data(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n%s\\n\' "$OUTPUT_DIR" "$DOWNLOAD_SAES_DIR" > "$CAPTURE"\n',
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)

    discovery_root = tmp_path / "data/induction_feature_outputs/run"
    result_root = tmp_path / "results/run"
    capture = tmp_path / "capture.txt"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "REPO_ID": "example/saes",
        "SAE_LOCATIONS": "sae/location",
        "TASKS": "lsc",
        "STAGES": "discovery",
        "MODEL_NAME": "gemma-2-2b",
        "DISCOVERY_ROOT": str(discovery_root),
        "DOWNLOAD_SAES_DIR": str(tmp_path / "data/downloaded_saes"),
        "RESULT_ROOT": str(result_root),
        "RESUME": "0",
    }
    subprocess.run(
        [
            "/usr/bin/bash",
            str(
                REPO_ROOT
                / "external/sae_bench/evals/icl_features/scripts/run_icl_pipeline.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    output_dir, download_dir = capture.read_text(encoding="utf-8").splitlines()
    assert Path(output_dir) == discovery_root / "gemma-2-2b/lsc"
    assert Path(download_dir) == tmp_path / "data/downloaded_saes"
    assert not (result_root / "gemma-2-2b/discovery").exists()


def test_generated_fega_config_uses_portable_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "data/icl_features/gemma-2-2b/lsc.json"
    summary = (
        tmp_path / "data/induction_feature_outputs/run/gemma-2-2b/lsc/summary.json"
    )
    output = tmp_path / "results/run/gemma-2-2b/sae/lsc/fega_config_smoke.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("[]\n", encoding="utf-8")
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "output_dir": "results/removed/discovery/lsc",
                "feature_sets": {"sae": {"candidate_feature_ids": [1]}},
            }
        ),
        encoding="utf-8",
    )
    metrics = summary.parent / "per_sae/sae_feature_metrics.csv"
    metrics.parent.mkdir()
    metrics.write_text(
        "feature_id,example_prevalence\n1,1.0\n2,0.5\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "external/sae_bench/evals/icl_features/scripts/write_fega_config.py"
            ),
            "--task",
            "lsc",
            "--dataset",
            "data/icl_features/gemma-2-2b/lsc.json",
            "--summary",
            "data/induction_feature_outputs/run/gemma-2-2b/lsc/summary.json",
            "--sae-repo-id",
            "example/saes",
            "--sae-uid",
            "sae",
            "--download-saes-dir",
            "cache/custom-saes",
            "--smoke-profile",
            "--smoke-max-features",
            "2",
            "--base-config",
            str(REPO_ROOT / "fega/config/induction/gemma2b_relu_trainer5.yaml"),
            "--output",
            "results/run/gemma-2-2b/sae/lsc/fega_config_smoke.yaml",
            "--output-root",
            "results/run/gemma-2-2b/sae/lsc/fega",
        ],
        cwd=tmp_path,
        check=True,
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert set(generated) == {
        "cache_dir",
        "device",
        "download_saes_dir",
        "entity_attribute_selection",
        "induction",
        "output_root",
        "phases",
        "reference_json",
        "reuse_model_across_phases",
        "run_id",
        "sae_repo_id",
        "seed",
        "source_kind",
    }
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    assert (output.parent / generated["reference_json"]).resolve() == dataset
    assert (output.parent / generated["induction"]["summary_json"]).resolve() == summary
    assert generated["output_root"] == "results/run/gemma-2-2b/sae/lsc/fega"
    assert generated["download_saes_dir"] == "cache/custom-saes"
    assert generated["phases"]["data_prep"]["readouts"] == ["final_resid"]
    assert generated["phases"]["vmf"]["effect_space"] == "pre_softcap_logits"
    assert generated["induction"]["explicit_feature_ids"] == [1, 2]


def test_python_consumers_resolve_one_fega_geometry_tree(tmp_path: Path) -> None:
    result_root = tmp_path / "results/run"
    model_name = "gemma-2-2b"
    sae_uid = "release__sae"
    task = "lsc"
    geometry_map = (
        result_root
        / model_name
        / sae_uid
        / task
        / "fega"
        / model_name
        / f"{task}.json"
        / f"{task}_pointer_like"
        / "geometry_reporting"
        / "geometry_map_data.json"
    )
    geometry_map.parent.mkdir(parents=True)
    geometry_map.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "feature_id": 7,
                        "primary_label": "directed_ray",
                        "embedding": {"x": 1.0, "y": 2.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_candidates = _task_source_candidates(
        result_root, model_name, sae_uid, task
    )

    counts, source = load_task_counts(
        result_root,
        tmp_path / "discovery" / model_name,
        model_name,
        sae_uid,
        task,
        geometry_feature_set="candidate",
    )
    plot_rows = geometry_plots.load_joint_rows([(task, geometry_map)])
    grid_rows = ravel_geometry_sae_grid_plots._load_map_rows(
        geometry_map,
        sae_label="ReLU",
        include_insufficient_evidence=True,
    )
    feature_ids_path = geometry_map.parent / "feature_ids.json"
    feature_ids_path.write_text(json.dumps({"feature_ids": [7]}), encoding="utf-8")

    assert counts["directed_ray"] == 1
    assert source == str(geometry_map)
    assert [kind for kind, _path in source_candidates] == [
        "fega_counts",
        "fega_records",
        "fega_map",
        "cross_task_plot",
        "pointer_plot",
    ]
    assert plot_rows[0]["source_map"] == str(geometry_map.resolve())
    assert grid_rows[0]["source_map"] == str(geometry_map.resolve())
    assert ravel_feature_summary._feature_ids(feature_ids_path) == [7]


def test_paper_audit_reconstructs_fega_geometry_root(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results/run"
    output = tmp_path / "audit.json"
    seen: list[Path] = []
    discovery_summary = tmp_path / "discovery/gemma-2-2b/lsc/summary.json"
    discovery_summary.parent.mkdir(parents=True)
    discovery_summary.write_text(
        json.dumps(
            {
                "feature_sets": {
                    "release__sae": {
                        "selection_thresholds": {
                            "min_example_fraction": 0.9,
                            "min_query_fraction_per_context": 0.9,
                            "min_context_fraction": 0.9,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_paper_artifacts, "TASKS", ("lsc",))
    monkeypatch.setattr(
        audit_paper_artifacts,
        "parse_args",
        lambda: argparse.Namespace(
            result_root=result_root,
            discovery_root=tmp_path / "discovery",
            model_name="gemma-2-2b",
            sae_uid=["release__sae"],
            feature_set="threshold",
            expected_examples=1,
            expected_random_trials=1,
            min_example_fraction=0.9,
            min_query_fraction=0.9,
            min_family_fraction=0.9,
            output=output,
        ),
    )
    monkeypatch.setattr(
        audit_paper_artifacts,
        "_require",
        lambda path, _missing: seen.append(path),
    )

    audit_paper_artifacts.main()

    assert (
        result_root
        / "gemma-2-2b/release__sae/lsc/fega/gemma-2-2b"
        / "lsc.json/lsc_pointer_like/geometry_reporting/geometry_map_data.json"
    ) in seen


def test_aggregate_reads_feature_counts_from_discovery_root(tmp_path: Path) -> None:
    summary_path = tmp_path / "discovery/gemma-2-2b/lsc/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "model_name": "gemma-2-2b",
                "dataset_metadata": {"task_name": "lsc"},
                "feature_sets": {
                    "sae": {
                        "candidate_feature_count": 2,
                        "strict_common_feature_count": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = collect_feature_counts(tmp_path / "discovery")
    assert [(row["task"], row["threshold_feature_count"]) for row in rows] == [
        ("lsc", 2)
    ]
