import argparse
import json
import sys
from pathlib import Path

import pytest
import yaml
from sae_bench.evals.icl_features import ravel_value_geometry
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


def test_ravel_value_geometry_defaults_to_data_download_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ravel_value_geometry",
            "--model-name",
            "gemma-2-2b",
            "--source",
            "sae_lens",
            "--sae-release",
            "release",
            "--sae-id",
            "sae",
            "--result-root",
            "results/run",
            "--resume-fega",
        ],
    )

    args = ravel_value_geometry.parse_args()
    assert args.download_saes_dir == Path("data/downloaded_saes")
    assert args.cache_dir is None
    assert args.stages == ["ravel", "fega", "summarize"]
    assert args.resume_fega is True
    assert args.fega_base_config == ravel_value_geometry.DEFAULT_FEGA_CONFIG
    assert args.fega_base_config.is_file()


def test_ravel_value_geometry_accepts_fega_workflow_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ravel_value_geometry",
            "--model-name",
            "gemma-2-2b",
            "--source",
            "sae_lens",
            "--sae-release",
            "release",
            "--sae-id",
            "sae",
            "--result-root",
            "results/run",
            "--stages",
            "fega",
            "summarize",
            "--no-resume-fega",
            "--fega-phases",
            "compute_effect,geometry_reporting",
            "--fega-base-config",
            "custom.yaml",
        ],
    )

    args = ravel_value_geometry.parse_args()

    assert args.stages == ["fega", "summarize"]
    assert args.resume_fega is False
    assert args.fega_phases == "compute_effect,geometry_reporting"
    assert args.fega_base_config == Path("custom.yaml")


@pytest.mark.parametrize(
    "extra",
    [
        ["--stages", "unsupported-stage"],
        ["--unsupported-workflow-option"],
    ],
)
def test_ravel_value_geometry_rejects_unsupported_workflow_forms(
    monkeypatch, extra: list[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ravel_value_geometry",
            "--model-name",
            "gemma-2-2b",
            "--source",
            "sae_lens",
            "--sae-release",
            "release",
            "--sae-id",
            "sae",
            "--result-root",
            "results/run",
            *extra,
        ],
    )

    with pytest.raises(SystemExit):
        ravel_value_geometry.parse_args()


def test_custom_ravel_entrypoints_use_data_download_cache() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    custom_eval = (repository_root / "external" / "custom_eval_instance.py").read_text()
    checked_in_config = (
        repository_root
        / "fega"
        / "config"
        / "ravel"
        / "city_country.yaml"
    ).read_text()

    assert 'DOWNLOAD_SAES = Path("data/downloaded_saes")' in custom_eval
    assert custom_eval.count("download_location=str(DOWNLOAD_SAES)") == 2
    assert "download_saes_dir: data/downloaded_saes" in checked_in_config


def test_ravel_value_geometry_paths_are_portable_from_other_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(ravel_value_geometry, "REPO_ROOT", repo)
    monkeypatch.chdir(outside)

    reference = repo / "data/instance_eval_results/ravel/ref.json"
    mdbm_root = repo / "data/mdbm/ravel"
    config_path = repo / "results/run/model/sae/ravel/fega_config.yaml"
    weight = ravel_value_geometry._mdbm_weight_path(
        mdbm_root,
        reference,
        "city",
        "Country",
        RAVELEvalConfig(model_name="gpt2"),
    )

    assert weight == (
        mdbm_root / "ref.json" / "city_Country_downsampled-None_top-500.pt"
    )
    assert ravel_value_geometry._rel(reference) == (
        "data/instance_eval_results/ravel/ref.json"
    )
    assert not Path(
        ravel_value_geometry._config_rel(reference, config_path)
    ).is_absolute()


def test_generated_fega_config_keeps_repo_paths_relative(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(ravel_value_geometry, "REPO_ROOT", repo)
    base_config = repo / "fega/base.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(
        yaml.safe_dump(
            {
                "cache_dir": None,
                "induction": {},
                "phases": {
                    name: {}
                    for name in (
                        "data_prep",
                        "compute_effect",
                        "geometry_metrics",
                        "vmf",
                        "stability",
                        "geometry_reporting",
                    )
                },
            }
        )
    )
    config_path = repo / "results/run/model/sae/ravel/fega_config.yaml"
    reference = repo / "data/instance_eval_results/ravel/ref.json"
    mdbm_root = repo / "data/mdbm/ravel"
    weight = mdbm_root / "ref.json/city_Country_downsampled-None_top-500.pt"
    output_root = config_path.parent / "fega"
    args = argparse.Namespace(
        fega_base_config=base_config,
        entity="city",
        attribute="Country",
        device="cuda:0",
        cache_dir=None,
        download_saes_dir=repo / "data/downloaded_saes",
    )

    ravel_value_geometry._write_fega_config(
        args,
        reference_json=reference,
        weight_path=weight,
        output_root=output_root,
        mdbm_root=mdbm_root,
        config_path=config_path,
        custom_repo_id="org/saes",
    )

    generated = yaml.safe_load(config_path.read_text())
    assert set(generated) == {
        "cache_dir",
        "device",
        "download_saes_dir",
        "entity_attribute_selection",
        "mdbm_root",
        "mdbm_weight_path",
        "output_root",
        "phases",
        "reference_json",
        "reuse_model_across_phases",
        "run_id",
        "sae_repo_id",
        "source_kind",
    }
    for key in (
        "reference_json",
        "output_root",
        "download_saes_dir",
        "mdbm_root",
        "mdbm_weight_path",
    ):
        assert not Path(generated[key]).is_absolute()
    assert (config_path.parent / generated["reference_json"]).resolve() == reference
    assert (config_path.parent / generated["mdbm_root"]).resolve() == mdbm_root
    assert (config_path.parent / generated["mdbm_weight_path"]).resolve() == weight
    assert generated["cache_dir"] is None
    assert generated["phases"]["data_prep"]["readouts"] == ["final_resid"]
    assert generated["phases"]["vmf"]["effect_space"] == "pre_softcap_logits"


def test_run_fega_invokes_module_with_resume(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(ravel_value_geometry, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        ravel_value_geometry.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    config_path = tmp_path / "fega_config.yaml"

    ravel_value_geometry._run_fega(
        config_path,
        argparse.Namespace(resume_fega=True, fega_phases="compute_effect,geometry_reporting"),
    )

    assert calls == [
        (
            [
                sys.executable,
                "-m",
                "fega.cli",
                "run",
                "--config",
                str(config_path),
                "--resume",
                "--phases",
                "compute_effect,geometry_reporting",
            ],
            {"check": True, "cwd": tmp_path},
        )
    ]


def test_summarize_only_main_discovers_fega_run_and_rejects_neutral_sibling(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    reference = repo / "data/instance_eval_results/ravel/reference.json"
    weight = repo / "data/mdbm/ravel/weights.pt"
    result_root = repo / "results/run"
    args = argparse.Namespace(
        result_root=result_root,
        data_root=repo / "data",
        download_saes_dir=repo / "data/downloaded_saes",
        fega_base_config=repo / "fega/base.yaml",
        attribute="Country",
        ravel_attributes=["Country", "Continent"],
        model_name="gemma-2-2b",
        entity="city",
        stages=["summarize"],
        source="sae_lens",
        repo_id=None,
        resume_fega=True,
    )
    monkeypatch.setattr(ravel_value_geometry, "REPO_ROOT", repo)
    monkeypatch.setattr(ravel_value_geometry, "parse_args", lambda: args)
    monkeypatch.setattr(ravel_value_geometry.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        ravel_value_geometry,
        "_selected_saes",
        lambda _args: [("release", "sae")],
    )
    monkeypatch.setattr(
        ravel_value_geometry,
        "_run_ravel_for_sae",
        lambda *_args, **_kwargs: ("release", "sae", reference, weight),
    )
    summaries = []
    monkeypatch.setattr(
        ravel_value_geometry,
        "_summarize",
        lambda *_args, **kwargs: summaries.append(kwargs),
    )

    sae_root = result_root / "gemma-2-2b/release__sae/ravel/city_Country"
    summary = (
        sae_root
        / "fega"
        / reference.parent.name
        / reference.name
        / "city_Country/data_prep/select/feature_contexts_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")

    ravel_value_geometry.main()

    assert summaries[0]["fega_output_root"] == sae_root / "fega"
    assert summaries[0]["summary_path"] == summary

    summary.unlink()
    neutral_summary = (
        sae_root
        / "unsupported-engine"
        / reference.parent.name
        / reference.name
        / "city_Country/data_prep/select/feature_contexts_summary.json"
    )
    neutral_summary.parent.mkdir(parents=True)
    neutral_summary.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="FEGA context summary not found"):
        ravel_value_geometry.main()
    assert len(summaries) == 1


def test_fega_summary_preserves_empty_feature_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(ravel_value_geometry, "REPO_ROOT", tmp_path)
    context_summary = tmp_path / "context_summary.json"
    context_summary.write_text(json.dumps({"feature_stats": {}}), encoding="utf-8")
    output_root = tmp_path / "results/run/fega"
    output_dir = tmp_path / "results/run/value_features"

    ravel_value_geometry._summarize(
        argparse.Namespace(
            model_name="gemma-2-2b",
            entity="city",
            attribute="Country",
            kept_threshold=8.0,
            kept_op="ge",
        ),
        uid="release__sae",
        release="release",
        sae_id="sae",
        source_release="release",
        source_sae_id="sae",
        source_kind="sae_lens",
        reference_json=tmp_path / "data/reference.json",
        weight_path=tmp_path / "data/weights.pt",
        fega_output_root=output_root,
        summary_path=context_summary,
        output_dir=output_dir,
    )

    feature_ids = json.loads((output_dir / "feature_ids.json").read_text())
    summary = json.loads((output_dir / "value_feature_summary.json").read_text())
    assert feature_ids["feature_ids"] == []
    assert summary["fega_output_root"] == "results/run/fega"
    assert summary["value_like_feature_ids"] == []
    assert summary["value_like_feature_count"] == 0
