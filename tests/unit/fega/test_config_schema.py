from __future__ import annotations

import json
from pathlib import Path

import pytest

from fega.config_schema import (
    DirectionalMixtureFitConfig,
    FEGAPipelineConfig,
    GeometryMetricsConfig,
    GeometryMetricsCRayConfig,
    GeometryMetricsEffectiveRankConfig,
    GeometryMetricsResidConfig,
    GeometryMetricsSpanConfig,
    GeometryReportingConfig,
)


def _write_reference_json(path: Path) -> None:
    payload = {
        "eval_config": {
            "model_name": "gpt2",
            "llm_dtype": "float32",
            "entity_attribute_selection": {"city": ["Country"]},
        }
    }
    path.write_text(json.dumps(payload))


def _base_yaml(reference_json: Path, output_root: Path) -> str:
    return (
        f"reference_json: {reference_json}\n"
        f"output_root: {output_root}\n"
        "device: cpu\n"
        "entity_attribute_selection:\n"
        "  city: ['Country']\n"
    )


def test_geometry_threshold_profile_is_paper_only(
    tmp_path: Path,
) -> None:
    """Default to the sole paper profile and reject retired profile names."""
    # Parse the default, then prove an old profile cannot enter the main schema.
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    default_path = tmp_path / "default.yaml"
    default_path.write_text(_base_yaml(reference_json, tmp_path / "default-out"))

    default = FEGAPipelineConfig.from_file(default_path)
    assert isinstance(default.phases.geometry_reporting, GeometryReportingConfig)
    assert default.phases.geometry_reporting.threshold_profile == "paper"

    retired_path = tmp_path / "retired.yaml"
    retired_path.write_text(
        _base_yaml(reference_json, tmp_path / "retired-out")
        + "phases:\n"
        + "  geometry_reporting:\n"
        + "    threshold_profile: exploratory\n"
    )
    with pytest.raises(ValueError, match="must be `paper`"):
        FEGAPipelineConfig.from_file(retired_path)


def _write_induction_summary(path: Path, *, multi_sae: bool = False) -> None:
    feature_set = {
        "sae_release": "canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
        "sae_id": "gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_5",
        "layer": 12,
        "hook_name": "blocks.12.hook_resid_post",
        "candidate_feature_ids": [1378, 1985, 7678],
        "strict_common_feature_ids": [1378, 7678],
        "candidate_feature_count": 3,
        "strict_common_feature_count": 2,
    }
    feature_sets = {"sae_a": feature_set}
    if multi_sae:
        feature_sets["sae_b"] = {**feature_set, "candidate_feature_ids": [1]}
    path.write_text(
        json.dumps(
            {
                "dataset_path": "prontoqa-1/prontoqa_induction_rule.json",
                "model_name": "gemma-2-2b",
                "llm_dtype": "bfloat16",
                "sae_selection_mode": "custom_repo",
                "filtering": {
                    "answer_prefix": " ",
                    "require_model_correct": True,
                    "max_contexts": None,
                    "max_examples": None,
                },
                "activation_definition": {
                    "active_if_post_encode_activation_exceeds": 0.0,
                    "token_position": "final_prompt_token_before_first_answer_token",
                },
                "overall_feature_counts": {"total_features": 65536},
                "feature_sets": feature_sets,
            }
        )
    )


def test_post_softcap_logits_pipeline_readout_is_unsupported(tmp_path: Path) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  data_prep:\n"
        + "    readouts: ['post_softcap_logits']\n"
    )

    with pytest.raises(
        ValueError, match="Unsupported data_prep readout 'post_softcap_logits'"
    ):
        FEGAPipelineConfig.from_file(cfg_path)


def test_compute_effect_enabled_requires_gram_for_final_resid(
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  data_prep:\n"
        + "    readouts: ['final_resid']\n"
        + "  compute_effect:\n"
        + "    enabled: true\n"
    )

    with pytest.raises(ValueError, match="gram_cache|final_resid"):
        FEGAPipelineConfig.from_file(cfg_path)


def test_stability_config_parses_nested_values_and_serializes(
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  stability:\n"
        + "    enabled: true\n"
        + "    effect_space: final_resid\n"
        + "    workers: 3\n"
        + "    resume: true\n"
        + "    checkpoint_flush_features: 17\n"
        + "    seed: 123\n"
        + "    scalar:\n"
        + "      enabled: true\n"
        + "      bootstrap_rounds: 7\n"
        + "      ci_quantiles: [0.1, 0.9]\n"
        + "    subspace:\n"
        + "      enabled: true\n"
        + "      k_values: [1, 3]\n"
        + "      resample_fraction: 0.5\n"
        + "      resample_rounds: 4\n"
        + "      angle_p90_quantile: 0.8\n"
        + "      eig_floor: 1.0e-6\n"
        + "    sample_size:\n"
        + "      enabled: true\n"
        + "      target_sizes: [8, 16]\n"
        + "      subset_rounds: 5\n"
        + "      strong_subset_rounds: 6\n"
        + "      max_enumerated_subsets: 3\n"
        + "    leave_out:\n"
        + "      enabled: true\n"
        + "      max_leave_two_out_pairs: 2\n"
        + "      min_group_count: 3\n"
        + "      min_group_size: 4\n"
    )

    cfg = FEGAPipelineConfig.from_file(cfg_path)
    stability = cfg.phases.stability
    assert stability.enabled is True
    assert stability.effect_space == "final_resid"
    assert stability.workers == 3
    assert stability.resume is True
    assert stability.checkpoint_flush_features == 17
    assert stability.seed == 123
    assert stability.scalar.bootstrap_rounds == 7
    assert stability.scalar.ci_quantiles == [0.1, 0.9]
    assert stability.subspace.k_values == [1, 3]
    assert stability.subspace.resample_fraction == pytest.approx(0.5)
    assert stability.subspace.resample_rounds == 4
    assert stability.subspace.angle_p90_quantile == pytest.approx(0.8)
    assert stability.subspace.eig_floor == pytest.approx(1.0e-6)
    assert stability.sample_size.target_sizes == [8, 16]
    assert stability.sample_size.subset_rounds == 5
    assert stability.sample_size.strong_subset_rounds == 6
    assert stability.sample_size.max_enumerated_subsets == 3
    assert stability.leave_out.max_leave_two_out_pairs == 2
    assert stability.leave_out.min_group_count == 3
    assert stability.leave_out.min_group_size == 4
    serialized = cfg.to_dict()["phases"]["stability"]
    assert serialized["seed"] == 123
    assert serialized["workers"] == 3
    assert serialized["resume"] is True
    assert serialized["checkpoint_flush_features"] == 17


@pytest.mark.parametrize(
    "body, match",
    [
        ("    made_up: true\n", "phases.stability.made_up"),
        ("    effect_space: bad\n", "effect_space"),
        ("    effect_spaces: ['final_resid']\n", "effect_spaces"),
        ("    workers: 0\n", "phases.stability.workers"),
        ("    workers: -1\n", "phases.stability.workers"),
        ("    workers: true\n", "phases.stability.workers"),
        ("    workers: 1.5\n", "phases.stability.workers"),
        ("    resume: 'yes'\n", "phases.stability.resume"),
        ("    resume: 1\n", "phases.stability.resume"),
        ("    resume: 0\n", "phases.stability.resume"),
        ("    resume: null\n", "phases.stability.resume"),
        (
            "    checkpoint_flush_features: 0\n",
            "phases.stability.checkpoint_flush_features",
        ),
        (
            "    checkpoint_flush_features: -1\n",
            "phases.stability.checkpoint_flush_features",
        ),
        (
            "    checkpoint_flush_features: true\n",
            "phases.stability.checkpoint_flush_features",
        ),
        (
            "    checkpoint_flush_features: 1.5\n",
            "phases.stability.checkpoint_flush_features",
        ),
        ("    scalar:\n      ci_quantiles: [0.9, 0.1]\n", "ci_quantiles"),
        ("    scalar:\n      ci_quantiles: [-0.1, 0.9]\n", "ci_quantiles"),
        ("    subspace:\n      resample_fraction: 0\n", "resample_fraction"),
        ("    subspace:\n      angle_p90_quantile: 1.5\n", "angle_p90_quantile"),
        ("    subspace:\n      k_values: [1, 1]\n", "k_values"),
        ("    sample_size:\n      target_sizes: [8, 0]\n", "target_sizes"),
        ("    scalar:\n      bootstrap_rounds: -1\n", "bootstrap_rounds"),
        ("    leave_out:\n      max_leave_two_out_pairs: -1\n", "max_leave_two"),
    ],
)
def test_stability_config_rejects_invalid_values(
    tmp_path: Path, body: str, match: str
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  stability:\n"
        + body
    )

    with pytest.raises(ValueError, match=match):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize("workers", ["0", "-1", "true", "1.5"])
def test_vmf_rejects_invalid_workers(tmp_path: Path, workers: str) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    workers: {workers}\n"
    )

    with pytest.raises(ValueError, match="phases.vmf.workers"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize("max_buffers", ["0", "-1", "true", "1.5"])
def test_vmf_rejects_invalid_max_vocab_buffers(
    tmp_path: Path, max_buffers: str
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    max_vocab_buffers: {max_buffers}\n"
    )

    with pytest.raises(ValueError, match="phases.vmf.max_vocab_buffers"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize("resume", ["'yes'", "1", "0", "null"])
def test_vmf_rejects_invalid_resume(tmp_path: Path, resume: str) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    resume: {resume}\n"
    )

    with pytest.raises(ValueError, match="phases.vmf.resume"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize("checkpoint_flush_features", ["0", "-1", "true", "1.5"])
def test_vmf_rejects_invalid_checkpoint_flush_features(
    tmp_path: Path, checkpoint_flush_features: str
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    checkpoint_flush_features: {checkpoint_flush_features}\n"
    )

    with pytest.raises(ValueError, match="phases.vmf.checkpoint_flush_features"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize("k_values", ["[]", "[0]", "[1, 1]", "[1.5]", "[5]"])
def test_selected_mode_count_values_must_be_feasible_fega_subset(
    tmp_path: Path, k_values: str
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    k_values: {k_values}\n"
    )

    with pytest.raises(ValueError, match="k_values"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize(
    "removed_field",
    [
        "selection",
        "eps",
        "n_min",
        "min_mode_count",
        "min_mode_frac",
        "tau_ray",
        "tau_axis",
        "tau_b_axis",
        "tau_mix",
        "tau_mode_mass",
        "tau_mode_fers",
        "tau_mode_c_ray",
        "tau_assignment_stability",
        "label_agreement",
    ],
)
def test_vmf_rejects_every_removed_fit_stage_field(
    tmp_path: Path, removed_field: str
) -> None:
    """Reject every former fit-stage threshold or acceptance key without aliases."""
    # Inject one removed field at a time through the public file-loading boundary.
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  vmf:\n"
        + f"    {removed_field}: 1\n"
    )

    with pytest.raises(ValueError, match=removed_field):
        FEGAPipelineConfig.from_file(cfg_path)


def test_geometry_metrics_effective_rank_requires_final_resid_span_and_resid(
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)

    logits_cfg = tmp_path / "logits.yaml"
    logits_cfg.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  geometry_metrics:\n"
        + "    effect_space: logits\n"
        + "    effective_rank:\n"
        + "      enabled: true\n"
    )
    with pytest.raises(ValueError, match="geometry_metrics.effect_space.*final_resid"):
        FEGAPipelineConfig.from_file(logits_cfg)

    no_span_cfg = tmp_path / "no_span.yaml"
    no_span_cfg.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  geometry_metrics:\n"
        + "    effect_space: final_resid\n"
        + "    span:\n"
        + "      enabled: false\n"
        + "    resid:\n"
        + "      enabled: true\n"
        + "    effective_rank:\n"
        + "      enabled: true\n"
    )
    with pytest.raises(ValueError, match="span.enabled"):
        FEGAPipelineConfig.from_file(no_span_cfg)

    no_resid_cfg = tmp_path / "no_resid.yaml"
    no_resid_cfg.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  geometry_metrics:\n"
        + "    effect_space: final_resid\n"
        + "    span:\n"
        + "      enabled: true\n"
        + "    resid:\n"
        + "      enabled: false\n"
        + "    effective_rank:\n"
        + "      enabled: true\n"
    )
    with pytest.raises(ValueError, match="resid.enabled"):
        FEGAPipelineConfig.from_file(no_resid_cfg)


def test_geometry_metrics_residual_k_values_default_to_supported_diagnostics(
    tmp_path: Path,
) -> None:
    """Make the residual diagnostic default exactly the implemented FEGA subset."""
    # Parse the minimal pipeline so the dataclass default is exercised through YAML.
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "default_resid.yaml"
    cfg_path.write_text(_base_yaml(reference_json, tmp_path / "out"))

    cfg = FEGAPipelineConfig.from_file(cfg_path)

    assert isinstance(cfg.phases.geometry_metrics, GeometryMetricsConfig)
    assert isinstance(
        cfg.phases.geometry_metrics.c_ray, GeometryMetricsCRayConfig
    )
    assert isinstance(cfg.phases.geometry_metrics.span, GeometryMetricsSpanConfig)
    assert isinstance(cfg.phases.geometry_metrics.resid, GeometryMetricsResidConfig)
    assert isinstance(
        cfg.phases.geometry_metrics.effective_rank,
        GeometryMetricsEffectiveRankConfig,
    )
    assert isinstance(cfg.phases.vmf, DirectionalMixtureFitConfig)
    assert cfg.phases.geometry_metrics.resid.k_values == [1, 2, 3, 4]


@pytest.mark.parametrize("k_values", ["[8]", "[1, 2, 8]"])
def test_geometry_metrics_residual_k_values_reject_unsupported_members(
    tmp_path: Path, k_values: str
) -> None:
    """Reject residual diagnostics outside the exact supported k subset."""
    # Write only the residual section so span and stability k=8 remain independent.
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "bad_resid.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out")
        + "phases:\n"
        + "  geometry_metrics:\n"
        + "    resid:\n"
        + f"      k_values: {k_values}\n"
    )

    with pytest.raises(ValueError, match="positive unique subset.*1, 2, 3, 4"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("  max_contexts: 10\n", "max_source_contexts"),
        ("  max_examples: 10\n", "max_source_examples"),
        ("  namespace_entity: induction\n", "entity_attribute_selection"),
        ("  namespace_attribute: rule_completion\n", "entity_attribute_selection"),
        ("  feature_set: bad\n", "feature_set"),
        (
            "  feature_set: explicit\n  explicit_feature_ids: []\n",
            "explicit_feature_ids",
        ),
        (
            "  feature_set: explicit\n  explicit_feature_ids: [1, 1]\n",
            "explicit_feature_ids",
        ),
        ("  materialize_only_selected: false\n", "materialize_only_selected"),
    ],
)
def test_induction_config_rejects_ambiguous_knobs(
    tmp_path: Path, body: str, match: str
) -> None:
    reference_json = tmp_path / "prontoqa_induction_rule.json"
    reference_json.write_text("{}")
    summary_json = tmp_path / "summary.json"
    _write_induction_summary(summary_json)
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        f"source_kind: induction\n"
        f"reference_json: {reference_json}\n"
        f"output_root: {tmp_path / 'out'}\n"
        "device: cpu\n"
        "entity_attribute_selection:\n"
        "  induction: ['rule_completion']\n"
        "induction:\n"
        f"  summary_json: {summary_json}\n" + body
    )

    with pytest.raises(ValueError, match=match):
        FEGAPipelineConfig.from_file(cfg_path)


def test_induction_config_requires_summary_json(tmp_path: Path) -> None:
    reference_json = tmp_path / "prontoqa_induction_rule.json"
    reference_json.write_text("{}")
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        f"source_kind: induction\n"
        f"reference_json: {reference_json}\n"
        f"output_root: {tmp_path / 'out'}\n"
        "device: cpu\n"
        "entity_attribute_selection:\n"
        "  induction: ['rule_completion']\n"
        "induction: {}\n"
    )

    with pytest.raises(ValueError, match="summary_json"):
        FEGAPipelineConfig.from_file(cfg_path)


def test_induction_config_requires_sae_uid_for_multi_sae_summary(
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "prontoqa_induction_rule.json"
    reference_json.write_text("{}")
    summary_json = tmp_path / "summary.json"
    _write_induction_summary(summary_json, multi_sae=True)
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        f"source_kind: induction\n"
        f"reference_json: {reference_json}\n"
        f"output_root: {tmp_path / 'out'}\n"
        "device: cpu\n"
        "entity_attribute_selection:\n"
        "  induction: ['rule_completion']\n"
        "induction:\n"
        f"  summary_json: {summary_json}\n"
    )

    with pytest.raises(ValueError, match="sae_uid"):
        FEGAPipelineConfig.from_file(cfg_path)


@pytest.mark.parametrize(
    ("phase_body", "match"),
    [
        (
            "  data_prep:\n    readouts: ['final_resid']\n"
            "  compute_effect:\n    enabled: true\n"
            "  geometry_metrics:\n    enabled: true\n    effect_space: final_resid\n",
            "gram_cache",
        ),
    ],
)
def test_enabled_downstream_effect_spaces_require_data_prep_readouts(
    tmp_path: Path, phase_body: str, match: str
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        _base_yaml(reference_json, tmp_path / "out") + "phases:\n" + phase_body
    )

    with pytest.raises(ValueError, match=match):
        FEGAPipelineConfig.from_file(cfg_path)


def test_readout_and_effect_space_defaults_reject_legacy_names(tmp_path: Path) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    default_path = tmp_path / "default.yaml"
    default_path.write_text(_base_yaml(reference_json, tmp_path / "out"))

    default = FEGAPipelineConfig.from_file(default_path)
    assert default.phases.data_prep.readouts == ["final_resid"]
    assert default.phases.geometry_metrics.effect_space == "final_resid"
    assert default.phases.vmf.effect_space == "pre_softcap_logits"
    assert default.phases.stability.effect_space == "final_resid"

    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(
        _base_yaml(reference_json, tmp_path / "legacy_out")
        + "phases:\n  data_prep:\n    readouts: ['logits']\n"
    )
    with pytest.raises(ValueError, match="Unsupported data_prep readout 'logits'"):
        FEGAPipelineConfig.from_file(legacy_path)

    dir_logits_path = tmp_path / "dir_logits.yaml"
    dir_logits_path.write_text(
        _base_yaml(reference_json, tmp_path / "dir_logits_out")
        + "phases:\n  geometry_metrics:\n    effect_space: logits\n"
    )
    with pytest.raises(ValueError, match="geometry_metrics.effect_space.*final_resid"):
        FEGAPipelineConfig.from_file(dir_logits_path)

    vmf_resid_path = tmp_path / "vmf_resid.yaml"
    vmf_resid_path.write_text(
        _base_yaml(reference_json, tmp_path / "vmf_resid_out")
        + "phases:\n  vmf:\n    effect_space: final_resid\n"
    )
    with pytest.raises(ValueError, match="vmf.effect_space.*pre_softcap_logits"):
        FEGAPipelineConfig.from_file(vmf_resid_path)
