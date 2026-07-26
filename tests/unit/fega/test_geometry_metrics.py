from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    gram_fingerprint,
)
from fega.core.geometry_metrics.metrics import (
    c_ray_fast_final_resid,
    c_ray_pairwise_final_resid,
    centered_residual_spectrum_final_resid,
    effective_rank_from_spectrum,
    span_spectrum_final_resid,
)
from fega.core.geometry_metrics.runner import run_geometry_metrics
from fega.paths import (
    effect_summary_path,
    effect_tensors_manifest_path,
    geometry_metrics_scores_path,
    gram_cache_meta_path,
)


def _write_reference_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eval_config": {
                    "model_name": "gpt2",
                    "llm_dtype": "float32",
                    "entity_attribute_selection": {"city": ["Country"]},
                }
            }
        )
    )


def _write_config(
    tmp_path: Path,
    *,
    effect_space: str,
    method: str = "pairwise",
    cosine_enabled: bool = True,
    store_r2: bool = True,
    span_enabled: bool | None = None,
    k_values: list[int] | None = None,
    resid_enabled: bool | None = None,
    resid_k_values: list[int] | None = None,
    effective_rank_enabled: bool | None = None,
    effective_rank_eps: float | None = None,
) -> FEGAPipelineConfig:
    reference_json = tmp_path / "refs" / "ref.json"
    _write_reference_json(reference_json)
    readouts = ["final_resid"]
    span_lines = []
    if span_enabled is not None or k_values is not None:
        span_lines.append("    span:")
        if span_enabled is not None:
            span_lines.append(f"      enabled: {str(span_enabled).lower()}")
        if k_values is not None:
            span_lines.append(f"      k_values: {k_values}")
    resid_lines = []
    if resid_enabled is not None or resid_k_values is not None:
        resid_lines.append("    resid:")
        if resid_enabled is not None:
            resid_lines.append(f"      enabled: {str(resid_enabled).lower()}")
        if resid_k_values is not None:
            resid_lines.append(f"      k_values: {resid_k_values}")
    effective_rank_lines = []
    if effective_rank_enabled is not None or effective_rank_eps is not None:
        effective_rank_lines.append("    effective_rank:")
        if effective_rank_enabled is not None:
            effective_rank_lines.append(
                f"      enabled: {str(effective_rank_enabled).lower()}"
            )
        if effective_rank_eps is not None:
            effective_rank_lines.append(f"      eps: {effective_rank_eps}")
    cfg_path = tmp_path / f"cfg_{effect_space}.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"reference_json: {reference_json}",
                f"output_root: {tmp_path / 'out'}",
                "device: cpu",
                "entity_attribute_selection:",
                "  city: ['Country']",
                "phases:",
                "  data_prep:",
                f"    readouts: {readouts}",
                "  geometry_metrics:",
                "    enabled: true",
                f"    effect_space: {effect_space}",
                "    c_ray:",
                f"      enabled: {str(cosine_enabled).lower()}",
                f"      method: {method}",
                f"      store_r2: {str(store_r2).lower()}",
                "      eps: 1.0e-12",
                *span_lines,
                *resid_lines,
                *effective_rank_lines,
            ]
        )
        + "\n"
    )
    return FEGAPipelineConfig.from_file(cfg_path)


class _NoModelResources:
    def __init__(self) -> None:
        self._json_cache: dict[str, object] = {}
        self.json_cache_hits = 0
        self.json_cache_misses = 0
        self.json_cache_writes = 0

    def get_cached_json(self, path: Path):
        key = str(Path(path).resolve())
        if key in self._json_cache:
            self.json_cache_hits += 1
            return self._json_cache[key]
        self.json_cache_misses += 1
        return None

    def cache_json(self, path: Path, payload) -> None:
        self._json_cache[str(Path(path).resolve())] = payload
        self.json_cache_writes += 1

    def get_model_and_sae(self):  # pragma: no cover - failure path only
        raise AssertionError("geometry_metrics must not load model or SAE resources")


def _write_effect_artifacts(
    config: FEGAPipelineConfig,
    effect_space: str,
    *,
    rows: torch.Tensor,
    value_key: str,
    gram_path: Path | None = None,
    gram: torch.Tensor | None = None,
    skipped_feature: bool = False,
) -> Path:
    artifact_dir = effect_tensors_manifest_path(config, effect_space).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shard_path = artifact_dir / "effect_tensors_00000.pt"
    payload = {
        "feature_ids": torch.tensor([1], dtype=torch.long),
        "row_offsets": torch.tensor([0, int(rows.shape[0])], dtype=torch.long),
        value_key: rows.to(dtype=torch.float32),
    }
    if effect_space == "final_resid":
        identities = [
            {
                "attribute_label": "Country",
                "pair_role": "cause_base",
                "pair_index": index,
            }
            for index in range(int(rows.shape[0]))
        ]
        payload.setdefault("delta", rows.to(dtype=torch.float32))
        payload.setdefault("magnitude", torch.ones(rows.shape[0], dtype=torch.float32))
        payload.update(
            {
                "attribute_labels": ["Country"] * int(rows.shape[0]),
                "pair_roles": ["cause_base"] * int(rows.shape[0]),
                "pair_indices": torch.arange(rows.shape[0], dtype=torch.long),
                "candidate_identity": [identities],
                "retained_mask": [[True] * int(rows.shape[0])],
            }
        )
    torch.save(payload, shard_path)

    per_feature = {
        "1": {
            "feature_id": 1,
            "usable_effects": int(rows.shape[0]),
            "tensor_shard": "effect_tensors_00000.pt",
            "row_start": 0,
            "row_end": int(rows.shape[0]),
            "candidate_identity": (
                identities if effect_space == "final_resid" else []
            ),
            "retained_mask": (
                [True] * int(rows.shape[0])
                if effect_space == "final_resid"
                else []
            ),
        }
    }
    skipped = []
    if skipped_feature:
        per_feature["2"] = {
            "feature_id": 2,
            "usable_effects": 0,
            "skipped_reason": "below_min_coverage_2",
            "tensor_shard": None,
            "row_start": None,
            "row_end": None,
            "candidate_identity": [],
            "retained_mask": [],
        }
        skipped.append({"feature_id": 2, "skipped_reason": "below_min_coverage_2"})
    summary = {
        "summary": {
            "readout_name": effect_space,
            "features_total": len(per_feature),
            "features_with_effects": 1,
            "features_skipped": len(skipped),
            "total_effect_rows": int(rows.shape[0]),
            "shard_count": 1,
        },
        "per_feature": per_feature,
        "skipped_features": skipped,
    }
    manifest = {
        "schema_version": 1,
        "readout_name": effect_space,
        "effect_space": effect_space,
        "metric_space": "logit_l2" if effect_space == "logits" else "residual_gram",
        "dtype": "float32",
        "inputs": {},
        "outputs": {
            "effect_summary_path": str(effect_summary_path(config, effect_space))
        },
        "counts": {
            "features_total": len(per_feature),
            "features_with_effects": 1,
            "features_skipped": len(skipped),
            "total_effect_rows": int(rows.shape[0]),
            "shard_count": 1,
        },
        "shards": [
            {
                "shard": 0,
                "path": str(shard_path),
                "rows": int(rows.shape[0]),
                "feature_ids": [1],
                "row_start": 0,
                "row_end": int(rows.shape[0]),
            }
        ],
    }
    if gram_path is not None:
        manifest["inputs"]["gram_path"] = str(gram_path)
    if effect_space == "final_resid":
        if gram is None or gram_path is None:
            raise ValueError("final_resid fixture requires Gram tensor and path")
        gram_metadata = {
            "checkpoint_identity": "test-model",
            "readout_name": "final_resid",
            "hidden_width": int(gram.shape[0]),
            "gram_dtype": "float32",
            "construction_recipe": GRAM_CONSTRUCTION_RECIPE,
            "unembedding_fingerprint": "test-unembedding",
            "unembedding_dtype": "torch.float32",
            "unembedding_shape": [int(gram.shape[0]), int(gram.shape[0])],
            "gram_shape": list(gram.shape),
            "gram_sha256": gram_fingerprint(gram),
        }
        meta_path = gram_cache_meta_path(config)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(gram_metadata))
        manifest["inputs"]["gram_meta_path"] = str(meta_path)
        manifest["vector_size"] = int(gram.shape[0])
        manifest["gram_metadata"] = gram_metadata
        summary["gram_metadata"] = gram_metadata
    effect_summary_path(config, effect_space).write_text(json.dumps(summary))
    effect_tensors_manifest_path(config, effect_space).write_text(json.dumps(manifest))
    return shard_path


def test_final_resid_pairwise_fast_r2_and_span_use_residual_gram() -> None:
    gram = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    directions = torch.tensor(
        [[1.0 / (2.0**0.5), 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    cross = 0.5 / (2.0**0.5)
    s_norm_sq = 2.0 + 2.0 * cross

    pairwise = c_ray_pairwise_final_resid(directions, gram, eps=1.0e-12)
    fast = c_ray_fast_final_resid(directions, gram, eps=1.0e-12)
    span = span_spectrum_final_resid(directions, gram, k_values=[1, 2, 8], eps=1.0e-12)

    assert pairwise.c_ray == pytest.approx(cross)
    assert fast.c_ray == pytest.approx(cross)
    assert fast.r2 == pytest.approx(s_norm_sq / 4.0)
    assert span.eigenvalues == pytest.approx([1.0 + cross, 1.0 - cross])
    assert span.s_span[1] == pytest.approx((1.0 + cross) / 2.0)
    assert span.s_span[2] == pytest.approx(1.0)
    assert span.s_span[8] == pytest.approx(1.0)
    assert span.u_span[1] == pytest.approx((1.0 + cross) / 2.0)
    assert span.u_span[2] == pytest.approx((1.0 - cross) / 2.0)
    assert span.u_span[8] is None
    assert span.d_span[1] == pytest.approx(
        (1.0 - cross) / (1.0 + cross + 1.0e-12)
    )
    assert span.d_span[2] is None
    assert span.d_span[8] is None


def test_pairwise_c_ray_preserves_cancellation_sensitive_float64_gram() -> None:
    """Retain exact-coordinate directions requiring float64 Gram cancellation."""
    # Use identical exact [1, -1] float32 directions with unit float64 Gram norm.
    directions = torch.tensor(
        [[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32
    )
    gram = torch.tensor(
        [[1.0e12, 1.0e12 - 0.5], [1.0e12 - 0.5, 1.0e12]],
        dtype=torch.float64,
    )

    result = c_ray_pairwise_final_resid(directions, gram, eps=1.0e-12)

    assert result.n_valid == 2
    assert result.c_ray == pytest.approx(1.0)


def test_span_spectrum_component_share_and_eigengap_null_zero_lambda() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    span = span_spectrum_final_resid(directions, gram, k_values=[1, 2], eps=1.0e-12)

    assert span.eigenvalues == pytest.approx([2.0, 0.0])
    assert span.u_span[1] == pytest.approx(1.0)
    assert span.u_span[2] == pytest.approx(0.0)
    assert span.d_span[1] == pytest.approx(0.0)
    assert span.d_span[2] is None


def test_span_spectrum_zero_pads_complete_dual_spectrum_to_ambient_width() -> None:
    """Distinguish known zero tails from indices beyond the ambient direction width."""
    # Two orthogonal rows give two unit eigenvalues and seven known zero ambient tails.
    gram = torch.eye(9, dtype=torch.float32)
    directions = torch.eye(9, dtype=torch.float32)[:2]

    span = span_spectrum_final_resid(
        directions, gram, k_values=[2, 3, 4, 8, 9], eps=1.0e-12
    )

    assert span.eigenvalues == pytest.approx([1.0, 1.0])
    assert span.d_span[2] == pytest.approx(0.0)
    for k in [3, 4, 8]:
        assert span.u_span[k] == pytest.approx(0.0)
        assert span.d_span[k] == pytest.approx(0.0)
    assert span.u_span[9] == pytest.approx(0.0)
    assert span.d_span[9] is None


def test_span_spectrum_symmetrizes_psd_kernel_before_negative_guard() -> None:
    generator = torch.Generator().manual_seed(4)
    gram = torch.diag(torch.logspace(0, 6, 16, dtype=torch.float32))
    directions = torch.randn(256, 16, generator=generator, dtype=torch.float32)
    magnitudes = torch.sqrt(
        torch.sum((directions @ gram) * directions, dim=1).clamp_min(1.0e-30)
    )
    directions = directions / magnitudes.unsqueeze(1)

    span = span_spectrum_final_resid(
        directions,
        gram,
        k_values=[1, 2, 8],
        eps=1.0e-12,
    )

    assert span.n_valid == 256
    assert min(span.eigenvalues) >= 0.0
    assert span.s_span[1] is not None
    assert span.u_span[1] is not None
    assert span.d_span[1] is not None


def test_span_spectrum_materially_negative_eigenvalue_guard() -> None:
    directions = torch.eye(2, dtype=torch.float32)
    bad_gram = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="materially negative eigenvalue"):
        span_spectrum_final_resid(directions, bad_gram, k_values=[1], eps=1.0e-12)


def test_span_spectrum_b_axis_uses_strict_signs_and_valid_denominator() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )

    span = span_spectrum_final_resid(directions, gram, k_values=[1], eps=1.0e-12)
    flipped = span_spectrum_final_resid(-directions, gram, k_values=[1], eps=1.0e-12)

    assert span.eigenvalues == pytest.approx([3.0, 1.0, 0.0, 0.0], abs=1.0e-6)
    assert span.b_axis == pytest.approx(1.0 / 4.0)
    assert flipped.b_axis == pytest.approx(span.b_axis)


def test_span_spectrum_no_valid_rows_returns_null_diagnostics() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32)

    span = span_spectrum_final_resid(directions, gram, k_values=[1, 8], eps=1.0e-12)

    assert span.n_valid == 0
    assert span.eigenvalues == []
    assert span.s_span == {1: None, 8: None}
    assert span.u_span == {1: None, 8: None}
    assert span.d_span == {1: None, 8: None}
    assert span.b_axis is None


def test_effective_rank_from_spectrum_uses_source_formulas() -> None:
    # Keep the positive total below eps so the visible-eps normalization is exercised.
    spectrum = [1.0e-13, 1.0e-13]
    eps = 1.0e-12
    weights = [value / (sum(spectrum) + eps) for value in spectrum]
    expected_ent = math.exp(
        -sum(p_value * math.log(p_value + eps) for p_value in weights)
    )
    expected_pr = 1.0 / sum(p_value * p_value for p_value in weights)

    result = effective_rank_from_spectrum(spectrum, eps=eps)

    assert result.r_ent == pytest.approx(expected_ent)
    assert result.r_pr == pytest.approx(expected_pr)


def test_effective_rank_empty_and_zero_spectra() -> None:
    empty = effective_rank_from_spectrum([], eps=1.0e-12)
    single_zero = effective_rank_from_spectrum([0.0], eps=1.0e-12)
    two_zeros = effective_rank_from_spectrum([0.0, 0.0], eps=1.0e-12)

    assert empty.r_ent is None
    assert empty.r_pr is None
    assert single_zero.r_ent is None
    assert single_zero.r_pr is None
    assert two_zeros.r_ent is None
    assert two_zeros.r_pr is None


def test_effective_rank_extreme_positive_spectrum_has_defined_ranks() -> None:
    """Keep a positive subnormal weight evaluable when its square underflows."""
    # The literal probability is positive, but its squared float representation is zero.
    result = effective_rank_from_spectrum(
        [float.fromhex("0x0.0000000000001p-1022")], eps=1.0
    )

    assert result.r_ent == pytest.approx(1.0)
    assert result.r_pr == math.inf


def test_centered_residual_spectrum_matches_source_hkh_fixture() -> None:
    gram = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)

    result = centered_residual_spectrum_final_resid(
        directions, gram, k_values=[1, 2, 3, 4], eps=1.0e-12
    )

    valid64 = directions.to(dtype=torch.float64)
    gram64 = gram.to(dtype=torch.float64)
    k_matrix = valid64 @ gram64 @ valid64.T
    n = int(k_matrix.shape[0])
    ones_n = torch.ones((n, 1), dtype=torch.float64)
    h_matrix = torch.eye(n, dtype=torch.float64) - (ones_n @ ones_n.T) / float(n)
    k_ctr = h_matrix @ k_matrix @ h_matrix
    k_ctr = (k_ctr + k_ctr.T) / 2.0
    expected_eigenvalues = torch.sort(
        torch.linalg.eigvalsh(k_ctr), descending=True
    ).values
    expected_eigenvalues = torch.clamp(expected_eigenvalues, min=0.0)
    expected_total = float(expected_eigenvalues.sum().item())
    expected_denom = expected_total + 1.0e-12

    assert result.eigenvalues == pytest.approx(
        [float(v) for v in expected_eigenvalues.tolist()]
    )
    assert result.s_res[1] == pytest.approx(
        float(expected_eigenvalues[:1].sum().item()) / expected_denom
    )
    assert result.s_res[2] == pytest.approx(
        float(expected_eigenvalues[:2].sum().item()) / expected_denom
    )
    assert result.s_res[3] == pytest.approx(1.0)
    assert result.s_res[4] == pytest.approx(1.0)
    assert result.e_res == pytest.approx(
        float(torch.trace(k_ctr).item())
        / (float(torch.trace(k_matrix).item()) + 1.0e-12)
    )


def test_centered_residual_spectrum_filters_invalid_rows() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [float("nan"), 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )

    result = centered_residual_spectrum_final_resid(
        directions, gram, k_values=[1, 2], eps=1.0e-12
    )

    assert result.n_total == 4
    assert result.n_valid == 2
    assert result.skipped_nonfinite == 1
    assert result.skipped_zero_norm == 1
    assert result.eigenvalues == pytest.approx([1.0, 0.0])
    assert result.s_res[1] == pytest.approx(1.0)
    assert result.s_res[2] == pytest.approx(1.0)
    assert result.e_res == pytest.approx(0.5)


def test_centered_residual_spectrum_single_valid_direction_is_zero() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor(
        [[1.0, 0.0], [float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32
    )

    result = centered_residual_spectrum_final_resid(
        directions, gram, k_values=[1, 2, 3, 4], eps=1.0e-12
    )

    assert result.n_valid == 1
    assert result.eigenvalues == pytest.approx([0.0])
    assert result.e_res == pytest.approx(0.0)
    assert result.s_res == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_centered_residual_spectrum_no_valid_rows_returns_none_scores() -> None:
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor([[float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32)

    result = centered_residual_spectrum_final_resid(
        directions, gram, k_values=[1, 2], eps=1.0e-12
    )

    assert result.n_valid == 0
    assert result.eigenvalues == []
    assert result.e_res is None
    assert result.s_res == {1: None, 2: None}


@pytest.mark.parametrize("has_valid_row", [True, False])
def test_centered_residual_spectrum_rejects_unsupported_k_at_api_boundary(
    has_valid_row: bool,
) -> None:
    """Reject unsupported residual cutoffs before valid-row availability matters."""
    # Exercise both an ordinary call and the formerly early-returning no-valid path.
    gram = torch.eye(2, dtype=torch.float32)
    directions = torch.tensor(
        [[1.0, 0.0]] if has_valid_row else [[float("nan"), 0.0]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match=r"only \{1, 2, 3, 4\}.*\[8\]"):
        centered_residual_spectrum_final_resid(
            directions, gram, k_values=[1, 8], eps=1.0e-12
        )


def test_centered_residual_spectrum_negative_eigenvalue_guard() -> None:
    directions = torch.eye(2, dtype=torch.float32)
    bad_gram = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="materially negative eigenvalue"):
        centered_residual_spectrum_final_resid(
            directions, bad_gram, k_values=[1], eps=1.0e-12
        )

    tiny_negative_gram = torch.tensor(
        [[1.0, 1.0 + 1.0e-7], [1.0 + 1.0e-7, 1.0]], dtype=torch.float32
    )
    result = centered_residual_spectrum_final_resid(
        directions, tiny_negative_gram, k_values=[1, 2], eps=1.0e-12
    )
    assert result.eigenvalues == pytest.approx([0.0, 0.0], abs=1.0e-6)
    assert result.s_res == {1: 0.0, 2: 0.0}


def test_run_geometry_metrics_rejects_logits_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="geometry_metrics.effect_space.*final_resid"):
        _write_config(tmp_path, effect_space="logits")


def test_run_geometry_metrics_final_resid_writes_effective_rank_from_spectra(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path,
        effect_space="final_resid",
        method="fast_formula",
        k_values=[1, 2],
        resid_enabled=True,
        resid_k_values=[1, 2],
        effective_rank_enabled=True,
        effective_rank_eps=1.0e-12,
    )
    gram_path = tmp_path / "missing_because_cache_is_used.pt"
    gram = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    _write_effect_artifacts(
        config,
        "final_resid",
        rows=rows,
        value_key="direction",
        gram_path=gram_path,
        gram=gram,
        skipped_feature=True,
    )
    resources = _NoModelResources()
    resources._compute_effect_gram_cache = {str(gram_path.resolve()): gram}

    run_geometry_metrics(config, resources)
    first = json.loads(geometry_metrics_scores_path(config, "final_resid").read_text())
    run_geometry_metrics(config, resources)
    second = json.loads(geometry_metrics_scores_path(config, "final_resid").read_text())

    assert first["summary"]["effective_rank"] == {
        "enabled": True,
        "eps": 1.0e-12,
    }
    assert first["canonical_source_fingerprint"]["components"][
        "gram_readout_metadata"
    ]["gram_sha256"] == gram_fingerprint(gram)
    feature = first["per_feature"]["1"]
    expected_span = effective_rank_from_spectrum(
        feature["span_eigenvalues"], eps=1.0e-12
    )
    expected_ctr = effective_rank_from_spectrum(
        feature["resid_eigenvalues"], eps=1.0e-12
    )
    assert feature["r_span_ent"] == pytest.approx(expected_span.r_ent)
    assert feature["r_span_pr"] == pytest.approx(expected_span.r_pr)
    assert feature["r_ctr_ent"] == pytest.approx(expected_ctr.r_ent)
    assert feature["r_ctr_pr"] == pytest.approx(expected_ctr.r_pr)
    assert "s_res_8" not in feature
    assert not any(key.startswith("r_raw") for key in feature)
    assert "r_raw_ent" not in json.dumps(first)
    assert "r_raw_pr" not in json.dumps(first)

    skipped_feature = first["per_feature"]["2"]
    assert skipped_feature["skipped_reason"] == "below_min_coverage_2"
    assert "b_axis" not in skipped_feature
    assert not any(key.startswith("u_span_") for key in skipped_feature)
    assert not any(key.startswith("d_span_") for key in skipped_feature)
    assert skipped_feature["r_span_ent"] is None
    assert skipped_feature["r_span_pr"] is None
    assert skipped_feature["r_ctr_ent"] is None
    assert skipped_feature["r_ctr_pr"] is None

    effective_rank_keys = {
        "r_span_ent",
        "r_span_pr",
        "r_ctr_ent",
        "r_ctr_pr",
    }
    assert first["summary"]["effective_rank"] == second["summary"]["effective_rank"]
    assert {key: first["per_feature"]["1"][key] for key in effective_rank_keys} == {
        key: second["per_feature"]["1"][key] for key in effective_rank_keys
    }


def test_run_geometry_metrics_rejects_mutated_residual_k_values(tmp_path: Path) -> None:
    """Keep the runner boundary safe when callers bypass config-file validation."""
    # Parse a valid config, then simulate a direct caller mutating the dataclass.
    config = _write_config(
        tmp_path,
        effect_space="final_resid",
        resid_enabled=True,
        resid_k_values=[1, 2, 3, 4],
    )
    config.phases.geometry_metrics.resid.k_values = [1, 2, 3, 4, 8]

    with pytest.raises(ValueError, match="resid.k_values"):
        run_geometry_metrics(config)
