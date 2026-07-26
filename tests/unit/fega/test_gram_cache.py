from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.compute_effect.runner import (
    _gram_to_device,
    _validate_gram,
    _validate_gram_readout,
)
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    canonical_unembedding,
    gram_fingerprint,
    unembedding_fingerprint,
    write_gram_cache,
)
from fega.core.geometry_metrics.artifacts import (
    GeometryMetricsInputs,
    resolve_final_resid_gram,
)
from fega.paths import gram_cache_meta_path, gram_cache_tensor_path


class _FakeOutputEmbeddings:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


class _FakeModel:
    def __init__(self, weight: torch.Tensor) -> None:
        self._output_embeddings = _FakeOutputEmbeddings(weight)
        self.config = type("Config", (), {"name_or_path": "fake-model"})()

    def get_output_embeddings(self):
        return self._output_embeddings


def _write_reference_json(path: Path) -> None:
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
    cfg_path: Path,
    reference_json: Path,
    output_root: Path,
    *,
    gram_cache: bool = True,
    gram_cache_dtype: str = "float32",
) -> None:
    cfg_path.write_text(
        "\n".join(
            [
                f"reference_json: {reference_json}",
                f"output_root: {output_root}",
                "device: cpu",
                "entity_attribute_selection:",
                "  city: ['Country']",
                "phases:",
                "  data_prep:",
                f"    gram_cache: {str(gram_cache).lower()}",
                "    readouts: ['final_resid']",
                f"    gram_cache_dtype: {gram_cache_dtype}",
            ]
        )
        + "\n"
    )


def test_canonical_unembedding_and_gram_cache_write(tmp_path: Path) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model = _FakeModel(weight)

    w_u = canonical_unembedding(model)
    assert w_u.shape == (3, 2)

    meta_path = write_gram_cache(model, config, final_resid_width=2)
    assert meta_path == gram_cache_meta_path(config)
    gram = torch.load(gram_cache_tensor_path(config), map_location="cpu")
    expected = weight.T @ weight
    assert torch.equal(gram, expected)
    assert gram.shape == (2, 2)

    meta = json.loads(gram_cache_meta_path(config).read_text())
    assert meta["model_name"] == "fake-model"
    assert meta["readout_name"] == "final_resid"
    assert meta["checkpoint_identity"] == "fake-model"
    assert meta["hidden_width"] == 2
    assert meta["gram_dtype"] == "float32"
    assert meta["construction_recipe"] == GRAM_CONSTRUCTION_RECIPE
    assert meta["unembedding_dtype"] == "torch.float32"
    assert meta["unembedding_fingerprint"] == unembedding_fingerprint(weight)
    assert meta["unembedding_shape"] == [3, 2]
    assert meta["gram_shape"] == [2, 2]
    assert meta["gram_sha256"] == gram_fingerprint(gram)
    assert meta["dtype"] == "float32"
    assert meta["compute_device"] == "cpu"
    assert meta["tensor_path"] == str(gram_cache_tensor_path(config))

    first_fingerprint = meta["unembedding_fingerprint"]
    write_gram_cache(model, config, final_resid_width=2)
    second_meta = json.loads(gram_cache_meta_path(config).read_text())
    assert second_meta["unembedding_fingerprint"] == first_fingerprint
    _validate_gram(gram, second_meta)
    _validate_gram_readout(model, second_meta)


def test_float64_gram_cache_preserves_cancellation_sensitive_values(
    tmp_path: Path,
) -> None:
    """Preserve configured float64 precision through every Gram load boundary."""
    # Build a float32 readout whose off-diagonal is lost if multiplication is float32.
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(
        cfg_path,
        reference_json,
        tmp_path / "out",
        gram_cache_dtype="float64",
    )
    config = FEGAPipelineConfig.from_file(cfg_path)
    assert config.phases.data_prep.gram_cache_dtype == "float64"
    weight = torch.tensor(
        [[1.0e8, 1.0], [1.0, 1.0], [1.0e8, -1.0]], dtype=torch.float32
    )
    model = _FakeModel(weight)

    write_gram_cache(model, config, final_resid_width=2)
    gram = torch.load(gram_cache_tensor_path(config), map_location="cpu")
    expected = weight.double().T @ weight.double()
    assert gram.dtype == torch.float64
    assert torch.equal(gram, expected)
    assert gram[0, 1].item() == 1.0

    meta = json.loads(gram_cache_meta_path(config).read_text())
    assert meta["gram_dtype"] == "float64"
    _validate_gram(gram, meta)
    device_gram = _gram_to_device(gram, "cpu")
    assert device_gram.dtype == torch.float64

    inputs = GeometryMetricsInputs(
        effect_space="final_resid",
        artifact_dir=gram_cache_tensor_path(config).parent,
        manifest_path=tmp_path / "manifest.json",
        summary_path=tmp_path / "summary.json",
        manifest={
            "inputs": {"gram_path": str(gram_cache_tensor_path(config))},
            "gram_metadata": meta,
        },
        summary={"gram_metadata": meta},
    )
    resolved = resolve_final_resid_gram(inputs)
    assert resolved.dtype == torch.float64
    assert torch.equal(resolved, expected)


def test_gram_cache_enforces_final_resid_width(tmp_path: Path) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    model = _FakeModel(torch.ones(4, 3))

    with pytest.raises(ValueError, match="final_resid"):
        write_gram_cache(model, config, final_resid_width=2)


def test_gram_metadata_rejects_missing_field_and_readout_digest_mismatch(
    tmp_path: Path,
) -> None:
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    model = _FakeModel(torch.arange(6, dtype=torch.float32).reshape(3, 2))
    write_gram_cache(model, config, final_resid_width=2)
    gram = torch.load(gram_cache_tensor_path(config), map_location="cpu")
    meta = json.loads(gram_cache_meta_path(config).read_text())

    missing = dict(meta)
    missing.pop("checkpoint_identity")
    with pytest.raises(ValueError, match="missing required"):
        _validate_gram(gram, missing)

    mismatched = dict(meta)
    mismatched["unembedding_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _validate_gram_readout(model, mismatched)

    corrupted = gram.clone()
    corrupted[0, 0] += 1.0
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_gram(corrupted, meta)

    torch.save(corrupted, gram_cache_tensor_path(config))
    inputs = GeometryMetricsInputs(
        effect_space="final_resid",
        artifact_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        summary_path=tmp_path / "summary.json",
        manifest={
            "inputs": {"gram_path": str(gram_cache_tensor_path(config))},
            "gram_metadata": meta,
        },
        summary={},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_final_resid_gram(inputs)
