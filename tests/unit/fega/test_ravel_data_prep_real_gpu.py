from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from fega.config_schema import FEGAPipelineConfig
from fega.core.data_prep import run_data_prep
from fega.core.resources import ModelResources, resolve_mdbm_path
from fega.core.utils import ChunkProcessor
from fega.paths import (
    data_prep_activations_dir,
    data_prep_pairs_path,
    data_prep_select_dir,
)

ROOT = Path(__file__).resolve().parents[3]
RAVEL_CONFIG = ROOT / "fega/config/ravel/city_country.yaml"
BASE_CONFIG = FEGAPipelineConfig.from_file(RAVEL_CONFIG)
BASE_PAIRS = data_prep_pairs_path(BASE_CONFIG)

GPU_REPEAT_ATOL = 0.0
GPU_REPEAT_RTOL = 0.0
GPU_BATCH_TENSOR_TOLERANCES = {
    "x": {"atol": 1.0, "rtol": 2.0e-2},
    "z": {"atol": 3.5e-1, "rtol": 3.0e-2},
    "final_resid": {"atol": 1.0, "rtol": 2.0e-2},
}
GPU_BATCH_SELECTED_Z_ATOL = 3.5e-1
GPU_BATCH_SELECTED_Z_RTOL = 3.0e-2
GPU_BATCH_THRESHOLD_ATOL = 1.25e-1


def _require_real_ravel_gpu() -> None:
    if os.environ.get("FEGA_RUN_REAL_RAVEL_GPU") != "1":
        pytest.skip("Set FEGA_RUN_REAL_RAVEL_GPU=1 for the real RAVEL/Gemma GPU smoke.")
    assert torch.cuda.is_available(), "CUDA is required for real RAVEL GPU smoke."
    assert RAVEL_CONFIG.exists(), f"Missing RAVEL config: {RAVEL_CONFIG}"
    assert BASE_PAIRS.exists(), f"Missing real RAVEL pairs cache: {BASE_PAIRS}"


def _write_reduced_config(tmp_path: Path, run_name: str, *, batch_size: int) -> Path:
    raw = yaml.safe_load(RAVEL_CONFIG.read_text())
    raw["reference_json"] = str(BASE_CONFIG.reference_json)
    raw["download_saes_dir"] = str(BASE_CONFIG.download_saes_dir)
    raw["mdbm_root"] = str(BASE_CONFIG.mdbm_root)
    raw["output_root"] = str(tmp_path / run_name)
    raw["device"] = "cuda:0"
    raw["reuse_model_across_phases"] = True
    phases = raw.setdefault("phases", {})
    phases["data_prep"] = {
        "enabled": True,
        "batch_size": batch_size,
        "save_chunk_size": 3,
        "single_file": False,
        "limit": 8,
        "tau_act": 0.0,
        "max_contexts": 8,
        "min_contexts": 0,
        "readouts": ["final_resid"],
        "gram_cache": False,
    }
    for phase in ("compute_effect", "geometry_metrics", "vmf", "stability", "geometry_reporting"):
        phases.setdefault(phase, {})
        phases[phase]["enabled"] = False
    config_path = tmp_path / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config_path


def _seed_pairs_cache(config: FEGAPipelineConfig) -> None:
    pairs_path = data_prep_pairs_path(config)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_PAIRS, pairs_path)


def _run_reduced_data_prep(
    config_path: Path, resources: ModelResources | None = None
) -> tuple[FEGAPipelineConfig, ModelResources]:
    config = FEGAPipelineConfig.from_file(config_path)
    _seed_pairs_cache(config)
    active_resources = resources or ModelResources(config)
    run_data_prep(config, active_resources)
    return config, active_resources


def _activation_rows(config: FEGAPipelineConfig) -> list[tuple[dict, dict[str, torch.Tensor]]]:
    manifest_path = data_prep_activations_dir(config) / "activations_manifest.json"
    rows: list[tuple[dict, dict[str, torch.Tensor]]] = []
    for tensors_path, meta_path in ChunkProcessor.stream(
        manifest_path, data_prep_activations_dir(config)
    ):
        payload = torch.load(tensors_path, map_location="cpu")
        meta_rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
        assert payload["z"].shape[0] == len(meta_rows)
        assert payload["x"].shape[0] == len(meta_rows)
        for offset, meta in enumerate(meta_rows):
            assert int(payload["index"][offset]) == int(meta["index"])
            assert meta["padded_target_position"] == (
                meta["pad_length"] + meta["unpadded_target_position"]
            )
            rows.append(
                (
                    meta,
                    {
                        "x": payload["x"][offset],
                        "z": payload["z"][offset],
                        "final_resid": payload["final_resid"][offset],
                    },
                )
            )
    return rows


def _load_contexts(config: FEGAPipelineConfig) -> dict[str, list[dict]]:
    contexts_path = data_prep_select_dir(config) / "feature_contexts.json"
    return json.loads(contexts_path.read_text())


def _load_summary(config: FEGAPipelineConfig) -> dict:
    summary_path = data_prep_select_dir(config) / "feature_contexts_summary.json"
    return json.loads(summary_path.read_text())


def _semantic_signature(config: FEGAPipelineConfig) -> dict:
    rows = _activation_rows(config)
    contexts = _load_contexts(config)
    summary = _load_summary(config)
    row_by_index = {int(row["index"]): row for row, _ in rows}
    tensors_by_index = {int(row["index"]): tensors for row, tensors in rows}
    assert len(row_by_index) == len(rows)

    selected: list[tuple] = []
    for feature_id, ctxs in contexts.items():
        for ctx in ctxs:
            fid = int(feature_id)
            meta = row_by_index[int(ctx["index"])]
            assert ctx["pair_index"] == meta["pair_index"]
            assert ctx["prompt"] == meta["prompt"]
            assert ctx["entity_label"] == meta["entity_label"]
            assert ctx["attribute_type"] == meta["attribute_type"]
            assert ctx["attribute_label"] == meta["attribute_label"]
            assert float(ctx["z"]) == pytest.approx(
                float(tensors_by_index[int(ctx["index"])]["z"][fid])
            )
            selected.append(
                (
                    fid,
                    int(ctx["index"]),
                    int(ctx["pair_index"]),
                    ctx["prompt"],
                    ctx["entity_label"],
                    ctx["attribute_type"],
                    ctx["attribute_label"],
                    float(ctx["z"]),
                )
            )

    return {
        "rows": [
            (
                int(row["index"]),
                int(row["pair_index"]),
                int(row["original_index"]),
                row["prompt"],
                row["entity_label"],
                row["attribute_type"],
                row["attribute_label"],
                row["raw_target_position"],
                int(row["unpadded_target_position"]),
            )
            for row, _ in rows
        ],
        "selected": sorted(selected),
        "target_features": sorted(int(fid) for fid in summary["feature_stats"]),
        "features_with_contexts": int(summary["features_with_contexts"]),
        "total_features": int(summary["total_features"]),
        "tensors": {
            index: {
                name: tensor.float()
                for name, tensor in tensors.items()
            }
            for index, tensors in tensors_by_index.items()
        },
    }


def _assert_gpu_signatures_close(
    left: dict,
    right: dict,
    *,
    compare_selected: bool,
) -> None:
    assert left["rows"] == right["rows"]
    assert left["target_features"] == right["target_features"]
    assert left["features_with_contexts"] == right["features_with_contexts"]
    assert left["total_features"] == right["total_features"]
    if not compare_selected:
        return
    assert left["tensors"].keys() == right["tensors"].keys()
    for index in sorted(left["tensors"]):
        for key in ("x", "z", "final_resid"):
            torch.testing.assert_close(
                left["tensors"][index][key],
                right["tensors"][index][key],
                atol=GPU_REPEAT_ATOL,
                rtol=GPU_REPEAT_RTOL,
                msg=f"same-config tensor mismatch for index={index}, key={key}",
            )
    assert len(left["selected"]) == len(right["selected"])
    for l_ctx, r_ctx in zip(left["selected"], right["selected"], strict=True):
        assert l_ctx[:-1] == r_ctx[:-1]
        assert l_ctx[-1] == pytest.approx(
            r_ctx[-1], abs=GPU_REPEAT_ATOL, rel=GPU_REPEAT_RTOL
        )


def _assert_batch_signatures_are_positionally_aligned(left: dict, right: dict) -> None:
    assert left["rows"] == right["rows"]
    assert left["target_features"] == right["target_features"]
    assert left["total_features"] == right["total_features"]

    left_by_identity = {ctx[:-1]: ctx[-1] for ctx in left["selected"]}
    right_by_identity = {ctx[:-1]: ctx[-1] for ctx in right["selected"]}
    for identity in sorted(set(left_by_identity) & set(right_by_identity)):
        index = identity[1]
        for key in ("x", "z", "final_resid"):
            tolerances = GPU_BATCH_TENSOR_TOLERANCES[key]
            torch.testing.assert_close(
                left["tensors"][index][key],
                right["tensors"][index][key],
                atol=tolerances["atol"],
                rtol=tolerances["rtol"],
                msg=f"batch-size tensor mismatch for identity={identity}, key={key}",
            )
        assert left_by_identity[identity] == pytest.approx(
            right_by_identity[identity],
            abs=GPU_BATCH_SELECTED_Z_ATOL,
            rel=GPU_BATCH_SELECTED_Z_RTOL,
        )
    drifting = sorted(set(left_by_identity) ^ set(right_by_identity))
    for identity in drifting:
        observed = [
            abs(value)
            for value in (
                left_by_identity.get(identity),
                right_by_identity.get(identity),
            )
            if value is not None
        ]
        assert observed
        assert max(observed) <= GPU_BATCH_THRESHOLD_ATOL


@pytest.mark.real_ravel_gpu
def test_real_ravel_gpu_data_prep_repeat_and_batch_signatures(tmp_path: Path) -> None:
    _require_real_ravel_gpu()
    repeat_a_path = _write_reduced_config(tmp_path, "repeat_a", batch_size=2)
    repeat_b_path = _write_reduced_config(tmp_path, "repeat_b", batch_size=2)
    batch_one_path = _write_reduced_config(tmp_path, "batch_one", batch_size=1)
    batch_four_path = _write_reduced_config(tmp_path, "batch_four", batch_size=4)

    repeat_a, resources = _run_reduced_data_prep(repeat_a_path)
    repeat_b, resources = _run_reduced_data_prep(repeat_b_path, resources)
    batch_one, resources = _run_reduced_data_prep(batch_one_path, resources)
    batch_four, _ = _run_reduced_data_prep(batch_four_path, resources)

    weight_path = resolve_mdbm_path(repeat_a, "city", "Country")
    assert weight_path.exists()

    repeat_a_sig = _semantic_signature(repeat_a)
    repeat_b_sig = _semantic_signature(repeat_b)
    batch_one_sig = _semantic_signature(batch_one)
    batch_four_sig = _semantic_signature(batch_four)

    assert repeat_a_sig["features_with_contexts"] > 0
    _assert_gpu_signatures_close(
        repeat_a_sig, repeat_b_sig, compare_selected=True
    )
    _assert_batch_signatures_are_positionally_aligned(batch_one_sig, batch_four_sig)
