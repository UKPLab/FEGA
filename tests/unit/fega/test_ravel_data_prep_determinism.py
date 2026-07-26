from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sae_bench.evals.ravel.instance import Prompt

from fega.config_schema import FEGAPipelineConfig
from fega.core.config import FEGAConfig
from fega.core.data_prep import collection, selection
from fega.core.positioning import POSITIONING_SCHEMA_VERSION
from fega.core.utils import ChunkProcessor
from fega.paths import (
    data_prep_activations_dir,
    data_prep_collect_dir,
    data_prep_select_dir,
)


class _ForwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.layer = torch.nn.Identity()
        self.output_embeddings = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.output_embeddings.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
        self.device = torch.device("cpu")

    def get_output_embeddings(self) -> torch.nn.Module:
        """Return the test LM head used to expose the final-residual boundary."""
        # Match the Hugging Face model interface consumed by data preparation.
        return self.output_embeddings

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
    ):
        assert use_cache is False
        assert attention_mask.shape == input_ids.shape
        hidden = torch.stack((input_ids.float(), position_ids.float()), dim=-1)
        hidden = self.layer(hidden)
        logits = self.output_embeddings(hidden)
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden] if output_hidden_states else None,
        )


class _IdentitySAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.eye(2))
        self.cfg = SimpleNamespace(hook_layer=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


class _MaskResources:
    def __init__(self, mask: list[float]) -> None:
        self.mask = torch.tensor(mask)

    def get_mdbm_mask(self, entity: str, attr: str, weight_path: Path):
        assert entity == "city"
        assert attr == "Country"
        assert weight_path.exists()
        return self.mask


def _write_reference_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "eval_config": {
                    "model_name": "gpt2",
                    "llm_dtype": "float32",
                    "num_pairs_per_attribute": 3,
                    "random_seed": 11,
                    "entity_attribute_selection": {"city": ["Country"]},
                }
            }
        )
    )


def _write_pipeline_config(
    path: Path,
    reference_json: Path,
    output_root: Path,
    *,
    mdbm_weight_path: Path | None = None,
    tau_act: float = 0.5,
    max_contexts: int = 10,
    min_contexts: int = 0,
) -> None:
    lines = [
        f"reference_json: {reference_json}",
        f"output_root: {output_root}",
        "device: cpu",
        "entity_attribute_selection:",
        "  city: ['Country']",
    ]
    if mdbm_weight_path is not None:
        lines.append(f"mdbm_weight_path: {mdbm_weight_path}")
    lines.extend(
        [
            "phases:",
            "  data_prep:",
            f"    tau_act: {tau_act}",
            f"    max_contexts: {max_contexts}",
            f"    min_contexts: {min_contexts}",
            "    readouts: ['final_resid']",
            "    gram_cache: false",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _prompt(
    text: str,
    input_ids: list[int],
    *,
    entity_label: str,
    attribute_label: str,
    final_entity_token_pos: int | None,
) -> Prompt:
    return Prompt(
        text=text,
        template="%s ->",
        attribute_type="Country",
        attribute_label=attribute_label,
        entity_label=entity_label,
        context_split="train",
        entity_split="train",
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        final_entity_token_pos=final_entity_token_pos,
        attribute_generation=attribute_label,
        first_generated_token_id=input_ids[-1] + 100,
        is_correct=True,
    )


def _collection_prompts() -> list[Prompt]:
    return [
        _prompt(
            "Paris -> France",
            [11, 12, 13],
            entity_label="Paris",
            attribute_label="France",
            final_entity_token_pos=-1,
        ),
        _prompt(
            "Berlin -> Germany",
            [21],
            entity_label="Berlin",
            attribute_label="Germany",
            final_entity_token_pos=None,
        ),
        _prompt(
            "Rome -> Italy",
            [31, 32],
            entity_label="Rome",
            attribute_label="Italy",
            final_entity_token_pos=0,
        ),
    ]


def _run_collection_fixture(
    monkeypatch,
    tmp_path: Path,
    *,
    run_name: str,
    batch_size: int,
    save_chunk_size: int,
) -> tuple[Path, FEGAConfig, list[Prompt]]:
    reference_json = tmp_path / f"{run_name}_ref.json"
    _write_reference_json(reference_json)
    prompts = _collection_prompts()
    fega_cfg = FEGAConfig.from_reference(
        reference_json,
        device="cpu",
        output_dir=tmp_path / run_name,
        entity_attribute_selection={"city": ["Country"]},
        save_chunk_size=save_chunk_size,
        random_seed=11,
        llm_batch_size_override=batch_size,
    )
    model = _ForwardModel()
    sae = _IdentitySAE()
    tokenizer = SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(
        collection.ReplayContext,
        "from_file",
        staticmethod(
            lambda path: SimpleNamespace(
                eval_config=None,
                sae_lens_release_id=None,
                sae_lens_id=None,
                sae_cfg_dict=None,
            )
        ),
    )
    monkeypatch.setattr(
        collection,
        "load_model_and_sae",
        lambda *args, **kwargs: (model, tokenizer, sae),
    )
    monkeypatch.setattr(
        collection,
        "load_filtered_dataset_if_cached",
        lambda *args, **kwargs: (object(), tmp_path / f"{run_name}_dataset.json", "loaded"),
    )
    monkeypatch.setattr(
        collection,
        "load_or_build_pairs",
        lambda *args, **kwargs: {
            "Country": {
                "cause_base_prompts": prompts,
                "cause_source_prompts": [],
                "iso_base_prompts": [],
                "iso_source_prompts": [],
            }
        },
    )
    monkeypatch.setattr(
        collection.activation_collection,
        "get_module",
        lambda model_arg, layer: model_arg.layer,
    )

    manifest_path = collection.collect_activations(
        fega_cfg,
        readouts=["final_resid"],
    )
    return manifest_path, fega_cfg, prompts


def _stream_activation_rows(manifest_path: Path, activations_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for tensors_path, meta_path in ChunkProcessor.stream(manifest_path, activations_dir):
        payload = torch.load(tensors_path, map_location="cpu")
        meta_rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
        assert len(meta_rows) == payload["z"].shape[0]
        for offset, meta in enumerate(meta_rows):
            assert payload["index"][offset] == meta["index"]
            rows.append(
                {
                    "meta": meta,
                    "x": payload["x"][offset],
                    "z": payload["z"][offset],
                    "final_resid": payload.get("final_resid", torch.empty(0))[offset],
                }
            )
    return rows


def _prompt_signature(prompt: Prompt) -> tuple:
    return (
        prompt.text,
        prompt.entity_label,
        prompt.attribute_type,
        prompt.attribute_label,
        tuple(prompt.input_ids),
        prompt.final_entity_token_pos,
        prompt.first_generated_token_id,
    )


def _collection_signature(manifest_path: Path, fega_cfg: FEGAConfig) -> list[tuple]:
    rows = _stream_activation_rows(manifest_path, fega_cfg.output_dir / "collect" / "activations")
    return [
        (
            row["meta"]["index"],
            row["meta"]["pair_index"],
            row["meta"]["original_index"],
            row["meta"]["prompt"],
            row["meta"]["entity_label"],
            row["meta"]["attribute_type"],
            row["meta"]["attribute_label"],
            row["meta"]["raw_target_position"],
            row["meta"]["unpadded_target_position"],
            tuple(row["x"].tolist()),
            tuple(row["z"].tolist()),
            tuple(row["final_resid"].tolist()),
        )
        for row in rows
    ]


def _write_selection_manifest(
    activations_dir: Path,
    meta_rows: list[dict],
    z_rows: list[list[float]],
    *,
    chunk_size: int,
) -> Path:
    activations_dir.mkdir(parents=True, exist_ok=True)
    processor = ChunkProcessor(
        activations_dir,
        chunk_size,
        "activations_tensors_{:04d}.pt",
        "activations_meta_{:04d}.jsonl",
        single_file=False,
    )
    for meta, z_row in zip(meta_rows, z_rows):
        idx = int(meta["index"])
        processor.add(
            torch.tensor([float(idx), 0.0]),
            torch.tensor(z_row, dtype=torch.float32),
            None,
            meta,
        )
    processor.flush()
    manifest_path = activations_dir / "activations_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_records": len(meta_rows),
                "chunk_size": chunk_size,
                "chunk_count": len(processor.manifest_entries),
                "single_file": False,
                "tensor_keys": ["index", "x", "z"],
                "chunks": processor.manifest_entries,
            },
            indent=2,
        )
    )
    return manifest_path


def _selection_meta_rows() -> list[dict]:
    return [
        {
            "index": 0,
            "pair_index": 0,
            "original_index": 0,
            "pair_role": "cause_base_prompts",
            "prompt": "Paris -> France",
            "entity_label": "Paris",
            "attribute_type": "Country",
            "attribute_label": "France",
            "raw_target_position": -1,
            "unpadded_target_position": 2,
            "padded_target_position": 2,
        },
        {
            "index": 1,
            "pair_index": 1,
            "original_index": 1,
            "pair_role": "cause_base_prompts",
            "prompt": "Berlin -> Germany",
            "entity_label": "Berlin",
            "attribute_type": "Country",
            "attribute_label": "Germany",
            "raw_target_position": None,
            "unpadded_target_position": 0,
            "padded_target_position": 2,
        },
        {
            "index": 2,
            "pair_index": 2,
            "original_index": 2,
            "pair_role": "cause_base_prompts",
            "prompt": "Rome -> Italy",
            "entity_label": "Rome",
            "attribute_type": "Country",
            "attribute_label": "Italy",
            "raw_target_position": 0,
            "unpadded_target_position": 0,
            "padded_target_position": 1,
        },
    ]


def test_collected_rows_resolve_to_prompt_positions_and_sae_activations(
    monkeypatch, tmp_path: Path
):
    manifest_path, fega_cfg, prompts = _run_collection_fixture(
        monkeypatch,
        tmp_path,
        run_name="positions",
        batch_size=3,
        save_chunk_size=2,
    )
    rows = _stream_activation_rows(
        manifest_path, fega_cfg.output_dir / "collect" / "activations"
    )

    for row in rows:
        meta = row["meta"]
        prompt = prompts[meta["pair_index"]]
        assert meta["original_index"] == meta["pair_index"]
        assert meta["prompt"] == prompt.text
        assert meta["entity_label"] == prompt.entity_label
        assert meta["attribute_type"] == prompt.attribute_type
        assert meta["attribute_label"] == prompt.attribute_label
        assert meta["positioning_schema_version"] == POSITIONING_SCHEMA_VERSION
        target = meta["unpadded_target_position"]
        assert meta["padded_target_position"] == meta["pad_length"] + target
        expected = torch.tensor(
            [float(prompt.input_ids[target]), float(target)],
            dtype=row["x"].dtype,
        )
        assert torch.equal(row["x"], expected)
        assert torch.equal(row["z"], expected)
        assert torch.equal(row["final_resid"], expected)


def test_collect_activations_preserves_prompt_order_across_batch_sizes_and_chunks(
    monkeypatch, tmp_path: Path
):
    manifest_b1, cfg_b1, prompts = _run_collection_fixture(
        monkeypatch,
        tmp_path,
        run_name="batch_1_chunk_1",
        batch_size=1,
        save_chunk_size=1,
    )
    manifest_b3, cfg_b3, _ = _run_collection_fixture(
        monkeypatch,
        tmp_path,
        run_name="batch_3_chunk_10",
        batch_size=3,
        save_chunk_size=10,
    )

    assert _collection_signature(manifest_b3, cfg_b3) == _collection_signature(
        manifest_b1, cfg_b1
    )
    rows = _stream_activation_rows(
        manifest_b3, cfg_b3.output_dir / "collect" / "activations"
    )
    assert [row["meta"]["pair_index"] for row in rows] == list(range(len(prompts)))

    pairs = json.loads((cfg_b3.output_dir / "collect" / "pairs_full.json").read_text())
    pair_signatures = [
        _prompt_signature(prompt)
        for prompt in prompts
    ]
    serialized_signatures = [
        (
            prompt["text"],
            prompt["entity_label"],
            prompt["attribute_type"],
            prompt["attribute_label"],
            tuple(prompt["input_ids"]),
            prompt["final_entity_token_pos"],
            prompt["first_generated_token_id"],
        )
        for prompt in pairs["Country"]["cause_base_prompts"]
    ]
    assert serialized_signatures == pair_signatures


def test_select_contexts_enforces_thresholds_and_tensor_activation_parity(
    tmp_path: Path,
):
    activations_dir = tmp_path / "activations"
    meta_rows = _selection_meta_rows()
    manifest_path = _write_selection_manifest(
        activations_dir,
        meta_rows,
        [
            [0.0, 0.49, 0.0, 0.0, 0.70],
            [0.0, 0.50, 0.0, 0.0, 0.10],
            [0.0, 0.60, 0.0, 0.0, 0.50],
        ],
        chunk_size=2,
    )

    contexts, feature_stats = selection.select_contexts(
        manifest_path,
        activations_dir,
        target_features=[1, 4],
        tau_act=0.5,
        max_contexts=10,
        min_contexts=0,
        seed=123,
    )

    assert set(contexts) == {1, 4}
    assert [ctx["index"] for ctx in contexts[1]] == [2]
    assert [ctx["index"] for ctx in contexts[4]] == [0]
    assert feature_stats[1]["skipped_tau"] == 2
    assert feature_stats[4]["skipped_tau"] == 2

    tensors_by_index: dict[int, torch.Tensor] = {}
    meta_by_index = {row["index"]: row for row in meta_rows}
    for tensors_path, meta_path in ChunkProcessor.stream(manifest_path, activations_dir):
        payload = torch.load(tensors_path, map_location="cpu")
        chunk_meta = [json.loads(line) for line in meta_path.read_text().splitlines()]
        for offset, meta in enumerate(chunk_meta):
            tensors_by_index[meta["index"]] = payload["z"][offset]

    for feature_id, ctxs in contexts.items():
        for ctx in ctxs:
            meta = meta_by_index[ctx["index"]]
            assert ctx["pair_index"] == meta["pair_index"]
            assert ctx["prompt"] == meta["prompt"]
            assert ctx["entity_label"] == meta["entity_label"]
            assert ctx["attribute_type"] == meta["attribute_type"]
            assert ctx["attribute_label"] == meta["attribute_label"]
            assert tensors_by_index[ctx["index"]][feature_id].item() == ctx["z"]


def test_run_context_selection_writes_only_mdbm_mask_feature_ids(tmp_path: Path):
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    weight_path = tmp_path / "mdbm.pt"
    weight_path.write_bytes(b"mask")
    cfg_path = tmp_path / "cfg.yaml"
    _write_pipeline_config(
        cfg_path,
        reference_json,
        tmp_path / "out",
        mdbm_weight_path=weight_path,
    )
    config = FEGAPipelineConfig.from_file(cfg_path)
    collect_dir = data_prep_collect_dir(config)
    activations_dir = data_prep_activations_dir(config)
    collect_dir.mkdir(parents=True, exist_ok=True)
    pairs = {
        "Country": {
            "cause_base_prompts": [prompt.__dict__ for prompt in _collection_prompts()],
            "cause_source_prompts": [],
            "iso_base_prompts": [],
            "iso_source_prompts": [],
        }
    }
    (collect_dir / "pairs_full.json").write_text(json.dumps(pairs))
    _write_selection_manifest(
        activations_dir,
        _selection_meta_rows(),
        [
            [0.0, 0.49, 0.0, 0.0, 0.70],
            [0.0, 0.50, 0.0, 0.0, 0.10],
            [0.0, 0.60, 0.0, 0.0, 0.50],
        ],
        chunk_size=1,
    )

    contexts_path = selection._run_context_selection(
        config,
        resources=_MaskResources([0.0, 1.0, 0.0, 0.0, 1.0]),
    )
    contexts = json.loads(contexts_path.read_text())
    summary = json.loads(
        (data_prep_select_dir(config) / "feature_contexts_summary.json").read_text()
    )

    assert list(contexts.keys()) == ["1", "4"]
    assert summary["total_features"] == 2
    assert list(summary["feature_stats"].keys()) == ["1", "4"]
    assert summary["features_with_contexts"] == 2
    assert {ctx["index"] for ctx in contexts["1"]} == {2}
    assert {ctx["index"] for ctx in contexts["4"]} == {0}


def test_stratified_topk_orders_activation_ties_by_stable_prompt_identity():
    records = [
        (
            1.0,
            {
                "index": 30,
                "pair_index": 2,
                "prompt": "Rome -> Italy",
                "entity_label": "city",
                "attribute_type": "Country",
            },
        ),
        (
            1.0,
            {
                "index": 20,
                "pair_index": 1,
                "prompt": "Berlin -> Germany",
                "entity_label": "city",
                "attribute_type": "Country",
            },
        ),
        (
            1.0,
            {
                "index": 10,
                "pair_index": 0,
                "prompt": "Paris -> France",
                "entity_label": "city",
                "attribute_type": "Country",
            },
        ),
    ]

    selected = selection._stratified_topk(records, max_contexts=3, seed=42)

    assert [ctx["pair_index"] for ctx in selected] == [0, 1, 2]
