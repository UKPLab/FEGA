from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from fega.config_schema import FEGAPipelineConfig
from fega.core.data_prep.induction import (
    ResolvedInductionFeatureSet,
    _align_selected_feature_activations,
    _assert_scan_materialization_parity,
    _selected_z_by_source,
    load_induction_examples,
    materialize_selected_induction_rows,
    prepare_scan_examples,
    resolve_induction_feature_set,
    run_induction_data_prep,
    scan_induction_records,
    select_induction_contexts,
    write_induction_artifacts,
)
from fega.paths import (
    data_prep_pairs_path,
)


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, **_kwargs):
        if text.startswith(" "):
            return [100 + len(text.strip())]
        return [ord(ch) % 97 + 1 for ch in text]


class _HookLayer(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return (hidden_states,)


class _ScanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(architectures=["Gemma2ForCausalLM"])
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([_HookLayer()]))

    def forward(self, input_ids, attention_mask, position_ids, **_kwargs):
        _ = (attention_mask, position_ids)
        hidden_states = torch.nn.functional.one_hot(
            input_ids % 4, num_classes=4
        ).float()
        hidden_states = self.model.layers[0](hidden_states)[0]
        logits = torch.zeros(
            (*input_ids.shape, 8), device=input_ids.device, dtype=hidden_states.dtype
        )
        logits[..., 3] = 10.0
        return SimpleNamespace(logits=logits)


class _BatchTrackingSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(hook_layer=0)
        self.encode_shapes: list[tuple[int, ...]] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_shapes.append(tuple(x.shape))
        base = torch.arange(6, device=x.device, dtype=x.dtype).expand(x.shape[0], 6)
        offsets = torch.arange(x.shape[0], device=x.device, dtype=x.dtype).unsqueeze(1)
        return base + offsets


class _InputStableSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(hook_layer=0)
        self.encode_shapes: list[tuple[int, ...]] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_shapes.append(tuple(x.shape))
        weights = torch.tensor([11.0, 13.0, 17.0, 19.0], device=x.device, dtype=x.dtype)
        score = x @ weights
        z = torch.zeros((x.shape[0], 6), device=x.device, dtype=x.dtype)
        z[:, 1] = score + 1.0
        z[:, 4] = score + 4.0
        return z


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"seed": 123},
                "contexts": [
                    {
                        "context_id": 0,
                        "query_style": "rule",
                        "queries": [
                            {
                                "query_id": "q0",
                                "prompt": "alpha",
                                "answer": "A",
                                "x": "x0",
                                "y": "y0",
                                "entity": "e0",
                                "source_concept": "source0",
                                "target_concept": "target0",
                                "support_example_index": 0,
                                "lookup_rule": "x0 -> y0",
                                "induction_prefix": "alpha",
                            }
                        ],
                    },
                    {
                        "context_id": 1,
                        "query_style": "rule",
                        "queries": [
                            {
                                "query_id": "q1",
                                "prompt": "beta",
                                "answer": "B",
                                "x": "x1",
                                "y": "y1",
                                "entity": "e1",
                                "source_concept": "source1",
                                "target_concept": "target1",
                                "support_example_index": 1,
                                "lookup_rule": "x1 -> y1",
                                "induction_prefix": "beta",
                            }
                        ],
                    },
                ],
            }
        )
    )


def _write_capped_dataset(path: Path) -> None:
    contexts = []
    for context_id in range(4):
        queries = []
        for support_example_index in range(2):
            row_id = context_id * 2 + support_example_index
            queries.append(
                {
                    "query_id": f"q{row_id}",
                    "prompt": f"prompt-{row_id}",
                    "answer": f"A{row_id}",
                    "x": f"x{row_id}",
                    "y": f"y{row_id}",
                    "entity": f"e{row_id}",
                    "source_concept": f"source{context_id}",
                    "target_concept": f"target{support_example_index}",
                    "support_example_index": support_example_index,
                    "lookup_rule": f"x{row_id} -> y{row_id}",
                    "induction_prefix": f"prefix-{context_id}",
                }
            )
        contexts.append(
            {
                "context_id": context_id,
                "query_style": "rule",
                "queries": queries,
            }
        )
    path.write_text(json.dumps({"metadata": {"seed": 123}, "contexts": contexts}))


def _write_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_path": "dataset.json",
                "model_name": "gemma-2-2b",
                "llm_dtype": "bfloat16",
                "sae_selection_mode": "custom_repo",
                "dataset_metadata": {"seed": 123},
                "filtering": {
                    "answer_prefix": " ",
                    "require_model_correct": True,
                    "single_token_only": False,
                    "max_contexts": None,
                    "max_examples": None,
                },
                "activation_definition": {
                    "active_if_post_encode_activation_exceeds": 0.0
                },
                "overall_feature_counts": {"total_features": 32},
                "feature_sets": {
                    "sae_a": {
                        "sae_release": "owner/repo",
                        "sae_id": "loc/layer_0/trainer_5",
                        "layer": 0,
                        "hook_name": "blocks.0.hook_resid_post",
                        "candidate_feature_ids": [7, 3],
                        "strict_common_feature_ids": [7],
                        "total_features": 32,
                    }
                },
            }
        )
    )


def _write_config(
    cfg_path: Path,
    reference_json: Path,
    summary_json: Path,
    output_root: Path,
    *,
    feature_set: str = "candidate",
    explicit_feature_ids: list[int] | None = None,
    save_chunk_size: int = 2,
) -> FEGAPipelineConfig:
    explicit = ""
    if explicit_feature_ids is not None:
        explicit = f"  explicit_feature_ids: {explicit_feature_ids}\n"
    cfg_path.write_text(
        f"source_kind: induction\n"
        f"reference_json: {reference_json}\n"
        f"output_root: {output_root}\n"
        "device: cpu\n"
        "sae_repo_id: owner/repo\n"
        "entity_attribute_selection:\n"
        "  induction: ['rule_completion']\n"
        "induction:\n"
        f"  summary_json: {summary_json}\n"
        f"  feature_set: {feature_set}\n"
        "  sae_uid: sae_a\n"
        f"{explicit}"
        "phases:\n"
        "  data_prep:\n"
        f"    save_chunk_size: {save_chunk_size}\n"
        "    readouts: ['final_resid']\n"
        "    gram_cache: true\n"
        "  compute_effect:\n"
        "    enabled: true\n"
    )
    return FEGAPipelineConfig.from_file(cfg_path)


def test_checked_summary_feature_resolution_candidate_and_strict_common(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    summary = tmp_path / "summary.json"
    _write_dataset(dataset)
    _write_summary(summary)
    cfg = _write_config(
        tmp_path / "config.yaml",
        dataset,
        summary,
        tmp_path / "out",
    )

    _, candidate = resolve_induction_feature_set(cfg)
    assert candidate.selected_feature_ids == [7, 3]

    cfg.induction.feature_set = "strict_common"  # type: ignore[union-attr]
    _, strict = resolve_induction_feature_set(cfg)
    assert strict.selected_feature_ids == [7]


def test_explicit_feature_resolution_preserves_validated_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    summary = tmp_path / "summary.json"
    _write_dataset(dataset)
    _write_summary(summary)
    cfg = _write_config(
        tmp_path / "cfg.yaml",
        dataset,
        summary,
        tmp_path / "out",
        feature_set="explicit",
        explicit_feature_ids=[9, 2],
    )

    _, feature_set = resolve_induction_feature_set(cfg)

    assert feature_set.selected_feature_ids == [9, 2]


def test_induction_dataset_sampling_and_target_tokens_are_stable(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    summary = tmp_path / "summary.json"
    _write_dataset(dataset)
    _write_summary(summary)
    cfg = _write_config(tmp_path / "cfg.yaml", dataset, summary, tmp_path / "out")
    assert cfg.induction is not None
    examples = load_induction_examples(dataset)

    first = prepare_scan_examples(
        examples,
        _Tokenizer(),
        induction=cfg.induction,
        summary=json.loads(summary.read_text()),
        seed=42,
    )
    second = prepare_scan_examples(
        list(reversed(examples)),
        _Tokenizer(),
        induction=cfg.induction,
        summary=json.loads(summary.read_text()),
        seed=42,
    )

    assert [row["prompt"] for row in first] == ["alpha", "beta"]
    assert [row["prompt"] for row in first] == [row["prompt"] for row in second]
    assert [row["target_first_token_id"] for row in first] == [101, 101]
    assert [row["source_row_index"] for row in first] == [0, 1]

    provenance_only_summary = json.loads(summary.read_text())
    provenance_only_summary["filtering"]["max_examples"] = 1
    summary_limited = prepare_scan_examples(
        examples,
        _Tokenizer(),
        induction=cfg.induction,
        summary=provenance_only_summary,
        seed=42,
    )
    assert [row["prompt"] for row in summary_limited] == ["alpha", "beta"]

    limited = prepare_scan_examples(
        examples,
        _Tokenizer(),
        induction=cfg.induction,
        summary=json.loads(summary.read_text()),
        seed=42,
        limit=1,
    )
    assert [row["prompt"] for row in limited] == ["alpha"]


def test_induction_source_caps_are_seeded_and_input_order_invariant(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    summary = tmp_path / "summary.json"
    _write_capped_dataset(dataset)
    _write_summary(summary)
    cfg = _write_config(tmp_path / "cfg.yaml", dataset, summary, tmp_path / "out")
    assert cfg.induction is not None
    cfg.induction.max_source_contexts = 2
    cfg.induction.max_source_examples = 3
    examples = load_induction_examples(dataset)

    first = prepare_scan_examples(
        examples,
        _Tokenizer(),
        induction=cfg.induction,
        summary=json.loads(summary.read_text()),
        seed=11,
    )
    second = prepare_scan_examples(
        list(reversed(examples)),
        _Tokenizer(),
        induction=cfg.induction,
        summary=json.loads(summary.read_text()),
        seed=11,
    )

    assert [row["source_row_index"] for row in first] == [4, 5, 7]
    assert [row["source_row_index"] for row in second] == [4, 5, 7]
    assert [row["context_id"] for row in first] == [2, 2, 3]
    assert [row["prompt"] for row in first] == ["prompt-4", "prompt-5", "prompt-7"]


def test_induction_scan_batches_sae_encode_and_records_only_selected_features() -> None:
    examples = [
        {
            "source_row_index": 0,
            "example_id": "q0",
            "context_id": 0,
            "support_example_index": 0,
            "prompt": "alpha",
            "answer": "A",
            "target_first_token_id": 3,
            "target_token_length": 1,
        },
        {
            "source_row_index": 1,
            "example_id": "q1",
            "context_id": 1,
            "support_example_index": 1,
            "prompt": "beta",
            "answer": "B",
            "target_first_token_id": 3,
            "target_token_length": 1,
        },
    ]
    sae = _BatchTrackingSAE()

    records, stats = scan_induction_records(
        model=_ScanModel(),
        tokenizer=_Tokenizer(),
        sae=sae,
        examples=examples,
        feature_ids=[1, 4],
        require_model_correct=True,
        batch_size=8,
        device="cpu",
    )

    assert stats["kept_after_correctness"] == 2
    assert sae.encode_shapes == [(2, 4)]
    assert [record["predicted_first_token_id"] for record in records] == [3, 3]
    assert [set(record["feature_activations"]) for record in records] == [
        {1, 4},
        {1, 4},
    ]
    assert records[0]["feature_activations"] == {1: 1.0, 4: 4.0}
    assert records[1]["feature_activations"] == {1: 2.0, 4: 5.0}


def test_induction_scan_identity_and_selection_are_batch_size_invariant() -> None:
    examples = [
        {
            "source_row_index": idx,
            "example_id": f"q{idx}",
            "context_id": idx,
            "support_example_index": 0,
            "prompt": prompt,
            "answer": "A",
            "target_first_token_id": 3,
            "target_token_length": 1,
        }
        for idx, prompt in enumerate(["a", "bb", "ccc"])
    ]

    def scan_payload(
        batch_size: int,
    ) -> tuple[list[tuple], list[int], list[tuple[int, ...]]]:
        sae = _InputStableSAE()
        records, stats = scan_induction_records(
            model=_ScanModel(),
            tokenizer=_Tokenizer(),
            sae=sae,
            examples=examples,
            feature_ids=[1, 4],
            require_model_correct=True,
            batch_size=batch_size,
            device="cpu",
        )
        selected, _ = select_induction_contexts(
            records,
            [1],
            tau_act=0.0,
            max_contexts=3,
            min_contexts=1,
            stratify_by=["context_id"],
        )
        record_payload = [
            (
                record["source_row_index"],
                record["unpadded_target_position"],
                record["feature_activations"],
            )
            for record in records
        ]
        selected_sources = [record["source_row_index"] for record in selected[1]]
        assert stats["kept_after_correctness"] == 3
        return record_payload, selected_sources, sae.encode_shapes

    one_by_one_payload, one_by_one_selected, one_by_one_shapes = scan_payload(1)
    batched_payload, batched_selected, batched_shapes = scan_payload(3)

    assert one_by_one_payload == batched_payload
    assert one_by_one_selected == batched_selected
    assert one_by_one_shapes == [(1, 4), (1, 4), (1, 4)]
    assert batched_shapes == [(3, 4)]


def test_source_activation_threshold_is_provenance_not_fega_selection(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset.json"
    summary = tmp_path / "summary.json"
    _write_dataset(dataset)
    _write_summary(summary)
    cfg_low = _write_config(
        tmp_path / "cfg_low.yaml", dataset, summary, tmp_path / "out_low"
    )
    cfg_high = _write_config(
        tmp_path / "cfg_high.yaml", dataset, summary, tmp_path / "out_high"
    )
    assert cfg_low.induction is not None
    assert cfg_high.induction is not None
    cfg_low.induction.source_activation_threshold = 0.0
    cfg_high.induction.source_activation_threshold = 99.0
    cfg_low.phases.data_prep.tau_act = 1.0
    cfg_high.phases.data_prep.tau_act = 1.0
    cfg_low.phases.data_prep.max_contexts = 2
    cfg_high.phases.data_prep.max_contexts = 2
    cfg_low.phases.data_prep.min_contexts = 1
    cfg_high.phases.data_prep.min_contexts = 1
    scan_records = [
        {
            "scan_record_id": 0,
            "source_row_index": 0,
            "example_id": "q0",
            "context_id": 0,
            "support_example_index": 0,
            "prompt": "alpha",
            "answer": "A",
            "target_first_token_id": 101,
            "target_token_length": 1,
            "model_correct_first_token": True,
            "feature_activations": {7: 0.5},
        },
        {
            "scan_record_id": 1,
            "source_row_index": 1,
            "example_id": "q1",
            "context_id": 1,
            "support_example_index": 0,
            "prompt": "beta",
            "answer": "B",
            "target_first_token_id": 101,
            "target_token_length": 1,
            "model_correct_first_token": True,
            "feature_activations": {7: 1.5},
        },
    ]
    for record in scan_records:
        record["feature_activations"][3] = record["feature_activations"][7]
    captured: dict[str, tuple[dict[int, list[int]], dict[int, dict]]] = {}

    class _Resources:
        def get_model_and_sae(self):
            return _ScanModel(), _Tokenizer(), _InputStableSAE()

    def fake_scan(**kwargs):
        assert kwargs["feature_ids"] == [7, 3]
        return [dict(record) for record in scan_records], {"kept_after_correctness": 2}

    def fake_materialize(**kwargs):
        cfg = kwargs["config"]
        selected_by_feature = kwargs["selected_by_feature"]
        selection_stats = kwargs["selection_stats"]
        captured[str(cfg.output_root)] = (
            {
                int(feature_id): [int(record["source_row_index"]) for record in records]
                for feature_id, records in selected_by_feature.items()
            },
            selection_stats,
        )
        return []

    def fake_write(**_kwargs):
        return tmp_path / "manifest.json", tmp_path / "feature_contexts.json"

    monkeypatch.setattr(
        "fega.core.data_prep.induction.scan_induction_records", fake_scan
    )
    monkeypatch.setattr(
        "fega.core.data_prep.induction.materialize_selected_induction_rows",
        fake_materialize,
    )
    monkeypatch.setattr(
        "fega.core.data_prep.induction.write_induction_artifacts", fake_write
    )

    run_induction_data_prep(cfg_low, _Resources())
    run_induction_data_prep(cfg_high, _Resources())

    low_selected, low_stats = captured[str(cfg_low.output_root)]
    high_selected, high_stats = captured[str(cfg_high.output_root)]
    assert low_selected == {7: [1], 3: [1]}
    assert high_selected == {7: [1], 3: [1]}
    assert low_stats == high_stats


def test_selected_scan_activations_are_aligned_into_materialized_z() -> None:
    selected_by_feature = {
        2: [
            {
                "scan_record_id": 3,
                "source_row_index": 4,
                "z": 9.5,
            }
        ],
        5: [
            {
                "scan_record_id": 3,
                "source_row_index": 4,
                "z": 1.25,
            }
        ],
    }
    materialized_z = torch.arange(8, dtype=torch.float32)

    aligned_z = _align_selected_feature_activations(
        materialized_z,
        {"scan_record_id": 3, "source_row_index": 4},
        _selected_z_by_source(selected_by_feature),
    )

    assert float(aligned_z[2].item()) == 9.5
    assert float(aligned_z[5].item()) == 1.25
    assert float(aligned_z[1].item()) == 1.0
    assert float(materialized_z[2].item()) == 2.0
    _assert_scan_materialization_parity(
        [{"index": 0, "z": aligned_z}], selected_by_feature
    )


def test_scan_materialize_and_write_contract_is_batch_and_chunk_invariant(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset.json"
    summary_path = tmp_path / "summary.json"
    _write_dataset(dataset)
    _write_summary(summary_path)
    summary = json.loads(summary_path.read_text())
    feature_set = ResolvedInductionFeatureSet(
        sae_uid="sae_a",
        sae_release="owner/repo",
        sae_id="loc/layer_0/trainer_5",
        layer=0,
        hook_name="blocks.0.hook_resid_post",
        selected_feature_ids=[1],
        feature_set_name="candidate",
        raw={},
    )

    def fake_reconstruction(
        _model,
        _sae,
        input_ids,
        _attention_mask,
        _target_positions,
        *,
        position_ids,
        readouts,
    ):
        rows = input_ids.shape[0]
        reps = torch.stack(
            [
                torch.tensor([float(input_ids[row].sum().item()), 1.0])
                for row in range(rows)
            ]
        )
        zs = torch.full((rows, 6), -100.0)
        payload = {}
        if "final_resid" in readouts:
            payload["final_resid"] = (
                position_ids.float().sum(dim=-1, keepdim=True).repeat(1, 2)
            )
        return reps, zs, payload

    monkeypatch.setattr(
        "fega.core.data_prep.induction.run_sae_reconstruction", fake_reconstruction
    )

    def run_variant(
        *,
        scan_batch_size: int,
        materialize_batch_size: int,
        save_chunk_size: int,
    ) -> tuple[list[int], list[int], dict[int, float], list[str], list[str]]:
        cfg = _write_config(
            tmp_path
            / f"cfg_{scan_batch_size}_{materialize_batch_size}_{save_chunk_size}.yaml",
            dataset,
            summary_path,
            tmp_path
            / f"out_{scan_batch_size}_{materialize_batch_size}_{save_chunk_size}",
            save_chunk_size=save_chunk_size,
        )
        cfg.phases.data_prep.batch_size = materialize_batch_size
        assert cfg.induction is not None
        examples = prepare_scan_examples(
            load_induction_examples(dataset),
            _Tokenizer(),
            induction=cfg.induction,
            summary=summary,
            seed=42,
        )
        scan_records, scan_stats = scan_induction_records(
            model=_ScanModel(),
            tokenizer=_Tokenizer(),
            sae=_InputStableSAE(),
            examples=examples,
            feature_ids=[1],
            require_model_correct=False,
            batch_size=scan_batch_size,
            device="cpu",
        )
        selected_by_feature, selection_stats = select_induction_contexts(
            scan_records,
            [1],
            tau_act=0.0,
            max_contexts=2,
            min_contexts=1,
            stratify_by=["context_id", "support_example_index"],
        )
        dense_rows = materialize_selected_induction_rows(
            config=cfg,
            model=_ScanModel(),
            tokenizer=_Tokenizer(),
            sae=_InputStableSAE(),
            selected_by_feature=selected_by_feature,
            feature_set=feature_set,
            summary=summary,
            scan_stats=scan_stats,
            selection_stats=selection_stats,
        )
        manifest_path, _ = write_induction_artifacts(
            config=cfg,
            tokenizer=_Tokenizer(),
            dense_rows=dense_rows,
            selected_by_feature=selected_by_feature,
            feature_set=feature_set,
            summary=summary,
            scan_stats=scan_stats,
            selection_stats=selection_stats,
        )
        manifest = json.loads(manifest_path.read_text())
        dense_order: list[int] = []
        selected_z: dict[int, float] = {}
        meta_prompts: list[str] = []
        for chunk in manifest["chunks"]:
            tensors = torch.load(chunk["tensors"], map_location="cpu")
            with open(chunk["meta"]) as mf:
                metas = [json.loads(line) for line in mf]
            for local_idx, dense_idx in enumerate(tensors["index"]):
                dense_idx = int(dense_idx)
                dense_order.append(dense_idx)
                selected_z[dense_idx] = float(tensors["z"][local_idx, 1].item())
                meta_prompts.append(metas[local_idx]["prompt"])
        pairs = json.loads(data_prep_pairs_path(cfg).read_text())
        pair_prompts = [
            row["text"] for row in pairs["rule_completion"]["cause_base_prompts"]
        ]
        selected_sources = [
            int(record["source_row_index"]) for record in selected_by_feature[1]
        ]
        return selected_sources, dense_order, selected_z, pair_prompts, meta_prompts

    small_batches = run_variant(
        scan_batch_size=1, materialize_batch_size=1, save_chunk_size=1
    )
    larger_batches = run_variant(
        scan_batch_size=2, materialize_batch_size=2, save_chunk_size=2
    )

    assert small_batches == larger_batches
    selected_sources, dense_order, selected_z, pair_prompts, meta_prompts = (
        small_batches
    )
    assert selected_sources == [0, 1]
    assert dense_order == [0, 1]
    assert pair_prompts == meta_prompts == ["alpha", "beta"]
    assert selected_z == {0: 14.0, 1: 14.0}
