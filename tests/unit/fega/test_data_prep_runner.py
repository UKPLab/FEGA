from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sae_bench.evals.ravel.instance import Prompt

from fega.config_schema import FEGAPipelineConfig
from fega.core.config import FEGAConfig
from fega.core.data_prep import collection, runner
from fega.core.positioning import POSITIONING_SCHEMA_VERSION
from fega.paths import (
    data_prep_activations_dir,
    data_prep_select_dir,
    gram_cache_dir,
)


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


def _write_config(cfg_path: Path, reference_json: Path, output_root: Path) -> None:
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
                "    batch_size: 3",
                "    tau_act: 0.4",
                "    max_contexts: 9",
                "    min_contexts: 2",
                "    readouts: ['final_resid']",
                "    gram_cache: false",
            ]
        )
        + "\n"
    )


def _write_induction_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "model_name": "gemma-2-2b",
                "llm_dtype": "bfloat16",
                "sae_selection_mode": "custom_repo",
                "filtering": {"answer_prefix": " ", "require_model_correct": True},
                "activation_definition": {
                    "active_if_post_encode_activation_exceeds": 0.0
                },
                "overall_feature_counts": {"total_features": 16},
                "feature_sets": {
                    "sae_a": {
                        "sae_release": "owner/repo",
                        "sae_id": "loc/layer_0/trainer_5",
                        "layer": 0,
                        "hook_name": "blocks.0.hook_resid_post",
                        "candidate_feature_ids": [1],
                        "strict_common_feature_ids": [1],
                    }
                },
            }
        )
    )


def _write_induction_config(
    cfg_path: Path,
    reference_json: Path,
    summary_json: Path,
    output_root: Path,
    *,
    gram_cache: bool,
) -> None:
    cfg_path.write_text(
        "\n".join(
            [
                "source_kind: induction",
                f"reference_json: {reference_json}",
                f"output_root: {output_root}",
                "device: cpu",
                "entity_attribute_selection:",
                "  induction: ['rule_completion']",
                "induction:",
                f"  summary_json: {summary_json}",
                "  sae_uid: sae_a",
                "phases:",
                "  data_prep:",
                "    readouts: ['final_resid']",
                f"    gram_cache: {str(gram_cache).lower()}",
            ]
        )
        + "\n"
    )


class _ForwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.layer = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
        self.device = torch.device("cpu")
        self.calls: list[dict[str, torch.Tensor]] = []

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
        assert output_hidden_states is False
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
            }
        )
        hidden = self.layer(
            torch.stack((input_ids.float(), position_ids.float()), dim=-1)
        )
        readout_input = hidden + 10.0
        raw_linear_logits = self.lm_head(readout_input)
        returned_model_logits = raw_linear_logits + 100.0
        return SimpleNamespace(
            logits=returned_model_logits,
            hidden_states=[hidden - 10.0],
        )

    def get_output_embeddings(self):
        """Return the actual linear readout module used by ``forward``."""
        # Expose the same module so the collection pre-hook captures its input.
        return self.lm_head


class _IdentitySAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.eye(2))
        self.cfg = SimpleNamespace(hook_layer=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


def _prompt(
    input_ids: list[int],
    *,
    final_entity_token_pos: int | None = None,
) -> Prompt:
    return Prompt(
        text="x",
        template="",
        attribute_type="Country",
        attribute_label="Country",
        entity_label="city",
        context_split="",
        entity_split="",
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        final_entity_token_pos=final_entity_token_pos,
        attribute_generation=None,
        first_generated_token_id=None,
        is_correct=True,
    )


def test_run_data_prep_calls_collection_then_selection(monkeypatch, tmp_path: Path):
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(cfg_path, reference_json, tmp_path / "out")
    config = FEGAPipelineConfig.from_file(cfg_path)
    calls: list[str] = []
    manifest_path = data_prep_activations_dir(config) / "activations_manifest.json"
    contexts_path = data_prep_select_dir(config) / "feature_contexts.json"

    def fake_collect(cfg, resources):
        assert cfg.phases.data_prep.tau_act == 0.4
        assert cfg.phases.data_prep.readouts == ["final_resid"]
        calls.append("collect")
        return manifest_path

    def fake_select(cfg, resources):
        assert cfg.phases.data_prep.max_contexts == 9
        assert cfg.phases.data_prep.min_contexts == 2
        calls.append("select")
        return contexts_path

    monkeypatch.setattr(runner, "_collect_data_prep_artifacts", fake_collect)
    monkeypatch.setattr(runner, "_run_context_selection", fake_select)

    result = runner.run_data_prep(config, resources=None)
    assert result == contexts_path
    assert calls == ["collect", "select"]


def test_run_data_prep_dispatches_induction_without_ravel_selection(
    monkeypatch, tmp_path: Path
) -> None:
    reference_json = tmp_path / "dataset.json"
    reference_json.write_text("{}")
    summary_json = tmp_path / "summary.json"
    _write_induction_summary(summary_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_induction_config(
        cfg_path, reference_json, summary_json, tmp_path / "out", gram_cache=False
    )
    config = FEGAPipelineConfig.from_file(cfg_path)
    manifest_path = data_prep_activations_dir(config) / "activations_manifest.json"
    contexts_path = data_prep_select_dir(config) / "feature_contexts.json"
    calls: list[str] = []

    def fail_ravel_collect(*args, **kwargs):
        raise AssertionError("RAVEL collection should not run for induction")

    def fail_ravel_select(*args, **kwargs):
        raise AssertionError("RAVEL selection should not run for induction")

    def fake_induction(cfg, resources):
        calls.append("induction")
        return manifest_path, contexts_path

    monkeypatch.setattr(runner, "_collect_data_prep_artifacts", fail_ravel_collect)
    monkeypatch.setattr(runner, "_run_context_selection", fail_ravel_select)
    monkeypatch.setattr(runner, "run_induction_data_prep", fake_induction)

    result = runner.run_data_prep(config, resources=None)

    assert result == contexts_path
    assert calls == ["induction"]


def test_run_data_prep_writes_shared_gram_once_for_induction(
    monkeypatch, tmp_path: Path
) -> None:
    reference_json = tmp_path / "dataset.json"
    reference_json.write_text("{}")
    summary_json = tmp_path / "summary.json"
    _write_induction_summary(summary_json)
    cfg_path = tmp_path / "cfg.yaml"
    _write_induction_config(
        cfg_path, reference_json, summary_json, tmp_path / "out", gram_cache=True
    )
    config = FEGAPipelineConfig.from_file(cfg_path)
    activations_dir = data_prep_activations_dir(config)
    activations_dir.mkdir(parents=True)
    tensors_path = activations_dir / "activations_tensors_0000.pt"
    meta_path = activations_dir / "activations_meta_0000.jsonl"
    torch.save(
        {
            "index": [0],
            "x": torch.ones(1, 2),
            "z": torch.ones(1, 3),
            "final_resid": torch.ones(1, 5),
        },
        tensors_path,
    )
    meta_path.write_text(json.dumps({"index": 0}) + "\n")
    manifest_path = activations_dir / "activations_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_records": 1,
                "chunks": [
                    {
                        "chunk": 0,
                        "count": 1,
                        "start_index": 0,
                        "end_index": 0,
                        "tensors": str(tensors_path),
                        "meta": str(meta_path),
                    }
                ]
            }
        )
    )
    contexts_path = data_prep_select_dir(config) / "feature_contexts.json"
    gram_calls: list[int] = []

    class Resources:
        def get_model_and_sae(self):
            return object(), object(), object()

    def fake_induction(cfg, resources):
        return manifest_path, contexts_path

    def fake_write_gram_cache(model, cfg, *, final_resid_width):
        gram_calls.append(final_resid_width)
        return gram_cache_dir(cfg) / "gram_meta.json"

    monkeypatch.setattr(runner, "run_induction_data_prep", fake_induction)
    monkeypatch.setattr(runner, "write_gram_cache", fake_write_gram_cache)

    result = runner.run_data_prep(config, resources=Resources())

    assert result == contexts_path
    assert gram_calls == [5]


def test_run_sae_reconstruction_forwards_position_ids(monkeypatch) -> None:
    model = _ForwardModel()
    sae = _IdentitySAE()
    monkeypatch.setattr(
        collection.activation_collection,
        "get_module",
        lambda model_arg, layer: model_arg.layer,
    )
    tokens = torch.tensor([[10, 11]], dtype=torch.long)
    attention = torch.tensor([[1, 1]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)

    _, _, readouts = collection.run_sae_reconstruction(
        model,
        sae,
        tokens,
        attention,
        [1],
        position_ids=position_ids,
        readouts=["final_resid"],
    )

    assert torch.equal(model.calls[0]["position_ids"], position_ids)
    assert torch.equal(readouts["final_resid"][0], torch.tensor([21.0, 11.0]))

    _, _, default_readouts = collection.run_sae_reconstruction(
        model,
        sae,
        tokens,
        attention,
        [1],
        position_ids=position_ids,
    )
    assert set(default_readouts) == {"final_resid"}


def test_collect_activations_writes_positioning_metadata(monkeypatch, tmp_path: Path):
    reference_json = tmp_path / "ref.json"
    _write_reference_json(reference_json)
    fega_cfg = FEGAConfig.from_reference(
        reference_json,
        device="cpu",
        output_dir=tmp_path / "out",
        entity_attribute_selection={"city": ["Country"]},
        llm_batch_size_override=2,
    )
    model = _ForwardModel()
    sae = _IdentitySAE()
    tokenizer = SimpleNamespace(pad_token_id=42)
    prompts = [
        _prompt([1, 2, 3], final_entity_token_pos=-1),
        _prompt([4], final_entity_token_pos=None),
    ]

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
        lambda *args, **kwargs: (object(), tmp_path / "dataset.json", "loaded"),
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

    manifest_path = collection.collect_activations(fega_cfg)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["readouts"] == ["final_resid"]
    positioning = manifest["positioning"]
    assert positioning["schema_version"] == POSITIONING_SCHEMA_VERSION
    assert positioning["pad_token_id"] == 42
    assert positioning["batch_size_provenance"]["configured_batch_size"] == 2
    meta = json.loads(
        (fega_cfg.output_dir / "collect" / "collection_meta.json").read_text()
    )
    assert meta["positioning"]["position_id_scheme"] == "prompt_local_attention_cumsum"
    meta_path = Path(manifest["chunks"][0]["meta"])
    rows = [json.loads(line) for line in meta_path.read_text().splitlines()]
    row_by_pair = {row["pair_index"]: row for row in rows}
    assert row_by_pair[0]["final_entity_token_pos"] == -1
    assert row_by_pair[0]["raw_target_position"] == -1
    assert row_by_pair[0]["unpadded_target_position"] == 2
    assert row_by_pair[0]["padded_target_position"] == 2
    assert row_by_pair[1]["prompt_length"] == 1
    assert row_by_pair[1]["pad_length"] == 2
    assert row_by_pair[1]["padded_target_position"] == 2
