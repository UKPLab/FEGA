from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from sae_bench.evals.ravel.instance import Prompt

import fega.core.compute_effect.effects as effects_module
import fega.core.compute_effect.runner as runner_module
from fega.config_schema import (
    ComputeEffectConfig,
    DataPrepConfig,
    FEGAPipelineConfig,
    PhasesConfig,
)
from fega.core.compute_effect.artifacts import (
    effect_direction,
    gram_magnitude,
    summarize_magnitudes,
)
from fega.core.compute_effect.effects import (
    EffectContextRecord,
    build_effect_context_records,
    compute_feature_effect_rows,
    run_ablation_readouts_batch,
)
from fega.core.compute_effect.runner import (
    _validate_examples_against_gram,
    _validate_gram,
    run_compute_effect,
)
from fega.core.data_prep.gram_cache import (
    GRAM_CONSTRUCTION_RECIPE,
    gram_fingerprint,
    unembedding_fingerprint,
)
from fega.core.positioning import (
    POSITIONING_SCHEMA_VERSION,
    build_positioning_metadata,
)
from fega.core.resources import ModelResources
from fega.paths import (
    compute_effect_readout_dir,
    data_prep_activations_dir,
    data_prep_pairs_path,
    data_prep_select_dir,
    effect_tensors_manifest_path,
    gram_cache_meta_path,
    gram_cache_tensor_path,
)


def _positioning() -> dict:
    return build_positioning_metadata(
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        batch_size_provenance={"phase": "test", "configured_batch_size": 8},
    )


def test_gram_magnitude_and_direction_normalization() -> None:
    gram = torch.tensor([[2.0, 0.0], [0.0, 8.0]])
    delta = torch.tensor([[3.0, 4.0]])

    magnitude = gram_magnitude(delta, gram)
    direction = effect_direction(delta, magnitude)

    assert magnitude.item() == pytest.approx((18.0 + 128.0) ** 0.5)
    assert direction.shape == delta.shape
    assert direction[0, 0].item() == pytest.approx(3.0 / magnitude.item())
    unit_q = torch.sum((direction @ gram.double()) * direction)
    assert unit_q.item() == pytest.approx(1.0, abs=1.0e-12)


def test_summarize_magnitudes_includes_cv_and_quantiles() -> None:
    stats = summarize_magnitudes([1.0, 2.0, 3.0])

    assert stats["mean_magnitude"] == pytest.approx(2.0)
    assert stats["std_magnitude"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert stats["median_magnitude"] == pytest.approx(2.0)
    assert stats["q10_magnitude"] == pytest.approx(1.2)
    assert stats["q90_magnitude"] == pytest.approx(2.8)
    assert stats["cv_magnitude"] == pytest.approx(((2.0 / 3.0) ** 0.5) / 2.0)


def test_gram_validation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="square"):
        _validate_gram(torch.ones(2, 3), {})
    with pytest.raises(ValueError, match="missing required"):
        _validate_gram(torch.eye(2), {"hidden_width": 3})
    with pytest.raises(ValueError, match="incompatible"):
        _validate_examples_against_gram(
            {0: {"readouts": {"final_resid": torch.ones(3)}}},
            hidden_size=2,
            manifest_path=Path("m"),
        )


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        self.lm_head.weight.data.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        )
        self.config = SimpleNamespace(name_or_path="fake-model")

    def get_output_embeddings(self):
        """Return the deterministic fake canonical unembedding."""
        # Share the exact module used to construct test Gram metadata.
        return self.lm_head


class _ReadoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        self.lm_head.weight.data.copy_(torch.eye(2))

    def get_output_embeddings(self):
        """Expose the exact linear module called by the model forward pass."""
        # Return the actual module so forward-pre-hook capture is unambiguous.
        return self.lm_head

    def forward(self, **kwargs):
        """Return diagnostics deliberately distinct from every internal tensor."""
        # Invoke the LM head on a distinct post-normalization representation.
        hidden = self.layer(torch.ones((*kwargs["input_ids"].shape, 2)))
        readout_input = hidden + 7.0
        raw_logits = self.lm_head(readout_input)
        return SimpleNamespace(
            logits=raw_logits + 50.0,
            hidden_states=[hidden * -3.0],
        )


class _FakeSAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.ones(1, 1))
        self.cfg = SimpleNamespace(hook_layer=0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return the test latent unchanged as a decoded residual row."""
        # Keep the ablation fixture focused on readout capture, not SAE behavior.
        return z


class _FakeResources:
    def __init__(self, tokenizer=None) -> None:
        self._json_cache = {}
        self._tokenizer = tokenizer or object()

    def get_cached_json(self, path: Path):
        return self._json_cache.get(str(path))

    def cache_json(self, path: Path, payload) -> None:
        self._json_cache[str(path)] = payload

    def get_model_and_sae(self):
        return _FakeModel(), self._tokenizer, _FakeSAE()


class _FakeTokenizer:
    def __call__(
        self, text: str, *, return_attention_mask: bool, add_special_tokens: bool
    ):
        _ = (return_attention_mask, add_special_tokens)
        input_ids = [ord(ch) % 31 + 1 for ch in text]
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
        )


def _prompt(
    input_ids: list[int] | None = None,
    *,
    final_entity_token_pos: int | None = 1,
) -> Prompt:
    tokens = input_ids or [1, 2]
    return Prompt(
        text="x",
        template="",
        attribute_type="",
        attribute_label="Country",
        entity_label="city",
        context_split="",
        entity_split="",
        input_ids=tokens,
        attention_mask=[1] * len(tokens),
        final_entity_token_pos=final_entity_token_pos,
        attribute_generation=None,
        first_generated_token_id=None,
        is_correct=True,
    )


def _context(index: int, prompt: Prompt | None = None) -> EffectContextRecord:
    return EffectContextRecord(
        index=index,
        pair_index=index,
        pair_role="clean",
        attribute_label="Country",
        feature_activation=0.5,
        prompt=prompt or _prompt(),
    )


def _kept_stats() -> dict[str, object]:
    return {
        "kept": 1,
        "skipped_near_zero": 0,
        "skipped_zero_norm": 0,
        "skipped_nonfinite": 0,
        "skipped_invalid": 0,
        "skipped_numerical_failure": 0,
        "candidate_identity": [],
        "retained_mask": [],
        "oom_adjustments": [],
    }


def test_country_value_label_resolves_canonical_pair_prompt_without_fallback() -> None:
    """Keep RAVEL lookup identity on the attribute type, not its value label."""
    # Provide the canonical BOS-bearing pair prompt under the Country lookup key.
    canonical_prompt = Prompt(
        text="Rome is in",
        template="",
        attribute_type="Country",
        attribute_label="Italy",
        entity_label="Rome",
        context_split="",
        entity_split="",
        input_ids=[2, 17, 23],
        attention_mask=[1, 1, 1],
        final_entity_token_pos=2,
        attribute_generation="Italy",
        first_generated_token_id=31,
        is_correct=True,
    )

    def fail_fallback_tokenization(*args, **kwargs):
        """Fail if prompt provenance falls back to metadata tokenization."""
        # Make any fallback invocation immediately visible to this regression.
        raise AssertionError("fallback tokenization must not run")

    records, stats = build_effect_context_records(
        feature_id=0,
        raw_contexts=[
            {
                "index": 0,
                "pair_index": 0,
                "pair_role": "cause_base_prompts",
                "attribute_type": "Country",
                "attribute_label": "Italy",
                "z": 0.5,
            }
        ],
        example_bank={
            0: {
                "z": torch.tensor([0.5]),
                "meta": {"prompt": "fallback text"},
            }
        },
        prompt_lookup={
            "Country": {"cause_base_prompts": [canonical_prompt]},
        },
        tokenizer=fail_fallback_tokenization,
        default_attr="Country",
    )

    assert stats == {
        "skipped_missing_example": 0,
        "skipped_missing_prompt": 0,
        "skipped_invalid": 0,
    }
    assert len(records) == 1
    assert records[0].attribute_label == "Italy"
    assert records[0].prompt is canonical_prompt
    assert records[0].prompt.input_ids[0] == 2


def test_ablation_captures_actual_lm_head_input_not_hidden_or_returned_logits(
    monkeypatch,
) -> None:
    model = _ReadoutModel()
    sae = _FakeSAE()
    monkeypatch.setattr(
        effects_module.activation_collection,
        "get_module",
        lambda model_arg, layer: model_arg.layer,
    )

    readouts = run_ablation_readouts_batch(
        model,
        sae,
        tokens=torch.tensor([[1, 2]]),
        attn=torch.ones(1, 2, dtype=torch.long),
        target_positions=[1],
        z_batch=torch.ones(1, 1),
        ablation_spec=effects_module.AblationSpec(feature_ids=torch.tensor([0])),
        position_ids=torch.tensor([[0, 1]]),
        requested_readouts=["final_resid"],
    )

    assert torch.equal(readouts["final_resid"][0], torch.tensor([7.0, 7.0]))


@pytest.mark.parametrize(
    ("gram", "counter"),
    [
        (torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]), "skipped_nonfinite"),
        (-torch.eye(2), "skipped_numerical_failure"),
    ],
)
def test_final_resid_rejects_invalid_quadratic_forms(
    monkeypatch, gram: torch.Tensor, counter: str
) -> None:
    record = _context(0)
    example_bank = {
        0: {
            "z": torch.ones(1),
            "readouts": {"final_resid": torch.zeros(2)},
            "meta": {},
        }
    }
    monkeypatch.setattr(
        effects_module,
        "run_ablation_readouts_batch",
        lambda **kwargs: {"final_resid": [torch.tensor([1.0, 0.0])]},
    )

    rows, stats = compute_feature_effect_rows(
        feature_id=0,
        context_records=[record],
        example_bank=example_bank,
        model=_FakeModel(),
        sae=_FakeSAE(),
        requested_readouts=["final_resid"],
        gram=gram,
        batch_size=1,
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        normalization_eps=1.0e-12,
        tau_zero=1.0e-12,
    )

    assert rows["final_resid"] == []
    assert stats["final_resid"]["retained_mask"] == [False]
    assert stats["final_resid"][counter] == 1


def test_final_resid_rows_use_one_mask_and_exact_gram_norm(monkeypatch) -> None:
    records = [_context(0), _context(1), _context(2)]
    example_bank = {
        idx: {
            "z": torch.ones(1),
            "readouts": {"final_resid": torch.tensor([1.0, 1.0])},
            "meta": {},
        }
        for idx in range(3)
    }

    def fake_ablation(**kwargs):
        assert kwargs["position_ids"].tolist() == [[0, 1], [0, 1], [0, 1]]
        assert kwargs["target_positions"] == [1, 1, 1]
        return {
            "final_resid": [
                torch.tensor([2.0, 3.0]),
                torch.tensor([float("nan"), 3.0]),
                torch.tensor([1.0, 1.0]),
            ]
        }

    monkeypatch.setattr(effects_module, "run_ablation_readouts_batch", fake_ablation)
    monkeypatch.setattr(
        effects_module,
        "effect_direction",
        lambda delta, magnitude: torch.tensor(
            [0.447213601234, 0.894427192345], dtype=torch.float64
        ),
    )

    rows_by_readout, stats_by_readout = compute_feature_effect_rows(
        feature_id=0,
        context_records=records,
        example_bank=example_bank,
        model=_FakeModel(),
        sae=_FakeSAE(),
        requested_readouts=["final_resid"],
        gram=torch.eye(2),
        batch_size=8,
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        normalization_eps=1.0e-12,
        tau_zero=1.0e-6,
    )

    rows = rows_by_readout["final_resid"]
    stats = stats_by_readout["final_resid"]
    assert len(rows) == 1
    assert torch.equal(rows[0]["delta"], torch.tensor([1.0, 2.0]))
    assert rows[0]["magnitude"] == pytest.approx(5.0**0.5)
    assert torch.sum(rows[0]["direction"].double().square()).item() == pytest.approx(
        1.0, abs=1.0e-7
    )
    persisted_direction = rows[0]["direction"]
    persisted_unit_q = torch.sum(
        (persisted_direction.double() @ torch.eye(2, dtype=torch.float64))
        * persisted_direction.double()
    )
    assert rows[0]["unit_gram_norm_error"] == abs(
        float(persisted_unit_q.item()) - 1.0
    )
    assert stats["retained_mask"] == [True, False, False]
    assert stats["candidate_identity"] == [
        {"attribute_label": "Country", "pair_role": "clean", "pair_index": idx}
        for idx in range(3)
    ]
    assert stats["kept"] == 1
    assert stats["skipped_nonfinite"] == 1
    assert stats["skipped_near_zero"] == 1


def test_final_resid_retains_cancellation_sensitive_float64_gram_row(
    monkeypatch,
) -> None:
    """Keep an exact-coordinate unit effect resolved only by float64 Gram arithmetic."""
    # Patch only the ablation boundary to produce the exact [1, -1] residual delta.
    record = _context(0)
    example_bank = {
        0: {
            "z": torch.ones(1),
            "readouts": {"final_resid": torch.zeros(2)},
            "meta": {},
        }
    }
    monkeypatch.setattr(
        effects_module,
        "run_ablation_readouts_batch",
        lambda **kwargs: {
            "final_resid": [torch.tensor([1.0, -1.0], dtype=torch.float32)]
        },
    )
    gram = torch.tensor(
        [[1.0e12, 1.0e12 - 0.5], [1.0e12 - 0.5, 1.0e12]],
        dtype=torch.float64,
    )

    rows_by_readout, stats_by_readout = compute_feature_effect_rows(
        feature_id=0,
        context_records=[record],
        example_bank=example_bank,
        model=_FakeModel(),
        sae=_FakeSAE(),
        requested_readouts=["final_resid"],
        gram=gram,
        batch_size=1,
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        normalization_eps=1.0e-12,
        tau_zero=1.0e-12,
    )

    rows = rows_by_readout["final_resid"]
    stats = stats_by_readout["final_resid"]
    assert len(rows) == 1
    assert rows[0]["magnitude"] == pytest.approx(1.0)
    assert stats["retained_mask"] == [True]
    assert rows[0]["unit_gram_norm_error"] < 1.0e-5


def test_final_resid_rows_use_one_ablation_call(monkeypatch) -> None:
    calls = []
    records = [_context(0)]
    example_bank = {
        0: {
            "z": torch.ones(1),
            "readouts": {"final_resid": torch.tensor([2.0, 0.0])},
            "meta": {},
        }
    }

    def fake_ablation(**kwargs):
        calls.append(
            {
                "requested_readouts": list(kwargs["requested_readouts"]),
                "position_ids": kwargs["position_ids"].clone(),
            }
        )
        return {"final_resid": [torch.tensor([4.0, 0.0])]}

    monkeypatch.setattr(effects_module, "run_ablation_readouts_batch", fake_ablation)

    rows_by_readout, stats_by_readout = compute_feature_effect_rows(
        feature_id=0,
        context_records=records,
        example_bank=example_bank,
        model=_FakeModel(),
        sae=_FakeSAE(),
        requested_readouts=["final_resid"],
        gram=torch.eye(2),
        batch_size=8,
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        normalization_eps=1.0e-12,
        tau_zero=1.0e-6,
    )

    assert calls[0]["requested_readouts"] == ["final_resid"]
    assert calls[0]["position_ids"].tolist() == [[0, 1]]
    assert rows_by_readout["final_resid"][0]["magnitude"] == pytest.approx(2.0)
    assert "direction" in rows_by_readout["final_resid"][0]
    assert stats_by_readout["final_resid"]["kept"] == 1


def test_oom_retry_preserves_prepared_row_semantics(monkeypatch) -> None:
    records = [
        _context(0, _prompt([1, 2], final_entity_token_pos=1)),
        _context(1, _prompt([3, 4], final_entity_token_pos=1)),
    ]
    example_bank = {
        0: {
            "z": torch.tensor([1.0]),
            "readouts": {"final_resid": torch.tensor([1.0, 0.0])},
            "meta": {},
        },
        1: {
            "z": torch.tensor([2.0]),
            "readouts": {"final_resid": torch.tensor([1.0, 0.0])},
            "meta": {},
        },
    }
    attempts = []

    def fake_ablation(**kwargs):
        attempts.append(
            {
                "tokens": kwargs["tokens"].clone(),
                "attention_mask": kwargs["attn"].clone(),
                "position_ids": kwargs["position_ids"].clone(),
                "target_positions": list(kwargs["target_positions"]),
                "z_batch": kwargs["z_batch"].clone(),
            }
        )
        if len(attempts) == 1:
            raise torch.cuda.OutOfMemoryError("forced")
        return {
            "final_resid": [
                torch.tensor([2.0, 0.0]) for _ in range(kwargs["tokens"].shape[0])
            ]
        }

    monkeypatch.setattr(effects_module, "run_ablation_readouts_batch", fake_ablation)

    rows_by_readout, stats_by_readout = compute_feature_effect_rows(
        feature_id=0,
        context_records=records,
        example_bank=example_bank,
        model=_FakeModel(),
        sae=_FakeSAE(),
        requested_readouts=["final_resid"],
        gram=torch.eye(2),
        batch_size=2,
        pad_token_id=0,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        normalization_eps=1.0e-12,
        tau_zero=1.0e-6,
    )

    assert len(attempts) == 3
    first_attempt_row0 = {
        key: value[0] for key, value in attempts[0].items() if key != "target_positions"
    }
    retry_row0 = {
        key: value[0] for key, value in attempts[1].items() if key != "target_positions"
    }
    for key in first_attempt_row0:
        assert torch.equal(first_attempt_row0[key], retry_row0[key])
    assert attempts[0]["target_positions"][0] == attempts[1]["target_positions"][0] == 1
    assert attempts[1]["z_batch"][0].item() == pytest.approx(1.0)
    assert attempts[2]["z_batch"][0].item() == pytest.approx(2.0)
    assert len(rows_by_readout["final_resid"]) == 2
    assert stats_by_readout["final_resid"]["oom_adjustments"] == [
        {"from": 2, "to": 1}
    ]


def test_run_compute_effect_accepts_induction_artifacts_and_prompt_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    reference_json = reference_dir / "induction.json"
    reference_json.write_text("{}")
    config = FEGAPipelineConfig(
        reference_json=reference_json,
        output_root=tmp_path / "out",
        device="cpu",
        source_kind="induction",
        entity_attribute_selection={"induction": ["rule_completion"]},
        phases=PhasesConfig(
            data_prep=DataPrepConfig(
                readouts=["final_resid"],
                gram_cache=True,
            ),
            compute_effect=ComputeEffectConfig(
                enabled=True,
                batch_size=8,
                min_coverage=1,
                effect_shard_size=8,
            ),
        ),
    )

    activations_dir = data_prep_activations_dir(config)
    activations_dir.mkdir(parents=True)
    tensors_path = activations_dir / "activations_tensors_0000.pt"
    meta_path = activations_dir / "activations_meta_0000.jsonl"
    z = torch.zeros(2, 8)
    z[0, 5] = 6.0
    z[1, 5] = 6.5
    torch.save(
        {
            "index": [0, 1],
            "z": z,
            "final_resid": torch.tensor([[2.0, 0.0], [3.0, 0.0]]),
        },
        tensors_path,
    )
    meta_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "prompt": "metadata-unused",
                        "attribute_label": "rule_completion",
                        "attribute_type": "rule_completion",
                        "entity_label": "induction",
                        "final_entity_token_pos": -1,
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "prompt": "from-meta",
                        "attribute_label": "rule_completion",
                        "attribute_type": "rule_completion",
                        "entity_label": "induction",
                        "final_entity_token_pos": -1,
                    }
                ),
            ]
        )
        + "\n"
    )
    (activations_dir / "activations_manifest.json").write_text(
        json.dumps(
            {
                "source_kind": "induction",
                "readouts": ["final_resid"],
                "tensor_keys": [
                    "index",
                    "z",
                    "final_resid",
                ],
                "total_records": 2,
                "chunk_count": 1,
                "positioning": _positioning(),
                "chunks": [
                    {
                        "chunk": 0,
                        "count": 2,
                        "start_index": 0,
                        "end_index": 1,
                        "tensors": str(tensors_path),
                        "meta": str(meta_path),
                    }
                ],
            }
        )
    )
    pairs_path = data_prep_pairs_path(config)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(
        json.dumps(
            {
                "rule_completion": {
                    "cause_base_prompts": [
                        {
                            "text": "from-pairs",
                            "template": "induction",
                            "attribute_type": "rule_completion",
                            "attribute_label": "rule_completion",
                            "entity_label": "induction",
                            "context_split": "0",
                            "entity_split": "0",
                            "input_ids": [9, 9],
                            "attention_mask": [1, 1],
                            "final_entity_token_pos": -1,
                            "first_generated_token_id": 101,
                            "attribute_generation": "A",
                            "is_correct": True,
                        }
                    ]
                }
            }
        )
    )
    select_dir = data_prep_select_dir(config)
    select_dir.mkdir(parents=True)
    (select_dir / "feature_contexts.json").write_text(
        json.dumps(
            {
                "5": [
                    {
                        "index": 0,
                        "pair_index": 0,
                        "pair_role": "cause_base_prompts",
                        "attribute_label": "rule_completion",
                        "source_kind": "induction",
                        "z": 5.5,
                    },
                    {
                        "index": 1,
                        "attribute_label": "rule_completion",
                        "source_kind": "induction",
                    },
                ]
            }
        )
    )
    gram_cache_tensor_path(config).parent.mkdir(parents=True)
    gram = torch.eye(2)
    torch.save(gram, gram_cache_tensor_path(config))
    readout = _FakeModel().get_output_embeddings().weight.detach()
    gram_cache_meta_path(config).write_text(
        json.dumps(
            {
                "checkpoint_identity": "fake-model",
                "readout_name": "final_resid",
                "hidden_width": 2,
                "gram_dtype": "float32",
                "construction_recipe": GRAM_CONSTRUCTION_RECIPE,
                "unembedding_fingerprint": unembedding_fingerprint(readout),
                "unembedding_dtype": str(readout.dtype),
                "unembedding_shape": [3, 2],
                "gram_shape": [2, 2],
                "gram_sha256": gram_fingerprint(gram),
            }
        )
    )

    captured = []

    def fake_compute_feature_effect_rows(**kwargs):
        records = list(kwargs["context_records"])
        captured.extend(
            [
                {
                    "index": record.index,
                    "pair_index": record.pair_index,
                    "feature_activation": record.feature_activation,
                    "prompt_text": record.prompt.text,
                    "input_ids": record.prompt.input_ids,
                }
                for record in records
            ]
        )
        return (
            {
                "final_resid": [
                    {
                        "context_index": record.index,
                        "pair_index": record.pair_index,
                        "pair_role": record.pair_role,
                        "attribute_label": record.attribute_label,
                        "feature_activation": record.feature_activation,
                        "delta": torch.tensor([2.0, 0.0]),
                        "magnitude": 2.0,
                        "direction": torch.tensor([1.0, 0.0]),
                        "unit_gram_norm_error": 0.0,
                    }
                    for record in records
                ],
            },
            {
                "final_resid": {
                    **_kept_stats(),
                    "kept": len(records),
                    "candidate_identity": [
                        {
                            "attribute_label": record.attribute_label,
                            "pair_role": record.pair_role,
                            "pair_index": record.pair_index,
                        }
                        for record in records
                    ],
                    "retained_mask": [True for record in records],
                },
            },
        )

    monkeypatch.setattr(
        runner_module, "compute_feature_effect_rows", fake_compute_feature_effect_rows
    )

    run_compute_effect(config, cast(ModelResources, _FakeResources(_FakeTokenizer())))

    assert captured == [
        {
            "index": 0,
            "pair_index": 0,
            "feature_activation": 5.5,
            "prompt_text": "from-pairs",
            "input_ids": [9, 9],
        },
    ]
    final_manifest = json.loads(
        effect_tensors_manifest_path(config, "final_resid").read_text()
    )
    assert final_manifest["inputs"]["source_chunk_count"] == 1
    final_shard = torch.load(
        compute_effect_readout_dir(config, "final_resid") / "effect_tensors_00000.pt",
        map_location="cpu",
    )
    assert final_shard["context_indices"].tolist() == [0]
    assert final_shard["feature_activations"].tolist() == [5.5]
    assert final_shard["candidate_identity"] == [[
        {
            "attribute_label": "rule_completion",
            "pair_role": "cause_base_prompts",
            "pair_index": 0,
        },
    ]]
    assert final_shard["retained_mask"] == [[True]]
