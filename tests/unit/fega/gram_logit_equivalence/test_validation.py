"""Vital per-feature contracts for Gram/logit equivalence validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from fega.core.gram_logit_equivalence import (
    evaluate_gram_logit_equivalence,
    evaluate_grouped_gram_logit_equivalence,
)
from scripts.validation import validate_fega_gram_logit_equivalence as validation_runner


@pytest.fixture
def grouped_equivalence_bundle() -> dict[str, Any]:
    """Build two distinct exact feature clouds with shared float32 readout tensors."""
    # Use clouds whose row counts and directions make accidental concatenation visible.
    unembedding = torch.tensor(
        [[1.0, 0.25], [0.5, 2.0], [-1.0, 0.75]], dtype=torch.float32
    )
    clouds = {
        9: torch.tensor(
            [[1.0, 2.0], [-2.0, 0.5], [0.25, -1.0]], dtype=torch.float32
        ),
        2: torch.tensor(
            [[3.0, -0.5], [1.5, 0.25]], dtype=torch.float32
        ),
    }
    feature_groups = {}
    for feature_id, hidden_deltas in clouds.items():
        explicit = hidden_deltas @ unembedding.T
        feature_groups[feature_id] = {
            "hidden_deltas": hidden_deltas,
            "explicit_logit_deltas": explicit,
            "returned_model_output_deltas": explicit.clone(),
            "row_identities": [
                {
                    "context_index": feature_id * 10 + row,
                    "attribute_label": "Country",
                    "pair_role": "positive",
                    "pair_index": row,
                }
                for row in range(hidden_deltas.shape[0])
            ],
        }
    return {
        "feature_groups": feature_groups,
        "unembedding": unembedding,
        "gram": unembedding.T @ unembedding,
        "expected_source_fingerprint": "deterministic-source-v1",
        "observed_source_fingerprint": "deterministic-source-v1",
    }


def test_exact_grouped_float32_clouds_pass_independently(
    grouped_equivalence_bundle: dict[str, Any],
) -> None:
    """Require numeric ordering and a complete passing result for each feature cloud."""
    # Evaluate the two clouds and assert their independent row identities are retained.
    result = evaluate_grouped_gram_logit_equivalence(**grouped_equivalence_bundle)

    assert result["status"] == "pass"
    assert result["observed_feature_ids"] == [2, 9]
    assert len(result["per_feature"]["2"]["row_identities"]) == 2
    assert len(result["per_feature"]["9"]["row_identities"]) == 3
    assert all(
        feature["status"] == "pass" for feature in result["per_feature"].values()
    )


def test_explicit_logit_profile_never_allocates_vocabulary_sized_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit-logit profiling bounded by rows rather than vocabulary width."""
    # Materialize the exact two-row, two-hidden, 64-vocabulary bundle before patching.
    retained_row_count = 2
    hidden_deltas = torch.eye(retained_row_count, dtype=torch.float64)
    unembedding = torch.zeros((64, 2), dtype=torch.float64)
    unembedding[:2] = hidden_deltas
    explicit_logit_deltas = hidden_deltas @ unembedding.T
    gram = unembedding.T @ unembedding
    returned_model_output_deltas = explicit_logit_deltas.clone()
    original_eye = torch.eye

    def bounded_eye(dimension: int, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Reject identities wider than the retained validation cloud."""
        # Permit the legitimate 2x2 centering identity and reject wider allocations.
        requested_dimensions = (dimension,) + (
            (args[0],) if args and isinstance(args[0], int) else ()
        )
        if any(size > retained_row_count for size in requested_dimensions):
            raise AssertionError("vocabulary-sized identity")
        return original_eye(dimension, *args, **kwargs)

    monkeypatch.setattr(
        "fega.core.gram_logit_equivalence.torch.eye",
        bounded_eye,
    )

    result = evaluate_gram_logit_equivalence(
        hidden_deltas=hidden_deltas,
        explicit_logit_deltas=explicit_logit_deltas,
        unembedding=unembedding,
        gram=gram,
        returned_model_output_deltas=returned_model_output_deltas,
        expected_source_fingerprint="deterministic-source-v1",
        observed_source_fingerprint="deterministic-source-v1",
    )

    assert result["status"] == "pass"


def test_perturbed_gram_fails_grouped_geometry_not_reconstruction(
    grouped_equivalence_bundle: dict[str, Any],
) -> None:
    """Detect shared Gram corruption without hiding exact linear reconstruction."""
    # Perturb only the Gram so both feature-local reconstruction checks remain exact.
    bundle = dict(grouped_equivalence_bundle)
    bundle["gram"] = grouped_equivalence_bundle["gram"].clone()
    bundle["gram"][0, 0] += 0.5

    result = evaluate_grouped_gram_logit_equivalence(**bundle)

    assert result["status"] == "fail"
    for feature in result["per_feature"].values():
        assert feature["checks"]["reconstruction"]["passed"] is True
        assert any(
            feature["checks"][name]["passed"] is False
            for name in ("norms", "inner_products", "cosines", "c_ray")
        )


def test_negative_hidden_quadratic_form_raises_instead_of_zero_norm() -> None:
    """Reject an invalid hidden quadratic form instead of clamping its norm."""
    # Supply a finite row whose Gram quadratic form is exactly negative.
    identity = torch.eye(2, dtype=torch.float32)
    hidden_deltas = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ArithmeticError, match="negative final_resid Gram quadratic form"):
        evaluate_gram_logit_equivalence(
            hidden_deltas=hidden_deltas,
            explicit_logit_deltas=hidden_deltas.clone(),
            unembedding=identity,
            gram=torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float32)),
            returned_model_output_deltas=hidden_deltas.clone(),
            expected_source_fingerprint="deterministic-source-v1",
            observed_source_fingerprint="deterministic-source-v1",
        )


def test_indefinite_gram_is_rejected_without_validation_cutoff() -> None:
    """Reject an indefinite Gram without a validation-only cutoff."""
    # Keep both row norms positive while making the unit Gram indefinite.
    identity = torch.eye(2, dtype=torch.float32)
    invalid_gram = torch.tensor(
        [[1.0, 1.000005], [1.000005, 1.0]], dtype=torch.float32
    )

    with pytest.raises(ValueError, match="negative eigenvalue"):
        evaluate_gram_logit_equivalence(
            hidden_deltas=identity,
            explicit_logit_deltas=identity,
            unembedding=identity,
            gram=invalid_gram,
            returned_model_output_deltas=identity,
            expected_source_fingerprint="deterministic-source-v1",
            observed_source_fingerprint="deterministic-source-v1",
        )


def test_grouped_source_fingerprint_mismatch_fails_closed(
    grouped_equivalence_bundle: dict[str, Any],
) -> None:
    """Reject all grouped evaluation before science when source fingerprints differ."""
    # Change only the observed source binding and require an input-validation failure.
    bundle = dict(grouped_equivalence_bundle)
    bundle["observed_source_fingerprint"] = "different-source-v2"

    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        evaluate_grouped_gram_logit_equivalence(**bundle)


def test_returned_output_disagreement_is_diagnostic_only_per_feature(
    grouped_equivalence_bundle: dict[str, Any],
) -> None:
    """Keep returned-output disagreement visible but outside aggregate pass status."""
    # Perturb one feature's returned output without changing any scientific tensor.
    bundle = copy.deepcopy(grouped_equivalence_bundle)
    bundle["feature_groups"][9]["returned_model_output_deltas"] += 1.0

    result = evaluate_grouped_gram_logit_equivalence(**bundle)

    assert result["status"] == "pass"
    diagnostic = result["per_feature"]["9"]["diagnostics"]
    assert diagnostic["returned_model_output_delta_vs_linear"]["equivalent"] is False
    assert result["per_feature"]["2"]["status"] == "pass"


def test_selection_computes_in_numeric_order_and_stops_at_bounded_sample() -> None:
    """Select first retained rows from lowest eligible IDs without a full sweep."""
    # Record computation order while making the first numeric feature undersized.
    contexts = {9: [{}], 2: [{}], 1: [{}], 5: [{}], 12: [{}]}
    retained_counts = {1: 7, 2: 9, 5: 8, 9: 8, 12: 8}
    computed = []

    def compute_candidate(feature_id: int, raw_contexts: list[dict]) -> dict:
        """Return deterministic in-memory rows for one candidate feature."""
        # Preserve the supplied context object and expose ordered row ordinals.
        computed.append(feature_id)
        assert raw_contexts is contexts[feature_id]
        return {
            "feature_id": feature_id,
            "rows": [
                {"ordinal": ordinal} for ordinal in range(retained_counts[feature_id])
            ],
        }

    selected = validation_runner.select_bounded_feature_rows(
        contexts, compute_candidate, feature_count=2, row_count=8
    )

    assert computed == [1, 2, 5]
    assert [bundle["feature_id"] for bundle in selected] == [2, 5]
    assert [[row["ordinal"] for row in bundle["rows"]] for bundle in selected] == [
        list(range(8)),
        list(range(8)),
    ]


@pytest.mark.parametrize("changed_input", ["activation_tensor", "activation_meta", "pairs"])
def test_scientific_input_fingerprint_tracks_referenced_input_bytes(
    tmp_path: Path, changed_input: str
) -> None:
    """Bind R12 provenance to every referenced activation and prompt input byte."""
    # Build one manifest-backed shard pair and mutate exactly one referenced input.
    activations_dir = tmp_path / "activations"
    activations_dir.mkdir()
    tensor_path = activations_dir / "activations_tensors_0000.pt"
    tensor_path.write_bytes(b"tensor-shard-v1")
    meta_path = activations_dir / "activations_meta_0000.jsonl"
    meta_path.write_text('{"index": 0}\n')
    activation_manifest_path = activations_dir / "activations_manifest.json"
    activation_manifest_path.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "tensors": str(tensor_path),
                        "meta": str(meta_path),
                    }
                ]
            }
        )
    )
    feature_contexts_path = tmp_path / "feature_contexts.json"
    feature_contexts_path.write_text('{"1": [{"index": 0}]}')
    pairs_path = tmp_path / "pairs_full.json"
    pairs_path.write_text('{"Country": {"cause_base_prompts": []}}')
    fingerprint_args = {
        "activation_manifest_path": activation_manifest_path,
        "activations_dir": activations_dir,
        "feature_contexts_path": feature_contexts_path,
        "pairs_path": pairs_path,
    }
    before = validation_runner._scientific_input_fingerprint(**fingerprint_args)

    changed_path = {
        "activation_tensor": tensor_path,
        "activation_meta": meta_path,
        "pairs": pairs_path,
    }[changed_input]
    changed_path.write_bytes(changed_path.read_bytes() + b"\n")

    after = validation_runner._scientific_input_fingerprint(**fingerprint_args)

    assert after != before


def test_gemma_softcap_returned_deltas_differ_from_raw_linear_deltas() -> None:
    """Expose Gemma's returned-output transform under deterministic saturation."""
    # Compare the explicit linear delta with Gemma's tanh-softcapped output delta.
    unembedding = torch.tensor(
        [[15.0, 15.0], [20.0, 20.0]], dtype=torch.float32
    )
    base_head = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    ablated_head = torch.tensor([[3.0, 3.0]], dtype=torch.float32)
    linear_delta = (ablated_head - base_head) @ unembedding.T
    softcap = 30.0
    returned_delta = softcap * (
        torch.tanh((ablated_head @ unembedding.T) / softcap)
        - torch.tanh((base_head @ unembedding.T) / softcap)
    )

    assert not torch.allclose(returned_delta, linear_delta, rtol=1e-4, atol=1e-4)
