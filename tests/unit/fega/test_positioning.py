from __future__ import annotations

import pytest
import torch

from fega.core.positioning import (
    POSITIONING_SCHEMA_VERSION,
    build_padded_prompt_batch,
    require_compatible_positioning,
)
from sae_bench.evals.ravel.instance import Prompt


def _prompt(
    input_ids: list[int],
    *,
    attention_mask: list[int] | None = None,
    final_entity_token_pos: int | None = None,
) -> Prompt:
    return Prompt(
        text="x",
        template="",
        attribute_type="",
        attribute_label="Country",
        entity_label="city",
        context_split="",
        entity_split="",
        input_ids=input_ids,
        attention_mask=attention_mask,
        final_entity_token_pos=final_entity_token_pos,
        attribute_generation=None,
        first_generated_token_id=None,
        is_correct=True,
    )


def test_left_padding_uses_prompt_local_position_ids() -> None:
    batch = build_padded_prompt_batch(
        [_prompt([11, 12, 13]), _prompt([21])],
        device=torch.device("cpu"),
        pad_token_id=99,
        original_indices=[10, 11],
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
    )

    assert batch.input_ids.tolist() == [[11, 12, 13], [99, 99, 21]]
    assert batch.attention_mask.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch.position_ids.tolist() == [[0, 1, 2], [0, 0, 0]]
    assert batch.target_positions == [2, 2]
    assert batch.row_metadata(1) == {
        "positioning_schema_version": 1,
        "prompt_length": 1,
        "pad_length": 2,
        "raw_target_position": None,
        "unpadded_target_position": 0,
        "padded_target_position": 2,
        "original_index": 11,
    }


def test_negative_ravel_target_normalizes_before_padded_conversion() -> None:
    batch = build_padded_prompt_batch(
        [_prompt([1, 2, 3, 4], final_entity_token_pos=-2), _prompt([5])],
        device="cpu",
        pad_token_id=0,
        original_indices=None,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
    )

    assert batch.unpadded_target_positions == [2, 0]
    assert batch.pad_lengths == [0, 3]
    assert batch.target_positions == [2, 3]
    assert batch.metadata["positioning"]["raw_target_positions"] == [-2, None]


def test_positive_target_converts_to_padded_physical_index() -> None:
    batch = build_padded_prompt_batch(
        [_prompt([1, 2, 3]), _prompt([4, 5], final_entity_token_pos=1)],
        device="cpu",
        pad_token_id=7,
        original_indices=[0, 1],
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
    )

    assert batch.unpadded_target_positions == [2, 1]
    assert batch.target_positions == [2, 2]
    assert batch.attention_mask[1, batch.target_positions[1]].item() == 1


def test_invalid_target_reports_original_index() -> None:
    with pytest.raises(ValueError, match="original index 42.*outside prompt length 2"):
        build_padded_prompt_batch(
            [_prompt([1, 2], final_entity_token_pos=-3)],
            device="cpu",
            pad_token_id=0,
            original_indices=[42],
            positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        )


def test_attention_mask_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="attention_mask length mismatch.*original index 7"):
        build_padded_prompt_batch(
            [_prompt([1, 2], attention_mask=[1])],
            device="cpu",
            pad_token_id=0,
            original_indices=[7],
            positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        )


def test_pad_token_id_is_configurable_and_recorded() -> None:
    batch = build_padded_prompt_batch(
        [_prompt([1]), _prompt([2, 3])],
        device="cpu",
        pad_token_id=123,
        original_indices=None,
        positioning_schema_version=POSITIONING_SCHEMA_VERSION,
    )

    positioning = batch.metadata["positioning"]
    assert batch.input_ids.tolist()[0] == [123, 1]
    assert positioning["pad_token_id"] == 123
    assert positioning["padding_side"] == "left"
    assert positioning["position_id_scheme"] == "prompt_local_attention_cumsum"


def test_require_compatible_positioning_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError, match="lacks positioning metadata.*Rerun data_prep"):
        require_compatible_positioning(None, artifact_name="manifest")
