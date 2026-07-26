from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from sae_bench.evals.ravel.instance import Prompt

POSITIONING_SCHEMA_VERSION = 1
POSITIONING_METADATA_KEY = "positioning"
PADDING_SIDE = "left"
POSITION_ID_SCHEME = "prompt_local_attention_cumsum"
TARGET_POSITION_SCHEME = "ravel_prompt_relative_to_padded_physical"


@dataclass(frozen=True)
class PaddedPromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_positions: list[int]
    lengths: list[int]
    pad_lengths: list[int]
    raw_target_positions: list[int | None]
    unpadded_target_positions: list[int]
    original_indices: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def row_metadata(self, row_idx: int) -> dict[str, Any]:
        data: dict[str, Any] = {
            "positioning_schema_version": self.metadata[POSITIONING_METADATA_KEY][
                "schema_version"
            ],
            "prompt_length": self.lengths[row_idx],
            "pad_length": self.pad_lengths[row_idx],
            "raw_target_position": self.raw_target_positions[row_idx],
            "unpadded_target_position": self.unpadded_target_positions[row_idx],
            "padded_target_position": self.target_positions[row_idx],
        }
        if self.original_indices is not None:
            data["original_index"] = self.original_indices[row_idx]
        return data


def build_padded_prompt_batch(
    prompts: Sequence[Prompt],
    *,
    device: torch.device | str,
    pad_token_id: int,
    original_indices: Sequence[int] | None,
    positioning_schema_version: int,
) -> PaddedPromptBatch:
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("Cannot build a padded prompt batch from zero prompts.")
    if int(positioning_schema_version) != POSITIONING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported positioning schema version {positioning_schema_version}; "
            f"expected {POSITIONING_SCHEMA_VERSION}."
        )
    index_list = None if original_indices is None else [int(idx) for idx in original_indices]
    if index_list is not None and len(index_list) != len(prompt_list):
        raise ValueError(
            "original_indices length must match prompts length: "
            f"{len(index_list)} != {len(prompt_list)}."
        )

    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    lengths: list[int] = []
    pad_lengths: list[int] = []
    raw_target_positions: list[int | None] = []
    unpadded_target_positions: list[int] = []
    target_positions: list[int] = []

    token_rows: list[list[int]] = []
    attn_rows: list[list[int]] = []
    for row_idx, prompt in enumerate(prompt_list):
        input_ids = _require_input_ids(prompt, row_idx, index_list)
        attention_mask = _attention_mask(prompt, input_ids, row_idx, index_list)
        token_rows.append(input_ids)
        attn_rows.append(attention_mask)
        lengths.append(len(input_ids))

    max_len = max(lengths)
    for row_idx, (prompt, input_ids, attention_mask) in enumerate(
        zip(prompt_list, token_rows, attn_rows)
    ):
        pad_len = max_len - len(input_ids)
        raw_pos = prompt.final_entity_token_pos
        unpadded_target = _normalize_target_position(
            raw_pos,
            prompt_len=len(input_ids),
            row_idx=row_idx,
            original_indices=index_list,
        )
        padded_target = pad_len + unpadded_target
        input_rows.append([int(pad_token_id)] * pad_len + input_ids)
        mask_rows.append([0] * pad_len + attention_mask)
        pad_lengths.append(pad_len)
        raw_target_positions.append(raw_pos)
        unpadded_target_positions.append(unpadded_target)
        target_positions.append(padded_target)

    input_ids_t = torch.tensor(input_rows, device=device, dtype=torch.long)
    attention_mask_t = torch.tensor(mask_rows, device=device, dtype=torch.long)
    position_ids_t = attention_mask_t.long().cumsum(dim=-1) - 1
    position_ids_t = position_ids_t.masked_fill(attention_mask_t == 0, 0)

    _validate_batch_invariants(
        input_ids=input_ids_t,
        attention_mask=attention_mask_t,
        position_ids=position_ids_t,
        target_positions=target_positions,
        unpadded_target_positions=unpadded_target_positions,
        lengths=lengths,
    )
    positioning = build_positioning_metadata(
        pad_token_id=int(pad_token_id),
        positioning_schema_version=positioning_schema_version,
        batch_size_provenance=None,
    )
    positioning.update(
        {
            "lengths": lengths,
            "pad_lengths": pad_lengths,
            "raw_target_positions": raw_target_positions,
            "unpadded_target_positions": unpadded_target_positions,
            "padded_target_positions": target_positions,
            "original_indices": index_list,
        }
    )
    return PaddedPromptBatch(
        input_ids=input_ids_t,
        attention_mask=attention_mask_t,
        position_ids=position_ids_t,
        target_positions=target_positions,
        lengths=lengths,
        pad_lengths=pad_lengths,
        raw_target_positions=raw_target_positions,
        unpadded_target_positions=unpadded_target_positions,
        original_indices=index_list,
        metadata={POSITIONING_METADATA_KEY: positioning},
    )


def build_positioning_metadata(
    *,
    pad_token_id: int,
    positioning_schema_version: int,
    batch_size_provenance: Mapping[str, Any] | None,
    source_data_prep_positioning: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": int(positioning_schema_version),
        "padding_side": PADDING_SIDE,
        "pad_token_id": int(pad_token_id),
        "position_id_scheme": POSITION_ID_SCHEME,
        "target_position_scheme": TARGET_POSITION_SCHEME,
    }
    if batch_size_provenance is not None:
        metadata["batch_size_provenance"] = dict(batch_size_provenance)
    if source_data_prep_positioning is not None:
        metadata["source_data_prep_positioning"] = dict(source_data_prep_positioning)
    return metadata


def require_compatible_positioning(
    positioning: Mapping[str, Any] | None, *, artifact_name: str
) -> dict[str, Any]:
    if not isinstance(positioning, Mapping):
        raise ValueError(
            f"{artifact_name} lacks positioning metadata. Rerun data_prep with "
            "current FEGA positioning support before running compute_effect."
        )
    expected = {
        "schema_version": POSITIONING_SCHEMA_VERSION,
        "padding_side": PADDING_SIDE,
        "position_id_scheme": POSITION_ID_SCHEME,
        "target_position_scheme": TARGET_POSITION_SCHEME,
    }
    for key, expected_value in expected.items():
        actual = positioning.get(key)
        if actual != expected_value:
            raise ValueError(
                f"{artifact_name} has incompatible positioning metadata: "
                f"{key}={actual!r}, expected {expected_value!r}. Rerun data_prep "
                "with current FEGA positioning support."
            )
    if "pad_token_id" not in positioning:
        raise ValueError(
            f"{artifact_name} positioning metadata is missing pad_token_id. "
            "Rerun data_prep with current FEGA positioning support."
        )
    return dict(positioning)


def _require_input_ids(
    prompt: Prompt, row_idx: int, original_indices: list[int] | None
) -> list[int]:
    if prompt.input_ids is None:
        raise ValueError(f"Prompt {_row_label(row_idx, original_indices)} has no input_ids.")
    input_ids = [int(token_id) for token_id in prompt.input_ids]
    if not input_ids:
        raise ValueError(f"Prompt {_row_label(row_idx, original_indices)} has empty input_ids.")
    return input_ids


def _attention_mask(
    prompt: Prompt,
    input_ids: list[int],
    row_idx: int,
    original_indices: list[int] | None,
) -> list[int]:
    if prompt.attention_mask is None:
        return [1] * len(input_ids)
    attention_mask = [int(value) for value in prompt.attention_mask]
    if len(attention_mask) != len(input_ids):
        raise ValueError(
            f"attention_mask length mismatch for prompt {_row_label(row_idx, original_indices)}: "
            f"got {len(attention_mask)} vs input_ids {len(input_ids)}."
        )
    return attention_mask


def _normalize_target_position(
    raw_pos: int | None,
    *,
    prompt_len: int,
    row_idx: int,
    original_indices: list[int] | None,
) -> int:
    if raw_pos is None:
        unpadded_target = prompt_len - 1
    elif int(raw_pos) < 0:
        unpadded_target = prompt_len + int(raw_pos)
    else:
        unpadded_target = int(raw_pos)
    if unpadded_target < 0 or unpadded_target >= prompt_len:
        raise ValueError(
            f"Target position {raw_pos!r} for prompt {_row_label(row_idx, original_indices)} "
            f"normalizes to {unpadded_target}, outside prompt length {prompt_len}."
        )
    return unpadded_target


def _validate_batch_invariants(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    target_positions: list[int],
    unpadded_target_positions: list[int],
    lengths: list[int],
) -> None:
    if position_ids.shape != input_ids.shape:
        raise RuntimeError(
            f"position_ids shape {tuple(position_ids.shape)} != input_ids shape {tuple(input_ids.shape)}."
        )
    if attention_mask.shape != input_ids.shape:
        raise RuntimeError(
            f"attention_mask shape {tuple(attention_mask.shape)} != input_ids shape {tuple(input_ids.shape)}."
        )
    padded_len = int(input_ids.shape[1])
    for row_idx, (padded_target, unpadded_target, prompt_len) in enumerate(
        zip(target_positions, unpadded_target_positions, lengths)
    ):
        if unpadded_target < 0 or unpadded_target >= prompt_len:
            raise RuntimeError(
                f"Unpadded target {unpadded_target} out of bounds for row {row_idx} length {prompt_len}."
            )
        if padded_target < 0 or padded_target >= padded_len:
            raise RuntimeError(
                f"Padded target {padded_target} out of bounds for row {row_idx} length {padded_len}."
            )
        if int(attention_mask[row_idx, padded_target].item()) != 1:
            raise RuntimeError(
                f"Padded target {padded_target} for row {row_idx} points at a masked token."
            )


def _row_label(row_idx: int, original_indices: list[int] | None) -> str:
    if original_indices is None:
        return str(row_idx)
    return f"{row_idx} (original index {original_indices[row_idx]})"
