from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sae_bench.evals.ravel.instance import Prompt

from fega.core.compute_effect.artifacts import EffectArtifactWriter
from fega.core.compute_effect.effects import (
    EffectContextRecord,
    compute_feature_effect_rows,
)
from fega.core.data_prep.collection import run_sae_reconstruction
from fega.core.positioning import (
    POSITIONING_SCHEMA_VERSION,
    build_padded_prompt_batch,
)

READOUTS = ["final_resid"]
FEATURE_ID = 0
EFFECT_DELTA_ATOL = 1.0e-3
EFFECT_DELTA_RTOL = 1.0e-3


class _IdentitySAE(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        self.cfg = SimpleNamespace(hook_layer=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.W_dec.device, dtype=self.W_dec.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z


def _prompt(tokenizer, text: str) -> Prompt:
    encoded = tokenizer(text, return_attention_mask=True, add_special_tokens=False)
    return Prompt(
        text=text,
        template="",
        attribute_type="Country",
        attribute_label="Country",
        entity_label=text.split()[0],
        context_split="",
        entity_split="",
        input_ids=list(encoded.input_ids),
        attention_mask=list(encoded.attention_mask),
        final_entity_token_pos=-1,
        attribute_generation=None,
        first_generated_token_id=None,
        is_correct=True,
    )


def _warmup_cuda_blas(model, device: torch.device) -> None:
    if device.type != "cuda":
        return
    dtype = next(model.parameters()).dtype
    hidden_size = int(model.config.n_embd)
    hidden = torch.zeros((3, hidden_size), device=device, dtype=dtype)
    _ = torch.nn.functional.linear(hidden, model.lm_head.weight)
    input_ids = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(3, device=device, dtype=torch.long).unsqueeze(0)
    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=True,
    ).logits
    torch.cuda.synchronize()


@torch.no_grad()
def _collect_examples_for_batch_size(
    *,
    model,
    sae: _IdentitySAE,
    prompts: list[Prompt],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[int, dict]:
    examples: dict[int, dict] = {}
    start = 0
    while start < len(prompts):
        batch_slice = list(enumerate(prompts[start : start + batch_size], start=start))
        batch_chunk = sorted(batch_slice, key=lambda kv: len(kv[1].input_ids))
        prompt_batch = build_padded_prompt_batch(
            [prompt for _, prompt in batch_chunk],
            device=device,
            pad_token_id=pad_token_id,
            original_indices=[idx for idx, _ in batch_chunk],
            positioning_schema_version=POSITIONING_SCHEMA_VERSION,
        )
        reps, zs, readouts = run_sae_reconstruction(
            model,
            sae,
            prompt_batch.input_ids,
            prompt_batch.attention_mask,
            prompt_batch.target_positions,
            position_ids=prompt_batch.position_ids,
            readouts=READOUTS,
        )
        for local_idx, (orig_idx, _) in enumerate(batch_chunk):
            examples[orig_idx] = {
                "x": reps[local_idx].detach().cpu(),
                "z": zs[local_idx].detach().cpu(),
                "readouts": {
                    readout: values[local_idx].detach().cpu()
                    for readout, values in readouts.items()
                },
                "meta": {"index": orig_idx, "pair_index": orig_idx},
            }
        start += batch_size
    return examples


def _context_records(prompts: list[Prompt], examples: dict[int, dict]) -> list[EffectContextRecord]:
    return [
        EffectContextRecord(
            index=idx,
            pair_index=idx,
            pair_role="cause_base_prompts",
            attribute_label="Country",
            feature_activation=float(examples[idx]["z"][FEATURE_ID].item()),
            prompt=prompt,
        )
        for idx, prompt in enumerate(prompts)
    ]


def _write_delta_artifacts(
    output_dir: Path,
    rows_by_readout: dict[str, list[dict]],
    stats_by_readout: dict[str, dict],
) -> dict[tuple[str, int, int], torch.Tensor]:
    deltas: dict[tuple[str, int, int], torch.Tensor] = {}
    for readout, rows in rows_by_readout.items():
        stats = stats_by_readout[readout]
        writer = EffectArtifactWriter(
            output_dir / readout,
            shard_size=1024,
            include_magnitude_direction=(readout == "final_resid"),
        )
        writer.add_feature_rows(
            FEATURE_ID,
            rows,
            candidate_identity=stats["candidate_identity"],
            retained_mask=stats["retained_mask"],
        )
        writer.flush()
        shard_path = output_dir / readout / "effect_tensors_00000.pt"
        shard = torch.load(shard_path, map_location="cpu")
        for row_idx, context_index in enumerate(shard["context_indices"].tolist()):
            deltas[(readout, FEATURE_ID, int(context_index))] = shard["delta"][row_idx]
    return deltas


def _assert_deltas_close(
    *,
    baseline: dict[tuple[str, int, int], torch.Tensor],
    candidate: dict[tuple[str, int, int], torch.Tensor],
    baseline_batch_size: int,
    candidate_batch_size: int,
    artifact_paths: dict[int, Path],
    device: torch.device,
) -> None:
    assert baseline.keys() == candidate.keys()
    for readout in READOUTS:
        keys = sorted(key for key in baseline if key[0] == readout)
        assert keys, f"No compared rows for readout={readout}"
        expected = torch.stack([baseline[key] for key in keys])
        actual = torch.stack([candidate[key] for key in keys])
        diff = actual - expected
        max_error = float(torch.max(torch.abs(diff)).item())
        rel_error = float(
            torch.linalg.vector_norm(diff)
            / (torch.linalg.vector_norm(expected) + 1.0e-12)
        )
        # A100 fp32 GPT-2 singleton vs padded-batch forwards differ at roughly
        # 2e-4 from GEMM shape changes; the missing-position_ids regression is
        # orders of magnitude larger on left-padded prompts.
        assert torch.allclose(
            actual,
            expected,
            rtol=EFFECT_DELTA_RTOL,
            atol=EFFECT_DELTA_ATOL,
        ), (
            f"batch invariance mismatch: readout={readout}, "
            f"baseline_batch_size={baseline_batch_size}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"max_abs_error={max_error:.6e}, relative_error={rel_error:.6e}, "
            f"rtol={EFFECT_DELTA_RTOL:.1e}, atol={EFFECT_DELTA_ATOL:.1e}, "
            f"compared_rows={len(keys)}, model=gpt2, device={device}, "
            f"baseline_artifact={artifact_paths[baseline_batch_size]}, "
            f"candidate_artifact={artifact_paths[candidate_batch_size]}"
        )


@pytest.mark.local_gpt2_gpu
def test_local_gpt2_effect_deltas_are_batch_size_invariant(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "CUDA is required for opted-in local GPT-2 smoke."

    from transformers import AutoTokenizer, GPT2LMHeadModel

    device = torch.device("cuda")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).to(device)
    except OSError as exc:
        pytest.fail(f"Missing local GPT-2 resources: {exc}")
    model.eval()
    _warmup_cuda_blas(model, device)
    sae = _IdentitySAE(device=device, dtype=next(model.parameters()).dtype)
    sae.eval()
    pad_token_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    prompts = [
        _prompt(tokenizer, "Paris is in"),
        _prompt(tokenizer, "The city of Berlin is located in"),
        _prompt(tokenizer, "Tokyo belongs to the country of"),
    ]
    artifact_paths: dict[int, Path] = {}
    deltas_by_batch_size: dict[int, dict[tuple[str, int, int], torch.Tensor]] = {}
    hidden_size = int(model.config.n_embd)
    gram = torch.eye(hidden_size, dtype=torch.float32)
    phase = "startup"
    try:
        for batch_size in (1, 8, 256):
            phase = f"batch_size={batch_size} data_prep"
            examples = _collect_examples_for_batch_size(
                model=model,
                sae=sae,
                prompts=prompts,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                device=device,
            )
            torch.cuda.empty_cache()
            phase = f"batch_size={batch_size} compute_effect"
            rows_by_readout, stats_by_readout = compute_feature_effect_rows(
                feature_id=FEATURE_ID,
                context_records=_context_records(prompts, examples),
                example_bank=examples,
                model=model,
                sae=sae,
                requested_readouts=READOUTS,
                gram=gram,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                positioning_schema_version=POSITIONING_SCHEMA_VERSION,
                normalization_eps=1.0e-12,
                tau_zero=0.0,
            )
            phase = f"batch_size={batch_size} artifact_write"
            artifact_paths[batch_size] = tmp_path / f"batch_{batch_size}"
            deltas_by_batch_size[batch_size] = _write_delta_artifacts(
                artifact_paths[batch_size], rows_by_readout, stats_by_readout
            )
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "CUBLAS_STATUS_NOT_INITIALIZED" in str(exc):
            if os.environ.get("FEGA_RERAISE_CUBLAS") == "1":
                raise
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            pytest.skip(
                "CUDA/cuBLAS failed during local GPT-2 smoke "
                f"at {phase}: allocated={allocated}, reserved={reserved}: {exc}"
            )
        raise

    for candidate_batch_size in (8, 256):
        _assert_deltas_close(
            baseline=deltas_by_batch_size[1],
            candidate=deltas_by_batch_size[candidate_batch_size],
            baseline_batch_size=1,
            candidate_batch_size=candidate_batch_size,
            artifact_paths=artifact_paths,
            device=device,
        )
