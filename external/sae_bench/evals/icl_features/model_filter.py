from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


MODEL_ALIASES = {
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
}


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


@dataclass
class CorrectnessResult:
    correct: bool
    target_first_token_id: int
    target_token_length: int
    predicted_first_token_id: int
    predicted_first_token: str


class GemmaFirstTokenFilter:
    """Filter prompts by exact first-answer-token correctness."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype_name: str,
        batch_size: int,
        answer_prefix: str,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_name = resolve_model_name(model_name)
        dtype = getattr(torch, dtype_name)
        self.model_name = model_name
        self.resolved_model_name = resolved_name
        self.device = device
        self.batch_size = batch_size
        self.answer_prefix = answer_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither a pad token nor an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def answer_token_ids(self, answer: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.tokenizer.encode(
                f"{self.answer_prefix}{answer}",
                add_special_tokens=False,
            )
        ]

    @torch.no_grad()
    def score(self, rows: list[dict[str, Any]]) -> list[CorrectnessResult]:
        if not rows:
            return []
        output: list[CorrectnessResult] = []
        for start in range(0, len(rows), self.batch_size):
            batch_rows = rows[start : start + self.batch_size]
            target_token_lists = [
                self.answer_token_ids(str(row["answer"])) for row in batch_rows
            ]
            if any(not token_ids for token_ids in target_token_lists):
                raise ValueError("Tokenizer produced an empty answer token sequence")
            encoded = self.tokenizer(
                [str(row["prompt"]) for row in batch_rows],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            final_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=self.device)
            predicted_ids = (
                model_output.logits[batch_indices, final_positions, :]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            for target_ids, predicted_id in zip(
                target_token_lists, predicted_ids, strict=True
            ):
                predicted_id = int(predicted_id)
                output.append(
                    CorrectnessResult(
                        correct=predicted_id == int(target_ids[0]),
                        target_first_token_id=int(target_ids[0]),
                        target_token_length=len(target_ids),
                        predicted_first_token_id=predicted_id,
                        predicted_first_token=self.tokenizer.decode([predicted_id]),
                    )
                )
            del model_output, input_ids, attention_mask
        return output


def single_token_words(tokenizer: Any, words: list[str], *, limit: int) -> list[str]:
    """Return deterministic full words represented by one leading-space token."""
    selected = []
    seen = set()
    for raw_word in words:
        word = str(raw_word).strip().lower()
        if not word.isalpha() or word in seen:
            continue
        token_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        seen.add(word)
        selected.append(word)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"Only found {len(selected)} one-token words; requested {limit}"
        )
    return selected


def load_brown_words() -> list[str]:
    try:
        from nltk.corpus import brown

        return list(brown.words())
    except (ImportError, LookupError) as exc:
        raise RuntimeError(
            "Brown-corpus generation requires `nltk` and its Brown corpus. "
            "Install nltk and run `python -m nltk.downloader brown` once."
        ) from exc
