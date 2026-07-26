import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.main import LLM_NAME_MAP


@dataclass
class FEGAConfig:
    """Scoped config for FEGA activation collection."""

    reference_json: Path
    eval_config: RAVELEvalConfig
    device: str
    output_dir: Path
    save_chunk_size: int | None = 512
    single_file: bool = False

    @property
    def hf_model_name(self) -> str:
        return LLM_NAME_MAP.get(
            self.eval_config.model_name, self.eval_config.model_name
        )

    @classmethod
    def from_reference(
        cls,
        ref_path: str | Path,
        device: str,
        output_dir: str | Path,
        entity_attribute_selection: dict | None = None,
        save_chunk_size: int | None = 512,
        single_file: bool = False,
        random_seed: int | None = 42,
        llm_batch_size_override: int | None = None,
    ):
        ref_path = Path(ref_path)
        with open(ref_path) as f:
            data: Dict[str, Any] = json.load(f)
        eval_config_raw: Dict[str, Any] = data.get("eval_config", {})
        eval_config = RAVELEvalConfig(**eval_config_raw)
        if entity_attribute_selection:
            eval_config.entity_attribute_selection = entity_attribute_selection
        # Allow overriding/forcing reproducibility seed.
        eval_config.random_seed = random_seed
        if llm_batch_size_override is not None:
            eval_config.llm_batch_size = llm_batch_size_override
        return cls(
            reference_json=ref_path,
            eval_config=eval_config,
            device=device,
            output_dir=Path(output_dir),
            save_chunk_size=save_chunk_size,
            single_file=single_file,
        )
