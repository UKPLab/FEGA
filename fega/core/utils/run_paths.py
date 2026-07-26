from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from .misc import ensure_dir, seed_everything

if TYPE_CHECKING:
    from fega.core.config import FEGAConfig


@dataclass
class RunPaths:
    """Resolved output paths for a single FEGA run."""

    collect_dir: Path
    activations_dir: Path
    manifest_path: Path
    run_pairs_path: Path
    pairs_cache_path: Path
    meta_tpl: str = "activations_meta_{:04d}.jsonl"
    tensor_tpl: str = "activations_tensors_{:04d}.pt"


def setup_run(fega_cfg: "FEGAConfig", entity_class: str, attrs: List[str]) -> RunPaths:
    """Seed randomness (if configured) and return run-relative output paths."""
    seed = getattr(fega_cfg.eval_config, "random_seed", None)
    if seed is not None:
        seed_everything(seed)

    collect_dir = fega_cfg.output_dir / "collect"
    activations_dir = collect_dir / "activations"
    ensure_dir(activations_dir)
    pairs_cache_path = collect_dir / "pairs_full.json"
    return RunPaths(
        collect_dir=collect_dir,
        activations_dir=activations_dir,
        manifest_path=activations_dir / "activations_manifest.json",
        run_pairs_path=collect_dir / "pairs_full.json",
        pairs_cache_path=pairs_cache_path,
    )
