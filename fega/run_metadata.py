from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from fega.config_schema import FEGAPipelineConfig


@dataclass
class RunMetadata:
    config_path: str
    resolved_config: dict[str, Any]
    global_seed: int | None = None
    stage_seeds: dict[str, int] = field(default_factory=dict)
    git_commit: str | None = None
    python_version: str = platform.python_version()
    cuda_version: str | None = None
    libraries: dict[str, str] = field(default_factory=dict)
    local_sae: dict[str, Any] | None = None


def write_run_metadata(path: Path, meta: RunMetadata) -> None:
    """Write run metadata to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(meta), f, indent=2)


def read_run_metadata(path: Path) -> RunMetadata:
    """Read run metadata from JSON."""
    with open(path) as f:
        raw = json.load(f)
    return RunMetadata(
        config_path=raw.get("config_path", ""),
        resolved_config=raw.get("resolved_config", {}),
        global_seed=raw.get("global_seed"),
        stage_seeds=raw.get("stage_seeds", {}),
        git_commit=raw.get("git_commit"),
        python_version=raw.get("python_version", platform.python_version()),
        cuda_version=raw.get("cuda_version"),
        libraries=raw.get("libraries", {}),
        local_sae=raw.get("local_sae"),
    )


def build_base_metadata(
    config_path: Path, resolved_config: FEGAPipelineConfig
) -> RunMetadata:
    """Collect basic environment metadata for a pipeline run."""
    resolved_dict = resolved_config.to_dict()
    local_sae = _build_local_sae_provenance(resolved_config)
    return RunMetadata(
        config_path=str(config_path),
        resolved_config=resolved_dict,
        global_seed=None,
        stage_seeds={},
        git_commit=_detect_git_commit(config_path),
        python_version=platform.python_version(),
        cuda_version=_detect_cuda_version(),
        libraries=_collect_library_versions(
            ["transformers", "torch", "sae_lens", "sae_bench"]
        ),
        local_sae=local_sae,
    )


def _build_local_sae_provenance(
    config: FEGAPipelineConfig,
) -> dict[str, Any] | None:
    source = (config.sae_source or "auto").lower()
    use_local = source == "local_checkpoint" or (
        source == "auto" and config.local_sae_checkpoint_path is not None
    )
    if not use_local:
        return None
    checkpoint_path = config.local_sae_checkpoint_path
    if checkpoint_path is None:
        return None
    payload: dict[str, Any] = {
        "source": "local_checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    if config.local_sae_resolved_config_path is not None:
        payload["resolved_config_path"] = str(config.local_sae_resolved_config_path)
    return payload


def _sha256_file(path: Path) -> str:
    hasher = sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _detect_git_commit(config_path: Path) -> str | None:
    """capture current Git commit hash for run metadata. This is vital to now which code version makes the outputs"""
    repo_root = _find_git_root(config_path.parent)
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _detect_cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return getattr(torch.version, "cuda", None)


def _collect_library_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            continue
    return versions


def _find_git_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None
