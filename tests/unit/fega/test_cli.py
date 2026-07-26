from argparse import Namespace
from pathlib import Path

from fega.cli import _apply_run_overrides
from fega.config_schema import FEGAPipelineConfig


def test_dense_cpu_worker_override_keeps_concurrency_atomic() -> None:
    """Keep launcher-derived feature workers, buffers, and stability aligned."""
    # Build the smallest config needed to exercise the run-boundary override.
    config = FEGAPipelineConfig(
        reference_json=Path("reference.json"),
        output_root=Path("results"),
        device="cpu",
        entity_attribute_selection={"city": ["Country"]},
    )
    args = Namespace(device="cuda:0", dense_cpu_workers=4)

    # Apply the same atomic override used by the Slurm launchers.
    _apply_run_overrides(config, args)

    assert config.device == "cuda:0"
    assert config.phases.vmf.backend == "dense_cpu"
    assert config.phases.vmf.workers == 4
    assert config.phases.vmf.max_vocab_buffers == 4
    assert config.phases.stability.workers == 4


def test_reporting_only_does_not_override_upstream_concurrency() -> None:
    """Keep reporting-only resource choices out of cached upstream identities."""
    # Preserve the configured vMF and stability topology when neither phase will run.
    config = FEGAPipelineConfig(
        reference_json=Path("reference.json"),
        output_root=Path("results"),
        device="cpu",
        entity_attribute_selection={"city": ["Country"]},
    )
    args = Namespace(device="cpu", dense_cpu_workers=4)

    _apply_run_overrides(config, args, phases=["geometry_reporting"])

    assert config.phases.vmf.workers == 1
    assert config.phases.vmf.max_vocab_buffers == 1
    assert config.phases.stability.workers == 1
