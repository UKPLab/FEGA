from __future__ import annotations

import argparse
import json
from pathlib import Path

from fega.config_schema import FEGAPipelineConfig
from fega.orchestrator import PHASES, run_pipeline
from fega.paths import run_metadata_path, run_root, run_status_path
from fega.run_metadata import read_run_metadata


def main() -> None:
    """Entry point for the pipeline CLI."""
    args = _build_arg_parser().parse_args()
    if args.command == "run":
        _handle_run(args)
    elif args.command == "status":
        _handle_status(args)
    elif args.command == "visualize":
        _handle_visualize(args)
    else:  # pragma: no cover - argparse enforces known commands
        raise ValueError(f"Unknown command: {args.command}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments and subcommands."""
    parser = argparse.ArgumentParser(description="FEGA pipeline orchestrator CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    # Run subcommand configuration.
    run_parser = sub.add_parser("run", help="Run one or more pipeline phases.")
    run_parser.add_argument(
        "--config", required=True, type=Path, help="Path to pipeline YAML config."
    )
    run_parser.add_argument(
        "--phases",
        default=None,
        help="Comma-separated list of phases to run or 'all' (default).",
    )
    run_parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop on first failure (default: true). Use --no-fail-fast to continue.",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip phases already marked success in run_status.json.",
    )
    run_parser.add_argument(
        "--device",
        default=None,
        help="Override the config device for this run, for example cpu or cuda:0.",
    )
    run_parser.add_argument(
        "--dense-cpu-workers",
        type=_positive_int,
        default=None,
        help=(
            "Use dense_cpu and atomically set vMF feature workers, vocabulary "
            "buffers, and stability workers to this positive count."
        ),
    )

    # Status subcommand configuration.
    status_parser = sub.add_parser(
        "status", help="Show latest run status for a config."
    )
    status_parser.add_argument(
        "--config", required=True, type=Path, help="Path to pipeline YAML config."
    )

    # Cached visualization subcommand configuration.
    visualize_parser = sub.add_parser(
        "visualize", help="Render per-feature views from one completed FEGA run."
    )
    visualize_parser.add_argument(
        "--run-dir", required=True, type=Path, help="One FEGA attribute run directory."
    )
    visualize_parser.add_argument(
        "--top-n",
        type=_positive_int,
        default=5,
        help="Candidates to render per atlas family (default: 5).",
    )
    visualize_parser.add_argument(
        "--palette-json",
        type=Path,
        default=None,
        help="Optional partial atlas-label to #RRGGBB palette JSON.",
    )
    visualize_parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=300,
        help="PNG resolution in dots per inch (default: 300).",
    )
    return parser


def _handle_run(args: argparse.Namespace) -> None:
    """Load config and invoke orchestrator run_pipeline."""
    # Load the checked-in experiment first, then record launcher-specific overrides.
    config_path = args.config.expanduser().resolve()
    config = FEGAPipelineConfig.from_file(config_path)
    phases_list = _parse_phases_arg(args.phases)
    _apply_run_overrides(config, args, phases=phases_list)

    # Preserve the source path so run metadata can record it.
    setattr(config, "_config_path", config_path)
    run_pipeline(
        config,
        phases=phases_list,
        fail_fast=bool(args.fail_fast),
        resume=bool(args.resume),
        config_path=config_path,
    )


def _positive_int(raw: str) -> int:
    """Parse one strictly positive integer supplied at the CLI boundary."""
    # Reject invalid worker topology before a scheduler allocation starts work.
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def _apply_run_overrides(
    config: FEGAPipelineConfig,
    args: argparse.Namespace,
    *,
    phases: list[str] | None = None,
) -> None:
    """Apply launcher resource overrides before resolved-config provenance is written.

    The dense worker override is atomic because feature workers and vocabulary
    buffers must remain aligned, while stability uses the same feature-level
    concurrency contract.
    """
    # Override the model/materialization device only when the launcher names one.
    device = getattr(args, "device", None)
    if device is not None:
        config.device = str(device)

    # Bind concurrency only for phases selected by this invocation.
    workers = getattr(args, "dense_cpu_workers", None)
    if workers is None:
        return
    selected = set(PHASES if phases is None else phases)
    if "vmf" in selected:
        config.phases.vmf.backend = "dense_cpu"
        config.phases.vmf.workers = int(workers)
        config.phases.vmf.max_vocab_buffers = int(workers)
    if "stability" in selected:
        config.phases.stability.workers = int(workers)


def _handle_status(args: argparse.Namespace) -> None:
    """Load status/metadata for a config and print a concise summary."""
    config_path = args.config.expanduser().resolve()
    config = FEGAPipelineConfig.from_file(config_path)
    status_path = run_status_path(config)
    print(f"Run root: {run_root(config)}")
    if not status_path.exists():
        print(f"No run_status.json found at {status_path}")
        return
    with open(status_path) as f:
        status = json.load(f)
    overall = status.get("status", "unknown")
    updated = status.get("last_updated")
    print(f"Status: {overall}" + (f" (updated {updated})" if updated else ""))
    phases = status.get("phases", {})
    for phase in PHASES:
        info = phases.get(phase, {})
        line = f"- {phase}: {info.get('status', 'unknown')}"
        if info.get("reason"):
            line += f" | {info['reason']}"
        if info.get("error"):
            line += f" | error: {info['error']}"
        print(line)
    meta_path = run_metadata_path(config)
    if meta_path.exists():
        meta = read_run_metadata(meta_path)
        if meta.global_seed is not None:
            print(f"Global seed: {meta.global_seed}")
        if meta.stage_seeds:
            seeds = ", ".join(f"{k}={v}" for k, v in meta.stage_seeds.items())
            print(f"Stage seeds: {seeds}")


def _handle_visualize(args: argparse.Namespace) -> None:
    """Generate cached per-feature views without entering the phase orchestrator."""
    # Import the plotting stack only for the standalone visualization command.
    from fega.core.visualizations import run_visualizations

    index_path = run_visualizations(
        args.run_dir,
        top_n=int(args.top_n),
        palette_path=args.palette_json,
        dpi=int(args.dpi),
    )
    print(f"Visualization index: {index_path}")


def _parse_phases_arg(raw: str | None) -> list[str] | None:
    """Convert comma-separated --phases argument into a list or None."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return None
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [raw]


if __name__ == "__main__":
    main()
