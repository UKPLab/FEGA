#!/usr/bin/env python3
"""Check whether this machine exactly reproduces a promoted vMF backend."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from fega.core.vmf.backend_policy import (
    BackendPolicyManifestError,
    backend_policy_sha256,
)
from fega.core.vmf.fit import production_factor_policy, production_gpu_factor_policy
from fega.core.vmf.utils._spherecluster._vmfm_factor import FactorIneligible


def _cpu_preflight() -> dict[str, Any]:
    """Validate the tracked CPU numerical and implementation identities."""
    # Exercise the same fail-closed checks run before a promoted CPU factor fit.
    policy = production_factor_policy()
    return {
        "accepted": True,
        "numerical_fingerprint": dict(policy.require_current_cpu_fingerprint()),
        "source_fingerprint": dict(policy.require_current_source_fingerprint()),
    }


def _gpu_preflight(device: str) -> dict[str, Any]:
    """Validate the inherited CPU identity and pinned CUDA execution surface."""
    # Exercise every admission check before any feature coordinates are transferred.
    policy = production_gpu_factor_policy()
    return {
        "accepted": True,
        "factor_cpu_numerical_fingerprint": dict(
            policy.require_current_cpu_fingerprint()
        ),
        "factor_source_fingerprint": dict(
            policy.require_current_factor_source_fingerprint()
        ),
        "gpu_source_fingerprint": dict(policy.require_current_source_fingerprint()),
        "gpu_numerical_fingerprint": dict(
            policy.require_current_gpu_fingerprint(device)
        ),
    }


def _policy_preflight() -> dict[str, Any]:
    """Return the digest of the validated packaged backend authority."""
    # Keep policy-loading failures inside the same machine-readable rejection path.
    return {"accepted": True, "policy_sha256": backend_policy_sha256()}


def _checked(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Return one machine-readable accepted or rejected backend record."""
    # Keep expected admission failures concise while preserving their exact reason.
    try:
        return operation()
    except (BackendPolicyManifestError, FactorIneligible, ValueError) as error:
        return {
            "accepted": False,
            "error_type": type(error).__name__,
            "reason": str(error),
            "backend": name,
        }


def _parse_args() -> argparse.Namespace:
    """Parse the backend scope and CUDA device selected for public preflight."""
    # Require an explicit scope so CPU-only reproducers do not probe unavailable CUDA.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cpu-factor", "gpu-factor", "all"),
        required=True,
        help="strict backend identity to validate",
    )
    parser.add_argument("--gpu-device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Print a JSON preflight report and fail when any requested backend drifts."""
    # Run only the requested strict policies and make shell automation unambiguous.
    args = _parse_args()
    policy_report = _checked("policy-manifest", _policy_preflight)
    report: dict[str, Any] = {
        "schema_version": 1,
        "policy_sha256": policy_report.get("policy_sha256"),
        "policy_manifest": policy_report,
        "backends": {},
    }
    if args.backend in {"cpu-factor", "all"}:
        report["backends"]["cpu-factor"] = _checked("cpu-factor", _cpu_preflight)
    if args.backend in {"gpu-factor", "all"}:
        report["backends"]["gpu-factor"] = _checked(
            "gpu-factor", lambda: _gpu_preflight(args.gpu_device)
        )
    report["accepted"] = policy_report["accepted"] and all(
        result["accepted"] for result in report["backends"].values()
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
