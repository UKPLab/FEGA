"""Load the tracked calibration authority for strict optimized vMF backends."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from fega.core.source_fingerprint import canonical_json_digest

_POLICY_PATH = Path(__file__).with_name("backend_policy.json")
_CPU_POLICY_KEYS = frozenset(
    {
        "promoted",
        "observed_error_envelopes",
        "cpu_numerical_fingerprint",
        "source_fingerprint",
        "validated_domain",
    }
)
_GPU_POLICY_KEYS = frozenset(
    {
        "promoted",
        "observed_error_envelopes",
        "gpu_numerical_fingerprint",
        "factor_cpu_numerical_fingerprint",
        "source_fingerprint",
        "factor_source_fingerprint",
        "validated_domain",
    }
)


class BackendPolicyManifestError(RuntimeError):
    """Report a missing, malformed, or internally inconsistent policy manifest."""


def validate_backend_policy_manifest(payload: Any) -> dict[str, Any]:
    """Return the policy body only when its schema and canonical digest are valid."""
    # Keep the public manifest self-validating without depending on OMX evidence.
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BackendPolicyManifestError("vMF backend policy schema mismatch")
    policy = payload.get("policy")
    if not isinstance(policy, dict):
        raise BackendPolicyManifestError("vMF backend policy body is missing")
    expected = payload.get("policy_sha256")
    observed = canonical_json_digest(policy)
    if not isinstance(expected, str) or expected != observed:
        raise BackendPolicyManifestError("vMF backend policy digest mismatch")
    if set(policy) != {"policy_version", "cpu_factor", "gpu_factor"}:
        raise BackendPolicyManifestError("vMF backend policy sections are incomplete")
    if policy.get("policy_version") != 1:
        raise BackendPolicyManifestError("vMF backend policy version mismatch")
    cpu_policy = policy.get("cpu_factor")
    gpu_policy = policy.get("gpu_factor")
    if not isinstance(cpu_policy, dict) or not isinstance(gpu_policy, dict):
        raise BackendPolicyManifestError("vMF backend policies must be objects")
    if set(cpu_policy) != _CPU_POLICY_KEYS or set(gpu_policy) != _GPU_POLICY_KEYS:
        raise BackendPolicyManifestError("vMF backend policy fields are incomplete")
    if cpu_policy.get("promoted") is not True or gpu_policy.get("promoted") is not True:
        raise BackendPolicyManifestError("vMF optimized backends are not promoted")
    # Require structured evidence before production constructors index each section.
    structured_fields = (
        cpu_policy["observed_error_envelopes"],
        cpu_policy["cpu_numerical_fingerprint"],
        cpu_policy["source_fingerprint"],
        cpu_policy["validated_domain"],
        gpu_policy["observed_error_envelopes"],
        gpu_policy["gpu_numerical_fingerprint"],
        gpu_policy["factor_cpu_numerical_fingerprint"],
        gpu_policy["source_fingerprint"],
        gpu_policy["factor_source_fingerprint"],
        gpu_policy["validated_domain"],
    )
    if any(not isinstance(field, dict) for field in structured_fields):
        raise BackendPolicyManifestError("vMF backend policy evidence must be objects")
    return policy


@cache
def _load_backend_policy_authority() -> tuple[str, str]:
    """Cache the validated digest and canonical policy bytes as immutable values."""
    # Keep the cached authority immutable so callers cannot bypass digest validation.
    try:
        payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendPolicyManifestError("cannot load vMF backend policy") from error
    policy = validate_backend_policy_manifest(payload)
    return str(payload["policy_sha256"]), json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_backend_policy_manifest() -> dict[str, Any]:
    """Return a fresh copy of the validated installed production policy."""
    # Decode the cached canonical bytes so no caller can mutate cached authority.
    _, canonical_policy = _load_backend_policy_authority()
    policy = json.loads(canonical_policy)
    if not isinstance(policy, dict):
        raise BackendPolicyManifestError("vMF backend policy body is missing")
    return policy


def backend_policy_sha256() -> str:
    """Return the canonical digest attested by the validated public manifest."""
    # Reuse the digest paired with the immutable cached policy bytes.
    digest, _ = _load_backend_policy_authority()
    return digest
