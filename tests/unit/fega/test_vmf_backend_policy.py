from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import fega.core.vmf.fit as vmf_fit
from fega.core.source_fingerprint import canonical_json_digest
from fega.core.vmf import backend_policy
from fega.core.vmf.backend_policy import (
    BackendPolicyManifestError,
    backend_policy_sha256,
    load_backend_policy_manifest,
    validate_backend_policy_manifest,
)


def test_tracked_backend_policy_is_self_validating_and_promoted() -> None:
    """Require the installed strict backend authority to pass its own digest gate."""
    # Load the packaged file through the same cached production boundary.
    policy = load_backend_policy_manifest()

    assert policy["cpu_factor"]["promoted"] is True
    assert policy["gpu_factor"]["promoted"] is True
    assert len(backend_policy_sha256()) == 64


def test_backend_policy_load_returns_defensive_copy() -> None:
    """Prevent callers from mutating the cached post-validation authority."""
    # Mutate one returned tree, then require a fresh load to retain tracked policy.
    first = load_backend_policy_manifest()
    first["cpu_factor"]["promoted"] = False

    second = load_backend_policy_manifest()

    assert second["cpu_factor"]["promoted"] is True


def test_missing_packaged_backend_policy_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject strict backend construction when its installed authority is absent."""
    # Clear the private immutable cache before pointing the loader at a missing file.
    backend_policy._load_backend_policy_authority.cache_clear()
    missing_path = tmp_path / "missing.json"
    monkeypatch.setattr(backend_policy, "_POLICY_PATH", missing_path)

    with pytest.raises(BackendPolicyManifestError, match="cannot load") as exc_info:
        load_backend_policy_manifest()
    assert str(missing_path) not in str(exc_info.value)

    backend_policy._load_backend_policy_authority.cache_clear()


def test_backend_policy_rejects_modified_evidence_without_a_new_digest() -> None:
    """Fail closed when policy evidence changes independently of its authority hash."""
    # Reconstruct the complete body while replacing only its accepted digest.
    policy = deepcopy(load_backend_policy_manifest())
    payload = {
        "schema_version": 1,
        "policy_sha256": "0" * 64,
        "policy": policy,
    }

    with pytest.raises(BackendPolicyManifestError, match="digest mismatch"):
        validate_backend_policy_manifest(payload)


def test_public_preflight_rejects_construction_invalid_policy_as_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject unconstructable promoted data without a traceback or ambiguous exit."""
    # Keep the manifest self-digested so this exercises semantic construction failure.
    from scripts.validation import check_vmf_backends

    policy = deepcopy(load_backend_policy_manifest())
    policy["cpu_factor"]["observed_error_envelopes"]["bic"] = None
    invalid_policy = validate_backend_policy_manifest(
        {
            "schema_version": 1,
            "policy_sha256": canonical_json_digest(policy),
            "policy": policy,
        }
    )
    monkeypatch.setattr(
        vmf_fit,
        "load_backend_policy_manifest",
        lambda: invalid_policy,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_vmf_backends.py", "--backend", "cpu-factor"],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_vmf_backends.main()

    report = json.loads(capsys.readouterr().out)
    assert exc_info.value.code == 1
    assert report["accepted"] is False
    assert report["backends"]["cpu-factor"]["error_type"] == (
        "BackendPolicyManifestError"
    )
