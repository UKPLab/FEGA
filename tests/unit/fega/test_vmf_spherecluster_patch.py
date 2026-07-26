from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_patcher():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "scripts/bootstrap/install_vmf_spherecluster_patch.py"
    )
    spec = importlib.util.spec_from_file_location("fega_vmf_patch_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()


def test_default_patch_source_uses_fega_package() -> None:
    repository_root = Path(__file__).resolve().parents[3]

    assert patcher.source_patch_dir() == (
        repository_root / "fega/core/vmf/utils/_spherecluster"
    )


def _write_patch_source(
    path: Path,
    spk: str = "patched-spk",
    numerics: str = "patched-numerics",
    vmfm: str = "patched-vmfm",
    factor: str = "patched-factor",
    factor_em: str = "patched-factor-em",
) -> None:
    """Write the complete standalone patch file set used by installer tests."""
    # Mirror the production install manifest so missing siblings are detected.
    path.mkdir(parents=True)
    (path / "_spk.py").write_text(spk)
    (path / "_vmf_numerics.py").write_text(numerics)
    (path / "_vmfm.py").write_text(vmfm)
    (path / "_vmfm_factor.py").write_text(factor)
    (path / "_vmfm_factor_em.py").write_text(factor_em)


def test_missing_spherecluster_reports_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(patcher, "find_spherecluster_dir", lambda: None)

    status = patcher.assess_patch()

    assert status.missing_package is True
    with pytest.raises(RuntimeError, match="spherecluster is not installed"):
        patcher.apply_patch_files()


def test_already_patched_files_are_up_to_date(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = tmp_path / "spherecluster"
    _write_patch_source(source)
    _write_patch_source(package)

    status = patcher.assess_patch(package_dir=package, patch_dir=source)

    assert status.up_to_date is True
    assert status.stale_files == ()


def test_stale_files_are_patched_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = tmp_path / "spherecluster"
    _write_patch_source(source)
    _write_patch_source(package, spk="old-spk", vmfm="old-vmfm")

    status = patcher.apply_patch_files(package_dir=package, patch_dir=source)

    assert status.up_to_date is True
    assert (package / "_spk.py").read_text() == "patched-spk"
    assert (package / "_vmf_numerics.py").read_text() == "patched-numerics"
    assert (package / "_vmfm.py").read_text() == "patched-vmfm"


def test_patched_vmfm_imports_and_evaluates_without_fega(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the installed spherecluster patch numerically usable without FEGA."""
    # Install the real patch into a temporary package while making FEGA unavailable.
    package = tmp_path / "spherecluster"
    package.mkdir()
    (package / "__init__.py").write_text("")
    status = patcher.apply_patch_files(package_dir=package)
    assert status.up_to_date is True

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "fega", None)
    monkeypatch.delitem(sys.modules, "spherecluster", raising=False)
    monkeypatch.delitem(sys.modules, "spherecluster._vmfm", raising=False)
    module = importlib.import_module("spherecluster._vmfm")
    factor_module = importlib.import_module("spherecluster._vmfm_factor")

    values = module._vmf_log(
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        0.0,
        np.asarray([1.0, 0.0], dtype=np.float64),
    )
    assert values == pytest.approx(np.asarray([-np.log(2.0 * np.pi)]))
    rows = np.eye(3, dtype=np.float64)
    factor = factor_module.factor_from_explicit_rows(rows)
    assert factor.z.shape == (3, 3)
    assert np.max(np.abs(factor.z @ factor.z.T - rows @ rows.T)) <= 1.0e-10


def test_check_mode_reports_stale_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    package = tmp_path / "spherecluster"
    _write_patch_source(source)
    _write_patch_source(package, spk="old-spk", vmfm="old-vmfm")
    monkeypatch.setattr(patcher, "source_patch_dir", lambda: source)
    monkeypatch.setattr(patcher, "find_spherecluster_dir", lambda: package)

    assert patcher.main(["--check"]) == 1
    assert (package / "_spk.py").read_text() == "old-spk"
    assert (package / "_vmf_numerics.py").read_text() == "patched-numerics"
    assert (package / "_vmfm.py").read_text() == "old-vmfm"
