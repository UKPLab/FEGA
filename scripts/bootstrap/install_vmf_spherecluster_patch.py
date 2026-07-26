#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PATCH_FILENAMES = (
    "_spk.py",
    "_vmf_numerics.py",
    "_vmfm.py",
    "_vmfm_factor.py",
    "_vmfm_factor_em.py",
)


@dataclass(frozen=True)
class PatchStatus:
    package_dir: Path | None
    missing_package: bool
    stale_files: tuple[str, ...]

    @property
    def up_to_date(self) -> bool:
        return not self.missing_package and not self.stale_files


def source_patch_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "fega" / "core" / "vmf" / "utils" / "_spherecluster"


def find_spherecluster_dir() -> Path | None:
    spec = importlib.util.find_spec("spherecluster")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def assess_patch(
    package_dir: Path | None = None, patch_dir: Path | None = None
) -> PatchStatus:
    package_dir = package_dir or find_spherecluster_dir()
    if package_dir is None:
        return PatchStatus(package_dir=None, missing_package=True, stale_files=())
    patch_dir = patch_dir or source_patch_dir()
    stale = []
    for name in PATCH_FILENAMES:
        source = patch_dir / name
        target = package_dir / name
        if not target.exists() or _sha256(source) != _sha256(target):
            stale.append(name)
    return PatchStatus(
        package_dir=package_dir, missing_package=False, stale_files=tuple(stale)
    )


def apply_patch_files(
    package_dir: Path | None = None, patch_dir: Path | None = None
) -> PatchStatus:
    status = assess_patch(package_dir=package_dir, patch_dir=patch_dir)
    if status.missing_package:
        raise RuntimeError(
            "spherecluster is not installed; install it in the active environment "
            "before applying the FEGA vMF compatibility patch."
        )
    if status.package_dir is None:
        raise RuntimeError("Could not resolve spherecluster package directory.")
    patch_dir = patch_dir or source_patch_dir()
    for name in status.stale_files:
        shutil.copy2(patch_dir / name, status.package_dir / name)
    return assess_patch(package_dir=status.package_dir, patch_dir=patch_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install the FEGA vMF spherecluster compatibility patch."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report patch status without writing files.",
    )
    args = parser.parse_args(argv)

    status = assess_patch()
    if status.missing_package:
        print(
            "spherecluster is not installed; install it in the active environment "
            "before applying the FEGA vMF compatibility patch.",
            file=sys.stderr,
        )
        return 2
    if args.check:
        if status.up_to_date:
            print(f"spherecluster patch is up to date: {status.package_dir}")
            return 0
        print(
            "spherecluster patch is stale: "
            + ", ".join(status.stale_files)
            + f" in {status.package_dir}"
        )
        return 1

    final = apply_patch_files(package_dir=status.package_dir)
    if not final.up_to_date:
        print(
            "spherecluster patch did not converge: "
            + ", ".join(final.stale_files),
            file=sys.stderr,
        )
        return 1
    print(f"spherecluster patch installed: {final.package_dir}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
