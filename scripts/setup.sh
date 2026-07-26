#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv sync --frozen --all-extras --python 3.10.19
uv run --frozen python scripts/bootstrap/install_vmf_spherecluster_patch.py
uv run --frozen python scripts/bootstrap/install_vmf_spherecluster_patch.py --check
