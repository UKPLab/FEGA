#!/usr/bin/env bash
#
# Activate the project's uv virtual environment.
#
# One-time setup for a fresh clone:
#   cd <repo-root>
#   bash scripts/setup.sh
#
# Usage: source scripts/activate.sh

# Repo root = parent of this script's directory.
_SAE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Preserve the OMX CLI across virtual-environment activation.
_SAE_OMX_BIN="$(command -v omx 2>/dev/null || true)"

# --- uv venv activation ----------------------------------------------------
if [[ -f "${_SAE_REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${_SAE_REPO_ROOT}/.venv/bin/activate"
else
    echo "[sae_env] .venv not found at ${_SAE_REPO_ROOT}/.venv" >&2
    echo "[sae_env] Create it: cd ${_SAE_REPO_ROOT} && bash scripts/setup.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# Restore OMX's bin directory if activation replaced the incoming PATH.
if [[ -n "${_SAE_OMX_BIN}" ]]; then
    _SAE_OMX_BIN_DIR="$(dirname "${_SAE_OMX_BIN}")"
    case ":${PATH}:" in
        *":${_SAE_OMX_BIN_DIR}:"*) ;;
        *) export PATH="${_SAE_OMX_BIN_DIR}:${PATH}" ;;
    esac
fi

# --- caches ----------------------------------------------------------------
# Set SAE_CACHE_ROOT to a shared filesystem when compute nodes cannot write $HOME.
_SAE_CACHE="${SAE_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/fega}"
export HF_HOME="${HF_HOME:-${_SAE_CACHE}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export NLTK_DATA="${NLTK_DATA:-${_SAE_CACHE}/nltk_data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_SAE_CACHE}/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_SAE_CACHE}}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${NLTK_DATA}" "${MPLCONFIGDIR}"

# Authenticate with external services before activation, for example with
# `hf auth login`, or export credentials in the calling shell.

echo "[sae_env] uv env ready: ${_SAE_REPO_ROOT}/.venv"
