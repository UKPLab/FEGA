#!/usr/bin/env bash

# CPU downstream launcher for dense vMF, stability, and geometry reporting.

set -euo pipefail

export FEGA_CONFIG="${FEGA_CONFIG:-fega/config/ravel/relu/city_country_2pow16.yaml}"
export FEGA_PHASES="${FEGA_PHASES:-geometry_reporting}"
export FEGA_DEVICE="${FEGA_DEVICE:-cpu}"
export FEGA_NUMERICAL_THREADS="${FEGA_NUMERICAL_THREADS:-16}"

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

LOGS_PATH="${repo_root}/data/logs"
mkdir -p "${LOGS_PATH}"

sbatch \
  --output="${LOGS_PATH}/slurm_%j.out" \
  --error="${LOGS_PATH}/slurm_%j.err" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --gpus=0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=fega

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

export SAE_CACHE_ROOT="${SAE_CACHE_ROOT:-${PWD}/data/cache}"
source scripts/activate.sh

# Divide allocated CPUs into measured 16-thread dense feature fits.
fit_threads="${FEGA_NUMERICAL_THREADS}"
allocated_cpus="${SLURM_CPUS_PER_TASK}"
if ! [[ "${fit_threads}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'FEGA_NUMERICAL_THREADS must be a positive integer: %s\n' "${fit_threads}" >&2
  exit 2
fi
if (( allocated_cpus < fit_threads || allocated_cpus % fit_threads != 0 )); then
  printf 'SLURM_CPUS_PER_TASK=%s must be divisible by fit threads=%s\n' \
    "${allocated_cpus}" "${fit_threads}" >&2
  exit 2
fi
feature_workers=$((allocated_cpus / fit_threads))

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${fit_threads}"
export MKL_NUM_THREADS="${fit_threads}"
export OPENBLAS_NUM_THREADS="${fit_threads}"
export NUMEXPR_NUM_THREADS="${fit_threads}"

set -u

cmd=(
  python -m fega.cli run
  --config "${FEGA_CONFIG}"
  --phases "${FEGA_PHASES}"
  --device "${FEGA_DEVICE}"
  --dense-cpu-workers "${feature_workers}"
)
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'Dense CPU topology: cpus=%s fit_threads=%s feature_workers=%s\n' \
  "${allocated_cpus}" "${fit_threads}" "${feature_workers}"
"${cmd[@]}"
SBATCH
