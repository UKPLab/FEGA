#!/usr/bin/env bash
set -euo pipefail

# Submit the Gemma-2B induction-feature FEGA config.
#
# Optional overrides:
#   FEGA_INDUCTION_CONFIG=fega/config/induction/gemma2b_relu_trainer5.yaml
#   FEGA_PHASES=data_prep,compute_effect,geometry_metrics
#   FEGA_RESUME=0
#   FEGA_DEVICE=cuda:0
#   FEGA_NUMERICAL_THREADS=16
#   SAE_CACHE_ROOT=/path/on/a/shared/filesystem

export FEGA_INDUCTION_CONFIG="${FEGA_INDUCTION_CONFIG:-fega/config/induction/gemma2b_relu_trainer5.yaml}"
export FEGA_PHASES="${FEGA_PHASES:-}"
export FEGA_RESUME="${FEGA_RESUME:-1}"
export FEGA_DEVICE="${FEGA_DEVICE:-cuda:0}"
export FEGA_NUMERICAL_THREADS="${FEGA_NUMERICAL_THREADS:-16}"

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

LOGS_PATH="${repo_root}/data/logs"
mkdir -p "${LOGS_PATH}"

sbatch \
  --output="${LOGS_PATH}/induction_%j.out" \
  --error="${LOGS_PATH}/induction_%j.err" <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=fega-induction

set -eo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
fi

source scripts/activate.sh

# Derive feature concurrency from allocated CPUs while retaining 16-thread fits.
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

export OMP_NUM_THREADS="${fit_threads}"
export MKL_NUM_THREADS="${fit_threads}"
export OPENBLAS_NUM_THREADS="${fit_threads}"
export NUMEXPR_NUM_THREADS="${fit_threads}"

set -u

cmd=(
  python -m fega.cli run
  --config "${FEGA_INDUCTION_CONFIG}"
  --device "${FEGA_DEVICE}"
  --dense-cpu-workers "${feature_workers}"
)
if [[ -n "${FEGA_PHASES}" ]]; then
  cmd+=(--phases "${FEGA_PHASES}")
fi
if [[ "${FEGA_RESUME}" != "0" && "${FEGA_RESUME}" != "false" ]]; then
  cmd+=(--resume)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
printf 'Dense CPU topology: cpus=%s fit_threads=%s feature_workers=%s\n' \
  "${allocated_cpus}" "${fit_threads}" "${feature_workers}"
"${cmd[@]}"
SBATCH
