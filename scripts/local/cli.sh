#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

# Paths/config
CONFIG_PATH="fega/config/ravel/city_country.yaml"
LOGS_PATH="${repo_root}/data/logs"

# Make sure logs dir exists
mkdir -p "${LOGS_PATH}"

# Timestamped log file name, e.g. run_20251206_193045.log
timestamp="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOGS_PATH}/run_${timestamp}.log"

echo "Logging to ${LOG_FILE}"

# Run full pipeline:
python -m fega.cli run --config "${CONFIG_PATH}" \
  2>&1 | tee "${LOG_FILE}"
