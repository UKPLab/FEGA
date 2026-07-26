#!/usr/bin/env bash
set -euo pipefail

cmd=(
  python -m sae_bench.evals.icl_features.validate_datasets
  --data-root "${DATA_ROOT:-data/icl_features}"
  --model-name "${MODEL_NAME:-gemma-2-2b}"
  --expected-examples "${TARGET_EXAMPLES:-50000}"
  --expected-families "${NUM_FAMILIES:-1000}"
  --batch-size "${BATCH_SIZE:-32}"
  --device "${DEVICE:-cuda:0}"
)
if [[ -n "${TASKS:-}" ]]; then
  read -r -a task_list <<< "${TASKS}"
  cmd+=(--tasks "${task_list[@]}")
fi
if [[ -n "${OUTPUT_PATH:-}" ]]; then
  cmd+=(--output "${OUTPUT_PATH}")
fi
if [[ "${RECHECK_MODEL_CORRECT:-0}" != "0" ]]; then
  cmd+=(--recheck-model-correct)
fi
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
