#!/usr/bin/env bash
set -euo pipefail

# Required:
#   TASK=lsc
#   REPO_ID=owner/sae-repo and optionally SAE_LOCATIONS="location_a location_b"
# Or:
#   SAE_REGEX_PATTERN='gemma-scope-.*'
#
# Common overrides:
# MODEL_NAME=gemma-2-2b, DEVICE=cuda:0, TARGET_EXAMPLES=50000,
# MIN_QUERY_FRACTION=0.9, MIN_FAMILY_FRACTION=0.9,
# MIN_EXAMPLE_FRACTION=0.9, ACTIVATION_THRESHOLD=0.0, RECHECK_MODEL_CORRECT=1

task="${TASK:?Set TASK to lsc, wc, tt, or prontoqa}"
model_name="${MODEL_NAME:-gemma-2-2b}"
device="${DEVICE:-cuda:0}"
target_examples="${TARGET_EXAMPLES:-50000}"
dataset="${DATASET_PATH:-data/icl_features/${model_name}/${task}.json}"
output_dir="${OUTPUT_DIR:-data/induction_feature_outputs/${model_name}/${task}}"
download_saes_dir="${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"

cmd=(
  python -m sae_bench.evals.icl_features.main
  --dataset-path "${dataset}"
  --output-dir "${output_dir}"
  --download-saes-dir "${download_saes_dir}"
  --model-name "${model_name}"
  --device "${device}"
  --batch-size "${BATCH_SIZE:-32}"
  --activation-threshold "${ACTIVATION_THRESHOLD:-0.0}"
  --min-example-fraction "${MIN_EXAMPLE_FRACTION:-0.9}"
  --min-query-fraction-per-context "${MIN_QUERY_FRACTION:-0.9}"
  --min-context-fraction "${MIN_FAMILY_FRACTION:-0.9}"
  --expected-examples "${target_examples}"
  --max-family-size-difference "${MAX_FAMILY_SIZE_DIFFERENCE:-1}"
)

if [[ "${task}" != "prontoqa" ]]; then
  cmd+=(--single-token-only)
fi
if [[ "${RECHECK_MODEL_CORRECT:-1}" != "0" ]]; then
  cmd+=(--require-model-correct)
fi

if [[ -n "${REPO_ID:-}" ]]; then
  cmd+=(--repo-id "${REPO_ID}")
  if [[ -n "${SAE_LOCATIONS:-}" ]]; then
    read -r -a sae_locations <<< "${SAE_LOCATIONS}"
    for location in "${sae_locations[@]}"; do
      cmd+=(--sae-location "${location}")
    done
  fi
elif [[ -n "${SAE_REGEX_PATTERN:-}" ]]; then
  cmd+=(--sae-regex-pattern "${SAE_REGEX_PATTERN}")
else
  echo "Set REPO_ID or SAE_REGEX_PATTERN." >&2
  exit 2
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
