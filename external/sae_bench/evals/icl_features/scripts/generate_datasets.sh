#!/usr/bin/env bash
set -euo pipefail

# Environment overrides:
# TASKS="lsc wc tt prontoqa", MODEL_NAME=gemma-2-2b, DEVICE=cuda:0,
# TARGET_EXAMPLES=50000, NUM_FAMILIES=1000, SEED=0, BATCH_SIZE=32,
# OUTPUT_ROOT=data/icl_features, EXTRA_ARGS="..."

read -r -a task_list <<< "${TASKS:-lsc wc tt prontoqa}"
model_name="${MODEL_NAME:-gemma-2-2b}"
device="${DEVICE:-cuda:0}"
target_examples="${TARGET_EXAMPLES:-50000}"
num_families="${NUM_FAMILIES:-1000}"
seed="${SEED:-0}"
batch_size="${BATCH_SIZE:-32}"
output_root="${OUTPUT_ROOT:-data/icl_features}"
extra_args=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
fi

mkdir -p "${output_root}/${model_name}"

for task in "${task_list[@]}"; do
  output="${output_root}/${model_name}/${task}.json"
  cmd=(
    python -m sae_bench.evals.icl_features.generate
    --task "${task}"
    --output "${output}"
    --model-name "${model_name}"
    --device "${device}"
    --target-examples "${target_examples}"
    --num-families "${num_families}"
    --seed "${seed}"
    --batch-size "${batch_size}"
  )
  cmd+=("${extra_args[@]}")
  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done
