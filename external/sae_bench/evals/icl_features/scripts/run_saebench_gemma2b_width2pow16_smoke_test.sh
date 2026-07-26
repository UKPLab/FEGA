#!/usr/bin/env bash
set -euo pipefail

model_name="gemma-2-2b"
repo_id="canrager/saebench_gemma-2-2b_width-2pow16_date-0107"
relu_location="gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
topk_location="gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
matryoshka_location="gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
sae_locations="${relu_location} ${topk_location} ${matryoshka_location}"
tasks="${TASKS:-lsc wc tt prontoqa}"
target_examples="${SMOKE_EXAMPLES:-16}"
num_families="${SMOKE_FAMILIES:-4}"
data_root="${SMOKE_DATA_ROOT:-data/icl_features_smoke}"
result_root="${RESULT_ROOT:-results/sae_geometry_smoke}"
discovery_root="${DISCOVERY_ROOT:-data/induction_feature_outputs/$(basename "${result_root}")}"
download_saes_dir="${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"
device="${DEVICE:-cuda:0}"

sae_uid() {
  local location="$1"
  printf '%s__%s' "${repo_id//\//_}" "${location//\//_}"
}

relu_uid="$(sae_uid "${relu_location}")"
topk_uid="$(sae_uid "${topk_location}")"
matryoshka_uid="$(sae_uid "${matryoshka_location}")"

mkdir -p "${result_root}/preflight"
python scripts/bootstrap/install_vmf_spherecluster_patch.py --check
python -m sae_bench.evals.icl_features.preflight \
  --model-name "${model_name}" \
  --repo-id "${repo_id}" \
  --sae-spec "relu::${relu_location}::StandardTrainerAprilUpdate" \
  --sae-spec "topk::${topk_location}::TopKTrainer" \
  --sae-spec "matryoshka_topk::${matryoshka_location}::MatryoshkaBatchTopKTrainer" \
  --device "${device}" \
  --download-saes-dir "${download_saes_dir}" \
  --output "${result_root}/preflight/preflight.json"

for task in ${tasks}; do
  task_extra_args="--max-candidates-per-family ${SMOKE_MAX_CANDIDATES_PER_FAMILY:-1000}"
  task_num_families="${num_families}"
  if [[ "${task}" == "tt" ]]; then
    # Tiny TT smoke runs can be brittle because some translation families have
    # very low Gemma first-token acceptance. Keep the smoke end-to-end while
    # leaving the full dataset-generation settings unchanged.
    task_num_families="${SMOKE_TT_FAMILIES:-1}"
    task_extra_args="--max-candidates-per-family ${SMOKE_TT_MAX_CANDIDATES_PER_FAMILY:-8000}"
  elif [[ "${task}" == "prontoqa" ]]; then
    # One-shot PrOntoQA keeps the smoke test focused on environment and path
    # health instead of being dominated by per-support-slot acceptance tails.
    task_extra_args="--max-candidates-per-family ${SMOKE_PRONTOQA_MAX_CANDIDATES_PER_FAMILY:-8000} --prontoqa-shots ${SMOKE_PRONTOQA_SHOTS:-1}"
  fi
  TASKS="${task}" \
  MODEL_NAME="${model_name}" \
  DEVICE="${device}" \
  TARGET_EXAMPLES="${target_examples}" \
  NUM_FAMILIES="${task_num_families}" \
  BATCH_SIZE="${SMOKE_GENERATION_BATCH_SIZE:-16}" \
  OUTPUT_ROOT="${data_root}" \
  EXTRA_ARGS="${task_extra_args}" \
    bash external/sae_bench/evals/icl_features/scripts/generate_datasets.sh
done

MODEL_NAME="${model_name}" \
REPO_ID="${repo_id}" \
SAE_LOCATIONS="${sae_locations}" \
TASKS="${tasks}" \
DATA_ROOT="${data_root}" \
RESULT_ROOT="${result_root}" \
DISCOVERY_ROOT="${discovery_root}" \
DOWNLOAD_SAES_DIR="${download_saes_dir}" \
STAGES="discovery iou" \
TARGET_EXAMPLES="${target_examples}" \
DISCOVERY_BATCH_SIZE="${SMOKE_DISCOVERY_BATCH_SIZE:-16}" \
RECHECK_MODEL_CORRECT=0 \
RESUME=0 \
DEVICE="${device}" \
  bash external/sae_bench/evals/icl_features/scripts/run_icl_pipeline.sh

for uid in "${relu_uid}" "${topk_uid}" "${matryoshka_uid}"; do
  python -m sae_bench.evals.icl_features.ablation \
    --task lsc \
    --dataset-path "${data_root}/${model_name}/lsc.json" \
    --discovery-summary "${discovery_root}/${model_name}/lsc/summary.json" \
    --sae-uid "${uid}" \
    --feature-set threshold \
    --output-dir "${result_root}/${model_name}/${uid}/lsc/ablation/threshold" \
    --model-name "${model_name}" \
    --download-saes-dir "${download_saes_dir}" \
    --device "${device}" \
    --batch-size "${SMOKE_ABLATION_BATCH_SIZE:-16}" \
    --random-trials "${SMOKE_RANDOM_TRIALS:-2}" \
    --random-match-pool-size "${SMOKE_RANDOM_MATCH_POOL_SIZE:-10}" \
    --expected-examples "${target_examples}" \
    --overwrite
done

run_geometry() {
  local task="$1"
  local uid="$2"
  local task_root="${result_root}/${model_name}/${uid}/${task}"
  TASK="${task}" \
  MODEL_NAME="${model_name}" \
  DATASET_PATH="${data_root}/${model_name}/${task}.json" \
  SUMMARY_PATH="${discovery_root}/${model_name}/${task}/summary.json" \
  SAE_REPO_ID="${repo_id}" \
  SAE_UID="${uid}" \
  FEGA_CONFIG="${task_root}/fega_config_smoke.yaml" \
  FEGA_OUTPUT_ROOT="${task_root}/fega" \
  FEGA_SMOKE=1 \
  FEGA_SMOKE_MAX_FEATURES="${SMOKE_GEOMETRY_FEATURES:-2}" \
  FEGA_RESUME=0 \
  DOWNLOAD_SAES_DIR="${download_saes_dir}" \
  DEVICE="${device}" \
    bash external/sae_bench/evals/icl_features/scripts/run_fega_geometry.sh
}

read -r -a task_list <<< "${tasks}"
for task in "${task_list[@]}"; do
  run_geometry "${task}" "${relu_uid}"
done
run_geometry "lsc" "${topk_uid}"
run_geometry "lsc" "${matryoshka_uid}"

relu_map_args=()
for task in "${task_list[@]}"; do
  relu_map_args+=(
    --task-map
    "${task}=${result_root}/${model_name}/${relu_uid}/${task}/fega/${model_name}/${task}.json/${task}_pointer_like/geometry_reporting/geometry_map_data.json"
  )
done
python -m sae_bench.evals.icl_features.geometry_plots \
  "${relu_map_args[@]}" \
  --output-dir "${result_root}/${model_name}/${relu_uid}/cross_task/geometry_plots" \
  --model-name "${model_name}" \
  --sae-uid "${relu_uid}" \
  --embedding pca \
  --title "Smoke Test: SAE Feature Effect Geometry Across ICL Tasks"

for uid in "${topk_uid}" "${matryoshka_uid}"; do
  python -m sae_bench.evals.icl_features.geometry_plots \
    --task-map "lsc=${result_root}/${model_name}/${uid}/lsc/fega/${model_name}/lsc.json/lsc_pointer_like/geometry_reporting/geometry_map_data.json" \
    --output-dir "${result_root}/${model_name}/${uid}/cross_task/geometry_plots" \
    --model-name "${model_name}" \
    --sae-uid "${uid}" \
    --embedding pca \
    --title "Smoke Test: LSC SAE Feature Effect Geometry"
done

python -m sae_bench.evals.icl_features.aggregate_results \
  --result-root "${result_root}" \
  --discovery-root "${discovery_root}"

required_files=(
  "${result_root}/preflight/preflight.json"
  "${discovery_root}/${model_name}/lsc/summary.json"
  "${result_root}/${model_name}/${relu_uid}/lsc/ablation/threshold/selected_ablation_table.csv"
  "${result_root}/${model_name}/${topk_uid}/lsc/ablation/threshold/random_ablation_table.csv"
  "${result_root}/${model_name}/${topk_uid}/lsc/ablation/threshold/random_ablation_aggregate.csv"
  "${result_root}/${model_name}/${matryoshka_uid}/lsc/ablation/threshold/ablation_summary.json"
  "${result_root}/${model_name}/${relu_uid}/cross_task/geometry_plots/geometry_all_tasks.png"
  "${result_root}/${model_name}/${topk_uid}/cross_task/geometry_plots/geometry_all_tasks.png"
  "${result_root}/${model_name}/${matryoshka_uid}/cross_task/geometry_plots/geometry_all_tasks.png"
  "${result_root}/tables/manifest.json"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Smoke test failed: missing or empty ${required_file}" >&2
    exit 1
  fi
done

printf '%s\n' "${required_files[@]}" > "${result_root}/smoke_verified_files.txt"
echo "Mini end-to-end smoke test passed. Results: ${result_root}"
