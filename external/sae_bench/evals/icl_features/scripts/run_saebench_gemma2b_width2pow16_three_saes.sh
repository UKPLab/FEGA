#!/usr/bin/env bash
set -euo pipefail

# Useful overrides:
#   DATA_ROOT=data/icl_features  directory containing pre-generated 50k datasets
#   RECHECK_MODEL_CORRECT=0      avoid Transformers-vs-TransformerLens drift

model_name="gemma-2-2b"
repo_id="canrager/saebench_gemma-2-2b_width-2pow16_date-0107"
relu_location="gemma-2-2b_standard_new_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
topk_location="gemma-2-2b_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
matryoshka_location="gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107/resid_post_layer_12/trainer_2"
sae_locations="${relu_location} ${topk_location} ${matryoshka_location}"
result_root="${RESULT_ROOT:-results/sae_geometry_gemma2b_65k}"
discovery_root="${DISCOVERY_ROOT:-data/induction_feature_outputs/$(basename "${result_root}")}"
download_saes_dir="${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"
data_root="${DATA_ROOT:-data/icl_features}"
stages="${STAGES:-discovery ablation iou geometry plots aggregate}"
tasks="${TASKS:-lsc wc tt prontoqa}"
target_examples="${TARGET_EXAMPLES:-50000}"
num_families="${NUM_FAMILIES:-1000}"
device="${DEVICE:-cuda:0}"

mkdir -p "${result_root}/preflight"

MODEL_NAME="${model_name}" \
DATA_ROOT="${data_root}" \
TASKS="${tasks}" \
TARGET_EXAMPLES="${target_examples}" \
NUM_FAMILIES="${num_families}" \
OUTPUT_PATH="${result_root}/preflight/datasets.json" \
RECHECK_MODEL_CORRECT="${DATASET_FULL_RECHECK:-0}" \
DEVICE="${device}" \
  bash external/sae_bench/evals/icl_features/scripts/validate_datasets.sh

if [[ "${RUN_PREFLIGHT:-1}" != "0" ]]; then
  python -m sae_bench.evals.icl_features.preflight \
    --model-name "${model_name}" \
    --repo-id "${repo_id}" \
    --sae-spec "relu::${relu_location}::StandardTrainerAprilUpdate" \
    --sae-spec "topk::${topk_location}::TopKTrainer" \
    --sae-spec "matryoshka_topk::${matryoshka_location}::MatryoshkaBatchTopKTrainer" \
    --device "${device}" \
    --download-saes-dir "${download_saes_dir}" \
    --output "${result_root}/preflight/preflight.json"
fi

MODEL_NAME="${model_name}" \
REPO_ID="${repo_id}" \
SAE_LOCATIONS="${sae_locations}" \
TASKS="${tasks}" \
DATA_ROOT="${data_root}" \
RESULT_ROOT="${result_root}" \
DISCOVERY_ROOT="${discovery_root}" \
DOWNLOAD_SAES_DIR="${download_saes_dir}" \
STAGES="${stages}" \
TARGET_EXAMPLES="${target_examples}" \
MIN_EXAMPLE_FRACTION="${MIN_EXAMPLE_FRACTION:-0.9}" \
MIN_QUERY_FRACTION="${MIN_QUERY_FRACTION:-0.9}" \
MIN_FAMILY_FRACTION="${MIN_FAMILY_FRACTION:-0.9}" \
ACTIVATION_THRESHOLD="${ACTIVATION_THRESHOLD:-0.0}" \
ABLATION_FEATURE_SETS="${ABLATION_FEATURE_SETS:-threshold}" \
ABLATION_POSITION="${ABLATION_POSITION:-final}" \
RANDOM_TRIALS="${RANDOM_TRIALS:-20}" \
RANDOM_MATCH_POOL_SIZE="${RANDOM_MATCH_POOL_SIZE:-100}" \
GEOMETRY_FEATURE_SET="${GEOMETRY_FEATURE_SET:-candidate}" \
PLOT_EMBEDDING="${PLOT_EMBEDDING:-auto}" \
RECHECK_MODEL_CORRECT="${RECHECK_MODEL_CORRECT:-0}" \
RESUME="${RESUME:-1}" \
DEVICE="${device}" \
  bash external/sae_bench/evals/icl_features/scripts/run_icl_pipeline.sh

if [[ "${RUN_PAPER_AUDIT:-1}" != "0" ]]; then
  audit_args=(
    python -m sae_bench.evals.icl_features.audit_paper_artifacts
    --result-root "${result_root}"
    --discovery-root "${discovery_root}"
    --model-name "${model_name}"
    --sae-uid "${repo_id//\//_}__${relu_location//\//_}"
    --sae-uid "${repo_id//\//_}__${topk_location//\//_}"
    --sae-uid "${repo_id//\//_}__${matryoshka_location//\//_}"
    --expected-examples "${target_examples}"
    --expected-random-trials "${RANDOM_TRIALS:-20}"
    --min-example-fraction "${MIN_EXAMPLE_FRACTION:-0.9}"
    --min-query-fraction "${MIN_QUERY_FRACTION:-0.9}"
    --min-family-fraction "${MIN_FAMILY_FRACTION:-0.9}"
    --output "${result_root}/paper_artifact_audit.json"
  )
  "${audit_args[@]}"
fi
