#!/usr/bin/env bash
set -euo pipefail

# Run the ICL pointer-feature pipeline for single Gemma Scope residual SAEs.
#
# This is intentionally separate from run_saebench_gemma2b_width2pow16_three_saes.sh:
# Gemma Scope width sweeps usually compare one SAE at a time rather than the
# ReLU/TopK/Matryoshka triplet from the SAEBench custom checkpoints.
#
# Common 2B launch:
#   MODEL_NAME=gemma-2-2b DEVICE=cuda:1 \
#     bash external/sae_bench/evals/icl_features/scripts/run_gemmascope_width_sweep.sh
#
# Common 9B launch:
#   MODEL_NAME=gemma-2-9b DEVICE=cuda:1 \
#     bash external/sae_bench/evals/icl_features/scripts/run_gemmascope_width_sweep.sh
#
# Useful overrides:
#   WIDTHS="65k 131k"
#   GEMMASCOPE_LAYER=12                         # default: 12 for 2B, 20 for 9B
#   GEMMASCOPE_65K_L0=72 GEMMASCOPE_131K_L0=67 # 2B defaults
#   GEMMASCOPE_65K_L0=55 GEMMASCOPE_131K_L0=62 # 9B defaults
#   STAGES="validate discovery ablation iou geometry plots aggregate"
#   RESULT_ROOT=results/sae_geometry_gemmascope
#   DISCOVERY_ROOT=data/induction_feature_outputs
#   DATA_ROOT=data/icl_features
#   DOWNLOAD_SAES_DIR=data/downloaded_saes

read -r -a tasks <<< "${TASKS:-lsc wc tt prontoqa}"
read -r -a stages <<< "${STAGES:-validate discovery ablation iou geometry plots aggregate}"
read -r -a widths <<< "${WIDTHS:-65k 131k}"
read -r -a ablation_feature_sets <<< "${ABLATION_FEATURE_SETS:-threshold}"

model_name="${MODEL_NAME:-gemma-2-2b}"
data_root="${DATA_ROOT:-data/icl_features}"
base_result_root="${RESULT_ROOT:-results/sae_geometry_gemmascope}"
base_discovery_root="${DISCOVERY_ROOT:-data/induction_feature_outputs}"
download_saes_dir="${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"
target_examples="${TARGET_EXAMPLES:-50000}"
num_families="${NUM_FAMILIES:-1000}"
device="${DEVICE:-cuda:0}"
resume="${RESUME:-1}"

has_stage() {
  local requested="$1"
  local stage
  for stage in "${stages[@]}"; do
    [[ "${stage}" == "${requested}" ]] && return 0
  done
  return 1
}

default_release() {
  case "${model_name}" in
    gemma-2-2b) printf 'gemma-scope-2b-pt-res' ;;
    gemma-2-9b) printf 'gemma-scope-9b-pt-res' ;;
    *)
      echo "Unsupported MODEL_NAME=${model_name}; set SAE_RELEASE explicitly." >&2
      return 2
      ;;
  esac
}

default_layer() {
  case "${model_name}" in
    gemma-2-2b) printf '12' ;;
    gemma-2-9b) printf '20' ;;
    *)
      echo "Unsupported MODEL_NAME=${model_name}; set GEMMASCOPE_LAYER explicitly." >&2
      return 2
      ;;
  esac
}

default_l0() {
  local width="$1"
  case "${model_name}:${width}" in
    gemma-2-2b:65k) printf '72' ;;
    gemma-2-2b:131k) printf '67' ;;
    gemma-2-9b:65k) printf '55' ;;
    gemma-2-9b:131k) printf '62' ;;
    *)
      echo "No default L0 for MODEL_NAME=${model_name}, width=${width}; set GEMMASCOPE_${width^^}_L0." >&2
      return 2
      ;;
  esac
}

configured_l0() {
  local width="$1"
  case "${width}" in
    65k) printf '%s' "${GEMMASCOPE_65K_L0:-$(default_l0 "${width}")}" ;;
    131k) printf '%s' "${GEMMASCOPE_131K_L0:-$(default_l0 "${width}")}" ;;
    *)
      local env_name="GEMMASCOPE_${width^^}_L0"
      env_name="${env_name//-/_}"
      local value="${!env_name:-}"
      if [[ -z "${value}" ]]; then
        echo "Set ${env_name} for WIDTHS entry ${width}." >&2
        return 2
      fi
      printf '%s' "${value}"
      ;;
  esac
}

sanitize() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  printf '%s' "${value}"
}

sae_uid() {
  local release="$1"
  local sae_id="$2"
  printf '%s__%s' "$(sanitize "${release}")" "$(sanitize "${sae_id}")"
}

run_root_for() {
  local width="$1"
  local l0="$2"
  printf '%s/%s_width-%s_l0-%s' "${base_result_root}" "${model_name}" "${width}" "${l0}"
}

run_discovery() {
  local release="$1"
  local sae_id="$2"
  local run_root="$3"
  local discovery_root="$4"
  local task
  for task in "${tasks[@]}"; do
    local dataset="${data_root}/${model_name}/${task}.json"
    local output_dir="${discovery_root}/${model_name}/${task}"
    if [[ "${resume}" != "0" && -f "${output_dir}/summary.json" ]]; then
      echo "Skipping completed discovery: ${model_name} ${sae_id} ${task}"
      continue
    fi
    mkdir -p "${output_dir}"
    local cmd=(
      python -m sae_bench.evals.icl_features.main
      --dataset-path "${dataset}"
      --output-dir "${output_dir}"
      --model-name "${model_name}"
      --device "${device}"
      --batch-size "${DISCOVERY_BATCH_SIZE:-32}"
      --activation-threshold "${ACTIVATION_THRESHOLD:-0.0}"
      --min-example-fraction "${MIN_EXAMPLE_FRACTION:-0.9}"
      --min-query-fraction-per-context "${MIN_QUERY_FRACTION:-0.9}"
      --min-context-fraction "${MIN_FAMILY_FRACTION:-0.9}"
      --expected-examples "${target_examples}"
      --max-family-size-difference "${MAX_FAMILY_SIZE_DIFFERENCE:-1}"
      --sae-regex-pattern "^${release}$"
      --sae-block-pattern "^${sae_id}$"
    )
    if [[ "${task}" != "prontoqa" ]]; then
      cmd+=(--single-token-only)
    fi
    if [[ "${RECHECK_MODEL_CORRECT:-0}" != "0" ]]; then
      cmd+=(--require-model-correct)
    fi
    printf 'Running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  done
}

run_ablation() {
  local uid="$1"
  local run_root="$2"
  local discovery_root="$3"
  local task
  local feature_set
  for task in "${tasks[@]}"; do
    for feature_set in "${ablation_feature_sets[@]}"; do
      python -m sae_bench.evals.icl_features.ablation \
        --task "${task}" \
        --dataset-path "${data_root}/${model_name}/${task}.json" \
        --discovery-summary "${discovery_root}/${model_name}/${task}/summary.json" \
        --sae-uid "${uid}" \
        --feature-set "${feature_set}" \
        --output-dir "${run_root}/${model_name}/${uid}/${task}/ablation/${feature_set}" \
        --model-name "${model_name}" \
        --download-saes-dir "${download_saes_dir}" \
        --device "${device}" \
        --batch-size "${ABLATION_BATCH_SIZE:-32}" \
        --ablation-position "${ABLATION_POSITION:-final}" \
        --random-trials "${RANDOM_TRIALS:-20}" \
        --random-seed "${RANDOM_SEED:-0}" \
        --random-match-pool-size "${RANDOM_MATCH_POOL_SIZE:-100}" \
        --expected-examples "${target_examples}" \
        --allow-imperfect-baseline \
        --baseline-correct-only
    done
  done
}

run_iou() {
  local run_root="$1"
  local discovery_root="$2"
  local task_summary_args=()
  local task
  for task in "${tasks[@]}"; do
    task_summary_args+=(--task-summary "${task}=${discovery_root}/${model_name}/${task}/summary.json")
  done
  python -m sae_bench.evals.icl_features.iou \
    "${task_summary_args[@]}" \
    --output-root "${run_root}"
}

run_geometry() {
  local release="$1"
  local uid="$2"
  local run_root="$3"
  local discovery_root="$4"
  local task
  for task in "${tasks[@]}"; do
    local task_root="${run_root}/${model_name}/${uid}/${task}"
    TASK="${task}" \
    MODEL_NAME="${model_name}" \
    DATASET_PATH="${data_root}/${model_name}/${task}.json" \
    SUMMARY_PATH="${discovery_root}/${model_name}/${task}/summary.json" \
    SAE_REPO_ID="${release}" \
    SAE_UID="${uid}" \
    FEGA_CONFIG="${task_root}/fega_config.yaml" \
    FEGA_OUTPUT_ROOT="${task_root}/fega" \
    FEATURE_SET="${GEOMETRY_FEATURE_SET:-candidate}" \
    FEGA_RESUME="${resume}" \
    FEGA_PHASES="${FEGA_PHASES:-}" \
    DOWNLOAD_SAES_DIR="${download_saes_dir}" \
    DEVICE="${device}" \
      bash external/sae_bench/evals/icl_features/scripts/run_fega_geometry.sh
  done
}

run_plots() {
  local uid="$1"
  local run_root="$2"
  local map_args=()
  local task
  for task in "${tasks[@]}"; do
    map_args+=(
      --task-map
      "${task}=${run_root}/${model_name}/${uid}/${task}/fega/${model_name}/${task}.json/${task}_pointer_like/geometry_reporting/geometry_map_data.json"
    )
  done
  python -m sae_bench.evals.icl_features.geometry_plots \
    "${map_args[@]}" \
    --output-dir "${run_root}/${model_name}/${uid}/cross_task/geometry_plots" \
    --model-name "${model_name}" \
    --sae-uid "${uid}" \
    --embedding "${PLOT_EMBEDDING:-auto}" \
    --seed "${PLOT_SEED:-0}" \
    --title "${PLOT_TITLE:-SAE Feature Effect Geometry Across ICL Tasks}"
}

if has_stage geometry; then
  python - <<'PY'
import importlib.util

missing = [
    name
    for name in ("umap", "spherecluster")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing geometry dependencies: "
        + ", ".join(missing)
        + ". Install the project and apply the spherecluster compatibility patch."
    )
PY
  python scripts/bootstrap/install_vmf_spherecluster_patch.py --check
fi

release="${SAE_RELEASE:-$(default_release)}"
layer="${GEMMASCOPE_LAYER:-$(default_layer)}"

if has_stage validate; then
  mkdir -p "${base_result_root}/preflight"
  MODEL_NAME="${model_name}" \
  DATA_ROOT="${data_root}" \
  TASKS="${TASKS:-lsc wc tt prontoqa}" \
  TARGET_EXAMPLES="${target_examples}" \
  NUM_FAMILIES="${num_families}" \
  OUTPUT_PATH="${base_result_root}/preflight/${model_name}_datasets.json" \
  RECHECK_MODEL_CORRECT="${DATASET_FULL_RECHECK:-0}" \
  DEVICE="${device}" \
    bash external/sae_bench/evals/icl_features/scripts/validate_datasets.sh
fi

for width in "${widths[@]}"; do
  l0="$(configured_l0 "${width}")"
  sae_id="layer_${layer}/width_${width}/average_l0_${l0}"
  uid="$(sae_uid "${release}" "${sae_id}")"
  run_root="$(run_root_for "${width}" "${l0}")"
  discovery_root="${base_discovery_root}/$(basename "${run_root}")"
  mkdir -p "${run_root}/${model_name}"

  echo "=== Gemma Scope run: model=${model_name} release=${release} sae_id=${sae_id} uid=${uid} root=${run_root} discovery=${discovery_root} ==="

  if [[ "${RUN_SAE_LOAD_CHECK:-1}" != "0" ]]; then
    python - "${release}" "${sae_id}" "${device}" <<'PY'
import sys
from sae_lens import SAE

release, sae_id, device = sys.argv[1:4]
sae, cfg, sparsity = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=device,
)
print(
    "Loaded Gemma Scope SAE:",
    f"release={release}",
    f"sae_id={sae_id}",
    f"d_sae={getattr(sae.cfg, 'd_sae', 'unknown')}",
    f"hook={getattr(sae.cfg, 'hook_name', 'unknown')}",
)
del sae, cfg, sparsity
PY
  fi

  if has_stage discovery; then
    run_discovery "${release}" "${sae_id}" "${run_root}" "${discovery_root}"
  fi
  if has_stage ablation; then
    run_ablation "${uid}" "${run_root}" "${discovery_root}"
  fi
  if has_stage iou; then
    run_iou "${run_root}" "${discovery_root}"
  fi
  if has_stage geometry; then
    run_geometry "${release}" "${uid}" "${run_root}" "${discovery_root}"
  fi
  if has_stage plots; then
    run_plots "${uid}" "${run_root}"
  fi
  if has_stage aggregate; then
    python -m sae_bench.evals.icl_features.aggregate_results \
      --result-root "${run_root}" \
      --discovery-root "${discovery_root}"
  fi
done
