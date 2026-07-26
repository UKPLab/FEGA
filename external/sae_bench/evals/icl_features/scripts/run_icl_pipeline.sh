#!/usr/bin/env bash
set -euo pipefail

# Required:
#   REPO_ID=owner/sae-repo
#   SAE_LOCATIONS="location/a location/b"
#
# Common overrides:
#   MODEL_NAME=gemma-2-2b
#   TASKS="lsc wc tt prontoqa"
#   DATA_ROOT=data/icl_features
#   DISCOVERY_ROOT=data/induction_feature_outputs/sae_geometry
#   DOWNLOAD_SAES_DIR=data/downloaded_saes
#   RESULT_ROOT=results/sae_geometry
#   STAGES="discovery ablation iou geometry plots aggregate"
#   RANDOM_TRIALS=20
#   ABLATION_FEATURE_SETS="threshold"
#   GEOMETRY_FEATURE_SET=candidate
#   RESUME=1

repo_id="${REPO_ID:?Set REPO_ID}"
read -r -a sae_locations <<< "${SAE_LOCATIONS:?Set SAE_LOCATIONS}"
read -r -a tasks <<< "${TASKS:-lsc wc tt prontoqa}"
read -r -a stages <<< "${STAGES:-discovery ablation iou geometry plots aggregate}"
read -r -a ablation_feature_sets <<< "${ABLATION_FEATURE_SETS:-threshold}"
model_name="${MODEL_NAME:-gemma-2-2b}"
data_root="${DATA_ROOT:-data/icl_features}"
result_root="${RESULT_ROOT:-results/sae_geometry}"
discovery_root="${DISCOVERY_ROOT:-data/induction_feature_outputs/$(basename "${result_root}")}"
download_saes_dir="${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"
resume="${RESUME:-1}"

has_stage() {
  local requested="$1"
  local stage
  for stage in "${stages[@]}"; do
    [[ "${stage}" == "${requested}" ]] && return 0
  done
  return 1
}

sae_uid() {
  local location="$1"
  local left="${repo_id//\//_}"
  local right="${location//\//_}"
  printf '%s__%s' "${left}" "${right}"
}

sae_label() {
  local location="$1"
  if [[ "${location}" == *"matryoshka"* ]]; then
    printf 'Matryoshka Batch TopK'
  elif [[ "${location}" == *"top_k"* ]]; then
    printf 'TopK'
  elif [[ "${location}" == *"standard"* || "${location}" == *"relu"* ]]; then
    printf 'ReLU'
  else
    printf '%s' "${location}"
  fi
}

mkdir -p "${discovery_root}/${model_name}"

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
if has_stage plots && [[ "${PLOT_EMBEDDING:-auto}" =~ ^(auto|umap)$ ]]; then
  python - <<'PY'
import importlib.util

if importlib.util.find_spec("umap") is None:
    raise SystemExit(
        "Missing plotting dependency: umap. Install the project or use "
        "PLOT_EMBEDDING=pca."
    )
PY
fi

if has_stage discovery; then
  for task in "${tasks[@]}"; do
    dataset="${data_root}/${model_name}/${task}.json"
    output_dir="${discovery_root}/${model_name}/${task}"
    if [[ "${resume}" != "0" && -f "${output_dir}/summary.json" ]]; then
      echo "Skipping completed discovery: ${task}"
      continue
    fi
    TASK="${task}" \
    MODEL_NAME="${model_name}" \
    DATASET_PATH="${dataset}" \
    OUTPUT_DIR="${output_dir}" \
    DOWNLOAD_SAES_DIR="${download_saes_dir}" \
    REPO_ID="${repo_id}" \
    SAE_LOCATIONS="${SAE_LOCATIONS}" \
    TARGET_EXAMPLES="${TARGET_EXAMPLES:-50000}" \
    BATCH_SIZE="${DISCOVERY_BATCH_SIZE:-32}" \
    ACTIVATION_THRESHOLD="${ACTIVATION_THRESHOLD:-0.0}" \
    MIN_EXAMPLE_FRACTION="${MIN_EXAMPLE_FRACTION:-0.9}" \
    MIN_QUERY_FRACTION="${MIN_QUERY_FRACTION:-0.9}" \
    MIN_FAMILY_FRACTION="${MIN_FAMILY_FRACTION:-0.9}" \
    RECHECK_MODEL_CORRECT="${RECHECK_MODEL_CORRECT:-1}" \
    DEVICE="${DEVICE:-cuda:0}" \
      bash external/sae_bench/evals/icl_features/scripts/discover_pointer_features.sh
  done
fi

if has_stage ablation; then
  for location in "${sae_locations[@]}"; do
    uid="$(sae_uid "${location}")"
    for task in "${tasks[@]}"; do
      for feature_set in "${ablation_feature_sets[@]}"; do
        python -m sae_bench.evals.icl_features.ablation \
          --task "${task}" \
          --dataset-path "${data_root}/${model_name}/${task}.json" \
          --discovery-summary "${discovery_root}/${model_name}/${task}/summary.json" \
          --sae-uid "${uid}" \
          --feature-set "${feature_set}" \
          --output-dir "${result_root}/${model_name}/${uid}/${task}/ablation/${feature_set}" \
          --model-name "${model_name}" \
          --download-saes-dir "${download_saes_dir}" \
          --device "${DEVICE:-cuda:0}" \
          --batch-size "${ABLATION_BATCH_SIZE:-32}" \
          --ablation-position "${ABLATION_POSITION:-final}" \
          --random-trials "${RANDOM_TRIALS:-20}" \
          --random-seed "${RANDOM_SEED:-0}" \
          --random-match-pool-size "${RANDOM_MATCH_POOL_SIZE:-100}" \
          --expected-examples "${TARGET_EXAMPLES:-50000}" \
          --allow-imperfect-baseline \
          --baseline-correct-only
      done
    done
  done
fi

task_summary_args=()
for task in "${tasks[@]}"; do
  task_summary_args+=(
    --task-summary
    "${task}=${discovery_root}/${model_name}/${task}/summary.json"
  )
done
if has_stage iou; then
  if [[ -n "${RAVEL_SUMMARY_PATH:-}" ]]; then
    python -m sae_bench.evals.icl_features.iou \
      "${task_summary_args[@]}" \
      --task-summary "ravel=${RAVEL_SUMMARY_PATH}" \
      --feature-sets threshold \
      --output-root "${result_root}"
    python -m sae_bench.evals.icl_features.iou \
      "${task_summary_args[@]}" \
      --feature-sets strict \
      --output-root "${result_root}"
  else
    python -m sae_bench.evals.icl_features.iou \
      "${task_summary_args[@]}" \
      --output-root "${result_root}"
  fi
fi

if has_stage geometry; then
  for location in "${sae_locations[@]}"; do
    uid="$(sae_uid "${location}")"
    for task in "${tasks[@]}"; do
      task_root="${result_root}/${model_name}/${uid}/${task}"
      TASK="${task}" \
      MODEL_NAME="${model_name}" \
      DATASET_PATH="${data_root}/${model_name}/${task}.json" \
      SUMMARY_PATH="${discovery_root}/${model_name}/${task}/summary.json" \
      SAE_REPO_ID="${repo_id}" \
      SAE_UID="${uid}" \
      FEGA_CONFIG="${task_root}/fega_config.yaml" \
      FEGA_OUTPUT_ROOT="${task_root}/fega" \
      FEATURE_SET="${GEOMETRY_FEATURE_SET:-candidate}" \
      FEGA_RESUME="${resume}" \
      FEGA_PHASES="${FEGA_PHASES:-}" \
      DOWNLOAD_SAES_DIR="${download_saes_dir}" \
      DEVICE="${DEVICE:-cuda:0}" \
        bash external/sae_bench/evals/icl_features/scripts/run_fega_geometry.sh
    done
  done
fi

if has_stage plots; then
  for location in "${sae_locations[@]}"; do
    uid="$(sae_uid "${location}")"
    map_args=()
    pointer_map_args=()
    pointer_task_count=0
    for task in "${tasks[@]}"; do
      map_path="${result_root}/${model_name}/${uid}/${task}/fega/${model_name}/${task}.json/${task}_pointer_like/geometry_reporting/geometry_map_data.json"
      map_args+=(--task-map "${task}=${map_path}")
      pointer_map_args+=(--task-map "${task}=${map_path}")
      pointer_task_count=$((pointer_task_count + 1))
    done
    python -m sae_bench.evals.icl_features.geometry_plots \
      "${map_args[@]}" \
      --output-dir "${result_root}/${model_name}/${uid}/cross_task/geometry_plots" \
      --model-name "${model_name}" \
      --sae-uid "${uid}" \
      --embedding "${PLOT_EMBEDDING:-auto}" \
      --seed "${PLOT_SEED:-0}" \
      --title "${PLOT_TITLE:-SAE Feature Effect Geometry Across ICL Tasks}"
    if [[ "${pointer_task_count}" -gt 0 ]]; then
      python -m sae_bench.evals.icl_features.geometry_plots \
        "${pointer_map_args[@]}" \
        --output-dir "${result_root}/${model_name}/${uid}/cross_task/pointer_only_geometry_plots" \
        --model-name "${model_name}" \
        --sae-uid "${uid}" \
        --embedding "${PLOT_EMBEDDING:-auto}" \
        --seed "${PLOT_SEED:-0}" \
        --title "SAE Feature Effect Geometry for Pointer-Like ICL Tasks"
    fi
  done
  sae_plot_args=()
  for location in "${sae_locations[@]}"; do
    uid="$(sae_uid "${location}")"
    sae_plot_args+=(
      --sae-plot
      "$(sae_label "${location}")=${result_root}/${model_name}/${uid}/cross_task/geometry_plots/geometry_plot_data.csv"
    )
  done
  if [[ "${#sae_locations[@]}" -gt 1 ]]; then
    python -m sae_bench.evals.icl_features.geometry_sae_grid_plots \
      "${sae_plot_args[@]}" \
      --output-dir "${result_root}/${model_name}/cross_sae/geometry_plots" \
      --basename "geometry_three_saes" \
      --embedding "${PLOT_EMBEDDING:-auto}" \
      --seed "${PLOT_SEED:-0}"
  fi
fi

if has_stage aggregate; then
  python -m sae_bench.evals.icl_features.aggregate_results \
    --result-root "${result_root}" \
    --discovery-root "${discovery_root}"
fi
