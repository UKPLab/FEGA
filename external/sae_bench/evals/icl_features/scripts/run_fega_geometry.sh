#!/usr/bin/env bash
set -euo pipefail

task="${TASK:?Set TASK to lsc, wc, tt, or prontoqa}"
model_name="${MODEL_NAME:-gemma-2-2b}"
sae_repo_id="${SAE_REPO_ID:?Set SAE_REPO_ID to the SAE repository used for discovery}"
dataset="${DATASET_PATH:-data/icl_features/${model_name}/${task}.json}"
summary="${SUMMARY_PATH:-data/induction_feature_outputs/${model_name}/${task}/summary.json}"
config="${FEGA_CONFIG:-fega/config/icl/${model_name}_${task}.yaml}"

config_cmd=(
  python external/sae_bench/evals/icl_features/scripts/write_fega_config.py
  --task "${task}"
  --dataset "${dataset}"
  --summary "${summary}"
  --sae-repo-id "${sae_repo_id}"
  --output "${config}"
  --output-root "${FEGA_OUTPUT_ROOT:-results/sae_geometry/${model_name}/${task}}"
  --device "${DEVICE:-cuda:0}"
  --download-saes-dir "${DOWNLOAD_SAES_DIR:-data/downloaded_saes}"
  --feature-set "${FEATURE_SET:-candidate}"
)
if [[ -n "${SAE_UID:-}" ]]; then
  config_cmd+=(--sae-uid "${SAE_UID}")
fi
if [[ "${FEGA_SMOKE:-0}" != "0" ]]; then
  config_cmd+=(
    --smoke-profile
    --smoke-max-features "${FEGA_SMOKE_MAX_FEATURES:-2}"
  )
fi
"${config_cmd[@]}"

zero_features="$(
  python - "${config}" <<'PY'
import csv
import json
import sys

from fega.config_schema import FEGAPipelineConfig
from fega.core.data_prep.induction import resolve_induction_feature_set
from fega.paths import (
    geometry_reporting_counts_path,
    geometry_reporting_dir,
    geometry_reporting_gate_diagnostics_json_path,
    geometry_reporting_gate_diagnostics_md_path,
    geometry_reporting_map_data_path,
    geometry_reporting_records_csv_path,
    geometry_reporting_records_path,
    geometry_reporting_stats_path,
)

config = FEGAPipelineConfig.from_file(sys.argv[1])
_, feature_set = resolve_induction_feature_set(config)
if feature_set.selected_feature_ids:
    print("0")
    raise SystemExit(0)

out_dir = geometry_reporting_dir(config)
out_dir.mkdir(parents=True, exist_ok=True)
records_payload = {
    "phase": "geometry_reporting",
    "schema_version": 1,
    "label_version": "none",
    "threshold_profile": config.phases.geometry_reporting.threshold_profile,
    "config": config.to_dict()["phases"]["geometry_reporting"],
    "source_paths": {},
    "summary": {
        "features_total": 0,
        "primary_label_counts": {},
        "terminal_reason_counts": {},
        "global_flag_counts": {},
    },
    "diagnostics_paths": {
        "gate_diagnostics_json": str(geometry_reporting_gate_diagnostics_json_path(config)),
        "gate_diagnostics_markdown": str(geometry_reporting_gate_diagnostics_md_path(config)),
    },
    "features": [],
    "empty_reason": "no_selected_features",
}
geometry_reporting_records_path(config).write_text(json.dumps(records_payload, indent=2) + "\n")
with geometry_reporting_records_csv_path(config).open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["feature_id", "primary_label", "terminal_reason"])
with geometry_reporting_counts_path(config).open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["kind", "name", "count", "fraction"])
geometry_reporting_stats_path(config).write_text(
    "# Geometry Reporting Statistics\n\n"
    "- total_features: 0\n"
    "- empty_reason: no_selected_features\n"
)
geometry_reporting_gate_diagnostics_json_path(config).write_text(
    json.dumps({"features": [], "empty_reason": "no_selected_features"}, indent=2) + "\n"
)
geometry_reporting_gate_diagnostics_md_path(config).write_text(
    "# Geometry Gate Diagnostics\n\n- empty_reason: no_selected_features\n"
)
map_payload = {
    "phase": "geometry_reporting",
    "schema_version": 1,
    "embedding": {"method": "none", "reason": "no_selected_features"},
    "preprocessing": {},
    "palette": {},
    "features": [],
    "figure_metadata": {
        "atlas": None,
        "atlas_point_count": 0,
        "atlas_excluded_point_count": 0,
        "atlas_label_counts": {},
        "zooms": {},
        "stats": str(geometry_reporting_stats_path(config)),
        "counts": str(geometry_reporting_counts_path(config)),
    },
}
geometry_reporting_map_data_path(config).write_text(json.dumps(map_payload, indent=2) + "\n")
print("1")
PY
)"
if [[ "${zero_features}" == "1" ]]; then
  echo "Skipping FEGA geometry because selected feature set is empty: ${config}"
  exit 0
fi

cmd=(python -m fega.cli run --config "${config}")
if [[ "${FEGA_RESUME:-1}" != "0" ]]; then
  cmd+=(--resume)
fi
if [[ -n "${FEGA_PHASES:-}" ]]; then
  cmd+=(--phases "${FEGA_PHASES}")
fi
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
