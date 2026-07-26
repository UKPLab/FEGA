#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

#Actual path to the JSON output file from the instance-level evaluation run.
INPUT_JSON="data/instance_eval_results/ravel/saebench_gemma-2-2b_width-2pow16_date-0107_gemma-2-2b_standard_new_width-2pow16_date-0107_resid_post_layer_12_trainer_0_custom_sae_eval_results.json"

# output path
OUTPUT_JSON="data/artifacts/ravel/failure_atlas/ravel_failure_atlas.json"

echo "Creating Failure Atlas from: $INPUT_JSON"
echo "-----------------------------------------------------"

# Execute the Python script
python "${SCRIPT_DIR}/create_failure_atlas_ravel.py" "$INPUT_JSON" --output_json "$OUTPUT_JSON"

echo "Failure Atlas created at: $OUTPUT_JSON"
