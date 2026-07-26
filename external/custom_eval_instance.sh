#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER

# Configuration
DEVICE="cuda:0"
BATCH_SIZE=128

echo "Starting SAEBench instance-level evaluation run for Gemma-2-2B..."
echo "Using Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "-----------------------------------------------------"

python "${SCRIPT_DIR}/custom_eval_instance.py" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

echo "All evaluations complete!"
echo "Check the 'data/instance_eval_results/' directory for detailed, per-example output."
