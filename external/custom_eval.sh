#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="cuda:0"
BATCH_SIZE=256

# --- Hugging Face Hub Configuration ---
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER

echo "Starting SAEBench evaluation run for Gemma-2-2B..."
echo "Using Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "-----------------------------------------------------"

python "${script_dir}/custom_eval.py" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"
