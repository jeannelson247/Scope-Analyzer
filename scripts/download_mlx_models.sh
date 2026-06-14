#!/bin/zsh
# download_mlx_models.sh - download only the benchmark-selected shipping set.
#
# Usage:
#   ./scripts/download_mlx_models.sh
#   MODELS_DIR=/Volumes/MySSD/ScopeStudioModels/mlx ./scripts/download_mlx_models.sh
#   INCLUDE_OPTIONAL=1 ./scripts/download_mlx_models.sh
#
# Destination default:
#   scripts/sync_shipping_models.py auto-prefers a plugged-in model vault:
#     /Volumes/<drive>/ScopeStudioModels/mlx
#     /Volumes/<drive>/models/mlx
#     /Volumes/<drive>/mlx
#   and falls back to ~/models/mlx when no model drive is mounted.
#
# The selected models live in config/shipping_models.json and are justified by
# backtests/mlx_model_benchmark_report.txt. This script does NOT download the
# exploratory benchmark fleet.
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c "import huggingface_hub" 2>/dev/null || "$PYTHON" -m pip install -U huggingface_hub
"$PYTHON" -c "import hf_transfer" 2>/dev/null || "$PYTHON" -m pip install -q hf_transfer
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ -z "$HF_TOKEN" ]; then
  echo "NOTE: unauthenticated - downloads may be rate-limited."
  echo "      Set HF_TOKEN or run Hugging Face login once for full speed."
fi

ARGS=(--download-missing)
if [ -n "$MODELS_DIR" ]; then
  ARGS+=(--dest "$MODELS_DIR")
fi
if [ "$INCLUDE_OPTIONAL" = "1" ]; then
  ARGS+=(--include-optional)
fi

python3 scripts/sync_shipping_models.py "${ARGS[@]}"
