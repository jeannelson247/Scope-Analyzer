#!/bin/zsh
# download_mlx_models.sh - pull the six MLX benchmark candidates.
#
# Usage:
#   ./scripts/download_mlx_models.sh                    # to ~/models/mlx
#   MODELS_DIR=/Volumes/MySSD/mlx ./scripts/download_mlx_models.sh
#
# External SSD: set MODELS_DIR to a folder on the SSD, then (optional)
#   ln -s /Volumes/MySSD/mlx ~/models/mlx
# Loading from an external SSD is fine - the model is read once into
# unified memory; only the first-load time is slower.
#
# NOTE: repo names on mlx-community drift as new quants land. If a
# download 404s, search https://huggingface.co/mlx-community for the
# model family and substitute the current 4-bit repo name.
set -e
MODELS_DIR="${MODELS_DIR:-$HOME/models/mlx}"
mkdir -p "$MODELS_DIR"
command -v hf >/dev/null 2>&1 || pip install -U huggingface_hub

# ---- speed: accelerated transfer + auth ------------------------------
# newer hub versions use the Xet backend (HF_XET_HIGH_PERFORMANCE);
# HF_HUB_ENABLE_HF_TRANSFER is kept for older versions (harmless warning)
python3 -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# token lifts the anonymous rate limit; set it up ONCE with:
#   hf auth login        (paste a free 'Read' token from
#                         huggingface.co -> Settings -> Access Tokens)
# NEVER hard-code a token in this script or commit one to the repo.
if ! hf auth whoami >/dev/null 2>&1 && [ -z "$HF_TOKEN" ]; then
  echo "NOTE: unauthenticated - downloads are rate-limited."
  echo "      Run 'hf auth login' once for full speed, then re-run."
fi

# Six candidates, all <= ~18 GB on disk (4-bit MLX), chosen to span:
# coding MoE, Nemotron (requested), dense coder, balanced generalist,
# current champion, and a small router-class control.
MODELS=(
  "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"      # ~17 GB coding MoE - top coder at 24 GB
  "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"    # ~16 GB Nemotron MoE (requested)
  "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"        # ~8 GB dense coder
  "mlx-community/Qwen3.5-9B-4bit"                        # ~6 GB current champion (baseline)
  "mlx-community/gemma-3-12b-it-4bit"                    # ~7 GB diversity pick
  "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit"         # ~2.5 GB small control
  # ---- lightweight additions (small MoEs + dense ladder) ----
  "mlx-community/LFM2-8B-A1B-MoE-4bit"                   # ~4.5 GB MoE, only 1B ACTIVE - fast
  "mlx-community/Qwen3.5-4B-MLX-4bit"                    # ~2.5 GB dense ladder step
  # ---- reasoning experiments (analyst/math roles ONLY: R1-style
  # models think before answering and make poor tool routers - judge
  # them on families F/G/physics, not routing B/E) -------------------
  "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"     # ~1 GB tiny reasoner; LITE-version candidate
  "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"       # ~4.5 GB mid reasoner (failed ROUTING in CS45 - retest as analyst)
)
# DeepScaleR-1.5B (math-RL reasoner): only third-party MLX conversions
# exist on the hub - cleaner supply chain is converting the official
# weights yourself (one command, ~2 min):
#   python3 -m mlx_lm convert --hf-path agentica-org/DeepScaleR-1.5B-Preview \
#       -q --mlx-path "$MODELS_DIR/DeepScaleR-1.5B-Preview-4bit"

for repo in "${MODELS[@]}"; do
  name="${repo##*/}"
  dest="$MODELS_DIR/$name"
  # ALWAYS run hf download: it verifies and resumes - a complete model
  # is a fast no-op, a partial one (e.g. interrupted at 42%) continues.
  # (A bare folder-exists check wrongly skipped incomplete downloads.)
  echo "== syncing $repo -> $dest"
  hf download "$repo" --local-dir "$dest" || \
    echo "!! $repo failed - check the current repo name on huggingface.co/mlx-community"
done

echo ""
echo "Done. Disk usage:"
du -sh "$MODELS_DIR"/* 2>/dev/null
echo ""
echo "Benchmark them with:"
echo "  python3 scripts/benchmark_data_analysis.py --backend mlx \\"
echo "      --models \"$MODELS_DIR/<name1>,$MODELS_DIR/<name2>,...\""
