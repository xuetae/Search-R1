#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"

"${SCRIPT_DIR}/setup_envs.sh"

echo "Downloading local model copies"
"${CONDA_BIN}" run -n searchr1 bash "${SCRIPT_DIR}/download_models.sh"

echo "Downloading training data and wiki-18 retrieval files"
"${CONDA_BIN}" run -n searchr1 bash "${SCRIPT_DIR}/prepare_data.sh"

"${SCRIPT_DIR}/preflight_check.sh"

echo "Bootstrap complete."
echo "Start the retriever in one shell:"
echo "  conda activate retriever"
echo "  bash reproduction/7b_base/launch_retriever.sh"
echo "Start training in another shell:"
echo "  conda activate searchr1"
echo "  bash reproduction/7b_base/train_grpo_7b_base.sh"
