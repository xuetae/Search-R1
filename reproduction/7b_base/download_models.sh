#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

HF_BASE_MODEL_ID="${HF_BASE_MODEL_ID:-Qwen/Qwen2.5-7B}"
HF_RETRIEVER_MODEL_ID="${HF_RETRIEVER_MODEL_ID:-intfloat/e5-base-v2}"

mkdir -p "${MODEL_ROOT}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli was not found. Activate searchr1 or retriever env first." >&2
  exit 1
fi

echo "Downloading base model ${HF_BASE_MODEL_ID} to ${LOCAL_BASE_MODEL}"
huggingface-cli download "${HF_BASE_MODEL_ID}" \
  --local-dir "${LOCAL_BASE_MODEL}" \
  --local-dir-use-symlinks False

echo "Downloading retriever model ${HF_RETRIEVER_MODEL_ID} to ${LOCAL_RETRIEVER_MODEL}"
huggingface-cli download "${HF_RETRIEVER_MODEL_ID}" \
  --local-dir "${LOCAL_RETRIEVER_MODEL}" \
  --local-dir-use-symlinks False

echo "Model download complete."
echo "BASE_MODEL=${LOCAL_BASE_MODEL}"
echo "RETRIEVER_MODEL=${LOCAL_RETRIEVER_MODEL}"

