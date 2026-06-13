#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${MODEL_ROOT}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli was not found. Activate searchr1 or retriever env first." >&2
  exit 1
fi

if [[ "${SKIP_BASE_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "Skipping base model download. LOCAL_BASE_MODEL=${LOCAL_BASE_MODEL}"
else
  echo "Downloading base model ${HF_BASE_MODEL_ID} to ${LOCAL_BASE_MODEL}"
  huggingface-cli download "${HF_BASE_MODEL_ID}" \
    --local-dir "${LOCAL_BASE_MODEL}" \
    --local-dir-use-symlinks False
fi

if [[ "${SKIP_RETRIEVER_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "Skipping retriever model download. LOCAL_RETRIEVER_MODEL=${LOCAL_RETRIEVER_MODEL}"
else
  echo "Downloading retriever model ${HF_RETRIEVER_MODEL_ID} to ${LOCAL_RETRIEVER_MODEL}"
  huggingface-cli download "${HF_RETRIEVER_MODEL_ID}" \
    --local-dir "${LOCAL_RETRIEVER_MODEL}" \
    --local-dir-use-symlinks False
fi

echo "Model download complete."
echo "BASE_MODEL=${LOCAL_BASE_MODEL}"
echo "RETRIEVER_MODEL=${LOCAL_RETRIEVER_MODEL}"
