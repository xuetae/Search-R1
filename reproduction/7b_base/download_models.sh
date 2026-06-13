#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${MODEL_ROOT}"

download_base_model() {
  case "${BASE_MODEL_SOURCE}" in
    hf)
      if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli was not found. Activate searchr1 or retriever env first." >&2
        exit 1
      fi
      echo "Downloading base model ${HF_BASE_MODEL_ID} to ${LOCAL_BASE_MODEL}"
      huggingface-cli download "${HF_BASE_MODEL_ID}" \
        --local-dir "${LOCAL_BASE_MODEL}" \
        --local-dir-use-symlinks False
      ;;
    modelscope)
      if ! command -v modelscope >/dev/null 2>&1; then
        echo "modelscope was not found. Install it first:" >&2
        echo "  python -m pip install --no-cache-dir modelscope" >&2
        exit 1
      fi
      echo "Downloading base model ${MS_BASE_MODEL_ID} to ${LOCAL_BASE_MODEL}"
      modelscope download --model "${MS_BASE_MODEL_ID}" --local_dir "${LOCAL_BASE_MODEL}"
      ;;
    *)
      echo "Unsupported BASE_MODEL_SOURCE=${BASE_MODEL_SOURCE}. Use hf or modelscope." >&2
      exit 2
      ;;
  esac
}

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli was not found. Activate searchr1 or retriever env first." >&2
  exit 1
fi

download_base_model

echo "Downloading retriever model ${HF_RETRIEVER_MODEL_ID} to ${LOCAL_RETRIEVER_MODEL}"
huggingface-cli download "${HF_RETRIEVER_MODEL_ID}" \
  --local-dir "${LOCAL_RETRIEVER_MODEL}" \
  --local-dir-use-symlinks False

echo "Model download complete."
echo "BASE_MODEL=${LOCAL_BASE_MODEL}"
echo "RETRIEVER_MODEL=${LOCAL_RETRIEVER_MODEL}"
