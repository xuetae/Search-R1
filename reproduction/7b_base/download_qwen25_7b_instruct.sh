#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
BASE_MODEL_NAME=qwen2.5-7b-instruct source "${SCRIPT_DIR}/env.sh"

TARGET="${LOCAL_BASE_MODEL:-${MODEL_ROOT}/qwen2.5-7b-instruct}"
MODEL_ID="${HF_BASE_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"

mkdir -p "${TARGET}"

if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${TARGET}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" \
    --local-dir "${TARGET}" \
    --local-dir-use-symlinks False
else
  echo "Neither hf nor huggingface-cli was found." >&2
  exit 1
fi

echo "Qwen2.5-7B-Instruct downloaded to ${TARGET}"
ls -lh "${TARGET}/config.json"
