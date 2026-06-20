#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

TARGET="${LLAMA_INSTRUCT_LOCAL_MODEL:-${MODEL_ROOT}/llama-7b-instruct}"
MODEL_ID="${LLAMA_INSTRUCT_MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}"

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

echo "Llama-2-7B-Chat downloaded to ${TARGET}"
ls -lh "${TARGET}/config.json"
