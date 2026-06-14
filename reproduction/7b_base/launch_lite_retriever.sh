#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

CORPUS_FILE="${CORPUS_FILE:-${WIKI18_DIR}/wiki-18.jsonl}"
LITE_RETRIEVER_MAX_DOCS="${LITE_RETRIEVER_MAX_DOCS:-20000}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Neither python nor python3 was found in PATH." >&2
    exit 1
  fi
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus file: ${CORPUS_FILE}" >&2
  exit 1
fi

cd "${WORK_DIR}"
"${PYTHON_BIN}" search_r1/search/lite_retrieval_server.py \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RETRIEVER_TOPK}" \
  --max_docs "${LITE_RETRIEVER_MAX_DOCS}"
