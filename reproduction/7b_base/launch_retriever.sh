#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

INDEX_FILE="${INDEX_FILE:-${WIKI18_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${WIKI18_DIR}/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-0}"
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

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing index file: ${INDEX_FILE}" >&2
  echo "Run reproduction/7b_base/prepare_data.sh first." >&2
  exit 1
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus file: ${CORPUS_FILE}" >&2
  echo "Run reproduction/7b_base/prepare_data.sh first." >&2
  exit 1
fi

FAISS_ARGS=()
if [[ "${RETRIEVER_FAISS_GPU}" == "1" || "${RETRIEVER_FAISS_GPU}" == "true" ]]; then
  FAISS_ARGS+=(--faiss_gpu)
fi

cd "${WORK_DIR}"
"${PYTHON_BIN}" search_r1/search/retrieval_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RETRIEVER_TOPK}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  "${FAISS_ARGS[@]}"
