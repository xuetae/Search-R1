#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

INDEX_FILE="${INDEX_FILE:-${WIKI18_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${WIKI18_DIR}/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"

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

cd "${WORK_DIR}"
python search_r1/search/retrieval_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RETRIEVER_TOPK}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --faiss_gpu
