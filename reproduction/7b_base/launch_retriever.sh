#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

INDEX_FILE="${INDEX_FILE:-${WIKI18_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${WIKI18_DIR}/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-0}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cpu}"
RETRIEVER_MAX_RETURN_TOKENS="${RETRIEVER_MAX_RETURN_TOKENS:-400}"
RETRIEVER_CPU_THREADS="${RETRIEVER_CPU_THREADS:-16}"
PYTHON_BIN="${PYTHON_BIN:-}"

echo "[retriever-launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
echo "[retriever-launch] device=${RETRIEVER_DEVICE} faiss_gpu=${RETRIEVER_FAISS_GPU} index=${INDEX_FILE}" >&2

# CPU FAISS can trigger OpenBLAS "too many memory regions" crashes when the
# runtime creates one BLAS thread per visible CPU. Keep defaults conservative;
# override these env vars explicitly when the host has been validated.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${RETRIEVER_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${RETRIEVER_CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${RETRIEVER_CPU_THREADS}}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf_datasets}"
export RETRIEVER_DATASETS_CACHE="${RETRIEVER_DATASETS_CACHE:-${HF_DATASETS_CACHE}}"
export RETRIEVER_CORPUS_NUM_PROC="${RETRIEVER_CORPUS_NUM_PROC:-1}"

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

if [[ "${EXPECTED_RETRIEVER_INDEX:-}" == "hnsw" && "$(basename "${INDEX_FILE}")" != *HNSW* ]]; then
  echo "Expected an HNSW index for this run mode, got: ${INDEX_FILE}" >&2
  echo "Use e5_HNSW64.index; CPU Flat search is not practical for online RL." >&2
  exit 2
fi
if [[ "${EXPECTED_RETRIEVER_INDEX:-}" == "flat" && "$(basename "${INDEX_FILE}")" != *Flat* ]]; then
  echo "Expected an exact Flat index for this run mode, got: ${INDEX_FILE}" >&2
  echo "Use e5_Flat.index for paper-compatible exact retrieval." >&2
  exit 2
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
  --retriever_device "${RETRIEVER_DEVICE}" \
  --max_return_tokens "${RETRIEVER_MAX_RETURN_TOKENS}" \
  "${FAISS_ARGS[@]}"
