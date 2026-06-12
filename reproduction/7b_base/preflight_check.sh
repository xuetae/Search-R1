#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

missing=0

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "${path}" ]]; then
    echo "[ok] ${label}: ${path}"
  else
    echo "[missing] ${label}: ${path}" >&2
    missing=1
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    echo "[ok] ${label}: ${path}"
  else
    echo "[missing] ${label}: ${path}" >&2
    missing=1
  fi
}

check_file "${DATA_DIR}/train.parquet" "train parquet"
check_file "${DATA_DIR}/test.parquet" "test parquet"
check_file "${WIKI18_DIR}/wiki-18.jsonl" "wiki-18 corpus"
check_file "${WIKI18_DIR}/e5_Flat.index" "e5 FAISS index"
check_dir "${LOCAL_BASE_MODEL}" "local 7B base model"
check_dir "${LOCAL_RETRIEVER_MODEL}" "local e5 retriever model"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "[warn] nvidia-smi not found; GPU availability was not checked." >&2
fi

if [[ "${missing}" -ne 0 ]]; then
  echo "Preflight failed. Run setup_envs.sh, download_models.sh, and prepare_data.sh as needed." >&2
  exit 1
fi

echo "Preflight passed."

