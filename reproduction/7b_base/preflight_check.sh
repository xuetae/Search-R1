#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

missing=0
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN=""
  fi
fi

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
check_dir "${LOCAL_BASE_MODEL}" "local ${BASE_MODEL_NAME} base model"
check_dir "${LOCAL_RETRIEVER_MODEL}" "local e5 retriever model"

if [[ -n "${PYTHON_BIN}" ]]; then
  if "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = ["torch", "transformers", "vllm", "faiss", "flash_attn"]
failed = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"[ok] python package: {name}")
    except Exception as exc:
        print(f"[missing] python package: {name}: {exc}", file=sys.stderr)
        failed.append(name)
sys.exit(1 if failed else 0)
PY
  then
    :
  else
    missing=1
  fi
else
  echo "[missing] python executable: python/python3 not found" >&2
  missing=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "[warn] nvidia-smi not found; GPU availability was not checked." >&2
fi

if [[ "${missing}" -ne 0 ]]; then
  echo "Preflight failed. Run setup_envs.sh, download_models.sh, and prepare_data.sh as needed." >&2
  echo "Current model settings:" >&2
  echo "  BASE_MODEL_NAME=${BASE_MODEL_NAME}" >&2
  echo "  LOCAL_BASE_MODEL=${LOCAL_BASE_MODEL}" >&2
  echo "  HF_BASE_MODEL_ID=${HF_BASE_MODEL_ID}" >&2
  echo "For a pre-uploaded LLaMA-7B directory, set LOCAL_BASE_MODEL=/path/to/llama-7b or put it at ${LOCAL_BASE_MODEL}." >&2
  exit 1
fi

echo "Preflight passed."
