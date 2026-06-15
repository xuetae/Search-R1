#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${DATA_DIR}" "${WIKI18_DIR}"

echo "Downloading training/evaluation parquet files to ${DATA_DIR}"
if command -v hf >/dev/null 2>&1; then
  hf download PeterJinGo/nq_hotpotqa_train \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download \
    --repo-type dataset \
    PeterJinGo/nq_hotpotqa_train \
    --local-dir "${DATA_DIR}"
else
  echo "Neither hf nor huggingface-cli was found. Install huggingface_hub first." >&2
  exit 1
fi

echo "Downloading wiki-18 corpus and e5 Flat index parts to ${WIKI18_DIR}"
python "${WORK_DIR}/scripts/download.py" --save_path "${WIKI18_DIR}"

if [[ ! -f "${WIKI18_DIR}/e5_Flat.index" ]]; then
  echo "Merging FAISS index parts into ${WIKI18_DIR}/e5_Flat.index"
  cat "${WIKI18_DIR}"/part_* > "${WIKI18_DIR}/e5_Flat.index"
fi

if [[ -f "${WIKI18_DIR}/wiki-18.jsonl.gz" && ! -f "${WIKI18_DIR}/wiki-18.jsonl" ]]; then
  echo "Decompressing wiki-18.jsonl.gz"
  gzip -d "${WIKI18_DIR}/wiki-18.jsonl.gz"
fi

echo "Data preparation complete."
echo "DATA_DIR=${DATA_DIR}"
echo "WIKI18_DIR=${WIKI18_DIR}"

{
  echo "DATA_DIR=${DATA_DIR}"
  find "${DATA_DIR}" -maxdepth 1 -type f -print | sort
  echo "WIKI18_DIR=${WIKI18_DIR}"
  find "${WIKI18_DIR}" -maxdepth 1 -type f -print | sort
} > "${WIKI18_DIR}/../data_manifest.txt"
echo "Manifest: ${WIKI18_DIR}/../data_manifest.txt"
