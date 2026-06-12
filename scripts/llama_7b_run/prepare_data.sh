#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${DATA_DIR}" "${WIKI18_DIR}"

echo "Downloading training/evaluation parquet files to ${DATA_DIR}"
huggingface-cli download \
  --repo-type dataset \
  PeterJinGo/nq_hotpotqa_train \
  --local-dir "${DATA_DIR}"

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

