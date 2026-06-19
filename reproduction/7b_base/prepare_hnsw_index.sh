#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

TARGET="${INDEX_FILE:-${WIKI18_DIR}/e5_HNSW64.index}"
PARTS_DIR="${HNSW_PARTS_DIR:-${WIKI18_DIR}/hnsw64_parts}"

if [[ -f "${TARGET}" ]]; then
  echo "HNSW64 index already exists: ${TARGET}"
  exit 0
fi

mkdir -p "${WIKI18_DIR}" "${PARTS_DIR}"

if command -v hf >/dev/null 2>&1; then
  hf download PeterJinGo/wiki-18-e5-index-HNSW64 \
    --repo-type dataset \
    --local-dir "${PARTS_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download \
    --repo-type dataset \
    PeterJinGo/wiki-18-e5-index-HNSW64 \
    --local-dir "${PARTS_DIR}"
else
  echo "Neither hf nor huggingface-cli was found." >&2
  exit 1
fi

mapfile -t PARTS < <(find "${PARTS_DIR}" -maxdepth 1 -type f -name 'part_*' | sort)
if [[ "${#PARTS[@]}" -eq 0 ]]; then
  echo "No HNSW64 index parts were downloaded to ${PARTS_DIR}." >&2
  exit 1
fi

echo "Merging ${#PARTS[@]} HNSW64 parts into ${TARGET}"
cat "${PARTS[@]}" > "${TARGET}"
ls -lh "${TARGET}"
