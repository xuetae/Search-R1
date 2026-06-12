#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"

docker run --rm -it \
  --gpus all \
  --network host \
  -v "${REPO_ROOT}/data:/workspace/Search-R1/data" \
  -v "${REPO_ROOT}/models/7b_base:/workspace/Search-R1/models/7b_base" \
  "${RETRIEVER_IMAGE}" \
  bash reproduction/7b_base/launch_retriever.sh

