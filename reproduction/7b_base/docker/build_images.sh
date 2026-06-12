#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"

cd "${REPO_ROOT}"

docker buildx build \
  --platform "${IMAGE_PLATFORM}" \
  -f reproduction/7b_base/docker/Dockerfile.searchr1 \
  -t "${SEARCHR1_IMAGE}" \
  --load \
  .

docker buildx build \
  --platform "${IMAGE_PLATFORM}" \
  -f reproduction/7b_base/docker/Dockerfile.retriever \
  -t "${RETRIEVER_IMAGE}" \
  --load \
  .

docker image ls "${SEARCHR1_IMAGE}"
docker image ls "${RETRIEVER_IMAGE}"

