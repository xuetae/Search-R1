#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"

mkdir -p "${REPO_ROOT}/${IMAGE_BUNDLE_DIR}"

docker save "${SEARCHR1_IMAGE}" | gzip -c > "${REPO_ROOT}/${IMAGE_BUNDLE_DIR}/searchr1-7b-cuda121.tar.gz"
docker save "${RETRIEVER_IMAGE}" | gzip -c > "${REPO_ROOT}/${IMAGE_BUNDLE_DIR}/searchr1-retriever-cuda121.tar.gz"

cat > "${REPO_ROOT}/${IMAGE_BUNDLE_DIR}/load_images_on_server.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gunzip -c "${SCRIPT_DIR}/searchr1-7b-cuda121.tar.gz" | docker load
gunzip -c "${SCRIPT_DIR}/searchr1-retriever-cuda121.tar.gz" | docker load

docker image ls 'searchr1*'
EOF
chmod +x "${REPO_ROOT}/${IMAGE_BUNDLE_DIR}/load_images_on_server.sh"

echo "Saved image archives under ${REPO_ROOT}/${IMAGE_BUNDLE_DIR}"

