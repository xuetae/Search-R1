#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
# shellcheck source=../env.sh
source "${REPO_ROOT}/reproduction/7b_base/env.sh"

INCLUDE_DATA="${INCLUDE_DATA:-1}"
INCLUDE_MODELS="${INCLUDE_MODELS:-1}"
SPLIT_SIZE="${SPLIT_SIZE:-}"

cd "${REPO_ROOT}"

bash reproduction/7b_base/docker/save_images.sh

mkdir -p "${DATA_BUNDLE_DIR}"

if [[ "${INCLUDE_DATA}" == "1" ]]; then
  for path in "${DATA_DIR}" "${WIKI18_DIR}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing data path: ${path}" >&2
      echo "Run reproduction/7b_base/prepare_data.sh first, or set INCLUDE_DATA=0." >&2
      exit 1
    fi
  done
  tar -czf "${DATA_BUNDLE_DIR}/searchr1-7b-data.tar.gz" \
    -C "${WORK_DIR}" \
    "data/${DATA_NAME}" \
    "data/wiki-18"
fi

if [[ "${INCLUDE_MODELS}" == "1" ]]; then
  if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "Missing model path: ${MODEL_ROOT}" >&2
    echo "Run reproduction/7b_base/download_models.sh first, or set INCLUDE_MODELS=0." >&2
    exit 1
  fi
  tar -czf "${DATA_BUNDLE_DIR}/searchr1-7b-models.tar.gz" \
    -C "${WORK_DIR}" \
    "models/7b_base"
fi

cat > "${UPLOAD_BUNDLE_DIR}/restore_on_server.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(pwd)"

bash "${SCRIPT_DIR}/images/load_images_on_server.sh"

if [[ -f "${SCRIPT_DIR}/data/searchr1-7b-data.tar.gz" ]]; then
  tar -xzf "${SCRIPT_DIR}/data/searchr1-7b-data.tar.gz" -C "${REPO_ROOT}"
fi

if [[ -f "${SCRIPT_DIR}/data/searchr1-7b-models.tar.gz" ]]; then
  tar -xzf "${SCRIPT_DIR}/data/searchr1-7b-models.tar.gz" -C "${REPO_ROOT}"
fi

bash reproduction/7b_base/preflight_check.sh
EOF
chmod +x "${UPLOAD_BUNDLE_DIR}/restore_on_server.sh"

if [[ -n "${SPLIT_SIZE}" ]]; then
  tar -czf - "${UPLOAD_BUNDLE_DIR}" | split -b "${SPLIT_SIZE}" - "${UPLOAD_BUNDLE_DIR}.tar.gz.part-"
  echo "Created split archive parts: ${UPLOAD_BUNDLE_DIR}.tar.gz.part-*"
else
  tar -czf "${UPLOAD_BUNDLE_DIR}.tar.gz" "${UPLOAD_BUNDLE_DIR}"
  echo "Created upload bundle: ${REPO_ROOT}/${UPLOAD_BUNDLE_DIR}.tar.gz"
fi

