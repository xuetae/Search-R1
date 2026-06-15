#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash reproduction/7b_base/rebuild_platform_project.sh [options]

Options:
  --root PATH              Persistent project root. Default: /workspace/filesdir/search-r1
  --code-archive PATH      Search-R1 code archive with a Search-R1/ top folder.
  --llama-archive PATH     LLaMA-7B model tar.gz archive.
  --e5-archive PATH        e5-base-v2 model tar.gz archive.
  --skip-code              Do not restore code. Use this after git clone.
  --skip-models            Do not restore model archives.
  --download-data          Download train/test parquet, wiki-18, and e5 index.
  --preflight              Run preflight_check.sh after restore/download.
  --help                   Show this message.

The script creates a clean platform layout:
  ROOT/projects/Search-R1
  ROOT/data/nq_hotpotqa_train
  ROOT/data/wiki-18
  ROOT/models/7b_base/llama-7b
  ROOT/models/7b_base/e5-base-v2
  ROOT/outputs, ROOT/logs, ROOT/cache, ROOT/uploads
USAGE
}

ROOT="/workspace/filesdir/search-r1"
CODE_ARCHIVE=""
LLAMA_ARCHIVE=""
E5_ARCHIVE=""
SKIP_CODE=0
SKIP_MODELS=0
DOWNLOAD_DATA=0
RUN_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --code-archive)
      CODE_ARCHIVE="$2"
      shift 2
      ;;
    --llama-archive)
      LLAMA_ARCHIVE="$2"
      shift 2
      ;;
    --e5-archive)
      E5_ARCHIVE="$2"
      shift 2
      ;;
    --skip-code)
      SKIP_CODE=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --download-data)
      DOWNLOAD_DATA=1
      shift
      ;;
    --preflight)
      RUN_PREFLIGHT=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$(dirname "${ROOT}")" ]]; then
  echo "Parent directory does not exist: $(dirname "${ROOT}")" >&2
  echo "Mount the xFusion file-management directory first, usually at /filesdir." >&2
  exit 1
fi

PROJECTS_DIR="${ROOT}/projects"
PROJECT_DIR="${PROJECTS_DIR}/Search-R1"
MODEL_DIR="${ROOT}/models/7b_base"
DATA_DIR="${ROOT}/data/nq_hotpotqa_train"
WIKI_DIR="${ROOT}/data/wiki-18"

mkdir -p \
  "${PROJECTS_DIR}" \
  "${DATA_DIR}" \
  "${WIKI_DIR}" \
  "${MODEL_DIR}" \
  "${ROOT}/outputs/runs" \
  "${ROOT}/outputs/checkpoints" \
  "${ROOT}/logs" \
  "${ROOT}/cache/huggingface" \
  "${ROOT}/cache/modelscope" \
  "${ROOT}/cache/pip" \
  "${ROOT}/uploads"

if [[ "${SKIP_CODE}" -eq 0 ]]; then
  if [[ -z "${CODE_ARCHIVE}" ]]; then
    for candidate in \
      "${ROOT}/uploads/Search-R1-code-llama-7b-run-fbed069.tar.gz" \
      "/filesdir/Search-R1-code-llama-7b-run-fbed069.tar.gz" \
      "/filesdir/uploads/Search-R1-code-llama-7b-run-fbed069.tar.gz"; do
      if [[ -f "${candidate}" ]]; then
        CODE_ARCHIVE="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${CODE_ARCHIVE}" || ! -f "${CODE_ARCHIVE}" ]]; then
    echo "Missing code archive. Pass --code-archive /path/to/Search-R1-code-llama-7b-run-fbed069.tar.gz" >&2
    exit 1
  fi

  echo "Restoring code from ${CODE_ARCHIVE}"
  rm -rf "${PROJECT_DIR}"
  tar -xzf "${CODE_ARCHIVE}" -C "${PROJECTS_DIR}"
fi

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  if [[ -z "${LLAMA_ARCHIVE}" ]]; then
    for candidate in "${ROOT}/uploads/llama-7b.tar.gz" "/filesdir/llama-7b.tar.gz" "/filesdir/uploads/llama-7b.tar.gz"; do
      if [[ -f "${candidate}" ]]; then
        LLAMA_ARCHIVE="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${E5_ARCHIVE}" ]]; then
    for candidate in "${ROOT}/uploads/e5-base-v2.tar.gz" "/filesdir/e5-base-v2.tar.gz" "/filesdir/uploads/e5-base-v2.tar.gz"; do
      if [[ -f "${candidate}" ]]; then
        E5_ARCHIVE="${candidate}"
        break
      fi
    done
  fi

  if [[ -n "${LLAMA_ARCHIVE}" && -f "${LLAMA_ARCHIVE}" ]]; then
    echo "Restoring LLaMA model from ${LLAMA_ARCHIVE}"
    rm -rf "${MODEL_DIR}/llama-7b"
    tar -xzf "${LLAMA_ARCHIVE}" -C "${MODEL_DIR}"
  else
    echo "[warn] LLaMA archive not found. Expected final path: ${MODEL_DIR}/llama-7b" >&2
  fi

  if [[ -n "${E5_ARCHIVE}" && -f "${E5_ARCHIVE}" ]]; then
    echo "Restoring e5 retriever model from ${E5_ARCHIVE}"
    rm -rf "${MODEL_DIR}/e5-base-v2"
    tar -xzf "${E5_ARCHIVE}" -C "${MODEL_DIR}"
  else
    echo "[warn] e5 archive not found. Expected final path: ${MODEL_DIR}/e5-base-v2" >&2
  fi
fi

if [[ "${DOWNLOAD_DATA}" -eq 1 ]]; then
  if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Project directory is missing: ${PROJECT_DIR}" >&2
    echo "Restore code first or pass --skip-code only after code already exists." >&2
    exit 1
  fi
  echo "Downloading parquet files, wiki-18 corpus, and e5 index to ${ROOT}/data"
  cd "${PROJECT_DIR}"
  SEARCH_R1_ROOT="${ROOT}" \
  WORK_DIR="${PROJECT_DIR}" \
  HF_HOME="${ROOT}/cache/huggingface" \
  HUGGINGFACE_HUB_CACHE="${ROOT}/cache/huggingface/hub" \
  bash reproduction/7b_base/prepare_data.sh
fi

cat > "${ROOT}/README_LAYOUT.txt" <<EOF
Search-R1 persistent root:
${ROOT}

Expected paths:
${PROJECT_DIR}
${DATA_DIR}/train.parquet
${DATA_DIR}/test.parquet
${WIKI_DIR}/wiki-18.jsonl
${WIKI_DIR}/e5_Flat.index
${MODEL_DIR}/llama-7b
${MODEL_DIR}/e5-base-v2

Prepare missing data:
cd ${PROJECT_DIR}
SEARCH_R1_ROOT=${ROOT} WORK_DIR=${PROJECT_DIR} bash reproduction/7b_base/prepare_data.sh

Preflight:
cd ${PROJECT_DIR}
SEARCH_R1_ROOT=${ROOT} WORK_DIR=${PROJECT_DIR} bash reproduction/7b_base/preflight_check.sh

xFusion training command:
SEARCH_R1_ROOT=${ROOT} WORK_DIR=${PROJECT_DIR} python3 ${PROJECT_DIR}/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
EOF

echo
echo "Platform layout under ${ROOT}:"
find "${ROOT}" -maxdepth 3 -type d | sort
echo
echo "Expected file status:"
for path in \
  "${DATA_DIR}/train.parquet" \
  "${DATA_DIR}/test.parquet" \
  "${WIKI_DIR}/wiki-18.jsonl" \
  "${WIKI_DIR}/e5_Flat.index" \
  "${MODEL_DIR}/llama-7b" \
  "${MODEL_DIR}/e5-base-v2"; do
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null || ls -lh "${path}"
  else
    echo "[missing] ${path}"
  fi
done

if [[ "${RUN_PREFLIGHT}" -eq 1 ]]; then
  cd "${PROJECT_DIR}"
  SEARCH_R1_ROOT="${ROOT}" WORK_DIR="${PROJECT_DIR}" bash reproduction/7b_base/preflight_check.sh
fi

cat <<EOF

Next data command if not downloaded yet:
cd ${PROJECT_DIR}
SEARCH_R1_ROOT=${ROOT} WORK_DIR=${PROJECT_DIR} bash reproduction/7b_base/prepare_data.sh

Next training-task command:
SEARCH_R1_ROOT=${ROOT} WORK_DIR=${PROJECT_DIR} python3 ${PROJECT_DIR}/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
EOF
