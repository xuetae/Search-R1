#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
elif [[ "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: bash reproduction/7b_base/organize_online_dev_files.sh [--dry-run|--apply]" >&2
  exit 2
fi

MANIFEST_DIR="${OUTPUT_ROOT}/layout_manifests"
MANIFEST_FILE="${MANIFEST_DIR}/layout_manifest_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "${MANIFEST_DIR}"

log() {
  echo "$*" | tee -a "${MANIFEST_FILE}"
}

run_cmd() {
  log "+ $*"
  if [[ "${APPLY}" -eq 1 ]]; then
    "$@"
  fi
}

ensure_layout() {
  run_cmd mkdir -p "${SEARCH_R1_ROOT}/projects"
  run_cmd mkdir -p "${SEARCH_R1_ROOT}/venvs"
  run_cmd mkdir -p "${SEARCH_R1_ROOT}/data"
  run_cmd mkdir -p "${MODEL_ROOT}"
  run_cmd mkdir -p "${OUTPUT_ROOT}/runs"
  run_cmd mkdir -p "${OUTPUT_ROOT}/checkpoints"
  run_cmd mkdir -p "${LOG_ROOT}"
  run_cmd mkdir -p "${CACHE_ROOT}"
}

move_dir_to_target() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ ! -d "${src}" ]]; then
    log "[skip] ${label}: source not found: ${src}"
    return 0
  fi

  if [[ "${src}" == "${dst}" ]]; then
    log "[ok] ${label}: already in target: ${dst}"
    return 0
  fi

  log "[move] ${label}"
  log "       from: ${src}"
  log "       to:   ${dst}"

  if [[ "${APPLY}" -eq 1 ]]; then
    mkdir -p "$(dirname "${dst}")"
    if [[ ! -e "${dst}" ]]; then
      mv "${src}" "${dst}"
    else
      mkdir -p "${dst}"
      shopt -s dotglob nullglob
      local items=("${src}"/*)
      if [[ "${#items[@]}" -gt 0 ]]; then
        mv -n "${items[@]}" "${dst}/"
      fi
      shopt -u dotglob nullglob
    fi
  fi
}

write_inventory() {
  log ""
  log "## Current Layout"
  log "WORK_DIR=${WORK_DIR}"
  log "SEARCH_R1_ROOT=${SEARCH_R1_ROOT}"
  log "DATA_DIR=${DATA_DIR}"
  log "WIKI18_DIR=${WIKI18_DIR}"
  log "MODEL_ROOT=${MODEL_ROOT}"
  log "OUTPUT_ROOT=${OUTPUT_ROOT}"
  log "LOG_ROOT=${LOG_ROOT}"
  log "CACHE_ROOT=${CACHE_ROOT}"

  log ""
  log "## Top-Level Search-R1 Root"
  if [[ -d "${SEARCH_R1_ROOT}" ]]; then
    find "${SEARCH_R1_ROOT}" -maxdepth 2 -mindepth 1 -print | sort | tee -a "${MANIFEST_FILE}"
  else
    log "[missing] ${SEARCH_R1_ROOT}"
  fi

  log ""
  log "## Project Top-Level"
  if [[ -d "${WORK_DIR}" ]]; then
    find "${WORK_DIR}" -maxdepth 2 -mindepth 1 -print | sort | tee -a "${MANIFEST_FILE}"
  else
    log "[missing] ${WORK_DIR}"
  fi
}

write_expected_files() {
  log ""
  log "## Expected Files After Organization"
  for path in \
    "${DATA_DIR}/train.parquet" \
    "${DATA_DIR}/test.parquet" \
    "${WIKI18_DIR}/wiki-18.jsonl" \
    "${WIKI18_DIR}/e5_Flat.index" \
    "${LOCAL_BASE_MODEL}" \
    "${LOCAL_RETRIEVER_MODEL}" \
    "${OUTPUT_ROOT}/runs" \
    "${OUTPUT_ROOT}/checkpoints"; do
    if [[ -e "${path}" ]]; then
      log "[ok] ${path}"
    else
      log "[missing] ${path}"
    fi
  done
}

log "Search-R1 online development layout organizer"
log "mode=$([[ "${APPLY}" -eq 1 ]] && echo apply || echo dry-run)"
log "manifest=${MANIFEST_FILE}"

write_inventory

log ""
log "## Planned Organization"
ensure_layout

move_dir_to_target "${WORK_DIR}/data/${DATA_NAME}" "${DATA_DIR}" "NQ/HotpotQA parquet data"
move_dir_to_target "${WORK_DIR}/data/wiki-18" "${WIKI18_DIR}" "wiki-18 retrieval corpus/index"
move_dir_to_target "${WORK_DIR}/models/7b_base" "${MODEL_ROOT}" "7B/retriever model cache"
move_dir_to_target "${WORK_DIR}/verl_checkpoints" "${OUTPUT_ROOT}/checkpoints" "legacy veRL checkpoints"
move_dir_to_target "${WORK_DIR}/reproduction/7b_base/runs" "${OUTPUT_ROOT}/runs" "legacy profiled run outputs"
move_dir_to_target "${WORK_DIR}/outputs" "${OUTPUT_ROOT}/project_outputs" "legacy project outputs"
move_dir_to_target "${WORK_DIR}/logs" "${LOG_ROOT}/project_logs" "legacy project logs"
move_dir_to_target "${WORK_DIR}/wandb" "${LOG_ROOT}/wandb" "wandb logs"
move_dir_to_target "${WORK_DIR}/.cache" "${CACHE_ROOT}/project_cache" "project cache"

write_expected_files

log ""
if [[ "${APPLY}" -eq 1 ]]; then
  log "Organization applied."
else
  log "Dry run only. Re-run with --apply to move files:"
  log "  bash reproduction/7b_base/organize_online_dev_files.sh --apply"
fi
