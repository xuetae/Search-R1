#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

APPLY=0
FAILED_ONLY=1
EXPERIMENT_FILTER=""

usage() {
  cat <<'EOF'
Usage:
  bash reproduction/7b_base/cleanup_failed_runs.sh [--dry-run|--apply] [--all|--failed-only] [--experiment NAME]

Deletes incomplete run artifacts under:
  ${OUTPUT_ROOT}/runs
  ${OUTPUT_ROOT}/checkpoints

Default is --dry-run --failed-only.

Examples:
  bash reproduction/7b_base/cleanup_failed_runs.sh
  bash reproduction/7b_base/cleanup_failed_runs.sh --apply
  bash reproduction/7b_base/cleanup_failed_runs.sh --experiment nq_hotpotqa_train-search-r1-grpo-llama-7b-h20_smoke-20260614_074816 --apply
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      APPLY=0
      shift
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --failed-only)
      FAILED_ONLY=1
      shift
      ;;
    --all)
      FAILED_ONLY=0
      shift
      ;;
    --experiment)
      EXPERIMENT_FILTER="${2:-}"
      if [[ -z "${EXPERIMENT_FILTER}" ]]; then
        echo "--experiment requires a value" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
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

RUNS_DIR="${OUTPUT_ROOT}/runs"
CKPTS_DIR="${OUTPUT_ROOT}/checkpoints"

is_under_output_root() {
  local target="$1"
  case "${target}" in
    "${OUTPUT_ROOT}"/*) return 0 ;;
    *) return 1 ;;
  esac
}

delete_path() {
  local target="$1"
  if [[ -z "${target}" || ! -e "${target}" ]]; then
    return 0
  fi
  if ! is_under_output_root "${target}"; then
    echo "[skip] refuses to delete outside OUTPUT_ROOT: ${target}" >&2
    return 1
  fi

  if [[ "${APPLY}" -eq 1 ]]; then
    echo "[delete] ${target}"
    rm -rf -- "${target}"
  else
    echo "[planned] ${target}"
  fi
}

run_is_failed() {
  local run_dir="$1"
  local summary="${run_dir}/summary.txt"

  if [[ "${FAILED_ONLY}" -eq 0 ]]; then
    return 0
  fi

  if [[ ! -f "${summary}" ]]; then
    return 0
  fi

  local exit_code
  exit_code="$(awk -F= '$1 == "exit_code" {print $2; exit}' "${summary}")"
  if [[ -z "${exit_code}" || "${exit_code}" != "0" ]]; then
    return 0
  fi

  return 1
}

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "CKPTS_DIR=${CKPTS_DIR}"
if [[ "${APPLY}" -eq 1 ]]; then
  echo "mode=apply"
else
  echo "mode=dry-run"
fi

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "[ok] no runs directory: ${RUNS_DIR}"
  exit 0
fi

FOUND=0
while IFS= read -r run_dir; do
  experiment_name="$(basename "${run_dir}")"
  if [[ -n "${EXPERIMENT_FILTER}" && "${experiment_name}" != "${EXPERIMENT_FILTER}" ]]; then
    continue
  fi
  if ! run_is_failed "${run_dir}"; then
    continue
  fi

  FOUND=1
  echo "== ${experiment_name} =="
  delete_path "${run_dir}"

  summary_ckpt=""
  if [[ -f "${run_dir}/summary.txt" ]]; then
    summary_ckpt="$(awk -F= '$1 == "checkpoint_dir" {print $2; exit}' "${run_dir}/summary.txt")"
  fi
  if [[ -n "${summary_ckpt}" ]]; then
    delete_path "${summary_ckpt}"
  fi
  delete_path "${CKPTS_DIR}/${experiment_name}"
done < <(find "${RUNS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ "${FOUND}" -eq 0 ]]; then
  echo "[ok] no matching run artifacts found"
fi

if [[ "${APPLY}" -eq 0 ]]; then
  echo
  echo "Dry run only. Re-run with --apply to delete the listed paths."
fi
