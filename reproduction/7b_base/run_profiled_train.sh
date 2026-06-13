#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

ALGO="${ALGO:-grpo}"
RUN_MODE="${RUN_MODE:-smoke}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/reproduction/7b_base/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

case "${ALGO}" in
  grpo)
    TRAIN_SCRIPT="${SCRIPT_DIR}/train_grpo_7b_base.sh"
    ;;
  ppo)
    TRAIN_SCRIPT="${SCRIPT_DIR}/train_ppo_7b_base.sh"
    ;;
  *)
    echo "Unsupported ALGO=${ALGO}. Use ALGO=grpo or ALGO=ppo." >&2
    exit 2
    ;;
esac

case "${RUN_MODE}" in
  smoke)
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-32}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-16}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-2}"
    export N_AGENT="${N_AGENT:-1}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
    export SAVE_FREQ="${SAVE_FREQ:-1}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    ;;
  full)
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}. Use RUN_MODE=smoke or RUN_MODE=full." >&2
    exit 2
    ;;
esac

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATA_NAME}-search-r1-${ALGO}-qwen2.5-7b-${RUN_MODE}-${RUN_ID}}"
RUN_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"

export CKPT_DIR="${CKPT_DIR:-${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}}"
export TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${RUN_DIR}/train.log}"

GPU_LOG="${RUN_DIR}/gpu_memory.csv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
CKPT_LIST="${RUN_DIR}/checkpoints.txt"
ENV_FILE="${RUN_DIR}/env.txt"
REPORT_IMAGE="${RUN_DIR}/report.png"

write_env_snapshot() {
  {
    echo "ALGO=${ALGO}"
    echo "RUN_MODE=${RUN_MODE}"
    echo "RUN_ID=${RUN_ID}"
    echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
    echo "WORK_DIR=${WORK_DIR}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "RETRIEVER_URL=${RETRIEVER_URL}"
    echo "RETRIEVER_TOPK=${RETRIEVER_TOPK}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
    echo "NNODES=${NNODES}"
    echo "TRAIN_DATA_NUM=${TRAIN_DATA_NUM}"
    echo "VAL_DATA_NUM=${VAL_DATA_NUM}"
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    echo "VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
    echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}"
    echo "PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE}"
    echo "LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE}"
    echo "N_AGENT=${N_AGENT:-}"
    echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
    echo "SAVE_FREQ=${SAVE_FREQ}"
    echo "TEST_FREQ=${TEST_FREQ}"
    echo "VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
    echo "TRAIN_LOGGER=${TRAIN_LOGGER}"
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
    echo "GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL}"
  } > "${ENV_FILE}"
}

monitor_gpu() {
  echo "wall_time,timestamp,gpu_index,gpu_name,memory_used_mb,memory_total_mb,gpu_util_percent" > "${GPU_LOG}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),nvidia-smi not found,,,,," >> "${GPU_LOG}"
    return 0
  fi

  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -v wall_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -F', ' '{print wall_time "," $0}' >> "${GPU_LOG}" || true
    sleep "${GPU_SAMPLE_INTERVAL}"
  done
}

format_epoch() {
  local epoch="$1"
  date -u -d "@${epoch}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "${epoch}" +%Y-%m-%dT%H:%M:%SZ
}

summarize_run() {
  local exit_code="$1"
  local start_epoch="$2"
  local end_epoch="$3"
  local duration_sec=$((end_epoch - start_epoch))
  local peak_mem="n/a"

  if [[ -s "${GPU_LOG}" ]]; then
    peak_mem="$(awk -F',' 'NR > 1 && $5 ~ /^[0-9]+$/ { if ($5 > max) max=$5 } END { if (max == "") print "n/a"; else print max " MB" }' "${GPU_LOG}")"
  fi

  if [[ -d "${CKPT_DIR}" ]]; then
    find "${CKPT_DIR}" -maxdepth 4 -type d -name 'global_step_*' | sort > "${CKPT_LIST}"
  else
    : > "${CKPT_LIST}"
  fi

  {
    echo "experiment_name=${EXPERIMENT_NAME}"
    echo "algo=${ALGO}"
    echo "run_mode=${RUN_MODE}"
    echo "exit_code=${exit_code}"
    echo "start_time=$(format_epoch "${start_epoch}")"
    echo "end_time=$(format_epoch "${end_epoch}")"
    echo "duration_seconds=${duration_sec}"
    echo "peak_gpu_memory=${peak_mem}"
    echo "train_log=${TRAIN_LOG_FILE}"
    echo "gpu_memory_log=${GPU_LOG}"
    echo "checkpoint_dir=${CKPT_DIR}"
    echo "checkpoint_list=${CKPT_LIST}"
  } > "${SUMMARY_FILE}"
}

write_env_snapshot
START_EPOCH="$(date +%s)"
monitor_gpu &
MONITOR_PID="$!"
trap 'kill "${MONITOR_PID}" >/dev/null 2>&1 || true' EXIT

set +e
bash "${TRAIN_SCRIPT}"
EXIT_CODE="$?"
set -e

END_EPOCH="$(date +%s)"
kill "${MONITOR_PID}" >/dev/null 2>&1 || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

summarize_run "${EXIT_CODE}" "${START_EPOCH}" "${END_EPOCH}"

if python3 "${SCRIPT_DIR}/render_training_report.py" "${RUN_DIR}" --output "${REPORT_IMAGE}" >/dev/null 2>&1; then
  echo "Training report: ${REPORT_IMAGE}"
else
  echo "Training report generation failed. Install matplotlib in the active environment to enable report.png." >&2
fi

echo "Run summary: ${SUMMARY_FILE}"
echo "GPU memory log: ${GPU_LOG}"
echo "Checkpoint list: ${CKPT_LIST}"
exit "${EXIT_CODE}"
