#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

if [[ -d /workspace/model_out ]]; then
  export OUTPUT_ROOT="${PLATFORM_OUTPUT_ROOT:-/workspace/model_out/search-r1/outputs}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-llama7b-grpo-exact-gpu-paper-data}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PAPER_PROMPT_SAMPLES="${PAPER_PROMPT_SAMPLES:-514048}"
ACTUAL_UPDATES=$(((PAPER_PROMPT_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$((ACTUAL_UPDATES + 1))}"
FINAL_UPDATE=$((TOTAL_TRAINING_STEPS - 1))
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/workspace/filesdir/miniforge3/envs/retriever-gpu/bin/python}"
EVAL_LOG="${EVAL_LOG:-${OUTPUT_ROOT}/eval-${EXPERIMENT_NAME}-full.log}"

cd "${WORK_DIR}"
mkdir -p "${OUTPUT_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
TRAIN_DATA_NUM=null \
VAL_DATA_NUM=512 \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
VAL_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}" \
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}" \
N_AGENT=5 \
MAX_TURNS=4 \
MAX_START_LENGTH=2048 \
MAX_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=500 \
MAX_OBS_LENGTH=500 \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
RETRIEVER_TOPK=3 \
PAPER_PROMPT_SAMPLES="${PAPER_PROMPT_SAMPLES}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
VAL_BEFORE_TRAIN=false \
FINAL_VALIDATION=false \
FINAL_SAVE=true \
TEST_FREQ=-1 \
SAVE_FREQ="${SAVE_FREQ:-400}" \
PYTHON_BIN="${RETRIEVER_PYTHON}" \
python3 "${SCRIPT_DIR}/training_task_entry.py" \
  --algo grpo \
  --run-mode two_gpu_llama_instruct_exact_gpu_paper_data \
  --cuda-visible-devices 0,1 \
  --train-cuda-visible-devices 0,1 \
  --retriever-cuda-visible-devices 0,1

EVAL_MODEL="${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}/actor/global_step_${FINAL_UPDATE}"
if [[ ! -d "${EVAL_MODEL}" ]]; then
  echo "Final checkpoint does not exist: ${EVAL_MODEL}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1 \
RETRIEVER_DEVICE=cuda \
RETRIEVER_FAISS_GPU=1 \
EXPECTED_RETRIEVER_INDEX=flat \
RETRIEVER_TOPK=3 \
RETRIEVER_MAX_RETURN_TOKENS=0 \
INDEX_FILE="${WIKI18_DIR}/e5_Flat.index" \
CORPUS_FILE="${WIKI18_DIR}/wiki-18.jsonl" \
PYTHON_BIN="${RETRIEVER_PYTHON}" \
bash "${SCRIPT_DIR}/launch_retriever.sh" \
  > "${OUTPUT_ROOT}/retriever-${EXPERIMENT_NAME}-eval.log" 2>&1 &

RETRIEVER_PID=$!
cleanup() {
  kill "${RETRIEVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 360); do
  if curl -fsS http://127.0.0.1:8000/docs >/dev/null; then
    break
  fi
  if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    echo "Exact-GPU retriever failed; inspect ${OUTPUT_ROOT}/retriever-${EXPERIMENT_NAME}-eval.log" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8000/docs >/dev/null

CUDA_VISIBLE_DEVICES=0,1 \
EVAL_MODEL="${EVAL_MODEL}" \
EVAL_DATA_NUM=null \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
MAX_TURNS=4 \
EVAL_MAX_TURNS=4 \
MAX_START_LENGTH=2048 \
MAX_PROMPT_LENGTH=4096 \
MAX_RESPONSE_LENGTH=500 \
MAX_OBS_LENGTH=500 \
MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-32}" \
MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS:-4096}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.5}" \
LOG_PROB_MICRO_BATCH_SIZE="${EVAL_LOG_PROB_MICRO_BATCH_SIZE:-16}" \
RETRIEVER_TOPK=3 \
N_GPUS_PER_NODE=2 \
NNODES=1 \
bash "${SCRIPT_DIR}/evaluate_7b_base.sh" \
  2>&1 | tee "${EVAL_LOG}"
