#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

ALGO="${ALGO:-grpo}"
RUN_MODE="${RUN_MODE:-smoke}"
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/runs}"
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
  two_gpu_llama_instruct_time_budget)
    # Llama-2-7B-Chat feasibility profile using the main v0.2 experiment
    # parameters. Only the training GPU count (8 -> 2), backbone, and requested
    # 400-update budget differ from the paper configuration.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    # global_steps starts at 1; a limit of 401 executes steps 1 through 400.
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-401}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_llama_instruct_exact_gpu_paper_data)
    # Two-H20 paper-data-volume profile with exact E5 Flat retrieval sharded
    # across the same two GPUs used by FSDP/vLLM. The upstream run consumes
    # 512 prompts x 1,004 updates = 514,048 prompt samples. At batch 128 this
    # requires 4,016 updates; global_steps starts at 1, hence the 4,017 limit.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    # Keep the default within Llama-2-7B's native 4,096-token context:
    # MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH <= 4096.
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    # Leave room for the exact Flat index (roughly 20 GB/GPU when evenly
    # sharded) plus FSDP model state and transient actor-update allocations.
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export PAPER_PROMPT_SAMPLES="${PAPER_PROMPT_SAMPLES:-514048}"
    if [[ -z "${TOTAL_TRAINING_STEPS:-}" ]]; then
      if (( PAPER_PROMPT_SAMPLES % TRAIN_BATCH_SIZE != 0 )); then
        echo "[profile] PAPER_PROMPT_SAMPLES=${PAPER_PROMPT_SAMPLES} is not divisible by TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; rounding updates up." >&2
      fi
      PAPER_ACTUAL_UPDATES=$(((PAPER_PROMPT_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
      export TOTAL_TRAINING_STEPS=$((PAPER_ACTUAL_UPDATES + 1))
    else
      export TOTAL_TRAINING_STEPS
    fi
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    # Upstream saves every 100 * 512 = 51,200 prompt samples. At batch 128,
    # save every 400 updates to preserve the same sample-based cadence.
    export SAVE_FREQ="${SAVE_FREQ:-400}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export FINAL_VALIDATION="${FINAL_VALIDATION:-false}"
    export FINAL_SAVE="${FINAL_SAVE:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_llama_instruct_exact_gpu_time_budget)
    # Exact-GPU retrieval with a bounded sample budget. This keeps the paper's
    # key rollout semantics: GRPO, 5 agents, 4 search turns, search, KL, and
    # exact Flat retrieval. The time budget is controlled mainly by prompt
    # sample volume and train batch size, not by shrinking the agent behavior.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export PAPER_PROMPT_SAMPLES="${PAPER_PROMPT_SAMPLES:-514048}"
    if [[ -z "${TOTAL_TRAINING_STEPS:-}" ]]; then
      if (( PAPER_PROMPT_SAMPLES % TRAIN_BATCH_SIZE != 0 )); then
        echo "[profile] PAPER_PROMPT_SAMPLES=${PAPER_PROMPT_SAMPLES} is not divisible by TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; rounding updates up." >&2
      fi
      PAPER_ACTUAL_UPDATES=$(((PAPER_PROMPT_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
      export TOTAL_TRAINING_STEPS=$((PAPER_ACTUAL_UPDATES + 1))
    else
      export TOTAL_TRAINING_STEPS
    fi
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export SAVE_FREQ="${SAVE_FREQ:-400}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export FINAL_VALIDATION="${FINAL_VALIDATION:-false}"
    export FINAL_SAVE="${FINAL_SAVE:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_llama_instruct_exact_gpu_step10)
    # Exact-GPU retrieval profile targeting roughly 10-minute updates on two
    # H20 GPUs. This intentionally reduces rollout width/depth while retaining
    # GRPO, multi-agent comparison, online search, exact Flat retrieval, and KL.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-1792}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-3}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    # Keep the default prompt volume aligned with the 1/3 paper-volume run.
    # With TRAIN_BATCH_SIZE=32 this becomes 5,360 actual updates.
    export PAPER_PROMPT_SAMPLES="${PAPER_PROMPT_SAMPLES:-171520}"
    # By default, bound the materialized training dataset to the same prompt
    # budget used for the update count. Set TRAIN_DATA_NUM=null explicitly to
    # sample from the full pool.
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-${PAPER_PROMPT_SAMPLES}}"
    if [[ -z "${TOTAL_TRAINING_STEPS:-}" ]]; then
      if (( PAPER_PROMPT_SAMPLES % TRAIN_BATCH_SIZE != 0 )); then
        echo "[profile] PAPER_PROMPT_SAMPLES=${PAPER_PROMPT_SAMPLES} is not divisible by TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; rounding updates up." >&2
      fi
      PAPER_ACTUAL_UPDATES=$(((PAPER_PROMPT_SAMPLES + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE))
      export TOTAL_TRAINING_STEPS=$((PAPER_ACTUAL_UPDATES + 1))
    else
      export TOTAL_TRAINING_STEPS
    fi
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    # Preserve the upstream sample-based save cadence:
    # 51,200 prompt samples / batch 32 = 1,600 updates.
    export SAVE_FREQ="${SAVE_FREQ:-1600}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export FINAL_VALIDATION="${FINAL_VALIDATION:-false}"
    export FINAL_SAVE="${FINAL_SAVE:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export REWARD_SCORE_PRINT_PROB="${REWARD_SCORE_PRINT_PROB:-0}"
    ;;
  two_gpu_llama_time_budget)
    # Llama-2-7B adaptation for a two-H20 time budget. Reduce rollout width,
    # turns, and token limits because the base model often fails to terminate
    # Search-R1 actions, while retaining GRPO, search, KL, and full data pool.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-1792}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-2}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-4}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_qwen_main_v02)
    # Search-R1 main/scripts/nq_hotpotqa/v0.2/train_grpo.sh with only the
    # training GPU count changed from 8 to 2. All paper experiment parameters
    # below intentionally remain unchanged.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    ;;
  two_gpu_qwen_hnsw_full_epoch)
    # Qwen2.5-7B time-budget profile for two H20 GPUs. Keep the complete
    # training split available and preserve the paper's model, sequence
    # lengths, GRPO/search semantics and 1,005-step schedule. Smaller
    # train/optimizer batches make the workload practical enough to benchmark
    # on two GPUs; this does not constitute a complete pass over the split.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    # Qwen uses a 4,096-token prompt plus a 500-token response. Keep the
    # scheduler token budget above the resulting 4,596-token model length.
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4608}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_qwen_instruct_exact_gpu_step100)
    # Qwen2.5-7B-Instruct exact-GPU retrieval profile for a short 400-update
    # GRPO run on two H20 GPUs. Keep the upstream main/paper rollout semantics
    # aligned (GRPO, search, 5 agents, 4 turns, long context), but lower the
    # actor/log-prob micro batches for two shared GPUs. The paper micro batches
    # OOM during actor backward once exact FAISS and GPU E5 are resident.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-$((TRAIN_BATCH_SIZE * 400))}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    # Trainer exits when global_steps reaches this value. Since global_steps
    # starts at 1, 401 corresponds to 400 actual updates.
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-401}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export FINAL_VALIDATION="${FINAL_VALIDATION:-false}"
    export FINAL_SAVE="${FINAL_SAVE:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  three_gpu_qwen_instruct_exact_gpu_step200)
    # Qwen2.5-7B-Instruct exact-GPU retrieval profile for 3 shared GPUs.
    # Compared with the two-GPU profile, this raises the prompt batch to 64
    # while keeping the same 12,800 prompt budget via 200 optimizer updates.
    export CUDA_VISIBLE_DEVICES="${THREE_GPU_CUDA_VISIBLE_DEVICES:-0,1,2}"
    export N_GPUS_PER_NODE="${THREE_GPU_N_GPUS_PER_NODE:-3}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-$((TRAIN_BATCH_SIZE * 200))}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    # Trainer exits when global_steps reaches this value. Since global_steps
    # starts at 1, 201 corresponds to 200 actual updates.
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-201}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export FINAL_VALIDATION="${FINAL_VALIDATION:-false}"
    export FINAL_SAVE="${FINAL_SAVE:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_hnsw_full_epoch)
    # Resource-constrained full-data profile recommended by main/docs:
    # both H20 GPUs run FSDP + vLLM, while E5 and HNSW64 ANN retrieval run on
    # CPU. The trainer starts global_steps at 1, so a limit of 2,652 executes
    # 2,651 optimizer steps and covers the 169,615-example split once.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-32}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2652}"
    export SAVE_FREQ="${SAVE_FREQ:-250}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  one_gpu_exact_flat)
    # Strictly isolate the two physical GPUs:
    #   GPU 0: one-GPU actor/ref training and vLLM rollout
    #   GPU 1: E5 query encoder and exact Flat FAISS search
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
    export N_GPUS_PER_NODE="${ONE_GPU_N_GPUS_PER_NODE:-1}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-512}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-8}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-4}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-true}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
    export SAVE_FREQ="${SAVE_FREQ:-50}"
    export TEST_FREQ="${TEST_FREQ:-50}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_balanced)
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${TWO_GPU_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-2048}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-128}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-640}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-192}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-float16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2560}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
    export N_AGENT="${N_AGENT:-3}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${TWO_GPU_RETRIEVER_TOPK:-3}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
    export SAVE_FREQ="${SAVE_FREQ:-25}"
    export TEST_FREQ="${TEST_FREQ:-25}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-float16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_fast)
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${TWO_GPU_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-1024}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-512}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-128}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-float16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
    export N_AGENT="${N_AGENT:-2}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${TWO_GPU_RETRIEVER_TOPK:-3}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-120}"
    export SAVE_FREQ="${SAVE_FREQ:-20}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-float16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_paper)
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${TWO_GPU_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-2048}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-128}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-768}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-256}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-float16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${TWO_GPU_RETRIEVER_TOPK:-3}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
    export SAVE_FREQ="${SAVE_FREQ:-200}"
    export TEST_FREQ="${TEST_FREQ:-50}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-float16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_vllm_h20)
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${TWO_GPU_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-4096}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-256}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-384}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    # H20 has shown SIGFPEs in vLLM's float16 generation path. Prefer bf16 for
    # rollout/actor compute while keeping CLI/env overrides for compatibility.
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-true}"
    # max_model_len is MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = 2432 here.
    # Keep batched tokens above that floor while limiting H20 rollout concurrency.
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${TWO_GPU_RETRIEVER_TOPK:-3}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
    export SAVE_FREQ="${SAVE_FREQ:-200}"
    export TEST_FREQ="${TEST_FREQ:-50}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bfloat16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  two_gpu_vllm_h20_flash)
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${TWO_GPU_NNODES:-1}"
    # Use the complete 169,615-example training split. With batch size 32,
    # 5,301 steps cover approximately one full pass over the dataset.
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-384}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-8}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
    # H20 has shown SIGFPEs in vLLM's float16 generation path. Prefer bf16 for
    # rollout/actor compute while keeping CLI/env overrides for compatibility.
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
    export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-true}"
    # max_model_len is MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH = 2432 here.
    # Keep batched tokens above that floor while limiting H20 rollout concurrency.
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-3072}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-2}"
    export RETRIEVER_TOPK="${TWO_GPU_RETRIEVER_TOPK:-3}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5301}"
    export SAVE_FREQ="${SAVE_FREQ:-500}"
    export TEST_FREQ="${TEST_FREQ:-500}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-bfloat16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
  h20_smoke)
    export CUDA_VISIBLE_DEVICES="${H20_CUDA_VISIBLE_DEVICES:-0}"
    export N_GPUS_PER_NODE="${H20_N_GPUS_PER_NODE:-1}"
    export NNODES="${H20_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-16}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-8}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-256}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-96}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-1}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-1}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}"
    export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-float16}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-false}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
    export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
    export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    export N_AGENT="${N_AGENT:-1}"
    export MAX_TURNS="${MAX_TURNS:-1}"
    export RETRIEVER_TOPK="${H20_RETRIEVER_TOPK:-1}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}"
    export SAVE_FREQ="${SAVE_FREQ:-1}"
    export TEST_FREQ="${TEST_FREQ:--1}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-[]}"
    export USE_KL_LOSS="${USE_KL_LOSS:-false}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-true}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-float16}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
    export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
    export HF_USE_CACHE="${HF_USE_CACHE:-false}"
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
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
  full_stable_native_4k)
    # Full-data GRPO profile using Llama-2-7B's native 4,096-token context.
    # Preserve the paper's 500-token response budget and reduce only the
    # maximum prompt budget so prompt + response never exceeds 4,096 tokens.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-1024}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3596}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-null}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-null}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    ;;
  full_stable)
    # Full-data GRPO profile for two H20 GPUs. Keep the paper's training,
    # rollout, search, and optimizer semantics, but remove the blocking
    # full-validation pass before step 1 and bound periodic validation and
    # vLLM scheduler concurrency.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-1024}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-8192}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-2.0}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    ;;
  full)
    # Search-R1 v0.2 GRPO paper configuration with the backbone replaced by
    # Llama-2-7B and the GPU allocation reduced to two H20 GPUs. Validation
    # batching and vLLM scheduler concurrency are reduced to avoid sustained
    # KV-cache recomputation; training/reward semantics remain unchanged.
    export CUDA_VISIBLE_DEVICES="${TWO_GPU_CUDA_VISIBLE_DEVICES:-0,1}"
    export N_GPUS_PER_NODE="${TWO_GPU_N_GPUS_PER_NODE:-2}"
    export NNODES="${PAPER_NNODES:-1}"
    export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
    export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
    export MAX_START_LENGTH="${MAX_START_LENGTH:-2048}"
    export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
    export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
    export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
    export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
    export CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE:-8}"
    export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
    export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
    export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
    export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
    export N_AGENT="${N_AGENT:-5}"
    export MAX_TURNS="${MAX_TURNS:-4}"
    export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
    export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TEST_FREQ="${TEST_FREQ:-100}"
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
    # Preserve the paper's W&B logger without requiring an API key inside the
    # non-interactive training task. Offline files are stored with run outputs.
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    export USE_KL_LOSS="${USE_KL_LOSS:-true}"
    export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
    export DO_SEARCH="${DO_SEARCH:-true}"
    # Llama-2-7B has a native 4,096-token context, while the paper profile
    # requests 4,096 prompt tokens plus 500 response tokens. Extend Llama's
    # RoPE context without changing the paper's sequence-length parameters.
    export MODEL_MAX_POSITION_EMBEDDINGS="${MODEL_MAX_POSITION_EMBEDDINGS:-8192}"
    export MODEL_ROPE_SCALING_FACTOR="${MODEL_ROPE_SCALING_FACTOR:-2.0}"
    # vLLM separately reads the checkpoint's original 4,096-token config.json.
    # Explicitly allow the 4,596-token rollout after applying RoPE scaling.
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
    # The upstream script does not force a Transformers attention
    # implementation; null leaves model loading at its library default.
    export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-null}"
    export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
    ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}. Use RUN_MODE=two_gpu_llama_instruct_exact_gpu_paper_data, two_gpu_llama_instruct_exact_gpu_time_budget, two_gpu_llama_instruct_exact_gpu_step10, two_gpu_llama_instruct_time_budget, two_gpu_llama_time_budget, two_gpu_qwen_main_v02, two_gpu_qwen_hnsw_full_epoch, two_gpu_qwen_instruct_exact_gpu_step100, three_gpu_qwen_instruct_exact_gpu_step200, two_gpu_hnsw_full_epoch, one_gpu_exact_flat, two_gpu_balanced, two_gpu_fast, two_gpu_paper, two_gpu_vllm_h20, two_gpu_vllm_h20_flash, h20_smoke, smoke, full_stable_native_4k, full_stable, or full." >&2
    exit 2
    ;;
esac

echo "[profile] run_mode=${RUN_MODE} train_batch_size=${TRAIN_BATCH_SIZE} val_batch_size=${VAL_BATCH_SIZE}" >&2
echo "[profile] max_num_seqs=${MAX_NUM_SEQS:-} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-} rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-}" >&2
export TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-python3}"
echo "[profile] train_python=${TRAIN_PYTHON_BIN}" >&2

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATA_NAME}-search-r1-${ALGO}-${BASE_MODEL_NAME}-${RUN_MODE}-${RUN_ID}}"
RUN_DIR="${RUN_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
export TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${RUN_DIR}/train.log}"
export RAY_TMPDIR="${RAY_TMPDIR:-${OUTPUT_ROOT}/ray_tmp}"
export TMPDIR="${TMPDIR:-${OUTPUT_ROOT}/tmp}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export REPORT_REFRESH_INTERVAL="${REPORT_REFRESH_INTERVAL:-300}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

GPU_LOG="${RUN_DIR}/gpu_memory.csv"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
CKPT_LIST="${RUN_DIR}/checkpoints.txt"
ENV_FILE="${RUN_DIR}/env.txt"
REPORT_IMAGE="${RUN_DIR}/report.png"
METRICS_FILE="${RUN_DIR}/training_metrics.csv"

write_env_snapshot() {
  {
    echo "ALGO=${ALGO}"
    echo "RUN_MODE=${RUN_MODE}"
    echo "RUN_ID=${RUN_ID}"
    echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
    echo "WORK_DIR=${WORK_DIR}"
    echo "SEARCH_R1_ROOT=${SEARCH_R1_ROOT}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "WIKI18_DIR=${WIKI18_DIR}"
    echo "MODEL_ROOT=${MODEL_ROOT}"
    echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
    echo "LOG_ROOT=${LOG_ROOT}"
    echo "CACHE_ROOT=${CACHE_ROOT}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "RETRIEVER_URL=${RETRIEVER_URL}"
    echo "RETRIEVER_TOPK=${RETRIEVER_TOPK}"
    echo "RETRIEVER_DEVICE=${RETRIEVER_DEVICE:-}"
    echo "RETRIEVER_FAISS_GPU=${RETRIEVER_FAISS_GPU:-}"
    echo "RETRIEVER_FAISS_TEMP_MEMORY_MB=${RETRIEVER_FAISS_TEMP_MEMORY_MB:-}"
    echo "RETRIEVER_MAX_RETURN_TOKENS=${RETRIEVER_MAX_RETURN_TOKENS:-}"
    echo "RETRIEVER_CPU_THREADS=${RETRIEVER_CPU_THREADS:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-}"
    echo "RETRIEVER_CUDA_VISIBLE_DEVICES=${RETRIEVER_CUDA_VISIBLE_DEVICES:-}"
    echo "TRAIN_PYTHON_BIN=${TRAIN_PYTHON_BIN}"
    echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
    echo "NNODES=${NNODES}"
    echo "TRAIN_DATA_NUM=${TRAIN_DATA_NUM}"
    echo "VAL_DATA_NUM=${VAL_DATA_NUM}"
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
    echo "VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
    echo "MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-}"
    echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-}"
    echo "MAX_START_LENGTH=${MAX_START_LENGTH:-}"
    echo "MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-}"
    echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}"
    echo "PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE}"
    echo "LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE}"
    echo "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-}"
    echo "ROLLOUT_NAME=${ROLLOUT_NAME:-}"
    echo "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-}"
    echo "ROLLOUT_DTYPE=${ROLLOUT_DTYPE:-}"
    echo "ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-}"
    echo "ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-}"
    echo "ROLLOUT_REMOVE_INVALID_VALUES=${ROLLOUT_REMOVE_INVALID_VALUES:-}"
    echo "ROLLOUT_STOP_STRINGS=${ROLLOUT_STOP_STRINGS:-}"
    echo "ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-}"
    echo "ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-}"
    echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}"
    echo "MAX_NUM_SEQS=${MAX_NUM_SEQS:-}"
    echo "N_AGENT=${N_AGENT:-}"
    echo "MAX_TURNS=${MAX_TURNS:-}"
    echo "USE_KL_LOSS=${USE_KL_LOSS:-}"
    echo "DISABLE_REFERENCE_POLICY=${DISABLE_REFERENCE_POLICY:-}"
    echo "DO_SEARCH=${DO_SEARCH:-}"
    echo "ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-}"
    echo "MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-}"
    echo "MODEL_MAX_POSITION_EMBEDDINGS=${MODEL_MAX_POSITION_EMBEDDINGS:-}"
    echo "MODEL_ROPE_SCALING_FACTOR=${MODEL_ROPE_SCALING_FACTOR:-}"
    echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-}"
    echo "USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-}"
    echo "HF_SUMMON_FULL_PARAMS=${HF_SUMMON_FULL_PARAMS:-}"
    echo "HF_USE_CACHE=${HF_USE_CACHE:-}"
    echo "RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-}"
    echo "RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-}"
    echo "RAY_TMPDIR=${RAY_TMPDIR:-}"
    echo "TMPDIR=${TMPDIR:-}"
    echo "TEMP=${TEMP:-}"
    echo "TMP=${TMP:-}"
    echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}"
    echo "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
    echo "PAPER_PROMPT_SAMPLES=${PAPER_PROMPT_SAMPLES:-}"
    echo "SAVE_FREQ=${SAVE_FREQ}"
    echo "TEST_FREQ=${TEST_FREQ}"
    echo "VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}"
    echo "FINAL_VALIDATION=${FINAL_VALIDATION:-}"
    echo "FINAL_SAVE=${FINAL_SAVE:-}"
    echo "TRAIN_LOGGER=${TRAIN_LOGGER}"
    echo "WANDB_MODE=${WANDB_MODE:-}"
    echo "WANDB_DIR=${WANDB_DIR:-}"
    echo "CKPT_DIR=${CKPT_DIR}"
    echo "TRAIN_LOG_FILE=${TRAIN_LOG_FILE}"
    echo "GPU_SAMPLE_INTERVAL=${GPU_SAMPLE_INTERVAL}"
    echo "REPORT_REFRESH_INTERVAL=${REPORT_REFRESH_INTERVAL}"
    echo "METRICS_FILE=${METRICS_FILE}"
  } > "${ENV_FILE}"
}

monitor_gpu() {
  echo "wall_time,timestamp,gpu_index,gpu_name,memory_used_mb,memory_total_mb,gpu_util_percent" > "${GPU_LOG}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),nvidia-smi not found,,,,," >> "${GPU_LOG}"
    return 0
  fi
  if ! nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),nvidia-smi unavailable,,,,," >> "${GPU_LOG}"
    echo "[profile] nvidia-smi is unavailable; GPU telemetry disabled." >&2
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

refresh_report_once() {
  "${TRAIN_PYTHON_BIN}" "${SCRIPT_DIR}/extract_training_metrics.py" "${TRAIN_LOG_FILE}" "${METRICS_FILE}" >/dev/null 2>&1 || true
  "${TRAIN_PYTHON_BIN}" "${SCRIPT_DIR}/render_training_report.py" "${RUN_DIR}" --output "${REPORT_IMAGE}" >/dev/null 2>&1 || true
}

refresh_report_loop() {
  if [[ "${REPORT_REFRESH_INTERVAL}" -le 0 ]]; then
    return 0
  fi
  while true; do
    refresh_report_once
    sleep "${REPORT_REFRESH_INTERVAL}"
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
    echo "training_metrics=${METRICS_FILE}"
    echo "checkpoint_dir=${CKPT_DIR}"
    echo "checkpoint_list=${CKPT_LIST}"
  } > "${SUMMARY_FILE}"
}

check_required_inputs() {
  local missing=0
  if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "[missing] train parquet: ${DATA_DIR}/train.parquet" | tee -a "${TRAIN_LOG_FILE}" >&2
    missing=1
  fi
  if [[ ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "[missing] test parquet: ${DATA_DIR}/test.parquet" | tee -a "${TRAIN_LOG_FILE}" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    {
      echo
      echo "Data files are missing. Prepare the dataset before training:"
      echo "  cd ${WORK_DIR}"
      echo "  bash reproduction/7b_base/prepare_data.sh"
      echo
      echo "If your data is stored outside the repository, rerun with:"
      echo "  DATA_DIR=/path/to/nq_hotpotqa_train ALGO=${ALGO} RUN_MODE=${RUN_MODE} bash reproduction/7b_base/run_profiled_train.sh"
    } | tee -a "${TRAIN_LOG_FILE}" >&2
    return 1
  fi
  return 0
}

write_env_snapshot
if ! check_required_inputs; then
  echo "wall_time,timestamp,gpu_index,gpu_name,memory_used_mb,memory_total_mb,gpu_util_percent" > "${GPU_LOG}"
  START_EPOCH="$(date +%s)"
  END_EPOCH="${START_EPOCH}"
  summarize_run 2 "${START_EPOCH}" "${END_EPOCH}"
  if "${TRAIN_PYTHON_BIN}" "${SCRIPT_DIR}/render_training_report.py" "${RUN_DIR}" --output "${REPORT_IMAGE}" >/dev/null 2>&1; then
    echo "Training report: ${REPORT_IMAGE}"
  fi
  echo "Run summary: ${SUMMARY_FILE}"
  echo "GPU memory log: ${GPU_LOG}"
  echo "Checkpoint list: ${CKPT_LIST}"
  exit 2
fi

START_EPOCH="$(date +%s)"
monitor_gpu &
MONITOR_PID="$!"
refresh_report_loop &
REPORT_PID="$!"
trap 'kill "${MONITOR_PID}" "${REPORT_PID}" >/dev/null 2>&1 || true' EXIT

set +e
bash "${TRAIN_SCRIPT}"
EXIT_CODE="$?"
set -e

END_EPOCH="$(date +%s)"
kill "${MONITOR_PID}" "${REPORT_PID}" >/dev/null 2>&1 || true
wait "${MONITOR_PID}" 2>/dev/null || true
wait "${REPORT_PID}" 2>/dev/null || true
trap - EXIT

summarize_run "${EXIT_CODE}" "${START_EPOCH}" "${END_EPOCH}"

if ! "${TRAIN_PYTHON_BIN}" "${SCRIPT_DIR}/extract_training_metrics.py" "${TRAIN_LOG_FILE}" "${METRICS_FILE}" >/dev/null 2>&1; then
  echo "Training metric extraction failed; report will contain GPU telemetry only." >&2
fi

if "${TRAIN_PYTHON_BIN}" "${SCRIPT_DIR}/render_training_report.py" "${RUN_DIR}" --output "${REPORT_IMAGE}" >/dev/null 2>&1; then
  echo "Training report: ${REPORT_IMAGE}"
else
  echo "Training report generation failed. Install matplotlib in the active environment to enable report.png." >&2
fi

echo "Run summary: ${SUMMARY_FILE}"
echo "GPU memory log: ${GPU_LOG}"
echo "Training metrics: ${METRICS_FILE}"
echo "Checkpoint list: ${CKPT_LIST}"
exit "${EXIT_CODE}"
