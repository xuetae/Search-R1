#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATA_NAME}-search-r1-grpo-${BASE_MODEL_NAME}-em}"
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
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
export ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-true}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export ROLLOUT_REMOVE_INVALID_VALUES="${ROLLOUT_REMOVE_INVALID_VALUES:-true}"
export ROLLOUT_STOP_STRINGS="${ROLLOUT_STOP_STRINGS:-[\"</search>\",\"</answer>\"]}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
export N_AGENT="${N_AGENT:-5}"
export USE_KL_LOSS="${USE_KL_LOSS:-true}"
export DISABLE_REFERENCE_POLICY="${DISABLE_REFERENCE_POLICY:-false}"
export DO_SEARCH="${DO_SEARCH:-true}"
export ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-null}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-flash_attention_2}"
export USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
export HF_SUMMON_FULL_PARAMS="${HF_SUMMON_FULL_PARAMS:-true}"
export HF_USE_CACHE="${HF_USE_CACHE:-true}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export FINAL_VALIDATION="${FINAL_VALIDATION:-true}"
export FINAL_SAVE="${FINAL_SAVE:-false}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-100}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
export CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
export MAX_TURNS="${MAX_TURNS:-4}"
export TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${EXPERIMENT_NAME}.log}"

cd "${WORK_DIR}"

MODEL_OVERRIDE_ARGS=()
if [[ -n "${MODEL_MAX_POSITION_EMBEDDINGS:-}" && "${MODEL_MAX_POSITION_EMBEDDINGS}" != "null" ]]; then
  MODEL_OVERRIDE_ARGS+=(
    "+actor_rollout_ref.model.override_config.max_position_embeddings=${MODEL_MAX_POSITION_EMBEDDINGS}"
  )
fi
if [[ -n "${MODEL_ROPE_SCALING_FACTOR:-}" && "${MODEL_ROPE_SCALING_FACTOR}" != "null" ]]; then
  MODEL_OVERRIDE_ARGS+=(
    "+actor_rollout_ref.model.override_config.rope_scaling={type:linear,factor:${MODEL_ROPE_SCALING_FACTOR}}"
  )
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num="${TRAIN_DATA_NUM}" \
  data.val_data_num="${VAL_DATA_NUM}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  +actor_rollout_ref.model.attn_implementation="${MODEL_ATTN_IMPLEMENTATION}" \
  "${MODEL_OVERRIDE_ARGS[@]}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  +actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}" \
  actor_rollout_ref.rollout.do_sample="${ROLLOUT_DO_SAMPLE}" \
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
  +actor_rollout_ref.rollout.disable_custom_all_reduce="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}" \
  +actor_rollout_ref.rollout.remove_invalid_values="${ROLLOUT_REMOVE_INVALID_VALUES}" \
  +actor_rollout_ref.rollout.stop_strings="${ROLLOUT_STOP_STRINGS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  +actor_rollout_ref.rollout.hf_summon_full_params="${HF_SUMMON_FULL_PARAMS}" \
  +actor_rollout_ref.rollout.hf_use_cache="${HF_USE_CACHE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  do_search="${DO_SEARCH}" \
  actor_rollout_ref.rollout.n_agent="${N_AGENT}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="${TRAIN_LOGGER}" \
  +trainer.val_only=false \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.final_validation="${FINAL_VALIDATION}" \
  trainer.final_save="${FINAL_SAVE}" \
  +trainer.disable_reference_policy="${DISABLE_REFERENCE_POLICY}" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${CKPT_DIR}" \
  max_turns="${MAX_TURNS}" \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk="${RETRIEVER_TOPK}" \
  2>&1 | tee "${TRAIN_LOG_FILE}"
