#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATA_NAME}-search-r1-grpo-qwen2.5-7b-em}"
export TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-null}"
export VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-64}"
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-128}"
export N_AGENT="${N_AGENT:-5}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-['wandb']}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export TEST_FREQ="${TEST_FREQ:-100}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1005}"
export CKPT_DIR="${CKPT_DIR:-verl_checkpoints/${EXPERIMENT_NAME}}"
export MAX_TURNS="${MAX_TURNS:-4}"
export TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${EXPERIMENT_NAME}.log}"

cd "${WORK_DIR}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num="${TRAIN_DATA_NUM}" \
  data.val_data_num="${VAL_DATA_NUM}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent="${N_AGENT}" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="${TRAIN_LOGGER}" \
  +trainer.val_only=false \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
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
