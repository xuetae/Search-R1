#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

EVAL_MODEL="${EVAL_MODEL:-${BASE_MODEL}}"
EVAL_DATA_NUM="${EVAL_DATA_NUM:-null}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1792}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-16}"
ROLLOUT_STOP_STRINGS="${ROLLOUT_STOP_STRINGS:-[\"</search>\",\"</answer>\"]}"

cd "${WORK_DIR}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num="${EVAL_DATA_NUM}" \
  data.train_batch_size=512 \
  data.val_batch_size="${EVAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path="${EVAL_MODEL}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size=64 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  +actor_rollout_ref.rollout.stop_strings="${ROLLOUT_STOP_STRINGS}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  critic.optim.lr=1e-5 \
  critic.model.use_remove_padding=True \
  critic.optim.lr_warmup_steps_ratio=0.05 \
  critic.model.path="${EVAL_MODEL}" \
  critic.model.enable_gradient_checkpointing=true \
  critic.ppo_micro_batch_size=8 \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.no_think_rl=false \
  trainer.critic_warmup=0 \
  trainer.logger="[]" \
  +trainer.val_only=true \
  +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  max_turns="${MAX_TURNS}" \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk="${RETRIEVER_TOPK}"
