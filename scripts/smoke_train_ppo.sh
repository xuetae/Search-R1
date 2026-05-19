#!/usr/bin/env bash
# 中文注释：启用严格模式，遇到错误、未定义变量或管道失败时立即退出。
set -euo pipefail

# 中文注释：允许调用者覆盖 GPU 列表，默认使用 8 张 GPU 以匹配原始训练脚本。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# 中文注释：允许调用者覆盖训练/验证数据目录，默认使用 NQ search 数据。
export DATA_DIR="${DATA_DIR:-data/nq_search}"
# 中文注释：允许调用者覆盖基础模型；若 Llama 无权限，可传入 Qwen/Qwen2.5-3B。
export BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B}"
# 中文注释：设置 smoke test 实验名，避免覆盖正式训练日志。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-smoke-search-r1-ppo}"
# 中文注释：设置 wandb project；默认与主训练脚本保持一致。
export WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
# 中文注释：vLLM 在部分 Qwen/flash attention 组合上需要 XFORMERS 后端。
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# 中文注释：N_GPUS 需要和 CUDA_VISIBLE_DEVICES 中可用 GPU 数一致。
N_GPUS="${N_GPUS:-8}"
# 中文注释：smoke test 默认只跑 2 个训练 step，用于检查端到端链路。
TOTAL_STEPS="${TOTAL_STEPS:-2}"
# 中文注释：默认关闭训练过程中的保存，避免 smoke test 产生大 checkpoint。
SAVE_FREQ="${SAVE_FREQ:--1}"
# 中文注释：默认关闭周期性验证，缩短 smoke test 时间。
TEST_FREQ="${TEST_FREQ:--1}"
# 中文注释：默认执行训练前验证；若只想最快检查训练链路，可设为 false。
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
# 中文注释：默认检索地址与配置文件保持一致。
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
# 中文注释：默认每次搜索返回 3 篇文档。
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

# 中文注释：以下 batch/长度参数比正式训练更小，便于快速暴露依赖、Ray、vLLM、检索连接问题。
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
CRITIC_MICRO_BATCH_SIZE="${CRITIC_MICRO_BATCH_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-256}"
MAX_TURNS="${MAX_TURNS:-2}"

# 中文注释：启动最小 PPO 搜索训练链路；所有覆盖参数都通过 Hydra 传入，不修改 train_ppo.sh。
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_start_length="$MAX_START_LENGTH" \
    data.max_obs_length="$MAX_OBS_LENGTH" \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.0 \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size="$CRITIC_MICRO_BATCH_SIZE" \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns="$MAX_TURNS" \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk="$RETRIEVER_TOPK" \
    2>&1 | tee "$EXPERIMENT_NAME.log"
