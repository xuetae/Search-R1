#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

MODE="${MODE:-search}"
TOPK="${TOPK:-1}"
VAL_DATA_NUM="${VAL_DATA_NUM:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.85}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
SEARCH_DATA_DIR="${SEARCH_DATA_DIR:-data/nq_search}"
DIRECT_DATA_DIR="${DIRECT_DATA_DIR:-data/nq}"

if [[ "$MODE" == "direct" ]]; then
    DATA_DIR="$DIRECT_DATA_DIR"
    DO_SEARCH=false
    MAX_TURNS=1
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-direct-baseline}"
elif [[ "$MODE" == "search" ]]; then
    DATA_DIR="$SEARCH_DATA_DIR"
    DO_SEARCH=true
    MAX_TURNS="${MAX_TURNS:-1}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-bm25-top${TOPK}-baseline}"
else
    echo "Unsupported MODE=$MODE. Use MODE=direct or MODE=search." >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Missing $DATA_DIR/test.parquet" >&2
    if [[ "$MODE" == "direct" ]]; then
        echo "Prepare it with: python scripts/data_process/nq.py --local_dir $DIRECT_DATA_DIR" >&2
    else
        echo "Prepare it with: python scripts/data_process/nq_search.py --local_dir $SEARCH_DATA_DIR" >&2
    fi
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=32 \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$VAL_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_start_length="$MAX_START_LENGTH" \
    data.max_obs_length="$MAX_OBS_LENGTH" \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="$VAL_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TOP_P" \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=Search-R1-baseline \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    do_search="$DO_SEARCH" \
    max_turns="$MAX_TURNS" \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk="$TOPK" \
    2>&1 | tee "eval-${EXPERIMENT_NAME}.log"
