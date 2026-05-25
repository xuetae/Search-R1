#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export SEARCH_R1_SHORT_INVALID_FEEDBACK="${SEARCH_R1_SHORT_INVALID_FEEDBACK:-1}"
export SEARCH_R1_MAX_DOC_CHARS="${SEARCH_R1_MAX_DOC_CHARS:-900}"
export SEARCH_R1_COMPRESS_EVIDENCE="${SEARCH_R1_COMPRESS_EVIDENCE:-1}"
export SEARCH_R1_EVIDENCE_SENTENCES_PER_DOC="${SEARCH_R1_EVIDENCE_SENTENCES_PER_DOC:-2}"
export SEARCH_R1_EVIDENCE_MAX_CHARS_PER_DOC="${SEARCH_R1_EVIDENCE_MAX_CHARS_PER_DOC:-360}"

DATA_DIR="${DATA_DIR:-data/nq_search}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen2.5-1.5b-bm25-shaped}"

TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-2048}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"
MAX_START_LENGTH="${MAX_START_LENGTH:-1024}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-384}"
TOPK="${TOPK:-1}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
STRUCTURE_FORMAT_SCORE="${STRUCTURE_FORMAT_SCORE:-0.2}"
FINAL_FORMAT_SCORE="${FINAL_FORMAT_SCORE:-0.1}"
RETRIEVAL_SCORE="${RETRIEVAL_SCORE:-0.2}"
ANSWER_GROUNDING_SCORE="${ANSWER_GROUNDING_SCORE:-0.1}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"
LLDS_ENABLE="${LLDS_ENABLE:-false}"
LLDS_COEF="${LLDS_COEF:-0.02}"
LLDS_REDUCE_THRES="${LLDS_REDUCE_THRES:-0.0}"
LLDS_ADV_GATE="${LLDS_ADV_GATE:-positive}"
LLDS_CHUNK="${LLDS_CHUNK:-false}"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Missing $DATA_DIR/train.parquet or $DATA_DIR/test.parquet" >&2
    echo "Prepare them with: python scripts/data_process/nq_search.py --local_dir $DATA_DIR" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_start_length="$MAX_START_LENGTH" \
    data.max_obs_length="$MAX_OBS_LENGTH" \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
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
    actor_rollout_ref.actor.llds_enable="$LLDS_ENABLE" \
    actor_rollout_ref.actor.llds_coef="$LLDS_COEF" \
    actor_rollout_ref.actor.llds_reduce_thres="$LLDS_REDUCE_THRES" \
    actor_rollout_ref.actor.llds_adv_gate="$LLDS_ADV_GATE" \
    actor_rollout_ref.actor.llds_chunk="$LLDS_CHUNK" \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TOP_P" \
    actor_rollout_ref.actor.state_masking=true \
    reward_model.structure_format_score="$STRUCTURE_FORMAT_SCORE" \
    reward_model.final_format_score="$FINAL_FORMAT_SCORE" \
    reward_model.retrieval_score="$RETRIEVAL_SCORE" \
    +reward_model.answer_grounding_score="$ANSWER_GROUNDING_SCORE" \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.project_name=Search-R1-lite \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=1 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk="$TOPK" \
    2>&1 | tee "$EXPERIMENT_NAME.log"
