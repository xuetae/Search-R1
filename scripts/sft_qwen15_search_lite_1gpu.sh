#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="${DATA_DIR:-data/nq_search_lite_sft}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-search-lite-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-verl_checkpoints/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1536}"
LR="${LR:-1e-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LORA_RANK="${LORA_RANK:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CPU_OFFLOAD="${CPU_OFFLOAD:-True}"
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-True}"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
    echo "Missing $DATA_DIR/train.parquet or $DATA_DIR/test.parquet" >&2
    echo "Prepare them with scripts/data_process/nq_search_lite_sft.py" >&2
    exit 1
fi

PYTHONUNBUFFERED=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size="$MICRO_BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=right \
    model.partial_pretrain="$BASE_MODEL" \
    model.enable_gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
    model.fsdp_config.cpu_offload="$CPU_OFFLOAD" \
    model.fsdp_config.offload_params="$OFFLOAD_PARAMS" \
    model.lora_rank="$LORA_RANK" \
    optim.lr="$LR" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name=Search-R1-lite \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.validate_before_training=False \
    trainer.logger=['console'] \
    2>&1 | tee "$EXPERIMENT_NAME.log"
