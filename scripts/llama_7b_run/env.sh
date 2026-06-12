#!/usr/bin/env bash
set -euo pipefail

# Shared configuration for reproducing the paper's 7B-base Search-R1 run.
# The official v0.2 scripts in this repository use Qwen/Qwen2.5-7B as the 7B base model.

export WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export DATA_NAME="${DATA_NAME:-nq_hotpotqa_train}"
export DATA_DIR="${DATA_DIR:-${WORK_DIR}/data/${DATA_NAME}}"
export WIKI18_DIR="${WIKI18_DIR:-${WORK_DIR}/data/wiki-18}"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B}"
export WAND_PROJECT="${WAND_PROJECT:-Search-R1}"

export RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"

# vLLM + Qwen2.5-7B is configured this way in the upstream scripts.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
