#!/usr/bin/env bash
set -euo pipefail

# Shared configuration for reproducing a 7B-base Search-R1 run.
# The upstream v0.2 scripts used Qwen/Qwen2.5-7B; this branch also supports
# setting BASE_MODEL_NAME=llama-7b for a local LLaMA-family 7B checkpoint.

export WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Keep code, environments, data, models, and outputs separated on platforms that
# mount a persistent filesdir root such as /workspace/filesdir/code/search-r1.
if [[ -z "${SEARCH_R1_ROOT:-}" ]]; then
  if [[ "$(basename "${WORK_DIR}")" == "Search-R1" && "$(basename "$(dirname "${WORK_DIR}")")" == "projects" ]]; then
    export SEARCH_R1_ROOT="$(dirname "$(dirname "${WORK_DIR}")")"
  else
    export SEARCH_R1_ROOT="${WORK_DIR}"
  fi
else
  export SEARCH_R1_ROOT
fi

export DATA_NAME="${DATA_NAME:-nq_hotpotqa_train}"
export DATA_DIR="${DATA_DIR:-${SEARCH_R1_ROOT}/data/${DATA_NAME}}"
export WIKI18_DIR="${WIKI18_DIR:-${SEARCH_R1_ROOT}/data/wiki-18}"
export MODEL_ROOT="${MODEL_ROOT:-${SEARCH_R1_ROOT}/models/7b_base}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SEARCH_R1_ROOT}/outputs}"
export LOG_ROOT="${LOG_ROOT:-${SEARCH_R1_ROOT}/logs}"
export CACHE_ROOT="${CACHE_ROOT:-${SEARCH_R1_ROOT}/cache}"
export BASE_MODEL_NAME="${BASE_MODEL_NAME:-llama-7b}"
case "${BASE_MODEL_NAME}" in
  qwen2.5-7b)
    DEFAULT_BASE_MODEL_ID="Qwen/Qwen2.5-7B"
    ;;
  llama-7b)
    DEFAULT_BASE_MODEL_ID="${HF_BASE_MODEL_ID:-meta-llama/Llama-2-7b-hf}"
    ;;
  *)
    DEFAULT_BASE_MODEL_ID="${HF_BASE_MODEL_ID:-${BASE_MODEL_NAME}}"
    ;;
esac

export HF_BASE_MODEL_ID="${HF_BASE_MODEL_ID:-${DEFAULT_BASE_MODEL_ID}}"
export HF_RETRIEVER_MODEL_ID="${HF_RETRIEVER_MODEL_ID:-intfloat/e5-base-v2}"
export LOCAL_BASE_MODEL="${LOCAL_BASE_MODEL:-${MODEL_ROOT}/${BASE_MODEL_NAME}}"
export LOCAL_RETRIEVER_MODEL="${LOCAL_RETRIEVER_MODEL:-${MODEL_ROOT}/e5-base-v2}"

if [[ -z "${BASE_MODEL:-}" && -d "${LOCAL_BASE_MODEL}" ]]; then
  export BASE_MODEL="${LOCAL_BASE_MODEL}"
else
  export BASE_MODEL="${BASE_MODEL:-${HF_BASE_MODEL_ID}}"
fi

if [[ -z "${RETRIEVER_MODEL:-}" && -d "${LOCAL_RETRIEVER_MODEL}" ]]; then
  export RETRIEVER_MODEL="${LOCAL_RETRIEVER_MODEL}"
else
  export RETRIEVER_MODEL="${RETRIEVER_MODEL:-${HF_RETRIEVER_MODEL_ID}}"
fi

export WAND_PROJECT="${WAND_PROJECT:-Search-R1}"

export RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export NNODES="${NNODES:-1}"

# vLLM + Qwen2.5-7B is configured this way in the upstream scripts.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
