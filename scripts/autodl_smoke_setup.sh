#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
PROJECTS_DIR="$DATA_ROOT/projects"
REPO_URL="${SEARCH_R1_REPO_URL:-https://github.com/xuetae/Search-R1.git}"
REPO_BRANCH="${SEARCH_R1_REPO_BRANCH:-codex/smoke-small-run}"
REPO_DIR="$PROJECTS_DIR/Search-R1"
CONDA_ENV_DIR="$DATA_ROOT/conda_envs"
SEARCHR1_ENV="$CONDA_ENV_DIR/searchr1"
RETRIEVER_ENV="$CONDA_ENV_DIR/retriever"
BM25_DIR="$DATA_ROOT/wiki18_bm25"

mkdir -p "$PROJECTS_DIR" \
         "$DATA_ROOT/cache/huggingface" \
         "$DATA_ROOT/cache/torch" \
         "$DATA_ROOT/cache/pip" \
         "$CONDA_ENV_DIR" \
         "$BM25_DIR"

if ! grep -q "SEARCH_R1_AUTODL_CACHE" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# SEARCH_R1_AUTODL_CACHE
export DATA_ROOT=/root/autodl-tmp
export HF_HOME=$DATA_ROOT/cache/huggingface
export HF_HUB_CACHE=$DATA_ROOT/cache/huggingface/hub
export TRANSFORMERS_CACHE=$DATA_ROOT/cache/huggingface/transformers
export TORCH_HOME=$DATA_ROOT/cache/torch
export XDG_CACHE_HOME=$DATA_ROOT/cache
export PIP_CACHE_DIR=$DATA_ROOT/cache/pip
EOF
fi

export HF_HOME="$DATA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$DATA_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export XDG_CACHE_HOME="$DATA_ROOT/cache"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
  git -C "$REPO_DIR" switch "$REPO_BRANCH"
  git -C "$REPO_DIR" pull --ff-only
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$SEARCHR1_ENV" ]; then
  conda create -p "$SEARCHR1_ENV" python=3.10 -y
fi
conda activate "$SEARCHR1_ENV"
python -m pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e "$REPO_DIR"
pip install wandb
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import vllm; print('vllm ok')"

if [ ! -d "$RETRIEVER_ENV" ]; then
  conda create -p "$RETRIEVER_ENV" python=3.10 -y
fi
conda activate "$RETRIEVER_ENV"
conda install -c conda-forge faiss-cpu openjdk=21 -y
pip install transformers datasets pyserini uvicorn fastapi pydantic
java -version
python -c "import faiss, pyserini; print('retriever deps ok')"

conda activate "$SEARCHR1_ENV"
cd "$REPO_DIR"
mkdir -p data/nq_search
python scripts/data_process/nq_search.py --local_dir ./data/nq_search
ls -lh data/nq_search

cd "$BM25_DIR"
huggingface-cli download PeterJinGo/wiki-18-bm25-index \
  --repo-type dataset \
  --local-dir "$BM25_DIR"
huggingface-cli download PeterJinGo/wiki-18-corpus \
  --repo-type dataset \
  --local-dir "$BM25_DIR"
if [ -f "$BM25_DIR/wiki-18.jsonl.gz" ] && [ ! -f "$BM25_DIR/wiki-18.jsonl" ]; then
  gzip -d "$BM25_DIR/wiki-18.jsonl.gz"
fi
ls -lh "$BM25_DIR"

cat <<EOF

AutoDL smoke setup complete.

Start the retriever in one terminal:
  source ~/.bashrc
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$RETRIEVER_ENV"
  cd "$REPO_DIR"
  export SEARCH_R1_BM25_DIR="$BM25_DIR"
  bash retrieval_launch_bm25_cpu_smoke.sh

Verify the retriever in another terminal:
  curl -X POST http://127.0.0.1:8000/retrieve \\
    -H "Content-Type: application/json" \\
    -d '{"queries":["who is the president of france"],"topk":1,"return_scores":true}'

Start the smoke training:
  source ~/.bashrc
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$SEARCHR1_ENV"
  cd "$REPO_DIR"
  bash train_grpo_smoke_1gpu.sh
EOF
