# Online Development Layout

Use this layout in the platform JupyterLab Terminal. Keep code, virtual
environments, data, models, logs, and outputs separated under the persistent
`filesdir` path.

## Directory Layout

```text
/workspace/filesdir/code/search-r1/
  projects/
    Search-R1/              # git clone, branch llama-7b-run
  venvs/
    searchr1/               # Python virtual environment
  data/
    nq_hotpotqa_train/      # train.parquet, test.parquet
    wiki-18/                # wiki-18.jsonl, e5_Flat.index
  models/
    7b_base/
      llama-7b/
      e5-base-v2/
  outputs/
    runs/                   # summary.txt, gpu_memory.csv, report.png
    checkpoints/            # saved actor/critic checkpoints
  logs/
  cache/
```

The reproduction scripts infer this root automatically when the repository is
cloned to:

```text
/workspace/filesdir/code/search-r1/projects/Search-R1
```

That means these defaults are used:

```text
WORK_DIR=/workspace/filesdir/code/search-r1/projects/Search-R1
SEARCH_R1_ROOT=/workspace/filesdir/code/search-r1
DATA_DIR=/workspace/filesdir/code/search-r1/data/nq_hotpotqa_train
WIKI18_DIR=/workspace/filesdir/code/search-r1/data/wiki-18
MODEL_ROOT=/workspace/filesdir/code/search-r1/models/7b_base
OUTPUT_ROOT=/workspace/filesdir/code/search-r1/outputs
```

## First-Time Setup

Run commands in JupyterLab Terminal, not in a notebook cell.

```bash
mkdir -p /workspace/filesdir/code/search-r1/projects
mkdir -p /workspace/filesdir/code/search-r1/venvs
mkdir -p /workspace/filesdir/code/search-r1/data
mkdir -p /workspace/filesdir/code/search-r1/models
mkdir -p /workspace/filesdir/code/search-r1/outputs
mkdir -p /workspace/filesdir/code/search-r1/logs
mkdir -p /workspace/filesdir/code/search-r1/cache
```

Create and activate the Python environment:

```bash
python3 -m venv /workspace/filesdir/code/search-r1/venvs/searchr1
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
python -m pip install --no-cache-dir --upgrade pip setuptools wheel
```

Clone the project:

```bash
cd /workspace/filesdir/code/search-r1/projects
git clone -b llama-7b-run --depth=1 https://github.com/xuetae/Search-R1.git
cd /workspace/filesdir/code/search-r1/projects/Search-R1
git branch
```

Install dependencies:

```bash
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
cd /workspace/filesdir/code/search-r1/projects/Search-R1
python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir "transformers<4.48" datasets pyserini uvicorn fastapi huggingface_hub faiss-cpu wandb IPython matplotlib
python -m pip install --no-cache-dir "vllm==0.6.3"
python -m pip install --no-cache-dir flash-attn --no-build-isolation || true
```

## Base Model

This branch defaults to:

```text
BASE_MODEL_NAME=llama-7b
LOCAL_BASE_MODEL=/workspace/filesdir/code/search-r1/models/7b_base/llama-7b
```

If you already uploaded LLaMA-7B weights, put the Hugging Face format model
files under:

```text
/workspace/filesdir/code/search-r1/models/7b_base/llama-7b/
```

If your model is stored somewhere else, pass it explicitly:

```bash
LOCAL_BASE_MODEL=/path/to/llama-7b \
ALGO=grpo \
RUN_MODE=h20_smoke \
bash reproduction/7b_base/run_profiled_train.sh
```

If you want `download_models.sh` to download from Hugging Face, set the exact
authorized repo id:

```bash
HF_BASE_MODEL_ID=meta-llama/Llama-2-7b-hf bash reproduction/7b_base/download_models.sh
```

If Hugging Face access is blocked, download the LLaMA-7B base model from
ModelScope instead:

```bash
python -m pip install --no-cache-dir modelscope
BASE_MODEL_SOURCE=modelscope \
MS_BASE_MODEL_ID=modelscope/Llama-2-7b-ms \
bash reproduction/7b_base/download_models.sh
```

This stores the base model at:

```text
/workspace/filesdir/code/search-r1/models/7b_base/llama-7b/
```

## Prepare Data

```bash
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
cd /workspace/filesdir/code/search-r1/projects/Search-R1
bash reproduction/7b_base/prepare_data.sh
```

## Organize Existing Files

If files were already downloaded under the old project-local layout, inspect and
move them into the persistent layout.

First inspect only:

```bash
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
cd /workspace/filesdir/code/search-r1/projects/Search-R1
bash reproduction/7b_base/organize_online_dev_files.sh --dry-run
```

Then apply the move:

```bash
bash reproduction/7b_base/organize_online_dev_files.sh --apply
```

The script writes a manifest to:

```text
/workspace/filesdir/code/search-r1/outputs/layout_manifests/
```

Expected files:

```text
/workspace/filesdir/code/search-r1/data/nq_hotpotqa_train/train.parquet
/workspace/filesdir/code/search-r1/data/nq_hotpotqa_train/test.parquet
/workspace/filesdir/code/search-r1/data/wiki-18/wiki-18.jsonl
/workspace/filesdir/code/search-r1/data/wiki-18/e5_Flat.index
```

## Run H20 Smoke Training

Start the retriever first in one terminal:

```bash
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
cd /workspace/filesdir/code/search-r1/projects/Search-R1
bash reproduction/7b_base/launch_retriever.sh
```

The launcher defaults to CPU FAISS because many venv installs provide
`faiss-cpu`. If your environment has a GPU-enabled FAISS build, run:

```bash
RETRIEVER_FAISS_GPU=1 bash reproduction/7b_base/launch_retriever.sh
```

Run smoke training in another terminal:

```bash
. /workspace/filesdir/code/search-r1/venvs/searchr1/bin/activate
cd /workspace/filesdir/code/search-r1/projects/Search-R1
ALGO=grpo RUN_MODE=h20_smoke bash reproduction/7b_base/run_profiled_train.sh
```

`h20_smoke` is a single-GPU connectivity run. It uses 1 training sample per
batch, shorter generation limits, a smaller vLLM cache budget, and disables the
reference-policy worker by default:

```text
USE_KL_LOSS=false
DISABLE_REFERENCE_POLICY=true
ROLLOUT_GPU_MEMORY_UTILIZATION=0.25
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=4
```

This mode is intended to confirm the retriever, rollout, training step,
checkpoint saving, GPU memory logging, and `report.png` generation. It is not
the paper-faithful final run. Use `RUN_MODE=full` on a multi-GPU allocation for
the full reproduction settings.

Results are written to:

```text
/workspace/filesdir/code/search-r1/outputs/runs/<experiment_name>/
/workspace/filesdir/code/search-r1/outputs/checkpoints/<experiment_name>/
```

Open `report.png` in the run directory for the visual training summary.

## Clean Failed Runs

If a smoke run fails before completing, clean only the failed run artifacts.
This does not touch datasets, wiki index files, or model directories.

Inspect first:

```bash
cd /workspace/filesdir/code/search-r1/projects/Search-R1
bash reproduction/7b_base/cleanup_failed_runs.sh
```

Delete the listed failed runs and matching checkpoints:

```bash
bash reproduction/7b_base/cleanup_failed_runs.sh --apply
```

Delete one known failed experiment:

```bash
bash reproduction/7b_base/cleanup_failed_runs.sh \
  --experiment nq_hotpotqa_train-search-r1-grpo-llama-7b-h20_smoke-20260614_074816 \
  --apply
```
