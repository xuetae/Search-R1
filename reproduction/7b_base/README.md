# 7B Base Search-R1 Reproduction Setup

This directory contains the self-contained launcher set for testing the paper-style 7B base result.

For platform online development with persistent storage under
`/workspace/filesdir/code/search-r1`, follow `ONLINE_DEV_LAYOUT.md`.

## What To Prepare

- Python/Search-R1 environment: `searchr1`, created by `setup_envs.sh`.
- Retriever environment: `retriever`, created by `setup_envs.sh`.
- GPUs: the upstream 7B scripts assume 1 node with 8 GPUs.
- Dataset: `PeterJinGo/nq_hotpotqa_train`, downloaded to `data/nq_hotpotqa_train`.
- Retrieval corpus/index: `wiki-18.jsonl` plus `e5_Flat.index`, downloaded to `data/wiki-18`.
- Base model: default local directory is `models/7b_base/llama-7b`; set `LOCAL_BASE_MODEL` if your LLaMA-7B weights are elsewhere.
- Retriever model: default is `intfloat/e5-base-v2`, downloaded to `models/7b_base/e5-base-v2`.

This branch is configured for a LLaMA-family 7B base model by default. The upstream paper scripts used Qwen2.5-7B, so keep the exact base model path recorded in each run's `env.txt`.

## One-Command Server Bootstrap

After cloning this branch on the server, run:

```bash
bash reproduction/7b_base/bootstrap_server.sh
```

This creates/updates the `searchr1` and `retriever` conda environments, downloads the 7B base model, downloads the e5 retriever model, prepares the NQ/HotpotQA data and wiki-18 retrieval files, and runs a preflight check.

If the server needs a Hugging Face token, log in first:

```bash
huggingface-cli login
```

## Local Upload Bundle With Docker Images

If the server cannot download dependencies directly, prepare an upload bundle locally:

```bash
bash reproduction/7b_base/docker/build_images.sh
conda activate searchr1
bash reproduction/7b_base/download_models.sh
bash reproduction/7b_base/prepare_data.sh
bash reproduction/7b_base/docker/make_upload_bundle.sh
```

For a Linux x86_64 GPU server, keep the default `IMAGE_PLATFORM=linux/amd64`. On Apple Silicon/macOS Docker Desktop, this cross-platform CUDA build can be very slow and may require more Docker memory/disk than the default allocation. If possible, build the upload bundle on a Linux x86_64 machine with a stable network.

This creates:

```text
offline_upload.tar.gz
```

Upload `offline_upload.tar.gz` to the server, put it at the repository root, then run:

```bash
tar -xzf offline_upload.tar.gz
bash offline_upload/restore_on_server.sh
```

The upload bundle contains Docker image archives plus optional local data/model archives:

- `searchr1-7b:cuda121`
- `searchr1-retriever:cuda121`
- `data/nq_hotpotqa_train`
- `data/wiki-18`
- `models/7b_base`

If the archive is too large for your upload channel, split it locally:

```bash
SPLIT_SIZE=10G bash reproduction/7b_base/docker/make_upload_bundle.sh
```

Then upload all `offline_upload.tar.gz.part-*` files and reassemble on the server:

```bash
cat offline_upload.tar.gz.part-* > offline_upload.tar.gz
tar -xzf offline_upload.tar.gz
bash offline_upload/restore_on_server.sh
```

After restore, run with Docker:

```bash
bash reproduction/7b_base/docker/run_retriever_container.sh
bash reproduction/7b_base/docker/run_grpo_container.sh
```

## Step-By-Step Setup

Create environments:

```bash
bash reproduction/7b_base/setup_envs.sh
```

Download local model copies:

```bash
conda activate searchr1
bash reproduction/7b_base/download_models.sh
```

## Prepare Data

```bash
conda activate searchr1
bash reproduction/7b_base/prepare_data.sh
```

Optional overrides:

```bash
WORK_DIR=/path/to/Search-R1 \
DATA_DIR=/path/to/data/nq_hotpotqa_train \
WIKI18_DIR=/path/to/data/wiki-18 \
bash reproduction/7b_base/prepare_data.sh
```

## Launch Retriever

Run this in the retriever environment:

```bash
conda activate retriever
bash reproduction/7b_base/launch_retriever.sh
```

The server listens on `http://127.0.0.1:8000/retrieve` by default. Override with `RETRIEVER_URL` in the training/evaluation shell if needed.

## Train 7B Base

Recommended workflow: run a small smoke training first, inspect runtime, GPU memory, and checkpoints, then run the full dataset.

Single-H20 smoke run:

```bash
conda activate searchr1
ALGO=grpo RUN_MODE=h20_smoke bash reproduction/7b_base/run_profiled_train.sh
```

`h20_smoke` is the safest first run for a 1-GPU online development instance. It sets:

- `CUDA_VISIBLE_DEVICES=0`
- `N_GPUS_PER_NODE=1`
- `TRAIN_DATA_NUM=8`
- `VAL_DATA_NUM=4`
- `TRAIN_BATCH_SIZE=1`
- `PPO_MICRO_BATCH_SIZE=1`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=128`
- `MAX_TURNS=2`
- `DO_SEARCH=true`
- `ROLLOUT_NAME=hf`
- `ROLLOUT_DTYPE=float16`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.25`
- `RAY_memory_usage_threshold=0.99`
- `TOTAL_TRAINING_STEPS=2`
- `SAVE_FREQ=1`

Keep the local retriever running for this mode. The 61GB FAISS index and HF 7B
rollout/training are close to the limit on a 128GB online-dev node, so this mode
raises Ray's memory kill threshold to 0.99. Use this smoke mode to validate
online retrieval, training, checkpointing, GPU memory logs, and report
generation.

If the runner reports missing parquet files, prepare the dataset first:

```bash
bash reproduction/7b_base/prepare_data.sh
```

If the dataset is stored in a persistent directory outside the repository, pass it explicitly:

```bash
DATA_DIR=/workspace/filesdir/code/search-r1/data/nq_hotpotqa_train \
ALGO=grpo \
RUN_MODE=h20_smoke \
bash reproduction/7b_base/run_profiled_train.sh
```

Smoke run with local profiling:

```bash
conda activate searchr1
ALGO=grpo RUN_MODE=smoke bash reproduction/7b_base/run_profiled_train.sh
```

The smoke defaults are intentionally small:

- `TRAIN_DATA_NUM=32`
- `VAL_DATA_NUM=16`
- `TRAIN_BATCH_SIZE=32`
- `TOTAL_TRAINING_STEPS=2`
- `SAVE_FREQ=1`
- `TEST_FREQ=-1`
- `TRAIN_LOGGER=[]`

Each profiled run writes:

```text
outputs/runs/<experiment_name>/summary.txt
outputs/runs/<experiment_name>/gpu_memory.csv
outputs/runs/<experiment_name>/train.log
outputs/runs/<experiment_name>/checkpoints.txt
outputs/runs/<experiment_name>/report.png
```

Check these fields before starting full training:

- `report.png`: visual summary with status, wall-clock time, peak GPU memory, checkpoint count, memory curve, and GPU utilization curve.
- `summary.txt`: exit code, wall-clock duration, peak GPU memory, ckpt directory.
- `gpu_memory.csv`: sampled `nvidia-smi` memory and utilization.
- `checkpoints.txt`: saved `global_step_*` checkpoint directories.
- `train.log`: full stdout/stderr from veRL.

Full profiled run:

```bash
conda activate searchr1
ALGO=grpo RUN_MODE=full bash reproduction/7b_base/run_profiled_train.sh
```

PPO is also supported:

```bash
ALGO=ppo RUN_MODE=smoke bash reproduction/7b_base/run_profiled_train.sh
ALGO=ppo RUN_MODE=full bash reproduction/7b_base/run_profiled_train.sh
```

Useful profiling overrides:

```bash
GPU_SAMPLE_INTERVAL=5 \
EXPERIMENT_NAME=nq_hotpotqa-grpo-7b-smoke-test \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
N_GPUS_PER_NODE=8 \
ALGO=grpo \
RUN_MODE=smoke \
bash reproduction/7b_base/run_profiled_train.sh
```

GRPO, matching the upstream v0.2 7B-base GRPO settings:

```bash
conda activate searchr1
bash reproduction/7b_base/train_grpo_7b_base.sh
```

PPO, matching the upstream v0.2 7B-base PPO settings:

```bash
conda activate searchr1
bash reproduction/7b_base/train_ppo_7b_base.sh
```

Useful overrides:

```bash
LOCAL_BASE_MODEL=/workspace/filesdir/code/search-r1/models/7b_base/llama-7b \
EXPERIMENT_NAME=nq_hotpotqa_train-search-r1-grpo-llama-7b-em \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash reproduction/7b_base/train_grpo_7b_base.sh
```

## Evaluate

Evaluate the base model:

```bash
conda activate searchr1
bash reproduction/7b_base/evaluate_7b_base.sh
```

Evaluate a trained checkpoint:

```bash
EVAL_MODEL=/workspace/filesdir/code/search-r1/outputs/checkpoints/nq_hotpotqa_train-search-r1-grpo-llama-7b-em/actor/global_step_1000 \
bash reproduction/7b_base/evaluate_7b_base.sh
```

## Notes

- Data, wiki corpus, FAISS index, model weights, checkpoints, logs, and wandb output are intentionally not committed.
- `data/`, `*.log`, checkpoints, and wandb directories are already covered by `.gitignore`.
- `models/7b_base/` is ignored so local downloaded model weights are not committed.
- The retriever must be running before training or evaluation, because rollouts call the `/retrieve` API whenever the model emits `<search>...</search>`.
