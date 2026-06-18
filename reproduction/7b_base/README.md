# 7B Base Search-R1 Reproduction Setup

This directory contains the self-contained launcher set for testing the paper-style 7B base result.

For platform online development with persistent storage under
`/filesdir/code/search-r1`, follow `ONLINE_DEV_LAYOUT.md`.

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

## xFusion Platform Rebuild

After deleting an xFusion algorithm entry, rebuild the project under the
file-management mount instead of `/workspace`:

```text
/filesdir/code/search-r1
```

Code and data should be downloaded on the platform. Upload only the local model
archives:

```text
/filesdir/llama-7b.tar.gz
/filesdir/e5-base-v2.tar.gz
```

Create the clean directory layout and clone this branch on the platform:

```bash
mkdir -p /filesdir/code/search-r1/projects
mkdir -p /filesdir/code/search-r1/uploads
mv /filesdir/llama-7b.tar.gz /filesdir/code/search-r1/uploads/ 2>/dev/null || true
mv /filesdir/e5-base-v2.tar.gz /filesdir/code/search-r1/uploads/ 2>/dev/null || true

cd /filesdir/code/search-r1/projects
git clone -b llama-7b-run https://github.com/xuetae/Search-R1.git
cd /filesdir/code/search-r1/projects/Search-R1
```

Then unpack uploaded models and download platform-side data:

```bash
bash reproduction/7b_base/rebuild_platform_project.sh \
  --skip-code \
  --llama-archive /filesdir/code/search-r1/uploads/llama-7b.tar.gz \
  --e5-archive /filesdir/code/search-r1/uploads/e5-base-v2.tar.gz \
  --download-data
```

The script creates this final layout:

```text
/filesdir/code/search-r1/projects/Search-R1
/filesdir/code/search-r1/data/nq_hotpotqa_train/train.parquet
/filesdir/code/search-r1/data/nq_hotpotqa_train/test.parquet
/filesdir/code/search-r1/data/wiki-18/wiki-18.jsonl
/filesdir/code/search-r1/data/wiki-18/e5_Flat.index
/filesdir/code/search-r1/models/7b_base/llama-7b
/filesdir/code/search-r1/models/7b_base/e5-base-v2
/filesdir/code/search-r1/outputs
/filesdir/code/search-r1/logs
/filesdir/code/search-r1/cache
```

If models are already unpacked, skip model extraction:

```bash
bash reproduction/7b_base/rebuild_platform_project.sh --skip-code --skip-models --download-data
```

If you only want to create directories and unpack models before downloading data
later, omit `--download-data`:

```bash
bash reproduction/7b_base/rebuild_platform_project.sh --skip-code
```

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

For single-H20 method validation on a 128GB online-dev node, use the lite
retriever instead of the full 61GB FAISS index:

```bash
RETRIEVER_TOPK=1 LITE_RETRIEVER_MAX_DOCS=20000 bash reproduction/7b_base/launch_lite_retriever.sh
```

The server listens on `http://127.0.0.1:8000/retrieve` by default. Override with `RETRIEVER_URL` in the training/evaluation shell if needed.

## Train 7B Base

Recommended workflow: run a small smoke training first, inspect runtime, GPU memory, and checkpoints, then run the full dataset.

Two-GPU paper-style pilot:

```bash
conda activate searchr1
ALGO=grpo RUN_MODE=two_gpu_paper bash reproduction/7b_base/run_profiled_train.sh
```

On xFusion training tasks, the run command field only accepts a Python command.
Use this entrypoint so the task starts the retriever and then launches training:

```bash
python reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

When the task is created from File Management, xFusion mounts the selected
algorithm path under `/workspace/algorithm`. Use the platform path in the run
command:

```bash
python3 /workspace/algorithm/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

If the selected file-management path is `Search_R1/projects` instead of
`Search_R1/projects/Search-R1`, use:

```bash
python3 /workspace/algorithm/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

The entrypoint reads data and models from `/workspace/filesdir` and writes run
outputs, checkpoints, logs, and reports under `/workspace/model_out/search-r1`
when the platform training-task output directory exists.
It also accepts xFusion injected arguments such as `--data_url`, `--train_out`,
and `--train_log`; `--train_out` and `--train_log` are used for persistent task
outputs when provided by the platform.
Some xFusion deployments mount the persistent output directory as
`/workspace/model-out`; the entrypoint detects that path too.

If the algorithm mount contains an unexpected extra directory level, run the
entrypoint directly from the file-management mount:

```bash
python3 /workspace/filesdir/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

For the validated 2x H20 setup, `two_gpu_paper`, `two_gpu_balanced`, and
`two_gpu_fast` use the HF rollout backend with `float16` and `sdpa`, because
older vLLM/FlashAttention wheels can raise `SIGFPE` on this platform. Use
`two_gpu_paper` when the goal is to stay as close as possible to the paper
while still fitting 2x H20.

If the image has been rebuilt with `torch==2.4.0+cu121` and `vllm==0.6.3`
or the local `vllm` package reports `dev` while
`verl.third_party.vllm.vllm_version == 0.6.3`, first run the standalone LLaMA
generation test. If it passes, use the H20 vLLM mode:

```bash
python3 /workspace/filesdir/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_vllm_h20
```

Recommended xFusion form values:

- Training task type: single-node training.
- Image: the Search-R1 image/version validated in online development.
- Run command: `python3 /workspace/filesdir/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper`.
- Compute type: GPU / full card.
- GPU specification: one node with 2 H20 GPUs.
- Max failed restarts: 0 or 1 while validating the pilot.
- Scheduler queue/node group: select the H20 queue/node group.
- File-management mount: mount the `Search_R1` file-management directory at
  `/workspace/filesdir`; the project should be under
  `/workspace/filesdir/projects/Search-R1`.

`two_gpu_paper` is the recommended GRPO run on 2x H20 when you need the closest
practical match to the paper setup:

- `CUDA_VISIBLE_DEVICES=0,1`
- `N_GPUS_PER_NODE=2`
- `TRAIN_DATA_NUM=2048`
- `VAL_DATA_NUM=128`
- `TRAIN_BATCH_SIZE=4`
- `VAL_BATCH_SIZE=4`
- `PPO_MINI_BATCH_SIZE=4`
- `PPO_MICRO_BATCH_SIZE=1`
- `LOG_PROB_MICRO_BATCH_SIZE=8`
- `MAX_PROMPT_LENGTH=1536`
- `MAX_RESPONSE_LENGTH=256`
- `MAX_START_LENGTH=768`
- `MAX_OBS_LENGTH=256`
- `ROLLOUT_NAME=hf`
- `ROLLOUT_DTYPE=float16`
- `ROLLOUT_TEMPERATURE=1`
- `MODEL_ATTN_IMPLEMENTATION=sdpa`
- `USE_REMOVE_PADDING=false`
- `HF_USE_CACHE=false`
- `TENSOR_MODEL_PARALLEL_SIZE=1`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.35`
- `MAX_NUM_BATCHED_TOKENS=3072`
- `MAX_NUM_SEQS=16`
- `DO_SEARCH=true`
- `RETRIEVER_TOPK=3`
- `USE_KL_LOSS=true`
- `DISABLE_REFERENCE_POLICY=false`
- `N_AGENT=5`
- `MAX_TURNS=2`
- `TOTAL_TRAINING_STEPS=1000`
- `SAVE_FREQ=200`
- `TEST_FREQ=50`
- `VAL_BEFORE_TRAIN=true`

This mode keeps the paper-critical parts: GRPO, retrieval, `topk=3`,
multi-agent rollout with `n_agent=5`, two search turns, KL loss, state masking,
periodic validation, and a longer training horizon. The main differences from
the upstream 8-GPU scripts are the unavoidable resource adaptations:
`TRAIN_BATCH_SIZE=4` instead of 512, HF rollout instead of vLLM, shorter
sequence lengths, and nearly the same step count: 1000 steps versus the
upstream 1005 steps. Checkpoint saving is relaxed to every 200 steps to reduce
I/O pressure under `/workspace/model_out`.

`two_gpu_vllm_h20` is the recommended vLLM run for the rebuilt H20 image with
the vLLM 0.6.3-compatible veRL wrapper. It keeps the same Search-R1 method
settings as `two_gpu_paper`, but uses a larger 2-GPU vLLM configuration:

- `TRAIN_DATA_NUM=4096`
- `VAL_DATA_NUM=256`
- `TRAIN_BATCH_SIZE=32`
- `VAL_BATCH_SIZE=16`
- `PPO_MINI_BATCH_SIZE=32`
- `PPO_MICRO_BATCH_SIZE=1`
- `LOG_PROB_MICRO_BATCH_SIZE=8`
- `MAX_PROMPT_LENGTH=2048`
- `MAX_RESPONSE_LENGTH=384`
- `MAX_START_LENGTH=1024`
- `MAX_OBS_LENGTH=384`
- `ROLLOUT_NAME=vllm`
- `ROLLOUT_DTYPE=float16`
- `ROLLOUT_ENFORCE_EAGER=true`
- `ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=true`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`
- `MODEL_ATTN_IMPLEMENTATION=sdpa`
- `USE_REMOVE_PADDING=false`
- `HF_USE_CACHE=false`
- `MAX_NUM_BATCHED_TOKENS=4096`
- `MAX_NUM_SEQS=64`
- `N_AGENT=5`
- `MAX_TURNS=2`
- `RETRIEVER_TOPK=3`
- `TRAIN_DATA_NUM=null` (all 169,615 training examples)
- `VAL_DATA_NUM=null` (all 51,713 validation examples)
- `TOTAL_TRAINING_STEPS=5301`
- `SAVE_FREQ=500`
- `TEST_FREQ=500`

At batch size 32, 5,301 steps consume about 169,632 training examples, which
is approximately one complete pass over the full training split. The effective
rollout per step is `32 x 5 = 160` trajectories. Validation uses the complete
51,713-example test split, so validation before training and every 500 steps
can take substantial time on two H20 GPUs.

Use this mode only after validating the image with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 reproduction/7b_base/test_vllm_llama.py
```

Do not use `vllm==0.7.3` for this mode unless the veRL hybrid rollout wrapper
has been ported to the 0.7 internal API. Standalone `from vllm import LLM`
generation can work on 0.7.3, but Search-R1 training synchronizes FSDP weights
through veRL's vendored vLLM wrapper, which is aligned with 0.6.3.

After installing `flash_attn`, use `two_gpu_vllm_h20_flash` for the closest
2-GPU H20 match to the upstream vLLM/FlashAttention path:

```bash
python3 /workspace/filesdir/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_vllm_h20_flash
```

It uses the full training split with the two-H20 batch and rollout settings,
and enables:

- `MODEL_ATTN_IMPLEMENTATION=flash_attention_2`
- `USE_REMOVE_PADDING=true`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.35`
- `ROLLOUT_DTYPE=bfloat16`
- `ACTOR_MODEL_DTYPE=bfloat16`

If this mode hits a FlashAttention/H20 kernel error or `SIGFPE`, rerun the
same image with `--run-mode two_gpu_vllm_h20` to keep vLLM while disabling
FlashAttention-dependent padding removal.

### Exact v0.2 paper profile

`full` records the upstream v0.2 GRPO settings without two-GPU adaptations:
full train/validation splits, batch size 512, PPO mini batch 256, micro batch
64, prompt length 4096, response length 500, five agents, four search turns,
top-3 retrieval, vLLM memory utilization 0.6, 1,005 steps, and eight GPUs.
This reproduction changes only the backbone from Qwen2.5-7B to Llama-2-7B:

```bash
BASE_MODEL_NAME=llama-7b \
LOCAL_BASE_MODEL=/workspace/filesdir/models/7b_base/llama-7b \
python3 /workspace/filesdir/projects/Search-R1/reproduction/7b_base/training_task_entry.py \
  --algo grpo \
  --run-mode full
```

This is a controlled backbone substitution, not an exact reproduction of the
paper's Qwen2.5-7B result. It still requires an eight-GPU training task because
the paper's GPU count, global/micro batches, sequence lengths, and rollout
settings are unchanged. For two H20 GPUs, use `two_gpu_vllm_h20_flash`.

`two_gpu_balanced` remains available as a faster fallback if `two_gpu_paper`
is too slow: it uses `N_AGENT=3`, `MAX_RESPONSE_LENGTH=192`, and
`TOTAL_TRAINING_STEPS=200`.

`two_gpu_fast` remains available for debugging only. It uses `N_AGENT=2`,
`MAX_RESPONSE_LENGTH=128`, and disables validation by default, so it is faster
but less suitable for reporting experiment quality.

Single-H20 smoke run:

```bash
conda activate searchr1
ALGO=grpo RUN_MODE=h20_smoke bash reproduction/7b_base/run_profiled_train.sh
```

`h20_smoke` is the safest first run for a 1-GPU online development instance. It sets:

- `CUDA_VISIBLE_DEVICES=0`
- `N_GPUS_PER_NODE=1`
- `TRAIN_DATA_NUM=16`
- `VAL_DATA_NUM=8`
- `TRAIN_BATCH_SIZE=1`
- `PPO_MICRO_BATCH_SIZE=1`
- `MAX_PROMPT_LENGTH=512`
- `MAX_RESPONSE_LENGTH=64`
- `MAX_TURNS=1`
- `RETRIEVER_TOPK=1`
- `DO_SEARCH=true`
- `ROLLOUT_NAME=hf`
- `ROLLOUT_DTYPE=float16`
- `ROLLOUT_DO_SAMPLE=false`
- `ROLLOUT_REMOVE_INVALID_VALUES=true`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.25`
- `ACTOR_MODEL_DTYPE=float16`
- `HF_SUMMON_FULL_PARAMS=true`
- `HF_USE_CACHE=false`
- `RAY_memory_usage_threshold=0.99`
- `RAY_memory_monitor_refresh_ms=0`
- `TOTAL_TRAINING_STEPS=4`
- `SAVE_FREQ=1`

Keep a retriever running for this mode. On a 128GB online-dev node, use
`launch_lite_retriever.sh` for method validation; the full FAISS retriever and
HF 7B rollout/training do not reliably fit in the same container. This smoke
mode validates online retrieval, several training updates, checkpointing, GPU
memory logs, and report generation.

If the runner reports missing parquet files, prepare the dataset first:

```bash
bash reproduction/7b_base/prepare_data.sh
```

If the dataset is stored in a persistent directory outside the repository, pass it explicitly:

```bash
DATA_DIR=/filesdir/code/search-r1/data/nq_hotpotqa_train \
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
LOCAL_BASE_MODEL=/filesdir/code/search-r1/models/7b_base/llama-7b \
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
EVAL_MODEL=/filesdir/code/search-r1/outputs/checkpoints/nq_hotpotqa_train-search-r1-grpo-llama-7b-em/actor/global_step_1000 \
bash reproduction/7b_base/evaluate_7b_base.sh
```

## Notes

- Data, wiki corpus, FAISS index, model weights, checkpoints, logs, and wandb output are intentionally not committed.
- `data/`, `*.log`, checkpoints, and wandb directories are already covered by `.gitignore`.
- `models/7b_base/` is ignored so local downloaded model weights are not committed.
- The retriever must be running before training or evaluation, because rollouts call the `/retrieve` API whenever the model emits `<search>...</search>`.
