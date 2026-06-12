# 7B Base Search-R1 Reproduction Setup

This branch adds a self-contained launcher set under `scripts/llama_7b_run/` for testing the paper-style 7B base result.

## What To Prepare

- Python/Search-R1 environment: install the main repo requirements as described in `README.md`.
- Retriever environment: install `faiss-gpu`, `pyserini`, `fastapi`, `uvicorn`, and the retriever dependencies from `README.md`.
- GPUs: the upstream 7B scripts assume 1 node with 8 GPUs.
- Dataset: `PeterJinGo/nq_hotpotqa_train`, downloaded to `data/nq_hotpotqa_train`.
- Retrieval corpus/index: `wiki-18.jsonl` plus `e5_Flat.index`, downloaded to `data/wiki-18`.
- Base model: default is `Qwen/Qwen2.5-7B`, matching the repository's v0.2 7B-base scripts.

The branch name is `llama-7b-run`, but this repository's paper reproduction scripts do not contain an official LLaMA-7B setting. The available 7B-base paper configuration is Qwen2.5-7B. You can override `BASE_MODEL` if you want to test a local LLaMA-family 7B checkpoint.

## Prepare Data

```bash
bash scripts/llama_7b_run/prepare_data.sh
```

Optional overrides:

```bash
WORK_DIR=/path/to/Search-R1 \
DATA_DIR=/path/to/data/nq_hotpotqa_train \
WIKI18_DIR=/path/to/data/wiki-18 \
bash scripts/llama_7b_run/prepare_data.sh
```

## Launch Retriever

Run this in the retriever environment:

```bash
bash scripts/llama_7b_run/launch_retriever.sh
```

The server listens on `http://127.0.0.1:8000/retrieve` by default. Override with `RETRIEVER_URL` in the training/evaluation shell if needed.

## Train 7B Base

GRPO, matching the upstream v0.2 7B-base GRPO settings:

```bash
bash scripts/llama_7b_run/train_grpo_7b_base.sh
```

PPO, matching the upstream v0.2 7B-base PPO settings:

```bash
bash scripts/llama_7b_run/train_ppo_7b_base.sh
```

Useful overrides:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B \
EXPERIMENT_NAME=nq_hotpotqa_train-search-r1-grpo-qwen2.5-7b-em \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/llama_7b_run/train_grpo_7b_base.sh
```

## Evaluate

Evaluate the base model:

```bash
bash scripts/llama_7b_run/evaluate_7b_base.sh
```

Evaluate a trained checkpoint:

```bash
EVAL_MODEL=verl_checkpoints/nq_hotpotqa_train-search-r1-grpo-qwen2.5-7b-em/global_step_1000/actor \
bash scripts/llama_7b_run/evaluate_7b_base.sh
```

## Notes

- Data, wiki corpus, FAISS index, model weights, checkpoints, logs, and wandb output are intentionally not committed.
- `data/`, `*.log`, checkpoints, and wandb directories are already covered by `.gitignore`.
- The retriever must be running before training or evaluation, because rollouts call the `/retrieve` API whenever the model emits `<search>...</search>`.

