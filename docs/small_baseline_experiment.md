# Small Baseline Experiment

This experiment checks whether the small Search-R1 chain improves over practical baselines on a single 48GB GPU.

## Compared Runs

Use the same base model, validation split, and metric for all runs:

- Base model: `Qwen2.5-1.5B`
- Dataset: `data/nq_search/test.parquet` for search runs, `data/nq/test.parquet` for direct runs
- Metric: `val/test_score/nq` exact match
- Suggested validation size: `VAL_DATA_NUM=256`

| Run | Model | Search | Training | Script |
| --- | --- | --- | --- | --- |
| Direct baseline | base model | no | no | `scripts/smoke_eval_baseline_1gpu.sh` with `MODE=direct` |
| BM25 baseline | base model | BM25 top1/top3 | no | `scripts/smoke_eval_baseline_1gpu.sh` with `MODE=search` |
| Search-R1 top1 | trained checkpoint | BM25 top1 | yes | `scripts/smoke_eval_searchr1_checkpoint_1gpu.sh` |
| Search-R1 top3 | trained checkpoint | BM25 top3 | yes | `scripts/smoke_eval_searchr1_checkpoint_1gpu.sh` |

## Data Preparation

Prepare both prompt formats if you want to run direct and search baselines:

```bash
python scripts/data_process/nq.py --local_dir ./data/nq
python scripts/data_process/nq_search.py --local_dir ./data/nq_search
```

## Retriever

For BM25 top3, start the retriever with `--topk 3`, then request `TOPK=3` in evaluation or training:

```bash
export SEARCH_R1_BM25_DIR=/root/autodl-tmp/wiki18_bm25
export OPENAI_API_KEY=dummy
python search_r1/search/retrieval_server.py \
  --index_path $SEARCH_R1_BM25_DIR/bm25 \
  --corpus_path $SEARCH_R1_BM25_DIR/wiki-18.jsonl \
  --topk 3 \
  --retriever_name bm25
```

## Example Commands

Direct baseline:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B \
MODE=direct \
VAL_DATA_NUM=256 \
bash scripts/smoke_eval_baseline_1gpu.sh
```

BM25 top1 baseline without training:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B \
MODE=search \
TOPK=1 \
VAL_DATA_NUM=256 \
bash scripts/smoke_eval_baseline_1gpu.sh
```

Small Search-R1 top3 training:

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B \
TOPK=3 \
MAX_OBS_LENGTH=384 \
bash scripts/train_grpo_qwen15_bm25_top3_1gpu.sh
```

Trained Search-R1 checkpoint evaluation:

```bash
MODEL_PATH=/path/to/trained/checkpoint \
TOPK=3 \
VAL_DATA_NUM=256 \
MAX_OBS_LENGTH=384 \
bash scripts/smoke_eval_searchr1_checkpoint_1gpu.sh
```

## Decision Rule

Use the same `VAL_DATA_NUM` and `TOPK` when comparing runs. Search-R1 shows a useful gain only if:

```text
trained Search-R1 EM > untrained BM25 baseline EM
```

If the trained model is not better, increase training steps before changing more variables:

```text
TOTAL_TRAINING_STEPS=400 -> 1000
TRAIN_DATA_NUM=2048 -> 4096 or 8192
```
