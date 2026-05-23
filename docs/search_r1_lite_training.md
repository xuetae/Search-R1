# Search-R1-lite Training

This path is intended for a single 48GB GPU and small models such as
`Qwen2.5-1.5B`. It separates the hard end-to-end Search-R1 behavior into
supervised sub-skills before running GRPO.

## What Changes

- Query SFT trains `question -> <think>...</think><search> concise query </search>`.
- Evidence QA SFT trains `question + retrieved evidence -> <think>...</think><answer> answer </answer>`.
- Decision SFT mixes search and answer actions.
- Shaped GRPO uses the existing format-aware reward implementation so small
  models receive non-zero signal before exact-match answers become frequent.

## Data

Start from the usual Search-R1 NQ parquet files:

```bash
python scripts/data_process/nq_search.py --local_dir data/nq_search
```

With the BM25 retriever running, build a mixed SFT dataset:

```bash
python scripts/data_process/nq_search_lite_sft.py \
  --input_dir data/nq_search \
  --local_dir data/nq_search_lite_sft \
  --task mixed \
  --topk 3 \
  --train_limit 5000 \
  --test_limit 512 \
  --filter_answer_in_evidence
```

Filtering keeps evidence-QA examples only when retrieved evidence contains a
golden answer. This intentionally trades data size for cleaner small-model
supervision.

## SFT

```bash
BASE_MODEL=/root/autodl-tmp/models/Qwen2.5-1.5B \
DATA_DIR=data/nq_search_lite_sft \
EXPERIMENT_NAME=qwen2.5-1.5b-search-lite-sft \
bash scripts/sft_qwen15_search_lite_1gpu.sh
```

Use the saved SFT actor checkpoint as `BASE_MODEL` for the next GRPO stage.

## Shaped GRPO

```bash
BASE_MODEL=verl_checkpoints/qwen2.5-1.5b-search-lite-sft/global_step_1000 \
TOPK=1 \
MAX_RESPONSE_LENGTH=192 \
MAX_OBS_LENGTH=384 \
TOTAL_TRAINING_STEPS=400 \
EXPERIMENT_NAME=nq-search-r1-lite-qwen15-bm25-top1-shaped \
bash scripts/train_grpo_qwen15_bm25_shaped_1gpu.sh
```

After top1 is stable, try `TOPK=3` with `MAX_OBS_LENGTH=512`.
