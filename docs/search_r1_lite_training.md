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

## Grounding Enhancement

If the model can search but fails to summarize evidence, build a grounding SFT
set with `post_search_answer` examples. This directly trains the missing state
transition:

```text
question + <search>query</search> + <information>evidence</information>
-> <think> supporting fact </think>
-> <answer> answer </answer>
```

Build the grounding data:

```bash
python scripts/data_process/nq_search_lite_sft.py \
  --input_dir data/nq_search \
  --local_dir data/nq_search_lite_grounding_sft \
  --task grounding \
  --topk 3 \
  --train_limit 3000 \
  --test_limit 384 \
  --filter_answer_in_evidence
```

Continue SFT from the previous Search-R1-lite SFT checkpoint:

```bash
BASE_MODEL=/root/autodl-tmp/projects/Search-R1/verl_checkpoints/qwen2.5-1.5b-search-lite-sft/global_step_1000 \
DATA_DIR=data/nq_search_lite_grounding_sft \
EXPERIMENT_NAME=qwen2.5-1.5b-search-lite-grounding-sft \
TRAIN_BATCH_SIZE=1 \
MICRO_BATCH_SIZE=1 \
MAX_LENGTH=1536 \
TOTAL_TRAINING_STEPS=500 \
SAVE_FREQ=100 \
bash scripts/sft_qwen15_search_lite_1gpu.sh
```

Then run shaped GRPO from the grounding SFT checkpoint:

```bash
BASE_MODEL=/root/autodl-tmp/projects/Search-R1/verl_checkpoints/qwen2.5-1.5b-search-lite-grounding-sft/global_step_500 \
TOPK=3 \
MAX_OBS_LENGTH=512 \
MAX_RESPONSE_LENGTH=192 \
TRAIN_DATA_NUM=2048 \
VAL_DATA_NUM=64 \
TOTAL_TRAINING_STEPS=400 \
TRAIN_BATCH_SIZE=2 \
VAL_BATCH_SIZE=2 \
TEMPERATURE=0.3 \
TOP_P=0.8 \
EXPERIMENT_NAME=nq-search-r1-qwen15-grounding-sft500-grpo-shaped-top3-400step \
bash scripts/train_grpo_qwen15_bm25_shaped_1gpu.sh
```

Compare against the previous result:

```text
SFT+GRPO shaped top3, obs512, 400 steps: val/test_score/nq = 0.203125
```
