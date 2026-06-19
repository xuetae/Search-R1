# Search-R1 Platform Recovery

This folder is for rebuilding the xFusion platform project after deleting the
old algorithm entry.

## Files

- `Search-R1-code-llama-7b-run-fbed069.tar.gz`: clean source-code archive from
  branch `llama-7b-run`, commit `fbed069`.
- `rebuild_platform_project.sh`: standalone platform rebuild helper.

The archive intentionally excludes local `.git` history, model weights, wiki
index files, training outputs, and cache directories.

## Recommended Platform Layout

Use the file-management mount as the persistent root:

```text
/filesdir/code/search-r1
```

Create this layout:

```text
/filesdir/code/search-r1/projects/Search-R1
/filesdir/code/search-r1/data/nq_hotpotqa_train
/filesdir/code/search-r1/data/wiki-18
/filesdir/code/search-r1/models/7b_base/llama-7b
/filesdir/code/search-r1/models/7b_base/e5-base-v2
/filesdir/code/search-r1/outputs
/filesdir/code/search-r1/logs
/filesdir/code/search-r1/cache
```

Do not store large data under `/workspace`. On xFusion training tasks,
`/workspace` is usually the algorithm mount and can disappear when the algorithm
entry is deleted.

## Restore Code On The Platform

Upload these files to the file manager root, which is mounted as `/filesdir`:

```text
Search-R1-code-llama-7b-run-fbed069.tar.gz
rebuild_platform_project.sh
llama-7b.tar.gz
e5-base-v2.tar.gz
```

Then in the online-development terminal run:

```bash
bash /filesdir/rebuild_platform_project.sh \
  --root /filesdir/code/search-r1 \
  --code-archive /filesdir/Search-R1-code-llama-7b-run-fbed069.tar.gz \
  --llama-archive /filesdir/llama-7b.tar.gz \
  --e5-archive /filesdir/e5-base-v2.tar.gz
```

## Restore Models

Upload local model archives, then extract them under:

```text
/filesdir/code/search-r1/models/7b_base
```

Expected final paths:

```text
/filesdir/code/search-r1/models/7b_base/llama-7b/config.json
/filesdir/code/search-r1/models/7b_base/e5-base-v2/config.json
```

The project now includes a platform rebuild helper. After uploading
`llama-7b.tar.gz` and `e5-base-v2.tar.gz` to `/filesdir`, run:

```bash
cd /filesdir/code/search-r1/projects/Search-R1
bash reproduction/7b_base/rebuild_platform_project.sh
```

This creates the clean directory layout, extracts uploaded models, downloads
the training parquet files, downloads wiki-18, merges `e5_Flat.index`, and
prints the final training-task command.

## Restore Data

Expected final paths:

```text
/filesdir/code/search-r1/data/nq_hotpotqa_train/train.parquet
/filesdir/code/search-r1/data/nq_hotpotqa_train/test.parquet
/filesdir/code/search-r1/data/wiki-18/wiki-18.jsonl
/filesdir/code/search-r1/data/wiki-18/e5_Flat.index
```

If these data files are gone, rerun:

```bash
cd /filesdir/code/search-r1/projects/Search-R1
SEARCH_R1_ROOT=/filesdir/code/search-r1 \
WORK_DIR=/filesdir/code/search-r1/projects/Search-R1 \
bash reproduction/7b_base/prepare_data.sh
```

The wiki corpus and FAISS index are large; verify the file-manager free space
before downloading. Expected final size is roughly 134GB after decompression
and index merge.

## Preflight

Run:

```bash
cd /filesdir/code/search-r1/projects/Search-R1
SEARCH_R1_ROOT=/filesdir/code/search-r1 \
WORK_DIR=/filesdir/code/search-r1/projects/Search-R1 \
bash reproduction/7b_base/preflight_check.sh
```

## xFusion Training Task Command

Use an absolute path and override the project root because this platform mounts
file management at `/filesdir`:

```bash
SEARCH_R1_ROOT=/filesdir/code/search-r1 WORK_DIR=/filesdir/code/search-r1/projects/Search-R1 python3 /filesdir/code/search-r1/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

Recommended resources:

```text
Training type: single-node training
Compute type: GPU / full card
GPU: 2 x H20
Basic resource: 8CPU 520GBMEM preferred, 8CPU 360GBMEM acceptable
Visualization resource: disabled
Max failed restarts: 0 or 1 while validating
```
