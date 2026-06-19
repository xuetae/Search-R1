# Search-R1 Code Upload And Platform Rebuild

Upload this code archive to the xFusion file manager `Search_R1` directory:

```text
Search-R1-code-llama-7b-run-1f10069.tar.gz
```

Also upload local model archives:

```text
llama-7b.tar.gz
e5-base-v2.tar.gz
```

On the platform, mount the file-management directory to:

```text
/filesdir
```

Then run these commands in Terminal.

## 1. Create Clear Project Layout

```bash
mkdir -p /filesdir/code/search-r1/projects
mkdir -p /filesdir/code/search-r1/uploads
mkdir -p /filesdir/code/search-r1/data
mkdir -p /filesdir/code/search-r1/models
mkdir -p /filesdir/code/search-r1/outputs
mkdir -p /filesdir/code/search-r1/logs
mkdir -p /filesdir/code/search-r1/cache
```

## 2. Move Uploaded Archives

```bash
mv /filesdir/Search-R1-code-llama-7b-run-1f10069.tar.gz /filesdir/code/search-r1/uploads/ 2>/dev/null || true
mv /filesdir/llama-7b.tar.gz /filesdir/code/search-r1/uploads/ 2>/dev/null || true
mv /filesdir/e5-base-v2.tar.gz /filesdir/code/search-r1/uploads/ 2>/dev/null || true
```

## 3. Extract Code

```bash
cd /filesdir/code/search-r1/projects
rm -rf Search-R1
tar -xzf /filesdir/code/search-r1/uploads/Search-R1-code-llama-7b-run-1f10069.tar.gz
cd /filesdir/code/search-r1/projects/Search-R1
```

## 4. Extract Models And Download Platform Data

This downloads `train.parquet`, `test.parquet`, `wiki-18.jsonl`, and
`e5_Flat.index` on the platform.

```bash
bash reproduction/7b_base/rebuild_platform_project.sh \
  --skip-code \
  --llama-archive /filesdir/code/search-r1/uploads/llama-7b.tar.gz \
  --e5-archive /filesdir/code/search-r1/uploads/e5-base-v2.tar.gz \
  --download-data
```

## 5. Check

```bash
SEARCH_R1_ROOT=/filesdir/code/search-r1 \
WORK_DIR=/filesdir/code/search-r1/projects/Search-R1 \
bash reproduction/7b_base/preflight_check.sh
```

## 6. xFusion Training Task Command

```bash
SEARCH_R1_ROOT=/filesdir/code/search-r1 WORK_DIR=/filesdir/code/search-r1/projects/Search-R1 python3 /filesdir/code/search-r1/projects/Search-R1/reproduction/7b_base/training_task_entry.py --algo grpo --run-mode two_gpu_paper
```

Expected final layout:

```text
/filesdir/code/search-r1/
  projects/Search-R1/
  uploads/
  data/nq_hotpotqa_train/train.parquet
  data/nq_hotpotqa_train/test.parquet
  data/wiki-18/wiki-18.jsonl
  data/wiki-18/e5_Flat.index
  models/7b_base/llama-7b/
  models/7b_base/e5-base-v2/
  outputs/
  logs/
  cache/
```
