#!/usr/bin/env bash
set -euo pipefail

file_path="${SEARCH_R1_BM25_DIR:-/the/path/you/save/bm25}"
index_file="${SEARCH_R1_BM25_INDEX:-$file_path/bm25}"
corpus_file="${SEARCH_R1_BM25_CORPUS:-$file_path/wiki-18.jsonl}"
retriever_name=bm25

python search_r1/search/retrieval_server.py \
    --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk 1 \
    --retriever_name "$retriever_name"
