# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
corpus_file=/your/corpus/jsonl/file # jsonl
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
save_dir=/the/path/to/save/index
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
retriever_name=e5 # this is for indexing naming
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
retriever_model=intfloat/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
