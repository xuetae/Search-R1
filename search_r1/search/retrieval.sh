# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
DATA_NAME=nq

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
DATASET_PATH="/home/peterjin/mnt/data/$DATA_NAME"

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
SPLIT='test'
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
TOPK=3

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
INDEX_PATH=/home/peterjin/mnt/index/wiki-18
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
CORPUS_PATH=/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
SAVE_NAME=e5_${TOPK}_wiki18.json

# INDEX_PATH=/home/peterjin/rm_retrieval_corpus/index/wiki-21
# CORPUS_PATH=/home/peterjin/rm_retrieval_corpus/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
# SAVE_NAME=e5_${TOPK}_wiki21.json

# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retrieval.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
