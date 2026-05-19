# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
file_path=/the/path/you/save/corpus
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
index_file=$file_path/e5_HNSW64.index
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
corpus_file=$file_path/wiki-18.jsonl
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
retriever_name=e5
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
retriever_path=intfloat/e5-base-v2

# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path
