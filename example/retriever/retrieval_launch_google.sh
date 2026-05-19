# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python search_r1/search/internal_google_server.py --api_key $api_key \
                                            --topk 5 \
                                            --cse_id $cse_id \
                                            --snippet_only
