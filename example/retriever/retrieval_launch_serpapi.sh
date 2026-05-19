# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
search_url=https://serpapi.com/search
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
serp_api_key="" # put your serp api key here (https://serpapi.com/)

# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python search_r1/search/online_search_server.py --search_url $search_url \
                                            --topk 3 \
                                            --serp_api_key $serp_api_key
