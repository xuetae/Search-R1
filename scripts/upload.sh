# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===

# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
index=/home/peterjin/mnt/index/wiki-18/e5_Flat.index

# 中文注释：下一行执行脚本步骤，保持原有复现流程不变。
split -b 40G $index part_

# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python upload.py
