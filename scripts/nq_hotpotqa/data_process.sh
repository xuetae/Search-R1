# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始配置/脚本逻辑保持不变。===
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
WORK_DIR=your/work/dir
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

## process multiple dataset search format train file
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
DATA=nq,hotpotqa
# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
# 中文注释：下一行定义 Shell 变量，用于集中管理路径、模型或实验参数。
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# 中文注释：下一行启动 Python 程序，是脚本的主要执行命令。
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
