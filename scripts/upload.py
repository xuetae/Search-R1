# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from huggingface_hub import upload_file

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
repo_id = "PeterJinGo/wiki-18-e5-index"
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
path = "/home/peterjin/mnt/index/wiki-18"
# 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
for file in ["part_aa", "part_ab"]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    upload_file(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        path_or_fileobj=os.path.join(path, file),  # File path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        path_in_repo=file,  # Destination filename in the repo
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_id=repo_id,  # Your dataset repo ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_type="dataset"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
