# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
下载脚本
========
功能：从Hugging Face Hub下载预构建的检索索引和文档库
用于本地检索系统的快速部署

下载内容：
1. E5编码器的FAISS索引（Wiki-18语料）
2. Wiki-18文档库JSONL文件

注意：索引文件较大，建议有良好的网络连接
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse  # 命令行参数解析
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from huggingface_hub import hf_hub_download  # Hugging Face Hub文件下载

# 设置命令行参数
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", 
                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                   help="Hugging Face repository ID")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument("--save_path", type=str, required=True, 
                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                   help="Local directory to save files")
    
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
args = parser.parse_args()

# ==================== 下载E5索引 ====================
# 下载Wiki-18 E5 FAISS索引（分为两部分）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
repo_id = "PeterJinGo/wiki-18-e5-index"
# 索引被分成两部分以便传输（part_aa和part_ab）
# 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
for file in ["part_aa", "part_ab"]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f"Downloading {file}...")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hf_hub_download(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_id=repo_id,  # 仓库ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        filename=file,  # 文件名
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_type="dataset",  # 这是一个数据集仓库
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_dir=args.save_path,  # 本地保存目录
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f"Downloaded {file} to {args.save_path}")

# ==================== 下载文档库 ====================
# 下载Wiki-18文档库（JSONL格式，gzip压缩）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
print("Downloading corpus...")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
repo_id = "PeterJinGo/wiki-18-corpus"
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
hf_hub_download(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_id=repo_id,  # 仓库ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        filename="wiki-18.jsonl.gz",  # 文件名（gzip压缩）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repo_type="dataset",  # 数据集仓库
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_dir=args.save_path,  # 本地保存目录
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
print(f"Downloaded corpus to {args.save_path}")

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
print("Download completed!")  # 下载完成提示
