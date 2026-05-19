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

import argparse  # 命令行参数解析
from huggingface_hub import hf_hub_download  # Hugging Face Hub文件下载

# 设置命令行参数
parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", 
                   help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, 
                   help="Local directory to save files")
    
args = parser.parse_args()

# ==================== 下载E5索引 ====================
# 下载Wiki-18 E5 FAISS索引（分为两部分）
repo_id = "PeterJinGo/wiki-18-e5-index"
# 索引被分成两部分以便传输（part_aa和part_ab）
for file in ["part_aa", "part_ab"]:
    print(f"Downloading {file}...")
    hf_hub_download(
        repo_id=repo_id,  # 仓库ID
        filename=file,  # 文件名
        repo_type="dataset",  # 这是一个数据集仓库
        local_dir=args.save_path,  # 本地保存目录
    )
    print(f"Downloaded {file} to {args.save_path}")

# ==================== 下载文档库 ====================
# 下载Wiki-18文档库（JSONL格式，gzip压缩）
print("Downloading corpus...")
repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
        repo_id=repo_id,  # 仓库ID
        filename="wiki-18.jsonl.gz",  # 文件名（gzip压缩）
        repo_type="dataset",  # 数据集仓库
        local_dir=args.save_path,  # 本地保存目录
)
print(f"Downloaded corpus to {args.save_path}")

print("Download completed!")  # 下载完成提示
