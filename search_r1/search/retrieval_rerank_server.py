# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
检索和重排服务器 (Retrieval and Reranking Server)
========================================
功能：结合检索和交叉编码重排的联合服务器

架构：
1. 检索阶段：使用BM25或FAISS获取初始候选文档
2. 重排阶段：使用交叉编码器重新排序文档

工作流：
1. 接收查询列表
2. 使用检索器批量获取文档
3. 使用重排器重新评分和排序
4. 返回最终排序结果

依赖：
- sentence-transformers: 交叉编码器模型
- FastAPI: Web服务框架
"""

# pip install -U sentence-transformers
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 操作系统接口
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re  # 正则表达式
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse  # 命令行参数解析
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass, field  # 数据类
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Optional  # 类型注解
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import defaultdict  # 默认字典

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from fastapi import FastAPI  # FastAPI框架
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pydantic import BaseModel  # 数据验证模型
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from sentence_transformers import CrossEncoder  # 交叉编码器

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from retrieval_server import get_retriever, Config as RetrieverConfig  # 检索器
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from rerank_server import SentenceTransformerCrossEncoder  # 重排器

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
app = FastAPI()  # 创建FastAPI应用

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def convert_title_format(text):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    转换标题格式
    
    功能：将'(Title: xxx) content'格式转换为'"xxx"\\ncontent'格式
    
    参数：
        text: 原始文本
        
    返回：
        转换后的文本
    """
    # 使用正则表达式提取标题和内容
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if match:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        title, content = match.groups()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return f'\"{title}\"\n{content}'
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return text

# 中文注释：下一行定义类，用于组织相关状态与行为。
class SearchRequest(BaseModel):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    搜索请求数据模型
    
    属性：
        queries: 查询文本列表
        topk_retrieval: 检索阶段返回的文档数
        topk_rerank: 重排后返回的文档数
        return_scores: 是否返回相似度分数
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    queries: List[str]  # 查询列表
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    topk_retrieval: Optional[int] = 10  # 检索Top-K
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    topk_rerank: Optional[int] = 3  # 重排Top-K
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return_scores: bool = False  # 是否返回分数

# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class RerankerArguments:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    重排器配置参数
    
    属性：
        max_length: 最大输入长度
        rerank_topk: 重排返回文档数
        rerank_model_name_or_path: 重排模型路径
        batch_size: 批处理大小
        reranker_type: 重排器类型
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_length: int = field(default=512)  # 最大长度
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rerank_topk: int = field(default=3)  # 重排Top-K
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")  # 模型
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size: int = field(default=32)  # 批大小
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranker_type: str = field(default="sentence_transformer")  # 重排器类型

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_reranker(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    获取重排器实例
    
    参数：
        config: 重排器配置对象
        
    返回：
        交叉编码器重排器实例
        
    说明：
    - 支持sentence_transformer交叉编码器
    - 自动选择GPU或CPU
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.reranker_type == "sentence_transformer":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return SentenceTransformerCrossEncoder.load(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            config.rerank_model_name_or_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size=config.batch_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device="cuda" if torch.cuda.is_available() else "cpu"  # GPU优先
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")

# ----------- Endpoint -----------
# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@app.post("/retrieve")
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def search_endpoint(request: SearchRequest):
    # Step 1: Retrieve documents
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retrieved_docs = retriever.batch_search(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_list=request.queries,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num=request.topk_retrieval,
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_score=False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )

    # Step 2: Rerank
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranked = reranker.rerank(request.queries, retrieved_docs)

    # Step 3: Format response
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i, doc_scores in reranked.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        doc_scores = doc_scores[:request.topk_rerank]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if request.return_scores:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            combined = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for doc, score in doc_scores:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                combined.append({"document": convert_title_format(doc), "score": score})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response.append(combined)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response.append([convert_title_format(doc) for doc, _ in doc_scores])

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {"result": response}


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # retriever
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of retrieved passages for one query.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    # reranker
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--reranking_topk", type=int, default=3, help="Number of reranked passages for one query.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2", help="Path of the reranker model.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Batch size for the reranker inference.")

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()
    
    # ----------- Load Retriever and Reranker -----------
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retriever_config = RetrieverConfig(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_method = args.retriever_name,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index_path=args.index_path,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        corpus_path=args.corpus_path,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_topk=args.retrieval_topk,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        faiss_gpu=args.faiss_gpu,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_model_path=args.retriever_model,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_pooling_method="mean",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_query_max_length=256,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_use_fp16=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_batch_size=512,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retriever = get_retriever(retriever_config)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranker_config = RerankerArguments(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rerank_topk = args.reranking_topk,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rerank_model_name_or_path = args.reranker_model,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = args.reranker_batch_size,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranker = get_reranker(reranker_config)
    
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import uvicorn
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    uvicorn.run(app, host="0.0.0.0", port=8000)
