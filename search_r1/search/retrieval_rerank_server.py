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
import os  # 操作系统接口
import re  # 正则表达式
import argparse  # 命令行参数解析
from dataclasses import dataclass, field  # 数据类
from typing import List, Optional  # 类型注解
from collections import defaultdict  # 默认字典

import torch  # PyTorch
import numpy as np  # NumPy
from fastapi import FastAPI  # FastAPI框架
from pydantic import BaseModel  # 数据验证模型
from sentence_transformers import CrossEncoder  # 交叉编码器

from retrieval_server import get_retriever, Config as RetrieverConfig  # 检索器
from rerank_server import SentenceTransformerCrossEncoder  # 重排器

app = FastAPI()  # 创建FastAPI应用

def convert_title_format(text):
    """
    转换标题格式
    
    功能：将'(Title: xxx) content'格式转换为'"xxx"\\ncontent'格式
    
    参数：
        text: 原始文本
        
    返回：
        转换后的文本
    """
    # 使用正则表达式提取标题和内容
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'\"{title}\"\n{content}'
    else:
        return text

class SearchRequest(BaseModel):
    """
    搜索请求数据模型
    
    属性：
        queries: 查询文本列表
        topk_retrieval: 检索阶段返回的文档数
        topk_rerank: 重排后返回的文档数
        return_scores: 是否返回相似度分数
    """
    queries: List[str]  # 查询列表
    topk_retrieval: Optional[int] = 10  # 检索Top-K
    topk_rerank: Optional[int] = 3  # 重排Top-K
    return_scores: bool = False  # 是否返回分数

@dataclass
class RerankerArguments:
    """
    重排器配置参数
    
    属性：
        max_length: 最大输入长度
        rerank_topk: 重排返回文档数
        rerank_model_name_or_path: 重排模型路径
        batch_size: 批处理大小
        reranker_type: 重排器类型
    """
    max_length: int = field(default=512)  # 最大长度
    rerank_topk: int = field(default=3)  # 重排Top-K
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")  # 模型
    batch_size: int = field(default=32)  # 批大小
    reranker_type: str = field(default="sentence_transformer")  # 重排器类型

def get_reranker(config):
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
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"  # GPU优先
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")

# ----------- Endpoint -----------
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    # Step 1: Retrieve documents
    retrieved_docs = retriever.batch_search(
        query_list=request.queries,
        num=request.topk_retrieval,
        return_score=False
    )

    # Step 2: Rerank
    reranked = reranker.rerank(request.queries, retrieved_docs)

    # Step 3: Format response
    response = []
    for i, doc_scores in reranked.items():
        doc_scores = doc_scores[:request.topk_rerank]
        if request.return_scores:
            combined = []
            for doc, score in doc_scores:
                combined.append({"document": convert_title_format(doc), "score": score})
            response.append(combined)
        else:
            response.append([convert_title_format(doc) for doc, _ in doc_scores])

    return {"result": response}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # retriever
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--retrieval_topk", type=int, default=10, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    # reranker
    parser.add_argument("--reranking_topk", type=int, default=3, help="Number of reranked passages for one query.")
    parser.add_argument("--reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2", help="Path of the reranker model.")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Batch size for the reranker inference.")

    args = parser.parse_args()
    
    # ----------- Load Retriever and Reranker -----------
    retriever_config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.retrieval_topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    retriever = get_retriever(retriever_config)

    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
    )
    reranker = get_reranker(reranker_config)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
