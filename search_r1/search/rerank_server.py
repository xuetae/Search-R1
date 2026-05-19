# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import defaultdict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass, field

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from sentence_transformers import CrossEncoder
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import HfArgumentParser
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import uvicorn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from fastapi import FastAPI
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pydantic import BaseModel


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BaseCrossEncoder:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model, batch_size=32, device="cuda"):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch_size = batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model.to(device)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _passage_to_string(self, doc_item):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "document" not in doc_item:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content = doc_item['contents']
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content = doc_item['document']['contents']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        title = content.split("\n")[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        text = "\n".join(content.split("\n")[1:])

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return f"(Title: {title}) {text}"

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def rerank(self, 
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               queries: list[str], 
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               documents: list[list[dict]]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Assume documents is a list of list of dicts, where each dict is a document with keys "id" and "contents".
        This asumption is made to be consistent with the output of the retrieval server.
        """ 
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(queries) == len(documents)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pairs = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        qids = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for qid, query in enumerate(queries):
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for document in documents:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for doc_item in document:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    doc = self._passage_to_string(doc_item)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pairs.append((query, doc))
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    qids.append(qid)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = self._predict(pairs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_to_doc_scores = defaultdict(list)

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(scores) == len(pairs) == len(qids)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(len(pairs)):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query, doc = pairs[i]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = scores[i] 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            qid = qids[i]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_to_doc_scores[qid].append((doc, score))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sorted_query_to_doc_scores = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for query, doc_scores in query_to_doc_scores.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return sorted_query_to_doc_scores

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _predict(self, pairs: list[tuple[str, str]]):
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError 

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load(cls, model_name_or_path, **kwargs):
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError


# 中文注释：下一行定义类，用于组织相关状态与行为。
class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model, batch_size=32, device="cuda"):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(model, batch_size, device)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _predict(self, pairs: list[tuple[str, str]]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return scores

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load(cls, model_name_or_path, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = CrossEncoder(model_name_or_path)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return cls(model, **kwargs)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RerankRequest(BaseModel):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    queries: list[str]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    documents: list[list[dict]]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rerank_topk: Optional[int] = None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return_scores: bool = False


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass 
# 中文注释：下一行定义类，用于组织相关状态与行为。
class RerankerArguments:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_length: int = field(default=512)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rerank_topk: int = field(default=3)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size: int = field(default=32)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranker_type: str = field(default="sentence_transformer")

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_reranker(config):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.reranker_type == "sentence_transformer":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return SentenceTransformerCrossEncoder.load(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            config.rerank_model_name_or_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size=config.batch_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device="cuda" if torch.cuda.is_available() else "cpu"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
app = FastAPI()

# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@app.post("/rerank")
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def rerank_endpoint(request: RerankRequest):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [[doc_item_1, ..., doc_item_k], [doc_item_1, ..., doc_item_k]],
      "rerank_topk": 3,
      "return_scores": true
    }
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not request.rerank_topk:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        request.rerank_topk = config.rerank_topk  # fallback to default

    # Perform batch re reranking
    # doc_scores already sorted by score
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    query_to_doc_scores = reranker.rerank(request.queries, request.documents) 

    # Format response 
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resp = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, doc_scores in query_to_doc_scores.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        doc_scores = doc_scores[:request.rerank_topk]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if request.return_scores:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            combined = [] 
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for doc, score in doc_scores:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                combined.append({"document": doc, "score": score})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resp.append(combined)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resp.append([doc for doc, _ in doc_scores])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {"result": resp}


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = HfArgumentParser((RerankerArguments))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config = parser.parse_args_into_dataclasses()[0]

    # 2) Instantiate a global retriever so it is loaded once and reused.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reranker = get_reranker(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    uvicorn.run(app, host="0.0.0.0", port=6980)
