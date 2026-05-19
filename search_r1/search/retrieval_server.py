"""
检索服务器 (Retrieval Server)
============================
功能：提供多种检索后端的统一接口

支持的检索方法：
1. BM25: 稀疏检索，基于Lucene索引（Pyserini）
2. Dense: 稠密检索，基于FAISS向量索引

主要特性：
- FastAPI Web服务器提供HTTP接口
- 支持查询编码和批量检索
- 支持多个编码器模型（E5、BGE、DPR等）
- GPU加速（FAISS GPU索引）

典型工作流：
1. 加载语料库（JSONL格式）
2. 初始化检索器（BM25或Dense）
3. 对查询进行编码
4. 搜索索引获取相关文档
5. 返回排名结果
"""

import json  # JSON序列化
import os  # 操作系统接口
import warnings  # 警告管理
from typing import List, Dict, Optional  # 类型注解
import argparse  # 命令行参数解析

import faiss  # Facebook AI Similarity Search
import torch  # PyTorch张量库
import numpy as np  # NumPy数组库
from transformers import AutoConfig, AutoTokenizer, AutoModel  # Hugging Face模型加载
from tqdm import tqdm  # 进度条显示
import datasets  # Hugging Face数据集库

import uvicorn  # ASGI服务器
from fastapi import FastAPI  # FastAPI Web框架
from pydantic import BaseModel  # 数据验证模型


def load_corpus(corpus_path: str):
    """
    加载语料库
    
    参数：
        corpus_path: JSONL格式语料库文件路径
        
    返回：
        datasets.Dataset: 加载后的数据集对象
        
    说明：
    - 使用4个进程并行加载
    - JSONL格式：每行一个JSON对象
    """
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4  # 4进程并行加载
    )
    return corpus


def read_jsonl(file_path):
    """
    读取JSONL文件
    
    参数：
        file_path: 文件路径
        
    返回：
        list: 文件中所有JSON对象的列表
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    """
    从语料库中获取指定索引的文档
    
    参数：
        corpus: 语料库数据集
        doc_idxs: 文档索引列表
        
    返回：
        list: 对应索引的文档列表
    """
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    """
    加载预训练模型和分词器
    
    参数：
        model_path: 模型路径（Hugging Face模型ID或本地路径）
        use_fp16: 是否使用FP16半精度
        
    返回：
        (model, tokenizer): 模型和分词器元组
        
    说明：
    - 模型放在GPU上
    - 设置为eval模式（禁用Dropout等）
    - 支持trust_remote_code用于自定义代码
    """
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()  # 评估模式
    model.cuda()  # 移到GPU
    if use_fp16: 
        model = model.half()  # 转换为FP16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    """
    从模型输出中提取句子表示
    
    参数：
        pooler_output: [CLS]标记对应的输出（来自BERT类模型）
        last_hidden_state: 最后一层隐藏状态 [batch_size, seq_len, hidden_size]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        pooling_method: 池化方法
            - "mean": 平均池化（考虑掩码）
            - "cls": 使用[CLS]标记
            - "pooler": 使用模型的pooler输出
        
    返回：
        torch.Tensor: 句子表示 [batch_size, hidden_size]
    """
    if pooling_method == "mean":
        # 平均池化：掩码位置置0，计算平均值
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        # 使用第一个标记（通常是[CLS]）
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        # 使用模型的pooler输出
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    """
    文本编码器
    
    功能：将文本转换为向量表示
    
    支持的模型：
    - E5: 支持通用查询编码
    - BGE: 中文检索优化
    - DPR: Dense Passage Retrieval
    - T5-based: T5-Retriever等基于T5的模型
    
    工作流：
    1. 对文本添加模型特定前缀（如"query: "用于E5）
    2. 分词并填充
    3. 输入模型获取表示
    4. 使用指定池化方法提取句子表示
    5. 正则化（除DPR外）
    """
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        """
        初始化编码器
        
        参数：
            model_name: 模型名称（用于检测模型类型）
            model_path: 模型路径
            pooling_method: 池化方法（mean/cls/pooler）
            max_length: 最大序列长度
            use_fp16: 是否使用FP16
        """
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """
        编码文本列表
        
        参数：
            query_list: 文本列表或单个文本
            is_query: 是否为查询（vs文档）
                      影响添加的前缀
        
        返回：
            np.ndarray: 编码表示 [batch_size, hidden_size]
                       dtype为float32，C顺序
        
        说明：
        - 自动处理模型特定的前缀
        - E5: "query: "或"passage: "前缀
        - BGE: 查询添加特殊提示语
        - T5模型需要decoder_input_ids
        - 结果正则化（L2），除DPR外
        """
        # 处理单个文本转列表
        if isinstance(query_list, str):
            query_list = [query_list]

        # E5模型：添加查询/文档前缀
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        # BGE模型：查询添加检索提示
        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # 分词
        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,  # 填充到最大长度
                                truncation=True,  # 截断超过最大长度的部分
                                return_tensors="pt"  # 返回PyTorch张量
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}  # 移到GPU

        if "T5" in type(self.model).__name__:
            # T5模型需要decoder_input_ids初始化
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]  # 使用第一个token
        else:
            # 其他模型的标准流程
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            # DPR不需要正则化
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)  # L2正则化

        # 转换为NumPy数组并清理GPU内存
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")  # C顺序用于FAISS
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    """
    检索器基类
    
    定义检索器的标准接口
    """
    def __init__(self, config):
        """初始化检索器配置"""
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        """单个查询搜索（子类实现）"""
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        """批量查询搜索（子类实现）"""
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        """单个查询搜索接口"""
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """批量查询搜索接口"""
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    """
    BM25稀疏检索器
    
    功能：基于Lucene索引的BM25算法检索
    
    工作原理：
    - BM25：统计学检索模型，考虑TF-IDF和文档长度
    - Lucene索引：倒排索引，支持快速词项查询
    - Pyserini：LuceneSearcher调用Lucene Java接口
    """
    def __init__(self, config):
        """
        初始化BM25检索器
        
        参数：
            config: 配置对象，需要包含：
                - index_path: Lucene索引路径
                - corpus_path: 语料库文件路径
        """
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)  # 初始化Lucene搜索器
        self.contain_doc = self._check_contain_doc()  # 检查索引是否包含完整文档
        if not self.contain_doc:
            # 如果索引不包含文档，加载外部语料库
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        """检查Lucene索引是否包含完整文档"""
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        """
        单个查询BM25搜索
        
        参数：
            query: 查询文本
            num: 返回结果数（默认为topk）
            return_score: 是否返回相似度分数
            
        返回：
            如果return_score=True: (结果列表, 分数列表)
            否则: 结果列表
        """
        if num is None:
            num = self.topk
        
        # 使用Lucene进行BM25搜索
        hits = self.searcher.search(query, num)
        
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        
        scores = [hit.score for hit in hits]  # 提取BM25分数
        
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        # 获取文档内容
        if self.contain_doc:
            # 从Lucene索引获取文档
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            # 从外部语料库加载文档
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """
        批量查询BM25搜索
        
        参数：
            query_list: 查询文本列表
            num: 返回结果数
            return_score: 是否返回分数
            
        返回：
            如果return_score=True: (结果列表的列表, 分数列表的列表)
            否则: 结果列表的列表
        """
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    """
    稠密检索器
    
    功能：基于FAISS向量索引的稠密检索
    
    工作原理：
    1. 编码器：将查询和文档转换为向量
    2. FAISS索引：使用高效的向量相似度搜索
    3. GPU加速：支持GPU上的FAISS操作
    """
    def __init__(self, config):
        """
        初始化稠密检索器
        
        参数：
            config: 配置对象，包含模型、索引等信息
        """
        super().__init__(config)
        
        # 加载FAISS索引
        self.index = faiss.read_index(self.index_path)
        
        if config.faiss_gpu:
            # 转移到GPU加速
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True  # 使用FP16节省显存
            co.shard = True  # 分片到多GPU
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # 加载语料库
        self.corpus = load_corpus(self.corpus_path)
        
        # 初始化编码器
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        """
        单个查询稠密检索
        
        参数：
            query: 查询文本
            num: 返回结果数
            return_score: 是否返回相似度分数
            
        返回：
            如果return_score=True: (结果列表, 相似度分数)
            否则: 结果列表
        """
        if num is None:
            num = self.topk
        
        # 编码查询
        query_emb = self.encoder.encode(query)
        
        # FAISS搜索
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        
        # 加载对应的文档
        results = load_docs(self.corpus, idxs)
        
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """
        批量查询稠密检索
        
        支持大规模批量查询，自动分批处理
        """
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        
        # 分批处理查询
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            
            # 批量编码
            batch_emb = self.encoder.encode(query_batch)
            
            # 批量FAISS搜索
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # 批量加载文档
            flat_idxs = sum(batch_idxs, [])  # 扁平化索引列表
            batch_results = load_docs(self.corpus, flat_idxs)
            # 重新分块为原始批大小
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            # 清理内存
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    """
    检索器工厂函数
    
    根据配置返回相应的检索器实例
    
    参数：
        config: 配置对象，config.retrieval_method决定返回的类型
        
    返回：
        BM25Retriever 或 DenseRetriever 实例
    """
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
