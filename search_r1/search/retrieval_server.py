# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json  # JSON序列化
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 操作系统接口
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings  # 警告管理
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Dict, Optional  # 类型注解
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse  # 命令行参数解析

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import faiss  # Facebook AI Similarity Search
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch张量库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoConfig, AutoTokenizer, AutoModel  # Hugging Face模型加载
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tqdm import tqdm  # 进度条显示
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import datasets  # Hugging Face数据集库

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import uvicorn  # ASGI服务器
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from fastapi import FastAPI  # FastAPI Web框架
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pydantic import BaseModel  # 数据验证模型


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_corpus(corpus_path: str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    corpus = datasets.load_dataset(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'json', 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_files=corpus_path,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        split="train",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_proc=4  # 4进程并行加载
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return corpus


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def read_jsonl(file_path):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    读取JSONL文件
    
    参数：
        file_path: 文件路径
        
    返回：
        list: 文件中所有JSON对象的列表
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data = []
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with open(file_path, "r") as f:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for line in f:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.append(json.loads(line))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_docs(corpus, doc_idxs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    从语料库中获取指定索引的文档
    
    参数：
        corpus: 语料库数据集
        doc_idxs: 文档索引列表
        
    返回：
        list: 对应索引的文档列表
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = [corpus[int(idx)] for idx in doc_idxs]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return results


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_model(model_path: str, use_fp16: bool = False):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.eval()  # 评估模式
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.cuda()  # 移到GPU
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if use_fp16: 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.half()  # 转换为FP16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model, tokenizer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pooling(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pooler_output,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    last_hidden_state,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attention_mask = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pooling_method = "mean"
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pooling_method == "mean":
        # 平均池化：掩码位置置0，计算平均值
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "cls":
        # 使用第一个标记（通常是[CLS]）
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden_state[:, 0]
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "pooler":
        # 使用模型的pooler输出
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return pooler_output
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError("Pooling method not implemented!")

# 中文注释：下一行定义类，用于组织相关状态与行为。
class Encoder:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化编码器
        
        参数：
            model_name: 模型名称（用于检测模型类型）
            model_path: 模型路径
            pooling_method: 池化方法（mean/cls/pooler）
            max_length: 最大序列长度
            use_fp16: 是否使用FP16
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_name = model_name
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_path = model_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pooling_method = pooling_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_length = max_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_fp16 = use_fp16

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model.eval()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(query_list, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list = [query_list]

        # E5模型：添加查询/文档前缀
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "e5" in self.model_name.lower():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_query:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"query: {query}" for query in query_list]
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"passage: {query}" for query in query_list]

        # BGE模型：查询添加检索提示
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "bge" in self.model_name.lower():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_query:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # 分词
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = self.tokenizer(query_list,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                max_length=self.max_length,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                padding=True,  # 填充到最大长度
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                truncation=True,  # 截断超过最大长度的部分
                                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                return_tensors="pt"  # 返回PyTorch张量
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = {k: v.cuda() for k, v in inputs.items()}  # 移到GPU

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "T5" in type(self.model).__name__:
            # T5模型需要decoder_input_ids初始化
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decoder_input_ids = torch.zeros(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ).to(inputs['input_ids'].device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.model(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_emb = output.last_hidden_state[:, 0, :]  # 使用第一个token
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 其他模型的标准流程
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.model(**inputs, return_dict=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_emb = pooling(output.pooler_output,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                output.last_hidden_state,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                inputs['attention_mask'],
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                self.pooling_method)
            # DPR不需要正则化
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "dpr" not in self.model_name.lower():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)  # L2正则化

        # 转换为NumPy数组并清理GPU内存
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.detach().cpu().numpy()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.astype(np.float32, order="C")  # C顺序用于FAISS
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        del inputs, output
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return query_emb


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BaseRetriever:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    检索器基类
    
    定义检索器的标准接口
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """初始化检索器配置"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_method = config.retrieval_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = config.retrieval_topk
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index_path = config.index_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus_path = config.corpus_path

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int, return_score: bool):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """单个查询搜索（子类实现）"""
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """批量查询搜索（子类实现）"""
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def search(self, query: str, num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """单个查询搜索接口"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._search(query, num, return_score)
    
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """批量查询搜索接口"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._batch_search(query_list, num, return_score)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BM25Retriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    BM25稀疏检索器
    
    功能：基于Lucene索引的BM25算法检索
    
    工作原理：
    - BM25：统计学检索模型，考虑TF-IDF和文档长度
    - Lucene索引：倒排索引，支持快速词项查询
    - Pyserini：LuceneSearcher调用Lucene Java接口
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化BM25检索器
        
        参数：
            config: 配置对象，需要包含：
                - index_path: Lucene索引路径
                - corpus_path: 语料库文件路径
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from pyserini.search.lucene import LuceneSearcher
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.searcher = LuceneSearcher(self.index_path)  # 初始化Lucene搜索器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.contain_doc = self._check_contain_doc()  # 检查索引是否包含完整文档
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.contain_doc:
            # 如果索引不包含文档，加载外部语料库
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.corpus = load_corpus(self.corpus_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_process_num = 8

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_contain_doc(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """检查Lucene索引是否包含完整文档"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.searcher.doc(0).raw() is not None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 使用Lucene进行BM25搜索
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hits = self.searcher.search(query, num)
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < 1:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if return_score:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return [], []
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return []
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = [hit.score for hit in hits]  # 提取BM25分数
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < num:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.warn('Not enough documents retrieved!')
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hits = hits[:num]

        # 获取文档内容
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.contain_doc:
            # 从Lucene索引获取文档
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_contents = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for hit in hits
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'title': content.split("\n")[0].strip("\""),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'text': "\n".join(content.split("\n")[1:]),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'contents': content
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                } 
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for content in all_contents
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 从外部语料库加载文档
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for query in query_list:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            item_result, item_score = self._search(query, num, True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.append(item_result)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores.append(item_score)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DenseRetriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    稠密检索器
    
    功能：基于FAISS向量索引的稠密检索
    
    工作原理：
    1. 编码器：将查询和文档转换为向量
    2. FAISS索引：使用高效的向量相似度搜索
    3. GPU加速：支持GPU上的FAISS操作
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化稠密检索器
        
        参数：
            config: 配置对象，包含模型、索引等信息
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        
        # 加载FAISS索引
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index = faiss.read_index(self.index_path)
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.faiss_gpu:
            # 转移到GPU加速
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co = faiss.GpuMultipleClonerOptions()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.useFloat16 = True  # 使用FP16节省显存
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.shard = True  # 分片到多GPU
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # 加载语料库
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus = load_corpus(self.corpus_path)
        
        # 初始化编码器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.encoder = Encoder(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_name = self.retrieval_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_path = config.retrieval_model_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pooling_method = config.retrieval_pooling_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_length = config.retrieval_query_max_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_fp16 = config.retrieval_use_fp16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = config.retrieval_topk
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch_size = config.retrieval_batch_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 编码查询
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = self.encoder.encode(query)
        
        # FAISS搜索
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores, idxs = self.index.search(query_emb, k=num)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idxs = idxs[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = scores[0]
        
        # 加载对应的文档
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = load_docs(self.corpus, idxs)
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores.tolist()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        批量查询稠密检索
        
        支持大规模批量查询，自动分批处理
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(query_list, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list = [query_list]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []
        
        # 分批处理查询
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            
            # 批量编码
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_emb = self.encoder.encode(query_batch)
            
            # 批量FAISS搜索
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores = batch_scores.tolist()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_idxs = batch_idxs.tolist()

            # 批量加载文档
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            flat_idxs = sum(batch_idxs, [])  # 扁平化索引列表
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = load_docs(self.corpus, flat_idxs)
            # 重新分块为原始批大小
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.extend(batch_results)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores.extend(batch_scores)
            
            # 清理内存
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.empty_cache()
            
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_retriever(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    检索器工厂函数
    
    根据配置返回相应的检索器实例
    
    参数：
        config: 配置对象，config.retrieval_method决定返回的类型
        
    返回：
        BM25Retriever 或 DenseRetriever 实例
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.retrieval_method == "bm25":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return BM25Retriever(config)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

# 中文注释：下一行定义类，用于组织相关状态与行为。
class Config:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self, 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_method: str = "bm25", 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_topk: int = 10,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index_path: str = "./index/bm25",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        corpus_path: str = "./data/corpus.jsonl",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataset_path: str = "./data",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_split: str = "train",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        faiss_gpu: bool = True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_model_path: str = "./model",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_pooling_method: str = "mean",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_query_max_length: int = 256,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_use_fp16: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_batch_size: int = 128
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_method = retrieval_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_topk = retrieval_topk
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index_path = index_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus_path = corpus_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dataset_path = dataset_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.data_split = data_split
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.faiss_gpu = faiss_gpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_model_path = retrieval_model_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_pooling_method = retrieval_pooling_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_query_max_length = retrieval_query_max_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_use_fp16 = retrieval_use_fp16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_batch_size = retrieval_batch_size


# 中文注释：下一行定义类，用于组织相关状态与行为。
class QueryRequest(BaseModel):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    queries: List[str]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    topk: Optional[int] = None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return_scores: bool = False


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
app = FastAPI()

# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@app.post("/retrieve")
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def retrieve_endpoint(request: QueryRequest):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not request.topk:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results, scores = retriever.batch_search(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_list=request.queries,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num=request.topk,
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_score=request.return_scores
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    
    # Format response
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resp = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i, single_result in enumerate(results):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if request.return_scores:
            # If scores are returned, combine them with results
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            combined = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for doc, score in zip(single_result, scores[i]):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                combined.append({"document": doc, "score": score})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resp.append(combined)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resp.append(single_result)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {"result": resp}


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config = Config(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_method = args.retriever_name,  # or "dense"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index_path=args.index_path,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        corpus_path=args.corpus_path,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        retrieval_topk=args.topk,
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

    # 2) Instantiate a global retriever so it is loaded once and reused.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    uvicorn.run(app, host="0.0.0.0", port=8000)
