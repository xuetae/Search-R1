# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
检索系统模块
============
功能：实现多种检索方法（BM25、密集向量检索）
用于从文档库中检索与查询相关的文档

主要类：
- Encoder: 文本编码器（将文本转换为向量）
- BaseRetriever: 检索器基类
- BM25Retriever: 稀疏检索器（BM25算法）
- DenseRetriever: 密集检索器（向量相似度）

工作流程：
1. 加载或构建索引（Lucene/FAISS）
2. 将查询编码为向量或BM25表示
3. 在索引中搜索相似文档
4. 返回排名靠前的文档

关键概念：
- 稀疏检索(BM25): 关键词匹配，快速但准确度受限
- 密集检索(FAISS): 向量相似度，准确度高但需要GPU加速
- 融合检索: 结合两种方法的优点
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json  # JSON格式处理
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 文件路径操作
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings  # 警告信息
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Dict  # 类型注解
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import functools  # 函数工具（如缓存装饰器）
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tqdm import tqdm  # 进度条显示
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from multiprocessing import Pool  # 多进程处理
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import faiss  # Facebook AI相似度搜索库（密集检索）
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch张量库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoConfig, AutoTokenizer, AutoModel  # Hugging Face模型
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse  # 命令行参数解析
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import datasets  # Hugging Face数据集库


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_corpus(corpus_path: str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    从JSON文件加载文档库
    
    功能：加载包含文档的JSON行格式文件
    
    参数：
        corpus_path: JSON文件路径（每行一个JSON对象）
        
    返回：
        datasets.Dataset对象，可通过索引访问文档
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    corpus = datasets.load_dataset(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'json',  # 数据格式为JSON
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_files=corpus_path,  # 文件路径
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            split="train",  # 使用训练集分割
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_proc=4)  # 使用4个进程并行加载
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return corpus
    

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def read_jsonl(file_path):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    读取JSONL文件（每行一个JSON对象）
    
    参数：
        file_path: 文件路径
        
    返回：
        列表，每个元素是一个JSON对象
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data = []
    
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with open(file_path, "r") as f:  # 打开文件
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        readin = f.readlines()  # 读取所有行
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for line in readin:  # 逐行处理
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.append(json.loads(line))  # 解析JSON并添加到列表
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_docs(corpus, doc_idxs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    根据索引从文档库中加载文档
    
    参数：
        corpus: 文档库（Dataset对象）
        doc_idxs: 文档索引列表
        
    返回：
        文档列表
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = [corpus[int(idx)] for idx in doc_idxs]  # 批量检索文档

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return results


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_path: str,  # 模型路径
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_fp16: bool = False  # 是否使用半精度浮点数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    加载编码模型和分词器
    
    功能：从预训练模型加载编码器用于将文本转换为向量
    
    参数：
        model_path: 模型在Hugging Face或本地的路径
        use_fp16: 是否使用FP16混合精度以节省显存
        
    返回：
        (model, tokenizer) 元组
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # 加载配置
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)  # 加载模型
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.eval()  # 设置为评估模式（禁用Dropout等）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.cuda()  # 移到GPU
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if use_fp16:  # 如果需要FP16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.half()  # 转换为半精度
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)  # 加载分词器

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model, tokenizer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pooling(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pooler_output,  # 模型的池化输出
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        last_hidden_state,  # 最后一层隐藏状态
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = None,  # 注意力掩码
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pooling_method = "mean"  # 池化方法
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    对隐藏状态进行池化，得到固定长度的向量表示
    
    功能：将可变长度的序列转换为固定长度的向量
    
    参数：
        pooler_output: 模型的标准池化输出
        last_hidden_state: 最后一层隐藏状态 [batch_size, seq_len, hidden_dim]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        pooling_method: 池化方法 ("mean"/"cls"/"pooler")
        
    返回：
        池化后的向量 [batch_size, hidden_dim]
        
    池化方法说明：
    - mean: 对所有真实令牌取平均
    - cls: 使用[CLS]令牌的向量
    - pooler: 使用模型的标准池化输出
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pooling_method == "mean":
        # 平均池化：将非真实令牌设为0，然后平均
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)  # 掩盖填充位置
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  # 平均
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "cls":
        # CLS池化：使用第一个令牌的向量（通常是[CLS]）
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden_state[:, 0]
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "pooler":
        # 模型池化：使用模型自带的池化层输出
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
    ==========
    将文本转换为向量表示（嵌入），用于密集检索
    
    支持不同的模型和池化方法
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化编码器
        
        参数：
            model_name: 模型名称（用于确定预处理方法）
            model_path: 模型路径
            pooling_method: 池化方法
            max_length: 最大输入长度
            use_fp16: 是否使用半精度
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_name = model_name  # 保存模型名称
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_path = model_path  # 保存模型路径
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pooling_method = pooling_method  # 保存池化方法
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_length = max_length  # 保存最大长度
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_fp16 = use_fp16  # 保存精度标志

        # 加载模型和分词器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model, self.tokenizer = load_model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_path=model_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_fp16=use_fp16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()  # 禁用梯度计算（推理模式）
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        将文本编码为向量
        
        功能：将输入文本转换为固定维度的向量表示
        
        参数：
            query_list: 文本字符串列表（或单个字符串）
            is_query: 是否为查询（True）还是文档（False）
                     某些模型需要不同的前缀
            
        返回：
            向量矩阵 [num_texts, embedding_dim]
            
        支持的模型特殊处理：
        - E5模型: 查询和文档需要不同前缀
        - BGE模型: 查询需要特定前缀
        - T5模型: 需要解码器输入ID
        """
        # 处理单个查询的情况
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(query_list, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list = [query_list]

        # E5模型特殊处理：添加"query:"或"passage:"前缀
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "e5" in self.model_name.lower():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_query:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"query: {query}" for query in query_list]  # 查询添加前缀
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"passage: {query}" for query in query_list]  # 文档添加前缀

        # BGE模型特殊处理：查询添加特定前缀
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "bge" in self.model_name.lower():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_query:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"Represent this sentence for searching relevant passages: {query}" 
                            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                            for query in query_list]

        # 分词：将文本转换为令牌ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = self.tokenizer(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_length=self.max_length,  # 截断过长文本
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padding=True,  # 填充过短文本
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            truncation=True,  # 启用截断
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return_tensors="pt"  # 返回PyTorch张量
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 将输入移到GPU
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # T5模型需要特殊处理（编码器-解码器架构）
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "T5" in type(self.model).__name__:
            # T5基础检索模型
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decoder_input_ids = torch.zeros(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ).to(inputs['input_ids'].device)  # 解码器输入（初始化为0）
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.model(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                **inputs, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                decoder_input_ids=decoder_input_ids, 
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return_dict=True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )  # 前向传播
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_emb = output.last_hidden_state[:, 0, :]  # 取解码器第一个令牌作为表示

        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 其他模型（如BERT、E5、BGE）
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.model(**inputs, return_dict=True)  # 前向传播
            # 使用指定的池化方法获取向量
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_emb = pooling(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output.pooler_output,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output.last_hidden_state,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                inputs['attention_mask'],
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.pooling_method
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 大多数密集检索模型使用L2归一化
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "dpr" not in self.model_name.lower():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)  # L2归一化

        # 转换为NumPy数组并转到CPU
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.detach().cpu().numpy()  # 分离梯度并转到CPU
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.astype(np.float32, order="C")  # 转换为float32并使用C顺序
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return query_emb


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BaseRetriever:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    检索器基类
    ==========
    定义检索器的通用接口，所有具体的检索器都继承自此类
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化检索器
        
        参数：
            config: 配置对象，包含：
                - retrieval_method: 检索方法（bm25或embedding模型名）
                - retrieval_topk: 返回的文档数
                - index_path: 索引文件路径
                - corpus_path: 文档库文件路径
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config  # 保存配置
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_method = config.retrieval_method  # 检索方法
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = config.retrieval_topk  # 返回前K个结果
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index_path = config.index_path  # 索引路径
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus_path = config.corpus_path  # 文档库路径

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int, return_score:bool) -> List[Dict[str, str]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        单个查询搜索（内部方法）
        
        子类需要实现此方法
        
        参数：
            query: 查询字符串
            num: 返回的文档数
            return_score: 是否返回相关性分数
            
        返回：
            文档列表，每个文档是字典，包含：
            - contents: 用于构建索引的原始内容
            - title: 文档标题（如果提供）
            - text: 文档文本（如果提供）
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list, num, return_score):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """批量查询搜索（内部方法），子类可选实现"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def search(self, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """单个查询搜索（外部接口）"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._search(*args, **kwargs)
    
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """批量查询搜索（外部接口）"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._batch_search(*args, **kwargs)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BM25Retriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""
    BM25检索器
    ==========
    基于BM25算法的稀疏检索器，使用预构建的Lucene索引
    
    BM25是一种经典的信息检索算法，基于关键词匹配，快速高效
    适合处理大规模文档库
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化BM25检索器
        
        参数：
            config: 配置对象
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 导入pyserini库的LuceneSearcher
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from pyserini.search.lucene import LuceneSearcher
        # 加载Lucene索引
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.searcher = LuceneSearcher(self.index_path)
        # 检查索引中是否包含文档内容
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.contain_doc = self._check_contain_doc()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.contain_doc:
            # 如果索引不包含内容，加载单独的文档库
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.corpus = load_corpus(self.corpus_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_process_num = 8  # 最大进程数
        
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_contain_doc(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""
        检查索引是否包含文档内容
        
        某些Lucene索引可能只包含文档ID，需要单独加载文档
        """
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.searcher.doc(0).raw() is not None  # 尝试获取第一个文档

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        BM25单查询搜索
        
        参数：
            query: 查询字符串
            num: 返回的文档数（默认使用topk）
            return_score: 是否返回BM25分数
            
        返回：
            (文档列表, 分数列表) 或 文档列表
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk  # 使用默认的topk
        
        # 使用Lucene搜索
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hits = self.searcher.search(query, num)
        
        # 处理无结果的情况
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < 1:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if return_score:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return [],[]
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return []
        
        # 提取分数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = [hit.score for hit in hits]
        
        # 检查结果数量
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < num:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.warn('Not enough documents retrieved!')  # 文档不足警告
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hits = hits[:num]  # 截断到num个

        # 获取文档
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.contain_doc:
            # 从索引中获取文档
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_contents = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for hit in hits
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]
            # 解析文档（第一行是标题，剩余是文本）
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
            # 从单独的文档库中获取
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        # 返回结果
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list, num: int = None, return_score = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        BM25批量查询搜索
        
        TODO: 可以优化为真正的并行处理
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []
        # 逐个查询进行搜索（当前实现是串行的）
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

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_available_gpu_memory():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    获取可用的GPU显存
    
    返回：
        列表，每个元素是(GPU_ID, 可用显存GB)
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    memory_info = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(torch.cuda.device_count()):  # 遍历所有GPU
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_memory = torch.cuda.get_device_properties(i).total_memory  # 总显存
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        allocated_memory = torch.cuda.memory_allocated(i)  # 已分配显存
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        free_memory = total_memory - allocated_memory  # 可用显存
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_info.append((i, free_memory / 1e9))  # 转换为GB并保存
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return memory_info


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DenseRetriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""
    密集检索器
    ==========
    基于FAISS的密集向量检索
    
    使用深度学习模型将查询和文档编码为向量，通过向量相似度检索
    准确度高但需要更多计算资源
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化密集检索器
        
        参数：
            config: 配置对象
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 加载预构建的FAISS索引
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index = faiss.read_index(self.index_path)
        
        # 如果配置为使用GPU，将索引转到GPU
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.faiss_gpu:
            # GPU选项配置
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co = faiss.GpuMultipleClonerOptions()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.useFloat16 = True  # 使用FP16节省显存
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.shard = True  # 分片到多个GPU
            # 复制索引到所有可用GPU
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # 加载文档库
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus = load_corpus(self.corpus_path)
        
        # 初始化编码器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.encoder = Encoder(
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             model_name = self.retrieval_method,  # 模型名称
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             model_path = config.retrieval_model_path,  # 模型路径
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             pooling_method = config.retrieval_pooling_method,  # 池化方法
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             max_length = config.retrieval_query_max_length,  # 最大长度
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             use_fp16 = config.retrieval_use_fp16  # 是否使用FP16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = config.retrieval_topk  # 返回前K个
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch_size = self.config.retrieval_batch_size  # 批处理大小

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        密集检索单查询搜索
        
        参数：
            query: 查询字符串
            num: 返回的文档数
            return_score: 是否返回距离分数
            
        返回：
            (文档列表, 分数列表) 或 文档列表
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 将查询编码为向量
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = self.encoder.encode(query)
        # 在FAISS索引中搜索最近邻
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores, idxs = self.index.search(query_emb, k=num)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idxs = idxs[0]  # 取第一个查询的结果
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = scores[0]  # 取第一个查询的分数

        # 加载文档
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = load_docs(self.corpus, idxs)
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list: List[str], num: int = None, return_score = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        密集检索批量查询搜索
        
        功能：批量处理多个查询以提高效率
        
        参数：
            query_list: 查询字符串列表
            num: 返回的文档数
            return_score: 是否返回分数
            
        返回：
            批量搜索结果
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
        batch_size = self.batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []

        # 按批处理查询
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for start_idx in tqdm(range(0, len(query_list), batch_size), 
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            desc='Retrieval process: '):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_batch = query_list[start_idx:start_idx + batch_size]
            
            # 编码批查询
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_emb = self.encoder.encode(query_batch)
            # 搜索
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            # 转换为Python列表
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores = batch_scores.tolist()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_idxs = batch_idxs.tolist()
            
            # 展平索引列表
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            flat_idxs = sum(batch_idxs, [])
            # 批量加载文档
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = load_docs(self.corpus, flat_idxs)
            # 重新组织为列表的列表
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch_results[i*num : (i+1)*num] 
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(len(batch_idxs))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores.extend(batch_scores)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.extend(batch_results)
        
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
    r"""
    根据配置自动选择检索器类
    
    参数：
        config: 配置对象，包含'retrieval_method'键

    返回：
        检索器实例
        
    支持的方法：
    - bm25: 返回BM25Retriever
    - 其他（embedding模型名）: 返回DenseRetriever
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.retrieval_method == "bm25":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return BM25Retriever(config)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DenseRetriever(config)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_dataset(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    从配置加载数据集
    
    参数：
        config: 配置对象
        
    返回：
        数据集列表
    """
     
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    split_path = os.path.join(config.dataset_path, f'{config.data_split}.jsonl')
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return read_jsonl(split_path)


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    命令行使用示例
    """
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser(description = "Retrieval")

    # 基础参数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_method', type=str)  # 检索方法
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_topk', type=int, default=10)  # 返回前K个
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--index_path', type=str, default=None)  # 索引路径
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--corpus_path', type=str)  # 文档库路径
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--dataset_path', default=None, type=str)  # 数据集路径

    # 检索参数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_gpu', default=True, type=bool)  # 是否使用GPU
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--data_split', default="train", type=str)  # 数据集分割
    
    # 编码器参数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_model_path', type=str, default=None)  # 模型路径
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_pooling_method', default='mean', type=str)  # 池化方法
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_query_max_length', default=256, type=str)  # 最大查询长度
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_use_fp16', action='store_true', default=False)  # 是否使用FP16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_batch_size', default=512, type=int)  # 批大小
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()

    # 构建索引路径
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args.index_path = os.path.join(args.index_path, f'{args.retrieval_method}_Flat.index') \
        if args.retrieval_method != 'bm25' \
        else os.path.join(args.index_path, 'bm25')

    # 加载数据集
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_split = get_dataset(args)
    
    # 提取前512个样本的问题
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_query = [sample['question'] for sample in all_split[:512]]
    
    # 初始化检索器并执行检索
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retriever = get_retriever(args)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print('Start Retrieving ...')    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results, scores = retriever.batch_search(input_query, return_score=True)

    # from IPython import embed
    # embed()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    # corpus = datasets.load_dataset(
    #         'json',
    #         data_files=corpus_path,
    #         split="train",
    #         num_proc=4)
    # return corpus
    

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def read_jsonl(file_path):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data = []
    
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with open(file_path, "r") as f:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        readin = f.readlines()
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for line in readin:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.append(json.loads(line))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_docs(corpus, doc_idxs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = [corpus[int(idx)] for idx in doc_idxs]

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return results


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_path: str, 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_fp16: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.eval()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model.cuda()
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if use_fp16: 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.half()
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
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pooling_method == "mean":
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "cls":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return last_hidden_state[:, 0]
    # 中文注释：下一行继续判断其他条件分支。
    elif pooling_method == "pooler":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return pooler_output
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError("Pooling method not implemented!")


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Encoder:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
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
        self.model, self.tokenizer = load_model(model_path=model_path,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                use_fp16=use_fp16)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(query_list, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list = [query_list]

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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "bge" in self.model_name.lower():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_query:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = self.tokenizer(query_list,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                max_length=self.max_length,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                padding=True,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                truncation=True,
                                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                return_tensors="pt"
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
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
            query_emb = output.last_hidden_state[:, 0, :]

        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
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
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "dpr" not in self.model_name.lower():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.detach().cpu().numpy()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = query_emb.astype(np.float32, order="C")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return query_emb


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BaseRetriever:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Base object for all retrievers."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
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

        # self.cache_save_path = os.path.join(config.save_dir, 'retrieval_cache.json')

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int, return_score:bool) -> List[Dict[str, str]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""Retrieve topk relevant documents in corpus.
        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list, num, return_score):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def search(self, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._search(*args, **kwargs)
    
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._batch_search(*args, **kwargs)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BM25Retriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""BM25 retriever based on pre-built pyserini index."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from pyserini.search.lucene import LuceneSearcher
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.searcher = LuceneSearcher(self.index_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.contain_doc = self._check_contain_doc()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.contain_doc:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.corpus = load_corpus(self.corpus_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_process_num = 8
        
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_contain_doc(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""Check if the index contains document content
        """
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.searcher.doc(0).raw() is not None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hits = self.searcher.search(query, num)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < 1:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if return_score:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return [],[]
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return []
            
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = [hit.score for hit in hits]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(hits) < num:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.warn('Not enough documents retrieved!')
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hits = hits[:num]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.contain_doc:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results = [{'title': content.split("\n")[0].strip("\""), 
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        'text': "\n".join(content.split("\n")[1:]),
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        'contents': content} for content in all_contents]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
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
    def _batch_search(self, query_list, num: int = None, return_score = False):
        # TODO: modify batch method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for query in query_list:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            item_result, item_score = self._search(query, num,True)
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

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_available_gpu_memory():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    memory_info = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(torch.cuda.device_count()):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_memory = torch.cuda.get_device_properties(i).total_memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        allocated_memory = torch.cuda.memory_allocated(i)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        free_memory = total_memory - allocated_memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_info.append((i, free_memory / 1e9))  # Convert to GB
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return memory_info


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DenseRetriever(BaseRetriever):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""Dense retriever based on pre-built faiss index."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index = faiss.read_index(self.index_path)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.faiss_gpu:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co = faiss.GpuMultipleClonerOptions()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.useFloat16 = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.shard = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            # self.index = faiss.index_cpu_to_all_gpus(self.index)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus = load_corpus(self.corpus_path)
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
        self.batch_size = self.config.retrieval_batch_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search(self, query: str, num: int = None, return_score = False):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_emb = self.encoder.encode(query)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores, idxs = self.index.search(query_emb, k=num)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idxs = idxs[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = scores[0]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = load_docs(self.corpus, idxs)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if return_score:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results, scores
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, query_list: List[str], num: int = None, return_score = False):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(query_list, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_list = [query_list]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num = self.topk
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = self.batch_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = []

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_batch = query_list[start_idx:start_idx + batch_size]
            
            # from time import time
            # a = time()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_emb = self.encoder.encode(query_batch)
            # b = time()
            # print(f'################### encode time {b-a} #####################')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_scores = batch_scores.tolist()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_idxs = batch_idxs.tolist()
            # print(f'################### search time {time()-b} #####################')
            # exit()
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            flat_idxs = sum(batch_idxs, [])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = load_docs(self.corpus, flat_idxs)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores.extend(batch_scores)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.extend(batch_results)
        
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
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.retrieval_method == "bm25":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return BM25Retriever(config)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DenseRetriever(config)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_dataset(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Load dataset from config."""
     
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    split_path = os.path.join(config.dataset_path, f'{config.data_split}.jsonl')
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return read_jsonl(split_path)


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser(description = "Retrieval")

    # Basic parameters
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_method', type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_topk', type=int, default=10)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--index_path', type=str, default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--corpus_path', type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--dataset_path', default=None, type=str)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_gpu', default=True, type=bool)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--data_split', default="train", type=str)
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_model_path', type=str, default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_pooling_method', default='mean', type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_query_max_length', default=256, type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_use_fp16', action='store_true', default=False)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_batch_size', default=512, type=int)
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args.index_path = os.path.join(args.index_path, f'{args.retrieval_method}_Flat.index') if args.retrieval_method != 'bm25' else os.path.join(args.index_path, 'bm25')

    # load dataset
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_split = get_dataset(args)
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_query = [sample['question'] for sample in all_split[:512]]
    
    # initialize the retriever and conduct retrieval
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retriever = get_retriever(args)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print('Start Retrieving ...')    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results, scores = retriever.batch_search(input_query, return_score=True)

    # from IPython import embed
    # embed()
