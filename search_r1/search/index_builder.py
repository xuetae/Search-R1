# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import faiss
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import cast, List, Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import shutil
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import subprocess
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tqdm import tqdm
# from LongRAG.retriever.utils import load_model, load_corpus, pooling
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import datasets
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoTokenizer, AutoModel, AutoConfig


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


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_corpus(corpus_path: str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    corpus = datasets.load_dataset(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'json', 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_files=corpus_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            split="train",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_proc=4)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return corpus


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Index_Builder:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    r"""A tool class used to build an index used in retrieval.
    
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self, 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            retrieval_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            corpus_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            save_dir,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_fp16,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pooling_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_type=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            embedding_path=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            save_embedding=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_gpu=False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ):
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.retrieval_method = retrieval_method.lower()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_path = model_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus_path = corpus_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.save_dir = save_dir
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_length = max_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch_size = batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_fp16 = use_fp16
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pooling_method = pooling_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.embedding_path = embedding_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.save_embedding = save_embedding
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.faiss_gpu = faiss_gpu

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(self.save_dir)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not os.path.exists(self.save_dir):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(self.save_dir)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not self._check_dir(self.save_dir):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.corpus = load_corpus(self.corpus_path)
       
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("Finish loading...")
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_dir(dir_path):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""Check if the dir path exists and if there is content.
        
        """
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if os.path.isdir(dir_path):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if len(os.listdir(dir_path)) > 0:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return False
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(dir_path, exist_ok=True)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return True

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def build_index(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""Constructing different indexes based on selective retrieval method.

        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.retrieval_method == "bm25":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.build_bm25_index()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.build_dense_index()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def build_bm25_index(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.save_dir = os.path.join(self.save_dir, "bm25")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.makedirs(self.save_dir, exist_ok=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temp_dir = self.save_dir + "/temp"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temp_file_path = temp_dir + "/temp.jsonl"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.makedirs(temp_dir)

        # if self.have_contents:
        #     shutil.copyfile(self.corpus_path, temp_file_path)
        # else:
        #     with open(temp_file_path, "w") as f:
        #         for item in self.corpus:
        #             f.write(json.dumps(item) + "\n")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shutil.copyfile(self.corpus_path, temp_file_path)
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("Start building bm25 index...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pyserini_args = ["--collection", "JsonCollection",
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "--input", temp_dir,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "--index", self.save_dir,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "--generator", "DefaultLuceneDocumentGenerator",
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "--threads", "1"]
       
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shutil.rmtree(temp_dir)
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("Finish!")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_embeddings = np.memmap(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                embedding_path,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                mode="r",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=np.float32
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ).reshape(corpus_size, hidden_size)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return all_embeddings

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _save_embedding(self, all_embeddings):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memmap = np.memmap(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.embedding_save_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            shape=all_embeddings.shape,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mode="w+",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=all_embeddings.dtype
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        length = all_embeddings.shape[0]
        # add in batch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        save_batch_size = 10000
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if length > save_batch_size:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                j = min(i + save_batch_size, length)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                memmap[i: j] = all_embeddings[i: j]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            memmap[:] = all_embeddings

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode_all(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.gpu_num > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print("Use multi gpu!")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.encoder = torch.nn.DataParallel(self.encoder)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.batch_size = self.batch_size * self.gpu_num

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_embeddings = []

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):

            # batch_data_title = self.corpus[start_idx:start_idx+self.batch_size]['title']
            # batch_data_text = self.corpus[start_idx:start_idx+self.batch_size]['text']
            # batch_data = ['"' + title + '"\n' + text for title, text in zip(batch_data_title, batch_data_text)]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.retrieval_method == "e5":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch_data = [f"passage: {doc}" for doc in batch_data]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs = self.tokenizer(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch_data,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        padding=True,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        truncation=True,
                        # 中文注释：下一行返回当前函数的计算结果或控制信号。
                        return_tensors='pt',
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        max_length=self.max_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ).to('cuda')

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs = {k: v.cuda() for k, v in inputs.items()}

            #TODO: support encoder-only T5 model
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "T5" in type(self.encoder).__name__:
                # T5-based retrieval model
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                decoder_input_ids = torch.zeros(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    (inputs['input_ids'].shape[0], 1), dtype=torch.long
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ).to(inputs['input_ids'].device)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.encoder(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                embeddings = output.last_hidden_state[:, 0, :]

            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.encoder(**inputs, return_dict=True)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                embeddings = pooling(output.pooler_output, 
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    output.last_hidden_state, 
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    inputs['attention_mask'],
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    self.pooling_method)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if  "dpr" not in self.retrieval_method:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            embeddings = cast(torch.Tensor, embeddings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            embeddings = embeddings.detach().cpu().numpy()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_embeddings.append(embeddings)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_embeddings = all_embeddings.astype(np.float32)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return all_embeddings

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def build_dense_index(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        """
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if os.path.exists(self.index_save_path):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print("The index file already exists and will be overwritten.")
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.encoder, self.tokenizer = load_model(model_path = self.model_path, 
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  use_fp16 = self.use_fp16)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.embedding_path is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_size = self.encoder.config.hidden_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            corpus_size = len(self.corpus)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_embeddings = self.encode_all()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.save_embedding:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._save_embedding(all_embeddings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            del self.corpus

        # build index
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("Creating index")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dim = all_embeddings.shape[-1]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.faiss_gpu:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co = faiss.GpuMultipleClonerOptions()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.useFloat16 = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            co.shard = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not faiss_index.is_trained:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                faiss_index.train(all_embeddings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_index.add(all_embeddings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not faiss_index.is_trained:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                faiss_index.train(all_embeddings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            faiss_index.add(all_embeddings)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        faiss.write_index(faiss_index, self.index_save_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("Finish!")


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
MODEL2POOLING = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "e5": "mean",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "bge": "cls",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "contriever": "mean",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'jina': 'mean'
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser(description = "Creating index.")

    # Basic parameters
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--retrieval_method', type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--model_path', type=str, default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--corpus_path', type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--save_dir', default= 'indexes/',type=str)

    # Parameters for building dense index
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--max_length', type=int, default=180)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--batch_size', type=int, default=512)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--use_fp16', default=False, action='store_true')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--pooling_method', type=str, default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_type',default=None,type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--embedding_path', default=None, type=str)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--save_embedding', action='store_true', default=False)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--faiss_gpu', default=False, action='store_true')
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if args.pooling_method is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pooling_method = 'mean'
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for k,v in MODEL2POOLING.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if k in args.retrieval_method.lower():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pooling_method = v
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                break
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if args.pooling_method not in ['mean','cls','pooler']:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pooling_method = args.pooling_method


    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    index_builder = Index_Builder(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        retrieval_method = args.retrieval_method,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        model_path = args.model_path,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        corpus_path = args.corpus_path,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        save_dir = args.save_dir,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        max_length = args.max_length,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch_size = args.batch_size,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        use_fp16 = args.use_fp16,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        pooling_method = pooling_method,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        faiss_type = args.faiss_type,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        embedding_path = args.embedding_path,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        save_embedding = args.save_embedding,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        faiss_gpu = args.faiss_gpu
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    index_builder.build_index()


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    main()
