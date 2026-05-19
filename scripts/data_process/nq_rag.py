# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Preprocess the nq dataset to parquet format
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import datasets

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.hdfs_io import copy, makedirs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def make_prefix(dp, template_type):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    question = dp['question']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    context = dp['context']

    # NOTE: also need to change reward_score/countdown.py
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if template_type == 'base':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """This works for any base model"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefix = f"""Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question} Context: {context} \n"""
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError    
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return prefix


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def format_reference(retrieval_result):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    format_reference = ''
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for idx, doc_item in enumerate(retrieval_result):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        content = doc_item['contents']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        title = content.split("\n")[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        text = "\n".join(content.split("\n")[1:])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return format_reference


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--local_dir', default='./data/nq_rag')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--hdfs_dir', default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--template_type', type=str, default='base')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--topk', type=int, default=3)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--corpus_path', type=str, default='/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--train_retrieval_cache', type=str, default='/home/peterjin/rag_retrieval_cache/nq/e5_train_retrieval_cache_2048.json')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--test_retrieval_cache', type=str, default='/home/peterjin/rag_retrieval_cache/nq/e5_test_retrieval_cache_10000.json')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data_source = 'nq'

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    train_dataset = dataset['train']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    test_dataset = dataset['test']

    # read retrieval cache
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print('reading retrieval cache...')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retrieval_cache = json.load(open(args.train_retrieval_cache))
    # test_retrieval_cache = json.load(open(args.test_retrieval_cache))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    retrieval_cache.update(json.load(open(args.test_retrieval_cache)))

    # read corpus
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print('reading corpus...')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    corpus = {}
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with open(args.corpus_path) as f:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        readin = f.readlines()
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for line in readin:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tmp = json.loads(line)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            corpus[tmp['id']] = tmp

    # add a column for the retrieval context
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_context(example):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        example['context'] = format_reference([corpus[docs["id"]] for docs in retrieval_cache[example['question']][:args.topk]])
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return example
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    train_dataset = train_dataset.map(function=add_context)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    test_dataset = test_dataset.map(function=add_context)

    # add a row to each data item that represents a unique id
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def make_map_fn(split):

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def process_fn(example, idx):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            example['question'] = example['question'].strip()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if example['question'][-1] != '?':
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                example['question'] += '?'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            question = make_prefix(example, template_type=args.template_type)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            solution = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "target": example['golden_answers'],
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "data_source": data_source,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "prompt": [{
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    "role": "user",
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    "content": question,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }],
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "ability": "fact-reasoning",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "reward_model": {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    "style": "rule",
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    "ground_truth": solution
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                },
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "extra_info": {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'split': split,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'index': idx,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return data

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return process_fn

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_dir = args.local_dir
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hdfs_dir = args.hdfs_dir

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if hdfs_dir is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        makedirs(hdfs_dir)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        copy(src=local_dir, dst=hdfs_dir)
