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
"""
NQ数据集预处理脚本
===================
功能：将NQ（Natural Questions）数据集转换为parquet格式
数据集链接：https://ai.google.com/research/NaturalQuestions/

处理步骤：
1. 加载NQ数据集
2. 生成问题前缀（包含系统提示）
3. 转换为parquet格式便于后续使用
4. 可选：上传到HDFS分布式存储
"""

import re  # 正则表达式处理
import os  # 文件路径操作
import datasets  # Hugging Face数据集库

from verl.utils.hdfs_io import copy, makedirs  # HDFS文件操作
import argparse  # 命令行参数解析


def make_prefix(dp, template_type):
    """
    为数据点生成提示词前缀
    
    功能：根据问题和模板类型生成系统提示词
    
    参数：
        dp: 数据点（字典），包含'question'键
        template_type: 模板类型（'base'或其他）
        
    返回：
        完整的提示词前缀字符串
        
    模板说明：
    - base模板：通用模板，适用于任何基础模型
           包含思考和答案的标签
    """
    question = dp['question']  # 提取问题

    # 注意：也需要在reward_score/countdown.py中更改相应模板
    if template_type == 'base':
        """This works for any base model"""
        # 构造提示词模板
        prefix = f"""Answer the given question. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question}\n"""
    else:
        raise NotImplementedError  # 其他模板未实现
        
    return prefix


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq', help='本地存储目录')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS存储目录（可选）')
    parser.add_argument('--template_type', type=str, default='base', help='提示词模板类型')

    args = parser.parse_args()

    data_source = 'nq'  # 数据源标识

    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
