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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re  # 正则表达式处理
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 文件路径操作
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import datasets  # Hugging Face数据集库

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.hdfs_io import copy, makedirs  # HDFS文件操作
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse  # 命令行参数解析


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def make_prefix(dp, template_type):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
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
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    question = dp['question']  # 提取问题

    # 注意：也需要在reward_score/countdown.py中更改相应模板
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if template_type == 'base':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """This works for any base model"""
        # 构造提示词模板
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefix = f"""Answer the given question. \
You should first have a reasoning process in mind and then provides the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>. \
Question: {question}\n"""
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError  # 其他模板未实现
        
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return prefix


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 命令行参数解析
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser = argparse.ArgumentParser()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--local_dir', default='./data/nq', help='本地存储目录')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--hdfs_dir', default=None, help='HDFS存储目录（可选）')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parser.add_argument('--template_type', type=str, default='base', help='提示词模板类型')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = parser.parse_args()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data_source = 'nq'  # 数据源标识

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    train_dataset = dataset['train']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    test_dataset = dataset['test']

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
