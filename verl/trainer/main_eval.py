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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import hydra
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.reward_score import math, gsm8k
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import pandas as pd
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def select_reward_fn(data_source):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if data_source == 'lighteval/MATH':
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return math.compute_score
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@hydra.main(config_path='config', config_name='evaluation', version_base=None)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_path = copy_local_path_from_hdfs(config.data.path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset = pd.read_parquet(local_path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompts = dataset[config.data.prompt_key]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    responses = dataset[config.data.response_key]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data_sources = dataset[config.data.data_source_key]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reward_model_data = dataset[config.data.reward_model_key]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    passes = 0

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total = len(dataset)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(total):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_lst = responses[i]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_source = data_sources[i]
        # select reward score based on data_source
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt = prompts[i]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_data = reward_model_data[i]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_fn = select_reward_fn(data_source)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ground_truth = reward_data['ground_truth']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        score_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for r in response_lst:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score = reward_fn(r, ground_truth)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            score_lst.append(score)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_score = np.max(score_lst)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_score == 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            passes += 1

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f'pass@5: {passes / total}')


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    main()
