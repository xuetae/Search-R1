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
Contains commonly used utilities for ray
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import concurrent.futures


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parallel_put(data_list, max_workers=None):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def put_data(index, data):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return index, ray.put(data)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if max_workers is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_workers = min(len(data_list), 16)

    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        res_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for future in concurrent.futures.as_completed(data_list_f):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            res_lst.append(future.result())

        # reorder based on index
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = [None for _ in range(len(data_list))]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for res in res_lst:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            index, data_ref = res
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output[index] = data_ref

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output
