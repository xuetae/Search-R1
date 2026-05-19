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
Contain small python utility functions
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from types import SimpleNamespace


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def union_two_dict(dict1: Dict, dict2: Dict):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in dict2.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if key in dict1:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert dict2[key] == dict1[key], \
                f'{key} in meta_dict1 and meta_dict2 are not the same object'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dict1[key] = val

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dict1


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def append_to_dict(data: Dict, new_data: Dict):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in new_data.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if key not in data:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data[key] = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data[key].append(val)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class NestedNamespace(SimpleNamespace):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, dictionary, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(**kwargs)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, value in dictionary.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(value, dict):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.__setattr__(key, NestedNamespace(value))
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.__setattr__(key, value)
