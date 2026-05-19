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
A Ray logger will receive logging info from different processes.
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numbers
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def concat_dict_to_str(dict: Dict, step):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = [f'step:{step}']
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for k, v in dict.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(v, numbers.Number):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.append(f'{k}:{v:.3f}')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_str = ' - '.join(output)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output_str


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LocalLogger:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.print_to_console = print_to_console
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if print_to_console:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print('Using LocalLogger is deprecated. The constructor API will change ')

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def flush(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def log(self, data, step):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.print_to_console:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(concat_dict_to_str(data, step=step), flush=True)