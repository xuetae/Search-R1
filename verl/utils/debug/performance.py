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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        message = f'{head}, memory allocated (GB): {memory_allocated}, memory reserved (GB): {memory_reserved}'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if logger is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(message)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.log(msg=message, level=level)
