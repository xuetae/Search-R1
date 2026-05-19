# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import torch.nn.functional as F
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def mark_parameter_as_sequence_parallel(parameter):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    setattr(parameter, 'sequence_parallel', True)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_sequence_parallel_param(param):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return hasattr(param, 'sequence_parallel') and param.sequence_parallel


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pad_to_sequence_parallel(unpad_tokens: torch.Tensor):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """pad the tokens such that the total length is a multiple of sp world size

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding

    Returns:

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_nnz = unpad_tokens.shape[0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_world_size = mpu.get_tensor_model_parallel_world_size()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if total_nnz % sp_world_size == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = sp_world_size - total_nnz % sp_world_size

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pad_size > 0:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if unpad_tokens.ndim == 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        # 中文注释：下一行继续判断其他条件分支。
        elif unpad_tokens.ndim == 2:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError(f'Padding dim {unpad_tokens.ndim()} is not supported')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return unpad_tokens
