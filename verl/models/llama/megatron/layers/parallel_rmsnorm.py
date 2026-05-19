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
import numbers
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import LlamaConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import sequence_parallel as sp_utils


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaRMSNorm(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(config.hidden_size, numbers.Integral):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            normalized_shape = (config.hidden_size,)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.normalized_shape = torch.Size(normalized_shape)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.variance_epsilon = config.rms_norm_eps

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self, hidden_states):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return fused_rms_norm_affine(input=hidden_states,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     weight=self.weight,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     normalized_shape=self.normalized_shape,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     eps=self.variance_epsilon,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     memory_efficient=True)