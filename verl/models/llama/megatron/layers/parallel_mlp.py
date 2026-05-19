# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import tensor_parallel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.activations import ACT2FN
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.models.llama.megatron.layers.parallel_linear import MergedColumnParallelLinear

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import tensor_parallel as tp_utils


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaMLP(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config, megatron_config: ModelParallelConfig = None) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hidden_size = config.hidden_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.intermediate_size = config.intermediate_size
        # The weight is only [hidden_size, intermediate_size // model_parallel_world_size]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert row_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gate_up_proj = MergedColumnParallelLinear(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_size=self.hidden_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gate_ouput_size=self.intermediate_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            up_output_size=self.intermediate_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            bias=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gather_output=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            skip_bias_add=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            **column_kwargs,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gate_size = self.intermediate_size // tp_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.down_proj = tensor_parallel.RowParallelLinear(input_size=self.intermediate_size,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           output_size=self.hidden_size,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           bias=False,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           input_is_parallel=True,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           skip_bias_add=False,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           **row_kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.act_fn = ACT2FN[config.hidden_act]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self, x):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gate_up = self.gate_up_proj(x)[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gate, up = gate_up.split(self.gate_size, dim=-1)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.down_proj(self.act_fn(gate) * up)[0]
