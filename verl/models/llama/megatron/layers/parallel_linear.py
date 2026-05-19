# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import tensor_parallel


# 中文注释：下一行定义类，用于组织相关状态与行为。
class QKVParallelLinear(tensor_parallel.ColumnParallelLinear):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 input_size,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 num_heads,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 num_key_value_heads,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 head_dim,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 *,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 bias=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 gather_output=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 skip_bias_add=False,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 **kwargs):
        # Keep input parameters, and already restrict the head numbers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.input_size = input_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.q_output_size = num_heads * head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.kv_output_size = num_key_value_heads * head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.head_dim = head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gather_output = gather_output
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.skip_bias_add = skip_bias_add

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_size = self.input_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_size = (num_heads + 2 * num_key_value_heads) * self.head_dim

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(input_size=input_size,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         output_size=output_size,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         bias=bias,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         gather_output=gather_output,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         skip_bias_add=skip_bias_add,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         **kwargs)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MergedColumnParallelLinear(tensor_parallel.ColumnParallelLinear):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 input_size,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 gate_ouput_size,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 up_output_size,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 *,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 bias=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 gather_output=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 skip_bias_add=False,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 **kwargs):
        # Keep input parameters, and already restrict the head numbers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.input_size = input_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.output_size = gate_ouput_size + up_output_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gather_output = gather_output
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.skip_bias_add = skip_bias_add

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(input_size=self.input_size,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         output_size=self.output_size,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         bias=bias,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         gather_output=gather_output,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         skip_bias_add=skip_bias_add,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         **kwargs)
